"""Adapt ZeroPoint to your own labelled texts.

    pip install "zeropoint[fit]"           # adds torch and onnx; deciding never needs them

    from zeropoint import ZeroPoint
    zp = ZeroPoint()
    mine, report = zp.fit(texts, labels)   # labels: the right option for each text, or None for "none of these"
    mine.save("my-model")                  # later: ZeroPoint("my-model")

What happens:
  * Your labels become the options. Each example becomes lists of 4 to 20 of them, like in the original training;
    in 30% of the lists the right option is removed, so the model keeps learning to say "none".
  * A held-out part of your examples (20%, chosen by a hash of the group id, or of the text when there are no
    groups) is never trained on; the report measures the model before and after on it.
  * The word embeddings and the first `frozen_layers` encoder layers stay as shipped; the remaining layers, the head,
    the temperature and the "none" level are trained for exactly `steps` steps. Same data, same seed and same number
    of threads give the same model, bit for bit.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from typing import Sequence

import numpy as np

from ._runtime import MODEL_FILES, Runtime, canonical_order


def _need(module: str):
    try:
        return __import__(module)
    except ImportError:
        raise ImportError(f'fit() needs {module}: pip install "zeropoint[fit]"') from None


# ---------------------------------------------------------------- the encoder in PyTorch, read from encoder.onnx

def _encoder_weights(model_dir: str) -> dict[str, np.ndarray]:
    onnx = _need("onnx")
    from onnx import numpy_helper
    m = onnx.load(os.path.join(model_dir, "encoder.onnx"))
    out = {}
    for init in m.graph.initializer:
        name = init.name
        if name.endswith("__fp16"):
            name = name[: -len("__fp16")]
        if name.startswith("enc."):
            out[name[4:]] = numpy_helper.to_array(init).astype(np.float32)
    return out


def _build_torch_encoder(weights: dict[str, np.ndarray], manifest: dict):
    torch = _need("torch")
    nn = torch.nn
    H, heads, n_layers, eps = (manifest["hidden_size"], manifest["attention_heads"], manifest["layers"],
                               manifest["layer_norm_eps"])

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q, self.k, self.v, self.o = (nn.Linear(H, H) for _ in range(4))
            self.ln1 = nn.LayerNorm(H, eps=eps)
            inter = weights["encoder.layer.0.intermediate.dense.weight"].shape[0]
            self.fc1, self.fc2 = nn.Linear(H, inter), nn.Linear(inter, H)
            self.ln2 = nn.LayerNorm(H, eps=eps)

        def forward(self, x, mask):
            B, L, _ = x.shape
            d = H // heads
            q = self.q(x).view(B, L, heads, d).transpose(1, 2)
            k = self.k(x).view(B, L, heads, d).transpose(1, 2)
            v = self.v(x).view(B, L, heads, d).transpose(1, 2)
            att = (q @ k.transpose(-1, -2)) / math.sqrt(d) + mask[:, None, None, :]
            ctx = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, L, H)
            x = self.ln1(x + self.o(ctx))
            return self.ln2(x + self.fc2(torch.nn.functional.gelu(self.fc1(x))))

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.pos = nn.Embedding(*weights["embeddings.position_embeddings.weight"].shape)
            self.typ = nn.Embedding(*weights["embeddings.token_type_embeddings.weight"].shape)
            self.ln = nn.LayerNorm(H, eps=eps)
            self.layers = nn.ModuleList(Layer() for _ in range(n_layers))

        def forward(self, embeds, pos, mask):
            x = self.ln(embeds + self.pos(pos) + self.typ.weight[0])
            for layer in self.layers:
                x = layer(x, mask)
            return x

    enc = Encoder()
    names = {"pos.weight": "embeddings.position_embeddings.weight",
             "typ.weight": "embeddings.token_type_embeddings.weight",
             "ln.weight": "embeddings.LayerNorm.weight", "ln.bias": "embeddings.LayerNorm.bias"}
    parts = {"q": "attention.self.query", "k": "attention.self.key", "v": "attention.self.value",
             "o": "attention.output.dense", "ln1": "attention.output.LayerNorm", "fc1": "intermediate.dense",
             "fc2": "output.dense", "ln2": "output.LayerNorm"}
    for i in range(n_layers):
        for ours, theirs in parts.items():
            for wb in ("weight", "bias"):
                names[f"layers.{i}.{ours}.{wb}"] = f"encoder.layer.{i}.{theirs}.{wb}"
    state = {ours: torch.from_numpy(weights[theirs].copy()) for ours, theirs in names.items()}
    enc.load_state_dict(state, strict=True)
    return enc, names


# ---------------------------------------------------------------- examples to lists

def _held_out(key: str) -> bool:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 5 == 0


def _lists(texts, labels, label_set, rng, repeat=1, none_share=0.3):
    """Lists of 4-20 options from the user's labels, in canonical order, with the right option or without it."""
    out = []
    for _ in range(repeat):
        for t, y in zip(texts, labels):
            others = [l for l in label_set if l != y]
            k = int(rng.integers(min(4, len(label_set)), min(20, len(label_set)) + 1))
            gold = y if (y is not None and rng.random() >= none_share) else None
            n_wrong = k - (1 if gold is not None else 0)
            opts = [str(o) for o in rng.choice(others, min(n_wrong, len(others)), replace=False)]
            if gold is not None:
                opts.append(gold)
            if not opts:
                continue
            canon = [opts[i] for i in canonical_order(opts)]
            out.append((t, canon, gold))
    return out


# ---------------------------------------------------------------- report

def _evaluate(zp, cases) -> dict:
    """Decision quality on held-out lists, with the real inference path."""
    if not cases:
        return {}
    rows = [(zp.decide(t, opts), gold) for t, opts, gold in cases]
    right = np.array([d.choice == g for d, g in rows], float)
    conf = np.array([d.confidence for d, _ in rows])
    neg = [d for d, g in rows if g is None]
    pos = [(d, g) for d, g in rows if g is not None]
    sure = conf >= 0.9
    ece = 0.0
    for lo in np.linspace(0, 1, 10, endpoint=False):
        m = (conf > lo) & (conf <= lo + 0.1)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - right[m].mean())
    return {"lists": len(rows),
            "decision_accuracy": float(right.mean()),
            "choice_accuracy": float(np.mean([max(d.probabilities, key=d.probabilities.get) == g for d, g in pos]))
            if pos else None,
            "false_admission_090": float(np.mean([d.choice is not None and d.confidence >= 0.9 for d in neg]))
            if neg else None,
            "sure_share": float(sure.mean()),
            "wrong_when_sure": float(1 - right[sure].mean()) if sure.any() else None,
            "ece": float(ece)}


# ---------------------------------------------------------------- fit

def fit(zp, texts: Sequence[str], labels: Sequence[str | None], *, groups: Sequence[str] | None = None,
        steps: int = 200, frozen_layers: int = 8, seed: int = 0, threads: int | None = None,
        lr: float = 3e-5, batch: int = 16):
    torch = _need("torch")
    from .model import ZeroPoint

    texts, labels = list(texts), list(labels)
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    if groups is not None and len(groups) != len(texts):
        raise ValueError("groups must have one entry per text")
    for t in texts:
        zp._check(t, ["x"])
    label_set = sorted({l for l in labels if l is not None}, key=lambda s: s.encode("utf-8"))
    if len(label_set) < 2:
        raise ValueError("fit needs at least two different labels")
    keys = list(groups) if groups is not None else texts
    hold = [_held_out(str(k)) for k in keys]
    tr = [(t, y) for t, y, h in zip(texts, labels, hold) if not h]
    ho = [(t, y) for t, y, h in zip(texts, labels, hold) if h]
    if len(tr) < 20:
        raise ValueError(f"fit needs at least 20 training examples after holding out 20%, got {len(tr)}")

    rt: Runtime = zp._rt
    manifest = rt.manifest
    n_layers = manifest["layers"]
    if not 0 <= frozen_layers <= n_layers:
        raise ValueError(f"frozen_layers must be between 0 and {n_layers}")

    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads or rt.threads)
    torch.manual_seed(seed)
    det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    t0 = time.perf_counter()
    try:
        weights = _encoder_weights(rt.model_dir)
        enc, names = _build_torch_encoder(weights, manifest)
        for n, p in enc.named_parameters():
            layer = int(n.split(".")[1]) if n.startswith("layers.") else -1
            p.requires_grad_(layer >= frozen_layers)
        w = torch.nn.Parameter(torch.tensor(rt.w, dtype=torch.float32))
        bias = torch.nn.Parameter(torch.tensor(rt.bias, dtype=torch.float32))
        log_t = torch.nn.Parameter(torch.tensor(zp._log_temperature, dtype=torch.float32))
        b = torch.nn.Parameter(torch.tensor(zp._none_level, dtype=torch.float32))
        params = [p for p in enc.parameters() if p.requires_grad] + [w, bias]
        opt = torch.optim.AdamW([{"params": params, "lr": lr}, {"params": [log_t, b], "lr": 1e-2}],
                                weight_decay=0.01)
        warm = max(1, steps // 10)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / steps))

        rng = np.random.default_rng(seed)
        pool = []
        while len(pool) < steps * batch:
            pool += _lists([t for t, _ in tr], [y for _, y in tr], label_set, rng)
        pool = pool[: steps * batch]

        def loss_on(cases):
            seqs, owner, golds, sizes = [], [], [], []
            for ci, (t, canon, gold) in enumerate(cases):
                ss, _ = rt.sequences(t, canon)
                for s in ss:
                    seqs.append(s)
                    owner.append(ci)
                golds.append(canon.index(gold) if gold is not None else len(canon))
                sizes.append(len(canon))
            L = max(len(ids) for ids, _, _ in seqs)
            E = np.zeros((len(seqs), L, manifest["hidden_size"]), np.float32)
            P = np.zeros((len(seqs), L), np.int64)
            M = np.full((len(seqs), L), -1e4, np.float32)
            for i, (ids, pos, _) in enumerate(seqs):
                E[i, :len(ids)] = rt._embed(ids)[0]
                P[i, :len(ids)] = pos
                M[i, :len(ids)] = 0.0
            h = enc(torch.from_numpy(E), torch.from_numpy(P), torch.from_numpy(M))
            depth = [[] for _ in cases]
            for i, (_, _, spans) in enumerate(seqs):
                for a, z in spans:
                    depth[owner[i]].append(torch.nn.functional.softplus(h[i, a:z].mean(0) @ w + bias))
            km = max(sizes)
            rows = []
            for d in depth:
                d = torch.stack(d)
                x = (d / log_t.exp()).clamp(1e-6, 60)
                lw = x + torch.log(-torch.expm1(-x))
                r = torch.full((km + 1,), -1e4)
                r = torch.cat([lw, r[len(d):km], b.reshape(1)])
                rows.append(r)
            return torch.nn.functional.cross_entropy(torch.stack(rows), torch.tensor(golds))

        enc.train()
        for step in range(steps):
            loss = loss_on(pool[step * batch:(step + 1) * batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(det)
    seconds = time.perf_counter() - t0

    trained = {theirs: enc.state_dict()[ours].detach().numpy() for ours, theirs in names.items()}
    new = ZeroPoint._from_fit(zp, trained, w.detach().numpy().astype(np.float64), float(bias.detach()),
                              float(log_t.detach()), float(b.detach()),
                              fit_info={"examples": len(texts), "trained_on": len(tr), "held_out": len(ho),
                                        "labels": label_set, "steps": steps, "frozen_layers": frozen_layers,
                                        "seed": seed, "threads": threads or rt.threads, "seconds": seconds})
    hrng = np.random.default_rng(seed + 1)
    held = _lists([t for t, _ in ho], [y for _, y in ho], label_set, hrng)
    report = {"before": _evaluate(zp, held), "after": _evaluate(new, held), "held_out_examples": len(ho),
              "trained_on": len(tr), "seconds": round(seconds, 1), "steps": steps, "frozen_layers": frozen_layers,
              "seed": seed, "labels": label_set,
              "note": "Measured on your held-out examples only. Thresholds and confidences fitted on one set of "
                      "options and one language do not carry over to others: measure again when either changes."}
    new.report = report
    return new, report


def write_model_dir(base_dir: str, out_dir: str, encoder_bytes: bytes | None, head: dict, manifest: dict) -> None:
    """Write a complete model folder: changed files from memory, unchanged ones copied, fresh hashes."""
    os.makedirs(out_dir, exist_ok=True)
    for f in MODEL_FILES:
        dst = os.path.join(out_dir, f)
        if f == "encoder.onnx" and encoder_bytes is not None:
            open(dst, "wb").write(encoder_bytes)
        elif f == "head.json":
            json.dump(head, open(dst, "w"), indent=1)
        elif os.path.abspath(base_dir) != os.path.abspath(out_dir):
            shutil.copyfile(os.path.join(base_dir, f), dst)
    from ._runtime import _sha256
    manifest = dict(manifest)
    manifest["files"] = {f: _sha256(os.path.join(out_dir, f)) for f in MODEL_FILES}
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
