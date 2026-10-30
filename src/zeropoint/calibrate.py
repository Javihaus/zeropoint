"""Calibrating ZeroPoint on your own labelled texts.

Needs PyTorch: ``pip install zeropoint[train]``. Add ``[monitor]`` to watch training with ndt.

What is learned, and from what:

* the wells (centre, depth, width of each option) and the text map, from your labelled texts;
* the temperature, on a held-out tenth of them;
* the edge of your data (class means, a shared covariance and two percentiles of distance), from in-domain
  texts only. No example of "out of scope" is needed or used.

The encoder stays frozen. Its embeddings are computed once and training runs on them, so a few thousand
examples train in a minute or two, even on a CPU.
"""

from __future__ import annotations

import datetime
import math
import warnings
from typing import Sequence

import numpy as np

from . import __version__
from .encoder import EncoderSpec, get_encoder

RECIPE = {
    "dim": 256,             # size of the learned space
    "hidden": 512,
    "epochs": 60,
    "min_epochs": 10,
    "patience": 8,
    "warmup": 3,            # first epochs without "none" examples
    "batch": None,          # None: 256, or fewer for small datasets so every epoch has enough steps
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "none_share": 0.125,    # share of "none" examples once warm-up is over
    "max_options": 20,      # options shown per example when there are more labels
    "depth0": 3.0,          # every well starts deep and wide, so every text feels a slope towards it
    "width0": 2.0,
    "learn_margin": 0.10,   # validation accuracy must beat chance by this much
    "kappa": 2.0,           # how fast the vacuum deepens beyond the edge of the data
}


class DidNotLearn(RuntimeError):
    """Raised when the model does not beat chance on held-out texts. A broken model is never returned."""


def ledoit_wolf_precision(x: np.ndarray) -> np.ndarray:
    """Inverse of the Ledoit-Wolf shrunk covariance of the rows of x."""
    x = x - x.mean(0)
    n, p = x.shape
    s = x.T @ x / n
    mu = np.trace(s) / p
    x2 = x ** 2
    beta_ = (x2.T @ x2).sum()
    delta_ = ((x.T @ x) ** 2).sum() / n ** 2
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2 * mu * np.trace(s) + p * mu ** 2) / p
    shrink = 0.0 if beta == 0 else min(beta, delta) / delta
    cov = (1 - shrink) * s + shrink * mu * np.eye(p)
    return np.linalg.inv(cov)


def _edge(emb: np.ndarray, y: np.ndarray, n_labels: int, seed: int):
    """Class means and shared precision from 90% of the texts; distance percentiles from the other 10%."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    ncal = max(len(y) // 10, 1)
    cal, fit = order[:ncal], order[ncal:]
    classes = [c for c in range(n_labels) if (y[fit] == c).sum() >= 5]
    means = np.stack([emb[fit][y[fit] == c].mean(0) for c in classes])
    pos = {c: k for k, c in enumerate(classes)}
    keep = np.array([c in pos for c in y[fit]])
    resid = emb[fit][keep] - means[[pos[c] for c in y[fit][keep]]]
    prec = ledoit_wolf_precision(resid)
    zp = emb[cal] @ prec
    d = ((zp * emb[cal]).sum(1)[:, None] - 2 * zp @ means.T + ((means @ prec) * means).sum(1)[None]).min(1)
    return means.astype(np.float32), prec.astype(np.float32), float(np.percentile(d, 50)), float(np.percentile(d, 99))


def _layouts(y: np.ndarray, n: int, catch_all: set[int], rng, with_none: bool, r: dict):
    """Training examples: (text index, option indices, target). Target == len(options) means "none"."""
    specific = [j for j in range(n) if j not in catch_all]
    q = r["none_share"] / (1 - r["none_share"])
    out = []
    for i, g in enumerate(y):
        g = int(g)
        if n > r["max_options"]:
            others = rng.choice([j for j in range(n) if j != g], r["max_options"] - 1, replace=False)
            opts = sorted([g] + [int(o) for o in others])
        elif g in catch_all or n <= 2:
            opts = list(range(n))
        else:
            u = rng.random()
            wrong = [j for j in specific if j != g]
            if u < 1 / 3 or not wrong:
                opts = list(range(n))
            elif u < 2 / 3 and catch_all:
                opts = list(specific)
            else:
                drop = int(rng.choice(wrong))
                opts = [j for j in specific if j != drop] if catch_all else [j for j in range(n) if j != drop]
        out.append((i, opts, opts.index(g)))
        if with_none and n > 2 and rng.random() < q:
            rest = [j for j in opts if j != g and j not in catch_all]
            if rest:
                out.append((i, rest, len(rest)))
    return out


def calibrate(texts: Sequence[str], labels: Sequence[str], *, descriptions: dict[str, str] | None = None,
              catch_all: Sequence[str] = (), encoder: str = "small", precision: str = "fp32",
              seed: int = 0, monitor: bool = False, device: str | None = None, embeddings: np.ndarray | None = None,
              recipe: dict | None = None, edge_int8: bool = True, verbose: bool = True):
    """Calibrate ZeroPoint on labelled texts and return the model.

    texts, labels  one label per text; every distinct label becomes an option
    descriptions   optional {label: short description}; the well of each option is built from
                   "label: description", which helps when label names are terse
    catch_all      labels that are right for anything (e.g. "General Inquiry"); they are never offered
                   in a "none" example, because with them on the list some option always fits
    encoder        "small" or "base" (or any E5-style model on the Hugging Face Hub with an onnx/ folder)
    precision      precision of the encoder while calibrating; fp32 is recommended
    seed           training seed
    monitor        watch the effective dimensionality of the learned space with ndt
                   (``pip install zeropoint[monitor]``); a collapse shows up even when the loss looks fine
    embeddings     precomputed "query" embeddings of `texts`, if you have them
    recipe         overrides for any value in ``zeropoint.calibrate.RECIPE``
    edge_int8      also measure the edge of the data with the 8-bit encoder, so the model is ready for the
                   default ``ZeroPoint.load(path)``; costs one more pass of the (faster) 8-bit encoder
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from .model import FORMAT, ZeroPoint

    r = {**RECIPE, **(recipe or {})}
    texts, labels = list(texts), list(labels)
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    names = sorted(set(labels))
    if len(names) < 2:
        raise ValueError("At least two different labels are needed.")
    descriptions = descriptions or {}
    desc = [f"{k}: {descriptions[k]}" if descriptions.get(k) else k for k in names]
    ca = {names.index(c) for c in catch_all}
    y = np.array([names.index(lab) for lab in labels])
    log = print if verbose else (lambda *a, **k: None)

    spec = EncoderSpec.resolve(encoder, precision)
    enc = get_encoder(spec)
    if embeddings is None:
        log(f"Embedding {len(texts)} texts with {spec.repo} ({precision})...")
        embeddings = enc.encode(texts, "query")
    emb = np.asarray(embeddings, dtype=np.float32)
    opt_emb = enc.encode(desc, "passage")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = np.random.default_rng(99 + seed).permutation(len(y))
    nval = max(20, len(y) // 10)
    val_i, tr_i = perm[:nval], perm[nval:]
    if r["batch"] is None:
        r["batch"] = int(np.clip(len(tr_i) // 50, 16, 256))

    def mlp(i, o):
        return nn.Sequential(nn.Linear(i, r["hidden"]), nn.GELU(approximate="tanh"), nn.Linear(r["hidden"], o))

    def inv_softplus(v):
        return math.log(math.expm1(v))

    class Net(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.f, self.g = mlp(d_in, r["dim"]), mlp(d_in, r["dim"])
            self.depth, self.width = nn.Linear(d_in, 1), nn.Linear(d_in, 1)
            for lin, v in [(self.depth, r["depth0"]), (self.width, r["width0"])]:
                nn.init.zeros_(lin.weight)
                nn.init.constant_(lin.bias, inv_softplus(v))
            self.log_t = nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward(self, x, o, opt):
            z = F.normalize(self.f(x), dim=-1)
            c = F.normalize(self.g(opt), dim=-1)[o]
            a = (F.softplus(self.depth(opt)).squeeze(-1) + 0.05)[o]
            s2 = (F.softplus(self.width(opt)).squeeze(-1) + 0.05)[o]
            e = -a * torch.exp(-((z.unsqueeze(1) - c) ** 2).sum(-1) / s2)
            logw = torch.log(torch.expm1((-e / self.log_t.exp()).clamp_min(1e-6)))
            return torch.cat([logw, torch.zeros_like(logw[:, :1])], 1)

    X = torch.tensor(emb, device=device)
    OPT = torch.tensor(opt_emb, device=device)

    def batches(examples, shuffle):
        groups: dict[int, list] = {}
        for ex in examples:
            groups.setdefault(len(ex[1]), []).append(ex)
        out = []
        for g in groups.values():
            idx = rng.permutation(len(g)) if shuffle else np.arange(len(g))
            for b in range(0, len(g), r["batch"]):
                sel = [g[j] for j in idx[b:b + r["batch"]]]
                out.append((torch.tensor([s[0] for s in sel], device=device),
                            torch.tensor([s[1] for s in sel], device=device),
                            torch.tensor([s[2] for s in sel], device=device)))
        if shuffle:
            rng.shuffle(out)
        return out

    def remap(examples, idx):
        return [(int(idx[i]), o, t) for i, o, t in examples]

    with_none = len(names) > 2
    val_b = batches(remap(_layouts(y[val_i], len(names), ca, np.random.default_rng(10_000 + seed), with_none, r),
                          val_i), False)
    net = Net(emb.shape[1]).to(device)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=r["lr"],
                            weight_decay=r["weight_decay"])

    @torch.no_grad()
    def val_loss():
        tot = n = 0
        for ti, o, t in val_b:
            tot += F.cross_entropy(net(X[ti], o, OPT), t, reduction="sum").item()
            n += len(t)
        return tot / n

    @torch.no_grad()
    def val_accuracy():
        hit = n = 0
        chance = 0.0
        for ti, o, t in val_b:
            k = o.shape[1]
            m = t < k
            if m.any():
                hit += (net(X[ti[m]], o[m], OPT)[:, :k].argmax(1) == t[m]).sum().item()
                n += int(m.sum())
                chance += m.sum().item() / k
        return hit / n, chance / n

    tracker = None
    if monitor:
        try:
            from ndt import HighFrequencyTracker
        except ImportError as e:
            raise ImportError("monitor=True needs ndt: pip install zeropoint[monitor]") from e
        tracker = HighFrequencyTracker(net, layers=[net.f[2]], layer_names=["text_map"], sampling_frequency=20,
                                       enable_jump_detection=False)

    best, bad, state, step, history = float("inf"), 0, None, 0, []
    for ep in range(r["epochs"]):
        net.train()
        tot = n = 0
        examples = remap(_layouts(y[tr_i], len(names), ca, rng, with_none and ep >= r["warmup"], r), tr_i)
        for ti, o, t in batches(examples, True):
            loss = F.cross_entropy(net(X[ti], o, OPT), t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if tracker is not None:
                tracker.log(step, loss.item())
            step += 1
            tot += loss.item() * len(t)
            n += len(t)
        net.eval()
        vl = val_loss()
        history.append({"epoch": ep, "train_loss": tot / n, "val_loss": vl})
        log(f"  epoch {ep:2d}  train {tot / n:.4f}  validation {vl:.4f}")
        if ep < r["warmup"]:
            continue  # the loss changes meaning once "none" examples arrive
        if vl < best - 1e-4:
            best, bad = vl, 0
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= r["patience"] and ep + 1 >= r["min_epochs"]:
                break
    net.load_state_dict(state)
    net.eval()

    participation = None
    if tracker is not None:
        dims = tracker.get_results().get("text_map")
        tracker.close()
        if dims is not None and len(dims):
            participation = [float(v) for v in dims["participation_ratio"]]
            floor = min(3, len(names) - 1)
            if participation[-1] < floor:
                warnings.warn(f"The learned space uses fewer than {floor} effective dimensions (ndt). "
                              "Possible collapse.")

    grid = np.linspace(-1.5, 1.5, 61)
    losses = []
    for lt in grid:
        net.log_t.data = torch.tensor(float(lt), device=device)
        losses.append(val_loss())
    net.log_t.data = torch.tensor(float(grid[int(np.argmin(losses))]), device=device)
    acc, chance = val_accuracy()
    log(f"Validation accuracy {acc:.3f} (chance {chance:.3f}), temperature {math.exp(net.log_t.item()):.3f}")
    if acc < chance + r["learn_margin"]:
        raise DidNotLearn(f"Validation accuracy {acc:.3f} does not beat chance {chance:.3f} by {r['learn_margin']}.")

    means, prec, q50, q99 = _edge(emb, y, len(names), seed + 5)
    weights = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items() if k != "log_t"}
    weights.update({"option_emb": opt_emb, f"edge.{precision}.means": means, f"edge.{precision}.precision": prec})
    config = {
        "format": FORMAT,
        "zeropoint_version": __version__,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "labels": names,
        "descriptions": desc,
        "catch_all": sorted(catch_all),
        "encoder": spec.to_dict(),
        "temperature": float(math.exp(net.log_t.item())),
        "kappa": r["kappa"],
        "edge": {precision: {"q50": q50, "q99": q99}},
        "recipe": r,
        "training": {"examples": len(texts), "seed": seed, "epochs": len(history),
                     "validation_accuracy": acc, "chance": chance, "history": history,
                     "participation_ratio": participation},
    }
    zp = ZeroPoint(weights, config)
    if precision != "int8" and edge_int8:
        log("Measuring the edge of the data with the 8-bit encoder as well (the default at inference)...")
        add_edge(zp, texts, labels, "int8", seed=seed)
    return zp


def add_edge(zp, texts: Sequence[str], labels: Sequence[str], precision: str,
             embeddings: np.ndarray | None = None, seed: int = 0) -> None:
    """Measure the edge of the data again with the encoder at another precision, and store it in `zp`.

    The edge is a distance in the encoder's own space, so it shifts a little between fp32 and int8. With an
    edge per precision, the model uses the one that matches the encoder it runs with. Needs no PyTorch.
    """
    spec = EncoderSpec(**{**zp.config["encoder"], "precision": precision})
    if embeddings is None:
        embeddings = get_encoder(spec).encode(list(texts), "query")
    y = np.array([zp.labels.index(lab) for lab in labels])
    means, prec, q50, q99 = _edge(np.asarray(embeddings, dtype=np.float32), y, len(zp.labels), seed + 5)
    zp.weights[f"edge.{precision}.means"], zp.weights[f"edge.{precision}.precision"] = means, prec
    zp.config["edge"][precision] = {"q50": q50, "q99": q99}
