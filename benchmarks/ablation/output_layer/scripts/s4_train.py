"""Stage 4: train the head variants on frozen features. python -I s4_train.py ENC [VARIANT ...]

Every run (variant x seed) writes cache/preds/<ENC>/<V>_s<seed>.npz and is skipped if that file exists.
See results/PROTOCOL.md for the definitions; this file implements them.
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from common import CACHE, LONG, SEEDS, TESTS, VARIANTS, jdump

torch.set_num_threads(1)   # small ops: one thread each, E1 and E2 trained in parallel
ENC = sys.argv[1]
VARS = sys.argv[2:] or VARIANTS
SMOKE = os.environ.get("SMOKE") == "1"   # code test only: dev_stop as train, 2 epochs, separate folder
OUT = os.path.join(CACHE, "preds_smoke" if SMOKE else "preds", ENC)
os.makedirs(OUT, exist_ok=True)
EVAL_SPLITS = ["dev_stop", "dev_cal"] + TESTS + LONG

LR, LR_SCALARS, BATCH, MAX_EPOCHS, PATIENCE = 1e-3, 1e-2, 256, 40, 4
if SMOKE:
    MAX_EPOCHS = 2


def load(split):
    parts = sorted(glob.glob(os.path.join(CACHE, ENC, f"{split}.part*.npz")),
                   key=lambda p: int(p.rsplit("part", 1)[1].split(".")[0]))
    if not parts:
        raise FileNotFoundError(f"no cache for {ENC}/{split}")
    Xs, offs, cls, gold = [], [0], [], []
    for p in parts:
        z = np.load(p)
        Xs.append(z["X"])
        offs += list(z["off"][1:] + offs[-1])
        cls.append(z["cls"])
        gold.append(z["gold"])
    off = np.array(offs, np.int64)
    k = np.diff(off)
    seg = np.repeat(np.arange(len(k)), k)
    posn = np.arange(off[-1]) - off[seg]
    return {"X": torch.from_numpy(np.concatenate(Xs)), "off": off, "k": k, "seg": seg, "pos": posn,
            "cls": torch.from_numpy(np.concatenate(cls)), "gold": np.concatenate(gold)}


class Head(torch.nn.Module):
    def __init__(self, v, H, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.v = v
        self.w = torch.nn.Parameter(torch.randn(H, generator=g) * 0.02)
        c0 = {"V1": np.log(np.expm1(1.0)), "V2": np.log(np.expm1(1.0)), "V5": 1.0}.get(v, 0.0)
        self.c = torch.nn.Parameter(torch.tensor(float(c0)))
        self.log_t = torch.nn.Parameter(torch.zeros(()))
        self.b = torch.nn.Parameter(torch.zeros(()))          # none logit (V3: offset d of r)
        if v == "V3":
            self.u = torch.nn.Parameter(torch.randn(H, generator=g) * 0.02)

    def logits(self, X, cls, seg, pos, n, kmax, exact_zero=False):
        """(n, kmax + 1) log-weights, 'none' last (V4: no none column, returns (n, kmax)). Padding = -inf."""
        z = X @ self.w + self.c
        T = self.log_t.exp()
        v = self.v
        if v in ("V1", "V5"):
            a = F.softplus(z) if v == "V1" else F.relu(z)
            x = (a / T).clamp(1e-6, 60)
            lw = x + torch.log(-torch.expm1(-x))
            if exact_zero:
                lw = torch.where(a > 0, lw, torch.full_like(lw, -float("inf")))
        elif v == "V2":
            lw = F.softplus(z) / T
        else:  # V3, V4
            lw = z / T
        L = torch.full((n, kmax), -float("inf"))
        L = L.index_put((torch.from_numpy(seg), torch.from_numpy(pos)), lw)
        if v == "V4":
            return L
        if v == "V3":
            nl = cls @ self.u + self.b
        else:
            nl = self.b.expand(n)
        return torch.cat([L, nl[:, None]], 1)


def batch_of(D, ids):
    """Sub-batch of lists `ids` (numpy array)."""
    k = D["k"][ids]
    seg = np.repeat(np.arange(len(ids)), k)
    pos = np.arange(int(k.sum())) - np.r_[0, np.cumsum(k)[:-1]][seg]
    rows = D["off"][ids][seg] + pos
    return D["X"][torch.from_numpy(rows)], D["cls"][torch.from_numpy(ids)], seg, pos, len(ids), int(k.max()), \
        D["gold"][ids], k


def nll(head, D, ids, exact_zero=False):
    X, cls, seg, pos, n, km, gold, k = batch_of(D, ids)
    L = head.logits(X, cls, seg, pos, n, km, exact_zero)
    t = np.where(gold >= 0, gold, km)
    return F.cross_entropy(L, torch.from_numpy(t))


def eval_nll(head, D, ids):
    with torch.no_grad():
        return float(sum(nll(head, D, ids[i:i + 1024]).item() * len(ids[i:i + 1024])
                         for i in range(0, len(ids), 1024)) / len(ids))


def predict(head, D):
    """Per-list option probabilities (flat, canonical order) and p_none (V4: 1 - max option probability)."""
    ids = np.arange(len(D["k"]))
    P, PN = [], []
    with torch.no_grad():
        for s in range(0, len(ids), 512):
            b = ids[s:s + 512]
            X, cls, seg, pos, n, km, gold, k = batch_of(D, b)
            L = head.logits(X, cls, seg, pos, n, km, exact_zero=(head.v == "V5")).double()
            p = L.softmax(-1).numpy()
            for r in range(n):
                P.append(p[r, :k[r]])
                PN.append(1 - p[r, :k[r]].max() if head.v == "V4" else p[r, km])
    return np.concatenate(P), np.array(PN)


def calibrate(head, D):
    """Refit the two scalars (log temperature and the none level / offset; V4: temperature only) on dev_cal by
    likelihood, with the rest of the head frozen. Same step as zp1.calibrate."""
    ids = np.arange(len(D["k"])) if head.v != "V4" else np.where(D["gold"] >= 0)[0]
    params = [head.log_t] if head.v == "V4" else [head.log_t, head.b]
    for p in head.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.LBFGS(params, lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nll(head, D, ids)
        loss.backward()
        return loss
    before = eval_nll(head, D, ids)
    opt.step(closure)
    return {"dev_cal_nll_before": before, "dev_cal_nll_after": eval_nll(head, D, ids)}


def fit_threshold(P, off, gold):
    """V4: threshold on the max option probability that maximises correct decisions on dev_cal (ties: lowest)."""
    maxp = np.array([P[off[i]:off[i + 1]].max() for i in range(len(gold))])
    arg = np.array([P[off[i]:off[i + 1]].argmax() for i in range(len(gold))])
    cands = np.unique(np.r_[0.0, maxp, np.nextafter(1.0, 2)])
    best, best_th = -1, None
    for th in cands:
        dec = np.where(maxp >= th, arg, -1)
        acc = np.mean(dec == gold)
        if acc > best:
            best, best_th = acc, float(th)
    return best_th, float(best)


def run(v, seed, data):
    path = os.path.join(OUT, f"{v}_s{seed}.npz")
    if os.path.exists(path):
        return
    t0 = time.time()
    torch.manual_seed(seed)
    tr, ds = data["train"], data["dev_stop"]
    H = tr["X"].shape[1]
    head = Head(v, H, seed)
    tr_ids = np.arange(len(tr["k"])) if v != "V4" else np.where(tr["gold"] >= 0)[0]
    ds_ids = np.arange(len(ds["k"])) if v != "V4" else np.where(ds["gold"] >= 0)[0]
    scal = [head.log_t, head.b]
    rest = [p for n, p in head.named_parameters() if n not in ("log_t", "b")]
    opt = torch.optim.Adam([{"params": rest, "lr": LR}, {"params": scal, "lr": LR_SCALARS}])
    rng = np.random.default_rng(seed)
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(MAX_EPOCHS):
        perm = rng.permutation(tr_ids)
        for s in range(0, len(perm), BATCH):
            loss = nll(head, tr, perm[s:s + BATCH])
            opt.zero_grad()
            loss.backward()
            opt.step()
        dl = eval_nll(head, ds, ds_ids)
        hist.append(dl)
        if dl < best - 1e-5:
            best, bad = dl, 0
            best_state = {k: t.detach().clone() for k, t in head.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    head.load_state_dict(best_state)
    cal = calibrate(head, data["dev_cal"])
    out, meta = {}, {"variant": v, "seed": seed, "encoder": ENC, "epochs_run": len(hist), "dev_stop_nll": hist,
                     "best_dev_stop_nll": best, "calibration": cal,
                     "T": float(head.log_t.detach().exp()), "b_or_offset": float(head.b.detach())}
    for split in EVAL_SPLITS:
        P, PN = predict(head, data[split])
        out[f"P_{split}"] = P
        out[f"N_{split}"] = PN
    if v == "V4":
        th, acc = fit_threshold(out["P_dev_cal"], data["dev_cal"]["off"], data["dev_cal"]["gold"])
        meta["threshold"], meta["threshold_dev_cal_correct"] = th, acc
    if v == "V5":
        meta["share_options_exact_zero"] = {s: float(np.mean(out[f"P_{s}"] == 0)) for s in ["dev_cal"] + TESTS}
    meta["seconds"] = time.time() - t0
    tmp = path + ".tmp.npz"
    np.savez(tmp, meta=json.dumps(meta), **out)
    os.replace(tmp, path)
    print(f"[{ENC}] {v} seed {seed}: epochs {len(hist)} dev_stop nll {best:.4f} T {meta['T']:.4f} "
          f"{meta['seconds']:.0f}s", flush=True)


if not SMOKE and all(os.path.exists(os.path.join(OUT, f"{v}_s{s}.npz")) for v in VARS for s in SEEDS):
    print(f"[{ENC}] all runs already saved", flush=True)
    sys.exit(0)
data = {s: load(s) for s in EVAL_SPLITS}
data["train"] = data["dev_stop"] if SMOKE else load("train")
SEEDS = SEEDS[:1] if SMOKE else SEEDS
for v in VARS:
    for seed in SEEDS:
        run(v, seed, data)
print(f"[{ENC}] training done", flush=True)
