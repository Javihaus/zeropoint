"""Stage 5: measures per run, mean/sd over seeds, seed-averaged paired bootstrap with Holm.
Outputs: results/metrics.json, results/paired.json"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import CACHE, ENCODERS, LONG, RESULTS, SEEDS, TESTS, VARIANTS, auroc, jdump, metrics

SMOKE = os.environ.get("SMOKE") == "1"    # code test only: smoke predictions, seed 0, outputs in cache/
PREDS = "preds_smoke" if SMOKE else "preds"
if SMOKE:
    ENCODERS, SEEDS = ["E1"], [0]
    M_OUT, P_OUT = os.path.join(CACHE, "smoke_metrics.json"), os.path.join(CACHE, "smoke_paired.json")
else:
    M_OUT, P_OUT = os.path.join(RESULTS, "metrics.json"), os.path.join(RESULTS, "paired.json")
SPLITS = ["dev_cal"] + TESTS + LONG
MEASURES = ["correct_decisions", "ranking", "auroc_none", "auprc_none", "fpr95", "silent_mistakes_09",
            "false_rejections", "ece", "aurc"]


def struct(enc, split):
    parts = sorted(glob.glob(os.path.join(CACHE, enc, f"{split}.part*.npz")),
                   key=lambda p: int(p.rsplit("part", 1)[1].split(".")[0]))
    offs, G, D = [0], [], []
    for p in parts:
        z = np.load(p)
        offs += list(z["off"][1:] + offs[-1])
        G.append(z["gold"])
        if "depth_rel" in z.files:
            D.append(z["depth_rel"])
    return np.array(offs), np.concatenate(G), (np.concatenate(D) if D else None)


S = {e: {s: struct(e, s) for s in SPLITS} for e in ENCODERS}


def split_lists(P, off):
    return [P[off[i]:off[i + 1]] for i in range(len(off) - 1)]


def decide(v, Pl, pn, th=None):
    best = np.array([int(np.argmax(p)) for p in Pl])       # argmax: first of ties (canonical order)
    pb = np.array([p.max() for p in Pl])
    if v == "V4":
        act = pb >= th
        return np.where(act, best, -1), np.where(act, pb, 1 - pb)
    act = pb >= pn
    return np.where(act, best, -1), np.maximum(pb, pn)


def fit_threshold(Pl, gold):
    maxp = np.array([p.max() for p in Pl])
    arg = np.array([int(p.argmax()) for p in Pl])
    best, best_th = -1, None
    for th in np.unique(np.r_[0.0, maxp, np.nextafter(1.0, 2)]):
        acc = np.mean(np.where(maxp >= th, arg, -1) == gold)
        if acc > best:
            best, best_th = acc, float(th)
    return best_th


def case_arrays(enc, v, P, PN, th):
    """Per split: dict with option prob lists, pnone, decision, conf, gold."""
    out = {}
    for s in SPLITS:
        off, gold, _ = S[enc][s]
        Pl = split_lists(P[s], off)
        dec, conf = decide(v, Pl, PN[s], th)
        out[s] = {"P": Pl, "pn": PN[s], "dec": dec, "conf": conf, "gold": gold}
    return out


def measure(ca):
    res = {s: metrics(ca[s]["gold"], ca[s]["P"], ca[s]["pn"], ca[s]["dec"], ca[s]["conf"]) for s in TESTS + LONG}
    for name, grp in [("pooled_tests", TESTS), ("pooled_long", LONG)]:
        cat = lambda k: np.concatenate([ca[s][k] for s in grp])
        res[name] = metrics(cat("gold"), sum([ca[s]["P"] for s in grp], []), cat("pn"), cat("dec"), cat("conf"))
    res["macro_tests"] = {m: float(np.mean([res[t][m] for t in TESTS if res[t][m] is not None]))
                          for m in MEASURES}
    return res


def released():
    """Reference: the released head on E1 (package arithmetic, runtime clip)."""
    from zeropoint import ZeroPoint
    rt = ZeroPoint()._rt
    P, PN = {}, {}
    for s in SPLITS:
        off, gold, d = S["E1"][s]
        ps, pn = [], []
        for i in range(len(gold)):
            p = rt.distribution(d[off[i]:off[i + 1]])
            ps.append(p[:-1])
            pn.append(p[-1])
        P[s], PN[s] = np.concatenate(ps), np.array(pn)
    return P, PN


if not os.path.exists(M_OUT):
    allm = {"per_run": {}, "summary": {}, "seed_averaged": {}, "run_meta": {}}
    Pr, PNr = released()
    allm["released_E1"] = measure(case_arrays("E1", "V1", Pr, PNr, None))
    for enc in ENCODERS:
        for v in VARIANTS:
            runs = []
            for seed in SEEDS:
                z = np.load(os.path.join(CACHE, PREDS, enc, f"{v}_s{seed}.npz"))
                meta = json.loads(str(z["meta"]))
                P = {s: z[f"P_{s}"] for s in SPLITS}
                PN = {s: z[f"N_{s}"] for s in SPLITS}
                m = measure(case_arrays(enc, v, P, PN, meta.get("threshold")))
                allm["per_run"][f"{enc}/{v}/s{seed}"] = m
                allm["run_meta"][f"{enc}/{v}/s{seed}"] = {k: meta[k] for k in meta if k != "dev_stop_nll"}
                runs.append((P, PN))
            summ = {}
            for scope in ["pooled_tests", "pooled_long", "macro_tests"] + TESTS + LONG:
                summ[scope] = {}
                for meas in MEASURES:
                    vals = [allm["per_run"][f"{enc}/{v}/s{s}"][scope][meas] for s in SEEDS]
                    vals = [x for x in vals if x is not None]
                    summ[scope][meas] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None} if vals else None
            allm["summary"][f"{enc}/{v}"] = summ
            # seed-averaged probabilities
            Pa = {s: np.mean([r[0][s] for r in runs], axis=0) for s in SPLITS}
            if v == "V4":
                PNa = {s: 1 - np.array([p.max() for p in split_lists(Pa[s], S[enc][s][0])]) for s in SPLITS}
                th = fit_threshold(split_lists(Pa["dev_cal"], S[enc]["dev_cal"][0]), S[enc]["dev_cal"][1])
            else:
                PNa = {s: np.mean([r[1][s] for r in runs], axis=0) for s in SPLITS}
                th = None
            ca = case_arrays(enc, v, Pa, PNa, th)
            allm["seed_averaged"][f"{enc}/{v}"] = measure(ca)
            if v == "V4":
                allm["seed_averaged"][f"{enc}/{v}"]["threshold"] = th
            np.savez(os.path.join(CACHE, PREDS, enc, f"{v}_avg_cases.npz"),
                     dec=np.concatenate([ca[t]["dec"] for t in TESTS]),
                     gold=np.concatenate([ca[t]["gold"] for t in TESTS]),
                     pn=np.concatenate([ca[t]["pn"] for t in TESTS]),
                     strata=np.concatenate([np.full(len(ca[t]["gold"]), i) for i, t in enumerate(TESTS)]))
    jdump(allm, M_OUT)
    print("metrics done", flush=True)

# ---------------------------------------------------------------- paired bootstrap
B = 200 if SMOKE else 2000


def stats_of(dec, gold, pn, idx):
    d, g, p = dec[idx], gold[idx], pn[idx]
    neg = g < 0
    cin = 1 - p
    th = np.quantile(cin[~neg], 0.05, method="lower")
    return np.array([np.mean(d == g), auroc(neg, p), np.mean(cin[neg] >= th)])


cases = {(e, v): np.load(os.path.join(CACHE, PREDS, e, f"{v}_avg_cases.npz")) for e in ENCODERS for v in VARIANTS}
strata = cases[("E1", "V1")]["strata"]
rng = np.random.default_rng(13)
groups = [np.where(strata == i)[0] for i in range(len(TESTS))]
boots = [np.concatenate([rng.choice(g, len(g), replace=True) for g in groups]) for _ in range(B)]
full = np.arange(len(strata))
names = ["correct_decisions", "auroc_none", "fpr95"]
comps = []
for enc in ENCODERS:
    base = cases[(enc, "V1")]
    sb = stats_of(base["dec"], base["gold"], base["pn"], full)
    bb = np.array([stats_of(base["dec"], base["gold"], base["pn"], ix) for ix in boots])
    for v in ["V2", "V3", "V4", "V5"]:
        c = cases[(enc, v)]
        assert np.array_equal(c["gold"], base["gold"])
        so = stats_of(c["dec"], c["gold"], c["pn"], full)
        bo = np.array([stats_of(c["dec"], c["gold"], c["pn"], ix) for ix in boots])
        diff = bb - bo
        for j, m in enumerate(names):
            dj = diff[:, j]
            p = min(1.0, (1 + 2 * min(np.sum(dj <= 0), np.sum(dj >= 0))) / (B + 1))
            comps.append({"encoder": enc, "contrast": f"V1-{v}", "measure": m, "V1": float(sb[j]),
                          "other": float(so[j]), "diff": float(sb[j] - so[j]),
                          "ci95": [float(np.percentile(dj, 2.5)), float(np.percentile(dj, 97.5))], "p": float(p)})
# Holm
order = np.argsort([c["p"] for c in comps], kind="mergesort")
m = len(comps)
running = 0.0
for rank, i in enumerate(order):
    adj = min(1.0, (m - rank) * comps[i]["p"])
    running = max(running, adj)
    comps[i]["p_holm"] = running
for c in comps:
    good = c["diff"] > 0 if c["measure"] != "fpr95" else c["diff"] < 0
    ci_side = (c["ci95"][0] > 0) if c["measure"] != "fpr95" else (c["ci95"][1] < 0)
    ci_other = (c["ci95"][1] < 0) if c["measure"] != "fpr95" else (c["ci95"][0] > 0)
    c["reading"] = ("V1 better" if (c["p_holm"] < 0.05 and ci_side) else
                    "V1 worse" if (c["p_holm"] < 0.05 and ci_other) else "no significant difference")
jdump({"B": B, "rng_seed": 13, "cases": int(len(strata)), "stratified_by": TESTS,
       "note": "Seed-averaged per-case probabilities; pooled over the eight tests; Holm over all comparisons.",
       "comparisons": comps}, P_OUT)
print("paired done", flush=True)
