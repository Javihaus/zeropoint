"""Stage 3: depth analysis of the released model, and package vs released 1.0c per-case outputs.
Output: results/depths.json"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import CACHE, DATA, LONG, RESULTS, TESTS, jdump

OUT = os.path.join(RESULTS, "depths.json")
if os.path.exists(OUT):
    sys.exit(0)
from zeropoint import ZeroPoint

zp = ZeroPoint()
rt = zp._rt
T, b = rt.temperature, rt.none_level
eb = float(np.exp(b))
RUNS = "/home/claude/zeropoint/training/runs/1.0c"


def load(split):
    parts = sorted(glob.glob(os.path.join(CACHE, "E1", f"{split}.part*.npz")),
                   key=lambda p: int(p.rsplit("part", 1)[1].split(".")[0]))
    D, offs, G = [], [0], []
    for p in parts:
        z = np.load(p)
        D.append(z["depth_rel"])
        offs += list(z["off"][1:] + offs[-1])
        G.append(z["gold"])
    return np.concatenate(D), np.array(offs), np.concatenate(G)


def summarise(depths, off, gold, which):
    sel = np.where(gold < 0)[0] if which == "none" else np.where(gold >= 0)[0]
    if len(sel) == 0:
        return None
    a_all, ratio, wrong_a = [], [], []
    for i in sel:
        a = depths[off[i]:off[i + 1]]
        x = np.clip(a / T, 1e-6, 60.0)            # the runtime's clip
        wsum = float(np.sum(np.expm1(x)))
        if which == "none":
            a_all.append(a)
            ratio.append(wsum / eb)
        else:
            g = gold[i]
            wrong = np.delete(a, g)
            wrong_a.append(wrong)
            ratio.append(float(np.sum(np.expm1(np.clip(wrong / T, 1e-6, 60.0)))) / eb)
    a = np.concatenate(a_all if which == "none" else wrong_a)
    r = np.array(ratio)
    k = np.diff(off)[sel]
    return {"cases": int(len(sel)), "options_per_list_mean": float(k.mean()),
            "depth_median": float(np.median(a)), "depth_p95": float(np.percentile(a, 95)),
            "depth_min": float(a.min()), "depth_max": float(a.max()),
            "share_depth_below_0.05": float(np.mean(a < 0.05)),
            "weight_per_option_median": float(np.median(np.expm1(np.clip(a / T, 1e-6, 60)))),
            "sum_weight_over_exp_b_median": float(np.median(r)), "sum_weight_over_exp_b_p95": float(np.percentile(r, 95)),
            "share_cases_sum_weight_gt_exp_b": float(np.mean(r > 1)),
            "mean_p_none_if_only_these_options": float(np.mean(1 / (1 + r)))}


res = {"T": T, "b": b, "exp_b": eb,
       "note": "Depths from the released ONNX in the PyPI package, cached through zeropoint._runtime and checked "
               "bit for bit against ZeroPoint.decide. 'none' = cases without a right option, all their options. "
               "'with_right_option' = the wrong options of cases with a right option. sum_weight = "
               "sum(exp(a/T) - 1) with the runtime clip; > exp(b) means the options together outweigh none.",
       "per_test": {}, "per_long": {}, "package_vs_1.0c_checkpoint": {}}
for t in TESTS + LONG:
    d, off, g = load(t)
    key = "per_test" if t in TESTS else "per_long"
    res[key][t] = {"none": summarise(d, off, g, "none"), "with_right_option": summarise(d, off, g, "pos")}
# pooled
for name, group in [("pooled_tests", TESTS), ("pooled_long", LONG)]:
    ds, offs, gs, base = [], [0], [], 0
    for t in group:
        d, off, g = load(t)
        ds.append(d)
        offs += list(off[1:] + offs[-1])
        gs.append(g)
    d, off, g = np.concatenate(ds), np.array(offs), np.concatenate(gs)
    res[name] = {"none": summarise(d, off, g, "none"), "with_right_option": summarise(d, off, g, "pos")}

# package (int8 vocabulary, ONNX, CPU) vs the released per-case outputs of the 1.0c checkpoint (torch, GPU)
for t in TESTS:
    p = os.path.join(RUNS, f"s0_cases_{t}.json")
    if not os.path.exists(p):
        continue
    ref = json.load(open(p, encoding="utf-8"))
    items = json.load(open(os.path.join(DATA, "tests", t + ".json"), encoding="utf-8"))
    d, off, g = load(t)
    from common import canonical_order
    diffs, agree = [], []
    for i, (it, r) in enumerate(zip(items, ref)):
        canon = [it["options"][j] for j in canonical_order(it["options"])]
        pr = rt.distribution(d[off[i]:off[i + 1]])
        diffs.append(abs(pr[-1] - r["p_none"]))
        best = max(range(len(canon)), key=lambda j: (pr[j], -j))
        dec = canon[best] if pr[best] >= pr[-1] else None
        agree.append(dec == r["decision"])
    diffs = np.array(diffs)
    res["package_vs_1.0c_checkpoint"][t] = {"cases": len(items), "p_none_abs_diff_max": float(diffs.max()),
                                           "p_none_abs_diff_median": float(np.median(diffs)),
                                           "decision_agreement": float(np.mean(agree))}
jdump(res, OUT)
print("depths done", flush=True)
