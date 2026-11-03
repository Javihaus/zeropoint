"""Measure the shipped model files on the test sets, and compare case by case with the training checkpoint.

    python tools/measure_model.py --tests tests_dir/ [--long tests_long_dir/] [--reference cases_dir/] --out results.json

tests_dir holds one JSON file per test: a list of {"text", "options", "gold"} (gold is null when no option is right).
--reference points at the per-case outputs of the training checkpoint (<tag>_cases_<test>.json); the report then
says how many decisions changed because of the fp16 / int8 storage.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from zeropoint._runtime import Runtime, canonical_order  # noqa: E402


def decide(rt, text, options):
    order = canonical_order(options)
    canon = [options[i] for i in order]
    d, reading = rt.depths(text, canon)
    p = rt.distribution(d)
    best = max(range(len(canon)), key=lambda j: (p[j], -j))
    probs = {canon[j]: float(p[j]) for j in range(len(canon))}
    return {"p": {o: probs[o] for o in options}, "p_none": float(p[-1]),
            "decision": canon[best] if p[best] >= p[-1] else None}


def scores(items, rows):
    from sklearn.metrics import roc_auc_score
    y = np.array([it["gold"] is not None for it in items], int)
    conf_any = np.array([1 - r["p_none"] for r in rows])
    out = {"n": len(items),
           "choice": float(np.mean([max(r["p"], key=r["p"].get) == it["gold"] for r, it in zip(rows, items)
                                    if it["gold"]])),
           "decision": float(np.mean([r["decision"] == it["gold"] for r, it in zip(rows, items)]))}
    if y.min() != y.max():
        th = np.quantile(conf_any[y == 1], 0.05, method="lower")
        out["auroc"] = float(roc_auc_score(y, conf_any))
        out["fpr95"] = float(np.mean(conf_any[y == 0] >= th))
        neg = [r for r, it in zip(rows, items) if it["gold"] is None]
        out["false_admission_090"] = float(np.mean([r["decision"] is not None and max(r["p"].values()) >= 0.9
                                                    for r in neg]))
    conf = np.array([max(max(r["p"].values()), r["p_none"]) for r in rows])
    right = np.array([r["decision"] == it["gold"] for r, it in zip(rows, items)], float)
    sure = conf >= 0.9
    out["sure_share"] = float(sure.mean())
    out["wrong_when_sure"] = float(1 - right[sure].mean()) if sure.any() else None
    ece = 0.0
    for lo in np.linspace(0, 1, 10, endpoint=False):
        m = (conf > lo) & (conf <= lo + 0.1)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - right[m].mean())
    out["ece"] = float(ece)
    ms = [r["ms"] for r in rows]
    out["ms_median"], out["ms_p99"] = float(np.median(ms)), float(np.percentile(ms, 99))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", required=True)
    ap.add_argument("--long")
    ap.add_argument("--reference")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rt = Runtime(threads=args.threads)
    report = {"threads": args.threads, "tests": {}, "long": {}, "against_reference": {}}
    sets = [("tests", f) for f in sorted(glob.glob(os.path.join(args.tests, "*.json")))]
    if args.long:
        sets += [("long", f) for f in sorted(glob.glob(os.path.join(args.long, "*.json")))]
    for kind, f in sets:
        name = os.path.basename(f)[:-5]
        items = json.load(open(f, encoding="utf-8"))
        rows = []
        for it in items:
            t0 = time.perf_counter()
            r = decide(rt, it["text"], it["options"])
            r["ms"] = (time.perf_counter() - t0) * 1000
            rows.append(r)
        report[kind][name] = scores(items, rows)
        if args.reference and os.path.exists(os.path.join(args.reference, f"s0_cases_{name}.json")):
            ref = json.load(open(os.path.join(args.reference, f"s0_cases_{name}.json")))
            diff = max(max(abs(a["p"][o] - b["p"][o]) for o in a["p"]) for a, b in zip(rows, ref))
            report["against_reference"][name] = {
                "decision_changes": int(sum(a["decision"] != b["decision"] for a, b in zip(rows, ref))),
                "max_abs_diff_p": float(diff), "cases": len(rows)}
        print(kind, name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                      for k, v in report[kind][name].items()}),
              report["against_reference"].get(name, ""), flush=True)
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
