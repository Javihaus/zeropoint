"""Accuracy, AUROC and FPR at 95% recall for every system in out/, with 95% bootstrap intervals.

Definitions (1,000 cases with a right option, 1,000 without):
  Accuracy (choice)    on cases with a right option: the top-scored option is the right one
  Accuracy (default)   on all 2,000 cases: the system's own decision at its default settings is right
                       (on a case with no right option, only "none" is right)
  Accuracy @ TPR95     on all 2,000 cases, after setting the system's threshold so that 95% of the cases
                       with a right option are still acted on; below it, the answer is "none"
  AUROC                how well `conf` separates cases with a right option from cases without (1 perfect, 0.5 chance)
  FPR95                share of cases with no right option still acted on, at the threshold that keeps 95% of
                       the cases with a right option (lower is better)
Intervals: 2,000 bootstrap resamples of the 552 distinct texts (cases that share a text move together).
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = {c["id"]: c for c in json.load(open(os.path.join(HERE, "data.json")))}
B = 2000
RNG = np.random.default_rng(13)

NAMES = {
    "zeropoint": "ZeroPoint 0.1.0 (κ = 2, default)",
    "zeropoint_k025": "ZeroPoint 0.1.0 (κ = 0.25)",
    "laya": "Laya",
    "sr_names": "Semantic Router, option names only",
    "sr_fewshot": "Semantic Router, 10 examples per option",
    "nli_bart": "Zero-shot NLI, BART-large-MNLI",
    "nli_deberta": "Zero-shot NLI, DeBERTa-v3-base",
}


def threshold_at_tpr(conf_pos, tpr=0.95):
    """Largest threshold that keeps at least `tpr` of the positive cases (conf >= threshold)."""
    return np.quantile(conf_pos, 1 - tpr, method="lower")


def point(rows):
    ids = np.array([r["id"] for r in rows])
    pos = np.array([DATA[i]["expected_choice"] is not None for i in ids])
    conf = np.array([r["conf"] for r in rows], dtype=float)
    top = np.array([max(r["scores"], key=r["scores"].get) for r in rows], dtype=object)
    gold = np.array([DATA[i]["expected_choice"] for i in ids], dtype=object)
    default = np.array([r["choice"] for r in rows], dtype=object)
    return ids, pos, conf, top, gold, default


def metrics(pos, conf, top, gold, default):
    out = {}
    out["acc_choice"] = float(np.mean(top[pos] == gold[pos]))
    out["acc_default"] = float(np.mean([(d == g) if g is not None else (d is None) for d, g in zip(default, gold)]))
    th = threshold_at_tpr(conf[pos])
    act = conf >= th
    right = np.where(pos, act & (top == gold), ~act)
    out["acc_tpr95"] = float(np.mean(right))
    out["auroc"] = float(roc_auc_score(pos.astype(int), conf))
    out["fpr95"] = float(np.mean(conf[~pos] >= th))
    return out


def evaluate(rows, mask=None):
    ids, pos, conf, top, gold, default = point(rows)
    if mask is not None:
        keep = np.array([mask(DATA[i]) for i in ids])
        ids, pos, conf, top, gold, default = ids[keep], pos[keep], conf[keep], top[keep], gold[keep], default[keep]
    est = metrics(pos, conf, top, gold, default)
    texts = np.array([DATA[i]["text"] for i in ids])
    groups = defaultdict(list)
    for k, t in enumerate(texts):
        groups[t].append(k)
    keys = list(groups)
    boots = defaultdict(list)
    for _ in range(B):
        pick = RNG.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[j]] for j in pick])
        if pos[idx].all() or (~pos[idx]).all():
            continue
        for k, v in metrics(pos[idx], conf[idx], top[idx], gold[idx], default[idx]).items():
            boots[k].append(v)
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in boots.items()}
    est["n"] = int(len(ids))
    est["n_pos"] = int(pos.sum())
    est["ms_median"] = float(np.median([r["ms"] for r in rows]))
    return est, ci


def load(name):
    return json.load(open(os.path.join(HERE, "out", f"{name}.json")))


if __name__ == "__main__":
    names = sys.argv[1:] or [n for n in NAMES if os.path.exists(os.path.join(HERE, "out", f"{n}.json"))]
    results = {}
    for n in names:
        rows = load(n)
        res = {"all": evaluate(rows)}
        for ls in ("everyday", "support"):
            res[ls] = evaluate(rows, lambda c, ls=ls: c["landscape"] == ls)
        res["no_clinc_overlap"] = evaluate(rows, lambda c: c["id"] not in OVERLAP) if (OVERLAP := set(
            json.load(open(os.path.join(HERE, "clinc_overlap_ids.json"))))) else None
        results[n] = res
        e, ci = res["all"]
        print(f"{NAMES[n]:42s} acc(choice) {e['acc_choice']:.3f}  acc(default) {e['acc_default']:.3f}  "
              f"acc@TPR95 {e['acc_tpr95']:.3f}  AUROC {e['auroc']:.3f} [{ci['auroc'][0]:.3f},{ci['auroc'][1]:.3f}]  "
              f"FPR95 {e['fpr95']:.3f} [{ci['fpr95'][0]:.3f},{ci['fpr95'][1]:.3f}]  {e['ms_median']:.0f} ms")
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1)
