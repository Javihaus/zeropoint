"""How well the shipped models choose, and how well they say "none".

    pip install -e . datasets scikit-learn
    python benchmarks/quality.py

Four tests, each with an answerable side and a side where no listed option is right:

    tickets_far   support tickets vs real unrelated texts (IMDB reviews, newsgroup posts) of the same length
    tickets_made  support tickets vs tickets whose right queue was removed from the list
    clinc_real    CLINC150 requests vs CLINC150 out-of-scope requests, written by people
    clinc_made    CLINC150 requests vs the same requests with the right intent swapped for another

Score for "one of the options is right": 1 - p_none. Measures: AUROC (1 = perfect, 0.5 = coin toss) and the
false positive rate at 95% recall (FPR95: share of no-right-answer cases you would still act on if you keep
95% of the answerable ones; lower is better). The baseline is similarity with the same encoder: the highest
cosine between the text and an option.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data  # noqa: E402

from zeropoint import ZeroPoint  # noqa: E402
from zeropoint.encoder import EncoderSpec, get_encoder  # noqa: E402

CACHE = os.path.join(HERE, "..", ".cache")
OUT = os.path.join(HERE, "results")
BOOT = 1000


def fpr95(y, s):
    f, t, _ = roc_curve(y, s)
    return float(f[np.searchsorted(t, 0.95)])


def boot_ci(y, s, fn, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = [fn(y[i], s[i]) for i in (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
                                     for _ in range(BOOT))]
    return np.percentile(vals, [2.5, 97.5])


def embed(texts, encoder, precision, tag, kind="query"):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{tag}-{encoder}-{precision}.npy")
    if os.path.exists(path):
        e = np.load(path)
        if len(e) == len(texts):
            return e
    e = get_encoder(EncoderSpec.resolve(encoder, precision)).encode(list(texts), kind)
    np.save(path, e)
    return e


def main():
    os.makedirs(OUT, exist_ok=True)
    _, tt = data.tickets()
    _, tc = data.clinc150()
    tests = {**{k: ("support", v) for k, v in tt.items()}, **{k: ("everyday", v) for k, v in tc.items()}}
    option_text = {**{q: f"{q}: {d}" for q, d in data.QUEUES.items()}}
    rows, acc_rows = [], []
    for encoder in ["small", "base"]:
        for precision in ["int8", "fp32"]:
            models = {n: ZeroPoint.load(n, encoder=encoder, precision=precision) for n in ["support", "everyday"]}
            enc = get_encoder(EncoderSpec.resolve(encoder, precision))
            for test, (task, (ans, none)) in tests.items():
                zp = models[task]
                items = ans + none
                y = np.array([1] * len(ans) + [0] * len(none))
                emb = embed([t for t, _, _ in items], encoder, precision, f"test-{test}")
                opts = [o for _, o, _ in items]
                scores = {}
                for kappa, system in [(zp.config["kappa"], "zeropoint"), (0.0, "zeropoint_no_edge")]:
                    zp.kappa = kappa
                    dec = zp._decide_embedded(emb, opts)
                    scores[system] = np.array([1 - d.p_none for d in dec])
                    if system == "zeropoint" and test in ("tickets_far", "tickets_made", "clinc_real"):
                        hit = [max(d.probabilities, key=d.probabilities.get) == g for d, (_, _, g) in zip(dec, ans)]
                        acc_rows.append({"encoder": encoder, "precision": precision, "test": test,
                                         "system": "zeropoint", "accuracy": float(np.mean(hit))})
                zp.kappa = zp.config["kappa"]
                labels = sorted({o for o in sum(opts, [])})
                lab_emb = dict(zip(labels, enc.encode([option_text.get(o, o) for o in labels], "passage")))
                cos = [np.array([lab_emb[o] @ e for o in op]) for e, op in zip(emb, opts)]
                scores["similarity"] = np.array([c.max() for c in cos])
                if test in ("tickets_far", "tickets_made", "clinc_real"):
                    hit = [op[int(c[:len(op)].argmax())] == g for c, (_, op, g) in zip(cos, ans)]
                    acc_rows.append({"encoder": encoder, "precision": precision, "test": test,
                                     "system": "similarity", "accuracy": float(np.mean(hit))})
                for system, s in scores.items():
                    lo, hi = boot_ci(y, s, roc_auc_score)
                    rows.append({"encoder": encoder, "precision": precision, "test": test, "system": system,
                                 "auroc": roc_auc_score(y, s), "auroc_lo": lo, "auroc_hi": hi, "fpr95": fpr95(y, s),
                                 "n_answerable": len(ans), "n_none": len(none)})
                print(encoder, precision, test, {k: round(roc_auc_score(y, v), 3) for k, v in scores.items()})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "none_detection.csv"), index=False)
    pd.DataFrame(acc_rows).to_csv(os.path.join(OUT, "accuracy.csv"), index=False)


if __name__ == "__main__":
    main()
