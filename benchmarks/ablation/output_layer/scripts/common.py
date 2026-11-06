"""Shared paths, list construction (a faithful copy of zp1.py's build_data logic) and metrics.

Scripts are run with `python -I`, so each script inserts this folder into sys.path explicitly.
"""
import hashlib
import json
import os
import random
import unicodedata

import numpy as np


def _json_load(*args, **kw):
    with open(*args, **kw) as f:
        return json.load(f)


def _jsonl(*args, **kw):
    with open(*args, **kw) as f:
        return [json.loads(line) for line in f]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_zip")
CACHE = os.path.join(ROOT, "cache")
RESULTS = os.path.join(ROOT, "results")
TESTS = ["bench2000", "snips", "atis", "mtop_es", "hwu64", "clinc_real_indomain", "tickets_made", "tickets_far"]
LONG = ["clinc_20", "clinc_50", "clinc_100", "clinc_150", "mtop_es_20", "mtop_es_50"]
VARIANTS = ["V1", "V2", "V3", "V4", "V5"]
SEEDS = [0, 1, 2, 3, 4]
ENCODERS = ["E1", "E2"]

# zp1.py defaults (training/zp1.py argparse)
N_BASE, N_BROAD, SYNTH_REPEAT = 50000, 12000, 3
LONG_SHARE, HARD_SHARE, NONE_SHARE = 0.08, 0.5, 0.3
TEXT_LEN, OPT_LEN, CHUNK = 192, 16, 20


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def canonical_order(options):
    keys = [unicodedata.normalize("NFC", o) for o in options]
    if any(not k.strip() for k in keys):
        raise ValueError("empty option")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate options")
    return sorted(range(len(options)), key=lambda i: keys[i].encode("utf-8"))


def canon_case(text, opts, gold):
    opts = [str(o) for o in opts]
    canon = [opts[i] for i in canonical_order(opts)]
    return {"text": text, "opts": canon, "gold": gold}


# ---------------------------------------------------------------- copy of zp1.py list construction

NB = _json_load(os.path.join(DATA, "data", "neighbors.json"), encoding="utf-8")


def wrong_options(pool, lab, same, k, rng):
    k = min(k, len(same))
    near = [x for x in NB.get(pool, {}).get(lab, []) if x in same]
    if near and rng.random() < HARD_SHARE:
        h = min(len(near), max(1, k // 2))
        picked = list(rng.choice(near, h, replace=False))
        rest = [x for x in same if x not in picked]
        return picked + list(rng.choice(rest, k - h, replace=False)) if k > h else picked
    return list(rng.choice(same, k, replace=False))


def lists_from_rows(rows, label_key, pools, n, rng, cross=True):
    names = sorted(pools)
    out = []
    for i in rng.choice(len(rows), min(n, len(rows)), replace=False):
        r = rows[i]
        pool, lab = r["_pool"], r[label_key]
        same = [l for l in pools[pool] if l != lab]
        if not same:
            continue
        k = int(rng.integers(4, 21))
        if len(same) > 21 and rng.random() < LONG_SHARE:
            k = int(rng.integers(21, len(same) + 1))
        u = rng.random()
        if u < NONE_SHARE / 2 or (u < NONE_SHARE and not cross):
            opts, gold, kind = wrong_options(pool, lab, same, k, rng), None, "none_removed"
        elif u < NONE_SHARE:
            p2 = names[rng.integers(len(names))]
            other = [l for l in pools[p2] if l != lab]
            opts, gold, kind = list(rng.choice(other, min(k, len(other)), replace=False)), None, "none_other_pool"
        else:
            opts, gold, kind = wrong_options(pool, lab, same, k - 1, rng) + [lab], lab, "gold"
        c = canon_case(r["text"], opts, gold)
        c["kind"] = kind
        c["src"] = pool
        out.append(c)
    return out


def synth_lists(rows, pools, repeat, rng):
    out = []
    for r in rows:
        labs = pools[r["pool"]]
        for _ in range(repeat):
            hard = list(r["hard_negatives"])
            rest = [l for l in labs if l != r["label"] and l not in hard]
            k = int(rng.integers(max(4, len(hard) + 1), len(labs) + 1))
            gold = r["label"] if (r["label"] is not None and rng.random() >= NONE_SHARE) else None
            n_rest = k - len(hard) - (1 if gold else 0)
            opts = hard + list(rng.choice(rest, max(0, min(n_rest, len(rest))), replace=False))
            if gold:
                opts.append(gold)
            c = canon_case(r["text"], opts, gold)
            c["kind"] = "synth_gold" if gold else "synth_none"
            c["src"] = r["pool"]
            out.append(c)
    return out


def held_out(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 20 == 0


def build_data(seed=0):
    """Exactly zp1.build_data(seed) with default arguments (limit=0)."""
    rng = np.random.default_rng(seed)
    tr = _jsonl(os.path.join(DATA, "data", "train.jsonl"), encoding="utf-8")
    dv = _jsonl(os.path.join(DATA, "data", "dev.jsonl"), encoding="utf-8")
    cmap = _json_load(os.path.join(DATA, "data", "coarse_map.json"), encoding="utf-8")
    fine_pools, broad_pools = {}, {}
    for r in tr:
        fine_pools.setdefault(r["source"], set()).add(r["label"])
    fine_pools = {k: sorted(v) for k, v in fine_pools.items()}
    for src, m in cmap.items():
        broad_pools[src + "#broad"] = sorted(set(m.values()))
    for rows in (tr, dv):
        for r in rows:
            r["_pool"] = r["source"]
            r["broad"] = cmap[r["source"]][r["label"]]
    tr = [r for r in tr if r["label"] in fine_pools[r["source"]]]
    dv = [r for r in dv if r["label"] in fine_pools.get(r["source"], [])]
    tr_b = [{**r, "_pool": r["source"] + "#broad"} for r in tr]
    dv_b = [{**r, "_pool": r["source"] + "#broad"} for r in dv]
    synth_pools = {k: v["labels"] for k, v in
                   _json_load(os.path.join(DATA, "data", "synth_pools.json"), encoding="utf-8").items()
                   if not k.startswith("_")}
    syn = _jsonl(os.path.join(DATA, "data", "synth.jsonl"), encoding="utf-8")
    syn_tr = [r for r in syn if not held_out(r["text"])]
    syn_dv = [r for r in syn if held_out(r["text"])]
    train = (lists_from_rows(tr, "label", fine_pools, N_BASE, rng)
             + lists_from_rows(tr_b, "broad", broad_pools, N_BROAD, rng)
             + synth_lists(syn_tr, synth_pools, SYNTH_REPEAT, rng))
    drng = np.random.default_rng(1000)
    dev = (lists_from_rows(dv, "label", fine_pools, 1500, drng)
           + lists_from_rows(dv_b, "broad", broad_pools, 500, drng)
           + synth_lists(syn_dv, synth_pools, 1, drng))
    random.Random(seed).shuffle(train)
    half = lambda e: int(hashlib.sha256(e["text"].encode("utf-8")).hexdigest(), 16) % 2
    dev_stop = [e for e in dev if half(e) == 0]
    dev_cal = [e for e in dev if half(e) == 1]
    return train, dev_stop, dev_cal


def load_test(name):
    sub = "tests" if name in TESTS else "tests_long"
    items = _json_load(os.path.join(DATA, sub, name + ".json"), encoding="utf-8")
    out = []
    for it in items:
        c = canon_case(it["text"], it["options"], it["gold"])
        out.append(c)
    return out


def load_split(split):
    """split: 'train', 'dev_stop', 'dev_cal', or a test name. Lists in canonical order."""
    if split in ("train", "dev_stop", "dev_cal", "full_train"):
        return _json_load(os.path.join(CACHE, f"lists_{split}.json"), encoding="utf-8")
    return load_test(split)


# ---------------------------------------------------------------- metrics

def auroc(y, s):
    """AUROC of score s for label y (1 = positive), ties counted as one half (rank average)."""
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return None
    from scipy.stats import rankdata
    ranks = rankdata(s, method="average")
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y, s):
    """Average precision (step-wise, as sklearn.metrics.average_precision_score), positives = y."""
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    if y.sum() == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    # thresholds at distinct score values
    last = np.r_[np.where(np.diff(ss) != 0)[0], len(ss) - 1]
    tp, fp = tp[last], fp[last]
    prec = tp / (tp + fp)
    rec = tp / y.sum()
    rec_prev = np.r_[0.0, rec[:-1]]
    return float(np.sum((rec - rec_prev) * prec))


def metrics(gold_idx, P, pnone, decision, conf, acc_score=None):
    """All pre-registered measures for one set of cases.

    gold_idx  int array, index of the right option or -1 when none is right
    P         list of arrays, probability of each option (canonical order)
    pnone     array, the abstention score (p_none; for V4: 1 - max option probability)
    decision  int array, chosen option index or -1 for abstention
    conf      array, probability attached to the decision
    """
    gold_idx = np.asarray(gold_idx)
    decision = np.asarray(decision)
    pnone = np.asarray(pnone, float)
    conf = np.asarray(conf, float)
    pos = gold_idx >= 0
    neg = ~pos
    right = decision == gold_idx
    out = {"n": int(len(gold_idx)), "n_none": int(neg.sum())}
    out["correct_decisions"] = float(right.mean())
    # ranking: right option strictly above every other option
    rk = []
    for g, p in zip(gold_idx, P):
        if g >= 0:
            rk.append(bool(np.all(np.delete(p, g) < p[g])))
    out["ranking"] = float(np.mean(rk)) if rk else None
    out["auroc_none"] = auroc(neg, pnone)
    out["auprc_none"] = auprc(neg, pnone)
    if pos.any() and neg.any():
        cin = 1 - pnone
        th = np.quantile(cin[pos], 0.05, method="lower")
        out["fpr95"] = float(np.mean(cin[neg] >= th))
    else:
        out["fpr95"] = None
    maxp = np.array([p.max() for p in P])
    if neg.any():
        out["silent_mistakes_09"] = float(np.mean((decision[neg] >= 0) & (maxp[neg] >= 0.9)))
    else:
        out["silent_mistakes_09"] = None
    out["false_rejections"] = float(np.mean(decision[pos] < 0)) if pos.any() else None
    # ECE of the final decision, 10 equal-width bins (zp1.py ece)
    tot = 0.0
    for lo in np.linspace(0, 1, 10, endpoint=False):
        m = (conf > lo) & (conf <= lo + 0.1)
        if m.any():
            tot += m.mean() * abs(conf[m].mean() - right[m].mean())
    out["ece"] = float(tot)
    # risk-coverage: sort by confidence of the decision (desc, stable), risk = error rate among the top k
    order = np.argsort(-conf, kind="mergesort")
    err = (~right[order]).astype(float)
    out["aurc"] = float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))
    return out
