"""Stage 1: rebuild the full 1.0c training and dev lists exactly as zp1.py does (seed 0), then draw the fixed
training subset. Outputs: cache/lists_full_train.json, cache/lists_train.json, cache/lists_dev_stop.json,
cache/lists_dev_cal.json, results/lists_info.json."""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import CACHE, RESULTS, build_data, jdump

N_SUB = int(os.environ.get("N_SUB", "0"))   # set from the protocol

full_p = os.path.join(CACHE, "lists_full_train.json")
if not os.path.exists(full_p):
    train, dev_stop, dev_cal = build_data(0)
    jdump(train, full_p)
    jdump(dev_stop, os.path.join(CACHE, "lists_dev_stop.json"))
    jdump(dev_cal, os.path.join(CACHE, "lists_dev_cal.json"))
    print("full lists built", len(train), len(dev_stop), len(dev_cal), flush=True)

sub_p = os.path.join(CACHE, "lists_train.json")
if N_SUB and not os.path.exists(sub_p):
    import json
    train = json.load(open(full_p, encoding="utf-8"))
    rng = np.random.default_rng(13)
    idx = sorted(rng.choice(len(train), N_SUB, replace=False).tolist())
    sub = [train[i] for i in idx]
    jdump(sub, sub_p)

    def desc(lists):
        k = np.array([len(e["opts"]) for e in lists])
        return {"lists": len(lists), "none_share": float(np.mean([e["gold"] is None for e in lists])),
                "long_share_gt20": float(np.mean(k > 20)), "options_mean": float(k.mean()),
                "options_max": int(k.max()), "kinds": dict(collections.Counter(e["kind"] for e in lists))}
    dev_stop = json.load(open(os.path.join(CACHE, "lists_dev_stop.json"), encoding="utf-8"))
    dev_cal = json.load(open(os.path.join(CACHE, "lists_dev_cal.json"), encoding="utf-8"))
    jdump({"full_train": desc(train), "train_subset": desc(sub), "dev_stop": desc(dev_stop),
           "dev_cal": desc(dev_cal), "subset_seed": 13}, os.path.join(RESULTS, "lists_info.json"))
    print("subset drawn", N_SUB, flush=True)
