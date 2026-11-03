"""Build the golden cases and their exact expected outputs.

    python tools/make_golden.py            # writes tests/golden/cases.json and tests/golden/expected.json

The 500 cases are ZeroPoint's own written examples (no third-party text): 400 lists of 4 to 20 options from
the same label set, a third of them with the right option removed ("none" cases), 50 long lists of 21 to 60
options, and 50 cases that stress the reading rules (long texts that get cut, accents in both Unicode forms,
one-option lists, very short texts). Expected outputs are written as hexadecimal floats, so the comparison in
tests/test_golden.py is exact to the last bit.
"""
import json
import os
import sys
import unicodedata

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from zeropoint import ZeroPoint  # noqa: E402


def build_cases(synth_path, pools_path, seed=20261103):
    rng = np.random.default_rng(seed)
    with open(synth_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    with open(pools_path, encoding="utf-8") as f:
        pools = {k: v["labels"] for k, v in json.load(f).items() if not k.startswith("_")}
    rng.shuffle(rows)
    cases = []
    for r in rows[:400]:
        labs = pools[r["pool"]]
        k = int(rng.integers(4, min(20, len(labs)) + 1))
        others = [l for l in labs if l != r["label"]]
        opts = list(rng.choice(others, min(k, len(others)), replace=False))
        if r["label"] is not None and rng.random() > 1 / 3:
            opts[0] = r["label"]
        rng.shuffle(opts)
        cases.append({"text": r["text"], "options": [str(o) for o in opts]})
    everything = sorted({l for v in pools.values() for l in v})
    for r in rows[400:450]:
        k = int(rng.integers(21, 61))
        opts = list(rng.choice(everything, k, replace=False))
        cases.append({"text": r["text"], "options": [str(o) for o in opts]})
    for i, r in enumerate(rows[450:500]):
        labs = pools[r["pool"]]
        kind = i % 5
        if kind == 0:
            text = (r["text"] + " ") * 12
            opts = labs[:5]
        elif kind == 1:
            text = unicodedata.normalize("NFD", r["text"])
            opts = [unicodedata.normalize("NFD", o) for o in labs[:6]]
        elif kind == 2:
            text, opts = r["text"], [labs[0]]
        elif kind == 3:
            text, opts = r["text"].split()[0], labs[:4]
        else:
            text = r["text"]
            opts = [o + " and everything related to it, including the special cases nobody thinks about" for o in labs[:3]]
        cases.append({"text": text, "options": list(opts)})
    return cases


def main():
    src = os.environ.get("ZP_SYNTH", "data/synth.jsonl")
    pools = os.environ.get("ZP_POOLS", "data/synth_pools.json")
    cases = build_cases(src, pools)
    zp = ZeroPoint(threads=1)
    expected = []
    for c in cases:
        d = zp.decide(c["text"], c["options"])
        expected.append({"choice": d.choice, "p_none": d.p_none.hex(),
                         "p": [d.probabilities[o].hex() for o in c["options"]]})
    out = os.path.join(ROOT, "tests", "golden")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cases.json"), "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=0)
    made_with = {"onnxruntime": __import__("onnxruntime").__version__, "numpy": np.__version__,
                 "tokenizers": __import__("tokenizers").__version__}
    with open(os.path.join(out, "expected.json"), "w") as f:
        json.dump({"model": zp.model_id, "made_with": made_with, "cases": expected}, f, indent=0)
    print(len(cases), "cases;", sum(e["choice"] is None for e in expected), "decided as none")


if __name__ == "__main__":
    main()
