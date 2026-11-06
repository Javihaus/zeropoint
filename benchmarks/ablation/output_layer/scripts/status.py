"""Write results/STATUS.md from what is on disk (so it is always true, whoever runs it)."""
import datetime
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, LONG, RESULTS, SEEDS, TESTS, VARIANTS


def _json_load(*args, **kw):
    with open(*args, **kw) as f:
        return json.load(f)


def _jsonl(*args, **kw):
    with open(*args, **kw) as f:
        return [json.loads(line) for line in f]

note = " ".join(sys.argv[1:])
splits = ["dev_stop", "dev_cal"] + TESTS + LONG + ["train"]


def cache_state(enc):
    done = [s for s in splits if glob.glob(os.path.join(CACHE, enc, f"{s}.part*.npz"))]
    tp = len(glob.glob(os.path.join(CACHE, enc, "train.part*.npz")))
    return f"{len(done)}/{len(splits)} splits started; train parts {tp}/30"


def preds(enc):
    return len([1 for v in VARIANTS for s in SEEDS if os.path.exists(os.path.join(CACHE, "preds", enc, f"{v}_s{s}.npz"))])


rows = [
    ("PROTOCOL.md written (frozen)", os.path.exists(os.path.join(RESULTS, "PROTOCOL.md"))),
    ("1 lists (cache/lists_*.json, results/lists_info.json)", os.path.exists(os.path.join(RESULTS, "lists_info.json"))),
    ("0 packaging (results/packaging.json)", os.path.exists(os.path.join(RESULTS, "packaging.json"))),
    (f"2 cache E1 ({cache_state('E1')})", os.path.exists(os.path.join(CACHE, "E1", "DONE"))),
    (f"2 cache E2 ({cache_state('E2')})", os.path.exists(os.path.join(CACHE, "E2", "DONE"))),
    ("3 depth analysis (results/depths.json)", os.path.exists(os.path.join(RESULTS, "depths.json"))),
    (f"4 train E1 ({preds('E1')}/25 runs)", preds("E1") == 25),
    (f"4 train E2 ({preds('E2')}/25 runs)", preds("E2") == 25),
    ("5 eval + bootstrap (results/metrics.json, results/paired.json)", os.path.exists(os.path.join(RESULTS, "paired.json"))),
    ("6 report (results/REPORT.md, results/summary.json)", os.path.exists(os.path.join(RESULTS, "summary.json"))),
]
lines = ["# STATUS", "", f"Updated {datetime.datetime.now().isoformat(timespec='seconds')}." + (f" Last event: {note}" if note else ""),
         "", "| stage | done |", "|---|---|"] + [f"| {a} | {'yes' if b else 'no'} |" for a, b in rows]
lines += ["", "## How to continue", "",
          "Run `bash scripts/driver.sh` from the ablation folder (in the background). Every stage skips work whose",
          "output file exists, so a rerun resumes. Logs are in `logs/`. Python is `pkg/venv/bin/python -I`.",
          "Deviations from PROTOCOL.md, if any, are listed below and in REPORT.md.", ""]
dev_p = os.path.join(RESULTS, "deviations.json")
if os.path.exists(dev_p):
    lines += ["## Deviations", ""] + [f"- {d}" for d in _json_load(dev_p)] + [""]
with open(os.path.join(RESULTS, "STATUS.md"), "w") as f:
    f.write("\n".join(lines))
