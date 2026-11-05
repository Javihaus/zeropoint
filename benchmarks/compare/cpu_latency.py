"""Latency of ZeroPoint and Strands Decider on the same CPU, case by case.

    python cpu_latency.py [N]        # the first N cases of the 2,000-case test's 300-case sample (default 118)

Strands Decider needs a GPU to be fast; this measures what it costs on the 2-core CPU that ZeroPoint is made for.
Both read the same cases, in the same process conditions as run.py (2 threads, one case at a time, three warm-up
calls not timed). Strands Decider is kept in bfloat16, as in run.py: the library would cast it to float32 on CPU,
which needs about 9 GB of memory; the library's own notes give about half the time in float32 (on an Apple M3
Pro). Rows are saved as they come in out/cpu_latency/<system>.jsonl, so an interrupted run continues.
"""
import json
import os
import sys
import time

from run import HERE, SYSTEMS, environment, load, sample, set_torch_threads

TEST = "bench2000"


def main(n):
    out = os.path.join(HERE, "out", "cpu_latency")
    os.makedirs(out, exist_ok=True)
    items = load(TEST)
    idx = sample(items)[:n]
    set_torch_threads()
    for system in ("zeropoint", "strands_none"):
        path = os.path.join(out, f"{system}.jsonl")
        done = set()
        if os.path.exists(path):
            with open(path) as f:
                done = {json.loads(line)["i"] for line in f if line.strip()}
        if done >= set(idx):
            continue
        decide = SYSTEMS[system][0](TEST, items)
        for it in items[:3]:
            decide(it["text"], it["options"])
        with open(path, "a") as f:
            for i in idx:
                if i in done:
                    continue
                s = time.perf_counter()
                row = decide(items[i]["text"], items[i]["options"])
                row.update(i=i, ms=(time.perf_counter() - s) * 1000)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
    with open(os.path.join(out, "environment.json"), "w") as f:
        json.dump(environment(), f, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 118)
