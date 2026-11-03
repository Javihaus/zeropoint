"""Latency of ZeroPoint on this machine, with the hardware it ran on.

    python benchmarks/latency.py                  # prints a table and writes latency.json
    python benchmarks/latency.py --check          # also fails if the budget is exceeded (used in CI)

Each cell: one text of 14 words against a list of options, decided 200 times after 20 warm-up calls. The time
covers everything a user waits for: reading, encoder, Boltzmann distribution. Median and 99th percentile in ms.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import onnxruntime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from zeropoint import ZeroPoint, __version__  # noqa: E402

TEXT = "I was charged twice for the same order and I would like the second payment refunded please"
OPTIONS = ["ask for a refund", "track an order", "cancel an order", "change the delivery address", "return an item",
           "update payment details", "apply a discount code", "contact the seller", "reorder a past purchase",
           "exchange an item for another size", "report a damaged item", "check delivery times", "close my account",
           "reset my password", "change my email", "subscribe to the newsletter", "leave a review",
           "ask about warranty", "find a store", "talk to a person"]
# Budget on a GitHub-hosted runner (4 vCPU), 2 threads: generous enough for a shared machine, tight enough to
# catch a regression of 2x or more.
BUDGET_MS = {5: 40.0, 20: 90.0}


def cpu_name():
    try:
        if sys.platform.startswith("linux"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return os.popen("sysctl -n machdep.cpu.brand_string").read().strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def measure(zp, k, runs=200, warm=20):
    opts = OPTIONS[:k] if k <= len(OPTIONS) else [f"{o} ({i})" for i, o in enumerate(OPTIONS * (k // len(OPTIONS) + 1))][:k]
    for _ in range(warm):
        zp.decide(TEXT, opts)
    ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        zp.decide(TEXT, opts)
        ms.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ms)), float(np.percentile(ms, 99))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default="latency.json")
    args = ap.parse_args()
    info = {"zeropoint": __version__, "cpu": cpu_name(), "cores": os.cpu_count(), "platform": platform.platform(),
            "python": platform.python_version(), "onnxruntime": onnxruntime.__version__}
    rows = []
    for threads in (1, 2):
        zp = ZeroPoint(threads=threads)
        for k in (5, 20, 60):
            med, p99 = measure(zp, k)
            rows.append({"threads": threads, "options": k, "median_ms": round(med, 1), "p99_ms": round(p99, 1)})
    print(json.dumps(info))
    print("| threads | options | median ms | p99 ms |\n|---|---|---|---|")
    for r in rows:
        print(f"| {r['threads']} | {r['options']} | {r['median_ms']} | {r['p99_ms']} |")
    json.dump({"machine": info, "latency": rows}, open(args.out, "w"), indent=1)
    if args.check:
        bad = [r for r in rows if r["threads"] == 2 and r["options"] in BUDGET_MS and r["median_ms"] > BUDGET_MS[r["options"]]]
        if bad:
            sys.exit(f"latency budget exceeded: {bad}")


if __name__ == "__main__":
    main()
