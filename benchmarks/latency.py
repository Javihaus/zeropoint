"""Latency of one decision, split into encoder and engine.

    pip install -e . datasets
    python benchmarks/latency.py                       # CPU
    python benchmarks/latency.py --providers CUDAExecutionProvider --precisions fp32   # GPU, needs onnxruntime-gpu

The texts are real support tickets (median about 400 characters, the long end of typical inputs) with 10
options. Reported: median (p50) and 95th percentile (p95) in milliseconds per call, for a single text and for
a batch of 32.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data  # noqa: E402

from zeropoint import ZeroPoint  # noqa: E402


def timed(fn, reps, warmup=5):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(t, 50)), float(np.percentile(t, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["small", "base"])
    ap.add_argument("--precisions", nargs="+", default=["int8", "fp32"])
    ap.add_argument("--providers", nargs="+", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "latency.csv"))
    a = ap.parse_args()
    pool, _ = data.tickets()
    texts = [t for t, _ in pool[:2000]]
    rng = np.random.default_rng(0)
    rows = []
    for encoder in a.encoders:
        for precision in a.precisions:
            zp = ZeroPoint.load("tickets", encoder=encoder, precision=precision, providers=a.providers,
                               threads=a.threads)
            enc = zp.encoder
            for batch in [1, 32]:
                pick = lambda: [texts[i] for i in rng.integers(len(texts), size=batch)]  # noqa: E731
                e_p50, e_p95 = timed(lambda: enc.encode(pick(), "query"), a.reps)
                emb = enc.encode(pick(), "query")
                opts = [zp.labels] * batch
                g_p50, g_p95 = timed(lambda: zp._decide_embedded(emb, opts), a.reps)
                t_p50, t_p95 = timed(lambda: zp.decide_many(pick()), a.reps)
                rows.append({"encoder": encoder, "precision": precision, "batch": batch,
                             "encoder_p50_ms": e_p50, "encoder_p95_ms": e_p95,
                             "engine_p50_ms": g_p50, "engine_p95_ms": g_p95,
                             "total_p50_ms": t_p50, "total_p95_ms": t_p95,
                             "per_text_p50_ms": t_p50 / batch,
                             "providers": ",".join(enc.session.get_providers()),
                             "machine": f"{platform.processor() or platform.machine()}, {os.cpu_count()} threads"})
                print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rows[-1].items()})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
