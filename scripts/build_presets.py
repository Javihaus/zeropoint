"""Rebuild the calibrated models shipped with the package.

    pip install -e ".[monitor]" datasets
    python scripts/build_presets.py --encoder small
    python scripts/build_presets.py --encoder base

Writes src/zeropoint/presets/{support,everyday}-{encoder}.safetensors. Embeddings are cached in
.cache/ so the benchmark can reuse them.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))
import data  # noqa: E402

from zeropoint import ZeroPoint  # noqa: E402
from zeropoint.calibrate import add_edge  # noqa: E402
from zeropoint.encoder import EncoderSpec, get_encoder  # noqa: E402

CACHE = os.path.join(HERE, "..", ".cache")
PRESETS = os.path.join(HERE, "..", "src", "zeropoint", "presets")


def embed(texts, encoder, precision, tag):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{tag}-{encoder}-{precision}.npy")
    if os.path.exists(path):
        e = np.load(path)
        if len(e) == len(texts):
            return e
    t0 = time.time()
    e = get_encoder(EncoderSpec.resolve(encoder, precision)).encode(list(texts), "query")
    np.save(path, e)
    print(f"  embedded {len(texts)} {tag} texts in {time.time() - t0:.0f}s")
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="small")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(PRESETS, exist_ok=True)
    jobs = {
        "support": (data.tickets()[0], {"descriptions": data.QUEUES, "catch_all": [data.GI]}),
        "everyday": (data.clinc150()[0], {}),
    }
    for name, (pool, opts) in jobs.items():
        texts, labels = [t for t, _ in pool], [y for _, y in pool]
        print(f"{name}: {len(texts)} training texts, {len(set(labels))} labels")
        emb = embed(texts, a.encoder, "fp32", f"pool-{name}")
        zp = ZeroPoint.fit(texts, labels, encoder=a.encoder, precision="fp32", seed=a.seed, monitor=True,
                           embeddings=emb, edge_int8=False, verbose=False, **opts)
        t = zp.config["training"]
        print(f"  {t['epochs']} epochs, validation accuracy {t['validation_accuracy']:.3f} (chance {t['chance']:.3f}), "
              f"temperature {zp.temperature:.3f}, ndt participation ratio {t['participation_ratio'][-1]:.1f}")
        add_edge(zp, texts, labels, "int8", embeddings=embed(texts, a.encoder, "int8", f"pool-{name}"), seed=a.seed)
        zp.save(os.path.join(PRESETS, f"{name}-{a.encoder}.safetensors"))


if __name__ == "__main__":
    main()
