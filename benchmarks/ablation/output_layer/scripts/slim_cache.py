"""Disk limit workaround: once all 25 runs of an encoder are saved, drop the feature arrays (X, cls) from its cache
parts, keeping off, gold, nblocks and (E1) depth_rel, which evaluation and the depth analysis need.
A part that was slimmed carries slim=1. Re-training that encoder needs its features cached again
(delete cache/<ENC> and rerun the cache stage). python -I slim_cache.py ENC"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import CACHE, SEEDS, VARIANTS

enc = sys.argv[1]
if not all(os.path.exists(os.path.join(CACHE, "preds", enc, f"{v}_s{s}.npz")) for v in VARIANTS for s in SEEDS):
    sys.exit(f"{enc}: not all runs saved, nothing slimmed")
for p in sorted(glob.glob(os.path.join(CACHE, enc, "*.part*.npz"))):
    z = np.load(p)
    if "slim" in z.files:
        continue
    keep = {k: z[k] for k in z.files if k not in ("X", "cls")}
    z.close()
    tmp = p + ".tmp.npz"
    np.savez(tmp, slim=np.array(1), **keep)
    os.replace(tmp, p)
print(f"{enc} slimmed", flush=True)
