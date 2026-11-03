"""Fuzz model loading with damaged files.

Any change to any byte of any model file must be refused with ModelFilesError before the file is parsed,
never crash the process or load a different model.

    pip install atheris && python fuzz/fuzz_load.py -max_total_time=60
"""

import os
import shutil
import sys
import tempfile

import atheris

with atheris.instrument_imports():
    from zeropoint._runtime import MODEL_DIR, ModelFilesError, Runtime

WORK = os.path.join(tempfile.mkdtemp(prefix="zp-fuzz-"), "model")
shutil.copytree(MODEL_DIR, WORK)
SMALL = ["head.json", "manifest.json", "vocab_scale.npy"]
ORIGINAL = {f: open(os.path.join(MODEL_DIR, f), "rb").read() for f in SMALL}


def test_one_input(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    name = SMALL[fdp.ConsumeIntInRange(0, len(SMALL) - 1)]
    content = fdp.ConsumeBytes(fdp.ConsumeIntInRange(0, 4096))
    if content == ORIGINAL[name]:
        return
    path = os.path.join(WORK, name)
    with open(path, "wb") as f:
        f.write(content)
    try:
        Runtime(WORK, threads=1)
        raise AssertionError(f"a damaged {name} was accepted")
    except ModelFilesError:
        pass
    finally:
        with open(path, "wb") as f:
            f.write(ORIGINAL[name])


atheris.Setup(sys.argv, test_one_input)
atheris.Fuzz()
