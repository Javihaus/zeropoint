"""Fuzz model loading with damaged and hostile files.

A file that is not a valid ZeroPoint model must be rejected with a ValueError, never crash the process or
raise anything else. Seeds are the shipped models, so the fuzzer starts from realistic files and mutates them.

    pip install atheris && python fuzz/fuzz_load.py -max_total_time=60
"""

import os
import sys
import tempfile

import atheris

with atheris.instrument_imports():
    from zeropoint import ZeroPoint

PATH = os.path.join(tempfile.mkdtemp(prefix="zp-fuzz-"), "model.safetensors")


def test_one_input(data: bytes) -> None:
    with open(PATH, "wb") as f:
        f.write(data)
    try:
        ZeroPoint.load(PATH)
    except ValueError:
        return  # rejected cleanly, as it should be


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
