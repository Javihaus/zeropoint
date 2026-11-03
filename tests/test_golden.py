"""500 fixed cases with their exact expected outputs (tests/golden, made by tools/make_golden.py).

On every platform the decisions must be the same and no probability may move by more than 1e-5. Whether the
outputs are also identical to the last bit is measured and written to the report (ZP_GOLDEN_REPORT), so the
statement about bit-identical results across machines rests on data from every CI platform."""
import json
import os
import platform
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime
import pytest

from zeropoint import ZeroPoint

GOLDEN = Path(__file__).parent / "golden"
CASES = json.loads((GOLDEN / "cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((GOLDEN / "expected.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def results():
    zp = ZeroPoint()
    assert zp.model_id == EXPECTED["model"], "the golden outputs belong to another model"
    return [zp.decide(c["text"], c["options"]) for c in CASES]


def test_golden_outputs(results):
    choice_changes, bit_identical, max_diff = [], 0, 0.0
    for i, (case, exp, d) in enumerate(zip(CASES, EXPECTED["cases"], results)):
        if d.choice != exp["choice"]:
            choice_changes.append(i)
        got = [d.p_none] + [d.probabilities[o] for o in case["options"]]
        want = [float.fromhex(exp["p_none"])] + [float.fromhex(x) for x in exp["p"]]
        bit_identical += got == want
        max_diff = max(max_diff, float(np.max(np.abs(np.array(got) - np.array(want)))))
    report = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
              "onnxruntime": onnxruntime.__version__, "numpy": np.__version__, "cases": len(CASES),
              "bit_identical": bit_identical, "choice_changes": choice_changes, "max_abs_diff": max_diff}
    print("\ngolden:", json.dumps(report))
    if os.environ.get("ZP_GOLDEN_REPORT"):
        Path(os.environ["ZP_GOLDEN_REPORT"]).write_text(json.dumps(report, indent=1))
    assert not choice_changes, f"decisions changed in cases {choice_changes}"
    assert max_diff <= 1e-5, f"a probability moved by {max_diff:.2e}"


def test_unicode_forms_give_the_same_bits():
    zp = ZeroPoint(threads=1)
    text, options = "Quiero cambiar la dirección de entrega", ["cambiar la dirección de entrega", "devolver un artículo"]
    nfc = zp.decide(text, options)
    nfd = zp.decide(unicodedata.normalize("NFD", text), [unicodedata.normalize("NFD", o) for o in options])
    assert nfc.p_none == nfd.p_none and list(nfc.probabilities.values()) == list(nfd.probabilities.values())
