"""Fuzz the decision rule and the option ordering with arbitrary inputs.

Whatever the depths of the wells, the probabilities must be finite, non-negative and sum to one. Whatever the
option strings, the canonical order must be a permutation that does not depend on the order they arrive in, and
anything that is not a valid list (empty or duplicate options) must raise ValueError, never anything else.

    pip install atheris && python fuzz/fuzz_engine.py -max_total_time=60
"""

import sys

import atheris

with atheris.instrument_imports():
    import numpy as np

    from zeropoint._runtime import Runtime, canonical_order

RT = Runtime(threads=1)


def test_one_input(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    n = fdp.ConsumeIntInRange(1, 200)
    depths = np.array(fdp.ConsumeFloatListInRange(n, 0.0, 1e6), dtype=np.float64)
    if depths.size:
        p = RT.distribution(depths)
        assert p.shape == (depths.size + 1,)
        assert np.all(np.isfinite(p)) and np.all(p >= 0)
        assert abs(p.sum() - 1.0) < 1e-9
    options = [fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 30)) for _ in range(fdp.ConsumeIntInRange(1, 12))]
    try:
        order = canonical_order(options)
    except ValueError:
        return
    assert sorted(order) == list(range(len(options)))
    canon = [options[i] for i in order]
    shuffled = options[::-1]
    assert [shuffled[i] for i in canonical_order(shuffled)] == canon


atheris.Setup(sys.argv, test_one_input)
atheris.Fuzz()
