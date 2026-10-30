"""Fuzz the decision rule with arbitrary numbers.

Whatever the energies, temperature and distance to the edge, the probabilities must be finite, non-negative
and sum to one, and a deeper vacuum must never change the ranking of the options.

    pip install atheris && python fuzz/fuzz_engine.py -max_total_time=60
"""

import sys

import atheris

with atheris.instrument_imports():
    import numpy as np

    from zeropoint import engine


def test_one_input(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    n_texts = fdp.ConsumeIntInRange(1, 4)
    n_options = fdp.ConsumeIntInRange(1, 24)
    energy = -np.abs(np.array(fdp.ConsumeFloatListInRange(n_texts * n_options, 0.0, 1e4), dtype=np.float64))
    if energy.size != n_texts * n_options:
        return
    energy = energy.reshape(n_texts, n_options)
    temperature = fdp.ConsumeFloatInRange(1e-3, 1e3)
    edge = np.array(fdp.ConsumeFloatListInRange(n_texts, -50.0, 50.0), dtype=np.float64)
    if edge.size != n_texts:
        return
    kappa = fdp.ConsumeFloatInRange(0.0, 10.0)

    p = engine.probabilities(energy, temperature, kappa * np.maximum(edge, 0.0))
    assert p.shape == (n_texts, n_options + 1)
    assert np.all(np.isfinite(p)) and np.all(p >= 0)
    assert np.allclose(p.sum(1), 1.0)

    p0 = engine.probabilities(energy, temperature, np.zeros(n_texts))
    order = np.argsort(-p0[:, :-1], axis=1, kind="stable")
    for r in range(n_texts):
        ranked = p[r, :-1][order[r]]
        assert np.all(np.diff(ranked) <= 1e-12 + 1e-9 * np.abs(ranked[:-1]))


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
