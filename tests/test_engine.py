import numpy as np
import pytest

from zeropoint import ZeroPoint, engine
from zeropoint.calibrate import ledoit_wolf_precision
from zeropoint.model import PRESETS


def test_probabilities_sum_to_one_and_vacuum_wins_far_from_every_well():
    energy = np.array([[-3.0, -0.5, -0.1], [0.0, 0.0, 0.0]])
    p = engine.probabilities(energy, 1.0, np.zeros(2))
    assert np.allclose(p.sum(1), 1)
    assert p[0].argmax() == 0
    assert p[1, -1] > 0.999  # no well reached: "none"


def test_a_deeper_vacuum_only_moves_probability_to_none():
    energy = np.array([[-3.0, -1.0]])
    p0 = engine.probabilities(energy, 1.0, np.array([0.0]))
    p1 = engine.probabilities(energy, 1.0, np.array([4.0]))
    assert p1[0, -1] > p0[0, -1]
    assert np.isclose(p0[0, 0] / p0[0, 1], p1[0, 0] / p1[0, 1])  # the ranking of options never changes


def test_log_weights_are_stable_for_deep_wells():
    lw = engine.log_weights(np.array([-1e4, -1e-9]), 0.5)
    assert np.all(np.isfinite(lw))


def test_gelu_matches_pytorch():
    torch = pytest.importorskip("torch")
    x = np.linspace(-6, 6, 101).astype(np.float32)
    ref = torch.nn.functional.gelu(torch.tensor(x), approximate="tanh").numpy()
    assert np.allclose(engine.gelu(x), ref, atol=1e-6)


def test_ledoit_wolf_matches_scikit_learn():
    sk = pytest.importorskip("sklearn.covariance")
    x = np.random.default_rng(0).normal(size=(200, 12)) * np.arange(1, 13)
    assert np.allclose(ledoit_wolf_precision(x), sk.LedoitWolf().fit(x).precision_, atol=1e-8)


@pytest.mark.parametrize("name", PRESETS)
@pytest.mark.parametrize("encoder", ["small", "base"])
def test_presets_load_and_decide_on_embeddings(name, encoder):
    zp = ZeroPoint.load(name, encoder=encoder)
    dim = zp.weights["option_emb"].shape[1]
    emb = engine.unit(np.random.default_rng(0).normal(size=(4, dim)).astype(np.float32))
    out = zp._decide_embedded(emb, [zp.labels[:5]] * 4)
    for d in out:
        assert np.isclose(sum(d.probabilities.values()) + d.p_none, 1, atol=1e-5)
        assert d.choice is None or d.choice in zp.labels[:5]
