"""fit(), recalibrate() and save(): they run, they are deterministic, and a saved model gives the same bits."""
import hashlib
import os

import numpy as np
import pytest

from zeropoint import ZeroPoint

from ._data import LABELS, TEXTS

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.fixture(scope="module")
def base():
    return ZeroPoint(threads=1)


def test_fit_runs_reports_and_is_deterministic(base, tmp_path):
    a, report = base.fit(TEXTS, LABELS, steps=4, frozen_layers=10, threads=1)
    b, _ = base.fit(TEXTS, LABELS, steps=4, frozen_layers=10, threads=1)
    assert a.model_id == b.model_id                       # same data, seed and threads: same model
    assert set(report) >= {"before", "after", "held_out_examples", "trained_on", "seconds"}
    assert report["after"]["lists"] == report["before"]["lists"] > 0
    a.save(tmp_path / "mine")
    c = ZeroPoint(str(tmp_path / "mine"), threads=1)
    for t in ["will it rain", "play jazz", "what is two plus two"]:
        d1, d2 = a.decide(t, ["weather", "music", "alarm"]), c.decide(t, ["weather", "music", "alarm"])
        assert d1.probabilities == d2.probabilities and d1.p_none == d2.p_none
    assert c.report is not None and c._rt.manifest["derived_from"] == base.model_id


def test_fit_leaves_frozen_layers_untouched(base):
    from zeropoint.fit import _encoder_weights
    fitted, _ = base.fit(TEXTS, LABELS, steps=3, frozen_layers=11, threads=1)
    w0, w1 = _encoder_weights(base._rt.model_dir), _encoder_weights(fitted._rt.model_dir)
    for name in w0:
        changed = not np.array_equal(w0[name], w1[name])
        if name.startswith("encoder.layer.11."):
            continue
        assert not changed, f"{name} changed although it is frozen"


def test_bad_fit_input_is_refused(base):
    with pytest.raises(ValueError):
        base.fit(TEXTS[:5], LABELS[:5])
    with pytest.raises(ValueError):
        base.fit(TEXTS, ["weather"] * len(TEXTS))
    with pytest.raises(ValueError):
        base.fit(TEXTS, LABELS[:-1])


def test_fit_warns_when_the_held_out_part_is_small(base):
    with pytest.warns(UserWarning, match="held-out examples"):
        base.fit(TEXTS, LABELS, steps=1, frozen_layers=11, threads=1)


def test_fit_folder_is_deleted_with_the_model(base):
    import gc
    fitted, _ = base.fit(TEXTS, LABELS, steps=1, frozen_layers=11, threads=1)
    folder = fitted._rt.model_dir
    assert os.path.isdir(folder)
    del fitted
    gc.collect()
    assert not os.path.exists(folder)


def test_fit_folder_stays_while_a_recalibrated_copy_lives(base):
    import gc
    fitted, _ = base.fit(TEXTS, LABELS, steps=1, frozen_layers=11, threads=1)
    folder = fitted._rt.model_dir
    recal, _ = fitted.recalibrate(TEXTS, LABELS)
    del fitted
    gc.collect()
    assert os.path.isdir(folder)                          # recal still reads it
    del recal
    gc.collect()
    assert not os.path.exists(folder)
