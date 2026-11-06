"""recalibrate() needs no torch: it re-fits the temperature and the "none" level only."""
import pytest

from zeropoint import ZeroPoint

from ._data import LABELS, TEXTS


@pytest.fixture(scope="module")
def base():
    return ZeroPoint(threads=1)


def test_recalibrate_changes_only_two_numbers(base, tmp_path):
    new, report = base.recalibrate(TEXTS, LABELS)
    again, _ = base.recalibrate(TEXTS, LABELS)
    assert (new._log_temperature, new._none_level) == (again._log_temperature, again._none_level)
    assert new._rt is base._rt                              # same encoder, same files
    assert set(report) >= {"before", "after", "temperature", "none_level"}
    new.save(tmp_path / "recal")
    loaded = ZeroPoint(str(tmp_path / "recal"), threads=1)
    d1 = new.decide("is it going to rain", ["weather", "music"])
    d2 = loaded.decide("is it going to rain", ["weather", "music"])
    assert d1.probabilities == d2.probabilities and d1.p_none == d2.p_none


def test_base_model_is_unchanged_by_recalibrate(base):
    before = base.decide("play jazz", ["music", "weather"])
    base.recalibrate(TEXTS, LABELS)
    after = base.decide("play jazz", ["music", "weather"])
    assert before.probabilities == after.probabilities


def test_recalibrate_warns_when_the_held_out_part_is_small(base):
    with pytest.warns(UserWarning, match="held-out examples"):
        base.recalibrate(TEXTS, LABELS)               # 52 examples: about 10 held out
