"""The public API: what decide() returns, the input rules, and the guarantees that never depend on circumstances."""
import itertools
import socket

import pytest

from zeropoint import Decision, UnsupportedLanguage, ZeroPoint

BANK = ["block a card", "check the balance", "order a new card", "transfer money"]


@pytest.fixture(scope="module")
def zp():
    return ZeroPoint(threads=1)


def same(a: Decision, b: Decision) -> bool:
    return (a.choice == b.choice and a.p_none == b.p_none and a.confidence == b.confidence
            and a.probabilities == b.probabilities and a.depths == b.depths)


def test_decision_is_a_distribution(zp):
    d = zp.decide("my card was swallowed by the cash machine", BANK)
    assert isinstance(d, Decision)
    assert list(d.probabilities) == BANK                       # the caller's order
    assert abs(sum(d.probabilities.values()) + d.p_none - 1.0) < 1e-12
    assert d.choice in BANK or d.choice is None
    assert d.confidence == (d.probabilities[d.choice] if d.choice else d.p_none)
    assert all(v >= 0 for v in d.depths.values())


def test_obvious_cases(zp):
    assert zp.decide("please send 200 euros to my brother's account", BANK).choice == "transfer money"
    assert zp.decide("what's the weather like in Lisbon tomorrow?", BANK).choice is None


def test_order_never_matters(zp):
    text = "Necesito anular la reserva del hotel de mañana"
    options = ["reservar un hotel", "anular una reserva de hotel", "alquilar un coche", "comprar un billete"]
    ref = zp.decide(text, options)
    for perm in itertools.permutations(options):
        d = zp.decide(text, list(perm))
        assert d.choice == ref.choice and d.p_none == ref.p_none
        assert all(d.probabilities[o] == ref.probabilities[o] for o in options)


def test_threads_never_matter():
    text, options = "turn off the lights in the kitchen", ["lights off", "lights on", "set an alarm"]
    ref = ZeroPoint(threads=1).decide(text, options)
    for n in (2, 4):
        assert same(ZeroPoint(threads=n).decide(text, options), ref)


def test_decide_many_equals_decide(zp):
    texts = ["block my card please", "how much money do I have", "hola"]
    many = zp.decide_many(texts, BANK)
    for t, d in zip(texts, many):
        assert same(d, zp.decide(t, BANK))
    per_text = [BANK, BANK[:2], ["say hello", "say goodbye"]]
    for t, o, d in zip(texts, per_text, zp.decide_many(texts, per_text)):
        assert same(d, zp.decide(t, o))


@pytest.mark.parametrize("text,options,error", [
    ("", BANK, ValueError),
    ("   ", BANK, ValueError),
    (None, BANK, TypeError),
    ("hello", [], ValueError),
    ("hello", "block a card", TypeError),
    ("hello", ["a", 3], TypeError),
    ("hello", ["block a card", "block a card"], ValueError),
    ("hello", ["block a card", ""], ValueError),
])
def test_bad_input_is_refused(zp, text, options, error):
    with pytest.raises(error):
        zp.decide(text, options)


@pytest.mark.parametrize("text", ["Заблокируйте мою карту", "请冻结我的银行卡", "أريد حظر بطاقتي", "Μπλοκάρετε την κάρτα μου"])
def test_other_scripts_are_refused(zp, text):
    with pytest.raises(UnsupportedLanguage):
        zp.decide(text, BANK)
    with pytest.raises(UnsupportedLanguage):
        zp.decide("block my card", BANK[:2] + [text])


def test_a_foreign_name_inside_latin_text_is_fine(zp):
    zp.decide("Pedí unos auriculares Sony 東京 edition y no han llegado", ["seguir un pedido", "devolver un artículo"])


def test_truncation_is_reported(zp):
    long_text = "My card does not work. " * 120
    d = zp.decide(long_text, BANK)
    assert d.text_truncated and not zp.decide("short", BANK).text_truncated
    long_option = "an option with a very long description that goes on and on well past sixteen pieces of text"
    d = zp.decide("anything", ["short", long_option])
    assert d.options_truncated == (long_option,)


def test_long_lists_are_read_in_blocks(zp):
    options = [f"topic {i}" for i in range(55)]
    assert zp.decide("topic 7", options).blocks == 3


def test_no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("ZeroPoint tried to open a network connection")
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    z = ZeroPoint(threads=1)
    z.decide("block my card", BANK)
