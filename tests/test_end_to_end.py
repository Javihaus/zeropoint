"""End-to-end tests. They download the small encoder (about 120 MB) on first run."""

import numpy as np
import pytest

from zeropoint import ZeroPoint


@pytest.fixture(scope="module")
def tickets():
    return ZeroPoint.load("support")


def test_chooses_the_obvious_queue(tickets):
    d = tickets.decide("Subject: charged twice\nI was billed two times for my March invoice. Please refund one.")
    assert d.choice == "Billing and Payments"


def test_says_none_for_a_text_from_another_world(tickets):
    d = tickets.decide("Subject: Great film\nThe cinematography is stunning and the lead actor gives the "
                       "performance of his career. I laughed, I cried, and I would watch it again tomorrow.")
    assert d.choice is None
    assert d.beyond_edge > 0


def test_says_none_when_the_right_option_is_missing(tickets):
    text = "Subject: charged twice\nI was billed two times for my March invoice. Please refund one."
    full = tickets.decide(text)
    missing = tickets.decide(text, options=["Human Resources", "Returns and Exchanges"])
    assert missing.p_none > full.p_none


def test_batch_and_single_agree(tickets):
    texts = ["My laptop cannot connect to the office VPN.", "How much does the enterprise plan cost?"]
    many = tickets.decide_many(texts)
    for t, d in zip(texts, many):
        assert np.isclose(tickets.decide(t).p_none, d.p_none, atol=1e-5)


def test_fit_save_load_roundtrip(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    seeds = {"billing": ["I was charged twice", "refund my payment", "the invoice is wrong", "card declined"],
             "tech": ["the app crashes on start", "I cannot log in", "server error 500", "sync is broken"],
             "shipping": ["where is my parcel", "the delivery is late", "package arrived damaged", "tracking fails"]}
    texts, labels = [], []
    for k, v in seeds.items():
        for i in range(60):
            texts.append(f"{v[i % 4]} (order {rng.integers(10_000)})")
            labels.append(k)
    zp = ZeroPoint.fit(texts, labels, verbose=False)
    path = tmp_path / "m.safetensors"
    zp.save(str(path))
    back = ZeroPoint.load(str(path), precision=None)
    zp.kappa = back.kappa = 0.0
    assert back.decide("refund my card payment").choice == "billing"
    assert np.isclose(back.decide("app crash").p_none, zp.decide("app crash").p_none)


def test_default_landscape_and_latency():
    zp = ZeroPoint.load()
    d = zp.decide("Can you move 200 euros from my savings to my checking account?",
                  options=["transfer", "balance", "pay bill"])
    assert d.choice == "transfer"
    assert 0 < d.latency_ms < 5000
