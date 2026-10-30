"""Training pools and test cases for the presets and the benchmark.

Everything comes from public datasets on the Hugging Face Hub and is fixed by seeds, so anyone can rebuild
it. The test cases are the same ones used in the research runs (and, for tickets, in the Laya audit):
no test case is ever in a training pool, and no out-of-scope text is ever used for training.
"""

from __future__ import annotations

import random
import re

N = 800                 # cases per side in each test
TRAIN_CAP = 20000
LEN_MIN, LEN_MAX = 350, 700
N_FAR_EACH = 150

QUEUES = {"Technical Support": "technical problems, bugs, outages, integrations",
          "Product Support": "help using a product or feature",
          "Customer Service": "general account or service questions",
          "IT Support": "internal IT, devices, access, networks",
          "Billing and Payments": "invoices, charges, refunds, payment methods",
          "Returns and Exchanges": "returning or exchanging an item",
          "Service Outages and Maintenance": "downtime, outages, scheduled maintenance",
          "Sales and Pre-Sales": "pricing, quotes, buying",
          "Human Resources": "employment, payroll, leave, hiring",
          "General Inquiry": "anything else"}
GI = "General Inquiry"
NINE = [q for q in QUEUES if q != GI]


def ticket(r):
    return ("Subject: " + (r["subject"] or "") + "\n" + r["body"].replace("\\n", "\n"))[:3000]


def _cap(pool, seed=13):
    pool = list(pool)
    random.Random(seed).shuffle(pool)
    return pool[:TRAIN_CAP]


def tickets():
    """Training pool and test items for support tickets.

    Test items are (text, options, right option or None).
    """
    from datasets import load_dataset

    d = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    rows = [(i, r) for i, r in enumerate(d) if r.get("language") == "en" and r.get("queue") in QUEUES and r.get("body")]
    triage = rows[:N]
    rest = [(i, r) for i, r in rows[N:] if r["queue"] != GI]
    r8 = random.Random(13)
    made_ans, made_none = [], []
    for i, r in rest[:N]:
        drop = r8.choice([k for k in QUEUES if k not in (r["queue"], GI)])
        made_ans.append((ticket(r), [k for k in QUEUES if k not in (GI, drop)], r["queue"]))
    for i, r in rest[N:2 * N]:
        made_none.append((ticket(r), [k for k in QUEUES if k not in (r["queue"], GI)], None))
    used = {i for i, _ in triage} | {i for i, _ in rest[:2 * N]}
    pool = _cap([(ticket(r), r["queue"]) for i, r in rows if i not in used])

    ans9 = [(ticket(r), NINE, r["queue"]) for _, r in triage if r["queue"] != GI]
    ans9 = [it for it in ans9 if LEN_MIN <= len(it[0]) <= LEN_MAX]
    tests = {"tickets_far": (ans9, [(t, NINE, None) for t in far_texts()]),
             "tickets_made": (made_ans, made_none)}
    return pool, tests


def _as_ticket(text):
    t = re.sub(r"<br\s*/?>", "\n", text).strip()
    first = re.split(r"(?<=[.!?])\s", t, maxsplit=1)[0]
    return "Subject: " + " ".join(first.split()[:8]) + "\n" + t


def far_texts():
    """Real texts written by people, far from customer support, with the length and format of a ticket."""
    from datasets import load_dataset

    rf = random.Random(21)
    imdb = [_as_ticket(r["text"]) for r in load_dataset("stanfordnlp/imdb", split="test")]
    imdb = [t for t in imdb if LEN_MIN <= len(t) <= LEN_MAX]
    rf.shuffle(imdb)
    keep = {"rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey", "sci.space", "sci.med",
            "talk.politics.guns", "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc", "alt.atheism",
            "soc.religion.christian"}
    ng = load_dataset("SetFit/20_newsgroups", split="test")
    news = [_as_ticket(r["text"]) for r in ng if r["label_text"] in keep and len(r["text"].split()) > 20]
    news = [t for t in news if LEN_MIN <= len(t) <= LEN_MAX]
    rf.shuffle(news)
    return imdb[:N_FAR_EACH] + news[:N_FAR_EACH]


def clinc150():
    """Training pool and test items for CLINC150 intents (20 options per case)."""
    from datasets import load_dataset

    try:
        c = load_dataset("clinc_oos", "plus")
    except Exception:
        c = load_dataset("clinc/clinc_oos", "plus")
    lab = c["train"].features["intent"].names
    oos = lab.index("oos")
    name = {i: n.replace("_", " ") for i, n in enumerate(lab)}
    names = sorted(name[i] for i in range(len(lab)) if i != oos)
    pool = _cap([(r["text"], name[r["intent"]]) for r in c["train"] if r["intent"] != oos])

    rc = random.Random(13)
    test = list(c["test"])
    rc.shuffle(test)
    ins = [r for r in test if r["intent"] != oos][:N]
    out = [r for r in test if r["intent"] == oos][:N]

    def opts20(gold=None):
        others = [n for n in names if n != gold]
        o = rc.sample(others, 19 if gold is not None else 20) + ([gold] if gold is not None else [])
        rc.shuffle(o)
        return o

    ans, swapped = [], []
    for r in ins:
        g = name[r["intent"]]
        ans.append((r["text"], opts20(g), g))
        swapped.append((r["text"], rc.sample([n for n in names if n != g], 20), None))
    real = [(r["text"], opts20(), None) for r in out]
    return pool, {"clinc_real": (ans, real), "clinc_made": (ans, swapped)}
