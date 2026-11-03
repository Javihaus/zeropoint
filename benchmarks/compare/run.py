"""Run one system on the test sets and save what it decided on every case.

    python run.py SYSTEM [TEST ...]       # all eight tests when none is given

Systems
  zeropoint        ZeroPoint, this repository, default settings
  laya_none        Laya, its recommended Router, with one more option "None of these" on every case
  laya             Laya without that option (it then always picks one of the listed options)
  semantic_router  Semantic Router, one route per option, the option name as its only utterance
  nli              Hugging Face zero-shot classification, DeBERTa-v3-base MNLI, multi_label=True

Every system runs on CPU with 2 threads, one case at a time, in its own process, on the same machine. Each case
is timed from the call with the text and options to the answer, with nothing cached between cases. Three calls
run before the timed cases so that one-off loading is not counted.

Zero-shot NLI runs one pass of a 184M-parameter model per option, so it reads a fixed sample of 300 cases per
test (150 with a right option, 150 without; see `sample()`); the other systems read every case.

Each system also reads the same 300-case sample a second time with the options in another order (`shuffled()`),
to measure how often its decision depends on the order of the options.

Output: out/<system>/<test>.json and out/<system>/<test>.shuffled.json, a list with, for each case:
  i       index of the case in tests/<test>.json
  p       the system's score for every listed option (a probability, except for Semantic Router: a similarity)
  p_none  the probability that no listed option fits (only systems that give one)
  choice  the system's decision at its default settings (null = no option fits)
  ms      time for that case, in milliseconds
"""

import json
import os
import platform
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = ["bench2000", "snips", "atis", "mtop_es", "hwu64", "clinc_real_indomain", "tickets_made", "tickets_far"]
THREADS = 2
NONE = "None of these"

# Hugging Face Hub commits of the models compared.
REVISIONS = {
    "convaiinnovations/laya": "7b928d828b7b0e022f929d9bd2e44165aa270148",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli": "6f5cf0a2b59cabb106aca4c287eed12e357e90eb",
}


def load(test):
    with open(os.path.join(HERE, "tests", f"{test}.json"), encoding="utf-8") as f:
        return json.load(f)


def sample(items, n=300, seed=13):
    """Indices of a fixed sample: n/2 cases with a right option and n/2 without, or every case if fewer."""
    if len(items) <= n:
        return list(range(len(items)))
    r = random.Random(seed)
    pos = [i for i, it in enumerate(items) if it["gold"] is not None]
    neg = [i for i, it in enumerate(items) if it["gold"] is None]
    k = min(n // 2, len(neg))
    return sorted(r.sample(pos, n - k) + r.sample(neg, k))


def shuffled(options, i):
    """The options in another order, fixed per case: a seeded shuffle, rotated if it came out unchanged."""
    o = list(options)
    random.Random(1000 + i).shuffle(o)
    return o if o != list(options) else o[1:] + o[:1]


def asks(test):
    """How Laya is asked, as in the Laya audit: a ticket for the ticket tests, an utterance otherwise."""
    if test.startswith("tickets"):
        return "ticket", "Which support queue should handle this ticket?"
    return "utterance", "Which intent does `utterance` express?"


# ------------------------------------------------------------------ the systems
# Each one is built once per test by a function that returns decide(text, options, test) -> row.

def build_zeropoint(test, items):
    from zeropoint import ZeroPoint
    zp = ZeroPoint(threads=THREADS)

    def decide(text, options):
        d = zp.decide(text, options)
        return {"p": dict(d.probabilities), "p_none": d.p_none, "choice": d.choice}
    return decide


_laya = None


def build_laya(with_none):
    def build(test, items):
        global _laya
        import laya
        if _laya is None:
            _laya = laya.Router(device="cpu", revision=REVISIONS["convaiinnovations/laya"])
        key, instructions = asks(test)

        def decide(text, options):
            crit = {o: None for o in options + ([NONE] if with_none else [])}
            a = _laya.predict({key: text}, {"q": {"type": "choice", "instructions": instructions, "criteria": crit}})
            pr = {o: float(v) for o, v in a["answers"]["q"]["probabilities"].items()}
            p = {o: pr[o] for o in options}
            choice = max(pr, key=pr.get)                      # Laya's answer: its most probable option
            return {"p": p, "p_none": pr[NONE] if with_none else None,
                    "choice": None if choice == NONE else choice, "laya_model": a["routing"]["model"]}
        return decide
    return build


_encoder = None


def build_semantic_router(test, items):
    """The router a user would build for this test: one route per option name, nothing else. The library's own
    index query runs with `route_filter` set to the case's options; route scores are aggregated as IndexedRouter
    does (mean of the route's hits among the top_k nearest utterances, 0.0 for a route with no hit). The library's
    default has no score threshold, so it always picks a route."""
    global _encoder
    import numpy as np
    from semantic_router import Route
    from semantic_router.encoders import HuggingFaceEncoder
    from semantic_router.routers import IndexedRouter
    rev = REVISIONS["sentence-transformers/all-MiniLM-L6-v2"]
    if _encoder is None:
        _encoder = HuggingFaceEncoder(tokenizer_kwargs={"revision": rev}, model_kwargs={"revision": rev})
    labels = sorted({o for it in items for o in it["options"]})
    router = IndexedRouter(encoder=_encoder, routes=[Route(name=x, utterances=[x]) for x in labels],
                           top_k=5, aggregation="mean")

    def decide(text, options):
        v = np.array(_encoder([text])[0])
        scores, names = router.index.query(vector=v, top_k=router.top_k, route_filter=list(options))
        hits = {o: [] for o in options}
        for n, s in zip(names, scores):
            hits[n].append(float(s))
        p = {o: (float(router.aggregate(h)) if h else 0.0) for o, h in hits.items()}
        return {"p": p, "p_none": None, "choice": max(p, key=p.get)}
    return decide


_nli = None


def build_nli(test, items):
    global _nli
    from transformers import pipeline
    mid = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    if _nli is None:
        _nli = pipeline("zero-shot-classification", model=mid, revision=REVISIONS[mid], device=-1)

    def decide(text, options):
        r = _nli(text, candidate_labels=list(options), multi_label=True)
        p = {lab: float(s) for lab, s in zip(r["labels"], r["scores"])}
        # multi_label=True scores each option on its own (entailment against contradiction); the usual default
        # is to take the top option, so it never says that no option fits
        return {"p": p, "p_none": None, "choice": max(p, key=p.get)}
    return decide


SYSTEMS = {
    "zeropoint": (build_zeropoint, False),
    "laya_none": (build_laya(True), False),
    "laya": (build_laya(False), False),
    "semantic_router": (build_semantic_router, False),
    "nli": (build_nli, True),           # True: the 300-case sample only
}


def environment():
    import importlib.metadata as md
    info = {"python": platform.python_version(), "machine": platform.machine(), "system": platform.system(),
            "threads": THREADS, "cpus": os.cpu_count()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    for p in ["zeropoint", "onnxruntime", "numpy", "laya", "semantic-router", "transformers", "torch"]:
        try:
            info[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return info


def run(system, tests):
    build, sample_only = SYSTEMS[system]
    try:                                                      # torch-based systems: the same 2 threads
        import torch
        torch.set_num_threads(THREADS)
        torch.set_num_interop_threads(1)
    except ImportError:
        pass
    out = os.path.join(HERE, "out", system)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "environment.json"), "w") as f:
        json.dump(environment(), f, indent=1)
    for test in tests:
        path = os.path.join(out, f"{test}.json")
        if os.path.exists(path) and os.path.exists(path.replace(".json", ".shuffled.json")):
            print(f"{system} {test}: already done", flush=True)
            continue
        items = load(test)
        decide = build(test, items)
        for it in items[:3]:                                  # warm-up, not timed, not saved
            decide(it["text"], it["options"])
        sam = sample(items)
        for name, idx, order in [("", sam if sample_only else range(len(items)), False),
                                 (".shuffled", sam, True)]:
            rows, t0 = [], time.time()
            for n, i in enumerate(idx):
                it = items[i]
                opts = shuffled(it["options"], i) if order else it["options"]
                s = time.perf_counter()
                row = decide(it["text"], opts)
                row["ms"] = (time.perf_counter() - s) * 1000
                row["i"] = i
                rows.append(row)
                if n % 200 == 0:
                    print(f"{system} {test}{name} {n}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
            with open(os.path.join(out, f"{test}{name}.json"), "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False)
            print(f"{system} {test}{name}: {len(rows)} cases, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2:] or TESTS)
