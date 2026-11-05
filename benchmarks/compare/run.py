"""Run one system on the test sets and save what it decided on every case.

    python run.py SYSTEM [TEST ...]       # all eight tests when none is given

Systems
  zeropoint        ZeroPoint, this repository, default settings
  laya_none        Laya, its recommended Router, with one more option "None of these" on every case
  semantic_router  Semantic Router, one route per option, the option name as its only utterance
  nli              Hugging Face zero-shot classification, DeBERTa-v3-base MNLI, multi_label=True
  strands_none     AWS Strands Decider 2B (Qwen3.5-2B base), with one more option "None of these", like laya_none
  clef_flash_none  Cloudflare Clef-flash (9B, Qwen3.5-9B base), the same way

Strands Decider and Clef-flash run on the device in DECIDER_DEVICE ("cpu" by default; "cuda" on a GPU).

Every system runs on CPU with 2 threads, one case at a time, in its own process, on the same machine. Each case
is timed from the call with the text and options to the answer, with nothing cached between cases. Three calls
run before the timed cases so that one-off loading is not counted.

Zero-shot NLI runs one pass of a 184M-parameter model per option, so it reads a fixed sample of 300 cases per
test (150 with a right option, 150 without; see `sample()`); the other systems read every case. Strands Decider
and Clef-flash also read the 300-case sample, and only on LLM_TESTS: Strands Decider's training data includes CLINC150
and 20 Newsgroups, Clef-flash reports results on CLINC150 and BANKING77, and the other public tests are as likely to be
in their training data as in an LLM's.

Each system also reads the same 300-case sample a second time with the options in another order (`shuffled()`),
to measure how often its decision depends on the order of the options.

Output: out/<system>/<test>.json and out/<system>/<test>.shuffled.json, a list with, for each case:
  i       index of the case in tests/<test>.json
  p       the system's score for every listed option (a probability, except for Semantic Router: a similarity)
  p_none  the probability that no listed option fits (only systems that give one)
  choice  the system's decision at its default settings (null = no option fits)
  ms      time for that case, in milliseconds
"""

import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = ["bench2000", "snips", "atis", "mtop_es", "hwu64", "clinc_real_indomain", "tickets_made", "tickets_far"]
# Hosted LLMs (benchmarks/llm) are compared on these three only: the other five tests are public since 1990-2020 and
# very likely in their training data.
LLM_TESTS = ["bench2000", "tickets_made", "tickets_far"]
THREADS = 2
NONE = "None of these"

# Hugging Face Hub commits of the models compared.
REVISIONS = {
    "convaiinnovations/laya": "7b928d828b7b0e022f929d9bd2e44165aa270148",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli": "6f5cf0a2b59cabb106aca4c287eed12e357e90eb",
    "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610": "216d1ccb46a9e9b036eb9187b248db98480f77d4",
    "Cloudflare/clef-flash": "fb4417d31c1d0f9e7e77f1141b40aaede7491e00",
}
DEVICE = os.environ.get("DECIDER_DEVICE", "cpu")


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


def systemone_decide(ask, test):
    """Strands Decider and Clef take the same request as Laya (the System One format of Jev): a state and a `choice`
    question. They are asked exactly as Laya is: the same instructions, and the options plus "None of these"."""
    key, instructions = asks(test)

    def decide(text, options):
        crit = {o: None for o in options + [NONE]}
        pr = ask({key: text}, {"q": {"type": "choice", "instructions": instructions, "criteria": crit}})
        choice = max(pr, key=pr.get)
        return {"p": {o: pr[o] for o in options}, "p_none": pr[NONE], "choice": None if choice == NONE else choice}
    return decide


_strands = None


def build_strands(test, items):
    """strands-decider 0.2.0. On CPU the base model (Qwen3.5-2B-Base, the revision the checkpoint's provenance.json
    pins) is kept in bfloat16: the library would cast it to float32, which needs about 9 GB of memory. Its authors
    report 106 ms (median) on an RTX 3090."""
    global _strands
    if _strands is None:
        from huggingface_hub import snapshot_download
        from strands_decider import infer
        repo = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
        if DEVICE == "cpu":
            infer.SystemOneEngine._upcast_torso_for_cpu = lambda self: None
        _strands = infer.load_engine(snapshot_download(repo, revision=REVISIONS[repo]), device=DEVICE)

    def ask(state, questions):
        a = _strands.ask(state, questions)
        return {o: float(v) for o, v in a.answers["q"].probabilities.items()}
    return systemone_decide(ask, test)


_clef = None


def build_clef_flash(test, items):
    """Clef-flash with the code its Hugging Face release ships (joint_schema_model.py, `systemone`), in bfloat16."""
    global _clef
    repo = "Cloudflare/clef-flash"
    if _clef is None:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo, revision=REVISIONS[repo])
        sys.path.insert(0, path)
        from joint_schema_model import load_release_model, systemone
        model, processor = load_release_model(path, device=DEVICE)
        _clef = (model, processor, systemone)
    model, processor, systemone = _clef

    def ask(state, questions):
        a = systemone(model, processor, {"model": "clef-flash", "state": state, "questions": questions})
        return {o: float(v) for o, v in a["answers"]["q"]["probabilities"].items()}
    return systemone_decide(ask, test)


SYSTEMS = {
    "zeropoint": (build_zeropoint, False),
    "laya_none": (build_laya(True), False),
    "semantic_router": (build_semantic_router, False),
    "nli": (build_nli, True),           # True: the 300-case sample only
    "strands_none": (build_strands, True),
    "clef_flash_none": (build_clef_flash, True),
}


def has(module):
    """True when `module` can be imported, without importing it."""
    return importlib.util.find_spec(module) is not None


def set_torch_threads():
    """Torch-based systems use the same 2 threads as the others."""
    if has("torch"):
        import torch
        torch.set_num_threads(THREADS)
        torch.set_num_interop_threads(1)


def cpu_name():
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or None


def environment():
    info = {"python": platform.python_version(), "machine": platform.machine(), "system": platform.system(),
            "threads": THREADS, "cpus": os.cpu_count(), "device": DEVICE}
    if has("torch"):
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    if cpu_name():
        info["cpu"] = cpu_name()
    if has("zeropoint"):                   # the code actually imported, which may not be the installed copy
        from zeropoint import __version__ as zeropoint_code
        info["zeropoint_code"] = zeropoint_code
    installed = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in importlib.metadata.distributions()}
    for p in ["zeropoint", "onnxruntime", "numpy", "laya", "semantic-router", "strands-decider", "transformers",
              "peft", "torch"]:
        if p in installed:
            info[p] = installed[p]
    return info


def run(system, tests):
    build, sample_only = SYSTEMS[system]
    set_torch_threads()
    out = os.path.join(HERE, "out", system)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "environment.json"), "w") as f:
        json.dump(environment(), f, indent=1)
    for test in tests:
        path = os.path.join(out, f"{test}.json")
        done = [p for p in (path, path.replace(".json", ".shuffled.json"))
                if os.path.exists(p) or os.path.exists(p + ".gz")]
        if len(done) == 2:
            print(f"{system} {test}: already done", flush=True)
            continue
        items = load(test)
        decide = build(test, items)
        for it in items[:3]:                                  # warm-up, not timed, not saved
            decide(it["text"], it["options"])
        sam = sample(items)
        for name, idx, order in [("", sam if sample_only else range(len(items)), False),
                                 (".shuffled", sam, True)]:
            final = os.path.join(out, f"{test}{name}.json")
            if os.path.exists(final) or os.path.exists(final + ".gz"):
                continue
            partial = os.path.join(out, f"{test}{name}.partial.jsonl")   # cases are saved as they come, so an
            rows = []                                                     # interrupted run continues where it stopped
            if os.path.exists(partial):
                with open(partial, encoding="utf-8") as f:
                    rows = [json.loads(line) for line in f if line.strip()]
            done, t0 = {r["i"] for r in rows}, time.time()
            with open(partial, "a", encoding="utf-8") as pf:
                for n, i in enumerate(idx):
                    if i in done:
                        continue
                    it = items[i]
                    opts = shuffled(it["options"], i) if order else it["options"]
                    s = time.perf_counter()
                    row = decide(it["text"], opts)
                    row["ms"] = (time.perf_counter() - s) * 1000
                    row["i"] = i
                    rows.append(row)
                    pf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    pf.flush()
                    if n % 50 == 0:
                        print(f"{system} {test}{name} {n}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
            os.makedirs(out, exist_ok=True)                   # in case the folder was removed during the run
            rows = sorted({r["i"]: r for r in rows}.values(), key=lambda r: list(idx).index(r["i"]))
            with open(final, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False)
            os.remove(partial)
            print(f"{system} {test}{name}: {len(rows)} cases, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2:] or TESTS)
