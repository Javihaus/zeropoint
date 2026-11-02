"""Run every system on the 2,000 cases and save, per case, one score per option and the default choice.

Usage: python run_systems.py SYSTEM [SYSTEM ...]
Systems: zeropoint, zeropoint_k025, laya, sr_names, sr_fewshot, nli_deberta, nli_bart
(nli_bart is included but was not run for the published results: too slow on CPU.)

Output: out/<system>.json, a list of {"id", "scores": {option: score}, "choice", "conf", "ms"}.
  scores  the system's score for each listed option (higher = more likely right)
  choice  the system's own decision at its default settings (None = it says no option fits)
  conf    the system's confidence that one of the listed options is right; used for AUROC and FPR95
"""

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "data.json")))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
SEED = 13

# Model revisions behind the published results (Hugging Face Hub commits).
REVISIONS = {
    "convaiinnovations/laya": "7b928d828b7b0e022f929d9bd2e44165aa270148",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli": "6f5cf0a2b59cabb106aca4c287eed12e357e90eb",
}


def save(name, rows):
    json.dump(rows, open(os.path.join(OUT, f"{name}.json"), "w"))
    print(f"{name}: {len(rows)} cases saved", flush=True)


# ------------------------------------------------------------------ ZeroPoint 0.1.0 (PyPI), small encoder, int8
def run_zeropoint(kappa=None):
    from zeropoint import ZeroPoint
    models = {ls: ZeroPoint.load(ls) for ls in ("everyday", "support")}
    if kappa is not None:
        for m in models.values():
            m.kappa = kappa
    rows = []
    for c in DATA:
        d = models[c["landscape"]].decide(c["text"], options=c["options"])
        rows.append({"id": c["id"], "scores": d.probabilities, "choice": d.choice, "conf": 1 - d.p_none,
                     "ms": d.latency_ms})
    return rows


# ------------------------------------------------------------------ Laya (its recommended Router), as in the Laya audit
def run_laya():
    import laya
    router = laya.Router(device="cpu", revision=REVISIONS["convaiinnovations/laya"])
    ask = {"everyday": ("utterance", "Which intent does `utterance` express?"),
           "support": ("ticket", "Which support queue should handle this ticket?")}
    rows = []
    for k, c in enumerate(DATA):
        key, instructions = ask[c["landscape"]]
        q = {"type": "choice", "instructions": instructions, "criteria": {o: None for o in c["options"]}}
        t0 = time.perf_counter()
        a = router.predict({key: c["text"]}, {"q": q})["answers"]["q"]
        ms = (time.perf_counter() - t0) * 1000
        p = {o: float(a["probabilities"][o]) for o in c["options"]}
        rows.append({"id": c["id"], "scores": p, "choice": max(p, key=p.get), "conf": max(p.values()), "ms": ms,
                     "answer_confidence": float(a.get("answer_confidence", float("nan")))})
        if k % 200 == 0:
            print(f"  laya {k}/{len(DATA)}", flush=True)
    return rows


# ------------------------------------------------------------------ Semantic Router (Aurelio AI), local HuggingFace encoder
def _fewshot_utterances(n=10):
    """n training examples per option: CLINC150 train split for everyday, the ticket dataset for support."""
    from datasets import load_dataset
    rng = np.random.default_rng(SEED)
    out = {}
    c = load_dataset("clinc/clinc_oos", "plus")["train"]
    names = c.features["intent"].names
    by = defaultdict(list)
    for r in c:
        by[names[r["intent"]].replace("_", " ")].append(r["text"])
    for lab, texts in by.items():
        out[lab] = [texts[i] for i in rng.choice(len(texts), min(n, len(texts)), replace=False)]
    t = load_dataset("Tobi-Bueck/customer-support-tickets")["train"]
    by = defaultdict(list)
    for r in t:
        if r["language"] == "en":
            by[r["queue"]].append(f"{r['subject'] or ''}\n{r['body'] or ''}".strip()[:1000])
    for lab, texts in by.items():
        out[lab] = [texts[i] for i in rng.choice(len(texts), min(n, len(texts)), replace=False)]
    return out


def run_semantic_router(fewshot):
    """Each case gets the router a user would build: one route per listed option, nothing else.

    The utterances of every option are embedded once; for each case the library's own index query runs with
    `route_filter` set to the case's options, and route scores are aggregated as IndexedRouter does (mean of
    the route's hits among the top_k nearest utterances; 0.0 for a route with no hit).
    """
    import numpy as np
    from semantic_router import Route
    from semantic_router.encoders import HuggingFaceEncoder
    from semantic_router.routers import IndexedRouter

    rev = REVISIONS["sentence-transformers/all-MiniLM-L6-v2"]
    encoder = HuggingFaceEncoder(tokenizer_kwargs={"revision": rev}, model_kwargs={"revision": rev})  # library default model
    examples = _fewshot_utterances() if fewshot else {}
    rows = {}
    for ls in ("everyday", "support"):
        cases = [c for c in DATA if c["landscape"] == ls]
        labels = sorted({o for c in cases for o in c["options"]})
        routes = [Route(name=lab, utterances=examples.get(lab, [lab]) if fewshot else [lab]) for lab in labels]
        router = IndexedRouter(encoder=encoder, routes=routes, top_k=5, aggregation="mean")
        t0 = time.perf_counter()
        vecs = encoder([c["text"] for c in cases])
        for c, v in zip(cases, vecs):
            scores, names = router.index.query(vector=np.array(v), top_k=router.top_k, route_filter=c["options"])
            hits = {o: [] for o in c["options"]}
            for n, sc in zip(names, scores):
                hits[n].append(float(sc))
            p = {o: (router.aggregate(h) if h else 0.0) for o, h in hits.items()}
            # the library's default router has no confidence threshold, so it always picks the top route
            rows[c["id"]] = {"id": c["id"], "scores": p, "choice": max(p, key=p.get), "conf": max(p.values())}
        ms = (time.perf_counter() - t0) * 1000 / len(cases)
        for c in cases:
            rows[c["id"]]["ms"] = ms
    return [rows[c["id"]] for c in DATA]


# ------------------------------------------------------------------ Hugging Face zero-shot NLI
def run_nli(model_id):
    from transformers import pipeline
    clf = pipeline("zero-shot-classification", model=model_id, revision=REVISIONS.get(model_id), device=-1)
    rows = []
    cache = {}
    for k, c in enumerate(DATA):
        need = [o for o in c["options"] if (c["text"], o) not in cache]
        t0 = time.perf_counter()
        if need:
            r = clf(c["text"], candidate_labels=need, multi_label=True)
            for lab, s in zip(r["labels"], r["scores"]):
                cache[(c["text"], lab)] = float(s)
        ms = (time.perf_counter() - t0) * 1000
        p = {o: cache[(c["text"], o)] for o in c["options"]}
        # multi_label=True scores each option on its own (entailment vs contradiction); the usual
        # default is to accept the top option, so the default choice never abstains
        rows.append({"id": c["id"], "scores": p, "choice": max(p, key=p.get), "conf": max(p.values()), "ms": ms})
        if k % 200 == 0:
            print(f"  {model_id} {k}/{len(DATA)}", flush=True)
    return rows


SYSTEMS = {
    "zeropoint": lambda: run_zeropoint(),
    "zeropoint_k025": lambda: run_zeropoint(kappa=0.25),
    "laya": run_laya,
    "sr_names": lambda: run_semantic_router(False),
    "sr_fewshot": lambda: run_semantic_router(True),
    "nli_bart": lambda: run_nli("facebook/bart-large-mnli"),
    "nli_deberta": lambda: run_nli("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"),
}

if __name__ == "__main__":
    for name in sys.argv[1:]:
        if os.path.exists(os.path.join(OUT, f"{name}.json")):
            print(f"{name}: already done", flush=True)
            continue
        t0 = time.time()
        print(f"{name}: start", flush=True)
        save(name, SYSTEMS[name]())
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)
