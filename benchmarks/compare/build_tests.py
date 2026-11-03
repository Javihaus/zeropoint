"""Build the eight test sets of the comparison from public datasets, pinned to fixed revisions.

    python build_tests.py            # writes tests/<name>.json and tests/SHA256SUMS

The test files are not in the repository: several source datasets do not allow redistribution, and the
support-ticket dataset is CC BY-NC (fine for testing, never used for training). Everything is fixed by seeds
and dataset revisions, so the files come out the same on every machine; `SHA256SUMS` lets you check that
against the hashes in `results/summary.json`.

Each test is a list of {"text", "options", "gold"}; gold is None when no listed option is right.

No test text is in ZeroPoint's training data: the training texts are loaded only to remove any test text
that appears in them word for word (the counts are in `tests/overlap_report.json`).
The CLINC150 test is in-domain: CLINC150 train is part of the training mix (its test split is not).
"""

import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict

import datasets
import pandas as pd
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import data as bdata  # noqa: E402  (benchmarks/data.py: tickets and CLINC150)

SEED = 13
CAP_BITEXT = 6000
DEV_SHARE = 0.05
TESTS = ["bench2000", "snips", "atis", "mtop_es", "hwu64", "clinc_real_indomain", "tickets_made", "tickets_far"]

# Hugging Face Hub commits of every dataset read here.
REVISIONS = {
    "clinc/clinc_oos": "155b9c710419136e17307b80d0a13e68cd46b4ec",
    "clinc_oos": "155b9c710419136e17307b80d0a13e68cd46b4ec",
    "PolyAI/banking77": "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6",
    "mteb/amazon_massive_intent": "940fd47a81eaa7f2cc7b129674d945d618ac38c2",
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset": "430d1a89bd93bd1fa23c16f29dd53e73f0087443",
    "bitext/Bitext-retail-banking-llm-chatbot-training-dataset": "3e3621092fc6baaf7f53ceb6f091c60ae99acb67",
    "bitext/Bitext-insurance-llm-chatbot-training-dataset": "3ce42aaa134857c47c7ee6ad821be1d1fcf5a75b",
    "bitext/Bitext-telco-llm-chatbot-training-dataset": "eacc593a96d75ec6da0887f081ff0dfc442fccab",
    "benayas/snips": "16915b895754dd4028068abe52b6851a904977d7",
    "DeepPavlov/hwu64": "0dd289ccdeb185ec065d1ebcf5de1c443cd1620f",
    "tuetschek/atis": "d44649dde31b4c9d2104aac42a35ed83956e7dec",
    "mteb/mtop_intent": "9a200bf94b01afa0438a39c54f6a3aceaf78856f",
    "Tobi-Bueck/customer-support-tickets": "ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb",
    "stanfordnlp/imdb": "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
    "SetFit/20_newsgroups": "f1b91292074e7cfb69be58b642d583ec262f30ed",
}

_load = datasets.load_dataset


def load_dataset(repo, *args, **kw):
    return _load(repo, *args, revision=REVISIONS[repo], **kw)


def hub_file(repo, filename):
    return hf_hub_download(repo, filename, repo_type="dataset", revision=REVISIONS[repo])


datasets.load_dataset = load_dataset          # benchmarks/data.py reads through this too, so it is pinned as well


def norm(label: str) -> str:
    s = label.replace("+", " and ").replace("?", "")
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    s = s.replace("_", " ").replace("-", " ").lower()
    return re.sub(r"\s+", " ", s).strip()


def stratified_cap(rows, cap, rng):
    if len(rows) <= cap:
        return rows
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    per = max(1, cap // len(by))
    out = []
    for lab, rs in by.items():
        rng.shuffle(rs)
        out += rs[:per]
    rng.shuffle(out)
    return out[:cap]


def training_texts(rng):
    """The texts of ZeroPoint's training and development data (both are excluded from every test).
    Read in the same order, with the same random draws, as the training run, so the seed state that the
    test sampling below starts from is the one the published test sets were made with."""
    texts = []
    c = load_dataset("clinc/clinc_oos", "plus")
    names = c["train"].features["intent"].names
    oos = names.index("oos")
    texts += [r["text"] for s in ("train", "validation") for r in c[s] if r["intent"] != oos]
    b = load_dataset("PolyAI/banking77")
    tr = list(b["train"])
    rng.shuffle(tr)
    texts += [r["text"] for r in tr]
    for lang in ["en", "es"]:
        m = load_dataset("mteb/amazon_massive_intent", lang)
        texts += [r["text"] for s in ("train", "validation") for r in m[s]]
    for repo in ["bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                 "bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
                 "bitext/Bitext-insurance-llm-chatbot-training-dataset",
                 "bitext/Bitext-telco-llm-chatbot-training-dataset"]:
        d = load_dataset(repo, split="train")
        rows = [{"text": r["instruction"], "label": norm(r["intent"])} for r in d]
        texts += [r["text"] for r in stratified_cap(rows, int(CAP_BITEXT * (1 + DEV_SHARE * 2)), rng)]
    return {t.strip().lower() for t in texts}


def make_items(rows, labels, n_ans, n_none, k_opts, rng):
    """rows: (text, gold). With a right option: gold + (k-1) others. Without: k others, gold left out."""
    pool = sorted(labels)
    rows = list(rows)
    rng.shuffle(rows)
    by = defaultdict(list)
    for t, g in rows:
        by[g].append((t, g))
    cap = max(3, int(3 * (n_ans + n_none) / max(1, len(by))))     # at most 3x the average per intent
    bal = []
    for g, rs in by.items():
        bal += rs[:cap]
    rng.shuffle(bal)
    ans, none = [], []
    for t, g in bal:
        others = [o for o in pool if o != g]
        if len(ans) < n_ans:
            k = min(k_opts, len(pool))
            opts = rng.sample(others, k - 1) + [g]
            rng.shuffle(opts)
            ans.append({"text": t, "options": opts, "gold": g})
        elif len(none) < n_none:
            k = min(k_opts, len(others))
            none.append({"text": t, "options": rng.sample(others, k), "gold": None})
        if len(ans) >= n_ans and len(none) >= n_none:
            break
    return ans + none


def main():
    rng = random.Random(SEED)
    out = os.path.join(HERE, "tests")
    os.makedirs(out, exist_ok=True)
    seen = training_texts(rng)
    report, tests = {}, {}

    def clean(rows, name):
        keep = [(t, g) for t, g in rows if t.strip().lower() not in seen]
        report[name] = {"test_rows": len(rows), "removed_as_in_training": len(rows) - len(keep)}
        return keep

    sn = load_dataset("benayas/snips", split="test")
    rows = clean([(r["text"], norm(r["category"])) for r in sn], "snips")
    tests["snips"] = make_items(rows, {g for _, g in rows}, 500, 500, 7, rng)

    names = pd.read_parquet(hub_file("DeepPavlov/hwu64", "intents/intents-00000-of-00001.parquet"))
    hn = dict(zip(names["id"], names["name"]))
    hw = load_dataset("DeepPavlov/hwu64", split="test")
    rows = clean([(r["utterance"], norm(hn[r["label"]])) for r in hw], "hwu64")
    tests["hwu64"] = make_items(rows, {g for _, g in rows}, 450, 450, 10, rng)

    at = load_dataset("tuetschek/atis", split="test")
    rows = clean([(r["text"], norm(r["intent"])) for r in at], "atis")
    tests["atis"] = make_items(rows, {g for _, g in rows}, 300, 300, 8, rng)

    mt = pd.read_parquet(hub_file("mteb/mtop_intent", "es/test-00000-of-00001.parquet"))
    rows = clean(list(zip(mt["text"], [norm(x) for x in mt["label_text"]])), "mtop_es")
    tests["mtop_es"] = make_items(rows, {g for _, g in rows}, 500, 500, 10, rng)

    _, tt = bdata.tickets()
    for k, (ans, none) in tt.items():
        # 300 tickets with a right queue and 300 without; the tickets left over are kept for the fit() trials
        a = [{"text": t, "options": list(o), "gold": g} for t, o, g in ans]
        n = [{"text": t, "options": list(o), "gold": g} for t, o, g in none]
        r7 = random.Random(7)
        r7.shuffle(a)
        r7.shuffle(n)
        tests[k] = a[:300] + n[:300]
        report[k] = {"test_rows": len(tests[k]),
                     "in_training": sum(it["text"].strip().lower() in seen for it in tests[k])}
    _, tc = bdata.clinc150()
    ans, none = tc["clinc_real"]
    tests["clinc_real_indomain"] = [{"text": t, "options": [norm(x) for x in o], "gold": norm(g) if g else None}
                                    for t, o, g in ans + none]

    with open(os.path.join(HERE, "..", "benchmark_2000", "data.json"), encoding="utf-8") as f:
        b2 = json.load(f)
    tests["bench2000"] = [{"text": c["text"], "options": c["options"], "gold": c["expected_choice"]} for c in b2]
    report["bench2000"] = {"test_rows": len(b2), "in_training": sum(c["text"].strip().lower() in seen for c in b2)}

    sums = []
    for k in TESTS:
        p = os.path.join(out, f"{k}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(tests[k], f, ensure_ascii=False)
        with open(p, "rb") as f:
            sums.append(f"{hashlib.sha256(f.read()).hexdigest()}  {k}.json")
        print(f"{k:22s} {len(tests[k]):5d} cases, {sum(it['gold'] is not None for it in tests[k]):4d} with a right option")
    with open(os.path.join(out, "SHA256SUMS"), "w") as f:
        f.write("\n".join(sums) + "\n")
    with open(os.path.join(out, "overlap_report.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
