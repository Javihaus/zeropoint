"""Score what every system decided (out/) and write results/summary.json and results/tables.md.

    python report.py

Measures, per system and test
  choice          on cases with a right option: the option the system scores highest is the right one
  decision        on every case: the system's own decision at its default settings is right (on a case with no
                  right option, only "no option fits" is right; a system that always picks one gets those wrong)
  sure_wrong      on cases with no right option: the system picks an option with probability 0.9 or more
                  (only systems that give probabilities)
  fpr95           on cases with no right option: share still acted on at the threshold that keeps 95% of the
                  cases with a right option (lower is better; any score works, so every system has one)
  auroc           how well the system's score separates cases with a right option from cases without
  order_changed   on the 300-case sample: the decision changes when the same options come in another order
  ms              time per case, median and 95th percentile
Intervals: 95% bootstrap (2,000 resamples of the cases).

The score used for fpr95 and auroc is 1 - p(none) for systems that give p(none), the top option's score otherwise.
"""

import gzip
import hashlib
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = ["bench2000", "snips", "atis", "mtop_es", "hwu64", "clinc_real_indomain", "tickets_made", "tickets_far"]
TEST_ABOUT = {
    "bench2000": "everyday requests and support tickets, with hard wording, typos and unrelated texts",
    "snips": "voice-assistant requests (English)",
    "atis": "airline travel questions (English)",
    "mtop_es": "assistant requests in Spanish",
    "hwu64": "home-assistant requests (English)",
    "clinc_real_indomain": "banking, travel and everyday requests; cases with no right option are real out-of-scope "
                           "questions. ZeroPoint was trained on other texts of this dataset (its train split)",
    "tickets_made": "support tickets; for cases with no right option, the ticket's own queue is removed from the list",
    "tickets_far": "support queues; cases with no right option are film reviews and newsgroup posts",
}
TEST_NAMES = {"bench2000": "2,000-case benchmark", "snips": "SNIPS", "atis": "ATIS", "mtop_es": "MTOP (Spanish)",
              "hwu64": "HWU64", "clinc_real_indomain": "CLINC150 (20 options)", "tickets_made": "Tickets, queue removed",
              "tickets_far": "Tickets, unrelated texts"}
SYSTEMS = ["zeropoint", "laya_none", "laya", "semantic_router", "nli", "strands_none"]
NAMES = {"zeropoint": "ZeroPoint 1.0", "laya_none": "Laya + \"None of these\"", "laya": "Laya",
         "semantic_router": "Semantic Router", "nli": "Zero-shot NLI (DeBERTa-v3-base)",
         "strands_none": "Strands Decider 2B + \"None of these\"",
         "clef_flash_none": "Clef-flash + \"None of these\""}
# Systems run only on LLM_TESTS: they are left out of the eight-test tables and shown in the LLM comparison.
LLM_TESTS_ONLY = ["strands_none", "clef_flash_none"]
PAIRED_WITH = ("laya_none", "laya")          # systems compared with ZeroPoint case by case
PROBABILITIES = {"zeropoint": True, "laya_none": True, "laya": True, "semantic_router": False, "nli": True,
                 "strands_none": True, "clef_flash_none": True}
TOGETHER = {"zeropoint": "yes", "laya_none": "yes", "laya": "yes", "semantic_router": "no", "nli": "no",
            "strands_none": "yes", "clef_flash_none": "yes"}
B = 2000
EXTRA_PAIRED = set()
from run import LLM_TESTS  # noqa: E402  (hosted LLMs are compared on these tests only, see run.py)
LLM_BASELINES = ["zeropoint", "laya_none"]


def extra_systems():
    """Other systems found in out/, each described by its own system.json (written by benchmarks/llm/run_llm.py):
    {"name": ..., "probabilities": true|false, "reads_options_together": "yes"|"no", "paired": true|false}."""
    out = []
    root = os.path.join(HERE, "out")
    for s in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        f = os.path.join(root, s, "system.json")
        if s in SYSTEMS or not os.path.exists(f):
            continue
        meta = read_json(f)
        NAMES[s] = meta["name"]
        PROBABILITIES[s] = bool(meta.get("probabilities", True))
        TOGETHER[s] = meta.get("reads_options_together", "yes")
        if meta.get("paired"):
            EXTRA_PAIRED.add(s)
        out.append(s)
    return out


def model_size(system, rows_by_test):
    """Bytes of model weights the system read in this run."""
    from huggingface_hub import hf_hub_download
    from run import REVISIONS

    def hub(repo, f):
        return os.path.getsize(hf_hub_download(repo, f, revision=REVISIONS[repo], local_files_only=True))
    try:
        if system == "zeropoint":
            import zeropoint
            d = os.path.join(os.path.dirname(zeropoint.__file__), "model")
            return sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
        if system.startswith("laya"):
            used = {r.get("laya_model") for rows in rows_by_test.values() for r in rows}
            files = {"english": "model.safetensors", "multilingual": "multilingual/model.safetensors"}
            return sum(hub("convaiinnovations/laya", files[m]) for m in used if m in files)
        if system == "semantic_router":
            return hub("sentence-transformers/all-MiniLM-L6-v2", "model.safetensors")
        if system == "strands_none":
            rev = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"        # Qwen3.5-2B-Base, as the checkpoint's provenance.json pins it
            base = sum(os.path.getsize(hf_hub_download("Qwen/Qwen3.5-2B-Base", f, revision=rev, local_files_only=True))
                       for f in ["model.safetensors-00001-of-00001.safetensors"])
            repo = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
            return base + sum(os.path.getsize(hf_hub_download(repo, f, revision=REVISIONS[repo], local_files_only=True))
                              for f in ["lora/adapter_model.safetensors", "head.safetensors"])
        if system == "clef_flash_none":
            from huggingface_hub import snapshot_download
            repo = "Cloudflare/clef-flash"
            d = snapshot_download(repo, revision=REVISIONS[repo], local_files_only=True)
            return sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".safetensors"))
        if system == "nli":
            return hub("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "model.safetensors")
    except Exception:                                          # weights not on this machine: leave it out
        return None


def read_rows(path):
    """out/ files are kept gzipped in the repository; a fresh run writes plain JSON. Either is read."""
    if os.path.exists(path):
        return read_json(path)
    if os.path.exists(path + ".gz"):
        with gzip.open(path + ".gz", "rt", encoding="utf-8") as f:
            return json.load(f)
    return None


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fpr95(y, s):
    th = np.quantile(s[y == 1], 0.05, method="lower")
    return float(np.mean(s[y == 0] >= th))


def ci(values_fn, n, rng):
    stats = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = values_fn(idx)
        if v is not None:
            stats.append(v)
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))] if stats else None


def score(items, rows, rng, probabilities=True):
    gold = [items[r["i"]]["gold"] for r in rows]
    y = np.array([g is not None for g in gold])
    probs = rows[0]["p_none"] is not None
    top = [max(r["p"], key=r["p"].get) for r in rows]
    conf = np.array([1 - r["p_none"] if probs else max(r["p"].values()) for r in rows])
    choice_ok = np.array([t == g for t, g in zip(top, gold)])
    decision_ok = np.array([r["choice"] == g for r, g in zip(rows, gold)])
    out = {"cases": len(rows), "with_right_option": int(y.sum()),
           "choice": float(choice_ok[y].mean()), "decision": float(decision_ok.mean())}
    out["choice_ci"] = ci(lambda i: choice_ok[i][y[i]].mean() if y[i].any() else None, len(rows), rng)
    sure = np.array([r["choice"] is not None and r["p"][r["choice"]] >= 0.9 for r in rows])
    if (~y).any():
        out["fpr95"] = fpr95(y.astype(int), conf)
        out["auroc"] = float(roc_auc_score(y.astype(int), conf))
        if probabilities:
            out["sure_wrong"] = float(sure[~y].mean())
            out["sure_wrong_ci"] = ci(lambda i: sure[i][~y[i]].mean() if (~y[i]).any() else None, len(rows), rng)
    ms = np.array([r["ms"] for r in rows])
    out["ms_median"], out["ms_p95"] = float(np.median(ms)), float(np.percentile(ms, 95))
    return out


def paired(items, a_rows, b_rows, rng):
    """ZeroPoint minus another system, case by case on the same cases, with a 95% paired bootstrap interval
    (2,000 resamples), for choice (cases with a right option) and sure_wrong (cases without one)."""
    a = {r["i"]: r for r in a_rows}
    b = {r["i"]: r for r in b_rows}
    common = sorted(set(a) & set(b))
    top = lambda r: max(r["p"], key=r["p"].get)
    sure = lambda r: r["choice"] is not None and r["p"][r["choice"]] >= 0.9
    out = {}
    pos = [i for i in common if items[i]["gold"] is not None]
    neg = [i for i in common if items[i]["gold"] is None]
    for key, idx, f in [("choice", pos, lambda r, i: top(r) == items[i]["gold"]),
                        ("sure_wrong", neg, lambda r, i: sure(r))]:
        if not idx:
            continue
        d = np.array([float(f(a[i], i)) - float(f(b[i], i)) for i in idx])
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
        out[key] = {"diff": float(d.mean()), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                    "cases": len(idx)}
    return out


def order_changed(rows, shuffled_rows):
    a = {r["i"]: r["choice"] for r in rows}
    b = {r["i"]: r["choice"] for r in shuffled_rows}
    common = sorted(set(a) & set(b))
    return float(np.mean([a[i] != b[i] for i in common])), len(common)


def llm_comparison(tests, systems):
    """Every system on the same cases: the 300-case sample of each test in LLM_TESTS. Right decision (none counted as
    an answer), silent mistakes (an option picked with probability or stated confidence >= 0.9 when none is right),
    latency, and ZeroPoint minus each system case by case with a 95% paired bootstrap interval (2,000 resamples)."""
    from run import sample
    cases = {t: sample(tests[t]) for t in LLM_TESTS}

    def rows_of(s):
        return {t: {r["i"]: r for r in read_rows(os.path.join(HERE, "out", s, f"{t}.json"))} for t in LLM_TESTS}

    def right(t, r, i):
        return float(r["choice"] == tests[t][i]["gold"])

    def silent(r):
        return float(r["choice"] is not None and r["p"][r["choice"]] >= 0.9)

    def interval(d, seed=13):
        rng = np.random.default_rng(seed)
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
        return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    zp = rows_of("zeropoint")
    out = {"tests": LLM_TESTS, "cases_per_test": {t: len(cases[t]) for t in LLM_TESTS}, "systems": {}}
    for s in systems:
        R = rows_of(s)
        all_cases = [(t, i) for t in LLM_TESTS for i in cases[t]]
        missing = [(t, i) for t, i in all_cases if i not in R[t]]
        if missing:
            raise SystemExit(f"{s}: {len(missing)} cases of the sample have no answer; run it again to finish them")
        neg = [(t, i) for t, i in all_cases if tests[t][i]["gold"] is None]
        ms = np.array([R[t][i]["ms"] for t, i in all_cases])
        res = {"right": float(np.mean([right(t, R[t][i], i) for t, i in all_cases])),
               "silent": float(np.mean([silent(R[t][i]) for t, i in neg])),
               "ms_median": float(np.median(ms)), "ms_p95": float(np.percentile(ms, 95)),
               "cases": len(all_cases),
               "invalid": int(sum(bool(R[t][i].get("invalid")) for t, i in all_cases)),
               "no_tool_call": int(sum(bool((R[t][i].get("stated") or {}).get("no_tool_call")) for t, i in all_cases)),
               "per_test": {t: {"right": float(np.mean([right(t, R[t][i], i) for i in cases[t]])),
                                "silent": float(np.mean([silent(R[t][i]) for i in cases[t]
                                                         if tests[t][i]["gold"] is None]))} for t in LLM_TESTS}}
        tin = [R[t][i].get("tokens_in") for t, i in all_cases if R[t][i].get("tokens_in") is not None]
        if tin:
            res["tokens_in"] = float(np.mean(tin))
            res["tokens_out"] = float(np.mean([R[t][i]["tokens_out"] for t, i in all_cases]))
        env = os.path.join(HERE, "out", s, "environment.json")
        res["mode"] = read_json(env).get("mode") if os.path.exists(env) else None
        if s != "zeropoint":
            d1 = np.array([right(t, zp[t][i], i) - right(t, R[t][i], i) for t, i in all_cases])
            d2 = np.array([silent(zp[t][i]) - silent(R[t][i]) for t, i in neg])
            res["zeropoint_minus"] = {"right": {"diff": float(d1.mean()), "ci": interval(d1)},
                                      "silent": {"diff": float(d2.mean()), "ci": interval(d2)}}
        out["systems"][s] = res
    return out


def options_range(items):
    n = [len(it["options"]) for it in items]
    return str(min(n)) if min(n) == max(n) else f"{min(n)}–{max(n)}"


def main():
    from run import sample
    rng = np.random.default_rng(13)
    tests = {t: read_json(os.path.join(HERE, "tests", f"{t}.json")) for t in TESTS}
    summary = {"tests": {t: {"cases": len(tests[t]),
                             "with_right_option": sum(it["gold"] is not None for it in tests[t]),
                             "options": options_range(tests[t]),
                             "sha256": sha256(os.path.join(HERE, "tests", f"{t}.json"))} for t in TESTS},
               "systems": {}}
    for s in SYSTEMS + extra_systems():
        d = os.path.join(HERE, "out", s)
        if not os.path.isdir(d):
            continue
        res = {"name": NAMES[s], "reads_options_together": TOGETHER[s], "tests": {}}
        rows_by_test = {}
        for t in TESTS:
            p = os.path.join(d, f"{t}.json")
            rows = read_rows(p)
            if rows is None:
                continue
            rows_by_test[t] = rows
            r = score(tests[t], rows, rng, PROBABILITIES[s])
            sam = set(sample(tests[t]))
            rs = score(tests[t], [x for x in rows if x["i"] in sam], rng, PROBABILITIES[s])
            r["on_sample"] = {k: v for k, v in rs.items() if not k.startswith("_") and not k.endswith("_ci")}
            shuffled_rows = read_rows(p.replace(".json", ".shuffled.json"))
            if shuffled_rows is not None:
                r["order_changed"], r["order_cases"] = order_changed(rows, shuffled_rows)
            res["tests"][t] = r
            if s in PAIRED_WITH or s in EXTRA_PAIRED:
                zp_rows = read_rows(os.path.join(HERE, "out", "zeropoint", f"{t}.json"))
                if zp_rows is not None:
                    r["zeropoint_minus"] = paired(tests[t], zp_rows, rows, np.random.default_rng(13))
        if not res["tests"]:
            continue
        all_ms = np.concatenate([[x["ms"] for x in rows] for rows in rows_by_test.values()])
        res["ms_median_all"] = float(np.median(all_ms))
        res["model_bytes"] = model_size(s, rows_by_test)
        costs = [x["cost_usd"] for rows in rows_by_test.values() for x in rows if x.get("cost_usd") is not None]
        if costs:
            res["cost_usd_per_decision"] = float(np.mean(costs))
        env = os.path.join(d, "environment.json")
        res["environment"] = read_json(env) if os.path.exists(env) else None
        summary["systems"][s] = res
    llms = [s for s in summary["systems"] if s.startswith("llm_")]
    if llms:
        local = LLM_BASELINES + [s for s in LLM_TESTS_ONLY if s in summary["systems"]]
        summary["llm_comparison"] = llm_comparison(tests, local + llms)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(HERE, "results", "tables.md"), "w", encoding="utf-8") as f:
        f.write(tables(summary))
    print(tables(summary))


def runs_on(env, s):
    """Where the latency was measured."""
    if s.startswith("llm_"):
        return "API, from Google Colab"
    if env.get("device", "cpu") != "cpu" and env.get("gpu"):
        return env["gpu"]
    return f"{env.get('cpus', 2)}-core CPU, {env.get('threads', 2)} threads"


def llm_tables(summary):
    C = summary["llm_comparison"]
    name = lambda s: summary["systems"][s]["name"]
    diff = lambda x: f"{100 * x['diff']:+.1f} ({100 * x['ci'][0]:+.1f} to {100 * x['ci'][1]:+.1f})"
    lines = ["", "## Compared with LLMs and decision models, on " + ", ".join(C["tests"]), "",
             "| System | Right decision | ZeroPoint − system | Silent mistakes | ZeroPoint − system | "
             + "Latency, median | Latency, 95th percentile | Runs on |", "|---|---|---|---|---|---|---|---|"]
    for s, r in C["systems"].items():
        zm = r.get("zeropoint_minus")
        lines.append(f"| {name(s)} | {pct(r['right'])} | {diff(zm['right']) if zm else ''} | {pct(r['silent'])} | "
                     f"{diff(zm['silent']) if zm else ''} | {r['ms_median']:,.0f} ms | {r['ms_p95']:,.0f} ms | "
                     f"{runs_on(summary['systems'][s].get('environment') or {}, s)} |")
    lines += ["", "*" + LLM_NOTE + "*", "", "| System | " + " | ".join(
        f"Right decision, {TEST_NAMES[t]}" for t in C["tests"]) + " | " + " | ".join(
        f"Silent mistakes, {TEST_NAMES[t]}" for t in C["tests"]) + " |", "|---|" + "---|" * (2 * len(C["tests"]))]
    for s, r in C["systems"].items():
        lines.append(f"| {name(s)} | " + " | ".join(pct(r["per_test"][t]["right"]) for t in C["tests"]) + " | "
                     + " | ".join(pct(r["per_test"][t]["silent"]) for t in C["tests"]) + " |")
    order = [s for s in C["systems"] if all(summary["systems"][s]["tests"].get(t, {}).get("order_changed") is not None
                                             for t in C["tests"])]
    if order:
        lines += ["", "| System | Decision changes when the options come in another order |", "|---|---|"]
        for s in order:
            v = np.mean([summary["systems"][s]["tests"][t]["order_changed"] for t in C["tests"]])
            lines.append(f"| {name(s)} | {pct(v)} |")
    lines += ["", "| LLM | How it was asked | Answers outside the list | Tokens per decision (in / out) |",
              "|---|---|---|---|"]
    for s, r in C["systems"].items():
        if not s.startswith("llm_"):
            continue
        m = r.get("mode") or {}
        how = (("forced tool call" if m.get("tool_choice") == "forced" else "tool offered, not forced")
               + (", temperature 0" if m.get("temperature_0") else ", default temperature")) if m else "strict JSON schema (temperature not recorded)"
        lines.append(f"| {name(s)} | {how} | {r['invalid']} of {r['cases']} | "
                     f"{r.get('tokens_in', 0):,.0f} / {r.get('tokens_out', 0):,.0f} |")
    return lines


def pct(x, nd=1):
    return "–" if x is None else f"{100 * x:.{nd}f}%"


def mean_over(res, key, sample=False):
    v = [(r["on_sample"] if sample else r).get(key) for r in res["tests"].values()]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if len(v) == len(TESTS) else None


def conditions(S):
    env = next(iter(S.values()))["environment"] or {}
    return (f"All systems ran on the same machine ({env.get('cpu', 'CPU')}, {env.get('system', '')} "
            f"{env.get('machine', '')}, Python {env.get('python', '')}), on CPU with {env.get('threads')} threads, "
            "one case at a time, at their default settings, on the same test cases.")


NLI_NOTE = ("Zero-shot NLI reads a fixed sample of 300 cases per test (150 with a right option, 150 without), "
            "because it runs the model once per option and is too slow for every case; the others read every case.")
DASH = "A dash means the measure does not apply to that system or the run has not finished."

CAPTIONS = {
    "choice": "Share of the cases with a right option in which the system scores the right option highest. "
              "Higher is better. It ignores whether the system would also say that no option fits.",
    "sure_wrong": "Share of the cases with no right option in which the system still picks an option and gives it "
                  "a probability of 0.9 or more: a confident wrong answer. Lower is better. Semantic Router has no "
                  "value because its scores are similarities, not probabilities. Laya without \"None of these\" "
                  "always picks an option, so its value is the share of those cases where it does so at 0.9 or more.",
    "fpr95": "Share of the cases with no right option that the system still acts on, when its threshold is set so "
             "that it keeps 95% of the cases with a right option. Lower is better. It uses 1 − p(none) for systems "
             "that give p(none), and the top option's score otherwise, so every system can be measured.",
    "decision": "Share of all cases in which the system's own answer at its default settings is right. On a case "
                "with no right option only \"no option fits\" is right, so a system that always picks an option "
                "(Semantic Router, zero-shot NLI, Laya without \"None of these\") gets none of those cases right.",
    "order_changed": "Share of cases whose decision changes when the system reads the same text and the same "
                     "options in another order (a fixed shuffle per case), on the 300-case sample of each test. "
                     "Lower is better; 0% means the order never matters.",
    "ms_median": "Median time per case in milliseconds, from the call with the text and the options to the answer, "
                 "with nothing cached between cases. Lower is better. It grows with the number of options and the "
                 "length of the text, so the tests differ.",
}


SUMMARY_NOTE = (
    "Each percentage is the mean of the eight per-test values in the tables below. "
    "**Choice**: the right option is scored highest, on cases that have one (higher is better). "
    "**Sure and wrong**: on cases with no right option, an option picked with probability 0.9 or more "
    "(lower is better). **FPR95**: on cases with no right option, the share still acted on when the "
    "threshold keeps 95% of the cases with a right option (lower is better). "
    "**Order changes the decision**: share of cases whose decision changes when the options come in "
    "another order (lower is better). **Median latency**: over every case of every test. "
    "**Model size**: the model weights the system read during the run. "
    "**Reads the options together**: the model sees all options at once rather than one at a time.")
SAMPLE_NOTE = (
    "The same measures as the summary, but every system is scored only on the 300-case sample that "
    "zero-shot NLI reads (150 cases with a right option and 150 without, per test), so all rows come from "
    "identical cases. Mean over the eight tests.")
PAIRED_NOTE = (
    "Difference in percentage points between ZeroPoint and the other system on the same cases, with its 95% interval "
    "from a paired bootstrap (2,000 resamples of the cases). Choice: positive means ZeroPoint picks the right option "
    "more often. Sure and wrong: negative means ZeroPoint is confidently wrong less often. When the interval does not "
    "include zero, the difference is not due to chance in the choice of cases.")
LLM_NOTE = (
    "The 300-case sample of each test, the same cases for every system, 900 in all. Only these three tests: the other "
    "five are public since 1990-2020 and very likely in the LLMs' training data. Right decision: none counted as an "
    "answer. Silent mistakes: on cases with no right option, an option picked with probability (stated confidence for "
    "the LLMs) of 0.9 or more. ZeroPoint − system: percentage points on the same cases, with the 95% interval of a "
    "paired bootstrap; for silent mistakes, negative means ZeroPoint makes fewer. Latency is measured where "
    "*Runs on* says: for an API, the call from a Google Colab machine, network included.")
TESTS_NOTE = (
    "In a case with a right option, the right option is on the list among others. In a case without "
    "one, the list is made of other options only. The test files are rebuilt by `build_tests.py`; their "
    "sha256 are in `summary.json`.")


def tables(summary):
    S = {s: r for s, r in summary["systems"].items() if not s.startswith("llm_") and s not in LLM_TESTS_ONLY}
    lines = ["# Comparison results", "",
             "Generated by `report.py` from `out/`. " + conditions(S), ""]
    lines += ["## Summary (mean over the eight tests)", "",
              "| System | Choice | Sure and wrong | FPR95 | Order changes the decision | Median latency | "
              + "Model size | Reads the options together |", "|---|---|---|---|---|---|---|---|"]
    for s, r in S.items():
        nli = s == "nli"
        oc = [x.get("order_changed") for x in r["tests"].values()]
        oc = float(np.mean(oc)) if len(oc) == len(TESTS) and all(v is not None for v in oc) else None
        size = f"{r['model_bytes'] / 1e6:.0f} MB" if r.get("model_bytes") else "–"
        lines.append(f"| {r['name']}{' ¹' if nli else ''} | {pct(mean_over(r, 'choice'))} | "
                     f"{pct(mean_over(r, 'sure_wrong'))} | {pct(mean_over(r, 'fpr95'))} | {pct(oc)} | "
                     f"{r['ms_median_all']:.0f} ms | {size} | {r['reads_options_together']} |")
    lines += ["",
              "*" + SUMMARY_NOTE + " "
              + conditions(S) + " ¹ " + NLI_NOTE + " " + DASH + "*", ""]
    lines += ["## The same, on the 300-case sample of every test", "",
              "| System | Choice | Sure and wrong | FPR95 |", "|---|---|---|---|"]
    for s, r in S.items():
        lines.append(f"| {r['name']} | {pct(mean_over(r, 'choice', True))} | "
                     f"{pct(mean_over(r, 'sure_wrong', True))} | {pct(mean_over(r, 'fpr95', True))} |")
    lines += ["", "*" + SAMPLE_NOTE + " " + DASH + "*"]
    for key, title in [("choice", "Choice: the right option scored highest (cases with a right option)"),
                       ("sure_wrong", "Sure and wrong: an option picked with probability ≥ 0.9 when none is right"),
                       ("fpr95", "FPR95: cases with no right option still acted on, at 95% recall"),
                       ("decision", "Decision at default settings, all cases"),
                       ("order_changed", "Decision changes when the options come in another order (300 cases)"),
                       ("ms_median", "Median latency (ms)")]:
        lines += ["", f"## {title}", "", "| Test | " + " | ".join(r["name"] for r in S.values()) + " |",
                  "|---|" + "---|" * len(S)]
        for t in TESTS:
            cells = []
            for r in S.values():
                x = r["tests"].get(t, {})
                v = x.get(key)
                if key == "ms_median":
                    cells.append("–" if v is None else f"{v:.0f}")
                else:
                    c = x.get(f"{key}_ci")
                    cells.append(pct(v) + (f" ({100 * c[0]:.1f}–{100 * c[1]:.1f})" if c else ""))
            lines.append(f"| {TEST_NAMES[t]} | " + " | ".join(cells) + " |")
        ci_note = " In brackets, the 95% confidence interval (2,000 bootstrap resamples of the cases)." \
            if key in ("choice", "sure_wrong") else ""
        nli_note = " " + NLI_NOTE if key not in ("order_changed",) else ""
        lines += ["", "*" + CAPTIONS[key] + ci_note + nli_note + " " + DASH + "*"]
    for s in list(PAIRED_WITH) + sorted(EXTRA_PAIRED):
        if s not in S:
            continue
        lines += ["", f"## ZeroPoint − {S[s]['name']}, case by case", "",
                  "| Test | Choice | Sure and wrong |", "|---|---|---|"]
        for t in TESTS:
            zm = S[s]["tests"].get(t, {}).get("zeropoint_minus", {})
            cells = []
            for key in ("choice", "sure_wrong"):
                x = zm.get(key)
                cells.append("–" if x is None else
                             f"{100 * x['diff']:+.1f} ({100 * x['ci'][0]:+.1f} to {100 * x['ci'][1]:+.1f})")
            lines.append(f"| {TEST_NAMES[t]} | " + " | ".join(cells) + " |")
        lines += ["", "*" + PAIRED_NOTE + "*"]
    if "llm_comparison" in summary:
        lines += llm_tables(summary)
    lines += ["", "## The tests", "",
              "| Test | Cases | With a right option | Options per case | What it is |", "|---|---|---|---|---|"]
    for t in TESTS:
        info = summary["tests"][t]
        lines.append(f"| {TEST_NAMES[t]} | {info['cases']:,} | {info['with_right_option']:,} | "
                     f"{info['options']} | {TEST_ABOUT[t]} |")
    lines += ["", "*" + TESTS_NOTE + "*", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
