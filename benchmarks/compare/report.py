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
TEST_NAMES = {"bench2000": "2,000-case benchmark", "snips": "SNIPS", "atis": "ATIS", "mtop_es": "MTOP (Spanish)",
              "hwu64": "HWU64", "clinc_real_indomain": "CLINC150 (20 options)", "tickets_made": "Tickets, queue removed",
              "tickets_far": "Tickets, unrelated texts"}
SYSTEMS = ["zeropoint", "laya_none", "laya", "semantic_router", "nli"]
NAMES = {"zeropoint": "ZeroPoint 1.0", "laya_none": "Laya + \"None of these\"", "laya": "Laya",
         "semantic_router": "Semantic Router", "nli": "Zero-shot NLI (DeBERTa-v3-base)"}
PROBABILITIES = {"zeropoint": True, "laya_none": True, "laya": True, "semantic_router": False, "nli": True}
TOGETHER = {"zeropoint": "yes", "laya_none": "yes", "laya": "yes", "semantic_router": "no", "nli": "no"}
B = 2000


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
        if system == "nli":
            return hub("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "model.safetensors")
    except Exception:                                          # weights not on this machine: leave it out
        return None


def read_rows(path):
    """out/ files are kept gzipped in the repository; a fresh run writes plain JSON. Either is read."""
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    if os.path.exists(path + ".gz"):
        return json.load(gzip.open(path + ".gz", "rt", encoding="utf-8"))
    return None


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


def order_changed(rows, shuffled_rows):
    a = {r["i"]: r["choice"] for r in rows}
    b = {r["i"]: r["choice"] for r in shuffled_rows}
    common = sorted(set(a) & set(b))
    return float(np.mean([a[i] != b[i] for i in common])), len(common)


def main():
    from run import sample
    rng = np.random.default_rng(13)
    tests = {t: json.load(open(os.path.join(HERE, "tests", f"{t}.json"), encoding="utf-8")) for t in TESTS}
    summary = {"tests": {t: {"cases": len(tests[t]),
                             "sha256": hashlib.sha256(open(os.path.join(HERE, "tests", f"{t}.json"), "rb")
                                                      .read()).hexdigest()} for t in TESTS},
               "systems": {}}
    for s in SYSTEMS:
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
        if not res["tests"]:
            continue
        all_ms = np.concatenate([[x["ms"] for x in rows] for rows in rows_by_test.values()])
        res["ms_median_all"] = float(np.median(all_ms))
        res["model_bytes"] = model_size(s, rows_by_test)
        env = os.path.join(d, "environment.json")
        res["environment"] = json.load(open(env)) if os.path.exists(env) else None
        summary["systems"][s] = res
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(summary, open(os.path.join(HERE, "results", "summary.json"), "w"), indent=1)
    open(os.path.join(HERE, "results", "tables.md"), "w", encoding="utf-8").write(tables(summary))
    print(tables(summary))


def pct(x, nd=1):
    return "–" if x is None else f"{100 * x:.{nd}f}%"


def mean_over(res, key, sample=False):
    v = [(r["on_sample"] if sample else r).get(key) for r in res["tests"].values()]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if len(v) == len(TESTS) else None


def tables(summary):
    S = summary["systems"]
    lines = ["# Comparison results", "",
             "Generated by `report.py` from `out/`. Same machine, same test cases, CPU, 2 threads, one case at a time.",
             ""]
    lines += ["## Summary (mean over the eight tests)", "",
              "| System | Choice | Sure and wrong | FPR95 | Order changes the decision | Median latency | "
              "Model size | Reads the options together |", "|---|---|---|---|---|---|---|---|"]
    for s, r in S.items():
        nli = s == "nli"
        oc = [x.get("order_changed") for x in r["tests"].values()]
        oc = float(np.mean(oc)) if all(v is not None for v in oc) else None
        size = f"{r['model_bytes'] / 1e6:.0f} MB" if r.get("model_bytes") else "–"
        lines.append(f"| {r['name']}{' ¹' if nli else ''} | {pct(mean_over(r, 'choice'))} | "
                     f"{pct(mean_over(r, 'sure_wrong'))} | {pct(mean_over(r, 'fpr95'))} | {pct(oc)} | "
                     f"{r['ms_median_all']:.0f} ms | {size} | {r['reads_options_together']} |")
    lines += ["", "¹ 300-case sample per test (150 with a right option, 150 without).", ""]
    lines += ["## The same, on the 300-case sample of every test (identical cases for every system)", "",
              "| System | Choice | Sure and wrong | FPR95 |", "|---|---|---|---|"]
    for s, r in S.items():
        lines.append(f"| {r['name']} | {pct(mean_over(r, 'choice', True))} | "
                     f"{pct(mean_over(r, 'sure_wrong', True))} | {pct(mean_over(r, 'fpr95', True))} |")
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
    env = next(iter(S.values()))["environment"] or {}
    lines += ["", f"Machine: {env.get('cpu', '?')}, {env.get('system', '')} {env.get('machine', '')}, "
              f"Python {env.get('python', '')}, {env.get('threads')} threads.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
