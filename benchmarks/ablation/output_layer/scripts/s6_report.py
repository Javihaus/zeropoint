"""Stage 6: results/summary.json (last) and results/REPORT.md, every number read from the JSON files.
The prose in READING (no numbers) is written after looking at the results and is labelled as such."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ENCODERS, LONG, RESULTS, TESTS, VARIANTS, jdump


def _json_load(*args, **kw):
    with open(*args, **kw) as f:
        return json.load(f)


def _jsonl(*args, **kw):
    with open(*args, **kw) as f:
        return [json.loads(line) for line in f]

J = lambda n: _json_load(os.path.join(RESULTS, n), encoding="utf-8")
M, P, D, K, L = J("metrics.json"), J("paired.json"), J("depths.json"), J("packaging.json"), J("lists_info.json")
DEV = J("deviations.json") if os.path.exists(os.path.join(RESULTS, "deviations.json")) else []
NOTES = J("packaging_notes.json") if os.path.exists(os.path.join(RESULTS, "packaging_notes.json")) else []
READING = "(reading not written yet)"
if os.path.exists(os.path.join(RESULTS, "reading.md")):
    with open(os.path.join(RESULTS, "reading.md"), encoding="utf-8") as f:
        READING = f.read()

MEAS = [("correct_decisions", "Correct", 3), ("ranking", "Ranking", 3), ("auroc_none", "AUROC", 3),
        ("auprc_none", "AUPRC", 3), ("fpr95", "FPR95", 3), ("silent_mistakes_09", "Silent@0.9", 3),
        ("false_rejections", "False rej.", 3), ("ece", "ECE", 3), ("aurc", "AURC", 3)]
NAMES = {"V1": "V1 fixed zero", "V2": "V2 plain Boltzmann", "V3": "V3 learned rejection", "V4": "V4 softmax+threshold",
         "V5": "V5 ReLU depths"}


def cell(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, dict):
        return f"{x['mean']:.{nd}f} ± {x['sd']:.{nd}f}" if x["sd"] is not None else f"{x['mean']:.{nd}f}"
    return f"{x:.{nd}f}"


def table(scope):
    head = "| encoder | variant | " + " | ".join(m[1] for m in MEAS) + " |"
    rows = [head, "|" + "---|" * (len(MEAS) + 2)]
    rows.append("| E1 | released 1.0c (reference) | " + " | ".join(cell(M["released_E1"][scope][m]) for m, _, _ in MEAS) + " |")
    for e in ENCODERS:
        for v in VARIANTS:
            s = M["summary"][f"{e}/{v}"][scope]
            rows.append(f"| {e} | {NAMES[v]} | " + " | ".join(cell(s[m]) for m, _, _ in MEAS) + " |")
    return "\n".join(rows)


def per_test_table(measure):
    rows = ["| encoder | variant | " + " | ".join(TESTS) + " |", "|" + "---|" * (len(TESTS) + 2)]
    rows.append("| E1 | released | " + " | ".join(cell(M["released_E1"][t][measure]) for t in TESTS) + " |")
    for e in ENCODERS:
        for v in VARIANTS:
            rows.append(f"| {e} | {v} | " + " | ".join(cell(M["summary"][f"{e}/{v}"][t][measure], 2) for t in TESTS) + " |")
    return "\n".join(rows)


def paired_table():
    rows = ["| encoder | contrast | measure | V1 | other | V1 − other | 95% CI | p | p (Holm) | reading |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for c in P["comparisons"]:
        rows.append(f"| {c['encoder']} | {c['contrast']} | {c['measure']} | {c['V1']:.4f} | {c['other']:.4f} | "
                    f"{c['diff']:+.4f} | [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | {c['p']:.4f} | "
                    f"{c['p_holm']:.4f} | {c['reading']} |")
    return "\n".join(rows)


def depth_table():
    rows = [("| test | options | depth median | depth p95 | depth min | S/e^b median | S/e^b p95 | share S > e^b | "
             "S/e^b median, wrong options of cases with a right one |"), "|---|---|---|---|---|---|---|---|---|"]
    for sec, names in [("per_test", TESTS), ("per_long", LONG)]:
        for t in names:
            n, p = D[sec][t]["none"], D[sec][t]["with_right_option"]
            rows.append(f"| {t} | {n['options_per_list_mean']:.0f} | {n['depth_median']:.3f} | {n['depth_p95']:.3f} | "
                        f"{n['depth_min']:.4f} | {n['sum_weight_over_exp_b_median']:.3f} | "
                        f"{n['sum_weight_over_exp_b_p95']:.2f} | {n['share_cases_sum_weight_gt_exp_b']:.3f} | "
                        f"{p['sum_weight_over_exp_b_median']:.3f} |")
    for name in ["pooled_tests", "pooled_long"]:
        n, p = D[name]["none"], D[name]["with_right_option"]
        rows.append(f"| **{name}** | {n['options_per_list_mean']:.1f} | {n['depth_median']:.3f} | {n['depth_p95']:.3f} | "
                    f"{n['depth_min']:.4f} | {n['sum_weight_over_exp_b_median']:.3f} | "
                    f"{n['sum_weight_over_exp_b_p95']:.2f} | {n['share_cases_sum_weight_gt_exp_b']:.3f} | "
                    f"{p['sum_weight_over_exp_b_median']:.3f} |")
    return "\n".join(rows)


pkg = K["checks"]
pkg_lines = [f"- Automated checks that failed: {', '.join(K['issues']) if K['issues'] else 'none'}.",
             f"- Import time {pkg['import_seconds']:.2f} s; import and load {pkg['import_and_load_seconds']:.2f} s.",
             f"- Quick start output equals README: {all(q['matches_readme'] for q in pkg['quick_start'])}. Order "
             f"invariance (bits): {pkg['order_invariant_bits']}. 1 vs 2 threads (bits): {pkg['threads_1_vs_2_bits']}. "
             f"decide_many = decide: {pkg['decide_many_equals_decide']}. fit save/reload (bits): "
             f"{pkg['fit_save_reload_same_bits']}. fit repeatable with the same seed: {pkg['fit_deterministic_same_seed']}.",
             f"- fit, {pkg['fit']['examples']} examples, {pkg['fit']['steps']} steps: {pkg['fit']['report_seconds']:.1f} s, "
             f"projected {pkg['fit']['projected_minutes_for_200_steps']:.1f} min for the default 200 steps.",
             f"- Warnings raised by the package: {', '.join(w for w in K['warnings'] if 's0_packaging' not in w) or 'none'}.",
             f"- ONNX outputs: {pkg['onnx_outputs']}."]
pkg_lines += [f"- {n}" for n in NOTES]
cmp_lines = [f"- {t}: decision agreement {v['decision_agreement']:.4f}, |Δp_none| median {v['p_none_abs_diff_median']:.5f}, "
             f"max {v['p_none_abs_diff_max']:.4f}" for t, v in D["package_vs_1.0c_checkpoint"].items()]

ts = L["train_subset"]
report = f"""# Fixed-zero ablation of ZeroPoint 1.0: report

Protocol: `results/PROTOCOL.md` (written before any result). Every number below is read from the JSON files in
`results/` by `scripts/s6_report.py`.

## What was run

- `zeropoint==1.0.0` from PyPI in a fresh venv (Python 3.13, CPU, 2 cores), plus the `fit` extras (CPU torch).
- Frozen features of {ts['lists']} training lists (of {L['full_train']['lists']} rebuilt with `zp1.py`'s list code;
  none share {ts['none_share']:.3f}, lists longer than 20 options {ts['long_share_gt20']:.3f}), {L['dev_stop']['lists']}
  dev lists for early stopping and {L['dev_cal']['lists']} for calibration, the eight tests ({M['released_E1']['pooled_tests']['n']}
  cases) and `tests_long` ({M['released_E1']['pooled_long']['n']} cases), with two frozen encoders: E1, the released
  ONNX encoder through the package's own runtime, and E2, the original multilingual-e5-small.
- Five heads (V1 fixed zero, V2 plain Boltzmann, V3 learned rejection class from [CLS], V4 softmax with a threshold
  fitted on dev, V5 ReLU depths with an exact zero), 5 seeds each, per encoder: 50 trainings.

## Results, pooled over the eight tests (mean ± sd over 5 seeds)

{table('pooled_tests')}

## Results on tests_long (pooled, mean ± sd over 5 seeds)

{table('pooled_long')}

## Correct decisions per test (mean over 5 seeds)

{per_test_table('correct_decisions')}

## AUROC for "no right option" per test (mean over 5 seeds)

{per_test_table('auroc_none')}

## Paired comparisons (seed-averaged probabilities, {P['cases']} cases, {P['B']} stratified bootstrap resamples, Holm over {len(P['comparisons'])})

{paired_table()}

## Depths of the released model on cases without a right option

T = {D['T']:.4f}, b = {D['b']:.4f}, e^b = {D['exp_b']:.2f}. S = Σ(e^(a/T) − 1) over the options of a case. S > e^b
means the options together outweigh *none* even though none of them is right.

{depth_table()}

Package (ONNX, int8 vocabulary, CPU) against the released per-case outputs of the 1.0c checkpoint:

{chr(10).join(cmp_lines)}

## Packaging

{chr(10).join(pkg_lines)}

## Reading (written after the results)

{READING}

## Deviations from the protocol

{chr(10).join('- ' + d for d in DEV) if DEV else '- none'}

## Limitations

- Only the head is retrained on frozen features. The fixed zero might act through encoder fine-tuning, which this
  design cannot see. E1 has been fine-tuned with the fixed-zero head, so it favours V1; E2 has never seen any head.
- A linear head on mean-pooled states, {ts['lists']} of {L['full_train']['lists']} training lists.
- Seed variation covers head initialisation and batch order only; the data subset and features are fixed, so the
  standard deviations understate the real variability. The paired bootstrap covers test-case sampling only.
- Pooled measures weight large tests (bench2000, clinc) more; per-test numbers are in `results/metrics.json`.
- The V4 threshold and all calibration scalars are fitted on `dev_cal`, which (as in `zp1.py`) contains no written
  examples: the hash split sends all of them to `dev_stop`.
- E2 is run in PyTorch with padded batches, E1 one sequence at a time through onnxruntime; both are deterministic,
  but tiny numeric differences from batching are possible in E2.
"""
with open(os.path.join(RESULTS, "REPORT.md"), "w", encoding="utf-8") as f:
    f.write(report)

summary = {"protocol": "results/PROTOCOL.md", "report": "results/REPORT.md", "lists": L,
           "packaging": {"failed_checks": K["issues"], "notes": NOTES, "checks": {k: pkg[k] for k in [
               "import_seconds", "import_and_load_seconds", "order_invariant_bits", "threads_1_vs_2_bits",
               "decide_many_equals_decide", "fit_save_reload_same_bits", "fit_deterministic_same_seed"]},
               "fit": {k: v for k, v in pkg["fit"].items() if k != "report"},
               "package_vs_1.0c_checkpoint": D["package_vs_1.0c_checkpoint"]},
           "released_E1": {s: M["released_E1"][s] for s in ["pooled_tests", "pooled_long", "macro_tests"]},
           "variants": {k: {s: v[s] for s in ["pooled_tests", "pooled_long", "macro_tests"]} for k, v in M["summary"].items()},
           "seed_averaged": {k: {s: v[s] for s in ["pooled_tests", "pooled_long"]} for k, v in M["seed_averaged"].items()},
           "paired": P["comparisons"],
           "depths": {"T": D["T"], "b": D["b"], "pooled_tests": D["pooled_tests"], "pooled_long": D["pooled_long"],
                      "per_test": D["per_test"], "per_long": D["per_long"]},
           "deviations": DEV}
jdump(summary, os.path.join(RESULTS, "summary.json"))
print("report done", flush=True)
