# Fixed-zero ablation of ZeroPoint 1.0: report

Protocol: `results/PROTOCOL.md` (written before any result). Every number below is read from the JSON files in
`results/` by `scripts/s6_report.py`.

## What was run

- `zeropoint==1.0.0` from PyPI in a fresh venv (Python 3.13, CPU, 2 cores), plus the `fit` extras (CPU torch).
- Frozen features of 30000 training lists (of 76211 rebuilt with `zp1.py`'s list code;
  none share 0.301, lists longer than 20 options 0.052), 1281
  dev lists for early stopping and 982 for calibration, the eight tests (7537
  cases) and `tests_long` (3600 cases), with two frozen encoders: E1, the released
  ONNX encoder through the package's own runtime, and E2, the original multilingual-e5-small.
- Five heads (V1 fixed zero, V2 plain Boltzmann, V3 learned rejection class from [CLS], V4 softmax with a threshold
  fitted on dev, V5 ReLU depths with an exact zero), 5 seeds each, per encoder: 50 trainings.

## Results, pooled over the eight tests (mean ± sd over 5 seeds)

| encoder | variant | Correct | Ranking | AUROC | AUPRC | FPR95 | Silent@0.9 | False rej. | ECE | AURC |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | released 1.0c (reference) | 0.695 | 0.758 | 0.858 | 0.843 | 0.521 | 0.053 | 0.235 | 0.156 | 0.132 |
| E1 | V1 fixed zero | 0.697 ± 0.000 | 0.760 ± 0.000 | 0.861 ± 0.000 | 0.849 ± 0.000 | 0.483 ± 0.003 | 0.052 ± 0.001 | 0.238 ± 0.000 | 0.154 ± 0.001 | 0.130 ± 0.000 |
| E1 | V2 plain Boltzmann | 0.692 ± 0.000 | 0.753 ± 0.000 | 0.850 ± 0.001 | 0.830 ± 0.003 | 0.635 ± 0.006 | 0.058 ± 0.001 | 0.234 ± 0.004 | 0.165 ± 0.001 | 0.143 ± 0.002 |
| E1 | V3 learned rejection | 0.696 ± 0.000 | 0.759 ± 0.001 | 0.859 ± 0.000 | 0.847 ± 0.000 | 0.503 ± 0.003 | 0.039 ± 0.001 | 0.217 ± 0.001 | 0.151 ± 0.001 | 0.128 ± 0.000 |
| E1 | V4 softmax+threshold | 0.613 ± 0.005 | 0.759 ± 0.001 | 0.711 ± 0.002 | 0.642 ± 0.002 | 0.849 ± 0.003 | 0.289 ± 0.002 | 0.298 ± 0.018 | 0.271 ± 0.004 | 0.288 ± 0.004 |
| E1 | V5 ReLU depths | 0.693 ± 0.002 | 0.723 ± 0.026 | 0.850 ± 0.008 | 0.811 ± 0.027 | 0.810 ± 0.260 | 0.038 ± 0.003 | 0.198 ± 0.011 | 0.156 ± 0.007 | 0.172 ± 0.035 |
| E2 | V1 fixed zero | 0.447 ± 0.001 | 0.452 ± 0.004 | 0.676 ± 0.000 | 0.633 ± 0.001 | 0.830 ± 0.001 | 0.018 ± 0.001 | 0.340 ± 0.006 | 0.097 ± 0.002 | 0.400 ± 0.003 |
| E2 | V2 plain Boltzmann | 0.447 ± 0.001 | 0.451 ± 0.004 | 0.675 ± 0.001 | 0.632 ± 0.001 | 0.831 ± 0.001 | 0.017 ± 0.001 | 0.340 ± 0.005 | 0.099 ± 0.003 | 0.401 ± 0.003 |
| E2 | V3 learned rejection | 0.438 ± 0.003 | 0.457 ± 0.006 | 0.672 ± 0.003 | 0.629 ± 0.003 | 0.825 ± 0.004 | 0.008 ± 0.002 | 0.297 ± 0.013 | 0.074 ± 0.006 | 0.391 ± 0.002 |
| E2 | V4 softmax+threshold | 0.389 ± 0.005 | 0.465 ± 0.002 | 0.587 ± 0.004 | 0.511 ± 0.002 | 0.917 ± 0.003 | 0.057 ± 0.001 | 0.323 ± 0.017 | 0.312 ± 0.004 | 0.469 ± 0.002 |
| E2 | V5 ReLU depths | 0.326 ± 0.026 | 0.313 ± 0.019 | 0.571 ± 0.016 | 0.505 ± 0.018 | 0.927 ± 0.012 | 0.000 ± 0.000 | 0.373 ± 0.059 | 0.054 ± 0.025 | 0.585 ± 0.028 |

## Results on tests_long (pooled, mean ± sd over 5 seeds)

| encoder | variant | Correct | Ranking | AUROC | AUPRC | FPR95 | Silent@0.9 | False rej. | ECE | AURC |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | released 1.0c (reference) | 0.674 | 0.764 | 0.818 | 0.784 | 0.786 | 0.061 | 0.184 | 0.107 | 0.195 |
| E1 | V1 fixed zero | 0.675 ± 0.001 | 0.766 ± 0.001 | 0.820 ± 0.000 | 0.791 ± 0.000 | 0.755 ± 0.002 | 0.059 ± 0.001 | 0.190 ± 0.001 | 0.105 ± 0.001 | 0.189 ± 0.000 |
| E1 | V2 plain Boltzmann | 0.669 ± 0.003 | 0.755 ± 0.001 | 0.812 ± 0.001 | 0.774 ± 0.002 | 0.790 ± 0.006 | 0.074 ± 0.002 | 0.181 ± 0.005 | 0.112 ± 0.004 | 0.210 ± 0.002 |
| E1 | V3 learned rejection | 0.660 ± 0.001 | 0.765 ± 0.001 | 0.821 ± 0.000 | 0.793 ± 0.001 | 0.753 ± 0.003 | 0.048 ± 0.001 | 0.168 ± 0.001 | 0.106 ± 0.002 | 0.188 ± 0.001 |
| E1 | V4 softmax+threshold | 0.661 ± 0.003 | 0.764 ± 0.001 | 0.752 ± 0.002 | 0.716 ± 0.002 | 0.816 ± 0.002 | 0.132 ± 0.001 | 0.391 ± 0.023 | 0.207 ± 0.001 | 0.256 ± 0.005 |
| E1 | V5 ReLU depths | 0.636 ± 0.009 | 0.738 ± 0.017 | 0.817 ± 0.002 | 0.782 ± 0.003 | 0.803 ± 0.110 | 0.046 ± 0.003 | 0.147 ± 0.008 | 0.133 ± 0.022 | 0.202 ± 0.004 |
| E2 | V1 fixed zero | 0.249 ± 0.003 | 0.204 ± 0.004 | 0.551 ± 0.001 | 0.548 ± 0.001 | 0.931 ± 0.001 | 0.002 ± 0.001 | 0.229 ± 0.007 | 0.096 ± 0.008 | 0.676 ± 0.007 |
| E2 | V2 plain Boltzmann | 0.252 ± 0.004 | 0.204 ± 0.005 | 0.551 ± 0.001 | 0.548 ± 0.001 | 0.932 ± 0.001 | 0.003 ± 0.000 | 0.231 ± 0.009 | 0.097 ± 0.007 | 0.675 ± 0.006 |
| E2 | V3 learned rejection | 0.241 ± 0.004 | 0.209 ± 0.008 | 0.548 ± 0.002 | 0.549 ± 0.001 | 0.924 ± 0.006 | 0.000 ± 0.000 | 0.206 ± 0.006 | 0.049 ± 0.006 | 0.637 ± 0.010 |
| E2 | V4 softmax+threshold | 0.496 ± 0.004 | 0.190 ± 0.003 | 0.556 ± 0.003 | 0.537 ± 0.004 | 0.933 ± 0.006 | 0.000 ± 0.000 | 0.826 ± 0.015 | 0.262 ± 0.002 | 0.447 ± 0.002 |
| E2 | V5 ReLU depths | 0.140 ± 0.015 | 0.072 ± 0.013 | 0.515 ± 0.006 | 0.512 ± 0.005 | 0.950 ± 0.005 | 0.000 ± 0.000 | 0.205 ± 0.026 | 0.033 ± 0.009 | 0.758 ± 0.015 |

## Correct decisions per test (mean over 5 seeds)

| encoder | variant | bench2000 | snips | atis | mtop_es | hwu64 | clinc_real_indomain | tickets_made | tickets_far |
|---|---|---|---|---|---|---|---|---|---|
| E1 | released | 0.789 | 0.633 | 0.621 | 0.561 | 0.755 | 0.890 | 0.223 | 0.618 |
| E1 | V1 | 0.79 ± 0.00 | 0.64 ± 0.00 | 0.62 ± 0.00 | 0.56 ± 0.00 | 0.75 ± 0.00 | 0.89 ± 0.00 | 0.23 ± 0.00 | 0.62 ± 0.00 |
| E1 | V2 | 0.79 ± 0.00 | 0.63 ± 0.00 | 0.62 ± 0.00 | 0.55 ± 0.00 | 0.76 ± 0.00 | 0.89 ± 0.00 | 0.21 ± 0.00 | 0.62 ± 0.00 |
| E1 | V3 | 0.79 ± 0.00 | 0.64 ± 0.00 | 0.62 ± 0.00 | 0.56 ± 0.00 | 0.76 ± 0.00 | 0.89 ± 0.00 | 0.21 ± 0.00 | 0.62 ± 0.00 |
| E1 | V4 | 0.62 ± 0.01 | 0.55 ± 0.01 | 0.51 ± 0.03 | 0.55 ± 0.00 | 0.72 ± 0.01 | 0.78 ± 0.01 | 0.40 ± 0.02 | 0.53 ± 0.00 |
| E1 | V5 | 0.79 ± 0.00 | 0.63 ± 0.01 | 0.62 ± 0.01 | 0.56 ± 0.00 | 0.77 ± 0.00 | 0.89 ± 0.00 | 0.20 ± 0.00 | 0.61 ± 0.00 |
| E2 | V1 | 0.54 ± 0.00 | 0.51 ± 0.00 | 0.50 ± 0.00 | 0.28 ± 0.00 | 0.29 ± 0.00 | 0.55 ± 0.00 | 0.15 ± 0.00 | 0.42 ± 0.01 |
| E2 | V2 | 0.54 ± 0.00 | 0.50 ± 0.00 | 0.49 ± 0.00 | 0.28 ± 0.00 | 0.29 ± 0.01 | 0.55 ± 0.00 | 0.15 ± 0.00 | 0.42 ± 0.01 |
| E2 | V3 | 0.52 ± 0.01 | 0.49 ± 0.01 | 0.50 ± 0.00 | 0.27 ± 0.01 | 0.30 ± 0.01 | 0.55 ± 0.00 | 0.14 ± 0.01 | 0.40 ± 0.01 |
| E2 | V4 | 0.39 ± 0.01 | 0.28 ± 0.01 | 0.50 ± 0.01 | 0.37 ± 0.01 | 0.38 ± 0.01 | 0.56 ± 0.01 | 0.14 ± 0.01 | 0.31 ± 0.01 |
| E2 | V5 | 0.35 ± 0.03 | 0.44 ± 0.05 | 0.38 ± 0.05 | 0.27 ± 0.02 | 0.18 ± 0.01 | 0.39 ± 0.03 | 0.14 ± 0.07 | 0.27 ± 0.07 |

## AUROC for "no right option" per test (mean over 5 seeds)

| encoder | variant | bench2000 | snips | atis | mtop_es | hwu64 | clinc_real_indomain | tickets_made | tickets_far |
|---|---|---|---|---|---|---|---|---|---|
| E1 | released | 0.912 | 0.777 | 0.692 | 0.723 | 0.899 | 0.963 | 0.507 | 0.993 |
| E1 | V1 | 0.91 ± 0.00 | 0.79 ± 0.00 | 0.69 ± 0.00 | 0.73 ± 0.00 | 0.90 ± 0.00 | 0.96 ± 0.00 | 0.51 ± 0.00 | 0.99 ± 0.00 |
| E1 | V2 | 0.91 ± 0.00 | 0.78 ± 0.00 | 0.69 ± 0.00 | 0.72 ± 0.00 | 0.90 ± 0.00 | 0.96 ± 0.00 | 0.50 ± 0.00 | 0.99 ± 0.00 |
| E1 | V3 | 0.91 ± 0.00 | 0.78 ± 0.00 | 0.68 ± 0.00 | 0.73 ± 0.00 | 0.90 ± 0.00 | 0.96 ± 0.00 | 0.51 ± 0.00 | 0.99 ± 0.00 |
| E1 | V4 | 0.76 ± 0.00 | 0.60 ± 0.00 | 0.63 ± 0.00 | 0.64 ± 0.00 | 0.78 ± 0.00 | 0.89 ± 0.00 | 0.47 ± 0.00 | 0.66 ± 0.00 |
| E1 | V5 | 0.90 ± 0.01 | 0.75 ± 0.02 | 0.68 ± 0.00 | 0.71 ± 0.01 | 0.90 ± 0.00 | 0.96 ± 0.00 | 0.51 ± 0.00 | 0.99 ± 0.00 |
| E2 | V1 | 0.74 ± 0.00 | 0.66 ± 0.00 | 0.60 ± 0.00 | 0.56 ± 0.00 | 0.60 ± 0.00 | 0.64 ± 0.00 | 0.54 ± 0.00 | 0.89 ± 0.00 |
| E2 | V2 | 0.74 ± 0.00 | 0.65 ± 0.00 | 0.60 ± 0.00 | 0.56 ± 0.00 | 0.60 ± 0.00 | 0.63 ± 0.00 | 0.54 ± 0.00 | 0.89 ± 0.00 |
| E2 | V3 | 0.73 ± 0.01 | 0.66 ± 0.00 | 0.60 ± 0.00 | 0.56 ± 0.00 | 0.61 ± 0.00 | 0.64 ± 0.00 | 0.53 ± 0.00 | 0.87 ± 0.00 |
| E2 | V4 | 0.63 ± 0.00 | 0.59 ± 0.01 | 0.67 ± 0.01 | 0.55 ± 0.00 | 0.60 ± 0.01 | 0.63 ± 0.00 | 0.45 ± 0.01 | 0.77 ± 0.03 |
| E2 | V5 | 0.58 ± 0.01 | 0.68 ± 0.01 | 0.54 ± 0.02 | 0.54 ± 0.01 | 0.59 ± 0.01 | 0.52 ± 0.01 | 0.53 ± 0.02 | 0.76 ± 0.05 |

## Paired comparisons (seed-averaged probabilities, 7537 cases, 2000 stratified bootstrap resamples, Holm over 24)

| encoder | contrast | measure | V1 | other | V1 − other | 95% CI | p | p (Holm) | reading |
|---|---|---|---|---|---|---|---|---|---|
| E1 | V1-V2 | correct_decisions | 0.6968 | 0.6925 | +0.0044 | [+0.0015, +0.0072] | 0.0055 | 0.0385 | V1 better |
| E1 | V1-V2 | auroc_none | 0.8612 | 0.8501 | +0.0111 | [+0.0097, +0.0126] | 0.0005 | 0.0120 | V1 better |
| E1 | V1-V2 | fpr95 | 0.4824 | 0.6380 | -0.1556 | [-0.1722, -0.1379] | 0.0005 | 0.0120 | V1 better |
| E1 | V1-V3 | correct_decisions | 0.6968 | 0.6959 | +0.0009 | [-0.0021, +0.0037] | 0.6252 | 1.0000 | no significant difference |
| E1 | V1-V3 | auroc_none | 0.8612 | 0.8593 | +0.0019 | [+0.0015, +0.0023] | 0.0005 | 0.0120 | V1 better |
| E1 | V1-V3 | fpr95 | 0.4824 | 0.5025 | -0.0201 | [-0.0303, -0.0077] | 0.0005 | 0.0120 | V1 better |
| E1 | V1-V4 | correct_decisions | 0.6968 | 0.6109 | +0.0860 | [+0.0748, +0.0971] | 0.0005 | 0.0120 | V1 better |
| E1 | V1-V4 | auroc_none | 0.8612 | 0.7113 | +0.1499 | [+0.1383, +0.1617] | 0.0005 | 0.0120 | V1 better |
| E1 | V1-V4 | fpr95 | 0.4824 | 0.8474 | -0.3649 | [-0.3930, -0.3368] | 0.0005 | 0.0120 | V1 better |
| E1 | V1-V5 | correct_decisions | 0.6968 | 0.6939 | +0.0029 | [-0.0015, +0.0073] | 0.2274 | 1.0000 | no significant difference |
| E1 | V1-V5 | auroc_none | 0.8612 | 0.8567 | +0.0045 | [+0.0036, +0.0055] | 0.0005 | 0.0120 | V1 better |
| E1 | V1-V5 | fpr95 | 0.4824 | 0.5267 | -0.0443 | [-0.0622, -0.0277] | 0.0005 | 0.0120 | V1 better |
| E2 | V1-V2 | correct_decisions | 0.4471 | 0.4466 | +0.0005 | [-0.0013, +0.0024] | 0.5852 | 1.0000 | no significant difference |
| E2 | V1-V2 | auroc_none | 0.6760 | 0.6752 | +0.0008 | [+0.0004, +0.0012] | 0.0005 | 0.0120 | V1 better |
| E2 | V1-V2 | fpr95 | 0.8299 | 0.8323 | -0.0024 | [-0.0057, +0.0035] | 0.7581 | 1.0000 | no significant difference |
| E2 | V1-V3 | correct_decisions | 0.4471 | 0.4408 | +0.0064 | [+0.0013, +0.0115] | 0.0135 | 0.0810 | no significant difference |
| E2 | V1-V3 | auroc_none | 0.6760 | 0.6727 | +0.0033 | [+0.0013, +0.0053] | 0.0005 | 0.0120 | V1 better |
| E2 | V1-V3 | fpr95 | 0.8299 | 0.8252 | +0.0047 | [-0.0066, +0.0159] | 0.4043 | 1.0000 | no significant difference |
| E2 | V1-V4 | correct_decisions | 0.4471 | 0.3873 | +0.0598 | [+0.0484, +0.0716] | 0.0005 | 0.0120 | V1 better |
| E2 | V1-V4 | auroc_none | 0.6760 | 0.5879 | +0.0881 | [+0.0723, +0.1035] | 0.0005 | 0.0120 | V1 better |
| E2 | V1-V4 | fpr95 | 0.8299 | 0.9167 | -0.0868 | [-0.1071, -0.0608] | 0.0005 | 0.0120 | V1 better |
| E2 | V1-V5 | correct_decisions | 0.4471 | 0.3456 | +0.1015 | [+0.0901, +0.1133] | 0.0005 | 0.0120 | V1 better |
| E2 | V1-V5 | auroc_none | 0.6760 | 0.5772 | +0.0988 | [+0.0877, +0.1102] | 0.0005 | 0.0120 | V1 better |
| E2 | V1-V5 | fpr95 | 0.8299 | 0.9277 | -0.0977 | [-0.1139, -0.0792] | 0.0005 | 0.0120 | V1 better |

## Depths of the released model on cases without a right option

T = 0.2575, b = 4.5173, e^b = 91.59. S = Σ(e^(a/T) − 1) over the options of a case. S > e^b
means the options together outweigh *none* even though none of them is right.

| test | options | depth median | depth p95 | depth min | S/e^b median | S/e^b p95 | share S > e^b | S/e^b median, wrong options of cases with a right one |
|---|---|---|---|---|---|---|---|---|
| bench2000 | 5 | 0.052 | 0.907 | 0.0064 | 0.052 | 22.64 | 0.173 | 0.085 |
| snips | 6 | 0.052 | 0.904 | 0.0065 | 0.071 | 8.61 | 0.224 | 0.064 |
| atis | 8 | 0.412 | 2.517 | 0.0064 | 107.095 | 1359.33 | 0.922 | 31.560 |
| mtop_es | 10 | 0.040 | 0.751 | 0.0063 | 0.108 | 10.71 | 0.185 | 0.108 |
| hwu64 | 10 | 0.061 | 0.849 | 0.0064 | 0.273 | 8.27 | 0.286 | 0.145 |
| clinc_real_indomain | 20 | 0.019 | 0.285 | 0.0063 | 0.075 | 2.01 | 0.080 | 0.128 |
| tickets_made | 8 | 0.381 | 2.480 | 0.0085 | 25.823 | 2642.64 | 0.790 | 13.859 |
| tickets_far | 9 | 0.062 | 0.385 | 0.0065 | 0.039 | 0.62 | 0.037 | 68.722 |
| clinc_20 | 20 | 0.019 | 0.292 | 0.0063 | 0.072 | 1.47 | 0.063 | 0.137 |
| clinc_50 | 50 | 0.019 | 0.329 | 0.0063 | 0.357 | 15.44 | 0.267 | 0.542 |
| clinc_100 | 100 | 0.019 | 0.327 | 0.0063 | 1.051 | 26.92 | 0.517 | 1.673 |
| clinc_150 | 150 | 0.021 | 0.347 | 0.0063 | 2.065 | 94.31 | 0.743 | 3.842 |
| mtop_es_20 | 20 | 0.044 | 0.737 | 0.0063 | 0.441 | 27.10 | 0.347 | 0.380 |
| mtop_es_50 | 50 | 0.051 | 0.856 | 0.0063 | 4.377 | 409.96 | 0.710 | 2.936 |
| **pooled_tests** | 10.1 | 0.037 | 0.837 | 0.0063 | 0.085 | 95.69 | 0.216 | 0.190 |
| **pooled_long** | 65.0 | 0.024 | 0.416 | 0.0063 | 0.724 | 67.48 | 0.441 | 1.013 |

Package (ONNX, int8 vocabulary, CPU) against the released per-case outputs of the 1.0c checkpoint:

- bench2000: decision agreement 0.9985, |Δp_none| median 0.00023, max 0.0366
- snips: decision agreement 0.9990, |Δp_none| median 0.00041, max 0.0211
- atis: decision agreement 0.9972, |Δp_none| median 0.00002, max 0.0209
- mtop_es: decision agreement 0.9965, |Δp_none| median 0.00074, max 0.0308
- hwu64: decision agreement 0.9981, |Δp_none| median 0.00046, max 0.0185
- clinc_real_indomain: decision agreement 1.0000, |Δp_none| median 0.00034, max 0.0351
- tickets_made: decision agreement 0.9983, |Δp_none| median 0.00037, max 0.0157
- tickets_far: decision agreement 0.9983, |Δp_none| median 0.00029, max 0.0150

## Packaging

- Automated checks that failed: none.
- Import time 0.08 s; import and load 0.64 s.
- Quick start output equals README: True. Order invariance (bits): True. 1 vs 2 threads (bits): True. decide_many = decide: True. fit save/reload (bits): True. fit repeatable with the same seed: True.
- fit, 60 examples, 10 steps: 12.9 s, projected 4.3 min for the default 200 steps.
- Warnings raised by the package: none.
- ONNX outputs: [['hidden', ['LayerNormalizationhidden_dim_0', 'L', 384], 'tensor(float)']].
- Installation of zeropoint==1.0.0 (core) worked in a fresh Python 3.13 venv with no error or warning; README quick start reproduced exactly.
- `pip install "zeropoint[fit]"` on Linux resolves torch from PyPI, which brings the CUDA build and its NVIDIA wheels (cuda-toolkit, cudnn, nccl, cublas, triton and others; see pkg/dryrun_fit.json), several GB, although fit runs on a CPU. The docs do not mention the CPU wheel index (pip install torch --index-url https://download.pytorch.org/whl/cpu). We installed CPU torch first.
- README and docs say an option's depth is 0 when the text does not belong there 'at all', and that 'an option that fits no better than zero gets no probability'. In the shipped runtime the depth is softplus(z), always > 0, and the Boltzmann weight is computed with x clipped to at least 1e-6, so every option keeps a positive weight; the depth analysis measures how large it is (depth_min > 0 on every test).
- The ONNX encoder exposes only the per-token last hidden states ('hidden'); pooling, head, depths and probabilities are numpy code in _runtime.py. Its batch dimension is fixed to 1, so there is no batched inference (consistent with the docs' 'one case at a time').
- recalibrate() on 60 support tickets returned an extreme temperature and a negative none level (results/packaging.json, checks.recalibrate), without any warning; its report is computed on the held-out 20%, here a handful of lists. fit() on the same 60 examples also reports on that handful of lists, without a warning about the size.
- docs/calibration.md says fit takes 'about 6 minutes for 300 examples on 2 threads'; training/runs/1.0c/fit_results.json records a longer time per fit for the same setting (cpu_freeze8_minutes_for_fit_2_threads). Our projection from 10 timed steps is in packaging.json (fit.projected_minutes_for_200_steps); time depends on steps, not on the number of examples.
- The example in usage.md and in the model.py docstring ('my card was swallowed by the cash machine' with three card/balance options) returns none with the shipped model (packaging.json, usage_md_example).
- The shipped model is not bit-identical to the 1.0c checkpoint whose results are published (training/runs/1.0c): the int8 vocabulary and ONNX/CPU arithmetic change p_none slightly and flip a small share of decisions (results/depths.json, package_vs_1.0c_checkpoint). zp1.py's docstring says 'the model measured is the model shipped'.
- Training-code finding (zp1.py, not the package): written examples go to dev only when sha256(text) % 20 == 0, and the dev set is then halved by sha256(text) % 2. Every number divisible by 20 is even, so all written-example dev lists land in dev_stop and none in dev_cal: the released temperature and none level were calibrated without any written example (results/lists_info.json: dev_cal has no synth kinds).

## Reading (written after the results)

This reading was written after the results were computed. It has no numbers of its own; every number it refers to is in the tables above.

**Sanity.** The V1 head retrained on E1 reproduces the released model on every measure within a small margin (compare the "released 1.0c" row with "E1 / V1 fixed zero"), so the frozen-feature setup recovers the shipped system.

**V1 against V2 (does the −1 matter?).** On E1, the encoder fine-tuned with the fixed zero, V1 beats V2 on all three pre-registered measures, most clearly on FPR95. On E2, the encoder that never saw any head, V1 and V2 are the same in correct decisions and FPR95; the AUROC difference is significant but tiny. So the −1 helps only on the encoder that was trained together with it. On neutral features it adds nothing measurable.

**V1 against V3 (learned rejection class).** Correct decisions are not different on either encoder after Holm. V1 has a slightly higher AUROC on both encoders and a lower FPR95 on E1 (not on E2). V3, in turn, makes fewer silent mistakes and fewer false rejections on both encoders and has a lower ECE. These are small differences in both directions. A learned *none* class fed by the [CLS] state abstains about as well as the fixed zero.

**V1 against V4 (softmax plus threshold).** V1 wins clearly on both encoders and all three measures. But V4 was trained, as pre-registered, only on lists with a right option. The gap shows the value of training with lists that have no right option and giving *none* its own place in the distribution. It does not show the value of the fixed zero, because V3, which also trains on those lists, closes almost all of the gap.

**V5 (ReLU, exact zero).** An exact zero does not help. On E1, correct decisions are not different, and V5 is worse on AUROC and FPR95, with a very large spread across seeds: dead ReLU units make the training unstable. On E2, V5 is far worse on everything. Its "zero silent mistakes" there comes from a model that is almost never confident, not from better abstention.

**Depths of the released model.** Softplus never reaches zero. Every option of every case without a right option keeps a positive depth, so it keeps a small positive weight. On short lists the summed residual weight is a small fraction of the *none* weight. But it adds up with list length. On CLINC with 20, 50, 100 and 150 options, the median summed weight grows steadily and passes the *none* weight on the longest lists, and the share of no-right-option cases in which the options together outweigh *none* rises with it. On ATIS and tickets_made, the options of no-right-option cases are not near zero at all: there the model is simply confused, not limited by the floor.

**Answer to the question.** On this evidence, the fixed zero is not what gives ZeroPoint 1.0 its abstention. What matters is training with lists that lack a right option and giving *none* an explicit outcome: a learned rejection class does about as well. The −1 shows a benefit only on the encoder that was fine-tuned with it, so the head and the encoder adapted to each other. An exact zero (ReLU) makes things worse. The README's description ("an option that fits no better than zero gets no probability") is not literally true of the shipped model, and the gap grows with list length. The head-only design cannot rule out that the fixed zero helps through encoder fine-tuning. Testing that needs full fine-tuning runs with V2 and V3 heads on a GPU.

**Caveats specific to these results.** The E2 heads are weak: their dev NLL is far above E1's, and several E2 V1 and V2 runs stopped at the 40-epoch cap, not by early stopping. So the E2 comparisons are made in a regime of poor features. The seed standard deviations are tiny for V1 to V4 because only the head initialisation and the batch order vary.


## Deviations from the protocol

- Operational, not a change of protocol: the container's disk filled up (about 43 GB usable, mostly taken by earlier sessions' files) while caching E2. After all 25 E1 runs were saved, the E1 feature arrays (X, cls) were dropped from cache/E1 by scripts/slim_cache.py, keeping offsets, gold indices and released depths. The E2 caching was relaunched and resumed from its saved parts. Retraining E1 would require caching its features again.
- Operational: torch was set to one thread per training process (small operations are about 35 times faster than with two threads on this CPU); this changes speed, not the computation.

## Limitations

- Only the head is retrained on frozen features. The fixed zero might act through encoder fine-tuning, which this
  design cannot see. E1 has been fine-tuned with the fixed-zero head, so it favours V1; E2 has never seen any head.
- A linear head on mean-pooled states, 30000 of 76211 training lists.
- Seed variation covers head initialisation and batch order only; the data subset and features are fixed, so the
  standard deviations understate the real variability. The paired bootstrap covers test-case sampling only.
- Pooled measures weight large tests (bench2000, clinc) more; per-test numbers are in `results/metrics.json`.
- The V4 threshold and all calibration scalars are fitted on `dev_cal`, which (as in `zp1.py`) contains no written
  examples: the hash split sends all of them to `dev_stop`.
- E2 is run in PyTorch with padded batches, E1 one sequence at a time through onnxruntime; both are deterministic,
  but tiny numeric differences from batching are possible in E2.
