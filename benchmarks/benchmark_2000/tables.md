## Overall (2,000 cases)

| System | Accuracy, choosing | Accuracy, default | Accuracy @ TPR95 | AUROC | FPR95 |
|---|---|---|---|---|---|
| ZeroPoint 0.1.0 (κ = 2, default) | 0.773 [0.73, 0.82] | 0.764 | 0.656 | 0.895 [0.86, 0.92] | 0.443 [0.32, 0.58] |
| ZeroPoint 0.1.0 (κ = 0.25) | 0.773 [0.73, 0.82] | 0.738 | 0.635 | 0.890 [0.85, 0.92] | 0.465 [0.38, 0.54] |
| Laya | 0.883 [0.85, 0.91] | 0.442 | 0.505 | 0.791 [0.75, 0.83] | 0.858 [0.75, 0.93] |
| Semantic Router, option names only | 0.711 [0.66, 0.76] | 0.355 | 0.462 | 0.823 [0.78, 0.86] | 0.777 [0.65, 0.88] |
| Semantic Router, 10 examples per option | 0.753 [0.71, 0.80] | 0.377 | 0.586 | 0.855 [0.82, 0.89] | 0.568 [0.48, 0.66] |
| Zero-shot NLI, DeBERTa-v3-base | 0.760 [0.71, 0.80] | 0.380 | 0.462 | 0.788 [0.74, 0.83] | 0.814 [0.72, 0.89] |

## everyday (1,415 cases)

| System | Accuracy, choosing | AUROC | FPR95 |
|---|---|---|---|
| ZeroPoint 0.1.0 (κ = 2, default) | 0.929 | 0.935 [0.90, 0.97] | 0.269 [0.17, 0.35] |
| ZeroPoint 0.1.0 (κ = 0.25) | 0.929 | 0.930 [0.89, 0.96] | 0.269 [0.18, 0.35] |
| Laya | 0.980 | 0.886 [0.86, 0.91] | 0.411 [0.32, 0.49] |
| Semantic Router, option names only | 0.852 | 0.912 [0.88, 0.94] | 0.491 [0.29, 0.61] |
| Semantic Router, 10 examples per option | 0.901 | 0.920 [0.88, 0.95] | 0.307 [0.20, 0.45] |
| Zero-shot NLI, DeBERTa-v3-base | 0.876 | 0.871 [0.82, 0.91] | 0.509 [0.38, 0.61] |

## support (585 cases)

| System | Accuracy, choosing | AUROC | FPR95 |
|---|---|---|---|
| ZeroPoint 0.1.0 (κ = 2, default) | 0.382 | 0.774 [0.71, 0.83] | 0.777 [0.54, 0.88] |
| ZeroPoint 0.1.0 (κ = 0.25) | 0.382 | 0.790 [0.73, 0.84] | 0.863 [0.55, 0.95] |
| Laya | 0.639 | 0.647 [0.57, 0.72] | 0.983 [0.91, 1.00] |
| Semantic Router, option names only | 0.358 | 0.645 [0.57, 0.72] | 0.873 [0.75, 0.96] |
| Semantic Router, 10 examples per option | 0.382 | 0.752 [0.69, 0.81] | 0.780 [0.56, 0.85] |
| Zero-shot NLI, DeBERTa-v3-base | 0.470 | 0.619 [0.54, 0.70] | 0.847 [0.78, 0.92] |

## without the 23 cases whose text is in CLINC150

| System | Accuracy, choosing | AUROC | FPR95 |
|---|---|---|---|
| ZeroPoint 0.1.0 (κ = 2, default) | 0.770 | 0.893 [0.86, 0.92] | 0.443 [0.33, 0.59] |
| ZeroPoint 0.1.0 (κ = 0.25) | 0.770 | 0.888 [0.85, 0.92] | 0.465 [0.38, 0.53] |
| Laya | 0.881 | 0.789 [0.75, 0.83] | 0.858 [0.75, 0.94] |
| Semantic Router, option names only | 0.707 | 0.821 [0.78, 0.86] | 0.775 [0.65, 0.87] |
| Semantic Router, 10 examples per option | 0.749 | 0.854 [0.82, 0.89] | 0.565 [0.48, 0.67] |
| Zero-shot NLI, DeBERTa-v3-base | 0.757 | 0.786 [0.74, 0.83] | 0.813 [0.72, 0.89] |

## Per scenario, at the threshold that keeps 95% of answerable cases

Answerable scenarios: share acted on with the right option. No-answer scenarios: share correctly answered *none*.

| System | In-Scope | Boundary-Answerable | Adversarial-Edge | Out-of-Scope | Boundary-NearMiss | OOD-Noise |
|---|---|---|---|---|---|---|
| ZeroPoint 0.1.0 (κ = 2, default) | 0.79 | 0.68 | 0.62 | 0.47 | 0.34 | 0.90 |
| ZeroPoint 0.1.0 (κ = 0.25) | 0.76 | 0.68 | 0.62 | 0.57 | 0.23 | 0.68 |
| Laya | 0.89 | 0.82 | 0.47 | 0.13 | 0.14 | 0.16 |
| Semantic Router, option names only | 0.75 | 0.56 | 0.37 | 0.13 | 0.14 | 0.50 |
| Semantic Router, 10 examples per option | 0.78 | 0.65 | 0.61 | 0.20 | 0.44 | 0.87 |
| Zero-shot NLI, DeBERTa-v3-base | 0.75 | 0.69 | 0.43 | 0.19 | 0.20 | 0.18 |
