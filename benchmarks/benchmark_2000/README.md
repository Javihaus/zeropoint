# Benchmark: 2,000 cases

A second test of ZeroPoint, separate from the CLINC150 and ticket tests in [BENCHMARK.md](../../BENCHMARK.md).
Every case is a text and a short list of options (4 to 9). In 1,000 cases one option is right; in the other
1,000 none is. The results and how to read them are in BENCHMARK.md, section
*A second benchmark: 2,000 cases*.

## The data

`data.json`: 2,000 cases built on the option names of the two shipped landscapes (1,415 `everyday`, 585
`support`), from 552 distinct texts. Each case has a scenario:

| With a right option (1,000) | Without one (1,000) |
|---|---|
| In-Scope (700) | Out-of-Scope (500): the right option is not on the list |
| Boundary-Answerable (250): harder wording | Boundary-NearMiss (200): close to an option, but none fits |
| Adversarial-Edge (100): typos, fragments | OOD-Noise (250): unrelated text, other languages |

23 `everyday` cases (9 texts) appear word for word in CLINC150; their ids are in `clinc_overlap_ids.json`
and the results are also reported without them.

## Systems

| Name | System | Score used for AUROC and FPR95 |
|---|---|---|
| `zeropoint` | ZeroPoint 0.1.0 from PyPI, small encoder, 8-bit, default κ = 2 | 1 − p(none) |
| `zeropoint_k025` | the same with κ = 0.25 | 1 − p(none) |
| `laya` | Laya 0.4.21, its recommended `Router`, CPU | top option probability |
| `sr_names` | Semantic Router 1.1.1, one route per option, utterance = option name | top route score |
| `sr_fewshot` | the same, 10 training examples per option (CLINC150 train, English tickets) | top route score |
| `nli_deberta` | Hugging Face zero-shot pipeline, DeBERTa-v3-base MNLI, `multi_label=True` | top entailment score |
| `nli_bart` | the same with BART-large-MNLI: in the code, not in the results (too slow on CPU) | |

Model revisions are pinned in `run_systems.py`. Liquid d1 and Microsoft Decision-1 are hosted only;
`run_api.py` runs them with your own API key and writes the same output format.

## Run it

```bash
pip install zeropoint==0.1.0 laya==0.4.21 semantic-router==1.1.1 transformers sentence-transformers \
    datasets scikit-learn torch
python run_systems.py zeropoint zeropoint_k025 laya sr_names sr_fewshot nli_deberta   # about 40 min on 2 CPU cores
python metrics.py        # writes results.json
python tables.py         # writes tables.md
python chart.py results.json ../../assets/benchmark-2000-chart.png
```

`out/` holds every system's score for every option of every case, so the metrics can be checked without
running the models again.

## Metrics

- **Accuracy, choosing**: on the 1,000 cases with a right option, the top-scored option is the right one.
- **Accuracy, default**: on all 2,000 cases, the system's own decision with its default settings. Only
  ZeroPoint answers *none* by default; the others always pick an option.
- **AUROC**: how well the score separates cases with a right option from cases without (1 is perfect,
  0.5 is chance).
- **FPR95**: the share of cases with no right option still acted on, at the threshold that keeps 95% of the
  cases with one.
- Intervals: 95%, from 2,000 bootstrap resamples of the 552 texts (cases that share a text are resampled
  together).
