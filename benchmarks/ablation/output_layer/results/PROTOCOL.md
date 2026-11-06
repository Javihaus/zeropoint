# Pre-registered protocol: is the fixed zero responsible for ZeroPoint 1.0's abstention?

Written on 2026-11-05, before any result below was computed. It is not changed afterwards. Any departure that
becomes necessary is recorded in `results/STATUS.md` and in `results/REPORT.md` under "Deviations", never by editing
this file.

What had been run before this file was written: (a) installation of `zeropoint==1.0.0` from PyPI and the README
quick start; (b) a timing pilot of the feature caching on 150 lists of three splits for each encoder (only seconds
per list were looked at); (c) a check, on 5 cases, that depths computed from the cached hidden states equal, bit for
bit, the depths `ZeroPoint.decide` returns. No head was trained and no measure below was computed.

## Question

ZeroPoint 1.0 gives every option a depth `a_y = softplus(w · h_y + c) >= 0`, a weight `w_y = exp(a_y / T) - 1`, and
gives *none* the fixed weight `exp(b)`. Is this fixed-zero construction what makes it abstain well, or would a
plain Boltzmann head, a learned rejection class, or a softmax with a threshold do as well on the same features?

## Design

Only the head is retrained, on frozen features, so every variant sees exactly the same encoder outputs and data.

**Encoders.**
- **E1**: the released fine-tuned encoder, `encoder.onnx` inside the PyPI wheel `zeropoint==1.0.0`, run through the
  package's own `zeropoint._runtime.Runtime` (its int8 vocabulary, tokenizer, canonical order, shared position ids,
  blocks of 20). The ONNX graph outputs the per-token last hidden states (`hidden`, 1 x L x 384), so pooled states
  are taken directly from that output; the graph is not modified.
- **E2**: the original `intfloat/multilingual-e5-small` from Hugging Face, with its original tokenizer and full
  vocabulary, in PyTorch on CPU, never trained with the fixed zero. Inputs are built the same way:
  `[CLS] query: <text, max 192 pieces> [SEP] <option, max 16 pieces> [SEP] ...`, NFC, canonical order (UTF-8 bytes),
  every option block with the same position ids starting right after the text, blocks of 20 options. Sequences
  are batched with padding and an attention mask (E1 runs one sequence at a time, as the package does).

**Features cached per list.** For every option, the mean of the last hidden states over the option's own pieces
(closing `[SEP]` excluded, as in `_runtime.py`), in canonical order; and the `[CLS]` state, averaged over the blocks
of the list. For E1, the released depth of every option, computed in float64 exactly as `Runtime.depths`.

**Data.**
- The full 1.0c training lists are rebuilt with a copy of `zp1.py`'s `build_data(seed=0)` and default arguments
  (50,000 fine lists, 12,000 broad lists, 3 lists per written example; 30% of lists without a right option, half of
  them with the right option removed and half with options from another pool; hard negatives from
  `neighbors.json` in half of the lists; 8% long lists of 21 to all options).
- **Training subset**: 30,000 lists drawn uniformly without replacement from these (numpy `default_rng(13)`). A
  uniform draw keeps the same mix of *none* lists, hard negatives and long lists; the realised mix is written to
  `results/lists_info.json`. Size chosen from the timing pilot (about 20 ms per list per encoder on 2 cores) so that
  caching takes about 15 minutes per encoder and the 50 head trainings fit in the 2.5-hour budget.
- **Dev**: the 1.0c dev lists (`zp1.py`, `default_rng(1000)`), split by text hash into `dev_stop` (early stopping)
  and `dev_cal` (post-hoc calibration of two scalars and, for V4, the threshold), exactly as `zp1.py` splits them.
- **Tests**: the eight files in `tests/` (7,537 cases) and the six files in `tests_long/` (3,600 cases).

## Head variants

`z_y = w · h_y + c` with `h_y` the pooled option state. `T = exp(log_t)`. Probabilities are a softmax over the
log-weights of the options and, except V4, of *none*.

| | option log-weight | *none* log-weight | trained on |
|---|---|---|---|
| V1 fixed zero | `log(exp(x) - 1)`, `x = clamp(softplus(z)/T, 1e-6, 60)` (as `zp1.py`) | `b` | all lists |
| V2 plain Boltzmann | `softplus(z)/T` | `b` | all lists |
| V3 learned rejection | `z/T` | `r(X) = u · h_CLS + b` | all lists |
| V4 softmax + threshold | `z/T` | none (abstain if max prob < threshold) | lists with a right option |
| V5 ReLU depths | as V1 with `relu(z)`; at evaluation an option with `relu(z) = 0` gets weight exactly 0 | `b` | all lists |

V5 in training keeps V1's clamp at `1e-6` (so the loss stays finite when the right option has depth 0); the exact
zero is applied only when predicting. V4's threshold is the one on the maximum option probability that maximises
correct decisions on `dev_cal` (ties: lowest threshold).

**Training.** Negative log-likelihood (cross-entropy over options + *none*; V4 over options only). Adam, learning
rate 1e-3 for `w, c, u` and 1e-2 for `log_t, b` (as `zp1.py`), batches of 256 lists, at most 40 epochs, early
stopping on `dev_stop` NLL with patience 4, best epoch kept. Initialisation: `w, u ~ N(0, 0.02^2)` from the seed,
`log_t = 0`, `b = 0`, `c = log(e - 1)` for V1 and V2 (depth 1, as `zp1.py`), `c = 1` for V5, `c = 0` for V3 and V4.
Then, as in `zp1.py`'s `calibrate`, the two scalars `log_t` and `b` (V4: `log_t` only) are refitted by L-BFGS on
`dev_cal` NLL with everything else frozen. Seeds 0 to 4 change the initialisation and the batch order only; data and
features are the same for every seed.

**Decision rule** (V1, V2, V3, V5, as the package): the most probable option (ties: first in canonical order) if
its probability is at least `p_none`, otherwise abstain. Confidence = the larger of the two. V4: the most probable
option if its probability is at least the threshold, otherwise abstain; abstention score `1 - max option
probability`; confidence = max probability when acting, `1 - max probability` when abstaining.

**Reference row.** The released model itself (E1 + released head, `T = 0.2575`, `b = 4.517`), from the cached
released depths, evaluated with the same code. It is not part of the paired tests.

## Measures

Computed per test, and pooled over the 7,537 cases of the eight tests (primary), and pooled over the 3,600 cases of
`tests_long` (secondary).

1. **Correct decisions**: the right option chosen, or abstention when no option is right.
2. **Ranking**: on cases with a right option, the right option's probability is strictly higher than every other
   option's.
3. **AUROC** and **AUPRC** for detecting cases without a right option, score `p_none` (V4: `1 - max prob`).
4. **FPR95**: share of cases without a right option whose `1 - p_none` is at or above the 5% quantile of `1 - p_none`
   on cases with one (as `zp1.py`).
5. **Silent mistakes at 0.9**: on cases without a right option, share where an option is chosen with probability
   at least 0.9.
6. **False rejections**: on cases with a right option, share of abstentions.
7. **ECE**: 10 equal-width bins on the confidence of the final decision against its correctness (as `zp1.py`).
8. **AURC**: cases sorted by decreasing confidence of the final decision; mean over k of the error rate of the
   decisions among the top k.

Reported: mean and standard deviation over the 5 seeds of each measure, for each variant and encoder.

## Paired tests

- For each variant and encoder, per-case option probabilities and `p_none` are **averaged over the 5 seeds**; the
  decision rule is applied to the averaged probabilities (V4: the threshold is refitted on the seed-averaged
  `dev_cal` probabilities).
- Contrasts: V1 minus V2, V1 minus V3, V1 minus V4, V1 minus V5, for each encoder, on correct decisions, AUROC and
  FPR95, pooled over the 7,537 cases of the eight tests. That is 4 x 3 x 2 = 24 pre-registered comparisons.
- Paired bootstrap: 2,000 resamples, `numpy.random.default_rng(13)`, cases resampled with replacement within each
  test (stratified, sizes kept), the same resample for both variants. 95% interval = 2.5 and 97.5 percentiles.
  Two-sided p = `(1 + 2 * min(#{diff <= 0}, #{diff >= 0})) / (B + 1)`, capped at 1.
- Holm correction across the 24 comparisons, family-wise alpha 0.05.
- Readings fixed in advance: "V1 beats Vk" on a measure only if the Holm-corrected p < 0.05 and the interval is on
  V1's favourable side (higher for correct decisions and AUROC, lower for FPR95). E2 is the encoder that is not
  biased toward V1; if E1 and E2 disagree, E2 is the one that speaks to the head, and E1 to the shipped system.

## Depth analysis of the released model

With the released ONNX through the PyPI package (cached released depths, verified against `ZeroPoint.decide`), on
cases without a right option, per test and per `tests_long` file: median and 95th percentile of the option depths
`a`; per case the summed weight `S = sum(exp(a/T) - 1)` against `exp(b)`: median and 95th percentile of `S / exp(b)`,
share of cases with `S > exp(b)` (the options together outweigh *none*); and, for comparison, the same on cases with
a right option. This measures how far softplus is from an exact zero and how the residual weight grows with list
length.

## Packaging check

Install `zeropoint==1.0.0` from PyPI into a fresh venv, run the README quick start and the documented examples,
errors and attributes, `recalibrate`, and `fit` on a few examples (10 steps, timed and projected to 200), save and
reload. Record errors, warnings, missing files, slow import and mismatches with `README.md` and `docs/*.md`. Also
compare the package's outputs on the eight tests with the released per-case outputs of the 1.0c checkpoint
(`training/runs/1.0c/s0_cases_*.json`).

## Known limitations, stated in advance

- Only the head is retrained. The fixed zero might matter through the encoder fine-tuning, which this design does
  not test; E1 carries that training, E2 has none of it.
- 30,000 of 76,211 training lists; a linear head on frozen states.
- Seed variation covers the head only; the data, features and subset are fixed.
- Pooling over tests weights large tests (bench2000, clinc) more.
