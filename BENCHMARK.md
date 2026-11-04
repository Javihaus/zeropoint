# Benchmark

How ZeroPoint 1.0 compares with three common ways of choosing an option for a text: Laya with a "None of these"
option, Semantic Router, and zero-shot classification with an NLI model. Everything here can be run again with the
scripts in [benchmarks/compare](benchmarks/compare), and every per-case output is in the repository.

## How it is measured

- **Eight tests, 7,537 cases.** In each case a system gets a text and a list of options. In about half the cases one
  option is right; in the others none is, and the right answer is *none*.
- **The same machine for every system:** a 2-core cloud CPU (Intel Xeon at 2.1 GHz), 2 threads, one case at a time,
  default settings.
- **Zero-shot NLI reads a fixed sample of 300 cases per test** (150 with a right option, 150 without), because it
  runs its model once per option and takes 1 to 3 seconds per case. Every system is also scored on that same sample,
  in [the full tables](benchmarks/compare/results/tables.md).
- **Order.** Every system reads the 300-case sample a second time with the options in another order.
- **Versions.** Model versions are pinned by their Hugging Face commit, dataset versions in `build_tests.py`.

| Test | Cases | With a right option | Options per case | What it is |
|---|---|---|---|---|
| 2,000-case test | 2,000 | 1,000 | 4–9 | everyday requests and support tickets, with hard wording, typos and unrelated texts ([details](benchmarks/benchmark_2000)) |
| SNIPS | 1,000 | 500 | 6–7 | voice-assistant requests (English) |
| ATIS | 351 | 300 | 8 | airline travel questions (English) |
| MTOP | 852 | 500 | 10 | assistant requests in Spanish |
| HWU64 | 534 | 450 | 10 | home-assistant requests (English) |
| CLINC150 | 1,600 | 800 | 20 | everyday requests; cases with no right option are real out-of-scope questions |
| Tickets, queue removed | 600 | 300 | 8 | support tickets; when no option is right, the ticket's own queue was removed from the list |
| Tickets, unrelated texts | 600 | 300 | 9 | support queues; the cases with no right option are film reviews and newsgroup posts |

No test text is in ZeroPoint's training data, with one exception: CLINC150's training split is part of the training
data, so its test split, used here, is in a domain ZeroPoint has seen.

## Systems

| System | Version | What it gets |
|---|---|---|
| ZeroPoint | 1.0.0 | the text and the options |
| Laya + "None of these" | `laya` 0.4.21, its recommended `Router`, model commit `7b928d82` | the options plus one more, "None of these" |
| Semantic Router | `semantic-router` 1.1.1, default encoder all-MiniLM-L6-v2 | one route per option, with the option name as its only example |
| Zero-shot NLI | DeBERTa-v3-base MNLI, Hugging Face zero-shot pipeline, `multi_label=True` | the text and the options |

Makers and licences are in [benchmarks/compare/README.md](benchmarks/compare/README.md).

## Summary

| | ZeroPoint | Laya + "None of these" | Semantic Router | Zero-shot NLI |
|---|---|---|---|---|
| Picks the right option, when there is one | 68.5% | **72.2%** | 56.8% | 50.6% |
| Picks an option with probability ≥ 0.9 when none is right | **6.9%** | 21.4% | – | 30.2% |
| Cases with no right option still acted on, at 95% recall (FPR95) | **53.6%** | 60.1% | 69.0% | 87.1% |
| Right decision at default settings, all cases | **63.6%** | 59.9% | 34.6% | 28.6% |
| Decision changes when the options come in another order | **0%** | 14.9% | 0.2% | 0% |
| Time per decision (median, all cases) | 21 ms | 383 ms | **9 ms** | 1,316 ms |
| Model size | **89 MB** | 1,486 MB | 91 MB | 369 MB |

Mean over the eight tests. Semantic Router gives similarities, not probabilities, so it has no value in the second
row. For zero-shot NLI, the 300-case sample.

## Choosing: the right option, when there is one

Share of cases with a right option in which the system scores the right option highest. In brackets, the 95%
confidence interval.

| Test | ZeroPoint | Laya + "None of these" | ZeroPoint − Laya | Semantic Router | Zero-shot NLI |
|---|---|---|---|---|---|
| 2,000-case test | 85.4% (83.1–87.6) | 87.5% (85.3–89.6) | −2.1 (−4.5 to +0.2) | 71.1% (68.2–73.8) | 68.7% (61.1–75.8) |
| SNIPS | 80.4% (76.7–83.8) | 89.0% (86.3–91.6) | **−8.6** (−12.0 to −5.4) | 76.4% (72.5–80.2) | 58.0% (49.7–65.9) |
| ATIS | 76.0% (71.2–80.9) | 77.3% (72.5–82.1) | −1.3 (−7.7 to +4.7) | 59.0% (53.5–64.4) | 35.7% (29.9–41.5) |
| MTOP (Spanish) | 72.8% (68.7–76.5) | 68.0% (63.8–72.2) | **+4.8** (+0.2 to +9.4) | 31.2% (27.0–35.1) | 52.7% (44.4–60.9) |
| HWU64 | 86.7% (83.5–89.6) | 82.0% (78.5–85.7) | **+4.7** (+0.4 to +8.9) | 74.2% (70.2–78.2) | 66.7% (60.2–72.7) |
| CLINC150 | 93.5% (91.7–95.1) | 94.9% (93.3–96.3) | −1.4 (−3.2 to +0.4) | 87.1% (84.5–89.3) | 68.0% (60.1–75.2) |
| Tickets, queue removed | 24.3% (19.5–29.4) | 39.0% (33.7–44.7) | **−14.7** (−21.0 to −8.3) | 27.3% (22.3–32.5) | 31.3% (23.6–38.8) |
| Tickets, unrelated texts | 28.7% (23.9–34.2) | 40.0% (34.0–45.2) | **−11.3** (−17.3 to −5.7) | 28.3% (23.6–33.7) | 24.0% (17.8–30.5) |

*ZeroPoint − Laya* is the difference in percentage points, with its 95% interval from a paired bootstrap over the
same cases; in bold where the interval does not include zero. Laya chooses better on SNIPS and on support tickets;
ZeroPoint on Spanish requests (MTOP) and on HWU64. On the 2,000-case test, ATIS and CLINC150 the difference is
within the interval.

## Saying *none*: confident answers when no option is right

Share of cases with no right option in which the system still picks an option and gives it a probability of 0.9 or
more.

| Test | ZeroPoint | Laya + "None of these" | ZeroPoint − Laya | Zero-shot NLI |
|---|---|---|---|---|
| 2,000-case test | 5.8% (4.4–7.2) | 12.2% (10.2–14.4) | **−6.4** (−8.7 to −4.2) | 20.0% (13.5–26.3) |
| SNIPS | 4.6% (2.9–6.5) | 29.4% (25.6–33.3) | **−24.8** (−28.8 to −21.0) | 26.7% (19.5–33.8) |
| ATIS | 17.6% (7.8–29.5) | 47.1% (33.3–61.1) | **−29.4** (−45.1 to −11.8) | 88.2% (78.7–96.0) |
| MTOP (Spanish) | 3.1% (1.5–5.1) | 25.0% (20.5–29.7) | **−21.9** (−26.4 to −17.3) | 25.3% (18.2–32.2) |
| HWU64 | 2.4% (0.0–6.3) | 32.1% (22.4–41.7) | **−29.8** (−40.5 to −19.0) | 16.7% (9.2–25.6) |
| CLINC150 | 1.9% (1.0–2.9) | 24.2% (21.4–27.3) | **−22.4** (−25.4 to −19.4) | 18.0% (12.1–24.5) |
| Tickets, queue removed | 19.7% (15.4–24.3) | 1.3% (0.3–2.8) | **+18.3** (+14.0 to +23.0) | 38.0% (30.0–45.4) |
| Tickets, unrelated texts | 0.3% (0.0–1.1) | 0.0% (0.0–0.0) | +0.3 (0.0 to +1.0) | 8.7% (4.4–13.4) |

Lower is better. ZeroPoint is far less often confidently wrong on six tests, by 6 to 30 points. The exception is
support tickets whose own queue was removed from the list: there ZeroPoint often picks a neighbouring queue with
high confidence, and Laya does not. [Adapting ZeroPoint to the tickets](#adapting-to-your-labels) brings this to 0%.

FPR95, which needs no probabilities and so covers every system: the share of cases with no right option still
acted on, with the threshold set to keep 95% of the cases with a right option (lower is better).

| Test | ZeroPoint | Laya + "None of these" | Semantic Router | Zero-shot NLI |
|---|---|---|---|---|
| 2,000-case test | 34.8% | 47.0% | 77.7% | 79.3% |
| SNIPS | 59.2% | 54.8% | 72.0% | 89.3% |
| ATIS | 94.1% | 88.2% | 92.2% | 88.2% |
| MTOP (Spanish) | 77.3% | 85.5% | 96.0% | 85.3% |
| HWU64 | 50.0% | 84.5% | 78.6% | 86.9% |
| CLINC150 | 18.5% | 21.4% | 25.0% | 81.3% |
| Tickets, queue removed | 91.3% | 92.3% | 95.3% | 95.3% |
| Tickets, unrelated texts | 3.7% | 7.0% | 15.3% | 91.3% |

When the case without a right option is close to the options (ATIS, tickets with their queue removed), no system
separates it well at 95% recall.

## Order of the options

Share of cases whose decision changes when the same text and the same options come in another order (300 cases
per test).

| Test | ZeroPoint | Laya + "None of these" | Semantic Router | Zero-shot NLI |
|---|---|---|---|---|
| 2,000-case test | 0% | 9.3% | 0% | 0% |
| SNIPS | 0% | 13.0% | 0% | 0% |
| ATIS | 0% | 16.7% | 0% | 0% |
| MTOP (Spanish) | 0% | 22.3% | 0% | 0% |
| HWU64 | 0% | 15.0% | 0% | 0% |
| CLINC150 | 0% | 13.0% | 0% | 0% |
| Tickets, queue removed | 0% | 14.7% | 0% | 0% |
| Tickets, unrelated texts | 0% | 15.3% | 1.7% | 0% |

ZeroPoint and Laya both read the options together with the text. Laya reads them in the order they are given, and
its answer depends on that order. ZeroPoint gives every option the same positions and reads them in an order fixed
by their text, so the order they are given in cannot matter. Semantic Router and zero-shot NLI score each option
on its own; Semantic Router's rare changes come from ties.

## Speed

Median time per case, in milliseconds, on the same machine.

| Test | ZeroPoint | Laya + "None of these" | Semantic Router | Zero-shot NLI |
|---|---|---|---|---|
| 2,000-case test | 16 | 345 | 9 | 780 |
| SNIPS | 16 | 351 | 8 | 848 |
| ATIS | 17 | 352 | 8 | 1,021 |
| MTOP (Spanish) | 20 | 152 | 9 | 1,312 |
| HWU64 | 20 | 380 | 7 | 1,354 |
| CLINC150 (20 options) | 27 | 472 | 8 | 2,697 |
| Tickets, queue removed | 43 | 620 | 18 | 1,705 |
| Tickets, unrelated texts | 53 | 755 | 24 | 2,296 |

ZeroPoint's time grows with the length of the text and the number of options, because it reads them together.
Semantic Router encodes only the text and compares it with option vectors computed once, which makes it the fastest
and also explains why it chooses less well.

## Adapting to your labels

`fit` with 300 labelled tickets (6 minutes on 2 CPU threads), measured on the two ticket tests:

| | Before | After |
|---|---|---|
| Right queue, tickets with their queue removed | 24.3% | 36.0% |
| Confident answers when the queue was removed | 19.7% | 0% |
| Right queue, tickets against unrelated texts | 28.7% | 46.0% |

## The same numbers on every machine

On every change, CI runs 500 reference cases on Linux x86, Linux ARM, macOS (Apple silicon) and Windows, with
Python 3.10 to 3.13. The decisions are the same on all of them, and every probability agrees with the reference
to within 3 millionths.

## Limits

- **Support tickets.** Without adaptation, ZeroPoint picks the right queue in about a quarter of tickets.
- **Training languages.** The training texts are English and Spanish. The other 23 languages are read through the
  multilingual encoder and were not tested one by one.
- **CLINC150 is in-domain** for ZeroPoint, as said above.
- **Laya is used through its recommended Router at default settings.** A different setup could change its numbers.

## Run it again

```bash
pip install -e . laya==0.4.21 semantic-router==1.1.1 transformers torch datasets pandas scikit-learn
python benchmarks/compare/build_tests.py
python benchmarks/compare/run.py zeropoint      # then laya_none, semantic_router, nli
python benchmarks/compare/report.py
```

Details in [benchmarks/compare/README.md](benchmarks/compare/README.md).
