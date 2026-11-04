# Benchmark

How ZeroPoint 1.0 compares with three common ways of choosing an option for a text: Laya with a "None of these"
option, Semantic Router, and zero-shot classification with an NLI model, and, on three tests, with six hosted LLMs
from Anthropic and OpenAI ([section below](#compared-with-hosted-llms)). Everything here can be run again with the
scripts in [benchmarks/compare](benchmarks/compare) and [benchmarks/llm](benchmarks/llm), and every per-case output
is in the repository.

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
| Laya | the same | only the listed options ([section below](#laya-without-none-of-these)) |
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

What the summary shows:

- **Confident wrong answers.** When no option fits, ZeroPoint still answers with 90% confidence or more in 7% of
  cases; Laya with "None of these" in 21%, zero-shot NLI in 30%. In production these are the answers that pass a
  confidence threshold and are acted on without review. ZeroPoint does better on six of the eight tests, by 6 to 30
  points; on support tickets whose own queue was removed it does worse (see below).
- **Choosing.** When one option is right, Laya picks it more often on SNIPS and on support tickets; ZeroPoint on
  Spanish requests and on HWU64; on the other three tests the difference is within the confidence interval.
- **Right decision, *none* included.** Counting *none* as the right answer when no option fits, ZeroPoint is right
  most often: 63.6% against 59.9% for Laya with "None of these".
- **Order.** Laya's answer depends on the order of the options in 9 to 22% of cases, depending on the test.
  ZeroPoint's never does.
- **Speed.** ZeroPoint takes 21 ms per decision (median; mean 25 ms, 95th percentile 54 ms) on 2 CPU cores, 18 times
  less than Laya on the same machine (its authors report 33 ms on a T4 GPU). Semantic Router is faster, 9 ms, and
  chooses less well, especially in Spanish (31% on MTOP).

## Choosing: the right option, when there is one

Share of cases with a right option in which the system scores the right option highest. In brackets, the 95%
confidence interval.

| Test | ZeroPoint | Laya + "None of these" | ZeroPoint − Laya | Semantic Router | Zero-shot NLI |
|---|---|---|---|---|---|
| 2,000-case test | 85.4% (83.1–87.6) | 87.5% (85.3–89.6) | −2.1 (−4.6 to +0.2) | 71.1% (68.2–73.8) | 68.7% (61.1–75.8) |
| SNIPS | 80.4% (76.7–83.8) | 89.0% (86.3–91.6) | **−8.6** (−11.8 to −5.4) | 76.4% (72.5–80.2) | 58.0% (49.7–65.9) |
| ATIS | 76.0% (71.2–80.9) | 77.3% (72.5–82.1) | −1.3 (−7.7 to +5.0) | 59.0% (53.5–64.4) | 35.7% (29.9–41.5) |
| MTOP (Spanish) | 72.8% (68.7–76.5) | 68.0% (63.8–72.2) | **+4.8** (+0.2 to +9.2) | 31.2% (27.0–35.1) | 52.7% (44.4–60.9) |
| HWU64 | 86.7% (83.5–89.6) | 82.0% (78.5–85.7) | **+4.7** (+0.4 to +8.9) | 74.2% (70.2–78.2) | 66.7% (60.2–72.7) |
| CLINC150 | 93.5% (91.7–95.1) | 94.9% (93.3–96.3) | −1.4 (−3.4 to +0.4) | 87.1% (84.5–89.3) | 68.0% (60.1–75.2) |
| Tickets, queue removed | 24.3% (19.5–29.4) | 39.0% (33.7–44.7) | **−14.7** (−20.7 to −8.0) | 27.3% (22.3–32.5) | 31.3% (23.6–38.8) |
| Tickets, unrelated texts | 28.7% (23.9–34.2) | 40.0% (34.0–45.2) | **−11.3** (−17.0 to −5.3) | 28.3% (23.6–33.7) | 24.0% (17.8–30.5) |

*ZeroPoint − Laya* is the difference in percentage points, with its 95% interval from a paired bootstrap over the
same cases (2,000 resamples, computed by `report.py`); in bold where the interval does not include zero. Laya chooses better on SNIPS and on support tickets;
ZeroPoint on Spanish requests (MTOP) and on HWU64. On the 2,000-case test, ATIS and CLINC150 the difference is
within the interval.

## Saying *none*: confident answers when no option is right

Share of cases with no right option in which the system still picks an option and gives it a probability of 0.9 or
more.

| Test | ZeroPoint | Laya + "None of these" | ZeroPoint − Laya | Zero-shot NLI |
|---|---|---|---|---|
| 2,000-case test | 5.8% (4.4–7.2) | 12.2% (10.2–14.4) | **−6.4** (−8.7 to −4.1) | 20.0% (13.5–26.3) |
| SNIPS | 4.6% (2.9–6.5) | 29.4% (25.6–33.3) | **−24.8** (−28.8 to −20.8) | 26.7% (19.5–33.8) |
| ATIS | 17.6% (7.8–29.5) | 47.1% (33.3–61.1) | **−29.4** (−45.1 to −13.7) | 88.2% (78.7–96.0) |
| MTOP (Spanish) | 3.1% (1.5–5.1) | 25.0% (20.5–29.7) | **−21.9** (−26.7 to −17.6) | 25.3% (18.2–32.2) |
| HWU64 | 2.4% (0.0–6.3) | 32.1% (22.4–41.7) | **−29.8** (−40.5 to −19.0) | 16.7% (9.2–25.6) |
| CLINC150 | 1.9% (1.0–2.9) | 24.2% (21.4–27.3) | **−22.4** (−25.2 to −19.4) | 18.0% (12.1–24.5) |
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

## Laya without "None of these"

Laya used as it comes, with only the listed options: it then always picks one of them. Same machine, same cases.

| | ZeroPoint | Laya + "None of these" | Laya |
|---|---|---|---|
| Picks the right option, when there is one | 68.5% | 72.2% | 72.3% |
| Picks an option with probability ≥ 0.9 when none is right | **6.9%** | 21.4% | 28.4% |
| FPR95 | **53.6%** | 60.1% | 79.9% |
| Decision changes when the options come in another order | **0%** | 14.9% | 16.7% |
| Time per decision (median) | 21 ms | 383 ms | 380 ms |

Adding "None of these" does not change how often Laya picks the right option, and lowers its confident wrong
answers when no option fits from 28% to 21%. It does not close the gap with ZeroPoint: per test, ZeroPoint is
confidently wrong less often than Laya without the option on the same six tests, and the case-by-case differences
are in [the full tables](benchmarks/compare/results/tables.md).

## Compared with hosted LLMs

Six hosted LLMs used as decision engines: Claude Haiku 5.5, Claude Sonnet 5.5 and Claude Opus 5.5 (Anthropic), and
GPT-6 Luna, GPT-6.1 Sol and GPT-6.1 Terra (OpenAI).

### Which tests, and why only three

Five of the eight tests (SNIPS, ATIS, MTOP, HWU64, CLINC150) have been public since 1990–2020, with their labels.
They are very likely in the training data of these LLMs, and a model that has seen a test is not measured by it.
The LLMs are therefore compared only on the three tests they are unlikely to have seen: the 2,000-case test and the
two support-ticket tests. This choice was fixed before any LLM result on these tests was seen.

On each test, every system reads the same fixed 300-case sample used for zero-shot NLI (150 cases with a right
option, 150 without): 900 cases per system, one pass, options in their original order.

### How the LLMs are asked

The same prompt for every model, in [run_llm.py](benchmarks/llm/run_llm.py): the text, the list of options plus
*none of these*, and the instruction to choose one and state a confidence from 0 to 1. The answer must be
structured, with the choice restricted to the listed strings: a tool call for Claude, a strict JSON schema for GPT.
No examples and no reasoning step are asked for. One call per case.

LLM APIs give no probabilities over the options, so the confidence used for silent mistakes is the one the model
states. LLM latency is the time of the API call from a Google Colab machine, network included; ZeroPoint and Laya
run on the 2-core CPU of the other tables.

### Results

| System | Right decision | ZeroPoint − system | Silent mistakes | ZeroPoint − system | Latency, median | Latency, 95th percentile |
|---|---|---|---|---|---|---|
| ZeroPoint 1.0 | 53.6% |  | 7.8% |  | 42 ms | 64 ms |
| Laya + "None of these" | 49.9% | +3.7 (+0.1 to +7.3) | 4.9% | +2.9 (−0.4 to +6.0) | 593 ms | 988 ms |
| Claude Haiku 5.5 | 57.0% | −3.4 (−6.8 to −0.1) | 1.1% | +6.7 (+4.2 to +9.3) | 778 ms | 1,285 ms |
| Claude Sonnet 5.5 | 59.6% | −6.0 (−9.4 to −2.7) | 1.8% | +6.0 (+3.3 to +8.7) | 1,520 ms | 2,265 ms |
| Claude Opus 5.5 | 60.3% | −6.8 (−10.2 to −3.4) | 0.9% | +6.9 (+4.4 to +9.6) | 2,123 ms | 3,433 ms |
| GPT-6 Luna | 58.0% | −4.4 (−7.9 to −1.1) | 22.7% | −14.9 (−18.9 to −10.9) | 1,590 ms | 3,801 ms |
| GPT-6.1 Terra | 58.9% | −5.3 (−8.7 to −2.0) | 24.4% | −16.7 (−20.7 to −12.4) | 2,010 ms | 3,149 ms |
| GPT-6.1 Sol | 59.9% | −6.3 (−9.7 to −3.0) | 26.0% | −18.2 (−22.4 to −14.2) | 2,329 ms | 3,753 ms |

Mean over the three tests. Right decision: *none* counted as an answer. Silent mistakes: on cases with no right
option, an option picked with probability (for the LLMs, stated confidence) of 0.9 or more. ZeroPoint − system:
difference in percentage points on the same cases, with the 95% interval of a paired bootstrap (2,000 resamples);
for silent mistakes, a negative difference means ZeroPoint makes fewer.

Per test:

| System | Right decision: 2,000-case test | Tickets, queue removed | Tickets, unrelated texts | Silent mistakes: 2,000-case test | Tickets, queue removed | Tickets, unrelated texts |
|---|---|---|---|---|---|---|
| ZeroPoint 1.0 | 76.7% | **22.3%** | 61.7% | 4.7% | 18.7% | 0.0% |
| Laya + "None of these" | 72.3% | 19.3% | 58.0% | 14.0% | 0.7% | 0.0% |
| Claude Haiku 5.5 | 88.7% | 16.0% | 66.3% | 1.3% | 2.0% | 0.0% |
| Claude Sonnet 5.5 | **93.7%** | 17.3% | 67.7% | 0.7% | 4.7% | 0.0% |
| Claude Opus 5.5 | 92.7% | 18.7% | **69.7%** | 2.0% | 0.7% | 0.0% |
| GPT-6 Luna | 91.3% | 20.3% | 62.3% | 11.3% | 52.7% | 4.0% |
| GPT-6.1 Terra | 92.7% | 19.0% | 65.0% | 8.7% | 61.3% | 3.3% |
| GPT-6.1 Sol | 93.0% | 21.0% | 65.7% | 9.3% | 66.0% | 2.7% |

How each LLM answered:

| LLM | How it was asked | Answers outside the list | Tokens per decision (in / out) |
|---|---|---|---|
| Claude Haiku 5.5 | forced tool call, default temperature | 0 of 900 | 885 / 62 |
| Claude Sonnet 5.5 | tool offered, not forced, default temperature | 0 of 900 | 761 / 78 |
| Claude Opus 5.5 | tool offered, not forced, default temperature | 1 of 900 | 761 / 109 |
| GPT-6 Luna | strict JSON schema | 0 of 900 | 256 / 59 |
| GPT-6.1 Terra | strict JSON schema | 0 of 900 | 256 / 75 |
| GPT-6.1 Sol | strict JSON schema | 0 of 900 | 256 / 38 |

An answer outside the list is counted as *none*.

### What the results show

- **Right decision.** The LLMs are right more often: 57.0 to 60.3% against 53.6% for ZeroPoint, 3 to 7 points, and
  every interval excludes zero (for Haiku, only just). Most of the gap comes from the 2,000-case test, where
  the LLMs reach 89 to 94% and ZeroPoint 77%. On support tickets whose own queue was removed, ZeroPoint is right
  most often, 22.3% against 16.0 to 21.0%; on tickets against unrelated texts it is 0.6 to 8 points behind.
- **Silent mistakes, Claude.** The three Claude models are rarely confidently wrong when no option fits: 0.9 to 1.8%,
  6 to 7 points fewer than ZeroPoint. Most of ZeroPoint's come from tickets whose own queue was removed (18.7%).
- **Silent mistakes, GPT.** The three GPT models are confidently wrong in 22.7 to 26.0% of cases with no right option,
  15 to 18 points more than ZeroPoint, and in 53 to 66% of tickets whose own queue was removed: they pick a
  neighbouring queue and state 90% confidence or more.
- **Laya with "None of these"** is right less often than ZeroPoint on these three tests (+3.7 points for ZeroPoint)
  and makes slightly fewer silent mistakes; that interval includes zero.
- **Latency.** ZeroPoint takes 42 ms (median) on a 2-core CPU. The LLMs take 0.8 to 2.3 s through their APIs, 18 to
  55 times longer, and their 95th percentile reaches 1.3 to 3.8 s.

### Limits of this comparison

- **Three tests, 900 cases per system.** The intervals above are for these cases; another sample or other tests
  could move the numbers by a few points.
- **The 2,000-case test was written with an LLM** (Gemini). Its wording may suit LLMs better than real user texts
  would; it is also the test where they lead ZeroPoint most.
- **Not every model was asked in the same mode.** Claude Sonnet 5.5 and Opus 5.5 rejected a forced tool call, so
  they were offered the tool without forcing it, with room to write before calling it. No Claude model accepted
  temperature 0, so all three ran at their default temperature. For the GPT models, whether temperature 0 was
  accepted was not recorded.
- **Stated confidence is not a probability.** The silent-mistake threshold of 0.9 applies to the confidence the LLM
  writes, which is not calibrated in the way ZeroPoint's probabilities are.
- **Latency was measured from Colab, network included.** It depends on where the calls are made from and on the
  load of the API at that time. ZeroPoint and Laya were timed on a different machine.
- **No cost.** Prices were not recorded with the run, so this section does not give a cost per decision. The token
  counts of every call are saved with the results.
- **One pass, original order.** The LLMs were not run a second time with the options in another order, so the order
  table below does not include them.

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
- **Hosted LLMs** are compared on three tests only; their own limits are [listed above](#limits-of-this-comparison).

## Run it again

```bash
pip install -e . laya==0.4.21 semantic-router==1.1.1 transformers torch datasets pandas scikit-learn
python benchmarks/compare/build_tests.py
python benchmarks/compare/run.py zeropoint      # then laya_none, laya, semantic_router, nli
python benchmarks/compare/report.py
```

The hosted LLMs, with your own API keys (or with the notebook [llm_compare.ipynb](benchmarks/llm/llm_compare.ipynb)):

```bash
pip install anthropic openai
python benchmarks/llm/run_llm.py anthropic claude-haiku-5-5     # the three tests, the 300-case sample: 900 calls
python benchmarks/llm/run_llm.py openai gpt-6-luna
python benchmarks/compare/report.py
```

Details in [benchmarks/compare/README.md](benchmarks/compare/README.md) and [benchmarks/llm/README.md](benchmarks/llm/README.md).
