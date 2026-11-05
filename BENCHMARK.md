# Benchmark

How ZeroPoint 1.0 compares with other ways of choosing one option from a list for a text, or saying that none
fits. Every table on this page is written by [report.py](benchmarks/compare/report.py) from the per-case outputs in
the repository, and every run can be repeated with the scripts in [benchmarks/compare](benchmarks/compare) and
[benchmarks/llm](benchmarks/llm).

## How it is measured

**Systems.**

| System | Made by | What it is | Runs on |
|---|---|---|---|
| ZeroPoint 1.0 | this repository | energy-based decision model; answers *none* on its own | CPU |
| Strands Decider 2B | AWS | open decision model, `strands-decider` 0.2.0, commit `216d1ccb` | GPU |
| Clef-flash | Cloudflare | open 9B decision model, commit `fb4417d3` | GPU |
| Laya | Convai Innovations | decision model, `laya` 0.4.21, its recommended `Router` | CPU or GPU |
| Semantic Router | Aurelio AI | `semantic-router` 1.1.1, default encoder, one route per option | CPU |
| Zero-shot NLI | Hugging Face pipeline | DeBERTa-v3-base MNLI, `multi_label=True` | CPU or GPU |
| Claude Haiku, Sonnet and Opus 5.5 | Anthropic | hosted LLMs, through their API | API |
| GPT-6 Luna, GPT-6.1 Terra and Sol | OpenAI | hosted LLMs, through their API | API |

**"None of these".** Strands Decider, Clef-flash, Laya and the LLMs have no *none* answer of their own. Every list
they read has one more option, "None of these", so that they can answer *none*. ZeroPoint reads the list as it is.
Semantic Router and zero-shot NLI, used as they come, always pick one of the options.

**Tests.** Every case is a text and a list of options; in about half the cases one option is right, in the others
none is and the right answer is *none*. The comparison uses three tests that no model is likely to have seen in
training:

| Test | Cases | With a right option | Options per case | What it is |
|---|---|---|---|---|
| 2,000-case test | 2,000 | 1,000 | 4–9 | everyday requests and support tickets, with hard wording, typos and unrelated texts ([details](benchmarks/benchmark_2000)) |
| Tickets, queue removed | 600 | 300 | 8 | support tickets; when no option is right, the ticket's own queue was removed from the list |
| Tickets, unrelated texts | 600 | 300 | 9 | support queues; the cases with no right option are film reviews and newsgroup posts |

Five other tests (SNIPS, ATIS, MTOP in Spanish, HWU64, CLINC150) have been public with their labels since 1990–2020
and are very likely in the training data of the larger models: Strands Decider lists CLINC150 and 20 Newsgroups
among its training sets, and Clef-flash reports results on CLINC150. They are used only for the systems that run on
a CPU, in [the last section](#eight-tests-the-systems-that-run-on-a-cpu). This choice was fixed before any result on
the three tests was seen.

**Cases.** On each of the three tests every system reads the same fixed sample of 300 cases (150 with a right
option, 150 without): 900 cases. Every system except the LLMs reads them a second time with the options in another
order.

**Machines.** ZeroPoint, Laya, Semantic Router and zero-shot NLI on a 2-core CPU (Intel Xeon at 2.1 GHz), 2
threads, one case at a time. Strands Decider and Clef-flash on an NVIDIA L4 GPU. The LLMs through their APIs, from
Google Colab, network included. Model and dataset versions are pinned in `run.py` and `build_tests.py`.

## All systems, on the three tests

<!-- table:main -->
| System | Right decision | Answer changes with the order of the options | Right option, when one is right | Silent mistakes | Latency, median | Runs on | Model size |
|---|---|---|---|---|---|---|---|
| ZeroPoint 1.0 | 53.6% | 0.0% | 43.8% | 7.8% | 42 ms | 2-core CPU, 2 threads | 89 MB |
| Strands Decider 2B | 58.6% | 7.3% | 58.4% | 0.0% | 93 ms | NVIDIA L4 | 4.8 GB |
| Clef-flash | 53.3% | 0.0% | 56.2% | 5.8% | 156 ms | NVIDIA L4 | 19.1 GB |
| Laya | 49.9% | 13.1% | 55.6% | 4.9% | 593 ms | 2-core CPU, 2 threads | 1.5 GB |
| Semantic Router | 19.4% | 0.6% | 38.9% | – | 17 ms | 2-core CPU, 2 threads | 91 MB |
| Zero-shot NLI | 20.7% | 0.0% | 41.3% | 22.2% | 1,697 ms | 2-core CPU, 2 threads | 369 MB |
| Claude Haiku 5.5 | 57.0% | not measured | – | 1.1% | 778 ms | API, from Google Colab | – |
| Claude Sonnet 5.5 | 59.6% | not measured | – | 1.8% | 1,520 ms | API, from Google Colab | – |
| Claude Opus 5.5 | 60.3% | not measured | – | 0.9% | 2,123 ms | API, from Google Colab | – |
| GPT-6 Luna | 58.0% | not measured | – | 22.7% | 1,590 ms | API, from Google Colab | – |
| GPT-6.1 Terra | 58.9% | not measured | – | 24.4% | 2,010 ms | API, from Google Colab | – |
| GPT-6.1 Sol | 59.9% | not measured | – | 26.0% | 2,329 ms | API, from Google Colab | – |
<!-- /table -->

*Right decision: the system's own answer is right, none included. Answer changes with the order of the options:
share of decisions that change when the same options are listed in another order. Right option, when one is right:
the right option has the highest score, on the 450 cases with one. Silent mistakes: on the 450 cases with no right
option, an option picked with a probability of 0.9 or more (for the LLMs, the confidence they state). A dash: the
measure does not apply (Semantic Router gives similarities, not probabilities; an LLM returns a single answer).*

What the results show:

- **Right decision.** The LLMs and Strands Decider are right most often, 57 to 60%; ZeroPoint and Clef-flash 53%.
- **Order.** ZeroPoint, Clef-flash and zero-shot NLI never change their answer with the order of the options;
  Strands Decider does in 7% of cases and Laya in 13%.
- **When one option is right**, Strands Decider, Clef-flash and Laya pick it more often than ZeroPoint, by 12 to 15
  points. ZeroPoint makes up for part of it by answering *none* more reliably.
- **Silent mistakes.** Strands Decider and the Claude models make the fewest, 0 to 2%; ZeroPoint 8%; the GPT models
  23 to 26%.
- **Where it runs.** ZeroPoint is the only decision model here that runs on a CPU at the speed of a request: 42 ms,
  with an 89 MB model. On the same CPU, Strands Decider takes 6.8 s and ZeroPoint 18 ms per decision; Clef-flash
  needs a GPU and weighs 19 GB.

## Right decision, *none* included

The share of cases in which the system's own answer is right: the right option when there is one, *none* when there
is not.

<!-- table:right -->
| System | 2,000-case test | Tickets, queue removed | Tickets, unrelated texts | All three | ZeroPoint − system |
|---|---|---|---|---|---|
| ZeroPoint 1.0 | 76.7% | 22.3% | 61.7% | 53.6% |  |
| Strands Decider 2B | 88.0% | 21.7% | 66.0% | 58.6% | −5.0 (−8.6 to −1.6) |
| Clef-flash | 85.3% | 19.3% | 55.3% | 53.3% | +0.2 (−3.2 to +3.8) |
| Laya | 72.3% | 19.3% | 58.0% | 49.9% | +3.7 (+0.1 to +7.3) |
| Semantic Router | 32.3% | 12.3% | 13.7% | 19.4% | +34.1 (+30.1 to +38.0) |
| Zero-shot NLI | 34.3% | 15.7% | 12.0% | 20.7% | +32.9 (+28.7 to +36.8) |
| Claude Haiku 5.5 | 88.7% | 16.0% | 66.3% | 57.0% | −3.4 (−6.8 to −0.1) |
| Claude Sonnet 5.5 | 93.7% | 17.3% | 67.7% | 59.6% | −6.0 (−9.4 to −2.7) |
| Claude Opus 5.5 | 92.7% | 18.7% | 69.7% | 60.3% | −6.8 (−10.2 to −3.4) |
| GPT-6 Luna | 91.3% | 20.3% | 62.3% | 58.0% | −4.4 (−7.9 to −1.1) |
| GPT-6.1 Terra | 92.7% | 19.0% | 65.0% | 58.9% | −5.3 (−8.7 to −2.0) |
| GPT-6.1 Sol | 93.0% | 21.0% | 65.7% | 59.9% | −6.3 (−9.7 to −3.0) |
<!-- /table -->

*ZeroPoint − system: difference in percentage points on the same cases, with the 95% interval of a paired bootstrap
(2,000 resamples). When the interval does not include zero, the difference is not due to the choice of cases.*

Strands Decider is right 5 points more often than ZeroPoint, and the six LLMs 3 to 7 points more. Clef-flash and
ZeroPoint are level. Most of the gap is on the 2,000-case test. On support tickets whose own queue is not on the
list, ZeroPoint is right most often.

## Same answer whatever the order of the options

The share of cases whose answer changes when the same text and the same options come in another order (a fixed
shuffle per case).

<!-- table:order -->
| System | 2,000-case test | Tickets, queue removed | Tickets, unrelated texts | All three |
|---|---|---|---|---|
| ZeroPoint 1.0 | 0.0% | 0.0% | 0.0% | 0.0% |
| Strands Decider 2B | 3.3% | 12.7% | 6.0% | 7.3% |
| Clef-flash | 0.0% | 0.0% | 0.0% | 0.0% |
| Laya | 9.3% | 14.7% | 15.3% | 13.1% |
| Semantic Router | 0.0% | 0.0% | 1.7% | 0.6% |
| Zero-shot NLI | 0.0% | 0.0% | 0.0% | 0.0% |
<!-- /table -->

ZeroPoint gives every option the same positions and reads the options in an order fixed by their text, so the
order they are given in cannot matter. Clef-flash sorts the options by name before reading them. Zero-shot NLI
scores each option on its own; Semantic Router's rare changes come from ties. Strands Decider and Laya read the
options in the order given, and their answer depends on it most on support tickets whose own queue is missing. The
LLMs were not run a second time.

## Right option, when one is right

On the 450 cases with a right option: the right option has the highest score, whether or not the system would also
answer *none*.

<!-- table:choice -->
| System | 2,000-case test | Tickets, queue removed | Tickets, unrelated texts | All three | ZeroPoint − system |
|---|---|---|---|---|---|
| ZeroPoint 1.0 | 80.0% | 22.7% | 28.7% | 43.8% |  |
| Strands Decider 2B | 94.7% | 38.7% | 42.0% | 58.4% | −14.7 (−20.2 to −9.3) |
| Clef-flash | 96.0% | 34.7% | 38.0% | 56.2% | −12.4 (−17.8 to −7.3) |
| Laya | 84.7% | 38.7% | 43.3% | 55.6% | −11.8 (−16.4 to −6.9) |
| Semantic Router | 64.7% | 24.7% | 27.3% | 38.9% | +4.9 (+0.2 to +10.0) |
| Zero-shot NLI | 68.7% | 31.3% | 24.0% | 41.3% | +2.4 (−2.4 to +7.8) |
<!-- /table -->

This is where ZeroPoint is weakest. When the right queue is on the list, Strands Decider, Clef-flash and Laya rank it
first 12 to 15 points more often. `fit` closes part of the gap on your own labels ([below](#adapting-to-your-labels)).

## Silent mistakes

On the 450 cases with no right option: the system picks an option anyway, with a probability of 0.9 or more. These
are the answers that pass a confidence threshold and are acted on without review.

<!-- table:silent -->
| System | 2,000-case test | Tickets, queue removed | Tickets, unrelated texts | All three | ZeroPoint − system |
|---|---|---|---|---|---|
| ZeroPoint 1.0 | 4.7% | 18.7% | 0.0% | 7.8% |  |
| Strands Decider 2B | 0.0% | 0.0% | 0.0% | 0.0% | +7.8 (+5.3 to +10.4) |
| Clef-flash | 8.0% | 9.3% | 0.0% | 5.8% | +2.0 (−0.9 to +4.9) |
| Laya | 14.0% | 0.7% | 0.0% | 4.9% | +2.9 (−0.4 to +6.0) |
| Zero-shot NLI | 20.0% | 38.0% | 8.7% | 22.2% | −14.4 (−18.7 to −10.4) |
| Claude Haiku 5.5 | 1.3% | 2.0% | 0.0% | 1.1% | +6.7 (+4.2 to +9.3) |
| Claude Sonnet 5.5 | 0.7% | 4.7% | 0.0% | 1.8% | +6.0 (+3.3 to +8.7) |
| Claude Opus 5.5 | 2.0% | 0.7% | 0.0% | 0.9% | +6.9 (+4.4 to +9.6) |
| GPT-6 Luna | 11.3% | 52.7% | 4.0% | 22.7% | −14.9 (−18.9 to −10.9) |
| GPT-6.1 Terra | 8.7% | 61.3% | 3.3% | 24.4% | −16.7 (−20.7 to −12.4) |
| GPT-6.1 Sol | 9.3% | 66.0% | 2.7% | 26.0% | −18.2 (−22.4 to −14.2) |
<!-- /table -->

*For silent mistakes, a negative ZeroPoint − system means ZeroPoint makes fewer. The LLMs' confidence is the one
they state, not a computed probability.*

Systems state high confidence at very different rates, so the same mistakes are also counted at 0.7 and 0.5:

<!-- table:thresholds -->
| System | Picks an option at all | ≥ 0.9 | ≥ 0.7 | ≥ 0.5 | Right answers given with ≥ 0.9 |
|---|---|---|---|---|---|
| ZeroPoint 1.0 | 31.8% | 7.8% | 19.1% | 26.0% | 66.0% |
| Strands Decider 2B | 39.3% | 0.0% | 4.2% | 21.1% | 36.8% |
| Clef-flash | 49.1% | 5.8% | 16.9% | 34.2% | 47.1% |
| Laya | 54.9% | 4.9% | 11.3% | 23.6% | 36.7% |
| Zero-shot NLI | 100.0% | 22.2% | 34.0% | 44.7% | 57.0% |
| Claude Haiku 5.5 | 40.2% | 1.1% | 14.4% | 38.4% | 57.9% |
| Claude Sonnet 5.5 | 38.2% | 1.8% | 15.6% | 34.0% | 59.5% |
| Claude Opus 5.5 | 38.7% | 0.9% | 7.3% | 30.2% | 58.9% |
| GPT-6 Luna | 39.6% | 22.7% | 38.2% | 39.6% | 93.3% |
| GPT-6.1 Terra | 38.2% | 24.4% | 37.8% | 38.2% | 92.3% |
| GPT-6.1 Sol | 36.7% | 26.0% | 36.7% | 36.7% | 92.8% |
<!-- /table -->

*Picks an option at all: on the cases with no right option, at any confidence. Last column: the share of the
system's right answers given with 0.9 or more, which shows how often it states high confidence at all.*

The threshold of 0.9 was fixed before the results were seen. Strands Decider picks an option more often than
ZeroPoint when none is right, 39% against 32%, but always with less than 0.9: it rarely states high confidence,
even when it is right. Its silent mistakes stay the fewest at every threshold. The GPT models state 0.9 or more in
over 90% of their right answers, and 0.7 or more on almost every option they pick when none is right.

## Latency and size

Time per decision, from the call with the text and the options to the answer.

<!-- table:latency -->
| System | Latency, median | Latency, 95th percentile | Runs on | Model size |
|---|---|---|---|---|
| ZeroPoint 1.0 | 42 ms | 64 ms | 2-core CPU, 2 threads | 89 MB |
| Strands Decider 2B | 93 ms | 96 ms | NVIDIA L4 | 4.8 GB |
| Clef-flash | 156 ms | 196 ms | NVIDIA L4 | 19.1 GB |
| Laya | 593 ms | 988 ms | 2-core CPU, 2 threads | 1.5 GB |
| Semantic Router | 17 ms | 30 ms | 2-core CPU, 2 threads | 91 MB |
| Zero-shot NLI | 1,697 ms | 2,743 ms | 2-core CPU, 2 threads | 369 MB |
| Claude Haiku 5.5 | 778 ms | 1,285 ms | API, from Google Colab | – |
| Claude Sonnet 5.5 | 1,520 ms | 2,265 ms | API, from Google Colab | – |
| Claude Opus 5.5 | 2,123 ms | 3,433 ms | API, from Google Colab | – |
| GPT-6 Luna | 1,590 ms | 3,801 ms | API, from Google Colab | – |
| GPT-6.1 Terra | 2,010 ms | 3,149 ms | API, from Google Colab | – |
| GPT-6.1 Sol | 2,329 ms | 3,753 ms | API, from Google Colab | – |
<!-- /table -->

*The 900 cases above. LLMs: the API call from Google Colab, network included.*

The GPU times of Strands Decider and Clef-flash are not comparable with the CPU times. To compare them, ZeroPoint and
Strands Decider were timed on the same 2-core CPU (an Intel Xeon at 2.8 GHz, 2 threads), on the first 118 cases of
the 2,000-case test's sample, with [cpu_latency.py](benchmarks/compare/cpu_latency.py):

<!-- table:same_cpu -->
| System | Latency, median | Latency, 95th percentile |
|---|---|---|
| ZeroPoint 1.0 | 18 ms | 29 ms |
| Strands Decider 2B | 6,770 ms | 10,040 ms |
<!-- /table -->

*Strands Decider ran in bfloat16: in float32, which its library uses on a CPU, it needs about 9 GB of memory, more
than that machine has; the library's own notes give about half the time in float32, on an Apple M3 Pro. On these
118 cases its answers on the CPU are the same as on the GPU.*

## The hosted LLMs

Each LLM gets the text, the options plus "None of these", and the instruction to choose one and state a confidence
from 0 to 1, in [run_llm.py](benchmarks/llm/run_llm.py). The answer must be structured, with the choice restricted
to the listed strings: a tool call for Claude, a strict JSON schema for GPT. No examples, no reasoning step, one
call per case. An answer outside the list counts as *none*.

<!-- table:llm_how -->
| LLM | How it was asked | Answers outside the list | Tokens per decision (in / out) |
|---|---|---|---|
| Claude Haiku 5.5 | forced tool call, default temperature | 0 of 900 | 885 / 62 |
| Claude Sonnet 5.5 | tool offered, not forced, default temperature | 0 of 900 | 761 / 78 |
| Claude Opus 5.5 | tool offered, not forced, default temperature | 1 of 900 | 761 / 109 |
| GPT-6 Luna | strict JSON schema (temperature not recorded) | 0 of 900 | 256 / 59 |
| GPT-6.1 Terra | strict JSON schema (temperature not recorded) | 0 of 900 | 256 / 75 |
| GPT-6.1 Sol | strict JSON schema (temperature not recorded) | 0 of 900 | 256 / 38 |
<!-- /table -->

Claude Sonnet 5.5 and Opus 5.5 rejected a forced tool call, so they were offered the tool without forcing it. No
Claude model accepted temperature 0. For the GPT models, whether temperature 0 was accepted was not recorded.

Strands Decider, a 2B model, against each LLM on the same cases:

<!-- table:strands_minus -->
| LLM | Right decision | Silent mistakes ≥ 0.9 | ≥ 0.7 | ≥ 0.5 |
|---|---|---|---|---|
| Claude Haiku 5.5 | +1.6 (−0.6 to +3.8) | −1.1 (−2.2 to −0.2) | −10.2 (−13.6 to −7.1) | −17.3 (−21.3 to −13.3) |
| Claude Sonnet 5.5 | −1.0 (−3.1 to +1.1) | −1.8 (−3.1 to −0.7) | −11.3 (−14.7 to −8.2) | −12.9 (−16.7 to −9.1) |
| Claude Opus 5.5 | −1.8 (−3.9 to +0.2) | −0.9 (−1.8 to −0.2) | −3.1 (−5.8 to −0.7) | −9.1 (−12.7 to −5.8) |
| GPT-6 Luna | +0.6 (−1.6 to +2.6) | −22.7 (−26.2 to −18.9) | −34.0 (−38.4 to −29.6) | −18.4 (−22.7 to −14.4) |
| GPT-6.1 Terra | −0.3 (−2.4 to +1.8) | −24.4 (−28.4 to −20.7) | −33.6 (−38.2 to −28.9) | −17.1 (−21.3 to −13.1) |
| GPT-6.1 Sol | −1.3 (−3.4 to +0.7) | −26.0 (−30.0 to −22.0) | −32.4 (−37.1 to −27.8) | −15.6 (−19.8 to −11.8) |
<!-- /table -->

*Percentage points, with the 95% interval of a paired bootstrap; for silent mistakes, negative means Strands
Decider makes fewer.*

In right decisions, Strands Decider is level with all six: every interval includes zero.

## Eight tests: the systems that run on a CPU

ZeroPoint, Laya, Semantic Router and zero-shot NLI on all eight tests, 7,537 cases: the three above and SNIPS, ATIS,
MTOP (Spanish), HWU64 and CLINC150 (20 options). Mean over the eight tests.

<!-- table:eight -->
| System | Right decision | Answer changes with the order of the options | Right option, when one is right | Silent mistakes | Latency, median | Model size |
|---|---|---|---|---|---|---|
| ZeroPoint 1.0 | 63.6% | 0.0% | 68.5% | 6.9% | 21 ms | 89 MB |
| Laya | 59.8% | 14.9% | 72.2% | 21.4% | 383 ms | 1.5 GB |
| Semantic Router | 34.6% | 0.2% | 56.8% | – | 9 ms | 91 MB |
| Zero-shot NLI | 28.6% | 0.0% | 50.6% | 30.2% | 1,316 ms | 369 MB |
<!-- /table -->

*Every case of every test, except zero-shot NLI, which reads the 300-case sample of each test because it runs its
model once per option. CLINC150's training split is in ZeroPoint's training data, so its test split is in a domain
ZeroPoint has seen. Per-test values, FPR95 and AUROC: [the full tables](benchmarks/compare/results/tables.md).*

## Adapting to your labels

`fit` with 300 labelled tickets (6 minutes on 2 CPU threads), measured on the two ticket tests:

| | Before | After |
|---|---|---|
| Right queue, tickets with their queue removed | 24.3% | 36.0% |
| Confident answers when the queue was removed | 19.7% | 0% |
| Right queue, tickets against unrelated texts | 28.7% | 46.0% |

## The same answers on every machine

On every change, CI runs 500 reference cases on Linux x86, Linux ARM, macOS (Apple silicon) and Windows, with
Python 3.10, 3.13 and 3.14. The decisions are the same on all of them, and every probability agrees with the reference to
within 3 millionths.

## Limits

- **Three tests, 900 cases per system.** Another sample or other tests could move the numbers by a few points.
- **The 2,000-case test was written with an LLM** (Gemini). It is the test where Strands Decider, Clef-flash and the
  LLMs lead ZeroPoint most.
- **Different machines.** CPU, GPU and API times are not comparable with each other; the same-CPU comparison covers
  118 cases.
- **"None of these" is a text option** for Strands Decider, Clef-flash, Laya and the LLMs. Other wording, or a
  different number of options, could change how often they choose it; this was not measured.
- **Stated confidence is not a probability.** For the LLMs, the 0.9 threshold applies to the confidence they write.
- **Training data.** Strands Decider's training data includes 20 Newsgroups; some cases with no right option in
  *Tickets, unrelated texts* are newsgroup posts.
- **Default settings.** Every system is used as it comes. A different setup could change its numbers.
- **Support tickets.** Without adaptation, ZeroPoint picks the right queue in about a quarter of tickets.
- **Training languages.** ZeroPoint's training texts are English and Spanish; the other 23 languages are read
  through its multilingual encoder and were not tested one by one.

## Run it again

```bash
pip install -e . laya==0.4.21 semantic-router==1.1.1 transformers torch datasets pandas scikit-learn
python benchmarks/compare/build_tests.py
python benchmarks/compare/run.py zeropoint      # then laya_none, semantic_router, nli
python benchmarks/compare/report.py             # writes results/ and the tables of this page
```

Strands Decider and Clef-flash, on a GPU (or with [deciders_colab.ipynb](benchmarks/compare/deciders_colab.ipynb),
on Google Colab):

```bash
pip install "strands-decider[cuda]==0.2.0"
DECIDER_DEVICE=cuda python benchmarks/compare/run.py strands_none bench2000 tickets_made tickets_far
DECIDER_DEVICE=cuda python benchmarks/compare/run.py clef_flash_none bench2000 tickets_made tickets_far
python benchmarks/compare/cpu_latency.py        # ZeroPoint and Strands Decider on the same CPU
```

The hosted LLMs, with your own API keys (or with [llm_compare.ipynb](benchmarks/llm/llm_compare.ipynb)):

```bash
pip install anthropic openai
python benchmarks/llm/run_llm.py anthropic claude-haiku-5-5     # the three tests, 900 calls
python benchmarks/llm/run_llm.py openai gpt-6-luna
```

Details in [benchmarks/compare/README.md](benchmarks/compare/README.md) and [benchmarks/llm/README.md](benchmarks/llm/README.md).
