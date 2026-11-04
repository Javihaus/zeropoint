<div align="center">

<img src="https://raw.githubusercontent.com/Javihaus/zeropoint/main/assets/zeropoint-circle-large.png" width="200" alt="ZeroPoint logo">

<br>

# ZeroPoint

<br>

[![License](https://img.shields.io/github/license/Javihaus/zeropoint?color=0000FF&cacheSeconds=86400)](https://github.com/Javihaus/zeropoint/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/zeropoint?color=4169E1&cacheSeconds=3600)](https://pypi.org/project/zeropoint/)
[![Python](https://img.shields.io/pypi/pyversions/zeropoint?color=1E90FF)](https://pypi.org/project/zeropoint/)
[![Docs](https://img.shields.io/badge/docs-javihaus.github.io-6a8bb5)](https://javihaus.github.io/zeropoint/)
[![Tests](https://github.com/Javihaus/zeropoint/actions/workflows/tests.yml/badge.svg)](https://github.com/Javihaus/zeropoint/actions/workflows/tests.yml)
[![SLSA 3](https://slsa.dev/images/gh-badge-level3.svg)](https://slsa.dev)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/Javihaus/zeropoint/badge)](https://scorecard.dev/viewer/?uri=github.com/Javihaus/zeropoint)

<br>

> Give ZeroPoint a text and a list of options. It picks the option that fits, or answers that none of them does.

<br>

</div>

Many systems need to pick one option from a list: the queue for a support ticket, the intent behind a request,
the process for a document. A classifier always picks something, even when the right option is not on the list or
the text has nothing to do with the task. ZeroPoint can answer *none*, and the probability of *none* comes out of
the same calculation as the probability of each option.

- **It runs on a CPU.** About 20 ms per decision on 2 cores. The model ships inside the package (89 MB); nothing is
  downloaded at run time.
- **It reads the text and all the options together**, like a person reading the whole list before choosing.
- **The order of the options never changes the answer**, and the same input gives the same numbers on Linux, macOS
  and Windows.
- **It works in 25 European languages**, and you can adapt it to your own labels with a few hundred examples.

<br>

## Quick start

```bash
pip install zeropoint
```

```python
from zeropoint import ZeroPoint

zp = ZeroPoint()
zp.decide("Can you move 200 euros from my savings to my checking account?",
          ["transfer", "balance", "pay bill"])
```

```text
transfer  0.77  ███████████████·····
balance   0.02  ····················
pay bill  0.01  ····················
none      0.20  ████················
→ transfer · 17 ms
```

When none of the options fits, ZeroPoint says so:

```python
zp.decide("Set an alarm for 6:30 tomorrow morning", ["transfer", "balance", "pay bill"])
```

```text
pay bill  0.00  ····················
transfer  0.00  ····················
balance   0.00  ····················
none      1.00  ████████████████████
→ none · 16 ms
```

The options can be any short text, in any of the supported languages:

```python
zp.decide("Me han cobrado dos veces la suscripción de marzo, ¿podéis devolverme el duplicado?",
          ["facturación", "soporte técnico", "recursos humanos"])
```

```text
facturación       0.76  ███████████████·····
soporte técnico   0.00  ····················
recursos humanos  0.00  ····················
none              0.24  █████···············
→ facturación · 20 ms
```

The full documentation is at [javihaus.github.io/zeropoint](https://javihaus.github.io/zeropoint/).

<br>

## How it compares with common approaches

The same eight tests for every system, on the same machine (CPU, 2 threads, one case at a time). In about half the
cases one option on the list is right; in the other half none is.

| | ZeroPoint | Laya + "None of these" | Semantic Router | Zero-shot NLI |
|---|---|---|---|---|
| Picks the right option, when there is one | 68.5% | **72.2%** | 56.8% | 50.6% ¹ |
| Picks an option with probability ≥ 0.9 when none is right (lower is better) | **6.9%** | 21.4% | – ² | 30.2% ¹ |
| Decision changes when the options come in another order | **0%** | 14.9% | 0.2% | 0% ¹ |
| Time per decision (median) | 21 ms | 383 ms | **9 ms** | 1,316 ms |
| Model size | **89 MB** | 1,486 MB | 91 MB | 369 MB |
| Reads all the options together | yes | yes | no | no |

<sub>Mean over eight tests: SNIPS, ATIS, HWU64, MTOP (Spanish), CLINC150 with 20 options, two support-ticket tests
and a 2,000-case test, 7,537 cases in all. Laya 0.4.21 with "None of these" added as one more option; Semantic
Router 1.1.1 with its default encoder and the option names as routes; zero-shot NLI with DeBERTa-v3-base MNLI.
¹ Zero-shot NLI reads a fixed sample of 300 cases per test, because it runs its model once per option.
² Semantic Router gives similarities, not probabilities. Per-test results, confidence intervals and how to run it
all again: [benchmarks/compare](https://github.com/Javihaus/zeropoint/tree/main/benchmarks/compare).</sub>

What the table shows:

- **ZeroPoint is far less often confidently wrong** when no option fits: 2–5% of those cases against 24–32% for
  Laya on SNIPS, MTOP, HWU64 and CLINC150.
- **Laya picks the right option more often** on SNIPS (89% against 80%) and on support tickets (39–40% against
  24–29%). ZeroPoint does better on Spanish (MTOP, 73% against 68%) and on HWU64 (87% against 82%); on the other
  tests the two are within their confidence intervals.
- **Support tickets are ZeroPoint's weak spot.** It picks the right queue in only about a quarter of tickets, and
  when a ticket's own queue is missing from the list it still answers with high confidence in 20% of them (Laya:
  1%). [Adapting it to your own tickets](#adapt-it-to-your-labels) fixes most of this.
- **Laya's answer depends on the order of the options**: the same text and the same options, listed in another
  order, change its decision in 9–22% of cases. ZeroPoint reads the options in an order that does not depend on how
  they are given, so this cannot happen.
- **Semantic Router is the fastest** and the weakest at picking the right option, especially in Spanish (31% on
  MTOP).

<br>

## Using it in your code

A common pattern is to act when ZeroPoint is confident and to pass the case to a person when it is not:

```python
d = zp.decide(text, ["transfer", "balance", "pay bill"])

if d.choice is None or d.confidence < 0.8:
    send_to_person(text)        # no option fits, or ZeroPoint is not sure enough
else:
    route(text, d.choice)       # e.g. "transfer"
```

The right threshold depends on what a wrong action costs you; pick it on a few hundred of your own labelled texts.

| Field | What it holds |
|---|---|
| `choice` | the chosen option, or `None` when *none* is the most probable answer |
| `confidence` | probability of `choice` (of *none* when `choice` is `None`) |
| `p_none` | probability that none of the options fits |
| `probabilities` | probability of each option, in the order you gave them; with `p_none` they add up to 1 |
| `depths` | how strongly the text belongs to each option (0: not at all) |
| `text_truncated` | `True` if the text was longer than ZeroPoint reads (192 word pieces) and its end was ignored |
| `options_truncated` | options longer than 16 word pieces, whose end was ignored |
| `latency_ms` | time the decision took |

For many texts, `zp.decide_many(texts, options)` takes one list of options for all texts, or one list per text.
Each text is computed on its own, so every result is exactly what `decide` gives for that text alone.

Lists can be of any length. ZeroPoint reads the options in blocks of 20 together with the text, and every option's
probability comes from one distribution over the whole list.

`ZeroPoint(threads=2)` sets how many CPU threads the encoder uses. The decisions do not depend on it.

<br>

## Languages

ZeroPoint reads the 22 official languages of the European Union written in the Latin script, plus Catalan, Basque
and Galician:

Basque, Catalan, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Hungarian,
Irish, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish.

A text written mostly in another script (Greek, Cyrillic, Arabic, Chinese and others) raises `UnsupportedLanguage`
instead of returning a guess. Other languages in the Latin script, such as Turkish or Vietnamese, are not detected:
ZeroPoint will answer, but it was not trained or tested on them.

The texts ZeroPoint was trained on are in English and Spanish. The other languages are read through the
multilingual encoder it is built on and were not tested one by one.

<br>

## Adapt it to your labels

With a few hundred labelled texts, `fit` adapts ZeroPoint to your own options. It runs on a CPU.

```bash
pip install "zeropoint[fit]"        # adds PyTorch and onnx, only needed to adapt
```

```python
mine, report = zp.fit(texts, labels)   # labels: the right option for each text, or None for "none of these"
print(report["before"], report["after"])
mine.save("my-model")

zp = ZeroPoint("my-model")
```

`fit` keeps 20% of your examples apart and reports how the model does on them before and after. Same data, same
seed and same number of threads give the same model, bit for bit.

On the support-ticket tests above, 300 labelled tickets and 6 minutes on 2 CPU threads raised the share of right
queues from 24% to 36% when the queue is on the list, and from 29% to 46% against unrelated texts. Confident
answers when the ticket's own queue is missing went from 20% to 0%.

If your labels are already the ones ZeroPoint knows and you only want to move the balance between acting and
answering *none*, `zp.recalibrate(texts, labels)` refits just two numbers, in seconds, without PyTorch.

<br>

## How it works

<img src="https://raw.githubusercontent.com/Javihaus/zeropoint/main/assets/how-it-works.png" width="100%" alt="Which team should handle the message: Can I change my delivery address? The options Billing and Technical are energy wells; None of these is the flat ground at energy zero. The text lands on the ground, so ZeroPoint answers none with 0.96, while softmax splits it between billing 0.55 and technical 0.45.">

ZeroPoint is an energy-based model ([LeCun et al., 2006](http://yann.lecun.com/exdb/publis/orig/lecun-06.pdf)).
Each option is a well, and *none of the options* is flat ground at energy zero. A text either falls into a well or
stays on the ground. A useful image is a ball on a floor with holes.

#### 1. Reading the text and the options together

The text and the options go through one encoder, [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small),
in a single sequence: the text first, then every option. Each option can look at the text and at the other options,
so it is judged against the alternatives. All options get the same positions in the sequence, so the encoder
cannot tell which one came first.

#### 2. The depth of each well

For each option, ZeroPoint reads how strongly the text belongs to it, a depth $a_y \ge 0$. Its energy is
$E_y = -a_y$. A depth of zero means the text does not belong to that option at all: the well is flat ground.

#### 3. Boltzmann's rule, with the ground at zero

In physics, Boltzmann's rule gives how likely each state is: the lower its energy, the more likely, exponentially
so. A softmax classifier applies this rule to the options only, so the probabilities of the options always add up
to 1 and one of them always wins.

ZeroPoint adds *none* as a state at energy zero and weighs each option by how far it falls below the ground:

$$w_y = e^{a_y/T} - 1, \qquad w_{\text{none}} = e^{b}, \qquad p(y) = \frac{w_y}{w_{\text{none}} + \sum_k w_k},
\qquad p(\text{none}) = \frac{w_{\text{none}}}{w_{\text{none}} + \sum_k w_k}$$

An option at depth zero gets weight zero, however many such options there are. With the plain rule, each of them
would get weight $e^0 = 1$ and *none* would be drowned out by long lists. The temperature $T$ and the level $b$
are two numbers learned in training and refit by `recalibrate`.

Example: two options, Billing and Technical, at temperature 1 and $b = 0$.

| Text | Depth in Billing | Depth in Technical | Softmax | ZeroPoint |
|---|---|---|---|---|
| *I was charged twice this month* | 3.00 | 0.00 | billing 0.95, technical 0.05 | **billing 0.95**, none 0.05 |
| *Can I change my delivery address?* | 0.05 | 0.04 | billing 0.50, technical 0.50 | billing 0.05, technical 0.04, **none 0.92** |

With the second text, softmax has to split all the probability between the two options. ZeroPoint finds that
neither option falls below the ground and answers *none*.

#### 4. The same numbers every time

The options are put in one fixed order (by their text) before reading, whatever order they arrive in. Each case is
computed on its own, never padded together with other texts, and the last steps run in 64-bit precision. On the
test machines (Linux on x86 and ARM, macOS, Windows), 500 reference cases give the same decisions everywhere, and
the probabilities agree to within 3 millionths.

<br>

## What it was trained on

Only data whose licence allows commercial use: CLINC150 (CC BY 3.0), Banking77 (CC BY 4.0), MASSIVE in English and
Spanish (CC BY 4.0), four Bitext customer-service datasets (CDLA-Sharing 1.0), and 5,000 examples written for this
project. None of the test texts above is in the training data. Details, with the criteria used to write the 5,000
examples: [docs/training-data.md](https://github.com/Javihaus/zeropoint/blob/main/docs/training-data.md).

<br>

## Coming from 0.x

1.0 replaces the landscapes of 0.x with one model that reads the options together with the text. `ZeroPoint.load()`
becomes `ZeroPoint()`, and options are always given in each call. The full list of changes is in the
[changelog](https://github.com/Javihaus/zeropoint/blob/main/CHANGELOG.md).

<br>

## References

- LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. (2006). A tutorial on energy-based learning. *Predicting Structured Data*, 1(0).
- Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., & Wei, F. (2024). Multilingual E5 text embeddings: a technical report. arXiv preprint arXiv:2402.05672.

<br>

## Citation

```bibtex
@software{marin_zeropoint_2026,
  author = {Marín, Javier},
  title  = {ZeroPoint: an energy-based decision engine},
  year   = {2026},
  url    = {https://github.com/Javihaus/zeropoint}
}
```

<br>

## License

Apache 2.0. The model inside the package is derived from
[multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) (MIT licence).

<br>

<div align="center">

<sub>Copyright Javier Marín, 2026</sub>

</div>
