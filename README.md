<div align="center">

<img src="https://raw.githubusercontent.com/Javihaus/zeropoint/main/assets/zeropoint-circle-large.png" width="200" alt="ZeroPoint logo">

<br>

# ZeroPoint: a lightweight System 1 decision engine

**Picks the option that fits, or says none does**

<br>

[![License](https://img.shields.io/github/license/Javihaus/zeropoint?color=0000FF&cacheSeconds=86400)](https://github.com/Javihaus/zeropoint/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/zeropoint?color=4169E1&cacheSeconds=3600)](https://pypi.org/project/zeropoint/)
[![Python](https://img.shields.io/pypi/pyversions/zeropoint?color=1E90FF)](https://pypi.org/project/zeropoint/)
[![Docs](https://img.shields.io/badge/docs-javihaus.github.io-6a8bb5)](https://javihaus.github.io/zeropoint/)

[![Tests](https://github.com/Javihaus/zeropoint/actions/workflows/tests.yml/badge.svg)](https://github.com/Javihaus/zeropoint/actions/workflows/tests.yml)
[![SLSA 3](https://slsa.dev/images/gh-badge-level3.svg)](https://slsa.dev)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/Javihaus/zeropoint/badge)](https://scorecard.dev/viewer/?uri=github.com/Javihaus/zeropoint)

<br>

</div>

<table>
<thead>
<tr><th></th><th>ZeroPoint<br><sub>decision engine</sub></th><th>Laya + "None of these"<br><sub>decision engine</sub></th><th>Jev ¹<br><sub>decision engine</sub></th><th>Semantic Router<br><sub>zero-shot classification</sub></th><th>Zero-shot NLI ²<br><sub>zero-shot classification</sub></th></tr>
</thead>
<tbody>
<tr><td><b>Accuracy</b>: right answer, <i>none</i> included</td><td><b>63.6%</b></td><td>59.9%</td><td>not published</td><td>34.6%</td><td>28.6%</td></tr>
<tr><td><b>Accuracy</b> when one option is right</td><td>68.5%</td><td><b>72.2%</b></td><td>not published ³</td><td>56.8%</td><td>50.6%</td></tr>
<tr><td><b>Silent mistakes</b>: no option fits, yet it answers with ≥ 90% confidence</td><td><b>6.9%</b></td><td>21.4%</td><td>not published</td><td>– ⁴</td><td>30.2%</td></tr>
<tr><td><b>Latency</b>, median</td><td>21 ms</td><td>383 ms</td><td>70–500 ms</td><td><b>9 ms</b></td><td>1,316 ms</td></tr>
<tr><td><b>Runs on</b></td><td>CPU, 89 MB model inside the package</td><td>CPU or GPU, 1.5 GB of models</td><td>hosted API</td><td>CPU, 91 MB</td><td>CPU or GPU, 369 MB</td></tr>
</tbody>
</table>

<sub>Measured on the same machine (2-core CPU, 2 threads, one case at a time) on eight tests and 7,537 cases,
about half of them with no right option: SNIPS, ATIS, HWU64, MTOP (Spanish), CLINC150, two support-ticket tests
and a 2,000-case test. Mean over the eight tests. Laya 0.4.21 with "None of these" added as one more option (its
authors report 33 ms on a T4 GPU); Semantic Router 1.1.1 with its default encoder; zero-shot NLI with
DeBERTa-v3-base. ¹ Jev is a hosted model whose licence does not allow benchmarking it here; latency as published by
its maker. ² A fixed sample of 300 cases per test, because it runs its model once per option. ³ Its maker reports
67.8% on its own tasks, which are not public. ⁴ Semantic Router gives similarities, not probabilities. Per-test
results and the scripts to run it all again: [BENCHMARK.md](https://github.com/Javihaus/zeropoint/blob/main/BENCHMARK.md).</sub>

<br>

## ZeroPoint at a glance

<table>
<tr>
<th width="25%">Reliability</th>
<th width="25%">Latency</th>
<th width="25%">Deterministic</th>
<th width="25%">Multilingual</th>
</tr>
<tr>
<td valign="top"><h3>3× fewer silent mistakes</h3><sub>When no option fits, it answers wrongly with 90% confidence or more in 7% of cases; Laya in 21%. Those are the errors that pass a confidence threshold and reach your users unreviewed.</sub></td>
<td valign="top"><h3>21 ms on a CPU</h3><sub>18 times faster than Laya, no GPU. The 89 MB model ships inside the package: nothing is downloaded at run time.</sub></td>
<td valign="top"><h3>Same input, same answer</h3><sub>The order of the options never changes the decision; Laya's changes in 15% of cases. Same decisions on Linux, macOS and Windows, tested on every change.</sub></td>
<td valign="top"><h3>25 languages</h3><sub>EU languages in the Latin script, plus Catalan, Basque and Galician. <a href="https://github.com/Javihaus/zeropoint/blob/main/LANGUAGES.md">Which ones, and why</a>.</sub></td>
</tr>
</table>
<br>

Routing a support ticket, finding the intent behind a request, sending a document to the right process: each means picking one option from a list. A classifier always picks something, even when the right option is not on the list or the text has nothing to do with the task. ZeroPoint can answer *none*, and the probability of *none* comes out of the same calculation as the probability of each option.

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
→ transfer · 12 ms
```

<br>

When none of the options fits, ZeroPoint says so:

```python
zp.decide("Set an alarm for 6:30 tomorrow morning", ["transfer", "balance", "pay bill"])
```

```text
pay bill  0.00  ····················
transfer  0.00  ····················
balance   0.00  ····················
none      1.00  ████████████████████
→ none · 12 ms
```

<br>

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
→ facturación · 16 ms
```

Documentation: [javihaus.github.io/zeropoint](https://javihaus.github.io/zeropoint/).

<br>

## Integration

ZeroPoint runs in your process: one call per decision, no network, no GPU, no service to run.
In a routing pipeline, gate on confidence: auto-route above your threshold and escalate the rest
to human review.

```python
d = zp.decide(text, ["transfer", "balance", "pay bill"])

if d.choice is None or d.confidence < 0.8:
    send_to_person(text)        # no option fits, or ZeroPoint is not sure enough
else:
    route(text, d.choice)       # e.g. "transfer"
```

The right threshold depends on what a wrong action costs you; pick it on a few hundred of your own labelled texts.

| Attribute | Description |
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

## Calibration

You can calibrate ZeroPoint on your own labelled data, on a CPU.

```bash
pip install "zeropoint[fit]"        # adds PyTorch and onnx; deciding never needs them
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

ZeroPoint is an energy-based model, not a generative LLM. It does not write an answer: it measures one number 
per option, its energy, which is lower the better the option fits the text. The text and all the options are
read together, in one pass of a small multilingual encoder, so each option is measured against the text and 
against the other options. *None of the options* is a state of its own, with its energy fixed at zero: the 
flat ground around the wells the options form.

Energies become probabilities with Boltzmann’s rule, the rule physics uses for the states of a system. A 
classifier applies it to the options alone, so one of them always wins, even when the text has nothing 
to do with any of them. ZeroPoint applies it with the ground included, and counts each option only by 
how far it falls below zero: an option that does not fall below the ground gets nothing, however many 
such options there are. *None* is an answer in its own right, with its probability computed in the same 
step as every option’s.

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

ZeroPoint was trained only on data whose licence allows commercial use:

| Data | Licence |
|---|---|
| CLINC150 | CC BY 3.0 |
| Banking77, MASSIVE (English and Spanish) | CC BY 4.0 |
| Bitext customer support, banking, insurance and telco | CDLA-Sharing 1.0 |
| 5,000 examples written for this project | Apache 2.0 |

No data with a non-commercial licence was used for training, and none of the test texts above is in the training
data. Details, with the criteria used to write the 5,000 examples, in
[docs/training-data.md](https://github.com/Javihaus/zeropoint/blob/main/docs/training-data.md).

The code is Apache 2.0. The model is built on
[multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) (MIT licence); the notices are in
[NOTICE](https://github.com/Javihaus/zeropoint/blob/main/NOTICE).

<br>

<div align="center">

<sub>Copyright Javier Marín, 2026</sub>

</div>