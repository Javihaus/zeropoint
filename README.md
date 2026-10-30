<div align="center">

<img src="assets/zeropoint-circle-large.png" width="300" alt="ZeroPoint logo">

<br>

<br>

[![License](https://img.shields.io/github/license/Javihaus/zeropoint?color=0000FF&cacheSeconds=86400)](https://github.com/Javihaus/zeropoint/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/zeropoint?color=4169E1&cacheSeconds=3600)](https://pypi.org/project/zeropoint/)
[![Python](https://img.shields.io/pypi/pyversions/zeropoint?color=1E90FF)](https://pypi.org/project/zeropoint/)
[![Tests](https://github.com/Javihaus/zeropoint/actions/workflows/tests.yml/badge.svg)](https://github.com/Javihaus/zeropoint/actions/workflows/tests.yml)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/16100/badge)](https://www.bestpractices.dev/projects/16100)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/Javihaus/zeropoint/badge)](https://scorecard.dev/viewer/?uri=github.com/Javihaus/zeropoint)

# ZeroPoint: energy-based System 1 decision engine
One forward pass turns a text into a choice among a list of options, with “none of them” as one of the possible answers.

<br>

### :dart: A classifier that picks an option when none is right is hallucinating

<br>


| | ZeroPoint | Laya | Jev | Similarity |
|---|---|---|---|---|
| Latency per decision | **19 ms** on a 2-core CPU | 33 ms on a T4 GPU; 193–464 ms on CPU | 70–500 ms | 15 ms on a 2-core CPU |
| Picks the right option when it is on the list | **97%** | 95% | not public | 80% |
| Out-of-scope requests still acted on, keeping 95% of the good ones | **10%** | 32% | not public | 52% |

<sub> Accuracy rows: CLINC150, 1,600 cases never seen in training (800 answerable requests, 800 out-of-scope
requests, 20 options per case), the same cases for every system measured. Similarity uses the same encoder as
ZeroPoint. Latency: ZeroPoint and similarity measured by us; Laya and Jev as published by their authors
([Laya](https://pypi.org/project/laya/), [Jev](https://www.truefoundry.com/blog/typesafe-ai-jev)), on their
own hardware. Jev is in early access and could not be tested. Full results, including support tickets, where
Laya chooses better, are in [BENCHMARK.md](BENCHMARK.md).</sub>

<br>

</div>
Many systems have to pick from a list: the support queue for a ticket, the intent behind a request, the route
for a document. A classifier always picks something, even when the right option is not on the list or the
text has nothing to do with the task. ZeroPoint returns the best option or `None`, and the probability of
"none" comes out of the same calculation as the probability of each option.

It runs on a CPU in milliseconds, needs no PyTorch to make decisions, and can be calibrated on your own data
in a few minutes.

```bash
pip install zeropoint
```
<br>

## A first decision

```python
from zeropoint import ZeroPoint

zp = ZeroPoint.load("clinc150")     # 150 everyday requests: banking, travel, home, work, small talk

zp.decide("Can you move 200 euros from my savings to my checking account?")
# Decision(choice='transfer', confidence=0.88, p_none=0.01)

zp.decide("Who won the football match yesterday?")
# Decision(choice=None, confidence=0.38, p_none=0.38)
```

Give it the options that are open at that moment. If the right one is missing, it says so:

```python
zp.decide("Can you move 200 euros from my savings to my checking account?",
          options=["balance", "pay bill", "freeze account"])
# Decision(choice=None, confidence=0.50, p_none=0.50)

zp.decide("Can you move 200 euros from my savings to my checking account?",
          options=["balance", "transfer", "freeze account"])
# Decision(choice='transfer', confidence=0.98, p_none=0.01)
```

With many options, a text that fits none of them can still leave a weak favourite. A low confidence is the
warning, and `min_confidence` turns it into a `None`:

```python
zp.decide("Write me a poem about the sea")
# Decision(choice='timezone', confidence=0.28, p_none=0.16)

zp.decide("Write me a poem about the sea", min_confidence=0.5)
# Decision(choice=None, confidence=0.28, p_none=0.16)
```

The right threshold depends on what a wrong action costs you.
[notebooks/calibrate_your_own.ipynb](notebooks/calibrate_your_own.ipynb) shows how to read it off your own data.

For many texts, `decide_many(texts)` takes the same arguments, plus one list of options per text if they
differ. Each `Decision` carries `choice`, `confidence`, `p_none`, the `probabilities` and `energies` of every
option, and `beyond_edge`, which says how far the text lies outside the data the model was calibrated on
(zero or below means inside).

### The models that come with it

| name | what it decides | learned from |
|---|---|---|
| `"clinc150"` | 150 intents of everyday requests | [CLINC150](https://huggingface.co/datasets/clinc_oos), 15,000 requests written by people |
| `"tickets"` | 10 support queues | a [public dataset](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) of 20,000 support tickets |

The tickets model is there to try the engine on longer texts. It knows tickets in the style of its dataset
(a subject line and a body) and treats very different writing as outside what it knows. For your own
tickets, calibrate on your own data.

Each model comes with two encoders: `encoder="small"` (default, faster) and `encoder="base"` (more accurate,
slower).

<br>

## How it works

ZeroPoint is an energy-based model ([LeCun et al., 2006](http://yann.lecun.com/exdb/publis/orig/lecun-06.pdf)).
Instead of scoring options against each other, it gives every option an energy and compares it with a
reference state whose energy is fixed.

#### 1. The text
We use [Multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small), a frozen multilingual encoder that
turns the text into a vector. A small learned map places it as a point $z$ in a space of 256 dimensions.

#### 2. The options
Each option $y$ is a well in that space, built from the option's own text: a centre $c_y$, a
depth $A_y$ and a width $s_y$.

$$E_y(z) = -A_y \exp\left(-\frac{\lVert z - c_y \rVert^2}{s_y^2}\right)$$

Near the centre the energy is low. Far from it the well is flat and the energy is exactly zero.

#### 3. The vacuum
"None of the options" is a state of its own, with energy fixed at zero. This is the zero
point. Each option is measured against it with Boltzmann's rule at a learned temperature $T$:

$$w_y = e^{-E_y/T} - 1, \qquad w_{\text{none}} = 1, \qquad p(y) = \frac{w_y}{w_{\text{none}} + \sum_k w_k}$$

A text that falls inside a well gives that option a large weight. A text that falls in no well gives every
option a weight close to zero, and the probability stays with "none". With an ordinary softmax the options
share all the probability among themselves, so the model cannot say that none of them fits; here it can,
because the vacuum's energy does not move.

#### 4. The edge of the data
On the unit sphere where the learned points live, a text from another world can
still land near a well. So the vacuum also deepens when the text is far from everything the model has seen,
measured in the encoder's own space (Mahalanobis distance to the nearest known class):

$$w_{\text{none}} = \exp\big(\kappa \cdot \max(0, u)\big)$$

where $u$ is that distance rescaled so that 0 marks the 99th percentile of in-domain texts, and $\kappa = 2$. Inside the data
nothing changes. The edge is set from in-domain texts only: no example of "out of scope" is needed.

**Calibration** learns the map, the wells and the temperature from labelled texts. During training a
share of the examples have their right option removed from the list, with "none" as the answer, so the
model learns to leave the probability with the vacuum when nothing fits.

Related ideas: deterministic uncertainty with distances to class centroids
([Van Amersfoort et al., 2020](https://arxiv.org/abs/2003.02037)), and world models whose energy is learned up to
a constant ([LeHami et al., 2026](https://arxiv.org/abs/2610.15267)). In ZeroPoint the constant is
fixed, and that is what makes it possible to say "none".

<br>

## Calibrate it on your own data

You need texts with their right label, a few hundred per label to start. No examples of "none".

```bash
pip install "zeropoint[train]"        # adds PyTorch, only needed to calibrate
```

```python
zp = ZeroPoint.fit(texts, labels,
                   descriptions={"BIL": "billing and invoices"},   # optional, helps with terse labels
                   catch_all=["Other"])                            # optional, labels that fit anything
zp.save("my_model.safetensors")

ZeroPoint.load("my_model.safetensors").decide("...")
```

The encoder stays frozen and its embeddings are computed once, so a few thousand texts calibrate in a few
minutes on a laptop. If the model does not beat chance on held-out texts by ten points, `fit` stops with an
error instead of returning a broken model.

To watch training from the inside, install `zeropoint[monitor]` and pass `monitor=True`.
[ndt](https://pypi.org/project/ndtracker/) tracks how many dimensions the learned space really uses; a
collapse shows up there even when the loss looks fine.

A complete walk-through, with a held-out check and a table to choose your threshold, is in
[notebooks/calibrate_your_own.ipynb](notebooks/calibrate_your_own.ipynb).

<br>

## Latency

On a modest 2-core cloud CPU, one decision takes about **19 ms** with the default settings (small encoder,
8-bit). The base encoder takes about 30 ms. No GPU is needed; a recent laptop is faster still.

The engine itself takes a millisecond or less. Almost all the time goes to the encoder, which runs with
[ONNX Runtime](https://onnxruntime.ai) in 8-bit precision by default (`precision="fp32"` for full
precision). Details in [BENCHMARK.md](BENCHMARK.md).

<br>

## Accuracy

How often it tells cases with a right option apart from cases without one (AUROC: 1 is perfect, 0.5 a coin
toss), on texts it never saw. Small encoder, default settings:

| test | ZeroPoint | similarity, same encoder |
|---|---|---|
| support tickets vs unrelated texts (reviews, forum posts) | **1.00** | 0.99 |
| support tickets whose right queue was removed from the list | **0.71** | 0.59 |
| everyday requests vs out-of-scope requests (CLINC150) | **0.98** | 0.90 |
| everyday requests whose right intent was swapped for a wrong one | **0.97** | 0.78 |

When the right option is on the list it chooses about as well as a classifier trained on the same data:
97% on CLINC150 with 20 options per request.

Full results, the baselines and the limits are in [BENCHMARK.md](BENCHMARK.md).

<br>

## References

- LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. (2006). A tutorial on energy-based learning. Predicting structured data, 1(0).
- Van Amersfoort, J., Smith, L., Teh, Y. W., & Gal, Y. (2020, November). Uncertainty estimation using a single deep deterministic neural network. In International conference on machine learning (pp. 9690-9700). PMLR.
- Yalovetzky, R., Balestriero, R., & LeCun, Y. (2026). LeHami: Symplectic Latent Dynamics for World Models. arXiv preprint arXiv:2610.15267.

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

Apache 2.0. The encoders are downloaded from the Hugging Face Hub at first use and keep their own MIT
license.

<br>

<div align="center">

<sub> Copyright Javier Marín, 2024 <sub>

</div>
