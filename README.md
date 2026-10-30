# ZeroPoint

**An energy-based decision engine. It chooses one option from a list, or says that none of them fits.**

Many systems have to pick from a list: the support queue for a ticket, the intent behind a request, the route
for a document. A classifier always picks something, even when the right option is not on the list or the
text has nothing to do with the task. ZeroPoint returns the best option or `None`, and the probability of
"none" comes out of the same calculation as the probability of each option.

It runs on a CPU in milliseconds, needs no PyTorch to make decisions, and can be calibrated on your own data
in a few minutes.

```bash
pip install zeropoint
```

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

## How it works

ZeroPoint is an energy-based model ([LeCun et al., 2006](https://yann.lecun.org/exdb/publis/pdf/lecun-06.pdf)).
Instead of scoring options against each other, it gives every option an energy and compares it with a
reference state whose energy is fixed.

**The text.** A frozen multilingual encoder ([E5](https://huggingface.co/intfloat/multilingual-e5-small))
turns the text into a vector. A small learned map places it as a point $z$ in a space of 256 dimensions.

**The options.** Each option $y$ is a well in that space, built from the option's own text: a centre $c_y$, a
depth $A_y$ and a width $s_y$.

$$E_y(z) = -A_y \exp\left(-\frac{\lVert z - c_y \rVert^2}{s_y^2}\right)$$

Near the centre the energy is low. Far from it the well is flat and the energy is exactly zero.

**The vacuum.** "None of the options" is a state of its own, with energy fixed at zero. This is the zero
point. Each option is measured against it with Boltzmann's rule at a learned temperature $T$:

$$w_y = e^{-E_y/T} - 1, \qquad w_{\text{none}} = 1, \qquad p(y) = \frac{w_y}{w_{\text{none}} + \sum_k w_k}$$

A text that falls inside a well gives that option a large weight. A text that falls in no well gives every
option a weight close to zero, and the probability stays with "none". With an ordinary softmax the options
share all the probability among themselves, so the model cannot say that none of them fits; here it can,
because the vacuum's energy does not move.

**The edge of the data.** On the unit sphere where the learned points live, a text from another world can
still land near a well. So the vacuum also deepens when the text is far from everything the model has seen,
measured in the encoder's own space (Mahalanobis distance to the nearest known class):

$$w_{\text{none}} = \exp\big(\kappa \cdot \max(0, u)\big)$$

where $u$ is that distance rescaled so that 0 marks the 99th percentile of in-domain texts, and $\kappa = 2$. Inside the data
nothing changes. The edge is set from in-domain texts only: no example of "out of scope" is needed.

**Calibration** learns the map, the wells and the temperature from labelled texts. During training a
share of the examples have their right option removed from the list, with "none" as the answer, so the
model learns to leave the probability with the vacuum when nothing fits.

Related ideas: deterministic uncertainty with distances to class centroids
([DUQ, van Amersfoort et al., 2020](https://arxiv.org/abs/2003.02037)), and world models whose energy is learned up to
a constant ([LeHami, Yalovetzky et al., 2026](https://arxiv.org/abs/2610.15267)). In ZeroPoint the constant is
fixed, and that is what makes it possible to say "none".

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

## Speed

On a modest 2-core cloud CPU, one decision takes about **19 ms** with the default settings (small encoder,
8-bit). The base encoder takes about 30 ms. No GPU is needed; a recent laptop is faster still.

The engine itself takes a millisecond or less. Almost all the time goes to the encoder, which runs with
[ONNX Runtime](https://onnxruntime.ai) in 8-bit precision by default (`precision="fp32"` for full
precision). Details in [BENCHMARK.md](BENCHMARK.md).

## How good it is

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

## What to keep in mind

- ZeroPoint sees texts through a frozen encoder. If the encoder does not tell two kinds of text apart,
  ZeroPoint will not either.
- The shipped models know their own options. You can pass new option texts, and they get a well built from
  their words, but only calibration on your data makes those wells reliable.
- When the right option is on the list, ZeroPoint chooses about as well as an ordinary classifier trained on
  the same data. What it adds is the "none".

## Calibration on your data, as a service

If you would rather have it done for you, with a report on where to set the threshold and what each
choice costs in errors, write to javier@jmarin.info.

## Citation

```bibtex
@software{marin_zeropoint_2026,
  author = {Marín, Javier},
  title  = {ZeroPoint: an energy-based decision engine},
  year   = {2026},
  url    = {https://github.com/Javihaus/zeropoint}
}
```

## License

Apache 2.0. The encoders are downloaded from the Hugging Face Hub at first use and keep their own MIT
license.
