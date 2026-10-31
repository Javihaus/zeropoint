# Calibrating your own landscape

The shipped landscapes know everyday requests and one style of support ticket. For your own options and your
own texts, calibrate a landscape on your data.

## What you need

- Texts with their right label. A few hundred per label is a good start.
- No examples of *none*. ZeroPoint learns where the edge of your data is from your labeled texts alone.
- PyTorch, only for calibrating:

```bash
pip install "zeropoint[train]"
```

## Calibrate, save, load

```python
from zeropoint import ZeroPoint

zp = ZeroPoint.fit(texts, labels)
zp.save("my_model.safetensors")

zp = ZeroPoint.load("my_model.safetensors")
zp.decide("...")
```

The encoder stays frozen. Its embeddings are computed once and training runs on them, so a few thousand texts
calibrate in a minute or two on a CPU.

The saved file holds the learned map, the wells, the temperature and the edge of the data, measured for both
8-bit and full precision. It does not hold the encoder, which is downloaded from the Hugging Face Hub on first
use.

## Options

`fit` takes these keyword arguments:

| Argument | Default | What it does |
|---|---|---|
| `descriptions` | `None` | `{label: description}`. Each well is built from "label: description", which helps when labels are short codes such as `"BIL"`. |
| `catch_all` | `()` | Labels that fit anything, such as `"Other"` or `"General Inquiry"`. They are never removed from a list during training, because with them on the list some option always fits. |
| `encoder` | `"small"` | `"small"`, `"base"`, or the name of an E5-style model on the Hugging Face Hub with an `onnx/` folder. |
| `precision` | `"fp32"` | Precision of the encoder while calibrating. Full precision is recommended. |
| `seed` | `0` | Training seed. |
| `monitor` | `False` | Track how many dimensions the learned space uses, with [ndt](https://pypi.org/project/ndtracker/). Needs `pip install "zeropoint[monitor]"`. |
| `embeddings` | `None` | Precomputed `query` embeddings of `texts`, to skip the encoding pass. |
| `recipe` | `None` | Overrides for any value in `zeropoint.calibrate.RECIPE` (epochs, learning rate, size of the learned space). |
| `device` | `None` | PyTorch device for training, such as `"cuda"`. |
| `verbose` | `True` | Print progress. |

```python
zp = ZeroPoint.fit(texts, labels,
                   descriptions={"BIL": "billing and invoices", "TEC": "technical problems"},
                   catch_all=["Other"])
```

## When calibration fails

If the model does not beat chance on held-out texts by ten percentage points, `fit` raises
`zeropoint.calibrate.DidNotLearn` instead of returning a model that has not learned. The usual causes are too
few texts per label, labels that overlap a lot, or labels that the encoder cannot tell apart.

## Choosing a threshold

Keep part of your labeled texts out of calibration. On those texts, compare `confidence` for right and wrong
answers, and pick the `min_confidence` where the share of wrong actions is one you can accept.

[notebooks/calibrate_your_own.ipynb](https://github.com/Javihaus/zeropoint/blob/main/notebooks/calibrate_your_own.ipynb)
walks through the whole process on Banking77, a dataset ZeroPoint never saw: calibration, a held-out check
and a table for choosing the threshold.
