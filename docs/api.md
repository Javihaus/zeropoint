# API reference

Everything below is importable from `zeropoint`:

```python
from zeropoint import ZeroPoint, Decision, Landscape, landscapes, __version__
```

## `ZeroPoint.load`

```python
ZeroPoint.load(name_or_path="everyday", encoder="small", precision="int8",
               providers=None, threads=None) -> ZeroPoint
```

Load a landscape that ships with the package, or one saved with `save`.

| Argument | Meaning |
|---|---|
| `name_or_path` | `"everyday"` (default), `"support"`, or the path to a `.safetensors` file written by `save` |
| `encoder` | `"small"` (default) or `"base"`; shipped landscapes only |
| `precision` | `"int8"` (default) or `"fp32"`; `None` keeps the precision the model was calibrated with |
| `providers` | ONNX Runtime execution providers; by default CUDA when available with `fp32`, otherwise CPU |
| `threads` | number of CPU threads for ONNX Runtime; by default ONNX Runtime chooses |

Raises `ValueError` when the name is neither a shipped landscape nor an existing file, or when the file is not
a valid ZeroPoint model.

## `ZeroPoint.decide`

```python
zp.decide(text, options=None, min_confidence=None) -> Decision
```

| Argument | Meaning |
|---|---|
| `text` | the text to decide on |
| `options` | list of options to choose from; `None` means every option the landscape knows |
| `min_confidence` | if set, `choice` is `None` unless the chosen option has at least this probability |

Raises `ValueError` when `options` is empty or has repeated entries.

## `ZeroPoint.decide_many`

```python
zp.decide_many(texts, options=None, min_confidence=None, batch_size=None) -> list[Decision]
```

| Argument | Meaning |
|---|---|
| `texts` | list of texts |
| `options` | `None`, one list shared by all texts, or one list per text |
| `min_confidence` | as in `decide` |
| `batch_size` | texts per encoder run; by default 1 with 8-bit, 4 with `fp32` on CPU, 32 on GPU |

## `ZeroPoint.fit`

```python
ZeroPoint.fit(texts, labels, *, descriptions=None, catch_all=(), encoder="small", precision="fp32",
              seed=0, monitor=False, device=None, embeddings=None, recipe=None, verbose=True) -> ZeroPoint
```

Calibrate a landscape on labeled texts. Needs `pip install "zeropoint[train]"`. Every argument is described in
[Calibrating your own landscape](calibration.md#options). Raises `zeropoint.calibrate.DidNotLearn` when the
model does not beat chance on held-out texts.

## `ZeroPoint.save`

```python
zp.save(path)
```

Write the landscape to a `.safetensors` file. Load it back with `ZeroPoint.load(path)`.

## `ZeroPoint` attributes

| Attribute | Meaning |
|---|---|
| `labels` | the options the landscape knows |
| `descriptions` | the text each known option's well was built from |
| `temperature` | the temperature of Boltzmann's rule, learned during calibration |
| `encoder_spec` | the encoder repository, revision and precision |

## `Decision`

The result of one decision. Frozen dataclass.

| Field | Type | Meaning |
|---|---|---|
| `choice` | `str` or `None` | the chosen option, or `None` |
| `confidence` | `float` | probability of the most probable outcome: the chosen option, or *none* |
| `p_none` | `float` | probability that none of the options fits |
| `probabilities` | `dict[str, float]` | probability of each option; with `p_none` they add up to 1 |
| `energies` | `dict[str, float]` | energy of the text in each option's well; 0 means not in the well |
| `beyond_edge` | `float` | distance beyond the edge of the calibration data; zero or below means inside |
| `latency_ms` | `float` | time to decide, in milliseconds |

## `landscapes`

```python
zeropoint.landscapes() -> list[Landscape]
```

The landscapes that ship with the package. Each `Landscape` has `name`, `options` (how many), `description`,
`examples` (a few of its options) and `source` (the data it was calibrated on).
