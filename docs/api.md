# API reference

```python
from zeropoint import ZeroPoint, Decision, UnsupportedLanguage, ModelFilesError, __version__
```

## `ZeroPoint(model_dir, threads=2)`

Loads a model: the one shipped with the package when `model_dir` is not given, or a folder written by `save`.
Every model file is checked against its sha256.

| Argument | Default | What it does |
|---|---|---|
| `model_dir` | the shipped model | folder of a saved model |
| `threads` | 2 | CPU threads for the encoder; does not change the results |

Raises `ModelFilesError` if a model file is missing or does not match its hash, and `ValueError` if `threads` is
not a positive integer.

## `ZeroPoint.decide(text, options)`

Chooses the option that fits `text`, or `None`. Returns a `Decision`.

Raises `ValueError` for an empty text, an empty list, an empty option or the same option twice, `TypeError` when
the text or an option is not a string, and `UnsupportedLanguage` when more than 20% of the letters of the text or of
an option are outside the Latin script.

## `ZeroPoint.decide_many(texts, options)`

Decides several texts. `options` is one list for all texts, or one list per text. Returns a list of `Decision`,
each exactly what `decide` gives for that text alone.

## `ZeroPoint.fit(texts, labels, *, groups=None, steps=200, frozen_layers=8, seed=0, threads=None)`

Adapts the model to your labelled texts. `labels` holds the right option for each text, or `None` for texts that
fit none of them. Returns `(model, report)`; the report compares the model before and after on a held-out 20% of
the examples. Needs `pip install "zeropoint[fit]"`. See [Adapting to your labels](calibration.md).

## `ZeroPoint.recalibrate(texts, labels, *, groups=None, seed=0)`

Refits only the temperature and the *none* level. Needs no PyTorch. Returns `(model, report)`.

## `ZeroPoint.save(path)`

Writes the model to the folder `path`, with a manifest holding the sha256 of every file and the report of `fit` or
`recalibrate`. Load it with `ZeroPoint(path)`.

## `ZeroPoint` attributes

| Attribute | What it holds |
|---|---|
| `languages` | codes of the 25 supported languages |
| `model_id` | first 12 characters of the hash of the model's training checkpoint |
| `threads` | CPU threads used |
| `report` | the report of `fit` or `recalibrate`, for a model made by them; `None` otherwise |

## `Decision`

| Field | Type | What it holds |
|---|---|---|
| `choice` | `str` or `None` | the chosen option, or `None` when *none* is the most probable answer |
| `confidence` | `float` | probability of `choice` (of *none* when `choice` is `None`) |
| `p_none` | `float` | probability that none of the options fits |
| `probabilities` | `dict[str, float]` | probability of each option, in the order given; with `p_none` they add up to 1 |
| `depths` | `dict[str, float]` | depth of each option's well, 0 or more |
| `text_truncated` | `bool` | the text was longer than 192 word pieces and its end was ignored |
| `options_truncated` | `tuple[str, ...]` | options longer than 16 word pieces, whose end was ignored |
| `blocks` | `int` | reads of 20 options the list needed |
| `latency_ms` | `float` | time the decision took |

Printing a `Decision` shows the options with their probabilities, *none*, the choice and the time.

## Exceptions

| Exception | Raised when |
|---|---|
| `UnsupportedLanguage` (a `ValueError`) | the text or an option is mostly outside the Latin script |
| `ModelFilesError` (a `RuntimeError`) | a model file is missing or does not match its sha256 |
