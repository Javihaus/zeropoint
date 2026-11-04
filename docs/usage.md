# Making decisions

## Loading the model

```python
from zeropoint import ZeroPoint

zp = ZeroPoint()
```

Load it once and reuse it. Loading checks every model file against its sha256 and takes about a second.
`ZeroPoint("my-model")` loads a model you adapted and saved yourself.

## Choosing among options

```python
d = zp.decide("my card was swallowed by the cash machine",
              ["block a card", "check the balance", "order a new card"])
d.choice        # the chosen option, or None
```

The options are any short texts: intents, queue names, categories. ZeroPoint reads the text and all the options
together, so the wording of every option matters. Write them as a person would read them ("Billing and
Payments", "order a new card"), not as codes ("BIL_02").

- The order of the options does not change the result.
- A list can be of any length. Options are read in blocks of 20 with the text, and all probabilities come from one
  distribution over the whole list.
- An empty option, or the same option twice, raises `ValueError`.

## What a decision holds

| Field | What it holds |
|---|---|
| `choice` | the chosen option, or `None` when *none* is the most probable answer |
| `confidence` | probability of `choice` (of *none* when `choice` is `None`) |
| `p_none` | probability that none of the options fits |
| `probabilities` | probability of each option, in the order you gave them; with `p_none` they add up to 1 |
| `depths` | how strongly the text belongs to each option (0: not at all) |
| `text_truncated` | `True` if the text was longer than 192 word pieces and its end was ignored |
| `options_truncated` | options longer than 16 word pieces, whose end was ignored |
| `blocks` | how many reads of 20 options the list needed |
| `latency_ms` | time the decision took |

## Setting a threshold

By default ZeroPoint returns its most probable answer. To act only when it is confident enough, compare
`confidence` with your own threshold:

```python
d = zp.decide(text, options)
if d.choice is None or d.confidence < 0.8:
    send_to_person(text)
else:
    route(text, d.choice)
```

The right threshold depends on what a wrong action costs. Choose it on a few hundred of your own labelled texts:
for each threshold, count how many cases would be acted on and how many of those would be wrong.

## Many texts at once

```python
decisions = zp.decide_many(texts, ["transfer", "balance", "pay bill"])        # one list for all texts
decisions = zp.decide_many(texts, [options_1, options_2, options_3])          # one list per text
```

Each text is computed on its own, so every result is exactly what `decide` gives for that text alone.

## Languages

ZeroPoint reads 25 languages written in the Latin script: Basque, Catalan, Croatian, Czech, Danish, Dutch, English,
Estonian, Finnish, French, Galician, German, Hungarian, Irish, Italian, Latvian, Lithuanian, Maltese, Polish,
Portuguese, Romanian, Slovak, Slovenian, Spanish and Swedish (`zp.languages`).

A text or an option written mostly in another script raises `UnsupportedLanguage`:

```python
from zeropoint import UnsupportedLanguage

try:
    d = zp.decide(text, options)
except UnsupportedLanguage:
    send_to_person(text)
```

Other languages in the Latin script, such as Turkish or Vietnamese, are not detected: ZeroPoint answers, but it
was not trained or tested on them. Its training texts are in English and Spanish; the other languages are read
through the multilingual encoder it is built on.
