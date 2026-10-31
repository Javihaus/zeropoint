# Making decisions

## Loading a landscape

A landscape is what ZeroPoint knows: one well for each option it can choose, and flat ground at energy zero
that means *none of these*. Two landscapes ship with the package.

```python
import zeropoint
from zeropoint import ZeroPoint

zeropoint.landscapes()
```

```text
everyday    150 options   everyday requests to an assistant: banking, travel, home, work, small talk   (default)
support      10 options   customer support tickets, routed to a queue
```

```python
zp = ZeroPoint.load()             # everyday, small encoder, 8-bit
zp = ZeroPoint.load("support")    # support tickets
zp = ZeroPoint.load("my_model.safetensors")   # a landscape you calibrated yourself
```

`zp.labels` lists the options a landscape knows.

## Choosing among options

`decide` takes a text and, optionally, a list of options.

```python
d = zp.decide("Can you move 200 euros from my savings to my checking account?",
              options=["transfer", "balance", "pay bill"])
d.choice         # 'transfer'
d.confidence     # 0.98
```

- **With `options`**, ZeroPoint chooses among them. An option the landscape does not know gets a well built
  from its own text, so `options` may include new names. Known options work best.
- **Without `options`**, ZeroPoint chooses among every option the landscape knows.

## What a decision holds

| Field | What it holds |
|---|---|
| `choice` | the chosen option, or `None` |
| `confidence` | probability of the most probable outcome: the chosen option, or *none* |
| `p_none` | probability that none of the options fits |
| `probabilities` | one probability per option; together with `p_none` they add up to 1 |
| `energies` | energy of the text in each option's well; 0 means the text is not in that well at all |
| `beyond_edge` | how far the text lies from the texts the landscape was calibrated on; zero or below means familiar |
| `latency_ms` | time the decision took, in milliseconds |

Printing a decision shows a small table with the five most probable options, *none*, the choice and the
time.

## Setting a threshold

By default `choice` is the most probable outcome, even when its probability is low. `min_confidence` makes
ZeroPoint return `None` unless the chosen option reaches that probability.

```python
d = zp.decide("Write me a poem about the sea", min_confidence=0.5)
d.choice         # None
d.confidence     # 0.28, below the threshold
```

The right threshold depends on the cost of acting on a wrong answer. Choose it on held-out data:
[Calibrating your own landscape](calibration.md#choosing-a-threshold) shows how.

## Many texts at once

`decide_many` takes a list of texts and returns one decision per text.

```python
decisions = zp.decide_many(texts, options=["transfer", "balance", "pay bill"])
```

`options` can be `None` (every known option), one list shared by all texts, or one list per text:

```python
decisions = zp.decide_many(["text one", "text two"],
                           options=[["transfer", "balance"], ["alarm", "timer", "reminder"]])
```

In `decide_many`, `latency_ms` is the time of the whole call divided by the number of texts.

## A pattern for production

Act when ZeroPoint is confident, and hand the case to a person when it is not:

```python
d = zp.decide(text, options=queues, min_confidence=0.6)

if d.choice is None:
    send_to_person(text)          # no option fits, or the confidence is below the threshold
else:
    route(text, d.choice)
```

Logging `p_none` and `beyond_edge` over time shows requests that your options do not cover and changes in the
kind of texts you receive.
