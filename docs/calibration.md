# Adapting to your labels

ZeroPoint works with any list of options out of the box. With a few hundred labelled texts from your own work, it
can learn your labels and how your texts are written.

## What you need

- Texts with their right label: a few hundred in all, and at least a few dozen per label.
- Optionally, texts that fit none of your labels, with the label `None`.
- `pip install "zeropoint[fit]"`, which adds PyTorch and onnx. Deciding never needs them.

## Adapt, save, load

```python
from zeropoint import ZeroPoint

zp = ZeroPoint()
mine, report = zp.fit(texts, labels)
print(report["before"], report["after"])

mine.save("my-model")
zp = ZeroPoint("my-model")
```

What `fit` does:

- Your labels become the options. Each example becomes lists of 4 to 20 of them; in 30% of the lists the right
  option is removed and the answer is *none*, so the model keeps learning to say *none*.
- 20% of your examples are kept apart and never trained on. The report measures the model on them before and
  after.
- The word embeddings and the first 8 encoder layers stay as shipped; the rest of the model is trained for 200
  steps. It runs on a CPU: about 6 minutes for 300 examples on 2 threads.
- Same data, same seed and same number of threads give the same model, bit for bit.

## Options

| Argument | Default | What it does |
|---|---|---|
| `groups` | `None` | one id per text (a customer, a conversation); texts with the same id are kept apart together, so the held-out check is not flattered by near-copies |
| `steps` | 200 | training steps |
| `frozen_layers` | 8 | encoder layers that stay as shipped (0 to 12); fewer frozen layers learn more and need more data |
| `seed` | 0 | random seed |
| `threads` | the model's | CPU threads for training |

## Only the balance between acting and *none*

If your options already work well and you only want ZeroPoint to say *none* more or less often on your texts,
`recalibrate` refits just two numbers, the temperature and the *none* level. It takes seconds and needs no
PyTorch:

```python
mine, report = zp.recalibrate(texts, labels)
```

## An example

On the two support-ticket tests of the [benchmark](https://github.com/Javihaus/zeropoint/blob/main/BENCHMARK.md),
`fit` with 300 labelled tickets raised the share of right queues from 24% to 36% when the right queue is on the
list, and from 29% to 46% when the cases without one are unrelated texts. Confident answers on tickets whose own
queue was missing from the list went from 20% to 0%.
