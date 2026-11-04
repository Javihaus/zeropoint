# ZeroPoint

ZeroPoint chooses the option that fits a text from a list of options, or answers that none of them does. It runs on
a CPU in about 20 ms, and the model ships inside the package.

Typical uses are routing a support ticket to a queue, finding the intent behind a request, or sending a document
to the right process. A classifier always picks something. ZeroPoint returns `None` when the right option is not on
the list or the text has nothing to do with the task.

## Install

```bash
pip install zeropoint
```

ZeroPoint needs Python 3.10 or newer and runs on Linux, macOS and Windows, with
[ONNX Runtime](https://onnxruntime.ai). The package holds the model (89 MB); nothing is downloaded when it runs.
PyTorch is only needed to [adapt it to your labels](calibration.md).

## First decision

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

## Where to go next

| Page | What it covers |
|---|---|
| [Making decisions](usage.md) | options, what a decision holds, thresholds, many texts at once, languages |
| [Adapting to your labels](calibration.md) | `fit` and `recalibrate` on your own labelled texts |
| [Running in production](deployment.md) | threads, long texts, the same numbers on every machine, offline use |
| [API reference](api.md) | every class, method and field |
| [Training data](training-data.md) | what the model was trained on, and under which licences |
| [Benchmark](https://github.com/Javihaus/zeropoint/blob/main/BENCHMARK.md) | ZeroPoint against Strands Decider, Clef-flash, Laya, Semantic Router, zero-shot NLI and hosted LLMs |
