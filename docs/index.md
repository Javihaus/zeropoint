# ZeroPoint

ZeroPoint chooses one option from a list for a given text, or answers that none of the options fits. It does
this in a single forward pass of a small encoder, on a CPU, in about 19 ms.

Typical uses are routing a support ticket to a queue, finding the intent behind a request, or sending a
document to the right process. A classifier always picks something. ZeroPoint returns `None` when the right
option is not on the list or the text has nothing to do with the task.

## Install

```bash
pip install zeropoint
```

ZeroPoint needs Python 3.10 or newer. It runs on Linux, macOS and Windows, with
[ONNX Runtime](https://onnxruntime.ai); PyTorch is only needed to [calibrate your own landscape](calibration.md).

## First decision

```python
from zeropoint import ZeroPoint

zp = ZeroPoint.load()
zp.decide("Can you move 200 euros from my savings to my checking account?",
          options=["transfer", "balance", "pay bill"])
```

```text
transfer  0.98  ███████████████████▋
balance   0.01  ▏
pay bill  0.00
none      0.01  ▏
→ transfer · 8 ms
```

```python
zp.decide("Set an alarm for 6:30 tomorrow morning",
          options=["transfer", "balance", "pay bill"])
```

```text
pay bill  0.00
transfer  0.00
balance   0.00
none      1.00  ███████████████████▉
→ none · 7 ms
```

The first call downloads the encoder from the Hugging Face Hub (about 130 MB for the small one) and takes a
few seconds. The encoder is cached, and later calls take milliseconds.

## Where to go next

| Page | What it covers |
|---|---|
| [Making decisions](usage.md) | options, thresholds, many texts at once, what a decision holds |
| [Calibrating your own landscape](calibration.md) | training ZeroPoint on your own labeled texts |
| [Running in production](deployment.md) | encoders, precision, threads, GPU, offline use |
| [API reference](api.md) | every public class, function and argument |
| [How it works](https://github.com/Javihaus/zeropoint#energy-based-models) | the energy landscape, Boltzmann's rule and the edge of the data |
| [Benchmark](https://github.com/Javihaus/zeropoint/blob/main/BENCHMARK.md) | accuracy, *none* detection and latency, with the scripts to rebuild them |
