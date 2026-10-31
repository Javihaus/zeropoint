# Running in production

## Encoder and precision

| Setting | Values | Default | Effect |
|---|---|---|---|
| `encoder` | `"small"`, `"base"` | `"small"` | The base encoder is somewhat more accurate and about 1.5 times slower. |
| `precision` | `"int8"`, `"fp32"` | `"int8"` | 8-bit is up to three times faster on a CPU and costs at most about 0.01 AUROC. |

```python
zp = ZeroPoint.load("support", encoder="base", precision="fp32")
```

Each landscape stores the edge of its data measured at both precisions, so neither setting is penalized.

Time per decision on a 2-core cloud CPU, median:

| Encoder, precision | Time per decision |
|---|---|
| small, int8 (default) | 19 ms |
| small, fp32 | 26 ms |
| base, int8 | 29 ms |
| base, fp32 | 106 ms |

Almost all of it is the encoder; the engine itself takes a millisecond or less. Run
`python benchmarks/latency.py` from the repository to measure on your own machine.

## Threads

`threads` sets how many CPU threads ONNX Runtime uses. By default ONNX Runtime chooses, usually one per physical core.

```python
zp = ZeroPoint.load(threads=2)
```

When several processes share a machine, give each one a few threads instead of letting all of them use every
core.

## GPU

Install `onnxruntime-gpu` instead of `onnxruntime` and use full precision. ZeroPoint picks CUDA when it is
available and the precision is `fp32`.

```python
zp = ZeroPoint.load(precision="fp32")
```

To choose the execution providers yourself, pass `providers`, for example
`providers=["CUDAExecutionProvider", "CPUExecutionProvider"]`.

## Same text, same decision

With the 8-bit encoder, ZeroPoint encodes one text at a time. An 8-bit encoder quantizes each batch as a
whole, so a text's result could otherwise change slightly with the other texts in its batch. One text per run
keeps every decision identical, whether it comes from `decide` or `decide_many`.

The encoders are pinned to a fixed revision on the Hugging Face Hub, so an embedding computed today is the
same embedding next year.

## Offline use

The encoder is downloaded once and kept in the Hugging Face cache (`~/.cache/huggingface` by default; set
`HF_HOME` to move it). On a machine without network access:

1. Load the landscape once on a machine with network access, and copy the cache folder.
2. On the offline machine, set `HF_HOME` to the copied folder and `HF_HUB_OFFLINE=1`.

The landscapes that ship with the package are inside the package itself and need no download.

## Startup

Loading the encoder takes about a second; ZeroPoint does it on the first decision and keeps one encoder per
process. To keep that second out of the first request, make one decision at startup:

```python
zp = ZeroPoint.load()
zp.decide("warm up")
```

## Saved models across versions

A model saved with `save` records the format it was written in. A newer version of ZeroPoint reads models saved
by older ones. An older version refuses a model saved in a newer format, with a message asking to upgrade.
