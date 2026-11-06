# Training the released model

`zp1.py` is the script that trained ZeroPoint 1.0 (run `1.0c`, seed 0) on an NVIDIA L4 GPU in Google Colab
(`zeropoint_1.0c.ipynb`). The released ONNX files were exported from that checkpoint with
`tools/export_model.py`.

    python zp1.py train --seed 0          # trains, then evaluates every test
    python zp1.py checks --ckpt results/s0.pt
    python zp1.py fit --ckpt results/s0.pt

The training texts themselves are not in this repository. Their sources and licences are listed in
[docs/training-data.md](../docs/training-data.md). `build_hierarchy.py`, `build_neighbors.py` and
`forbidden_labels.json` build the broad labels, the confusable options and the list of labels that must never
match a test label. `synth/` describes how the 5,000 written examples were produced.

## Results

`runs/1.0c/` is the released model:

- `s0_train_info.json`: every setting of the run, training time, and the refit of the temperature `T` and the
  *none* level `b` on 982 held-out lists (before and after, with log-likelihood and calibration error).
- `s0_results.json` and `s0_cases_*.json`: the eight tests, summary and case by case, for the training checkpoint.
- `checks.json`: answers under five orders of the options, lists of 20 to 150 options, 1, 2 and 4 CPU threads, and
  GPU against CPU.
- `fit_results.json`: adaptation with 100 to 1,000 labelled tickets, with all layers trained or the first 8
  frozen.

`runs/1.0b/` holds two seeds of a preliminary version with the same architecture, trained before the vocabulary
reduction, the confusable options, the long lists and the refit were added. It measures the variation between
training seeds.

## A known flaw in the dev split

The dev data is split in two halves by a hash of the text: one half decides when to stop training, the other fits
`T` and `b` afterwards. Written examples reach dev when `sha256(text) % 20 == 0`, and every such number is even, so
the split by `sha256(text) % 2` sends all 263 of them to the stopping half. The calibration half has no written
examples; `T` and `b` were fitted on public texts only. The script is left as it ran, so that run `1.0c` can be
repeated exactly; a future run should split dev with a different hash.

## The released files

The released files store weights in float16 and the vocabulary in int8. Their own measurements, including how
many decisions that storage changes, are in `benchmarks/results/v1_model_measure.json`.
