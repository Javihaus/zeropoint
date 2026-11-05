# Changelog

Every change that reaches a release is listed here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) as described in
[CONTRIBUTING.md](CONTRIBUTING.md#versions-and-releases).

Changes merged since the last release go under **Unreleased** and move to a numbered section when the next
version is published.

## [Unreleased]

## [1.0.0] - 2026-11-05

1.0 is a new model and a new, smaller API. Code written for 0.x needs the changes listed under
*Moving from 0.x*.

### Added

- One model that reads the text and all the options together (a listwise reading) and gives each option a well
  depth; *none* is a state at energy zero in the same Boltzmann distribution. It replaces the landscapes of 0.x.
- The model ships inside the package (89 MB: an encoder in ONNX with 16-bit weights, and its vocabulary reduced to
  25 languages). Nothing is downloaded at run time, and every model file is checked against its sha256 when it
  is loaded (`ModelFilesError` if one was changed).
- 25 languages: the 22 official languages of the European Union written in the Latin script, plus Catalan, Basque
  and Galician. A text mostly in another script raises `UnsupportedLanguage`.
- The decision does not depend on the order of the options, on the number of threads, or on other texts decided
  at the same time. 500 reference cases give the same decisions on Linux (x86 and ARM), macOS and Windows; CI
  checks it on every change.
- `ZeroPoint.fit(texts, labels)`, an instance method that adapts the model to your labels on a CPU and returns the
  new model and a report on a held-out 20% (`pip install "zeropoint[fit]"`). Deterministic for the same data,
  seed and threads.
- `ZeroPoint.recalibrate(texts, labels)`: refits only the temperature and the *none* level, without PyTorch.
- `Decision.depths`, `Decision.text_truncated`, `Decision.options_truncated`, `Decision.blocks`.
- `benchmarks/compare` and `benchmarks/llm`: ZeroPoint against Strands Decider 2B, Clef-flash, Laya, Semantic
  Router, zero-shot NLI and six hosted LLMs, on three tests no model is likely to have seen, and against the CPU
  systems on eight tests; with every per-case output and the scripts and notebooks to run it all again.
  `report.py` writes the tables of BENCHMARK.md.
- `LANGUAGES.md`: the supported languages, why these, and what happens with the others.
- `docs/training-data.md`: the training data, all under licences that allow commercial use.

### Changed

- Lists of any length: options are read in blocks of 20, and all probabilities come from one distribution.
- A duplicate or empty option raises `ValueError`.
- `ZeroPoint(threads=2)` by default.

### Removed

- Landscapes (`everyday`, `support`), `ZeroPoint.load`, `zeropoint.landscapes()`, the `encoder`, `kappa` and
  `precision` options, and the edge of the calibration data with `Decision.beyond_edge`.
- `min_confidence`: compare `Decision.confidence` with your own threshold.
- Deciding without a list of options: the options are always given.
- `Decision.energies`, `Decision.unseen_options` and `Decision.truncated` (see the table below).
- The `train` and `monitor` extras.

### Moving from 0.x

| 0.x | 1.0 |
|---|---|
| `ZeroPoint.load()`, `ZeroPoint.load("support")` | `ZeroPoint()` |
| `ZeroPoint.load("my_model.safetensors")` | `ZeroPoint("my-model")`, a folder written by `save` |
| `zp.decide(text, options=[...])` | `zp.decide(text, [...])` |
| `zp.decide(text)` (all the options of the landscape) | pass the list of options |
| `zp.decide(text, options, min_confidence=0.5)` | `d = zp.decide(text, options)`, then act only if `d.confidence >= 0.5` |
| `ZeroPoint.fit(texts, labels, descriptions=..., catch_all=...)` | `mine, report = ZeroPoint().fit(texts, labels)`; labels can be `None` for *none of these* |
| `zp.save("my_model.safetensors")` | `zp.save("my-model")` |
| `d.energies` | `d.depths` (energy = − depth) |
| `d.truncated` | `d.text_truncated` |
| `d.beyond_edge`, `d.unseen_options` | no equivalent: every option is read from its text, with the text |
| `pip install "zeropoint[train]"` | `pip install "zeropoint[fit]"` |

## [0.2.0] - 2026-11-02

### Added

- `kappa` in `ZeroPoint.load`, and `zp.kappa` after loading: how fast the probability moves to *none* beyond
  the edge of the calibration data. The default stays at 2; 0 turns the edge off. Negative or non-numeric
  values raise `ValueError`.
- `Decision.unseen_options`: the listed options the landscape was not calibrated on, whose wells are built
  from the option's text alone.
- `Decision.truncated`: `True` when a text was longer than the 512 tokens the encoder reads.
- Documentation of κ, of long texts, and of the `support` landscape's weakness with overlapping queues.
- Releases carry signed SLSA level 3 provenance, and the wheel and the sdist are attached to the GitHub
  release. `SECURITY.md` explains how to verify them.

No decision changes: with the default settings, every text gets the same probabilities as in 0.1.0.

## [0.1.0] - 2026-10-31

First release.

### Added

- `ZeroPoint.decide` chooses one option from a list for a text, or returns `None` when none of the options
  fits. The probability of *none* comes from the same calculation as the probability of each option.
- `ZeroPoint.decide_many` for many texts at once, with one shared list of options or one list per text.
- `min_confidence` to return `None` when the chosen option is below a probability.
- Each decision holds the choice, its confidence, the probability of *none*, the probability and energy of
  every option, the distance beyond the edge of the calibration data, and the time it took.
- Two calibrated landscapes inside the package: `everyday` (150 kinds of everyday requests, calibrated on
  CLINC150; the default) and `support` (10 support queues, calibrated on public support tickets).
  `zeropoint.landscapes()` lists them.
- Two frozen encoders, multilingual-e5-small (default) and multilingual-e5-base, pinned to a fixed revision
  and run with ONNX Runtime in 8-bit or full precision. No PyTorch needed for decisions.
- `ZeroPoint.fit` calibrates a landscape on your own labeled texts, with no examples of *none*;
  `save` and `load` store it in a `.safetensors` file. Optional monitoring of training with ndt.
- Benchmark against similarity with the same encoder and against Laya, with the scripts to rebuild it.
- Documentation site at <https://javihaus.github.io/zeropoint/>.

[Unreleased]: https://github.com/Javihaus/zeropoint/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Javihaus/zeropoint/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/Javihaus/zeropoint/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Javihaus/zeropoint/releases/tag/v0.1.0
