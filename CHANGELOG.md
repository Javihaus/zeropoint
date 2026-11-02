# Changelog

Every change that reaches a release is listed here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) as described in
[CONTRIBUTING.md](CONTRIBUTING.md#versions-and-releases).

Changes merged since the last release go under **Unreleased** and move to a numbered section when the next
version is published.

## [Unreleased]

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

[Unreleased]: https://github.com/Javihaus/zeropoint/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Javihaus/zeropoint/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Javihaus/zeropoint/releases/tag/v0.1.0
