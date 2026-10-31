# Contributing to ZeroPoint

Thank you for your interest in ZeroPoint. Bug reports, questions, benchmark results on new data and code
changes are all welcome.

## Reporting a bug or asking a question

Open an issue on GitHub. For a bug, please include:

- what you ran (a few lines of code are enough),
- what you expected and what you got,
- your Python version, operating system, and the output of `pip show zeropoint onnxruntime`.

If the bug depends on your data, a few example texts that reproduce it help more than a description.

## Reporting a security problem

Please do not open a public issue. Write to javier@jmarin.info with the details. You will get a reply within
a few days, and the fix will be published before the problem is disclosed.

## Setting up for development

```bash
git clone https://github.com/Javihaus/zeropoint.git
cd zeropoint
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" scikit-learn
pytest -q
```

The first test run downloads the small encoder (about 120 MB) from the Hugging Face Hub.

## Proposing a change

1. Open an issue first if the change is large or changes how ZeroPoint behaves, so we can agree on the
   approach before you spend time on it.
2. Create a branch from `main`. Changes never go directly to `main`.
3. Make the change and add a test that fails without it.
4. Run `pytest -q` and check that every test passes.
5. Open a pull request that explains what changes and why.

Please keep each pull request to one topic. Small pull requests are reviewed faster.

## Changes that affect the numbers

If a change can alter decisions (the engine, the encoder, the calibration recipe or the shipped models),
run the benchmark before and after and include both results in the pull request:

```bash
pip install datasets scikit-learn
python benchmarks/quality.py
python benchmarks/latency.py
```

The shipped models in `src/zeropoint/presets/` are rebuilt only with `scripts/build_presets.py`, never by
hand, so that anyone can reproduce them. If they change, `BENCHMARK.md` is updated in the same pull request.

## Benchmark results on new data

Results on datasets that ZeroPoint has not been tested on are very welcome, including results where it does
badly. Open an issue with the dataset, the code you ran and the numbers.

## Writing style

Documentation is written for someone who arrives knowing nothing about the project: plain words first,
technical terms after. Numbers in the documentation must come from a script in this repository.

## Versions and releases

ZeroPoint follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). A version number is
`MAJOR.MINOR.PATCH`, and it lives in one place only: `src/zeropoint/_version.py`.

| Change | Version step | Examples |
|---|---|---|
| Breaks code that works today | MAJOR (`1.4.2` → `2.0.0`) | a renamed or removed argument, a model file older versions cannot read |
| Adds something, or changes decisions | MINOR (`1.4.2` → `1.5.0`) | a new landscape, a new option, rebuilt shipped models, a new calibration recipe |
| Fixes a bug without changing decisions | PATCH (`1.4.2` → `1.4.3`) | a crash, a wrong error message, documentation |

While the version starts with `0.`, a change that breaks code steps MINOR instead of MAJOR.

Rebuilding a shipped landscape is never a patch: the same text can get a different decision, and users
need to see that in the version number. `FORMAT` in `_version.py` is a separate number for the model file
format; it goes up only when older versions of ZeroPoint can no longer read a saved model, and that is
always a MAJOR step.

Every pull request that users would notice adds a line under **Unreleased** in
[CHANGELOG.md](CHANGELOG.md).

A release is made by the maintainer:

1. Move the **Unreleased** lines to a new section `## [X.Y.Z] - YYYY-MM-DD`, update the links at the end
   of the changelog, and set `__version__ = "X.Y.Z"`, all in one pull request. The tests check that the
   version and the changelog agree.
2. After merging, create the GitHub release with tag `vX.Y.Z` and the changelog section as notes:
   `gh release create vX.Y.Z --title "ZeroPoint X.Y.Z" --notes "$(python scripts/release_notes.py)"`.
3. The `publish` workflow checks that the tag, the version and the changelog agree, builds the package and
   uploads it to PyPI. The tag and the PyPI version are always the same number.

## License

By contributing, you agree that your contribution is released under the Apache License 2.0, the license of
this project.

## Contact

Javier Marín, javier@jmarin.info
