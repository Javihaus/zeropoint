# Security Policy

## Supported versions

Security fixes go into the latest release. Earlier versions are not patched; the fix is to upgrade.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a vulnerability

Please do not report security problems in public issues, pull requests or discussions.

Report them privately, in either of these ways:

- through GitHub: the **Security** tab of this repository, then **Report a vulnerability**;
- by email to javier@jmarin.info, with "ZeroPoint security" in the subject.

Please include what you found, the version of ZeroPoint, and the steps or code that reproduce it.

## What happens next

- **Within 3 working days:** you get a reply confirming that the report was received.
- **Within 10 working days:** you get an assessment: whether the problem is confirmed, how serious it is,
  and the plan to fix it.
- **If it is confirmed:** a fixed version is published on PyPI, and a GitHub security advisory describes the
  problem and the fix. You are credited in the advisory, unless you prefer not to be.
- **If it is declined:** you get an explanation of why it is not considered a vulnerability in ZeroPoint.

Please give us the chance to publish a fix before you disclose the problem. We aim to do so within 90 days
of the report.

## What is in scope

- The `zeropoint` package and the code in this repository.
- The calibrated models shipped with the package (`src/zeropoint/presets/`).

Out of scope: vulnerabilities in dependencies (ONNX Runtime, tokenizers, NumPy, PyTorch) and in the E5
encoders on the Hugging Face Hub. Please report those to their own projects. If one of them affects
ZeroPoint in a specific way, we still want to hear about it.

## Verifying a release

Every release is built once by the `publish` workflow in this repository, from the tagged commit, with build
tools pinned by hash. The same files go to the GitHub release and to PyPI, with two kinds of signed proof of
where they were built:

- **SLSA level 3 provenance**, attached to each GitHub release as `multiple.intoto.jsonl`. To check a wheel
  downloaded from PyPI or from the release, with [slsa-verifier](https://github.com/slsa-framework/slsa-verifier):

  ```bash
  slsa-verifier verify-artifact zeropoint-X.Y.Z-py3-none-any.whl \
      --provenance-path multiple.intoto.jsonl \
      --source-uri github.com/Javihaus/zeropoint --source-tag vX.Y.Z
  ```

- **PyPI attestations**: PyPI records, for each file, that it was uploaded by this repository's `publish`
  workflow through trusted publishing. They are shown on the project's PyPI page, next to each file.

## How ZeroPoint limits risk

- Models are stored as `safetensors` files, which hold numbers only and cannot run code when loaded.
- Encoders are downloaded from the Hugging Face Hub at a pinned revision, so a change upstream cannot
  silently change what ZeroPoint runs.
- Making decisions does not need PyTorch or network access once the encoder has been downloaded.
