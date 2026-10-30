# Pinned dependencies

The `.txt` files pin every package that CI, fuzzing and releases install, with the hash of every file, so a
changed or tampered package is refused. The `.in` files say what is wanted; the `.txt` files are generated
from them. To regenerate after changing an `.in` file (delete the old `.txt` first, or uv keeps its pins):

```bash
pip install uv
cd requirements
rm -f *.txt
uv pip compile ci.in       --universal --python-version 3.10 --generate-hashes -o ci.txt
uv pip compile ci-torch.in --universal --python-version 3.10 --generate-hashes -o ci-torch.txt
uv pip compile build.in    --universal --python-version 3.10 --generate-hashes -o build.txt
uv pip compile fuzz.in     --python-version 3.11 --python-platform x86_64-manylinux_2_28 --generate-hashes -o fuzz.txt
uv pip compile torch-cpu.in --universal --python-version 3.10 --generate-hashes --no-deps \
    --index-url https://download.pytorch.org/whl/cpu --emit-index-url -o torch-cpu.txt
```

PyTorch comes from its own CPU index and is installed with `--no-deps`; its dependencies come from PyPI
through `ci-torch.txt`. Mixing the two indexes in one file fails, because the PyTorch index serves its own
copies of some common packages with different hashes.
