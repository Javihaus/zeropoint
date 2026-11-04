# Running in production

## Speed

On a 2-core cloud CPU (Intel Xeon at 2.1 GHz, 2 threads), the median time per decision is 16–20 ms with 4 to 10
short options, 27 ms with 20 options, and 43–53 ms for support tickets with a subject and a body. Almost all of it is
the encoder. No GPU is needed.

The time grows with the length of the text and with the number of options. Lists longer than 20 options are read
in blocks of 20, one encoder pass per block.

## Threads

`ZeroPoint(threads=2)` sets how many CPU threads the encoder uses for one decision. More threads make one decision
faster up to a point. To decide many texts in parallel, run several processes with one or two threads each.

The number of threads never changes the result.

## The same numbers on every machine

The options are put in one fixed order before reading, each case is computed on its own, and the last steps run in
64-bit precision. On every change, CI runs 500 reference cases on Linux (x86 and ARM), macOS and Windows: the
decisions are the same on all of them, and the probabilities agree with the reference to within 3 millionths.

## Long texts

ZeroPoint reads the first 192 word pieces of a text and the first 16 of each option. `text_truncated` and
`options_truncated` tell you when something was cut. For long documents, pass the part that decides the case
(the subject and the first paragraph of an email, for example).

## Offline use

The model is inside the package, and ZeroPoint never connects to the network. It works on machines without
internet access once the package is installed.

## Integrity of the model

Every model file is checked against the sha256 in `manifest.json` when the model is loaded, and the manifest of the
shipped model is pinned in the code. A changed or missing file raises `ModelFilesError`. Releases carry signed SLSA
level 3 provenance; `SECURITY.md` explains how to verify them.

## Saved models

`save` writes a folder with the model files and a manifest with their hashes, and the report of `fit` or
`recalibrate`. Models from 0.x (`.safetensors` landscapes) cannot be loaded by 1.0.
