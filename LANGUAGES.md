# Languages

ZeroPoint reads 25 languages: the 22 official languages of the European Union written in the Latin script, plus
Catalan, Basque and Galician.

| | | | | |
|---|---|---|---|---|
| Basque (`eu`) | Catalan (`ca`) | Croatian (`hr`) | Czech (`cs`) | Danish (`da`) |
| Dutch (`nl`) | English (`en`) | Estonian (`et`) | Finnish (`fi`) | French (`fr`) |
| Galician (`gl`) | German (`de`) | Hungarian (`hu`) | Irish (`ga`) | Italian (`it`) |
| Latvian (`lv`) | Lithuanian (`lt`) | Maltese (`mt`) | Polish (`pl`) | Portuguese (`pt`) |
| Romanian (`ro`) | Slovak (`sk`) | Slovenian (`sl`) | Spanish (`es`) | Swedish (`sv`) |

`zp.languages` gives the same list as codes.

## Why these 25

The model is built on [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small), which reads
about 100 languages with a vocabulary of 250,002 word pieces. Most of the model's size is that vocabulary. Keeping
only the pieces these 25 languages need makes the model small enough to ship inside the package (89 MB) instead of
downloading it at run time.

The pieces were chosen by counting, in up to 10,000 Wikipedia articles of each language, which pieces cover 99.9% of the
text, and adding every piece that appears in the training data. 107,299 pieces are kept. Wikipedia was used only
to count pieces, never to train.

Greek and Bulgarian are official EU languages but are written in other alphabets, which would have added many pieces
for two languages. They are left out.

## How well each language works

The training texts are in **English and Spanish**, and the tests in [BENCHMARK.md](BENCHMARK.md) are almost all in
those two languages. The other 23 are read through the multilingual encoder, which places texts with the same
meaning close together across languages; they were not tested one by one. If your
texts are in one of them, check ZeroPoint on a few hundred of your own labelled texts, and
[calibrate it](README.md#calibration) if needed.

Options and texts can be in different languages: the MTOP test in BENCHMARK.md gives Spanish texts with English
options.

## Texts in other languages

**Other alphabets.** A text or an option in which more than 20% of the letters are outside the Latin script raises
`UnsupportedLanguage`, instead of returning a guess. The limit was set on the SIB-200 dataset: texts in the 25
supported languages have at most 1.7% of such letters (a name, a symbol), and texts in Greek, Cyrillic, Arabic,
Hebrew, Chinese, Japanese, Korean, Devanagari, Thai, Georgian and Armenian start at 25.9%.

**Other languages in the Latin script**, such as Turkish, Vietnamese or Indonesian, are not detected. ZeroPoint
answers, but some of their words fall outside the reduced vocabulary and it was not trained or tested on them.
Expect clearly worse results.

## Long texts

ZeroPoint reads the first 192 word pieces of a text. Languages with long words (Finnish, Hungarian, Basque) use
more pieces per word than English, so less of a long text is read. `Decision.text_truncated` tells you when the end
of a text was ignored.
