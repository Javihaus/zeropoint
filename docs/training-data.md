# What ZeroPoint was trained on

ZeroPoint 1.0 was trained only on data whose licence allows commercial use. No test text used in
[benchmarks/compare](https://github.com/Javihaus/zeropoint/tree/main/benchmarks/compare) is in it: any test text
that also appears in the training data was removed from the tests (the counts are in
`benchmarks/compare/tests/overlap_report.json`).

## Public datasets

| Dataset | Licence | Language | Texts used |
|---|---|---|---|
| [CLINC150](https://huggingface.co/datasets/clinc/clinc_oos) (Larson et al., 2019) | CC BY 3.0 | English | 18,000 (train and validation, without the out-of-scope texts) |
| [Banking77](https://huggingface.co/datasets/PolyAI/banking77) (Casanueva et al., 2020) | CC BY 4.0 | English | 10,003 |
| [MASSIVE](https://huggingface.co/datasets/mteb/amazon_massive_intent) (FitzGerald et al., 2023) | CC BY 4.0 | English, Spanish | 13,547 per language |
| [Bitext customer support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | CDLA-Sharing 1.0 | English | 6,588 |
| [Bitext retail banking](https://huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset) | CDLA-Sharing 1.0 | English | 6,578 |
| [Bitext insurance](https://huggingface.co/datasets/bitext/Bitext-insurance-llm-chatbot-training-dataset) | CDLA-Sharing 1.0 | English | 6,591 |
| [Bitext telco](https://huggingface.co/datasets/bitext/Bitext-telco-llm-chatbot-training-dataset) | CDLA-Sharing 1.0 | English | 6,578 |

Each Bitext dataset was capped at about 6,600 texts, the same number per label, so that they do not outweigh the
texts written by people. About 13% of all texts were kept apart for choosing when to stop training and for
calibrating the two numbers that set how often ZeroPoint answers *none*.

The labels of these datasets are fine-grained intents ("activate my card"). ZeroPoint also learned broad labels
("card services") made by grouping the intents of each dataset by hand, so that it can choose among departments
and categories as well as among intents.

During training, every labelled text becomes a list of 4 to 20 options from the same dataset. In about half of the
lists the wrong options are the ones most easily confused with the right one. In 8% of the lists from datasets with more than
21 labels there are 21 to 150 options. In part of the lists the right option is removed, and the right answer is *none*.

## 5,000 examples written for this project

The public datasets have three gaps that showed up case by case in early tests: they contain almost no long texts
like support emails, almost no negations ("book the hotel but not the car"), and almost no broad categories that
need knowing what a name refers to ("play Hotel California" is music, not a hotel). The 5,000 examples cover
exactly these three gaps, in English (70%) and Spanish (30%). They were written with the help of a language model
(Claude, by Anthropic) following the rules below, and contain no text from third parties.

| Family | English | Spanish | Words (min / median / max) |
|---|---|---|---|
| Support tickets with a subject and a body, to departments | 1,500 | 700 | 25 / 44 / 134 |
| Requests with a negation or a contrast | 1,100 | 500 | 7 / 17 / 34 |
| Broad categories that depend on a name (a film, a band, a place) | 900 | 300 | 4 / 9 / 23 |

**Labels.** 30 sets of 6 to 12 labels, made up for this project: departments of a software company, a bank, a
shop, a city council, a clinic and others; intents for travel, banking, home, shopping, work and food; and broad
categories. No label is the same as a label of any test, and a program checks it. Departments have different names
in each set ("People Team", "Payroll", "Personas y Cultura"), so that the model learns what a department does and
not a particular name.

**Rules for each family.**

- Tickets: the department follows from the situation, not from its name; at most 10% of texts may contain their
  label word for word. Polite, curt, angry, with small typos, with made-up order numbers and amounts. At least 25%
  could seem to belong to a neighbouring department, and a detail decides. No real personal data.
- Negations: at least two intents of the set appear, one asked for and one negated or ruled out, in varied forms
  ("no", "instead of", "never mind the…", negation first). Up to 15% negate everything and have no right label.
- Broad categories: the text turns on a real, well-known name, and at least 25% are traps (a song called like a
  hotel, a series called like a room). Names are used only as references; no text of any work is copied.

**How they were written.** In 50 batches of 100. Each example got a score from 1 to 5 in six respects: realism,
whether the label is the only right one, difficulty, variety, language, and form. In odd batches, the worst example
and the 10 that fail in the same way were rewritten after writing down why the worst one was the worst. In even
batches, the 10 examples furthest from the best one were rewritten taking it as the reference. 525 examples were
rewritten this way.

**Checks against the tests.** Whoever wrote the examples never saw the test sets. Every example was compared with
every test sentence using the multilingual-e5 encoder; the 24 examples with a similarity of 0.92 or more (requests
of the same kind with similar words, none a copy) were rewritten with another situation and other words. After
that, the highest similarity is 0.919 and the median 0.871; two unrelated sentences on the same topic score about
0.85 with this encoder. Repeated texts were removed after normalising case, accents and punctuation.

## References

- Larson, S., et al. (2019). An evaluation dataset for intent classification and out-of-scope prediction. *EMNLP-IJCNLP 2019*.
- Casanueva, I., Temčinas, T., Gerz, D., Henderson, M., & Vulić, I. (2020). Efficient intent detection with dual sentence encoders. *Proceedings of the 2nd Workshop on NLP for Conversational AI*.
- FitzGerald, J., et al. (2023). MASSIVE: A 1M-example multilingual natural language understanding dataset with 51 typologically-diverse languages. *ACL 2023*.
