# Ablation: a fixed zero against a learned rejection class

Before ZeroPoint 1.0 we tested whether fixing the energy of the empty answer at zero helps a decision model say
*none of these*. A prototype was trained against softmax classifiers on the same data, with the same frozen encoder
and the same examples without a right answer. These files hold the results reported in Table 1 of the paper.

The prototype is not the 1.0 architecture. It uses a frozen e5-base encoder and places one well per label on the
normalized embeddings. ZeroPoint 1.0 reads the text and all options together and fine-tunes the encoder.

| Folder | Run | What it tests |
|---|---|---|
| `run1/` | `boltz1-20261029-1924` | Support tickets with their own queue removed. Fixed zero (5 seeds) against a softmax, a softmax with a learned *none of the above* class, Laya's energy, similarity and two LLMs. Hypotheses H1 to H3 fixed before training, Holm-corrected. Queues held out from training as missing (`generalisation_heldout_queues.csv`). |
| `run2/` | `boltz1c-20261029-2307` | Choice among options on eight datasets against a softmax trained on the same data (`choice_results.csv`, 2 seeds). Detection of *none* on CLINC150 out-of-scope questions and on tickets (`none_results.csv`, `none_primary_tests.csv`). |
| `run3/` | `boltz1d-20261030-0840` | The same model on tickets with their queue removed and on unrelated texts (film reviews, newsgroup posts), plus a variant with a distance term (`boltz_far`). |

Column names:

- `auroc_within_strata`: AUROC computed within each removed queue and averaged, so that a model cannot score by
  recognizing which list is shown.
- `lo`, `hi`: 95% bootstrap interval. `p_holm`: p-value after Holm correction over the tests of the same file.
- `fpr95`: share of cases without a right option still acted on when 95% of the cases with one are kept.

`tickets_real` in `run2/none_results.csv` is a test of 100 tickets written with a language model. It is not valid:
text length alone separates it (AUROC 0.84), so it is kept here only for completeness and is not used in the paper.
