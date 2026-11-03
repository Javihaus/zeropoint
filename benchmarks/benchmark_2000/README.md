# The 2,000-case test

One of the eight tests in the comparison ([benchmarks/compare](../compare)). Every case is a text and a short
list of options (4 to 9). In 1,000 cases one option is right; in the other 1,000 none is.

## The data

`data.json`: 2,000 cases from 552 distinct texts. The options are everyday intents (1,415 cases) and
support-ticket queues (585). Each case has a scenario:

| With a right option (1,000) | Without one (1,000) |
|---|---|
| In-Scope (700) | Out-of-Scope (500): the right option is not on the list |
| Boundary-Answerable (250): harder wording | Boundary-NearMiss (200): close to an option, but none fits |
| Adversarial-Edge (100): typos, fragments | OOD-Noise (250): unrelated text, other languages |

23 cases (9 texts) appear word for word in the CLINC150 dataset; their ids are in `clinc_overlap_ids.json`.
The texts of 4 cases are also in ZeroPoint's training data; the comparison counts them like any other case.
