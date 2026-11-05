# Comparison with common approaches

ZeroPoint against other ways of picking one option from a list for a text, or saying that none fits:

| System | What it is |
|---|---|
| **ZeroPoint 1.0** | this repository, default settings |
| **Laya** | Laya 0.4.21 (its recommended `Router`), with one more option, "None of these", added to every list |
| **Semantic Router** | Semantic Router 1.1.1, one route per option with the option name as its only example |
| **Zero-shot NLI** | the Hugging Face zero-shot classification pipeline with DeBERTa-v3-base MNLI, `multi_label=True` |
| **Strands Decider 2B** | AWS's open decision model, `strands-decider` 0.2.0, with "None of these" added like Laya; three tests only |
| **Clef-flash** | Cloudflare's open 9B decision model, with the code of its release, the same way; three tests only |

The results, with what every value means, are in [results/tables.md](results/tables.md).

## Where each system comes from

| System | Made by | Version used | Model | Licence |
|---|---|---|---|---|
| Laya | Convai Innovations ([convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)) | `laya` 0.4.21 | model commit `7b928d82`; its router chooses the English or the multilingual checkpoint per text | Apache 2.0 (library and model) |
| Semantic Router | Aurelio AI ([aurelio-labs/semantic-router](https://github.com/aurelio-labs/semantic-router)) | `semantic-router` 1.1.1 | its default encoder, [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), commit `1110a243` | MIT (library), Apache 2.0 (encoder) |
| Strands Decider | AWS ([StrandsAgents/strands-decider-2B-qwen3.5-v1-2610](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610)), base model Qwen3.5-2B-Base | `strands-decider` 0.2.0 | commit `216d1ccb` | Apache 2.0 |
| Clef-flash | Cloudflare ([Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash)), base model Qwen3.5-9B | the code of its release, `joint_schema_model.py` | commit `fb4417d3` | Apache 2.0 |
| Zero-shot NLI | Moritz Laurer, Vrije Universiteit Amsterdam ([MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)); base model DeBERTa-v3-base by Microsoft, fine-tuned on MultiNLI, FEVER-NLI and ANLI | Hugging Face `transformers` zero-shot pipeline | commit `6f5cf0a2` | MIT |

Zero-shot NLI model: Laurer, M., van Atteveldt, W., Salleras Casas, A. and Welbers, K. (2022). *Less Annotating,
More Classifying: Addressing the Data Scarcity Issue of Supervised Machine Learning with Deep Transfer Learning and
BERT-NLI*. Preprint, Open Science Framework. https://osf.io/74b8k

## How it is measured

- Every system runs on the same machine, on CPU with 2 threads, one case at a time, at its default settings.
- Every system reads the same cases: eight tests, from 351 to 2,000 cases each. About half the cases have a right
  option on the list and half do not.
- Zero-shot NLI runs its model once per option, so it reads a fixed sample of 300 cases per test (150 with a right
  option, 150 without). A second table scores every system on that same sample.
- Each system reads the 300-case sample a second time with the options in another order, to measure how often
  the order changes its decision.
- Model versions are fixed by their Hugging Face commit in `run.py`; dataset versions in `build_tests.py`.
- Strands Decider and Clef-flash read only the 300-case sample of the three tests in `LLM_TESTS`, on a GPU
  (`DECIDER_DEVICE=cuda`): their training data includes, or may include, the five public tests.
  [deciders_colab.ipynb](deciders_colab.ipynb) runs them on Google Colab. `cpu_latency.py` times ZeroPoint and
  Strands Decider on the same CPU.

## Run it

```bash
pip install -e . laya==0.4.21 semantic-router==1.1.1 transformers torch datasets pandas scikit-learn
python benchmarks/compare/build_tests.py          # rebuilds tests/*.json; check them against tests/SHA256SUMS
python benchmarks/compare/run.py zeropoint         # one system at a time; about 5 minutes
python benchmarks/compare/run.py semantic_router   # about 5 minutes
python benchmarks/compare/run.py laya_none         # about 1.5 hours on 2 CPU cores
python benchmarks/compare/run.py nli               # about 3 hours
DECIDER_DEVICE=cuda python benchmarks/compare/run.py strands_none bench2000 tickets_made tickets_far      # GPU
DECIDER_DEVICE=cuda python benchmarks/compare/run.py clef_flash_none bench2000 tickets_made tickets_far   # GPU
python benchmarks/compare/report.py                # writes results/summary.json and results/tables.md
```

`out/` holds what every system decided on every case (gzipped), so `report.py` can check the numbers without
running the models again. A test already in `out/` is not run again.

The test texts are not in the repository: some source datasets do not allow redistribution, and the support-ticket
dataset is CC BY-NC (used here for testing only, never for training). `build_tests.py` rebuilds them exactly.
