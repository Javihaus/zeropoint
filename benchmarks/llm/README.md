# ZeroPoint against hosted LLMs

Runs Anthropic and OpenAI models as decision engines on three of the tests of [benchmarks/compare](../compare), and
scores them with the same `report.py`, next to ZeroPoint and Laya with "None of these". The results are in
[BENCHMARK.md](../../BENCHMARK.md#compared-with-hosted-llms).

The three tests are the 2,000-case test and the two support-ticket tests (`LLM_TESTS` in `run.py`). The other five
have been public since 1990–2020 and are very likely in the LLMs' training data.

The easiest way to run it is the notebook, [llm_compare.ipynb](llm_compare.ipynb) (it works on Colab). It rebuilds the
tests, reads your API keys from Colab secrets, runs the models in parallel, scores everything and draws accuracy
against latency (and against cost, if you enter prices per token).

From a terminal:

```bash
pip install -e . anthropic openai datasets pandas scikit-learn
python benchmarks/compare/build_tests.py
export ANTHROPIC_API_KEY=...
python benchmarks/llm/run_llm.py anthropic claude-sonnet-5-5
python benchmarks/compare/report.py
```

## How the models are asked

The same for every model, fixed in `run_llm.py`:

- The text and the options, plus *none of these*. The model must answer with one structured call: a tool call
  (Anthropic) or a strict JSON schema (OpenAI), with the choice as an enum of the options and a stated confidence
  from 0 to 1. An answer outside the list is counted as *none* and marked `invalid`.
- The tool call is forced when the model accepts it. A model that rejects a forced tool call is offered the same
  tool with `tool_choice` auto; a model that rejects temperature 0 runs at its default temperature. The mode each
  model ran in is saved in its `environment.json`.
- No examples and no reasoning step. One call per case.
- LLM APIs give no probabilities over the options, so the confidence used for *silent mistakes* is the one the
  model states.

## What is measured

- The decision and the stated confidence, scored like the other systems.
- Latency: from sending the request to receiving the answer, network included, from wherever the run happens.
- Tokens per call, and the cost per decision if you enter prices in `PRICES`.
- By default, the same fixed 300-case sample per test as zero-shot NLI: 900 calls per model, one pass. This is
  exactly what the published results were made with. `--shuffle` adds a second pass with the options in another
  order; `--tests` and `--all-cases` change the cases.

Runs are written as they go; an interrupted run continues where it stopped.
