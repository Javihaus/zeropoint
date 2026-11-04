# ZeroPoint against hosted LLMs

Runs Anthropic and OpenAI models as decision engines on the eight tests of [benchmarks/compare](../compare), and
scores them with the same `report.py`, next to ZeroPoint, Laya, Semantic Router and zero-shot NLI.

The easiest way to run it is the notebook, [llm_compare.ipynb](llm_compare.ipynb) (it works on Colab). It rebuilds the
tests, asks for your API keys and the prices per token, runs the models, scores everything and draws accuracy
against cost and against latency.

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

- The text and the options, plus *none of these*. The model must answer with one structured call: a forced tool
  call (Anthropic) or a strict JSON schema (OpenAI), with the choice as an enum of the options and a stated
  confidence from 0 to 1. It cannot answer outside the list.
- No examples and no reasoning step. One call per case, temperature 0 where the model accepts it.
- LLM APIs give no probabilities over the options, so the confidence used for *silent mistakes* is the one the
  model states.

## What is measured

- The decision and the stated confidence, scored like the other systems.
- Latency: from sending the request to receiving the answer, network included, from wherever the run happens.
- Cost per decision: tokens reported by the API times the prices you enter.
- By default, the same fixed 300-case sample per test as zero-shot NLI, then the same cases with the options in
  another order: about 4,800 calls per model.

Runs are written as they go; an interrupted run continues where it stopped.
