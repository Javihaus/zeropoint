"""Run a hosted LLM (Anthropic or OpenAI) as a decision engine on the comparison tests, and save its decisions in
the same format as benchmarks/compare/run.py, so benchmarks/compare/report.py scores it next to the other systems.

    pip install anthropic openai
    export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY
    python benchmarks/llm/run_llm.py anthropic claude-opus-5-5
    python benchmarks/llm/run_llm.py openai <model id> --tests snips atis

How the model is asked (the same for every model, written once below):
  * The model gets the text and the list of options, plus "none of these", and must call one tool, `decide`, whose
    `choice` can only be one of those strings (an enum) and whose `confidence` is a number from 0 to 1. Forcing a
    tool call with an enum is the structured, System 1 way of asking: no free text, no answer outside the list.
  * No examples and no reasoning step: one call per case, temperature 0 where the model accepts it.
  * The confidence is the model's own stated probability that its choice is right. LLM APIs give no probabilities
    over options, so "silent mistakes" (confident wrong answers when no option fits) use this stated confidence.

What is measured per case: the decision, the stated confidence, the time from sending the request to receiving the
answer (it includes the network, from wherever this runs), and the cost from the tokens the API reports and the
prices in PRICES (USD per million tokens; check them against the provider's price page before running).

By default the model reads the three tests in LLM_TESTS (benchmarks/compare/run.py), the ones it is unlikely to
have seen in training, and on each the same fixed 300-case sample as zero-shot NLI (run.py sample()): 900 calls.
This is exactly what the published results were made with. --shuffle adds a second pass on the same cases with the
options in another order; --tests and --all-cases change the cases.
Results are written as they come, so an interrupted run continues where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
COMPARE = os.path.join(HERE, "..", "compare")
sys.path.insert(0, COMPARE)
from run import LLM_TESTS, asks, load, sample, shuffled  # noqa: E402  (same tests, sample and shuffles as the others)

NONE = "none of these"

# USD per million tokens (input, output). Fill in or correct from the provider's price page before running;
# a model missing here gets no cost.
PRICES = {
    "claude-opus-5-5": (None, None),
    "claude-sonnet-5-5": (None, None),
    "claude-haiku-5-5": (None, None),
}

SYSTEM = ("You route texts. You are given a text and a list of options. Choose the one option that fits the text, "
          f'or "{NONE}" if no option fits. Then give your confidence, from 0 to 1, that your choice is right. '
          "Answer only by calling the decide tool.")


def prompt(text: str, options: list[str], test: str) -> str:
    key, question = asks(test)
    lines = "\n".join(f"- {o}" for o in options)
    return f"{question}\n\n{key.capitalize()}:\n{text}\n\nOptions:\n{lines}\n- {NONE}"


def schema(options: list[str]) -> dict:
    return {"type": "object", "additionalProperties": False, "required": ["choice", "confidence"],
            "properties": {"choice": {"type": "string", "enum": list(options) + [NONE]},
                           "confidence": {"type": "number", "minimum": 0, "maximum": 1}}}


class Anthropic:
    """Forced tool call when the model accepts it. Some models reject a forced tool_choice (and some reject a
    temperature): the first such refusal switches that model, once and for all, to the same tool offered with
    tool_choice "auto" and the instruction to call it. The mode used is saved in environment.json."""

    def __init__(self, model: str):
        import anthropic
        self.client, self.model = anthropic.Anthropic(), model
        self.forced, self.temperature = True, True

    @property
    def mode(self):
        return {"tool_choice": "forced" if self.forced else "auto", "temperature_0": self.temperature}

    def __call__(self, text, options, test):
        tool = {"name": "decide", "description": "Give the decision.", "input_schema": schema(options)}
        while True:
            kw = dict(model=self.model, system=SYSTEM, tools=[tool],
                      messages=[{"role": "user", "content": prompt(text, options, test)}])
            if self.forced:
                kw.update(max_tokens=200, tool_choice={"type": "tool", "name": "decide"})
            else:                                # room for the model's own reasoning before the call
                kw.update(max_tokens=4000, tool_choice={"type": "auto"})
            if self.temperature:
                kw["temperature"] = 0
            try:
                r = self.client.messages.create(**kw)
                break
            except Exception as e:
                msg = str(e)
                if self.forced and "tool_choice" in msg:
                    self.forced = False
                elif self.temperature and "temperature" in msg:
                    self.temperature = False
                else:
                    raise
                print(f"  {self.model}: switching to {self.mode}", flush=True)
        call = next((b for b in r.content if b.type == "tool_use"), None)
        if call is not None:
            answer = call.input
        else:                                    # no tool call: take a JSON object from the text, if any
            text_out = "".join(getattr(b, "text", "") for b in r.content)
            m = re.search(r"\{[^{}]*\}", text_out)
            try:
                answer = json.loads(m.group(0)) if m else {}
            except ValueError:
                answer = {}
            answer["no_tool_call"] = True
        return answer, r.usage.input_tokens, r.usage.output_tokens


class OpenAI:
    def __init__(self, model: str):
        import openai
        self.client, self.model = openai.OpenAI(), model
        self.temperature = True

    @property
    def mode(self):
        return {"response_format": "strict JSON schema", "temperature_0": self.temperature}

    def __call__(self, text, options, test):
        kw = dict(model=self.model,
                  messages=[{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt(text, options, test)}],
                  response_format={"type": "json_schema",
                                   "json_schema": {"name": "decide", "strict": True, "schema": schema(options)}})
        if self.temperature:
            try:
                r = self.client.chat.completions.create(temperature=0, **kw)
            except Exception as e:               # some models do not accept a temperature: never send it again
                if "temperature" not in str(e):
                    raise
                self.temperature = False
        if not self.temperature:
            r = self.client.chat.completions.create(**kw)
        return json.loads(r.choices[0].message.content), r.usage.prompt_tokens, r.usage.completion_tokens


def row(answer: dict, options: list[str]) -> dict:
    """The answer in the format report.py reads: the chosen option gets the stated confidence."""
    choice = answer.get("choice")
    try:
        conf = min(1.0, max(0.0, float(answer.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0
    if choice not in options:                  # "none of these"; anything else is counted as no option
        return {"p": {o: 0.0 for o in options}, "p_none": conf if choice == NONE else 1.0, "choice": None,
                "invalid": choice != NONE}
    return {"p": {o: (conf if o == choice else 0.0) for o in options}, "p_none": 1.0 - conf, "choice": choice}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("provider", choices=["anthropic", "openai"])
    ap.add_argument("model")
    ap.add_argument("--tests", nargs="*", default=LLM_TESTS)
    ap.add_argument("--all-cases", action="store_true", help="every case instead of the 300-case sample")
    ap.add_argument("--shuffle", action="store_true", help="add a second pass with the options in another order")
    args = ap.parse_args()

    decide = (Anthropic if args.provider == "anthropic" else OpenAI)(args.model)
    name = "llm_" + args.model.replace("/", "_").replace(".", "_")
    out = os.path.join(COMPARE, "out", name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "system.json"), "w") as f:
        json.dump({"name": args.model, "probabilities": True, "reads_options_together": "yes",
                   "paired": True}, f, indent=1)
    def write_environment():
        with open(os.path.join(out, "environment.json"), "w") as f:
            json.dump({"provider": args.provider, "model": args.model, "python": platform.python_version(),
                       "prompt": SYSTEM, "prices_usd_per_mtok": PRICES.get(args.model),
                       "mode": getattr(decide, "mode", None)}, f, indent=1)
    write_environment()
    price_in, price_out = PRICES.get(args.model, (None, None))

    for test in args.tests:
        items = load(test)
        idx = list(range(len(items))) if args.all_cases else sample(items)
        passes = [("", False)] + ([(".shuffled", True)] if args.shuffle else [])
        for suffix, order in passes:
            path = os.path.join(out, f"{test}{suffix}.jsonl")
            done = set()
            if os.path.exists(path):
                with open(path) as f:
                    done = {json.loads(line)["i"] for line in f if line.strip()}
            with open(path, "a") as f:
                for n, i in enumerate(idx):
                    if i in done:
                        continue
                    it = items[i]
                    opts = shuffled(it["options"], i) if order else it["options"]
                    for attempt in range(5):
                        try:
                            t0 = time.perf_counter()
                            answer, tin, tout = decide(it["text"], opts, test)
                            ms = (time.perf_counter() - t0) * 1000
                            break
                        except Exception as e:  # rate limits and transient errors: wait and retry
                            status = getattr(e, "status_code", None)
                            if status in (400, 401, 403, 404):   # bad key, no access, unknown model: retrying cannot help
                                sys.exit(f"{args.provider} {args.model}: {e}")
                            if attempt == 4:
                                raise
                            print(f"  retry after error: {e}", flush=True)
                            time.sleep(5 * 2 ** attempt)
                    r = row(answer, list(opts))
                    r.update({"i": i, "ms": ms, "tokens_in": tin, "tokens_out": tout, "stated": answer})
                    if price_in is not None and price_out is not None:
                        r["cost_usd"] = (tin * price_in + tout * price_out) / 1e6
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    if n % 50 == 0:
                        print(f"{args.model} {test}{suffix} {n}/{len(idx)}", flush=True)
            with open(path) as f:                     # the .json that report.py reads: one row per case, the first
                rows = list({r["i"]: r for r in reversed([json.loads(line) for line in f if line.strip()])}.values())
                rows = [r for r in rows if r["i"] in set(idx)]
            with open(os.path.join(out, f"{test}{suffix}.json"), "w", encoding="utf-8") as f:
                json.dump(sorted(rows, key=lambda x: x["i"]), f, ensure_ascii=False)
            print(f"{args.model} {test}{suffix}: {len(rows)} cases", flush=True)
            write_environment()


if __name__ == "__main__":
    main()
