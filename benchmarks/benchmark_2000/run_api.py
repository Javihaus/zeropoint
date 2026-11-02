"""Run a hosted decision model (Liquid d1, Microsoft Decision-1) on the 2,000 cases, in the same format as
run_systems.py, so metrics.py can score it next to the others.

    export DECISIONS_API_KEY=...            # the key of the provider you use
    python run_api.py d1                    # Liquid d1 through AI/ML API (api.aimlapi.com)
    python run_api.py decision1             # Microsoft Decision-1 through OpenRouter

Both providers take the System One request shape: a `state`, and `questions` of type "choice" with
`criteria` (one key per option). If a provider changes its endpoint or model id, edit PROVIDERS below.
Writes out/<name>.json. Cases already done are kept, so an interrupted run can be resumed.
"""

import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "data.json")))

PROVIDERS = {
    "d1": {"url": "https://api.aimlapi.com/v1/decisions", "model": "liquid/d1"},
    "decision1": {"url": "https://openrouter.ai/api/v1/decisions", "model": "microsoft/microsoft-decision-1"},
}
ASK = {"everyday": ("utterance", "Which intent does `utterance` express?"),
       "support": ("ticket", "Which support queue should handle this ticket?")}


def call(provider, case, key):
    field, instructions = ASK[case["landscape"]]
    body = {"model": provider["model"], "state": {field: case["text"]},
            "questions": {"q": {"type": "choice", "instructions": instructions,
                                "criteria": {o: None for o in case["options"]}}}}
    req = urllib.request.Request(provider["url"], data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        ans = json.load(r)["answers"]["q"]
    ms = (time.perf_counter() - t0) * 1000
    p = {o: float(ans["probabilities"][o]) for o in case["options"]}
    return {"id": case["id"], "scores": p, "choice": max(p, key=p.get), "conf": max(p.values()), "ms": ms,
            "answer_confidence": float(ans.get("confidence", float("nan")))}


if __name__ == "__main__":
    name = sys.argv[1]
    key = os.environ["DECISIONS_API_KEY"]
    path = os.path.join(HERE, "out", f"{name}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = {r["id"]: r for r in json.load(open(path))} if os.path.exists(path) else {}
    for k, case in enumerate(DATA):
        if case["id"] in done:
            continue
        for attempt in range(5):
            try:
                done[case["id"]] = call(PROVIDERS[name], case, key)
                break
            except Exception as e:  # rate limit or network: wait and retry
                print(f"case {case['id']}: {e}; retrying", flush=True)
                time.sleep(2 ** attempt)
        if k % 50 == 0:
            json.dump([done[c["id"]] for c in DATA if c["id"] in done], open(path, "w"))
            print(f"{name}: {len(done)}/{len(DATA)}", flush=True)
    json.dump([done[c["id"]] for c in DATA if c["id"] in done], open(path, "w"))
    print(f"{name}: {len(done)} cases saved to {path}")
