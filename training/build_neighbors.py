"""For every label pool used in training, the 10 labels of the same pool closest to each label, by cosine of
multilingual-e5-small embeddings ("query: <label>"). Training draws half of the wrong options from these
neighbours, so the model learns to separate the options that are easiest to confuse.
    python build_neighbors.py  ->  data/neighbors.json  {pool: {label: [nearest labels]}}"""
import json, torch, numpy as np
from transformers import AutoModel, AutoTokenizer
M = "intfloat/multilingual-e5-small"
tok = AutoTokenizer.from_pretrained(M); enc = AutoModel.from_pretrained(M).eval()
pools = {}
for l in open("data/train.jsonl", encoding="utf-8"):
    r = json.loads(l); pools.setdefault(r["source"], set()).add(r["label"])
for src, m in json.load(open("data/coarse_map.json", encoding="utf-8")).items():
    pools[src + "#broad"] = set(m.values())
for k, v in json.load(open("data/synth_pools.json", encoding="utf-8")).items():
    if not k.startswith("_"):
        pools[k] = set(v["labels"])
@torch.no_grad()
def emb(xs):
    t = tok(["query: " + x for x in xs], padding=True, return_tensors="pt")
    h = enc(**t).last_hidden_state; m = t["attention_mask"].unsqueeze(-1).float()
    return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).numpy()
out = {}
for p in sorted(pools):
    labs = sorted(pools[p]); E = emb(labs); S = E @ E.T
    np.fill_diagonal(S, -9)
    out[p] = {l: [labs[j] for j in np.argsort(-S[i], kind="stable")[:10]] for i, l in enumerate(labs)}
json.dump(out, open("data/neighbors.json", "w", encoding="utf-8"), ensure_ascii=False)
print(len(out), "pools;", "example:", list(out["clinc150"].items())[:2])
