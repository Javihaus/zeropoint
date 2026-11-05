"""Stage 2: cache frozen encoder features, listwise, exactly as ZeroPoint reads a list.

    python -I s2_cache.py E1|E2 SPLIT [SPLIT ...]

For each list: the pooled last hidden state of every option (mean over the option's own pieces, the closing [SEP]
excluded, as in _runtime.py), in canonical order, and the [CLS] state of every block (we keep their mean).
E1 = the released encoder.onnx from the PyPI package, run through zeropoint._runtime.Runtime itself (its 'hidden'
output is the per-token last hidden state). E1 also stores the released depth of every option, computed in float64
exactly as Runtime.depths does.
E2 = intfloat/multilingual-e5-small from Hugging Face, original tokenizer and vocabulary, torch on CPU.

Splits with more than PART lists are cached in parts of PART lists; every part is skipped if its file exists.
Output: cache/<ENC>/<split>.part<k>.npz
"""
import os
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import CACHE, load_split

PART = 1000
enc_name, splits = sys.argv[1], sys.argv[2:]
LIMIT = int(os.environ.get("LIMIT", "0"))   # pilot timing only
OUT = os.path.join(CACHE, enc_name if not LIMIT else enc_name + "_pilot")
os.makedirs(OUT, exist_ok=True)


class E1:
    def __init__(self):
        from zeropoint import ZeroPoint
        self.zp = ZeroPoint(threads=2)
        self.rt = self.zp._rt

    def encode(self, text, canon):
        rt = self.rt
        seqs, _ = rt.sequences(text, canon)
        pooled, depth, cls = [], [], []
        for ids, pos, spans in seqs:
            h = rt.session.run(None, {"inputs_embeds": rt._embed(ids),
                                      "position_ids": np.asarray(pos, dtype=np.int64)[None]})[0][0]
            cls.append(h[0].astype(np.float64))
            for a, b in spans:
                m = h[a:b].astype(np.float64).mean(axis=0)
                pooled.append(m)
                z = float(m @ rt.w) + rt.bias
                depth.append(np.logaddexp(0.0, z))
        return np.array(pooled), np.mean(cls, axis=0), np.array(depth), len(seqs)


class E2:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(2)
        self.torch = torch
        name = "intfloat/multilingual-e5-small"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.enc = AutoModel.from_pretrained(name).eval()
        cfg = self.enc.config
        self.pos0 = (cfg.pad_token_id + 1) if cfg.model_type in ("xlm-roberta", "roberta") else 0
        self.CLS, self.SEP, self.PAD, self.UNK = (self.tok.cls_token_id, self.tok.sep_token_id,
                                                  self.tok.pad_token_id, self.tok.unk_token_id)

    def sequences(self, text, canon):
        t = self.tok("query: " + unicodedata.normalize("NFC", text), add_special_tokens=False, truncation=True,
                     max_length=192)["input_ids"]
        t = [self.CLS] + t + [self.SEP]
        out = []
        for s in range(0, len(canon), 20):
            ids, pos, spans = list(t), list(range(len(t))), []
            for o in canon[s:s + 20]:
                oi = (self.tok(unicodedata.normalize("NFC", o), add_special_tokens=False, truncation=True,
                               max_length=16)["input_ids"] or [self.UNK]) + [self.SEP]
                spans.append((len(ids), len(ids) + len(oi) - 1))
                ids += oi
                pos += list(range(len(t), len(t) + len(oi)))
            out.append((ids, pos, spans))
        return out

    def encode_many(self, cases):
        """cases: list of (text, canon). Sequences are batched (padded, masked) by length."""
        torch = self.torch
        seqs, owner = [], []
        for ci, (text, canon) in enumerate(cases):
            for s in self.sequences(text, canon):
                seqs.append(s)
                owner.append(ci)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
        hid = [None] * len(seqs)
        B = 16
        for s in range(0, len(order), B):
            idx = order[s:s + B]
            L = max(len(seqs[i][0]) for i in idx)
            ids = torch.full((len(idx), L), self.PAD, dtype=torch.long)
            pos = torch.zeros((len(idx), L), dtype=torch.long)
            mask = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                x, p, _ = seqs[i]
                ids[r, :len(x)] = torch.tensor(x)
                pos[r, :len(x)] = torch.tensor(p)
                mask[r, :len(x)] = 1
            with torch.inference_mode():
                h = self.enc(input_ids=ids, attention_mask=mask, position_ids=pos + self.pos0).last_hidden_state
            h = h.numpy()
            for r, i in enumerate(idx):
                hid[i] = h[r, :len(seqs[i][0])]
        res = [([], [], 0) for _ in cases]
        res = [{"pooled": [], "cls": [], "nb": 0} for _ in cases]
        for i, (_, _, spans) in enumerate(seqs):
            h = hid[i]
            d = res[owner[i]]
            d["cls"].append(h[0].astype(np.float64))
            d["nb"] += 1
            for a, b in spans:
                d["pooled"].append(h[a:b].astype(np.float64).mean(axis=0))
        return [(np.array(d["pooled"]), np.mean(d["cls"], axis=0), None, d["nb"]) for d in res]


model = E1() if enc_name == "E1" else E2()

for split in splits:
    lists = load_split(split)
    if LIMIT:
        lists = lists[:LIMIT]
    nparts = (len(lists) + PART - 1) // PART
    for k in range(nparts):
        path = os.path.join(OUT, f"{split}.part{k}.npz")
        if os.path.exists(path):
            continue
        part = lists[k * PART:(k + 1) * PART]
        t0 = time.time()
        if enc_name == "E1":
            enc = [model.encode(e["text"], e["opts"]) for e in part]
        else:
            enc = []
            for s in range(0, len(part), 32):
                enc += model.encode_many([(e["text"], e["opts"]) for e in part[s:s + 32]])
        X = np.concatenate([p for p, _, _, _ in enc]).astype(np.float32)
        off = np.r_[0, np.cumsum([len(p) for p, _, _, _ in enc])].astype(np.int64)
        assert all(len(p) == len(e["opts"]) for (p, _, _, _), e in zip(enc, part))
        cls = np.stack([c for _, c, _, _ in enc]).astype(np.float32)
        gold = np.array([e["opts"].index(e["gold"]) if e["gold"] is not None else -1 for e in part], np.int64)
        nb = np.array([n for _, _, _, n in enc], np.int64)
        extra = {}
        if enc_name == "E1":
            extra["depth_rel"] = np.concatenate([d for _, _, d, _ in enc])
        tmp = path + ".tmp.npz"
        np.savez(tmp, X=X, off=off, cls=cls, gold=gold, nblocks=nb, **extra)
        os.replace(tmp, path)
        dt = time.time() - t0
        print(f"[{enc_name}] {split} part {k + 1}/{nparts}: {len(part)} lists, {int(nb.sum())} blocks, "
              f"{dt:.0f}s ({dt / len(part) * 1000:.0f} ms/list)", flush=True)
print(f"[{enc_name}] done {' '.join(splits)}", flush=True)
