"""ZeroPoint 1.0 training and checks, on a GPU (Colab L4).

The model reads the text and every option in one sequence; "none" stays at energy zero:
    [CLS] query: <text> [SEP] <option 1> [SEP] ... <option K> [SEP]
    E_y = -a_y,  a_y = softplus(head(mean of option y's token states)),  w_y = exp(a_y / T) - 1,  w_none = exp(b)

What this run adds, against 1.0b:
  * The order of the options cannot change the answer (same position ids for every option, canonical order,
    duplicates rejected, ties to the first option in canonical order). Unchanged from 1.0b.
  * Vocabulary of 25 languages (EU official languages in Latin script + Catalan, Basque, Galician), about 90,000
    pieces instead of 250,000; the model is trained with it, so the model measured is the model shipped.
  * Wrong options: in half of the lists, half of them are the labels closest to the right one (data/neighbors.json).
  * Long lists: 8% of the lists have 21 to all options of the pool, read in blocks of 20 like at inference.
  * Calibration: the dev data is split in two halves by text. One decides when to stop; the other, after training,
    fits the single temperature and the "none" level (two numbers, encoder frozen).
  * Reports: share of confident answers (0.9 or more), wrong answers among them, and "false admissions": no right
    option in the list, yet an option chosen with confidence 0.9 or more.
  * fit(): 200 steps, with all layers or with the first 8 frozen (faster on a CPU).
  * Training is deterministic for a given seed and GPU type.

    python zp1.py train --seed 0          # trains, then evaluates every test
    python zp1.py checks --ckpt results/s0.pt   # order, repeat, long lists, calibration, CPU bit-stability
    python zp1.py fit --ckpt results/s0.pt      # adapt to a client's tickets with 100 to 2000 examples
    python zp1.py summary
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import glob
import hashlib
import json
import math
import random
import time
import unicodedata

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["train", "checks", "fit", "summary"])
ap.add_argument("--model", default="intfloat/multilingual-e5-small")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--n_base", type=int, default=50000, help="lists from the original training texts")
ap.add_argument("--n_broad", type=int, default=12000, help="lists with broad labels")
ap.add_argument("--synth_repeat", type=int, default=3, help="lists per written example")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--text_len", type=int, default=192)
ap.add_argument("--opt_len", type=int, default=16)
ap.add_argument("--chunk", type=int, default=20, help="options per sequence; training lists never exceed it")
ap.add_argument("--long_share", type=float, default=0.08, help="share of lists with 21 to all options of the pool")
ap.add_argument("--hard_share", type=float, default=0.5, help="share of lists whose wrong options include the most confusable ones")
ap.add_argument("--vocab", default="vocab", help="folder with the reduced tokenizer.json and keep_ids.npy")
ap.add_argument("--none_share", type=float, default=0.3)
ap.add_argument("--fit_steps", type=int, default=200)
ap.add_argument("--fit_freeze", type=int, default=0, help="encoder layers kept frozen during fit (0 = none)")
ap.add_argument("--ckpt", default="results/s0.pt")
ap.add_argument("--limit", type=int, default=0, help="smoke test: few examples, few cases")
args = ap.parse_args()

TESTS = ["bench2000", "snips", "atis", "mtop_es", "hwu64", "clinc_real_indomain", "tickets_made", "tickets_far"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("results", exist_ok=True)
if os.path.exists(os.path.join(args.vocab, "tokenizer.json")):
    # 25 languages (EU official languages in Latin script + Catalan, Basque, Galician); see vocab/report.json
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(args.vocab, "tokenizer.json"), bos_token="<s>",
                                  eos_token="</s>", unk_token="<unk>", pad_token="<pad>", cls_token="<s>",
                                  sep_token="</s>", mask_token="<mask>")
    KEEP = np.load(os.path.join(args.vocab, "keep_ids.npy"))
else:
    tok = AutoTokenizer.from_pretrained(args.model)
    KEEP = None
CLS, SEP, PAD = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
NB = json.load(open("data/neighbors.json", encoding="utf-8")) if os.path.exists("data/neighbors.json") else {}


# ---------------------------------------------------------------- canonical order and input encoding

def canonical_order(options):
    """Indices of the options in canonical order. Raises on empty or duplicate options."""
    keys = [unicodedata.normalize("NFC", o) for o in options]
    if any(not k.strip() for k in keys):
        raise ValueError("empty option")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate options (after Unicode NFC normalisation)")
    return sorted(range(len(options)), key=lambda i: keys[i].encode("utf-8"))


def text_ids(text):
    return tok("query: " + unicodedata.normalize("NFC", text), add_special_tokens=False, truncation=True,
               max_length=args.text_len)["input_ids"]


def option_ids(option):
    return tok(unicodedata.normalize("NFC", option), add_special_tokens=False, truncation=True,
               max_length=args.opt_len)["input_ids"] or [tok.unk_token_id]


class Joint(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(args.model)
        if KEEP is not None:   # keep only the rows of the 25-language vocabulary
            W = self.enc.embeddings.word_embeddings.weight.data[torch.tensor(KEEP)].clone()
            self.enc.embeddings.word_embeddings = torch.nn.Embedding.from_pretrained(W, freeze=False)
            self.enc.config.vocab_size = len(KEEP)
        # XLM-R counts positions from padding_idx + 1; BERT from 0.
        self.pos0 = (self.enc.config.pad_token_id + 1) if self.enc.config.model_type in ("xlm-roberta", "roberta") else 0
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.constant_(self.head.bias, math.log(math.expm1(1.0)))
        self.log_t = torch.nn.Parameter(torch.zeros(()))
        self.b = torch.nn.Parameter(torch.zeros(()))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=DEV))
        return self

    def sequences(self, text, options):
        """One sequence per chunk of options (already in canonical order). Every option block gets the same
        position ids, starting right after the text."""
        t = [CLS] + text_ids(text) + [SEP]
        out = []
        for s in range(0, len(options), args.chunk):
            ids, pos, spans = list(t), list(range(len(t))), []
            for o in options[s:s + args.chunk]:
                oi = option_ids(o) + [SEP]
                spans.append((len(ids), len(ids) + len(oi) - 1))
                ids += oi
                pos += list(range(len(t), len(t) + len(oi)))
            out.append((ids, pos, spans))
        return out

    def depths(self, cases):
        """cases: list of (text, options in canonical order). Returns one depth vector per case."""
        seqs, owner = [], []
        for ci, (text, options) in enumerate(cases):
            for s in self.sequences(text, options):
                seqs.append(s)
                owner.append(ci)
        L = max(len(x) for x, _, _ in seqs)
        ids = torch.full((len(seqs), L), PAD, dtype=torch.long)
        pos = torch.zeros((len(seqs), L), dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, (x, p, _) in enumerate(seqs):
            ids[i, :len(x)] = torch.tensor(x)
            pos[i, :len(x)] = torch.tensor(p)
            mask[i, :len(x)] = 1
        h = self.enc(input_ids=ids.to(DEV), attention_mask=mask.to(DEV),
                     position_ids=(pos + self.pos0).to(DEV)).last_hidden_state
        out = [[] for _ in cases]
        for i, (_, _, spans) in enumerate(seqs):
            pooled = torch.stack([h[i, s:e].mean(0) for s, e in spans])
            out[owner[i]].append(F.softplus(self.head(pooled)).squeeze(-1))
        return [torch.cat(o) for o in out]

    def logits(self, cases):
        """(B, Kmax+1) Boltzmann log-weights, vacuum last; padded options get -1e4."""
        ds = self.depths(cases)
        km = max(len(d) for d in ds)
        rows = []
        for d in ds:
            x = (d.float() / self.log_t.exp()).clamp(1e-6, 60)
            lw = x + torch.log(-torch.expm1(-x))
            r = torch.full((km + 1,), -1e4, device=lw.device)
            r[:len(d)] = lw
            r[km] = self.b
            rows.append(r)
        return torch.stack(rows), km

    @torch.no_grad()
    def predict(self, text, options):
        """Probabilities in the caller's order, p_none, and the decision. One case at a time, so the result
        never depends on what else is being processed."""
        order = canonical_order(options)
        canon = [options[i] for i in order]
        lg, km = self.logits([(text, canon)])
        p = lg.float().softmax(-1)[0].cpu().numpy()
        probs = {canon[j]: float(p[j]) for j in range(len(canon))}
        best = max(range(len(canon)), key=lambda j: (p[j], -j))   # ties: first in canonical order
        return {"p": {o: probs[o] for o in options}, "p_none": float(p[km]),
                "decision": canon[best] if p[best] >= p[km] else None}


# ---------------------------------------------------------------- training data

def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def canon_case(text, opts, gold):
    opts = [str(o) for o in opts]
    canon = [opts[i] for i in canonical_order(opts)]
    return {"text": text, "opts": canon, "gold": gold}


def wrong_options(pool, lab, same, k, rng):
    """k wrong options from the same pool. In a share of lists, half of them are the labels closest to the right
    one (the easiest to confuse), so the model has to tell near neighbours apart."""
    k = min(k, len(same))
    near = [x for x in NB.get(pool, {}).get(lab, []) if x in same]
    if near and rng.random() < args.hard_share:
        h = min(len(near), max(1, k // 2))
        picked = list(rng.choice(near, h, replace=False))
        rest = [x for x in same if x not in picked]
        return picked + list(rng.choice(rest, k - h, replace=False)) if k > h else picked
    return list(rng.choice(same, k, replace=False))


def lists_from_rows(rows, label_key, pools, n, rng, cross=True):
    """Lists of 4-20 options from the text's own pool (a share up to the whole pool); a share of "none" lists
    (right option removed, or options from another pool). Wrong options often include the closest labels."""
    names = sorted(pools)
    out = []
    for i in rng.choice(len(rows), min(n, len(rows)), replace=False):
        r = rows[i]
        pool, lab = r["_pool"], r[label_key]
        same = [l for l in pools[pool] if l != lab]
        if not same:
            continue
        k = int(rng.integers(4, 21))
        if len(same) > 21 and rng.random() < args.long_share:
            k = int(rng.integers(21, len(same) + 1))
        u = rng.random()
        if u < args.none_share / 2 or (u < args.none_share and not cross):
            opts, gold = wrong_options(pool, lab, same, k, rng), None
        elif u < args.none_share:
            p2 = names[rng.integers(len(names))]
            other = [l for l in pools[p2] if l != lab]
            opts, gold = list(rng.choice(other, min(k, len(other)), replace=False)), None
        else:
            opts, gold = wrong_options(pool, lab, same, k - 1, rng) + [lab], lab
        out.append(canon_case(r["text"], opts, gold))
    return out


def synth_lists(rows, pools, repeat, rng):
    """Lists for the written examples: the hard negatives are always in the list; a share of "none" lists keeps
    the hard negatives and drops the right option; null-label examples are always "none"."""
    out = []
    for r in rows:
        labs = pools[r["pool"]]
        for _ in range(repeat):
            hard = list(r["hard_negatives"])
            rest = [l for l in labs if l != r["label"] and l not in hard]
            k = int(rng.integers(max(4, len(hard) + 1), len(labs) + 1))
            gold = r["label"] if (r["label"] is not None and rng.random() >= args.none_share) else None
            n_rest = k - len(hard) - (1 if gold else 0)
            opts = hard + list(rng.choice(rest, max(0, min(n_rest, len(rest))), replace=False))
            if gold:
                opts.append(gold)
            out.append(canon_case(r["text"], opts, gold))
    return out


def held_out(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 20 == 0   # 5% for model selection


def build_data(seed):
    rng = np.random.default_rng(seed)
    lim = args.limit
    tr = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")]
    dv = [json.loads(l) for l in open("data/dev.jsonl", encoding="utf-8")]
    cmap = json.load(open("data/coarse_map.json", encoding="utf-8"))
    fine_pools, broad_pools = {}, {}
    for r in tr:
        fine_pools.setdefault(r["source"], set()).add(r["label"])
    fine_pools = {k: sorted(v) for k, v in fine_pools.items()}
    for src, m in cmap.items():
        broad_pools[src + "#broad"] = sorted(set(m.values()))
    for rows in (tr, dv):
        for r in rows:
            r["_pool"] = r["source"]
            r["broad"] = cmap[r["source"]][r["label"]]
    tr = [r for r in tr if r["label"] in fine_pools[r["source"]]]
    dv = [r for r in dv if r["label"] in fine_pools.get(r["source"], [])]
    tr_b = [{**r, "_pool": r["source"] + "#broad"} for r in tr]
    dv_b = [{**r, "_pool": r["source"] + "#broad"} for r in dv]
    synth_pools = {k: v["labels"] for k, v in json.load(open("data/synth_pools.json", encoding="utf-8")).items()
                   if not k.startswith("_")}
    syn = [json.loads(l) for l in open("data/synth.jsonl", encoding="utf-8")]
    syn_tr = [r for r in syn if not held_out(r["text"])]
    syn_dv = [r for r in syn if held_out(r["text"])]
    if lim:
        syn_tr, syn_dv = syn_tr[:lim], syn_dv[:lim // 4 or 1]
    train = (lists_from_rows(tr, "label", fine_pools, args.n_base if not lim else lim, rng)
             + lists_from_rows(tr_b, "broad", broad_pools, args.n_broad if not lim else lim // 2, rng)
             + synth_lists(syn_tr, synth_pools, args.synth_repeat, rng))
    drng = np.random.default_rng(1000)
    dev = (lists_from_rows(dv, "label", fine_pools, 1500 if not lim else 16, drng)
           + lists_from_rows(dv_b, "broad", broad_pools, 500 if not lim else 8, drng)
           + synth_lists(syn_dv, synth_pools, 1, drng))
    random.Random(seed).shuffle(train)
    # two disjoint halves: one decides when to stop, the other fits temperature and "none" level afterwards
    half = lambda e: int(hashlib.sha256(e["text"].encode("utf-8")).hexdigest(), 16) % 2
    dev_stop = [e for e in dev if half(e) == 0]
    dev_cal = [e for e in dev if half(e) == 1]
    print(f"train lists {len(train)}  dev lists {len(dev_stop)} (stop) + {len(dev_cal)} (calibration)  "
          f"(written examples: {len(syn_tr)} train, {len(syn_dv)} dev)", flush=True)
    return train, dev_stop, dev_cal


# ---------------------------------------------------------------- training

def loss_on(model, exs, reduction="mean"):
    lg, km = model.logits([(e["text"], e["opts"]) for e in exs])
    t = torch.tensor([e["opts"].index(e["gold"]) if e["gold"] else km for e in exs], device=lg.device)
    return F.cross_entropy(lg, t, reduction=reduction)


@torch.no_grad()
def dev_loss(model, dev):
    model.eval()
    tot = sum(loss_on(model, dev[i:i + 32], "sum").item() for i in range(0, len(dev), 32))
    model.train()
    return tot / len(dev)


def make_opt(model, lr, steps):
    params = [p for n, p in model.named_parameters() if n not in ("log_t", "b") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": params, "lr": lr}, {"params": [model.log_t, model.b], "lr": 1e-2}],
                            weight_decay=0.01)
    warm = min(200, max(1, steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / steps))
    return opt, sched


def train_steps(model, batches, opt, sched, tag, log_every=500):
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    model.train()
    for step, exs in enumerate(batches, 1):
        with torch.autocast(DEV, enabled=DEV == "cuda"):
            loss = loss_on(model, exs)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % log_every == 0:
            print(f"[{tag}] step {step}/{len(batches)} loss {loss.item():.3f}", flush=True)


@torch.no_grad()
def _depths_of(model, exs):
    model.eval()
    out = []
    for i in range(0, len(exs), 32):
        out += [d.float().cpu() for d in model.depths([(e["text"], e["opts"]) for e in exs[i:i + 32]])]
    return out


def _cal_nll_ece(depths, exs, log_t, b):
    nll, conf, right = 0.0, [], []
    for d, e in zip(depths, exs):
        x = (d / log_t.exp()).clamp(1e-6, 60)
        lg = torch.cat([x + torch.log(-torch.expm1(-x)), b.reshape(1)])
        t = e["opts"].index(e["gold"]) if e["gold"] else len(d)
        nll = nll + F.cross_entropy(lg.unsqueeze(0), torch.tensor([t]))
        p = lg.softmax(-1)
        j = int(torch.argmax(p))
        conf.append(float(p[j]))
        right.append(j == t)
    conf, right = np.array(conf), np.array(right, float)
    e_ = sum(((conf > lo) & (conf <= lo + .1)).mean() * abs(conf[(conf > lo) & (conf <= lo + .1)].mean()
             - right[(conf > lo) & (conf <= lo + .1)].mean())
             for lo in np.linspace(0, 1, 10, endpoint=False) if ((conf > lo) & (conf <= lo + .1)).any())
    return nll / len(exs), float(e_)


def calibrate(model, dev_cal, dev_stop):
    """Fit the single temperature and the "none" level on the calibration half only, encoder frozen.
    Two numbers, chosen by likelihood; the depths of every option stay as trained."""
    dc, ds = _depths_of(model, dev_cal), _depths_of(model, dev_stop)
    log_t = model.log_t.detach().float().cpu().clone().requires_grad_(True)
    b = model.b.detach().float().cpu().clone().requires_grad_(True)
    before = {"cal": _cal_nll_ece(dc, dev_cal, log_t.detach(), b.detach()),
              "stop": _cal_nll_ece(ds, dev_stop, log_t.detach(), b.detach())}
    opt = torch.optim.LBFGS([log_t, b], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = _cal_nll_ece(dc, dev_cal, log_t, b)[0]
        loss.backward()
        return loss
    opt.step(closure)
    after = {"cal": _cal_nll_ece(dc, dev_cal, log_t.detach(), b.detach()),
             "stop": _cal_nll_ece(ds, dev_stop, log_t.detach(), b.detach())}
    rep_ = {"T_before": float(model.log_t.exp()), "b_before": float(model.b), "T_after": float(log_t.exp()),
            "b_after": float(b), **{f"{w}_{k}": {"nll": float(v[0]), "ece": v[1]} for w, d in
                                    [("before", before), ("after", after)] for k, v in d.items()}}
    with torch.no_grad():
        model.log_t.fill_(float(log_t))
        model.b.fill_(float(b))
    print("[calibration]", json.dumps(rep_), flush=True)
    return rep_


def cmd_train():
    seed_all(args.seed)
    train, dev, dev_cal = build_data(args.seed)
    model = Joint().to(DEV)
    per_epoch = [train[i:i + args.batch] for i in range(0, len(train), args.batch)]
    opt, sched = make_opt(model, args.lr, args.epochs * len(per_epoch))
    tag, best, t0 = f"s{args.seed}", float("inf"), time.time()
    for ep in range(1, args.epochs + 1):
        order = list(range(len(per_epoch)))
        random.Random(args.seed * 100 + ep).shuffle(order)
        train_steps(model, [per_epoch[i] for i in order], opt, sched, tag)
        dl = dev_loss(model, dev)
        print(f"[{tag}] epoch {ep} dev loss {dl:.4f} {time.time() - t0:.0f}s", flush=True)
        if dl < best:
            best = dl
            model.save(f"results/{tag}.pt")
    model.load(f"results/{tag}.pt")
    cal = calibrate(model, dev_cal, dev)
    model.save(f"results/{tag}.pt")
    json.dump({"seed": args.seed, "calibration": cal, "epochs": args.epochs, "train_seconds": time.time() - t0, "device": DEV,
               "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else None, "dev_loss": best,
               "torch": torch.__version__, "transformers": __import__("transformers").__version__,
               "model": args.model, "args": vars(args),
               "sha256": hashlib.sha256(open(f"results/{tag}.pt", "rb").read()).hexdigest()},
              open(f"results/{tag}_train_info.json", "w"), indent=1)
    evaluate_all(model, tag)


# ---------------------------------------------------------------- evaluation

def scores(items, rows):
    y = np.array([it["gold"] is not None for it in items], int)
    conf = np.array([1 - r["p_none"] for r in rows])
    acc = float(np.mean([max(r["p"], key=r["p"].get) == it["gold"] for r, it in zip(rows, items) if it["gold"]]))
    out = {"choice": acc, "n": len(items)}
    if y.min() != y.max():
        th = np.quantile(conf[y == 1], 0.05, method="lower")
        out.update({"auroc": float(roc_auc_score(y, conf)), "fpr95": float(np.mean(conf[y == 0] >= th))})
    out["default_acc"] = float(np.mean([r["decision"] == it["gold"] for r, it in zip(rows, items)]))
    sure = [(r, it) for r, it in zip(rows, items) if max(max(r["p"].values()), r["p_none"]) >= 0.9]
    out["sure_share"] = len(sure) / len(items)
    out["wrong_when_sure"] = float(np.mean([r["decision"] != it["gold"] for r, it in sure])) if sure else None
    neg = [r for r, it in zip(rows, items) if it["gold"] is None]
    if neg:   # no right option in the list, yet an option chosen with confidence 0.9 or more
        out["false_admit_09"] = float(np.mean([r["decision"] is not None and max(r["p"].values()) >= 0.9
                                               for r in neg]))
    out["ece"] = ece(items, rows)
    ms = [r["ms"] for r in rows if "ms" in r]
    if ms:
        out["ms_median"] = float(np.median(ms))
        out["ms_p99"] = float(np.percentile(ms, 99))
    return out


def ece(items, rows, bins=10):
    """Calibration of the final decision (an option or none): in each confidence bin, how far is the stated
    confidence from the share of right decisions? 0 = perfectly calibrated."""
    conf = np.array([max(max(r["p"].values()), r["p_none"]) for r in rows])
    right = np.array([r["decision"] == it["gold"] for r, it in zip(rows, items)], float)
    tot = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            tot += m.mean() * abs(conf[m].mean() - right[m].mean())
    return float(tot)


def run_items(model, items):
    model.eval()
    rows = []
    for it in items:
        s = time.perf_counter()
        r = model.predict(it["text"], it["options"])
        if DEV == "cuda":
            torch.cuda.synchronize()
        r["ms"] = (time.perf_counter() - s) * 1000
        rows.append(r)
    return rows


def load_test(path):
    items = json.load(open(path, encoding="utf-8"))
    if args.limit:
        items = items[:args.limit // 2] + items[-args.limit // 2:]
    return items


def evaluate_all(model, tag):
    res = {}
    for t in TESTS:
        items = load_test(f"tests/{t}.json")
        rows = run_items(model, items)
        json.dump(rows, open(f"results/{tag}_cases_{t}.json", "w"))
        res[t] = scores(items, rows)
        print(f"[{tag}] {t} {json.dumps({k: (round(v, 3) if v is not None else None) for k, v in res[t].items()})}", flush=True)
    json.dump(res, open(f"results/{tag}_results.json", "w"), indent=1)


def cmd_checks():
    """Order, repeat, long lists, calibration, and whether the CPU gives the same bits with 1, 2 and 4 threads."""
    global DEV
    model = Joint().to(DEV).load(args.ckpt)
    model.eval()
    res = {"order": {}, "long": {}}
    for t in ["bench2000", "snips", "atis", "mtop_es", "clinc_real_indomain"]:
        items = load_test(f"tests/{t}.json")[:300]
        runs = [run_items(model, items)]
        for k in range(3):
            r_ = random.Random(100 + k)
            runs.append(run_items(model, [{**it, "options": r_.sample(it["options"], len(it["options"]))}
                                          for it in items]))
        runs.append(run_items(model, items))  # the same input again
        diff = max(abs(run[i]["p"][o] - runs[0][i]["p"][o]) for run in runs[1:] for i, it in enumerate(items)
                   for o in it["options"])
        diff_none = max(abs(run[i]["p_none"] - runs[0][i]["p_none"]) for run in runs[1:] for i in range(len(items)))
        changes = float(np.mean([len({run[i]["decision"] for run in runs}) > 1 for i in range(len(items))]))
        res["order"][t] = {"cases": len(items), "runs": len(runs), "decision_changes": changes,
                           "max_abs_diff_p": diff, "max_abs_diff_p_none": diff_none}
        print("[order]", t, res["order"][t], flush=True)
    for path in sorted(glob.glob("tests_long/*.json")):
        name = os.path.basename(path)[:-5]
        items = load_test(path)
        res["long"][name] = scores(items, run_items(model, items))
        print("[long]", name, {k: (round(v, 3) if v is not None else None) for k, v in res["long"][name].items()}, flush=True)
    # CPU: same bits with different thread counts?
    DEV = "cpu"
    cpu = Joint().load(args.ckpt).to("cpu").eval()
    items = load_test("tests/snips.json")[:100] + load_test("tests/mtop_es.json")[:100]
    outs = {}
    for th in [1, 2, 4]:
        torch.set_num_threads(th)
        outs[th] = [cpu.predict(it["text"], it["options"]) for it in items]
    flat = {th: np.concatenate([np.array([o["p_none"]] + [o["p"][x] for x in it["options"]], dtype=np.float64)
                                for o, it in zip(outs[th], items)]) for th in outs}
    res["cpu_threads"] = {f"{a}_vs_{b}": {"bit_identical": bool(np.array_equal(flat[a], flat[b])),
                                          "max_abs_diff": float(np.max(np.abs(flat[a] - flat[b]))),
                                          "decision_changes": int(sum(x["decision"] != y["decision"]
                                                                      for x, y in zip(outs[a], outs[b])))}
                          for a, b in [(1, 2), (1, 4)]}
    gpu_vs_cpu = None
    if torch.cuda.is_available():
        DEV = "cuda"
        g = [model.predict(it["text"], it["options"]) for it in items]
        gflat = np.concatenate([np.array([o["p_none"]] + [o["p"][x] for x in it["options"]]) for o, it in zip(g, items)])
        gpu_vs_cpu = {"max_abs_diff": float(np.max(np.abs(gflat - flat[1]))),
                      "decision_changes": int(sum(x["decision"] != y["decision"] for x, y in zip(g, outs[1])))}
    res["gpu_vs_cpu"] = gpu_vs_cpu
    print("[cpu]", res["cpu_threads"], "gpu vs cpu:", gpu_vs_cpu, flush=True)
    json.dump(res, open("results/checks.json", "w"), indent=1)


def freeze_layers(model, n):
    """Keep the word embeddings and the first n encoder layers fixed during fit (n = 0: train everything)."""
    if n <= 0:
        return
    for p in model.enc.embeddings.parameters():
        p.requires_grad_(False)
    for layer in model.enc.encoder.layer[:n]:
        for p in layer.parameters():
            p.requires_grad_(False)


def cmd_fit():
    """Adapt to a client's tickets with 100, 300, 1000 and 2000 examples. Always --fit_steps optimiser steps,
    so a small client set is seen many times and a large one once or twice. Also times one step on the CPU."""
    global DEV
    rows = [json.loads(l) for l in open("data/tickets_fit.jsonl", encoding="utf-8")]
    labels = sorted({r["label"] for r in rows})
    res = {}
    base = Joint().to(DEV).load(args.ckpt)
    for t in ["tickets_made", "tickets_far"]:
        res[f"before/{t}"] = scores(load_test(f"tests/{t}.json"), run_items(base, load_test(f"tests/{t}.json")))
    del base
    steps = args.fit_steps if not args.limit else 4
    grid = [(100, 0), (300, 0), (1000, 0), (300, 8), (1000, 8)] if not args.limit else [(16, 0), (16, 8)]
    for n, freeze in grid:
        seed_all(0)
        rng = np.random.default_rng(0)
        sub = [{**r, "_pool": "client"} for r in rows[:n]]
        batches = []
        while len(batches) < steps:
            ex = lists_from_rows(sub, "label", {"client": labels}, len(sub), rng, cross=False)
            batches += [ex[i:i + args.batch] for i in range(0, len(ex), args.batch)]
        batches = batches[:steps]
        model = Joint().to(DEV).load(args.ckpt)
        freeze_layers(model, freeze)
        opt, sched = make_opt(model, args.lr, steps)
        t0 = time.time()
        tag = f"fit{n}_freeze{freeze}"
        train_steps(model, batches, opt, sched, tag, log_every=100)
        res[f"{tag}/gpu_seconds"] = time.time() - t0
        for t in ["tickets_made", "tickets_far"]:
            items = load_test(f"tests/{t}.json")
            res[f"{tag}/{t}"] = scores(items, run_items(model, items))
        print("[fit]", tag, {k: v for k, v in res.items() if k.startswith(tag)}, flush=True)
    DEV = "cpu"
    torch.set_num_threads(2)
    for freeze in [0, 8]:
        model = Joint().load(args.ckpt).to("cpu")
        freeze_layers(model, freeze)
        exs = lists_from_rows([{**r, "_pool": "client"} for r in rows[:64]], "label", {"client": labels}, 64,
                              np.random.default_rng(2), cross=False)
        opt, sched = make_opt(model, args.lr, 4)
        t0 = time.time()
        train_steps(model, [exs[i:i + args.batch] for i in range(0, 64, args.batch)], opt, sched, "cpu")
        per_step = (time.time() - t0) / 4
        res[f"cpu_freeze{freeze}_seconds_per_step_2_threads"] = per_step
        res[f"cpu_freeze{freeze}_minutes_for_fit_2_threads"] = per_step * args.fit_steps / 60
        print(f"[fit] cpu, {freeze} frozen layers: {per_step:.1f} s per step, "
              f"{per_step * args.fit_steps / 60:.1f} min for a fit", flush=True)
    json.dump(res, open("results/fit_results.json", "w"), indent=1)


def cmd_summary():
    for f in sorted(glob.glob("results/s*_results.json")):
        tag = os.path.basename(f)[:-13]
        print(f"\n== {tag} ==")
        for t, r in json.load(open(f)).items():
            print(f"  {t:22s} choice {r['choice']:.3f}  AUROC {r.get('auroc', float('nan')):.3f}  "
                  f"FPR95 {r.get('fpr95', float('nan')):.2f}  ECE {r['ece']:.3f}")
    for name in ["checks", "fit_results"]:
        p = f"results/{name}.json"
        if os.path.exists(p):
            print(f"\n== {name} ==")
            print(json.dumps(json.load(open(p)), indent=1))


{"train": cmd_train, "checks": cmd_checks, "fit": cmd_fit, "summary": cmd_summary}[args.cmd]()
