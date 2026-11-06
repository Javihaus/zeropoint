"""Stage 0: packaging check of zeropoint==1.0.0 installed from PyPI. Output: results/packaging.json"""
import importlib.metadata as md
import io
import json
import os
import subprocess
import sys
import time
import warnings
from contextlib import redirect_stderr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, RESULTS, jdump

OUT = os.path.join(RESULTS, "packaging.json")
if os.path.exists(OUT):
    sys.exit(0)
rep = {"issues": [], "checks": {}}
W = []
warnings.simplefilter("always")
_show = warnings.showwarning
warnings.showwarning = lambda m, c, f, l, file=None, line=None: W.append(f"{c.__name__}: {m} ({os.path.basename(f)}:{l})")

# import time in a fresh interpreter
t = subprocess.run([sys.executable, "-I", "-c", "import time;t=time.perf_counter();import zeropoint;"
                    "print(time.perf_counter()-t)"], capture_output=True, text=True)
rep["checks"]["import_seconds"] = float(t.stdout.strip())
rep["checks"]["import_stderr"] = t.stderr
t = subprocess.run([sys.executable, "-I", "-c", "import time;t=time.perf_counter();from zeropoint import ZeroPoint;"
                    "ZeroPoint();print(time.perf_counter()-t)"], capture_output=True, text=True)
rep["checks"]["import_and_load_seconds"] = float(t.stdout.strip())

import zeropoint
from zeropoint import Decision, ModelFilesError, UnsupportedLanguage, ZeroPoint


def _json_load(*args, **kw):
    with open(*args, **kw) as f:
        return json.load(f)


def _jsonl(*args, **kw):
    with open(*args, **kw) as f:
        return [json.loads(line) for line in f]
rep["checks"]["version"] = zeropoint.__version__
rep["checks"]["dist_version"] = md.version("zeropoint")
files = [str(f) for f in md.files("zeropoint")]
rep["checks"]["wheel_files"] = files
for need in ["encoder.onnx", "head.json", "manifest.json", "tokenizer.json", "vocab_int8.npy", "vocab_scale.npy"]:
    if not any(f.endswith("model/" + need) for f in files):
        rep["issues"].append(f"wheel lacks model/{need}")
rep["checks"]["requires"] = md.requires("zeropoint")

err = io.StringIO()
with redirect_stderr(err):
    zp = ZeroPoint()
rep["checks"]["repr"] = repr(zp)

# README quick start: the printed tables in README.md
readme_examples = [
    ("Can you move 200 euros from my savings to my checking account?", ["transfer", "balance", "pay bill"],
     "transfer  0.77  ███████████████·····\nbalance   0.02  ····················\npay bill  0.01  ····················\nnone      0.20  ████················\n→ transfer"),
    ("Set an alarm for 6:30 tomorrow morning", ["transfer", "balance", "pay bill"],
     "pay bill  0.00  ····················\ntransfer  0.00  ····················\nbalance   0.00  ····················\nnone      1.00  ████████████████████\n→ none"),
    ("Me han cobrado dos veces la suscripción de marzo, ¿podéis devolverme el duplicado?",
     ["facturación", "soporte técnico", "recursos humanos"],
     "facturación       0.76  ███████████████·····\nsoporte técnico   0.00  ····················\nrecursos humanos  0.00  ····················\nnone              0.24  █████···············\n→ facturación"),
]
qs = []
for text, opts, expected in readme_examples:
    d = zp.decide(text, opts)
    got = repr(d).rsplit(" · ", 1)[0]
    qs.append({"text": text, "matches_readme": got == expected, "got": got, "latency_ms": d.latency_ms,
               "p_none": d.p_none, "depths": d.depths})
    if got != expected:
        rep["issues"].append(f"README quick start output differs for {text!r}: got {got!r}")
rep["checks"]["quick_start"] = qs
d = zp.decide("my card was swallowed by the cash machine", ["block a card", "check the balance", "order a new card"])
rep["checks"]["usage_md_example"] = {"choice": d.choice, "p_none": d.p_none}

# documented fields and attributes
doc_fields = ["choice", "confidence", "p_none", "probabilities", "depths", "text_truncated", "options_truncated",
              "blocks", "latency_ms"]
missing = [f for f in doc_fields if not hasattr(d, f)]
if missing:
    rep["issues"].append(f"Decision lacks documented fields {missing}")
for a in ["languages", "model_id", "threads", "report"]:
    if not hasattr(zp, a):
        rep["issues"].append(f"ZeroPoint lacks documented attribute {a}")
rep["checks"]["languages"] = list(zp.languages)

# documented errors
errs = {}
for name, call, exc in [
    ("duplicate", lambda: zp.decide("x y", ["a", "a"]), ValueError),
    ("empty_option", lambda: zp.decide("x y", ["a", " "]), ValueError),
    ("empty_text", lambda: zp.decide(" ", ["a", "b"]), ValueError),
    ("empty_list", lambda: zp.decide("x y", []), ValueError),
    ("non_string", lambda: zp.decide("x y", ["a", 3]), TypeError),
    ("greek", lambda: zp.decide("Καλημέρα, θέλω να αλλάξω την κάρτα μου", ["a", "b"]), UnsupportedLanguage),
    ("threads0", lambda: ZeroPoint(threads=0), ValueError),
    ("decide_many_str", lambda: zp.decide_many("abc", ["a", "b"]), TypeError),
]:
    try:
        call()
        errs[name] = "no exception"
        rep["issues"].append(f"documented error not raised: {name}")
    except exc as e:
        errs[name] = f"{type(e).__name__}: {e}"
    except Exception as e:   # noqa
        errs[name] = f"unexpected {type(e).__name__}: {e}"
        rep["issues"].append(f"unexpected exception for {name}: {type(e).__name__}")
rep["checks"]["errors"] = errs

# order invariance and decide_many == decide
opts = ["transfer", "balance", "pay bill", "card lost", "exchange rate"]
a = zp.decide("I lost my card yesterday", opts)
b = zp.decide("I lost my card yesterday", opts[::-1])
rep["checks"]["order_invariant_bits"] = all(a.probabilities[o] == b.probabilities[o] for o in opts) and a.p_none == b.p_none
many = zp.decide_many(["I lost my card yesterday", "what is the rate"], opts)
rep["checks"]["decide_many_equals_decide"] = many[0].probabilities == a.probabilities
zp4 = ZeroPoint(threads=1)
c = zp4.decide("I lost my card yesterday", opts)
rep["checks"]["threads_1_vs_2_bits"] = c.probabilities == a.probabilities and c.p_none == a.p_none

# 1.0c checkpoint values documented in the brief: T = 0.2575, b = 4.517
rep["checks"]["temperature"] = zp._rt.temperature
rep["checks"]["none_level"] = zp._rt.none_level
rep["checks"]["onnx_inputs"] = [(i.name, i.shape, i.type) for i in zp._rt.session.get_inputs()]
rep["checks"]["onnx_outputs"] = [(o.name, o.shape, o.type) for o in zp._rt.session.get_outputs()]
rep["checks"]["onnx_exposes"] = ("per-token last hidden states ('hidden', shape 1 x L x 384); depths, pooled "
                                 "states and probabilities are computed in numpy by _runtime.py")

# recalibrate and fit on a few examples (support tickets shipped with the training archive)
rows = _jsonl(os.path.join(DATA, "data", "tickets_fit.jsonl"), encoding="utf-8")
rows = rows[:60]
texts, labels = [r["text"] for r in rows], [r["label"] for r in rows]
t0 = time.time()
try:
    with redirect_stderr(err):
        new, rr = zp.recalibrate(texts, labels)
    rep["checks"]["recalibrate"] = {"seconds": time.time() - t0, "report": rr}
except Exception as e:   # noqa
    rep["issues"].append(f"recalibrate failed: {type(e).__name__}: {e}")
fit_steps = 10
try:
    t0 = time.time()
    with redirect_stderr(err):
        mine, fr = zp.fit(texts, labels, steps=fit_steps)
    dt = time.time() - t0
    rep["checks"]["fit"] = {"examples": len(texts), "steps": fit_steps, "seconds_total": dt,
                            "report_seconds": fr["seconds"],
                            "projected_minutes_for_200_steps": fr["seconds"] / fit_steps * 200 / 60,
                            "report": fr}
    out_dir = os.path.join(os.path.dirname(OUT), "..", "cache", "fit_model")
    mine.save(out_dir)
    again = ZeroPoint(out_dir)
    x1 = mine.decide(texts[0], sorted(set(labels)))
    x2 = again.decide(texts[0], sorted(set(labels)))
    rep["checks"]["fit_save_reload_same_bits"] = x1.probabilities == x2.probabilities and x1.p_none == x2.p_none
    if not rep["checks"]["fit_save_reload_same_bits"]:
        rep["issues"].append("a fitted model saved and reloaded gives different numbers")
    # second fit with same seed: bit identical?
    with redirect_stderr(err):
        mine2, _ = zp.fit(texts, labels, steps=fit_steps)
    y = mine2.decide(texts[0], sorted(set(labels)))
    rep["checks"]["fit_deterministic_same_seed"] = y.probabilities == x1.probabilities
except Exception as e:   # noqa
    import traceback
    rep["issues"].append(f"fit failed: {type(e).__name__}: {e}")
    rep["checks"]["fit_traceback"] = traceback.format_exc()
rep["checks"]["stderr"] = err.getvalue()[-5000:]
rep["warnings"] = sorted(set(W))
jdump(rep, OUT)
print(json.dumps(rep["issues"], indent=1))
