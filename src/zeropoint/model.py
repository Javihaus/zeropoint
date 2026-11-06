"""ZeroPoint 1.0: choose among a list of options, or say that none of them fits.

    from zeropoint import ZeroPoint
    zp = ZeroPoint()
    d = zp.decide("Can you move 200 euros from my savings to my checking account?",
                  ["transfer", "balance", "pay bill"])
    d.choice, d.confidence, d.p_none     # 'transfer', about 0.77, about 0.20

One model reads the text and all the options together. Each option gets a well whose depth says how well it fits;
"none of them" is a state with energy zero. The probabilities of the options and of "none" come from one Boltzmann
distribution and add up to 1.

What the result never depends on: the order of the options, the number of CPU threads, what else is being decided
at the same time. The same text and options always give the same numbers.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unicodedata
import warnings
import weakref
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ._runtime import MODEL_DIR, Runtime, canonical_order

# Share of letters outside the Latin script above which a text is refused. Measured on SIB-200 (all splits):
# the 25 supported languages reach at most 0.017; Greek, Cyrillic, Chinese, Japanese, Korean, Arabic, Hebrew,
# Devanagari, Thai, Georgian and Armenian texts start at 0.259.
NON_LATIN_LIMIT = 0.2


class UnsupportedLanguage(ValueError):
    """The text is not written in the Latin script, which every supported language uses."""


def _non_latin_share(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum("LATIN" not in unicodedata.name(c, "") for c in letters) / len(letters)


def _bar(p: float, width: int = 20) -> str:
    n = int(round(p * width))
    return "█" * n + "·" * (width - n)


@dataclass(frozen=True)
class Decision:
    """The outcome for one text.

    choice             the chosen option, or None when "none of them" is the most probable outcome
    confidence         probability of `choice` (of "none" when choice is None)
    p_none             probability that none of the options is right
    probabilities      probability of each option, in the order the options were given; with p_none they sum to 1
    depths             depth of each option's well (near 0 when the text does not belong there)
    text_truncated     True when the text was longer than the model reads (192 pieces) and its end was ignored
    options_truncated  options longer than the model reads (16 pieces), whose end was ignored
    blocks             how many reads of 20 options the list needed
    latency_ms         time to decide this text, in milliseconds
    """

    choice: str | None
    confidence: float
    p_none: float
    probabilities: dict[str, float] = field(repr=False)
    depths: dict[str, float] = field(repr=False)
    text_truncated: bool = field(repr=False, default=False)
    options_truncated: tuple[str, ...] = field(repr=False, default=())
    blocks: int = field(repr=False, default=1)
    latency_ms: float = field(repr=False, default=0.0)

    def __repr__(self) -> str:
        """A small table: each option with its probability, then "none", then the decision."""
        items = sorted(self.probabilities.items(), key=lambda kv: -kv[1])
        shown = items if len(items) <= 6 else items[:5]
        width = max([len(k) for k, _ in shown] + [4])
        lines = [f"{k:<{width}}  {v:.2f}  {_bar(v)}" for k, v in shown]
        if len(shown) < len(items):
            lines.append(f"{'':<{width}}  (+{len(items) - len(shown)} more options)")
        lines.append(f"{'none':<{width}}  {self.p_none:.2f}  {_bar(self.p_none)}")
        lines.append(f"→ {self.choice if self.choice is not None else 'none'} · {self.latency_ms:.0f} ms")
        return "\n".join(lines)


class ZeroPoint:
    """The decision model. Load it once and reuse it: loading checks every model file against its sha256."""

    def __init__(self, model_dir: str = MODEL_DIR, threads: int = 2):
        if not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        self._rt = Runtime(model_dir, threads=threads)
        self.languages: tuple[str, ...] = tuple(self._rt.manifest["vocabulary"]["languages"])
        self.model_id: str = self._rt.manifest["source_checkpoint_sha256"][:12]
        self._log_temperature = self._rt.log_temperature
        self._none_level = self._rt.none_level
        self.report: dict | None = self._rt.manifest.get("report")

    @property
    def threads(self) -> int:
        return self._rt.threads

    def decide(self, text: str, options: Sequence[str]) -> Decision:
        """Choose the option that fits the text, or None when none of them does."""
        t0 = time.perf_counter()
        options = self._check(text, options)
        order = canonical_order(options)
        canon = [options[i] for i in order]
        depths, reading = self._rt.depths(text, canon)
        p = self._rt.distribution(depths, float(np.exp(self._log_temperature)), self._none_level)
        best = max(range(len(canon)), key=lambda j: (p[j], -j))      # ties: first in canonical order
        p_canon = {canon[j]: float(p[j]) for j in range(len(canon))}
        d_canon = {canon[j]: float(depths[j]) for j in range(len(canon))}
        p_none = float(p[-1])
        choice = canon[best] if p[best] >= p[-1] else None
        return Decision(
            choice=choice,
            confidence=p_canon[choice] if choice is not None else p_none,
            p_none=p_none,
            probabilities={o: p_canon[o] for o in options},
            depths={o: d_canon[o] for o in options},
            text_truncated=reading.text_truncated,
            options_truncated=tuple(reading.options_truncated),
            blocks=reading.blocks,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def decide_many(self, texts: Sequence[str], options) -> list[Decision]:
        """Decide several texts. `options` is one list for all texts, or one list per text.

        Each text is computed on its own, never padded together with others, so every result is exactly the one
        `decide` gives for that text alone."""
        if isinstance(texts, str):
            raise TypeError("texts must be a list of strings; use decide() for one text")
        texts = list(texts)
        per_text = (len(options) == len(texts) and len(texts) > 0
                    and all(isinstance(o, (list, tuple)) for o in options))
        return [self.decide(t, options[i] if per_text else options) for i, t in enumerate(texts)]

    # ---------------------------------------------------------------- adapting to your data

    def fit(self, texts: Sequence[str], labels: Sequence[str | None], **options):
        """Adapt the model to your labelled texts. Returns (new model, report). Needs zeropoint[fit].
        See zeropoint.fit for what happens and for the options (groups, steps, frozen_layers, seed, threads)."""
        from .fit import fit
        return fit(self, texts, labels, **options)

    def recalibrate(self, texts: Sequence[str], labels: Sequence[str | None], *, groups=None, seed: int = 0):
        """Re-fit only the temperature and the "none" level on your labelled texts, nothing else.
        No torch needed. Returns (new model, report); the report compares before and after on a held-out 20%.
        Warns when that held-out part is too small to trust the report, or when the two numbers move far."""
        from .fit import _held_out, _lists, _evaluate, _warn_small_held_out
        texts, labels = list(texts), list(labels)
        label_set = sorted({l for l in labels if l is not None}, key=lambda s: s.encode("utf-8"))
        if len(label_set) < 2:
            raise ValueError("recalibrate needs at least two different labels")
        keys = list(groups) if groups is not None else texts
        hold = [_held_out(str(k)) for k in keys]
        rng = np.random.default_rng(seed)
        cal = _lists([t for t, h in zip(texts, hold) if not h], [y for y, h in zip(labels, hold) if not h],
                     label_set, rng, repeat=2)
        held = _lists([t for t, h in zip(texts, hold) if h], [y for y, h in zip(labels, hold) if h], label_set,
                      np.random.default_rng(seed + 1))
        if len(cal) < 20:
            raise ValueError("recalibrate needs at least 10 examples outside the held-out 20%")
        data = []
        for t, canon, gold in cal:
            d, _ = self._rt.depths(t, canon)
            data.append((d, canon.index(gold) if gold is not None else len(canon)))
        log_t, b = _fit_two_numbers(data, self._log_temperature, self._none_level)
        _warn_small_held_out(len(held), "recalibrate")
        t_ratio = float(np.exp(log_t - self._log_temperature))
        if not 0.5 <= t_ratio <= 2.0 or abs(b - self._none_level) > 2.0:
            warnings.warn(
                f"recalibrate moved the temperature from {np.exp(self._log_temperature):.3f} to {np.exp(log_t):.3f} "
                f"and the none level from {self._none_level:.2f} to {b:.2f}. A change this large usually means the "
                f"examples are few or unlike the texts you will decide; compare report['before'] and "
                f"report['after'] before using the new model.", UserWarning, stacklevel=2)
        new = self._derived(log_t=log_t, none_level=b)
        report = {"before": _evaluate(self, held), "after": _evaluate(new, held),
                  "temperature": {"before": float(np.exp(self._log_temperature)), "after": float(np.exp(log_t))},
                  "none_level": {"before": self._none_level, "after": b}, "calibration_lists": len(cal)}
        new.report = report
        return new, report

    def save(self, path: str) -> None:
        """Write this model to a folder; load it later with ZeroPoint(path)."""
        from .fit import write_model_dir
        with open(os.path.join(self._rt.model_dir, "head.json"), encoding="utf-8") as f:
            head = json.load(f)
        head["log_temperature"] = float(self._log_temperature).hex()
        head["none_level"] = float(self._none_level).hex()
        manifest = dict(self._rt.manifest)
        if self.report is not None:
            manifest["report"] = self.report
        write_model_dir(self._rt.model_dir, path, None, head, manifest)

    def _derived(self, log_t: float, none_level: float) -> "ZeroPoint":
        new = object.__new__(ZeroPoint)
        new.__dict__.update(self.__dict__)
        new._log_temperature, new._none_level, new.report = float(log_t), float(none_level), None
        return new

    @classmethod
    def _from_fit(cls, base: "ZeroPoint", weights, w, bias, log_t, none_level, fit_info) -> "ZeroPoint":
        import hashlib
        import onnx
        from onnx import numpy_helper
        from .fit import write_model_dir
        m = onnx.load(os.path.join(base._rt.model_dir, "encoder.onnx"))
        for i, init in enumerate(m.graph.initializer):
            name = init.name[:-6] if init.name.endswith("__fp16") else init.name
            if name.startswith("enc.") and name[4:] in weights:
                a = weights[name[4:]]
                arr = a.astype(np.float16) if init.name.endswith("__fp16") else a.astype(np.float32)
                m.graph.initializer[i].CopyFrom(numpy_helper.from_array(arr, init.name))
        enc_bytes = m.SerializeToString()
        head = {"weight": [float(x).hex() for x in np.asarray(w, np.float32)],
                "bias": float(np.float32(bias)).hex(),
                "log_temperature": float(np.float32(log_t)).hex(), "none_level": float(np.float32(none_level)).hex()}
        manifest = dict(base._rt.manifest)
        manifest["derived_from"] = base.model_id
        manifest["fit"] = fit_info
        manifest["source_checkpoint_sha256"] = hashlib.sha256(enc_bytes + json.dumps(head).encode()).hexdigest()
        out = tempfile.mkdtemp(prefix="zeropoint-fit-")
        try:
            write_model_dir(base._rt.model_dir, out, enc_bytes, head, manifest)
            new = cls(out, threads=base.threads)
        except BaseException:
            shutil.rmtree(out, ignore_errors=True)
            raise
        # The folder lives as long as any model that reads it (save() copies from it, and models made from this one
        # with recalibrate() share its runtime) and is deleted with the last of them.
        weakref.finalize(new._rt, shutil.rmtree, out, True)
        return new

    def _check(self, text, options) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("empty text")
        if isinstance(options, str) or not isinstance(options, Sequence):
            raise TypeError("options must be a list of strings")
        options = list(options)
        if not options:
            raise ValueError("options is empty: give at least one option")
        for o in options:
            if not isinstance(o, str):
                raise TypeError(f"every option must be a string, got {type(o).__name__}")
        for s, what in [(text, "text")] + [(o, f"option {o!r}") for o in options]:
            share = _non_latin_share(s)
            if share > NON_LATIN_LIMIT:
                raise UnsupportedLanguage(
                    f"the {what} is not in the Latin script ({share:.0%} of its letters are not Latin). "
                    f"Supported languages: {', '.join(self.languages)}.")
        return options

    def __repr__(self) -> str:
        return f"ZeroPoint(model={self.model_id}, languages={len(self.languages)}, threads={self.threads})"


def _fit_two_numbers(data, log_t: float, b: float, iters: int = 300):
    """Minimise the negative log-likelihood over (log temperature, none level) by gradient descent with a
    backtracking line search, in float64. Deterministic: no randomness, fixed number of iterations at most."""
    def nll_grad(lt, bb):
        t = np.exp(lt)
        tot, g_lt, g_b = 0.0, 0.0, 0.0
        for d, gold in data:
            x = d / t
            inside = (x > 1e-6) & (x < 60.0)
            xc = np.clip(x, 1e-6, 60.0)
            lw = xc + np.log(-np.expm1(-xc))
            lg = np.concatenate([lw, [bb]])
            m = lg.max()
            p = np.exp(lg - m)
            z = p.sum()
            p /= z
            tot += (m + np.log(z)) - lg[gold]
            dl = p.copy()
            dl[gold] -= 1.0
            dlw_dx = 1.0 / (-np.expm1(-xc))
            g_lt += float(np.sum(dl[:-1] * dlw_dx * (-xc) * inside))
            g_b += float(dl[-1])
        n = len(data)
        return tot / n, np.array([g_lt / n, g_b / n])
    x = np.array([log_t, b], dtype=np.float64)
    f, g = nll_grad(*x)
    for _ in range(iters):
        step = 1.0
        while step > 1e-8:
            cand = x - step * g
            fc, gc = nll_grad(*cand)
            if fc <= f - 1e-4 * step * float(g @ g):
                break
            step *= 0.5
        if step <= 1e-8 or abs(f - fc) < 1e-12:
            break
        x, f, g = cand, fc, gc
    return float(x[0]), float(x[1])
