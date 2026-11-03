"""ZeroPoint 1.0: choose among a list of options, or say that none of them fits.

    from zeropoint import ZeroPoint
    zp = ZeroPoint()
    d = zp.decide("my card was swallowed by the cash machine",
                  ["block a card", "check the balance", "order a new card"])
    d.choice, d.confidence, d.p_none

One model reads the text and all the options together. Each option gets a well whose depth says how well it fits;
"none of them" is a state with energy zero. The probabilities of the options and of "none" come from one Boltzmann
distribution and add up to 1.

What the result never depends on: the order of the options, the number of CPU threads, what else is being decided
at the same time. The same text and options always give the same numbers.
"""
from __future__ import annotations

import time
import unicodedata
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
    depths             depth of each option's well (0 = the text does not belong there at all)
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
    """The decision engine. Load it once and reuse it: loading checks every model file against its sha256."""

    def __init__(self, model_dir: str = MODEL_DIR, threads: int = 2):
        if not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        self._rt = Runtime(model_dir, threads=threads)
        self.languages: tuple[str, ...] = tuple(self._rt.manifest["vocabulary"]["languages"])
        self.model_id: str = self._rt.manifest["source_checkpoint_sha256"][:12]

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
        p = self._rt.distribution(depths)
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
