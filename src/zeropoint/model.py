"""ZeroPoint: choose among options, or say that none of them fits."""

from __future__ import annotations

import json
import os
import time
import dataclasses
from dataclasses import dataclass, field
from importlib import resources
from typing import Sequence

import numpy as np

from . import engine
from ._version import FORMAT
from .encoder import EncoderSpec, get_encoder

DEFAULT_PRECISION = "int8"
DEFAULT_LANDSCAPE = "everyday"


@dataclass(frozen=True)
class Landscape:
    """A calibrated energy landscape that ships with the package."""

    name: str
    options: int
    description: str
    examples: tuple[str, ...]
    source: str


LANDSCAPES: dict[str, Landscape] = {
    "everyday": Landscape(
        "everyday", 150, "everyday requests to an assistant: banking, travel, home, work, small talk",
        ("transfer", "book flight", "alarm", "weather"),
        "calibrated on CLINC150 (Larson et al., 2019): 15,000 requests written by people"),
    "support": Landscape(
        "support", 10, "customer support tickets, routed to a queue",
        ("Billing and Payments", "Technical Support", "Returns and Exchanges", "Human Resources"),
        "calibrated on 20,000 public support tickets (Tobi-Bueck/customer-support-tickets)"),
}


class _LandscapeList(list):
    def __repr__(self) -> str:
        rows = []
        for ls in self:
            default = "   (default)" if ls.name == DEFAULT_LANDSCAPE else ""
            rows.append(f"{ls.name:<10} {ls.options:>4} options   {ls.description}{default}")
        return "\n".join(rows)


def landscapes() -> list[Landscape]:
    """The calibrated landscapes that ship with ZeroPoint. Load one with ``ZeroPoint.load(name)``."""
    return _LandscapeList(LANDSCAPES.values())


def _bar(p: float, width: int = 20) -> str:
    cells = p * width
    full = int(cells)
    part = " ▏▎▍▌▋▊▉"[int((cells - full) * 8)]
    return ("█" * full + (part if full < width else "")).rstrip()


@dataclass(frozen=True)
class Decision:
    """The outcome for one text.

    choice        the chosen option, or None when "none of them" is the most probable outcome
    confidence    probability of `choice` (of "none" when choice is None)
    p_none        probability that none of the options is right
    probabilities probability of each option; together with p_none they sum to 1
    energies      energy of the text in each option's well; 0 means "not in this well at all"
    beyond_edge   how far the text lies outside the data the model was calibrated on (<= 0: inside)
    latency_ms    time to decide, in milliseconds (in decide_many, the batch time divided by its texts)
    """

    choice: str | None
    confidence: float
    p_none: float
    probabilities: dict[str, float] = field(repr=False)
    energies: dict[str, float] = field(repr=False)
    beyond_edge: float = field(repr=False, default=0.0)
    latency_ms: float = field(repr=False, default=0.0)

    def __repr__(self) -> str:
        """A small table: each option with its probability, then "none", then the decision.

        With more than 6 options, the 5 most probable are shown.
        """
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
    """An energy-based decision engine on top of a frozen text encoder.

    Load a calibrated landscape with ``ZeroPoint.load()`` (see ``zeropoint.landscapes()``) or calibrate your own with
    ``ZeroPoint.fit(texts, labels)``, then call ``decide``.
    """

    def __init__(self, weights: dict[str, np.ndarray], config: dict, precision: str | None = None,
                 providers: list[str] | None = None, threads: int | None = None):
        self.weights = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        self.config = config
        enc = dict(config["encoder"])
        if precision is not None:
            enc["precision"] = precision
        self.encoder_spec = EncoderSpec(**enc)
        self._providers, self._threads = providers, threads
        self.labels: list[str] = list(config["labels"])
        self.descriptions: list[str] = list(config["descriptions"])
        self.temperature = float(config["temperature"])
        self.kappa = float(config["kappa"])
        self._known = {lab: i for i, lab in enumerate(self.labels)}
        self._wells_cache: dict[str, tuple[np.ndarray, float, float]] = {}
        centre, depth, width2 = engine.wells(self.weights["option_emb"], self.weights)
        for i, lab in enumerate(self.labels):
            self._wells_cache[lab] = (centre[i], float(depth[i]), float(width2[i]))

    # ------------------------------------------------------------------ loading and saving

    @classmethod
    def load(cls, name_or_path: str = DEFAULT_LANDSCAPE, encoder: str = "small",
             precision: str | None = DEFAULT_PRECISION, providers: list[str] | None = None,
             threads: int | None = None) -> "ZeroPoint":
        """Load a calibrated energy landscape.

        name_or_path  a landscape that ships with the package ("everyday", the default, or "support"; see
                      ``zeropoint.landscapes()``), or the path to a file written by ``save``
        encoder       "small" (default, faster) or "base" (more accurate); shipped landscapes only
        precision     "int8" (default, fastest on CPU) or "fp32"; None keeps the precision it was calibrated with
        """
        if name_or_path in LANDSCAPES:
            path = str(resources.files("zeropoint") / "presets" / f"{name_or_path}-{encoder}.safetensors")
        elif os.path.exists(name_or_path):
            path = name_or_path
        else:
            raise ValueError(f"{name_or_path!r} is neither a file nor a shipped landscape. Available:\n"
                             f"{landscapes()!r}")
        from safetensors import safe_open

        try:
            with safe_open(path, framework="np") as f:
                config = json.loads((f.metadata() or {})["zeropoint"])
                weights = {k: f.get_tensor(k) for k in f.keys()}
        except Exception as e:  # damaged file, or not written by zeropoint
            raise ValueError(f"{path} is not a ZeroPoint model file ({type(e).__name__}).") from e
        if not isinstance(config, dict) or config.get("format", 1) > FORMAT:
            raise ValueError("This model was saved by a newer version of zeropoint, or is damaged. Please upgrade.")
        try:
            return cls(weights, config, precision=precision, providers=providers, threads=threads)
        except (KeyError, TypeError, ValueError, AttributeError, IndexError) as e:
            raise ValueError(f"{path} is not a valid ZeroPoint model ({type(e).__name__}: {e}).") from e

    def save(self, path: str) -> None:
        from safetensors.numpy import save_file

        save_file({k: np.ascontiguousarray(v) for k, v in self.weights.items()}, path,
                  metadata={"zeropoint": json.dumps(self.config)})

    @classmethod
    def fit(cls, texts: Sequence[str], labels: Sequence[str], **options) -> "ZeroPoint":
        """Calibrate a model on your own labelled texts. Needs ``pip install zeropoint[train]``.

        See ``zeropoint.calibrate.train`` for every option.
        """
        from .calibrate import train

        weights, config = train(texts, labels, **options)
        return cls(weights, config)

    # ------------------------------------------------------------------ deciding

    @property
    def encoder(self):
        return get_encoder(self.encoder_spec, providers=self._providers, threads=self._threads)

    def decide(self, text: str, options: Sequence[str] | None = None,
               min_confidence: float | None = None) -> Decision:
        """Choose one of `options` for `text`, or None.

        options         defaults to every label the model knows
        min_confidence  if set, `choice` is None unless the chosen option has at least this probability
        """
        return self.decide_many([text], options, min_confidence=min_confidence)[0]

    def decide_many(self, texts: Sequence[str], options=None, min_confidence: float | None = None,
                    batch_size: int | None = None) -> list[Decision]:
        """Decide for many texts at once.

        options         None (every known label), one list shared by all texts, or one list per text
        min_confidence  if set, `choice` is None unless the chosen option has at least this probability
        """
        texts = list(texts)
        if options is None:
            per_text = [self.labels] * len(texts)
        elif len(options) and not isinstance(options[0], str):
            per_text = [list(o) for o in options]
            if len(per_text) != len(texts):
                raise ValueError("Give one list of options per text, or a single shared list.")
        else:
            per_text = [list(options)] * len(texts)
        for o in per_text:
            if len(o) == 0 or len(set(o)) != len(o):
                raise ValueError("Each list of options must be non-empty and without repeats.")
        encoder = self.encoder  # loading the encoder is a one-time cost, kept out of the latency
        start = time.perf_counter()
        emb = encoder.encode(texts, "query", batch_size=batch_size)
        out = self._decide_embedded(emb, per_text, min_confidence)
        ms = (time.perf_counter() - start) * 1000 / max(len(texts), 1)
        return [dataclasses.replace(d, latency_ms=ms) for d in out]

    def _decide_embedded(self, emb: np.ndarray, per_text: list[list[str]],
                         min_confidence: float | None = None) -> list[Decision]:
        self._add_wells({o for opts in per_text for o in opts})
        w = self.weights
        z = engine.unit(engine.mlp(emb, w, "f"))
        prec = self._edge_precision()
        q = self.config["edge"][prec]
        edge = engine.distance_beyond_edge(emb, w[f"edge.{prec}.means"], w[f"edge.{prec}.precision"],
                                           q["q50"], q["q99"])
        log_vacuum = self.kappa * np.maximum(edge, 0.0)
        out: list[Decision | None] = [None] * len(per_text)
        groups: dict[tuple, list[int]] = {}
        for i, o in enumerate(per_text):
            groups.setdefault(tuple(o), []).append(i)
        for opts, idx in groups.items():
            centre = np.stack([self._wells_cache[o][0] for o in opts])
            depth = np.array([self._wells_cache[o][1] for o in opts], dtype=np.float32)
            width2 = np.array([self._wells_cache[o][2] for o in opts], dtype=np.float32)
            d2 = ((z[idx][:, None, :] - centre[None]) ** 2).sum(-1)
            energy = -depth[None] * np.exp(-d2 / width2[None])
            p = engine.probabilities(energy, self.temperature, log_vacuum[idx])
            for r, i in enumerate(idx):
                k = int(p[r].argmax())
                choice = opts[k] if k < len(opts) else None
                if min_confidence is not None and p[r, k] < min_confidence:
                    choice = None
                out[i] = Decision(choice=choice, confidence=float(p[r, k]), p_none=float(p[r, -1]),
                                  probabilities={o: float(p[r, j]) for j, o in enumerate(opts)},
                                  energies={o: float(energy[r, j]) for j, o in enumerate(opts)},
                                  beyond_edge=float(edge[i]))
        return out  # type: ignore[return-value]

    def _edge_precision(self) -> str:
        """The edge measured with the precision the encoder runs at, or the calibration one."""
        edges = self.config["edge"]
        return self.encoder_spec.precision if self.encoder_spec.precision in edges else next(iter(edges))

    def _add_wells(self, options: set[str]) -> None:
        """Options the model has not seen get a well from their own text. Results are cached."""
        new = sorted(o for o in options if o not in self._wells_cache)
        if not new:
            return
        emb = self.encoder.encode(new, "passage")
        centre, depth, width2 = engine.wells(emb, self.weights)
        for i, o in enumerate(new):
            self._wells_cache[o] = (centre[i], float(depth[i]), float(width2[i]))

    def __repr__(self) -> str:
        enc = self.encoder_spec
        return (f"ZeroPoint({len(self.labels)} options, encoder={enc.repo.split('/')[-1]}, "
                f"precision={enc.precision})")
