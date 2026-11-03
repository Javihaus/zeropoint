"""The inference core: reads a text and a list of options exactly as the model was trained, runs the encoder with
onnxruntime, and turns the depths of the wells into one Boltzmann distribution with "none" at energy zero.

Only numpy, onnxruntime and tokenizers. No network, no torch.

Every step is fixed so the same input gives the same bits:
  * text and options are normalised to Unicode NFC;
  * options are read in one canonical order (UTF-8 bytes of the NFC form), whatever order they arrive in;
  * every option gets the same position ids, starting right after the text, so the encoder cannot see order;
  * one case at a time, never padded together with other cases;
  * pooling, head and Boltzmann weights in float64 with numpy.
"""
from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


class ModelFilesError(RuntimeError):
    """The model files are missing or do not match the hashes in manifest.json."""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_order(options: list[str]) -> list[int]:
    """Indices of the options in canonical order. Raises ValueError on an empty or duplicate option."""
    keys = [unicodedata.normalize("NFC", o) for o in options]
    for o, k in zip(options, keys):
        if not k.strip():
            raise ValueError(f"empty option: {o!r}")
    seen: dict[str, str] = {}
    for o, k in zip(options, keys):
        if k in seen:
            raise ValueError(f"duplicate options: {seen[k]!r} and {o!r} are the same text")
        seen[k] = o
    return sorted(range(len(options)), key=lambda i: keys[i].encode("utf-8"))


@dataclass
class Reading:
    """What the engine actually read, so nothing is cut or changed silently."""
    text_pieces: int
    text_truncated: bool
    options_truncated: list[str] = field(default_factory=list)
    blocks: int = 1


class Runtime:
    def __init__(self, model_dir: str = MODEL_DIR, threads: int = 2, verify: bool = True):
        mpath = os.path.join(model_dir, "manifest.json")
        if not os.path.exists(mpath):
            raise ModelFilesError(f"no manifest.json in {model_dir}")
        self.manifest = json.load(open(mpath, encoding="utf-8"))
        if verify:
            for name, digest in self.manifest["files"].items():
                p = os.path.join(model_dir, name)
                if not os.path.exists(p):
                    raise ModelFilesError(f"missing model file {name}")
                if _sha256(p) != digest:
                    raise ModelFilesError(f"{name} does not match its sha256 in manifest.json")
        r = self.manifest["reading"]
        self.prefix = r["text_prefix"]
        self.text_max = int(r["text_max_pieces"])
        self.opt_max = int(r["option_max_pieces"])
        self.block = int(r["options_per_block"])
        self.pos0 = int(r["position_offset"])
        sp = self.manifest["special_ids"]
        self.cls, self.sep, self.unk = sp["cls"], sp["sep"], sp["unk"]

        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.no_truncation()
        self.tok.no_padding()
        self.vocab = np.load(os.path.join(model_dir, "vocab_int8.npy"))
        self.scale = np.load(os.path.join(model_dir, "vocab_scale.npy"))
        head = json.load(open(os.path.join(model_dir, "head.json"), encoding="utf-8"))
        self.w = np.array([float.fromhex(x) for x in head["weight"]], dtype=np.float64)
        self.bias = float.fromhex(head["bias"])
        self.temperature = float(np.exp(float.fromhex(head["log_temperature"])))
        self.none_level = float.fromhex(head["none_level"])

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        self.session = ort.InferenceSession(os.path.join(model_dir, "encoder.onnx"), so,
                                            providers=["CPUExecutionProvider"])
        self.threads = int(threads)

    # ---------------------------------------------------------------- reading

    def _pieces(self, s: str) -> list[int]:
        return self.tok.encode(unicodedata.normalize("NFC", s), add_special_tokens=False).ids

    def _embed(self, ids: list[int]) -> np.ndarray:
        idx = np.asarray(ids, dtype=np.int64)
        return (self.vocab[idx].astype(np.float32) * self.scale[idx][:, None])[None]

    def depths(self, text: str, canon: list[str]) -> tuple[np.ndarray, Reading]:
        """Depth of the well of every option (canonical order), float64."""
        t = self._pieces(self.prefix + text)
        reading = Reading(text_pieces=min(len(t), self.text_max), text_truncated=len(t) > self.text_max)
        t = [self.cls] + t[:self.text_max] + [self.sep]
        opt_ids = []
        for o in canon:
            oi = self._pieces(o)
            if len(oi) > self.opt_max:
                reading.options_truncated.append(o)
            opt_ids.append((oi[:self.opt_max] or [self.unk]) + [self.sep])
        out = np.empty(len(canon), dtype=np.float64)
        reading.blocks = (len(canon) + self.block - 1) // self.block
        for s in range(0, len(canon), self.block):
            ids, pos, spans = list(t), list(range(len(t))), []
            for oi in opt_ids[s:s + self.block]:
                spans.append((len(ids), len(ids) + len(oi) - 1))
                ids += oi
                pos += range(len(t), len(t) + len(oi))
            h = self.session.run(None, {"inputs_embeds": self._embed(ids),
                                        "position_ids": (np.asarray(pos, dtype=np.int64) + self.pos0)[None]})[0][0]
            for j, (a, b) in enumerate(spans):
                z = float(h[a:b].astype(np.float64).mean(axis=0) @ self.w) + self.bias
                out[s + j] = np.logaddexp(0.0, z)          # softplus: depth >= 0
        return out, reading

    # ---------------------------------------------------------------- Boltzmann distribution

    def distribution(self, depths: np.ndarray) -> np.ndarray:
        """Probabilities of every option and, last, of "none" (energy zero)."""
        x = np.clip(depths / self.temperature, 1e-6, 60.0)
        logw = x + np.log(-np.expm1(-x))                    # log(exp(x) - 1), stable
        lg = np.concatenate([logw, [self.none_level]])
        p = np.exp(lg - lg.max())
        return p / p.sum()
