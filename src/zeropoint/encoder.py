"""Frozen text encoder, run with ONNX Runtime.

ZeroPoint never trains the encoder. It reads its embeddings, so inference needs no PyTorch:
ONNX Runtime, a tokenizer and NumPy are enough.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# Encoders ZeroPoint knows by a short name. Revisions are pinned so that an embedding computed
# today is the embedding computed next year.
ENCODERS = {
    "small": ("intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3"),
    "base": ("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a"),
}

ONNX_FILES = {
    "int8": "onnx/model_qint8_avx512_vnni.onnx",
    "fp32": "onnx/model.onnx",
}


@dataclass(frozen=True)
class EncoderSpec:
    """Everything needed to rebuild the exact same encoder."""

    repo: str
    revision: str | None
    precision: str
    max_length: int = 512

    @classmethod
    def resolve(cls, encoder: str = "small", precision: str = "int8", max_length: int = 512) -> "EncoderSpec":
        if precision not in ONNX_FILES:
            raise ValueError(f"precision must be one of {sorted(ONNX_FILES)}, got {precision!r}")
        repo, revision = ENCODERS.get(encoder, (encoder, None))
        return cls(repo=repo, revision=revision, precision=precision, max_length=max_length)

    def to_dict(self) -> dict:
        return {"repo": self.repo, "revision": self.revision, "precision": self.precision, "max_length": self.max_length}


class Encoder:
    """Turns texts into unit-length embeddings with an E5-style ONNX model.

    E5 models expect a prefix: ``query: `` for the text to decide on, ``passage: `` for the options.
    """

    def __init__(self, spec: EncoderSpec, providers: list[str] | None = None, threads: int | None = None):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.spec = spec
        fetch = lambda f: hf_hub_download(spec.repo, f, revision=spec.revision)  # noqa: E731
        self.tokenizer = Tokenizer.from_file(fetch("onnx/tokenizer.json"))
        self.tokenizer.enable_truncation(spec.max_length)
        self.tokenizer.no_padding()

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            options.intra_op_num_threads = threads
        model_path = fetch(ONNX_FILES[spec.precision])
        if spec.precision == "fp32" and _has_file(spec, ONNX_FILES["fp32"] + "_data"):
            fetch(ONNX_FILES["fp32"] + "_data")  # external weights, when the repo splits them out
        if providers is None:  # CUDA when present, else CPU; other providers only when asked for
            gpu = "CUDAExecutionProvider" in ort.get_available_providers() and spec.precision == "fp32"
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, options, providers=providers)
        self._inputs = {i.name for i in self.session.get_inputs()}
        self.encode(["warm up"])  # the first run initialises ONNX Runtime; keep it out of the first decision

    @property
    def dim(self) -> int:
        return int(self.session.get_outputs()[0].shape[-1])

    @property
    def on_cpu(self) -> bool:
        return self.session.get_providers()[0] == "CPUExecutionProvider"

    def encode(self, texts: list[str], kind: str = "query", batch_size: int | None = None) -> np.ndarray:
        """Embeddings of shape (len(texts), dim), float32, unit length.

        batch_size defaults to 1 for the 8-bit encoder, 4 on CPU and 32 on GPU. The 8-bit encoder quantizes
        each batch as a whole, so a text's embedding would shift slightly with its neighbours in the batch;
        one text per run keeps every decision identical however it is called.
        """
        if batch_size is None:
            batch_size = 1 if self.spec.precision == "int8" else (4 if self.on_cpu else 32)
        if kind not in ("query", "passage"):
            raise ValueError("kind must be 'query' or 'passage'")
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        encs = self.tokenizer.encode_batch([f"{kind}: {t}" for t in texts])
        order = np.argsort([len(e.ids) for e in encs])  # similar lengths together: less padding, faster
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            width = max(len(encs[i].ids) for i in idx)
            ids = np.zeros((len(idx), width), dtype=np.int64)
            mask = np.zeros((len(idx), width), dtype=np.int64)
            for r, i in enumerate(idx):
                n = len(encs[i].ids)
                ids[r, :n], mask[r, :n] = encs[i].ids, 1
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feeds)[0]
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(1) / np.maximum(m.sum(1), 1e-9)
            out[idx] = pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        return out


def _has_file(spec: EncoderSpec, filename: str) -> bool:
    """Whether the model repository contains `filename`. Without network access, assume it does not."""
    from huggingface_hub import file_exists

    try:
        return file_exists(spec.repo, filename, revision=spec.revision)
    except Exception:  # offline or Hub unreachable: the main file came from the local cache, so use it alone
        return False


_CACHE: dict = {}


def get_encoder(spec: EncoderSpec, providers: list[str] | None = None, threads: int | None = None) -> Encoder:
    """One encoder per spec and process: loading it is the slow part."""
    key = (spec, tuple(providers or ()), threads, os.getpid())
    if key not in _CACHE:
        _CACHE[key] = Encoder(spec, providers=providers, threads=threads)
    return _CACHE[key]
