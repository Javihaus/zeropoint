"""Turn a trained checkpoint into the model files shipped inside the zeropoint package.

    python tools/export_model.py --ckpt s0.pt --vocab vocab/ --out src/zeropoint/model/

Inputs
  --ckpt   the training checkpoint (state dict of the joint model: enc.*, head.*, log_t, b)
  --vocab  folder with the 25-language tokenizer.json, keep_ids.npy and report.json

Outputs (every file is listed with its sha256 in manifest.json, and the package checks them when it loads)
  encoder.onnx     the transformer layers. Weights stored in fp16 under their PyTorch names (no constant folding at
                   export, so fit() can find every weight by name); the Cast to fp32 in front of each one and the
                   transposes are folded when the session starts, so all arithmetic is fp32. Inputs: inputs_embeds (1, L, 384) and
                   position_ids (1, L); output: last hidden states (1, L, 384).
  vocab_int8.npy   the word-embedding table of the reduced vocabulary, int8, one row per piece
  vocab_scale.npy  one fp32 scale per row: embedding = int8 * scale
  head.json        the head (384 weights and a bias), log of the temperature and the "none" level, written as
                   exact hexadecimal floats so no decimal rounding happens anywhere
  tokenizer.json   the 25-language tokenizer
  manifest.json    reading rules, special ids, hashes, source checkpoint, versions

Needs torch, transformers and onnx (developer tools only; the package itself needs none of them).
"""
import argparse
import hashlib
import json
import os
import shutil

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper
from transformers import AutoConfig, AutoModel

ENCODER = "intfloat/multilingual-e5-small"
READING = {"text_prefix": "query: ", "text_max_pieces": 192, "option_max_pieces": 16, "options_per_block": 20,
           "unicode_normalization": "NFC", "position_offset": 0}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Encoder(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, inputs_embeds, position_ids):
        return self.enc(inputs_embeds=inputs_embeds, position_ids=position_ids).last_hidden_state


def fp16_weights(model_path):
    """Store every fp32 initializer of two or more dimensions in fp16, with a Cast back to fp32 in front of it."""
    m = onnx.load(model_path)
    g = m.graph
    keep, casts = [], []
    for init in g.initializer:
        a = numpy_helper.to_array(init)
        if a.dtype == np.float32 and a.ndim >= 2:
            name16 = init.name + "__fp16"
            keep.append(numpy_helper.from_array(a.astype(np.float16), name16))
            casts.append(helper.make_node("Cast", [name16], [init.name], to=onnx.TensorProto.FLOAT,
                                          name="cast_" + init.name.replace("/", "_")))
        else:
            keep.append(init)
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.checker.check_model(m)
    onnx.save(m, model_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    keep = np.load(os.path.join(args.vocab, "keep_ids.npy"))
    sd = torch.load(args.ckpt, map_location="cpu")

    cfg = AutoConfig.from_pretrained(ENCODER)
    cfg.vocab_size = len(keep)
    enc = AutoModel.from_config(cfg)
    enc.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("enc.")})
    enc.eval()

    # vocabulary table: int8 with one scale per row
    W = enc.embeddings.word_embeddings.weight.detach().numpy().astype(np.float32)
    scale = (np.abs(W).max(axis=1) / 127.0).astype(np.float32)
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(W / scale[:, None]), -127, 127).astype(np.int8)
    np.save(os.path.join(args.out, "vocab_int8.npy"), q)
    np.save(os.path.join(args.out, "vocab_scale.npy"), scale)

    # encoder layers
    onnx_path = os.path.join(args.out, "encoder.onnx")
    L = 8
    torch.onnx.export(Encoder(enc), (torch.zeros(1, L, cfg.hidden_size), torch.arange(L)[None]), onnx_path,
                      input_names=["inputs_embeds", "position_ids"], output_names=["hidden"],
                      dynamic_axes={"inputs_embeds": {1: "L"}, "position_ids": {1: "L"}, "hidden": {1: "L"}},
                      opset_version=17, dynamo=False, do_constant_folding=False)
    fp16_weights(onnx_path)

    # head, temperature and "none" level, exact
    head = {"weight": [float(x).hex() for x in sd["head.weight"].reshape(-1).float().numpy()],
            "bias": float(sd["head.bias"].reshape(-1)[0]).hex(),
            "log_temperature": float(sd["log_t"]).hex(),
            "none_level": float(sd["b"]).hex()}
    with open(os.path.join(args.out, "head.json"), "w") as f:
        json.dump(head, f, indent=1)
    shutil.copy(os.path.join(args.vocab, "tokenizer.json"), os.path.join(args.out, "tokenizer.json"))

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(args.out, "tokenizer.json"))
    special = {name: tk.token_to_id(tok) for name, tok in
               [("cls", "<s>"), ("sep", "</s>"), ("pad", "<pad>"), ("unk", "<unk>")]}
    files = ["encoder.onnx", "vocab_int8.npy", "vocab_scale.npy", "head.json", "tokenizer.json"]
    with open(os.path.join(args.vocab, "report.json")) as f:
        vocab_report = json.load(f)
    manifest = {
        "format": 1,
        "encoder": ENCODER,
        "hidden_size": cfg.hidden_size,
        "layers": cfg.num_hidden_layers,
        "attention_heads": cfg.num_attention_heads,
        "layer_norm_eps": cfg.layer_norm_eps,
        "vocabulary": {"pieces": int(len(keep)), "languages": sorted(
            vocab_report["per_language_needed"])},
        "reading": READING,
        "special_ids": special,
        "source_checkpoint_sha256": sha256(args.ckpt),
        "files": {f: sha256(os.path.join(args.out, f)) for f in files},
        "exported_with": {"torch": torch.__version__, "onnx": onnx.__version__},
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    total = sum(os.path.getsize(os.path.join(args.out, f)) for f in files)
    print(json.dumps({"MB": round(total / 1e6, 1), **manifest["files"]}, indent=1))


if __name__ == "__main__":
    main()
