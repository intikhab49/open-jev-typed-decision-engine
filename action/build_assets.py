"""Build the release assets the GitHub Action downloads. Maintainer-only.

  python action/build_assets.py --ckpt jevlite.pt --out dist

Needs the training stack (torch, transformers, scikit-learn, onnx). The Action
itself never does: it runs these files with onnxruntime + numpy on the runner.

Writes to --out:
  jevlite.onnx         fine-tuned model -> raw per-label logits (single file)
  encoder.onnx         frozen ModernBERT-base -> mean-pooled state embedding
  tokenizer.json       tokenizer with the <<q>>/<<l>> markers (fine-tuned model)
  tokenizer-base.json  stock tokenizer (frozen encoder)
  probes.json          one logistic regression per built-in question, all of train
  presets.json         the 20 built-in question definitions, keyed by workflow
  config.json          lengths, token ids, per-type temperatures, blend w and T
and action/SHA256SUMS, which the Action checks every download against.

The probes are fitted exactly as 07_ensemble.py fits the ones it scores on test:
torch embeddings, all 1200 training cases. The blend weight and sharpening
temperature default to the values 07_ensemble.py chose on validation (README).

--ckpt also takes the fp16 copy published on the model release; floating
weights are cast back to fp32 before export, and verify.py measures the result.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

from td_data import load_split, TypedDecisions, collate
from model import JevLite, require_ckpt
from probe import embed_states, fit_probes
from typed_schema import TYPES

ASSETS = ["jevlite.onnx", "encoder.onnx", "tokenizer.json", "tokenizer-base.json",
          "probes.json", "presets.json", "config.json"]


class Logits(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, label_pos):
        return self.m.logits(input_ids, attention_mask, label_pos)


class PooledEncoder(torch.nn.Module):
    """Same pooling as probe.embed_states, inside the graph."""

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1)


def export_single_file(module, args, path, input_names, output_names, dynamic_axes):
    """torch writes weights to a .data sidecar; fold them back in so the release
    has one file per model. 600 MB is well under protobuf's 2 GB ceiling."""
    import onnx
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "m.onnx")
        torch.onnx.export(module, args, raw, input_names=input_names,
                          output_names=output_names, dynamic_axes=dynamic_axes,
                          opset_version=18)
        onnx.save(onnx.load(raw), path, save_as_external_data=False)
    print(f"  {os.path.basename(path)}: {os.path.getsize(path) / 1e6:.0f} MB")


def parity(path, feeds, ref, what):
    import onnxruntime as ort
    got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(
        None, {k: v.numpy() for k, v in feeds.items()})[0]
    diff = float(np.abs(ref - got).max())
    print(f"  {what} parity vs pytorch: max abs diff {diff:.2e}")
    if diff > 1e-3:
        sys.exit(f"{what}: ONNX diverges from pytorch - do not ship this")


def probe_to_json(p):
    if isinstance(p, str):
        return {"constant": p}
    assert isinstance(p, LogisticRegression)
    return {"classes": [str(c) for c in p.classes_],
            "coef": p.coef_.tolist(), "intercept": p.intercept_.tolist()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "jevlite.pt"))
    ap.add_argument("--blend-w", type=float, default=0.60)
    ap.add_argument("--sharpen-t", type=float, default=0.35)
    ap.add_argument("--out", default=os.path.join(ROOT, "dist"))
    ap.add_argument("--probe-max-len", type=int, default=512)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ck = require_ckpt(a.ckpt)
    model = JevLite(ck["encoder"])
    model.load_state_dict({k: v.float() if v.is_floating_point() else v
                           for k, v in ck["state_dict"].items()})
    model.eval()
    train = load_split("train")
    test = load_split("test")

    print("exporting the fine-tuned model...")
    ds = TypedDecisions(test[:1], model.tok, ck["max_len"], model.qid, model.lid)
    b = collate([ds[0]], model.tok.pad_token_id)
    feeds = {k: b[k] for k in ("input_ids", "attention_mask", "label_pos")}
    path = os.path.join(a.out, "jevlite.onnx")
    export_single_file(Logits(model), tuple(feeds.values()), path, list(feeds),
                       ["label_logits"],
                       {"input_ids": {0: "batch", 1: "seq"},
                        "attention_mask": {0: "batch", 1: "seq"},
                        "label_pos": {0: "batch", 1: "labels"},
                        "label_logits": {0: "batch", 1: "labels"}})
    with torch.no_grad():
        parity(path, feeds, Logits(model)(**feeds).numpy(), "fine-tuned")

    print("exporting the frozen encoder...")
    base_tok = AutoTokenizer.from_pretrained(ck["encoder"])
    enc = PooledEncoder(AutoModel.from_pretrained(ck["encoder"]).eval())
    bt = base_tok(["state text for export"], return_tensors="pt")
    efeeds = {"input_ids": bt["input_ids"], "attention_mask": bt["attention_mask"]}
    path = os.path.join(a.out, "encoder.onnx")
    export_single_file(enc, tuple(efeeds.values()), path, list(efeeds), ["state_embedding"],
                       {"input_ids": {0: "batch", 1: "seq"},
                        "attention_mask": {0: "batch", 1: "seq"},
                        "state_embedding": {0: "batch"}})
    with torch.no_grad():
        parity(path, efeeds, enc(**efeeds).numpy(), "encoder")

    print("tokenizers...")
    with tempfile.TemporaryDirectory() as tmp:
        model.tok.save_pretrained(tmp)
        shutil.copy(os.path.join(tmp, "tokenizer.json"),
                    os.path.join(a.out, "tokenizer.json"))
    with tempfile.TemporaryDirectory() as tmp:
        base_tok.save_pretrained(tmp)
        shutil.copy(os.path.join(tmp, "tokenizer.json"),
                    os.path.join(a.out, "tokenizer-base.json"))

    print(f"fitting probes on all {len(train)} training cases...")
    X = embed_states(train, ck["encoder"], a.probe_max_len, device="cpu")
    probes = fit_probes(train, X)
    json.dump({f"{wf}/{qn}": probe_to_json(p) for (wf, qn), p in probes.items()},
              open(os.path.join(a.out, "probes.json"), "w"))
    print(f"  {len(probes)} probes")

    presets = {}
    for r in train:
        for qn, q in r["questions"].items():
            prev = presets.setdefault(r["workflow"], {}).setdefault(qn, q)
            if prev != q:
                sys.exit(f"{r['workflow']}/{qn} has two definitions - presets ambiguous")
    json.dump(presets, open(os.path.join(a.out, "presets.json"), "w"), indent=1,
              sort_keys=True)

    temps = model.log_temp.exp().tolist()
    tok = model.tok
    json.dump({
        "encoder": ck["encoder"],
        "max_len": ck["max_len"],
        "probe_max_len": a.probe_max_len,
        "max_questions_per_pass": 8,
        "temperatures": dict(zip(TYPES, temps)),
        "blend_w": a.blend_w,
        "sharpen_t": a.sharpen_t,
        "ids": {"cls": tok.cls_token_id, "sep": tok.sep_token_id,
                "pad": tok.pad_token_id, "q": model.qid, "l": model.lid,
                "base_cls": base_tok.cls_token_id, "base_sep": base_tok.sep_token_id},
    }, open(os.path.join(a.out, "config.json"), "w"), indent=1)

    lines = []
    for name in ASSETS:
        h = hashlib.sha256()
        with open(os.path.join(a.out, name), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {name}")
    with open(os.path.join(ROOT, "action", "SHA256SUMS"), "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote action/SHA256SUMS - commit it with the release tag it describes")
