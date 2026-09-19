"""Step 6 - export to ONNX so it runs anywhere for free.

ONNX Runtime Web gives you the same model in a browser tab over WebGPU, with
WASM as the fallback. That is the one advantage an API model structurally cannot
match: no network hop, no key, no per-call cost, and the data never leaves the
machine. Measured reference for a 150M encoder at this shape is ~35 ms p50 on
single-thread WASM, and less on WebGPU.

  python 06_export_onnx.py --quantize
"""
import argparse
import os
import torch
from td_data import load_split, TypedDecisions, collate
from model import JevLite, require_ckpt


class ExportWrapper(torch.nn.Module):
    """Raw per-label logits only. The grouped softmax is a dozen lines of JS and
    keeping it out of the graph avoids exporting a Python loop over questions."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, label_pos):
        return self.m.logits(input_ids, attention_mask, label_pos)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="jevlite.pt")
    p.add_argument("--out", default="jevlite.onnx")
    p.add_argument("--opset", type=int, default=18,
                   help="18 is the floor: torch emits opset-18 ops "
                        "(Split.num_outputs) and a lower request "
                        "silently produces a graph ORT refuses to load")
    p.add_argument("--quantize", action="store_true", help="int8 dynamic, ~4x smaller")
    a = p.parse_args()

    ck = require_ckpt(a.ckpt)
    model = JevLite(ck["encoder"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    row = load_split("test", limit=1)[0]
    ds = TypedDecisions([row], model.tok, ck["max_len"], model.qid, model.lid)
    b = collate([ds[0]], model.tok.pad_token_id)

    torch.onnx.export(
        ExportWrapper(model),
        (b["input_ids"], b["attention_mask"], b["label_pos"]),
        a.out,
        input_names=["input_ids", "attention_mask", "label_pos"],
        output_names=["label_logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "label_pos": {0: "batch", 1: "labels"},
                      "label_logits": {0: "batch", 1: "labels"}},
        opset_version=a.opset,
    )
    sidecar = a.out + ".data"
    extra = f" + {os.path.getsize(sidecar)/1e6:.0f} MB {sidecar}" if \
        os.path.exists(sidecar) else ""
    print(f"exported -> {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB){extra}")
    if extra:
        print("  note: weights live in the sidecar. Serve BOTH files from the same")
        print("  directory, or ORT Web will load a graph with no weights in it.")

    # Parity check. An export that silently diverges is worse than no export -
    # you would only find out from wrong decisions in production.
    import numpy as np
    import onnxruntime as ort
    with torch.no_grad():
        ref = ExportWrapper(model)(b["input_ids"], b["attention_mask"],
                                   b["label_pos"]).numpy()
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    got = sess.run(None, {k: b[k].numpy() for k in
                          ("input_ids", "attention_mask", "label_pos")})[0]
    diff = float(np.abs(ref - got).max())
    print(f"  parity vs pytorch: max abs diff {diff:.2e} "
          f"{'OK' if diff < 1e-3 else 'FAILED - do not ship this'}")

    if a.quantize:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        q = a.out.replace(".onnx", ".int8.onnx")
        quantize_dynamic(a.out, q, weight_type=QuantType.QUInt8)
        qsize = os.path.getsize(q) + (os.path.getsize(q + ".data")
                                      if os.path.exists(q + ".data") else 0)
        print(f"quantized -> {q} ({qsize/1e6:.0f} MB total)")
        qs = ort.InferenceSession(q, providers=["CPUExecutionProvider"])
        qgot = qs.run(None, {k: b[k].numpy() for k in
                             ("input_ids", "attention_mask", "label_pos")})[0]
        print(f"  int8 vs fp32: max abs diff {np.abs(got - qgot).max():.2e}")
        print("  int8 shifts logits. Re-run 03_calibrate.py against the quantized")
        print("  model before trusting its confidences.")

    temps = ck.get("temperatures")
    print("\nIn the browser, load this with onnxruntime-web, tokenize with")
    print("@huggingface/transformers, then for each question softmax its own")
    print(f"label logits after dividing by its type temperature: {temps}")
