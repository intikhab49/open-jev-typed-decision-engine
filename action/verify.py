"""Replay the typed-decisions test split through the Action's runtime.

  python action/verify.py --dir dist

The numbers the Action advertises come from here, not from the training run:
same 400 cases, but scored by onnxruntime + numpy exactly as the runner will
score them. Also asserts the runtime tokenizes byte-for-byte like td_data, so the
torch-free reimplementation cannot drift unnoticed. Needs the training stack
(torch, transformers) for those comparisons; the Action itself does not.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np
import torch
from transformers import AutoTokenizer

from jev_action import Engine
from td_data import load_split, encode, add_markers
from metrics import ece_confidence


def acc_ece(rows):
    correct = torch.tensor([c for c, _ in rows])
    conf = torch.tensor([p for _, p in rows])
    return correct.float().mean().item(), ece_confidence(conf, correct)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "dist"))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    eng = Engine(a.dir)
    test = load_split("test", limit=a.limit)
    cfg = eng.cfg

    print("tokenization parity against td_data...")
    hf = AutoTokenizer.from_pretrained(cfg["encoder"])
    _, qid, lid = add_markers(hf)
    base = AutoTokenizer.from_pretrained(cfg["encoder"])
    from typed_schema import state_to_text
    eng.embed(test[0]["state"])            # loads the frozen encoder's tokenizer
    for r in test:
        ref = encode(r, hf, cfg["max_len"], qid, lid, with_gold=False)
        ids, pos, _ = eng.encode(r["state"], r["questions"])
        assert ids == ref["input_ids"] and pos == ref["label_pos"], f"encode drift on {r['id']}"
        assert eng.presets[r["workflow"]] == r["questions"], f"preset drift on {r['id']}"
        text = state_to_text(r["state"])
        bref = base([text], truncation=True,
                    max_length=cfg["probe_max_len"])["input_ids"][0]
        body = eng._base_tok.encode(text, add_special_tokens=False).ids
        seq = [cfg["ids"]["base_cls"]] + body[:cfg["probe_max_len"] - 2] + \
            [cfg["ids"]["base_sep"]]
        assert seq == bref, f"probe tokenization drift on {r['id']}"
    print(f"  {len(test)} cases: fine-tuned and probe inputs identical")

    print(f"scoring {len(test)} cases on CPU...")
    ens, mod, t0 = [], [], time.time()
    for i, r in enumerate(test):
        out = eng.decide(r["state"], workflow=r["workflow"])["decisions"]
        md = eng.model_dists(r["state"], r["questions"])
        for qname in sorted(r["questions"]):
            gold = r["gold"][qname]["label"]
            d = out[qname]
            ens.append((d["answer"] == gold, d["confidence"]))
            labels, p = md[qname]
            k = int(np.argmax(p))
            mod.append((labels[k] == gold, float(p[k])))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(test)}  {(time.time() - t0) / (i + 1):.2f} s/case")
    secs = (time.time() - t0) / len(test)

    ea, ee = acc_ece(ens)
    ma, me = acc_ece(mod)
    print(f"\n=== Action runtime, test split ({len(test)} cases / {len(ens)} decisions) ===")
    print(f"  {'':<26}{'accuracy':>10}{'ECE':>9}")
    print(f"  {'built-in (ensemble)':<26}{ea:>10.4f}{ee:>9.4f}")
    print(f"  {'custom (model only)':<26}{ma:>10.4f}{me:>9.4f}")
    print(f"  {secs:.2f} s per case on this CPU (both paths)")
    full = a.limit is None and len(test) == 400
    if not full:
        print("  ! not the full 400-case split - do not quote these numbers")
    else:
        json.dump({"cases": len(test), "decisions": len(ens),
                   "ensemble": {"accuracy": ea, "ece": ee},
                   "model": {"accuracy": ma, "ece": me}},
                  open(os.path.join(HERE, "measured.json"), "w"), indent=1)
        print("  wrote action/measured.json")
