"""Shared split + dataset. Split is deterministic so every script sees the same test set."""
import json, hashlib
import numpy as np
from schema import ACTIONS

A2I = {a: i for i, a in enumerate(ACTIONS)}

def load(path="labeled.jsonl"):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def split(rows, test_frac=0.15, val_frac=0.15):
    """Hash-based split on id - stable if you add more data later."""
    def bucket(r):
        h = int(hashlib.sha1(r["id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if h < test_frac: return "test"
        if h < test_frac + val_frac: return "val"
        return "train"
    out = {"train": [], "val": [], "test": []}
    for r in rows:
        out[bucket(r)].append(r)
    return out

def targets(rows):
    return (
        np.array([float(r["is_terminal_error"]) for r in rows], dtype="float32"),
        np.array([A2I[r["action_required"]] for r in rows], dtype="int64"),
        np.array([r["confidence_score"] - 1 for r in rows], dtype="int64"),
    )
