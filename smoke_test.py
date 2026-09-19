"""Cheap correctness checks that do not need a GPU or a trained checkpoint.

  python smoke_test.py            # tokenization + batching, tokenizer download only
  python smoke_test.py --all      # validate EVERY row of train+test (no model)
  python smoke_test.py --full     # also runs an untrained forward pass (~600MB)

What it verifies: that every marker position really lands on a <<l>> token, that
grouped softmax sums to 1 inside each question and stays 0 on padding, that gold
targets line up with the labels they are supposed to score, and that the metrics
agree with hand-computed values.
"""
import argparse
import torch
from transformers import AutoTokenizer
from td_data import (load_split, TypedDecisions, collate, add_markers, encode,
                     pad_cat, L_TOKEN)
from typed_schema import iter_labels
from model import grouped_softmax, DEFAULT_ENCODER
from metrics import ece, brier

MAX_LEN = 1024


def validate_all(tok, qid, lid):
    """Encode every row in the dataset and check the schema invariants.

    Worth its runtime: 10% of typed-decisions is bare `noul` questions carrying
    no `criteria` key at all, and that shape does not appear in the first rows
    of the first workflow - so a 6-row smoke test passes while a full training
    run dies on row 300. Any assumption about question shape gets checked here,
    against all of it, before a GPU is involved.
    """
    from typed_schema import TYPES
    n_q = n_bad = 0
    for split in ("train", "test"):
        rows = load_split(split)
        for r in rows:
            enc = encode(r, tok, MAX_LEN, qid, lid)
            b = collate([enc], tok.pad_token_id)
            got = b["input_ids"].gather(1, b["label_pos"])[b["label_mask"]]
            assert (got == lid).all(), f"{r['id']}: marker misalignment"
            for gi, qname in enumerate(enc["qnames"]):
                q = r["questions"][qname]
                assert q["type"] in TYPES, f"{r['id']}/{qname}: type {q['type']}"
                labels = set(enc["labels"][gi])
                gold_keys = set(r["gold"][qname]["probabilities"])
                if labels != gold_keys:
                    print(f"  ! {r['id']}/{qname}: labels {sorted(labels)} "
                          f"!= gold keys {sorted(gold_keys)}")
                    n_bad += 1
                n_q += 1
    print(f"ok  encoded every row; {n_q} questions checked, {n_bad} label/gold "
          f"mismatches")
    assert n_bad == 0, "gold distributions do not key on the labels we score"


def main(full, check_all=False):
    rows = load_split("test", limit=6)
    print(f"loaded {len(rows)} rows  ({rows[0]['id']})")

    tok = AutoTokenizer.from_pretrained(DEFAULT_ENCODER)
    _, qid, lid = add_markers(tok)
    ds = TypedDecisions(rows, tok, MAX_LEN, qid, lid)
    batch = collate([ds[i] for i in range(len(ds))], tok.pad_token_id)

    # 1. every recorded label position must be a marker token
    ids, pos, mask = batch["input_ids"], batch["label_pos"], batch["label_mask"]
    at_marker = ids.gather(1, pos)[mask]
    assert (at_marker == lid).all(), "label_pos does not point at <<l>> markers"
    print(f"ok  {mask.sum().item()} label positions all land on {L_TOKEN}")

    # 2. label count must match the schema
    expected = sum(len(iter_labels(r["questions"][q]))
                   for r in rows for q in sorted(r["questions"]))
    assert mask.sum().item() == expected, f"{mask.sum().item()} != {expected}"
    print(f"ok  label count matches schema ({expected})")

    # 3. gold targets sum to 1 within each question, 0 on padding
    Q = batch["qtype"].size(1)
    for q in range(Q):
        m = (batch["group"] == q) & mask
        if not m.any():
            continue
        s = (batch["target"] * m).sum(-1)[m.any(-1)]
        assert torch.allclose(s, torch.ones_like(s), atol=2e-2), f"q{q} targets sum {s}"
    assert (batch["target"][~mask] == 0).all()
    print("ok  consensus targets normalised per question, zero on padding")

    # 4. grouped softmax is a valid distribution per question
    logits = torch.randn_like(batch["target"])
    probs = grouped_softmax(logits, batch["group"], mask, Q)
    assert (probs[~mask] == 0).all(), "softmax leaked into padding"
    for q in range(Q):
        m = (batch["group"] == q) & mask
        if not m.any():
            continue
        s = (probs * m).sum(-1)[m.any(-1)]
        assert torch.allclose(s, torch.ones_like(s), atol=1e-5), f"q{q} probs sum {s}"
    print("ok  grouped softmax normalises within questions only")

    # 5. metrics sanity - a perfect predictor scores 0 on both
    assert abs(brier(batch["target"], batch["target"], mask)) < 1e-6
    assert ece(batch["target"], batch["target"], mask) < 1e-6
    print("ok  Brier and ECE are 0 for a perfect predictor")

    seq = batch["input_ids"].size(1)
    print(f"\nbatch: {ids.size(0)} cases, {seq} tokens, {mask.sum().item()} labels")

    if full:
        from model import JevLite, decision_loss, argmax_per_question
        m = JevLite(DEFAULT_ENCODER)
        m.eval()
        with torch.no_grad():
            _, probs = m(batch)
        loss, ce, br = decision_loss(probs, batch)
        pred, conf = argmax_per_question(probs, batch)
        gold, _ = argmax_per_question(batch["target"], batch)
        print(f"ok  untrained forward pass: loss {loss:.4f} (ce {ce:.4f} brier {br:.4f})")
        print(f"    pred {pred[0].tolist()}  gold {gold[0].tolist()}  "
              f"conf {[round(c,3) for c in conf[0].tolist()]}")
        assert probs.shape == batch["target"].shape
        print("ok  output shape matches target shape")

    # 6. batches drawn from different workflows disagree on label width
    mixed, seen = [], set()
    for r in load_split("test"):
        if r["workflow"] not in seen:
            seen.add(r["workflow"])
            mixed.append(r)
        if len(seen) >= 4:
            break
    widths, parts = set(), []
    for r in mixed:
        bb = collate([encode(r, tok, MAX_LEN, qid, lid)], tok.pad_token_id)
        widths.add(bb["target"].size(1))
        parts.append(bb)
    assert len(widths) > 1, ("workflows now agree on label count - this test no "
                             "longer guards anything, rewrite it")
    stacked = pad_cat([b["target"] for b in parts])
    smask = pad_cat([b["label_mask"] for b in parts], False)
    assert stacked.shape == smask.shape == (len(parts), max(widths))
    assert (stacked[~smask] == 0).all()
    print(f"ok  pad_cat joins ragged batches (widths {sorted(widths)}); "
          "a plain torch.cat raises here")

    if check_all:
        print("\nvalidating every row of train + test...")
        validate_all(tok, qid, lid)

    print("\nall checks passed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true")
    p.add_argument("--all", action="store_true")
    a = p.parse_args()
    main(a.full, a.all)
