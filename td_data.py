"""Load LocalLLaMA/typed-decisions and turn it into label-position tensors.

One example = one state + every question about it. The encoder sees all of them
in a single sequence; each candidate label gets a marker token whose hidden state
becomes that label's logit.
"""
from __future__ import annotations
import hashlib
import json
import torch
from torch.utils.data import Dataset
from typed_schema import iter_labels, state_to_text, label_text, TYPE_ID

DATASET = "LocalLLaMA/typed-decisions"
Q_TOKEN, L_TOKEN = "<<q>>", "<<l>>"
MAX_Q = 8


def _parse(r):
    state = r["state"]
    return {
        "id": r["id"],
        "workflow": r["workflow"],
        "state": json.loads(state) if state.lstrip()[:1] in "{[" else state,
        "questions": json.loads(r["questions"]),
        "gold": json.loads(r["gold"]) if r.get("gold") else {},
    }


def _load_http(split, config, limit):
    """Fallback for environments without the `datasets` package (and for Colab
    cold starts, where the rows endpoint is faster than a parquet download)."""
    import time, urllib.error, urllib.parse, urllib.request

    def get(url, tries=5):
        # The anonymous rows endpoint rate-limits, and it does so exactly when
        # you are iterating quickly. Back off rather than dying mid-download.
        for i in range(tries):
            try:
                with urllib.request.urlopen(url, timeout=60) as f:
                    return json.load(f)
            except urllib.error.HTTPError as e:
                if e.code not in (429, 502, 503, 504) or i == tries - 1:
                    raise
                wait = 2 ** i * 5
                print(f"  HTTP {e.code} from the dataset server, retrying in {wait}s "
                      f"({i+1}/{tries-1})")
                time.sleep(wait)

    rows, offset = [], 0
    while limit is None or len(rows) < limit:
        n = 100 if limit is None else min(100, limit - len(rows))
        q = urllib.parse.urlencode({"dataset": DATASET, "config": config,
                                    "split": split, "offset": offset, "length": n})
        page = get(f"https://datasets-server.huggingface.co/rows?{q}")
        got = [_parse(x["row"]) for x in page["rows"]]
        rows += got
        offset += len(got)
        if not got or offset >= page.get("num_rows_total", offset):
            break
    return rows[:limit] if limit else rows


def load_split(split: str, config: str = "all", limit: int | None = None):
    try:
        from datasets import load_dataset
    except ImportError:
        return _load_http(split, config, limit)
    ds = load_dataset(DATASET, config, split=split)
    rows = []
    for r in ds:
        rows.append(_parse(r))
        if limit and len(rows) >= limit:
            break
    return rows


def add_markers(tok):
    """Register the two marker tokens. Caller must resize the model embeddings."""
    n = tok.add_tokens([Q_TOKEN, L_TOKEN], special_tokens=True)
    return n, tok.convert_tokens_to_ids(Q_TOKEN), tok.convert_tokens_to_ids(L_TOKEN)


def encode(row, tok, max_len, qid, lid, with_gold=True):
    """-> dict of python lists. Labels are laid out question by question."""
    qnames = sorted(row["questions"])[:MAX_Q]

    tail, label_pos, group, target, qtypes = [], [], [], [], []
    for gi, qname in enumerate(qnames):
        q = row["questions"][qname]
        qtypes.append(TYPE_ID[q["type"]])
        tail += [qid] + tok.encode(q["instructions"], add_special_tokens=False)
        gold = row.get("gold", {}).get(qname, {}) if with_gold else {}
        probs = gold.get("probabilities", {})
        for lab, desc in iter_labels(q):
            label_pos.append(len(tail))            # index of the <<l>> marker
            group.append(gi)
            target.append(float(probs.get(lab, 0.0)))
            tail += [lid] + tok.encode(label_text(lab, desc), add_special_tokens=False)

    # The questions are the part we must never truncate; the state absorbs the cut.
    budget = max_len - len(tail) - 2
    if budget < 32:
        raise ValueError(f"max_len={max_len} too small for {len(tail)} question tokens")
    head = [tok.cls_token_id] + tok.encode(
        state_to_text(row["state"]), add_special_tokens=False)[:budget]

    off = len(head)
    ids = head + tail + [tok.sep_token_id]
    return {
        "input_ids": ids,
        "label_pos": [p + off for p in label_pos],
        "group": group,
        "target": target,
        "qtype": qtypes,
        "qnames": qnames,
        "gold_label": [row["gold"][q]["label"] for q in qnames] if with_gold else [],
        "labels": [[l for l, _ in iter_labels(row["questions"][q])] for q in qnames],
    }


class TypedDecisions(Dataset):
    def __init__(self, rows, tok, max_len, qid, lid):
        self.enc = [encode(r, tok, max_len, qid, lid) for r in rows]
        self.rows = rows

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, i):
        return self.enc[i]


def collate(batch, pad_id):
    B = len(batch)
    L = max(len(b["input_ids"]) for b in batch)
    M = max(len(b["label_pos"]) for b in batch)
    Q = max(len(b["qtype"]) for b in batch)

    ids = torch.full((B, L), pad_id, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    pos = torch.zeros((B, M), dtype=torch.long)
    grp = torch.full((B, M), -1, dtype=torch.long)
    tgt = torch.zeros((B, M), dtype=torch.float)
    lmask = torch.zeros((B, M), dtype=torch.bool)
    qtype = torch.full((B, Q), -1, dtype=torch.long)
    qmask = torch.zeros((B, Q), dtype=torch.bool)

    for i, b in enumerate(batch):
        n, m, q = len(b["input_ids"]), len(b["label_pos"]), len(b["qtype"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        att[i, :n] = 1
        pos[i, :m] = torch.tensor(b["label_pos"])
        grp[i, :m] = torch.tensor(b["group"])
        tgt[i, :m] = torch.tensor(b["target"])
        lmask[i, :m] = True
        qtype[i, :q] = torch.tensor(b["qtype"])
        qmask[i, :q] = True

    return {"input_ids": ids, "attention_mask": att, "label_pos": pos,
            "group": grp, "target": tgt, "label_mask": lmask,
            "qtype": qtype, "qmask": qmask}


def val_split(rows, frac=0.15):
    """Deterministic train/val carve-out of the 1200-case train split.

    Hash-based on id so the same cases stay in validation when you add data or
    rerun on another machine - temperature fitted on a moving val set is not a
    calibration, it is a lottery.
    """
    def h(r):
        return int(hashlib.sha1(r["id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return [r for r in rows if h(r) >= frac], [r for r in rows if h(r) < frac]


def pad_cat(tensors, pad_value=0):
    """Concatenate [B, M] tensors whose M differs, padding to the widest.

    Batches are collated independently, so their label dimension is only as
    wide as their own widest example. A plain torch.cat across batches works
    right up until two batches contain different workflows - which is why this
    only shows up on the full dataset and never on a single-workflow slice.
    """
    width = max(t.size(1) for t in tensors)
    out = []
    for t in tensors:
        if t.size(1) < width:
            pad = torch.full((t.size(0), width - t.size(1)), pad_value,
                             dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad], dim=1)
        out.append(t)
    return torch.cat(out, dim=0)
