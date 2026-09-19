"""JevLite: one encoder pass, a logit per candidate label.

The schema is data, not architecture. Every candidate label is written into the
input sequence behind a marker token; the head is a single Linear(d, 1) applied
at each marker position. Adding a question, a label, or a whole new workflow at
inference time requires no retraining and no new parameters.

Compare to a fixed-head classifier, which needs one nn.Linear per question and
can only ever answer the questions it was built with.
"""
from __future__ import annotations
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from td_data import add_markers
from typed_schema import TYPES

DEFAULT_ENCODER = "answerdotai/ModernBERT-base"   # bidirectional, 8192 ctx, 150M
NEG = -1e4                                        # fp16-safe masking value


def grouped_softmax(logits, group, label_mask, n_groups):
    """Softmax within each question's own label set. -> probs [B, M]"""
    probs = torch.zeros_like(logits)
    for q in range(n_groups):
        m = (group == q) & label_mask
        if not m.any():
            continue
        masked = logits.masked_fill(~m, NEG)
        p = torch.softmax(masked, dim=-1)
        probs = probs + p * m
    return probs


class JevLite(nn.Module):
    def __init__(self, encoder_name=DEFAULT_ENCODER, tokenizer=None, dropout=0.1):
        super().__init__()
        self.encoder_name = encoder_name
        self.tok = tokenizer or AutoTokenizer.from_pretrained(encoder_name)
        _, self.qid, self.lid = add_markers(self.tok)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder.resize_token_embeddings(len(self.tok))
        d = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.score = nn.Linear(d, 1)
        # One temperature per primitive type. Jev's measured failure is that its
        # types miscalibrate in opposite directions, so a single global T cannot
        # fix both. These are frozen during training and fitted in 03_calibrate.py.
        self.log_temp = nn.Parameter(torch.zeros(len(TYPES)), requires_grad=False)

    def logits(self, input_ids, attention_mask, label_pos):
        h = self.encoder(input_ids=input_ids,
                         attention_mask=attention_mask).last_hidden_state
        d = h.size(-1)
        picked = h.gather(1, label_pos.unsqueeze(-1).expand(-1, -1, d))
        return self.score(self.drop(picked)).squeeze(-1)          # [B, M]

    def forward(self, batch, apply_temperature=False):
        lg = self.logits(batch["input_ids"], batch["attention_mask"], batch["label_pos"])
        if apply_temperature:
            lg = lg / self.temp_per_label(batch)
        n_groups = batch["qtype"].size(1)
        probs = grouped_softmax(lg.float(), batch["group"], batch["label_mask"], n_groups)
        return lg, probs

    def temp_per_label(self, batch):
        """Map each label slot to its question's temperature. -> [B, M]"""
        g = batch["group"].clamp(min=0)
        t = batch["qtype"].clamp(min=0).gather(1, g)              # type id per label
        return self.log_temp.exp()[t]


def decision_loss(probs, batch, brier_weight=1.0):
    """Soft cross-entropy against the consensus distribution, plus Brier.

    The dataset gives full annotator distributions, so training on the hard argmax
    throws away the signal that makes confidence mean something. Brier is a proper
    scoring rule and is what pushes probabilities toward honesty rather than
    toward whichever label happens to win.
    """
    p = probs.clamp_min(1e-8)
    t = batch["target"]
    m = batch["label_mask"].float()
    n_q = batch["qmask"].sum().clamp(min=1)
    ce = -(t * p.log() * m).sum() / n_q
    brier = (((probs - t) ** 2) * m).sum() / n_q
    return ce + brier_weight * brier, ce.detach(), brier.detach()


def argmax_per_question(probs, batch):
    """-> [B, Q] index of the winning label *within* each question, -1 if absent."""
    B, Q = batch["qtype"].shape
    out = torch.full((B, Q), -1, dtype=torch.long, device=probs.device)
    conf = torch.zeros((B, Q), device=probs.device)
    for q in range(Q):
        m = (batch["group"] == q) & batch["label_mask"]
        if not m.any():
            continue
        scores = probs.masked_fill(~m, -1.0)
        best = scores.argmax(dim=-1)
        first = torch.where(m.any(-1), m.float().argmax(dim=-1), torch.zeros_like(best))
        has = m.any(-1)
        out[:, q] = torch.where(has, best - first, torch.full_like(best, -1))
        conf[:, q] = torch.where(has, scores.max(dim=-1).values,
                                 torch.zeros_like(conf[:, q]))
    return out, conf


def require_ckpt(path):
    """Load a checkpoint, or say plainly which earlier step never finished."""
    if not os.path.exists(path):
        sys.exit(
            f"'{path}' does not exist - 02_train.py has not completed. "
            "Scroll up and fix the training step; every step after it depends "
            "on the checkpoint it writes.")
    return torch.load(path, map_location="cpu")


@torch.no_grad()
def predict_distributions(model, rows, max_len, device, batch_size=8):
    """Per-question probability distributions with label names attached.

    -> [{qname: {"labels": [...], "probs": [...]}}] in the same order as `rows`.

    argmax_per_question is enough for accuracy; an ensemble needs the whole
    distribution keyed by label so it can be combined with another model that
    orders its classes differently.
    """
    from td_data import TypedDecisions, collate
    from torch.utils.data import DataLoader

    ds = TypedDecisions(rows, model.tok, max_len, model.qid, model.lid)
    dl = DataLoader(ds, batch_size=batch_size,
                    collate_fn=lambda b: collate(b, model.tok.pad_token_id))
    model.eval()
    out, at = [], 0
    for b in dl:
        n = b["input_ids"].size(0)
        b = {k: v.to(device) for k, v in b.items()}
        _, probs = model(b, apply_temperature=True)
        probs = probs.cpu()
        for i in range(n):
            enc = ds[at + i]
            row, cursor = {}, 0
            for gi, qname in enumerate(enc["qnames"]):
                labels = enc["labels"][gi]
                row[qname] = {"labels": labels,
                              "probs": probs[i, cursor:cursor + len(labels)].tolist()}
                cursor += len(labels)
            out.append(row)
        at += n
    return out
