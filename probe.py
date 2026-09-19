"""The frozen-encoder probe: embeddings + one logistic regression per question.

Measured at 0.670 on the test split against the fine-tuned model's 0.6195, so
this is not a strawman baseline - it is the thing to beat, and the thing to
ensemble with.

What it cannot do is why the fine-tuned model still exists: a probe needs
labelled examples for every question before it can answer that question, and it
has no notion of a schema supplied at request time.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression

from typed_schema import state_to_text


def embed_states(rows, encoder, max_len=512, batch_size=16, device=None):
    """Mean-pooled frozen encoder representation of each state. -> [N, d]"""
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(encoder)
    enc = AutoModel.from_pretrained(encoder).to(dev).eval()
    out = []
    for i in range(0, len(rows), batch_size):
        texts = [state_to_text(r["state"]) for r in rows[i:i + batch_size]]
        b = tok(texts, truncation=True, max_length=max_len,
                padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            h = enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
    return np.vstack(out)


def fit_probes(rows, X):
    """One classifier per (workflow, question). -> {key: clf | constant label}"""
    probes = {}
    keys = sorted({(r["workflow"], qn) for r in rows for qn in r["gold"]})
    for wf, qn in keys:
        idx = [i for i, r in enumerate(rows) if r["workflow"] == wf and qn in r["gold"]]
        if not idx:
            continue
        y = [rows[i]["gold"][qn]["label"] for i in idx]
        if len(set(y)) < 2:
            probes[(wf, qn)] = y[0]        # degenerate question, always one answer
            continue
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        clf.fit(X[idx], y)
        probes[(wf, qn)] = clf
    return probes


def probe_distribution(probes, workflow, qname, labels, x):
    """Probe's distribution over `labels`, in that exact order. -> [len(labels)]

    Returns a uniform distribution when the probe has never seen this question,
    so the ensemble degrades to the neural model rather than crashing.
    """
    p = probes.get((workflow, qname))
    n = len(labels)
    if p is None:
        return np.full(n, 1.0 / n)
    if isinstance(p, str):
        out = np.zeros(n)
        out[labels.index(p)] = 1.0 if p in labels else 0.0
        return out if out.sum() else np.full(n, 1.0 / n)

    proba = p.predict_proba(x.reshape(1, -1))[0]
    by_label = dict(zip(p.classes_, proba))
    out = np.array([by_label.get(l, 0.0) for l in labels], dtype="float64")
    s = out.sum()
    return out / s if s > 0 else np.full(n, 1.0 / n)
