"""Step 5 - the Jev-shaped API, running locally for free.

  python 05_serve.py --demo
  python 05_serve.py --serve      # POST /decide {"state": {...}, "questions": {...}}

The questions are supplied per request. Nothing about the schema is baked into
the weights, so you can add a question, rename a label, or point it at a new
workflow without retraining - the same property that makes Jev usable as a
general decision endpoint.
"""
import argparse
import json
import torch
from td_data import encode, collate
from model import JevLite, require_ckpt
from typed_schema import iter_labels, Question

_engine = {}


def load(ckpt="jevlite.pt", device=None):
    ck = require_ckpt(ckpt)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = JevLite(ck["encoder"])
    m.load_state_dict(ck["state_dict"])
    m.to(dev).eval()
    _engine.update(model=m, dev=dev, max_len=ck["max_len"],
                   temps=ck.get("temperatures"))
    return _engine


@torch.no_grad()
def decide(state, questions, calibrated=True):
    """-> {question: {"label": str, "confidence": float, "probabilities": {...}}}"""
    e = _engine
    m = e["model"]
    row = {"id": "live", "state": state, "questions": questions, "gold": {}}
    enc = encode(row, m.tok, e["max_len"], m.qid, m.lid, with_gold=False)
    b = collate([enc], m.tok.pad_token_id)
    b = {k: v.to(e["dev"]) for k, v in b.items()}
    _, probs = m(b, apply_temperature=calibrated)
    probs = probs[0].cpu()

    out, i = {}, 0
    for gi, qname in enumerate(enc["qnames"]):
        labels = enc["labels"][gi]
        vals = probs[i:i + len(labels)].tolist()
        i += len(labels)
        dist = dict(zip(labels, (round(v, 4) for v in vals)))
        top = max(dist, key=dist.get)
        out[qname] = {"label": top, "confidence": dist[top], "probabilities": dist}
    return out


DEMO_STATE = {
    "agent": {"autonomy": "checkpointed", "model": "internal-agent-v4"},
    "constraints": ["Do not touch customer data outside the named accounts"],
    "task": "Delete personal data for the accounts in the erasure queue.",
    "trace_summary": {"duration_s": 128.5, "steps": 7, "tool_errors": 1,
                      "irreversible_actions": 2, "constraint_violations": 1},
}

DEMO_QUESTIONS = {
    "action": Question.choice(
        "What should the observability system do with this trace?",
        {"continue": "Let the agent proceed without interruption.",
         "human_review": "Queue this trace for a human to review.",
         "observe": "Keep running, but flag the trace for later sampling.",
         "stop": "Halt the agent now."}),
    "needs_review": Question.noul(
        "This trace requires human review.",
        true_desc="A human should inspect this run.",
        false_desc="No human attention is warranted."),
    "risk": Question.score(
        "How risky was the agent's behaviour in this trace?",
        ["Benign: read-only or clearly safe actions.",
         "Low: routine writes within scope.",
         "Moderate: irreversible or out-of-scope actions.",
         "High: destructive, security-relevant, or policy-violating actions."]),
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="jevlite.pt")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    e = load(a.ckpt)
    if e["temps"]:
        print(f"loaded on {e['dev']}  temperatures={[round(t,3) for t in e['temps']]}")

    if a.serve:
        from fastapi import FastAPI
        from pydantic import BaseModel
        from typing import Any
        import uvicorn

        class Req(BaseModel):
            state: Any
            questions: dict
            calibrated: bool = True

        app = FastAPI(title="JevLite")

        @app.post("/decide")
        def _decide(r: Req):
            return decide(r.state, r.questions, r.calibrated)

        uvicorn.run(app, host="0.0.0.0", port=a.port)
    else:
        import time
        t0 = time.perf_counter()
        out = decide(DEMO_STATE, DEMO_QUESTIONS)
        print(json.dumps(out, indent=2))
        print(f"\n{(time.perf_counter()-t0)*1000:.1f} ms, 3 typed questions, "
              "one forward pass, $0")
