"""Step 6 - serve it.

Local:   python 06_serve.py --text "Traceback (most recent call last): ..."
HTTP:    python 06_serve.py --serve      (FastAPI on :8000, POST /decide {"text": ...})
"""
import argparse, json
import torch
from model import AgentDecisionEngine, get_tokenizer
from schema import ACTIONS, AgentDecision

_state = {}

def load_engine(ckpt="engine.pt", device=None):
    ck = torch.load(ckpt, map_location="cpu")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = get_tokenizer(ck["encoder"])
    m = AgentDecisionEngine(ck["encoder"])
    m.load_state_dict(ck["state_dict"])
    m.to(dev).eval()
    _state.update(tok=tok, model=m, dev=dev, max_len=ck["max_len"])
    return _state

@torch.no_grad()
def decide(text: str) -> AgentDecision:
    s = _state
    enc = s["tok"](text, truncation=True, max_length=s["max_len"],
                   return_tensors="pt").to(s["dev"])
    le, la, ls = s["model"](enc["input_ids"], enc["attention_mask"])
    return AgentDecision(
        is_terminal_error=bool(torch.sigmoid(le).item() > 0.5),
        action_required=ACTIONS[int(la.argmax(-1))],
        confidence_score=int(ls.argmax(-1)) + 1,
    )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="engine.pt")
    p.add_argument("--text")
    p.add_argument("--serve", action="store_true")
    a = p.parse_args()
    load_engine(a.ckpt)

    if a.serve:
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn

        class Req(BaseModel):
            text: str

        app = FastAPI()

        @app.post("/decide")
        def _decide(r: Req):
            return decide(r.text).model_dump()

        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print(json.dumps(decide(a.text or "all tests passed").model_dump(), indent=2))
