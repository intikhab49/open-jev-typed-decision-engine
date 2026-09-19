"""Step 2 - bulk-label the full set with the Batch API (50% off, <24h).

  python 02_label.py submit --n 10000          -> prints a batch id
  python 02_label.py collect --batch-id msgbatch_...
"""
import argparse, json, sys
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from teacher import SYSTEM, OUTPUT_CONFIG, user_msg, cost
from schema import AgentDecision

client = anthropic.Anthropic()

def submit(model, states):
    reqs = [
        Request(
            custom_id=s["id"][:64],
            params=MessageCreateParamsNonStreaming(
                model=model, max_tokens=256, system=SYSTEM,
                messages=[{"role": "user", "content": user_msg(s["text"])}],
                output_config=OUTPUT_CONFIG,
            ),
        )
        for s in states
    ]
    b = client.messages.batches.create(requests=reqs)
    print(f"batch {b.id} submitted with {len(reqs)} requests")
    print(f"poll:  python 02_label.py collect --batch-id {b.id}")

def collect(batch_id, states, out, model):
    b = client.messages.batches.retrieve(batch_id)
    if b.processing_status != "ended":
        print(f"status: {b.processing_status} - not ready yet")
        sys.exit(0)

    by_id = {s["id"][:64]: s for s in states}
    n_ok = n_bad = in_tok = out_tok = 0
    with open(out, "w", encoding="utf-8") as f:
        for res in client.messages.batches.results(batch_id):
            if res.result.type != "succeeded":
                n_bad += 1
                continue
            m = res.result.message
            in_tok += m.usage.input_tokens
            out_tok += m.usage.output_tokens
            try:
                txt = next(bl.text for bl in m.content if bl.type == "text")
                d = AgentDecision(**json.loads(txt))
            except Exception:
                n_bad += 1
                continue
            src = by_id.get(res.custom_id)
            if src is None:
                n_bad += 1
                continue
            f.write(json.dumps({"id": res.custom_id, "text": src["text"],
                                **d.model_dump()}) + "\n")
            n_ok += 1
    print(f"{n_ok} labeled, {n_bad} dropped -> {out}")
    print(f"batch cost ~${cost(model, in_tok, out_tok, batch=True):.2f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["submit", "collect"])
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--states", default="states.jsonl")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--batch-id")
    p.add_argument("--out", default="labeled.jsonl")
    a = p.parse_args()

    states = [json.loads(l) for l in open(a.states, encoding="utf-8")][: a.n]
    if a.cmd == "submit":
        submit(a.model, states)
    else:
        collect(a.batch_id, states, a.out, a.model)
