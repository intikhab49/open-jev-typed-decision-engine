"""Step 1 - the go/no-go gate. Run this BEFORE generating 10k labels.

Labels the same N states R times and measures how often the teacher agrees with
itself. That agreement is the hard ceiling on your student model's accuracy.
If action_required agreement is ~70%, a student scoring 70% is already perfect.

Sampling params (temperature/top_p) are rejected by current models, so repeat
sampling is the only way to measure this - which is exactly what we want.
"""
import argparse, json, collections
from concurrent.futures import ThreadPoolExecutor
import anthropic
from teacher import SYSTEM, OUTPUT_CONFIG, user_msg, parse, cost

client = anthropic.Anthropic()

def label_one(args):
    model, text = args
    r = client.messages.create(
        model=model, max_tokens=256, system=SYSTEM,
        messages=[{"role": "user", "content": user_msg(text)}],
        output_config=OUTPUT_CONFIG,
    )
    return parse(r).model_dump(), r.usage.input_tokens, r.usage.output_tokens

def majority_rate(values):
    """Fraction of runs that match the modal value."""
    c = collections.Counter(values)
    return c.most_common(1)[0][1] / len(values)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--n", type=int, default=100, help="states to test")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--states", default="states.jsonl")
    a = p.parse_args()

    states = [json.loads(l) for l in open(a.states, encoding="utf-8")][: a.n]
    jobs = [(a.model, s["text"]) for s in states for _ in range(a.repeats)]

    with ThreadPoolExecutor(max_workers=8) as ex:
        out = list(ex.map(label_one, jobs))

    labels = [o[0] for o in out]
    in_tok = sum(o[1] for o in out)
    out_tok = sum(o[2] for o in out)

    per_field = {}
    for field in ("is_terminal_error", "action_required", "confidence_score"):
        rates = []
        for i in range(len(states)):
            runs = labels[i * a.repeats : (i + 1) * a.repeats]
            rates.append(majority_rate([r[field] for r in runs]))
        per_field[field] = sum(rates) / len(rates)

    print(f"\nteacher: {a.model}   states: {len(states)}   repeats: {a.repeats}")
    print("-" * 52)
    for k, v in per_field.items():
        print(f"  {k:<20} self-agreement {v:6.1%}   <- student ceiling")
    dist = collections.Counter(l["action_required"] for l in labels)
    print(f"\n  action distribution: {dict(dist)}")
    print(f"  spent ~${cost(a.model, in_tok, out_tok):.2f}")
    print("\nRead this as: if a field is below ~0.80, either tighten its definition")
    print("in teacher.py SYSTEM, or drop the field. Do not train on it as-is.")
