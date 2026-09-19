"""Decision console for the trained model. Paste a state, get typed answers.

Not a chat. The model has no decoder and cannot generate text - for each
question it picks one of the labels YOU defined and reports how confident it
is. Ask it a question and it will not answer; it will fill in your form about
the sentence you typed.

  python decide.py                 # agent schema by default
  python decide.py --schema risk   # start on another one

Inside the prompt:
  :schemas            list the built-in question sets
  :use <name>         switch schema
  :show               print the current questions
  :q                  quit

Paste multi-line states and finish with a blank line. Anything that parses as
JSON is treated as JSON, everything else as plain text - the model was trained
on JSON states but the encoder does not care.
"""
from __future__ import annotations
import argparse
import json
import os
import time

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from typed_schema import Question

SCHEMAS = {
    "agent": {
        "action": Question.choice(
            "What should the observability system do with this trace?",
            {"continue": "Let the agent proceed without interruption.",
             "human_review": "Queue this trace for a human to review.",
             "observe": "Keep running, but flag the trace for later sampling.",
             "stop": "Halt the agent now."}),
        "outcome": Question.choice(
            "How did this agent run turn out?",
            {"success": "The agent completed the task correctly.",
             "partial": "The agent made progress but did not finish.",
             "failure": "The agent did not accomplish the task.",
             "harmful": "The agent caused damage or violated a constraint."}),
        "needs_review": Question.noul(
            "This trace requires human review.",
            true_desc="A human should inspect this run.",
            false_desc="No human attention is warranted."),
        "risk": Question.score(
            "How risky was the agent's behaviour in this trace?",
            ["Benign: read-only or clearly safe actions.",
             "Low: routine writes within scope.",
             "Moderate: irreversible or out-of-scope actions.",
             "High: destructive, security-relevant or policy-violating."]),
    },
    "support": {
        "category": Question.choice(
            "What kind of support request is this?",
            {"billing": "Charges, invoices, refunds or payment methods.",
             "technical": "Something is broken or not behaving as documented.",
             "account": "Access, login, permissions or profile changes.",
             "other": "Anything that does not fit the categories above."}),
        "action": Question.choice(
            "What should happen to this ticket next?",
            {"auto_reply": "A templated answer resolves it.",
             "escalate": "Send it to a specialist queue.",
             "refund": "Issue a refund or credit.",
             "close": "No action needed, close it."}),
        "needs_human": Question.noul(
            "This ticket needs a human agent.",
            true_desc="A person should handle this.",
            false_desc="Automation can resolve it."),
        "urgency": Question.score(
            "How quickly does this need attention?",
            ["No time pressure.", "Routine, normal queue.",
             "Elevated, within the week.", "Critical, same day."]),
    },
    "risk": {
        "credential_compromise": Question.noul(
            "The evidence indicates a credential or account has been compromised."),
        "action": Question.choice(
            "What should the security team do?",
            {"monitor": "Keep watching, no action yet.",
             "investigate": "Open an investigation.",
             "contain": "Isolate the affected system or account now.",
             "dismiss": "This is benign, close it."}),
        "severity": Question.score(
            "How severe is this incident?",
            ["Informational.", "Low, routine noise.",
             "Moderate, needs a response.", "High, active compromise."]),
    },
}

EXAMPLE = {
    "agent": {"agent": {"autonomy": "checkpointed", "model": "internal-agent-v4"},
              "constraints": ["Do not touch customer data outside named accounts"],
              "task": "Delete personal data for accounts in the erasure queue.",
              "trace_summary": {"duration_s": 128.5, "steps": 7, "tool_errors": 1,
                                "irreversible_actions": 2, "constraint_violations": 1}},
    "support": {"message": "i was charged twice this month and the second charge "
                           "is still pending. can you refund it today please",
                "customer": {"plan": "pro", "tenure_months": 14}},
    "risk": {"alert": "impossible travel: login from two countries 8 minutes apart",
             "user": {"role": "admin", "mfa": False},
             "signals": {"failed_logins": 0, "new_device": True}},
}


def render(out, elapsed_ms):
    width = max(len(k) for k in out)
    for name, r in out.items():
        bar = "#" * int(round(r["confidence"] * 24))
        print(f"  {name:<{width}}  {r['label']:<16} {r['confidence']:.3f}  {bar}")
        others = {k: v for k, v in r["probabilities"].items() if k != r["label"]}
        rest = "  ".join(f"{k} {v:.2f}" for k, v in
                         sorted(others.items(), key=lambda kv: -kv[1])[:3])
        if rest:
            print(f"  {'':<{width}}  {rest}")
    print(f"\n  {elapsed_ms:.0f} ms, one forward pass, $0\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="jevlite.pt")
    p.add_argument("--schema", default="agent", choices=list(SCHEMAS))
    a = p.parse_args()

    # 05_serve.py starts with a digit so it cannot be imported by name
    import importlib.util
    spec = importlib.util.spec_from_file_location("serve", "05_serve.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    print("loading the model...", flush=True)
    e = S.load(a.ckpt)
    name = a.schema
    print(f"ready on {e['dev']}   schema: {name}   "
          f"temperatures {[round(t, 3) for t in e['temps']]}")
    print("type or paste a state, press enter. plain english works too.")
    print("  :ex           run the built-in example")
    print("  :use <name>   switch schema   (" + "  ".join(SCHEMAS) + ")")
    print("  :schemas  :show  :q\n")

    def send(raw):
        try:
            state = json.loads(raw)
        except json.JSONDecodeError:
            state = raw
        t0 = time.perf_counter()
        out = S.decide(state, SCHEMAS[name])
        render(out, (time.perf_counter() - t0) * 1000)

    def unbalanced(t):
        """True while a JSON-looking paste is still missing closers."""
        if t.lstrip()[:1] not in "{[":
            return False
        q = t.count('"') % 2
        return q or t.count("{") > t.count("}") or t.count("[") > t.count("]")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not line:
            continue
        if line in (":q", ":quit", ":exit"):
            return
        if line == ":schemas":
            print("  " + "   ".join(SCHEMAS))
            continue
        if line == ":show":
            print(json.dumps(SCHEMAS[name], indent=1))
            continue
        if line in (":ex", ":example"):
            print(f"  {json.dumps(EXAMPLE[name])}")
            send(json.dumps(EXAMPLE[name]))
            continue
        if line.startswith(":use"):
            want = line.split(None, 1)[1].strip() if " " in line else ""
            if want in SCHEMAS:
                name = want
                print(f"  schema: {name}   questions: "
                      f"{', '.join(SCHEMAS[name])}")
            else:
                print(f"  no such schema. have: {', '.join(SCHEMAS)}")
            continue

        # Enter sends. Only a half-finished JSON paste keeps reading, so a
        # pasted one-liner works and prose works, which is what people expect.
        while unbalanced(line):
            try:
                more = input("… ")
            except (EOFError, KeyboardInterrupt):
                break
            if not more.strip():
                break
            line += "\n" + more
        send(line)


if __name__ == "__main__":
    main()
