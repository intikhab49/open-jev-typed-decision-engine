"""Step 0 - get raw states to label.

Two sources:
  --source swebench   : SWE-bench Lite problem statements (public, no auth)
  --source local      : your own JSONL, one {"id": ..., "text": ...} per line

Writes states.jsonl: {"id": str, "text": str}
"""
import argparse, json, pathlib

def swebench(n):
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    for i, r in enumerate(ds):
        if i >= n:
            break
        yield {"id": r["instance_id"], "text": r["problem_statement"]}

def local(path, n):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            r = json.loads(line)
            yield {"id": r.get("id", str(i)), "text": r["text"]}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="swebench", choices=["swebench", "local"])
    p.add_argument("--path", default="raw.jsonl")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--out", default="states.jsonl")
    a = p.parse_args()

    rows = swebench(a.n) if a.source == "swebench" else local(a.path, a.n)
    with open(a.out, "w", encoding="utf-8") as f:
        c = 0
        for r in rows:
            r["text"] = r["text"][:20000]          # hard cap, see README on context
            f.write(json.dumps(r) + "\n")
            c += 1
    print(f"wrote {c} states -> {a.out}")
