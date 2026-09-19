"""Step 5 - score on the held-out test set and compare against step 3.

Also measures throughput, which is the actual reason this project exists.
"""
import argparse, time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from model import AgentDecisionEngine, get_tokenizer
from data import load, split, targets
from schema import ACTIONS

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="engine.pt")
    p.add_argument("--bs", type=int, default=16)
    a = p.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = get_tokenizer(ck["encoder"])
    model = AgentDecisionEngine(ck["encoder"])
    model.load_state_dict(ck["state_dict"])
    model.to(dev).eval()

    te = split(load())["test"]
    ye, ya, ys = targets(te)
    pe, pa, ps = [], [], []

    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(te), a.bs):
            batch = [r["text"] for r in te[i : i + a.bs]]
            enc = tok(batch, truncation=True, max_length=ck["max_len"],
                      padding=True, return_tensors="pt").to(dev)
            le, la, ls = model(enc["input_ids"], enc["attention_mask"])
            pe += (torch.sigmoid(le) > 0.5).int().cpu().tolist()
            pa += la.argmax(-1).cpu().tolist()
            ps += ls.argmax(-1).cpu().tolist()
    dt = time.perf_counter() - t0

    print(f"\ntest n={len(te)}   {dt:.2f}s total   "
          f"{len(te)/dt:.1f} decisions/sec   {dt/len(te)*1000:.1f} ms/decision\n")
    print(f"is_terminal_error  acc {accuracy_score(ye.astype(int), pe):.3f}")
    print(f"action_required    acc {accuracy_score(ya, pa):.3f}   "
          f"macro-F1 {f1_score(ya, pa, average='macro'):.3f}")
    print(f"confidence_score   acc {accuracy_score(ys, ps):.3f}   "
          f"MAE {np.abs(np.array(ps) - ys).mean():.2f}\n")
    print(classification_report(ya, pa, target_names=ACTIONS, zero_division=0))
    print("Compare action_required against 03_baseline.py. If it is not clearly")
    print("higher, the encoder is not paying for itself - ship the TF-IDF model.")
