"""Step 4 - score on the official 400-case test split and compare to Jev.

Reports the same headline metric Jev is published against (argmax accuracy over
all decisions), plus the things accuracy hides: calibration, distributional
agreement, the confidence/coverage tradeoff, and latency.
"""
import argparse
import json
import time
import torch
from torch.utils.data import DataLoader
from td_data import load_split, TypedDecisions, collate, pad_cat
from model import JevLite, argmax_per_question, require_ckpt
from typed_schema import TYPES
from metrics import (ece, ece_confidence, brier, nll, tvd, risk_coverage,
                     reliability_bins)

JEV_ACC = 0.727          # TypeSafe Jev 1.13.0, published on this test split
JEV_ECE_REPORTED = 0.144  # from the LocalLLaMA typed-decisions run


def evaluate(model, dl, dev, temp):
    P, T, M, TY, G = [], [], [], [], []
    pred_all, gold_all, conf_all, qtype_all = [], [], [], []
    n_cases = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            _, probs = model(b, apply_temperature=temp)
            pred, conf = argmax_per_question(probs, b)
            gold, _ = argmax_per_question(b["target"], b)
            m = b["qmask"]
            pred_all.append(pred[m]); gold_all.append(gold[m])
            conf_all.append(conf[m]); qtype_all.append(b["qtype"][m])
            P.append(probs.cpu()); T.append(b["target"].cpu())
            M.append(b["label_mask"].cpu()); G.append(b["group"].cpu())
            TY.append(b["qtype"].clamp(min=0).gather(1, b["group"].clamp(min=0)).cpu())
            n_cases += b["input_ids"].size(0)
    dt = time.perf_counter() - t0

    cat = lambda xs: torch.cat([x.reshape(-1) if x.dim() == 1 else x for x in xs])
    pred, gold = cat(pred_all).cpu(), cat(gold_all).cpu()
    conf, qt = cat(conf_all).cpu(), cat(qtype_all).cpu()
    correct = pred == gold

    # Pad to the widest label dimension: batches spanning different workflows
    # do not share one, and a bare torch.cat silently only works when they do.
    Pm, Tm = pad_cat(P), pad_cat(T)
    Mm = pad_cat(M, False)
    TYm = pad_cat(TY)
    Gm = pad_cat(G, -1)
    return dict(correct=correct, conf=conf, qtype=qt, dt=dt, n_cases=n_cases,
                P=Pm, T=Tm, M=Mm, TY=TYm, G=Gm)


def show(r, tag):
    acc = r["correct"].float().mean().item()
    print(f"\n=== {tag} ===")
    print(f"  accuracy   {acc:.4f}   ({int(r['correct'].sum())}/{len(r['correct'])} decisions)")
    print(f"  ECE        {ece_confidence(r['conf'], r['correct']):.4f}"
          "   (top-1 confidence vs correctness - comparable to published Jev)")
    print(f"  ECE-dist   {ece(r['P'], r['T'], r['M']):.4f}"
          "   (predicted vs consensus distribution)")
    print(f"  Brier      {brier(r['P'], r['T'], r['M']):.4f}")
    print(f"  NLL        {nll(r['P'], r['T'], r['M']):.4f}")
    print(f"  TVD        {tvd(r['P'], r['T'], r['M'], r['G'], r['P'].size(1)):.4f}"
          "   (distance from the consensus distribution)")
    print("  per type:")
    for i, name in enumerate(TYPES):
        sel = r["qtype"] == i
        selm = r["M"] & (r["TY"] == i)
        if sel.any():
            print(f"    {name:<8} acc {r['correct'][sel].float().mean():.4f}"
                  f"   ECE {ece_confidence(r['conf'][sel], r['correct'][sel]):.4f}"
                  f"   n={int(sel.sum())}")
    return acc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="jevlite.pt")
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--config", default="all",
                   help="one workflow (customer_service, invoice_processing, "
                        "security_incidents, agent_trace_observability) instead "
                        "of all four - a domain specialist is where a small model wins")
    a = p.parse_args()

    ck = require_ckpt(a.ckpt)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = JevLite(ck["encoder"])
    model.load_state_dict(ck["state_dict"])
    model.to(dev).eval()

    test = load_split("test", a.config, limit=a.limit)
    ds = TypedDecisions(test, model.tok, ck["max_len"], model.qid, model.lid)
    dl = DataLoader(ds, batch_size=a.bs,
                    collate_fn=lambda b: collate(b, model.tok.pad_token_id))

    raw = evaluate(model, dl, dev, temp=False)
    show(raw, "uncalibrated")
    cal = evaluate(model, dl, dev, temp=True)
    acc = show(cal, "calibrated (per-type temperature)")

    my_ece = ece_confidence(cal["conf"], cal["correct"])

    # Jev's 0.727 / 0.144 are full-split figures over all four configs. A subset run
    # is a different distribution, so don't print a delta someone could quote out of
    # context - say what the run actually was instead.
    full_split = a.config == "all" and a.limit is None
    if full_split:
        print(f"\n=== vs Jev 1.13.0 ===")
        print(f"  JevLite  {acc:.4f}        Jev  {JEV_ACC:.4f}"
              f"   ->  {'AHEAD' if acc > JEV_ACC else 'behind'} by {abs(acc-JEV_ACC):.4f}")
        print(f"  JevLite ECE {my_ece:.4f}    Jev ECE {JEV_ECE_REPORTED:.4f} (reported)"
              f"   ->  {'AHEAD' if my_ece < JEV_ECE_REPORTED else 'behind'}")
    else:
        scope = []
        if a.config != "all":
            scope.append(f"--config {a.config}")
        if a.limit is not None:
            scope.append(f"--limit {a.limit}")
        print(f"\n=== vs Jev 1.13.0: subset run, comparison suppressed ===")
        flags = " ".join(scope)
        print(f"  This run covered {cal['n_cases']} cases ({flags}).")
        print(f"  Jev's {JEV_ACC:.4f} / {JEV_ECE_REPORTED:.4f} are full-split figures over all 400")
        print(f"  test cases and all four configs - not a like-for-like baseline here.")
        print(f"  Run `python 04_eval.py` with no flags for the comparable number.")
    n_dec = len(cal["correct"])
    print(f"\n  {cal['dt']:.2f}s for {cal['n_cases']} cases / {n_dec} decisions"
          f"  ->  {cal['dt']/cal['n_cases']*1000:.1f} ms/case on {dev}, no network")
    print(f"  Jev API p50 measured from Europe: 239 ms/call")

    print("\n=== confidence -> coverage (can you autoroute?) ===")
    print(f"  {'threshold':>10}{'coverage':>11}{'accuracy':>11}")
    for t, cov, ac in risk_coverage(cal["conf"], cal["correct"]):
        print(f"  {t:>10.2f}{cov:>11.1%}{ac:>11.4f}")
    print("\n  Read the last column: if accuracy climbs sharply with the threshold,")
    print("  the confidence is real and you can route the top band automatically.")

    json.dump({
        "accuracy": acc,
        "ece": my_ece,
        "ece_dist": ece(cal["P"], cal["T"], cal["M"]),
        "brier": brier(cal["P"], cal["T"], cal["M"]),
        "nll": nll(cal["P"], cal["T"], cal["M"]),
        "tvd": tvd(cal["P"], cal["T"], cal["M"], cal["G"], cal["P"].size(1)),
        "ms_per_case": cal["dt"] / cal["n_cases"] * 1000,
        "n_cases": cal["n_cases"], "n_decisions": n_dec, "device": dev,
        "config": a.config, "limit": a.limit, "full_split": full_split,
        "per_type": [
            {"type": name,
             "accuracy": float(cal["correct"][cal["qtype"] == i].float().mean()),
             "ece": ece_confidence(cal["conf"][cal["qtype"] == i],
                                   cal["correct"][cal["qtype"] == i]),
             "n": int((cal["qtype"] == i).sum())}
            for i, name in enumerate(TYPES) if (cal["qtype"] == i).any()],
        "coverage": risk_coverage(cal["conf"], cal["correct"]),
        "reliability_raw": reliability_bins(raw["conf"], raw["correct"]),
        "reliability_cal": reliability_bins(cal["conf"], cal["correct"]),
    }, open("results.json", "w"), indent=1)
    print("\nwrote results.json  ->  python plots.py --show")
