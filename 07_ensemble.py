"""Step 7 - combine the fine-tuned model with the frozen probe.

Measured separately on the test split: fine-tune 0.6195, frozen probe 0.670.
Two models at similar accuracy that make *different* errors usually combine
above both, and this costs one afternoon rather than another training run.

  python 07_ensemble.py

The blend weight is chosen on validation, never on test. Probes used to pick the
weight are fitted on the training portion only; the probes scored against test
are refitted on all of train. Skipping either of those is how an ensemble comes
out looking better than it is.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import torch

from td_data import load_split, val_split
from model import JevLite, require_ckpt, predict_distributions
from probe import embed_states, fit_probes, probe_distribution
from typed_schema import TYPE_ID, TYPES
from metrics import ece_confidence, risk_coverage

JEV_ACC, JEV_ECE = 0.727, 0.144


def sharpen(p, temp):
    """Temperature in probability space: p**(1/T), renormalised.

    Mixing two distributions that disagree flattens the result, so the blend
    ends up *under*confident even as accuracy improves - measured ECE went
    0.105 (model) -> 0.156 (blend) while accuracy went 0.624 -> 0.697. One
    scalar fitted on validation puts the confidence back where it belongs.
    """
    if temp == 1.0:
        return p
    q = np.power(np.clip(p, 1e-12, 1.0), 1.0 / temp)
    s = q.sum()
    return q / s if s > 0 else p


def blend(model_dists, probe_dists, w, temp=1.0):
    """p = w * model + (1 - w) * probe, per question. -> [(pred, conf, qname)]"""
    out = []
    for md, pd in zip(model_dists, probe_dists):
        for qname, m in md.items():
            p = np.asarray(pd[qname])
            mp = np.asarray(m["probs"], dtype="float64")
            mix = w * mp + (1.0 - w) * p
            s = mix.sum()
            mix = mix / s if s > 0 else np.full(len(mix), 1.0 / len(mix))
            mix = sharpen(mix, temp)
            k = int(mix.argmax())
            out.append((m["labels"][k], float(mix[k]), qname))
    return out


def score(blended, rows, want_type=False):
    gold, qtypes = [], []
    for r in rows:
        for qname in sorted(r["questions"]):
            gold.append(r["gold"][qname]["label"])
            qtypes.append(TYPE_ID[r["questions"][qname]["type"]])
    pred = [b[0] for b in blended]
    conf = torch.tensor([b[1] for b in blended])
    correct = torch.tensor([p == g for p, g in zip(pred, gold)])
    acc = correct.float().mean().item()
    if want_type:
        return acc, ece_confidence(conf, correct), correct, conf, torch.tensor(qtypes)
    return acc, ece_confidence(conf, correct), correct, conf


def probe_dists_for(rows, probes, X):
    out = []
    for i, r in enumerate(rows):
        row = {}
        for qname in sorted(r["questions"]):
            from typed_schema import iter_labels
            labels = [l for l, _ in iter_labels(r["questions"][qname])]
            row[qname] = probe_distribution(probes, r["workflow"], qname, labels, X[i])
        out.append(row)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="jevlite.pt")
    p.add_argument("--config", default="all")
    p.add_argument("--probe-max-len", type=int, default=512)
    a = p.parse_args()

    ck = require_ckpt(a.ckpt)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = JevLite(ck["encoder"])
    model.load_state_dict(ck["state_dict"])
    model.to(dev).eval()

    train = load_split("train", a.config)
    test = load_split("test", a.config)
    tr_part, va_part = val_split(train)
    print(f"train {len(tr_part)} | val {len(va_part)} | test {len(test)}\n")

    print("embedding states with the frozen encoder...")
    Xtr = embed_states(tr_part, ck["encoder"], a.probe_max_len, device=dev)
    Xva = embed_states(va_part, ck["encoder"], a.probe_max_len, device=dev)
    Xall = embed_states(train, ck["encoder"], a.probe_max_len, device=dev)
    Xte = embed_states(test, ck["encoder"], a.probe_max_len, device=dev)

    print("scoring the neural model...")
    md_va = predict_distributions(model, va_part, ck["max_len"], dev)
    md_te = predict_distributions(model, test, ck["max_len"], dev)

    # Weight selection: probes fitted on the training portion only.
    probes_tr = fit_probes(tr_part, Xtr)
    pd_va = probe_dists_for(va_part, probes_tr, Xva)

    print("\nchoosing the blend weight on validation:")
    print(f"  {'w':>5}{'accuracy':>11}{'ECE':>9}   (w=1 is model only, w=0 probe only)")
    best_w, best_acc = 1.0, -1.0
    curve = []
    for w in [i / 20 for i in range(21)]:
        acc, e, _, _ = score(blend(md_va, pd_va, w), va_part)
        curve.append({"w": w, "accuracy": acc, "ece": e})
        if w * 20 % 2 == 0:
            print(f"  {w:>5.2f}{acc:>11.4f}{e:>9.4f}")
        if acc > best_acc:
            best_acc, best_w = acc, w
    print(f"\n  best on validation: w={best_w:.2f} at {best_acc:.4f}")

    # Sharpening cannot move the argmax, so accuracy is already fixed at this
    # point and the only thing left to choose is the temperature that makes the
    # confidence honest. Chosen on validation, like w.
    base_ece = score(blend(md_va, pd_va, best_w), va_part)[1]
    best_t, best_ece = 1.0, base_ece
    for i in range(35):
        t = 0.30 + 0.05 * i
        e = score(blend(md_va, pd_va, best_w, t), va_part)[1]
        if e < best_ece:
            best_t, best_ece = t, e
    note = "  (T < 1 means the blend was underconfident)" if best_t < 1 else ""
    print(f"  sharpening: T={best_t:.2f} takes validation ECE "
          f"{base_ece:.4f} -> {best_ece:.4f}{note}")

    # Test: probes refitted on ALL of train, weight frozen from validation.
    probes_all = fit_probes(train, Xall)
    pd_te = probe_dists_for(test, probes_all, Xte)

    rows = []
    for name, w, t in (("model only", 1.0, 1.0), ("probe only", 0.0, 1.0),
                       (f"ensemble w={best_w:.2f}", best_w, 1.0),
                       (f"  + sharpened T={best_t:.2f}", best_w, best_t)):
        acc, e, correct, conf, qt = score(blend(md_te, pd_te, w, t), test,
                                          want_type=True)
        rows.append((name, acc, e, correct, conf, qt))

    n_dec = sum(len(r["gold"]) for r in test)
    print(f"\n=== test split ({len(test)} cases / {n_dec} decisions) ===")
    print(f"  {'':<22}{'accuracy':>10}{'ECE':>9}")
    for name, acc, e, *_ in rows:
        print(f"  {name:<22}{acc:>10.4f}{e:>9.4f}")
    print(f"  {'TypeSafe Jev 1.13.0':<22}{JEV_ACC:>10.4f}{JEV_ECE:>9.4f}")
    if a.config != "all" or len(test) != 400:
        print("  ! not the full 400-case split - the Jev row is NOT comparable")

    name, acc, e, correct, conf, qt = rows[-1]
    gap = JEV_ACC - acc
    print(f"\n  ensemble vs Jev: behind by {gap:.4f}" if gap > 0
          else f"\n  ensemble vs Jev: AHEAD by {-gap:.4f}")
    print("  per type:")
    for i, tname in enumerate(TYPES):
        sel = qt == i
        if sel.any():
            print(f"    {tname:<8} acc {correct[sel].float().mean():.4f}"
                  f"   ECE {ece_confidence(conf[sel], correct[sel]):.4f}"
                  f"   n={int(sel.sum())}")

    print("\n=== confidence -> coverage ===")
    print(f"  {'threshold':>10}{'coverage':>11}{'accuracy':>11}")
    cov = risk_coverage(conf, correct)
    for t, c, ac in cov:
        print(f"  {t:>10.2f}{c:>11.1%}{ac:>11.4f}")

    json.dump({"best_w": best_w, "best_temp": best_t, "val_curve": curve,
               "config": a.config,
               "test": {n: {"accuracy": ac, "ece": ee}
                        for n, ac, ee, *_ in rows},
               "coverage": cov}, open("ensemble.json", "w"), indent=1)
    print("\nwrote ensemble.json  ->  python plots.py --outdir plots")
