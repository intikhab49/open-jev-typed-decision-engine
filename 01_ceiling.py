"""Step 1 - reference points and baselines, before you train anything. Costs $0.

  annotator   Mean probability that a randomly drawn annotator agrees with the
              consensus label. This is what ONE human rater scores against the
              consensus - a reference point, NOT a ceiling. A model that always
              predicts the consensus argmax scores 1.000, so exceeding this
              number is normal and expected; Jev's 0.727 already does.
  ambiguous   Share of decisions where the top two labels are within 0.1. On
              these the consensus label is close to a coin flip, so accuracy
              there is mostly luck. The `clear` column is the honest signal.
  majority    Always answer the most common label for that (workflow, question).
  frozen      Frozen encoder embeddings + logistic regression. This is the
              baseline that beat Jev by 10 points on Banking77 at 44x the speed.
              If your fine-tune cannot clear it, ship this instead.
"""
import argparse
import json
import numpy as np
from collections import defaultdict
from td_data import load_split
# state_to_text now lives behind probe.py; nothing here needs it directly

AMBIGUOUS_MARGIN = 0.1     # same threshold the dataset's own tooling uses


def annotator_agreement(rows):
    """Expected accuracy of a single random annotator judged against consensus."""
    per_type, allv = defaultdict(list), []
    for r in rows:
        for qn, g in r["gold"].items():
            top = max(g["probabilities"].values())
            per_type[g["type"]].append(top)
            allv.append(top)
    return float(np.mean(allv)), {k: float(np.mean(v)) for k, v in per_type.items()}


def ambiguous_fraction(rows):
    """Share of decisions whose consensus label is nearly a tie."""
    amb = tot = 0
    for r in rows:
        for qn, g in r["gold"].items():
            p = sorted(g["probabilities"].values(), reverse=True)
            margin = p[0] - (p[1] if len(p) > 1 else 0.0)
            amb += int(margin < AMBIGUOUS_MARGIN)
            tot += 1
    return amb / tot, tot


def majority(train, test):
    tally = defaultdict(lambda: defaultdict(int))
    for r in train:
        for qn, g in r["gold"].items():
            tally[(r["workflow"], qn)][g["label"]] += 1
    best = {k: max(v, key=v.get) for k, v in tally.items()}
    hit = tot = 0
    for r in test:
        for qn, g in r["gold"].items():
            tot += 1
            hit += int(best.get((r["workflow"], qn)) == g["label"])
    return hit / tot


def frozen_probe(train, test, encoder, max_len, batch_size=16):
    """Accuracy of embeddings + one logistic regression per (workflow, question).

    Shares its implementation with 07_ensemble.py via probe.py - two copies of
    the baseline would eventually disagree, and then the comparison is fiction.
    """
    from probe import embed_states, fit_probes

    Xtr = embed_states(train, encoder, max_len, batch_size)
    Xte = embed_states(test, encoder, max_len, batch_size)
    probes = fit_probes(train, Xtr)
    hit = tot = 0
    for i, r in enumerate(test):
        for qn, g in r["gold"].items():
            clf = probes.get((r["workflow"], qn))
            if clf is None:
                continue
            pred = clf if isinstance(clf, str) else clf.predict(Xte[i:i + 1])[0]
            hit += int(pred == g["label"])
            tot += 1
    return hit / max(tot, 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="answerdotai/ModernBERT-base")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--skip-frozen", action="store_true")
    p.add_argument("--config", default="all",
                   help="one workflow (customer_service, invoice_processing, "
                        "security_incidents, agent_trace_observability) instead "
                        "of all four - a domain specialist is where a small model wins")
    a = p.parse_args()

    train, test = load_split("train", a.config), load_split("test", a.config)
    n_dec = sum(len(r["gold"]) for r in test)
    print(f"train {len(train)} cases | test {len(test)} cases / {n_dec} decisions\n")

    c, per_type = annotator_agreement(test)
    spread = "  ".join(f"{k}={v:.3f}" for k, v in sorted(per_type.items()))
    amb, tot = ambiguous_fraction(test)
    print(f"  single annotator         {c:.3f}   {spread}")
    print(f"  ambiguous decisions      {amb:.1%}   "
          f"({int(amb*tot)}/{tot} within {AMBIGUOUS_MARGIN} of a tie)")
    maj = majority(train, test)
    print(f"  majority baseline        {maj:.3f}")
    fz = None
    if not a.skip_frozen:
        fz = frozen_probe(train, test, a.encoder, a.max_len)
        print(f"  frozen encoder + logreg  {fz:.3f}   (no fine-tuning)")

    json.dump({"annotator": c, "annotator_per_type": per_type, "ambiguous": amb,
               "majority": maj, "frozen": fz, "n_decisions": tot,
               "config": a.config},
              open("baselines.json", "w"), indent=1)

    print(f"\n  TypeSafe Jev 1.13.0      0.727   (published)")
    print("\nTargets: clear the frozen probe to justify fine-tuning at all, then")
    print("clear 0.727 to beat Jev. 'single annotator' is what one human rater")
    print("scores, not a cap - models routinely exceed it and Jev already does.")
    print(f"But {amb:.0%} of decisions are near-ties, so the last few points of")
    print("headline accuracy are largely luck. Judge on the calibration and")
    print("coverage tables in 04_eval.py, not on accuracy alone.")
