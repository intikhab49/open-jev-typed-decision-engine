"""Scoring. Accuracy is the headline; the rest is what makes confidence usable.

All functions take flat [B, M] tensors of per-label probabilities, the matching
consensus targets, and a boolean mask selecting real (non-padding) label slots.
"""
import torch


def _flat(probs, target, mask):
    return probs[mask].float(), target[mask].float()


def brier(probs, target, mask):
    """Multi-class Brier over label slots. Proper scoring rule, lower is better."""
    p, t = _flat(probs, target, mask)
    return ((p - t) ** 2).mean().item()


def nll(probs, target, mask):
    p, t = _flat(probs, target, mask)
    return (-(t * p.clamp_min(1e-8).log())).sum().item() / max(mask.sum().item(), 1)


def ece(probs, target, mask, bins=15):
    """Expected calibration error on the *chosen* label of each slot.

    Reliability is measured against the consensus probability of the label the
    model picked, which is the quantity an escalation threshold actually reads.
    """
    p, t = _flat(probs, target, mask)
    if p.numel() == 0:
        return float("nan")
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        sel = (p > edges[i]) & (p <= edges[i + 1])
        if sel.sum() == 0:
            continue
        total += (sel.float().mean() * (p[sel].mean() - t[sel].mean()).abs()).item()
    return total


def tvd(probs, target, mask, group, n_groups):
    """Total variation distance between predicted and consensus distributions.

    Accuracy only checks the argmax. TVD checks whether the whole distribution
    is right, which is what you are relying on when you route on confidence.
    """
    out = []
    for q in range(n_groups):
        m = (group == q) & mask
        if not m.any():
            continue
        d = ((probs - target).abs() * m).sum(-1) * 0.5
        out.append(d[m.any(-1)])
    return torch.cat(out).mean().item() if out else float("nan")


def risk_coverage(conf, correct, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9)):
    """Accuracy among decisions the model is confident about, and how many those are.

    This is the table that decides whether you can autoroute. A model that is
    96% accurate on the 40% of cases it is sure about is useful even if its
    overall accuracy is mediocre - you escalate the rest.
    """
    rows = []
    for t in thresholds:
        sel = conf >= t
        cov = sel.float().mean().item()
        acc = correct[sel].float().mean().item() if sel.any() else float("nan")
        rows.append((t, cov, acc))
    return rows


def ece_confidence(conf, correct, bins=15):
    """Standard ECE: top-1 confidence vs. empirical correctness.

    This is the number published benchmarks quote (Jev measured at 0.144 on
    typed-decisions, 0.154 on phishing), so it is the only one that can be
    compared against them. `ece` above answers a different question - how close
    the whole predicted distribution sits to the consensus - and the two can
    disagree sharply: a model that outputs near-uniform everywhere scores well
    on distributional ECE while being useless at ranking its own errors.
    """
    conf = conf.float().flatten()
    correct = correct.float().flatten()
    if conf.numel() == 0:
        return float("nan")
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        sel = (conf > edges[i]) & (conf <= edges[i + 1])
        if sel.sum() == 0:
            continue
        total += (sel.float().mean()
                  * (conf[sel].mean() - correct[sel].mean()).abs()).item()
    return total


def reliability_bins(conf, correct, bins=10):
    """-> [(bin_centre, mean_confidence, empirical_accuracy, count)] for non-empty bins.

    The raw material of a reliability diagram: a perfectly calibrated model puts
    every point on the diagonal, above it means underconfident, below means
    overconfident.
    """
    conf = conf.float().flatten()
    correct = correct.float().flatten()
    edges = torch.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        sel = (conf > edges[i]) & (conf <= edges[i + 1])
        n = int(sel.sum())
        if n == 0:
            continue
        out.append((float((edges[i] + edges[i + 1]) / 2),
                    float(conf[sel].mean()), float(correct[sel].mean()), n))
    return out
