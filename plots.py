"""Charts for the run. Reads the JSON each step writes; renders nothing it lacks.

  python plots.py              # every chart it has data for, saved to plots/
  python plots.py --show       # also display (Colab renders them inline)

Deliberately decoupled from the pipeline: the scripts dump numbers, this reads
them. You can re-style a chart without re-running a 15-minute training job.
"""
from __future__ import annotations
import argparse
import json
import pathlib

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

# --- palette -----------------------------------------------------------------
# Validated categorical slots 1-2 (worst adjacent CVD dE 24.7, normal 33.6).
# Matplotlib renders a fixed PNG that cannot follow Colab's theme, so every
# figure carries its own light surface and reads correctly on either background.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
BASELINE = "#c3c2b7"
S1 = "#2a78d6"    # blue
S2 = "#eb6834"    # orange
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

JEV_ACC = 0.727
JEV_ECE = 0.144


def style():
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": BASELINE, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": BASELINE, "grid.alpha": 0.45,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0, "lines.markersize": 8,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 12,
        "figure.dpi": 120,
    })


def hline_label(ax, y, text, color=None):
    """Caption a horizontal reference line at the right edge, just above it.

    Anchoring in data coordinates put the label outside the axes on some
    panels and it silently vanished; x here is axes-relative, y is data.
    """
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.annotate(text, xy=(0.995, y), xycoords=tr, xytext=(0, 4),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=9, color=color or CRITICAL, fontweight="bold")


def load(name):
    p = pathlib.Path(name)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _finish(fig, ax_or_axes, name, subtitle, outdir, show):
    axes = ax_or_axes if isinstance(ax_or_axes, (list, tuple)) else [ax_or_axes]
    for ax in axes:
        ax.set_axisbelow(True)
    if subtitle:
        fig.text(0.0, 1.0, subtitle, ha="left", va="bottom",
                 fontsize=10, color=INK_2, transform=fig.transFigure)
    fig.tight_layout()
    out = pathlib.Path(outdir) / name
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"  {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# --- 1. scoreboard ------------------------------------------------------------
def scoreboard(base, res, outdir, show, ens=None):
    """Magnitude across methods -> horizontal bars, one series, no legend.

    Jev is a reference line rather than a bar: it is the thing being compared
    against, not another entry in the same list.
    """
    rows = []
    if base:
        rows += [("Majority class", base["majority"], False),
                 ("Single annotator", base["annotator"], False)]
        if base.get("frozen") is not None:
            rows.append(("Frozen encoder + logreg", base["frozen"], False))
    if res:
        rows.append(("Fine-tuned model", res["accuracy"], False))
    if ens:
        # the headline configuration, so it is the one painted as ours
        best = max(ens["test"].items(), key=lambda kv: kv[1]["accuracy"])
        rows.append(("Ensemble (this repo)", best[1]["accuracy"], True))
    elif rows:
        rows[-1] = (rows[-1][0], rows[-1][1], True)
    if not rows:
        return
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(8.0, 0.62 * len(rows) + 2.0))
    ys = range(len(rows))
    colors = [S1 if mine else BASELINE for _, _, mine in rows]
    ax.barh(list(ys), [v for _, v, _ in rows], height=0.62, color=colors)

    # Values sit in their own column past every bar and past the Jev rule, so a
    # number can never land on top of the line it is being compared against.
    top = max(max(v for _, v, _ in rows), JEV_ACC)
    label_x = top * 1.06
    for y, (_, v, mine) in zip(ys, rows):
        ax.text(label_x, y, f"{v:.3f}", va="center", fontsize=11,
                color=INK if mine else INK_2,
                fontweight="bold" if mine else "normal")

    ax.axvline(JEV_ACC, color=CRITICAL, linewidth=2, linestyle=(0, (5, 3)), zorder=3)
    # Caption the rule above the bars: below the axis it lands in the tick
    # labels, and level with a bar it lands on that bar's value.
    head = len(rows) - 0.5 + 0.80
    ax.annotate(f"TypeSafe Jev {JEV_ACC:.3f}  ", xy=(JEV_ACC, len(rows) - 0.5 + 0.12),
                xytext=(-4, 0), textcoords="offset points",
                ha="right", va="bottom", color=CRITICAL, fontsize=10,
                fontweight="bold")

    ax.set_yticks(list(ys), [n for n, _, _ in rows], color=INK_2, fontsize=11)
    ax.set_ylim(-0.65, head)
    ax.set_xlim(0, top * 1.20)
    n = (res or {}).get("n_cases")
    ax.set_xlabel(f"accuracy on the {n}-case test split" if n
                  else "accuracy on the test split")
    ax.set_title("Accuracy vs TypeSafe Jev")
    ax.grid(axis="y", visible=False)
    _finish(fig, ax, "1_scoreboard.png",
            "Blue is the shipped configuration. Grey needs little or no training.",
            outdir, show)


# --- 2. training curve --------------------------------------------------------
def training(hist, outdir, show):
    """Two measures on different scales -> two panels, never two y-axes."""
    if not hist:
        return
    ep = [h["epoch"] for h in hist]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9))

    a1.plot(ep, [h["loss"] for h in hist], color=S1, marker="o",
            markeredgecolor=SURFACE, markeredgewidth=2)
    a1.set_title("Training loss")
    a1.set_xlabel("epoch")

    acc = [h["val_acc"] for h in hist]
    a2.plot(ep, acc, color=S2, marker="o",
            markeredgecolor=SURFACE, markeredgewidth=2)
    best = max(range(len(acc)), key=lambda i: acc[i])
    # Anchor the callout away from whichever edge the best epoch sits against.
    at_end = best >= len(acc) - 1
    a2.annotate(f"best {acc[best]:.3f}", (ep[best], acc[best]),
                textcoords="offset points",
                xytext=(-10 if at_end else 0, 12),
                ha="right" if at_end else "center",
                fontsize=10, color=INK, fontweight="bold")
    span = (max(acc) - min(acc)) or 0.02
    a2.set_ylim(min(acc) - span * 0.18, max(acc) + span * 0.38)
    a2.set_title("Validation accuracy")
    a2.set_xlabel("epoch")

    for ax in (a1, a2):
        ax.set_xticks(ep)
        ax.grid(axis="x", visible=False)
    _finish(fig, [a1, a2], "2_training.png",
            "Loss still falling at the last epoch means it is undertrained.",
            outdir, show)


# --- 3. reliability -----------------------------------------------------------
def reliability(res, outdir, show):
    """Calibration: two series (before/after) -> legend plus direct labels."""
    if not res or not res.get("reliability_raw"):
        return
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1.5,
            linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.97, 0.93, "perfectly calibrated", color=MUTED, fontsize=9,
            ha="right", rotation=38, rotation_mode="anchor")

    for key, colour, label in (("reliability_raw", BASELINE, "before"),
                               ("reliability_cal", S1, "after temperature")):
        pts = res.get(key)
        if not pts:
            continue
        ax.plot([p[1] for p in pts], [p[2] for p in pts], color=colour,
                marker="o", markeredgecolor=SURFACE, markeredgewidth=2,
                label=label, zorder=3 if colour == S1 else 2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence the model reported")
    ax.set_ylabel("how often it was actually right")
    ax.set_title("Reliability")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK_2)
    _finish(fig, ax, "3_reliability.png",
            "Below the diagonal = overconfident. Above = underconfident.",
            outdir, show)


# --- 4. coverage --------------------------------------------------------------
def coverage(res, outdir, show):
    """One series -> no legend; label the points that carry the decision."""
    rows = (res or {}).get("coverage")
    if not rows:
        return
    rows = [r for r in rows if r[1] > 0]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cov = [r[1] for r in rows]
    acc = [r[2] for r in rows]
    ax.plot(cov, acc, color=S1, marker="o",
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    # High thresholds crowd together at low coverage, so alternate the callout
    # above and below whenever two points sit close on the x axis.
    span = max(max(cov) - min(cov), 1e-6)
    last_x, above = None, True
    for t, c, a in sorted(rows, key=lambda r: r[1]):
        if last_x is not None and (c - last_x) / span < 0.12:
            above = not above
        else:
            above = True
        ax.annotate(f"≥{t:.2f}", (c, a), textcoords="offset points",
                    xytext=(0, 11 if above else -18), ha="center",
                    fontsize=9, color=INK_2)
        last_x = c

    overall = res.get("accuracy")
    if overall:
        ax.axhline(overall, color=BASELINE, linewidth=1.5, linestyle=(0, (4, 3)))
        hline_label(ax, overall, f"all decisions {overall:.3f}", MUTED)

    ax.set_xlabel("coverage - share of decisions kept")
    ax.set_ylabel("accuracy on the decisions kept")
    ax.set_title("Can you route on the confidence?")
    ax.set_xlim(0, 1.02)
    _finish(fig, ax, "4_coverage.png",
            "Up and to the left is the useful shape: high accuracy on a band "
            "you can autoroute, the rest escalated.", outdir, show)


# --- 5. per type --------------------------------------------------------------
def per_type(res, outdir, show):
    """Accuracy and ECE are different scales and opposite polarity -> two panels."""
    rows = (res or {}).get("per_type")
    if not rows:
        return
    names = [r["type"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    xs = range(len(names))

    a1.bar(list(xs), [r["accuracy"] for r in rows], width=0.58, color=S1)
    for x, r in zip(xs, rows):
        a1.text(x, r["accuracy"] + 0.012, f"{r['accuracy']:.3f}", ha="center",
                fontsize=10, color=INK_2)
    a1.set_title("Accuracy by question type")
    a1.set_ylim(0, 1.08)

    a2.bar(list(xs), [r["ece"] for r in rows], width=0.58, color=S2)
    for x, r in zip(xs, rows):
        a2.text(x, r["ece"] + 0.004, f"{r['ece']:.3f}", ha="center",
                fontsize=10, color=INK_2)
    a2.axhline(JEV_ECE, color=CRITICAL, linewidth=2, linestyle=(0, (5, 3)))
    hline_label(a2, JEV_ECE, f"Jev {JEV_ECE:.3f}")
    a2.set_ylim(0, max(JEV_ECE, max(r["ece"] for r in rows)) * 1.30)
    a2.set_title("Calibration error by type (lower is better)")

    for ax in (a1, a2):
        ax.set_xticks(list(xs), names, color=INK_2)
        ax.grid(axis="x", visible=False)
    _finish(fig, [a1, a2], "5_per_type.png",
            "Jev's own error runs opposite ways by type - that is why each gets "
            "its own temperature.", outdir, show)


# --- 6. ensemble ---------------------------------------------------------------
def ensemble(ens, outdir, show):
    """Two panels: the weight sweep that chose w, and what it bought on test."""
    if not ens:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ws = [c["w"] for c in ens["val_curve"]]
    accs = [c["accuracy"] for c in ens["val_curve"]]
    a1.plot(ws, accs, color=S1)
    bw = ens["best_w"]
    a1.scatter([bw], [max(accs)], color=S1, zorder=4, s=70,
               edgecolor=SURFACE, linewidth=2)
    a1.annotate(f"w={bw:.2f}", (bw, max(accs)), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=10, color=INK,
                fontweight="bold")
    a1.set_xlabel("w   (1.0 = neural model only, 0.0 = frozen probe only)")
    a1.set_ylabel("validation accuracy")
    a1.set_title("Choosing the blend weight")

    names = list(ens["test"])
    vals = [ens["test"][n]["accuracy"] for n in names]
    eces = [ens["test"][n]["ece"] for n in names]
    # Ties on accuracy break toward the better-calibrated configuration, which
    # is the one actually shipped - sharpening moves ECE, never the argmax.
    best = max(range(len(vals)), key=lambda i: (vals[i], -eces[i]))
    colors = [S1 if i == best else BASELINE for i in range(len(names))]
    a2.bar(range(len(names)), vals, width=0.58, color=colors)
    # Labels sit inside the bars: above them they collide with the Jev rule,
    # which on this chart runs only a hair above the tallest bar.
    for i, v in enumerate(vals):
        a2.text(i, v - 0.028, f"{v:.3f}", ha="center", va="top", fontsize=10,
                color=SURFACE if i == best else INK_2,
                fontweight="bold" if i == best else "normal")
        a2.text(i, v - 0.075, f"ECE {eces[i]:.3f}", ha="center", va="top",
                fontsize=8, color=SURFACE if i == best else MUTED)
    a2.axhline(JEV_ACC, color=CRITICAL, linewidth=2, linestyle=(0, (5, 3)))
    hline_label(a2, JEV_ACC, f"Jev {JEV_ACC:.3f}")
    a2.set_xticks(range(len(names)),
                  [n.strip().replace(" ", chr(10)) for n in names],
                  color=INK_2, fontsize=9)
    a2.set_ylim(0, max(max(vals), JEV_ACC) * 1.16)
    a2.set_title("Test accuracy")
    a2.grid(axis="x", visible=False)

    _finish(fig, [a1, a2], "6_ensemble.png",
            "Combining two models that fail differently, weight chosen on "
            "validation and never on test.", outdir, show)


def main(outdir="plots", show=False):
    style()
    base, hist, res = load("baselines.json"), load("history.json"), load("results.json")
    ens = load("ensemble.json")
    if not any((base, hist, res, ens)):
        raise SystemExit(
            "no run artifacts found - run 01_ceiling.py / 02_train.py / "
            "04_eval.py first; each writes the JSON this reads.")
    # Artifacts from different runs plot happily side by side and the chart
    # gives no hint that the bars are not comparable.
    cfgs = {k: v.get("config") for k, v in (("baselines", base), ("results", res))
            if v and v.get("config")}
    if len(set(cfgs.values())) > 1:
        print(f"  ! mixed runs: {cfgs} - the scoreboard compares bars measured on "
              "different data. Rerun 01_ceiling.py and 04_eval.py with the same "
              "--config before quoting it.")
    if res and res.get("n_cases") and res["n_cases"] < 400 and res.get("config") == "all":
        print(f"  ! results.json covers only {res['n_cases']} of the 400 test "
              "cases (--limit was set) - not comparable to Jev's published number.")

    print("charts written:")
    scoreboard(base, res, outdir, show, ens)
    training(hist, outdir, show)
    reliability(res, outdir, show)
    coverage(res, outdir, show)
    per_type(res, outdir, show)
    ensemble(ens, outdir, show)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="plots")
    p.add_argument("--show", action="store_true")
    a = p.parse_args()
    main(a.outdir, a.show)
