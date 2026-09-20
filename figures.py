"""The two figures the report is built around.

Figure 1 answers the question the experiment was designed to ask: what does a
point budget buy, and how does the answer depend on how the points are drawn.
The fully supervised ceiling and the majority-class floor are drawn on the same
axes, because a mIoU quoted without them cannot be read.

Figure 2 breaks the same runs down by class, which is where the aggregate hides
its story: the two strategies trade classes against each other rather than one
dominating, and agriculture is weak even under full supervision.

Colours are fixed per entity and carry a distinct marker as well, so identity
survives greyscale printing and colour-vision deficiency. Error bars are the
standard deviation over the three seeds.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loveda import CLASSES, OUTPUT


RESULTS = OUTPUT / "results.csv"
BUDGETS = (5, 20, 50, 200)

UNIFORM, STRATIFIED, DENSE = "#2a78d6", "#eb6834", "#1baf7a"
FLOOR_INK, GRID_INK = "#8a8a80", "#e2e2dc"
TEXT, MUTED = "#1a1a19", "#5c5c55"
STYLE = {"uniform": (UNIFORM, "o"), "stratified": (STRATIFIED, "s")}


def load(path=RESULTS):
    """Results split into the sweep, the ceiling and the floor."""
    frame = pd.read_csv(path)
    sweep = frame[frame.strategy.isin(STYLE)]
    ceiling = frame[frame.strategy == "dense"]
    floor = float(frame[frame.strategy == "majority"].mIoU.iloc[0])
    return sweep, ceiling, floor


def budget_curve(sweep, ceiling, floor, out_path=None):
    """mIoU against points per tile, both strategies, with the two anchors."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    top, spread = ceiling.mIoU.mean(), ceiling.mIoU.std()
    ax.axhspan(top - spread, top + spread, color=DENSE, alpha=0.10, zorder=0)
    ax.axhline(top, color=DENSE, lw=2, ls="--", zorder=1)
    ax.text(5.2, top + 0.012, f"fully supervised  {top:.3f}", color=MUTED,
            fontsize=9, va="bottom")

    ax.axhline(floor, color=FLOOR_INK, lw=1.5, ls=":", zorder=1)
    ax.text(5.2, floor + 0.012, f"majority class  {floor:.3f}", color=MUTED,
            fontsize=9, va="bottom")

    for name, (colour, marker) in STYLE.items():
        block = sweep[sweep.strategy == name].groupby("points").mIoU
        mean, error = block.mean(), block.std()
        ax.errorbar(mean.index, mean.values, yerr=error.values, color=colour,
                    marker=marker, markersize=8, lw=2, capsize=4, capthick=1.5,
                    elinewidth=1.5, label=name, zorder=3,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.2)

    ax.set_xscale("log")
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xlabel("labelled points per tile  (of 65,536 pixels)", color=TEXT)
    ax.set_ylabel("mIoU on held-out dense masks", color=TEXT)
    ax.set_title("What a point budget buys", color=TEXT, fontsize=13, pad=12)
    ax.set_ylim(0, top + 0.10)
    ax.grid(True, color=GRID_INK, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_INK)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.10),
              labelcolor=TEXT)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    return fig


def class_breakdown(sweep, ceiling, uniform_at=200, stratified_at=200, out_path=None):
    """Per-class IoU for both strategies at the same budget, against the ceiling.

    Holding the budget fixed is what isolates the strategy: comparing each
    strategy at its own best budget would confound how the points were drawn
    with how many there were.

    Classes are ordered by how well full supervision does on them, which puts
    the intrinsically hard ones at the bottom and makes it obvious when a gap
    belongs to the class rather than to the supervision.
    """
    rows = {
        "fully supervised": (ceiling[list(CLASSES)].mean(), DENSE, "^"),
        f"uniform N={uniform_at}": (
            sweep[(sweep.strategy == "uniform") & (sweep.points == uniform_at)][
                list(CLASSES)].mean(), UNIFORM, "o"),
        f"stratified N={stratified_at}": (
            sweep[(sweep.strategy == "stratified") & (sweep.points == stratified_at)][
                list(CLASSES)].mean(), STRATIFIED, "s"),
    }
    order = rows["fully supervised"][0].sort_values().index
    positions = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    lo = np.minimum.reduce([values[order].values for values, _, _ in rows.values()])
    hi = np.maximum.reduce([values[order].values for values, _, _ in rows.values()])
    ax.hlines(positions, lo, hi, color=GRID_INK, lw=2, zorder=1)

    # The ceiling is drawn hollow and oversized rather than filled. Where a
    # strategy matches it exactly -- stratified on agriculture does -- a filled
    # marker would sit on top and erase the very result worth seeing.
    for label, (values, colour, marker) in rows.items():
        if label == "fully supervised":
            ax.scatter(values[order].values, positions, facecolors="none",
                       edgecolors=colour, marker=marker, s=190, linewidths=2.2,
                       label=label, zorder=4)
        else:
            ax.scatter(values[order].values, positions, color=colour,
                       marker=marker, s=80, label=label, zorder=3,
                       edgecolors="#fcfcfb", linewidths=1.2)

    ax.set_yticks(positions)
    ax.set_yticklabels(order, color=TEXT)
    ax.set_xlabel("IoU on held-out dense masks", color=TEXT)
    ax.set_title("Which classes point supervision recovers", color=TEXT,
                 fontsize=13, pad=12)
    ax.grid(True, axis="x", color=GRID_INK, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID_INK)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="lower right", labelcolor=TEXT)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    return fig


if __name__ == "__main__":
    sweep, ceiling, floor = load()
    budget_curve(sweep, ceiling, floor, OUTPUT / "fig1_budget_curve.png")
    class_breakdown(sweep, ceiling, out_path=OUTPUT / "fig2_class_breakdown.png")
    print(f"wrote fig1_budget_curve.png, fig2_class_breakdown.png to {OUTPUT}")
