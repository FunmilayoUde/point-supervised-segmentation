"""Turning dense masks into simulated point labels.

Draws a small budget of pixels per tile and discards the rest of the mask, which
is the supervision a real point annotator would have produced. Output keeps the
convention loveda.to_train_ids established: a full-size map holding true class
ids at sampled pixels and IGNORE everywhere else, so the loss in losses.py consumes a
point map and a dense mask through exactly the same code path.

Two strategies are implemented because the choice between them is one of the
experimental factors: uniform sampling ignores class balance, stratified splits
the budget across the classes a tile actually contains.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loveda import (
    CLASSES,
    IGNORE,
    OUTPUT,
    PALETTE,
    index_tiles,
    read_image,
    read_mask,
    to_train_ids,
)


SEED = 0
BUDGETS = (5, 10, 20, 50, 100)
STRATEGIES = ("uniform", "stratified")
COVERAGE_TILES = 200


def tile_rng(seed, index):
    """An independent, reproducible random stream per tile.

    Seeding on the pair means run N of the experiment is repeatable, while two
    tiles in the same run never receive correlated draws.
    """
    return np.random.default_rng([seed, index])


def sample_points(labels, n_points, strategy="uniform", rng=None):
    """Return a point map: true ids at sampled pixels, IGNORE elsewhere."""
    rng = np.random.default_rng() if rng is None else rng
    flat = labels.ravel()
    valid = np.flatnonzero(flat != IGNORE)

    if valid.size == 0:
        return np.full_like(labels, IGNORE)
    if strategy == "uniform":
        chosen = _uniform(valid, n_points, rng)
    elif strategy == "stratified":
        chosen = _stratified(flat, valid, n_points, rng)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    points = np.full(flat.size, IGNORE, dtype=np.uint8)
    points[chosen] = flat[chosen]
    return points.reshape(labels.shape)


def _uniform(valid, n_points, rng):
    """Draw pixel indices uniformly from the valid pixels, without replacement."""
    return rng.choice(valid, size=min(n_points, valid.size), replace=False)


def _stratified(flat, valid, n_points, rng):
    """Split the budget across the classes this tile actually contains.

    A class absent from the tile cannot be sampled, so stratification cures
    within-tile imbalance only. Whether a rare class appears in a tile at all
    is a property of the data that no sampling rule can change, and the gap
    between those two effects is what the coverage table below measures.
    """
    present = np.unique(flat[valid])
    quota = np.full(present.size, n_points // present.size)
    quota[rng.permutation(present.size)[: n_points % present.size]] += 1

    chosen = []
    for cls, budget in zip(present, quota):
        pool = valid[flat[valid] == cls]
        chosen.append(rng.choice(pool, size=min(budget, pool.size), replace=False))
    return np.concatenate(chosen)


def class_coverage(pairs, n_points, strategy, seed=SEED, limit=COVERAGE_TILES):
    """Per class, the fraction of tiles that are labelled and that are present.

    "Labelled" is the fraction of tiles where sampling produced at least one
    point of the class; "present" is the fraction where the class exists in the
    dense mask at all. Present is the ceiling that labelled can reach.
    """
    labelled = np.zeros(len(CLASSES), dtype=np.int64)
    present = np.zeros(len(CLASSES), dtype=np.int64)

    for index, (_, mask_path) in enumerate(pairs[:limit]):
        labels = to_train_ids(read_mask(mask_path))
        points = sample_points(labels, n_points, strategy, tile_rng(seed, index))
        labelled[np.unique(points[points != IGNORE])] += 1
        present[np.unique(labels[labels != IGNORE])] += 1

    tiles = min(limit, len(pairs))
    return labelled / tiles, present / tiles


def show_points(image_path, mask_path, n_points=20, strategy="uniform", seed=SEED,
                out_path=None):
    """Contrast the dense mask against the handful of pixels actually kept."""
    image = read_image(image_path)
    labels = to_train_ids(read_mask(mask_path))
    points = sample_points(labels, n_points, strategy, np.random.default_rng(seed))

    rows, cols = np.nonzero(points != IGNORE)
    colours = PALETTE[points[rows, cols]] / 255

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    axes[0].imshow(image)
    axes[0].set_title(image_path.name, fontsize=10)

    axes[1].imshow(PALETTE[np.where(labels == IGNORE, len(CLASSES), labels)])
    axes[1].set_title("dense mask (discarded)", fontsize=10)

    axes[2].imshow(image)
    axes[2].scatter(cols, rows, c=colours, s=70, edgecolors="black", linewidths=1.2)
    axes[2].set_title(f"{len(rows)} {strategy} points (what trains)", fontsize=10)

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    pairs = index_tiles()
    rows = []

    for strategy in STRATEGIES:
        for budget in BUDGETS:
            labelled, present = class_coverage(pairs, budget, strategy)
            for name, got, ceiling in zip(CLASSES, labelled, present):
                rows.append(
                    {
                        "strategy": strategy,
                        "points_per_tile": budget,
                        "class": name,
                        "tiles_labelled": got,
                        "tiles_present": ceiling,
                    }
                )

    coverage = pd.DataFrame(rows)
    ceilings = coverage.groupby("class").tiles_present.first()

    print(f"--- tiles with at least one point of each class ({COVERAGE_TILES} tiles) ---")
    print("  'present' is the ceiling: the class is in the dense mask at all.\n")
    for strategy in STRATEGIES:
        block = coverage[coverage.strategy == strategy]
        wide = block.pivot(index="class", columns="points_per_tile",
                          values="tiles_labelled")
        wide = wide.reindex(ceilings.sort_values().index)
        header = "  ".join(f"N={n:<5}" for n in BUDGETS)
        print(f"  [{strategy}]")
        print(f"  {'class':<12} {'present':>7}  {header}")
        for name, row in wide.iterrows():
            cells = "  ".join(f"{row[n]:6.1%} " for n in BUDGETS)
            print(f"  {name:<12} {ceilings[name]:7.1%}  {cells}")
        print()

    OUTPUT.mkdir(exist_ok=True)
    coverage.to_csv(OUTPUT / "coverage.csv", index=False)
    show_points(*pairs[0], n_points=20, strategy="uniform",
                out_path=OUTPUT / "points_uniform.png")
    show_points(*pairs[0], n_points=20, strategy="stratified",
                out_path=OUTPUT / "points_stratified.png")
    print(f"wrote coverage.csv, points_*.png to {OUTPUT}")
