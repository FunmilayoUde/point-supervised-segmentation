"""The experiment grid: label budget against sampling strategy.

Two factors: how many points a tile receives, and how they
are drawn. Every run trains on Train/Urban point labels and is scored against
held-out Val/Urban DENSE masks, because the question is how far point
supervision recovers the full segmentation, not how tightly it fits its own
points.

The anchors matter more than the grid. A fully supervised run sets the ceiling
and a majority-class predictor sets the floor; an mIoU quoted without both says
nothing about what the point budget actually cost.

Model selection is on dense validation mIoU and never on training loss. The single-tile check in train.py
showed why: twenty points drove training loss to 0.0002 while dense mIoU sat at
0.237, so training loss announces convergence long after generalisation has
stopped improving.

Results append to results.csv one row at a time, and a finished
configuration is skipped on a rerun, so the sweep survives a closed lid.
"""

import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import LoveDAPoints
from losses import partial_focal_ce
from loveda import CLASSES, OUTPUT, index_tiles
from metrics import ConfusionMatrix
from train import build_model, device, evaluate, freeze_batchnorm


ARCH = "mobilenet"
TRAIN_TILES = 400
VAL_TILES = 250
EPOCHS = 12
BATCH = 4
LR = 3e-4
GAMMA = 2.0  # The brief's formula is focal; gamma=0 would give plain partial CE.

BUDGETS = (5, 20, 50, 200)
STRATEGIES = ("uniform", "stratified")
SEEDS = (0, 1, 2)
SPLIT_SEED = 1234  # Fixes which tiles are held out, independent of run seed.
RESULTS = OUTPUT / "results.csv"


def splits(train_tiles=TRAIN_TILES, val_tiles=VAL_TILES, seed=SPLIT_SEED):
    """Disjoint training and validation tiles, both drawn from Train/Urban.

    LoveDA's official Val/Urban is deliberately not used here. It runs 26%
    agriculture against 2% in Train/Urban, because the published splits are
    geographically disjoint to serve the domain-adaptation benchmark. That is
    fatal for this experiment: a gap between two point budgets would be swamped
    by a distribution shift that has nothing to do with either factor. Holding
    out a random slice of Train/Urban keeps train and validation on the same
    distribution, so the only things varying are the two factors under study.
    """
    pairs = index_tiles(split="Train", scene="Urban")
    order = np.random.default_rng(seed).permutation(len(pairs))
    train = [pairs[i] for i in order[:train_tiles]]
    val = [pairs[i] for i in order[train_tiles:train_tiles + val_tiles]]
    return train, val


def majority_floor(val_pairs):
    """Scores for always predicting the commonest class.

    Not a model, just the number a run has to beat to have learned anything.
    One class scores its own frequency as IoU and the other six score zero, so
    the floor sits far below any working segmenter.
    """
    matrix = ConfusionMatrix()
    loader = DataLoader(LoveDAPoints(val_pairs, supervision="dense"), batch_size=BATCH)
    counts = np.zeros(len(CLASSES), dtype=np.int64)
    for _, targets in loader:
        flat = targets.numpy().ravel()
        counts += np.bincount(flat[flat != 255], minlength=len(CLASSES))
    commonest = int(counts.argmax())

    for _, targets in loader:
        matrix.update(np.full(targets.shape, commonest), targets.numpy())
    return matrix.report()


def run_one(strategy, points, seed, train_pairs, val_pairs, dev, epochs=EPOCHS):
    """Train one configuration, returning the best dense-validation scores."""
    torch.manual_seed(seed)
    model = build_model(ARCH).to(dev)

    supervision = "dense" if strategy == "dense" else "points"
    train_set = LoveDAPoints(
        train_pairs, supervision=supervision, n_points=points,
        strategy=strategy if supervision == "points" else "uniform", seed=seed,
    )
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(
        LoveDAPoints(val_pairs, supervision="dense"), batch_size=BATCH
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR)

    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        freeze_batchnorm(model)
        for images, targets in train_loader:
            loss = partial_focal_ce(
                model(images.to(dev))["out"], targets.to(dev), gamma=GAMMA
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

        scores = evaluate(model, val_loader, dev).report()
        if best is None or scores["mIoU"] > best["mIoU"]:
            best = dict(scores, best_epoch=epoch)
    return best


def completed(path=RESULTS):
    """Configurations already present in the results file."""
    if not path.exists():
        return set()
    done = pd.read_csv(path)
    return set(zip(done.strategy, done.points, done.seed))


COLUMNS = ["strategy", "points", "seed", "seconds", "best_epoch", *CLASSES, "mIoU"]


def append(row, path=RESULTS):
    """Write one result immediately, so a crash costs one run and not the sweep.

    The column order is pinned rather than inherited from the dict. A headerless
    append writes whatever order the dict happens to carry, so rows built by
    different call sites file their values under each other's headings and the
    corruption is invisible until a metric reads back out of range.
    """
    path.parent.mkdir(exist_ok=True)
    frame = pd.DataFrame([row])[COLUMNS]
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def grid():
    """Anchors first, so the ceiling is known before the sweep is interpreted."""
    plan = [("dense", 0, seed) for seed in SEEDS]
    plan += [
        (strategy, budget, seed)
        for strategy in STRATEGIES
        for budget in BUDGETS
        for seed in SEEDS
    ]
    return plan


if __name__ == "__main__":
    dev = device()
    train_pairs, val_pairs = splits()
    done = completed()

    print(f"device {dev}  arch {ARCH}  gamma {GAMMA}")
    print(f"train {len(train_pairs)} tiles  val {len(val_pairs)} tiles  {EPOCHS} epochs")

    if ("majority", 0, -1) not in done:
        floor = majority_floor(val_pairs)
        append(dict(strategy="majority", points=0, seed=-1, best_epoch=0,
                    seconds=0.0, **floor))
        print(f"majority-class floor: mIoU {floor['mIoU']:.4f}")

    plan = [job for job in grid() if job not in done]
    print(f"{len(plan)} runs to go ({len(done)} already recorded)\n")

    for index, (strategy, points, seed) in enumerate(plan, start=1):
        started = time.time()
        best = run_one(strategy, points, seed, train_pairs, val_pairs, dev)
        elapsed = time.time() - started
        append(dict(strategy=strategy, points=points, seed=seed,
                    seconds=round(elapsed, 1), **best))
        print(
            f"[{index:>2}/{len(plan)}] {strategy:<11} N={points:<4} seed={seed}  "
            f"mIoU {best['mIoU']:.4f}  epoch {best['best_epoch']:>2}  "
            f"{elapsed/60:.1f} min"
        )
