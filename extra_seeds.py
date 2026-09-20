"""Additional seeds at N=50, where the curve was ambiguous.

Uniform at N=50 spanned 0.177 to 0.397 across three seeds, a spread wider than
the gap between N=20 and N=200. That single low run is the only reason uniform
looks flat across the middle of the curve, and three seeds cannot say whether it
was an unlucky sampling draw or ordinary noise. Both strategies get the extra
seeds so the comparison at N=50 stays balanced.

Rows append to the same results file and the sweep's own skip logic applies, so
this can be rerun safely.
"""

import time

from experiment import RESULTS, append, completed, run_one, splits
from train import device


BUDGET = 50
EXTRA_SEEDS = (3, 4, 5)
STRATEGIES = ("uniform", "stratified")


if __name__ == "__main__":
    dev = device()
    train_pairs, val_pairs = splits()
    done = completed()

    plan = [
        (strategy, BUDGET, seed)
        for strategy in STRATEGIES
        for seed in EXTRA_SEEDS
        if (strategy, BUDGET, seed) not in done
    ]
    print(f"{len(plan)} extra runs at N={BUDGET}")

    for index, (strategy, points, seed) in enumerate(plan, start=1):
        started = time.time()
        best = run_one(strategy, points, seed, train_pairs, val_pairs, dev)
        elapsed = time.time() - started
        append(dict(strategy=strategy, points=points, seed=seed,
                    seconds=round(elapsed, 1), **best))
        print(f"[{index}/{len(plan)}] {strategy:<11} N={points} seed={seed}  "
              f"mIoU {best['mIoU']:.4f}  epoch {best['best_epoch']:>2}  "
              f"{elapsed/60:.1f} min")
