"""How often does a uniform point budget miss a class?

Reads the class balance written by loveda.py and turns it into the number that
actually drives the sampling design: for a budget of N points per tile, the chance a tile
receives no point at all for a given class. Rare classes are invisible to the
loss on any tile where they are unlabelled, so this is the argument for
stratified sampling, made before a single model is trained.
"""

import numpy as np
import pandas as pd

from loveda import OUTPUT


BALANCE_CSV = OUTPUT / "class_balance.csv"
BUDGETS = (5, 10, 20, 50, 100)


def valid_class_shares(path=BALANCE_CSV):
    """Class shares renormalised over valid pixels, no-data dropped.

    No-data is already IGNORE by the time points are drawn, so it is never a
    sampling candidate and must leave the denominator too.
    """
    balance = pd.read_csv(path)
    classes = balance[balance["class"] != "no-data"].copy()
    classes["share"] = classes.pixels / classes.pixels.sum()
    return classes.sort_values("share").reset_index(drop=True)


def miss_probability(shares, budgets=BUDGETS):
    """P(a tile draws zero points of a class) under uniform sampling.

    Sampling N pixels independently, each lands on class c with probability
    share_c, so all N missing it is (1 - share_c) ** N.
    """
    table = pd.DataFrame({"class": shares["class"], "share": shares.share})
    for n in budgets:
        table[f"N={n}"] = (1 - shares.share) ** n
    return table


if __name__ == "__main__":
    shares = valid_class_shares()
    table = miss_probability(shares)

    print("--- share of valid (non no-data) pixels ---")
    for _, row in shares.iterrows():
        print(f"  {row['class']:<12} {row.share:7.2%}")

    print("\n--- P(tile receives zero points of this class) ---")
    header = "  ".join(f"N={n:<5}" for n in BUDGETS)
    print(f"  {'class':<12} {'share':>7}  {header}")
    for _, row in table.iterrows():
        cells = "  ".join(f"{row[f'N={n}']:6.1%} " for n in BUDGETS)
        print(f"  {row['class']:<12} {row.share:7.2%}  {cells}")

    print("\n--- expected points per tile on the rarest class ---")
    rarest = shares.iloc[0]
    for n in BUDGETS:
        print(f"  N={n:<4} -> {n * rarest.share:5.2f} points of {rarest['class']}")

    table.to_csv(OUTPUT / "sampling_budget.csv", index=False)
    print(f"\nwrote sampling_budget.csv to {OUTPUT}")
