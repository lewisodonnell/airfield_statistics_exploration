# experiments/task1_4.py
"""
Task 1.4 – Weighted-Gini Decision-Tree & pair-wise accuracy table.
Run:
    python -m experiments.task1_4
"""

import itertools
from pathlib import Path
import numpy as np

from data.loader import load_airfield_statistics
from models.DT import DecisionTreeClassifier           # standard
from models.DT_weighted import WeightedDecisionTreeClassifier  # new


LOSS_MATRIX = np.array(
    [[0.0, 0.1, 1.0, 1.0],
     [0.1, 0.0, 1.0, 1.0],
     [1.0, 1.0, 0.0, 1.0],
     [1.0, 1.0, 1.0, 0.0]], dtype=float
)

PAIR_LIST = list(itertools.combinations(range(4), 2))
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def pairwise_accuracy(tree, X, y):
    y_pred = tree.predict(X)
    acc = []
    for c1, c2 in PAIR_LIST:
        mask = (y == c1) | (y == c2)
        acc.append(float(np.mean(y_pred[mask] == y[mask])))
    return np.array(acc)


def main():
    X_tr, y_tr, X_te, y_te, cats = load_airfield_statistics()

    # --- vanilla tree (from Task 1.1) -------------------------------------
    dt_std = DecisionTreeClassifier(cats, random_state=0).fit(X_tr, y_tr)

    # --- weighted-Gini tree ----------------------------------------------
    dt_wgt = WeightedDecisionTreeClassifier(
        cats, loss_matrix=LOSS_MATRIX, random_state=0
    ).fit(X_tr, y_tr)

    # --- accuracies -------------------------------------------------------
    std_pair = pairwise_accuracy(dt_std, X_te, y_te)
    wgt_pair = pairwise_accuracy(dt_wgt, X_te, y_te)

    print("\nPair-wise accuracies (test set)")
    print(" pair | weighted | standard | Δ")
    for (c1, c2), wp, sp in zip(PAIR_LIST, wgt_pair, std_pair):
        print(f"({c1},{c2}) | {wp:.3f}    | {sp:.3f}   | {wp-sp:+.3f}")

    # --- save to JSON -----------------------------------------------------
    import json
    out = {
        "weighted": {f"{c1}_{c2}": round(float(v), 4) for (c1, c2), v in zip(PAIR_LIST, wgt_pair)},
        "standard": {f"{c1}_{c2}": round(float(v), 4) for (c1, c2), v in zip(PAIR_LIST, std_pair)},
    }
    with open(RESULTS_DIR / "task1_4_pairwise.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nJSON saved ➜ {RESULTS_DIR/'task1_4_pairwise.json'}")


if __name__ == "__main__":
    main()
