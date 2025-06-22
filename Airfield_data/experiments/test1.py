from __future__ import annotations

import json
import pathlib
from typing import Dict

import numpy as np

from data.loader import load_airfield_statistics
from data.preprocessing import StandardScaler
from metrics import accuracy_score
from models.DT import DecisionTreeClassifier
from models.KNN import KNNClassifier

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------

def _print_line(model_name: str, train_acc: float, test_acc: float) -> None:
    print(f"{model_name:>20} | train: {train_acc:6.3f} | test: {test_acc:6.3f}")


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main() -> Dict[str, float]:
    # 1. data ----------------------------------------------------------------
    X_train, y_train, X_test, y_test, cat_columns_dict = load_airfield_statistics()

    # 2. Decision‑tree --------------------------------------------------------
    dt = DecisionTreeClassifier(
        cat_columns_dict=cat_columns_dict,
        max_depth=10,
        min_samples_leaf=12,
        random_state=0,
    ).fit(X_train, y_train)

    dt_train_acc = dt.score(X_train, y_train)
    dt_test_acc = dt.score(X_test, y_test)

    # 3. k‑NN (raw) -----------------------------------------------------------
    knn_raw = KNNClassifier(k=25).fit(X_train, y_train)
    knn_raw_train_acc = knn_raw.score(X_train, y_train)
    knn_raw_test_acc = knn_raw.score(X_test, y_test)

    # 4. k‑NN (standardised) --------------------------------------------------
    scaler = StandardScaler().fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    knn_std = KNNClassifier(k=25).fit(X_train_std, y_train)
    knn_std_train_acc = knn_std.score(X_train_std, y_train)
    knn_std_test_acc = knn_std.score(X_test_std, y_test)

    # 5. Display --------------------------------------------------------------
    print("\n=== Task 1.1 accuracy scores ===")
    _print_line("Decision‑Tree", dt_train_acc, dt_test_acc)
    _print_line("k‑NN (raw)", knn_raw_train_acc, knn_raw_test_acc)
    _print_line("k‑NN (std)", knn_std_train_acc, knn_std_test_acc)

    # 6. Persist for README ---------------------------------------------------
    metrics = {
        "dt_train": dt_train_acc,
        "dt_test": dt_test_acc,
        "knn_raw_train": knn_raw_train_acc,
        "knn_raw_test": knn_raw_test_acc,
        "knn_std_train": knn_std_train_acc,
        "knn_std_test": knn_std_test_acc,
    }

    with (RESULTS_DIR / "task1_metrics.json").open("w") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


if __name__ == "__main__":
    np.random.seed(0)
    main()