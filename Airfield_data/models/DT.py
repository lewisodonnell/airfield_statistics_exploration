# models/DT.py
"""
Decision-Tree classifier (hard & soft outputs) with *real* Gini feature-importance.

Key points
----------
* `predict_proba` returns class probabilities.
* `feature_importances_` is populated exactly as in the notebook:
      importance(j) = Σ_nodes p_t · ΔGini_t   where  p_t = N_t / N_root
* `feature_importances_` is scaled so the maximum feature gets value 1
  (makes percentage plots easy – just multiply by 100).
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from models.base import BaseModel


# ──────────────────────────── helper funcs ────────────────────────────────
def _gini_index(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p ** 2)


def _split_samples(
    X: np.ndarray, y: np.ndarray, column: int, value, categorical: bool
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    if categorical:
        mask = X[:, column] == value
    else:
        mask = X[:, column] < value
    return (X[mask], y[mask]), (X[~mask], y[~mask])


def _gini_split_value(X: np.ndarray, y: np.ndarray, column: int, categorical: bool):
    unique_vals = np.unique(X[:, column])
    best_gini, best_val = np.inf, None
    for v in unique_vals:
        (X_l, y_l), (X_r, y_r) = _split_samples(X, y, column, v, categorical)
        if len(y_l) == 0 or len(y_r) == 0:
            continue
        p_l = len(y_l) / len(y)
        p_r = 1.0 - p_l
        g = p_l * _gini_index(y_l) + p_r * _gini_index(y_r)
        if g < best_gini:
            best_gini, best_val = g, v
    return best_gini, best_val


# ───────────────────────── Decision-Tree class ────────────────────────────
class DecisionTreeClassifier(BaseModel):
    def __init__(
        self,
        cat_columns_dict: Dict[int, bool],
        max_depth: int = 10,
        min_samples_leaf: int = 12,
        random_state: Optional[int] = None,
    ):
        self.cat_columns_dict = cat_columns_dict
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

        self.n_classes_: int = 0
        self.feature_names_: List[str] = []
        self._N_total: int = 0
        self.feature_importances_: np.ndarray | None = None
        self.tree_: Dict[str, Any] | None = None

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ):
        X_arr, y_arr = self._check_Xy(X, y)
        self._N_total = len(y_arr)
        self.n_classes_ = int(y_arr.max()) + 1
        self.feature_names_ = (
            X.columns.tolist()
            if isinstance(X, pd.DataFrame)
            else [f"x{i}" for i in range(X_arr.shape[1])]
        )
        self.feature_importances_ = np.zeros(X_arr.shape[1])

        self.tree_ = self._build_tree(X_arr, y_arr, depth=1)

        # scale importances so max = 1  (same convention you used in notebook)
        max_val = self.feature_importances_.max()
        if max_val > 0:
            self.feature_importances_ /= max_val
        return self

    # ------------------------------------------------------- predict_proba
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        return np.vstack([self._classify_soft(self.tree_, x) for x in X_arr])

    # ----------------------------------------------------------- predict
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    # ───────────────────── internal recursive builder ─────────────────────
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        # stopping conditions
        if (
            len(np.unique(y)) == 1
            or depth > self.max_depth
            or len(X) <= self.min_samples_leaf
        ):
            return {"counts": np.bincount(y, minlength=self.n_classes_)}

        # pick best split
        best_g, best_col, best_val = np.inf, None, None
        for col, categorical in self.cat_columns_dict.items():
            if len(np.unique(X[:, col])) < 2:
                continue
            g, v = _gini_split_value(X, y, col, categorical)
            if g < best_g:
                best_g, best_col, best_val = g, col, v

        if best_col is None:
            return {"counts": np.bincount(y, minlength=self.n_classes_)}

        categorical = self.cat_columns_dict[best_col]
        (X_l, y_l), (X_r, y_r) = _split_samples(X, y, best_col, best_val, categorical)

        # ─── accumulate feature importance (Δ Gini) ───
        g_parent = _gini_index(y)
        g_left = _gini_index(y_l)
        g_right = _gini_index(y_r)
        p_left = len(y_l) / len(y)
        p_right = 1.0 - p_left
        delta_g = g_parent - (p_left * g_left + p_right * g_right)
        p_node = len(y) / self._N_total
        self.feature_importances_[best_col] += p_node * delta_g

        return {
            "feature_index": best_col,
            "feature_name": self.feature_names_[best_col],
            "value": best_val,
            "categorical": categorical,
            "left": self._build_tree(X_l, y_l, depth + 1),
            "right": self._build_tree(X_r, y_r, depth + 1),
        }

    # ─────────────────────────────── classify helper ───────────────────────
    def _classify_soft(self, node: Dict[str, Any], x: np.ndarray) -> np.ndarray:
        if "counts" in node:  # leaf
            counts = node["counts"]
            return counts / counts.sum()
        col, val, categorical = (
            node["feature_index"],
            node["value"],
            node["categorical"],
        )
        branch = "left" if (x[col] == val if categorical else x[col] < val) else "right"
        return self._classify_soft(node[branch], x)
