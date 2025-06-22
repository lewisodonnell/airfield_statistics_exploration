# models/DT_weighted.py
"""
Weighted-Gini Decision-Tree   (Task 1.4 – MSc extension)

* Inherits everything from our plain DecisionTreeClassifier.
* Accepts a 4×4 loss-matrix L (np.ndarray).
* Overrides the Gini calculation so each split minimises the weighted
  impurity   Σ_{q≠q'} L_{qq'} π_q π_{q'}   as required.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from models.DT import DecisionTreeClassifier


class WeightedDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(
        self,
        cat_columns_dict: Dict[int, bool],
        loss_matrix: np.ndarray,
        max_depth: int = 10,
        min_samples_leaf: int = 12,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            cat_columns_dict,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        if loss_matrix.shape[0] != loss_matrix.shape[1]:
            raise ValueError("loss_matrix must be square.")
        self.L = loss_matrix.astype(float)

    # ───────────────────────── override the impurity calc ───────────────────
    def _gini_index(self, y: np.ndarray) -> float:  # type: ignore[override]
        counts = np.bincount(y, minlength=self.L.shape[0])
        p = counts / counts.sum()
        # Σ_{q,q'} L_q,q'  p_q  p_q'
        return float(np.sum(self.L * np.outer(p, p)))

    # ───────────────────────── override the split routine ───────────────────
    # we replace DecisionTreeClassifier._best_split with a version that calls
    # our new _gini_index.  Everything else is identical.

    def _best_split(  # type: ignore[override]
        self, X: np.ndarray, y: np.ndarray, cat_columns_dict: Dict[int, bool]
    ) -> Tuple[float, Optional[int], Optional[float | str]]:
        best_gini, best_col, best_val = np.inf, None, None
        for col, categorical in cat_columns_dict.items():
            if len(np.unique(X[:, col])) < 2:
                continue
            gini, val = self._gini_split_value(X, y, col, categorical)  # type: ignore
            if gini < best_gini:
                best_gini, best_col, best_val = gini, col, val
        return best_gini, best_col, best_val
