
"""
Weighted-Gini Decision-Tree   

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

    
    def _gini_index(self, y: np.ndarray) -> float:  
        counts = np.bincount(y, minlength=self.L.shape[0])
        p = counts / counts.sum()
        
        return float(np.sum(self.L * np.outer(p, p)))

  

    def _best_split(  
        self, X: np.ndarray, y: np.ndarray, cat_columns_dict: Dict[int, bool]
    ) -> Tuple[float, Optional[int], Optional[float | str]]:
        best_gini, best_col, best_val = np.inf, None, None
        for col, categorical in cat_columns_dict.items():
            if len(np.unique(X[:, col])) < 2:
                continue
            gini, val = self._gini_split_value(X, y, col, categorical) 
            if gini < best_gini:
                best_gini, best_col, best_val = gini, col, val
        return best_gini, best_col, best_val
