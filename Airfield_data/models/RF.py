
"""
Random-Forest built on DecisionTreeClassifier.
"""

from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from models.base import BaseModel
from models.DT import DecisionTreeClassifier


class RandomForestClassifier(BaseModel):
    def __init__(
        self,
        cat_columns_dict: Dict[int, bool],
        n_estimators: int = 20,
        max_depth: int = 10,
        min_samples_leaf: int = 12,
        max_features: str | int | float = "sqrt",
        random_state: Optional[int] = None,
    ):
        self.cat_columns_dict = cat_columns_dict
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

        self.estimators_: List[DecisionTreeClassifier] = []
        self.feature_importances_: np.ndarray | None = None

    
    def fit(self, X, y):
        X_arr, y_arr = self._check_Xy(X, y)
        n_samples, n_features = X_arr.shape
        m = self._resolve_max_features(n_features)

        self.feature_importances_ = np.zeros(n_features)
        self.estimators_.clear()

        for _ in range(self.n_estimators):
            # bootstrap
            idx = np.random.randint(0, n_samples, n_samples)
            X_boot, y_boot = X_arr[idx], y_arr[idx]

            # train tree
            dt = DecisionTreeClassifier(
                self.cat_columns_dict,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=None,
            )
            
            dt._n_subspace_features = m 
            dt.fit(X_boot, y_boot)
            self.estimators_.append(dt)

            
            self.feature_importances_ += dt.feature_importances_

        self.feature_importances_ /= self.n_estimators
        
        max_val = self.feature_importances_.max()
        if max_val > 0:
            self.feature_importances_ /= max_val
        return self

    
    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.estimators_], axis=0)

    
    def predict(self, X):
        preds = np.vstack([t.predict(X) for t in self.estimators_]).T
        return np.apply_along_axis(lambda row: np.bincount(row).argmax(), 1, preds)

    def _resolve_max_features(self, n_feats: int) -> int:
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_feats)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_feats)))
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_feats))
        if isinstance(self.max_features, int):
            return max(1, min(self.max_features, n_feats))
        return n_feats
