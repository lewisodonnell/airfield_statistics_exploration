# models/base.py
"""Light-weight abstract parent class for all custom models."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd

from metrics import accuracy_score


class BaseModel(ABC):
    # ─────────────────────────── public API ──────────────────────────────
    @abstractmethod
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        ...

    # optional override by subclasses
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # convenience wrapper shared by every subclass
    def score(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> float:
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    # ────────────────────────── shared utilities ─────────────────────────
    def _check_Xy(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        • Converts DataFrame / Series to NumPy arrays  
        • Ensures X.shape[0] == y.shape[0]  
        • Flattens y to shape (N,)
        """
        # convert to arrays
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y.to_numpy() if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)

        # flatten y
        y_arr = y_arr.ravel()

        # basic sanity check
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X has {X_arr.shape[0]} rows but y has {y_arr.shape[0]} elements."
            )
        return X_arr, y_arr
