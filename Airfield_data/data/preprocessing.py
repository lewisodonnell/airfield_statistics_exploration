
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

ArrayLike = Union[pd.DataFrame, np.ndarray]


class StandardScaler:
    """Remove mean and scale to unit variance feature‑wise.

    Parameters
    ----------
    with_mean : bool, default=True
        Whether to centre the data before scaling.
    with_std : bool, default=True
        Whether to scale the data to unit variance.

    Notes
    -----
    *Imitates the parts of :class:`sklearn.preprocessing.StandardScaler`
    we actually need, but stays NumPy‑only.*
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fitting / transforming
    # ------------------------------------------------------------------
    def fit(self, X: ArrayLike) -> "StandardScaler":
        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        if self.with_mean:
            self.mean_ = X_arr.mean(axis=0, keepdims=True)
        else:
            self.mean_ = np.zeros((1, X_arr.shape[1]))

        if self.with_std:
            self.scale_ = X_arr.std(axis=0, keepdims=True)
            self.scale_[self.scale_ == 0] = 1.0  # avoid /0
        else:
            self.scale_ = np.ones((1, X_arr.shape[1]))

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler instance is not fitted yet — call `fit` first.")

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return (X_arr - self.mean_) / self.scale_

    def inverse_transform(self, X_std: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler instance is not fitted yet — call `fit` first.")
        return X_std * self.scale_ + self.mean_

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        return self.fit(X).transform(X)


# ----------------------------------------------------------------------
# Functional wrapper keeping backwards compatibility with the notebook
# ----------------------------------------------------------------------

def standardise(
    X: ArrayLike,
    X_train_: Optional[ArrayLike] = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Replicates the original `standardise` function from the notebook.

    This is a *thin shim* around :class:`StandardScaler` so legacy cells
    copy‑pasted from the notebook still work, yet the rest of the code
    can use the object‑oriented API.

    Returns
    -------
    X_std : np.ndarray
    (mu, sigma) : tuple
        Mean and std that were used for scaling – handy for assertions.
    """

    if X_train_ is None:
        X_train_ = X

    scaler = StandardScaler().fit(X_train_)
    return scaler.transform(X), (scaler.mean_, scaler.scale_)