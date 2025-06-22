from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[np.ndarray]


class KNNClassifier:  
   

    def __init__(self, k: int = 25):
        if k < 1:
            raise ValueError("`k` must be a positive integer.")
        self.k = k
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    
    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNClassifier":
        """Store the training set (no training cost)."""
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("X and y must have the same length.")
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict labels for *each* row in *X*."""
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Model is not fitted yet – call `fit()` first.")
        X_test = np.asarray(X)
        neigh_ind = self._kneighbours(X_test)
        return np.array([
            np.argmax(np.bincount(self.y_train[idx])) for idx in neigh_ind
        ])

    def score(self, X: ArrayLike, y: ArrayLike) -> float: 
        """Return mean accuracy on the given test data and labels."""
        y_pred = self.predict(X)
        y_true = np.asarray(y)
        return float(np.mean(y_true == y_pred))

    
    @staticmethod
    def _euclidean_distance(x_i: np.ndarray, X_j: np.ndarray) -> np.ndarray:
        """Vectorised L2 distance between *x_i* and *all* rows of *X_j*."""
        return np.sqrt(np.sum((x_i - X_j) ** 2, axis=1))

    def _kneighbours(
        self, X_test: np.ndarray, *, return_distance: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Indices (and optionally distances) of *k* nearest neighbours."""
        dist_list: list[np.ndarray] = []
        idx_list: list[np.ndarray] = []
        for x in X_test:
            dists = self._euclidean_distance(x, self.X_train)  
            idx = np.argsort(dists)[: self.k]
            idx_list.append(idx)
            if return_distance:
                dist_list.append(dists[idx])
        if return_distance:
            return np.stack(dist_list), np.stack(idx_list)
        return np.stack(idx_list)
