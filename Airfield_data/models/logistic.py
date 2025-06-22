# models/logistic.py
"""
Single-layer logistic classifier (binary) used in Task 3 .2
– implements forward-pass, SGD step and helpers.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

# ───────────────────────── activation helpers ──────────────────────────
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ─────────────────────────── main class ────────────────────────────────
class LogisticClassifier:
    """
    A single fully-connected layer with sigmoid activation.
    Parameters are:
        W : shape (1, in_dim)
        b : shape (1,)
    """

    def __init__(self, in_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W = rng.normal(size=(1, in_dim)) * np.sqrt(2.0 / (1 + in_dim))
        self.b = np.zeros(1)

    # ─────────────────────── forward / predict ────────────────────────
    def forward(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.W.T + self.b      # shape (N, 1)
        return _sigmoid(z)

    predict_proba = forward

    def predict(self, X: np.ndarray, thresh: float = 0.5) -> np.ndarray:
        return (self.forward(X) >= thresh).astype(int).ravel()

    # ──────────────────────── loss & gradient ─────────────────────────
    @staticmethod
    def _loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-7) -> float:
        p = np.clip(p, eps, 1.0 - eps)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    def grad_step(self,
                  X: np.ndarray,
                  y: np.ndarray,
                  lr: float = 1e-3) -> float:
        """
        One mini-batch SGD step.  Returns batch loss.
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        p = self.forward(X)                 # (M,1)
        delta = p - y                       # dL/dz
        grad_W = (delta.T @ X) / len(X)
        grad_b = delta.mean(axis=0)

        self.W -= lr * grad_W
        self.b -= lr * grad_b
        return self._loss(y, p)
