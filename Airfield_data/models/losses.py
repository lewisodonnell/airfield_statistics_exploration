# models/losses.py
"""
Shared loss utilities for the whole repo.
"""

import numpy as np

# ───────────────────────── hinge ──────────────────────────
def hinge_raw(z: np.ndarray) -> np.ndarray:
    """Hinge loss on raw margin  z = y·f(x).   L = max(0, 1 - z)."""
    return np.maximum(0.0, 1.0 - z)


# ───────────────────── modified Huber ─────────────────────
def huber_raw(z: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Modified Huber (Zhang, 2004) on margin  z = y·f(x).

        if z >= 1       → 0
        if 0 <= z < 1   → 0.5 · (1 - z)²
        if z  < 0       → c · (1 - z) - 0.5·c²    (linear)

    Parameter
    ---------
    c : float
        “transition” slope parameter (task states c=1).
    """
    z = 1.0 - z        # switch to  q = 1 - z  formulation
    L = np.zeros_like(z)
    mask1 = (z > 0) & (z <= c)
    mask2 = z > c
    L[mask1] = 0.5 * z[mask1] ** 2
    L[mask2] = c * (z[mask2] - 0.5 * c)
    return L


# ─────────────────────── compound regression loss ───────────────────────
def compound_loss(y_true, y_pred, lam: float = 0.0):
    """
    Compound loss from Task 3:
        MSE(T_min) + MSE(T_max) + λ·max(0, T_min_hat – T_max_hat)
    """
    diff = y_true - y_pred
    mse = (diff ** 2).sum(axis=1)          # per-sample sum of two MSE terms
    penalty = lam * np.maximum(0.0, y_pred[:, 0] - y_pred[:, 1])
    return np.mean(mse + penalty)


def grad_compound_loss(y_true, y_pred, lam: float = 0.0):
    """
    Gradient wrt predictions.  Shape (K, 2).
    """
    diff = y_true - y_pred          # (K,2)
    grad = -2.0 * diff              # d/dŷ of MSE part
    mask = (y_pred[:, 0] > y_pred[:, 1]).astype(float)
    grad[:, 0] += lam * mask
    grad[:, 1] -= lam * mask
    grad /= y_true.shape[0]         # average over batch
    return grad
