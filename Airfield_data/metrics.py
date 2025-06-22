
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


ArrayLike = Sequence | np.ndarray



def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Simple mean accuracy.

    Parameters
    ----------
    y_true, y_pred : array‑like, shape (N,)
        Ground‑truth and predicted labels (integers).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, "Input vectors must share shape"
    return float(np.mean(y_true == y_pred))


def roc_curve(
    y_true: ArrayLike,
    y_score: ArrayLike,
    pos_label: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds) for a *binary* classifier.

    *y_score* are probability estimates or decision function values for
    the **positive** class (given by *pos_label*).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    assert y_true.ndim == 1 and y_score.ndim == 1, "Inputs must be 1‑D"
    assert y_true.shape == y_score.shape, "Mismatched shapes"

   
    y_true_bin = y_true == pos_label

    
    desc_sort = np.argsort(-y_score)
    y_score_sorted = y_score[desc_sort]
    y_true_sorted = y_true_bin[desc_sort]

    
    distinct_idx = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_idx, y_true.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  

    tpr = tps / tps[-1] if tps[-1] else np.zeros_like(tps)
    fpr = fps / fps[-1] if fps[-1] else np.zeros_like(fps)

    thresholds = y_score_sorted[threshold_idxs]
    return fpr, tpr, thresholds


def auc(x: ArrayLike, y: ArrayLike) -> float:
    """Compute Area Under Curve using the trapezoidal rule."""
    return float(np.trapezoid(y, x))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Balanced accuracy = (TPR + TNR) / 2
    Works for binary {-1, +1} labels.
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    tpr = tp / (tp + fn + 1e-10)
    tnr = tn / (tn + fp + 1e-10)
    return 0.5 * (tpr + tnr)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination for regression (supports multi-target).
    """
    y_true = y_true.reshape(y_pred.shape)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot
