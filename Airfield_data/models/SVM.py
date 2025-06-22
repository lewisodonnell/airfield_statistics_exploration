# models/SVM.py
"""
Mini-batch SGD for a *linear*, binary SVM with either Hinge or
Modified-Huber loss + L2 regularisation.  Includes a simple grid-search
cross-validation helper used in Task 2.2.
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Optional
import itertools
import numpy as np
import pandas as pd

from models.base import BaseModel
from models.losses import hinge_raw, huber_raw
from metrics import balanced_accuracy


class LinearSVM(BaseModel):
    # ─────────────────────────── constructor ────────────────────────────
    def __init__(
        self,
        loss: str = "huber",          # "huber" or "hinge"
        lambda_: float = 1e2,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        max_iter: int = 2000,
        stop_tol: float = 1e-3,
        c: float = 1.0,               # only for huber
        standardise: bool = True,
        random_state: Optional[int] = None,
    ):
        if loss not in {"huber", "hinge"}:
            raise ValueError("loss must be 'huber' or 'hinge'")
        self.loss = loss
        self.lambda_ = lambda_
        self.lr = learning_rate
        self.batch = batch_size
        self.max_iter = max_iter
        self.stop_tol = stop_tol
        self.c = c
        self.standardise = standardise
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

        # learned during fit
        self.w_: np.ndarray | None = None
        self.loss_history_: List[float] = []
        self.iter_history_: List[int] = []
        self._mu_: np.ndarray | None = None     # standardisation μ
        self._sigma_: np.ndarray | None = None  # standardisation σ

    # ───────────────────────────── public API ────────────────────────────
    def fit(self, X, y):
        X_arr, y_arr = self._check_Xy(X, y)
        if self.standardise:
            self._mu_ = X_arr.mean(axis=0, keepdims=True)
            self._sigma_ = X_arr.std(axis=0, keepdims=True) + 1e-12
            X_arr = (X_arr - self._mu_) / self._sigma_

        X_arr = np.hstack([X_arr, np.ones((len(X_arr), 1))])  # bias column
        self.w_ = np.zeros(X_arr.shape[1])
        self.loss_history_.clear()
        self.iter_history_.clear()

        prev_loss = np.inf
        check_iter = 1

        for it in range(1, self.max_iter + 1):
            perm = np.random.permutation(len(y_arr))
            X_arr, y_arr = X_arr[perm], y_arr[perm]

            for start in range(0, len(y_arr), self.batch):
                end = start + self.batch
                X_b = X_arr[start:end]
                y_b = y_arr[start:end]
                grad = self._grad(self.w_, X_b, y_b)
                self.w_ -= self.lr * grad

            if it == check_iter or it == self.max_iter:
                loss_now = self._loss(self.w_, X_arr, y_arr)
                self.loss_history_.append(loss_now)
                self.iter_history_.append(it)
                if abs(prev_loss - loss_now) < self.stop_tol * prev_loss:
                    break
                prev_loss = loss_now
                check_iter *= 2
        return self

    def predict(self, X):
        raw = self._decision_function(X)
        return np.where(raw >= 0, 1, -1)

    # balanced accuracy helper
    def balanced_accuracy(self, X, y):
        return balanced_accuracy(y, self.predict(X))

    # margin-violation count
    def margin_violations(self, X, y):
        y = y.ravel()
        raw = self._decision_function(X)
        return int(np.sum(y * raw < 1))

    # ──────────────────────── decision function ──────────────────────────
    def _decision_function(self, X):
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        if self.standardise and self._mu_ is not None:
            X_arr = (X_arr - self._mu_) / self._sigma_
        X_arr = np.hstack([X_arr, np.ones((len(X_arr), 1))])
        return X_arr @ self.w_

    # ───────────────────────── loss & gradient ───────────────────────────
    def _loss(self, w, X, y):
        margin = y * (X @ w)
        if self.loss == "huber":
            L = huber_raw(margin, c=self.c)
        else:
            L = hinge_raw(margin)
        reg = 0.5 * np.dot(w, w) - 0.5 * w[-1] ** 2
        return reg + self.lambda_ * L.sum()

    def _grad(self, w, X, y):
        margin = y * (X @ w)
        if self.loss == "huber":
            z = 1 - margin
            g = np.zeros_like(z)
            mask1 = (z > 0) & (z <= self.c)
            mask2 = z > self.c
            g[mask1] = z[mask1]
            g[mask2] = self.c
        else:
            g = (margin < 1).astype(float)

        grad = -self.lambda_ * (g * y) @ X
        grad += w
        grad[-1] -= w[-1]          # no reg on bias
        grad /= len(y)
        return grad

    # ─────────────────────── classmethod grid-search ─────────────────────
    @classmethod
    def grid_search_cv(
        cls,
        X,
        y,
        lambda_grid: List[float],
        c_grid: List[float],
        num_folds: int = 5,
        **common_kwargs,
    ) -> Tuple["LinearSVM", Dict[str, float]]:
        """
        Simple k-fold CV over λ × c grid for Modified-Huber SVM.
        Returns the best-fitted model and a {(λ,c): mean_val_acc} dict.
        """
        X_arr = X.to_numpy(copy=True) if hasattr(X, "to_numpy") else np.asarray(X)
        y_arr = y.to_numpy(copy=True) if hasattr(y, "to_numpy") else np.asarray(y)
        
        indices = np.arange(len(y_arr))
        np.random.seed(common_kwargs.get("random_state", 0))
        np.random.shuffle(indices)
        folds = np.array_split(indices, num_folds)

        cv_table: Dict[str, float] = {}
        best_acc, best_params = -np.inf, (None, None)

        for λ, c in itertools.product(lambda_grid, c_grid):
            val_scores = []
            for k in range(num_folds):
                val_idx = folds[k]
                train_idx = np.setdiff1d(indices, val_idx)
                X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
                X_val, y_val = X_arr[val_idx], y_arr[val_idx]

                model = cls(
                    loss="huber",
                    lambda_=λ,
                    c=c,
                    **common_kwargs,
                ).fit(X_tr, y_tr)
                val_scores.append(model.score(X_val, y_val))

            mean_acc = float(np.mean(val_scores))
            cv_table[f"λ={λ}_c={c}"] = mean_acc
            if mean_acc > best_acc:
                best_acc, best_params = mean_acc, (λ, c)

        # refit on full data with best params
        best_model = cls(
            loss="huber",
            lambda_=best_params[0],
            c=best_params[1],
            **common_kwargs,
        ).fit(X, y)
        best_model.cv_results_ = cv_table          # attach for convenience
        return best_model, cv_table

