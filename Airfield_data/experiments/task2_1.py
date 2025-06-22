# experiments/task2_1.py
"""
Task 2.1  –  Binary linear SVM: Hinge vs Modified-Huber.

Run:
    python -m experiments.task2_1
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

from data.loader import load_airfield_statistics
from models.SVM import LinearSVM

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def relabel_to_binary(y):
    return np.where(y == 0, 1, -1)


def main():
    # ─── data prep ──────────────────────────────────────────────
    X_tr, y_tr, X_te, y_te, _ = load_airfield_statistics()
    y_tr_bin = relabel_to_binary(y_tr)
    y_te_bin = relabel_to_binary(y_te)

    # ─── Huber SVM ──────────────────────────────────────────────
    huber_svm = LinearSVM(
        loss="huber", lambda_=100, random_state=0
    ).fit(X_tr, y_tr_bin)

    # ─── Hinge SVM ──────────────────────────────────────────────
    hinge_svm = LinearSVM(
        loss="hinge", lambda_=100, random_state=0
    ).fit(X_tr, y_tr_bin)

    # ─── plot loss curves ───────────────────────────────────────
    plt.figure(figsize=(6, 4))
    plt.plot(huber_svm.iter_history_, huber_svm.loss_history_, label="Huber")
    plt.plot(hinge_svm.iter_history_, hinge_svm.loss_history_, label="Hinge")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(r"Training loss for $\lambda=100$")
    plt.legend()
    plt.tight_layout()

    fig_path = RESULTS_DIR / "task2_1_loss_curves.png"
    plt.savefig(fig_path, dpi=300)
    print(f"Loss-curve plot ➜ {fig_path}")

    from models.losses import hinge_raw, huber_raw
    t_grid = np.linspace(-0.5, 2, 300)
    plt.figure(figsize=(6, 4))
    plt.plot(t_grid, hinge_raw(t_grid), label="Hinge")
    plt.plot(t_grid, huber_raw(t_grid, c=1.0), label="Modified Huber (c=1)")
    plt.xlabel(r"$y\,f(x)$")
    plt.ylabel("Loss")
    plt.title("Hinge vs. Modified Huber Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "task2_1_loss_functions.png", dpi=300)
    print(f"Loss-curve plot ➜ {RESULTS_DIR / 'task2_1_loss_functions.png'}")

    # ─── metrics ────────────────────────────────────────────────
    metrics = {
        "huber_train_acc":  huber_svm.score(X_tr, y_tr_bin),
        "huber_test_acc":   huber_svm.score(X_te, y_te_bin),
        "hinge_train_acc":  hinge_svm.score(X_tr, y_tr_bin),
        "hinge_test_acc":   hinge_svm.score(X_te, y_te_bin),
        "huber_margin_pts": huber_svm.margin_violations(X_tr, y_tr_bin),
        "hinge_margin_pts": hinge_svm.margin_violations(X_tr, y_tr_bin),
    }

    for k, v in metrics.items():
        print(f"{k:18}: {v:.6f}" if "acc" in k else f"{k:18}: {v}")

    with open(RESULTS_DIR / "task2_1_metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)


if __name__ == "__main__":
    main()
