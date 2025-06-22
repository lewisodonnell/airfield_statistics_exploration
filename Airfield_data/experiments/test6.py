

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from data.loader import load_airfield_statistics
from models.SVM import LinearSVM
from models.losses import huber_raw
from metrics import balanced_accuracy

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def relabel(y):
    return np.where(y == 0, 1, -1)


def main():
    
    X_tr, y_tr, X_te, y_te, _ = load_airfield_statistics()
    y_tr_bin = relabel(y_tr)
    y_te_bin = relabel(y_te)

   
    lambdas = [1, 100, 10_000]
    c_vals  = [0.5, 1, 10]

    best_model, cv_table = LinearSVM.grid_search_cv(
        X_tr,
        y_tr_bin,
        lambda_grid=lambdas,
        c_grid=c_vals,
        num_folds=5,
        random_state=0,
    )

    with open(RESULTS_DIR / "test6_cv_table.json", "w") as f:
        json.dump(cv_table, f, indent=2)
    print("CV table saved ➜ results/test6_cv_table.json")

    acc_test = best_model.score(X_te, y_te_bin)
    bal_acc  = balanced_accuracy(y_te_bin, best_model.predict(X_te))

    print(f"Best hyper-params  λ={best_model.lambda_} , c={best_model.c}")
    print(f"Test accuracy      {acc_test:.4f}")
    print(f"Balanced accuracy  {bal_acc:.4f}")

 
    t_grid = np.linspace(-2, 3, 500)
    plt.figure(figsize=(6, 4))
    for c in c_vals:
        plt.plot(t_grid, huber_raw(t_grid, c=c), label=f"c={c}")
    plt.xlabel(r"$y\,f(x)$")
    plt.ylabel("Modified Huber loss")
    plt.title("Modified Huber Loss for different $c$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "test6_loss_c_curves.png", dpi=300)
    print("Loss-shape plot ➜ results/test6_loss_c_curves.png")


if __name__ == "__main__":
    main()
