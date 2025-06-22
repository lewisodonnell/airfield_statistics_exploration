

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.loader import load_airfield_statistics
from models.MLP import MLP
from metrics import r2_score

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_regression_data():

   
    X_tr_df, _, X_te_df, _, _ = load_airfield_statistics()
    X_tr = X_tr_df.to_numpy(dtype=np.float64)  # ensure float
    X_te = X_te_df.to_numpy(dtype=np.float64)

    
    train_df = pd.read_csv("data/airfield_statistics_train.csv")
    test_df  = pd.read_csv("data/airfield_statistics_test.csv")
    y_tr = train_df.iloc[:, -2:].to_numpy(dtype=np.float64)
    y_te = test_df.iloc[:,  -2:].to_numpy(dtype=np.float64)

    return X_tr, y_tr, X_te, y_te


def main():
    X_tr, y_tr, X_te, y_te = load_regression_data()

    lambdas = [0, 100, 500]
    loss_curves = {}
    metrics = {}

    for lam in lambdas:
        mlp = MLP(seed=2)
        mlp.add_layer(6, 20)               
        mlp.add_layer(20, 20, "relu")      
        mlp.add_layer(20, 2,  "identity")  

        mlp.fit(
            X_tr, y_tr,
            X_te, y_te,
            lam=lam,
            lr=1e-5,
            epochs=200,
            batch=20,
            seed=0,
        )
        loss_curves[lam] = mlp.loss_history_

        y_hat_tr = mlp.predict(X_tr)
        y_hat_te = mlp.predict(X_te)
        metrics[lam] = {
            "R2_train": float(r2_score(y_tr, y_hat_tr)),
            "R2_test":  float(r2_score(y_te, y_hat_te)),
        }
        print(f"λ={lam}: R²(train)={metrics[lam]['R2_train']:.3f} | "
              f"R²(test)={metrics[lam]['R2_test']:.3f}")

    
    plt.figure(figsize=(8, 5))
    for lam, curve in loss_curves.items():
        plt.plot(curve, label=f"λ={lam}")
    plt.xlabel("Epoch")
    plt.ylabel("Compound loss (train)")
    plt.title("Training Compound Loss vs. Epoch")
    plt.legend()
    plt.tight_layout()
    plot_path = RESULTS_DIR / "task3_1_loss_by_lambda.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved ➜ {plot_path}")

    with open(RESULTS_DIR / "task3_1_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
