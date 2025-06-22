

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.loader import load_airfield_statistics
from models.MLP import MLP
from models.logistic import LogisticClassifier
from metrics import r2_score   


RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

def _binary_subset() -> Tuple[np.ndarray, np.ndarray,
                              np.ndarray, np.ndarray]:
    """
    Use only classes 1 and 3  → map {1 → 1, 3 → 0}
    Returns raw-feature numpy arrays.
    """
    X_tr_df, y_tr_ser, X_te_df, y_te_ser, _ = load_airfield_statistics()

    tr_mask = (y_tr_ser.isin([1, 3]))
    te_mask = (y_te_ser.isin([1, 3]))

    X_tr = X_tr_df[tr_mask].to_numpy(dtype=np.float64)
    X_te = X_te_df[te_mask].to_numpy(dtype=np.float64)

    y_tr = y_tr_ser[tr_mask].map({1: 1, 3: 0}).to_numpy(dtype=np.float64)
    y_te = y_te_ser[te_mask].map({1: 1, 3: 0}).to_numpy(dtype=np.float64)

    return X_tr, y_tr, X_te, y_te


def extract_features(X: np.ndarray, src_mlp: MLP) -> np.ndarray:
    """Pass X through the first two layers (0 and 1) of src_mlp."""
    h = X
    for i in (0, 1):
        lyr = src_mlp.layers[i]
        a = h @ lyr["W"].T + lyr["b"]
        h = np.maximum(a, 0.0) if lyr["act"] == "relu" else a
    return h



def main():
  
    X_tr, y_tr, X_te, y_te = _binary_subset()

    pretrained = MLP(seed=2)
    pretrained.add_layer(6, 20)
    pretrained.add_layer(20, 20, "relu")
    pretrained.add_layer(20, 2, "identity")

    w_file = Path("results/task3_1_mlp_lambda0.npz")
    if w_file.exists():
        ws = np.load(w_file)
        for lyr, idx in zip(pretrained.layers, range(3)):
            lyr["W"] = ws[f"W{idx}"]
            lyr["b"] = ws[f"b{idx}"]
    else:
      
        pretrained.fit(X_tr, np.column_stack([y_tr, y_tr]),
                       X_te, np.column_stack([y_te, y_te]),
                       lr=1e-5, epochs=10, batch=20)

   
    X_tr_feat = extract_features(X_tr, pretrained)
    X_te_feat = extract_features(X_te, pretrained)

    clf = LogisticClassifier(in_dim=20, seed=2)
    loss_transfer = []

    for epoch in range(50):
        loss = clf.grad_step(X_tr_feat, y_tr, lr=1e-3)
        loss_transfer.append(loss)

    acc_transfer = (clf.predict(X_te_feat) == y_te).mean()
    print(f"Transfer-learning test accuracy = {acc_transfer:.3f}")

    mlp = MLP(seed=2)
    mlp.add_layer(6, 20)
    mlp.add_layer(20, 20, "relu")
    mlp.add_layer(20, 1, "sigmoid")

    loss_scratch = []
    rng = np.random.default_rng(0)
    for epoch in range(50):

        y_hat, g = mlp._forward(X_tr)
        delta = (y_hat - y_tr.reshape(-1, 1))
        grads = mlp._backprop(g, delta)
        for lyr, grad in zip(mlp.layers, grads):
            lyr["W"] -= 1e-3 * grad["W"]
            lyr["b"] -= 1e-3 * grad["b"]
        loss_scratch.append(-np.mean(
            y_tr * np.log(y_hat+1e-7) + (1-y_tr)*np.log(1-y_hat+1e-7)))

    acc_scratch = ((mlp.predict(X_te) >= 0.5).astype(int).ravel() == y_te).mean()
    print(f"Scratch-MLP test accuracy    = {acc_scratch:.3f}")

  
    plt.figure(); plt.plot(loss_transfer)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Transfer-Learned Classifier Training")
    plt.savefig(RESULTS / "task3_2_transfer_loss.png", dpi=300)

    plt.figure(); plt.hist(clf.forward(X_te_feat).ravel(), bins=20)
    plt.xlabel("p"); plt.ylabel("Count")
    plt.title("Histogram of Predicted Probabilities (transfer)")
    plt.savefig(RESULTS / "task3_2_hist_transfer.png", dpi=300)

    plt.figure(); plt.plot(loss_scratch)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Loss of Non-Transfer Learning")
    plt.savefig(RESULTS / "task3_2_scratch_loss.png", dpi=300)

    plt.figure(figsize=(7,4))
    plt.plot(loss_transfer, label="Transfer Learning")
    plt.plot(loss_scratch,  label="No Transfer Learning")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Comparison of Training Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "task3_2_comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
