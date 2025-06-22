

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

from data.loader import load_airfield_statistics
from models.RF import RandomForestClassifier

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def main():
  
    X_tr, y_tr, X_te, y_te, cats = load_airfield_statistics()

    rf = RandomForestClassifier(
        cat_columns_dict=cats,
        n_estimators=20,
        max_depth=10,
        min_samples_leaf=12,
        max_features="sqrt",
        random_state=0,
    ).fit(X_tr, y_tr)

    acc_train = rf.score(X_tr, y_tr)
    acc_test  = rf.score(X_te, y_te)

    print(f"Random-Forest  : train={acc_train:.4f} | test={acc_test:.4f}")

    imp = rf.feature_importances_
    imp_pct = 100 * imp / imp.max() if imp.max() else imp
    idx = np.argsort(imp_pct)

    plt.figure(figsize=(6, 4))
    plt.barh(range(len(imp_pct)), imp_pct[idx])
    plt.yticks(range(len(imp_pct)), X_tr.columns[idx])
    plt.xlabel("Importance (% of max)")
    plt.tight_layout()

    fig_path = RESULTS_DIR / "test3_importances.png"
    plt.savefig(fig_path, dpi=300)
    print(f"Gini importance plot ➜ {fig_path}")

    json_path = RESULTS_DIR / "test3_metrics.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "train_accuracy": round(float(acc_train), 6),
                "test_accuracy":  round(float(acc_test), 6),
                "importances_pct": {col: round(float(v), 2)
                                    for col, v in zip(X_tr.columns, imp_pct)},
            },
            f, indent=2
        )
    print(f"Metrics JSON saved ➜ {json_path}")


if __name__ == "__main__":
    main()
