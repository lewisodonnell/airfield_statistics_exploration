


from pathlib import Path
import json
import itertools
import matplotlib.pyplot as plt
import numpy as np

from data.loader import load_airfield_statistics
from models.DT import DecisionTreeClassifier
from metrics import roc_curve, auc


def main():
  
    X_train, y_train, X_test, y_test, cat_columns = load_airfield_statistics()

    
    clf = DecisionTreeClassifier(
        cat_columns_dict=cat_columns,
        max_depth=10,
        min_samples_leaf=12,
        random_state=0,
    ).fit(X_train, y_train)

    prob_test = clf.predict_proba(X_test)       

   
    class_pairs = list(itertools.combinations(range(4), 2))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    auc_results = {}  

    for ax, (c1, c2) in zip(axes, class_pairs):
        
        mask = np.isin(y_test, [c1, c2])
        y_bin   = (y_test[mask] == c1).astype(int)   
        y_score = prob_test[mask, c1]             

        fpr, tpr, _ = roc_curve(y_bin, y_score)
        auc_val = auc(fpr, tpr)
        auc_results[f"{c1}_vs_{c2}"] = round(auc_val, 4)

        ax.plot(fpr, tpr, label=f"AUC = {auc_val:.2f}")
        ax.plot([0, 1], [0, 1], "r--", label="Identity function")
        ax.set_title(f"Class {c1} vs Class {c2}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(loc="lower right")

    plt.tight_layout()

    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    fig_path = results_dir / "test2_rocs.png"
    json_path = results_dir / "test2_auc.json"

    fig.savefig(fig_path, dpi=300)
    with open(json_path, "w") as f:
        json.dump(auc_results, f, indent=2)

    print(f"ROC grid saved to: {fig_path}")
    for pair, value in auc_results.items():
        c1, c2 = pair.split("_vs_")
        print(f"Class {c1} vs Class {c2}:  AUC = {value:.2f}")


if __name__ == "__main__":
    main()
