from __future__ import annotations

import numpy as np

from data.loader import load_airfield_statistics
from data.preprocessing import StandardScaler
from models.DT import DecisionTreeClassifier
from models.KNN import KNNClassifier
from metrics import roc_curve, auc
from models.RF import RandomForestClassifier
from models.DT_weighted import WeightedDecisionTreeClassifier
from models.SVM import LinearSVM
from data.preprocessing import StandardScaler
from models.SVM import LinearSVM
from metrics import balanced_accuracy
from models.MLP import MLP
from metrics import r2_score

# ---------------------------------------------------------------------
# Shared fixtures (no pytest fixtures to keep the file self‑contained)
# ---------------------------------------------------------------------

X_train, y_train, X_test, y_test, cat_columns_dict = load_airfield_statistics()


# ---------------------------------------------------------------------
# Decision‑Tree (Task 1.1)
# ---------------------------------------------------------------------

def test_decision_tree_test_accuracy() -> None:
    dt = DecisionTreeClassifier(
        cat_columns_dict=cat_columns_dict,
        max_depth=10,
        min_samples_leaf=12,
        random_state=0,
    ).fit(X_train, y_train)

    # Test‑set accuracy should equal ≈ 0.8530612244897959
    assert np.isclose(dt.score(X_test, y_test), 0.8530612244897959, atol=1e-6)


# ---------------------------------------------------------------------
# k‑Nearest‑Neighbour (standardised)
# ---------------------------------------------------------------------

def test_knn_standardised_test_accuracy() -> None:
    scaler = StandardScaler().fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    knn = KNNClassifier(k=25).fit(X_train_std, y_train)
    assert np.isclose(knn.score(X_test_std, y_test), 0.8081632653061225, atol=1e-6)



def test_predict_proba_and_auc():
    X_tr, y_tr, X_te, y_te, cats = load_airfield_statistics()
    dt = DecisionTreeClassifier(cats, random_state=0).fit(X_tr, y_tr)

    # predict_proba shape / sum checks
    proba = dt.predict_proba(X_te)
    assert proba.shape == (len(X_te), 4)
    assert np.allclose(proba.sum(axis=1), 1.0)

    # quick AUC sanity – class 0 vs 1 should be ~0.92
    mask = np.isin(y_te, [0, 1])
    y_bin = (y_te[mask] == 0).astype(int)
    y_score = proba[mask, 0]
    fpr, tpr, _ = roc_curve(y_bin, y_score)
    assert np.isclose(auc(fpr, tpr), 0.92, atol=0.03)



def test_random_forest_accuracy():
    X_tr, y_tr, X_te, y_te, cats = load_airfield_statistics()
    rf = RandomForestClassifier(cats, n_estimators=20, random_state=0).fit(X_tr, y_tr)
    assert np.isclose(rf.score(X_te, y_te), 0.873469, atol=1e-3)




def test_weighted_tree_pairwise_gain():
    X_tr, y_tr, X_te, y_te, cats = load_airfield_statistics()
    loss = np.array([[0, .1, 1, 1],[.1,0,1,1],[1,1,0,1],[1,1,1,0]])
    w_dt = WeightedDecisionTreeClassifier(cats, loss, random_state=0).fit(X_tr, y_tr)
    s_dt = DecisionTreeClassifier(cats, random_state=0).fit(X_tr, y_tr)

    # improvement for pair (0,1) should be ~+0.015
    mask = np.isin(y_te, [0,1])
    w_acc = np.mean(w_dt.predict(X_te[mask]) == y_te[mask])
    s_acc = np.mean(s_dt.predict(X_te[mask]) == y_te[mask])
    assert (w_acc - s_acc) > 0.01


from models.SVM import LinearSVM
from data.preprocessing import StandardScaler

def test_svm_huber_and_hinge():
    X_tr, y_tr, X_te, y_te, _ = load_airfield_statistics()
    y_tr_bin = np.where(y_tr == 0, 1, -1)
    y_te_bin = np.where(y_te == 0, 1, -1)

    huber = LinearSVM(loss="huber", lambda_=100, random_state=0).fit(X_tr, y_tr_bin)
    hinge = LinearSVM(loss="hinge", lambda_=100, random_state=0).fit(X_tr, y_tr_bin)

    # accuracies close to notebook
    assert np.isclose(huber.score(X_te, y_te_bin), 0.9388, atol=1e-3)
    assert np.isclose(hinge.score(X_te, y_te_bin), 0.9347, atol=1e-3)

    # margin-violation counts
    assert huber.margin_violations(X_tr, y_tr_bin) == 168
    assert hinge.margin_violations(X_tr, y_tr_bin) == 121




def test_svm_grid_search_and_balanced_acc():
    X_tr, y_tr, X_te, y_te, _ = load_airfield_statistics()
    y_tr_bin = np.where(y_tr == 0, 1, -1)
    y_te_bin = np.where(y_te == 0, 1, -1)

    best_model, _ = LinearSVM.grid_search_cv(
        X_tr, y_tr_bin,
        lambda_grid=[1, 100, 10000],
        c_grid=[0.5, 1, 10],
        num_folds=5,
        random_state=0,
    )
    # correct hyper-params
    assert best_model.lambda_ == 100
    assert best_model.c == 10
    # balanced accuracy close to notebook
    bal = balanced_accuracy(y_te_bin, best_model.predict(X_te))
    assert np.isclose(bal, 0.9057, atol=3e-3)




def test_mlp_r2_scores():
    X_tr, y_tr, X_te, y_te, _ = load_airfield_statistics()
    X_tr = X_tr.to_numpy(); X_te = X_te.to_numpy()
    y_tr = np.column_stack([y_tr, y_tr])[:, :2]
    y_te = np.column_stack([y_te, y_te])[:, :2]

    mlp = MLP(seed=2)
    mlp.add_layer(6, 20)
    mlp.add_layer(20, 20, "relu")
    mlp.add_layer(20, 2,  "identity")
    mlp.fit(X_tr, y_tr, X_te, y_te, lam=0, lr=5e-5, epochs=200, batch=20)

    r2_tst = r2_score(y_te, mlp.predict(X_te))
    assert r2_tst > 0.75 and r2_tst < 0.80   # ≈ 0.776 in notebook
