# Airfield Statistics Exploration

This repository contains a small research project analysing the *Airfield Statistics* dataset.  It provides
custom Python implementations of several machine‑learning algorithms and a set of scripts that perform experiments and produce plots.

The code is self‑contained and avoids external ML, all tools are implemented from scratch.

## Dataset

The `Airfield_data/data` directory holds the training and test CSV files:

- `airfield_statistics_train.csv`
- `airfield_statistics_test.csv`

Each file contains nine columns.  The first six are numeric features (days of air frost, precipitation, sunshine hours,
humidity, wind speed and aircraft movements).  Column 7 (`Weather and flight condition category`) is a four‑class
label used for the classification tasks.  The final two columns provide the runway surface minimum and maximum
temperature, which are used for regression in Task 3.

`data/loader.py` exposes a helper `load_airfield_statistics` that loads the CSV files, splits them into features and
target arrays and detects which columns are categorical.  A simple `StandardScaler` is available in
`data/preprocessing.py` for optional feature normalisation.

## Repository layout

```
Airfield_data/
 ├── data/            Dataset CSVs and loading utilities
 ├── experiments/     Stand‑alone scripts for each coursework task
 ├── metrics.py       Accuracy, ROC/AUC, balanced accuracy and R² metrics
 ├── models/          Lightweight implementations of DT, RF, kNN, SVM and MLP
 └── tests/           Unit tests exercising the main models
```

### Models

- **DecisionTreeClassifier** and **RandomForestClassifier** in `models/DT.py` and `models/RF.py`
  Implement CART‑style decision trees with real Gini feature importance.
- **WeightedDecisionTreeClassifier** extends the tree with a loss matrix for Task 1.4.
- **KNNClassifier** implements a brute‑force k‑nearest neighbour classifier.
- **LinearSVM** in `models/SVM.py` trains a binary linear SVM by mini‑batch SGD using either Hinge or
  Modified‑Huber loss.  A simple grid‑search CV helper is included.
- **MLP** in `models/MLP.py` provides a minimal multilayer perceptron with ReLU and sigmoid
  activations.  It is used for both regression and classification in Task 3.
- **LogisticClassifier** is a single dense layer with sigmoid activation used to demonstrate transfer
  learning in Task 3.2.

### Experiments

The `experiments/` directory contains files that, when run, perform experiments with the different models on the dataset.  Each script can be executed directly from the
repository root, for example

```bash
python -m experiments.test1
```

Results such as plots and JSON metrics are written to a `results/` directory created in the working folder.
The available experiments are:

1. **Test 1** – Compare decision trees and k‑NN (raw and standardised features).
2. **Test 2** – Compute ROC curves and AUC for the soft decision tree classifier.
3. **Test 3** – Random‑forest accuracy and Gini feature importance visualisation.
4. **Test 4** – Weighted‑Gini decision tree with pair‑wise accuracy table.
5. **Test 5** – Hinge vs. Modified‑Huber linear SVM including margin statistics.
6. **Test 6** – Hyper‑parameter grid search for the Modified‑Huber SVM.
7. **Test 7** – MLP regression for runway surface temperatures.
8. **Test 8** – Transfer‑learning versus scratch training for a binary classifier.

## Requirements

The code relies only on common scientific Python packages such as `numpy`, `pandas`, `matplotlib` and `tqdm`.
Install them with

```bash
pip install numpy pandas matplotlib tqdm
```

No external machine‑learning library is required.


## Running your own analysis

You can import the data loader and models in your own scripts, e.g.

```python
from data.loader import load_airfield_statistics
from models.DT import DecisionTreeClassifier

X_tr, y_tr, X_te, y_te, cats = load_airfield_statistics()
clf = DecisionTreeClassifier(cats).fit(X_tr, y_tr)
print("Test accuracy:", clf.score(X_te, y_te))
```



