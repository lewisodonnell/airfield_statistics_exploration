from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd

__all__ = [
    "load_airfield_statistics",
]

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]  # <repo>/data/loader.py → <repo>
_DATA_DIR = _REPO_ROOT / "data"


def _detect_categorical(df: pd.DataFrame) -> Dict[int, bool]:
    """Return a *dict* {column_index → is_categorical}.

    A column is flagged *categorical* if its dtype is not numeric.
    """
    return {
        idx: (dtype.kind not in ("i", "u", "f"))  # object, bool, category → True
        for idx, dtype in enumerate(df.dtypes)
    }


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def load_airfield_statistics(
    train_file: str = "airfield_statistics_train.csv",
    test_file: str = "airfield_statistics_test.csv",
    *,
    target_index: int = -3,
    data_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[int, bool]]:
    """Load the Airfield Statistics train/test CSVs.

    Parameters
    ----------
    train_file, test_file
        Filenames (within *data_dir*) of the coursework CSVs.
    target_index
        Position **relative to the end** of the dataframe that holds the
        classification target.  Task 1 places the label three columns from
        the right, hence the default ``-3``.
    data_dir
        Path to the directory containing the csvs.  Defaults to
        ``<repo>/data``.

    Returns
    -------
    X_train, y_train, X_test, y_test
        Features are returned as **new** dataframes so that subsequent
        in‑place operations don’t mutate the originals.
    cat_columns_dict
        Mapping *feature index* → *is_categorical* used by the DT code.
    """
    data_dir = Path(data_dir or _DATA_DIR)

    train_df = pd.read_csv(data_dir / train_file)
    test_df = pd.read_csv(data_dir / test_file)

    # Identify the target column by position (negative index from the right)
    target_col = train_df.columns[target_index]

    X_train = train_df.iloc[:, :target_index].copy()
    y_train = train_df[target_col].astype(int).copy()

    X_test = test_df.iloc[:, :target_index].copy()
    y_test = test_df[target_col].astype(int).copy()

    cat_columns_dict = _detect_categorical(X_train)

    return X_train, y_train, X_test, y_test, cat_columns_dict
