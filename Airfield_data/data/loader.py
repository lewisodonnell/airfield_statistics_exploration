from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd

__all__ = [
    "load_airfield_statistics",
]



_REPO_ROOT = Path(__file__).resolve().parents[1] 
_DATA_DIR = _REPO_ROOT / "data"


def _detect_categorical(df: pd.DataFrame) -> Dict[int, bool]:
    """Return a *dict* {column_index → is_categorical}.

    A column is flagged *categorical* if its dtype is not numeric.
    """
    return {
        idx: (dtype.kind not in ("i", "u", "f"))  
        for idx, dtype in enumerate(df.dtypes)
    }




def load_airfield_statistics(
    train_file: str = "airfield_statistics_train.csv",
    test_file: str = "airfield_statistics_test.csv",
    *,
    target_index: int = -3,
    data_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[int, bool]]:
    """Load the Airfield Statistics train/test CSVs."""
    data_dir = Path(data_dir or _DATA_DIR)

    train_df = pd.read_csv(data_dir / train_file)
    test_df = pd.read_csv(data_dir / test_file)

    
    target_col = train_df.columns[target_index]

    X_train = train_df.iloc[:, :target_index].copy()
    y_train = train_df[target_col].astype(int).copy()

    X_test = test_df.iloc[:, :target_index].copy()
    y_test = test_df[target_col].astype(int).copy()

    cat_columns_dict = _detect_categorical(X_train)

    return X_train, y_train, X_test, y_test, cat_columns_dict
