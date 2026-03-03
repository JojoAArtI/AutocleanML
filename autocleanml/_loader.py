"""
autocleanml._loader
~~~~~~~~~~~~~~~~~~~
load_data() and sample_data() — data ingestion helpers.
"""

import os
import pandas as pd

from ._exceptions import DataLoadError, InvalidTargetError, InsufficientDataError
from ._constants import MIN_ROWS


def load_data(source, target=None, verbose=True):
    """Load a CSV file or accept an existing DataFrame.

    Parameters
    ----------
    source : str or pd.DataFrame
        CSV file path or an existing DataFrame.
    target : str, optional
        If provided, validates that this column exists.
    verbose : bool
        Print load summary and next-step hint.

    Returns
    -------
    pd.DataFrame
    """
    # --- Accept DataFrame directly ---
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, str):
        if not os.path.isfile(source):
            raise DataLoadError(
                f'File not found: "{source}"\n'
                f"Check for typos in the file path and make sure the file exists."
            )
        try:
            df = pd.read_csv(source)
        except Exception as exc:
            raise DataLoadError(f"Could not read file: {exc}") from exc
    else:
        raise DataLoadError(
            f"source must be a file path (str) or a pandas DataFrame, "
            f"got {type(source).__name__}."
        )

    # --- Validate target ---
    if target is not None and target not in df.columns:
        raise InvalidTargetError(target, df.columns)

    # --- Validate size ---
    if len(df) < MIN_ROWS:
        raise InsufficientDataError(len(df), MIN_ROWS)

    # --- Verbose output ---
    if verbose:
        print(f"\n📂 Data loaded: {len(df):,} rows × {len(df.columns)} columns")
        if target:
            print(f"   Target column: \"{target}\"")
        print(f"   → Next: clean(df, target=\"{target or '<target>'}\")  |  guide(\"clean\") to learn more")

    return df


def sample_data(name="regression"):
    """Load a bundled sample dataset for practice.

    Parameters
    ----------
    name : str
        ``"regression"``, ``"classification"``, or ``"titanic"``.

    Returns
    -------
    pd.DataFrame
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    mapping = {
        "regression": "sample_regression.csv",
        "classification": "sample_classification.csv",
        "titanic": "sample_titanic.csv",
    }

    key = name.lower().strip()
    if key not in mapping:
        raise ValueError(
            f'Unknown sample dataset: "{name}"\n'
            f'Available: {", ".join(mapping.keys())}'
        )

    path = os.path.join(data_dir, mapping[key])
    df = pd.read_csv(path)

    targets = {"regression": "price", "classification": "label", "titanic": "Survived"}
    target = targets[key]
    print(f"\n📦 Loaded sample dataset: \"{key}\" — {len(df):,} rows × {len(df.columns)} columns")
    print(f'   Target column: "{target}"')
    print(f'   → Try: run_pipeline(df, target="{target}")')

    return df
