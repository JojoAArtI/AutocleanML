"""
autocleanml._splitter
~~~~~~~~~~~~~~~~~~~~~
split() — train/test split with auto-stratification.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from ._detector import detect_task


def split(df, target, test_size=0.2, random_state=42, stratify="auto", verbose=True):
    """Split a DataFrame into train and test sets.

    Automatically uses stratified splitting for classification targets.

    Parameters
    ----------
    df : pd.DataFrame
    target : str
        Target column name.
    test_size : float
        Fraction for test set (0.05–0.5).
    random_state : int
        Reproducibility seed.
    stratify : str or bool
        ``"auto"`` detects classification targets and stratifies.
        ``True`` forces stratification, ``False`` disables it.
    verbose : bool
        Print split summary and next-step hint.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)``
    """
    if target not in df.columns:
        from ._exceptions import InvalidTargetError
        raise InvalidTargetError(target, df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # Decide on stratification
    strat_col = None
    task = detect_task(y)
    if stratify == "auto":
        if task == "classification":
            strat_col = y
    elif stratify is True:
        strat_col = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_col,
    )

    if verbose:
        strat_msg = " (stratified)" if strat_col is not None else ""
        print(f"\n✂️  Split complete{strat_msg}. "
              f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
        print(f'   → Next: encode(X_train, fit=True)  |  guide("encode") to learn more')

    return X_train, X_test, y_train, y_test
