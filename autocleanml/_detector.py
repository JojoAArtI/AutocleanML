"""
autocleanml._detector
~~~~~~~~~~~~~~~~~~~~~
detect_task() — infer regression vs classification from the target.
"""

import numpy as np
import pandas as pd


def detect_task(y, threshold=20):
    """Infer whether the problem is regression or classification.

    Heuristic: if the target has ≤ *threshold* unique values **or** is
    non-numeric (object / category dtype), it's classification.

    Parameters
    ----------
    y : array-like
        Target values.
    threshold : int
        Max unique values to still be considered classification.

    Returns
    -------
    str
        ``"regression"`` or ``"classification"``.
    """
    if isinstance(y, pd.Series):
        if y.dtype == "object" or str(y.dtype) == "category":
            return "classification"
        n_unique = y.nunique()
    else:
        arr = np.asarray(y)
        if arr.dtype.kind in ("U", "S", "O"):  # string / object
            return "classification"
        n_unique = len(np.unique(arr[~np.isnan(arr)]) if np.issubdtype(arr.dtype, np.floating) else np.unique(arr))

    return "classification" if n_unique <= threshold else "regression"
