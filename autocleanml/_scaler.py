"""
autocleanml._scaler
~~~~~~~~~~~~~~~~~~~
scale() — feature scaling with fit/reuse pattern.
"""

import warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from ._exceptions import PreprocessingOrderWarning


_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def scale(X, method="standard", fit=True, scaler=None, verbose=True):
    """Scale numerical features.

    Designed to be fit on training data only and reused on test data.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    method : str
        ``"standard"``, ``"minmax"``, ``"robust"``, or ``"none"``.
    fit : bool
        Fit a new scaler.
    scaler : fitted scaler, optional
        Reuse a previously fitted scaler.
    verbose : bool
        Log scaling summary and next-step hint.

    Returns
    -------
    tuple
        ``(scaled_DataFrame, fitted_scaler)``
    """
    if method == "none":
        if verbose:
            print("\n📏 Scaling skipped (method='none').")
            print('   → Next: train(X_train, y_train)  |  guide("train") to learn more')
        return X.copy(), None

    if fit and scaler is not None:
        raise ValueError(
            "Cannot set fit=True and pass a scaler at the same time.\n"
            "Use fit=True to create a NEW scaler, or pass scaler=... to REUSE one."
        )

    X = X.copy()
    num_cols = X.select_dtypes(include="number").columns.tolist()

    if not num_cols:
        if verbose:
            print("\n📏 No numeric columns to scale.")
            print('   → Next: train(X_train, y_train)  |  guide("train") to learn more')
        return X, scaler

    # Leakage heuristic
    if fit and len(X) > 5000:
        warnings.warn(
            "⚠️  You appear to be scaling a large dataset. "
            "Make sure you have already called split() before scale().\n"
            '   Recommended order: clean() → split() → encode() → scale() → train()\n'
            '   Call guide("leakage") to learn more.',
            PreprocessingOrderWarning,
            stacklevel=2,
        )

    if method not in _SCALERS:
        raise ValueError(
            f'Unknown scaling method: "{method}"\n'
            f'Available: {", ".join(_SCALERS.keys())}, "none"'
        )

    if fit:
        sc = _SCALERS[method]()
        X[num_cols] = sc.fit_transform(X[num_cols])
    else:
        if scaler is None:
            raise ValueError("fit=False requires a scaler. Pass scaler=<fitted scaler>.")
        sc = scaler
        X[num_cols] = sc.transform(X[num_cols])

    if verbose:
        action = f"Fitted {method} scaler on" if fit else f"Applied {method} scaler to"
        print(f"\n📏 {action} {len(num_cols)} numeric features")
        print(f'   → Next: train(X_train, y_train)  |  guide("train") to learn more')

    return X, sc
