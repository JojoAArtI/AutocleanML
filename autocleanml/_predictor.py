"""
autocleanml._predictor
~~~~~~~~~~~~~~~~~~~~~~
predict() — make predictions from a live model or a saved .pkl file.
"""

import os
import numpy as np
import pandas as pd


def predict(model, data, return_proba=False, verbose=True):
    """Make predictions using a fitted model or a saved ``.pkl`` bundle.

    Parameters
    ----------
    model : fitted estimator or str
        A live model object or path to a ``.pkl`` file saved with
        ``save_model()``.
    data : array-like or str
        Features (DataFrame / array) or a CSV path.
    return_proba : bool
        Return class probabilities instead of labels (classification only).
    verbose : bool
        Print prediction summary.

    Returns
    -------
    np.ndarray
    """
    # --- Load model bundle if given a path ---
    encoder = None
    scaler = None

    if isinstance(model, str):
        from ._persistence import load_model
        bundle = load_model(model, verbose=False)
        estimator = bundle["model"]
        encoder = bundle.get("encoder")
        scaler = bundle.get("scaler")
    else:
        estimator = model

    # --- Load data if given a CSV path ---
    if isinstance(data, str):
        if not os.path.isfile(data):
            from ._exceptions import DataLoadError
            raise DataLoadError(f'File not found: "{data}"')
        data = pd.read_csv(data)

    if isinstance(data, pd.DataFrame):
        X = data.copy()
    else:
        X = np.asarray(data)

    # --- Apply saved preprocessors ---
    if encoder is not None and isinstance(X, pd.DataFrame):
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            from ._encoder import encode
            X, _ = encode(X, fit=False, encoder=encoder, verbose=False)

    if scaler is not None and isinstance(X, pd.DataFrame):
        from ._scaler import scale
        X, _ = scale(X, fit=False, scaler=scaler, verbose=False)

    X_arr = np.asarray(X)

    # --- Predict ---
    if return_proba:
        if not hasattr(estimator, "predict_proba"):
            from ._exceptions import NotSupportedError
            raise NotSupportedError(
                f"{type(estimator).__name__} does not support predict_proba.\n"
                f"Try predict(model, data) for class labels instead."
            )
        preds = estimator.predict_proba(X_arr)
    else:
        preds = estimator.predict(X_arr)

    if verbose:
        print(f"\n🔮 Predictions generated: {len(preds):,} samples")

    return preds
