"""
autocleanml._evaluator
~~~~~~~~~~~~~~~~~~~~~~
evaluate() — model evaluation with plain-English metric descriptions.
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
)

from ._detector import detect_task


def evaluate(model, X, y, task="auto", verbose=True):
    """Evaluate a fitted model on a test set.

    Parameters
    ----------
    model : fitted sklearn estimator
    X : array-like
        Test features.
    y : array-like
        Test targets.
    task : str
        ``"auto"``, ``"regression"``, or ``"classification"``.
    verbose : bool
        Print metrics with plain-English explanations.

    Returns
    -------
    dict
        Metric name → value.
    """
    if task == "auto":
        task = detect_task(y)

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    y_pred = model.predict(X_arr)

    if task == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_arr, y_pred)))
        mae = float(mean_absolute_error(y_arr, y_pred))
        r2 = float(r2_score(y_arr, y_pred))
        metrics = {"rmse": round(rmse, 2), "mae": round(mae, 2), "r2": round(r2, 4)}

        if verbose:
            print(f"\n📊 Evaluation on test set ({len(y_arr):,} samples):\n")
            print(f"  RMSE  :  {rmse:,.2f}   (lower is better — average prediction error in same units as target)")
            print(f"  MAE   :  {mae:,.2f}   (lower is better — more interpretable than RMSE)")
            print(f"  R²    :  {r2:.4f}    (higher is better — 1.0 is perfect)")
            print(f'\n  → guide("evaluate") to understand what these numbers mean')

    else:
        acc = float(accuracy_score(y_arr, y_pred))
        avg = "weighted" if len(np.unique(y_arr)) > 2 else "binary"
        prec = float(precision_score(y_arr, y_pred, average=avg, zero_division=0))
        rec = float(recall_score(y_arr, y_pred, average=avg, zero_division=0))
        f1 = float(f1_score(y_arr, y_pred, average=avg, zero_division=0))
        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

        if verbose:
            print(f"\n📊 Evaluation on test set ({len(y_arr):,} samples):\n")
            print(f"  Accuracy  :  {acc:.4f}   (fraction of correct predictions)")
            print(f"  Precision :  {prec:.4f}   (of predicted positives, how many were correct)")
            print(f"  Recall    :  {rec:.4f}   (of actual positives, how many were found)")
            print(f"  F1        :  {f1:.4f}   (harmonic mean of precision and recall)")
            print(f'\n  → guide("evaluate") to understand what these numbers mean')

    return metrics
