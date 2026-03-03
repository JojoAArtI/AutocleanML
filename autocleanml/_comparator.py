"""
autocleanml._comparator
~~~~~~~~~~~~~~~~~~~~~~~
compare_models() — cross-validated model comparison leaderboard.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from ._detector import detect_task
from ._constants import REGRESSION_MODELS, CLASSIFICATION_MODELS


def compare_models(X, y, task="auto", models=None, cv=5, metric="auto", verbose=True):
    """Train and cross-validate multiple models. Returns a ranked leaderboard.

    Parameters
    ----------
    X : array-like
        Training features.
    y : array-like
        Training targets.
    task : str
        ``"auto"``, ``"regression"``, or ``"classification"``.
    models : list[str], optional
        Model names to compare. ``None`` compares all built-in models.
    cv : int
        Number of cross-validation folds.
    metric : str
        Scoring metric. ``"auto"`` picks ``"r2"`` for regression,
        ``"accuracy"`` for classification.
    verbose : bool
        Print the leaderboard table.

    Returns
    -------
    pd.DataFrame
        Columns: ``rank``, ``model_name``, ``cv_score``, ``std``.
    """
    if task == "auto":
        task = detect_task(y)

    registry = REGRESSION_MODELS if task == "regression" else CLASSIFICATION_MODELS

    if models is not None:
        selected = {}
        for m in models:
            if m not in registry:
                from ._exceptions import ModelNotSupportedError
                raise ModelNotSupportedError(m, registry.keys())
            selected[m] = registry[m]
        registry = selected

    if metric == "auto":
        scoring = "r2" if task == "regression" else "accuracy"
    else:
        # Map user-friendly names to sklearn scorers
        _metric_map = {
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "accuracy": "accuracy",
            "f1": "f1_weighted",
            "f1_weighted": "f1_weighted",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
        }
        scoring = _metric_map.get(metric, metric)

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    if verbose:
        metric_display = metric if metric != "auto" else ("R²" if task == "regression" else "Accuracy")
        print(f"\n🤖 Comparing models ({cv}-fold CV) on {len(X_arr):,} training samples...\n")

    results = []
    for name, (cls, default_params) in registry.items():
        estimator = cls(**default_params)
        scores = cross_val_score(estimator, X_arr, y_arr, cv=cv, scoring=scoring)
        mean_score = scores.mean()
        std_score = scores.std()

        # For neg_* metrics, flip the sign for display
        if scoring.startswith("neg_"):
            mean_score = -mean_score
            std_score = std_score

        results.append({
            "model_name": name,
            "cv_score": round(mean_score, 4),
            "std": round(std_score, 4),
        })

    leaderboard = pd.DataFrame(results)
    # Sort: higher is better for most metrics, but for RMSE / MAE (displayed positive), lower is better
    ascending = scoring.startswith("neg_")
    leaderboard = leaderboard.sort_values("cv_score", ascending=ascending).reset_index(drop=True)
    leaderboard["rank"] = range(1, len(leaderboard) + 1)
    leaderboard = leaderboard[["rank", "model_name", "cv_score", "std"]]

    if verbose:
        metric_label = metric if metric != "auto" else ("R²" if task == "regression" else "Accuracy")
        print(f"  #   {'Model':<25} {'CV Score (' + metric_label + ')':<22} Std Dev")
        print(f"  {'─' * 60}")
        for _, row in leaderboard.iterrows():
            star = " ⭐ Best" if row["rank"] == 1 else ""
            print(f"  {row['rank']:<4} {row['model_name']:<25} {row['cv_score']:<22} ±{row['std']}{star}")
        best = leaderboard.iloc[0]["model_name"]
        print(f'\n   → Use train(X_train, y_train, model="{best}") to train the best model')
        print(f'   → guide("compare") to understand cross-validation')

    return leaderboard
