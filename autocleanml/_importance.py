"""
autocleanml._importance
~~~~~~~~~~~~~~~~~~~~~~~
get_feature_importance() — extract and optionally plot feature importances.
"""

import numpy as np
import pandas as pd

from ._exceptions import NotSupportedError


def get_feature_importance(model, feature_names=None, top_n=None, plot=False):
    """Return feature importances for tree-based or linear models.

    Parameters
    ----------
    model : fitted sklearn estimator
    feature_names : list-like, optional
        Feature names (e.g. ``X_train.columns``).
    top_n : int, optional
        Show only the top N most important features.
    plot : bool
        Requires ``matplotlib``. Plots a horizontal bar chart.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``importance`` — sorted descending.
    """
    # --- Extract importances ---
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            imp = np.mean(np.abs(coef), axis=0)
        else:
            imp = np.abs(coef)
    else:
        raise NotSupportedError(
            f"{type(model).__name__} does not expose feature importances or coefficients.\n"
            "Feature importance is available for tree-based models (RandomForest, "
            "GradientBoosting, DecisionTree) and linear models (LinearRegression, "
            "Ridge, Lasso, LogisticRegression)."
        )

    # --- Build DataFrame ---
    if feature_names is not None:
        names = list(feature_names)
    else:
        names = [f"feature_{i}" for i in range(len(imp))]

    df = pd.DataFrame({"feature": names, "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    if top_n is not None:
        df = df.head(top_n)

    # --- Optional plot ---
    if plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Plotting requires matplotlib. Install it with:\n"
                "  pip install matplotlib"
            )
        plot_df = df.sort_values("importance", ascending=True)
        plt.figure(figsize=(8, max(3, len(plot_df) * 0.35)))
        plt.barh(plot_df["feature"], plot_df["importance"], color="#4c72b0")
        plt.xlabel("Importance")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.show()

    return df
