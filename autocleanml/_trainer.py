"""
autocleanml._trainer
~~~~~~~~~~~~~~~~~~~~
train() — model training with named models or custom sklearn estimators.
"""

import numpy as np

from ._detector import detect_task
from ._constants import (
    REGRESSION_MODELS, CLASSIFICATION_MODELS,
    DEFAULT_REGRESSION_MODEL, DEFAULT_CLASSIFICATION_MODEL,
    ALL_MODEL_NAMES,
)
from ._exceptions import ModelNotSupportedError


def train(X, y, model="auto", task="auto", params=None, random_state=42, verbose=True):
    """Train a model.

    Parameters
    ----------
    X : array-like
        Training features.
    y : array-like
        Training targets.
    model : str or sklearn estimator
        ``"auto"`` picks a sensible default, a name string selects a built-in
        model, or pass any sklearn-compatible estimator directly.
    task : str
        ``"regression"``, ``"classification"``, or ``"auto"`` (inferred).
    params : dict, optional
        Hyperparameter overrides for named models.
    random_state : int
        Reproducibility seed.
    verbose : bool
        Log training summary and next-step hint.

    Returns
    -------
    fitted sklearn estimator
    """
    # --- Detect task ---
    if task == "auto":
        task = detect_task(y)

    # --- Resolve model ---
    if isinstance(model, str):
        registry = REGRESSION_MODELS if task == "regression" else CLASSIFICATION_MODELS

        if model == "auto":
            model_name = DEFAULT_REGRESSION_MODEL if task == "regression" else DEFAULT_CLASSIFICATION_MODEL
        else:
            model_name = model

        if model_name not in registry:
            raise ModelNotSupportedError(model_name, registry.keys())

        cls, default_params = registry[model_name]
        final_params = {**default_params}
        if params:
            final_params.update(params)
        # Inject random_state where applicable
        if "random_state" in final_params:
            final_params["random_state"] = random_state
        estimator = cls(**final_params)
        display_name = type(estimator).__name__
    else:
        # Custom sklearn estimator
        estimator = model
        display_name = type(estimator).__name__

    # --- Fit ---
    if verbose:
        print(f"\n🤖 Training {display_name}...")

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    estimator.fit(X_arr, y_arr)

    if verbose:
        print(f"   ↳ Trained on {X_arr.shape[0]:,} rows × {X_arr.shape[1]} features")
        print(f'   → Next: evaluate(model, X_test, y_test)  |  guide("evaluate") to learn more')

    return estimator
