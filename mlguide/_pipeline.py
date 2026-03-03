"""
mlguide._pipeline
~~~~~~~~~~~~~~~~~~~~~
run_pipeline() — the full autopilot orchestrator.
"""

from ._loader import load_data
from ._cleaner import clean
from ._splitter import split
from ._encoder import encode
from ._scaler import scale
from ._detector import detect_task
from ._trainer import train
from ._comparator import compare_models
from ._evaluator import evaluate
from ._persistence import save_model as _save_model
from ._importance import get_feature_importance
from ._exceptions import NotSupportedError


def run_pipeline(
    source,
    target,
    # Task
    task="auto",
    # Splitting
    test_size=0.2,
    random_state=42,
    # Preprocessing
    scale_method="standard",
    drop_threshold=0.8,
    cardinality_threshold=50,
    # Model
    model="auto",
    compare_all=True,
    cv_folds=5,
    # Output
    save_model=False,
    output_path="mlguide_model.pkl",
    verbose=True,
):
    """Run the full ML pipeline from data to trained model.

    Parameters
    ----------
    source : str or pd.DataFrame
        CSV path or DataFrame.
    target : str
        Target column name.
    task : str
        ``"auto"``, ``"regression"``, or ``"classification"``.
    test_size : float
        Fraction for test set.
    random_state : int
        Reproducibility seed.
    scale_method : str
        ``"standard"``, ``"minmax"``, ``"robust"``, or ``"none"``.
    drop_threshold : float
        Missing-value threshold for column dropping.
    cardinality_threshold : int
        Max unique values for string columns.
    model : str or sklearn estimator
        ``"auto"``, a model name, or a custom estimator.
    compare_all : bool
        Run ``compare_models()`` and pick the best.
    cv_folds : int
        Cross-validation folds for compare_models.
    save_model : bool
        Save the model bundle to disk when done.
    output_path : str
        Path for saved model.
    verbose : bool
        Print progress at every step.

    Returns
    -------
    dict
        Keys: ``best_model``, ``model_name``, ``metrics``, ``encoder``,
        ``scaler``, ``leaderboard``, ``feature_importance``,
        ``problem_type``, ``feature_names``, ``X_test``, ``y_test``.
    """
    if verbose:
        print("\n" + "━" * 56)
        print("  mlguide — Full Pipeline")
        print("━" * 56)

    # 1. Load
    df = load_data(source, target=target, verbose=verbose)

    # 2. Clean
    df = clean(
        df,
        target=target,
        drop_threshold=drop_threshold,
        cardinality_threshold=cardinality_threshold,
        verbose=verbose,
    )

    # 3. Split
    X_train, X_test, y_train, y_test = split(
        df, target=target, test_size=test_size,
        random_state=random_state, verbose=verbose,
    )

    # 4. Detect task
    if task == "auto":
        task = detect_task(y_train)
    if verbose:
        print(f"\n🔍 Detected task: {task}")

    # 5. Encode
    X_train, enc = encode(X_train, fit=True, verbose=verbose)
    X_test, _ = encode(X_test, encoder=enc, fit=False, verbose=False)

    # 6. Scale
    X_train, sc = scale(X_train, method=scale_method, fit=True, verbose=verbose)
    X_test, _ = scale(X_test, method=scale_method, fit=False, scaler=sc, verbose=False)

    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None

    # 7. Compare or Train
    leaderboard = None
    if compare_all and isinstance(model, str) and model == "auto":
        leaderboard = compare_models(
            X_train, y_train, task=task, cv=cv_folds, verbose=verbose,
        )
        best_model_name = leaderboard.iloc[0]["model_name"]
        fitted_model = train(
            X_train, y_train, model=best_model_name,
            task=task, random_state=random_state, verbose=verbose,
        )
    else:
        best_model_name = model if isinstance(model, str) else type(model).__name__
        fitted_model = train(
            X_train, y_train, model=model,
            task=task, random_state=random_state, verbose=verbose,
        )
        best_model_name = type(fitted_model).__name__

    # 8. Evaluate
    metrics = evaluate(fitted_model, X_test, y_test, task=task, verbose=verbose)

    # 9. Feature importance (best-effort)
    feat_imp = None
    try:
        feat_imp = get_feature_importance(
            fitted_model, feature_names=feature_names,
        )
    except (NotSupportedError, Exception):
        pass

    # 10. Save
    if save_model:
        _save_model(
            fitted_model,
            path=output_path,
            encoder=enc,
            scaler=sc,
            metadata={"target": target, "metrics": metrics, "task": task},
            verbose=verbose,
        )

    if verbose:
        print("\n" + "━" * 56)
        print("  ✅ Pipeline complete!")
        print("━" * 56)

    return {
        "best_model": fitted_model,
        "model_name": best_model_name,
        "metrics": metrics,
        "encoder": enc,
        "scaler": sc,
        "leaderboard": leaderboard,
        "feature_importance": feat_imp,
        "problem_type": task,
        "feature_names": feature_names,
        "X_test": X_test,
        "y_test": y_test,
    }
