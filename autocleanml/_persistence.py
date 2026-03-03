"""
autocleanml._persistence
~~~~~~~~~~~~~~~~~~~~~~~~
save_model() / load_model() — bundle model + preprocessors + metadata.
"""

import joblib


def save_model(model, path, encoder=None, scaler=None, metadata=None, verbose=True):
    """Save a model bundle (model + encoder + scaler + metadata) to disk.

    Parameters
    ----------
    model : fitted sklearn estimator
    path : str
        Output path (e.g. ``"my_model.pkl"``).
    encoder : fitted encoder, optional
    scaler : fitted scaler, optional
    metadata : dict, optional
        Any extra info to store alongside the model.
    verbose : bool
        Print save confirmation.
    """
    bundle = {
        "model": model,
        "encoder": encoder,
        "scaler": scaler,
        "metadata": metadata or {},
    }
    joblib.dump(bundle, path)

    if verbose:
        print(f'\n💾 Model saved to "{path}"')
        print(f'   Contents: model={type(model).__name__}', end="")
        if encoder is not None:
            print(f", encoder=yes", end="")
        if scaler is not None:
            print(f", scaler=yes", end="")
        if metadata:
            print(f", metadata={list(metadata.keys())}", end="")
        print()
        print(f'   → Load later: bundle = load_model("{path}")')
        print(f'   → Predict:    predict("{path}", new_data)')


def load_model(path, verbose=True):
    """Load a model bundle previously saved with ``save_model()``.

    Parameters
    ----------
    path : str
        Path to the ``.pkl`` file.
    verbose : bool
        Print load confirmation.

    Returns
    -------
    dict
        Keys: ``model``, ``encoder``, ``scaler``, ``metadata``.
    """
    bundle = joblib.load(path)

    if verbose:
        model_name = type(bundle.get("model", None)).__name__
        print(f'\n📂 Model loaded from "{path}" — {model_name}')

    return bundle
