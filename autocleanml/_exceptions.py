"""
autocleanml._exceptions
~~~~~~~~~~~~~~~~~~~~~~~
Custom exceptions with student-readable messages.
"""

import difflib


class AutoCleanMLError(Exception):
    """Base exception for autocleanml — catchable with a single except."""
    pass


class DataLoadError(AutoCleanMLError):
    """Raised when data cannot be loaded from the given source."""
    pass


class InvalidTargetError(AutoCleanMLError):
    """Raised when the target column is not found in the dataset."""

    def __init__(self, target, available_columns):
        close = difflib.get_close_matches(target, available_columns, n=3, cutoff=0.5)
        msg = f'Column "{target}" not found in dataset.\n'
        if close:
            msg += f'Did you mean: "{close[0]}"?\n'
        msg += f"Available columns: {list(available_columns)}"
        super().__init__(msg)


class InsufficientDataError(AutoCleanMLError):
    """Raised when the dataset has fewer than the minimum required rows."""

    def __init__(self, n_rows, minimum=50):
        msg = (
            f"Dataset has only {n_rows} rows.\n"
            f"autocleanml needs at least {minimum} rows to train reliably.\n"
            f"With very small datasets, results are unlikely to be meaningful."
        )
        super().__init__(msg)


class PreprocessingError(AutoCleanMLError):
    """Raised for general preprocessing failures."""
    pass


class ModelNotSupportedError(AutoCleanMLError):
    """Raised when a model name string is not recognised."""

    def __init__(self, name, available):
        msg = (
            f'"{name}" is not a built-in model name.\n'
            f"Available names: {', '.join(sorted(available))}\n"
            f"To use a custom model, pass the object directly:\n"
            f"  from sklearn.svm import SVR\n"
            f'  model = train(X_train, y_train, model=SVR())'
        )
        super().__init__(msg)


class NotSupportedError(AutoCleanMLError):
    """Raised when an operation is not supported for the given model type."""
    pass


class PreprocessingOrderWarning(UserWarning):
    """Warning emitted when encode/scale is called before split."""
    pass
