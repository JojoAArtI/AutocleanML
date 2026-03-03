"""
autocleanml._constants
~~~~~~~~~~~~~~~~~~~~~~
Model registry and default hyperparameters.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# ---------------------------------------------------------------------------
# Model registry: name → (class, default_params)
# ---------------------------------------------------------------------------

REGRESSION_MODELS = {
    "linear":            (LinearRegression, {}),
    "ridge":             (Ridge, {"alpha": 1.0}),
    "lasso":             (Lasso, {"alpha": 1.0}),
    "decision_tree":     (DecisionTreeRegressor, {"random_state": 42}),
    "random_forest":     (RandomForestRegressor, {"n_estimators": 100, "random_state": 42}),
    "gradient_boosting": (GradientBoostingRegressor, {"n_estimators": 100, "random_state": 42}),
    "knn":               (KNeighborsRegressor, {"n_neighbors": 5}),
}

CLASSIFICATION_MODELS = {
    "logistic_regression": (LogisticRegression, {"max_iter": 1000, "random_state": 42}),
    "decision_tree":       (DecisionTreeClassifier, {"random_state": 42}),
    "random_forest":       (RandomForestClassifier, {"n_estimators": 100, "random_state": 42}),
    "gradient_boosting":   (GradientBoostingClassifier, {"n_estimators": 100, "random_state": 42}),
    "svm":                 (SVC, {"random_state": 42}),
    "knn":                 (KNeighborsClassifier, {"n_neighbors": 5}),
    "naive_bayes":         (GaussianNB, {}),
}

# Quick lookup for name validation
ALL_MODEL_NAMES = set(REGRESSION_MODELS) | set(CLASSIFICATION_MODELS)

# Default auto-pick models (good general defaults)
DEFAULT_REGRESSION_MODEL = "random_forest"
DEFAULT_CLASSIFICATION_MODEL = "random_forest"

# Minimum rows to proceed
MIN_ROWS = 50
