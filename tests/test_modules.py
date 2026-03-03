"""Unit tests for individual autocleanml modules."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from autocleanml import (
    load_data, sample_data, clean, split, encode, scale,
    detect_task, train, compare_models, evaluate, predict,
    save_model, load_model, get_feature_importance,
)
from autocleanml._exceptions import (
    DataLoadError, InvalidTargetError, InsufficientDataError,
    ModelNotSupportedError, NotSupportedError,
)


# ──────────────────────────── Fixtures ─────────────────────────


@pytest.fixture
def regression_df():
    """A small regression DataFrame."""
    np.random.seed(0)
    n = 100
    df = pd.DataFrame({
        "num_a": np.random.randn(n),
        "num_b": np.random.rand(n) * 10,
        "cat_a": np.random.choice(["X", "Y", "Z"], n),
        "price": np.random.randn(n) * 100 + 500,
    })
    return df


@pytest.fixture
def classification_df():
    """A small classification DataFrame."""
    np.random.seed(1)
    n = 120
    df = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "cat": np.random.choice(["A", "B"], n),
        "label": np.random.choice([0, 1], n),
    })
    return df


# ──────────────────────────── load_data ────────────────────────


class TestLoadData:
    def test_load_from_dataframe(self, regression_df):
        result = load_data(regression_df, verbose=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(regression_df)

    def test_load_validates_target(self, regression_df):
        with pytest.raises(InvalidTargetError):
            load_data(regression_df, target="nonexistent", verbose=False)

    def test_load_file_not_found(self):
        with pytest.raises(DataLoadError):
            load_data("totally_fake_path.csv", verbose=False)

    def test_load_insufficient_rows(self):
        small_df = pd.DataFrame({"a": range(10), "b": range(10)})
        with pytest.raises(InsufficientDataError):
            load_data(small_df, verbose=False)


# ──────────────────────────── clean ────────────────────────────


class TestClean:
    def test_returns_copy(self, regression_df):
        cleaned = clean(regression_df, target="price", verbose=False)
        assert cleaned is not regression_df

    def test_removes_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2, 3], "b": [4, 4, 5, 6]})
        # Need at least 50 rows for load, but clean doesn't check that
        big_df = pd.concat([df] * 20, ignore_index=True)
        cleaned = clean(big_df, verbose=False)
        assert len(cleaned) < len(big_df)

    def test_drops_mostly_null_columns(self):
        n = 100
        df = pd.DataFrame({
            "good": np.random.randn(n),
            "mostly_null": [np.nan] * 90 + [1.0] * 10,
            "target": np.random.randn(n),
        })
        cleaned = clean(df, target="target", verbose=False)
        assert "mostly_null" not in cleaned.columns
        assert "target" in cleaned.columns

    def test_imputes_numeric_nulls(self):
        n = 100
        vals = list(np.random.randn(95)) + [np.nan] * 5
        df = pd.DataFrame({"a": vals, "b": np.random.randn(n)})
        cleaned = clean(df, verbose=False)
        assert cleaned["a"].isnull().sum() == 0


# ──────────────────────────── split ────────────────────────────


class TestSplit:
    def test_basic_split(self, regression_df):
        X_tr, X_te, y_tr, y_te = split(regression_df, target="price", verbose=False)
        assert len(X_tr) + len(X_te) == len(regression_df)
        assert "price" not in X_tr.columns

    def test_stratified_classification(self, classification_df):
        X_tr, X_te, y_tr, y_te = split(
            classification_df, target="label", verbose=False
        )
        # Check proportions are roughly preserved
        orig_ratio = classification_df["label"].mean()
        train_ratio = y_tr.mean()
        assert abs(orig_ratio - train_ratio) < 0.1


# ──────────────────────────── detect_task ──────────────────────


class TestDetectTask:
    def test_regression(self):
        y = pd.Series(np.random.randn(100))
        assert detect_task(y) == "regression"

    def test_classification(self):
        y = pd.Series(np.random.choice([0, 1], 100))
        assert detect_task(y) == "classification"

    def test_string_target(self):
        y = pd.Series(["cat", "dog", "cat", "dog"] * 25)
        assert detect_task(y) == "classification"


# ──────────────────────────── encode ───────────────────────────


class TestEncode:
    def test_encode_fit_and_reuse(self, regression_df):
        X = regression_df.drop(columns=["price"])
        X_enc, enc = encode(X, fit=True, verbose=False)
        assert "cat_a" not in X_enc.columns
        assert enc is not None

        # Reuse
        X_enc2, _ = encode(X, encoder=enc, fit=False, verbose=False)
        assert X_enc2.shape == X_enc.shape

    def test_encode_no_categoricals(self):
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        X_enc, enc = encode(X, fit=True, verbose=False)
        assert X_enc.shape == X.shape

    def test_fit_and_encoder_raises(self, regression_df):
        X = regression_df.drop(columns=["price"])
        _, enc = encode(X, fit=True, verbose=False)
        with pytest.raises(ValueError):
            encode(X, fit=True, encoder=enc, verbose=False)


# ──────────────────────────── scale ────────────────────────────


class TestScale:
    def test_standard_scaling(self, regression_df):
        X = regression_df[["num_a", "num_b"]]
        X_sc, sc = scale(X, method="standard", fit=True, verbose=False)
        assert sc is not None
        # Means should be near zero
        assert abs(X_sc["num_a"].mean()) < 0.2

    def test_none_method(self, regression_df):
        X = regression_df[["num_a", "num_b"]]
        X_sc, sc = scale(X, method="none", verbose=False)
        assert sc is None

    def test_invalid_method(self, regression_df):
        X = regression_df[["num_a", "num_b"]]
        with pytest.raises(ValueError):
            scale(X, method="invalid_method", fit=True, verbose=False)


# ──────────────────────────── train ────────────────────────────


class TestTrain:
    def test_auto_regression(self, regression_df):
        X = regression_df[["num_a", "num_b"]].values
        y = regression_df["price"].values
        model = train(X, y, task="regression", verbose=False)
        assert hasattr(model, "predict")

    def test_named_model(self, regression_df):
        X = regression_df[["num_a", "num_b"]].values
        y = regression_df["price"].values
        model = train(X, y, model="ridge", task="regression", verbose=False)
        assert "Ridge" in type(model).__name__

    def test_custom_estimator(self, regression_df):
        from sklearn.linear_model import LinearRegression
        X = regression_df[["num_a", "num_b"]].values
        y = regression_df["price"].values
        model = train(X, y, model=LinearRegression(), verbose=False)
        assert isinstance(model, LinearRegression)

    def test_unsupported_model(self, regression_df):
        X = regression_df[["num_a", "num_b"]].values
        y = regression_df["price"].values
        with pytest.raises(ModelNotSupportedError):
            train(X, y, model="xgboost", task="regression", verbose=False)


# ──────────────────────────── compare_models ───────────────────


class TestCompareModels:
    def test_compare_regression(self, regression_df):
        X = regression_df[["num_a", "num_b"]].values
        y = regression_df["price"].values
        lb = compare_models(X, y, task="regression", cv=2, verbose=False)
        assert isinstance(lb, pd.DataFrame)
        assert "rank" in lb.columns
        assert len(lb) > 0

    def test_compare_subset(self, regression_df):
        X = regression_df[["num_a", "num_b"]].values
        y = regression_df["price"].values
        lb = compare_models(
            X, y, task="regression",
            models=["linear", "ridge"], cv=2, verbose=False,
        )
        assert len(lb) == 2


# ──────────────────────────── evaluate ─────────────────────────


class TestEvaluate:
    def test_regression_metrics(self, regression_df):
        X = regression_df[["num_a", "num_b"]].values
        y = regression_df["price"].values
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(X, y)
        metrics = evaluate(model, X, y, task="regression", verbose=False)
        assert "rmse" in metrics
        assert "r2" in metrics

    def test_classification_metrics(self, classification_df):
        X = classification_df[["f1", "f2"]].values
        y = classification_df["label"].values
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=200).fit(X, y)
        metrics = evaluate(model, X, y, task="classification", verbose=False)
        assert "accuracy" in metrics
        assert "f1" in metrics


# ──────────────────────────── persistence ──────────────────────


class TestPersistence:
    def test_save_and_load(self, regression_df, tmp_path):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(
            regression_df[["num_a", "num_b"]].values,
            regression_df["price"].values,
        )
        path = str(tmp_path / "test_model.pkl")
        save_model(model, path, metadata={"test": True}, verbose=False)
        bundle = load_model(path, verbose=False)
        assert "model" in bundle
        assert bundle["metadata"]["test"] is True


# ──────────────────────────── predict ──────────────────────────


class TestPredict:
    def test_predict_from_model(self, regression_df):
        from sklearn.linear_model import LinearRegression
        X = regression_df[["num_a", "num_b"]].values
        y = regression_df["price"].values
        model = LinearRegression().fit(X, y)
        preds = predict(model, X, verbose=False)
        assert len(preds) == len(X)


# ──────────────────────────── importance ───────────────────────


class TestImportance:
    def test_tree_importance(self, regression_df):
        from sklearn.ensemble import RandomForestRegressor
        X = regression_df[["num_a", "num_b"]].values
        y = regression_df["price"].values
        model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
        imp = get_feature_importance(model, feature_names=["num_a", "num_b"])
        assert len(imp) == 2
        assert "importance" in imp.columns

    def test_linear_coefficients(self, regression_df):
        from sklearn.linear_model import LinearRegression
        X = regression_df[["num_a", "num_b"]].values
        y = regression_df["price"].values
        model = LinearRegression().fit(X, y)
        imp = get_feature_importance(model, feature_names=["num_a", "num_b"])
        assert len(imp) == 2

    def test_unsupported_model(self, regression_df):
        from sklearn.neighbors import KNeighborsRegressor
        X = regression_df[["num_a", "num_b"]].values
        y = regression_df["price"].values
        model = KNeighborsRegressor().fit(X, y)
        with pytest.raises(NotSupportedError):
            get_feature_importance(model)
