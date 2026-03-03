"""Integration tests — full pipeline from data to predictions."""

import numpy as np
import pandas as pd
import pytest

from autocleanml import (
    load_data, clean, split, encode, scale, train, evaluate,
    compare_models, predict, save_model, load_model, run_pipeline,
    sample_data, detect_task,
)


@pytest.fixture
def regression_df():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "sqft": np.random.randint(500, 3000, n),
        "bedrooms": np.random.choice([1, 2, 3, 4], n),
        "type": np.random.choice(["house", "condo", "townhouse"], n),
        "old": np.random.choice(["yes", "no"], n),
    })
    df["price"] = df["sqft"] * 100 + df["bedrooms"] * 20000 + np.random.randn(n) * 5000
    # inject a few nulls
    df.loc[0, "sqft"] = np.nan
    df.loc[5, "type"] = np.nan
    return df


@pytest.fixture
def classification_df():
    np.random.seed(7)
    n = 200
    df = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "color": np.random.choice(["red", "blue", "green"], n),
    })
    df["target"] = (df["f1"] + df["f2"] > 0).astype(int)
    return df


class TestFullPipelineManual:
    """Step-by-step pipeline (Layer 2)."""

    def test_regression_manual(self, regression_df):
        df = load_data(regression_df, target="price", verbose=False)
        df = clean(df, target="price", verbose=False)
        X_tr, X_te, y_tr, y_te = split(df, target="price", verbose=False)
        X_tr, enc = encode(X_tr, fit=True, verbose=False)
        X_te, _ = encode(X_te, encoder=enc, fit=False, verbose=False)
        X_tr, sc = scale(X_tr, fit=True, verbose=False)
        X_te, _ = scale(X_te, scaler=sc, fit=False, verbose=False)
        model = train(X_tr, y_tr, task="regression", verbose=False)
        metrics = evaluate(model, X_te, y_te, task="regression", verbose=False)
        assert "r2" in metrics
        assert metrics["r2"] > 0  # model should learn something

    def test_classification_manual(self, classification_df):
        df = load_data(classification_df, target="target", verbose=False)
        df = clean(df, target="target", verbose=False)
        X_tr, X_te, y_tr, y_te = split(df, target="target", verbose=False)
        X_tr, enc = encode(X_tr, fit=True, verbose=False)
        X_te, _ = encode(X_te, encoder=enc, fit=False, verbose=False)
        X_tr, sc = scale(X_tr, fit=True, verbose=False)
        X_te, _ = scale(X_te, scaler=sc, fit=False, verbose=False)
        model = train(X_tr, y_tr, task="classification", verbose=False)
        metrics = evaluate(model, X_te, y_te, task="classification", verbose=False)
        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.5


class TestRunPipeline:
    """Full autopilot (Layer 3)."""

    def test_regression_pipeline(self, regression_df):
        result = run_pipeline(regression_df, target="price", verbose=False)
        assert "best_model" in result
        assert "metrics" in result
        assert result["problem_type"] == "regression"
        assert result["metrics"]["r2"] > 0

    def test_classification_pipeline(self, classification_df):
        result = run_pipeline(classification_df, target="target", verbose=False)
        assert result["problem_type"] == "classification"
        assert result["metrics"]["accuracy"] > 0.5

    def test_pipeline_with_save(self, regression_df, tmp_path):
        out = str(tmp_path / "model.pkl")
        result = run_pipeline(
            regression_df, target="price",
            save_model=True, output_path=out, verbose=False,
        )
        bundle = load_model(out, verbose=False)
        assert bundle["model"] is not None

    def test_pipeline_no_compare(self, regression_df):
        result = run_pipeline(
            regression_df, target="price",
            model="ridge", compare_all=False, verbose=False,
        )
        assert "Ridge" in result["model_name"]


class TestSampleData:
    """Bundled datasets load correctly."""

    def test_regression_sample(self):
        df = sample_data("regression")
        assert "price" in df.columns
        assert len(df) >= 50

    def test_classification_sample(self):
        df = sample_data("classification")
        assert "label" in df.columns

    def test_titanic_sample(self):
        df = sample_data("titanic")
        assert "Survived" in df.columns
