# autocleanml

> **The ML toolkit that teaches you while it works.**

[![PyPI](https://img.shields.io/pypi/v/autocleanml)](https://pypi.org/project/autocleanml/)
[![Python](https://img.shields.io/pypi/pyversions/autocleanml)](https://pypi.org/project/autocleanml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**autocleanml** is a modular, student-first ML toolkit with three layers:

1. **Guided Learning** — a built-in interactive help system that explains ML concepts.
2. **Individual Modules** — every pipeline stage is independently importable.
3. **Full Autopilot** — `run_pipeline()` chains everything together.

## Installation

```bash
pip install autocleanml
```

## Quick Start

### Autopilot (one line)

```python
from autocleanml import sample_data, run_pipeline

df = sample_data("regression")
result = run_pipeline(df, target="price")
```

### Step by Step

```python
from autocleanml import load_data, clean, split, encode, scale, train, evaluate

df = load_data("housing.csv")
df = clean(df, target="price")
X_tr, X_te, y_tr, y_te = split(df, target="price")
X_tr, enc = encode(X_tr, fit=True)
X_te, _   = encode(X_te, encoder=enc)
X_tr, sc  = scale(X_tr, fit=True)
X_te, _   = scale(X_te, scaler=sc)
model     = train(X_tr, y_tr, model="random_forest")
metrics   = evaluate(model, X_te, y_te)
```

### Need Help?

```python
from autocleanml import guide

guide()              # overview
guide("ml_basics")   # what is ML?
guide("split")       # why we split data
guide("train")       # all available models
guide("cheatsheet")  # compact reference
```

## Features

- **Zero data leakage** — split before encode/scale is enforced by architecture.
- **Transparent** — every function logs what it did and why.
- **Modular** — use any function independently.
- **Student-friendly errors** — typo suggestions, available column listings, clear next-step guidance.
- **Lightweight** — only pandas, numpy, scikit-learn, and joblib.
- **Bundled datasets** — `sample_data("regression")` for instant practice.

## API Reference

| Function | Description |
|---|---|
| `guide(topic)` | Interactive help system |
| `run_pipeline(source, target)` | Full autopilot pipeline |
| `load_data(source)` | Load CSV or DataFrame |
| `sample_data(name)` | Bundled practice datasets |
| `clean(df, target)` | Clean data (nulls, duplicates, cardinality) |
| `split(df, target)` | Train/test split with auto-stratification |
| `encode(X, fit=True)` | One-Hot Encode categorical columns |
| `scale(X, method)` | Scale numeric features |
| `detect_task(y)` | Infer regression vs classification |
| `train(X, y, model)` | Train a model |
| `compare_models(X, y)` | Cross-validated model comparison |
| `evaluate(model, X, y)` | Evaluate on test set |
| `predict(model, data)` | Make predictions |
| `save_model(model, path)` | Save model bundle |
| `load_model(path)` | Load model bundle |
| `get_feature_importance(model)` | Feature importance table |

## License

MIT — Joel Inian Francis
