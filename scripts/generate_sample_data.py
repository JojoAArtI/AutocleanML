"""Generate sample CSV datasets for autocleanml.

Run this script once to create the bundled sample data files.
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "autocleanml", "data")
os.makedirs(data_dir, exist_ok=True)

# ── 1. Regression dataset (housing-style) ──
n = 500
df_reg = pd.DataFrame({
    "sqft": np.random.randint(600, 5000, n),
    "bedrooms": np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.2, 0.4, 0.25, 0.1]),
    "bathrooms": np.random.choice([1, 2, 3, 4], n, p=[0.15, 0.45, 0.3, 0.1]),
    "age": np.random.randint(0, 80, n),
    "garage": np.random.choice([0, 1, 2, 3], n, p=[0.1, 0.4, 0.35, 0.15]),
    "neighborhood": np.random.choice(["Downtown", "Suburb", "Rural", "Midtown", "Lakeside"], n),
    "condition": np.random.choice(["Excellent", "Good", "Fair", "Poor"], n, p=[0.15, 0.45, 0.3, 0.1]),
    "has_pool": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    "lot_size": np.random.randint(1000, 20000, n),
})
# Generate price as a function of features + noise
df_reg["price"] = (
    df_reg["sqft"] * 120
    + df_reg["bedrooms"] * 15000
    + df_reg["bathrooms"] * 12000
    - df_reg["age"] * 800
    + df_reg["garage"] * 20000
    + df_reg["has_pool"] * 30000
    + df_reg["lot_size"] * 3
    + np.random.normal(0, 25000, n)
).astype(int)

# Inject some nulls to make cleaning interesting
for col in ["age", "lot_size", "condition"]:
    mask = np.random.random(n) < 0.05
    df_reg.loc[mask, col] = np.nan

df_reg.to_csv(os.path.join(data_dir, "sample_regression.csv"), index=False)
print(f"✅ sample_regression.csv: {len(df_reg)} rows, target='price'")

# ── 2. Classification dataset ──
n = 400
df_cls = pd.DataFrame({
    "feature_1": np.random.randn(n),
    "feature_2": np.random.randn(n),
    "feature_3": np.random.randn(n),
    "feature_4": np.random.uniform(0, 10, n),
    "feature_5": np.random.uniform(-5, 5, n),
    "category_a": np.random.choice(["X", "Y", "Z"], n),
    "category_b": np.random.choice(["Low", "Medium", "High"], n),
})
# Label based on feature combination
score = (
    df_cls["feature_1"] * 0.5
    + df_cls["feature_2"] * 0.3
    - df_cls["feature_3"] * 0.2
    + df_cls["feature_4"] * 0.1
    + np.random.randn(n) * 0.3
)
df_cls["label"] = (score > score.median()).astype(int)

# Inject some nulls
for col in ["feature_3", "category_a"]:
    mask = np.random.random(n) < 0.04
    df_cls.loc[mask, col] = np.nan

df_cls.to_csv(os.path.join(data_dir, "sample_classification.csv"), index=False)
print(f"✅ sample_classification.csv: {len(df_cls)} rows, target='label'")

# ── 3. Titanic-style dataset ──
n = 400
df_tit = pd.DataFrame({
    "Pclass": np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
    "Sex": np.random.choice(["male", "female"], n, p=[0.65, 0.35]),
    "Age": np.random.normal(30, 14, n).clip(1, 80).round(0),
    "SibSp": np.random.choice([0, 1, 2, 3, 4], n, p=[0.6, 0.23, 0.1, 0.04, 0.03]),
    "Parch": np.random.choice([0, 1, 2, 3], n, p=[0.7, 0.15, 0.1, 0.05]),
    "Fare": np.random.exponential(35, n).round(2),
    "Embarked": np.random.choice(["S", "C", "Q"], n, p=[0.72, 0.19, 0.09]),
})
# Survival based roughly on Pclass + Sex
surv_prob = 0.2 + (df_tit["Pclass"] == 1) * 0.2 + (df_tit["Sex"] == "female") * 0.35
surv_prob = surv_prob.clip(0.05, 0.95)
df_tit["Survived"] = (np.random.random(n) < surv_prob).astype(int)

# Inject nulls in Age
mask = np.random.random(n) < 0.1
df_tit.loc[mask, "Age"] = np.nan

df_tit.to_csv(os.path.join(data_dir, "sample_titanic.csv"), index=False)
print(f"✅ sample_titanic.csv: {len(df_tit)} rows, target='Survived'")

print("\n🎉 All sample datasets generated!")
