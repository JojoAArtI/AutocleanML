
import pandas as pd


def clean(
    df,
    target=None,
    drop_threshold=0.8,
    cardinality_threshold=50,
    numeric_strategy="median",
    categorical_strategy="most_frequent",
    remove_duplicates=True,
    verbose=True,
):
    """Clean a DataFrame — handles nulls, duplicates, high-cardinality columns.

    Returns a *copy* — the original DataFrame is never modified.

    Parameters
    ----------
    df : pd.DataFrame
    target : str, optional
        Protected column — never dropped, never imputed by accident.
    drop_threshold : float
        Drop columns with more than this fraction of nulls (0–1).
    cardinality_threshold : int
        Drop string columns with more unique values than this.
    numeric_strategy : str
        ``"median"`` or ``"mean"`` for numeric null imputation.
    categorical_strategy : str
        ``"most_frequent"`` for categorical null imputation.
    remove_duplicates : bool
        Remove fully duplicate rows.
    verbose : bool
        Log each action and next-step hint.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    actions = []

    if verbose:
        print("\n🧹 Cleaning data...")

    # --- Remove duplicates ---
    if remove_duplicates:
        n_before = len(df)
        df = df.drop_duplicates()
        n_removed = n_before - len(df)
        if n_removed > 0:
            actions.append(f"Removed {n_removed:,} duplicate rows")

    # --- Drop columns with too many nulls ---
    null_frac = df.isnull().mean()
    cols_to_drop = null_frac[null_frac > drop_threshold].index.tolist()
    # Protect the target column
    if target and target in cols_to_drop:
        cols_to_drop.remove(target)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        actions.append(
            f"Dropped {len(cols_to_drop)} columns (>{drop_threshold*100:.0f}% missing): "
            f"{cols_to_drop}"
        )

    # --- Drop high-cardinality string columns (likely IDs/free text) ---
    obj_cols = df.select_dtypes(include=["object", "category"]).columns
    high_card = [
        c for c in obj_cols
        if df[c].nunique() > cardinality_threshold and c != target
    ]
    if high_card:
        df = df.drop(columns=high_card)
        actions.append(
            f"Dropped {len(high_card)} high-cardinality columns "
            f"(likely IDs or free text): {high_card}"
        )

    # --- Impute numeric nulls ---
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if target and target in num_cols:
        num_cols_impute = [c for c in num_cols if c != target]
    else:
        num_cols_impute = num_cols
    num_with_nulls = [c for c in num_cols_impute if df[c].isnull().any()]
    if num_with_nulls:
        if numeric_strategy == "median":
            df[num_with_nulls] = df[num_with_nulls].fillna(df[num_with_nulls].median())
        else:
            df[num_with_nulls] = df[num_with_nulls].fillna(df[num_with_nulls].mean())
        actions.append(f"Imputed {len(num_with_nulls)} numeric columns with {numeric_strategy}")

    # --- Impute categorical nulls ---
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target and target in cat_cols:
        cat_cols_impute = [c for c in cat_cols if c != target]
    else:
        cat_cols_impute = cat_cols
    cat_with_nulls = [c for c in cat_cols_impute if df[c].isnull().any()]
    if cat_with_nulls:
        for c in cat_with_nulls:
            df[c] = df[c].fillna(df[c].mode().iloc[0])
        actions.append(
            f"Imputed {len(cat_with_nulls)} categorical columns with most frequent value"
        )

    # --- Drop rows where target is null ---
    if target and target in df.columns and df[target].isnull().any():
        n_null_target = df[target].isnull().sum()
        df = df.dropna(subset=[target])
        actions.append(f"Dropped {n_null_target:,} rows with missing target")

    # --- Verbose output ---
    if verbose:
        for a in actions:
            print(f"   ↳ {a}")
        print(f"   ↳ Result: {len(df):,} rows × {len(df.columns)} columns")
        t = target or "<target>"
        print(f'   → Next: split(df, target="{t}")  |  guide("split") to learn more')

    return df
