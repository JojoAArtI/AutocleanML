"""
autocleanml._encoder
~~~~~~~~~~~~~~~~~~~~
encode() — One-Hot Encoding with fit/reuse pattern and leakage guard.
"""

import warnings
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from ._exceptions import PreprocessingOrderWarning


def encode(X, fit=True, encoder=None, drop="first", handle_unknown="ignore", verbose=True):
    """One-Hot Encode categorical columns.

    Designed to be fit on training data only and reused on test data.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    fit : bool
        Fit a new encoder on this data.
    encoder : fitted OneHotEncoder, optional
        Reuse a previously fitted encoder.
    drop : str
        Avoids the dummy variable trap (``"first"``).
    handle_unknown : str
        What to do with unseen categories at inference.
    verbose : bool
        Log encoding summary and next-step hint.

    Returns
    -------
    tuple
        ``(encoded_DataFrame, fitted_encoder)``
    """
    if fit and encoder is not None:
        raise ValueError(
            "Cannot set fit=True and pass an encoder at the same time.\n"
            "Use fit=True to create a NEW encoder, or pass encoder=... to REUSE one."
        )

    X = X.copy()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Nothing to encode
    if not cat_cols:
        if verbose:
            print("\n🔢 No categorical columns to encode.")
            print('   → Next: scale(X_train, fit=True)  |  guide("scale") to learn more')
        return X, encoder

    # Leakage heuristic: warn if the DataFrame is suspiciously large (> 80% of common full datasets)
    if fit and len(X) > 5000:
        warnings.warn(
            "⚠️  You appear to be encoding a large dataset. "
            "Make sure you have already called split() before encode().\n"
            '   Recommended order: clean() → split() → encode() → scale() → train()\n'
            '   Call guide("leakage") to learn more.',
            PreprocessingOrderWarning,
            stacklevel=2,
        )

    if fit:
        enc = OneHotEncoder(drop=drop, handle_unknown=handle_unknown, sparse_output=False)
        encoded = enc.fit_transform(X[cat_cols])
        feature_names = enc.get_feature_names_out(cat_cols)
    else:
        if encoder is None:
            raise ValueError("fit=False requires an encoder. Pass encoder=<fitted encoder>.")
        enc = encoder
        encoded = enc.transform(X[cat_cols])
        feature_names = enc.get_feature_names_out(cat_cols)

    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)
    X = X.drop(columns=cat_cols)
    X = pd.concat([X, encoded_df], axis=1)

    if verbose:
        action = "Fitted new encoder on" if fit else "Applied existing encoder to"
        print(f"\n🔢 {action} {len(cat_cols)} categorical columns → {len(feature_names)} encoded features")
        print(f'   → Next: scale(X_train, fit=True)  |  guide("scale") to learn more')

    return X, enc
