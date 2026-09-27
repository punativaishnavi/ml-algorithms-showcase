"""Feature engineering helpers shared across experiments."""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def make_scaler():
    return StandardScaler()


def scale_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit a StandardScaler on the TRAIN split only, then transform both.
    Returns (X_train_scaled, X_test_scaled, scaler).
    """
    scaler = make_scaler()
    Xtr = pd.DataFrame(scaler.fit_transform(X_train),
                       columns=X_train.columns, index=X_train.index)
    Xte = pd.DataFrame(scaler.transform(X_test),
                       columns=X_test.columns, index=X_test.index)
    return Xtr, Xte, scaler


def scale_all(X: pd.DataFrame):
    """Scale a full frame (for unsupervised learning)."""
    scaler = make_scaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return Xs, scaler
