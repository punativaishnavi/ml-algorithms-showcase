"""Smoke tests: every stage runs on a small slice and returns sane outputs."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import (  # noqa: E402
    classification_split,
    load_wine,
    regression_split,
)
from src.features import scale_all, scale_train_test  # noqa: E402


def _sample(n=120):
    df = load_wine()
    return df.sample(n=n, random_state=0).reset_index(drop=True)


def test_load_wine_shape_and_columns():
    df = load_wine()
    assert df.shape == (1599, 12)
    assert "quality" in df.columns
    assert df.isna().sum().sum() == 0


def test_regression_split_sizes():
    df = _sample()
    Xtr, Xte, ytr, yte = regression_split(df)
    assert len(Xtr) + len(Xte) == len(df)
    assert ytr.between(3, 8).all()


def test_classification_split_stratified():
    df = _sample(400)
    Xtr, Xte, ytr, yte = classification_split(df)
    assert set(ytr.unique()) <= {0, 1}
    # stratification keeps the positive rate close between splits
    assert abs(ytr.mean() - yte.mean()) < 0.1


def test_scaler_fit_on_train_only():
    df = _sample()
    Xtr, Xte, _, _ = regression_split(df)
    Xtr_s, Xte_s, scaler = scale_train_test(Xtr, Xte)
    assert np.allclose(Xtr_s.mean().values, 0, atol=1e-8)
    assert Xtr_s.shape == Xtr.shape and Xte_s.shape == Xte.shape
    assert scaler is not None


def test_scale_all_for_clustering():
    df = _sample()
    Xs, scaler = scale_all(df.drop(columns=["quality"]))
    assert Xs.shape == (len(df), 11)
    assert np.allclose(Xs.mean().values, 0, atol=1e-8)


def _load_experiment(mod_name: str):
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "experiments", mod_name)
    spec = importlib.util.spec_from_file_location(mod_name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_regression_experiment_end_to_end():
    mod = _load_experiment("01_regression.py")
    metrics = mod.main()
    assert metrics["Random Forest"]["r2"] > metrics["Linear Regression"]["r2"]
    assert os.path.exists("results/figures/regression_true_vs_pred.png")


def test_classification_experiment_end_to_end():
    mod = _load_experiment("02_classification.py")
    metrics = mod.main()
    assert metrics["Random Forest"]["roc_auc"] > 0.8
    assert os.path.exists("results/figures/classification_confusion_rf.png")
    assert os.path.exists("results/figures/classification_roc.png")


def test_clustering_experiment_end_to_end():
    mod = _load_experiment("03_clustering.py")
    metrics = mod.main()
    assert 2 <= metrics["kmeans"]["best_k"] <= 8
    assert os.path.exists("results/figures/clustering_pca.png")
    assert os.path.exists("results/metrics.json")
