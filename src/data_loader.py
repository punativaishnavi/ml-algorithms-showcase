"""Loading and splitting the wine quality dataset."""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "winequality-red.csv")
RANDOM_STATE = 42


def load_wine(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw wine quality CSV (semicolon-separated)."""
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip('" ') for c in df.columns]
    return df


def regression_split(df: pd.DataFrame, test_size: float = 0.2):
    """Features / continuous quality target split for regression."""
    X = df.drop(columns=["quality"])
    y = df["quality"].astype(float)
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


def classification_split(df: pd.DataFrame, good_threshold: int = 7, test_size: float = 0.2):
    """
    Binary target: 1 = "good" wine (quality >= threshold), 0 = rest.
    Stratified because good wines are a minority (~13%).
    """
    X = df.drop(columns=["quality"])
    y = (df["quality"] >= good_threshold).astype(int)
    return train_test_split(X, y, test_size=test_size,
                            random_state=RANDOM_STATE, stratify=y)
