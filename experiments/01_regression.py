"""
01 — Regression: predict the exact wine quality score (3-8).

Models: Linear Regression, Ridge, Random Forest Regressor.
Metrics: RMSE, MAE, R^2 (5-fold CV + holdout test).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from src.data_loader import RANDOM_STATE, load_wine, regression_split
from src.evaluate import fig_path, print_table, save_metrics

MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 13)),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE,
                                           n_jobs=-1),
}


def main() -> dict:
    df = load_wine()
    print(f"Loaded {len(df)} wines, {df.shape[1] - 1} features")
    X_train, X_test, y_train, y_test = regression_split(df)

    rows, metrics = [], {}
    best_name, best_rmse, best_model = None, float("inf"), None

    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))
        cv_r2 = float(cross_val_score(model, X_train, y_train, cv=5).mean())
        rows.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2, "cv_r2": cv_r2})
        metrics[name] = {"rmse": round(rmse, 4), "mae": round(mae, 4),
                         "r2": round(r2, 4), "cv_r2": round(cv_r2, 4)}
        if rmse < best_rmse:
            best_name, best_rmse, best_model = name, rmse, model

    print_table("REGRESSION — quality score prediction (holdout test)",
                rows, ["model", "rmse", "mae", "r2", "cv_r2"])

    # True vs predicted scatter for the winning model.
    pred = best_model.predict(X_test)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred, alpha=0.35, s=18)
    lo, hi = y_test.min(), y_test.max()
    plt.plot([lo, hi], [lo, hi], "r--", label="perfect prediction")
    plt.xlabel("True quality")
    plt.ylabel("Predicted quality")
    plt.title(f"Regression: true vs predicted ({best_name})")
    plt.legend()
    plt.tight_layout()
    out = fig_path("regression_true_vs_pred.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  figure -> {out}")

    # Feature importances for the random forest.
    rf = MODELS["Random Forest"]
    importances = sorted(zip(X_train.columns, rf.feature_importances_),
                         key=lambda t: t[1], reverse=True)
    print("\nTop features (Random Forest importance):")
    for feat, imp in importances[:5]:
        print(f"  {feat:<22} {imp:.3f}")
    metrics["top_features"] = [f for f, _ in importances[:5]]

    save_metrics("regression", metrics)
    return metrics


if __name__ == "__main__":
    main()
