"""
02 — Classification: "good" wine (quality >= 7) vs the rest.

Models: Logistic Regression, SVM (RBF), Random Forest, k-NN.
Metrics: accuracy, precision, recall, F1, ROC-AUC + confusion matrix.
The positive class is a minority (~13%), so F1 / ROC-AUC matter most.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             RocCurveDisplay)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.data_loader import RANDOM_STATE, load_wine, classification_split
from src.evaluate import fig_path, print_table, save_metrics
from src.features import scale_train_test

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "SVM (RBF)": CalibratedClassifierCV(SVC(kernel="rbf", random_state=RANDOM_STATE)),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                            n_jobs=-1),
    "k-NN": KNeighborsClassifier(n_neighbors=15),
}


def main() -> dict:
    df = load_wine()
    X_train, X_test, y_train, y_test = classification_split(df)
    print(f"Class balance: {(y_train == 1).mean():.1%} good wines in train")
    X_train_s, X_test_s, _ = scale_train_test(X_train, X_test)

    rows, metrics, fitted = [], {}, {}
    for name, model in MODELS.items():
        # Tree models use raw features; distance/linear models use scaled.
        use_scaled = name in {"Logistic Regression", "SVM (RBF)", "k-NN"}
        Xt_tr, Xt_te = (X_train_s, X_test_s) if use_scaled else (X_train, X_test)
        model.fit(Xt_tr, y_train)
        pred = model.predict(Xt_te)
        proba = model.predict_proba(Xt_te)[:, 1]
        row = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }
        rows.append(row)
        metrics[name] = {k: round(v, 4) for k, v in row.items() if k != "model"}
        fitted[name] = (model, Xt_te)

    print_table("CLASSIFICATION — good wine vs rest (holdout test)",
                rows, ["model", "accuracy", "precision", "recall", "f1", "roc_auc"])

    # Confusion matrix for the best-F1 model.
    best = max(rows, key=lambda r: r["f1"])["model"]
    model, Xt_te = fitted[best]
    cm = confusion_matrix(y_test, model.predict(Xt_te))
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion matrix — {best}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    plt.tight_layout()
    out = fig_path("classification_confusion_rf.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  figure -> {out}")

    # ROC curves for all models.
    plt.figure(figsize=(6, 5))
    for name in MODELS:
        model, Xt_te = fitted[name]
        RocCurveDisplay.from_estimator(model, Xt_te, y_test, name=name, ax=plt.gca())
    plt.plot([0, 1], [0, 1], "k--", label="chance")
    plt.title("ROC curves — good-wine classification")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = fig_path("classification_roc.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  figure -> {out}")

    metrics["best_model"] = best
    save_metrics("classification", metrics)
    return metrics


if __name__ == "__main__":
    main()
