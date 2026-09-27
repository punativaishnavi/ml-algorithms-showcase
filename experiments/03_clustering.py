"""
03 — Clustering (unsupervised): do wines group naturally by quality?

Methods: K-Means (k chosen by elbow + silhouette, k=2..8) and DBSCAN
(eps tuned on a small grid). PCA projects the 11-D feature space to 2-D
for visualization; true quality bands are overlaid for comparison.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.data_loader import RANDOM_STATE, load_wine
from src.evaluate import fig_path, print_table, save_metrics
from src.features import scale_all


def main() -> dict:
    df = load_wine()
    X = df.drop(columns=["quality"])
    Xs, _ = scale_all(X)
    metrics: dict = {}

    # ---- K-Means: pick k by silhouette ----
    rows = []
    inertias, sils, models = {}, {}, {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(Xs)
        inertias[k] = float(km.inertia_)
        sils[k] = float(silhouette_score(Xs, labels))
        models[k] = km
        rows.append({"k": k, "inertia": inertias[k], "silhouette": sils[k]})
    print_table("K-MEANS — model selection", rows, ["k", "inertia", "silhouette"])

    best_k = max(sils, key=sils.get)
    km = models[best_k]
    metrics["kmeans"] = {"best_k": best_k, "silhouette": round(sils[best_k], 4),
                         "inertia": round(inertias[best_k], 2)}
    print(f"\nBest k = {best_k} (silhouette {sils[best_k]:.3f})")

    # ---- DBSCAN: tune eps (ignore degenerate solutions: >=2 clusters, <50% noise) ----
    db_rows = []
    best = None
    n_total = len(Xs)
    for eps in [0.8, 1.0, 1.2, 1.5, 2.0]:
        db = DBSCAN(eps=eps, min_samples=10)
        labels = db.fit_predict(Xs)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        sil = None
        valid = n_clusters >= 2 and n_noise < 0.5 * n_total
        if valid:
            sil = float(silhouette_score(Xs[labels != -1], labels[labels != -1]))
        db_rows.append({"eps": eps, "clusters": n_clusters,
                        "noise_pts": n_noise,
                        "silhouette": round(sil, 4) if sil is not None else "n/a"})
        if sil is not None and (best is None or sil > best[0]):
            best = (sil, eps, labels)
    print_table("DBSCAN — eps tuning", db_rows,
                ["eps", "clusters", "noise_pts", "silhouette"])
    if best:
        metrics["dbscan"] = {"eps": best[1], "silhouette": round(best[0], 4)}
        print(f"\nBest eps = {best[1]} (silhouette {best[0]:.3f})")
    else:
        metrics["dbscan"] = {"eps": None, "silhouette": None}

    # ---- PCA visualization: clusters vs true quality ----
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X2 = pca.fit_transform(Xs)
    print(f"\nPCA explained variance (2 comps): {pca.explained_variance_ratio_.sum():.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].scatter(X2[:, 0], X2[:, 1], c=km.labels_, cmap="tab10", s=12, alpha=0.7)
    axes[0].set_title(f"K-Means clusters (k={best_k})")
    sc = axes[1].scatter(X2[:, 0], X2[:, 1], c=df["quality"], cmap="viridis",
                         s=12, alpha=0.7)
    axes[1].set_title("True quality scores")
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=axes[1], label="quality")
    fig.suptitle("Unsupervised structure vs ground truth (PCA)")
    fig.tight_layout()
    out = fig_path("clustering_pca.png")
    fig.savefig(out, dpi=120)
    plt.close()
    print(f"  figure -> {out}")

    # How pure are the K-Means clusters w.r.t. quality?
    df_tmp = df.copy()
    df_tmp["cluster"] = km.labels_
    purity = (df_tmp.groupby("cluster")["quality"]
              .agg(lambda s: (s == s.mode().iloc[0]).mean()).mean())
    metrics["kmeans"]["avg_cluster_purity"] = round(float(purity), 4)
    print(f"Avg. K-Means cluster purity vs majority quality: {purity:.3f}")

    save_metrics("clustering", metrics)
    return metrics


if __name__ == "__main__":
    main()
