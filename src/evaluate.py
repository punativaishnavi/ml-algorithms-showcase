"""Shared evaluation utilities: metrics tables, plots, metrics.json logging."""

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")

sns.set_theme(style="whitegrid")


def _ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def save_metrics(experiment: str, metrics: dict) -> None:
    """Merge this experiment's metrics into results/metrics.json."""
    _ensure_dirs()
    data = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    data[experiment] = metrics
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"  metrics -> {METRICS_PATH}")


def fig_path(name: str) -> str:
    _ensure_dirs()
    return os.path.join(FIGURES_DIR, name)


def print_table(title: str, rows: list[dict], columns: list[str]) -> None:
    print(f"\n{title}")
    widths = {c: max(len(c), max(len(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]))
                                 for r in rows)) for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(
            (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])).ljust(widths[c])
            for c in columns))
