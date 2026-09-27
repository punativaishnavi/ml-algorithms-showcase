#!/usr/bin/env bash
# Run all three ML experiments end to end.
set -euo pipefail
cd "$(dirname "$0")"
python experiments/01_regression.py
python experiments/02_classification.py
python experiments/03_clustering.py
echo ""
echo "All experiments complete. See results/metrics.json and results/figures/."
