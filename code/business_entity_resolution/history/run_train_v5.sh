#!/usr/bin/env bash
# v5: v4 candidates (with expansion) but no reverse-competition features (r_*), which shift on test
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
mkdir -p models_v5
{
echo "=== START $(date) ==="
python src/prune.py --mode train --models models_v5 --drop r_
python src/make_features.py --stage full --split train
python src/train.py --neg-rate 1.0 --models models_v5 --drop r_ --threads 2
python src/decide.py --mode val --frac 0.3 --models models_v5
echo "=== V5 TRAIN DONE $(date) ==="
} >> work/train_v5.log 2>&1 || echo "=== FAILED $(date) ===" >> work/train_v5.log
