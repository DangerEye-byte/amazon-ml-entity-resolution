#!/usr/bin/env bash
# v6: train/validate under simulated test conditions (19% of S1 deleted, orphaned S2/S3 clusters kept)
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
mkdir -p models_v6
{
echo "=== START $(date) ==="
python src/simulate_drop.py --drop-frac 0.19 --out-work work_sim
python src/prune.py --mode train --work work_sim --models models_v6
python src/make_features.py --stage full --split train --work work_sim
python src/train.py --neg-rate 1.0 --work work_sim --models models_v6 --threads 2
python src/decide.py --mode val --work work_sim --frac 0.3 --drop-frac 0.19 --models models_v6
echo "=== V6 TRAIN DONE $(date) ==="
} >> work/train_v6.log 2>&1 || echo "=== FAILED $(date) ===" >> work/train_v6.log
