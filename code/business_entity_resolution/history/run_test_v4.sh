#!/usr/bin/env bash
# v4 test pipeline: sibling expansion -> block -> prune (v4) -> full -> predict -> decide -> validate
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
LOG=work/test_v4.log
{
echo "=== START $(date) ==="
python src/expand.py --split test --workers 2
python src/make_features.py --stage block --split test --cands cands_all.parquet
echo "=== TEST BLOCK V4 DONE ==="
mkdir -p models_v3_backup && cp models/* models_v3_backup/ && cp models_v4/* models/
python src/prune.py --mode test
python src/make_features.py --stage full --split test
python src/predict.py
python src/decide.py --mode test
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
mkdir -p submissions/v4 && cp output/matching_results.tsv output/candidate_pairs.tsv submissions/v4/
echo "=== SUBMISSION V4 READY $(date) ==="
} >> "$LOG" 2>&1 || echo "=== FAILED $(date) ===" >> "$LOG"
