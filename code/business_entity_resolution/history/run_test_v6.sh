#!/usr/bin/env bash
# v6 test scoring (after the v4 test block stage and history/run_train_v6.sh)
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
LOG=work/test_v6.log
# The original run waited here for the v4 test chain (history/run_test_v4.sh) to finish its block stage, stopped
# that chain, and waited for history/run_train_v6.sh; run those first.
grep -q "V6 TRAIN DONE" work/train_v6.log || { echo "run history/run_train_v6.sh first"; exit 1; }
{
python src/prune.py --mode test --models models_v6
python src/make_features.py --stage full --split test
python src/predict.py --models models_v6
python src/decide.py --mode test --models models_v6
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
mkdir -p submissions/v6 && cp output/matching_results.tsv output/candidate_pairs.tsv submissions/v6/
echo "=== SUBMISSION V6 READY $(date) ==="
} >> $LOG 2>&1 || echo "=== FAILED $(date) ===" >> $LOG
