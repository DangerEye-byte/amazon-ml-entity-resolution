#!/usr/bin/env bash
# v7 = v6 (simulated S1 deletion) + reverse blocking (record -> S1)
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
LOG=work/v7.log
step() { echo "=== $1 $(date +%H:%M) ===" >> $LOG; }
fail() { echo "=== FAILED: $1 $(date +%H:%M) ===" >> $LOG; exit 1; }
step START
python src/retrieve_rev.py --split train --drop-frac 0.19 --workers 2 >> $LOG 2>&1 || fail "rev train"
[ -f work/test/cands_all.parquet ] || fail "missing work/test/cands_all.parquet (run src/expand.py --split test first)"
python src/retrieve_rev.py --split test --workers 2 >> $LOG 2>&1 || fail "rev test"
step "REVERSE DONE"
# (the original run waited here for history/run_train_v6.sh, which shared the CPU)
mkdir -p models_v7
python src/make_features.py --stage block --split train --frac 0.3 --cands cands_full.parquet >> $LOG 2>&1 || fail "block train"
python src/simulate_drop.py --drop-frac 0.19 --out-work work_sim --cands cands_full.parquet >> $LOG 2>&1 || fail "simulate"
python src/prune.py --mode train --work work_sim --models models_v7 >> $LOG 2>&1 || fail "prune train"
python src/make_features.py --stage full --split train --work work_sim >> $LOG 2>&1 || fail "full train"
python src/train.py --neg-rate 1.0 --work work_sim --models models_v7 >> $LOG 2>&1 || fail "train"
python src/decide.py --mode val --work work_sim --frac 0.3 --drop-frac 0.19 --models models_v7 >> $LOG 2>&1 || fail "decide val"
step "V7 TRAIN DONE"
# (the original run waited here for history/run_test_v6.sh, which shared work/test)
python src/make_features.py --stage block --split test --cands cands_full.parquet >> $LOG 2>&1 || fail "block test"
python src/prune.py --mode test --models models_v7 >> $LOG 2>&1 || fail "prune test"
python src/make_features.py --stage full --split test >> $LOG 2>&1 || fail "full test"
python src/predict.py --models models_v7 >> $LOG 2>&1 || fail "predict"
mkdir -p output_v7
python src/decide.py --mode test --models models_v7 --out-dir output_v7 >> $LOG 2>&1 || fail "decide test"
python utils/validate_submission.py --matching output_v7/matching_results.tsv --candidate output_v7/candidate_pairs.tsv --test-dir dataset/test >> $LOG 2>&1 || fail "validate"
mkdir -p submissions/v7 && cp output_v7/*.tsv submissions/v7/
step "SUBMISSION V7 READY"
