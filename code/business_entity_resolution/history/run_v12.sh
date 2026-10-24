#!/usr/bin/env bash
# v12: v11 with population-matched collective features (counts in the simulation run over a subsample sized like
# the test split per country, so frequency features mean the same thing in training and on test).
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
LOG=work/v12.log
step() { echo "=== $1 $(date +%H:%M) ===" >> $LOG; }
fail() { echo "=== FAILED: $1 $(date +%H:%M) ===" >> $LOG; exit 1; }
step START
python src/train_lex.py --work work10_sim --test-work work10 --models models_v10lxp --tag v10lxp --base-models models_v10 \
  --no-test --extra work10_sim/train/coll_p.parquet >> $LOG 2>&1 || fail "train lx"
python src/stage3.py --mode val --work work10_sim --models models_v10 --lexical --tag v10sp --in-tag v10lxp >> $LOG 2>&1 || fail "stage3 val"
python src/greedy_stack.py build s3=v10sp s2=v10lxp role=v10rac out=work10_sim/train/stackx_val_p.parquet >> $LOG 2>&1 || fail "matrix"
step "VAL READY"
python src/score_lex.py --work work10_sim --test-work work10 --models models_v10lxp --tag v10lxp --extra work10/test/coll.parquet >> $LOG 2>&1 || fail "score lx"
python src/stage3.py --mode test --work work10_sim --test-work work10 --models models_v10 --lexical --tag v10sp --in-tag v10lxp >> $LOG 2>&1 || fail "stage3 test"
step "V12 DONE"
