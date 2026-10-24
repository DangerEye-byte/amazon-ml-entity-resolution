#!/usr/bin/env bash
# v10c: stage-2 models (lexical India/US and word-role France) retrained with the collective / density /
# raw-name twin features (src/coll_feats.py), then stage-3 and a rebuilt stack search matrix.
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
LOG=work/v10c.log
step() { echo "=== $1 $(date +%H:%M) ===" >> $LOG; }
fail() { echo "=== FAILED: $1 $(date +%H:%M) ===" >> $LOG; exit 1; }
step START
python src/train_lex.py --work work10_sim --test-work work10 --models models_v10lxc --tag v10lxc --base-models models_v10 \
  --no-test --extra work10_sim/train/coll.parquet >> $LOG 2>&1 || fail "train lx"
python src/stage3.py --mode val --work work10_sim --models models_v10 --lexical --tag v10sc --in-tag v10lxc >> $LOG 2>&1 || fail "stage3 val"
python src/greedy_stack.py build s3=v10sc s2=v10lxc role=v10ra out=work10_sim/train/stackx_val_c.parquet >> $LOG 2>&1 || fail "matrix"
step "VAL READY"
python src/train_lex.py --work work10_sim --test-work work10 --models models_v10rac --tag v10rac --base-models models_v10 \
  --agnostic --role --no-test --extra work10_sim/train/coll.parquet >> $LOG 2>&1 || fail "train role"
step "ROLE TRAINED"
[ -f work10/test/coll.parquet ] || fail "missing work10/test/coll.parquet (run src/coll_feats.py --split-dir work10/test first)"
python src/score_lex.py --work work10_sim --test-work work10 --models models_v10lxc --tag v10lxc --extra work10/test/coll.parquet >> $LOG 2>&1 || fail "score lx"
python src/score_lex.py --work work10_sim --test-work work10 --models models_v10rac --tag v10rac --extra work10/test/coll.parquet >> $LOG 2>&1 || fail "score role"
python src/stage3.py --mode test --work work10_sim --test-work work10 --models models_v10 --lexical --tag v10sc --in-tag v10lxc >> $LOG 2>&1 || fail "stage3 test"
step "V10C DONE"
