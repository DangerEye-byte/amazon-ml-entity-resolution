#!/usr/bin/env bash
# v10 = v9 + dense-retrieval candidates (src/colab_dense.py on Colab -> src/merge_dense.py),
# rebuilt in work10/ + work10_sim/; stage-2 gets the dense rank/score features.
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
LOG=work/v10.log
DN=${DN:-dense_pkg/out_dense}
step() { echo "=== $1 $(date +%H:%M) ===" >> $LOG; }
fail() { echo "=== FAILED: $1 $(date +%H:%M) ===" >> $LOG; exit 1; }
step START
python src/merge_dense.py --split train --dense $DN/dense_train.parquet >> $LOG 2>&1 || fail "merge train"
step "MERGE TRAIN DONE"
mkdir -p models_v10
python src/make_features.py --stage block --split train --work work10 --frac 0.3 --cands cands_v10.parquet >> $LOG 2>&1 || fail "block train"
python src/simulate_drop.py --drop-frac 0.19 --work work10 --out-work work10_sim --cands cands_v10.parquet >> $LOG 2>&1 || fail "simulate"
python src/prune.py --mode train --work work10_sim --models models_v10 --max-loss 0.001 >> $LOG 2>&1 || fail "prune train"
python src/make_features.py --stage full --split train --work work10_sim >> $LOG 2>&1 || fail "full train"
python -c "
import json
f = json.load(open('models_v8/features.json'))
f += [c for c in ['dn_rrank', 'dn_frank', 'dn_score', 'dn_score_gmax', 'dn_score_gap', 'dn_score_grank'] if c not in f]
json.dump(f, open('models_v10/features.json', 'w'))
print(len(f), 'stage-2 base features')
" >> $LOG 2>&1 || fail "features.json"
python src/train_lex.py --work work10_sim --test-work work10 --models models_v10lx --tag v10lx --base-models models_v10 --no-test >> $LOG 2>&1 || fail "train_lex"
python src/decide.py --mode val --work work10_sim --frac 0.3 --drop-frac 0.19 --models models_tmp10 --val-pred work10_sim/train/val_predv10lx.parquet >> $LOG 2>&1 || fail "decide val lx"
python src/train_lex.py --work work10_sim --test-work work10 --models models_v10ra --tag v10ra --base-models models_v10 --agnostic --role --no-test >> $LOG 2>&1 || fail "train role"
step "V10 TRAIN DONE"
[ -f $DN/dense_test.parquet ] || fail "missing $DN/dense_test.parquet (run src/colab_dense.py first)"
python src/merge_dense.py --split test --dense $DN/dense_test.parquet >> $LOG 2>&1 || fail "merge test"
step "MERGE TEST DONE"
python src/make_features.py --stage block --split test --work work10 --cands cands_v10.parquet >> $LOG 2>&1 || fail "block test"
python src/prune.py --mode test --work work10 --models models_v10 >> $LOG 2>&1 || fail "prune test"
python src/make_features.py --stage full --split test --work work10 >> $LOG 2>&1 || fail "full test"
python src/score_lex.py --work work10_sim --test-work work10 --models models_v10lx --tag v10lx >> $LOG 2>&1 || fail "score lx"
python src/score_lex.py --work work10_sim --test-work work10 --models models_v10ra --tag v10ra >> $LOG 2>&1 || fail "score role"
python src/stage3.py --mode val --work work10_sim --models models_v10 --lexical --tag v10s --in-tag v10lx >> $LOG 2>&1 || fail "stage3 val"
python src/stage3.py --mode test --work work10_sim --test-work work10 --models models_v10 --lexical --tag v10s --in-tag v10lx >> $LOG 2>&1 || fail "stage3 test"
step "V10 DONE"
