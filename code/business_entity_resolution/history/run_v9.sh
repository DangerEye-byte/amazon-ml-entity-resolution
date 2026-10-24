#!/usr/bin/env bash
# v9 = v8 with a wider re-ranked reverse search (top-100, keep 3, looser tie rule, France included),
# rebuilt in work9/ + work9_sim/; India/US from the lexical model, France from the word-role model.
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
LOG=work/v9.log
step() { echo "=== $1 $(date +%H:%M) ===" >> $LOG; }
fail() { echo "=== FAILED: $1 $(date +%H:%M) ===" >> $LOG; exit 1; }
step START
RR="--k 100 --keep 3 --tie 0.8 --max-ties 3 --out-work work9 --out-name cands_v9.parquet --workers 2"
python src/retrieve_rerank.py --split train --drop-frac 0.19 $RR >> $LOG 2>&1 || fail "rerank train"
python src/retrieve_rerank.py --split test --all-countries $RR >> $LOG 2>&1 || fail "rerank test"
step "RERANK DONE"
mkdir -p models_v9
python src/make_features.py --stage block --split train --work work9 --frac 0.3 --cands cands_v9.parquet >> $LOG 2>&1 || fail "block train"
python src/simulate_drop.py --drop-frac 0.19 --work work9 --out-work work9_sim --cands cands_v9.parquet >> $LOG 2>&1 || fail "simulate"
python src/prune.py --mode train --work work9_sim --models models_v9 >> $LOG 2>&1 || fail "prune train"
python src/make_features.py --stage full --split train --work work9_sim >> $LOG 2>&1 || fail "full train"
cp models_v8/features.json models_v9/features.json
python src/train_lex.py --work work9_sim --test-work work9 --models models_v9lx --tag v9lx --base-models models_v9 --no-test >> $LOG 2>&1 || fail "train_lex"
python src/decide.py --mode val --work work9_sim --frac 0.3 --drop-frac 0.19 --models models_tmp9 --val-pred work9_sim/train/val_predv9lx.parquet >> $LOG 2>&1 || fail "decide val lx"
python src/train_lex.py --work work9_sim --test-work work9 --models models_v9ra --tag v9ra --base-models models_v9 --agnostic --role --no-test >> $LOG 2>&1 || fail "train role"
step "V9 TRAIN DONE"
python src/make_features.py --stage block --split test --work work9 --cands cands_v9.parquet >> $LOG 2>&1 || fail "block test"
python src/prune.py --mode test --work work9 --models models_v9 >> $LOG 2>&1 || fail "prune test"
python src/make_features.py --stage full --split test --work work9 >> $LOG 2>&1 || fail "full test"
python src/score_lex.py --work work9_sim --test-work work9 --models models_v9lx --tag v9lx >> $LOG 2>&1 || fail "score lx"
python src/score_lex.py --work work9_sim --test-work work9 --models models_v9ra --tag v9ra >> $LOG 2>&1 || fail "score role"
python src/stage3.py --mode val --work work9_sim --models models_v9 --lexical --tag v9s --in-tag v9lx >> $LOG 2>&1 || fail "stage3 val"
python src/stage3.py --mode test --work work9_sim --test-work work9 --models models_v9 --lexical --tag v9s --in-tag v9lx >> $LOG 2>&1 || fail "stage3 test"
step "V9 DONE"
