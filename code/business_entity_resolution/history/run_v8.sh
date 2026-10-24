#!/usr/bin/env bash
# v8 = v7 + wide reverse search with fine re-ranking for common-name records (retrieve_rerank.py),
# rebuilt end to end in separate work dirs (work8, work8_sim) so v7 artifacts stay intact.
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
LOG=work/v8.log
step() { echo "=== $1 $(date +%H:%M) ===" >> $LOG; }
fail() { echo "=== FAILED: $1 $(date +%H:%M) ===" >> $LOG; exit 1; }
step START
python src/retrieve_rerank.py --split train --drop-frac 0.19 --out-work work8 --workers 2 >> $LOG 2>&1 || fail "rerank train"
python src/retrieve_rerank.py --split test --out-work work8 --workers 2 >> $LOG 2>&1 || fail "rerank test"
step "RERANK DONE"
mkdir -p models_v8
python src/make_features.py --stage block --split train --work work8 --frac 0.3 --cands cands_v8.parquet >> $LOG 2>&1 || fail "block train"
python src/simulate_drop.py --drop-frac 0.19 --work work8 --out-work work8_sim --cands cands_v8.parquet >> $LOG 2>&1 || fail "simulate"
python src/prune.py --mode train --work work8_sim --models models_v8 >> $LOG 2>&1 || fail "prune train"
python src/make_features.py --stage full --split train --work work8_sim >> $LOG 2>&1 || fail "full train"
python -c "import json; f = json.load(open('models_v7/features.json')); json.dump(f + ['rr_rank', 'rr_score'], open('models_v8/features.json', 'w'))" || fail "features.json"
python src/train_lex.py --work work8_sim --test-work work8 --models models_v8lx --tag v8lx --base-models models_v8 --no-test >> $LOG 2>&1 || fail "train_lex"
python src/decide.py --mode val --work work8_sim --frac 0.3 --drop-frac 0.19 --models models_tmp8 --val-pred work8_sim/train/val_predv8lx.parquet >> $LOG 2>&1 || fail "decide val lx"
step "V8 TRAIN DONE"
python src/make_features.py --stage block --split test --work work8 --cands cands_v8.parquet >> $LOG 2>&1 || fail "block test"
python src/prune.py --mode test --work work8 --models models_v8 >> $LOG 2>&1 || fail "prune test"
python src/make_features.py --stage full --split test --work work8 >> $LOG 2>&1 || fail "full test"
python src/score_lex.py --work work8_sim --test-work work8 --models models_v8lx --tag v8lx >> $LOG 2>&1 || fail "score_lex"
python src/stage3.py --mode val --work work8_sim --models models_v8 --lexical --tag v8s --in-tag v8lx >> $LOG 2>&1 || fail "stage3 val"
mkdir -p models_v8s
python src/decide.py --mode val --work work8_sim --frac 0.3 --drop-frac 0.19 --models models_v8s --decoy-dup 2 --val-pred work8_sim/train/val_predv8s.parquet >> $LOG 2>&1 || fail "decide val s"
python src/stage3.py --mode test --work work8_sim --test-work work8 --models models_v8 --lexical --tag v8s --in-tag v8lx >> $LOG 2>&1 || fail "stage3 test"
python src/decide.py --mode test --work work8 --models models_v8s --pred work8/test/test_predv8s.parquet --out-dir output_v8s >> $LOG 2>&1 || fail "decide test"
python src/hybrid.py --seen output_v8s --unseen submissions/v7_swap --out submissions/v8h >> $LOG 2>&1 || fail "hybrid"
python utils/validate_submission.py -m submissions/v8h/matching_results.tsv -c submissions/v8h/candidate_pairs.tsv -t dataset/test >> $LOG 2>&1 || fail "validate"
step "SUBMISSION V8H READY"
