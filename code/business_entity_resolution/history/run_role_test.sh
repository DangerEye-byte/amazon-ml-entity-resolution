#!/usr/bin/env bash
# after run_role.sh: score test with the word-role model and build the France-from-role candidate
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
LOG=work/role.log
grep -q "=== DONE" $LOG || { echo "run history/run_role.sh first"; exit 1; }   # originally waited for it
echo "=== ROLE TEST START $(date +%H:%M)" >> $LOG
python src/score_lex.py --work work8_sim --test-work work8 --models models_v8ra --tag v8ra >> $LOG 2>&1 && \
python src/decide.py --mode test --work work8 --models models_v8ra --pred work8/test/test_predv8ra.parquet --out-dir output_v8ra >> $LOG 2>&1 && \
python src/swap_filter.py --work work8/test --pred work8/test/test_predv8ra.parquet --in-dir output_v8ra --out-dir submissions/v8ra_swap >> $LOG 2>&1 && \
cp output_v8ra/candidate_pairs.tsv submissions/v8ra_swap/ && echo "=== ROLE TEST DONE $(date +%H:%M)" >> $LOG
