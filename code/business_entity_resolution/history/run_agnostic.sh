#!/usr/bin/env bash
# after run_overnight.sh: language-agnostic lexical stage-2 (for countries unseen in training)
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
LOG=work/overnight.log
grep -q "ALL DONE" $LOG || { echo "run history/run_overnight.sh first"; exit 1; }   # originally waited for it
echo "=== AGNOSTIC START $(date +%H:%M)" >> $LOG
python src/train_lex.py --work work_sim --test-work work --models models_v7lxa --tag lxa --agnostic --no-test >> $LOG 2>&1 && \
python src/score_lex.py --work work_sim --test-work work --models models_v7lxa --tag lxa >> $LOG 2>&1 && echo "=== AGNOSTIC DONE $(date +%H:%M)" >> $LOG
