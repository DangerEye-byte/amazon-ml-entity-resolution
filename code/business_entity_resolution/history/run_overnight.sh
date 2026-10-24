#!/usr/bin/env bash
# overnight: decoy-aware and second-seed lexical stage-2 models, scored on val and test
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
LOG=work/overnight.log
echo "=== START $(date +%H:%M)" >> $LOG
python src/train_lex.py --work work_sim --test-work work --models models_v7lxd --tag lxd --decoy-weight 2.0 --no-test >> $LOG 2>&1 && \
python src/score_lex.py --work work_sim --test-work work --models models_v7lxd --tag lxd >> $LOG 2>&1 && echo "=== LXD DONE $(date +%H:%M)" >> $LOG
python src/train_lex.py --work work_sim --test-work work --models models_v7lxs --tag lxs --seed 7 --leaves 255 --lr 0.04 --no-test >> $LOG 2>&1 && \
python src/score_lex.py --work work_sim --test-work work --models models_v7lxs --tag lxs >> $LOG 2>&1 && echo "=== LXS DONE $(date +%H:%M)" >> $LOG
echo "=== ALL DONE $(date +%H:%M)" >> $LOG
