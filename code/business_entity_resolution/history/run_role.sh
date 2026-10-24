#!/usr/bin/env bash
# language-agnostic + word-role stage-2 model on v8 candidates (for France), validated on India/US
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
LOG=work/role.log
echo "=== START $(date +%H:%M)" > $LOG
python src/train_lex.py --work work8_sim --test-work work8 --models models_v8ra --tag v8ra --base-models models_v8 --agnostic --role --no-test >> $LOG 2>&1 && \
python src/decide.py --mode val --work work8_sim --frac 0.3 --drop-frac 0.19 --models models_v8ra --val-pred work8_sim/train/val_predv8ra.parquet >> $LOG 2>&1 && \
python src/train_lex.py --work work8_sim --test-work work8 --models models_v8a --tag v8a --base-models models_v8 --agnostic --no-test >> $LOG 2>&1 && \
python src/decide.py --mode val --work work8_sim --frac 0.3 --drop-frac 0.19 --models models_tmp8 --val-pred work8_sim/train/val_predv8a.parquet >> $LOG 2>&1
echo "=== DONE $(date +%H:%M)" >> $LOG
