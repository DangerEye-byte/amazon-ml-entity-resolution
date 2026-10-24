#!/usr/bin/env bash
# usage: build_stage.sh TAG  -> submissions/v7s${TAG}_swap (stage-3 with lexical features, decoy-tuned decision, France filter)
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
TAG=$1
LOG=work/stage${TAG}_test.log
: > $LOG
python src/stage3.py --mode test --work work_sim --test-work work --models models_v7 --lexical --tag $TAG --in-tag "${IN_TAG:-}" >> $LOG 2>&1 || { echo FAILED >> $LOG; exit 1; }
mkdir -p models_v7s$TAG
python src/decide.py --mode val --work work_sim --frac 0.3 --drop-frac 0.19 --models models_v7s$TAG --decoy-dup 2 --val-pred work_sim/train/val_pred$TAG.parquet >> $LOG 2>&1 || { echo FAILED >> $LOG; exit 1; }
python src/decide.py --mode test --models models_v7s$TAG --pred work/test/test_pred$TAG.parquet --out-dir output_v7s$TAG >> $LOG 2>&1 || { echo FAILED >> $LOG; exit 1; }
python src/swap_filter.py --pred work/test/test_pred$TAG.parquet --in-dir output_v7s$TAG --out-dir submissions/v7s${TAG}_swap >> $LOG 2>&1 || { echo FAILED >> $LOG; exit 1; }
cp output_v7s$TAG/candidate_pairs.tsv submissions/v7s${TAG}_swap/
python utils/validate_submission.py -m submissions/v7s${TAG}_swap/matching_results.tsv -c submissions/v7s${TAG}_swap/candidate_pairs.tsv -t dataset/test >> $LOG 2>&1 || { echo FAILED >> $LOG; exit 1; }
echo BUILD_DONE >> $LOG
