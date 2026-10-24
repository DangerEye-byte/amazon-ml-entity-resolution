#!/usr/bin/env bash
# End-to-end pipeline. Run from code/business_entity_resolution/.
# Every stage writes its artifacts under work/ and models/, so a crashed or
# disconnected session can resume from the last finished stage.
set -euo pipefail
export PYTHONHASHSEED=0
FRAC=${FRAC:-0.12}      # fraction of train S1 rows used for features/training
K=${K:-20}              # candidates kept per S1 per source (S2 and S3)

[ -f work/translit.json ]          || python src/translit.py --train-dir dataset/train --test-dir dataset/test
[ -f work/train/s3.parquet ]       || python src/prep.py --split train
[ -f work/test/s3.parquet ]        || python src/prep.py --split test
[ -f work/train/cands.parquet ]    || python src/retrieve.py --split train --k "$K"
[ -f work/test/cands.parquet ]     || python src/retrieve.py --split test --k "$K"
# stage 1: cheap blocking features -> learned pruning (final candidate set)
python src/make_features.py --stage block --split train --frac "$FRAC" --resume
python src/prune.py --mode train
# stage 2: full features on survivors -> matcher -> decision rule
python src/make_features.py --stage full --split train
python src/train.py
python src/decide.py --mode val --frac "$FRAC"
python src/make_features.py --stage block --split test --resume
python src/prune.py --mode test
python src/make_features.py --stage full --split test
python src/predict.py
python src/decide.py --mode test
python utils/validate_submission.py --matching output/matching_results.tsv     --candidate output/candidate_pairs.tsv --test-dir dataset/test
