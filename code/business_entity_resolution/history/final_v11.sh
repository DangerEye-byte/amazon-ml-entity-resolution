#!/usr/bin/env bash
# v11 submission: final stacks (src/final_v11.py) -> decisions -> France swap filter -> merge -> validate.
#   India/US: soft exclusivity, threshold 0.75.  France: hard exclusivity, expected F0.5 (t = 2), swap filter.
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
mkdir -p models_v8ra_fr && cp config/decision_hard_ef2.json models_v8ra_fr/decision.json   # France rule: hard exclusivity, expected F0.5, t=2
OUT=${OUT:-v11}
python src/final_v11.py
mkdir -p models_v11inus
echo '{"cfg": {"excl": "soft", "method": "thr", "t": 0.75, "m": 0.0}}' > models_v11inus/decision.json
python src/decide.py --mode test --work work10 --models models_v11inus --pred work10/test/test_predv11inus.parquet --out-dir output_v11inus
python src/decide.py --mode test --work work10 --models models_v8ra_fr --pred work10/test/test_predv11fr.parquet --out-dir output_v11fr
PYTHONIOENCODING=utf-8 python src/swap_filter.py --work work10/test --pred work10/test/test_predv11fr.parquet \
  --in-dir output_v11fr --out-dir submissions/v11fr_swap | grep dropping | cut -c1-100
cp output_v11fr/candidate_pairs.tsv submissions/v11fr_swap/
python src/hybrid.py --seen output_v11inus --unseen submissions/v11fr_swap --out submissions/$OUT
python utils/validate_submission.py -m submissions/$OUT/matching_results.tsv -c submissions/$OUT/candidate_pairs.tsv -t dataset/test | tail -1
echo "FINAL READY: submissions/$OUT"
