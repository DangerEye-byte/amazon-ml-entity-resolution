#!/usr/bin/env bash
# Final submission (v15): India/US v11 stack + France round-3 stack, "at most one owner" (odds) exclusivity for France,
# evidence check on contested records (France: drop coin flips; India/US: restore evidence-backed top claimants).
# Needs: final_v11.sh outputs (output_v11inus), fr3_integrate.sh outputs (work10/test/test_predv10rafr123.parquet).
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
mkdir -p models_v13ofr
echo '{"cfg": {"excl": "odds", "method": "ef", "t": 2.0, "m": 0.0}}' > models_v13ofr/decision.json
python src/decide.py --mode test --work work10 --models models_v13ofr --pred work10/test/test_predv10rafr123.parquet --out-dir output_v13ofr
PYTHONIOENCODING=utf-8 python src/swap_filter.py --work work10/test --pred work10/test/test_predv10rafr123.parquet \
  --in-dir output_v13ofr --out-dir submissions/v13ofr_swap
cp output_v13ofr/candidate_pairs.tsv submissions/v13ofr_swap/
python src/hybrid.py --seen output_v11inus --unseen submissions/v13ofr_swap --out submissions/v13o
python src/contest_filter.py --mode drop --countries fr --pred work10/test/test_predv10rafr123.parquet \
  --in-dir submissions/v13ofr_swap --out-dir submissions/v15bfr
python src/contest_filter.py --mode restore --countries in us --pred work10/test/test_predv11inus.parquet \
  --in-dir output_v11inus --out-dir output_v15binus --rival-p 0.3
python src/hybrid.py --seen output_v15binus --unseen submissions/v15bfr --out submissions/v15b
python utils/validate_submission.py -m submissions/v15b/matching_results.tsv -c submissions/v15b/candidate_pairs.tsv -t dataset/test
