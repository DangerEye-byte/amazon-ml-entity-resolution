#!/usr/bin/env bash
# France with the round-3 self-trained transformer: role (v10ra) + fr + fr2 + fr3 stack, hard ef t=2, swap filter;
# merge with v11 India/US into submissions/v12 (France changes were audited by reading samples of flipped decisions).
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
mkdir -p models_v8ra_fr && cp config/decision_hard_ef2.json models_v8ra_fr/decision.json   # France rule: hard exclusivity, expected F0.5, t=2
C=colab_pkg/comb10; F3=colab_pkg/out_fr3
CE="fr=$C/fr_val.parquet,$C/fr_test.parquet fr2=$C/fr2_val.parquet,$C/fr2_test.parquet fr3=$F3/ce_val.parquet,$F3/ce_testband.parquet"
python src/stack_ce.py --val work10_sim/train/val_predv10ra.parquet --test work10/test/test_predv10ra.parquet \
  --ce $CE --out-tag v10rafr123 --min-p 0.02 | grep -E "logloss|wrote"
for dup in 1 2; do
  echo -n "France-safe + fr3 (proxy) decoy x$dup: "
  python src/decide.py --mode val --work work10_sim --frac 0.3 --drop-frac 0.19 --models models_tmp --decoy-dup $dup \
    --val-pred work10_sim/train/val_predv10rafr123.parquet --cfg '{"excl":"hard","method":"ef","t":2.0,"m":0.0}' | grep "^best" | cut -c1-12
done
echo "(v10 France-safe: 0.98978 / 0.98944)"
python src/decide.py --mode test --work work10 --models models_v8ra_fr --pred work10/test/test_predv10rafr123.parquet --out-dir output_v12fr3 | tail -1
PYTHONIOENCODING=utf-8 python src/swap_filter.py --work work10/test --pred work10/test/test_predv10rafr123.parquet \
  --in-dir output_v12fr3 --out-dir submissions/v12fr3_swap | grep dropping | cut -c1-90
cp output_v12fr3/candidate_pairs.tsv submissions/v12fr3_swap/
python src/hybrid.py --seen output_v11inus --unseen submissions/v12fr3_swap --out submissions/v12
python utils/validate_submission.py -m submissions/v12/matching_results.tsv -c submissions/v12/candidate_pairs.tsv -t dataset/test | tail -1
echo "V12 READY"
