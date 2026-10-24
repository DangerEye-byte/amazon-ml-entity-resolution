#!/usr/bin/env bash
# v10 submission from the finished v10 chain (run_v10.sh) and the transformer scores of its new uncertain pairs
# (Colab colab_ce.py --model-dir <model> : validation -> colab_v10/out_new10v_<name>, test -> colab_v10/out_new10_<name>).
#   India/US: stage-3 (v10s) + transformer stack ($MODELS), hardsoft exclusivity, expected F0.5 (t=1)
#   France:   word-role model (v10ra) + both self-trained French transformers (fr, fr2), hard ef t=2, swap filter
#   Candidate set = pairs whose input probability (stage-3 for India/US, role model for France) exceeds $MINP;
#   the transformer stack (the final model) is trained and applied only on those pairs, so candidate_pairs.tsv
#   is exactly the set it scores.
# Usage: MINP=0.02 bash final_v10.sh
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
mkdir -p models_v8ra_fr && cp config/decision_hard_ef2.json models_v8ra_fr/decision.json   # France rule: hard exclusivity, expected F0.5, t=2
MINP=${MINP:-0.02}
MODELS=${MODELS:-"mini e5b e5l e5l2 labse e5lv10"}
OUT=${OUT:-v10}
COLL=""; [ "${COLLECTIVE:-1}" = "1" ] && COLL="--collective work10_sim/train work10/test"   # name/address frequency features
P=colab_pkg; N=colab_v10; C=$P/comb10
declare -A VAL=([mini]="$P/out/ce_val.parquet $P/out_new_mini/ce_val.parquet"
                [e5b]="$P/out_e5b/ce_val.parquet $P/out_new_e5b/ce_val.parquet"
                [e5l]="$P/out_e5l/ce_val.parquet $P/out_new_e5l/ce_val.parquet"
                [e5l2]="$P/out_e5l2/ce_val.parquet $P/out_new_e5l2/ce_val.parquet"
                [labse]="$P/out_labse/ce_val.parquet" [fr]="$P/out_fr/ce_val.parquet" [fr2]="$P/out_fr2/ce_val.parquet"
                [e5lv10]="$P/out_e5lv10/ce_val.parquet")
declare -A TEST=([mini]="$P/out/ce_testband.parquet $P/out_new_mini/ce_testband.parquet"
                 [e5b]="$P/out_e5b/ce_testband.parquet $P/out_new_e5b/ce_testband.parquet"
                 [e5l]="$P/out_e5l/ce_testband.parquet $P/out_new_e5l/ce_testband.parquet"
                 [e5l2]="$P/out_e5l2/ce_testband.parquet $P/out_new_e5l2/ce_testband.parquet"
                 [labse]="$P/out_labse/ce_testband.parquet" [fr]="$P/out_fr/ce_testband.parquet" [fr2]="$P/out_fr2/ce_testband.parquet"
                 [e5lv10]="")
for m in $MODELS; do
  python src/comb_ce.py --name $m --out $C --val ${VAL[$m]} $N/out_new10v_$m/ce_val.parquet \
    --test ${TEST[$m]} $N/out_new10_$m/ce_testband.parquet \
    --val-band work10_sim/train/val_predv10lx.parquet --test-band work10/test/test_predv10lx.parquet
done
for m in fr fr2; do
  python src/comb_ce.py --name $m --out $C --val ${VAL[$m]} $N/out_new10v_$m/ce_val.parquet \
    --test ${TEST[$m]} $N/out_new10_$m/ce_testband.parquet \
    --val-band work10_sim/train/val_predv10ra.parquet --test-band work10/test/test_predv10ra.parquet
done
# India/US
CES=""; for m in $MODELS; do CES="$CES $m=$C/${m}_val.parquet,$C/${m}_test.parquet"; done
python src/stack_ce.py --val work10_sim/train/val_predv10s.parquet --test work10/test/test_predv10s.parquet \
  --ce $CES --out-tag v10sce --min-p $MINP $COLL
for dup in 1 2; do
  echo -n "v10sce decoy x$dup hardsoft ef t1: "
  python src/decide.py --mode val --work work10_sim --frac 0.3 --drop-frac 0.19 --models models_tmp --decoy-dup $dup \
    --val-pred work10_sim/train/val_predv10sce.parquet --cfg '{"excl":"hardsoft","method":"ef","t":1.0,"m":0.0}' | grep "^best"
done
mkdir -p models_v10sce
echo '{"cfg": {"excl": "hardsoft", "method": "ef", "t": 1.0, "m": 0.0}}' > models_v10sce/decision.json
python src/decide.py --mode test --work work10 --models models_v10sce --pred work10/test/test_predv10sce.parquet --out-dir output_v10sce
# France
python src/stack_ce.py --val work10_sim/train/val_predv10ra.parquet --test work10/test/test_predv10ra.parquet \
  --ce fr=$C/fr_val.parquet,$C/fr_test.parquet fr2=$C/fr2_val.parquet,$C/fr2_test.parquet --out-tag v10rafr12 --min-p $MINP $COLL
python src/decide.py --mode test --work work10 --models models_v8ra_fr --pred work10/test/test_predv10rafr12.parquet --out-dir output_v10rafr12
PYTHONIOENCODING=utf-8 python src/swap_filter.py --work work10/test --pred work10/test/test_predv10rafr12.parquet \
  --in-dir output_v10rafr12 --out-dir submissions/v10rafr12_swap | grep dropping | cut -c1-100
cp output_v10rafr12/candidate_pairs.tsv submissions/v10rafr12_swap/
python src/hybrid.py --seen output_v10sce --unseen submissions/v10rafr12_swap --out submissions/$OUT
python utils/validate_submission.py -m submissions/$OUT/matching_results.tsv -c submissions/$OUT/candidate_pairs.tsv -t dataset/test | tail -1
python -c "
import pandas as pd
c = pd.read_csv('submissions/$OUT/candidate_pairs.tsv', sep='\t', dtype=str).fillna('')
n = c.candidate_entity_ids.map(lambda s: len([x for x in s.split(',') if x]))
m = pd.read_csv('submissions/$OUT/matching_results.tsv', sep='\t', dtype=str).fillna('')
k = m.matched_entity_ids.map(lambda s: len([x for x in s.split(',') if x]))
print(f'candidate_pairs.tsv: {n.sum()} pairs, {n.mean():.3f} per S1; matches {k.sum()} ({k.mean():.3f} per S1), empty {(k == 0).mean():.4f}')
"
echo "FINAL READY: submissions/$OUT"
