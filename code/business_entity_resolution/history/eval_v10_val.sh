#!/usr/bin/env bash
# v10 validation before the test side exists: India/US transformer stack on stage-3 and the France
# stack (role model + both French transformers), both behind the candidate cutoff MINP.
# Needs colab_v10/out_new10v_<name>/ce_val.parquet for every model.
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
MINP=${MINP:-0.02}
P=colab_pkg; N=colab_v10; C=$P/comb10
declare -A RAW=([mini]="$P/out/ce_val.parquet $P/out_new_mini/ce_val.parquet"
                [e5b]="$P/out_e5b/ce_val.parquet $P/out_new_e5b/ce_val.parquet"
                [e5l]="$P/out_e5l/ce_val.parquet $P/out_new_e5l/ce_val.parquet"
                [e5l2]="$P/out_e5l2/ce_val.parquet $P/out_new_e5l2/ce_val.parquet"
                [labse]="$P/out_labse/ce_val.parquet" [fr]="$P/out_fr/ce_val.parquet" [fr2]="$P/out_fr2/ce_val.parquet")
for m in mini e5b e5l e5l2 labse; do
  python src/comb_ce.py --name $m --out $C --val ${RAW[$m]} $N/out_new10v_$m/ce_val.parquet --val-band work10_sim/train/val_predv10lx.parquet
done
for m in fr fr2; do
  python src/comb_ce.py --name $m --out $C --val ${RAW[$m]} $N/out_new10v_$m/ce_val.parquet --val-band work10_sim/train/val_predv10ra.parquet
done
CES=""; for m in mini e5b e5l e5l2 labse; do CES="$CES $m=$C/${m}_val.parquet,unused"; done
python src/stack_ce.py --val work10_sim/train/val_predv10s.parquet --ce $CES --out-tag v10sce --min-p $MINP | grep -E "candidate|logloss"
for dup in 1 2; do
  echo -n "IN/US v10sce decoy x$dup, soft thr .75: "
  python src/decide.py --mode val --work work10_sim --frac 0.3 --drop-frac 0.19 --models models_tmp --decoy-dup $dup \
    --val-pred work10_sim/train/val_predv10sce.parquet --cfg '{"excl":"soft","method":"thr","t":0.75}' | grep "^best" | cut -c1-12
  echo -n "IN/US v10sce decoy x$dup, grid: "
  python src/decide.py --mode val --work work10_sim --frac 0.3 --drop-frac 0.19 --models models_tmp --decoy-dup $dup \
    --val-pred work10_sim/train/val_predv10sce.parquet | grep "^best"
done
python src/stack_ce.py --val work10_sim/train/val_predv10ra.parquet --ce fr=$C/fr_val.parquet,unused fr2=$C/fr2_val.parquet,unused \
  --out-tag v10rafr12 --min-p $MINP | grep -E "candidate|logloss"
for dup in 1 2; do
  echo -n "France-safe v10rafr12 (IN/US val) decoy x$dup, hard ef t2: "
  python src/decide.py --mode val --work work10_sim --frac 0.3 --drop-frac 0.19 --models models_tmp --decoy-dup $dup \
    --val-pred work10_sim/train/val_predv10rafr12.parquet --cfg '{"excl":"hard","method":"ef","t":2.0,"m":0.0}' | grep "^best" | cut -c1-12
done
