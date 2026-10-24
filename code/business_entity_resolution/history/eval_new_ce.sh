#!/usr/bin/env bash
# Does a newly scored cross-encoder add to the v8 stack?  Usage: bash eval_new_ce.sh mdb colab_pkg/out_mdb
# The new model's files must cover pairs_val_all (v7 val pairs + v8 new) and pairs_test_all (v7 band + v8 new).
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
NAME=$1; DIR=$2
C=colab_pkg/comb
python - "$NAME" "$DIR" <<'EOF'
import sys, numpy as np, pandas as pd
name, d = sys.argv[1:3]
K = lambda df: df.q.values.astype(np.int64)*8*(1<<32) + df.src.values.astype(np.int64)*(1<<32) + df.d.values.astype(np.int64)
keep = np.concatenate([K(pd.read_parquet("colab_pkg/val_band_mask.parquet")), K(pd.read_parquet("colab_new/pairs_val_v8new.parquet"))])
v = pd.read_parquet(f"{d}/ce_val.parquet")
v[np.isin(K(v), keep)].reset_index(drop=True).to_parquet(f"colab_pkg/comb/{name}_val.parquet", index=False)
pd.read_parquet(f"{d}/ce_testband.parquet").to_parquet(f"colab_pkg/comb/{name}_test.parquet", index=False)
print(f"{name}: val {np.isin(K(v), keep).sum()} band pairs")
EOF
BASE="mini=$C/mini_val.parquet,$C/mini_test.parquet e5b=$C/e5b_val.parquet,$C/e5b_test.parquet e5l=$C/e5l_val.parquet,$C/e5l_test.parquet e5l2=$C/e5l2_val.parquet,$C/e5l2_test.parquet"
python src/stack_ce.py --val work8_sim/train/val_predv8s.parquet --ce $BASE $NAME=$C/${NAME}_val.parquet,$C/${NAME}_test.parquet --out-tag v8s4p$NAME | grep "val logloss"
echo "baseline (mini,e5b,e5l,e5l2): dup1 0.98789 dup2 0.98770"
for dup in 1 2; do
  echo -n "+$NAME dup$dup: "
  python src/decide.py --mode val --work work8_sim --frac 0.3 --drop-frac 0.19 --models models_tmp --decoy-dup $dup \
    --val-pred work8_sim/train/val_predv8s4p$NAME.parquet --cfg '{"excl":"soft","method":"thr","t":0.75}' | grep "^best" | cut -c1-12
done
