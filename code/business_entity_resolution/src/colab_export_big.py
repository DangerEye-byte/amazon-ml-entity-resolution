"""Bigger transformer training set: existing hard pairs + unused training rows + top retrieval
candidates of training S1 rows outside the 30% sample (labels from the ground truth; never
validation rows or S1 rows dropped by the simulation). Writes pairs_train_big.parquet and
rec_train_s<k>_big.parquet (texts of records not yet exported)."""
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colab_export import write  # noqa: E402
from common import is_val, read_ground_truth  # noqa: E402
from simulate_drop import dropped_mask  # noqa: E402

OUT = "colab_big"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(3)
old = pd.read_parquet("colab_pkg/pairs_train.parquet")
key = lambda d: d["q"].astype(np.int64) * 8 * (1 << 32) + d["src"].astype(np.int64) * (1 << 32) + d["d"].astype(np.int64)  # noqa
# (b) unused training rows of the v8 feature set
f = pd.concat([pq.read_table(p, columns=["q", "src", "d", "label", "is_val"]).to_pandas()
               for p in sorted(glob.glob("work8_sim/train/feats/part-*.parquet"))], ignore_index=True)
f = f[~f["is_val"].values]
b = f[~np.isin(key(f).values, key(old).values)][["q", "src", "d", "label"]]
b = b.sample(min(500_000, len(b)), random_state=0)
# (c) top retrieval candidates of S1 rows outside the sample
ids = np.array(pq.read_table("work/train/s1.parquet", columns=["id"]).column("id").to_pylist(), dtype=object)
ok = ~is_val(ids) & ~is_val(ids, frac=0.3, seed=11) & ~dropped_mask(ids, 0.19)
qs = rng.choice(np.flatnonzero(ok), min(280_000, int(ok.sum())), replace=False)  # min(): small smoke-test samples
c = pq.read_table("work8/train/cands_v8.parquet", columns=["q", "src", "d", "score"]).to_pandas()
c = c[np.isin(c["q"].values, qs)]
c = c.sort_values(["q", "score"], ascending=[True, False]).groupby("q").head(5)
gt = read_ground_truth("dataset/train/train_ground_truth.tsv")
idl = {k: pq.read_table(f"work/train/s{k}.parquet", columns=["id"]).column("id").to_numpy() for k in (2, 3)}
did = np.where(c["src"].values == 2, idl[2][np.minimum(c["d"].values, len(idl[2]) - 1)], idl[3][np.minimum(c["d"].values, len(idl[3]) - 1)])
c["label"] = [int(d in set(gt[ids[q]].split(","))) for q, d in zip(c["q"].values, did)]
c = c[["q", "src", "d", "label"]]
big = pd.concat([old, b, c], ignore_index=True).drop_duplicates(["q", "src", "d"]).sample(frac=1.0, random_state=5)
big["label"] = big["label"].astype(np.int8)
write(big.reset_index(drop=True), os.path.join(OUT, "pairs_train_big.parquet"))
print(f"big training set: {len(big)} pairs (existing {len(old)}, unused rows {len(b)}, outside-sample {len(c)}), "
      f"positive rate {big['label'].mean():.3f}", flush=True)
# texts for records not in the existing rec files
have = {k: set(pd.read_parquet(f"colab_pkg/rec_train_s{k}.parquet", columns=["row"])["row"].values) for k in (1, 2, 3)}
need = {1: set(big["q"].values)}
for k in (2, 3):
    need[k] = set(big["d"].values[big["src"].values == k])
for k in (1, 2, 3):
    rows = np.array(sorted(need[k] - have[k]), dtype=np.int64)
    idsk = pq.read_table(f"work/train/s{k}.parquet", columns=["id"]).column("id").to_numpy()
    raw = pd.read_csv(f"dataset/train/train_source{k}.tsv", sep="\t", dtype=str,
                      usecols=["entity_id", "business_name", "business_address"]).fillna("").set_index("entity_id")
    sel = raw.loc[idsk[rows]]
    text = np.where(sel["business_address"].values != "", sel["business_name"].values + " | " + sel["business_address"].values,
                    sel["business_name"].values)
    write(pd.DataFrame({"row": rows.astype(np.int32), "text": text}), os.path.join(OUT, f"rec_train_s{k}_big.parquet"))
    print(f"s{k}: {len(rows)} new record texts", flush=True)
