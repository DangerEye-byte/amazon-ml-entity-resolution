"""Export the GPU cross-encoder package for Google Colab.

Files (all parquet, zstd):
  rec_<split>_s<k>.parquet   row, text        raw "name | address" of every record a pair uses
  pairs_train.parquet        q, src, d, label training rows (never validation rows): all hard
                                              pairs (0.01 < p1 < 0.99) plus a random sample of the rest
  pairs_val.parquet          q, src, d, label every validation candidate pair (stacking input)
  pairs_test.parquet         q, src, d        every test candidate pair
q is a row of <split>/s1.parquet, d a row of <split>/s<src>.parquet (same order as the TSVs).

Usage:
    python src/colab_export.py --out colab_pkg
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write(df, path):
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="zstd")


def records(split, work, k, rows, out):
    ids = pq.read_table(os.path.join(work, split, f"s{k}.parquet"), columns=["id"]).column("id").to_numpy()
    raw = pd.read_csv(os.path.join("dataset", split, f"{split}_source{k}.tsv"), sep="\t", dtype=str,
                      usecols=["entity_id", "business_name", "business_address"]).fillna("")
    raw = raw.set_index("entity_id")
    rows = np.unique(rows)
    sel = raw.loc[ids[rows]]
    text = np.where(sel["business_address"].values != "",
                    sel["business_name"].values + " | " + sel["business_address"].values,
                    sel["business_name"].values)
    write(pd.DataFrame({"row": rows.astype(np.int32), "text": text}), os.path.join(out, f"rec_{split}_s{k}.parquet"))
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="colab_pkg")
    ap.add_argument("--work-sim", default="work_sim")
    ap.add_argument("--work", default="work")
    ap.add_argument("--n-easy", type=int, default=250000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    parts = sorted(glob.glob(os.path.join(a.work_sim, "train", "feats", "part-*.parquet")))
    f = pd.concat([pq.read_table(p, columns=["q", "src", "d", "p1", "label", "is_val"]).to_pandas() for p in parts],
                  ignore_index=True)
    tr = f[~f["is_val"].values]
    hard = (tr["p1"] > 0.01) & (tr["p1"] < 0.99)
    easy = tr[~hard].sample(min(a.n_easy, int((~hard).sum())), random_state=0)
    ptr = pd.concat([tr[hard], easy])[["q", "src", "d", "label"]].sample(frac=1.0, random_state=1)
    ptr["label"] = ptr["label"].astype(np.int8)
    pva = f[f["is_val"].values][["q", "src", "d", "label"]].copy()
    pva["label"] = pva["label"].astype(np.int8)
    del f
    write(ptr.reset_index(drop=True), os.path.join(a.out, "pairs_train.parquet"))
    write(pva.reset_index(drop=True), os.path.join(a.out, "pairs_val.parquet"))
    print(f"train pairs {len(ptr)} (pos {ptr['label'].mean():.3f}), val pairs {len(pva)}", flush=True)
    both = pd.concat([ptr, pva])
    n = records("train", a.work_sim, 1, both["q"].values, a.out)
    for k in (2, 3):
        n += records("train", a.work_sim, k, both["d"].values[both["src"].values == k], a.out)
    print(f"train records {n}", flush=True)
    pte = pq.read_table(os.path.join(a.work, "test", "test_pred.parquet"), columns=["q", "src", "d"]).to_pandas()
    write(pte, os.path.join(a.out, "pairs_test.parquet"))
    n = records("test", a.work, 1, pte["q"].values, a.out)
    for k in (2, 3):
        n += records("test", a.work, k, pte["d"].values[pte["src"].values == k], a.out)
    print(f"test pairs {len(pte)}, test records {n}", flush=True)


if __name__ == "__main__":
    main()
