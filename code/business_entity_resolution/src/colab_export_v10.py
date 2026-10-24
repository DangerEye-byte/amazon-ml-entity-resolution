"""Cross-encoder package on v10's candidate distribution (dense-retrieved hard pairs included).

Training pairs: every non-validation pair of work10_sim with 0.01 < p1 < 0.99 (the blocking filter
is uncertain) plus a random sample of the rest, capped at --n. Validation file: every validation pair
in v10's uncertain band (the stack needs this model's score on all of them). Record texts are written
only for rows not in the text files already uploaded (--have).

Usage:
    python src/colab_export_v10.py --out colab_v10t
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colab_export import write  # noqa: E402


def texts(split, work, k, rows):
    ids = pq.read_table(os.path.join(work, split, f"s{k}.parquet"), columns=["id"]).column("id").to_numpy(
        zero_copy_only=False)
    raw = pd.read_csv(os.path.join("dataset", split, f"{split}_source{k}.tsv"), sep="\t", dtype=str,
                      usecols=["entity_id", "business_name", "business_address"]).fillna("").set_index("entity_id")
    sel = raw.loc[ids[rows]]
    t = np.where(sel["business_address"].values != "", sel["business_name"].values + " | " + sel["business_address"].values,
                 sel["business_name"].values)
    return pd.DataFrame({"row": rows.astype(np.int32), "text": t})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="colab_v10t")
    ap.add_argument("--work-sim", default="work10_sim")
    ap.add_argument("--val-pred", default="work10_sim/train/val_predv10lx.parquet")
    ap.add_argument("--n", type=int, default=1_400_000)
    ap.add_argument("--n-easy", type=int, default=250_000)
    ap.add_argument("--have", nargs="*", default=["colab_pkg/rec_train_s{k}.parquet", "colab_big/rec_train_s{k}_big.parquet",
                                                  "colab_new/rec_train_s{k}_v8.parquet", "colab_v10/rec_train_s{k}_v10v.parquet"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    parts = sorted(glob.glob(os.path.join(a.work_sim, "train", "feats", "part-*.parquet")))
    f = pd.concat([pq.read_table(p, columns=["q", "src", "d", "p1", "label", "is_val"]).to_pandas() for p in parts],
                  ignore_index=True)
    tr = f[~f["is_val"].values]
    hard = tr[(tr["p1"] > 0.01) & (tr["p1"] < 0.99)]
    if len(hard) > a.n - a.n_easy:
        hard = hard.sample(a.n - a.n_easy, random_state=0)
    easy = tr[~((tr["p1"] > 0.01) & (tr["p1"] < 0.99))].sample(a.n_easy, random_state=1)
    ptr = pd.concat([hard, easy])[["q", "src", "d", "label"]].sample(frac=1.0, random_state=2).reset_index(drop=True)
    ptr["label"] = ptr["label"].astype(np.int8)
    write(ptr, os.path.join(a.out, "pairs_train_v10.parquet"))
    vp = pd.read_parquet(a.val_pred)
    vb = vp[(vp["p"] > 0.005) & (vp["p"] < 0.995)][["q", "src", "d", "label"]].reset_index(drop=True)
    vb["label"] = vb["label"].astype(np.int8)
    write(vb, os.path.join(a.out, "pairs_val_v10band.parquet"))
    print(f"training pairs {len(ptr)} (hard {len(hard)}, positives {ptr['label'].mean():.3f}); validation band {len(vb)}",
          flush=True)
    both = pd.concat([ptr, vb])
    for k in (1, 2, 3):
        need = np.unique(both["q"].values if k == 1 else both["d"].values[both["src"].values == k])
        have = set()
        for h in a.have:
            p = h.format(k=k)
            if os.path.exists(p):
                have |= set(pd.read_parquet(p, columns=["row"])["row"].values.tolist())
        rows = np.array(sorted(set(need.tolist()) - have), dtype=np.int64)
        write(texts("train", a.work_sim, k, rows), os.path.join(a.out, f"rec_train_s{k}_v10t.parquet"))
        print(f"s{k}: {len(need)} records needed, {len(rows)} new texts", flush=True)


if __name__ == "__main__":
    main()
