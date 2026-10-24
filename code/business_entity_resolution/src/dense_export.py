"""Export the dense-retrieval package for Google Colab (src/colab_dense.py).

The bi-encoder is fine-tuned only on training S1 rows OUTSIDE the downstream
sample (make_features --frac 0.3 keeps is_val(id, frac=0.3, seed=11)), so the
retrieved candidates and dense scores of every row the pipeline trains or
validates on are as unbiased as on test.

Files (parquet, zstd), q/d are row positions in work/<split>/s<k>.parquet:
  txt_<split>_s<k>.parquet  text, ck       raw "name | address" and country code (0 in, 1 us, 2 fr)
  s1meta_train.parquet      drop, sample   19% simulated S1 deletion; downstream 30% sample
  pos_train.parquet         q, src, d, nsim  matches of S1 rows outside the sample (fine-tuning data);
                                              nsim = token_set_ratio of the normalized names (hard-positive sampling)
  pos_val.parquet           q, src, d, ret matches of validation S1 rows; ret = already in cands_v9

Usage:
    python src/dense_export.py --out dense_pkg
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rapidfuzz import fuzz  # noqa: E402
from rapidfuzz.process import cpdist  # noqa: E402

from common import is_val, read_ground_truth  # noqa: E402
from simulate_drop import dropped_mask  # noqa: E402

CK = {"in": 0, "us": 1, "fr": 2}


def write(df, path):
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="zstd")


def texts(split, k, work, out):
    path = os.path.join(out, f"txt_{split}_s{k}.parquet")
    if os.path.exists(path):
        return pq.ParquetFile(path).metadata.num_rows
    ids = pq.read_table(os.path.join(work, split, f"s{k}.parquet"), columns=["id"]).column("id").to_numpy(
        zero_copy_only=False)
    ck = pq.read_table(os.path.join(work, split, f"s{k}.parquet"), columns=["ck"]).column("ck").to_pandas()
    raw = pd.read_csv(os.path.join("dataset", split, f"{split}_source{k}.tsv"), sep="\t", dtype=str,
                      usecols=["entity_id", "business_name", "business_address"]).fillna("")
    raw = raw.set_index("entity_id").loc[ids]
    name, addr = raw["business_name"].values, raw["business_address"].values
    text = np.where(addr != "", name + " | " + addr, name)
    write(pd.DataFrame({"text": text, "ck": ck.map(CK).astype(np.int8).values}), path)
    return len(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dense_pkg")
    ap.add_argument("--work", default="work")
    ap.add_argument("--cands", default="work9/train/cands_v9.parquet")
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--drop-frac", type=float, default=0.19)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    for split in ("train", "test"):
        for k in (1, 2, 3):
            n = texts(split, k, a.work, a.out)
            print(f"txt_{split}_s{k}: {n} records ({time.time() - t0:.0f}s)", flush=True)
    ids = np.array(pq.read_table(os.path.join(a.work, "train", "s1.parquet"), columns=["id"]).column("id")
                   .to_pylist(), dtype=object)
    drop = dropped_mask(ids, a.drop_frac)
    sample = is_val(ids, frac=a.frac, seed=11)
    val = sample & is_val(ids) & ~drop
    write(pd.DataFrame({"drop": drop, "sample": sample}), os.path.join(a.out, "s1meta_train.parquet"))
    gt = read_ground_truth(os.path.join("dataset", "train", "train_ground_truth.tsv"))
    s = pd.Series([gt.get(i, "") for i in ids])
    s = s[s != ""].str.split(",").explode()
    s = s[s != ""]
    src = np.where(s.str[1].values == "2", 2, 3).astype(np.int8)
    d = np.full(len(s), -1, np.int64)
    for k in (2, 3):
        m = src == k
        rid = pq.read_table(os.path.join(a.work, "train", f"s{k}.parquet"), columns=["id"]).column("id").to_numpy(
            zero_copy_only=False)
        d[m] = pd.Index(rid).get_indexer(s.values[m])
    assert (d >= 0).all()
    P = pd.DataFrame({"q": s.index.values.astype(np.int32), "src": src, "d": d.astype(np.int32)})
    tr = P[~sample[P["q"].values]].reset_index(drop=True)
    core = {k: pq.read_table(os.path.join(a.work, "train", f"s{k}.parquet"), columns=["core"]).column("core")
            .to_numpy(zero_copy_only=False) for k in (1, 2, 3)}
    cd = np.empty(len(tr), dtype=object)
    for k in (2, 3):
        m = tr["src"].values == k
        cd[m] = core[k][tr["d"].values[m]]
    tr["nsim"] = cpdist(core[1][tr["q"].values].tolist(), cd.tolist(), scorer=fuzz.token_set_ratio,
                        workers=-1, dtype=np.uint8)
    del core, cd
    print(f"train positives: name similarity < 60 for {(tr['nsim'] < 60).mean():.3f}", flush=True)
    write(tr, os.path.join(a.out, "pos_train.parquet"))
    pv = P[val[P["q"].values]].reset_index(drop=True)
    # which validation matches the token retrieval (v9) already has
    vq = np.unique(pv["q"].values)
    c = ds.dataset(a.cands).to_table(columns=["q", "src", "d"], filter=pc.field("q").isin(pa.array(vq)))
    key = lambda q, s_, d_: np.asarray(q, np.int64) * 8 * (1 << 32) + np.asarray(s_, np.int64) * (1 << 32) + np.asarray(d_, np.int64)  # noqa
    kc = key(c.column("q").to_numpy(), c.column("src").to_numpy(), c.column("d").to_numpy())
    pv["ret"] = np.isin(key(pv["q"].values, pv["src"].values, pv["d"].values), kc)
    write(pv, os.path.join(a.out, "pos_val.parquet"))
    print(f"train positives {len(tr)} ({(~sample).sum()} S1 rows outside the sample); val positives {len(pv)}, "
          f"not retrieved by v9 {(~pv['ret']).sum()} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
