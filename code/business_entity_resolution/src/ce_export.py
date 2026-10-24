"""Export the cross-encoder package: hard training pairs and uncertain pairs to score.

Training pairs: labelled training rows (never validation rows) whose stage-1
probability p1 (out-of-fold) is uncertain. Scoring pairs: validation and test
rows whose stage-2 probability (--in-tag model) lies in (--lo, --hi); the same
rule is applied to both so the stacked feature means the same thing.
Texts are the raw source names/addresses (the transformer reads all scripts).

Usage:
    python src/ce_export.py --out ce_pkg --in-tag lx
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def raw(split, k):
    p = os.path.join("dataset", split, f"{split}_source{k}.tsv")
    return pd.read_csv(p, sep="\t", dtype=str, usecols=["entity_id", "business_name", "business_address"]).fillna("")


def texts_norm(split, work, pairs):
    """Normalised text (transliterated, lowercase, abbreviations expanded): 'core legal' and 'anorm'."""
    out = pairs.copy()
    t = {k: pq.read_table(os.path.join(work, split, f"s{k}.parquet"), columns=["core", "legal", "anorm"]).to_pandas()
         for k in (1, 2, 3)}
    q = pairs["q"].values
    out["a_name"] = (t[1]["core"].values[q] + " " + t[1]["legal"].values[q]).astype(object)
    out["a_addr"] = t[1]["anorm"].values[q]
    bn = np.empty(len(pairs), dtype=object)
    ba = np.empty(len(pairs), dtype=object)
    for k in (2, 3):
        m = pairs["src"].values == k
        d = pairs["d"].values[m]
        bn[m] = t[k]["core"].values[d] + " " + t[k]["legal"].values[d]
        ba[m] = t[k]["anorm"].values[d]
    out["b_name"], out["b_addr"] = bn, ba
    for c in ("a_name", "b_name"):
        out[c] = out[c].str.strip()
    return out


def texts(split, work, pairs):
    """pairs: q, src, d (row indices of work/<split>/s*.parquet) -> a_name, a_addr, b_name, b_addr."""
    out = pairs.copy()
    ids = {k: pq.read_table(os.path.join(work, split, f"s{k}.parquet"), columns=["id"]).column("id").to_numpy()
           for k in (1, 2, 3)}
    r1 = raw(split, 1).set_index("entity_id")
    a = r1.loc[ids[1][pairs["q"].values]]
    out["a_name"], out["a_addr"] = a["business_name"].values, a["business_address"].values
    del r1
    bn = np.empty(len(pairs), dtype=object)
    ba = np.empty(len(pairs), dtype=object)
    for k in (2, 3):
        m = pairs["src"].values == k
        rk = raw(split, k).set_index("entity_id")
        b = rk.loc[ids[k][pairs["d"].values[m]]]
        bn[m], ba[m] = b["business_name"].values, b["business_address"].values
        del rk
    out["b_name"], out["b_addr"] = bn, ba
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ce_pkg")
    ap.add_argument("--work-sim", default="work_sim")
    ap.add_argument("--work", default="work")
    ap.add_argument("--in-tag", default="lx")
    ap.add_argument("--n-train", type=int, default=300000)
    ap.add_argument("--lo", type=float, default=0.005)
    ap.add_argument("--hi", type=float, default=0.995)
    ap.add_argument("--normalized", action="store_true", help="export normalised text instead of raw source text")
    ap.add_argument("--skip-unseen", action="store_true", help="do not score test rows of countries absent from training")
    a = ap.parse_args()
    global texts
    if a.normalized:
        texts = texts_norm
    os.makedirs(a.out, exist_ok=True)
    parts = sorted(glob.glob(os.path.join(a.work_sim, "train", "feats", "part-*.parquet")))
    f = pd.concat([pq.read_table(p, columns=["q", "src", "d", "p1", "label", "is_val"]).to_pandas() for p in parts],
                  ignore_index=True)
    tr = f[~f["is_val"].values & (f["p1"] > 0.01) & (f["p1"] < 0.99)]
    tr = tr.sample(min(a.n_train, len(tr)), random_state=0)[["q", "src", "d", "label"]].reset_index(drop=True)
    del f
    t = texts("train", a.work_sim, tr)
    t["label"] = t["label"].astype(np.float32)
    t[["a_name", "a_addr", "b_name", "b_addr", "label"]].to_parquet(os.path.join(a.out, "ce_train.parquet"), index=False)
    print(f"train pairs {len(t)}, positive rate {t['label'].mean():.3f}", flush=True)
    rows = []
    vp = pd.read_parquet(os.path.join(a.work_sim, "train", f"val_pred{a.in_tag}.parquet"))
    vp = vp[(vp["p"] > a.lo) & (vp["p"] < a.hi)][["q", "src", "d"]].reset_index(drop=True)
    v = texts("train", a.work_sim, vp)
    v.insert(0, "split", "val")
    rows.append(v)
    tp = pd.read_parquet(os.path.join(a.work, "test", f"test_pred{a.in_tag}.parquet"))
    tp = tp[(tp["p"] > a.lo) & (tp["p"] < a.hi)][["q", "src", "d"]].reset_index(drop=True)
    if a.skip_unseen:
        seen = set(pq.read_table(os.path.join(a.work_sim, "train", "s1.parquet"), columns=["ck"]).column("ck").unique().to_pylist())
        ck = pq.read_table(os.path.join(a.work, "test", "s1.parquet"), columns=["ck"]).column("ck").to_numpy()
        tp = tp[np.isin(ck[tp["q"].values], list(seen))].reset_index(drop=True)
    te = texts("test", a.work, tp)
    te.insert(0, "split", "test")
    rows.append(te)
    sc = pd.concat(rows, ignore_index=True)
    sc.to_parquet(os.path.join(a.out, "ce_score.parquet"), index=False)
    print(f"score pairs: val {len(v)}, test {len(te)}")


if __name__ == "__main__":
    main()
