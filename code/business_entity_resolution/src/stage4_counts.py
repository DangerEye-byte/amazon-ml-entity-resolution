"""Stage 4: collective re-scoring with the generator's match-count distribution.

Each S1 row owns n matches with a fixed distribution (1: 6%, 2: 18%, 3: 26%, 4: 23%, 5: 16% ...)
and per source n2, n3 roughly independent (P(n2 = 0) 8%, 1: 38%, 2: 31% ...). When several S1
rows (often with identical names) compete for an ambiguous record, the likely owner is the one
whose count of confident matches in the record's source is still low: the log-likelihood ratio
log P(n_k = c + 1) - log P(n_k = c) is ~ +1.1 for c = 0 and ~ -1.1 for c = 4.

Features from the current probabilities p: confident / soft counts of the S1's other candidates
(all sources and the record's source), the count likelihood ratios, the same quantities for the
best competing S1 row of the record, the number of competitors with the S1's exact core name, and
the record's empty-address flag. A small LightGBM (2-fold cross-fit on validation S1 rows) re-scores
the pairs; test mode fits on all validation rows.

Usage:
    python src/stage4_counts.py --val work8_sim/train/val_predv8sce6.parquet \
        --test work8/test/test_predv8sce6.parquet --work-val work8_sim --work-test work8 --out-tag v8sce6c
"""
import argparse
import hashlib
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_ground_truth  # noqa: E402

FEATS = ["lp", "d_aempty", "c_all", "c_src", "s_all", "s_src", "lr_all", "lr_src", "r_n", "r_pmax_other",
         "r_best_c_src", "r_best_lr_src", "lr_src_diff", "r_n_same_core", "r_same_core_best"]


def count_priors(gt_path, max_n=12):
    """log P(n = k) for the per-S1 total and per-source match counts (Laplace-smoothed)."""
    gt = read_ground_truth(gt_path)
    n2 = np.array([sum(1 for x in v.split(",") if x[1:2] == "2") for v in gt.values()])
    n3 = np.array([sum(1 for x in v.split(",") if x[1:2] == "3") for v in gt.values()])
    out = {}
    for name, n in (("all", n2 + n3), ("src2", n2), ("src3", n3)):
        c = np.bincount(np.minimum(n, max_n), minlength=max_n + 2).astype(float) + 1.0
        out[name] = np.log(c / c.sum())
    return out


def features(df, s1_core, aempty, pri, max_n=12):
    p = df["p"].values
    conf = (p > 0.9).astype(np.int32)
    q = df["q"].values
    g_c = pd.Series(conf).groupby(q).transform("sum").values
    g_s = pd.Series(p).groupby(q).transform("sum").values
    qs = q.astype(np.int64) * 4 + df["src"].values
    gs_c = pd.Series(conf).groupby(qs).transform("sum").values
    gs_s = pd.Series(p).groupby(qs).transform("sum").values
    df["c_all"] = g_c - conf
    df["c_src"] = gs_c - conf
    df["s_all"] = g_s - p
    df["s_src"] = gs_s - p
    ca = np.minimum(df["c_all"].values, max_n - 1)
    cs = np.minimum(df["c_src"].values, max_n - 1)
    df["lr_all"] = pri["all"][ca + 1] - pri["all"][ca]
    lr2 = pri["src2"][cs + 1] - pri["src2"][cs]
    lr3 = pri["src3"][cs + 1] - pri["src3"][cs]
    df["lr_src"] = np.where(df["src"].values == 2, lr2, lr3)
    df["lp"] = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    m2 = df["src"].values == 2
    ae = np.empty(len(df), np.int8)
    ae[m2] = aempty[2][df["d"].values[m2]]
    ae[~m2] = aempty[3][df["d"].values[~m2]]
    df["d_aempty"] = ae
    # competition for the record: the best other S1 row and its counts
    rk = df["src"].values.astype(np.int64) * (1 << 32) + df["d"].values
    df["_rk"] = rk
    df["r_n"] = pd.Series(p).groupby(rk).transform("size").values
    o = df.sort_values(["_rk", "p"], ascending=[True, False])
    first = o.drop_duplicates("_rk")
    second = o[o.duplicated("_rk")].drop_duplicates("_rk")
    b1 = first.set_index("_rk")
    b2 = second.set_index("_rk")
    is_best = (df["q"].values == b1["q"].reindex(rk).values)
    other = lambda col, fill: np.where(is_best, b2[col].reindex(rk).fillna(fill).values,  # noqa: E731
                                       b1[col].reindex(rk).fillna(fill).values)
    df["r_pmax_other"] = other("p", 0.0)
    df["r_best_c_src"] = other("c_src", -1)
    df["r_best_lr_src"] = other("lr_src", 0.0)
    df["lr_src_diff"] = df["lr_src"].values - df["r_best_lr_src"].values
    core = s1_core[q]
    df["_core"] = core
    same = df.groupby(["_rk", "_core"])["p"].transform("size").values - 1
    df["r_n_same_core"] = same
    best_core = s1_core[b1["q"].reindex(rk).values.astype(np.int64)]
    df["r_same_core_best"] = ((~is_best) & (best_core == core)).astype(np.int8)
    return df.drop(columns=["_rk", "_core"])


def fold_of(q):
    return np.array([hashlib.blake2b(str(x).encode(), digest_size=2).digest()[0] & 1 for x in q])


def fit(X, y):
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200,
              "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "verbose": -1, "num_threads": 4, "seed": 0}
    return lgb.train(params, lgb.Dataset(X, y), num_boost_round=400)


def load(pred, work, pri):
    df = pd.read_parquet(pred)
    s1_core = pq.read_table(os.path.join(work, "s1.parquet"), columns=["core"]).column("core").to_numpy(zero_copy_only=False)
    aempty = {k: pq.read_table(os.path.join(work, f"s{k}.parquet"), columns=["aempty"]).column("aempty").to_numpy()
              for k in (2, 3)}
    return features(df, s1_core, aempty, pri)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--work-val", default="work8_sim")
    ap.add_argument("--work-test", default="work8")
    ap.add_argument("--out-tag", required=True)
    a = ap.parse_args()
    t0 = time.time()
    pri = count_priors(os.path.join("dataset", "train", "train_ground_truth.tsv"))
    v = load(a.val, os.path.join(a.work_val, "train"), pri)
    y = v["label"].values
    fold = fold_of(v["q"].values)
    X = v[FEATS].values.astype(np.float32)
    oof = np.zeros(len(v))
    for f in (0, 1):
        m = fit(X[fold != f], y[fold != f])
        oof[fold == f] = m.predict(X[fold == f])
    ll = lambda t, pp: -np.mean(t * np.log(np.clip(pp, 1e-7, 1)) + (1 - t) * np.log(np.clip(1 - pp, 1e-7, 1)))  # noqa
    print(f"val logloss {ll(y, v['p'].values):.5f} -> {ll(y, oof):.5f} ({time.time() - t0:.0f}s)", flush=True)
    out = v[["q", "src", "d", "label"]].copy()
    out["p"] = oof
    out.to_parquet(os.path.join(os.path.dirname(a.val), f"val_pred{a.out_tag}.parquet"), index=False)
    if a.test:
        model = fit(X, y)
        imp = sorted(zip(FEATS, model.feature_importance("gain")), key=lambda x: -x[1])[:10]
        print("top features:", [(n, int(g)) for n, g in imp], flush=True)
        t = load(a.test, os.path.join(a.work_test, "test"), pri)
        res = t[["q", "src", "d"]].copy()
        res["p"] = model.predict(t[FEATS].values.astype(np.float32))
        res.to_parquet(os.path.join(os.path.dirname(a.test), f"test_pred{a.out_tag}.parquet"), index=False)
        print(f"test: wrote test_pred{a.out_tag} ({len(res)} pairs, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
