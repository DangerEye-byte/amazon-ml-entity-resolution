"""Stage 3: group-aware re-scoring of stage-2 probabilities.

Stage 2 scores each (S1, record) pair mostly on its own. Many remaining
errors are visible only across the S1 row's candidate list or across the S1
rows competing for a record: a decoy business at a neighbouring house number
forms its own consistent cluster next to the true one; an empty-address
record is confirmed by the name of the confident cluster; a record is claimed
more strongly by another S1 row. Stage 3 is a small LightGBM trained on
held-out validation predictions (never on rows stage 2 was fit on) with
features describing those relations.

Val mode: 2-fold cross-fit on validation S1 rows, writes val_pred3 (OOF) for
decide.py. Test mode: fit on all validation rows, re-score test.

Usage:
    python src/stage3.py --mode val  --work work_sim --models models_v7
    python src/stage3.py --mode test --work work_sim --test-work work --models models_v7
"""
import argparse
import glob
import hashlib
import os

import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW = ["num1_eq", "num1_lev", "num1_logdiff", "num_q_empty", "num_d_empty", "d_aempty", "nm_ratio", "nm_tset",
       "ad_tset", "legal_ratio", "legal_d_empty", "d_oov_frac", "d_is_domain", "is_s3", "p1", "r_cnt",
       "r_score_ratio", "nmc_tset"]


def first_num(n):
    n = str(n).split()
    return n[0] if n else ""


def group_features(df, s):
    """df: q, src, d, p (+RAW). s: dict of per-source tables with n1 (first number), core, compact."""
    m2 = df["src"].values == 2
    for col in ("n1", "core"):
        v = np.empty(len(df), dtype=object)
        v[m2] = s[2][col].values[df["d"].values[m2]]
        v[~m2] = s[3][col].values[df["d"].values[~m2]]
        df["d_" + col] = v
    df["q_n1"] = s[1]["n1"].values[df["q"].values]
    p = df["p"].values
    g = df.groupby("q")["p"]
    df["g_max"] = g.transform("max")
    df["g_sum"] = g.transform("sum")
    df["g_n"] = g.transform("size")
    df["g_n90"] = (df["p"] > 0.9).groupby(df["q"]).transform("sum")
    df["g_rank"] = g.rank(ascending=False, method="first")
    df["g_gap"] = df["g_max"] - p
    df["g_sum_other"] = df["g_sum"] - p
    # house-number agreement with the confident part of the cluster
    has = df["d_n1"] != ""
    conf = (df["p"] > 0.9) & has
    key = df["q"].astype(str) + "|" + df["d_n1"].astype(str)
    conf_cnt = conf.groupby(key).transform("sum")
    df["n_conf_same_num"] = np.where(has, conf_cnt - (conf.astype(int)), -1)
    all_cnt = has.groupby(key).transform("sum")
    df["n_same_num"] = np.where(has, all_cnt - 1, -1)
    minp = df["p"].where(has).groupby(key).transform("min")
    df["same_num_minp"] = np.where(has, minp, -1)
    conf_at_q = ((df["d_n1"] == df["q_n1"]) & conf).groupby(df["q"]).transform("sum")
    df["n_conf_at_q_num"] = conf_at_q
    df["d_num_is_q"] = np.where(has & (df["q_n1"] != ""), (df["d_n1"] == df["q_n1"]).astype(int), -1)
    # name agreement with the best other candidate
    order = df.sort_values(["q", "p"], ascending=[True, False])
    top = order.groupby("q").head(1).set_index("q")
    df["top_core_eq"] = (df["d_core"].values == top["d_core"].reindex(df["q"].values).values).astype(int)
    df["top_num_eq"] = (df["d_n1"].values == top["d_n1"].reindex(df["q"].values).values).astype(int)
    # competition for the record across S1 rows
    rk = df["src"].astype(np.int64) * (1 << 32) + df["d"].astype(np.int64)
    rg = df["p"].groupby(rk)
    df["r_pmax"] = rg.transform("max")
    df["r_psum_other"] = rg.transform("sum") - p
    df["r_is_best"] = (p >= df["r_pmax"].values - 1e-12).astype(int)
    df["r_n"] = rg.transform("size")
    return df


FEATS = ["p", "g_max", "g_sum", "g_n", "g_n90", "g_rank", "g_gap", "g_sum_other", "n_conf_same_num", "n_same_num",
         "same_num_minp", "n_conf_at_q_num", "d_num_is_q", "top_core_eq", "top_num_eq", "r_pmax", "r_psum_other",
         "r_is_best", "r_n"] + RAW


def load_split(work, split, pred_path, val):
    parts = sorted(glob.glob(os.path.join(work, split, "feats", "part-*.parquet")))
    cols = ["q", "src", "d"] + RAW + (["is_val"] if val else [])
    f = pd.concat([pq.read_table(p, columns=cols).to_pandas() for p in parts], ignore_index=True)
    if val:
        f = f[f["is_val"].values].drop(columns="is_val")
    pr = pq.read_table(pred_path).to_pandas()
    df = pr.merge(f, on=["q", "src", "d"], how="left")
    s = {}
    for k in (1, 2, 3):
        t = pq.read_table(os.path.join(work, split, f"s{k}.parquet"), columns=["nums", "core"]).to_pandas()
        t["n1"] = t["nums"].map(first_num)
        s[k] = t
    return group_features(df, s)


def fold_of(q):
    return np.array([hashlib.blake2b(str(x).encode(), digest_size=2).digest()[0] & 1 for x in q])


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lexical  # noqa: E402

LX = ["lx_n_extra", "lx_n_missing", "lx_n_typo", "lx_n_realvar", "lx_n_shared", "lx_extra_min", "lx_extra_mean",
      "lx_missing_min", "lx_missing_mean", "lx_extra_unseen", "lx_missing_unseen", "lx_legal", "lx_numtype",
      "lx_aextra_min", "lx_amissing_min", "lx_n_aextra", "lx_n_amissing", "lx_extra_max", "lx_nq", "lx_nd",
      "lx_min_var", "lx_legnum", "lx_first", "lx_first_state", "g_num_lx_extra_min", "g_num_lx_legal_min",
      "g_num_lx_first_min"]


def add_lexical(df, split_dir, encoder, t0):
    s, voc = lexical.load_split(split_dir)
    diffs = lexical.pair_diffs(df, s, voc)
    del s
    f = lexical.features(diffs, encoder)
    for c in f.columns:
        df[c] = f[c].values
    # decoy evidence shared by the records at the same house number (a decoy business forms its own cluster)
    has = df["d_n1"] != ""
    key = np.where(has, df["q"].astype(str) + "|" + df["d_n1"].astype(str), "")
    for c, out in (("lx_extra_min", "g_num_lx_extra_min"), ("lx_legal", "g_num_lx_legal_min"),
                   ("lx_first", "g_num_lx_first_min")):
        m = df[c].groupby(key).transform("min")
        df[out] = np.where(has, m, df[c])
    print(f"lexical features for {len(df)} pairs ({time.time() - t0:.0f}s)", flush=True)
    return df


def fit_lexical(work, t0):
    parts = sorted(glob.glob(os.path.join(work, "train", "feats", "part-*.parquet")))
    tr = pd.concat([pq.read_table(p, columns=["q", "src", "d", "label", "is_val"]).to_pandas() for p in parts],
                   ignore_index=True)
    tr = tr[~tr["is_val"].values].reset_index(drop=True)
    s, voc = lexical.load_split(os.path.join(work, "train"))
    diffs = lexical.pair_diffs(tr, s, voc)
    enc = lexical.fit_encoder(diffs, tr["label"].values.astype(float))
    print(f"lexical encoder from {len(tr)} training pairs: {len(enc['extra'])} extra words, "
          f"{len(enc['missing'])} missing words ({time.time() - t0:.0f}s)", flush=True)
    return enc


PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "verbosity": -1, "num_threads": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["val", "test"])
    ap.add_argument("--work", default="work_sim")
    ap.add_argument("--test-work", default="work")
    ap.add_argument("--models", default="models_v7")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--lexical", action="store_true")
    ap.add_argument("--tag", default="3")
    ap.add_argument("--ce", nargs="*", default=[],
                    help="transformer scores, each NAME=VAL_PARQUET,TEST_PARQUET[,VAL_MASK_PARQUET] "
                         "(files with q, src, d, ce; the optional mask lists the validation pairs that were "
                         "eligible for scoring under the same rule as test)")
    ap.add_argument("--in-tag", default="", help="read val_pred<in-tag> / test_pred<in-tag> as stage-2 input")
    a = ap.parse_args()
    t0 = time.time()
    ces = []
    for spec in a.ce:
        name, paths = spec.split("=", 1)
        paths = paths.split(",")
        ces.append((name, paths[0], paths[1], paths[2] if len(paths) > 2 else None))
    feats = FEATS + (LX if a.lexical else []) + [c for n, *_ in ces for c in (f"ce_{n}", f"ce_{n}_scored")]

    def add_ce(df, split):
        for name, vpath, tpath, mpath in ces:
            c = pd.read_parquet(vpath if split == "val" else tpath, columns=["q", "src", "d", "ce"])
            c = c.rename(columns={"ce": f"ce_{name}"}).drop_duplicates(["q", "src", "d"])
            if split == "val" and mpath:
                m = pd.read_parquet(mpath, columns=["q", "src", "d"]).assign(_ok=1)
                c = c.merge(m, on=["q", "src", "d"], how="inner").drop(columns="_ok")
            df = df.merge(c, on=["q", "src", "d"], how="left")
            df[f"ce_{name}_scored"] = df[f"ce_{name}"].notna().astype(np.float32)
            print(f"ce_{name} [{split}]: {int(df[f'ce_{name}_scored'].sum())} of {len(df)} pairs scored", flush=True)
        return df
    va = load_split(a.work, "train", os.path.join(a.work, "train", f"val_pred{a.in_tag}.parquet"), True)
    if a.lexical:
        enc = fit_lexical(a.work, t0)
        va = add_lexical(va, os.path.join(a.work, "train"), enc, t0)
    va = add_ce(va, "val")
    X, y = va[feats].values.astype(np.float32), va["label"].values
    if a.mode == "val":
        fold = fold_of(va["q"].values)
        p3 = np.zeros(len(va))
        for f in (0, 1):
            tr, te = fold != f, fold == f
            m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), num_boost_round=a.rounds)
            p3[te] = m.predict(X[te])
        out = va[["q", "src", "d", "label"]].copy()
        out["p"] = p3
        out.to_parquet(os.path.join(a.work, "train", f"val_pred{a.tag}.parquet"), index=False)
        def log_loss(t, q):
            q = np.clip(q, 1e-7, 1 - 1e-7)
            return float(-np.mean(t * np.log(q) + (1 - t) * np.log(1 - q)))
        print(f"logloss stage2 {log_loss(y, np.clip(va['p'], 1e-7, 1 - 1e-7)):.5f} -> stage3 {log_loss(y, p3):.5f}")
    else:
        m = lgb.train(PARAMS, lgb.Dataset(X, y), num_boost_round=a.rounds)
        m.save_model(os.path.join(a.models, f"stage{a.tag}.txt"))
        te = load_split(a.test_work, "test", os.path.join(a.test_work, "test", f"test_pred{a.in_tag}.parquet"), False)
        if a.lexical:
            te = add_lexical(te, os.path.join(a.test_work, "test"), enc, t0)
        te = add_ce(te, "test")
        out = te[["q", "src", "d"]].copy()
        out["p"] = m.predict(te[feats].values.astype(np.float32))
        out.to_parquet(os.path.join(a.test_work, "test", f"test_pred{a.tag}.parquet"), index=False)
        print("wrote test_pred3; mean p", te["p"].mean(), "->", out["p"].mean())


if __name__ == "__main__":
    main()
