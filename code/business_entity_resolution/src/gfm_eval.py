"""A/B of the India/US decision rule on validation: the current soft-exclusivity 0.75 threshold vs the exact
expected-F0.5 maximiser (gfm.py), on out-of-fold probabilities of the v11a India/US stack.

For decoy x1 and x2 it prints macro F0.5, the paired difference to thr75 with its standard error, the same with
validation S1 rows reweighted to the test India/US composition (country x same-name group size), and the
difference by country. Variants:
  gfm_indep        GFM with independent labels, Poisson missing matches, P(singleton) = prod(1-p) * P(miss = 0)
  gfm_nonempty     GFM only sizes rows that thr75 already predicts (no rescue of empty rows)
  gfm_p0           GFM with a learned row-level P(singleton) (2-fold cross-fit, same folds as the stack)
  gfm_p0_nonempty  learned P(singleton) may empty a row, but never rescues one

--transfer also prints the gain of the collective/twin stack features (v11a's stack change), unweighted vs
test-weighted, to test whether the 62% LB transfer is a composition effect.

Usage (from code/business_entity_resolution; ~3 GB RAM, a few minutes):
    python src/gfm_eval.py [--seeds 1] [--transfer]
"""
import argparse
import hashlib
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ensure_hashseed, f05  # noqa: E402
import greedy_stack as g  # noqa: E402
from decide import exclusivity  # noqa: E402
from gfm import gfm_select  # noqa: E402
from simulate_drop import dropped_mask  # noqa: E402

KW = dict(base="stage3", s3="v10sc", s2="v10lxc", role="v10rac")  # as in final_v11.py
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 200, "verbosity": -1,
          "num_threads": 4, "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.7}
EDGES = [1, 2, 4, 9, 29]  # same-name group size buckets: 1, 2, 3-4, 5-9, 10-29, 30+


def oof(df, feats, fold, seeds, rounds=300):
    X = df[feats].values.astype(np.float32)
    y = df["label"].values
    p = np.zeros(len(df))
    for s in seeds:
        for f in (0, 1):
            m = lgb.train({**PARAMS, "seed": s}, lgb.Dataset(X[fold != f], y[fold != f]), rounds)
            p[fold == f] += m.predict(X[fold == f]) / len(seeds)
    return p


def bucket_of(ck, core):
    key = pd.Series(ck, dtype=object) + "|" + pd.Series(core, dtype=object)
    return np.digitize(key.map(key.value_counts()).values, EDGES, right=True)


def val_rows(ev):
    """One row per scored validation S1 id: country, same-name bucket, test-composition weight, true match count."""
    t1 = pq.read_table(os.path.join(g.W, "s1.parquet"), columns=["id", "ck", "core"]).to_pandas()
    t1 = t1[~dropped_mask(t1["id"].values.astype(object), 0.19)]
    t1["b"] = bucket_of(t1["ck"].values, t1["core"].values)
    te = pq.read_table("work10/test/s1.parquet", columns=["ck", "core"]).to_pandas()
    te = te[te["ck"].isin(["in", "us"])]
    te["b"] = bucket_of(te["ck"].values, te["core"].values)
    r = t1.set_index("id").reindex(ev.val_ids)[["ck", "b"]]
    tshare = te.groupby(["ck", "b"]).size() / len(te)
    vshare = r.groupby(["ck", "b"]).size() / len(r)
    r["w"] = (tshare / vshare).reindex(pd.MultiIndex.from_arrays([r["ck"].values, r["b"].values])).fillna(0.0).values
    r["w"] /= r["w"].mean()
    r["n_true"] = [len(ev.truth[s]) for s in ev.val_ids]
    return r


def pair_table(ev, p, dup):
    """Same decoy duplication and soft exclusivity as greedy_stack.Evaluator.score; column i = matrix row."""
    d = ev.df.assign(p=p, i=np.arange(len(p)))
    if dup > 1:
        extra = []
        for k in range(1, dup):
            c = d.iloc[ev.dec].copy()
            c["d"] = c["d"].values + k * 20_000_000
            c["did"] = c["did"].values + f"#dup{k}"
            extra.append(c)
        d = pd.concat([d] + extra, ignore_index=True)
    d = d.sort_values("q", kind="stable").reset_index(drop=True)
    d["pe"] = exclusivity(d, "soft")
    return d


def row_f(ev, d, sel):
    pred = d.loc[sel].groupby("sid")["did"].apply(set).to_dict()
    return np.array([f05(pred.get(s, set()), ev.truth[s]) for s in ev.val_ids])


def missing_lambda(df, ev, rows):
    """Expected number of true matches outside the candidate set, per country x same-name bucket."""
    sid = ev.df["sid"].values
    tp = pd.Series(df["label"].values).groupby(sid).sum()
    r = rows.assign(tp=tp.reindex(rows.index).values)
    r = r[r["tp"].notna()]  # rows with at least one candidate
    lam = (r["n_true"] - r["tp"]).groupby([r["ck"], r["b"]]).mean().clip(lower=0)
    out = lam.reindex(pd.MultiIndex.from_arrays([rows["ck"].values, rows["b"].values])).fillna(lam.mean()).values
    print("missing matches per row (lambda) by country x bucket:\n" + lam.round(4).to_string(), flush=True)
    return pd.Series(out, index=rows.index)


def p0_oof(d, df, fold, rows):
    """Row-level P(no true match at all), 2-fold cross-fit on the stack's folds."""
    d = d.assign(g50=(d["pe"].values >= 0.5).astype(np.int8), g75=(d["pe"].values >= 0.75).astype(np.int8))
    for c in ("lp3", "lpra", "c_daempty", "c_samecore", "c_raw_margin"):
        if c in df:
            d[c] = df[c].values[d["i"].values]
    gq = d.groupby("q", sort=True)
    R = gq.agg(sid=("sid", "first"), i=("i", "first"), pmax=("pe", "max"), psum=("pe", "sum"), n=("pe", "size"),
               n50=("g50", "sum"), n75=("g75", "sum"))
    rk = gq["pe"].rank(ascending=False, method="first").values
    R["p2nd"] = pd.Series(d["pe"].values[rk == 2], index=d["q"].values[rk == 2]).reindex(R.index).fillna(0.0).values
    for c, how in (("lp3", "max"), ("lpra", "max"), ("c_daempty", "mean"), ("c_samecore", "max"), ("c_raw_margin", "max")):
        if c in d:
            R[f"{c}_{how}"] = gq[c].agg(how).values
    for c in ("c_qname", "c_qaddr", "c_qnumst"):
        if c in df:
            R[c] = df[c].values[R["i"].values]
    feats = [c for c in R.columns if c not in ("sid", "i")]
    scored = R["sid"].isin(rows.index).values  # train only on S1 rows the evaluator scores
    y = (rows["n_true"].reindex(R["sid"].values).fillna(-1).values == 0).astype(np.float32)
    fr = fold[R["i"].values]
    X = R[feats].values.astype(np.float32)
    p0 = np.zeros(len(R))
    P = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 100, "verbosity": -1,
         "num_threads": 4}
    for f in (0, 1):
        tr = (fr != f) & scored
        m = lgb.train(P, lgb.Dataset(X[tr], y[tr]), 200)
        p0[fr == f] = m.predict(X[fr == f])
    ys, ps = y[scored], p0[scored]
    ll = -np.mean(ys * np.log(np.clip(ps, 1e-6, 1)) + (1 - ys) * np.log(np.clip(1 - ps, 1e-6, 1)))
    print(f"P(singleton) model: {int(scored.sum())} rows, base rate {ys.mean():.4f}, logloss {ll:.4f}", flush=True)
    return pd.Series(p0, index=R.index)


def report(name, dup, f, base, rows):
    w = rows["w"].values
    dl = f - base
    se = dl.std(ddof=1) / np.sqrt(len(dl))
    dw = float(np.sum(w * dl) / w.sum())
    sew = float(np.sqrt(np.sum((w * (dl - dw)) ** 2)) / w.sum())
    by = pd.Series(dl).groupby(rows["ck"].values).mean()
    print(f"x{dup} {name:16s} F {f.mean():.5f} (d {dl.mean():+.5f} +- {se:.5f})  test-weighted F "
          f"{np.sum(w * f) / w.sum():.5f} (d {dw:+.5f} +- {sew:.5f})  rows changed {int((dl != 0).sum())}  by country "
          + " ".join(f"{k} {v:+.5f}" for k, v in by.items()), flush=True)


def main():
    ensure_hashseed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--transfer", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    feats = json.load(open(os.path.join(g.W, "greedy_feats.json"))) + ["l_tw"]
    df = g.build_split(feats, split="val", **KW)
    ev = g.Evaluator(df)
    fold = np.array([hashlib.blake2b(str(int(q)).encode(), digest_size=2).digest()[0] & 1 for q in df["q"].values])
    seeds = tuple(range(a.seeds))
    p = oof(df, feats, fold, seeds)
    rows = val_rows(ev)
    rows["lam"] = missing_lambda(df, ev, rows)
    print(f"val S1 rows {len(rows)}; stack OOF ready ({time.time() - t0:.0f}s)", flush=True)
    for dup in (1, 2):
        d = pair_table(ev, p, dup)
        q, pe = d["q"].values, d["pe"].values
        rmax = d.groupby("q")["pe"].transform("max").values
        lam = d["sid"].map(rows["lam"]).fillna(rows["lam"].mean()).values
        p0 = d["q"].map(p0_oof(d, df, fold, rows)).values
        base = row_f(ev, d, pe >= 0.75)
        report("thr75", dup, base, base, rows)
        for name, sel in (("gfm_indep", gfm_select(q, pe, lam=lam)),
                          ("gfm_nonempty", gfm_select(q, pe, lam=lam, p0=np.where(rmax >= 0.75, 0.0, 1.0))),
                          ("gfm_p0", gfm_select(q, pe, lam=lam, p0=p0)),
                          ("gfm_p0_nonempty", gfm_select(q, pe, lam=lam, p0=np.where(rmax >= 0.75, p0, 1.0)))):
            report(name, dup, row_f(ev, d, sel), base, rows)
    if a.transfer:
        drop = {c for c in feats if c.startswith("c_")} | {"l_tw"}
        p_old = oof(df, [c for c in feats if c not in drop], fold, seeds)
        for dup in (1, 2):
            d_new, d_old = pair_table(ev, p, dup), pair_table(ev, p_old, dup)
            report("coll+twin gain", dup, row_f(ev, d_new, d_new["pe"].values >= 0.75),
                   row_f(ev, d_old, d_old["pe"].values >= 0.75), rows)
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
