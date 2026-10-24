"""Fast greedy search over the final stack on validation.

build: caches one wide validation matrix (work10_sim/train/stackx_val.parquet) with every signal for every
candidate pair past the candidate cutoff (stage-3 p > 0.02): stage-2 / stage-3 / role-model probabilities,
transformer logits, collective / density / twin features and all pairwise feature columns.
Experiments (python -c "import greedy_stack as g; g.run(...)") train the stack by 2-fold cross-fit on
validation S1 rows and report out-of-fold macro F0.5 (decoy x1 and x2) under a decision rule.
"""
import glob
import hashlib
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import f05, is_val, read_ground_truth  # noqa: E402
from decide import attach_ids, exclusivity, select  # noqa: E402
from simulate_drop import dropped_mask  # noqa: E402
from stack_ce import collective, logit  # noqa: E402

W = "work10_sim/train"
C = "colab_pkg/comb10"
CES = ["mini", "e5b", "e5l", "e5l2", "labse", "e5lv10", "fr", "fr2"]
MATRIX = os.path.join(W, "stackx_val.parquet")
META = {"q", "src", "d", "label", "is_val", "sid", "did"}


def build(minp=0.02, s3="v10s", s2="v10lx", role="v10ra", out=None):
    t0 = time.time()
    df = pd.read_parquet(os.path.join(W, f"val_pred{s3}.parquet")).rename(columns={"p": "p3"})
    df = df[df["p3"].values > minp].reset_index(drop=True)
    for tag, col in ((s2, "p2"), (role, "pra")):
        x = pd.read_parquet(os.path.join(W, f"val_pred{tag}.parquet"), columns=["q", "src", "d", "p"]).rename(columns={"p": col})
        df = df.merge(x, on=["q", "src", "d"], how="left")
    for c in ("p2", "p3", "pra"):
        df["l" + c] = logit(df[c].values)
    for m in CES:
        c = pd.read_parquet(os.path.join(C, f"{m}_val.parquet")).drop_duplicates(["q", "src", "d"]).rename(columns={"ce": "ce"})
        df = df.merge(c.rename(columns={"ce": f"ce_{m}"}), on=["q", "src", "d"], how="left")
        df[f"l_{m}"] = np.where(df[f"ce_{m}"].notna(), logit(df[f"ce_{m}"].fillna(0.5).values), np.nan)
        df = df.drop(columns=f"ce_{m}")
    print(f"base + CE ({time.time() - t0:.0f}s)", flush=True)
    df, _ = collective(df, W, 0.19)
    print(f"collective ({time.time() - t0:.0f}s)", flush=True)
    have = set(df.columns)
    parts = sorted(glob.glob(os.path.join(W, "feats", "part-*.parquet")))
    names = [n for n in pq.ParquetFile(parts[0]).schema_arrow.names if n not in META and n not in have]
    f = pd.concat([pq.read_table(p, columns=["q", "src", "d", "is_val"] + names).to_pandas() for p in parts], ignore_index=True)
    f = f[f["is_val"].values].drop(columns="is_val").drop_duplicates(["q", "src", "d"])
    df = df.merge(f, on=["q", "src", "d"], how="left")
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    out = out or MATRIX
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {df.shape} ({time.time() - t0:.0f}s)", flush=True)


class Evaluator:
    """Out-of-fold macro F0.5 on validation S1 rows, optionally with every decoy candidate counted twice."""

    def __init__(self, df):
        df = df.copy()
        s1 = attach_ids(df, W)
        ids = np.array(s1.to_pylist(), dtype=object)
        m = is_val(ids) & is_val(ids, frac=0.3, seed=11) & ~dropped_mask(ids, 0.19)
        self.val_ids = ids[m]
        gt = read_ground_truth(os.path.join("dataset", "train", "train_ground_truth.tsv"))
        self.truth = {s: set(x for x in gt[s].split(",") if x) for s in self.val_ids}
        owned = set(x for v in gt.values() for x in v.split(",") if x)
        self.dec = np.flatnonzero((df["label"].values == 0) & ~df["did"].isin(owned).values)
        self.df = df[["q", "src", "d", "sid", "did"]]

    def score(self, p, rule, dup=1):
        d = self.df.assign(p=p)
        if dup > 1:
            extra = []
            for i in range(1, dup):
                c = d.iloc[self.dec].copy()
                c["d"] = c["d"].values + i * 20_000_000
                c["did"] = c["did"].values + f"#dup{i}"
                extra.append(c)
            d = pd.concat([d] + extra, ignore_index=True)
        d = d.sort_values("q", kind="stable").reset_index(drop=True)
        sel = select(d["q"].values, exclusivity(d, rule["excl"]), rule["method"], rule["t"], rule.get("m", 0.0))
        pred = d[sel].groupby("sid")["did"].apply(set).to_dict()
        return float(np.mean([f05(pred.get(s, set()), self.truth[s]) for s in self.val_ids]))


RULES = {"thr75": {"excl": "soft", "method": "thr", "t": 0.75}, "ef1": {"excl": "hardsoft", "method": "ef", "t": 1.0, "m": 0.0}}
_CACHE = {}


def load(path=None):
    if path and _CACHE.get("path") != path:
        _CACHE.clear()
    if "df" not in _CACHE:
        _CACHE["path"] = path
        df = pd.read_parquet(path or MATRIX)
        _CACHE["df"] = df
        _CACHE["ev"] = Evaluator(df)
        _CACHE["fold"] = np.array([hashlib.blake2b(str(int(q)).encode(), digest_size=2).digest()[0] & 1
                                   for q in df["q"].values])
    return _CACHE["df"], _CACHE["ev"], _CACHE["fold"]


def oof(feats, params=None, rounds=300, seed=0):
    df, ev, fold = load()
    P = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 200, "verbosity": -1,
         "num_threads": 4, "seed": seed}
    P.update(params or {})
    X = df[feats].values.astype(np.float32)
    y = df["label"].values
    p = np.zeros(len(df))
    for f in (0, 1):
        m = lgb.train(P, lgb.Dataset(X[fold != f], y[fold != f]), rounds)
        p[fold == f] = m.predict(X[fold == f])
    return p


def run(name, feats, params=None, rounds=300, rules=("thr75",), dups=(1, 2), seeds=(0,)):
    t0 = time.time()
    df, ev, _ = load()
    p = np.mean([oof(feats, params, rounds, s) for s in seeds], axis=0)
    y = df["label"].values
    ll = float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))))
    res = {f"{r}x{d}": ev.score(p, RULES[r], d) for r in rules for d in dups}
    print(f"{name:40s} ll {ll:.5f} " + " ".join(f"{k} {v:.5f}" for k, v in res.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    return res, p


def comp_feats(df, p, prefix="i_"):
    """Neighbourhood features from a probability vector: competition for the record across S1 rows and the
    S1's other candidates (iterative collective classification)."""
    out = pd.DataFrame(index=df.index)
    rk = df["src"].values.astype(np.int64) * (1 << 32) + df["d"].values
    s = pd.Series(p)
    g = s.groupby(rk)
    mx = g.transform("max").values
    o = pd.DataFrame({"rk": rk, "p": p}).sort_values(["rk", "p"], ascending=[True, False])
    sec = o.groupby("rk")["p"].nth(1)
    sec_map = pd.Series(sec.values, index=o.loc[sec.index, "rk"].values) if len(sec) else pd.Series(dtype=float)
    second = pd.Series(rk).map(sec_map).fillna(0.0).values
    out[prefix + "r_n"] = g.transform("size").values
    out[prefix + "r_other"] = np.where(p >= mx - 1e-12, second, mx)
    out[prefix + "r_sum_other"] = g.transform("sum").values - p
    q = df["q"].values
    gq = s.groupby(q)
    out[prefix + "g_sum_other"] = gq.transform("sum").values - p
    c9 = (p > 0.9).astype(np.float32)
    out[prefix + "g_n90_other"] = pd.Series(c9).groupby(q).transform("sum").values - c9
    out[prefix + "g_rank"] = gq.rank(ascending=False, method="first").values
    out[prefix + "lp"] = logit(np.clip(p, 1e-6, 1 - 1e-6))
    return out


def run2(name, feats, params=None, rounds=300, rules=("thr75",), dups=(1, 2)):
    """Two-pass stack: neighbourhood features recomputed from the first pass's out-of-fold probabilities."""
    t0 = time.time()
    df, ev, fold = load()
    p1 = oof(feats, params, rounds)
    cf = comp_feats(df, p1)
    for c in cf.columns:
        df[c] = cf[c].values
    p = oof(feats + list(cf.columns), params, rounds)
    y = df["label"].values
    ll = float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))))
    res = {f"{r}x{d}": ev.score(p, RULES[r], d) for r in rules for d in dups}
    print(f"{name:40s} ll {ll:.5f} " + " ".join(f"{k} {v:.5f}" for k, v in res.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    return res, p


def columns():
    df, _, _ = load()
    return [c for c in df.columns if c not in META and c not in ("p2", "p3", "pra")]


if __name__ == "__main__":
    if sys.argv[1] == "build":
        kw = dict(a.split("=", 1) for a in sys.argv[2:])
        build(**kw)


def seg_thresholds(p, seg, grid=(0.6, 0.65, 0.7, 0.75, 0.8, 0.85), dups=(1, 2), excl="soft"):
    """Greedy per-segment thresholds (coordinate ascent on mean F0.5 over decoy x1/x2)."""
    df, ev, _ = load()
    segs = np.unique(seg)
    th = {s: 0.75 for s in segs}

    def score(th):
        pp = p.copy()
        # a per-row threshold t is applied as a 0.75 threshold on a monotone rescaling of p
        tt = np.array([th[s] for s in seg])
        pp = np.where(p >= tt, 0.75 + 0.25 * (p - tt) / np.maximum(1 - tt, 1e-9), 0.75 * p / np.maximum(tt, 1e-9))
        return np.mean([ev.score(pp, {"excl": excl, "method": "thr", "t": 0.75}, d) for d in dups])

    best = score(th)
    for _ in range(2):
        for s in segs:
            for t in grid:
                cand = dict(th)
                cand[s] = t
                sc = score(cand)
                if sc > best + 1e-6:
                    best, th = sc, cand
    print(f"segment thresholds {th}: mean F0.5 {best:.5f}", flush=True)
    return th, best


def build_split(feats, split="test", base="stage3", s3="v10sc", s2="v10lxc", role="v10ra", minp=0.02, tw="colab_pkg/out_tw"):
    """The stack's input matrix for one split, restricted to `feats`.
    base = "stage3": candidate set = stage-3 p > minp (India/US); base = "role": role-model p > minp (France)."""
    t0 = time.time()
    sd = "work10/test" if split == "test" else W
    pref = "test_pred" if split == "test" else "val_pred"
    main_tag = s3 if base == "stage3" else role
    df = pd.read_parquet(os.path.join(sd, f"{pref}{main_tag}.parquet"),
                         columns=["q", "src", "d", "p"] + (["label"] if split == "val" else []))
    df = df[df["p"].values > minp].reset_index(drop=True)
    df["p3" if base == "stage3" else "pra"] = df.pop("p")
    others = [(s2, "p2"), (role, "pra")] if base == "stage3" else [(s2, "p2")]
    for tag, col in others:
        if ("l" + col) in feats:
            x = pd.read_parquet(os.path.join(sd, f"{pref}{tag}.parquet"), columns=["q", "src", "d", "p"]).rename(columns={"p": col})
            df = df.merge(x, on=["q", "src", "d"], how="left")
    for c in ("p2", "p3", "pra"):
        if c in df:
            df["l" + c] = logit(df[c].values)
    for f in [f for f in feats if f.startswith("l_")]:
        m = f[2:]
        if m in ("tw", "qwen"):
            path = os.path.join("colab_pkg", f"out_{m}", "ce_val.parquet" if split == "val" else "ce_testband.parquet")
        else:
            path = os.path.join(C, f"{m}_{'val' if split == 'val' else 'test'}.parquet")
        c = pd.read_parquet(path).drop_duplicates(["q", "src", "d"]).rename(columns={"ce": "_ce"})
        df = df.merge(c, on=["q", "src", "d"], how="left")
        df[f] = np.where(df["_ce"].notna(), logit(df["_ce"].fillna(0.5).values), np.nan)
        df = df.drop(columns="_ce")
    coll = pd.read_parquet(os.path.join(sd, "coll.parquet"))
    cc = [f for f in feats if f in coll.columns]
    df = df.merge(coll[["q", "src", "d"] + cc], on=["q", "src", "d"], how="left")
    parts = sorted(glob.glob(os.path.join(sd, "feats", "part-*.parquet")))
    names = set(pq.ParquetFile(parts[0]).schema_arrow.names)
    pc_ = [f for f in feats if f in names and f not in df.columns]
    if pc_:
        f_ = pd.concat([pq.read_table(p, columns=["q", "src", "d"] + pc_).to_pandas() for p in parts], ignore_index=True)
        df = df.merge(f_.drop_duplicates(["q", "src", "d"]), on=["q", "src", "d"], how="left")
    missing = [f for f in feats if f not in df.columns]
    assert not missing, missing
    print(f"{split} matrix {df.shape} ({time.time() - t0:.0f}s)", flush=True)
    return df


NB_COLS = ["lp2", "l_e5l", "l_e5l2", "l_e5lv10", "l_tw", "l_mini", "l_e5b", "l_labse", "c_raw_self"]


def neighbour_feats(df, cols=NB_COLS):
    """Leak-free message passing over the candidate graph: for every pair, the same evidence columns of its
    competitors - the best other S1 row for the same record, and the best other record for the same S1 row -
    plus this pair's margin and rank against them. Uses features only, never labels or predictions."""
    out = {}
    rk = df["src"].values.astype(np.int64) * (1 << 32) + df["d"].values
    q = df["q"].values
    for c in cols:
        if c not in df:
            continue
        v = df[c].values.astype(np.float64)
        filled = np.where(np.isnan(v), -1e9, v)
        for key, tag in ((rk, "r"), (q, "s")):
            s = pd.Series(filled)
            mx = s.groupby(key).transform("max").values
            o = pd.DataFrame({"k": key, "v": filled}).sort_values(["k", "v"], ascending=[True, False])
            sec = o.groupby("k")["v"].nth(1)
            second = pd.Series(key).map(pd.Series(sec.values, index=o.loc[sec.index, "k"].values)).values
            other = np.where(filled >= mx - 1e-12, second, mx)
            other = np.where((other <= -1e8) | pd.isna(other), np.nan, other)
            out[f"nb_{tag}_{c}"] = other
            out[f"nb_{tag}_{c}_margin"] = np.where(np.isnan(v), np.nan, v - other)
    return pd.DataFrame(out, index=df.index)


def orphan_feats(df, split_dir, drop_frac):
    """Footprints of missing entities: per name, the records that carry a real address matching no present S1 row
    of that name (same house number + street word). Such records imply a deleted (or never listed) entity with the
    same name, so an empty-address record with that name may be its orphan."""
    t = {k: pq.read_table(os.path.join(split_dir, f"s{k}.parquet"), columns=["ck", "core", "anorm", "nums"]).to_pandas()
         for k in (1, 2, 3)}
    present = np.ones(len(t[1]), bool)
    if drop_frac > 0:
        ids = pq.read_table(os.path.join(split_dir, "s1.parquet"), columns=["id"]).column("id").to_numpy(zero_copy_only=False)
        present = ~dropped_mask(ids.astype(object), drop_frac)

    def akey(x):
        n1 = x["nums"].fillna("").str.split().str[0].fillna("")
        w1 = x["anorm"].fillna("").str.extract(r"([a-z]{4,})", expand=False).fillna("")
        return x["ck"] + "|" + x["core"] + "|" + n1 + "|" + w1, (n1 != "") & (w1 != "")

    k1, ok1 = akey(t[1])
    s1keys = set(k1[present & ok1.values].values)
    rec = pd.concat([t[2], t[3]], ignore_index=True)
    kr, okr = akey(rec)
    unexpl = okr.values & ~kr.isin(s1keys).values
    ncore = (rec["ck"] + "|" + rec["core"])
    cnt = pd.Series(unexpl.astype(np.int32)).groupby(ncore.values).sum()
    tot = pd.Series(okr.values.astype(np.int32)).groupby(ncore.values).sum()
    q = df["q"].values
    qn = t[1]["ck"].values[q] + "|" + t[1]["core"].values[q]
    dn = np.empty(len(df), dtype=object)
    for k in (2, 3):
        m = df["src"].values == k
        dn[m] = t[k]["ck"].values[df["d"].values[m]] + "|" + t[k]["core"].values[df["d"].values[m]]
    out = pd.DataFrame(index=df.index)
    out["o_unexpl_q"] = pd.Series(qn).map(cnt).fillna(0).values
    out["o_unexpl_d"] = pd.Series(dn).map(cnt).fillna(0).values
    out["o_addr_q"] = pd.Series(qn).map(tot).fillna(0).values
    out["o_frac_q"] = out["o_unexpl_q"].values / np.maximum(out["o_addr_q"].values, 1)
    return out
