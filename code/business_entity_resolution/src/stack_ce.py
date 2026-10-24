"""Light stacking of stage-2 probabilities with transformer scores (cross-fitted on validation).

Inputs: a stage-2 validation/test prediction file and any number of transformer
score sources NAME=VAL,TEST[,VAL_MASK]. Missing scores are NaN (LightGBM handles
them); on validation, scores outside the mask are dropped so validation matches
test (where only the uncertain band was scored). Writes the stacked validation
predictions (2-fold cross-fit, for decide.py) and, with --test, test predictions
from a model fit on all validation rows.

--min-p T makes the input probability a candidate filter: pairs with p <= T are dropped before
stacking, on validation and test alike, so the stack (the final model) scores exactly the pairs
written to candidate_pairs.tsv.

Usage:
    python src/stack_ce.py --val work8_sim/train/val_predv8lx.parquet --test work8/test/test_predv8lx.parquet \
        --ce gpu=colab_pkg/out/ce_val.parquet,colab_pkg/out/ce_testband.parquet,colab_pkg/val_band_mask.parquet \
        --out-tag v8ce
"""
import argparse
import hashlib
import os

import lightgbm as lgb
import numpy as np
import pandas as pd


def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def attach(df, ces, split):
    cols = ["lp"]
    for name, vpath, tpath, mpath in ces:
        path = vpath if split == "val" else tpath
        c = pd.read_parquet(path, columns=["q", "src", "d", "ce"]).drop_duplicates(["q", "src", "d"])
        if split == "val" and mpath:
            m = pd.read_parquet(mpath, columns=["q", "src", "d"]).assign(_ok=1)
            c = c.merge(m, on=["q", "src", "d"], how="inner").drop(columns="_ok")
        c = c.rename(columns={"ce": f"ce_{name}"})
        df = df.merge(c, on=["q", "src", "d"], how="left")
        df[f"l_{name}"] = np.where(df[f"ce_{name}"].notna(), logit(df[f"ce_{name}"].fillna(0.5)), np.nan)
        cols.append(f"l_{name}")
        print(f"{split}: {name} scores {int(df[f'ce_{name}'].notna().sum())} of {len(df)} pairs", flush=True)
    return df, cols


def collective(df, split_dir, drop_frac, ref_dir="work/test", ref_sizes=None):
    """Frequency features over the S1 rows present in this split (the 19% simulated deletion removed on
    validation): how many carry this S1's exact name / the record's exact name / this S1's exact address.
    Population matching (simulation only): counts run over a hash subsample of S1 rows and records sized like the
    reference (test) split of the same country, so a count means the same thing in training and on test; a pair's
    own S1 row and record are always counted."""
    import pyarrow.parquet as pq
    t = {k: pq.read_table(os.path.join(split_dir, f"s{k}.parquet"), columns=["ck", "core", "anorm", "aempty"]).to_pandas()
         for k in (1, 2, 3)}
    present = np.ones(len(t[1]), bool)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if drop_frac > 0:
        from simulate_drop import dropped_mask
        ids = pq.read_table(os.path.join(split_dir, "s1.parquet"), columns=["id"]).column("id").to_numpy(
            zero_copy_only=False).astype(object)
        present = ~dropped_mask(ids, drop_frac)
    samp1 = present.copy()
    sampr = {k: np.ones(len(t[k]), bool) for k in (2, 3)}
    if drop_frac > 0 and ref_dir and os.path.exists(os.path.join(ref_dir, "s1.parquet")):
        from common import is_val
        idsk = {k: pq.read_table(os.path.join(split_dir, f"s{k}.parquet"), columns=["id"]).column("id").to_numpy(
            zero_copy_only=False).astype(object) for k in (1, 2, 3)}
        ref1 = pd.Series(pq.read_table(os.path.join(ref_dir, "s1.parquet"), columns=["ck"]).column("ck")
                         .to_numpy(zero_copy_only=False)).value_counts()
        refr = pd.concat([pd.Series(pq.read_table(os.path.join(ref_dir, f"s{k}.parquet"), columns=["ck"]).column("ck")
                                    .to_numpy(zero_copy_only=False)) for k in (2, 3)]).value_counts()
        if ref_sizes:  # explicit target population per country, e.g. France's size for a France-only stack
            ref1 = pd.Series(ref_sizes["s1"])
            refr = pd.Series(ref_sizes["rec"])
        own1 = pd.Series(t[1]["ck"].values[present]).value_counts()
        ownr = pd.concat([t[2]["ck"], t[3]["ck"]]).value_counts()
        for c in own1.index:
            r = min(1.0, ref1.get(c, 0) / own1[c])
            if r < 1.0:
                m = t[1]["ck"].values == c
                samp1[m] &= is_val(idsk[1][m], frac=r, seed=99)
        for k in (2, 3):
            for c in ownr.index:
                r = min(1.0, refr.get(c, 0) / ownr[c])
                if r < 1.0:
                    m = t[k]["ck"].values == c
                    sampr[k][m] = is_val(idsk[k][m], frac=r, seed=98 + k)
    s1 = t[1][samp1]
    ncnt = s1.groupby(["ck", "core"]).size()
    acnt = s1[s1["anorm"] != ""].groupby(["ck", "anorm"]).size()
    q = df["q"].values
    ck = t[1]["ck"].values[q]
    dcore = np.empty(len(df), dtype=object)
    ae = np.zeros(len(df), np.float32)
    for k in (2, 3):
        m = df["src"].values == k
        dcore[m] = t[k]["core"].values[df["d"].values[m]]
        ae[m] = t[k]["aempty"].values[df["d"].values[m]]
    selfq = (~samp1[q]).astype(np.float64)  # the pair's own S1 row is always counted
    df["c_qname"] = ncnt.reindex(pd.MultiIndex.from_arrays([ck, t[1]["core"].values[q]])).fillna(0).values + selfq
    df["c_dname"] = (ncnt.reindex(pd.MultiIndex.from_arrays([ck, dcore])).fillna(0).values
                     + selfq * (dcore == t[1]["core"].values[q]))
    df["c_qaddr"] = (acnt.reindex(pd.MultiIndex.from_arrays([ck, t[1]["anorm"].values[q]])).fillna(0).values
                     + selfq * (t[1]["anorm"].values[q] != ""))
    df["c_daempty"] = ae
    df["c_samecore"] = (dcore == t[1]["core"].values[q]).astype(np.float32)
    # looser address key: first house number + first street word (>= 4 letters)
    n1 = t[1]["nums"].fillna("").str.split().str[0].fillna("") if "nums" in t[1] else None
    if n1 is None:
        t1n = pq.read_table(os.path.join(split_dir, "s1.parquet"), columns=["nums"]).to_pandas()
        n1 = t1n["nums"].fillna("").str.split().str[0].fillna("")
    w1 = t[1]["anorm"].str.extract(r"([a-z]{4,})", expand=False).fillna("")
    key = (t[1]["ck"] + "|" + n1 + "|" + w1).values
    kp = pd.Series(key[samp1])
    kcnt = kp[(n1.values[samp1] != "") & (w1.values[samp1] != "")].value_counts()
    kvalid = (n1.values[q] != "") & (w1.values[q] != "")
    df["c_qnumst"] = np.where(kvalid, pd.Series(key[q]).map(kcnt).fillna(0).values + selfq, 0.0)
    # record-side density: how many S2/S3 records share this record's / this S1's exact name or address.
    # Each entity owns ~3.5 records, so records-per-name well above 3.5 x S1-rows-per-name means orphans of deleted twins.
    rec = pd.concat([t[2][["ck", "core", "anorm"]][sampr[2]], t[3][["ck", "core", "anorm"]][sampr[3]]], ignore_index=True)
    rn = rec.groupby(["ck", "core"]).size()
    ra = rec[rec["anorm"] != ""].groupby(["ck", "anorm"]).size()
    danorm = np.empty(len(df), dtype=object)
    selfd = np.zeros(len(df))
    for k in (2, 3):
        m = df["src"].values == k
        danorm[m] = t[k]["anorm"].values[df["d"].values[m]]
        selfd[m] = ~sampr[k][df["d"].values[m]]  # the pair's own record is always counted
    df["c_rname_d"] = rn.reindex(pd.MultiIndex.from_arrays([ck, dcore])).fillna(0).values + selfd
    df["c_rname_q"] = rn.reindex(pd.MultiIndex.from_arrays([ck, t[1]["core"].values[q]])).fillna(0).values
    df["c_raddr_d"] = ra.reindex(pd.MultiIndex.from_arrays([ck, danorm])).fillna(0).values + selfd * (danorm != "")
    df["c_raddr_q"] = ra.reindex(pd.MultiIndex.from_arrays([ck, t[1]["anorm"].values[q]])).fillna(0).values
    df["c_ratio_name"] = df["c_rname_d"].values / np.maximum(df["c_dname"].values, 1)
    df["c_ratio_addr"] = df["c_raddr_q"].values / np.maximum(df["c_qaddr"].values, 1)
    dens = ["c_rname_d", "c_rname_q", "c_raddr_d", "c_raddr_q", "c_ratio_name", "c_ratio_addr"]
    # raw-name twins: the record's raw name (case, punctuation, legal-form spelling) against this S1's raw name and
    # against every other present S1 with the same normalized name, candidate or not
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    split = os.path.basename(os.path.normpath(split_dir))
    ids1 = pq.read_table(os.path.join(split_dir, "s1.parquet"), columns=["id"]).column("id").to_numpy(zero_copy_only=False)
    raw1 = pd.read_csv(os.path.join("dataset", split, f"{split}_source1.tsv"), sep="	", dtype=str,
                       usecols=["entity_id", "business_name"]).fillna("")
    name1 = raw1.set_index("entity_id")["business_name"].reindex(ids1).fillna("").values
    del raw1
    dname = np.empty(len(df), dtype=object)
    for k in (2, 3):
        m = df["src"].values == k
        idk = pq.read_table(os.path.join(split_dir, f"s{k}.parquet"), columns=["id"]).column("id").to_numpy(zero_copy_only=False)
        rk = pd.read_csv(os.path.join("dataset", split, f"{split}_source{k}.tsv"), sep="	", dtype=str,
                         usecols=["entity_id", "business_name"]).fillna("").set_index("entity_id")["business_name"]
        dname[m] = rk.reindex(idk[df["d"].values[m]]).fillna("").values
        del rk
    df["c_raw_self"] = cpdist(list(dname), list(name1[q]), scorer=fuzz.ratio, workers=-1, dtype=np.float32)
    df["c_raw_exact"] = (dname == name1[q]).astype(np.float32)
    pres = np.flatnonzero(samp1)
    gkey = t[1]["ck"].values[pres] + "|" + t[1]["core"].values[pres]
    gs = pd.Series(gkey).map(pd.Series(gkey).value_counts()).values
    ok = (gs >= 2) & (gs <= 30)
    grp = pd.DataFrame({"key": gkey[ok], "row": pres[ok]})
    pairs = pd.DataFrame({"i": np.arange(len(df)), "key": ck + "|" + t[1]["core"].values[q], "q": q})
    x = pairs.merge(grp, on="key")
    x = x[x["row"].values != x["q"].values]
    tw = np.full(len(df), -1.0, np.float32)
    twx = np.zeros(len(df), np.float32)
    if len(x):
        sims = cpdist(list(dname[x["i"].values]), list(name1[x["row"].values]), scorer=fuzz.ratio, workers=-1, dtype=np.float32)
        mx = pd.Series(sims).groupby(x["i"].values).max()
        tw[mx.index.values] = mx.values
        ex = pd.Series((dname[x["i"].values] == name1[x["row"].values]).astype(np.float32)).groupby(x["i"].values).max()
        twx[ex.index.values] = ex.values
    df["c_raw_twin"] = tw
    df["c_raw_twin_exact"] = twx
    df["c_raw_margin"] = np.where(tw >= 0, df["c_raw_self"].values - tw, 100.0)
    dens += ["c_raw_self", "c_raw_exact", "c_raw_twin", "c_raw_twin_exact", "c_raw_margin"]
    # pairwise evidence from the feature files, so the stack can combine it with the counts
    import glob
    cols = ["nm_tset", "ad_tset", "num1_eq", "d_oov_frac", "r_cnt", "abbr_ratio"]
    f = pd.concat([pq.read_table(p_, columns=["q", "src", "d"] + cols).to_pandas()
                   for p_ in sorted(glob.glob(os.path.join(split_dir, "feats", "part-*.parquet")))], ignore_index=True)
    df = df.merge(f.drop_duplicates(["q", "src", "d"]), on=["q", "src", "d"], how="left")
    return df, ["c_qname", "c_dname", "c_qaddr", "c_daempty", "c_samecore", "c_qnumst"] + dens + cols


PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 200, "verbosity": -1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--ce", nargs="+", required=True)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--min-p", type=float, default=None, help="candidate filter: keep pairs with input p > T")
    ap.add_argument("--collective", nargs=2, default=None, metavar=("VAL_SPLIT_DIR", "TEST_SPLIT_DIR"),
                    help="add name/address frequency features (e.g. work10_sim/train work10/test)")
    ap.add_argument("--drop-frac", type=float, default=0.19, help="simulated S1 deletion on validation")
    a = ap.parse_args()
    ces = []
    for spec in a.ce:
        name, paths = spec.split("=", 1)
        p = paths.split(",")
        ces.append((name, p[0], p[1], p[2] if len(p) > 2 else None))
    va = pd.read_parquet(a.val)
    if a.min_p is not None:
        n0 = len(va)
        va = va[va["p"].values > a.min_p].reset_index(drop=True)
        print(f"candidate filter p > {a.min_p}: validation {n0} -> {len(va)} pairs", flush=True)
    va["lp"] = logit(va["p"].values)
    va, cols = attach(va, ces, "val")
    if a.collective:
        va, ccols = collective(va, a.collective[0], a.drop_frac)
        cols = cols + ccols
    X, y = va[cols].values.astype(np.float32), va["label"].values
    fold = np.array([hashlib.blake2b(str(int(q)).encode(), digest_size=2).digest()[0] & 1 for q in va["q"].values])
    p2 = np.zeros(len(va))
    for f in (0, 1):
        m = lgb.train(PARAMS, lgb.Dataset(X[fold != f], y[fold != f]), a.rounds)
        p2[fold == f] = m.predict(X[fold == f])
    out = va[["q", "src", "d", "label"]].copy()
    out["p"] = p2
    vdir = os.path.dirname(a.val)
    out.to_parquet(os.path.join(vdir, f"val_pred{a.out_tag}.parquet"), index=False)
    ll = lambda t, q: float(-np.mean(t * np.log(np.clip(q, 1e-6, 1)) + (1 - t) * np.log(np.clip(1 - q, 1e-6, 1))))  # noqa
    print(f"val logloss {ll(y, va['p'].values):.5f} -> {ll(y, p2):.5f}")
    if a.test:
        m = lgb.train(PARAMS, lgb.Dataset(X, y), a.rounds)
        te = pd.read_parquet(a.test)
        if a.min_p is not None:
            n0 = len(te)
            te = te[te["p"].values > a.min_p].reset_index(drop=True)
            print(f"candidate filter p > {a.min_p}: test {n0} -> {len(te)} pairs", flush=True)
        te["lp"] = logit(te["p"].values)
        te, _ = attach(te, ces, "test")
        if a.collective:
            te, _ = collective(te, a.collective[1], 0.0)
        o = te[["q", "src", "d"]].copy()
        o["p"] = m.predict(te[cols].values.astype(np.float32))
        o.to_parquet(os.path.join(os.path.dirname(a.test), f"test_pred{a.out_tag}.parquet"), index=False)
        print(f"test: wrote test_pred{a.out_tag} ({len(o)} pairs)")


if __name__ == "__main__":
    main()
