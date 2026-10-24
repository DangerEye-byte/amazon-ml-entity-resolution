"""Stage 5: turn pair probabilities into match sets, optimised for macro F0.5.

1. Exclusivity: every S2/S3 record belongs to at most one S1 record, so a
   record's probability mass is shared among the S1 rows competing for it
   ("hard": keep only the best S1; "soft": renormalise when mass > 1).
2. Per-S1 selection: either a global threshold, or the subset (top-k by
   probability, including the empty set) maximising expected F0.5.

Validation mode grid-searches these choices on held-out S1 rows using the
official metric (singletons score 1.0 for an empty prediction) and saves the
best config to models/decision.json.
Test mode applies that config and writes the two submission files.

Usage:
    python src/decide.py --mode val --frac 0.25
    python src/decide.py --mode test
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import f05, is_val, read_ground_truth  # noqa: E402


def exclusivity(df, mode):
    if mode == "none":
        return df["p"].values
    key = df["src"].values.astype(np.int64) * (1 << 32) + df["d"].values.astype(np.int64)
    g = pd.Series(df["p"].values).groupby(key)
    if mode == "hard":
        mx = g.transform("max").values
        return np.where(df["p"].values >= mx, df["p"].values, df["p"].values * 0.0)
    if mode == "soft":
        sm = g.transform("sum").values
        return df["p"].values / np.maximum(1.0, sm)
    if mode == "hardsoft":
        mx = g.transform("max").values
        sm = g.transform("sum").values
        p = df["p"].values / np.maximum(1.0, sm)
        return np.where(df["p"].values >= mx, p, p * 0.3)
    if mode == "odds":  # posterior that this S1 owns the record, given at most one S1 does (independent pair odds)
        pp = np.clip(df["p"].values, 1e-6, 1 - 1e-6)
        o = pp / (1 - pp)
        so = pd.Series(o).groupby(key).transform("sum").values
        return o / (1 + so)
    raise ValueError(mode)


def select(q, p, method, t, m=0.0, floor=0.0):
    """Return boolean mask of selected rows. q must be sorted ascending (groups contiguous)."""
    if method == "thr":
        return p >= t
    order = np.lexsort((-p, q))
    qs, ps = q[order], p[order]
    first = np.r_[True, qs[1:] != qs[:-1]]
    gid = np.cumsum(first) - 1
    pos = np.arange(len(qs))
    start = np.maximum.accumulate(np.where(first, pos, 0))
    k = pos - start + 1
    cs = np.cumsum(ps)
    base = np.r_[0, cs][start]
    csg = cs - base
    tot = np.bincount(gid, weights=ps)
    et = tot[gid] + m
    ef = 1.25 * csg / (k + 0.25 * et)
    # P(no true match) ~ prod(1-p); temper with t (calibration knob for the empty set)
    logq = np.bincount(gid, weights=np.log(np.clip(1 - ps, 1e-9, 1)))
    p_empty = np.exp(logq) * t
    best_ef = np.full(gid.max() + 1 if len(gid) else 0, -1.0)
    np.maximum.at(best_ef, gid, ef)
    # best k per group = first position reaching max ef
    is_best = ef >= best_ef[gid] - 1e-12
    bestk = np.full(len(best_ef), 0)
    idx = np.flatnonzero(is_best)
    # first occurrence per group
    fg = gid[idx]
    firsts = np.r_[True, fg[1:] != fg[:-1]]
    bestk[fg[firsts]] = k[idx[firsts]]
    choose_empty = p_empty > best_ef
    sel_sorted = (k <= bestk[gid]) & ~choose_empty[gid] & (ps >= floor)
    mask = np.zeros(len(q), bool)
    mask[order] = sel_sorted
    return mask


def evaluate(df, sel, truth, s1_ids_all):
    """macro F0.5 over all S1 ids in `truth` (dict id->set); df has column sid."""
    pred = {}
    for sid, did in zip(df["sid"].values[sel], df["did"].values[sel]):
        pred.setdefault(sid, set()).add(did)
    tot = 0.0
    for sid in s1_ids_all:
        tot += f05(pred.get(sid, set()), truth[sid])
    return tot / len(s1_ids_all)


def attach_ids(df, split_dir):
    s1 = pq.read_table(os.path.join(split_dir, "s1.parquet"), columns=["id"]).column("id")
    s2 = pq.read_table(os.path.join(split_dir, "s2.parquet"), columns=["id"]).column("id")
    s3 = pq.read_table(os.path.join(split_dir, "s3.parquet"), columns=["id"]).column("id")
    df["sid"] = np.array(s1.take(df["q"].values).to_pylist(), dtype=object)
    did = np.empty(len(df), dtype=object)
    m2 = df["src"].values == 2
    did[m2] = s2.take(df["d"].values[m2]).to_pylist()
    did[~m2] = s3.take(df["d"].values[~m2]).to_pylist()
    df["did"] = did
    return s1


GRID = []
for ex in ("none", "soft", "hard", "hardsoft"):
    for t in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.96):
        GRID.append({"excl": ex, "method": "thr", "t": t})
    for t in (1.0, 2.0, 3.0, 5.0, 8.0):
        for m in (0.0, 0.3):
            GRID.append({"excl": ex, "method": "ef", "t": t, "m": m})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["val", "test"])
    ap.add_argument("--work", default="work")
    ap.add_argument("--models", default="models")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--frac", type=float, default=1.0, help="same --frac as make_features (train)")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--drop-frac", type=float, default=0.0, help="S1 fraction removed by simulate_drop.py")
    ap.add_argument("--decoy-dup", type=int, default=1, help="val mode: count each decoy candidate this many times")
    ap.add_argument("--val-pred", default=None, help="val mode: probabilities file (default val_pred.parquet)")
    ap.add_argument("--pred", default=None, help="test mode: probabilities file (default <work>/test/test_pred.parquet)")
    ap.add_argument("--cfg", default=None, help='val mode: evaluate only this JSON config, e.g. {"excl":"soft","method":"thr","t":0.75}')
    a = ap.parse_args()
    t0 = time.time()
    if a.mode == "val":
        split_dir = os.path.join(a.work, "train")
        df = pq.read_table(a.val_pred or os.path.join(split_dir, "val_pred.parquet")).to_pandas()
        s1 = attach_ids(df, split_dir)
        all_ids = np.array(s1.to_pylist(), dtype=object)
        m = is_val(all_ids)
        if a.frac < 1.0:
            m &= is_val(all_ids, frac=a.frac, seed=11)
        if a.drop_frac > 0:  # simulated test conditions: these S1 rows do not exist
            from simulate_drop import dropped_mask
            m &= ~dropped_mask(all_ids, a.drop_frac)
        val_ids = all_ids[m]
        gt = read_ground_truth(os.path.join(a.data_dir, "train", "train_ground_truth.tsv"))
        truth = {s: set(x for x in gt[s].split(",") if x) for s in val_ids}
        if a.decoy_dup > 1:
            # test-like validation: test holds ~2x as many decoy records (records owned by no S1);
            # add copies of every decoy candidate so each counts decoy_dup times
            owned = set(x for v in gt.values() for x in v.split(",") if x)
            dec = df[(df["label"].values == 0) & ~df["did"].isin(owned).values]
            extra = []
            for i in range(1, a.decoy_dup):
                c = dec.copy()
                c["d"] = c["d"].values + i * 20_000_000
                c["did"] = c["did"].values + f"#dup{i}"
                extra.append(c)
            df = pd.concat([df] + extra, ignore_index=True)
            print(f"decoy duplication x{a.decoy_dup}: {len(dec)} decoy candidates copied", flush=True)
        n_true = sum(len(v) for v in truth.values())
        print(f"val S1 rows {len(val_ids)}, with candidates {df['sid'].nunique()}, "
              f"blocking recall {df['label'].sum() / max(1, n_true):.4f}", flush=True)
        # upper bound: perfect classifier on retrieved candidates
        ub = evaluate(df, df["label"].values == 1, truth, val_ids)
        print(f"upper bound (oracle on candidates): {ub:.4f}")
        df = df.sort_values("q", kind="stable").reset_index(drop=True)
        q = df["q"].values
        res = []
        grid = [json.loads(a.cfg)] if a.cfg else GRID
        pex = {ex: exclusivity(df, ex) for ex in sorted(set(c["excl"] for c in grid))}
        for cfg in grid:
            p = pex[cfg["excl"]]
            sel = select(q, p, cfg["method"], cfg["t"], cfg.get("m", 0.0))
            sc = evaluate(df, sel, truth, val_ids)
            res.append((sc, cfg))
        res.sort(key=lambda x: -x[0])
        for sc, cfg in res[:10]:
            print(f"{sc:.5f} {cfg}")
        best = res[0][1]
        os.makedirs(a.models, exist_ok=True)
        with open(os.path.join(a.models, "decision.json"), "w") as f:
            json.dump({"cfg": best, "val_f05": res[0][0], "upper_bound": ub}, f, indent=2)
        print(f"best {res[0][0]:.5f} {best} ({time.time() - t0:.0f}s)")
    else:
        split_dir = os.path.join(a.work, "test")
        with open(os.path.join(a.models, "decision.json")) as f:
            cfg = json.load(f)["cfg"]
        df = pq.read_table(a.pred or os.path.join(split_dir, "test_pred.parquet")).to_pandas()
        s1 = attach_ids(df, split_dir)
        df = df.sort_values("q", kind="stable").reset_index(drop=True)
        p = exclusivity(df, cfg["excl"])
        sel = select(df["q"].values, p, cfg["method"], cfg["t"], cfg.get("m", 0.0))
        write_outputs(df, sel, s1.to_pylist(), a.out_dir)
        print(f"selected {sel.sum()} matches for {df.loc[sel, 'sid'].nunique()} S1 rows "
              f"({time.time() - t0:.0f}s)")


def write_outputs(df, sel, all_s1, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cand = df.groupby("sid", sort=False)["did"].agg(lambda x: ",".join(dict.fromkeys(x)))
    match = df[sel].groupby("sid", sort=False)["did"].agg(lambda x: ",".join(dict.fromkeys(x)))
    cand, match = cand.to_dict(), match.to_dict()
    with open(os.path.join(out_dir, "candidate_pairs.tsv"), "w", encoding="utf8", newline="\n") as fc, \
            open(os.path.join(out_dir, "matching_results.tsv"), "w", encoding="utf8", newline="\n") as fm:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1:
            fc.write(f"{sid}\t{cand.get(sid, '')}\n")
            fm.write(f"{sid}\t{match.get(sid, '')}\n")
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
