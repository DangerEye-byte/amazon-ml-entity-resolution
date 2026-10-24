"""Feature building, two stages.

--stage block : cheap blocking-level features for every retrieved candidate
                -> <work>/<split>/bfeats/part-*.parquet   (input of prune.py)
--stage full  : full string features for candidates that survived pruning
                (<work>/<split>/pruned/part-*.parquet)
                -> <work>/<split>/feats/part-*.parquet    (input of train.py / predict.py)

Train split: only a deterministic fraction of S1 rows (--frac); rows carry
`label` and `is_val` (20% hash split of S1 ids). Test: everything.

Usage:
    python src/make_features.py --stage block --split train --frac 0.12
    python src/make_features.py --stage full  --split train
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ensure_hashseed, is_val, read_ground_truth  # noqa: E402
from features import (BLOCK_GROUP_COLS, FULL_GROUP_COLS, Split, add_group_features,  # noqa: E402
                      add_reverse_features, anchor_features, block_features, string_features)


def chunk_starts(q, size):
    bounds = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    starts = [0]
    for b in bounds:
        if b - starts[-1] >= size:
            starts.append(b)
    starts.append(len(q))
    return starts


def label_pairs(S, gt, s1_ids, q, src, d):
    did = S.docs.column("id").take(pa.array(S.doc_index(src, d))).to_pylist()
    sid = s1_ids[q]
    cache = {}
    lab = np.zeros(len(sid), np.int8)
    for i, (si, x) in enumerate(zip(sid, did)):
        t = cache.get(si)
        if t is None:
            t = cache[si] = set(gt.get(si, "").split(","))
        lab[i] = x in t
    return lab


def stage_block(a, base, out_dir, t0):
    cands = pq.read_table(os.path.join(base, a.cands)).to_pandas()
    rrank, rcnt, rbest = add_reverse_features(cands)
    cands["r_rank"], cands["r_cnt"], cands["r_best_other"] = rrank, rcnt, rbest
    del rrank, rcnt, rbest
    s1_tab = pq.read_table(os.path.join(base, "s1.parquet"), columns=["id", "ck"])
    s1_ids = np.array(s1_tab.column("id").to_pylist(), dtype=object)
    s1_ck = np.array(s1_tab.column("ck").to_pylist(), dtype=object)
    del s1_tab
    uq = np.unique(cands["q"].values)
    val_flag = np.zeros(len(s1_ids), bool)
    val_flag[uq] = is_val(s1_ids[uq])
    if a.frac < 1.0:
        keep_q = uq[is_val(s1_ids[uq], frac=a.frac, seed=11)]
        cands = cands[np.isin(cands["q"].values, keep_q)]
    cands = cands.sort_values(["q", "src", "rank"], kind="stable").reset_index(drop=True)
    print(f"pairs: {len(cands)} for {cands['q'].nunique()} S1 rows", flush=True)
    gt = read_ground_truth(os.path.join(a.data_dir, "train", "train_ground_truth.tsv")) \
        if a.split == "train" else None
    q_ck = s1_ck[cands["q"].values]
    total, done = len(cands), 0
    for ck in sorted(set(q_ck)):
        cc = cands[q_ck == ck].reset_index(drop=True)
        starts = chunk_starts(cc["q"].values, a.chunk)
        outs = [os.path.join(out_dir, f"part-{ck}-{i:05d}.parquet") for i in range(len(starts) - 1)]
        if a.resume and all(os.path.exists(o) for o in outs):
            done += len(cc)
            continue
        S = Split(base, ck)
        print(f"[{ck}] tables loaded ({time.time() - t0:.0f}s), {len(cc)} pairs", flush=True)
        for pi, out in enumerate(outs):
            s, e = starts[pi], starts[pi + 1]
            done += e - s
            if a.resume and os.path.exists(out):
                continue
            c = cc.iloc[s:e]
            qq, src, d = c["q"].values, c["src"].values, c["d"].values
            df = block_features(S, qq, src, d, c["score"].values, c["rank"].values)
            df["r_rank"] = c["r_rank"].values
            df["r_cnt"] = c["r_cnt"].values
            df["r_best_other"] = c["r_best_other"].values
            df["r_score_ratio"] = c["score"].values / np.maximum(c["r_best_other"].values, 1e-3)
            if "n_sib" in c:
                df["n_sib"] = c["n_sib"].values
                df["exp_score"] = c["exp_score"].values
            if "rev_rank" in c:
                df["rev_rank"] = c["rev_rank"].values.astype(np.float32)
                df["rev_score"] = c["rev_score"].values
            if "nb" in c:
                df["nb"] = c["nb"].values.astype(np.float32)
            if "rr_rank" in c:
                df["rr_rank"] = c["rr_rank"].values.astype(np.float32)
                df["rr_score"] = c["rr_score"].values.astype(np.float32)
            if "dn_rrank" in c:
                df["dn_rrank"] = c["dn_rrank"].values.astype(np.float32)
                df["dn_frank"] = c["dn_frank"].values.astype(np.float32)
                df["dn_score"] = c["dn_score"].values.astype(np.float32)
            df = add_group_features(df, qq, BLOCK_GROUP_COLS + (["dn_score"] if "dn_score" in df else []))
            df["q"], df["src"], df["d"] = qq, src, d
            df["is_val"] = val_flag[qq]
            if gt is not None:
                df["label"] = label_pairs(S, gt, s1_ids, qq, src, d)
            df.to_parquet(out + ".tmp", index=False)
            os.replace(out + ".tmp", out)
            print(f"[{ck}] part {pi + 1}/{len(outs)}: {done}/{total} pairs, {time.time() - t0:.0f}s", flush=True)
        del S


def stage_full(a, base, out_dir, t0):
    in_parts = sorted(glob.glob(os.path.join(base, "pruned", "part-*.parquet")))
    by_ck = {}
    for p in in_parts:
        ck = os.path.basename(p).split("-")[1]
        by_ck.setdefault(ck, []).append(p)
    for ck, parts in sorted(by_ck.items()):
        outs = [os.path.join(out_dir, os.path.basename(p)) for p in parts]
        if a.resume and all(os.path.exists(o) for o in outs):
            continue
        S = Split(base, ck, tokens=False)
        print(f"[{ck}] tables loaded ({time.time() - t0:.0f}s), {len(parts)} parts", flush=True)
        for i, (p, out) in enumerate(zip(parts, outs)):
            if a.resume and os.path.exists(out):
                continue
            b = pq.read_table(p).to_pandas()
            if len(b) == 0:
                b.to_parquet(out, index=False)
                continue
            qq, src, d = b["q"].values, b["src"].values, b["d"].values
            sf = string_features(S, qq, src, d)
            for c in sf.columns:
                b[c] = sf[c].values
            af = anchor_features(S, qq, src, d, b["p1"].values)
            for c in af.columns:
                b[c] = af[c].values
            b = add_group_features(b, qq, FULL_GROUP_COLS, prefix="s_")
            b.to_parquet(out + ".tmp", index=False)
            os.replace(out + ".tmp", out)
            if i % 10 == 0:
                print(f"[{ck}] part {i + 1}/{len(parts)}, {time.time() - t0:.0f}s", flush=True)
        del S


def main():
    ensure_hashseed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["block", "full"])
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--work", default="work")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=300000, help="pairs per part (approx)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cands", default="cands.parquet", help="candidates file inside <work>/<split>/")
    a = ap.parse_args()
    base = os.path.join(a.work, a.split)
    out_dir = os.path.join(base, "bfeats" if a.stage == "block" else "feats")
    os.makedirs(out_dir, exist_ok=True)
    if not a.resume:
        for f in glob.glob(os.path.join(out_dir, "part-*.parquet")):
            os.remove(f)
    t0 = time.time()
    (stage_block if a.stage == "block" else stage_full)(a, base, out_dir, t0)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
