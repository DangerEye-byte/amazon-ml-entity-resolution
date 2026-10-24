"""Stage 2b: sibling query expansion (recall booster for blocking).

The S2/S3 records of one business resemble each other more than some of them
resemble the S1 record (garbled name, partial address). After the stage-1
filter, candidates with p1 >= --sib-thr are confident "siblings"; each sibling
is used as a query against the S2 and S3 indexes of its country, and the top-K
hits become extra candidates for the sibling's S1 row.

Work is split into (country, source) tasks that run in parallel and are
checkpointed to <work>/<split>/exp/<ck>_S<src>.parquet, so an interrupted run
resumes where it stopped.

Writes <work>/<split>/cands_all.parquet = original candidates + new ones, with
two extra columns: n_sib (how many sibling queries retrieved the record) and
exp_score (best sibling-query score). New pairs get their S1-query score and
rank = K_retrieve.

Usage:
    python src/expand.py --split train
    python src/expand.py --split test
"""
import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ensure_hashseed  # noqa: E402
from prep import TOKEN_TYPES  # noqa: E402
from retrieve import build_index, build_queries, country_rows, topk  # noqa: E402


def _tok(base, k):
    return pq.read_table(os.path.join(base, f"s{k}.parquet"), columns=TOKEN_TYPES + ["core"])


def task(args):
    base, ck, src, sib, k, chunk, out = args
    t0 = time.time()
    # sibling query table (siblings come from both sources)
    qtabs, owners = [], []
    for s in (2, 3):
        m = sib["src"].values == s
        if m.any():
            qtabs.append(_tok(base, s).take(pa.array(sib["d"].values[m])))
            owners.append(sib[m][["q", "src", "d"]].values)
    qtab = pa.concat_tables(qtabs)
    own = np.concatenate(owners)
    del qtabs
    rd = country_rows(os.path.join(base, f"s{src}.parquet"), ck)
    vocab, DT = build_index([_tok(base, src).take(pa.array(rd)).select(TOKEN_TYPES)])
    Q = build_queries(qtab, vocab, DT.shape[0])
    del qtab
    parts = []
    for s in range(0, Q.shape[0], chunk):
        R = Q[s:s + chunk] @ DT
        rows, _, idx, score, _ = topk(R, R.shape[1] + 1, k, rel=0.1)
        o = own[s + rows]
        keep = ~((o[:, 1] == src) & (o[:, 2] == rd[idx]))  # drop self-hits
        parts.append(pd.DataFrame({"q": o[keep, 0].astype(np.int32), "d": rd[idx[keep]].astype(np.int32),
                                   "exp_score": score[keep]}))
    del Q
    h = pd.concat(parts, ignore_index=True)
    agg = h.groupby(["q", "d"], sort=False).agg(n_sib=("exp_score", "size"),
                                                exp_score=("exp_score", "max")).reset_index()
    # S1-query score for every (q, d) hit, against the same index
    pos = np.full(int(rd.max()) + 1, -1, np.int64)
    pos[rd] = np.arange(len(rd))
    s1q = pq.read_table(os.path.join(base, "s1.parquet"), columns=TOKEN_TYPES + ["core"])
    Qn = build_queries(s1q.take(pa.array(agg["q"].values)), vocab, DT.shape[0])
    del s1q
    Dn = DT.T.tocsr()[pos[agg["d"].values]]
    agg["score"] = np.asarray(Qn.multiply(Dn).sum(1)).ravel().astype(np.float32)
    agg["src"] = np.int8(src)
    agg.to_parquet(out + ".tmp", index=False)
    os.replace(out + ".tmp", out)
    return f"[{ck}/S{src}] {len(own)} sibling queries -> {len(agg)} (q,doc) hits ({time.time() - t0:.0f}s)"


def main():
    ensure_hashseed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--work", default="work")
    ap.add_argument("--sib-thr", type=float, default=0.9)
    ap.add_argument("--k", type=int, default=5, help="hits per sibling query per source")
    ap.add_argument("--rank-new", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    base = os.path.join(a.work, a.split)
    exp_dir = os.path.join(base, "exp")
    os.makedirs(exp_dir, exist_ok=True)
    pr = pd.concat([pq.read_table(p, columns=["q", "src", "d", "p1"]).to_pandas()
                    for p in sorted(glob.glob(os.path.join(base, "pruned", "part-*.parquet")))],
                   ignore_index=True)
    sib = pr[pr["p1"].values >= a.sib_thr][["q", "src", "d"]].reset_index(drop=True)
    del pr
    print(f"siblings: {len(sib)} (p1>={a.sib_thr}) for {sib['q'].nunique()} S1 rows", flush=True)
    s1_ck = np.array(pq.read_table(os.path.join(base, "s1.parquet"), columns=["ck"]).column("ck").to_pylist(),
                     dtype=object)
    sib_ck = s1_ck[sib["q"].values]
    tasks = []
    for ck in sorted(set(sib_ck)):
        sc = sib[sib_ck == ck].reset_index(drop=True)
        for src in (2, 3):
            out = os.path.join(exp_dir, f"{ck}_S{src}.parquet")
            if os.path.exists(out):
                print(f"[{ck}/S{src}] checkpoint exists, skipping", flush=True)
                continue
            tasks.append((base, ck, src, sc, a.k, a.chunk, out))
    if tasks:
        with Pool(min(a.workers, len(tasks))) as pool:
            for msg in pool.imap_unordered(task, tasks):
                print(msg, flush=True)
    hits = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(exp_dir, "*.parquet")))],
                     ignore_index=True)
    cands = pq.read_table(os.path.join(base, "cands.parquet")).to_pandas()
    merged = cands.merge(hits[["q", "src", "d", "n_sib", "exp_score"]], on=["q", "src", "d"], how="left")
    merged["n_sib"] = merged["n_sib"].fillna(0).astype(np.float32)
    merged["exp_score"] = merged["exp_score"].fillna(0).astype(np.float32)
    key_c = cands["q"].values.astype(np.int64) * 8 * (1 << 32) + cands["src"].values.astype(np.int64) * (1 << 32) \
        + cands["d"].values.astype(np.int64)
    key_h = hits["q"].values.astype(np.int64) * 8 * (1 << 32) + hits["src"].values.astype(np.int64) * (1 << 32) \
        + hits["d"].values.astype(np.int64)
    new = hits[~np.isin(key_h, key_c)].copy()
    new["rank"] = np.int16(a.rank_new)
    print(f"expansion: {len(hits)} (q,doc) hits, {len(new)} new candidate pairs "
          f"for {new['q'].nunique()} S1 rows", flush=True)
    new = new[merged.columns]
    merged = pd.concat([merged, new.astype(merged.dtypes.to_dict())], ignore_index=True)
    out = os.path.join(base, "cands_all.parquet")
    pq.write_table(pa.Table.from_pandas(merged, preserve_index=False), out + ".tmp")
    os.replace(out + ".tmp", out)
    print(f"wrote {out}: {len(merged)} pairs ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
