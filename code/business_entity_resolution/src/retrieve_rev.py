"""Stage 2c: reverse blocking (S2/S3 record -> S1).

Forward retrieval scores each S1 row against 3-5M S2/S3 records per country;
in an index that large, common street and city words exceed the frequency cap
and are dropped, so a true record with a partial address ties with every
same-name record in the country. The S1 index is ~5x smaller and holds clean
addresses, so the same words stay discriminative. Every S2/S3 record belongs
to at most one S1, so its top-k S1 rows form a tiny, precise candidate list.
On validation, reverse top-3 recovers ~45% of the true links forward
retrieval misses.

Train: --drop-frac removes the same S1 rows as simulate_drop.py (test was built
by deleting ~19% of S1 entities, so deleted owners are absent from the index).

Work is split into (country, source, chunk) tasks, run in parallel and
checkpointed to <work>/<split>/rev/. The merge writes
<work>/<split>/cands_full.parquet = input candidates + reverse pairs, with
rev_rank (rank of this S1 in the record's reverse list, 9 if absent) and
rev_score. New pairs get score = rev_score and rank = --rank-new.

Usage:
    python src/retrieve_rev.py --split train --drop-frac 0.19
    python src/retrieve_rev.py --split test
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

KEY = lambda q, s, d: q.astype(np.int64) * 8 * (1 << 32) + s.astype(np.int64) * (1 << 32) + d.astype(np.int64)  # noqa


def task(args):
    base, ck, src, rows_d, r1, k, chunk, out = args
    t0 = time.time()
    vocab, DT = build_index([pq.read_table(os.path.join(base, "s1.parquet"), columns=TOKEN_TYPES)
                             .take(pa.array(r1))])
    qt = pq.read_table(os.path.join(base, f"s{src}.parquet"), columns=TOKEN_TYPES + ["core"]).take(pa.array(rows_d))
    Q = build_queries(qt, vocab, DT.shape[0])
    del qt, vocab
    parts = []
    for s in range(0, Q.shape[0], chunk):
        R = (Q[s:s + chunk] @ DT).tocsr()
        rows, _, idx, score, rank = topk(R, R.shape[1] + 1, k, rel=0.1)
        parts.append(pd.DataFrame({"q": r1[idx].astype(np.int32), "src": np.int8(src),
                                   "d": rows_d[s + rows].astype(np.int32),
                                   "rev_score": score.astype(np.float32), "rev_rank": rank.astype(np.int8)}))
    out_df = pd.concat(parts, ignore_index=True)
    out_df.to_parquet(out + ".tmp", index=False)
    os.replace(out + ".tmp", out)
    return f"[{ck}/S{src}] {len(rows_d)} record queries -> {len(out_df)} pairs ({time.time() - t0:.0f}s)"


def main():
    ensure_hashseed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--work", default="work")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--drop-frac", type=float, default=0.0)
    ap.add_argument("--in-cands", default="cands_all.parquet")
    ap.add_argument("--rank-new", type=int, default=20)
    ap.add_argument("--task-size", type=int, default=700000)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    t0 = time.time()
    base = os.path.join(a.work, a.split)
    rev_dir = os.path.join(base, "rev")
    os.makedirs(rev_dir, exist_ok=True)
    s1_path = os.path.join(base, "s1.parquet")
    drop = None
    if a.drop_frac > 0:
        from simulate_drop import dropped_mask
        ids = np.array(pq.read_table(s1_path, columns=["id"]).column("id").to_pylist(), dtype=object)
        drop = dropped_mask(ids, a.drop_frac)
        print(f"reverse index excludes {drop.sum()} dropped S1 rows", flush=True)
    cks = pq.read_table(s1_path, columns=["ck"]).column("ck").unique().to_pylist()
    tasks = []
    for ck in sorted(cks):
        r1 = country_rows(s1_path, ck)
        if drop is not None:
            r1 = r1[~drop[r1]]
        for src in (2, 3):
            rd = country_rows(os.path.join(base, f"s{src}.parquet"), ck)
            for i in range(0, len(rd), a.task_size):
                out = os.path.join(rev_dir, f"{ck}_S{src}_{i // a.task_size:03d}.parquet")
                if not os.path.exists(out):
                    tasks.append((base, ck, src, rd[i:i + a.task_size], r1, a.k, a.chunk, out))
    print(f"{len(tasks)} tasks to run", flush=True)
    if tasks:
        with Pool(min(a.workers, len(tasks))) as pool:
            for msg in pool.imap_unordered(task, tasks):
                print(msg, flush=True)
    rev = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(rev_dir, "*.parquet")))],
                    ignore_index=True)
    cands = pq.read_table(os.path.join(base, a.in_cands)).to_pandas()
    kc = KEY(cands["q"].values, cands["src"].values, cands["d"].values)
    kr = KEY(rev["q"].values, rev["src"].values, rev["d"].values)
    order = np.argsort(kr)
    kr_s = kr[order]
    pos = np.searchsorted(kr_s, kc)
    pos[pos >= len(kr_s)] = 0
    hit = kr_s[pos] == kc
    cands["rev_rank"] = np.where(hit, rev["rev_rank"].values[order][pos], 9).astype(np.int8)
    cands["rev_score"] = np.where(hit, rev["rev_score"].values[order][pos], 0).astype(np.float32)
    new = rev[~np.isin(kr, kc)].copy()
    new["score"] = new["rev_score"]
    new["rank"] = np.int16(a.rank_new)
    for c in ("n_sib", "exp_score"):
        if c in cands:
            new[c] = np.float32(0)
    new = new[cands.columns].astype(cands.dtypes.to_dict())
    print(f"reverse pairs {len(rev)}; already in candidates {hit.sum()}; new pairs {len(new)} "
          f"for {new['q'].nunique()} S1 rows", flush=True)
    merged = pd.concat([cands, new], ignore_index=True)
    out = os.path.join(base, "cands_full.parquet")
    pq.write_table(pa.Table.from_pandas(merged, preserve_index=False), out + ".tmp")
    os.replace(out + ".tmp", out)
    print(f"wrote {out}: {len(merged)} pairs ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
