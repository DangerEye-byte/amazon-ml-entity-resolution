"""Stage 2d: exact-name blocking (recall booster).

After forward, sibling and reverse retrieval, ~40% of the true records still
missing have exactly the S1 row's normalised core name; most have an empty or
heavily truncated address, so token retrieval ranks them below same-name
records with fuller addresses. Here every S1 row is paired with every S2/S3
record of its country whose core name is identical, as long as that name is
rare (at most --max-group records share it).

Appends the new pairs to --in-cands (in place, atomically) with score 0,
rank --rank-new, and zero expansion / reverse columns; downstream features
recompute everything else from the tokens.

Usage:
    python src/name_bucket.py --split train --drop-frac 0.19
    python src/name_bucket.py --split test
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ensure_hashseed  # noqa: E402

KEY = lambda q, s, d: q.astype(np.int64) * 8 * (1 << 32) + s.astype(np.int64) * (1 << 32) + d.astype(np.int64)  # noqa


def main():
    ensure_hashseed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--work", default="work")
    ap.add_argument("--in-cands", default="cands_full.parquet")
    ap.add_argument("--max-group", type=int, default=10)
    ap.add_argument("--rank-new", type=int, default=20)
    ap.add_argument("--drop-frac", type=float, default=0.0)
    a = ap.parse_args()
    t0 = time.time()
    base = os.path.join(a.work, a.split)
    t = {k: pq.read_table(os.path.join(base, f"s{k}.parquet"), columns=["ck", "core"]).to_pandas() for k in (1, 2, 3)}
    docs = pd.concat([t[k].assign(src=np.int8(k), d=np.arange(len(t[k]), dtype=np.int32)) for k in (2, 3)],
                     ignore_index=True)
    docs = docs[docs["core"].str.len() > 0]
    docs["gsz"] = docs.groupby(["ck", "core"])["d"].transform("size")
    docs = docs[docs["gsz"] <= a.max_group].drop(columns="gsz")
    s1 = t[1].assign(q=np.arange(len(t[1]), dtype=np.int32))
    s1 = s1[s1["core"].str.len() > 0]
    if a.drop_frac > 0:
        from simulate_drop import dropped_mask
        ids = np.array(pq.read_table(os.path.join(base, "s1.parquet"), columns=["id"]).column("id").to_pylist(),
                       dtype=object)
        s1 = s1[~dropped_mask(ids, a.drop_frac)[s1["q"].values]]
    pairs = s1.merge(docs, on=["ck", "core"])[["q", "src", "d"]]
    del docs, s1, t
    path = os.path.join(base, a.in_cands)
    cands = pq.read_table(path).to_pandas()
    if "nb" in cands:  # already applied: recompute from the original rows only
        cands = cands[cands["nb"] == 0].reset_index(drop=True)
    kc = np.sort(KEY(cands["q"].values, cands["src"].values, cands["d"].values))
    kp = KEY(pairs["q"].values, pairs["src"].values, pairs["d"].values)
    pos = np.searchsorted(kc, kp)
    pos[pos >= len(kc)] = 0
    new = pairs[kc[pos] != kp].copy()
    print(f"exact-name pairs {len(pairs)}, new {len(new)} for {new['q'].nunique()} S1 rows", flush=True)
    for c in cands.columns:
        if c not in new:
            new[c] = 0
    new["rank"] = a.rank_new
    new["rev_rank"] = 9 if "rev_rank" in cands else 0
    cands["nb"] = np.int8(0)
    new["nb"] = np.int8(1)
    new = new[cands.columns].astype(cands.dtypes.to_dict())
    merged = pd.concat([cands, new], ignore_index=True)
    pq.write_table(pa.Table.from_pandas(merged, preserve_index=False), path + ".tmp")
    os.replace(path + ".tmp", path)
    print(f"wrote {path}: {len(merged)} pairs ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
