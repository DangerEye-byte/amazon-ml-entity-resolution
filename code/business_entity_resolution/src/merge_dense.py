"""Stage 2f: add dense-retrieval candidates (src/colab_dense.py) to the token candidates.

Every candidate gets dn_rrank (rank of the S1 in the record's dense list, 99 if
absent), dn_frank (rank of the record in the S1's dense list, 99 if absent) and
dn_score (cosine; NaN when the pair is in neither list). Dense pairs that are not
candidates yet are added when dn_rrank < --add-rev or dn_frank < --add-fwd, with
token score 0, rank 20, rev_rank 9 and rr_rank 9 (no token evidence).
Streams the output in row groups (no full in-memory copy of ~100M candidates).

Usage:
    python src/merge_dense.py --split train --dense dense_pkg/out_dense/dense_train.parquet
    python src/merge_dense.py --split test --dense dense_pkg/out_dense/dense_test.parquet
"""
import argparse
import os
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

KEY = lambda q, s, d: np.asarray(q, np.int64) * 8 * (1 << 32) + np.asarray(s, np.int64) * (1 << 32) + np.asarray(d, np.int64)  # noqa
NEW_DEFAULTS = {"score": 0.0, "rank": 20, "rev_rank": 9, "rr_rank": 9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--in-work", default="work9")
    ap.add_argument("--in-cands", default="cands_v9.parquet")
    ap.add_argument("--dense", required=True)
    ap.add_argument("--out-work", default="work10")
    ap.add_argument("--out-name", default="cands_v10.parquet")
    ap.add_argument("--add-rev", type=int, default=3)
    ap.add_argument("--add-fwd", type=int, default=5)
    ap.add_argument("--rows", type=int, default=8_000_000, help="rows per output row group")
    a = ap.parse_args()
    t0 = time.time()
    base, outb = os.path.join(a.in_work, a.split), os.path.join(a.out_work, a.split)
    os.makedirs(outb, exist_ok=True)
    for f in ("s1.parquet", "s2.parquet", "s3.parquet"):
        dst = os.path.join(outb, f)
        if not os.path.exists(dst):
            try:
                os.link(os.path.join(base, f), dst)
            except OSError:
                shutil.copy(os.path.join(base, f), dst)
    cands = pq.read_table(os.path.join(base, a.in_cands))
    dn = pq.read_table(a.dense).to_pandas()
    kd = KEY(dn["q"].values, dn["src"].values, dn["d"].values)
    order = np.argsort(kd)
    kd = kd[order]
    rr, fr, sc = dn["rrank"].values[order], dn["frank"].values[order], dn["score"].values[order].astype(np.float32)
    del dn, order
    kc = KEY(cands.column("q").to_numpy(), cands.column("src").to_numpy(), cands.column("d").to_numpy())
    pos = np.clip(np.searchsorted(kd, kc), 0, len(kd) - 1)
    hit = kd[pos] == kc
    cands = cands.append_column("dn_rrank", pa.array(np.where(hit, rr[pos], 99).astype(np.int8)))
    cands = cands.append_column("dn_frank", pa.array(np.where(hit, fr[pos], 99).astype(np.int8)))
    cands = cands.append_column("dn_score", pa.array(np.where(hit, sc[pos], np.nan).astype(np.float32)))
    hit_rate = hit.mean()
    del pos, hit
    kc.sort()
    p2 = np.clip(np.searchsorted(kc, kd), 0, len(kc) - 1)
    known = kc[p2] == kd
    del kc, p2
    add = ~known & ((rr < a.add_rev) | (fr < a.add_fwd))
    ka = kd[add]
    new = {"q": (ka // (8 * (1 << 32))).astype(np.int32), "src": ((ka // (1 << 32)) % 8).astype(np.int8),
           "d": (ka % (1 << 32)).astype(np.int32), "dn_rrank": rr[add].astype(np.int8),
           "dn_frank": fr[add].astype(np.int8), "dn_score": sc[add]}
    cols = {}
    for f in cands.schema:
        cols[f.name] = new[f.name] if f.name in new else np.full(len(ka), NEW_DEFAULTS.get(f.name, 0),
                                                                     dtype=f.type.to_pandas_dtype())
    new_t = pa.table(cols, schema=cands.schema)
    print(f"dense pairs {len(kd)}: already candidates {int(known.sum())} ({hit_rate:.3f} of candidates are dense "
          f"pairs); new pairs {len(ka)} for {len(np.unique(new['q']))} S1 rows", flush=True)
    del kd, rr, fr, sc, known, add, ka, new, cols
    out = os.path.join(outb, a.out_name)
    with pq.ParquetWriter(out + ".tmp", cands.schema) as w:
        for s in range(0, cands.num_rows, a.rows):
            w.write_table(cands.slice(s, a.rows))
        w.write_table(new_t)
    os.replace(out + ".tmp", out)
    print(f"wrote {out}: {cands.num_rows + new_t.num_rows} pairs ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
