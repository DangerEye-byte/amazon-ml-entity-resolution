"""Stage 2e: wide reverse search with fine re-ranking for common-name records.

The remaining never-found true records are mostly records of common names
("Foot & Ankle Partners" x 264 S1 rows) with a partial address. In the reverse
search (record -> S1 index) their true S1 sits in a dense tie: ~36 S1 rows
score within 90% of the best, so the reverse top-3 misses it. On validation,
the true S1 of ~68% of the missed records is in the reverse top-50, and a fine
address comparison puts it first 82% / in the top-2 87% of the time.

For every record whose reverse top-3 is a near tie (all three within --tie of
the best), this retrieves the reverse top-k (--k 50), re-ranks them with
    token_set(address) + 0.5 * partial(address) + 40 * [same first house number]
    + 0.3 * token_set(name)
and keeps the best --keep S1 rows (skipping records whose best re-rank score is
shared by more than --max-ties rows: those are empty-address coin flips).
Only countries present in training are processed (--countries default).

Writes <out-work>/<split>/cands_v8.parquet = <work>/<split>/cands_full.parquet
+ new pairs, with rr_rank (0.. or 9 if the pair was not a re-ranked top pick)
and rr_score. New pairs get score = reverse score, rank = 20, rev_rank = 9.

Usage:
    python src/retrieve_rerank.py --split train --drop-frac 0.19 --out-work work8
    python src/retrieve_rerank.py --split test --out-work work8
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
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ensure_hashseed  # noqa: E402
from prep import TOKEN_TYPES  # noqa: E402
from retrieve import build_index, build_queries, country_rows, topk  # noqa: E402

KEY = lambda q, s, d: q.astype(np.int64) * 8 * (1 << 32) + s.astype(np.int64) * (1 << 32) + d.astype(np.int64)  # noqa


def first_num(n):
    n = str(n).split()
    return n[0] if n else ""


def task(args):
    base, ck, src, recs, r1, k, keep, max_ties, chunk, out = args
    t0 = time.time()
    vocab, DT = build_index([pq.read_table(os.path.join(base, "s1.parquet"), columns=TOKEN_TYPES).take(pa.array(r1))])
    qt = pq.read_table(os.path.join(base, f"s{src}.parquet"), columns=TOKEN_TYPES + ["core"]).take(pa.array(recs))
    Q = build_queries(qt, vocab, DT.shape[0])
    del qt, vocab
    s1 = pq.read_table(os.path.join(base, "s1.parquet"), columns=["core", "anorm", "nums"]).to_pandas()
    sd = pq.read_table(os.path.join(base, f"s{src}.parquet"), columns=["core", "anorm", "nums"]).take(
        pa.array(recs)).to_pandas()
    s1_n1 = s1["nums"].map(first_num).values
    sd_n1 = sd["nums"].map(first_num).values
    parts = []
    for s in range(0, Q.shape[0], chunk):
        R = (Q[s:s + chunk] @ DT).tocsr()
        rows, _, idx, score, rank = topk(R, R.shape[1] + 1, k, rel=0.3)
        del R
        qi = s + rows              # position in recs
        cand = r1[idx]             # S1 row
        da = sd["anorm"].values[qi].tolist()
        ca = s1["anorm"].values[cand].tolist()
        f = cpdist(da, ca, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) \
            + 0.5 * cpdist(da, ca, scorer=fuzz.partial_ratio, workers=-1, dtype=np.float32) \
            + 0.3 * cpdist(sd["core"].values[qi].tolist(), s1["core"].values[cand].tolist(),
                           scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        n1d, n1q = sd_n1[qi], s1_n1[cand]
        f = f + 40.0 * ((n1d == n1q) & (n1d != ""))
        df = pd.DataFrame({"qi": qi, "q": cand.astype(np.int32), "f": f, "rev_score": score.astype(np.float32)})
        df = df.sort_values(["qi", "f"], ascending=[True, False], kind="stable")
        df["rr_rank"] = df.groupby("qi").cumcount().astype(np.int8)
        top = df.groupby("qi")["f"].transform("max")
        nties = (df["f"] >= top - 1e-6).groupby(df["qi"]).transform("sum")
        df = df[(df["rr_rank"] < keep) & (nties <= max_ties)]
        parts.append(pd.DataFrame({"q": df["q"].values, "src": np.int8(src), "d": recs[df["qi"].values].astype(np.int32),
                                   "rr_rank": df["rr_rank"].values, "rr_score": df["f"].values.astype(np.float32),
                                   "rev_score_w": df["rev_score"].values}))
    out_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["q", "src", "d", "rr_rank", "rr_score", "rev_score_w"])
    out_df.to_parquet(out + ".tmp", index=False)
    os.replace(out + ".tmp", out)
    return f"[{ck}/S{src}] {len(recs)} tied records -> {len(out_df)} re-ranked pairs ({time.time() - t0:.0f}s)"


def main():
    ensure_hashseed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--work", default="work")
    ap.add_argument("--out-work", default="work8")
    ap.add_argument("--in-cands", default="cands_full.parquet")
    ap.add_argument("--drop-frac", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--tie", type=float, default=0.9)
    ap.add_argument("--max-ties", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--countries", nargs="*", default=None, help="default: countries present in training")
    ap.add_argument("--all-countries", action="store_true", help="also process countries absent from training")
    ap.add_argument("--out-name", default="cands_v8.parquet")
    a = ap.parse_args()
    t0 = time.time()
    base = os.path.join(a.work, a.split)
    outb = os.path.join(a.out_work, a.split)
    rr_dir = os.path.join(outb, "rerank")
    os.makedirs(rr_dir, exist_ok=True)
    for f in ("s1.parquet", "s2.parquet", "s3.parquet"):
        dst = os.path.join(outb, f)
        if not os.path.exists(dst):
            os.link(os.path.join(base, f), dst)
    s1_path = os.path.join(base, "s1.parquet")
    if a.countries is None:
        src_s1 = s1_path if a.all_countries else os.path.join(a.work, "train", "s1.parquet")
        a.countries = sorted(pq.read_table(src_s1, columns=["ck"]).column("ck").unique().to_pylist())
    drop = None
    if a.drop_frac > 0:
        from simulate_drop import dropped_mask
        ids = np.array(pq.read_table(s1_path, columns=["id"]).column("id").to_pylist(), dtype=object)
        drop = dropped_mask(ids, a.drop_frac)
    # records whose reverse top-3 is a near tie
    rev = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(base, "rev", "*.parquet")))],
                    ignore_index=True)
    g = rev.groupby(["src", "d"])["rev_score"].agg(["max", "min", "size"]).reset_index()
    g = g[(g["size"] >= 3) & (g["min"] >= a.tie * g["max"])]
    del rev
    tasks = []
    for ck in a.countries:
        r1 = country_rows(s1_path, ck)
        if drop is not None:
            r1 = r1[~drop[r1]]
        for src in (2, 3):
            rd = country_rows(os.path.join(base, f"s{src}.parquet"), ck)
            recs = np.intersect1d(rd, g[g["src"] == src]["d"].values).astype(np.int64)
            out = os.path.join(rr_dir, f"{ck}_S{src}.parquet")
            if not os.path.exists(out) and len(recs):
                tasks.append((base, ck, src, recs, r1, a.k, a.keep, a.max_ties, a.chunk, out))
    print(f"{len(tasks)} tasks, {sum(len(t[3]) for t in tasks)} tied records", flush=True)
    if tasks:
        with Pool(min(a.workers, len(tasks))) as pool:
            for msg in pool.imap_unordered(task, tasks):
                print(msg, flush=True)
    rr = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(rr_dir, "*.parquet")))],
                   ignore_index=True)
    cands = pq.read_table(os.path.join(base, a.in_cands)).to_pandas()
    kc = KEY(cands["q"].values, cands["src"].values, cands["d"].values)
    kr = KEY(rr["q"].values, rr["src"].values, rr["d"].values)
    order = np.argsort(kr)
    kr_s = kr[order]
    pos = np.searchsorted(kr_s, kc)
    pos[pos >= len(kr_s)] = 0
    hit = kr_s[pos] == kc
    cands["rr_rank"] = np.where(hit, rr["rr_rank"].values[order][pos], 9).astype(np.int8)
    cands["rr_score"] = np.where(hit, rr["rr_score"].values[order][pos], 0).astype(np.float32)
    new = rr[~np.isin(kr, kc)].copy()
    new["score"] = new["rev_score_w"]
    new["rank"] = np.int16(20)
    for c in cands.columns:
        if c not in new:
            new[c] = 9 if c == "rev_rank" else 0
    new = new[cands.columns].astype(cands.dtypes.to_dict())
    print(f"re-ranked pairs {len(rr)}; already candidates {int(hit.sum())}; new pairs {len(new)} "
          f"for {new['q'].nunique()} S1 rows", flush=True)
    merged = pd.concat([cands, new], ignore_index=True)
    out = os.path.join(outb, a.out_name)
    pq.write_table(pa.Table.from_pandas(merged, preserve_index=False), out + ".tmp")
    os.replace(out + ".tmp", out)
    print(f"wrote {out}: {len(merged)} pairs ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
