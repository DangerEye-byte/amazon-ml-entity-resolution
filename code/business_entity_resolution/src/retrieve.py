"""Stage 2: candidate generation (blocking).

Per country, S1 queries are scored against all S2+S3 records by an IDF-weighted
sparse dot product over hashed typed tokens (name words, name word pairs,
name char-4grams, address words, house-number x street pairs, numbers).
Tokens more frequent than a per-type cap are dropped from the index, which
bounds the cost of the sparse product. The top-K S2 and top-K S3 records per S1
are kept.

Output: <work>/<split>/cands.parquet with columns
    q (int32 row in s1.parquet), src (int8 2|3), d (int32 row in s{src}.parquet),
    score (float32), rank (int16, within source), rank_all (int16), n_tok (float32)

Usage:
    python src/retrieve.py --split train --k 25 [--s1-frac 0.05]
    python src/retrieve.py --split test  --k 25
"""
import argparse
import os
import sys
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ensure_hashseed, is_val  # noqa: E402
from prep import TOKEN_TYPES  # noqa: E402

CAPS = {"t_n": 4000, "t_nb": 2000, "t_c": 3000, "t_a": 3000, "t_h": 1500, "t_p": 1500}
# tuned on a US/S2 validation probe (recall@15: 95.6% -> 96.7%)
TYPE_W = {"t_n": 1.0, "t_nb": 0.3, "t_c": 0.4, "t_a": 2.0, "t_h": 1.05, "t_p": 1.4}


def abbrevs(core):
    """Abbreviated-name variants seen in S2/S3 ('shifflett capital mines' -> 'scmines', 'scm')."""
    toks = core.split()
    if len(toks) < 2:
        return []
    return ["".join(t[0] for t in toks[:-1]) + toks[-1], "".join(t[0] for t in toks)]


def flat(table, col):
    arr = table.column(col).combine_chunks()
    off = arr.offsets.to_numpy()
    vals = arr.values.to_numpy()
    return off, vals


def country_rows(path, ck):
    t = pq.read_table(path, columns=["ck"])
    return np.flatnonzero(pc.equal(t.column("ck"), ck).to_numpy(zero_copy_only=False))


def build_index(doc_tables, caps=CAPS):
    """doc_tables: list of pyarrow tables (S2 subset, S3 subset). Returns per-type vocab info + DT."""
    n_docs = sum(t.num_rows for t in doc_tables)
    vocab = {}
    rows_all, cols_all = [], []
    col_off = 0
    for typ in TOKEN_TYPES:
        vals_l, rows_l = [], []
        base = 0
        for t in doc_tables:
            off, vals = flat(t, typ)
            vals_l.append(vals)
            rows_l.append(np.repeat(np.arange(base, base + t.num_rows, dtype=np.int32), np.diff(off)))
            base += t.num_rows
        vals = np.concatenate(vals_l)
        rows = np.concatenate(rows_l)
        del vals_l, rows_l
        uniq, inv, df = np.unique(vals, return_inverse=True, return_counts=True)
        keep = df <= caps[typ]
        # remap kept tokens to compact column ids
        newid = np.full(len(uniq), -1, dtype=np.int64)
        newid[keep] = np.arange(keep.sum()) + col_off
        cols = newid[inv]
        m = cols >= 0
        rows_all.append(rows[m])
        cols_all.append(cols[m].astype(np.int32))
        idf = np.log(n_docs / df[keep]).astype(np.float32) * TYPE_W[typ]
        vocab[typ] = (uniq[keep], idf, col_off)
        col_off += int(keep.sum())
        print(f"  {typ}: {len(uniq)} tokens, {keep.sum()} kept (cap {caps[typ]}), "
              f"postings {m.sum()}", flush=True)
        del vals, rows, inv, cols, m
    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    del rows_all, cols_all
    DT = sp.csr_matrix((np.ones(len(rows), np.float32), (cols, rows)), shape=(col_off, n_docs))
    return vocab, DT


def build_queries(q_table, vocab, n_cols):
    rows_all, cols_all, w_all = [], [], []
    for typ in TOKEN_TYPES:
        off, vals = flat(q_table, typ)
        rows = np.repeat(np.arange(q_table.num_rows, dtype=np.int32), np.diff(off))
        if typ == "t_n" and "core" in q_table.column_names:
            ex_r, ex_v = [], []
            for i, c in enumerate(q_table.column("core").to_pylist()):
                for ab in abbrevs(c):
                    ex_r.append(i)
                    ex_v.append(hash("x:" + ab))
            rows = np.concatenate([rows, np.array(ex_r, np.int32)])
            vals = np.concatenate([vals, np.array(ex_v, np.int64)])
        uniq, idf, base = vocab[typ]
        pos = np.searchsorted(uniq, vals)
        pos[pos >= len(uniq)] = 0
        hit = uniq[pos] == vals if len(uniq) else np.zeros(len(vals), bool)
        rows_all.append(rows[hit])
        cols_all.append((pos[hit] + base).astype(np.int32))
        w_all.append(idf[pos[hit]])
    Q = sp.csr_matrix((np.concatenate(w_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
                      shape=(q_table.num_rows, n_cols))
    return Q


def topk(R, n2, k, rel=0.2):
    indptr, idx, data = R.indptr, R.indices, R.data
    rows = np.repeat(np.arange(R.shape[0], dtype=np.int32), np.diff(indptr))
    if len(data) > k * R.shape[0]:
        # cheap prefilter: drop entries far below their row's best score before sorting
        nz = np.diff(indptr) > 0
        rmax = np.zeros(R.shape[0], np.float32)
        rmax[nz] = np.maximum.reduceat(data, indptr[:-1][nz])
        m = data >= rel * rmax[rows]
        rows, idx, data = rows[m], idx[m], data[m]
    src = (idx >= n2).astype(np.int8)
    order = np.lexsort((-data, src, rows))
    rows, src, idx, data = rows[order], src[order], idx[order], data[order]
    key = rows.astype(np.int64) * 2 + src
    first = np.r_[True, key[1:] != key[:-1]]
    pos = np.arange(len(key))
    start = np.maximum.accumulate(np.where(first, pos, 0))
    rank = pos - start
    keep = rank < k
    return rows[keep], src[keep], idx[keep], data[keep], rank[keep].astype(np.int16)


def run(split, work, k, s1_frac, chunk, countries=None, only_val=False):
    base = os.path.join(work, split)
    s1_path = os.path.join(base, "s1.parquet")
    cks = pq.read_table(s1_path, columns=["ck"]).column("ck").unique().to_pylist()
    out_parts = []
    for ck in sorted(cks):
        if countries and ck not in countries:
            continue
        r1 = country_rows(s1_path, ck)
        if s1_frac < 1.0 or only_val:
            ids = np.array(pq.read_table(s1_path, columns=["id"]).column("id").take(r1).to_pylist(), dtype=object)
            m = is_val(ids) if only_val else is_val(ids, frac=s1_frac, seed=7)
            r1 = r1[m]
        t1 = pq.read_table(s1_path, columns=TOKEN_TYPES + ["core"]).take(r1)
        for src in (2, 3):
            t0 = time.time()
            path = os.path.join(base, f"s{src}.parquet")
            rd = country_rows(path, ck)
            print(f"[{split}/{ck}/S{src}] docs={len(rd)} queries={len(r1)}", flush=True)
            vocab, DT = build_index([pq.read_table(path, columns=TOKEN_TYPES).take(rd)])
            Q = build_queries(t1, vocab, DT.shape[0])
            del vocab
            print(f"  index built in {time.time() - t0:.0f}s; DT nnz={DT.nnz}, Q nnz={Q.nnz}", flush=True)
            t0 = time.time()
            for s in range(0, Q.shape[0], chunk):
                R = Q[s:s + chunk] @ DT
                rows, _, idx, score, rank = topk(R, R.shape[1] + 1, k)
                del R
                out_parts.append({
                    "q": r1[s + rows].astype(np.int32), "src": np.full(len(rows), src, np.int8),
                    "d": rd[idx].astype(np.int32), "score": score.astype(np.float32), "rank": rank,
                })
                done = min(s + chunk, Q.shape[0])
                if (s // chunk) % 25 == 0:
                    el = time.time() - t0
                    print(f"  {done}/{Q.shape[0]} queries, {el:.0f}s, eta {el / done * (Q.shape[0] - done):.0f}s",
                          flush=True)
            del Q, DT
        del t1
    import pyarrow as pa
    tbl = pa.table({c: np.concatenate([p[c] for p in out_parts]) for c in out_parts[0]})
    return tbl


def main():
    ensure_hashseed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--work", default="work")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--s1-frac", type=float, default=1.0, help="deterministic S1 subsample (train)")
    ap.add_argument("--only-val", action="store_true", help="only validation S1 rows")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--countries", nargs="*")
    ap.add_argument("--out")
    a = ap.parse_args()
    tbl = run(a.split, a.work, a.k, a.s1_frac, a.chunk, a.countries, a.only_val)
    out = a.out or os.path.join(a.work, a.split, "cands.parquet")
    pq.write_table(tbl, out)
    print("wrote", out, tbl.num_rows, "pairs")


if __name__ == "__main__":
    main()
