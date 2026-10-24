"""Stage 3: pairwise features for (S1, candidate) pairs, fully vectorized.

String similarities use rapidfuzz.process.cpdist (C++, multi-threaded).
Token overlaps (count and IDF-weighted, per token type) use sparse
element-wise products. Group features compare each candidate with the other
candidates of the same S1 row. Reverse features describe how contested a
candidate record is across all S1 rows (each S2/S3 record belongs to at most
one S1).
Country is deliberately NOT a feature (test contains an unseen country).
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import TOKEN_TYPES  # noqa: E402

STR_COLS = ["core", "compact", "alt", "legal", "anorm", "awords", "nums", "state"]
INT_COLS = ["is_domain", "nonlatin", "aempty", "anonlatin"]
HASH_BITS = 25
HMASK = (1 << HASH_BITS) - 1


class Split:
    """Normalized tables (strings + tokens) of one split, restricted to one country,
    plus hashed doc-frequency tables for IDF weights."""

    def __init__(self, split_dir, ck, tokens=True):
        self.ck = ck
        cols = ["id"] + STR_COLS + INT_COLS + (TOKEN_TYPES if tokens else [])
        self.rows = {}
        tabs = {}
        for k in (1, 2, 3):
            path = os.path.join(split_dir, f"s{k}.parquet")
            ckcol = pq.read_table(path, columns=["ck"]).column("ck")
            r = np.flatnonzero(pc.equal(ckcol, ck).to_numpy(zero_copy_only=False))
            self.rows[k] = r
            tabs[k] = pq.read_table(path, columns=cols).take(pa.array(r)).combine_chunks()
        self.s1 = tabs[1]
        # vocabulary of real business-name words (S1 is the clean reference source); S2/S3 names made
        # of words outside it are typically synthetic rebrands ("jaxorbi") of true matches
        self.vocab = set()
        for c in self.s1.column("core").to_pylist():
            self.vocab.update(c.split())
        self.n2 = tabs[2].num_rows
        self.docs = pa.concat_tables([tabs[2], tabs[3]]).combine_chunks()
        del tabs
        # global row -> local position lookups
        self.pos = {}
        for k in (1, 2, 3):
            n_total = int(self.rows[k].max()) + 1 if len(self.rows[k]) else 0
            lut = np.full(n_total, -1, np.int64)
            lut[self.rows[k]] = np.arange(len(self.rows[k]))
            self.pos[k] = lut
        n = self.docs.num_rows
        self.idf = {}
        for typ in (TOKEN_TYPES if tokens else []):
            vals = self.docs.column(typ).chunk(0).values.to_numpy()
            df = np.bincount((vals & HMASK).astype(np.int64), minlength=1 << HASH_BITS).astype(np.float32)
            self.idf[typ] = np.log((n + 1) / (df + 1)).astype(np.float32)
            del vals, df

    def q_index(self, q):
        return self.pos[1][q]

    def doc_index(self, src, d):
        out = np.empty(len(d), np.int64)
        m2 = src == 2
        out[m2] = self.pos[2][d[m2]]
        out[~m2] = self.pos[3][d[~m2]] + self.n2
        return out


def _take(table, idx):
    return table.take(pa.array(idx))


def _overlap(qt, dt, typ, idf):
    """Row-aligned token overlap between two list<int64> columns."""
    qa = qt.column(typ).combine_chunks() if isinstance(qt.column(typ), pa.ChunkedArray) else qt.column(typ)
    da = dt.column(typ).combine_chunks() if isinstance(dt.column(typ), pa.ChunkedArray) else dt.column(typ)
    qo, qv = qa.offsets.to_numpy(), qa.values.to_numpy()
    do, dv = da.offsets.to_numpy(), da.values.to_numpy()
    # offsets may not start at 0 after take -> normalise
    qv = qv[qo[0]:qo[-1]]
    qo = qo - qo[0]
    dv = dv[do[0]:do[-1]]
    do = do - do[0]
    n = len(qo) - 1
    allv = np.concatenate([qv, dv])
    uniq, inv = np.unique(allv, return_inverse=True)
    qi, di = inv[:len(qv)], inv[len(qv):]
    qw = idf[(qv & HMASK)]
    dw = idf[(dv & HMASK)]
    qr = np.repeat(np.arange(n), np.diff(qo))
    dr = np.repeat(np.arange(n), np.diff(do))
    Qm = sp.csr_matrix((qw, (qr, qi)), shape=(n, len(uniq)))
    Dm = sp.csr_matrix((np.ones(len(dv), np.float32), (dr, di)), shape=(n, len(uniq)))
    shared_w = np.asarray(Qm.multiply(Dm).sum(1)).ravel()
    Qb = Qm.copy()
    Qb.data[:] = 1
    shared_c = np.asarray(Qb.multiply(Dm).sum(1)).ravel()
    q_tot = np.bincount(qr, weights=qw, minlength=n)
    d_tot = np.bincount(dr, weights=dw, minlength=n)
    q_cnt = np.diff(qo).astype(np.float32)
    d_cnt = np.diff(do).astype(np.float32)
    return {
        f"{typ}_sw": shared_w,
        f"{typ}_sc": shared_c,
        f"{typ}_qc": np.divide(shared_w, q_tot, out=np.zeros(n), where=q_tot > 0),
        f"{typ}_dc": np.divide(shared_w, d_tot, out=np.zeros(n), where=d_tot > 0),
        f"{typ}_jac": np.divide(shared_c, q_cnt + d_cnt - shared_c, out=np.zeros(n),
                                where=(q_cnt + d_cnt - shared_c) > 0),
    }


def _sim(a, b, scorer, **kw):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def block_features(S, q, src, d, score, rank):
    """Cheap blocking-level features used by the stage-1 pruning model:
    token overlaps per type, retrieval score/rank and two fast similarities."""
    di = S.doc_index(src, d)
    qt = _take(S.s1, S.q_index(q))
    dt = _take(S.docs, di)
    f = {}
    for typ in TOKEN_TYPES:
        f.update(_overlap(qt, dt, typ, S.idf[typ]))
    f["cmp_ratio"] = _sim(qt.column("compact").to_pylist(), dt.column("compact").to_pylist(), fuzz.ratio)
    f["aw_tset"] = _sim(qt.column("awords").to_pylist(), dt.column("awords").to_pylist(), fuzz.token_set_ratio)
    f["is_s3"] = (src == 3).astype(np.float32)
    f["ret_score"] = score.astype(np.float32)
    f["ret_rank"] = rank.astype(np.float32)
    f["d_aempty"] = dt.column("aempty").to_numpy().astype(np.float32)
    f["d_is_domain"] = dt.column("is_domain").to_numpy().astype(np.float32)
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})


def string_features(S, q, src, d):
    """Full string-similarity features for the stage-2 matcher."""
    di = S.doc_index(src, d)
    qt = _take(S.s1, S.q_index(q))
    dt = _take(S.docs, di)
    f = {}
    col = lambda t, c: t.column(c).to_pylist()  # noqa: E731
    qc, dc = col(qt, "core"), col(dt, "core")
    f["nm_ratio"] = _sim(qc, dc, fuzz.ratio)
    f["nm_tset"] = _sim(qc, dc, fuzz.token_set_ratio)
    f["nm_tsort"] = _sim(qc, dc, fuzz.token_sort_ratio)
    f["nm_partial"] = _sim(qc, dc, fuzz.partial_ratio)
    f["nm_jw"] = _sim(qc, dc, JaroWinkler.normalized_similarity)
    f["nm_wr"] = _sim(qc, dc, fuzz.WRatio)
    qx, dx = col(qt, "compact"), col(dt, "compact")
    f["cmp_partial"] = _sim(qx, dx, fuzz.partial_ratio)
    f["cmp_jw"] = _sim(qx, dx, JaroWinkler.normalized_similarity)
    ab = ["".join(t[0] for t in c.split()[:-1]) + c.split()[-1] if " " in c else "" for c in qc]
    f["abbr_ratio"] = _sim(ab, dx, fuzz.ratio)
    qcl, dcl = [_clean_name(x) for x in qc], [_clean_name(x) for x in dc]
    f["nmc_tset"] = _sim(qcl, dcl, fuzz.token_set_ratio)
    f["nmc_ratio"] = _sim(qcl, dcl, fuzz.ratio)
    f["nm_extra_d"] = np.array([len(set(b.split()) - set(a.split())) for a, b in zip(qcl, dcl)], np.float32)
    f["nm_extra_q"] = np.array([len(set(a.split()) - set(b.split())) for a, b in zip(qcl, dcl)], np.float32)
    voc = S.vocab
    oov = [[w not in voc for w in x.split()] for x in dc]
    f["d_oov_frac"] = np.array([sum(o) / len(o) if o else -1 for o in oov], np.float32)
    f["d_oov_cnt"] = np.array([sum(o) for o in oov], np.float32)
    f["d_cmp_in_voc"] = np.array([int(x in voc) for x in dx], np.float32)
    dalt = col(dt, "alt")
    has_alt = np.array([1 if x else 0 for x in dalt], np.float32)
    f["alt_tset"] = np.where(has_alt > 0, _sim(qc, dalt, fuzz.token_set_ratio), -1)
    ql, dl = col(qt, "legal"), col(dt, "legal")
    f["legal_ratio"] = _sim(ql, dl, fuzz.token_set_ratio)
    f["legal_q_empty"] = np.array([not x for x in ql], np.float32)
    f["legal_d_empty"] = np.array([not x for x in dl], np.float32)
    qa, da = col(qt, "anorm"), col(dt, "anorm")
    f["ad_ratio"] = _sim(qa, da, fuzz.ratio)
    f["ad_tset"] = _sim(qa, da, fuzz.token_set_ratio)
    f["ad_tsort"] = _sim(qa, da, fuzz.token_sort_ratio)
    f["ad_partial"] = _sim(qa, da, fuzz.partial_ratio)
    admin = FR_ADMIN if S.ck == "fr" else set()
    qst, dst = col(qt, "state"), col(dt, "state")
    qa2 = [_strip_admin(x, st, admin) for x, st in zip(qa, qst)]
    da2 = [_strip_admin(x, st, admin) for x, st in zip(da, dst)]
    f["ad_core_tset"] = _sim(qa2, da2, fuzz.token_set_ratio)
    f["ad_core_tsort"] = _sim(qa2, da2, fuzz.token_sort_ratio)
    qw, dw = col(qt, "awords"), col(dt, "awords")
    f["aw_tsort"] = _sim(qw, dw, fuzz.token_sort_ratio)
    qn, dn = col(qt, "nums"), col(dt, "nums")
    f["num_tset"] = _sim(qn, dn, fuzz.token_set_ratio)
    q1 = np.array([x.split(" ", 1)[0] for x in qn], dtype=object)
    d1 = np.array([x.split(" ", 1)[0] for x in dn], dtype=object)
    f["num1_eq"] = ((q1 == d1) & (q1 != "")).astype(np.float32)
    both = (q1 != "") & (d1 != "")
    f["num1_lev"] = np.where(both, _sim(list(q1), list(d1), Levenshtein.distance), -1)
    f["num1_prefix"] = np.array([int(a != b and bool(a) and bool(b) and (a.startswith(b) or b.startswith(a)))
                                 for a, b in zip(q1, d1)], np.float32)
    f["num1_logdiff"] = np.array([np.log1p(abs(int(a[:9]) - int(b[:9]))) if a and b else -1
                                  for a, b in zip(q1, d1)], np.float32)
    qns = [set(x.split()) for x in qn]
    dns = [set(x.split()) for x in dn]
    f["num_shared"] = np.array([len(a & b) for a, b in zip(qns, dns)], np.float32)
    f["num_q_only"] = np.array([len(a - b) for a, b in zip(qns, dns)], np.float32)
    f["num_d_only"] = np.array([len(b - a) for a, b in zip(qns, dns)], np.float32)
    f["num_q_empty"] = (q1 == "").astype(np.float32)
    f["num_d_empty"] = (d1 == "").astype(np.float32)
    qs = np.array(col(qt, "state"), dtype=object)
    ds = np.array(col(dt, "state"), dtype=object)
    f["state_eq"] = np.where((qs == "") | (ds == ""), -1, (qs == ds).astype(np.float32))
    for c in ("nonlatin", "anonlatin"):
        f["d_" + c] = dt.column(c).to_numpy().astype(np.float32)
    f["q_nm_len"] = np.array([len(x) for x in qc], np.float32)
    f["d_nm_len"] = np.array([len(x) for x in dc], np.float32)
    f["q_nm_ntok"] = np.array([x.count(" ") + 1 if x else 0 for x in qc], np.float32)
    f["d_nm_ntok"] = np.array([x.count(" ") + 1 if x else 0 for x in dc], np.float32)
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})


# French administrative area words (régions / départements) as they appear after normalization;
# S1 tends to give the région, S2/S3 the département, so they only add noise to address similarity
FR_ADMIN = {"hauts", "pays", "loire", "atlantique", "nouvelle", "aquitaine", "nord", "pas", "gironde",
            "ile", "occitanie", "bretagne", "normandie", "grand", "est", "auvergne", "rhone", "alpes",
            "bourgogne", "franche", "comte", "centre", "val", "provence", "cote", "azur", "corse",
            "maritime", "finistere", "morbihan", "vendee", "charente", "landes", "pyrenees", "somme",
            "aisne", "oise", "herault", "gard", "var", "savoie", "isere", "moselle", "rhin", "marne"}


def _strip_admin(addr, state, admin):
    return " ".join(t for t in addr.split() if t != state and t not in admin and not t.isdigit())


WEB_TOKENS = {"www", "com", "http", "https", "net", "org", "in", "co", "c0m", "html"}


def _clean_name(s):
    return " ".join(t for t in s.split() if t not in WEB_TOKENS and not t.isdigit())


def anchor_features(S, q, src, d, p1):
    """Sibling consensus: compare each candidate with the strongest OTHER candidate of the same
    S1 (by stage-1 probability). True matches of one entity agree with each other; injected
    near-duplicates (house number off by one, different name at the same address) do not."""
    n = len(q)
    order = np.lexsort((-p1, q))
    qs = q[order]
    first = np.r_[True, qs[1:] != qs[:-1]]
    pos = np.arange(n)
    start = np.maximum.accumulate(np.where(first, pos, 0))
    grp = np.cumsum(first) - 1
    size = np.bincount(grp)[grp]
    rank = pos - start
    best = order[start]
    second = np.where(size > 1, order[np.minimum(start + 1, n - 1)], -1)
    anc_sorted = np.where(rank == 0, second, best)
    anc = np.empty(n, np.int64)
    anc[order] = anc_sorted
    has = anc >= 0
    f = {k: np.full(n, -1.0, np.float32) for k in
         ("anc_p1", "anc_nm_tset", "anc_ad_tset", "anc_num1_eq", "anc_cmp_ratio", "anc_aw_tset")}
    if has.any():
        i = np.flatnonzero(has)
        j = anc[i]
        di = S.doc_index(src[i], d[i])
        dj = S.doc_index(src[j], d[j])
        a = _take(S.docs, di)
        b = _take(S.docs, dj)
        col = lambda t, c: t.column(c).to_pylist()  # noqa: E731
        f["anc_p1"][i] = p1[j]
        f["anc_nm_tset"][i] = _sim(col(a, "core"), col(b, "core"), fuzz.token_set_ratio)
        f["anc_ad_tset"][i] = _sim(col(a, "anorm"), col(b, "anorm"), fuzz.token_set_ratio)
        f["anc_cmp_ratio"][i] = _sim(col(a, "compact"), col(b, "compact"), fuzz.ratio)
        f["anc_aw_tset"][i] = _sim(col(a, "awords"), col(b, "awords"), fuzz.token_set_ratio)
        n1 = np.array([x.split(" ", 1)[0] for x in col(a, "nums")], dtype=object)
        n2 = np.array([x.split(" ", 1)[0] for x in col(b, "nums")], dtype=object)
        f["anc_num1_eq"][i] = np.where((n1 == "") | (n2 == ""), -1, (n1 == n2).astype(np.float32))
    return pd.DataFrame(f)


BLOCK_GROUP_COLS = ["ret_score", "cmp_ratio", "aw_tset", "t_n_qc", "t_a_qc", "t_h_sw", "t_c_qc"]
FULL_GROUP_COLS = ["nm_tset", "nm_ratio", "ad_tset", "p1", "nmc_tset", "anc_ad_tset"]


def add_group_features(df, q, cols, prefix=""):
    """Within-S1 context: max over the S1's candidates, gap to it, and rank."""
    g = pd.Series(q)
    key = pd.Series(q).values
    for c in cols:
        mx = df[c].groupby(key).transform("max").values
        df[c + "_gmax"] = mx
        df[c + "_gap"] = mx - df[c].values
        df[c + "_grank"] = df[c].groupby(key).rank(ascending=False, method="min").values
    df[prefix + "g_n"] = g.groupby(key).transform("size").values.astype(np.float32)
    if "nm_tset" in df and "ad_tset" in df:
        strong = ((df["nm_tset"] >= 85) & (df["ad_tset"] >= 70)).astype(np.float32)
        df[prefix + "g_strong"] = strong.groupby(key).transform("sum").values
    return df


def add_reverse_features(cands):
    """For every candidate record: how many S1 rows retrieved it and this S1's rank among them.
    cands: DataFrame with q, src, d, score. Returns arrays aligned to cands rows."""
    key = cands["src"].values.astype(np.int64) * (1 << 32) + cands["d"].values.astype(np.int64)
    order = np.lexsort((-cands["score"].values, key))
    ks = key[order]
    first = np.r_[True, ks[1:] != ks[:-1]]
    pos = np.arange(len(ks))
    start = np.maximum.accumulate(np.where(first, pos, 0))
    rrank = np.empty(len(ks), np.float32)
    rrank[order] = pos - start
    grp = np.cumsum(first) - 1
    cnt = np.bincount(grp)
    rcnt = np.empty(len(ks), np.float32)
    rcnt[order] = cnt[grp]
    sc = cands["score"].values[order]
    best = sc[start]
    # second best score within the group (0 if none)
    second = np.zeros(len(ks), np.float32)
    has2 = (pos - start) == 1
    second_by_grp = np.zeros(len(cnt), np.float32)
    second_by_grp[grp[has2]] = sc[has2]
    rbest_other = np.empty(len(ks), np.float32)
    # best score among OTHER S1 rows for this doc
    other = np.where((pos - start) == 0, second_by_grp[grp], best)
    rbest_other[order] = other
    return rrank, rcnt, rbest_other
