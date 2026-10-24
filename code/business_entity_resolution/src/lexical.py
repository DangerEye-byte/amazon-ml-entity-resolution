"""Word-level (lexical) features for candidate pairs.

String similarities tell the matcher *how much* two names differ, not *which*
words differ. The data generator makes true records by adding alias words
(Center, Services, Partners, & Fils ...) or typos, and decoy businesses by
adding decoy words (Holdings, Group, East ...), swapping the business-type
word (Clinic -> Projects) or replacing a name with a similar real name
(Clyos -> Klyos, Barry -> Bandy). This module splits every pair's name
difference into:
  * variants: a record word within edit similarity >= 70 of a removed S1 word;
    'typo' if the new word is unknown to the country's S1 vocabulary,
    'real' if it is a real name word (confusable-name decoys);
  * extra words: new record words that are not variants;
  * missing words: S1 words absent from the record (not variants).
Extra and missing words are target-encoded with the true-match rate observed
for that word in labelled training pairs (never on validation rows), with
smoothing towards the prior.
"""
import collections
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz

LEG = {"sas", "sarl", "sa", "sasu", "eurl", "snc", "sci", "et", "ei", "pvt", "ltd", "private", "limited", "llp",
       "llc", "inc", "corp", "co", "company", "corporation", "incorporated", "lp", "pc", "pllc", "l", "p", "c"}


def first_num(n):
    n = str(n).split()
    return n[0] if n else ""


def num_type(q, d):
    if not q and not d:
        return "none"
    if not q:
        return "q_missing"
    if not d:
        return "d_missing"
    if q == d:
        return "equal"
    if len(d) == len(q) - 1 and any(q[:i] + q[i + 1:] == d for i in range(len(q))):
        return "del1"
    if len(d) == len(q) + 1 and any(d[:i] + d[i + 1:] == q for i in range(len(d))):
        return "ins1"
    near = q.isdigit() and d.isdigit() and abs(int(q) - int(d)) <= 30
    if len(d) == len(q) and sum(a != b for a, b in zip(q, d)) == 1:
        return "sub1_near" if near else "sub1_far"
    return "near" if near else "other"


def load_split(split_dir):
    s = {}
    for k in (1, 2, 3):
        cols = ["core", "legal", "nums", "awords"] + (["ck"] if k == 1 else [])
        s[k] = pq.read_table(os.path.join(split_dir, f"s{k}.parquet"), columns=cols).to_pandas()
    vocab = {}
    for ck in s[1]["ck"].unique():
        c = collections.Counter(w for x in s[1]["core"].values[s[1]["ck"].values == ck] for w in set(x.split()))
        vocab[ck] = {w for w, n in c.items() if n >= 3}
    return s, vocab


def diff(qcore, dcore, voc):
    qtoks = [t for t in qcore.split() if t not in LEG]
    qa = set(qtoks)
    db = set(dcore.split()) - LEG
    qo, do = qa - db, db - qa
    extra, missing, n_typo, n_real = [], set(qo), 0, 0
    first = qtoks[0] if qtoks else ""
    first_state, min_var = ("kept" if first in db else "lost"), 100.0
    for w in do:
        best, bx = 0, None
        for x in missing:
            r = fuzz.ratio(w, x)
            if r > best:
                best, bx = r, x
        if best >= 70:
            missing.discard(bx)
            real = w in voc and len(w) >= 3
            n_real += real
            n_typo += not real
            min_var = min(min_var, best)
            if bx == first:
                first_state = "real" if real else "typo"
        else:
            extra.append(w)
    return extra, sorted(missing), n_typo, n_real, len(qa & db), len(qa), len(db), first_state, min_var


def pair_diffs(df, s, vocab):
    ck = s[1]["ck"].values
    col = lambda c: (s[1][c].values, {2: s[2][c].values, 3: s[3][c].values})  # noqa: E731
    qc, dc = col("core")
    ql, dl = col("legal")
    qn, dn = col("nums")
    qa, da = col("awords")
    out = []
    for q, k, d in zip(df["q"].values, df["src"].values, df["d"].values):
        k = int(k)
        r = diff(qc[q], dc[k][d], vocab[ck[q]])
        legal = (ql[q] or "-") + ">" + (dl[k][d] or "-")
        nt = num_type(first_num(qn[q]), first_num(dn[k][d]))
        qw, dw = set(str(qa[q]).split()), set(str(da[k][d]).split())
        out.append(r + (legal, nt, sorted(dw - qw)[:6], sorted(qw - dw)[:6]))
    return out


def legal_rel(legal):
    q, d = legal.split(">")
    if q == "-" and d == "-":
        return "none"
    if q == "-":
        return "q_missing"
    if d == "-":
        return "d_missing"
    return "same" if q == d else "diff"


AGNOSTIC = ["lx_n_extra", "lx_n_missing", "lx_n_typo", "lx_n_realvar", "lx_n_shared", "lx_numtype", "lx_nq", "lx_nd",
            "lx_min_var", "lx_first", "lx_first_state", "lx_legrel", "lx_legrelnum"]


def fit_encoder(diffs, labels, a=20.0):
    prior = float(np.mean(labels))
    tabs = {k: collections.defaultdict(lambda: [0, 0])
            for k in ("extra", "missing", "legal", "num", "aextra", "amissing", "legnum", "first", "legrel", "legrelnum")}
    for (extra, missing, _, _, _, _, _, first, _, legal, nt, aex, ami), y in zip(diffs, labels):
        for key, items in (("extra", extra), ("missing", missing), ("legal", [legal]), ("num", [nt]),
                           ("aextra", aex), ("amissing", ami), ("legnum", [legal + "|" + nt]),
                           ("first", [first + "|" + nt]), ("legrel", [legal_rel(legal)]),
                           ("legrelnum", [legal_rel(legal) + "|" + nt])):
            for w in items:
                c = tabs[key][w]
                c[0] += 1
                c[1] += y
    enc = lambda d: {w: ((p + a * prior) / (n + a), n) for w, (n, p) in d.items()}  # noqa: E731
    out = {k: enc(v) for k, v in tabs.items()}
    out["prior"] = prior
    return out


def features(diffs, encoder):
    prior = encoder["prior"]
    ex, mi = encoder["extra"], encoder["missing"]
    lg, nm, ax, am = encoder["legal"], encoder["num"], encoder["aextra"], encoder["amissing"]
    ln, fs = encoder["legnum"], encoder["first"]
    lr, lrn = encoder["legrel"], encoder["legrelnum"]
    rows = []
    for extra, missing, n_typo, n_real, n_shared, nq, nd, first, min_var, legal, nt, aex, ami in diffs:
        e = [ex.get(w, (prior, 0)) for w in extra]
        m = [mi.get(w, (prior, 0)) for w in missing]
        er = [r for r, _ in e]
        mr = [r for r, _ in m]
        axr = [ax.get(w, (prior, 0))[0] for w in aex]
        amr = [am.get(w, (prior, 0))[0] for w in ami]
        rows.append((len(extra), len(missing), n_typo, n_real, n_shared,
                     min(er) if er else 1.0, float(np.mean(er)) if er else 1.0,
                     min(mr) if mr else 1.0, float(np.mean(mr)) if mr else 1.0,
                     sum(1 for _, n in e if n == 0), sum(1 for _, n in m if n == 0),
                     lg.get(legal, (prior, 0))[0], nm.get(nt, (prior, 0))[0],
                     min(axr) if axr else 1.0, min(amr) if amr else 1.0, len(aex), len(ami),
                     max(er) if er else 1.0, nq, nd, min_var, ln.get(legal + "|" + nt, (prior, 0))[0],
                     fs.get(first + "|" + nt, (prior, 0))[0], {"kept": 0, "typo": 1, "real": 2, "lost": 3}[first],
                     lr.get(legal_rel(legal), (prior, 0))[0], lrn.get(legal_rel(legal) + "|" + nt, (prior, 0))[0]))
    cols = ["lx_n_extra", "lx_n_missing", "lx_n_typo", "lx_n_realvar", "lx_n_shared", "lx_extra_min", "lx_extra_mean",
            "lx_missing_min", "lx_missing_mean", "lx_extra_unseen", "lx_missing_unseen", "lx_legal", "lx_numtype",
            "lx_aextra_min", "lx_amissing_min", "lx_n_aextra", "lx_n_amissing", "lx_extra_max", "lx_nq", "lx_nd",
            "lx_min_var", "lx_legnum", "lx_first", "lx_first_state", "lx_legrel", "lx_legrelnum"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Word-role statistics: a language-independent description of the words that differ.
# For every country separately (no labels), each word seen as an extra record word
# is described by how it behaves across all candidate pairs of that country:
#   swap share  - how often it replaces a name word rather than being added
#   eq share    - how often records carrying it keep the S1 house number
#   near share  - how often they sit at a nearby, different number (decoy pattern)
#   own rate    - how often the record name + number is another S1's name + number
#   s1 freq     - how many S1 names of the country contain the word (descriptor words)
#   count       - frequency among difference words (per million candidate pairs)
# The matcher learns on India/US what these statistics mean; they are computed the
# same way for a country absent from training.
ROLE_COLS = ["rl_swap_min", "rl_swap_max", "rl_eq_min", "rl_near_max", "rl_own_max", "rl_s1f_max", "rl_cnt_min",
             "rl_n_role_words"]


def role_stats(df, s, vocab_counts, ck_of_q):
    """df: q, src, d candidate pairs (one split). Returns {country: {word: stats tuple}}."""
    qc, dc = s[1]["core"].values, {2: s[2]["core"].values, 3: s[3]["core"].values}
    qn = s[1]["nums"].values
    dn = {2: s[2]["nums"].values, 3: s[3]["nums"].values}
    acc = collections.defaultdict(lambda: np.zeros(6))  # swap, add, eq, near, own, total
    npairs = collections.Counter()
    s1_key = collections.defaultdict(set)
    ck1 = s[1]["ck"].values
    for i in range(len(qc)):
        s1_key[ck1[i]].add(" ".join(sorted(set(qc[i].split()) - LEG)) + "|" + first_num(qn[i]))
    for q, k, d in zip(df["q"].values, df["src"].values, df["d"].values):
        k = int(k)
        ck = ck1[q]
        npairs[ck] += 1
        qa = set(qc[q].split()) - LEG
        db = set(dc[k][d].split()) - LEG
        qo, do = qa - db, db - qa
        if not do or not (qa & db):
            continue
        nt = num_type(first_num(qn[q]), first_num(dn[k][d]))
        own = (" ".join(sorted(db)) + "|" + first_num(dn[k][d])) in s1_key[ck]
        swap = len(qo) >= 1
        for w in do:
            v = acc[(ck, w)]
            v[0] += swap
            v[1] += not swap
            v[2] += nt == "equal"
            v[3] += nt in ("near", "sub1_near")
            v[4] += own
            v[5] += 1
    out = collections.defaultdict(dict)
    for (ck, w), v in acc.items():
        n = v[5]
        out[ck][w] = (v[0] / n, v[2] / n, v[3] / n, v[4] / n,
                      np.log1p(vocab_counts.get(ck, {}).get(w, 0)), np.log1p(1e6 * n / max(1, npairs[ck])))
    return out


def role_features(diffs, ck_of_pairs, stats):
    rows = []
    for (extra, *_), ck in zip(diffs, ck_of_pairs):
        st = [stats[ck][w] for w in extra if w in stats.get(ck, {})]
        if not st:
            rows.append((-1, -1, -1, -1, -1, -1, -1, 0))
            continue
        a = np.array(st)
        rows.append((a[:, 0].min(), a[:, 0].max(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max(), a[:, 4].max(),
                     a[:, 5].min(), len(st)))
    return pd.DataFrame(rows, columns=ROLE_COLS)


def s1_vocab_counts(s):
    out = {}
    ck1 = s[1]["ck"].values
    for ck in set(ck1):
        c = collections.Counter()
        for x in s[1]["core"].values[ck1 == ck]:
            c.update(set(x.split()))
        out[ck] = c
    return out
