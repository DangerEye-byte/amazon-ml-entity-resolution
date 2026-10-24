"""Post-filter for countries unseen in training: drop descriptor-swap matches.

The data generator builds name variants of a business in two ways that matter
here. A *true* record may replace the business-type word of the name with one
of a few alias words (US/India: Center, Services, Service, Partners). A
*different* business at the same address is created by replacing it with
another business-type word ("Cmp Club SA" -> "Cmp Section SA"). On US/India
validation, swaps to alias words are 86-90% true and swaps to any other word
~0% true; the matcher learned the English/Indian vocabulary from training, so
it rejects the latter. For a country absent from training (France) it has
never seen the local words and accepts such swaps.

For every unseen country we therefore derive its alias words from the test
data itself (no labels): among candidate pairs whose names differ by exactly
one replaced word, alias words are frequent (>= --alias-share of all swaps)
and almost never form the exact name + house number of another S1 business
("own rate" < --alias-own). A selected match that swaps to any other word is
removed, unless the two words are spelling variants or a known abbreviation
pair (Cie/Compagnie, Saint/St, Ets/Etablissements).

Usage:
    python src/swap_filter.py --in-dir submissions/v7 --out-dir submissions/v7_swap
"""
import argparse
import collections
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz

LEG = {"sas", "sarl", "sa", "sasu", "eurl", "snc", "sci", "et", "ei", "pvt", "ltd", "private", "limited", "llp",
       "llc", "inc", "corp", "co", "company", "corporation", "incorporated"}
ABBREV = [("cie", "compagnie"), ("saint", "st"), ("sainte", "ste"), ("etablissements", "ets"),
          ("etablissement", "ets"), ("etablissements", "etablissement")]
KEEP_PAIRS = set(ABBREV) | {(b, a) for a, b in ABBREV}


def words(core):
    return set(core.split()) - LEG


def first_num(n):
    n = str(n).split()
    return n[0] if n else ""


def swap(qa, db):
    """(old word, new word) if the names differ by exactly one replaced word, else None."""
    qo, do = qa - db, db - qa
    if len(qo) == 1 and len(do) == 1 and qa & db:
        x, w = next(iter(qo)), next(iter(do))
        if fuzz.ratio(w, x) < 60 and (x, w) not in KEEP_PAIRS:
            return x, w
    return None


def multi_swap(qa, db, alias, vocab):
    """New business-type word when several words were replaced (e.g. "Maison de Sante X" -> "Amicale X").
    The new word must be a real S1 name word (not typo garble) and not an alias or a variant of a removed word."""
    qo, do = qa - db, db - qa
    if not qo or not (qa & db):
        return None
    for w in sorted(do):
        if w in alias or len(w) < 3 or w not in vocab:
            continue
        if any(fuzz.ratio(w, x) >= 60 or (x, w) in KEEP_PAIRS for x in qo):
            continue
        return w
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work/test")
    ap.add_argument("--train-s1", default="work/train/s1.parquet")
    ap.add_argument("--pred", default="work/test/test_pred.parquet")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--countries", nargs="*", default=None, help="default: countries absent from training")
    ap.add_argument("--alias-share", type=float, default=0.02)
    ap.add_argument("--alias-own", type=float, default=0.10)
    a = ap.parse_args()
    t = {k: pq.read_table(os.path.join(a.work, f"s{k}.parquet"), columns=["id", "ck", "core", "nums"]).to_pandas()
         for k in (1, 2, 3)}
    if a.countries is None:
        seen = set(pq.read_table(a.train_s1, columns=["ck"]).column("ck").unique().to_pylist())
        a.countries = sorted(set(t[1]["ck"].unique()) - seen)
    print("countries:", a.countries)
    mr = pd.read_csv(os.path.join(a.in_dir, "matching_results.tsv"), sep="\t", dtype=str).fillna("")
    row1 = pd.Series(np.arange(len(t[1])), index=t[1]["id"].values)
    rowd = {k: pd.Series(np.arange(len(t[k])), index=t[k]["id"].values) for k in (2, 3)}
    pr = pq.read_table(a.pred, columns=["q", "src", "d"]).to_pandas()
    drop_by = collections.defaultdict(set)
    for ck in a.countries:
        m1 = t[1]["ck"].values == ck
        keys = collections.Counter(" ".join(sorted(words(c))) + "|" + first_num(n)
                                   for c, n in zip(t[1]["core"].values[m1], t[1]["nums"].values[m1]))
        # swap-word statistics over every scored candidate pair of this country
        p = pr[m1[pr["q"].values]]
        cnt = collections.defaultdict(lambda: [0, 0])
        for q, s, d in zip(p["q"].values, p["src"].values, p["d"].values):
            sw = swap(words(t[1]["core"].values[q]), words(t[int(s)]["core"].values[d]))
            if sw:
                db = words(t[int(s)]["core"].values[d])
                c = cnt[sw[1]]
                c[0] += 1
                c[1] += keys.get(" ".join(sorted(db)) + "|" + first_num(t[int(s)]["nums"].values[d]), 0) > 0
        st = pd.DataFrame(cnt, index=["n", "owned"]).T
        st["share"] = st["n"] / st["n"].sum()
        st["own"] = st["owned"] / st["n"]
        alias = set(st[(st["share"] >= a.alias_share) & (st["own"] < a.alias_own)].index)
        print(st[st["share"] >= 0.01].sort_values("n", ascending=False).round(3).head(16).to_string())
        print(f"[{ck}] swap pairs {int(st['n'].sum())}, alias words: "
              f"{st.loc[sorted(alias)][['n', 'own']].round(3).to_dict('index')}")
        vocab = collections.Counter(w for c in t[1]["core"].values[m1] for w in set(c.split()))
        vocab = {w for w, c in vocab.items() if c >= 20}
        sel = mr[(mr["matched_entity_ids"] != "")]
        sel = sel[m1[row1[sel["source1_entity_id"].values].values]]
        nd, n_typo, words_dropped = 0, 0, collections.Counter()
        for s1id, ids in zip(sel["source1_entity_id"].values, sel["matched_entity_ids"].values):
            qa = words(t[1]["core"].values[row1[s1id]])
            for did in ids.split(","):
                k = 2 if did.startswith("S2") else 3
                db = words(t[k]["core"].values[rowd[k][did]])
                sw = swap(qa, db)
                if sw:
                    # the new word must be a real business word of this country, not typo garble ("fbiysr")
                    w = sw[1] if (sw[1] not in alias and sw[1] in vocab) else None
                    if w is None and sw[1] not in alias:
                        n_typo += 1
                else:
                    w = multi_swap(qa, db, alias, vocab)
                if w:
                    drop_by[s1id].add(did)
                    nd += 1
                    words_dropped[w] += 1
        print(f"[{ck}] kept {n_typo} swaps whose new word is not a real name word (typos)")
        print(f"[{ck}] dropping {nd} swap matches from {sum(1 for s in drop_by if m1[row1[s]])} S1 rows "
              f"({nd / m1.sum():.3f} per S1); top words {words_dropped.most_common(12)}")
    idx = mr["source1_entity_id"].isin(drop_by.keys())
    mr.loc[idx, "matched_entity_ids"] = [",".join(x for x in ids.split(",") if x not in drop_by[s])
                                         for s, ids in zip(mr.loc[idx, "source1_entity_id"],
                                                           mr.loc[idx, "matched_entity_ids"])]
    os.makedirs(a.out_dir, exist_ok=True)
    mr.to_csv(os.path.join(a.out_dir, "matching_results.tsv"), sep="\t", index=False, lineterminator="\n")
    print("empty rows:", (mr["matched_entity_ids"] == "").sum())


if __name__ == "__main__":
    main()
