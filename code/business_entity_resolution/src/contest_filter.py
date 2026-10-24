"""Evidence check for contested records (records that two or more S1 rows claim).

A record can belong to only one S1 row. When several S1 rows claim it with a high probability, the decision
rule's pick between them is often a coin flip; the record itself usually says which S1 it belongs to or that
it cannot tell. For each claimant X and rival Y, the record votes on:
    house number shared with X but not Y (weight 2), initials of X's name (2), address words shared (fuzzy,
    order-free; 2), same legal form (1), closer name by character trigrams (1).
X beats Y when the weighted vote is positive; X keeps (or regains) the record only if it beats every rival.

--mode drop     (unseen countries, e.g. France): drop selected contested matches whose S1 does not beat every
                rival with p >= --rival-p.
--mode restore  (India/US): add back a record's top claimant with p >= --restore-p that soft exclusivity
                dropped, when it beats every rival with p >= --rival-p (at least one such rival) and no other S1 holds the record.

Usage:
    python src/contest_filter.py --mode drop --countries fr --pred work10/test/test_predv10rafr123.parquet \
        --in-dir submissions/v13ofr_swap --out-dir submissions/v15fr
    python src/contest_filter.py --mode restore --countries in us --pred work10/test/test_predv11inus.parquet \
        --in-dir output_v11inus --out-dir output_v15inus --rival-p 0.3
"""
import argparse
import os
import re
import shutil
import unicodedata

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def tri(s):
    s = f"  {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def jac(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def initials(core):
    return "".join(w[0] for w in core.split() if w)


def contain(rwords, xwords):
    """Share of the record's address words that appear in the S1 address (fuzzy: trigram Jaccard >= 0.4)."""
    if not rwords:
        return 0.0
    xt = [tri(w) for w in xwords]
    hit = 0
    for w in rwords:
        if w in xwords or any(jac(tri(w), t) >= 0.4 for t in xt):
            hit += 1
    return hit / len(rwords)


NUM_RE = re.compile(r"(?<![a-z0-9])(\d+)\s*(bis|ter|quater|[a-d](?![a-z]))?")


def raw_nums(addr):
    """House numbers with their bis/ter/letter suffix from a raw address ("56 bis" -> "56bis", "N° 56" -> "56")."""
    x = unicodedata.normalize("NFKD", str(addr)).encode("ascii", "ignore").decode().lower()
    return {n.lstrip("0") + (sfx or "") for n, sfx in NUM_RE.findall(x)} - {""}


def vote(R, X, Y):
    """Weighted evidence of record R for S1 X over S1 Y (> 0: X, < 0: Y)."""
    s = 0
    rn = R["rnums"] if "rnums" in R else set(R["nums"].split())
    if rn:
        xs = X["rnums"] if "rnums" in X else set(X["nums"].split())
        ys = Y["rnums"] if "rnums" in Y else set(Y["nums"].split())
        xn, yn = bool(rn & xs), bool(rn & ys)
        s += 2 * (int(xn) - int(yn))
    ix, iy, rc = initials(X["core"]), initials(Y["core"]), R["compact"]
    xi, yi = len(ix) >= 2 and rc.startswith(ix), len(iy) >= 2 and rc.startswith(iy)
    s += 2 * (int(xi) - int(yi))
    rw = set(R["awords"].split())
    if rw:  # distinctive address words: found (fuzzily) in one claimant's address but not the other's
        xw, yw = set(X["awords"].split()), set(Y["awords"].split())
        nx = sum(1 for w in rw if contain({w}, xw) > 0 and contain({w}, yw) == 0)
        ny = sum(1 for w in rw if contain({w}, yw) > 0 and contain({w}, xw) == 0)
        s += 2 * (int(nx > ny) - int(ny > nx))
    if R["legal"]:
        s += int(X["legal"] == R["legal"]) - int(Y["legal"] == R["legal"])
    jx, jy = jac(tri(rc), tri(X["compact"])), jac(tri(rc), tri(Y["compact"]))
    s += int(jx >= jy + 0.1) - int(jy >= jx + 0.1)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["drop", "restore"])
    ap.add_argument("--work", default="work10/test")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--countries", nargs="+", default=["fr"])
    ap.add_argument("--rival-p", type=float, default=0.5)
    ap.add_argument("--restore-p", type=float, default=0.75)
    ap.add_argument("--data-dir", default="dataset/test", help="raw TSVs: house-number suffixes (bis/ter/a-d)")
    a = ap.parse_args()
    cols = ["id", "ck", "core", "compact", "legal", "awords", "nums"]
    t = {k: pq.read_table(os.path.join(a.work, f"s{k}.parquet"), columns=cols).to_pandas() for k in (1, 2, 3)}
    for k in (1, 2, 3):  # raw house numbers with suffixes, for the rows of the selected countries only
        raw = pd.read_csv(os.path.join(a.data_dir, f"test_source{k}.tsv"), sep="	", dtype=str, quoting=3,
                          usecols=["entity_id", "business_address"]).set_index("entity_id")["business_address"]
        if k == 1:
            t[k]["rnums"] = [raw_nums(x) for x in raw.reindex(t[k]["id"].values).values]
        else:
            t[k]["_raw"] = raw.reindex(t[k]["id"].values).values
    s1 = t[1]
    ck = s1["ck"].values
    d = pd.read_parquet(a.pred, columns=["q", "src", "d", "p"])
    d = d[np.isin(ck[d["q"].values], a.countries)].reset_index(drop=True)
    d["sid"] = s1["id"].values[d["q"].values]
    d["did"] = np.where(d["src"].values == 2, t[2]["id"].values[np.where(d["src"].values == 2, d["d"].values, 0)],
                        t[3]["id"].values[np.where(d["src"].values == 3, d["d"].values, 0)])
    d["key"] = d["src"].values.astype(np.int64) * (1 << 32) + d["d"].values
    m_in = pd.read_csv(os.path.join(a.in_dir, "matching_results.tsv"), sep="\t", dtype=str).fillna("")
    ids_c = set(s1["id"].values[np.isin(ck, a.countries)])
    sel = set((s, x) for s, v in zip(m_in["source1_entity_id"], m_in["matched_entity_ids"])
              for x in v.split(",") if x and s in ids_c)
    d["sel"] = [(s, r) in sel for s, r in zip(d["sid"].values, d["did"].values)]
    strong = d[d["p"].values >= a.rival_p]
    riv = {}

    def beats_all(row, need_rival=False):
        R = t[int(row.src)].iloc[int(row.d)].copy()
        R["rnums"] = raw_nums(R["_raw"])
        X = s1.iloc[int(row.q)]
        others = [(yq, yp) for yq, yp in riv.get(row.key, []) if int(yq) != int(row.q)]
        if need_rival and not others:
            return False
        for yq, yp in others:
            if int(yq) == int(row.q):
                continue
            if vote(R, X, s1.iloc[int(yq)]) <= 0:
                return False
        return True

    final = set(sel)
    if a.mode == "drop":
        n_claim = strong.groupby("key").size()
        cand = d[d["sel"].values & (d["key"].map(n_claim).fillna(0).values >= 2)]
        riv.update({k: g[["q", "p"]].values for k, g in strong[strong["key"].isin(set(cand["key"]))].groupby("key")})
        drop = [(r.sid, r.did) for r in cand.itertuples(index=False) if not beats_all(r)]
        final -= set(drop)
        print(f"{a.countries}: selected {len(sel)}, contested {len(cand)}, dropped {len(drop)}", flush=True)
    else:
        mx = d.groupby("key")["p"].transform("max").values
        held = set(d.loc[d["sel"].values, "key"])
        cand = d[(d["p"].values >= a.restore_p) & (d["p"].values >= mx) & ~d["sel"].values & ~d["key"].isin(held).values]
        riv.update({k: g[["q", "p"]].values for k, g in strong[strong["key"].isin(set(cand["key"]))].groupby("key")})
        add = [(r.sid, r.did) for r in cand.itertuples(index=False) if beats_all(r, need_rival=True)]
        final |= set(add)
        print(f"{a.countries}: selected {len(sel)}, restore candidates {len(cand)}, restored {len(add)}", flush=True)
    by = {}
    for s, r in final:
        by.setdefault(s, []).append(r)
    out = m_in.copy()
    isc = out["source1_entity_id"].isin(ids_c)
    out.loc[isc, "matched_entity_ids"] = [",".join(sorted(by.get(s, []))) for s in out.loc[isc, "source1_entity_id"]]
    os.makedirs(a.out_dir, exist_ok=True)
    out.to_csv(os.path.join(a.out_dir, "matching_results.tsv"), sep="\t", index=False)
    shutil.copy(os.path.join(a.in_dir, "candidate_pairs.tsv"), os.path.join(a.out_dir, "candidate_pairs.tsv"))
    print(f"wrote {a.out_dir}: matches {len(sel)} -> {len(final)}")


if __name__ == "__main__":
    main()
