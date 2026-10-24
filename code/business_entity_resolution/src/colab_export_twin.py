"""Twin-aware cross-encoder package: every pair also carries its closest "twin" S1 row.

A twin is another S1 row present in the split (19% simulated deletion on train, as in test) with the
same normalized name in the same country, candidate of the record or not. The closest twin is the one
whose raw name (case, punctuation, legal-form spelling) is most similar to the record's raw name.
colab_ce.py appends its text to the S1 side ("S1 || twin"), so the transformer judges this S1 against
the best alternative explanation of the record (the "compare" strategy of ComEM).

Writes to --out: pairs_train_tw.parquet (q, src, d, label, tw), pairs_val_tw.parquet, pairs_test_tw.parquet
(tw = twin S1 row or -1) and rec_<split>_s1_tw.parquet (twin S1 texts not uploaded before).

Usage:
    python src/colab_export_twin.py --out colab_twin
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colab_export import write  # noqa: E402
from simulate_drop import dropped_mask  # noqa: E402


def raw_names(split, split_dir, k, rows=None):
    ids = pq.read_table(os.path.join(split_dir, f"s{k}.parquet"), columns=["id"]).column("id").to_numpy(zero_copy_only=False)
    r = pd.read_csv(os.path.join("dataset", split, f"{split}_source{k}.tsv"), sep="\t", dtype=str,
                    usecols=["entity_id", "business_name", "business_address"]).fillna("").set_index("entity_id")
    ids = ids if rows is None else ids[rows]
    r = r.reindex(ids).fillna("")
    return r["business_name"].values, r["business_address"].values


def best_twin(df, split, split_dir, drop_frac):
    t1 = pq.read_table(os.path.join(split_dir, "s1.parquet"), columns=["id", "ck", "core"]).to_pandas()
    present = np.ones(len(t1), bool) if drop_frac == 0 else ~dropped_mask(t1["id"].values.astype(object), drop_frac)
    name1, _ = raw_names(split, split_dir, 1)
    dname = np.empty(len(df), dtype=object)
    for k in (2, 3):
        m = df["src"].values == k
        dname[m] = raw_names(split, split_dir, k, df["d"].values[m])[0]
    pres = np.flatnonzero(present)
    gkey = t1["ck"].values[pres] + "|" + t1["core"].values[pres]
    gs = pd.Series(gkey).map(pd.Series(gkey).value_counts()).values
    ok = (gs >= 2) & (gs <= 30)
    grp = pd.DataFrame({"key": gkey[ok], "row": pres[ok]})
    q = df["q"].values
    x = pd.DataFrame({"i": np.arange(len(df)), "key": t1["ck"].values[q] + "|" + t1["core"].values[q], "q": q}).merge(grp, on="key")
    x = x[x["row"].values != x["q"].values].reset_index(drop=True)
    tw = np.full(len(df), -1, np.int64)
    if len(x):
        x["s"] = cpdist(list(dname[x["i"].values]), list(name1[x["row"].values]), scorer=fuzz.ratio, workers=-1, dtype=np.float32)
        best = x.sort_values(["i", "s"], ascending=[True, False]).drop_duplicates("i")
        tw[best["i"].values] = best["row"].values
    return tw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="colab_twin")
    ap.add_argument("--train", default="colab_v10t/pairs_train_v10.parquet")
    ap.add_argument("--val", default="colab_v10t/pairs_val_v10band.parquet")
    ap.add_argument("--test-inus", default="colab_v10/pairs_test_v10band.parquet")
    ap.add_argument("--test-role", default="work10/test/test_predv10ra.parquet", help="France band from the role model")
    ap.add_argument("--have-train", nargs="*", default=["colab_pkg/rec_train_s1.parquet", "colab_big/rec_train_s1_big.parquet",
                                                        "colab_new/rec_train_s1_v8.parquet", "colab_v10/rec_train_s1_v10v.parquet",
                                                        "colab_v10t/rec_train_s1_v10t.parquet"])
    ap.add_argument("--have-test", nargs="*", default=["colab_pkg/rec_test_s1.parquet", "colab_new/rec_test_s1_v8.parquet",
                                                       "colab_fr/rec_test_s1_fr.parquet", "colab_fr2/rec_test_s1_fr2.parquet",
                                                       "colab_v10/rec_test_s1_v10.parquet"])
    ap.add_argument("--n-plain", type=int, default=350_000, help="training pairs without a twin to keep")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    tr = pd.read_parquet(a.train)
    tr["tw"] = best_twin(tr, "train", "work10_sim/train", 0.19)
    has = tr["tw"].values >= 0
    keep = has | (rng.random(len(tr)) < a.n_plain / max(1, (~has).sum()))
    tr = tr[keep].sample(frac=1.0, random_state=1).reset_index(drop=True)
    write(tr, os.path.join(a.out, "pairs_train_tw.parquet"))
    print(f"train pairs {len(tr)} (with twin {int((tr['tw'] >= 0).sum())}, positives {tr['label'].mean():.3f})", flush=True)
    va = pd.read_parquet(a.val)
    va["tw"] = best_twin(va, "train", "work10_sim/train", 0.19)
    write(va, os.path.join(a.out, "pairs_val_tw.parquet"))
    ck = pq.read_table("work10/test/s1.parquet", columns=["ck"]).column("ck").to_numpy(zero_copy_only=False)
    te = pd.read_parquet(a.test_inus, columns=["q", "src", "d"])
    rp = pd.read_parquet(a.test_role, columns=["q", "src", "d", "p"])
    rp = rp[(ck[rp["q"].values] == "fr") & (rp["p"] > 0.005) & (rp["p"] < 0.995)][["q", "src", "d"]]
    te = pd.concat([te, rp], ignore_index=True).drop_duplicates(["q", "src", "d"]).reset_index(drop=True)
    te["tw"] = best_twin(te, "test", "work10/test", 0.0)
    write(te, os.path.join(a.out, "pairs_test_tw.parquet"))
    print(f"val pairs {len(va)} (with twin {int((va['tw'] >= 0).sum())}); test pairs {len(te)} (with twin "
          f"{int((te['tw'] >= 0).sum())})", flush=True)
    for split, frames, have, sdir in (("train", (tr, va), a.have_train, "work10_sim/train"), ("test", (te,), a.have_test, "work10/test")):
        need = set()
        for f in frames:
            need |= set(f["tw"].values[f["tw"].values >= 0].tolist()) | set(f["q"].values.tolist())
        got = set()
        for h in have:
            if os.path.exists(h):
                got |= set(pd.read_parquet(h, columns=["row"])["row"].values.tolist())
        rows = np.array(sorted(need - got), dtype=np.int64)
        nm, ad = raw_names(split, sdir, 1, rows)
        text = np.where(ad != "", nm + " | " + ad, nm)
        write(pd.DataFrame({"row": rows.astype(np.int32), "text": text}), os.path.join(a.out, f"rec_{split}_s1_tw.parquet"))
        print(f"{split}: {len(rows)} new S1 texts", flush=True)


if __name__ == "__main__":
    main()
