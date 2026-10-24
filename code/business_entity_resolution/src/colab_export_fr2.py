"""France self-training, round 2: pseudo-labels from the stacked France model (role model +
round-1 French transformer, submissions/v8rafr_swap), keeping only pairs both views accept.

  positives: kept in the final France output, stacked probability >= --pos, and neither the role
             model nor the round-1 transformer below --veto
  negatives: stacked probability <= --neg (not kept) with neither view above 1 - --veto, and every
             pair removed by the descriptor-swap filter
Mixed with the labelled India/US training pairs. Writes colab_fr2/pairs_train_fr2.parquet and
rec_test_s<k>_fr2.parquet (record texts not exported before). Score with pairs_test_fr.parquet.

Usage:
    python src/colab_export_fr2.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colab_export import write  # noqa: E402

KEY = lambda df: df["q"].values.astype(np.int64) * 8 * (1 << 32) + df["src"].values.astype(np.int64) * (1 << 32) + df["d"].values.astype(np.int64)  # noqa


def pairs_of(path, ids):
    m = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    m = m[m["matched_entity_ids"] != ""]
    s = m.set_index("source1_entity_id")["matched_entity_ids"].str.split(",").explode()
    q = pd.Series(np.arange(len(ids[1])), index=ids[1]).reindex(s.index).values
    src = np.where(s.str[1].values == "2", 2, 3)
    d = np.full(len(s), -1, np.int64)
    for k in (2, 3):
        mk = src == k
        d[mk] = pd.Index(ids[k]).get_indexer(s.values[mk])
    return KEY(pd.DataFrame({"q": q, "src": src, "d": d}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="colab_fr2")
    ap.add_argument("--stack", default="work8/test/test_predv8rafr.parquet")
    ap.add_argument("--role", default="work8/test/test_predv8ra.parquet")
    ap.add_argument("--ce", default="colab_pkg/out_fr/ce_testband.parquet")
    ap.add_argument("--final", default="submissions/v8rafr_swap/matching_results.tsv")
    ap.add_argument("--pre", default="output_v8rafr/matching_results.tsv")
    ap.add_argument("--pos", type=float, default=0.97)
    ap.add_argument("--neg", type=float, default=0.03)
    ap.add_argument("--veto", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=350_000)
    ap.add_argument("--train-name", default="pairs_train_fr2.parquet")
    ap.add_argument("--tag", default="fr2", help="suffix of the record-text files")
    ap.add_argument("--band-out", default=None, help="also write the France band (lo < role p < hi) for scoring")
    ap.add_argument("--have", nargs="*", default=["colab_pkg/rec_test_s{k}.parquet", "colab_fr/rec_test_s{k}_fr.parquet"],
                    help="record-text files already uploaded ({k} = source)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ck = pq.read_table("work8/test/s1.parquet", columns=["ck"]).column("ck").to_numpy(zero_copy_only=False)
    ids = {k: pq.read_table(f"work8/test/s{k}.parquet", columns=["id"]).column("id").to_numpy(zero_copy_only=False)
           for k in (1, 2, 3)}
    pr = pd.read_parquet(a.stack, columns=["q", "src", "d", "p"])
    pr = pr[ck[pr["q"].values] == "fr"].reset_index(drop=True)
    k = KEY(pr)
    role = pd.read_parquet(a.role, columns=["q", "src", "d", "p"])
    role = role[ck[role["q"].values] == "fr"]
    pr["role"] = pd.Series(role["p"].values, index=KEY(role)).reindex(k).values
    ce = pd.read_parquet(a.ce)
    pr["ce"] = pd.Series(ce["ce"].values, index=KEY(ce)).reindex(k).values   # NaN outside the scored band
    final, pre = pairs_of(a.final, ids), pairs_of(a.pre, ids)
    pr["kept"] = np.isin(k, final)
    pr["swapped_out"] = np.isin(k, pre) & ~pr["kept"].values
    lo_ok = ~(pr["role"].values < a.veto) & ~(pr["ce"].values < a.veto)          # NaN compares False
    hi_ok = ~(pr["role"].values > 1 - a.veto) & ~(pr["ce"].values > 1 - a.veto)
    pos = pr[pr["kept"].values & (pr["p"].values >= a.pos) & lo_ok]
    neg_sw = pr[pr["swapped_out"].values]
    neg = pr[~pr["kept"].values & ~pr["swapped_out"].values & (pr["p"].values <= a.neg) & hi_ok]
    rng = np.random.default_rng(21)
    pos = pos.iloc[rng.permutation(len(pos))[:a.n]]
    neg = neg.iloc[rng.permutation(len(neg))[:max(0, a.n - len(neg_sw))]]
    fr = pd.concat([pos.assign(label=1), neg_sw.assign(label=0), neg.assign(label=0)])[["q", "src", "d", "label"]]
    fr["split"] = "test"
    us = pd.read_parquet("colab_pkg/pairs_train.parquet").assign(split="train")
    tr = pd.concat([us, fr], ignore_index=True).sample(frac=1.0, random_state=23)
    tr["label"] = tr["label"].astype(np.int8)
    write(tr[["split", "q", "src", "d", "label"]].reset_index(drop=True), os.path.join(a.out, a.train_name))
    band = None
    if a.band_out:
        band = role[(role["p"] > 0.005) & (role["p"] < 0.995)][["q", "src", "d"]].reset_index(drop=True)
        write(band, os.path.join(a.out, a.band_out))
        print(f"France band to score: {len(band)} pairs", flush=True)
    n_vetoed = int((pr["kept"].values & (pr["p"].values >= a.pos) & ~lo_ok).sum())
    print(f"training pairs: India/US {len(us)}, France pseudo {len(fr)} (pos {len(pos)}, swap-filter negatives "
          f"{len(neg_sw)}, low-p negatives {len(neg)}; {n_vetoed} confident positives vetoed by one view)", flush=True)
    allp = fr if band is None else pd.concat([fr[["q", "src", "d"]], band])
    for kk in (1, 2, 3):
        need = set(allp["q"]) if kk == 1 else set(allp["d"][allp["src"] == kk])
        have = set()
        for h in a.have:
            f = h.format(k=kk)
            if os.path.exists(f):
                have |= set(pd.read_parquet(f, columns=["row"])["row"])
        rows = np.array(sorted(need - have), dtype=np.int64)
        raw = pd.read_csv(f"dataset/test/test_source{kk}.tsv", sep="\t", dtype=str,
                          usecols=["entity_id", "business_name", "business_address"]).fillna("").set_index("entity_id")
        sel = raw.loc[ids[kk][rows]]
        text = np.where(sel["business_address"].values != "",
                        sel["business_name"].values + " | " + sel["business_address"].values, sel["business_name"].values)
        write(pd.DataFrame({"row": rows.astype(np.int32), "text": text}), os.path.join(a.out, f"rec_test_s{kk}_{a.tag}.parquet"))
        print(f"s{kk}: {len(rows)} extra record texts", flush=True)


if __name__ == "__main__":
    main()
