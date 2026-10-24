"""France self-training package for the cross-encoder.

Pseudo-labels (no ground truth exists for France) from the France-safe pipeline:
  positives: French pairs kept in the final France output with word-role probability >= 0.97
  negatives: French pairs with probability <= 0.03, and every pair removed by the French
             descriptor-swap filter (a confirmed decoy pattern)
Mixed with the labelled India/US training pairs. Writes to colab_fr/:
  pairs_train_fr.parquet (split, q, src, d, label), pairs_test_fr.parquet (uncertain French pairs)
and rec_test_s<k>_fr.parquet for any French record text not exported before.
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colab_export import write  # noqa: E402

OUT = "colab_fr"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(11)
ck = pq.read_table("work8/test/s1.parquet", columns=["ck"]).column("ck").to_numpy()
ids = {k: pq.read_table(f"work8/test/s{k}.parquet", columns=["id"]).column("id").to_numpy() for k in (1, 2, 3)}
pr = pd.read_parquet("work8/test/test_predv8ra.parquet")
pr = pr[ck[pr["q"].values] == "fr"].reset_index(drop=True)
did = np.where(pr["src"].values == 2, ids[2][np.minimum(pr["d"].values, len(ids[2]) - 1)],
               ids[3][np.minimum(pr["d"].values, len(ids[3]) - 1)])
pr["sid"], pr["did"] = ids[1][pr["q"].values], did


def pairs_of(path):
    m = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    m = m[m["matched_entity_ids"] != ""]
    return set((s, d) for s, v in zip(m["source1_entity_id"], m["matched_entity_ids"]) for d in v.split(","))


final = pairs_of("submissions/v8ra_fr_swap/matching_results.tsv")
pre = pairs_of("output_v8ra_fr/matching_results.tsv")
key = list(zip(pr["sid"].values, pr["did"].values))
pr["kept"] = [k in final for k in key]
pr["swapped_out"] = [(k in pre) and (k not in final) for k in key]
pos = pr[pr["kept"] & (pr["p"] >= 0.97)]
neg = pd.concat([pr[pr["swapped_out"]], pr[(pr["p"] <= 0.03) & ~pr["kept"]]]).drop_duplicates(["q", "src", "d"])
n = 300_000
pos = pos.sample(min(n, len(pos)), random_state=1)
neg_sw = neg[neg["swapped_out"]]
neg_rest = neg[~neg["swapped_out"]].sample(min(n - len(neg_sw), int((~neg["swapped_out"]).sum())), random_state=2)
fr = pd.concat([pos.assign(label=1), neg_sw.assign(label=0), neg_rest.assign(label=0)])[["q", "src", "d", "label"]]
fr["split"] = "test"
us = pd.read_parquet("colab_pkg/pairs_train.parquet").assign(split="train")
tr = pd.concat([us, fr], ignore_index=True).sample(frac=1.0, random_state=3)
tr["label"] = tr["label"].astype(np.int8)
write(tr[["split", "q", "src", "d", "label"]].reset_index(drop=True), os.path.join(OUT, "pairs_train_fr.parquet"))
print(f"training pairs: India/US {len(us)}, France pseudo {len(fr)} (pos {len(pos)}, swap-filter negatives {len(neg_sw)}, "
      f"low-p negatives {len(neg_rest)})", flush=True)
unc = pr[(pr["p"] > 0.005) & (pr["p"] < 0.995)][["q", "src", "d"]].reset_index(drop=True)
write(unc, os.path.join(OUT, "pairs_test_fr.parquet"))
print(f"uncertain French pairs to score: {len(unc)}", flush=True)
need = {1: set(fr["q"]) | set(unc["q"])}
for k in (2, 3):
    need[k] = set(fr["d"][fr["src"] == k]) | set(unc["d"][unc["src"] == k])
for k in (1, 2, 3):
    have = set(pd.read_parquet(f"colab_pkg/rec_test_s{k}.parquet", columns=["row"])["row"])
    rows = np.array(sorted(need[k] - have), dtype=np.int64)
    raw = pd.read_csv(f"dataset/test/test_source{k}.tsv", sep="\t", dtype=str,
                      usecols=["entity_id", "business_name", "business_address"]).fillna("").set_index("entity_id")
    sel = raw.loc[ids[k][rows]]
    text = np.where(sel["business_address"].values != "", sel["business_name"].values + " | " + sel["business_address"].values,
                    sel["business_name"].values)
    write(pd.DataFrame({"row": rows.astype(np.int32), "text": text}), os.path.join(OUT, f"rec_test_s{k}_fr.parquet"))
    print(f"s{k}: {len(rows)} extra record texts", flush=True)
