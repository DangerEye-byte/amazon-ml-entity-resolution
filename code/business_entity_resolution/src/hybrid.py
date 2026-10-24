"""Combine two submissions by country: countries seen in training take rows from --seen,
countries absent from training take rows from --unseen (models with lexical features rely on
word statistics that do not exist for a new language).

Usage:
    python src/hybrid.py --seen submissions/v7s7_swap --unseen submissions/v7_swap --out submissions/v7h
"""
import argparse
import os

import pandas as pd
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seen", required=True)
    ap.add_argument("--unseen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-s1", default="work/train/s1.parquet")
    ap.add_argument("--test-s1", default="work/test/s1.parquet")
    a = ap.parse_args()
    seen_ck = set(pq.read_table(a.train_s1, columns=["ck"]).column("ck").unique().to_pylist())
    s1 = pq.read_table(a.test_s1, columns=["id", "ck"]).to_pandas()
    unseen_ids = set(s1["id"][~s1["ck"].isin(seen_ck)])
    x = pd.read_csv(os.path.join(a.seen, "matching_results.tsv"), sep="\t", dtype=str).fillna("")
    y = pd.read_csv(os.path.join(a.unseen, "matching_results.tsv"), sep="\t", dtype=str).fillna("").set_index("source1_entity_id")
    m = x["source1_entity_id"].isin(unseen_ids)
    x.loc[m, "matched_entity_ids"] = y.loc[x.loc[m, "source1_entity_id"], "matched_entity_ids"].values
    os.makedirs(a.out, exist_ok=True)
    x.to_csv(os.path.join(a.out, "matching_results.tsv"), sep="\t", index=False, lineterminator="\n")
    # candidate rows follow the same source as the matches, so every match stays inside its candidate list
    cx = pd.read_csv(os.path.join(a.seen, "candidate_pairs.tsv"), sep="\t", dtype=str).fillna("")
    cy = pd.read_csv(os.path.join(a.unseen, "candidate_pairs.tsv"), sep="\t", dtype=str).fillna("")
    cy = cy.set_index("source1_entity_id")
    mc = cx["source1_entity_id"].isin(unseen_ids)
    cx.loc[mc, "candidate_entity_ids"] = cy.loc[cx.loc[mc, "source1_entity_id"], "candidate_entity_ids"].values
    cx.to_csv(os.path.join(a.out, "candidate_pairs.tsv"), sep="\t", index=False, lineterminator="\n")
    print(f"{m.sum()} rows from {a.unseen} (unseen countries), {len(x) - m.sum()} rows from {a.seen}")


if __name__ == "__main__":
    main()
