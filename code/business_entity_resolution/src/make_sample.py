"""Small, structurally faithful sample of the challenge data for the end-to-end smoke test (smoke_test.sh).

Train: a hash sample of S1 rows with all of their true matches, plus a hash sample of every other S2/S3 record
(records owned by S1 rows outside the sample then behave like the orphans of deleted S1 rows in test).
Test: a hash sample of S1 rows with their candidate records from --test-cands (a previous candidate_pairs.tsv, so the
sample keeps true matches and near-miss decoys together), plus a hash sample of every other record.

Usage:
    python src/make_sample.py --data dataset --out /tmp/smoke/dataset --frac 0.015
"""
import argparse
import hashlib
import os

import pandas as pd


def keep(ids, frac, seed):
    thr = int(frac * 2 ** 64)
    key = f"{seed}".encode()
    return pd.Series([int.from_bytes(hashlib.blake2b(str(x).encode(), digest_size=8, key=key).digest(), "little") < thr
                      for x in ids], index=ids.index)


def read(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig")


def write(df, path):
    df.to_csv(path, sep="\t", index=False, lineterminator="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.015, help="share of S1 rows kept")
    ap.add_argument("--rec-frac", type=float, default=0.015, help="share of the remaining S2/S3 records kept")
    ap.add_argument("--test-cands", default="output/candidate_pairs.tsv")
    a = ap.parse_args()
    for split in ("train", "test"):
        os.makedirs(os.path.join(a.out, split), exist_ok=True)
        s1 = read(os.path.join(a.data, split, f"{split}_source1.tsv"))
        s1 = s1[keep(s1["entity_id"], a.frac, 1).values]
        linked = set()
        if split == "train":
            gt = read(os.path.join(a.data, "train", "train_ground_truth.tsv"))
            gt = gt[gt["source1_entity_id"].isin(set(s1["entity_id"]))]
            linked = set(x for v in gt["matched_entity_ids"] for x in v.split(",") if x)
            write(gt, os.path.join(a.out, "train", "train_ground_truth.tsv"))
        elif os.path.exists(a.test_cands):
            c = read(a.test_cands)
            c = c[c["source1_entity_id"].isin(set(s1["entity_id"]))]
            linked = set(x for v in c["candidate_entity_ids"] for x in v.split(",") if x)
        write(s1, os.path.join(a.out, split, f"{split}_source1.tsv"))
        n = [len(s1)]
        for k in (2, 3):
            r = read(os.path.join(a.data, split, f"{split}_source{k}.tsv"))
            r = r[r["entity_id"].isin(linked).values | keep(r["entity_id"], a.rec_frac, 2).values]
            write(r, os.path.join(a.out, split, f"{split}_source{k}.tsv"))
            n.append(len(r))
        print(f"{split}: S1 {n[0]}, S2 {n[1]}, S3 {n[2]}", flush=True)


if __name__ == "__main__":
    main()
