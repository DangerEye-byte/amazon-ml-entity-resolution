"""Blocking recall of a candidates parquet against training ground truth, at several K.

Usage:
    python src/block_recall.py --cands work/train/cands_probe.parquet
"""
import argparse
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_ground_truth  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--work", default="work")
    ap.add_argument("--gt", default="dataset/train/train_ground_truth.tsv")
    a = ap.parse_args()
    c = pq.read_table(a.cands).to_pandas()
    base = os.path.join(a.work, "train")
    s1 = pq.read_table(os.path.join(base, "s1.parquet"), columns=["id", "ck"])
    ids = {k: pq.read_table(os.path.join(base, f"s{k}.parquet"), columns=["id"]).column("id") for k in (2, 3)}
    c["sid"] = s1.column("id").take(c["q"].values).to_pylist()
    c["ck"] = s1.column("ck").take(c["q"].values).to_pylist()
    did = np.empty(len(c), object)
    for k in (2, 3):
        m = c["src"].values == k
        did[m] = ids[k].take(c["d"].values[m]).to_pylist()
    c["did"] = did
    gt = read_ground_truth(a.gt)
    qs = c["sid"].unique()
    truth = {s: set(x for x in gt[s].split(",") if x) for s in qs}
    c["hit"] = [d in truth[s] for s, d in zip(c["sid"].values, c["did"].values)]
    n_true = sum(len(v) for v in truth.values())
    print(f"S1 rows {len(qs)}, true links {n_true}, pairs {len(c)} ({len(c) / len(qs):.1f}/S1)")
    for k in (1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40):
        sub = c[c["rank"] < k]
        print(f"K={k:3d}/source: recall {sub['hit'].sum() / n_true:.4f}  pairs/S1 {len(sub) / len(qs):.1f}")
    for ck in sorted(c["ck"].unique()):
        cc = c[c["ck"] == ck]
        nt = sum(len(truth[s]) for s in cc["sid"].unique())
        print(f"  {ck}: recall@max {cc['hit'].sum() / nt:.4f}")
    # recall by source
    for k in (2, 3):
        cc = c[c["src"] == k]
        nt = sum(1 for v in truth.values() for x in v if x.startswith(f"S{k}-"))
        print(f"  S{k}: recall@max {cc['hit'].sum() / nt:.4f}")


if __name__ == "__main__":
    main()
