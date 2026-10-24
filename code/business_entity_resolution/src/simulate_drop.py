"""Make training look like test: remove a fraction of S1 entities.

Evidence: test has 5.75 S2/S3 records per S1 vs 4.68 in train, and 31% of
test records left unmatched have a near-identical unmatched sibling (orphan
clusters) vs 2.5% of train distractors. The test set was therefore built by
deleting ~19% of S1 entities while keeping all their S2/S3 records. Those
orphaned records are the hardest distractors, and they never occur in train.

Simulation: drop --drop-frac of train S1 rows (deterministic hash). Retrieval
per S1 does not depend on other S1 rows, so candidate lists are unchanged; we
only (1) remove the dropped S1 rows and (2) recompute the reverse-competition
features (r_*) without them. The result is written to <out-work>/train/, which
then goes through the normal prune -> full -> train -> decide stages.

Usage:
    python src/simulate_drop.py --drop-frac 0.19 --out-work work_sim
"""
import argparse
import glob
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import is_val  # noqa: E402
from features import add_reverse_features  # noqa: E402

DROP_SEED = 77


def dropped_mask(ids, frac):
    return is_val(ids, frac=frac, seed=DROP_SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--out-work", default="work_sim")
    ap.add_argument("--drop-frac", type=float, default=0.19)
    ap.add_argument("--cands", default="cands_all.parquet")
    a = ap.parse_args()
    t0 = time.time()
    src_dir = os.path.join(a.work, "train")
    out_dir = os.path.join(a.out_work, "train")
    os.makedirs(os.path.join(out_dir, "bfeats"), exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "bfeats", "part-*.parquet")):
        os.remove(f)  # never mix parts from an earlier run
    for f in ("s1.parquet", "s2.parquet", "s3.parquet"):
        dst = os.path.join(out_dir, f)
        if not os.path.exists(dst):
            try:
                os.link(os.path.join(src_dir, f), dst)  # hard link, no copy
            except OSError:
                shutil.copy(os.path.join(src_dir, f), dst)
    ids = np.array(pq.read_table(os.path.join(src_dir, "s1.parquet"), columns=["id"]).column("id").to_pylist(),
                   dtype=object)
    drop = dropped_mask(ids, a.drop_frac)
    print(f"dropping {drop.sum()} of {len(ids)} S1 rows ({drop.mean():.3f})", flush=True)
    c = pq.read_table(os.path.join(src_dir, a.cands), columns=["q", "src", "d", "score"]).to_pandas()
    c = c[~drop[c["q"].values]].reset_index(drop=True)
    rrank, rcnt, rbest = add_reverse_features(c)
    key = lambda q, s, d: q.astype(np.int64) * 8 * (1 << 32) + s.astype(np.int64) * (1 << 32) + d.astype(np.int64)  # noqa
    k_all = key(c["q"].values, c["src"].values, c["d"].values)
    order = np.argsort(k_all)
    k_sorted = k_all[order]
    score_sorted = c["score"].values[order]
    rr, rc, rb = rrank[order], rcnt[order], rbest[order]
    del c, rrank, rcnt, rbest, k_all, order
    print(f"reverse features recomputed ({time.time() - t0:.0f}s)", flush=True)
    for p in sorted(glob.glob(os.path.join(src_dir, "bfeats", "part-*.parquet"))):
        df = pq.read_table(p).to_pandas()
        df = df[~drop[df["q"].values]].reset_index(drop=True)
        k = key(df["q"].values, df["src"].values, df["d"].values)
        pos = np.searchsorted(k_sorted, k)
        assert np.all(k_sorted[pos] == k)
        df["r_rank"], df["r_cnt"], df["r_best_other"] = rr[pos], rc[pos], rb[pos]
        df["r_score_ratio"] = score_sorted[pos] / np.maximum(rb[pos], 1e-3)
        df.to_parquet(os.path.join(out_dir, "bfeats", os.path.basename(p)), index=False)
    print(f"wrote {out_dir}/bfeats ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
