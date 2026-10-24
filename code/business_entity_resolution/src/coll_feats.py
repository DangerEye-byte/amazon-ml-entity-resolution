"""Collective / density / raw-name twin features for every candidate pair of a split (see stack_ce.collective),
saved for stage-2 training (train_lex.py --extra) and scoring (score_lex.py --extra).

Usage:
    python src/coll_feats.py --split-dir work10_sim/train --drop-frac 0.19
    python src/coll_feats.py --split-dir work10/test --drop-frac 0
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stack_ce import collective  # noqa: E402

PAIRWISE = ["nm_tset", "ad_tset", "num1_eq", "d_oov_frac", "r_cnt", "abbr_ratio"]  # already in the feature files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--drop-frac", type=float, default=0.0)
    ap.add_argument("--out", default="coll.parquet")
    ap.add_argument("--ref-sizes", default=None, help='JSON {"s1": {ck: n}, "rec": {ck: n}}: target population per country')
    a = ap.parse_args()
    t0 = time.time()
    parts = sorted(glob.glob(os.path.join(a.split_dir, "feats", "part-*.parquet")))
    df = pd.concat([pq.read_table(p, columns=["q", "src", "d"]).to_pandas() for p in parts], ignore_index=True)
    df = df.drop_duplicates(["q", "src", "d"]).reset_index(drop=True)
    print(f"{len(df)} pairs ({time.time() - t0:.0f}s)", flush=True)
    import json
    df, cols = collective(df, a.split_dir, a.drop_frac, ref_sizes=json.loads(a.ref_sizes) if a.ref_sizes else None)
    keep = [c for c in cols if c not in PAIRWISE]
    out = df[["q", "src", "d"] + keep].copy()
    for c in keep:
        out[c] = out[c].astype(np.float32)
    out.to_parquet(os.path.join(a.split_dir, a.out), index=False)
    print(f"wrote {a.split_dir}/{a.out}: {len(keep)} features ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
