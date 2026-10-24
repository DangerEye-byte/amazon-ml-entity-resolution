"""Stage 4b: score test feature parts with the trained model -> work/test/test_pred.parquet.

Usage:
    python src/predict.py
"""
import argparse
import glob
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--models", default="models")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    t0 = time.time()
    model = lgb.Booster(model_file=os.path.join(a.models, "lgb.txt"))
    with open(os.path.join(a.models, "features.json")) as f:
        feats = json.load(f)
    outs = []
    parts = sorted(glob.glob(os.path.join(a.work, a.split, "feats", "part-*.parquet")))
    for i, p in enumerate(parts):
        df = pq.read_table(p).to_pandas()
        o = df[["q", "src", "d"]].copy()
        o["p"] = model.predict(df[feats].values.astype(np.float32))
        outs.append(o)
        if i % 20 == 0:
            print(f"{i + 1}/{len(parts)} parts, {time.time() - t0:.0f}s", flush=True)
    out = pd.concat(outs, ignore_index=True)
    out.to_parquet(os.path.join(a.work, a.split, f"{a.split}_pred.parquet"), index=False)
    print(f"scored {len(out)} pairs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
