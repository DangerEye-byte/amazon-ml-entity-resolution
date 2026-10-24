"""Stage 4: train the LightGBM pair classifier.

Reads work/train/feats/part-*.parquet (from make_features.py). Rows of
validation S1 ids (is_val) are held out for early stopping and for decide.py.
Writes models/lgb.txt, models/features.json and work/train/val_pred.parquet.

Usage:
    python src/train.py [--max-train-rows 12000000] [--rounds 3000]
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

META = ["q", "src", "d", "is_val", "label"]


def load(parts, max_rows, seed=0, neg_rate=1.0):
    tr, va = [], []
    rng = np.random.default_rng(seed)
    for p in parts:
        df = pq.read_table(p).to_pandas()
        t = df[~df["is_val"]]
        if neg_rate < 1.0:
            t = t[(t["label"].values == 1) | (rng.random(len(t)) < neg_rate)]
        tr.append(t)
        va.append(df[df["is_val"]])
        del df
    tr = pd.concat(tr, ignore_index=True)
    va = pd.concat(va, ignore_index=True)
    if max_rows and len(tr) > max_rows:
        # keep every positive, subsample negatives
        pos = tr[tr["label"] == 1]
        neg = tr[tr["label"] == 0].sample(n=max(0, max_rows - len(pos)), random_state=seed)
        tr = pd.concat([pos, neg], ignore_index=True)
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--models", default="models")
    ap.add_argument("--max-train-rows", type=int, default=12_000_000)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--neg-rate", type=float, default=0.5)
    ap.add_argument("--drop", nargs="*", default=[], help="feature-name prefixes to exclude")
    a = ap.parse_args()
    t0 = time.time()
    parts = sorted(glob.glob(os.path.join(a.work, "train", "feats", "part-*.parquet")))
    tr, va = load(parts, a.max_train_rows, neg_rate=a.neg_rate)
    feats = [c for c in tr.columns if c not in META and not any(c.startswith(x) for x in a.drop)]
    print(f"train rows {len(tr)} (pos {tr['label'].sum()}), val rows {len(va)} (pos {va['label'].sum()}), "
          f"{len(feats)} features, loaded in {time.time() - t0:.0f}s", flush=True)
    dtr = lgb.Dataset(tr[feats].values.astype(np.float32), label=tr["label"].values, feature_name=feats,
                      free_raw_data=True)
    dva = lgb.Dataset(va[feats].values.astype(np.float32), label=va["label"].values, reference=dtr)
    params = {
        "objective": "binary", "learning_rate": a.lr, "num_leaves": a.leaves, "min_data_in_leaf": 100,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
        "max_bin": 255, "verbosity": -1, "metric": ["binary_logloss", "auc"], "num_threads": a.threads,
    }
    del tr
    model = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dva], valid_names=["val"],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    os.makedirs(a.models, exist_ok=True)
    model.save_model(os.path.join(a.models, "lgb.txt"))
    with open(os.path.join(a.models, "features.json"), "w") as f:
        json.dump(feats, f)
    va_pred = va[["q", "src", "d", "label"]].copy()
    va_pred["p"] = model.predict(va[feats].values.astype(np.float32), num_threads=a.threads)
    va_pred.to_parquet(os.path.join(a.work, "train", "val_pred.parquet"), index=False)
    imp = sorted(zip(model.feature_importance("gain"), feats), reverse=True)[:30]
    print("top features:", [(f, round(float(g))) for g, f in imp])
    print(f"done in {time.time() - t0:.0f}s, best iter {model.best_iteration}")


if __name__ == "__main__":
    main()
