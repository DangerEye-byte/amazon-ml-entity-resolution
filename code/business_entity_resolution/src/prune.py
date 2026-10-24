"""Stage-1 learned blocking filter (supervised meta-blocking).

A small LightGBM model on cheap blocking-level features (token overlaps,
retrieval score/rank, reverse competition, two fast similarities) estimates
p1 = P(match) for every retrieved candidate. Candidates with p1 below a
threshold chosen on validation (target: lose at most --max-loss of the
retrieved true links) are dropped. The survivors are the final candidate set,
the exact set the stage-2 matcher runs inference over.

Train mode: 2-fold cross-fitting gives out-of-fold p1 on training rows, so the
stage-2 training data matches what the filter produces at test time.

Usage:
    python src/prune.py --mode train
    python src/prune.py --mode test
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
PARAMS = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 200,
          "feature_fraction": 0.9, "bagging_fraction": 0.7, "bagging_freq": 1, "verbosity": -1,
          "num_threads": 0}


def fold_of(q):
    # deterministic 2-fold split by S1 row (multiplicative hash, vectorised)
    return ((q.astype(np.uint64) * np.uint64(2654435761)) >> np.uint64(7) & np.uint64(1)).astype(np.int8)


def fit(X, y, rounds=400):
    return lgb.train(PARAMS, lgb.Dataset(X, label=y), num_boost_round=rounds)


def write_pruned(parts, p1_list, thr, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "part-*.parquet")):
        os.remove(f)
    kept = total = 0
    for p, p1 in zip(parts, p1_list):
        df = pq.read_table(p).to_pandas()
        df["p1"] = p1.astype(np.float32)
        df = df[df["p1"].values >= thr]
        kept += len(df)
        df.to_parquet(os.path.join(out_dir, os.path.basename(p)), index=False)
        total += len(p1)
    return kept, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["train", "test", "apply"],
                    help="apply = run a trained pruner on the train split (evaluation only)")
    ap.add_argument("--work", default="work")
    ap.add_argument("--models", default="models")
    ap.add_argument("--max-loss", type=float, default=0.002,
                    help="max fraction of retrieved true links the filter may drop (validation)")
    ap.add_argument("--neg-rate", type=float, default=0.3)
    ap.add_argument("--drop", nargs="*", default=[], help="feature-name prefixes to exclude")
    a = ap.parse_args()
    t0 = time.time()
    split = "test" if a.mode == "test" else "train"
    parts = sorted(glob.glob(os.path.join(a.work, split, "bfeats", "part-*.parquet")))
    out_dir = os.path.join(a.work, split, "pruned")
    if a.mode == "train":
        # pass 1: collect a float32 training sample (all positives + a fraction of negatives)
        feats = None
        Xs, ys, fs = [], [], []
        rng = np.random.default_rng(0)
        for p in parts:
            df = pq.read_table(p).to_pandas()
            if feats is None:
                feats = [c for c in df.columns if c not in META and not any(c.startswith(x) for x in a.drop)]
            m = (~df["is_val"].values) & ((df["label"].values == 1) | (rng.random(len(df)) < a.neg_rate))
            d = df[m]
            Xs.append(np.column_stack([d[c].values.astype(np.float32) for c in feats]))
            ys.append(d["label"].values)
            fs.append(fold_of(d["q"].values))
            del df, d
        X, y, fold = np.concatenate(Xs), np.concatenate(ys), np.concatenate(fs)
        del Xs, ys, fs
        print(f"pruner sample: {len(y)} rows ({y.sum()} pos), {len(feats)} features", flush=True)
        models = {}
        for f in (0, 1):
            models[f] = fit(X[fold != f], y[fold != f])
            print(f"fold {f} done ({time.time() - t0:.0f}s)", flush=True)
        model = fit(X, y)
        del X, y, fold
        # pass 2: out-of-fold p1 for training rows, full-model p1 for validation rows
        p1_list, pv, yv, nq = [], [], [], set()
        for p in parts:
            df = pq.read_table(p, columns=feats + ["q", "is_val", "label"]).to_pandas()
            X = np.column_stack([df[c].values.astype(np.float32) for c in feats])
            isv = df["is_val"].values
            fo = fold_of(df["q"].values)
            p1 = np.zeros(len(df), np.float32)
            for f in (0, 1):
                m = ~isv & (fo == f)
                if m.any():
                    p1[m] = models[f].predict(X[m])  # model trained on the other fold
            if isv.any():
                p1[isv] = model.predict(X[isv])
                pv.append(p1[isv])
                yv.append(df["label"].values[isv])
                nq.update(df["q"].values[isv].tolist())
            p1_list.append(p1)
            del df, X
        pv, yv, nq = np.concatenate(pv), np.concatenate(yv), len(nq)
        pos = np.sort(pv[yv == 1])
        thr = float(pos[int(a.max_loss * len(pos))]) if len(pos) else 0.0
        for t in sorted({thr, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1}):
            keep = pv >= t
            print(f"  thr {t:.4f}: keep {keep.sum() / nq:.2f} cand/S1, "
                  f"retrieved-positive recall {(keep & (yv == 1)).sum() / max(1, (yv == 1).sum()):.4f}")
        os.makedirs(a.models, exist_ok=True)
        model.save_model(os.path.join(a.models, "prune.txt"))
        with open(os.path.join(a.models, "prune.json"), "w") as f:
            json.dump({"features": feats, "threshold": thr}, f)
        kept, total = write_pruned(parts, p1_list, thr, out_dir)
    else:
        model = lgb.Booster(model_file=os.path.join(a.models, "prune.txt"))
        with open(os.path.join(a.models, "prune.json")) as f:
            cfg = json.load(f)
        feats, thr = cfg["features"], cfg["threshold"]
        p1_list = []
        for p in parts:
            df = pq.read_table(p, columns=feats).to_pandas()
            p1_list.append(model.predict(np.column_stack([df[c].values.astype(np.float32) for c in feats])))
        kept, total = write_pruned(parts, p1_list, thr, out_dir)
    print(f"threshold {thr:.5f}: kept {kept}/{total} candidates ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
