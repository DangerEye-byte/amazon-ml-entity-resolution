"""Stage 2 with lexical features: retrain the pair matcher on feats + word-level features.

Lexical features (lexical.py) are target encodings of the words / legal-form
changes / number-change types that differ between the two names. For training
rows they are computed out-of-fold (2 folds by S1 row), so a row never sees
its own label; validation and test rows use an encoder fit on all training
rows. Writes <models>/lgb.txt, features.json, <work>/train/val_pred<tag>.parquet
and <test-work>/test/test_pred<tag>.parquet.

Usage:
    python src/train_lex.py --work work_sim --test-work work --models models_v7lx --tag lx
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lexical  # noqa: E402

META = ["q", "src", "d", "is_val", "label"]


def fold_of(q):
    return np.array([hashlib.blake2b(str(int(x)).encode(), digest_size=2).digest()[0] & 1 for x in q])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work_sim")
    ap.add_argument("--test-work", default="work")
    ap.add_argument("--models", default="models_v7lx")
    ap.add_argument("--tag", default="lx")
    ap.add_argument("--base-models", default="models_v7", help="feature list of the stage-2 model to extend")
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decoy-weight", type=float, default=1.0,
                    help="weight of negatives whose record belongs to no S1 (test has ~2x more such decoys)")
    ap.add_argument("--no-test", action="store_true", help="skip test scoring (use score_lex.py)")
    ap.add_argument("--extra", default=None, help="parquet of extra per-pair features (q, src, d, ...) to add")
    ap.add_argument("--role", action="store_true",
                    help="add language-independent word-role statistics (lexical.role_stats)")
    ap.add_argument("--agnostic", action="store_true",
                    help="only language-independent lexical features (usable for countries unseen in training)")
    a = ap.parse_args()
    t0 = time.time()
    base = json.load(open(os.path.join(a.base_models, "features.json")))
    parts = sorted(glob.glob(os.path.join(a.work, "train", "feats", "part-*.parquet")))
    df = pd.concat([pq.read_table(p, columns=META + base).to_pandas() for p in parts], ignore_index=True)
    if a.extra:
        ex = pd.read_parquet(a.extra)
        xcols = [c for c in ex.columns if c not in ("q", "src", "d")]
        df = df.merge(ex, on=["q", "src", "d"], how="left")
        base = base + xcols
        print(f"extra features: {len(xcols)}", flush=True)
    print(f"loaded {len(df)} rows ({time.time() - t0:.0f}s)", flush=True)
    s, voc = lexical.load_split(os.path.join(a.work, "train"))
    diffs = lexical.pair_diffs(df, s, voc)
    if a.role:
        ck1 = s[1]["ck"].values
        stats = lexical.role_stats(df[["q", "src", "d"]], s, lexical.s1_vocab_counts(s), ck1)
        rf = lexical.role_features(diffs, ck1[df["q"].values], stats)
        for c in lexical.ROLE_COLS:
            df[c] = rf[c].values
        print(f"role features ({time.time() - t0:.0f}s)", flush=True)
    del s
    print(f"diffs ({time.time() - t0:.0f}s)", flush=True)
    tr = ~df["is_val"].values
    y = df["label"].values.astype(float)
    fold = fold_of(df["q"].values)
    tr_idx = np.flatnonzero(tr)
    va_idx = np.flatnonzero(~tr)
    # validation rows: encoder fit on all training rows
    enc_all = lexical.fit_encoder([diffs[i] for i in tr_idx], y[tr_idx])
    fv = lexical.features([diffs[i] for i in va_idx], enc_all)
    LX = list(lexical.AGNOSTIC) if a.agnostic else list(fv.columns)
    fv = fv[LX]
    lxm = np.zeros((len(df), len(LX)), np.float32)
    lxm[va_idx] = fv.values.astype(np.float32)
    # training rows: out-of-fold encoding (2 folds by S1 row)
    for f in (0, 1):
        fit = tr_idx[fold[tr_idx] != f]
        app = tr_idx[fold[tr_idx] == f]
        enc = lexical.fit_encoder([diffs[i] for i in fit], y[fit])
        lxm[app] = lexical.features([diffs[i] for i in app], enc)[LX].values.astype(np.float32)
    del diffs
    for j, c in enumerate(LX):
        df[c] = lxm[:, j]
    print(f"lexical features ready ({time.time() - t0:.0f}s)", flush=True)
    feats = base + LX + (list(lexical.ROLE_COLS) if a.role else [])
    trd = df[tr]
    vad = df[~tr]
    w = None
    if a.decoy_weight != 1.0:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from common import read_ground_truth
        gt = read_ground_truth(os.path.join("dataset", "train", "train_ground_truth.tsv"))
        owned_ids = set(x for v in gt.values() for x in v.split(",") if x)
        owned = {}
        for k in (2, 3):
            ids = pq.read_table(os.path.join(a.work, "train", f"s{k}.parquet"), columns=["id"]).column("id").to_pylist()
            owned[k] = np.array([i in owned_ids for i in ids])
        m2 = trd["src"].values == 2
        own = np.empty(len(trd), bool)
        own[m2] = owned[2][trd["d"].values[m2]]
        own[~m2] = owned[3][trd["d"].values[~m2]]
        w = np.where((trd["label"].values == 0) & ~own, a.decoy_weight, 1.0)
        print(f"decoy negatives weighted x{a.decoy_weight}: {int(((trd['label'].values == 0) & ~own).sum())} rows", flush=True)
    dtr = lgb.Dataset(trd[feats].values.astype(np.float32), label=trd["label"].values, feature_name=feats, weight=w)
    dva = lgb.Dataset(vad[feats].values.astype(np.float32), label=vad["label"].values, reference=dtr)
    params = {"objective": "binary", "learning_rate": a.lr, "num_leaves": a.leaves, "min_data_in_leaf": 100,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 255,
              "verbosity": -1, "metric": ["binary_logloss"], "num_threads": 0, "seed": a.seed}
    model = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dva], valid_names=["val"],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    os.makedirs(a.models, exist_ok=True)
    model.save_model(os.path.join(a.models, "lgb.txt"))
    json.dump(feats, open(os.path.join(a.models, "features.json"), "w"))
    vp = vad[["q", "src", "d", "label"]].copy()
    vp["p"] = model.predict(vad[feats].values.astype(np.float32))
    vp.to_parquet(os.path.join(a.work, "train", f"val_pred{a.tag}.parquet"), index=False)
    imp = sorted(zip(model.feature_importance("gain"), feats), reverse=True)[:25]
    print("top features:", [(f, round(float(g))) for g, f in imp], flush=True)
    print(f"stage-2 trained, best iter {model.best_iteration} ({time.time() - t0:.0f}s)", flush=True)
    del df, trd, vad, dtr, dva
    if a.no_test:
        return
    # test
    tparts = sorted(glob.glob(os.path.join(a.test_work, "test", "feats", "part-*.parquet")))
    te = pd.concat([pq.read_table(p, columns=["q", "src", "d"] + base).to_pandas() for p in tparts], ignore_index=True)
    s, voc = lexical.load_split(os.path.join(a.test_work, "test"))
    tf = lexical.features(lexical.pair_diffs(te, s, voc), enc_all)
    del s
    for c in LX:
        te[c] = tf[c].values
    out = te[["q", "src", "d"]].copy()
    out["p"] = model.predict(te[feats].values.astype(np.float32))
    out.to_parquet(os.path.join(a.test_work, "test", f"test_pred{a.tag}.parquet"), index=False)
    print(f"test scored: {len(out)} pairs ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
