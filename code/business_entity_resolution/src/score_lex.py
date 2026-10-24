"""Score test pairs with the lexical stage-2 model (train_lex.py) part by part.

The lexical encoder is refit on all training rows exactly as in train_lex.py,
then every test feats part gets its lexical features and probabilities; this
keeps memory bounded (train_lex scoring all 9.6M rows at once needs ~12 GB).

Usage:
    python src/score_lex.py --work work_sim --test-work work --models models_v7lx --tag lx
"""
import argparse
import glob
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work_sim")
    ap.add_argument("--test-work", default="work")
    ap.add_argument("--models", default="models_v7lx")
    ap.add_argument("--tag", default="lx")
    ap.add_argument("--extra", default=None, help="parquet of extra per-pair test features used by the model")
    a = ap.parse_args()
    t0 = time.time()
    feats = json.load(open(os.path.join(a.models, "features.json")))
    model = lgb.Booster(model_file=os.path.join(a.models, "lgb.txt"))
    parts = sorted(glob.glob(os.path.join(a.work, "train", "feats", "part-*.parquet")))
    tr = pd.concat([pq.read_table(p, columns=["q", "src", "d", "label", "is_val"]).to_pandas() for p in parts],
                   ignore_index=True)
    tr = tr[~tr["is_val"].values].reset_index(drop=True)
    s, voc = lexical.load_split(os.path.join(a.work, "train"))
    enc = lexical.fit_encoder(lexical.pair_diffs(tr, s, voc), tr["label"].values.astype(float))
    del s, tr
    print(f"encoder ready ({time.time() - t0:.0f}s)", flush=True)
    s, voc = lexical.load_split(os.path.join(a.test_work, "test"))
    base = [f for f in feats if not f.startswith("lx_") and not f.startswith("rl_")]
    ex = None
    if a.extra:
        ex = pd.read_parquet(a.extra)
        xcols = [c for c in ex.columns if c not in ("q", "src", "d")]
        base = [f for f in base if f not in xcols]
        xkey = ex["q"].values.astype(np.int64) * 8 * (1 << 32) + ex["src"].values.astype(np.int64) * (1 << 32) + ex["d"].values.astype(np.int64)
        xo = np.argsort(xkey)
        xkey, xval = xkey[xo], ex[xcols].values[xo]
        del ex
        ex = True
    tparts = sorted(glob.glob(os.path.join(a.test_work, "test", "feats", "part-*.parquet")))
    stats = None
    if any(f.startswith("rl_") for f in feats):
        allp = pd.concat([pq.read_table(p, columns=["q", "src", "d"]).to_pandas() for p in tparts], ignore_index=True)
        stats = lexical.role_stats(allp, s, lexical.s1_vocab_counts(s), s[1]["ck"].values)
        del allp
        print(f"role statistics ready ({time.time() - t0:.0f}s)", flush=True)
    outs = []
    for i, p in enumerate(tparts):
        te = pq.read_table(p, columns=["q", "src", "d"] + base).to_pandas()
        if ex is not None:  # sorted-key lookup of the extra features (same values as a left merge)
            k = te["q"].values.astype(np.int64) * 8 * (1 << 32) + te["src"].values.astype(np.int64) * (1 << 32) + te["d"].values.astype(np.int64)
            pos = np.clip(np.searchsorted(xkey, k), 0, len(xkey) - 1)
            hit = xkey[pos] == k
            vals = np.where(hit[:, None], xval[pos], np.nan)
            for j, c in enumerate(xcols):
                te[c] = vals[:, j].astype(np.float32)
        dd = lexical.pair_diffs(te, s, voc)
        lx = lexical.features(dd, enc)
        if stats is not None:
            rf = lexical.role_features(dd, s[1]["ck"].values[te["q"].values], stats)
            lx = pd.concat([lx, rf], axis=1)
        lx = lx[[c for c in lx.columns if c in feats]]
        for c in lx.columns:
            te[c] = lx[c].values
        o = te[["q", "src", "d"]].copy()
        o["p"] = model.predict(te[feats].values.astype(np.float32))
        outs.append(o)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(tparts)} parts ({time.time() - t0:.0f}s)", flush=True)
    out = pd.concat(outs, ignore_index=True)
    out.to_parquet(os.path.join(a.test_work, "test", f"test_pred{a.tag}.parquet"), index=False)
    print(f"wrote test_pred{a.tag}: {len(out)} pairs, mean p {out['p'].mean():.4f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
