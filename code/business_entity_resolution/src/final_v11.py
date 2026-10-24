"""v11 final stacks (see greedy_stack.py for the search that chose them).

India/US: candidates = stage-3 (v10sc) p > 0.02; stack on stage-2 / stage-3 / role logits, 7 transformer logits
          (incl. the twin-aware one), collective / density / twin and pairwise features; 3 bagged seeds.
France:   candidates = word-role (v10rac) p > 0.02; stack on the role logit, the French self-trained and twin-aware
          transformers and the language-agnostic collective features; 3 bagged seeds.
Both stacks are fit on all validation rows and applied to test. Writes work10/test/test_predv11{inus,fr}.parquet.
The India/US feature groups come from the greedy search (greedy_stack.py build + greedy_run.py), frozen in
config/stack_inus_features.json. The submitted France rows come from the round-3 France stack (reproduce.sh), so
--inus-only skips the France stack here.

Usage:
    python src/final_v11.py [--inus-only]
"""
import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import greedy_run as gr  # noqa: E402
import greedy_stack as g  # noqa: E402

PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 200, "verbosity": -1,
          "num_threads": 4, "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.7}
SEEDS = (0, 1, 2)


def fit_predict(feats, va, te):
    X, y = va[feats].values.astype(np.float32), va["label"].values
    Xt = te[feats].values.astype(np.float32)
    p = np.zeros(len(te))
    for s in SEEDS:
        m = lgb.train({**PARAMS, "seed": s}, lgb.Dataset(X, y), 300)
        p += m.predict(Xt) / len(SEEDS)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="config/stack_inus_features.json", help="India/US stack features (greedy search)")
    ap.add_argument("--inus-only", action="store_true", help="skip the France stack (France rows come from elsewhere)")
    a = ap.parse_args()
    t0 = time.time()
    inus = json.load(open(a.feats)) + ["l_tw"]
    kw = dict(base="stage3", s3="v10sc", s2="v10lxc", role="v10rac")
    va = g.build_split(inus, split="val", **kw)
    te = g.build_split(inus, split="test", **kw)
    out = te[["q", "src", "d"]].copy()
    out["p"] = fit_predict(inus, va, te)
    out.to_parquet("work10/test/test_predv11inus.parquet", index=False)
    print(f"India/US stack: {len(out)} test pairs ({time.time() - t0:.0f}s)", flush=True)
    del va, te
    if a.inus_only:
        return
    fr = ["lpra", "l_fr", "l_fr2", "l_tw"] + gr.COLL
    kw = dict(base="role", role="v10rac")
    va = g.build_split(fr, split="val", **kw)
    te = g.build_split(fr, split="test", **kw)
    out = te[["q", "src", "d"]].copy()
    out["p"] = fit_predict(fr, va, te)
    out.to_parquet("work10/test/test_predv11fr.parquet", index=False)
    print(f"France stack: {len(out)} test pairs ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
