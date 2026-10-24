"""Export candidate pairs that no transformer has scored yet (new in a later pipeline version).

Writes to --out: pairs_val_<tag>new.parquet, pairs_test_<tag>new.parquet (pairs whose stage-2
probability lies in the uncertain band and that are not in any --old-val / --old-test file) and
rec_<split>_s<k>_<tag>.parquet (texts of their records).

Usage:
    python src/colab_export_new.py                        # v8 (defaults)
    python src/colab_export_new.py --tag v10 --out colab_v10 --val-pred work10_sim/train/val_predv10lx.parquet \
        --test-pred work10/test/test_predv10lx.parquet --work-sim work10_sim --work work10 \
        --old-val colab_pkg/pairs_val.parquet colab_new/pairs_val_v8new.parquet \
        --old-test colab_pkg/pairs_test.parquet colab_new/pairs_test_v8new.parquet
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colab_export import records, write  # noqa: E402

KEY = lambda df: df["q"].values.astype(np.int64) * 8 * (1 << 32) + df["src"].values.astype(np.int64) * (1 << 32) + df["d"].values.astype(np.int64)  # noqa


def new_pairs(cur, olds):
    old = np.concatenate([KEY(pd.read_parquet(p, columns=["q", "src", "d"])) for p in olds])
    return cur[~np.isin(KEY(cur), old)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="colab_new")
    ap.add_argument("--tag", default="v8")
    ap.add_argument("--val-pred", default="work8_sim/train/val_predv8lx.parquet")
    ap.add_argument("--test-pred", default="work8/test/test_predv8lx.parquet")
    ap.add_argument("--work-sim", default="work8_sim")
    ap.add_argument("--work", default="work8")
    ap.add_argument("--old-val", nargs="+", default=["colab_pkg/pairs_val.parquet"])
    ap.add_argument("--old-test", nargs="+", default=["colab_pkg/pairs_test.parquet"])
    ap.add_argument("--test-pred-fr", default=None,
                    help="France band from this file (role model); India/US band from --test-pred")
    ap.add_argument("--old-test-fr", nargs="+", default=["colab_fr/pairs_test_fr.parquet"],
                    help="France pairs the France transformer already scored")
    ap.add_argument("--val-pred-fr", default=None, help="also the role model's validation band (France stack)")
    ap.add_argument("--no-test", action="store_true", help="validation pairs only (test predictions not ready yet)")
    ap.add_argument("--no-val", action="store_true", help="test pairs only (validation already scored)")
    ap.add_argument("--full-band", default=None, help="also write every India/US test band pair here (for new models)")
    ap.add_argument("--lo", type=float, default=0.005)
    ap.add_argument("--hi", type=float, default=0.995)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    vp = pd.read_parquet(a.val_pred)
    vp = vp[(vp["p"] > a.lo) & (vp["p"] < a.hi)]
    if a.val_pred_fr:
        rp = pd.read_parquet(a.val_pred_fr)
        vp = pd.concat([vp, rp[(rp["p"] > a.lo) & (rp["p"] < a.hi)]]).drop_duplicates(["q", "src", "d"])
    v = new_pairs(vp, a.old_val)[["q", "src", "d", "label"]].reset_index(drop=True)
    if a.no_test:
        write(v, os.path.join(a.out, f"pairs_val_{a.tag}new.parquet"))
        tmp = a.out + "_tmp"
        os.makedirs(tmp, exist_ok=True)
        records("train", a.work_sim, 1, v["q"].values, tmp)
        for k in (2, 3):
            records("train", a.work_sim, k, v["d"].values[v["src"].values == k], tmp)
        for k in (1, 2, 3):
            os.replace(os.path.join(tmp, f"rec_train_s{k}.parquet"), os.path.join(a.out, f"rec_train_s{k}_{a.tag}.parquet"))
        os.rmdir(tmp)
        print(f"new uncertain validation pairs: {len(v)} (no test)", flush=True)
        return
    tp = pd.read_parquet(a.test_pred, columns=["q", "src", "d", "p"])
    tp = tp[(tp["p"] > a.lo) & (tp["p"] < a.hi)]
    if a.test_pred_fr:
        import pyarrow.parquet as pq
        ck = pq.read_table(os.path.join(a.work, "test", "s1.parquet"), columns=["ck"]).column("ck").to_numpy(
            zero_copy_only=False)
        fp = pd.read_parquet(a.test_pred_fr, columns=["q", "src", "d", "p"])
        fp = fp[(ck[fp["q"].values] == "fr") & (fp["p"] > a.lo) & (fp["p"] < a.hi)]
        tp = tp[ck[tp["q"].values] != "fr"]
        fr_new = new_pairs(fp, a.old_test_fr)[["q", "src", "d"]]   # pairs the France transformer scored
        print(f"France band pairs {len(fp)}, not scored yet {len(fr_new)}", flush=True)
    t = new_pairs(tp, a.old_test)[["q", "src", "d"]]
    if a.test_pred_fr:
        t = pd.concat([t, fr_new])
    t = t.reset_index(drop=True)
    if a.full_band:
        write(tp[["q", "src", "d"]].reset_index(drop=True), os.path.join(a.out, a.full_band))
        print(f"full India/US test band: {len(tp)} pairs", flush=True)
    if not a.no_val:
        write(v, os.path.join(a.out, f"pairs_val_{a.tag}new.parquet"))
    write(t, os.path.join(a.out, f"pairs_test_{a.tag}new.parquet"))
    print(f"new uncertain pairs: val {0 if a.no_val else len(v)}, test {len(t)}", flush=True)
    for split, df, work in (("train", v, a.work_sim), ("test", t, a.work)):
        if split == "train" and a.no_val:
            continue
        tmp = a.out + "_tmp"
        os.makedirs(tmp, exist_ok=True)
        records(split, work, 1, df["q"].values, tmp)
        for k in (2, 3):
            records(split, work, k, df["d"].values[df["src"].values == k], tmp)
        for k in (1, 2, 3):
            os.replace(os.path.join(tmp, f"rec_{split}_s{k}.parquet"),
                       os.path.join(a.out, f"rec_{split}_s{k}_{a.tag}.parquet"))
        os.rmdir(tmp)
    print("done")


if __name__ == "__main__":
    main()
