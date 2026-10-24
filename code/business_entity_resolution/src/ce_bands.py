"""Scoring files for the cross-encoders: the uncertain band of the final pipeline, plus every record text it uses.

The transformers score only pairs whose stage-2 probability lies in (--lo, --hi), identically on validation and test:
  pairs_val_inus_band.parquet    validation band of the lexical stage-2 model (q, src, d, label)
  pairs_test_inus_band.parquet   test band of the lexical stage-2 model, India/US rows (q, src, d)
  pairs_val_fr_band.parquet      validation band of the word-role model (q, src, d, label)
  pairs_test_fr_band.parquet     test band of the word-role model, France rows (q, src, d)
  rec_<split>_s<k>_band.parquet  texts of every record these pairs use (colab_ce.py reads the union of rec_* files)

During the competition the same pairs were scored incrementally, as each pipeline version added new candidates
(colab_export_new.py); src/comb_ce.py then assembled one file per model restricted to this band. Scoring the band
directly gives the same inputs to the final stacks.

Usage:
    python src/ce_bands.py --out colab_pkg
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colab_export import records, write  # noqa: E402


def band(path, lo, hi, cols):
    p = pd.read_parquet(path, columns=["q", "src", "d", "p"] + (["label"] if "label" in cols else []))
    b = p[(p["p"] > lo) & (p["p"] < hi)][cols].reset_index(drop=True)
    if "label" in cols:
        b["label"] = b["label"].astype(np.int8)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="colab_pkg")
    ap.add_argument("--work-sim", default="work10_sim")
    ap.add_argument("--work", default="work10")
    ap.add_argument("--lx", default="v10lx", help="tag of the India/US stage-2 predictions")
    ap.add_argument("--role", default="v10ra", help="tag of the France-safe word-role predictions")
    ap.add_argument("--lo", type=float, default=0.005)
    ap.add_argument("--hi", type=float, default=0.995)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    vd, td = os.path.join(a.work_sim, "train"), os.path.join(a.work, "test")
    ck = pq.read_table(os.path.join(td, "s1.parquet"), columns=["ck"]).column("ck").to_numpy(zero_copy_only=False)
    vl = ["q", "src", "d", "label"]
    tl = ["q", "src", "d"]
    files = {
        "pairs_val_inus_band.parquet": band(os.path.join(vd, f"val_pred{a.lx}.parquet"), a.lo, a.hi, vl),
        "pairs_val_fr_band.parquet": band(os.path.join(vd, f"val_pred{a.role}.parquet"), a.lo, a.hi, vl),
    }
    t = band(os.path.join(td, f"test_pred{a.lx}.parquet"), a.lo, a.hi, tl)
    files["pairs_test_inus_band.parquet"] = t[ck[t["q"].values] != "fr"].reset_index(drop=True)
    t = band(os.path.join(td, f"test_pred{a.role}.parquet"), a.lo, a.hi, tl)
    files["pairs_test_fr_band.parquet"] = t[ck[t["q"].values] == "fr"].reset_index(drop=True)
    for name, df in files.items():
        write(df, os.path.join(a.out, name))
        print(f"{name}: {len(df)} pairs", flush=True)
    for split, work, names in (("train", a.work_sim, ["pairs_val_inus_band.parquet", "pairs_val_fr_band.parquet"]),
                               ("test", a.work, ["pairs_test_inus_band.parquet", "pairs_test_fr_band.parquet"])):
        df = pd.concat([files[n] for n in names], ignore_index=True)
        tmp = os.path.join(a.out, f"_tmp_{split}")
        os.makedirs(tmp, exist_ok=True)
        n = records(split, work, 1, df["q"].values, tmp)
        for k in (2, 3):
            n += records(split, work, k, df["d"].values[df["src"].values == k], tmp)
        for k in (1, 2, 3):
            os.replace(os.path.join(tmp, f"rec_{split}_s{k}.parquet"), os.path.join(a.out, f"rec_{split}_s{k}_band.parquet"))
        os.rmdir(tmp)
        print(f"{split}: {n} record texts", flush=True)


if __name__ == "__main__":
    main()
