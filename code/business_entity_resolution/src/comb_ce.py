"""Assemble one transformer's scores for a pipeline version: concatenate every scored file
(later files win on duplicates) and keep only the pairs in that version's uncertain band
(lo < p < hi of the stage-2 predictions), identically on validation and test.

Usage:
    python src/comb_ce.py --name mini --out colab_pkg/comb10 \
        --val colab_pkg/out/ce_val.parquet colab_pkg/out_new_mini/ce_val.parquet colab_v10/out_new10_mini/ce_val.parquet \
        --test colab_pkg/out/ce_testband.parquet colab_pkg/out_new_mini/ce_testband.parquet colab_v10/out_new10_mini/ce_testband.parquet \
        --val-band work10_sim/train/val_predv10lx.parquet --test-band work10/test/test_predv10lx.parquet
"""
import argparse
import os

import numpy as np
import pandas as pd

KEY = lambda df: df["q"].values.astype(np.int64) * 8 * (1 << 32) + df["src"].values.astype(np.int64) * (1 << 32) + df["d"].values.astype(np.int64)  # noqa


def assemble(files, band_file, lo, hi):
    found = [f for f in files if os.path.exists(f)]
    for f in files:
        if f not in found:
            print(f"WARNING: {f} not found, skipped (check the coverage line below)", flush=True)
    ce = pd.concat([pd.read_parquet(f, columns=["q", "src", "d", "ce"]) for f in found], ignore_index=True)
    ce = ce.iloc[::-1].drop_duplicates(["q", "src", "d"]).iloc[::-1]
    band = pd.read_parquet(band_file, columns=["q", "src", "d", "p"])
    band = band[(band["p"] > lo) & (band["p"] < hi)]
    kb = KEY(band)
    out = ce[np.isin(KEY(ce), kb)].reset_index(drop=True)
    out = out.astype({"q": np.int32, "src": np.int8, "d": np.int32, "ce": np.float32})
    return out, len(band)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", default=None, help="omit for a validation-only run")
    ap.add_argument("--val-band", required=True)
    ap.add_argument("--test-band", default=None)
    ap.add_argument("--out", default="colab_pkg/comb10")
    ap.add_argument("--lo", type=float, default=0.005)
    ap.add_argument("--hi", type=float, default=0.995)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for split, files, band in (("val", a.val, a.val_band), ("test", a.test, a.test_band)):
        if files is None:
            continue
        out, nb = assemble(files, band, a.lo, a.hi)
        out.to_parquet(os.path.join(a.out, f"{a.name}_{split}.parquet"), index=False)
        print(f"{a.name} {split}: {len(out)} scored of {nb} band pairs ({len(out) / max(1, nb):.4f})", flush=True)


if __name__ == "__main__":
    main()
