"""Per-stratum label-shift correction of test probabilities (EM, Saerens et al. 2002).

Test differs from validation in how often some kinds of candidate occur: e.g.
test India/US contain ~2x as many candidates at a nearby-but-different house
number (decoy businesses), while pairs with an equal number occur equally
often. Within a stratum the matcher is calibrated for the validation positive
rate pi_v; on test the rate pi_t is re-estimated by EM from the model's own
probabilities, and every probability is rescaled by the prior ratio:

    odds_t = odds_v * (pi_t / (1 - pi_t)) / (pi_v / (1 - pi_v))

Strata = country x (name-similarity band) x (house-number relation). Strata
with no shift get pi_t ~= pi_v and are left unchanged. Countries absent from
validation use the pooled validation rates of the stratum.

Usage:
    python src/label_shift.py --val-work work_sim --test-work work --out work/test/test_pred_ls.parquet
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLS = ["q", "src", "d", "num1_eq", "num1_lev", "num1_logdiff", "num_q_empty", "num_d_empty", "d_aempty",
        "nm_ratio", "d_oov_frac"]


def stratum(df):
    nm = np.select([df["d_oov_frac"] >= 0.5, df["nm_ratio"] >= 99.5, df["nm_ratio"] >= 85], ["oov", "nm100", "nm85"],
                   "nmlow")
    # S1 without a house number vs record without one behave very differently (decoys add numbers)
    qmiss = (df["num_q_empty"] == 1) & (df["num_d_empty"] == 0)
    dmiss = df["num_d_empty"] == 1
    # nearby number: |q - d| <= 30 (num1_logdiff = log1p|q - d|), the decoy-business pattern
    near = (df["num1_eq"] == 0) & (df["num1_logdiff"] >= 0) & (df["num1_logdiff"] <= np.log1p(30))
    num = np.select([df["d_aempty"] == 1, qmiss, dmiss, df["num1_eq"] == 1, near],
                    ["aempty", "qmiss", "dmiss", "eq", "near"], "far")
    return pd.Series(nm, index=df.index) + "|" + num


def em_prior(p, pi_v, iters=100):
    pi = pi_v
    for _ in range(iters):
        a = (pi / pi_v) * p
        b = ((1 - pi) / (1 - pi_v)) * (1 - p)
        new = float(np.mean(a / (a + b)))
        if abs(new - pi) < 1e-6:
            break
        pi = new
    return pi


def load(work, split, val):
    parts = sorted(glob.glob(os.path.join(work, split, "feats", "part-*.parquet")))
    cols = COLS + (["is_val"] if val else [])
    df = pd.concat([pq.read_table(p, columns=cols).to_pandas() for p in parts], ignore_index=True)
    if val:
        df = df[df["is_val"].values].drop(columns="is_val")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-work", default="work_sim")
    ap.add_argument("--test-work", default="work")
    ap.add_argument("--out", default="work/test/test_pred_ls.parquet")
    ap.add_argument("--min-n", type=int, default=2000)
    ap.add_argument("--clip", type=float, default=4.0, help="max odds scaling factor either way")
    a = ap.parse_args()
    ck_tr = pq.read_table(os.path.join(a.val_work, "train", "s1.parquet"), columns=["ck"]).column("ck").to_numpy()
    ck_te = pq.read_table(os.path.join(a.test_work, "test", "s1.parquet"), columns=["ck"]).column("ck").to_numpy()
    vp = pq.read_table(os.path.join(a.val_work, "train", "val_pred.parquet")).to_pandas()
    va = load(a.val_work, "train", True).merge(vp, on=["q", "src", "d"])
    va["st"] = stratum(va)
    va["ck"] = ck_tr[va["q"].values]
    tp = pq.read_table(os.path.join(a.test_work, "test", "test_pred.parquet")).to_pandas()
    te = load(a.test_work, "test", False).merge(tp, on=["q", "src", "d"], how="right")
    te["st"] = stratum(te.fillna({"nm_ratio": 0, "d_oov_frac": 0}))
    te["ck"] = ck_te[te["q"].values]
    pooled = va.groupby("st")["label"].mean()
    per = va.groupby(["ck", "st"])["label"].agg(["mean", "size"])
    p_new = te["p"].values.copy()
    rows = []
    for (ck, st), g in te.groupby(["ck", "st"]):
        if (ck, st) in per.index and per.loc[(ck, st), "size"] >= a.min_n:
            pi_v = per.loc[(ck, st), "mean"]
        elif st in pooled.index:
            pi_v = pooled[st]
        else:
            continue
        pi_v = float(np.clip(pi_v, 1e-4, 1 - 1e-4))
        p = np.clip(g["p"].values, 1e-6, 1 - 1e-6)
        pi_t = em_prior(p, pi_v)
        f = (pi_t / (1 - pi_t)) / (pi_v / (1 - pi_v))
        f = float(np.clip(f, 1 / a.clip, a.clip))
        odds = p / (1 - p) * f
        p_new[g.index.values] = odds / (1 + odds)
        rows.append((ck, st, len(g), pi_v, pi_t, f))
    r = pd.DataFrame(rows, columns=["ck", "st", "n", "pi_val", "pi_test", "odds_factor"])
    pd.set_option("display.width", 200)
    print(r[(r["odds_factor"] < 0.9) | (r["odds_factor"] > 1.1)].sort_values("n", ascending=False).round(3).to_string())
    out = te[["q", "src", "d"]].copy()
    out["p"] = p_new
    out.to_parquet(a.out, index=False)
    print(f"wrote {a.out}; mean p {te['p'].mean():.4f} -> {p_new.mean():.4f}")


if __name__ == "__main__":
    main()
