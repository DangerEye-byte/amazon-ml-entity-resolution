"""Show validation false positives / false negatives under the chosen decision rule.

Usage:
    python src/error_analysis.py --n 25
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decide import attach_ids, exclusivity, select  # noqa: E402

COLS = ["id", "core", "alt", "anorm", "ck"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--models", default="models")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    base = os.path.join(a.work, "train")
    df = pq.read_table(os.path.join(base, "val_pred.parquet")).to_pandas()
    attach_ids(df, base)
    with open(os.path.join(a.models, "decision.json")) as f:
        cfg = json.load(f)["cfg"]
    df = df.sort_values("q", kind="stable").reset_index(drop=True)
    p = exclusivity(df, cfg["excl"])
    df["p_adj"] = p
    df["sel"] = select(df["q"].values, p, cfg["method"], cfg["t"], cfg.get("m", 0.0))
    s1 = pq.read_table(os.path.join(base, "s1.parquet"), columns=COLS)
    docs = {k: pq.read_table(os.path.join(base, f"s{k}.parquet"), columns=COLS) for k in (2, 3)}
    fp = df[df["sel"] & (df["label"] == 0)]
    fn = df[~df["sel"] & (df["label"] == 1)]
    print(f"val pairs {len(df)}, selected {df['sel'].sum()}, FP {len(fp)}, FN(scored) {len(fn)}")
    ck = np.array(s1.column("ck").take(df["q"].values).to_pylist())
    for c in np.unique(ck):
        m = ck == c
        print(f"  {c}: FP {(df['sel'] & (df['label'] == 0) & m).sum()}, FN {(~df['sel'] & (df['label'] == 1) & m).sum()}, "
              f"TP {(df['sel'] & (df['label'] == 1) & m).sum()}")
    random.seed(0)

    def show(rows, title):
        print(f"\n===== {title}")
        for i in random.sample(range(len(rows)), min(a.n, len(rows))):
            r = rows.iloc[i]
            q = s1.slice(int(r["q"]), 1).to_pylist()[0]
            d = docs[int(r["src"])].slice(int(r["d"]), 1).to_pylist()[0]
            true_sel = df[(df["q"] == r["q"]) & (df["label"] == 1)]
            print(f"p={r['p']:.3f} adj={r['p_adj']:.3f} | S1 [{q['core']}] [{q['anorm']}]\n"
                  f"          S{int(r['src'])} [{d['core']}] alt[{d['alt']}] [{d['anorm']}]  "
                  f"(S1 has {len(true_sel)} true cands)")

    show(fp, "FALSE POSITIVES")
    show(fn, "FALSE NEGATIVES (retrieved but not selected)")


if __name__ == "__main__":
    main()
