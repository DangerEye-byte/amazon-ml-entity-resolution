"""Shared helpers: I/O, deterministic split, official metric."""
import os
import sys
import hashlib

import numpy as np
import pandas as pd

SRC_COLS = ["entity_id", "business_name", "business_address", "country"]


def ensure_hashseed():
    """Token hashing uses Python's hash(); it must be deterministic across runs/processes."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        import subprocess
        sys.exit(subprocess.call([sys.executable] + sys.argv, env=os.environ))


def read_source(path, nrows=None):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                     nrows=nrows, encoding="utf-8-sig")
    missing = set(SRC_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df[SRC_COLS]


def read_ground_truth(path):
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                     encoding="utf-8-sig")
    return dict(zip(gt["source1_entity_id"], gt["matched_entity_ids"]))


def is_val(ids, frac=0.2, seed=13):
    """Deterministic hash split of S1 ids -> boolean array (True = validation)."""
    # blake2b, not crc32: crc32 is affine, so different seeds give correlated splits
    thr = int(frac * 2 ** 64)
    salt = f"{seed}".encode()
    return np.fromiter((int.from_bytes(hashlib.blake2b(str(x).encode(), digest_size=8, key=salt).digest(),
                                       "little") < thr for x in ids), dtype=bool, count=len(ids))


def f05(pred, truth):
    """Official per-row F0.5, including the singleton rule."""
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_map, truth_map):
    """pred_map/truth_map: dict s1_id -> set of ids. Scored over truth_map keys."""
    tot = 0.0
    for sid, t in truth_map.items():
        tot += f05(pred_map.get(sid, set()), t)
    return tot / max(1, len(truth_map))


def parse_ids(s):
    return set(x for x in (s or "").split(",") if x)
