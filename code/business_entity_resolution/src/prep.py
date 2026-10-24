"""Stage 1: normalize every record and emit hashed blocking tokens -> parquet.

Output: <work>/<split>/s{1,2,3}.parquet with normalized string columns plus
token-hash list columns (t_n, t_nb, t_c, t_a, t_h, t_p) consumed by retrieve.py
and features.py.

Usage:
    python src/prep.py --split train --data-dir dataset --work work
    python src/prep.py --split test  --data-dir dataset --work work
"""
import argparse
import json
import os
import sys
import time
from itertools import combinations
from multiprocessing import Pool

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SRC_COLS, ensure_hashseed  # noqa: E402
from normalize import country_key, norm_name, norm_addr, street_words  # noqa: E402

_MAP = {"name": {}, "addr": {}}
TOKEN_TYPES = ["t_n", "t_nb", "t_c", "t_a", "t_h", "t_p"]
STR_COLS = ["core", "compact", "alt", "legal", "anorm", "awords", "nums", "state"]
INT_COLS = ["is_domain", "nonlatin", "aempty", "anonlatin"]


def _init(mapping):
    global _MAP
    _MAP = mapping


def tokens(nm, ad):
    core = nm["core"].split()
    alt = nm["alt"].split()
    t_n = {"n:" + t for t in core + alt if len(t) >= 2}
    if nm["compact"]:
        t_n.add("x:" + nm["compact"])
    if alt:
        t_n.add("x:" + "".join(alt))
    base = sorted(set(core[:6]))
    t_nb = {"b:" + a + "|" + b for a, b in combinations(base, 2)}
    comp = nm["compact"][:24]
    t_c = {"c:" + comp[i:i + 4] for i in range(max(0, len(comp) - 3))}
    aw = ad["awords"].split()
    t_a = {"a:" + w for w in aw if len(w) >= 3}
    sw = [w for w in street_words(ad["awords"]) if len(w) >= 3][:5]
    nums = ad["nums"].split()
    t_h = {"h:" + n + "|" + w for n in nums[:3] for w in sw}
    t_p = {"p:" + n for n in nums if len(n) >= 2}
    return [[hash(t) for t in s] for s in (t_n, t_nb, t_c, t_a, t_h, t_p)]


def process(rows):
    return to_table(_process(rows))


def _process(rows):
    out = {c: [] for c in ["id", "ck"] + STR_COLS + INT_COLS + TOKEN_TYPES}
    for eid, name, addr, country in rows:
        ck = country_key(country)
        nm = norm_name(name, _MAP.get("name"))
        ad = norm_addr(addr, ck, _MAP.get("addr"))
        out["id"].append(eid)
        out["ck"].append(ck)
        for c in ("core", "compact", "alt", "legal"):
            out[c].append(nm[c])
        for c in ("anorm", "awords", "nums", "state"):
            out[c].append(ad[c])
        out["is_domain"].append(nm["is_domain"])
        out["nonlatin"].append(nm["nonlatin"])
        out["aempty"].append(ad["aempty"])
        out["anonlatin"].append(ad["anonlatin"])
        for c, v in zip(TOKEN_TYPES, tokens(nm, ad)):
            out[c].append(v)
    return out


def to_table(out):
    cols = {}
    for c, vals in out.items():
        if c in TOKEN_TYPES:
            cols[c] = pa.array(vals, type=pa.list_(pa.int64()))
        elif c in INT_COLS:
            cols[c] = pa.array(vals, type=pa.int8())
        else:
            cols[c] = pa.array(vals, type=pa.string())
    return pa.table(cols)


def iter_chunks(path, nrows=None, chunk=25000):
    reader = pd.read_csv(path, sep="	", dtype=str, keep_default_na=False, na_filter=False,
                         encoding="utf-8-sig", chunksize=chunk, nrows=nrows)
    for df in reader:
        yield list(df[SRC_COLS].itertuples(index=False, name=None))


def run(split, data_dir, work, translit_path, workers, nrows=None):
    mapping = {"name": {}, "addr": {}}
    if translit_path and os.path.exists(translit_path):
        with open(translit_path, encoding="utf8") as f:
            mapping = json.load(f)
    else:
        print("WARNING: no translit dictionary found, falling back to unidecode only")
    os.makedirs(os.path.join(work, split), exist_ok=True)
    for k in (1, 2, 3):
        t0 = time.time()
        path = os.path.join(data_dir, split, f"{split}_source{k}.tsv")
        out = os.path.join(work, split, f"s{k}.parquet")
        n = 0
        writer = None
        with Pool(workers, initializer=_init, initargs=(mapping,)) as pool:
            for tbl in pool.imap(process, iter_chunks(path, nrows), chunksize=1):
                if writer is None:
                    writer = pq.ParquetWriter(out + ".tmp", tbl.schema)
                writer.write_table(tbl)
                n += tbl.num_rows
        writer.close()
        os.replace(out + ".tmp", out)
        print(f"{split} s{k}: {n} rows -> {out} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ensure_hashseed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--translit", default="work/translit.json")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--nrows", type=int)
    a = ap.parse_args()
    run(a.split, a.data_dir, a.work, a.translit, a.workers, a.nrows)


if __name__ == "__main__":
    main()
