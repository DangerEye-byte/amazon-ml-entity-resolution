"""Learn a non-Latin -> Latin token dictionary from training ground-truth pairs.

For every S2/S3 record whose name or address contains non-Latin script
(Devanagari, Kannada, Tamil, ...), we look at the matched S1 record (always
Latin) and count which Latin tokens co-occur with each non-Latin token.
Positionally aligned pairs (same token count) get extra weight. Only the
provided training data is used.

Usage:
    python src/translit.py --train-dir dataset/train --out work/translit.json
"""
import argparse
import collections
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import NONLATIN_RE, SPLIT_KEEP, _basic, NON_ALNUM  # noqa: E402
from common import read_source  # noqa: E402


def nl_tokens(s):
    return [p.strip().lower() for p in SPLIT_KEEP.split(s or "") if p.strip() and NONLATIN_RE.search(p)]


def latin_tokens(s):
    return [t for t in NON_ALNUM.split(_basic(s or "")) if t]


def learn(pairs_name, pairs_addr, min_count=2, min_share=0.5):
    out = {}
    for kind, pairs in (("name", pairs_name), ("addr", pairs_addr)):
        cooc = collections.defaultdict(collections.Counter)
        cnt = collections.Counter()
        for doc_raw, s1_raw in pairs:
            nl = nl_tokens(doc_raw)
            if not nl:
                continue
            lt = latin_tokens(s1_raw)
            lset = set(lt)
            doc_parts = [p.strip().lower() for p in re.split(r"\s+", doc_raw.strip()) if p.strip()]
            aligned = len(doc_parts) == len(lt)
            for i, t in enumerate(set(nl)):
                cnt[t] += 1
                for s in lset:
                    cooc[t][s] += 1
            if aligned:
                for dp, s in zip(doc_parts, lt):
                    if NONLATIN_RE.search(dp):
                        cooc[dp.strip(",;()")][s] += 2
        m = {}
        for t, c in cooc.items():
            s, v = c.most_common(1)[0]
            n = cnt[t]
            if n >= min_count and v >= min_share * n:
                m[t] = s
            elif n == 1 and v >= 3:  # single aligned example
                m[t] = s
        if kind == "addr":
            # a single non-Latin token can spell a two-word state name; map it to the full name
            from normalize import IN_STATES
            for t, s in list(m.items()):
                full = [n for n in IN_STATES if n.split()[-1] == s and len(n.split()) > 1]
                if len(full) >= 1 and " " not in s:
                    best, bv = None, 0
                    for n in full:
                        v = sum(cooc[t][w] for w in n.split()[:-1])
                        if v > bv:
                            best, bv = n, v
                    if best and bv >= 0.8 * cnt[t]:
                        m[t] = best
        out[kind] = m
        print(f"{kind}: {len(cnt)} non-latin tokens, {len(m)} mapped")
    return out


def skeleton(w):
    """Crude phonetic consonant skeleton, robust to Indic->Latin romanisation quirks."""
    w = re.sub(r"N", "n", w)
    w = re.sub(r"[^a-z]", "", w.lower())
    for a, b in (("ph", "f"), ("ck", "k"), ("q", "k"), ("x", "ks"), ("ce", "se"), ("ci", "si"),
                 ("c", "k"), ("z", "j"), ("g", "j"), ("w", "v"), ("th", "t"), ("dh", "d")):
        w = w.replace(a, b)
    w = re.sub(r"[aeiouyh]", "", w)
    return re.sub(r"(.)\1+", r"\1", w)


def extend_from_vocab(mapping, source_paths, s1_path, max_dist=1):
    """Map non-Latin name tokens unseen in training to the closest Latin word of the
    (unlabeled) S1 vocabulary by consonant skeleton. Uses only challenge data."""
    from rapidfuzz.distance import Levenshtein
    from unidecode import unidecode
    miss = collections.Counter()
    for p in source_paths:
        for s in read_source(p)["business_name"].values:
            for t in nl_tokens(s):
                if t not in mapping:
                    miss[t] += 1
    vocab = collections.Counter(w for n in read_source(s1_path)["business_name"].str.lower().values
                                for w in re.findall(r"[a-z]+", n) if len(w) >= 3)
    by = collections.defaultdict(int)
    word_of = {}
    for w, c in vocab.items():
        k = skeleton(w)
        if c > by[k]:
            by[k], word_of[k] = c, w
    keys = list(by)
    added = 0
    for t, c in miss.items():
        sk = skeleton(unidecode(t))
        if len(sk) < 3:
            continue
        best, bs = None, -1e9
        for k in keys:
            if abs(len(k) - len(sk)) > max_dist:
                continue
            d = Levenshtein.distance(sk, k, score_cutoff=max_dist)
            if d <= max_dist:
                if d > 0 and len(sk) < 5:
                    continue
                sc = np.log(by[k]) - 20.0 * d
                if sc > bs:
                    best, bs = k, sc
        if best is not None:
            mapping[t] = word_of[best]
            added += 1
    print(f"extended dictionary with {added}/{len(miss)} unseen tokens")
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--out", default="work/translit.json")
    ap.add_argument("--test-dir", default="dataset/test", help="unlabeled test data for vocab extension")
    a = ap.parse_args()
    docs = []
    for k in (2, 3):
        df = read_source(os.path.join(a.train_dir, f"train_source{k}.tsv"))
        mask = df["business_name"].str.contains(NONLATIN_RE) | df["business_address"].str.contains(NONLATIN_RE)
        docs.append(df[mask])
    docs = pd.concat(docs, ignore_index=True)
    print("non-latin docs:", len(docs))
    gt = pd.read_csv(os.path.join(a.train_dir, "train_ground_truth.tsv"), sep="\t", dtype=str,
                     keep_default_na=False)
    gt = gt[gt["matched_entity_ids"] != ""]
    gt["doc"] = gt["matched_entity_ids"].str.split(",")
    gt = gt.explode("doc")[["source1_entity_id", "doc"]]
    gt = gt[gt["doc"].isin(set(docs["entity_id"]))]
    s1 = read_source(os.path.join(a.train_dir, "train_source1.tsv"))
    j = gt.merge(docs, left_on="doc", right_on="entity_id").merge(
        s1, left_on="source1_entity_id", right_on="entity_id", suffixes=("_d", "_s"))
    print("aligned pairs:", len(j))
    res = learn(zip(j["business_name_d"], j["business_name_s"]),
                zip(j["business_address_d"], j["business_address_s"]))
    if a.test_dir and os.path.isdir(a.test_dir):
        extend_from_vocab(res["name"], [os.path.join(a.test_dir, f"test_source{k}.tsv") for k in (2, 3)],
                          os.path.join(a.test_dir, "test_source1.tsv"))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf8") as f:
        json.dump(res, f, ensure_ascii=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
