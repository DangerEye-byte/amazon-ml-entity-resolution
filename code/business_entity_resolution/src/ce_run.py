"""Cross-encoder (transformer) scorer for hard candidate pairs.

Fine-tunes a small multilingual transformer (default:
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, Apache-2.0,
118M parameters) as a pair classifier on hard labelled training pairs, then
scores the uncertain validation/test pairs. The score is stacked into the
stage-3 model as an extra feature. Runs on CPU; checkpoints the fine-tuned
model and the scores in chunks so an interrupted run resumes.

Input (made by src/ce_export.py):
    ce_train.parquet  a_name, a_addr, b_name, b_addr, label
    ce_score.parquet  split, q, src, d, a_name, a_addr, b_name, b_addr
Output:
    ce_scores.parquet split, q, src, d, ce

Usage:
    python ce_run.py --data ce_pkg --out ce_pkg/out [--max-train 250000] [--epochs 1]
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def text(name, addr):
    return f"{name} | {addr}" if addr else name


def batches(df, bs):
    for i in range(0, len(df), bs):
        yield i, df.iloc[i:i + bs]


def encode(tok, chunk, max_len):
    a = [text(n, x) for n, x in zip(chunk["a_name"].values, chunk["a_addr"].values)]
    b = [text(n, x) for n, x in zip(chunk["b_name"].values, chunk["b_addr"].values)]
    return tok(a, b, truncation=True, max_length=max_len, padding=True, return_tensors="pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ce_pkg")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max-train", type=int, default=100000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--score-bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--train-emb", action="store_true", help="also fine-tune the word-embedding table (slower)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--limit-score", type=int, default=0, help="smoke test: score only this many rows")
    a = ap.parse_args()
    out = a.out or os.path.join(a.data, "out")
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(a.threads or os.cpu_count())
    torch.manual_seed(0)
    t0 = time.time()
    model_dir = os.path.join(out, "model")
    tok = AutoTokenizer.from_pretrained(a.model)
    if os.path.exists(os.path.join(model_dir, "done")):
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        print("loaded fine-tuned model from checkpoint", flush=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1)
        if not a.train_emb:
            for prm in model.base_model.embeddings.word_embeddings.parameters():
                prm.requires_grad = False
        tr = pd.read_parquet(os.path.join(a.data, "ce_train.parquet")).fillna("")
        if len(tr) > a.max_train:
            tr = tr.sample(a.max_train, random_state=0)
        tr = tr.sample(frac=1.0, random_state=1).reset_index(drop=True)
        steps = int(len(tr) * a.epochs / a.bs)
        opt = torch.optim.AdamW([prm for prm in model.parameters() if prm.requires_grad], lr=a.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / max(1, int(0.06 * steps))) * max(0.0, (steps - s) / steps))
        lossf = torch.nn.BCEWithLogitsLoss()
        model.train()
        step, run = 0, 0.0
        while step < steps:
            for i, chunk in batches(tr, a.bs):
                if step >= steps:
                    break
                enc = encode(tok, chunk, a.max_len)
                logits = model(**enc).logits.squeeze(-1)
                loss = lossf(logits, torch.tensor(chunk["label"].values, dtype=torch.float32))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                lv = float(loss.detach())
                run = 0.98 * run + 0.02 * lv if step > 1 else lv
                if step % 50 == 0 or step == steps:
                    rate = step * a.bs / (time.time() - t0)
                    print(f"step {step}/{steps} loss {run:.4f} ({rate:.0f} pairs/s, "
                          f"eta {(steps - step) * a.bs / max(rate, 1e-9) / 60:.0f} min)", flush=True)
        model.save_pretrained(model_dir)
        tok.save_pretrained(model_dir)
        open(os.path.join(model_dir, "done"), "w").write("ok")
        print(f"fine-tuning done ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    sc = pd.read_parquet(os.path.join(a.data, "ce_score.parquet")).fillna("")
    if a.limit_score:
        sc = sc.head(a.limit_score)
    chunk_rows = 50000
    for ci, start in enumerate(range(0, len(sc), chunk_rows)):
        path = os.path.join(out, f"scores_{ci:04d}.parquet")
        if os.path.exists(path):
            continue
        part = sc.iloc[start:start + chunk_rows]
        # sort by text length so each batch needs little padding
        ln = (part["a_name"].str.len() + part["a_addr"].str.len() + part["b_name"].str.len() + part["b_addr"].str.len()).values
        order = np.argsort(ln, kind="stable")
        sp = part.iloc[order]
        vals = []
        with torch.inference_mode():
            for i, chunk in batches(sp, a.score_bs):
                vals.append(torch.sigmoid(model(**encode(tok, chunk, a.max_len)).logits.squeeze(-1)).numpy())
        res = sp[["split", "q", "src", "d"]].copy()
        res["ce"] = np.concatenate(vals).astype(np.float32)
        res.to_parquet(path, index=False)
        done = min(start + chunk_rows, len(sc))
        print(f"scored {done}/{len(sc)} ({done / (time.time() - t0):.0f} rows/s overall)", flush=True)
    allp = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(out, "scores_*.parquet")))],
                     ignore_index=True)
    allp.to_parquet(os.path.join(out, "ce_scores.parquet"), index=False)
    print(f"wrote {out}/ce_scores.parquet: {len(allp)} rows ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
