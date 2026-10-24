"""GPU cross-encoder for Google Colab (also runs on CPU for a smoke test).

Fine-tunes paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 118M params) as a
pair classifier on labelled training pairs, then scores every validation and
test candidate pair. Checkpoints the model and every scored chunk to --pkg/out,
so a disconnected session resumes where it stopped.

Usage (Colab cell):
    !python /content/drive/MyDrive/colab_pkg/colab_ce.py --pkg /content/drive/MyDrive/colab_pkg
Smoke test (CPU):
    python src/colab_ce.py --pkg colab_pkg --smoke
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_texts(pkg, split):
    """Union of rec_<split>_s<k>*.parquet files (extra files add rows for newer candidate pairs)."""
    t = {}
    for k in (1, 2, 3):
        r = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(pkg, f"rec_{split}_s{k}*.parquet")))],
                      ignore_index=True).drop_duplicates("row")
        arr = np.empty(int(r["row"].max()) + 1, dtype=object)
        arr[r["row"].values] = r["text"].values
        t[k] = arr
    return t


def encode(tok, a, b, max_len, dev):
    """Pair encoding; decoder tokenizers (no [SEP], e.g. Qwen) get one marked sequence instead."""
    if tok.sep_token is None:
        return tok([f"S1: {x} || Record: {y}" for x, y in zip(a, b)], truncation=True, max_length=max_len,
                   padding=True, return_tensors="pt").to(dev)
    return tok(a, b, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dev)


def pair_texts(t, df):
    a = t[1][df["q"].values]
    if "tw" in df.columns:  # twin-aware model: "S1 || closest twin S1" against the record
        tw = df["tw"].values
        a = np.array([x if j < 0 else f"{x} || {t[1][j]}" for x, j in zip(a, tw)], dtype=object)
    b = np.empty(len(df), dtype=object)
    for k in (2, 3):
        m = df["src"].values == k
        b[m] = t[k][df["d"].values[m]]
    return list(a), list(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default="colab_pkg")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--score-bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=3000, help="with --smoke: pairs scored per file")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--model-dir", default=None, help="score with this already fine-tuned model (no training)")
    ap.add_argument("--val-file", default="pairs_val.parquet")
    ap.add_argument("--train-file", default="pairs_train.parquet")
    ap.add_argument("--out-name", default="out", help="output folder inside --pkg (use a different one per model)")
    ap.add_argument("--test-file", default="pairs_test.parquet", help="e.g. pairs_test_band.parquet (uncertain pairs only)")
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast (DeBERTa-v3 overflows in float16)")
    ap.add_argument("--val-only", action="store_true", help="score only --val-file")
    ap.add_argument("--test-only", action="store_true", help="score only --test-file")
    a = ap.parse_args()
    out = os.path.join(a.pkg, "out_smoke" if (a.smoke and a.out_name == "out") else a.out_name)
    os.makedirs(out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)
    torch.manual_seed(0)
    t0 = time.time()
    mdir = a.model_dir or os.path.join(out, "model")
    if a.model_dir and not os.path.exists(os.path.join(mdir, "done")):
        raise SystemExit(f"--model-dir {mdir} has no finished model (missing 'done')")
    tok = AutoTokenizer.from_pretrained(mdir if os.path.exists(os.path.join(mdir, "done")) else a.model)
    use_amp = dev == "cuda"
    amp_dtype = torch.bfloat16 if a.bf16 else torch.float16
    model = None
    if os.path.exists(os.path.join(mdir, "done")):
        try:  # a VM crash right after saving can leave an incomplete file on Drive: verify it loads
            model = AutoModelForSequenceClassification.from_pretrained(mdir).to(dev)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model.config.pad_token_id = tok.pad_token_id
            print("loaded fine-tuned model (verified)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"saved model is unreadable ({e}); retraining from the last checkpoint", flush=True)
            os.remove(os.path.join(mdir, "done"))
            model = None
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1).to(dev)
        if tok.pad_token is None:  # decoder LLMs (e.g. Qwen2.5) classify from the last non-pad token
            tok.pad_token = tok.eos_token
        model.config.pad_token_id = tok.pad_token_id
        tr = pd.read_parquet(os.path.join(a.pkg, a.train_file))
        if a.smoke:
            tr = tr.head(256)
        if "split" not in tr.columns:
            tr["split"] = "train"
        A, B = [None] * len(tr), [None] * len(tr)
        for sp in tr["split"].unique():   # pairs may come from the train split or (pseudo-labelled) test split
            idx = np.flatnonzero(tr["split"].values == sp)
            tt = load_texts(a.pkg, sp)
            a_, b_ = pair_texts(tt, tr.iloc[idx])
            for j, i in enumerate(idx):
                A[i], B[i] = a_[j], b_[j]
            del tt
        y = tr["label"].values.astype(np.float32)
        steps = int(len(tr) * a.epochs / a.bs)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        warm = max(1, int(0.05 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / steps))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not a.bf16)
        lossf = torch.nn.BCEWithLogitsLoss()
        model.train()
        step, run, perm, nskip = 0, None, np.random.default_rng(0).permutation(len(tr)), 0
        ck = os.path.join(out, "ckpt.pt")
        start = 0
        if os.path.exists(ck):  # resume an interrupted training run
            st = torch.load(ck, map_location=dev, weights_only=False)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            step = start = st["step"]
            print(f"resumed training at step {step}", flush=True)
        nb = (len(perm) + a.bs - 1) // a.bs          # batches per pass over the data
        epoch = start // nb
        if epoch > 0:
            perm = np.random.default_rng(epoch).permutation(len(tr))
        first = (start % nb) * a.bs                   # resume inside the current pass
        while step < steps:
            for i in range(first, len(perm), a.bs):
                if step >= steps:
                    break
                idx = perm[i:i + a.bs]
                enc = encode(tok, [A[j] for j in idx], [B[j] for j in idx], a.max_len, dev)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(**enc).logits.squeeze(-1)
                loss = lossf(logits.float(), torch.tensor(y[idx], device=dev))
                opt.zero_grad()
                if torch.isfinite(loss):
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if torch.isfinite(gn) or scaler.is_enabled():   # the scaler skips inf steps itself
                        scaler.step(opt)
                    else:
                        nskip += 1
                    scaler.update()
                else:
                    nskip += 1   # never let a non-finite batch reach the weights
                sched.step()
                step += 1
                lv = float(loss.detach())
                if np.isfinite(lv):
                    run = lv if run is None else 0.98 * run + 0.02 * lv
                if step % a.ckpt_every == 0 and step < steps:
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                                "sched": sched.state_dict(), "step": step}, ck + ".tmp")
                    os.replace(ck + ".tmp", ck)
                if step % 200 == 0 or step == steps:
                    rate = (step - start) * a.bs / (time.time() - t0)
                    print(f"step {step}/{steps} loss {run:.4f} ({rate:.0f} pairs/s, eta "
                          f"{(steps - step) * a.bs / rate / 60:.0f} min, skipped {nskip})", flush=True)
            epoch += 1
            first = 0
            perm = np.random.default_rng(epoch).permutation(len(tr))
        model.save_pretrained(mdir)
        tok.save_pretrained(mdir)
        open(os.path.join(mdir, "done"), "w").write("ok")
        print(f"training done ({time.time() - t0:.0f}s)", flush=True)
    if a.train_only:
        return
    model.eval()
    for split, fname in (("val", a.val_file), ("test", a.test_file)):
        if (a.val_only and split != "val") or (a.test_only and split != "test"):
            continue
        if split == "test" and a.test_file != "pairs_test.parquet":
            split = "testband"
        df = pd.read_parquet(os.path.join(a.pkg, fname))
        if a.smoke:
            df = df.head(a.smoke_n)
        tt = load_texts(a.pkg, "train" if split == "val" else "test")
        A, B = pair_texts(tt, df)
        del tt
        chunk = 500000
        for ci, s in enumerate(range(0, len(df), chunk)):
            path = os.path.join(out, f"ce_{split}_{ci:04d}.parquet")
            e = min(s + chunk, len(df))
            if os.path.exists(path):
                try:  # skip only complete, readable chunks
                    if len(pd.read_parquet(path)) == e - s:
                        continue
                except Exception:  # noqa: BLE001
                    pass
                print(f"[{split}] chunk {ci} incomplete, redoing", flush=True)
            ln = np.array([len(A[j]) + len(B[j]) for j in range(s, e)])
            order = s + np.argsort(ln, kind="stable")
            vals = np.empty(e - s, np.float32)
            with torch.inference_mode():
                for i in range(0, len(order), a.score_bs):
                    idx = order[i:i + a.score_bs]
                    enc = encode(tok, [A[j] for j in idx], [B[j] for j in idx], a.max_len, dev)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                        p = torch.sigmoid(model(**enc).logits.squeeze(-1).float())
                    vals[idx - s] = p.cpu().numpy()
            res = df.iloc[s:e][["q", "src", "d"]].copy()
            res["ce"] = vals
            res.to_parquet(path + ".tmp", index=False)
            os.replace(path + ".tmp", path)
            print(f"[{split}] scored {e}/{len(df)} ({time.time() - t0:.0f}s)", flush=True)
        allp = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(out, f"ce_{split}_*.parquet")))],
                         ignore_index=True)
        allp.to_parquet(os.path.join(out, f"ce_{split}.parquet"), index=False)
        print(f"wrote {out}/ce_{split}.parquet ({len(allp)} rows)", flush=True)
    print(f"ALL DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
