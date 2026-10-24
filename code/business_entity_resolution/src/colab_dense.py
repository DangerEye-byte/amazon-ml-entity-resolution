"""Dense retrieval on Google Colab: fine-tune a multilingual-e5 bi-encoder, then retrieve candidates.

Recovers matches the token retrieval never finds: renamed records linked only by
their address, names written in another script, and empty-address records whose
name lost or gained words. Package from src/dense_export.py.

Steps (resumable; everything lands in <pkg>/<out-name>/):
 1. choose one match per fine-tuning S1 row (the least similar name for --hard-frac of them)
    and mine one hard negative S1 per pair with the base model (same country)       -> mined.parquet
 2. fine-tune: symmetric InfoNCE, in-batch + hard negatives                          -> model/
 3. per split and country: encode every record, search record -> S1 top --k-rev (reverse)
    and S1 -> record top --k-fwd (forward). Train: the S1 index excludes the simulated
    19% deletion (pairs of every remaining S1 row are kept: the reverse-competition
    features count candidates over all S1 rows, as on test)                         -> dense_<split>.parquet
    columns q, src, d, rrank, frank (99 = not in that list), score (cosine)
 4. recall of validation matches, overall and on those the token retrieval missed.

Usage (Colab cell):
    !python /content/drive/MyDrive/dense_pkg/colab_dense.py --pkg /content/drive/MyDrive/dense_pkg
Smoke test (CPU, tiny subset):
    python src/colab_dense.py --pkg dense_pkg --smoke
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL = "intfloat/multilingual-e5-small"
KEY = lambda q, s, d: np.asarray(q, np.int64) * 8 * (1 << 32) + np.asarray(s, np.int64) * (1 << 32) + np.asarray(d, np.int64)  # noqa


def log(t0, msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def load_txt(pkg, split, k):
    df = pd.read_parquet(os.path.join(pkg, f"txt_{split}_s{k}.parquet"))
    return np.array(["query: " + t for t in df["text"].tolist()], dtype=object), df["ck"].values


class Encoder:
    def __init__(self, path, dev, max_len):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(dev)
        self.dev, self.max_len = dev, max_len
        self.dt = torch.float16 if dev == "cuda" else torch.float32

    def embed(self, texts):
        enc = self.tok(list(texts), truncation=True, max_length=self.max_len, padding=True,
                       return_tensors="pt").to(self.dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.dev == "cuda"):
            h = self.model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        return F.normalize(((h * m).sum(1) / m.sum(1).clamp(min=1)).float(), dim=-1)

    @torch.inference_mode()
    def encode(self, texts, bs):
        self.model.eval()
        order = np.argsort(np.fromiter((len(t) for t in texts), np.int32, len(texts)), kind="stable")
        out = torch.empty((len(texts), self.model.config.hidden_size), dtype=self.dt)
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            out[torch.from_numpy(idx)] = self.embed(texts[idx]).to(self.dt).cpu()
        return out


def search(E1, R, k_rev, k_fwd, chunk, dev):
    """Cosine top-k both ways. E1: (n1, h) S1 embeddings on dev, R: (nr, h) record embeddings (CPU)."""
    n1, nr = E1.shape[0], R.shape[0]
    kr, kf = min(k_rev, n1), min(k_fwd, nr)
    rev_i = torch.empty((nr, kr), dtype=torch.int32)
    rev_v = torch.empty((nr, kr), dtype=torch.float32)
    fv = torch.full((n1, kf), -10.0, device=dev, dtype=E1.dtype)
    fi = torch.full((n1, kf), -1, device=dev, dtype=torch.int64)
    for s in range(0, nr, chunk):
        Rc = R[s:s + chunk].to(dev)
        S = Rc @ E1.T
        v, i = S.topk(kr, dim=1)
        rev_i[s:s + len(Rc)] = i.int().cpu()
        rev_v[s:s + len(Rc)] = v.float().cpu()
        del S
        St = E1 @ Rc.T
        v, i = St.topk(min(kf, St.shape[1]), dim=1)
        del St
        fv, sel = torch.cat([fv, v], dim=1).topk(kf, dim=1)
        fi = torch.cat([fi, i + s], dim=1).gather(1, sel)
    return rev_i.numpy(), rev_v.numpy(), fi.cpu().numpy(), fv.float().cpu().numpy()


def choose_pairs(pos, n, hard_frac, rng):
    """One match per S1 row: the least similar name for hard_frac of the rows, a random one otherwise."""
    pos = pos.assign(r=rng.random(len(pos)))
    hard = pos.sort_values(["q", "nsim", "r"]).drop_duplicates("q")
    rnd = pos.sort_values(["q", "r"]).drop_duplicates("q")
    use_hard = rng.random(len(hard)) < hard_frac
    ch = pd.concat([hard[use_hard], rnd[~rnd["q"].isin(hard["q"].values[use_hard])]])
    return ch.sample(min(n, len(ch)), random_state=0)[["q", "src", "d", "nsim"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default="dense_pkg")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out-name", default="out_dense")
    ap.add_argument("--n-train", type=int, default=1200000)
    ap.add_argument("--hard-frac", type=float, default=0.5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--scale", type=float, default=20.0, help="inverse temperature of the InfoNCE logits")
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--enc-bs", type=int, default=1024)
    ap.add_argument("--k-rev", type=int, default=3)
    ap.add_argument("--k-fwd", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=4096, help="records per search block (lower on small GPUs)")
    ap.add_argument("--splits", nargs="*", default=["train", "test"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_train, a.bs, a.enc_bs, a.chunk = 256, 32, 64, 256
        if a.out_name == "out_dense":
            a.out_name = "out_dense_smoke"
    out = os.path.join(a.pkg, a.out_name)
    mdir = os.path.join(out, "model")
    os.makedirs(out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    log(t0, f"device {dev} {torch.cuda.get_device_name(0) if dev == 'cuda' else ''}")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    meta = pd.read_parquet(os.path.join(a.pkg, "s1meta_train.parquet"))
    sample, drop = meta["sample"].values, meta["drop"].values
    tr_txt = {k: load_txt(a.pkg, "train", k) for k in (1, 2, 3)}
    log(t0, "train texts loaded")

    # ---- 1-2. mining + fine-tuning
    if not os.path.exists(os.path.join(mdir, "done")):
        enc = Encoder(a.model, dev, a.max_len)
        mined_path = os.path.join(out, "mined.parquet")
        if os.path.exists(mined_path):
            ch = pd.read_parquet(mined_path)
        else:
            pos = pd.read_parquet(os.path.join(a.pkg, "pos_train.parquet"))
            if a.smoke:
                u = pos["q"].drop_duplicates()
                pos = pos[pos["q"].isin(u.sample(min(2000, len(u)), random_state=0))]
            ch = choose_pairs(pos, a.n_train, a.hard_frac, rng)
            rec_txt = np.empty(len(ch), dtype=object)
            for k in (2, 3):
                m = ch["src"].values == k
                rec_txt[m] = tr_txt[k][0][ch["d"].values[m]]
            pool = np.flatnonzero(~sample)                      # S1 rows the model may learn from
            if a.smoke:
                pool = np.union1d(ch["q"].values, rng.choice(pool, min(3000, len(pool)), replace=False))
            EA = enc.encode(rec_txt, a.enc_bs)
            EP = enc.encode(tr_txt[1][0][pool], a.enc_bs)
            log(t0, f"mining: encoded {len(ch)} anchors + {len(pool)} pool S1")
            ck_a, ck_p = tr_txt[1][1][ch["q"].values], tr_txt[1][1][pool]
            neg = np.full(len(ch), -1, np.int64)
            for c in np.unique(ck_a):
                ia, ip = np.flatnonzero(ck_a == c), np.flatnonzero(ck_p == c)
                ri, _, _, _ = search(EP[ip].to(dev), EA[ia], 8, 1, a.chunk, dev)
                cand = pool[ip][ri]                             # (na, 8) S1 rows
                ok = cand != ch["q"].values[ia][:, None]
                first = ok.argmax(1)
                neg[ia] = np.where(ok.any(1), cand[np.arange(len(ia)), first], rng.choice(pool[ip], len(ia)))
            ch["neg"] = neg
            del EA, EP
            ch.to_parquet(mined_path, index=False)
            log(t0, f"mined {len(ch)} pairs (hard share {(ch['nsim'] < 60).mean():.3f})")
        rec_txt = np.empty(len(ch), dtype=object)
        for k in (2, 3):
            m = ch["src"].values == k
            rec_txt[m] = tr_txt[k][0][ch["d"].values[m]]
        pos_txt, neg_txt = tr_txt[1][0][ch["q"].values], tr_txt[1][0][ch["neg"].values]
        qp = torch.from_numpy(ch["q"].values.astype(np.int64)).to(dev)
        qn = torch.from_numpy(ch["neg"].values.astype(np.int64)).to(dev)
        n = len(ch)
        steps = max(1, int(n * a.epochs / a.bs))
        opt = torch.optim.AdamW(enc.model.parameters(), lr=a.lr, weight_decay=0.01)
        warm = max(1, int(0.05 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / steps))
        scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
        enc.model.train()
        perm, run, t1 = np.random.default_rng(1).permutation(n), None, time.time()
        for step in range(steps):
            i0 = (step * a.bs) % n
            idx = perm[i0:i0 + a.bs]
            b = len(idx)
            E = enc.embed(np.concatenate([rec_txt[idx], pos_txt[idx], neg_txt[idx]]))
            A, P, N = E[:b], E[b:2 * b], E[2 * b:]
            ti = torch.from_numpy(idx).to(dev)
            tgt = torch.arange(b, device=dev)
            # record -> S1: in-batch positives + hard negatives; mask other copies of the true S1
            l1 = a.scale * A @ torch.cat([P, N]).T
            same = torch.cat([qp[ti], qn[ti]])[None, :] == qp[ti][:, None]
            same[tgt, tgt] = False
            l1 = l1.masked_fill(same, -1e4)
            # S1 -> record
            l2 = a.scale * P @ A.T
            loss = 0.5 * (F.cross_entropy(l1, tgt) + F.cross_entropy(l2, tgt))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(enc.model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            lv = float(loss.detach())
            run = lv if run is None else 0.98 * run + 0.02 * lv
            if (step + 1) % 100 == 0 or step + 1 == steps:
                rate = (step + 1) * a.bs / (time.time() - t1)
                log(t0, f"step {step + 1}/{steps} loss {run:.4f} ({rate:.0f} pairs/s, eta "
                        f"{(steps - step - 1) * a.bs / rate / 60:.0f} min)")
        enc.model.save_pretrained(mdir)
        enc.tok.save_pretrained(mdir)
        open(os.path.join(mdir, "done"), "w").write("ok")
        log(t0, "fine-tuning done")
        del enc, opt
        if dev == "cuda":
            torch.cuda.empty_cache()
    enc = Encoder(mdir, dev, a.max_len)
    log(t0, "fine-tuned model loaded")

    # ---- 3. retrieval
    for split in a.splits:
        final = os.path.join(out, f"dense_{split}.parquet")
        if os.path.exists(final):
            log(t0, f"{final} exists, skipping")
            continue
        txt = tr_txt if split == "train" else {k: load_txt(a.pkg, split, k) for k in (1, 2, 3)}
        n1 = len(txt[1][0])
        in_index = ~drop if split == "train" else np.ones(n1, bool)
        rows = {k: np.arange(len(txt[k][0])) for k in (1, 2, 3)}
        if a.smoke:  # tiny consistent subset: some S1 rows, their matches, random records
            pv = pd.read_parquet(os.path.join(a.pkg, "pos_val.parquet")).head(400)
            rows[1] = np.union1d(pv["q"].values, rng.choice(n1, 2000, replace=False)) if split == "train" \
                else rng.choice(n1, 2000, replace=False)
            for k in (2, 3):
                own = pv["d"].values[pv["src"].values == k] if split == "train" else np.array([], int)
                rows[k] = np.union1d(own, rng.choice(len(txt[k][0]), 2000, replace=False))
        parts = []
        for c in np.unique(txt[1][1]):
            pp = os.path.join(out, f"dense_{split}_{c}.parquet")
            if os.path.exists(pp):
                parts.append(pd.read_parquet(pp))
                continue
            q_idx = rows[1][(txt[1][1][rows[1]] == c) & in_index[rows[1]]]
            r_src, r_d = [], []
            for k in (2, 3):
                rk = rows[k][txt[k][1][rows[k]] == c]
                r_src.append(np.full(len(rk), k, np.int8))
                r_d.append(rk)
            r_src, r_d = np.concatenate(r_src), np.concatenate(r_d)
            if len(q_idx) == 0 or len(r_d) == 0:
                continue
            E1 = enc.encode(txt[1][0][q_idx], a.enc_bs).to(dev)
            R = torch.cat([enc.encode(txt[k][0][r_d[r_src == k]], a.enc_bs) for k in (2, 3)])
            log(t0, f"[{split}/{c}] encoded {len(q_idx)} S1 + {len(r_d)} records")
            ri, rv, fi, fv = search(E1, R, a.k_rev, a.k_fwd, a.chunk, dev)
            del E1, R
            kr, kf = ri.shape[1], fi.shape[1]
            rev = pd.DataFrame({"q": q_idx[ri.ravel()].astype(np.int32), "src": np.repeat(r_src, kr),
                                "d": np.repeat(r_d, kr).astype(np.int32),
                                "rrank": np.tile(np.arange(kr, dtype=np.int8), len(r_d)), "frank": np.int8(99),
                                "score": rv.ravel()})
            ok = fi.ravel() >= 0
            fwd = pd.DataFrame({"q": np.repeat(q_idx, kf)[ok].astype(np.int32), "src": r_src[fi.ravel()[ok]],
                                "d": r_d[fi.ravel()[ok]].astype(np.int32), "rrank": np.int8(99),
                                "frank": np.tile(np.arange(kf, dtype=np.int8), len(q_idx))[ok],
                                "score": fv.ravel()[ok]})
            df = pd.concat([rev, fwd], ignore_index=True)
            df = df.groupby(["q", "src", "d"], as_index=False, sort=False).agg(
                rrank=("rrank", "min"), frank=("frank", "min"), score=("score", "max"))
            df["score"] = df["score"].astype(np.float16)
            df.to_parquet(pp + ".tmp", index=False, compression="zstd")
            os.replace(pp + ".tmp", pp)
            parts.append(df)
            log(t0, f"[{split}/{c}] {len(df)} pairs")
        allp = pd.concat(parts, ignore_index=True)
        allp.to_parquet(final + ".tmp", index=False, compression="zstd")
        os.replace(final + ".tmp", final)
        log(t0, f"wrote {final}: {len(allp)} pairs")

        # ---- 4. validation recall
        if split == "train":
            pv = pd.read_parquet(os.path.join(a.pkg, "pos_val.parquet"))
            if a.smoke:
                pv = pv.head(400)
            kp = KEY(pv["q"].values, pv["src"].values, pv["d"].values)
            ka = KEY(allp["q"].values, allp["src"].values, allp["d"].values)
            o = np.argsort(ka)
            ka = ka[o]
            pos = np.clip(np.searchsorted(ka, kp), 0, len(ka) - 1)
            hit = ka[pos] == kp
            rr = np.where(hit, allp["rrank"].values[o][pos], 99)
            fr = np.where(hit, allp["frank"].values[o][pos], 99)
            miss = ~pv["ret"].values
            for lab, m in (("all val matches", np.ones(len(pv), bool)), ("missed by token retrieval", miss)):
                print(f"{lab} ({m.sum()}): " + ", ".join(
                    f"{n_} {v[m].mean():.3f}" for n_, v in (
                        ("rev@1", rr < 1), ("rev@2", rr < 2), ("rev@3", rr < 3),
                        ("fwd@1", fr < 1), ("fwd@5", fr < 5), ("any", hit))), flush=True)
    log(t0, "ALL DONE")


if __name__ == "__main__":
    main()
