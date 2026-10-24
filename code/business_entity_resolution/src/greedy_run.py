"""Greedy forward selection of feature groups for the final stack (see greedy_stack.py).

A group is kept when the mean of validation F0.5 at decoy x1 and x2 (threshold 0.75 rule) improves by more
than --min-gain. Prints every trial and the final feature set.
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import greedy_stack as g  # noqa: E402

COLL = ["c_qname", "c_dname", "c_qaddr", "c_daempty", "c_samecore", "c_qnumst", "c_rname_d", "c_rname_q", "c_raddr_d",
        "c_raddr_q", "c_ratio_name", "c_ratio_addr", "c_raw_self", "c_raw_exact", "c_raw_twin", "c_raw_twin_exact",
        "c_raw_margin", "nm_tset", "ad_tset", "num1_eq", "d_oov_frac", "r_cnt", "abbr_ratio"]
BASE = ["lp3"] + [f"l_{m}" for m in ("mini", "e5b", "e5l", "e5l2", "labse", "e5lv10")] + COLL
GS = lambda *names: [f"{n}_{s}" for n in names for s in ("gmax", "gap", "grank")]  # noqa: E731
GROUPS = [
    ("stage2+role", ["lp2", "lpra"]),
    ("french CEs", ["l_fr", "l_fr2"]),
    ("dense", ["dn_rrank", "dn_frank", "dn_score"] + GS("dn_score")),
    ("retrieval", ["ret_score", "ret_rank", "r_rank", "r_best_other", "r_score_ratio", "n_sib", "exp_score", "rev_rank",
                   "rev_score", "rr_rank", "rr_score", "p1", "g_n"] + GS("ret_score", "p1")),
    ("name", ["nm_ratio", "nm_tsort", "nm_partial", "nm_jw", "nm_wr", "cmp_ratio", "cmp_partial", "cmp_jw", "nmc_tset",
              "nmc_ratio", "nm_extra_d", "nm_extra_q", "d_oov_cnt", "d_cmp_in_voc", "alt_tset", "legal_ratio",
              "legal_q_empty", "legal_d_empty", "d_is_domain", "q_nm_len", "d_nm_len", "q_nm_ntok", "d_nm_ntok"]
     + GS("cmp_ratio", "nm_tset", "nm_ratio", "nmc_tset")),
    ("address", ["ad_ratio", "ad_tsort", "ad_partial", "ad_core_tset", "ad_core_tsort", "aw_tset", "aw_tsort", "state_eq",
                 "d_aempty", "d_anonlatin", "d_nonlatin"] + GS("aw_tset", "ad_tset")),
    ("numbers", ["num_tset", "num1_lev", "num1_prefix", "num1_logdiff", "num_shared", "num_q_only", "num_d_only",
                 "num_q_empty", "num_d_empty"]),
    ("anchor", ["anc_p1", "anc_nm_tset", "anc_ad_tset", "anc_num1_eq", "anc_cmp_ratio", "anc_aw_tset"] + GS("anc_ad_tset")),
    ("tokens", [f"t_{t}_{k}" for t in ("n", "nb", "c", "a", "h", "p") for k in ("sw", "sc", "qc", "dc", "jac")]
     + GS("t_n_qc", "t_a_qc", "t_h_sw", "t_c_qc")),
    ("misc", ["s_g_n", "s_g_strong", "is_s3"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gain", type=float, default=0.00002)
    a = ap.parse_args()
    if os.environ.get("MATRIX"):
        g.load(os.environ["MATRIX"])
    cols = set(g.columns() + ["lp2", "lp3", "lpra"])
    feats = [f for f in BASE if f in cols]
    res, _ = g.run("BASE (current stack)", feats)
    best = (res["thr75x1"] + res["thr75x2"]) / 2
    for name, grp in GROUPS:
        add = [f for f in grp if f in cols and f not in feats]
        if not add:
            continue
        res, _ = g.run(f"+ {name} ({len(add)})", feats + add)
        sc = (res["thr75x1"] + res["thr75x2"]) / 2
        if sc > best + a.min_gain:
            feats, best = feats + add, sc
            print(f"   KEEP {name}: mean {best:.5f}", flush=True)
    print(f"FINAL mean {best:.5f} with {len(feats)} features", flush=True)
    json.dump(feats, open(os.environ.get("FEATS_OUT", "work10_sim/train/greedy_feats.json"), "w"))


if __name__ == "__main__":
    main()
