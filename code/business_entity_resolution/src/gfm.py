"""Exact expected-F0.5 decision per S1 row (General F-measure Maximizer, Dembczynski et al. 2011;
Waegeman et al. 2014, JMLR 15:3513).

The metric is the instance-wise F-beta of multi-label classification: for one S1 row with s true matches
(retrieved or not) and k predicted records, F = (1+b2) * TP / (b2 * s + k), with 0/0 = 1 (singleton, empty
prediction). The Bayes-optimal prediction needs only
    P_is = P(candidate i is a match, the row has s true matches in total)   and   P0 = P(s = 0):
    Delta_ik = sum_s (1+b2) * P_is / (b2 * s + k),  E[F(top-k by Delta_.k)] = sum of the k largest Delta_.k,
    E[F(empty)] = P0; predict the k with the largest expectation.
s counts every true match, so retrieval misses / the candidate cutoff are handled through P_is and P0.

joint_independent() builds P_is from pair probabilities assuming independent labels plus a Poisson number of
matches missing from the candidate set (the exact version of decide.py's ratio-of-expectations "ef" rule);
a learned P(s | match) or a learned row-level P0 can be plugged in instead.

Usage:
    from gfm import gfm_select
    sel = gfm_select(q, p, lam=0.03)                  # q sorted ascending, p after exclusivity
    sel = gfm_select(q, p, lam=lam_row, p0=p0_row)    # learned singleton probability per row
"""
import numpy as np

B2 = 0.25  # beta^2 for F0.5


def _pad(q, p, m_max):
    """Rows of q (sorted ascending) -> dense (R, M) matrix of probabilities sorted descending, plus index maps."""
    first = np.r_[True, q[1:] != q[:-1]]
    gid = np.cumsum(first) - 1
    order = np.lexsort((-p, gid))  # within each row, highest p first
    g, ps = gid[order], p[order]
    start = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    pos = np.arange(len(g)) - np.repeat(start, np.diff(np.r_[start, len(g)]))
    R = int(g.max()) + 1 if len(g) else 0
    M = int(min(m_max, pos.max() + 1)) if len(g) else 1
    keep = pos < M
    P = np.zeros((R, M))
    P[g[keep], pos[keep]] = ps[keep]
    return P, order, g, pos, keep, R, M


def joint_independent(P, lam, s_max=None):
    """P_is for independent labels: P_is = p_i * P(1 + sum_{j != i} y_j + Miss = s), Miss ~ Poisson(lam).
    P: (R, M) probabilities (0 = padding). lam: scalar or (R,). Returns J with shape (R, M, S+1), J[:, :, s]."""
    R, M = P.shape
    S = s_max or (M + 6)
    lam = np.broadcast_to(np.asarray(lam, dtype=np.float64), (R,))
    # Poisson pmf of the missing count, truncated at S
    miss = np.zeros((R, S + 1))
    miss[:, 0] = np.exp(-lam)
    for s in range(1, S + 1):
        miss[:, s] = miss[:, s - 1] * lam / s
    J = np.zeros((R, M, S + 1))
    for i in range(M):
        d = miss.copy()  # distribution of Miss + sum over j != i
        for j in range(M):
            if j == i:
                continue
            pj = P[:, j:j + 1]
            d[:, 1:] = d[:, 1:] * (1 - pj) + d[:, :-1] * pj
            d[:, 0] = d[:, 0] * (1 - pj[:, 0])
        J[:, i, 1:] = P[:, i:i + 1] * d[:, :-1]  # candidate i is a match: s = 1 + others
    return J


def gfm_from_joint(J, p0):
    """J: (R, M, S+1) with J[r, i, s] = P(y_i = 1, s); p0: (R,) = P(s = 0). Returns (R, M) bool selection and E[F]."""
    R, M, S1 = J.shape
    s = np.arange(S1, dtype=np.float64)
    best = np.asarray(p0, dtype=np.float64).copy()  # k = 0
    bestk = np.zeros(R, dtype=np.int64)
    choice = np.zeros((R, M), dtype=bool)
    for k in range(1, M + 1):
        w = (1 + B2) / (B2 * s + k)  # (S+1,)
        delta = J @ w  # (R, M)
        top = np.argsort(-delta, axis=1, kind="stable")[:, :k]
        ek = np.take_along_axis(delta, top, axis=1).sum(axis=1)
        better = ek > best + 1e-12
        if better.any():
            best[better] = ek[better]
            bestk[better] = k
            choice[better] = False
            rows = np.flatnonzero(better)
            choice[rows[:, None], top[better]] = True
    return choice, best


def gfm_select(q, p, lam=0.0, p0=None, m_max=12, chunk=200_000):
    """Boolean mask over pairs (q sorted ascending, p = pair probabilities after exclusivity).
    lam: expected number of true matches outside the candidate set (scalar, or one value per pair's row, taken
    from the first pair of each row). p0: optional learned P(row has no true match), one value per pair's row;
    default is independence: prod(1 - p) * exp(-lam)."""
    q = np.asarray(q)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    P, order, g, pos, keep, R, M = _pad(q, p, m_max)
    first = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    lam_r = np.broadcast_to(np.asarray(lam, dtype=np.float64), q.shape)[first] if np.ndim(lam) else np.full(R, float(lam))
    if p0 is None:
        p0_r = np.prod(1 - P, axis=1) * np.exp(-lam_r)  # padded zeros contribute 1
    else:
        p0_r = np.asarray(p0, dtype=np.float64)[first]
    sel_dense = np.zeros((R, M), dtype=bool)
    for a in range(0, R, chunk):
        b = min(R, a + chunk)
        J = joint_independent(P[a:b], lam_r[a:b])
        sel_dense[a:b], _ = gfm_from_joint(J, p0_r[a:b])
    mask_sorted = np.zeros(len(q), dtype=bool)
    mask_sorted[keep] = sel_dense[g[keep], pos[keep]]
    mask = np.zeros(len(q), dtype=bool)
    mask[order] = mask_sorted
    return mask


def _brute(pvec, lam, p0=None):
    """Brute-force expected F0.5 over all label vectors (for testing)."""
    from itertools import product
    from math import exp, factorial
    m = len(pvec)
    best, bh = -1, None
    for h in product([0, 1], repeat=m):
        e = 0.0
        for y in product([0, 1], repeat=m):
            py = np.prod([pi if yi else 1 - pi for pi, yi in zip(pvec, y)])
            for miss in range(0, 12):
                pm = exp(-lam) * lam ** miss / factorial(miss)
                s = sum(y) + miss
                k = sum(h)
                tp = sum(a * b for a, b in zip(h, y))
                f = 1.0 if (s == 0 and k == 0) else (1 + B2) * tp / (B2 * s + k) if (s + k) > 0 else 1.0
                e += py * pm * f
        if e > best + 1e-12:
            best, bh = e, h
    return best, bh


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    bad = 0
    for trial in range(300):
        m = rng.integers(1, 6)
        pv = np.round(rng.beta(0.6, 0.6, size=m), 3)
        lam = float(rng.choice([0.0, 0.05, 0.3]))
        eb, hb = _brute(list(pv), lam)
        q = np.zeros(m, dtype=np.int64)
        mask = gfm_select(q, pv, lam=lam)
        P, order, g, pos, keep, R, M = _pad(q, pv, 12)
        J = joint_independent(P, np.array([lam]))
        _, eg = gfm_from_joint(J, np.array([np.prod(1 - pv) * np.exp(-lam)]))
        if abs(eg[0] - eb) > 1e-6:
            bad += 1
            print("mismatch", pv, lam, eb, eg[0], hb, mask.astype(int))
    print(f"brute-force check: {300 - bad}/300 rows match the exact optimum")
    # multi-row call with unsorted p inside rows
    q = np.array([0, 0, 0, 1, 1, 2])
    p = np.array([0.6, 0.99, 0.3, 0.55, 0.52, 0.4])
    print("example selection:", gfm_select(q, p, lam=0.0).astype(int))
