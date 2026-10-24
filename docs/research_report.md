# Research report: from LB 0.990352 toward 0.992

Answers `research_brief.md`. Written 27 Sep 2026, around 20:00–20:15 IST.

## Bottom line

1. **0.992 is not reachable tonight by any technique I found.** It needs +0.0017 overall, which is about +0.0019 on India/US at the current France level. That is 30% of the entire remaining India/US gap to the candidate oracle (0.9974), and two thirds of that gap is unresolvable empty-address common names. A realistic target for the next 3 hours is **+0.0002 to +0.0006, giving about 0.9906–0.9910.** The top team is at 0.99181, so rank matters more than the absolute score. **Pick changes by robust validation, not by the public LB.**
2. **The best-grounded new lever is the decision rule itself.** Your metric is exactly the *instance-wise F-measure* of multi-label classification, and that problem has a known Bayes-optimal decision procedure: GFM (General F-measure Maximizer; Dembczyński et al. 2011, Waegeman et al. 2014).
   - The expected-F0.5 rule in `decide.py` is not GFM. It uses a ratio-of-expectations approximation plus a heuristic empty-set term, ∏(1−p)·t.
   - I implemented GFM (`src/gfm.py`, checked against brute force on 300/300 random rows) and an A/B harness (`src/gfm_eval.py`). Both are ready to run.
3. **Your transfer-gap hypothesis about density features is refuted by the data.** I checked this: see §1.
   - What does differ between validation and test is **composition**: the country mix, and US same-name group sizes.
   - A composition-weighted validation is cheap. It may explain the 62% transfer, and you should use it to choose tonight's final submissions.
4. **France's operating point already matches India/US.** Its empty share, matches per S1 and single-match share are all the same. There is no global France threshold fix to make. France gains can only come from better pair scores (round 3).

## 1. What I measured on your data (read-only, from `coll.parquet`, `s1.parquet` and `submissions/v11a`)

| Check | Validation (IN / US) | Test (IN / US) | Meaning |
|---|---|---|---|
| Records per same-name S1 (by `c_qname` 1..6+) | 2.88 2.83 2.84 2.77 2.80 3.98 / 2.86 3.08 2.98 2.93 2.89 3.20 | 2.92 2.77 2.70 2.65 2.67 3.70 / 2.98 2.94 2.89 2.85 2.92 2.97 | No test inflation from the ×2 decoys. Density features are **not** shifted up. |
| Mean `c_ratio_name` | 4.04 / 7.09 | 3.76 / 4.77 | Test is lower, not higher. |
| Mean `c_qname` (same-name S1 rows) | 16.6 / 18.7 | 18.3 / 12.4 | US test population is 38% smaller, so it has smaller name groups. |
| Adversarial AUC, validation vs test, on 17 collective features | — | 0.679 / 0.677 | Driven by `c_qname`, `c_qnumst`, `c_rname_q`, i.e. population-size counts. Without the density features it is still 0.650 / 0.634. |
| Empty-address same-core candidates, share of pairs | 5.5% / 6.1% | 5.0% / 4.8% | The ambiguous slice is rarer on test. |
| Scored S1 rows by country | IN 40% / US 60% | IN 55% / US 45% | The validation mean is US-weighted; test India/US is India-weighted. |
| US S1 rows with a unique name / 30+ same-name group | 54.8% / 14.6% | 60.1% / 10.7% | The US test has fewer hard twin rows. |
| Predicted empty / matches per S1 / one-match rows (v11a) | — | FR 5.78% / 3.365 / 6.3%; IN 5.78% / 3.381 / 6.1%; US 5.77% / 3.389 / 6.0% | France's decisions are calibrated in aggregate. |

**Implication for the transfer gap.** v11a's collective and twin features target twin-heavy rows. Validation over-represents those rows (US-heavy, bigger US name groups), so part of the validation gain would not appear on test. This is testable in minutes: `python src/gfm_eval.py --transfer` prints the collective/twin gain both unweighted and reweighted to the test composition. If the weighted gain is about 0.6× the unweighted one, the gap is explained. From then on, trust only the **test-weighted** numbers.

## 2. Ranked approaches

| # | Approach | Expected gain (overall LB) | Time | New angle vs. "tried and failed" |
|---|---|---|---|---|
| 1 | **GFM exact expected-F0.5 decision**, with a learned P(singleton) | +0.0000 to +0.00025 (roughly 50% chance of a positive result) | 30–45 min, no pipeline rerun | The tried EF rule was an approximation, and its empty-set term was a heuristic. GFM is exact and Bayes-optimal for this metric, even with dependent labels. |
| 2 | **Composition-weighted validation** (importance weighting) for every accept/reject decision and the final pick, plus paired standard errors | 0 directly; protects the private LB | 10 min (built into `gfm_eval.py`) | "Population matching" changed the features. This reweights the *evaluation* instead; the features stay untouched. |
| 3 | **Two-pass stack with component-grouped folds** (leak-free stacked graphical learning) | 0 to +0.00015 (about 30%) | 20–30 min | The −0.0004 result was the leak you diagnosed. Folds split by connected component remove it. |
| 4 | **France: round-3 CE into the France-safe stack**, keeping the decision rule | +0.0001 to +0.00015 | Already running | Evidence from round 1: about +0.002 France. |
| 5 | **Smaller candidate set** (stage-3 cutoff 0.02 → 0.03–0.05) | About −0.00002 F; helps the "smaller candidate set" criterion | 10 min | — |
| 6 | Re-add the count-prior stack features, **but only under GFM** | 0 to +0.0001 | 10 min after #1 | These improved logloss but not F0.5 under a fixed threshold. GFM is the rule that can turn logloss gains into decisions. |
| – | Not tonight: listwise record-centric CE, GNN post-processing, more CEs, EM label-shift correction, France collective features | — | — | See §4. |

### #1 GFM: the Bayes-optimal decision for per-row F0.5

**Why it applies.** For one S1 row with s true matches (retrieved or not) and k predicted records:

F0.5 = 1.25·TP / (0.25·s + k), with 0/0 = 1.

Waegeman et al. (JMLR 2014, arXiv:1310.4849), Theorem 5.1, show that the expected-F maximizer needs only two things:
- P_is = P(y_i = 1, s), the probability that candidate i is a match *and* the row has s true matches;
- P(s = 0), the probability that the row is a singleton.

Then:
- Δ_ik = Σ_s 1.25·P_is / (0.25·s + k);
- E[F(best set of size k)] = the sum of the k largest Δ_·k;
- E[F(∅)] = P(s = 0);
- predict the argmax over k. Inference is O(m²) per row.

This is Bayes-optimal *regardless of label dependence*. Because s counts matches outside the candidate set, retrieval misses are handled correctly.

**Why the current EF rule underperforms.** It approximates E[F] by 1.25·Σp / (k + 0.25·(Σp + m)), which is a ratio of expectations. That approximation is biased when s is small (about 3.5 here), and it prices the empty set with ∏(1−p)·t.

Ye et al. (ICML 2012, arXiv:1206.4625) find two things:
- thresholding (EUM) is more robust to a misspecified model;
- the exact decision-theoretic rule is better "given a good model… for rare classes and a common domain adaptation scenario".

Your stack is a good model: logloss 0.039, 2-fold out-of-fold.

**Two cautions from your own results.**

1. **Your threshold is already close to optimal at the population level.** For calibrated probabilities, adding an item helps iff p > F*/(1+β²), which is 0.8·F* ≈ 0.79 for β = 0.5 and F* ≈ 0.99. This extends Lipton et al. 2014 (arXiv:1402.1892), who show the F1 optimum is at F*/2. So gains can only come from two places:
   - per-row adaptivity: rows with a lower F* want a lower threshold;
   - the empty-set decision.
2. **Assuming independent labels reproduces your failed "rescue" rule.**
   - Example: a row with candidates at 0.55 and 0.52 gets both predicted, because ∏(1−p) is only 0.22.
   - Your rescue experiment shows P(singleton) is much higher than that in such rows: the labels are correlated, for example sibling decoys that are wrong together.
   - So use a **learned** row-level P(singleton), or let GFM only size rows that the threshold already fills.

**What is ready.**
- `src/gfm.py`: `gfm_select(q, p, lam, p0)`, vectorized and chunked. It pads rows to the top 12 candidates, and uses Poisson(λ) for true matches outside the candidate set.
- `src/gfm_eval.py`: rebuilds the v11a India/US stack matrix exactly as `final_v11.py` does and computes 2-fold out-of-fold p. It then scores five rules under decoy ×1 and ×2:

| Variant | What it does |
|---|---|
| `thr75` | Current rule (baseline). |
| `gfm_indep` | GFM with independent labels. Expected to lose, like the rescue rule. |
| `gfm_nonempty` | Sizes only rows that thr75 already predicts as non-empty. The safest variant. |
| `gfm_p0` | Uses a learned P(singleton): a 2-fold row-level LightGBM on max/second/sum of p, counts above 0.5 and 0.75, max lp3/lpra, `c_qname`, `c_qaddr`, `c_qnumst`, `c_daempty` and `c_samecore`. |
| `gfm_p0_nonempty` | The learned P(singleton) may empty a row but never rescues one. This targets predicted singleton rows, which score 0. |

For each variant it prints F, the paired Δ vs thr75 ± SE, the test-weighted Δ ± SE, and Δ by country. λ is estimated per country × same-name bucket from validation.

**How to run it.** From `code/business_entity_resolution`:
```
python src/gfm_eval.py                # ~3 GB RAM, a few minutes (wait for other jobs to free RAM)
python src/gfm_eval.py --transfer     # also prints the collective/twin gain, unweighted vs test-weighted
```

**Accept rule.** Take a variant only if **Δ ≥ 2 SE on both ×1 and ×2 and on the test-weighted score.** With about 107k scored validation rows, a paired SE is roughly 0.00005–0.0001. Several past "+0.0001" gains were within about 1 SE.

**Test-time path, if a variant wins.**
1. Fit P(singleton) on all validation rows, using the same row features computed from the final stack's p.
2. Apply it to `work10/test/test_predv11inus.parquet`, together with the λ table by country × bucket (test buckets come from the test `s1.parquet` name groups).
3. Apply soft exclusivity, then `gfm_select`, then `decide.write_outputs`.
4. Hybridise with France exactly as `final_v11.sh` does.

About 60 lines, mostly copied from `gfm_eval.py`. Change the India/US part only in the first upload, so the LB delta is attributable.

**France.** Run the same A/B with the France-safe stack (`base="role"`, features as in `final_v11.py`) on the India/US proxy, against the current hard-exclusivity EF with t = 2. Adopt it for France only if it wins there.

### #2 Composition-weighted validation and paired standard errors

**Method.** Importance-weighted validation: Sugiyama, Krauledat & Müller, JMLR 2007, "covariate shift adaptation by importance-weighted cross validation", and the Kaggle "adversarial validation" practice. Weight each validation S1 row by

w = P_test(country, same-name bucket) / P_val(country, bucket),

with buckets 1, 2, 3–4, 5–9, 10–29 and 30+. Report Σw·F / Σw. It is exact for the macro metric because the metric is an average over S1 rows. `gfm_eval.py` already does this for every variant.

**Use.**
1. Run `--transfer`. If the weighted collective/twin gain is about 0.6× the unweighted one, the 62% transfer is composition, not a model defect.
2. Rank every candidate change tonight by the *weighted* Δ, and require Δ ≥ 2 SE.
3. Report per-country Δs. The test is 55% India, and the brief shows India and US behave differently (0.99177 vs 0.99077).

**Optional.** Pass the same weights as `weight=` when fitting the final stack (importance-weighted ERM, Shimodaira 2000). The expected gain is small because LightGBM is flexible. Keep it only if the test-weighted score improves.

### #3 Leak-free two-pass stack (stacked graphical learning)

**Why it failed before.** `greedy_stack.run2` builds neighbourhood features from first-pass out-of-fold predictions, but the folds are a hash of q. A record's competing S1 rows therefore land in the other fold. Their first-pass predictions come from a model trained on this pair's label, which is the leak you saw.

Stacked graphical learning (Kou & Cohen 2007) requires every neighbour prediction used as a feature to come from a model that never saw the target's neighbourhood. Recent graph-stacking work stresses the same leakage control (Chen et al. 2022, arXiv:2206.08473).

**Fix.** Compute folds per connected component of the bipartite candidate graph, with S1 rows on one side and records on the other:
```python
import hashlib
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

def component_folds(df, min_p=None):
    """2-fold ids by connected component of the S1-record candidate graph (every pair of a component in one fold)."""
    e = df if min_p is None else df[df["p3"].values > min_p]      # thin edges only if one component is huge
    rk = e["src"].values.astype(np.int64) * (1 << 32) + e["d"].values
    uq, qi = np.unique(e["q"].values, return_inverse=True)
    ur, ri = np.unique(rk, return_inverse=True)
    n = len(uq) + len(ur)
    A = sp.coo_matrix((np.ones(len(qi)), (qi, len(uq) + ri)), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    comp_q = pd.Series(lab[: len(uq)], index=uq)                   # nodes 0..len(uq)-1 are S1 rows
    q = df["q"].values
    comp = comp_q.reindex(q).values
    comp = np.where(np.isnan(comp), -1 - q, comp).astype(np.int64)  # S1 rows without kept edges: own component
    sizes = pd.Series(comp).value_counts()
    print(f"components {len(sizes)}, largest holds {sizes.iloc[0] / len(df):.1%} of pairs")
    return np.array([hashlib.blake2b(str(c).encode(), digest_size=2).digest()[0] & 1 for c in comp])
```
Then:
1. Set `_CACHE["fold"] = component_folds(df)` after `g.load()`.
2. Rerun `g.run("base", feats)` and `g.run2("2pass", feats)` on the **same** folds.
3. If the largest component holds more than 40% of pairs, rebuild with `min_p=0.1`. The residual leak through edges with p < 0.1 is negligible.

**Expected gain is modest.** Your leak-free raw-evidence message passing gave 0. Predictions summarise more than raw evidence does, though, and under GFM probability gains can become decisions.

### #4 France

- **No global threshold fix.** The operating point already matches India/US. This is a moment check against the generator constants, in the spirit of black-box label-shift estimation (Lipton et al. 2018), so don't spend an upload on a France threshold.
- **Round 3.** Stack it into the France-safe stack. Accept it only if both hold:
  - the India/US proxy improves by at least 2 SE;
  - the audit looks like round 1: it adds initialisms and renames, drops legal-form-dropped empty-address pairs, and changes a moderate share of rows (v10 changed 11.8%).
- **Keep collective features out of France.** Your Pessac audit is the right evidence. Label-free ensemble estimators (the spectral meta-learner and Dawid–Skene-style estimators; Parisi et al. 2014, arXiv:1303.3257; Jaffe et al. 2015, arXiv:1407.7644) could re-weight France's CEs without labels. They assume conditionally independent models, which your correlated e5 family breaks, so skip them tonight.

### #5 Candidate-set size (organizer criterion)

Sweep the stage-3 cutoff `minp` over {0.02, 0.03, 0.05, 0.08} in `build_split`, retrain the stack on each set and score it. Pick the largest cutoff whose loss is at most 0.00002 (test-weighted).

Matches are 3.37 per S1 against 3.79 candidates, and GFM never selects candidates with p < 0.05. Tightening therefore costs almost nothing and may break ties among the roughly 20 teams at 0.991 or above.

## 3. Suggested schedule (19:55 → 23:25 work, then zip)

1. **Now → 20:40:** run `gfm_eval.py --transfer` once RAM is free. Decide on #1 (the accept rule above) and read the transfer diagnosis (#2).
2. **20:40 → 21:20:** if a GFM variant wins, build the test path and **upload 1**: India/US decision change only, France untouched.
3. **In parallel (Colab):** France round 3 → France-safe stack → audit.
4. **21:20 → 22:00:** #3 component folds, and #6 count-prior features under GFM. Keep only 2-SE, test-weighted winners.
5. **Upload 2:** France round 3, if the audit passes, on top of the best India/US.
6. **Upload 3:** reserve for the final combination, or a fallback. Choose the final submission by the test-weighted validation, with the LB as a no-regression check.
7. **22:45:** freeze, then apply the #5 cutoff to the final build. **23:25:** zip.

## 4. What I would not do tonight, and why

- **More or other cross-encoders, or a Qwen/LLM pair classifier.** Saturated in your data (+0.00001 to +0.00008).
- **A listwise, record-centric CE**, reading the record against all its S1 candidates plus "none". It is the right inductive bias for the "record of another present S1" FPs (+0.00076) and orphans (+0.00075), but with manual Colab steps it needs at least 2.5 h, which doesn't fit.
- **GNN post-processing.** In Foursquare Location Matching, a GNN node-classification stage took one top team from 0.907 to 0.946. That was many-to-many clustering inside one dataset with lots of transitive links. Your bipartite, exclusive structure is already exploited by exclusivity and sibling expansion.
- **Neighbourhood-blending / "min-2" tricks.** From Shopee's 1st place: min2 gave +0.019 LB and neighbourhood blending +0.008. Your analogues are sibling query expansion (done) and rescue (failed). GFM with a learned P(singleton) is the principled version of "min-k".
- **EM label-shift correction** (`label_shift.py`, Saerens 2002). Alexandari et al. 2020 (arXiv:1901.06852) show EM is fragile unless calibration is bias-corrected. Your threshold is already robust to decoy ×1/×2/×3, and on France it cannot be validated.
- **Dangling-entity detectors** (Sun et al. ACL 2021, arXiv:2106.02248: margin- and background-ranking abstention). These are equivalent to your competitor-margin features, which gave 0.
- **Recomputing collective features for the test population.** The data shows the drift is composition, and the per-k conditional is invariant, as you argued. Reweight the evaluation instead (#2).

## Sources (retrieved with Firecrawl)

- Waegeman, Dembczyński et al., *On the Bayes-optimality of F-measure maximizers*, JMLR 2014 — https://arxiv.org/abs/1310.4849 (GFM, Theorem 5.1, Algorithm 1; P_is with Δ = P·W)
- Ye, Chai, Lee, Chieu, *Optimizing F-measures: a tale of two approaches*, ICML 2012 — https://arxiv.org/abs/1206.4625 (exact O(n²) expected-F under independence; EUM vs DTA)
- Lipton, Elkan, Naryanaswamy, *Thresholding classifiers to maximize F1*, 2014 — https://arxiv.org/abs/1402.1892
- Natarajan et al., *Optimal decision-theoretic classification using non-decomposable metrics* — https://arxiv.org/abs/1505.01802 (probability ranking principle, which justifies top-k sets)
- Alexandari, Kundaje, Shrikumar, *Maximum likelihood with bias-corrected calibration is hard-to-beat at label shift adaptation*, ICML 2020 — https://arxiv.org/abs/1901.06852; Garg et al., *A unified view of label shift estimation* — https://arxiv.org/abs/2003.07554
- Chen et al., *A robust stacking framework for training deep graph models with multifaceted node features* (leakage control in graph stacking) — https://arxiv.org/abs/2206.08473
- Sun et al., *Knowing the no-match: entity alignment with dangling cases*, ACL 2021 — https://arxiv.org/abs/2106.02248
- Parisi et al., *Ranking and combining multiple predictors without labeled data* — https://arxiv.org/abs/1303.3257; Jaffe et al., *Estimating the accuracies of multiple classifiers without labeled data* — https://arxiv.org/abs/1407.7644
- Shopee Price Match, 1st-place write-up (min2, iterative neighbourhood blending, gains per step) — https://www.kaggle.com/competitions/shopee-product-matching/writeups/upstage-making-ai-beneficial-1st-place-solution-fr
- Foursquare Location Matching winners summary (GNN post-processing 0.907→0.946; three-stage LightGBM + transformer stacks) — https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/; 13th place (GNN) — https://www.kaggle.com/competitions/foursquare-location-matching/writeups/team-merge-master-13th-place-solution-gnn
- Adversarial validation (Kaggle) — https://www.kaggle.com/code/carlmcbrideellis/what-is-adversarial-validation; Splink term-frequency adjustments (the Fellegi–Sunter rarity principle behind your collective features) — https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html
- Domain adaptation for entity matching, for France after the deadline: DAME — https://arxiv.org/abs/2204.09244; Cycle Self-Training — https://arxiv.org/abs/2103.03571
- Cited from memory, not re-fetched: Dembczyński et al., *An exact algorithm for F-measure maximization*, NIPS 2011; Sugiyama, Krauledat & Müller, *Covariate shift adaptation by importance weighted cross validation*, JMLR 2007; Kou & Cohen, *Stacked graphical models*, SDM 2007; Shimodaira 2000.
