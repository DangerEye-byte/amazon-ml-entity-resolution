# Research brief: Amazon ML Challenge 2026 — Business Entity Resolution

**Current public leaderboard (LB): 0.990352. Target: 0.992. Top team: 0.991811; ~20 teams ≥ 0.991.**
Written 27 Sep 2026, ~19:40 IST. Deadline 23:59 IST the same day, with ~30 min needed for the final zip. **3 LB uploads left today.**

## Your task (the question)
Study this brief and do deep research (papers, Kaggle/competition write-ups, record-linkage literature) on techniques that could raise our score from 0.9904 to ≥ 0.992 in the ~3 hours of work time left. Return a document that ranks the approaches you think are best, with:
- the expected gain and the reasoning behind it;
- concrete implementation steps that fit our pipeline and constraints (below);
- how to validate each one on our simulated validation.

**Do not re-propose what is listed under "Tried and failed" unless you explain why a different variant would behave differently.**

---

## 1. Task and scoring
- **Input.** Three sources of business records: `entity_id, business_name, business_address, country`.
  - Source 1 (S1) is the reference.
  - For every S1 row, predict the matching S2/S3 records.
- **Metric.** Macro **F0.5** per S1 row, averaged over all S1 rows (precision-weighted).
  - A true singleton (no matches) scores 1.0 for an empty prediction and 0 otherwise.
  - Any S1 row with true matches but an empty prediction scores 0.
- **Candidate set counts too.** `candidate_pairs.tsv` must be exactly the pairs the final model scores. The organizers say a **smaller candidate set per S1 ranks higher in the final evaluation**, beyond the LB.
- **Final ranking uses the private LB** (the rest of the test set), so robust gains beat public-LB probing.
- **Rules.**
  - No external data, APIs or geocoding.
  - Models must be MIT/Apache-2.0 and ≤ 8B parameters.
- **Compute.**
  - Local: one laptop with 4 CPU cores and 16 GB RAM, running all pipeline stages.
  - Google Colab A100 (40/80 GB) for transformers, driven manually by the user (upload files to Drive, run one cell, download outputs).

## 2. Data facts we established
- **Sizes.**
  - Train: S1 2.21M (India 883k, US 1.32M); S2 5.0M; S3 5.3M.
  - Test S1: 1.73M (India 810k, US 663k, **France 259k, unseen in train**).
  - Test S2+S3: India 4.72M, US 3.82M, France 1.43M records.
- **Generator constants.** Identical in India and US train:
  - 3.46 matches per S1 on average;
  - singleton share 5.58%;
  - matches per source roughly independent: P(n2 = 0) = 7.9%, P(n2 = 1) = 38%, P(n2 = 2) = 31%;
  - empty address on 4.4% of matched records, independently per record.
- **Exclusivity.** Every S2/S3 record matches at most one S1; matches never cross countries.
- **How the test was built.** About 19% of S1 entities were deleted and their records kept, which creates "orphans". Test has 5.8 records per S1 vs 4.7 in train. The test also has about twice as many pure decoys.
- **How we validate.** Hash-drop 19% of train S1 rows. Validation is a hashed 20% of a 30% S1 sample. We score both normal and "decoy ×2" (every decoy candidate counted twice). **The LB tracked weighted decoy-×2 validation within 0.00005 for v10**, but only at 62% of the predicted gain for v11a (see §6).
- **Noise patterns.**
  - Typos, accents, shuffled words, legal-form churn.
  - Names in 9 Indic scripts.
  - Initialisms ("ESD").
  - Domain-style names.
  - **Renamed records.** Pseudo-word names built from a small syllable set (belo, lyra, quo, vera, gild, kor, onyx…) that match only by address.
  - Address abbreviations and house-number typos (728 ↔ 28).
- **Decoys.**
  - The same name at a nearby house number.
  - A different name at the same address.
  - A descriptor or legal-form swap. In France this is a one-word descriptor swap; alias words that do **not** signal a decoy are fils, cie, services, associés, développement, france, groupe.
  - Empty-address copies of common names.
- **Twins.** Several present S1 rows often share a name, since the generator reuses names across cities.
  - In India/US train, an empty-address record with the S1's core name but a dropped legal form is a true match only 29% of the time.
  - Raw-string cues (case, parentheses, legal-form spelling) often identify the owner among twins. Example: `'Choudhary (indit) Private Limited'` belongs to `'Choudhary (india) Private Limited'`, not to `'Choudhary India Private Limited'`.
- **France is structurally different.** French names use a small vocabulary (generic word + city). At equal population size, the mean number of S1 rows sharing a name is about 3.4× India/US (France test 21.7 vs India/US at France's size 6.4). Records sharing a name: 65.8 vs 14.6.
- **Population sizes differ.** Train-present (after the 19% drop) vs test: India 715k vs 810k S1 rows; US 1.07M vs 663k.
- **No leaks.** Checked and found nothing in file row order, entity-ID values (correlation 0.0003, all modular tests at chance) or structural counts.

## 3. Current pipeline (v11a, LB 0.990352)
1. **Normalization.**
   - Learned Indic → Latin dictionary; unidecode; legal forms separated.
   - Address normalization: abbreviations, state codes, house numbers.
2. **Four candidate generators.**
   - IDF-weighted sparse token retrieval, top-20 per source.
   - Sibling query expansion.
   - Reverse retrieval (record → S1 top-3) with fine address re-ranking of near-ties (reverse top-100, keep 3).
   - **Dense bi-encoder** (multilingual-e5-small fine-tuned with InfoNCE plus mined hard negatives), trained only on train S1 rows never used downstream: record → S1 top-3 and S1 → record top-5.

   Blocking recall is 0.9926 (was 0.9830 before dense retrieval).
3. **Pruner.** LightGBM on cheap features, 2-fold out-of-fold, max-loss 0.001: 5.03 candidates per S1.
4. **Stage-2 LightGBM** (127 leaves; trained on 2.1M simulated train pairs, validated on the val rows). Features:
   - ~150 string / number / address / retrieval / group / sibling-anchor features;
   - out-of-fold lexical target encoding (match rate of differing words, legal-form pairs, number-change types);
   - dense rank and score (a top-7 feature);
   - **collective features:**
     - same-name / same-address S1 counts;
     - records per name;
     - density ratios;
     - raw-name "twin" similarity: this S1's raw name vs the closest present same-name S1's raw name, and the margin between them.

   Stage-2 validation logloss went 0.0407 → 0.0388 with the collective features.
5. **Stage-3.** A group-aware LightGBM, cross-fitted on validation.
6. **Candidate cutoff.** Stage-3 p > 0.02 defines `candidate_pairs.tsv`: **3.79 per S1** (vs 5.51 before), costing −0.00002 F0.5.
7. **Final stack.** LightGBM (15 leaves, 300 rounds, 3 bagged seeds, feature fraction 0.7) cross-fitted on validation (~400k pairs). Inputs:
   - stage-2 / stage-3 / role-model logits;
   - 7 cross-encoder logits: MiniLM-L12-multilingual, e5-base, e5-large ×2, LaBSE, e5-large on v10 hard pairs, and a twin-aware e5-large that reads "S1 || closest twin S1" against the record;
   - collective, density and twin features, plus a few pairwise features.

   Cross-encoders score only the uncertain band (0.005 < stage-2 p < 0.995), identically on validation and test.
8. **Decision.** India/US: soft exclusivity with threshold 0.75. This beats expected-F0.5 rules under decoy ×1, ×2 and ×3; per-segment thresholds and a "rescue top candidate" rule were tested and don't help.
9. **France** (no labels):
   - a language-agnostic "word-role" LightGBM, stacked with two e5-large cross-encoders self-trained on French pseudo-labels (rounds 1 and 2);
   - hard exclusivity with expected F0.5 (t = 2);
   - a data-derived filter for French descriptor-swap decoys.

   France's LB contribution is estimated at ~0.989.

## 4. Score history
| Version | Main change | Validation India/US (decoy ×1 / ×2) | LB |
|---|---|---|---|
| v7h | reverse retrieval, lexical target encoding, France swap filter | 0.9835 | 0.981415 |
| v8ce5 | tie re-rank, stage-3, 5 cross-encoders | 0.98788 / 0.98769 | 0.98611 |
| v8ce5 + France role model | | | 0.986374 |
| v10 | dense retrieval, 6-CE stack, French self-trained CEs, candidate cutoff | 0.99017 / 0.98989 | **0.989766** |
| v11a | collective/density/twin features in stage-2 and stack, twin-aware CE, bagging (France unchanged) | **0.99117 / 0.99100** | **0.990352** |

## 5. What worked (validation gains, India/US)
- **Dense retrieval:** +0.0024 at stage-2.
- **Collective frequency features** (a Fellegi–Sunter-style rarity principle): +0.0002.
- **Density ratios:** +0.0001.
- **Raw-name twin margin:** +0.0003.
- **Collective features inside stage-2 instead of only in the stack:** +0.00015.
- **Bagging the stack:** +0.0001.
- **France:** self-trained French cross-encoders gave +0.0024 on the India/US proxy for the France-safe model, and about +0.002 France on the LB (round 1).

## 6. Tried and failed (please do not re-suggest without a new angle)
- **More or other cross-encoders.** Adding a 6th e5 model: +0.00008. LaBSE: +0.00005. e5-large trained on v10 hard pairs: +0.00001. **Qwen2.5-1.5B fine-tuned as a pair classifier: 0** (correlation 0.84 with e5-large, still no decision gain). mDeBERTa-v3: NaN in bf16, dropped.
- **Stack or stage-2 capacity.**
  - 511-leaf stage-2: worse logloss.
  - Bigger stack (31/63 leaves, more rounds): worse.
  - Decoy-weighted stage-2 (negatives of pure decoys ×2): −0.0001.
- **Collective / graph variants.**
  - Count-prior layer (per-source match-count likelihood ratios), as a separate stage-4: −0.0002. As stack features: logloss better, F0.5 unchanged.
  - Two-pass iterative classification (neighbourhood features from first-pass out-of-fold predictions): −0.0004, a label leak through shared folds.
  - Leak-free "message passing" features (competitors' raw evidence): 0. Most competing twins aren't candidates.
  - Missing-entity footprints (per name, records whose address matches no present same-name S1): 0. The exact address key is too strict.
  - Address twins: 0 (redundant with the address token-set similarity).
- **Decision layer.**
  - Per-segment thresholds (empty address × has-twin): 0.
  - Rescue rule (give empty-predicted S1 rows their top candidate): negative at every threshold.
  - Blending the India/US stack with the France-safe stack: negative.
- **Population matching.** Counting collective features over a subsample sized like the test population: validation −0.00013. On reflection it's theoretically wrong: P(match | k same-name S1 rows present) is population-invariant.
- **France with India/US-learned collective or twin features.** Better on the India/US proxy (+0.0003 / +0.0005), but **the French audit rejects them**. They drop 11–15k French matches that are true alias-word variants at identical addresses ("Pessac Union SCI" ↔ "Pessac SCI Développement"). French name repetition breaks their semantics.
- **Retrieval.** Of the 0.65% of true pairs still never retrieved, 94% are empty-address records, and 1,865 of 2,266 have names shared by 5+ entities, so they are unresolvable. Only ~140 have an address, and those carry just a house number and city.

## 7. Remaining error budget (validation, India/US, current model ≈ 0.9910)
- **Oracle on final candidates: 0.9974.** Most of the gap to 1 is unresolvable empty-address common names.
- **False negatives:**
  - p 0.2–0.5: 2,849 pairs, worth +0.0023;
  - p < 0.2: 1,614 pairs, worth +0.0015;
  - p ≥ 0.5 but not chosen: 1,382 pairs, worth +0.0009.

  About 64% of false negatives are "ambiguous": empty address and a name shared by ≥ 2 present S1 rows. Many have identical-name twins; the owner's twin often isn't even a candidate.
- **False positives:**
  - orphans (owner was deleted): 307 pairs, worth +0.00075;
  - record of another present S1: 273 pairs, worth +0.00076;
  - pure decoys: 173 pairs, worth +0.0005.
- **By country:** India 0.99177, US 0.99077. 0.15–0.17% of S1 rows have real matches but get an empty prediction, costing ~0.0016. The rescue rule shows those empties are nonetheless the right call in expectation.
- **Transfer gap:** v11a's India/US gain transferred at 62% (predicted +0.00094 overall, got +0.00059). v10 transferred ~99%. The cause is unknown; the decision rule was ruled out.

## 8. Ideas not yet done (my ranking)
1. **France self-training round 3** (running): pseudo-labels from the LB-verified v10 France model, including dense-retrieved French pairs; balanced 350k / 350k.
2. **A France-adapted role model** trained on India/US plus French pseudo-labels, so collective features learn France's own name-repetition semantics.
3. **Retrieval of rare-name empty-address misses** (~310 validation pairs): a second dense pass with larger top-k for empty-address records. Needs a pipeline rerun (~4 h), so probably too late.
4. **Understanding the v11a transfer gap.** Which v11 components transfer and which don't; this would need LB probes with 3 uploads left.

## 9. Key files and commands (for concrete proposals)
Code lives in `code/business_entity_resolution/src/`:
- **Stack/search harness:** `greedy_stack.py`. It caches the validation matrix; `build_split(feats, split, base, ...)` builds validation or test matrices; experiments train by 2-fold cross-fit and score out-of-fold F0.5.
- **Collective features:** `stack_ce.collective()`.
- **Stage-2 training and scoring:** `train_lex.py` and `score_lex.py`, with `--extra` for per-pair feature files.
- **Final build:** `final_v11.py` / `final_v11.sh`.
- **Colab cross-encoder:** `colab_ce.py`. It supports `--model-dir` scoring, twin context via a `tw` column, decoder LLMs and `--bf16`.

A new stack feature takes about 1 minute to validate. A new stage-2 feature set takes about 45 minutes end to end, including test scoring.
