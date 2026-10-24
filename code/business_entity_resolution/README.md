# Business Entity Resolution: code and reproduction

This folder regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the challenge data.
The method, the experiments behind each choice and the results are in `docs/METHODOLOGY.md` (it was `Documentation_template.md` in
the submission zip).

In one line: normalisation → four candidate generators (token search, sibling expansion, reverse search with
re-ranking, a fine-tuned dense bi-encoder) → learned pruner → LightGBM matchers with lexical target encoding and
collective features → seven fine-tuned cross-encoders on the uncertain pairs → LightGBM stack → a decision layer
derived from F0.5 and the one-owner-per-record constraint. France, absent from training, uses a language-agnostic
word-role model, three rounds of self-trained French cross-encoders and record-level checks on contested records.

## Folder contents

```text
reproduce.sh        runs the whole pipeline, raw data -> output/ (the only script you need)
smoke_test.sh       runs reproduce.sh end to end on a ~1.5% sample (about an hour on a 4-core CPU)
requirements.txt    pinned package versions
src/                all Python code (index below)
config/             choices frozen during development: stage-2 feature lists, final-stack features, decision rules
utils/              official submission validator
history/            the original per-version driver scripts (v3 ... v15b), kept for reference
```

## Setup

- Python 3.13 (tested with 3.13.5): `pip install -r requirements.txt`
- Challenge data, unchanged, in this folder:
  `dataset/train/train_source{1,2,3}.tsv`, `dataset/train/train_ground_truth.tsv`, `dataset/test/test_source{1,2,3}.tsv`
- CPU stages: 4 cores and 16 GB of RAM are enough; this is the machine the submission was built on.
- GPU stages: one CUDA GPU. We used Google Colab A100s; each transformer took one to two GPU hours.
- Pretrained weights are downloaded from the Hugging Face Hub on first use. All are MIT or Apache-2.0, at most
  560M parameters: `intfloat/multilingual-e5-small`, `-base`, `-large`,
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and `sentence-transformers/LaBSE`.
- No other external data or services are used.

## Reproduce the submission

```bash
cd code/business_entity_resolution      # every command runs from this folder
bash reproduce.sh
```

What the script does:

- It runs every stage in the order the submission was built.
- Each step writes a marker to `work/.done/` and its log to `logs/<step>.log`.
- A rerun skips finished steps, so an interrupted run resumes where it stopped.
- The final files are written to `output/` and checked with the official validator (`logs/validate.log`).
- The script sets `PYTHONHASHSEED=0` itself. Token hashes must agree across stages, and every script re-executes
  itself if the variable is missing.

GPU steps (`src/colab_dense.py`, `src/colab_ce.py`) run in place when CUDA is available. Without a GPU, the script
stops at the first GPU step and prints the exact command. You can then either:

- run that command on a GPU machine from a copy of this folder, copy the named output folder back, and rerun
  `bash reproduce.sh`; or
- set `CPU_OK=1` to run the GPU steps on the CPU (possible but very slow).

Expected time:

- CPU stages: about a day (20 to 25 hours) on a 4-core laptop, most of it retrieval and feature building.
- GPU stages: roughly 15 A100 hours in total (one bi-encoder and ten cross-encoder fine-tunes).

### Stages

The final submission grew over several versions, and later versions reuse artefacts of earlier ones. The first
pruner selects the confident pairs for sibling expansion. The v7 and v8 candidate sets supply the cross-encoder
training pairs and the first French pseudo-labels. `reproduce.sh` therefore rebuilds exactly those intermediate
versions, and skips the experiments that did not reach the submission zip).

| # | Stage | Scripts | Main outputs |
|---|---|---|---|
| 1 | Normalisation, learned Indic-to-Latin transliteration, typed blocking tokens | `translit.py`, `prep.py` (`normalize.py`) | `work/translit.json`, `work/<split>/s{1,2,3}.parquet` |
| 2 | Forward IDF-weighted sparse token search, top 20 per source | `retrieve.py` | `work/<split>/cands.parquet` |
| 3 | First pruner (v3); sibling query expansion from pairs with p ≥ 0.9 | `make_features.py --stage block`, `prune.py`, `expand.py` | `models_v3/`, `cands_all.parquet` |
| 4 | Reverse search, record → S1 top 3 (training: 19% of S1 rows deleted, as in test) | `retrieve_rev.py` | `cands_full.parquet` |
| 5 | v7 candidate set and matcher (source of the first cross-encoder training set) | `simulate_drop.py`, `prune.py`, `make_features.py`, `train.py`, `predict.py` | `work_sim/`, `models_v7/`, `work/test/test_pred.parquet` |
| 6 | v8: re-ranked reverse search; word-role model; the France decisions that seed round-1 pseudo-labels | `retrieve_rerank.py`, `train_lex.py --agnostic --role`, `score_lex.py`, `decide.py`, `swap_filter.py` | `work8/`, `work8_sim/`, `models_v8ra/` |
| 7 | v9: wider re-ranking (top 100, keep 3, France included) | `retrieve_rerank.py` | `work9/<split>/cands_v9.parquet` |
| 8 | Dense retrieval **[GPU]**: fine-tuned multilingual-e5-small, both directions | `dense_export.py`, `colab_dense.py` | `dense_pkg/out_dense/dense_{train,test}.parquet` |
| 9 | v10 candidate set, pruner (max loss 0.1%), lexical stage 2, word-role model | `merge_dense.py`, `make_features.py`, `simulate_drop.py`, `prune.py`, `train_lex.py`, `score_lex.py` | `work10/`, `work10_sim/`, `models_v10*/` |
| 10 | Collective features; stage 2 and role model retrained with them; stage-3 group model | `coll_feats.py`, `train_lex.py --extra`, `stage3.py` | `val_predv10sc`, `test_predv10sc`, `test_predv10rac` |
| 11 | Cross-encoder training sets and scoring files (all merged into `colab_pkg/`) | `colab_export*.py`, `ce_bands.py` | `colab_pkg/pairs_*.parquet`, `rec_*.parquet` |
| 12 | Cross-encoders, round A **[GPU]**: MiniLM-L12, e5-base, e5-large (×2), LaBSE, e5-large v10, twin-aware e5-large, French round 1 | `colab_ce.py` | `colab_pkg/out_<name>/ce_{val,testband}.parquet` |
| 13 | French round 2 **[GPU]**: labels from the v8 role model plus round 1 | `stack_ce.py`, `decide.py`, `swap_filter.py`, `colab_export_fr2.py`, `colab_ce.py` | `colab_pkg/out_fr2/` |
| 14 | One score file per model, restricted to the uncertain band | `comb_ce.py` | `colab_pkg/comb10/` |
| 15 | French round 3 **[GPU]**: labels from the v10 France stack | `stack_ce.py`, `colab_export_fr2.py --out colab_fr3`, `colab_ce.py` | `colab_pkg/out_fr3/` |
| 16 | Final stacks and decisions; merge by country; validate | `final_v11.py`, `stack_ce.py`, `decide.py`, `swap_filter.py`, `contest_filter.py`, `hybrid.py` | `output/*.tsv` |

The transformers score only pairs whose stage-2 probability lies between 0.005 and 0.995. Those pairs are listed in
`colab_pkg/pairs_{val,test}_{inus,fr}_band.parquet`, and the same rule applies on validation and test. During the
competition these pairs were scored incrementally, as each version added candidates (`colab_export_new.py`), and
`comb_ce.py` assembled one file per model. Scoring the band directly, as `reproduce.sh` does, gives the final stacks
the same inputs.

### The final model and the candidate file

- **India and the US**:
  - model: a LightGBM stack (`src/final_v11.py`, 15 leaves, 300 rounds, 3 seeds) on the pairs with stage-3
    probability above 0.02;
  - decision: soft exclusivity with a 0.75 threshold (`config/decision_soft_thr075.json`), then evidence-backed
    restores (`contest_filter.py --mode restore`).
- **France**:
  - model: a LightGBM stack (`stack_ce.py --out-tag v10rafr123`) on the pairs with word-role probability above 0.02;
  - decision: the one-owner posterior with expected-F0.5 selection (`config/decision_odds_ef2.json`), then the
    descriptor-swap filter and the evidence vote on contested records (`contest_filter.py --mode drop`).
- For every S1 row, `candidate_pairs.tsv` lists exactly the pairs scored by the final model that decides that row
  (the India/US stack for Indian and US rows, the France stack for French rows): 6,558,404 pairs, 3.79 per S1 row.

## Quick check: the smoke test

```bash
bash smoke_test.sh            # or: bash smoke_test.sh /path/to/empty/folder
```

The smoke test works in a separate folder:

1. It copies what this folder ships (`src/`, `config/`, `utils/`, `reproduce.sh`).
2. It builds a 1.5% sample of the data there (`src/make_sample.py`):
   - training S1 rows with all their true matches;
   - test S1 rows with their candidate records;
   - a random share of every other record.
3. It runs `SMOKE=1 bash reproduce.sh`. The GPU stages run on the CPU, with the small MiniLM model and a few hundred
   training pairs each.
4. It validates the result with `--check-ids`.

Every stage and script of the full run executes, in the same order, so this is the fastest way to confirm that the
code, configuration and environment are complete. The scores on the sample are meaningless.

## What is and is not bit-exact

- **CPU stages are deterministic.** The hash seed is fixed, the LightGBM seeds are fixed, and splits use keyed
  blake2b hashes of the entity ids. Given the same inputs they give the same results; multithreaded LightGBM sums can
  differ in the last bit of a probability (about 1e-16), which changed no decision when we replayed the final stage.
- **GPU fine-tuning is not bit-reproducible.** Nondeterministic CUDA kernels and fp16 arithmetic mean a retrained
  transformer gives slightly different scores. A few decisions near the thresholds can therefore flip, so a full rerun
  matches the submitted files closely rather than byte for byte.
- **Early stages now run with the final code.** The code evolved during the competition, and `reproduce.sh` runs
  every stage with its final version, so intermediates first built with older code (for example the v3 block
  features) can come out slightly different.
- **The final stages reproduce the submission exactly.** Starting from the intermediates of our run, the last stage
  (16) regenerates the submitted files byte for byte:
  - `matching_results.tsv` md5 `46c626b2078274b73f516d16a4adbf49`;
  - `candidate_pairs.tsv` md5 `0562f27f8567207a0630809cf4134ff3`.

## Source index (`src/`)

| Area | Files |
|---|---|
| Shared helpers, official metric, hash splits | `common.py` |
| Normalisation, transliteration, typed tokens | `normalize.py`, `translit.py`, `prep.py` |
| Candidate generation | `retrieve.py`, `expand.py`, `retrieve_rev.py`, `retrieve_rerank.py`, `dense_export.py`, `colab_dense.py`, `merge_dense.py` |
| Features and simulated test conditions | `features.py`, `make_features.py`, `simulate_drop.py`, `coll_feats.py` |
| Pruner and matchers | `prune.py`, `train.py`, `predict.py`, `lexical.py`, `train_lex.py`, `score_lex.py`, `stage3.py` |
| Cross-encoders | `colab_export.py`, `colab_export_big.py`, `colab_export_fr.py`, `colab_export_fr2.py`, `colab_export_v10.py`, `colab_export_twin.py`, `ce_bands.py`, `colab_ce.py`, `comb_ce.py` |
| Stacks and feature search | `stack_ce.py`, `greedy_stack.py`, `greedy_run.py` (produced `config/stack_inus_features.json`), `final_v11.py` |
| Decisions and France post-processing | `decide.py` (exclusivity modes, expected-F0.5 selection, validation grid), `swap_filter.py`, `contest_filter.py`, `hybrid.py` |
| Smoke test | `make_sample.py` |
| Diagnostics and experiments that did not reach the submission | `block_recall.py`, `error_analysis.py`, `colab_export_new.py`, `ce_export.py`, `ce_run.py`, `name_bucket.py`, `stage4_counts.py`, `label_shift.py`, `gfm.py`, `gfm_eval.py` |

The decision rules in `config/` were chosen on validation with `decide.py --mode val` (the grid, with
`--decoy-dup 2` for test-like decoy density), and the final-stack features with `greedy_stack.py build` followed by
`greedy_run.py`.

## `history/`

These are the driver scripts as they ran during the competition, one per version (`run_v7.sh`, `run_v8.sh`,
`run_v9.sh`, `run_v10.sh`, `run_v10c.sh`, `final_v10.sh`, `final_v11.sh`, `fr3_integrate.sh`, `final_v15.sh`, and
others). Only four things changed: paths, so that they run from the parent folder; waits on jobs that shared the machine,
which became explicit checks; one call to a local audit script, which was removed; and the France decision rule,
which had been written by hand and is now copied from `config/`. They document the path from v3 to v15b. `reproduce.sh` contains every step of theirs that
the submission depends on, so you do not need to run them.
