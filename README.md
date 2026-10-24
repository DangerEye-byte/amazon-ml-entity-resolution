# Business Entity Resolution (Amazon ML Challenge 2026)

Team Coder's Legion. For every business record in Source 1 (name, address, country), find the matching
records in Source 2 and Source 3. Scored by macro F0.5 per Source-1 row (precision-weighted); a record with no
true match scores 1.0 only for an empty prediction. The test set adds France, which never appears in training.

**Result:** finished in the **top 50** on the public leaderboard, F0.5 **0.990642** (validation ~0.990). Trained and run on CPU
(4 cores, 16 GB) plus Colab GPUs for the transformer stages.

## Approach

1. **Normalisation + transliteration**: Indic scripts mapped to Latin tokens, learned from training pairs.
2. **Candidate generation**: IDF-weighted token search, sibling-query expansion, reverse search with
   address re-ranking, and a fine-tuned dense bi-encoder (blocking recall 0.983 -> 0.993).
3. **Learned pruner** (LightGBM) to keep the candidate set small. A smaller set per record ranks higher.
4. **Matchers**: LightGBM on string, number, lexical target-encoded and collective features.
5. **Cross-encoders**: seven fine-tuned multilingual transformers (e5, MiniLM, LaBSE) on uncertain pairs,
   stacked with LightGBM.
6. **Decision layer**: expected-F0.5 selection under the one-owner-per-record constraint.
7. **France (unseen country)**: language-agnostic word-role model, self-trained French cross-encoders,
   and record-level checks on contested records.

Licences: only MIT / Apache-2.0 models, at most 560M parameters, no external data or APIs.

## Layout

```text
code/business_entity_resolution/   pipeline code, reproduce.sh, smoke_test.sh, requirements.txt
docs/METHODOLOGY.md                full write-up: method, experiments, results
docs/research_brief.md             state of the project, validation vs leaderboard analysis
docs/research_report.md            ranked list of what to try next
```

The challenge data and trained artefacts are not included (organiser data; large generated files).
See `code/business_entity_resolution/README.md` for setup and reproduction.
