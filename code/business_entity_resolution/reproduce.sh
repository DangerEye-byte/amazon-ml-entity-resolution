#!/usr/bin/env bash
# Rebuilds output/matching_results.tsv and output/candidate_pairs.tsv from dataset/train and dataset/test.
#
# The final submission was built over several pipeline versions (v3 ... v15b). Later versions reuse artefacts of
# earlier ones: the first pruner drives sibling expansion, and the v7 and v8 candidate sets supply the training
# pairs of the cross-encoders and the first French pseudo-labels. This script runs exactly those stages, in the
# order they were run, and skips experiments that did not reach the submission. See README.md.
#
#   bash reproduce.sh               full run (CPU stages: about a day on 4 cores / 16 GB; GPU stages: see below)
#   SMOKE=1 bash reproduce.sh       quick end-to-end check on a small sample (used by smoke_test.sh)
#
# Every step writes a marker to work/.done/ and its log to logs/<step>.log; a rerun skips finished steps.
# GPU steps (src/colab_dense.py, src/colab_ce.py) need a CUDA GPU (an A100 took 1 to 1.5 h per model). Without one
# the script stops at the first GPU step and prints the command; run it on a GPU machine with the same folder
# layout, copy the output folder back, and rerun. CPU_OK=1 runs GPU steps on the CPU anyway (very slow).
set -eo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8
if ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  python() { python3 "$@"; }; export -f python     # systems that only have python3
fi
SMOKE=${SMOKE:-0}
W=${WORKERS:-2}                 # worker processes for the retrieval stages (memory-bound: 2 fit in 16 GB)
DONE=work/.done
mkdir -p "$DONE" logs
SMOKE_CE="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

run() {  # run NAME COMMAND...: one pipeline step, skipped when already done
  local name=$1; shift
  if [ -f "$DONE/$name" ]; then echo "  skip  $name"; return 0; fi
  echo "  run   $name: $*"
  local t0=$SECONDS
  if ! "$@" > "logs/$name.log" 2>&1; then
    echo "FAILED: $name (last lines of logs/$name.log below)"; tail -n 25 "logs/$name.log"; exit 1
  fi
  touch "$DONE/$name"; echo "        done in $((SECONDS - t0)) s"
}
sh_() { bash -c "set -eo pipefail; $1"; }   # for steps made of several shell commands

have_gpu() { python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; }
gpu_gate() {  # stop before a GPU step when no GPU is available (unless SMOKE=1 or CPU_OK=1)
  [ "$SMOKE" = 1 ] || [ "${CPU_OK:-0}" = 1 ] || have_gpu && return 0
  echo; echo "No CUDA GPU found. Run this step on a GPU machine from this folder:"; echo "    $1"
  echo "then copy $2 back here and rerun: bash reproduce.sh (finished steps are skipped)."; exit 3
}
ce() {  # ce NAME ARGS...: fine-tune and/or score one cross-encoder with src/colab_ce.py into colab_pkg/out_NAME
  local name=$1; shift
  local out=colab_pkg/out_$name want=()
  case " $* " in *" --val-only "*) want=("$out/ce_val.parquet");; *" --test-only "*) want=("$out/ce_testband.parquet");;
                 *) want=("$out/ce_val.parquet" "$out/ce_testband.parquet");; esac
  local ok=1; for f in "${want[@]}"; do [ -f "$f" ] || ok=0; done
  if [ $ok = 1 ]; then touch "$DONE/ce_$name"; fi
  [ -f "$DONE/ce_$name" ] && { echo "  skip  ce_$name"; return 0; }
  gpu_gate "python src/colab_ce.py --pkg colab_pkg --out-name out_$name $*" "$out/"
  local extra=(); [ "$SMOKE" = 1 ] && extra=(--smoke --smoke-n 600 --model "$SMOKE_CE" --bs 16 --score-bs 64 --max-len 64)
  run "ce_$name" python src/colab_ce.py --pkg colab_pkg --out-name "out_$name" "$@" "${extra[@]}"
}
decision() { mkdir -p "$2" && cp "config/$1" "$2/decision.json"; }   # frozen decision rule -> models dir

[ -f dataset/train/train_source1.tsv ] && [ -f dataset/test/test_source1.tsv ] || {
  echo "Put the challenge data in dataset/train/ and dataset/test/ (see README.md)"; exit 1; }
echo "reproduce.sh (SMOKE=$SMOKE, workers=$W)"

# ---- 1. normalisation, learned transliteration, typed blocking tokens -------------------------------------------
run translit      python src/translit.py --train-dir dataset/train --test-dir dataset/test
run prep_train    python src/prep.py --split train
run prep_test     python src/prep.py --split test
# ---- 2. forward IDF token retrieval, top 20 per source ------------------------------------------------------------
run retr_train    python src/retrieve.py --split train --k 20
run retr_test     python src/retrieve.py --split test --k 20
# ---- 3. first learned pruner (v3); its confident pairs drive sibling query expansion -----------------------------
run block3_train  python src/make_features.py --stage block --split train --frac 0.3
run prune3_train  python src/prune.py --mode train --models models_v3
run block3_test   python src/make_features.py --stage block --split test
run prune3_test   python src/prune.py --mode test --models models_v3
run expand_train  python src/expand.py --split train --workers "$W"
run expand_test   python src/expand.py --split test --workers "$W"
# ---- 4. reverse retrieval: every record queries the S1 index (19% of training S1 rows deleted, as in test) -------
run rev_train     python src/retrieve_rev.py --split train --drop-frac 0.19 --workers "$W"
run rev_test      python src/retrieve_rev.py --split test --workers "$W"
# ---- 5. v7 candidate set: source of the first cross-encoder training package -------------------------------------
run block7_train  python src/make_features.py --stage block --split train --frac 0.3 --cands cands_full.parquet
run sim7          python src/simulate_drop.py --drop-frac 0.19 --out-work work_sim --cands cands_full.parquet
run prune7_train  python src/prune.py --mode train --work work_sim --models models_v7
run full7_train   python src/make_features.py --stage full --split train --work work_sim
run train7        python src/train.py --neg-rate 1.0 --work work_sim --models models_v7
run block7_test   python src/make_features.py --stage block --split test --cands cands_full.parquet
run prune7_test   python src/prune.py --mode test --models models_v7
run full7_test    python src/make_features.py --stage full --split test
run predict7      python src/predict.py --models models_v7
# ---- 6. v8 candidate set (reverse re-ranking, first version): larger CE training set, France round-1 labels -------
run rr8_train     python src/retrieve_rerank.py --split train --drop-frac 0.19 --out-work work8 --workers "$W"
run rr8_test      python src/retrieve_rerank.py --split test --out-work work8 --workers "$W"
run block8_train  python src/make_features.py --stage block --split train --work work8 --frac 0.3 --cands cands_v8.parquet
run sim8          python src/simulate_drop.py --drop-frac 0.19 --work work8 --out-work work8_sim --cands cands_v8.parquet
run prune8_train  python src/prune.py --mode train --work work8_sim --models models_v8
run full8_train   python src/make_features.py --stage full --split train --work work8_sim
run block8_test   python src/make_features.py --stage block --split test --work work8 --cands cands_v8.parquet
run prune8_test   python src/prune.py --mode test --work work8 --models models_v8
run full8_test    python src/make_features.py --stage full --split test --work work8
cp config/features_v8.json models_v8/features.json    # stage-2 base features of v8 (v7 list + rr_rank, rr_score)
run role8_train   python src/train_lex.py --work work8_sim --test-work work8 --models models_v8ra --tag v8ra \
                    --base-models models_v8 --agnostic --role --no-test
run role8_test    python src/score_lex.py --work work8_sim --test-work work8 --models models_v8ra --tag v8ra
decision decision_hard_ef2.json models_v8ra_fr
run fr_role8      sh_ "python src/decide.py --mode test --work work8 --models models_v8ra_fr --pred work8/test/test_predv8ra.parquet --out-dir output_v8ra_fr
                       python src/swap_filter.py --work work8/test --pred work8/test/test_predv8ra.parquet --in-dir output_v8ra_fr --out-dir submissions/v8ra_fr_swap
                       cp output_v8ra_fr/candidate_pairs.tsv submissions/v8ra_fr_swap/"
# ---- 7. v9: wider reverse re-ranking (top 100, keep 3, France included): the token candidate base of v10 ----------
RR="--k 100 --keep 3 --tie 0.8 --max-ties 3 --out-work work9 --out-name cands_v9.parquet --workers $W"
run rr9_train     python src/retrieve_rerank.py --split train --drop-frac 0.19 $RR
run rr9_test      python src/retrieve_rerank.py --split test --all-countries $RR
# ---- 8. dense retrieval: fine-tuned multilingual-e5-small bi-encoder [GPU] ----------------------------------------
run dense_export  python src/dense_export.py --out dense_pkg
DN=dense_pkg/out_dense; DARGS=(); [ "$SMOKE" = 1 ] && { DN=dense_pkg/out_dense_smoke; DARGS=(--smoke); }
if [ ! -f "$DONE/dense" ] && [ -f $DN/dense_train.parquet ] && [ -f $DN/dense_test.parquet ]; then touch "$DONE/dense"; fi
[ -f "$DONE/dense" ] || gpu_gate "python src/colab_dense.py --pkg dense_pkg" "$DN/"
run dense         python src/colab_dense.py --pkg dense_pkg "${DARGS[@]}"
# ---- 9. v10: final candidate set, pruner, lexical stage 2, word-role model -----------------------------------------
run merge_train   python src/merge_dense.py --split train --dense $DN/dense_train.parquet
run merge_test    python src/merge_dense.py --split test --dense $DN/dense_test.parquet
run block10_train python src/make_features.py --stage block --split train --work work10 --frac 0.3 --cands cands_v10.parquet
run sim10         python src/simulate_drop.py --drop-frac 0.19 --work work10 --out-work work10_sim --cands cands_v10.parquet
run prune10_train python src/prune.py --mode train --work work10_sim --models models_v10 --max-loss 0.001
run full10_train  python src/make_features.py --stage full --split train --work work10_sim
cp config/features_v10.json models_v10/features.json  # v8 list + the six dense-retrieval features
run lx10_train    python src/train_lex.py --work work10_sim --test-work work10 --models models_v10lx --tag v10lx \
                    --base-models models_v10 --no-test
run role10_train  python src/train_lex.py --work work10_sim --test-work work10 --models models_v10ra --tag v10ra \
                    --base-models models_v10 --agnostic --role --no-test
run block10_test  python src/make_features.py --stage block --split test --work work10 --cands cands_v10.parquet
run prune10_test  python src/prune.py --mode test --work work10 --models models_v10
run full10_test   python src/make_features.py --stage full --split test --work work10
run lx10_test     python src/score_lex.py --work work10_sim --test-work work10 --models models_v10lx --tag v10lx
run role10_test   python src/score_lex.py --work work10_sim --test-work work10 --models models_v10ra --tag v10ra
# ---- 10. collective features, stage 2 and word-role model retrained with them, stage-3 group model -----------------
run coll_train    python src/coll_feats.py --split-dir work10_sim/train --drop-frac 0.19
run coll_test     python src/coll_feats.py --split-dir work10/test --drop-frac 0
run lxc_train     python src/train_lex.py --work work10_sim --test-work work10 --models models_v10lxc --tag v10lxc \
                    --base-models models_v10 --no-test --extra work10_sim/train/coll.parquet
run s3c_val       python src/stage3.py --mode val --work work10_sim --models models_v10 --lexical --tag v10sc --in-tag v10lxc
run rac_train     python src/train_lex.py --work work10_sim --test-work work10 --models models_v10rac --tag v10rac \
                    --base-models models_v10 --agnostic --role --no-test --extra work10_sim/train/coll.parquet
run lxc_test      python src/score_lex.py --work work10_sim --test-work work10 --models models_v10lxc --tag v10lxc \
                    --extra work10/test/coll.parquet
run rac_test      python src/score_lex.py --work work10_sim --test-work work10 --models models_v10rac --tag v10rac \
                    --extra work10/test/coll.parquet
run s3c_test      python src/stage3.py --mode test --work work10_sim --test-work work10 --models models_v10 --lexical \
                    --tag v10sc --in-tag v10lxc
# ---- 11. cross-encoder packages (CPU); every export is merged into colab_pkg/, the folder the GPU steps read -------
V10N=(); T=(); [ "$SMOKE" = 1 ] && { V10N=(--n 6000 --n-easy 1000); T=(--n-plain 3000); }
run export_pkg    python src/colab_export.py --out colab_pkg
run export_big    sh_ "python src/colab_export_big.py && cp colab_big/* colab_pkg/"
run export_fr1    sh_ "python src/colab_export_fr.py && cp colab_fr/* colab_pkg/"
run export_v10    sh_ "python src/colab_export_v10.py --out colab_v10t ${V10N[*]} && cp colab_v10t/* colab_pkg/"
run export_bands  python src/ce_bands.py --out colab_pkg
run export_twin   sh_ "python src/colab_export_twin.py --out colab_twin --train colab_pkg/pairs_train_v10.parquet \
                         --val colab_pkg/pairs_val_inus_band.parquet --test-inus colab_pkg/pairs_test_inus_band.parquet ${T[*]} \
                       && cp colab_twin/* colab_pkg/"
# ---- 12. cross-encoders, round A [GPU]: each scores the uncertain band of the final pipeline ----------------------
IB=(--val-file pairs_val_inus_band.parquet --test-file pairs_test_inus_band.parquet)
FB=(--val-file pairs_val_fr_band.parquet --test-file pairs_test_fr_band.parquet)
ce mini    --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 "${IB[@]}"
ce e5b     --model intfloat/multilingual-e5-base  "${IB[@]}" --bs 256 --score-bs 2048
ce e5l     --model intfloat/multilingual-e5-large "${IB[@]}" --bs 256 --score-bs 4096
ce e5l2    --model intfloat/multilingual-e5-large "${IB[@]}" --train-file pairs_train_big.parquet --bs 256 --score-bs 4096
ce labse   --model sentence-transformers/LaBSE    "${IB[@]}" --bs 128 --score-bs 2048 --lr 3e-5
ce e5lv10  --model intfloat/multilingual-e5-large "${IB[@]}" --train-file pairs_train_v10.parquet --bs 128 --score-bs 2048
ce tw      --model intfloat/multilingual-e5-large --train-file pairs_train_tw.parquet --val-file pairs_val_tw.parquet \
           --test-file pairs_test_tw.parquet --bs 128 --score-bs 1024 --max-len 192
# France round 1: India/US labels + pseudo-labels of the v8 word-role model; scored on the v8 pairs that make the
# round-2 labels, then (same model) on the final France band
ce fr      --model intfloat/multilingual-e5-large --train-file pairs_train_fr.parquet --test-file pairs_test_fr.parquet \
           --bs 128 --score-bs 2048
ce fr_band --model-dir colab_pkg/out_fr/model "${FB[@]}" --score-bs 2048
# ---- 13. France round 2 labels: v8 word-role model + round-1 transformer, stacked, decided, swap-filtered ----------
run fr_stack8     sh_ "python -c \"
import pandas as pd
v = pd.read_parquet('work8_sim/train/val_predv8ra.parquet', columns=['q', 'src', 'd', 'p'])
v[(v.p > 0.005) & (v.p < 0.995)][['q', 'src', 'd']].reset_index(drop=True).to_parquet('colab_pkg/val_band_mask_v8ra.parquet', index=False)
\"
python src/stack_ce.py --val work8_sim/train/val_predv8ra.parquet --test work8/test/test_predv8ra.parquet \
  --ce fr=colab_pkg/out_fr/ce_val.parquet,colab_pkg/out_fr/ce_testband.parquet,colab_pkg/val_band_mask_v8ra.parquet --out-tag v8rafr
python src/decide.py --mode test --work work8 --models models_v8ra_fr --pred work8/test/test_predv8rafr.parquet --out-dir output_v8rafr
python src/swap_filter.py --work work8/test --pred work8/test/test_predv8rafr.parquet --in-dir output_v8rafr --out-dir submissions/v8rafr_swap
cp output_v8rafr/candidate_pairs.tsv submissions/v8rafr_swap/"
run export_fr2    sh_ "python src/colab_export_fr2.py && cp colab_fr2/* colab_pkg/"
ce fr2     --model intfloat/multilingual-e5-large --train-file pairs_train_fr2.parquet "${FB[@]}" --bs 256 --score-bs 4096
# ---- 14. transformer scores per model, restricted to the band (identical on validation and test) --------------------
run comb_ce       sh_ "for m in mini e5b e5l e5l2 labse e5lv10; do
  python src/comb_ce.py --name \$m --out colab_pkg/comb10 --val colab_pkg/out_\$m/ce_val.parquet --test colab_pkg/out_\$m/ce_testband.parquet \
    --val-band work10_sim/train/val_predv10lx.parquet --test-band work10/test/test_predv10lx.parquet
done
for m in fr:fr_band fr2:fr2; do
  python src/comb_ce.py --name \${m%%:*} --out colab_pkg/comb10 --val colab_pkg/out_\${m##*:}/ce_val.parquet \
    --test colab_pkg/out_\${m##*:}/ce_testband.parquet --val-band work10_sim/train/val_predv10ra.parquet --test-band work10/test/test_predv10ra.parquet
done"
# ---- 15. France round 3: labels from the v10 France stack (role model + rounds 1 and 2) [GPU] ----------------------
run fr_stack10    sh_ "C=colab_pkg/comb10
python src/stack_ce.py --val work10_sim/train/val_predv10ra.parquet --test work10/test/test_predv10ra.parquet \
  --ce fr=\$C/fr_val.parquet,\$C/fr_test.parquet fr2=\$C/fr2_val.parquet,\$C/fr2_test.parquet --out-tag v10rafr12 --min-p 0.02
python src/decide.py --mode test --work work10 --models models_v8ra_fr --pred work10/test/test_predv10rafr12.parquet --out-dir output_v10rafr12
python src/swap_filter.py --work work10/test --pred work10/test/test_predv10rafr12.parquet --in-dir output_v10rafr12 --out-dir submissions/v10rafr12_swap
cp output_v10rafr12/candidate_pairs.tsv submissions/v10rafr12_swap/"
run export_fr3    sh_ "python src/colab_export_fr2.py --out colab_fr3 --stack work10/test/test_predv10rafr12.parquet \
  --role work10/test/test_predv10ra.parquet --ce colab_pkg/comb10/fr2_test.parquet --final submissions/v10rafr12_swap/matching_results.tsv \
  --pre output_v10rafr12/matching_results.tsv --train-name pairs_train_fr3.parquet --tag fr3 --have 'colab_pkg/rec_test_s{k}_band.parquet' \
  && cp colab_fr3/* colab_pkg/"
ce fr3     --model intfloat/multilingual-e5-large --train-file pairs_train_fr3.parquet "${FB[@]}" --bs 256 --score-bs 2048
# ---- 16. final models and decisions ----------------------------------------------------------------------------------
# India/US: LightGBM stack (3 seeds) on stage-2/3/role logits, 7 cross-encoders, collective and pairwise features;
# soft exclusivity, threshold 0.75; then evidence-backed restores of contested records.
run final_inus    python src/final_v11.py --inus-only
decision decision_soft_thr075.json models_v11inus
run decide_inus   sh_ "python src/decide.py --mode test --work work10 --models models_v11inus --pred work10/test/test_predv11inus.parquet --out-dir output_v11inus
python src/contest_filter.py --mode restore --countries in us --pred work10/test/test_predv11inus.parquet --in-dir output_v11inus \
  --out-dir output_v15binus --rival-p 0.3"
# France: stack of the word-role model and the three self-trained French cross-encoders; one-owner posterior and
# expected-F0.5 selection, descriptor-swap filter, evidence vote on contested records.
run final_fr      python src/stack_ce.py --val work10_sim/train/val_predv10ra.parquet --test work10/test/test_predv10ra.parquet \
                    --ce fr=colab_pkg/comb10/fr_val.parquet,colab_pkg/comb10/fr_test.parquet \
                    fr2=colab_pkg/comb10/fr2_val.parquet,colab_pkg/comb10/fr2_test.parquet \
                    fr3=colab_pkg/out_fr3/ce_val.parquet,colab_pkg/out_fr3/ce_testband.parquet --out-tag v10rafr123 --min-p 0.02
decision decision_odds_ef2.json models_v13ofr
run decide_fr     sh_ "python src/decide.py --mode test --work work10 --models models_v13ofr --pred work10/test/test_predv10rafr123.parquet --out-dir output_v13ofr
python src/swap_filter.py --work work10/test --pred work10/test/test_predv10rafr123.parquet --in-dir output_v13ofr --out-dir submissions/v13ofr_swap
cp output_v13ofr/candidate_pairs.tsv submissions/v13ofr_swap/
python src/contest_filter.py --mode drop --countries fr --pred work10/test/test_predv10rafr123.parquet --in-dir submissions/v13ofr_swap \
  --out-dir submissions/v15bfr"
# Merge by country (India/US rows from the India/US model, France rows from the France model) and validate.
run merge_final   python src/hybrid.py --seen output_v15binus --unseen submissions/v15bfr --out output
run validate      python utils/validate_submission.py -m output/matching_results.tsv -c output/candidate_pairs.tsv -t dataset/test
python - <<'EOF'
import pandas as pd
n = lambda s: 0 if not s else len(s.split(","))
c = pd.read_csv("output/candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)["candidate_entity_ids"].map(n)
m = pd.read_csv("output/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)["matched_entity_ids"].map(n)
print(f"output/candidate_pairs.tsv: {c.sum()} pairs ({c.mean():.3f} per S1 row)")
print(f"output/matching_results.tsv: {m.sum()} matches ({m.mean():.3f} per S1 row), {(m == 0).mean():.2%} rows empty")
EOF
echo "DONE: output/matching_results.tsv, output/candidate_pairs.tsv (validator log: logs/validate.log)"
