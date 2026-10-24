#!/usr/bin/env bash
# End-to-end smoke test of reproduce.sh: every stage, in order, on a small sample of the challenge data.
# Copies only what the submission ships (src/, config/, utils/, reproduce.sh) into a separate folder, builds a
# ~1.5% sample of dataset/ there, runs `SMOKE=1 bash reproduce.sh` (GPU stages run on the CPU with a small
# transformer and a few hundred training pairs), then validates the two output files.
#
#   bash smoke_test.sh [DIR]     DIR defaults to ../smoke_run; about an hour on 4 CPU cores
set -eo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8
if ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  python() { python3 "$@"; }; export -f python     # systems that only have python3
fi
DIR=${1:-../smoke_run}
FRAC=${FRAC:-0.015}
[ -f dataset/train/train_source1.tsv ] || { echo "Put the challenge data in dataset/train and dataset/test first"; exit 1; }
CANDS=""   # a previous candidate_pairs.tsv keeps true matches and near-miss decoys together in the test sample
for c in ../../output/candidate_pairs.tsv output/candidate_pairs.tsv; do [ -f "$c" ] && { CANDS=$c; break; }; done
mkdir -p "$DIR"
cp -r src config utils reproduce.sh "$DIR"/
if [ ! -f "$DIR/dataset/test/test_source3.tsv" ]; then
  python src/make_sample.py --data dataset --out "$DIR/dataset" --frac "$FRAC" --rec-frac "$FRAC" ${CANDS:+--test-cands "$CANDS"}
fi
cd "$DIR"
SMOKE=1 bash reproduce.sh
python utils/validate_submission.py -m output/matching_results.tsv -c output/candidate_pairs.tsv -t dataset/test --check-ids
echo "SMOKE TEST PASSED in $DIR"
