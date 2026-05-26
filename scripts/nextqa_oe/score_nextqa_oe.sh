#!/usr/bin/env bash
# score_nextqa_oe.sh — batch WUPS scoring for all NExT-QA OE prediction files
#
# Scans PRED_DIR for every *.jsonl prediction file produced by
# infer_nextqa_oe_jsonl.py and runs score_nextqa_oe.py on each one.
# Already-scored files (*.wups.json present) are skipped unless FORCE=1.
#
# Usage (standalone):
#   bash scripts/nextqa_oe/score_nextqa_oe.sh
#
# Key env vars:
#   PRED_DIR     directory that holds prediction JSONL files
#                (default: results/nextqa_oe)
#   REF_CSV      override reference CSV for all files
#                (default: auto-detected from filename: val|test)
#   REF_BASE_DIR directory containing openend/<split>.csv files
#                (default: dataset/nextqa/openend)
#   ADD_REF_DIR  directory with add_reference_answer_<split>.json
#                (default: ../NExT-OE/dataset/nextqa)
#   FORCE        set to 1 to re-score even if .wups.json already exists
#   PYTHON_BIN   python interpreter  (default: python3)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# ── dependencies ──────────────────────────────────────────────────────────────
pip install nltk pandas pywsd --quiet
"$PYTHON_BIN" -c "
import nltk
nltk.download('wordnet',                        quiet=True)
nltk.download('punkt',                          quiet=True)
nltk.download('punkt_tab',                      quiet=True)
nltk.download('averaged_perceptron_tagger',     quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
"
PRED_DIR="${PRED_DIR:-results/nextqa_oe}"
REF_BASE_DIR="${REF_BASE_DIR:-dataset/nextqa/openend}"
ADD_REF_DIR="${ADD_REF_DIR:-../NExT-OE/dataset/nextqa}"
FORCE="${FORCE:-0}"

if [[ ! -d "$PRED_DIR" ]]; then
  echo "[error] PRED_DIR not found: $PRED_DIR" >&2
  exit 1
fi

# Collect prediction JSONL files — exclude any nested .json output files
mapfile -t PRED_FILES < <(find "$PRED_DIR" -maxdepth 1 -name "*.jsonl" | sort)

if [[ ${#PRED_FILES[@]} -eq 0 ]]; then
  echo "[warn] No *.jsonl files found in $PRED_DIR"
  exit 0
fi

echo "=== score_nextqa_oe: found ${#PRED_FILES[@]} prediction file(s) in $PRED_DIR ==="

scored=0
skipped=0

for PRED_JSONL in "${PRED_FILES[@]}"; do
  BASENAME="$(basename "$PRED_JSONL")"
  WUPS_JSON="${PRED_JSONL%.jsonl}.wups.json"

  # Skip already-scored unless FORCE=1
  if [[ -f "$WUPS_JSON" && "$FORCE" != "1" ]]; then
    echo "  [skip] $BASENAME  (wups.json exists; set FORCE=1 to re-score)"
    skipped=$((skipped + 1))
    continue
  fi

  # Auto-detect split (val / test) from filename
  if [[ "${REF_CSV:-}" != "" ]]; then
    THIS_REF_CSV="$REF_CSV"
  elif [[ "$BASENAME" == *test* ]]; then
    THIS_REF_CSV="$REF_BASE_DIR/test.csv"
  else
    THIS_REF_CSV="$REF_BASE_DIR/val.csv"
  fi

  if [[ ! -f "$THIS_REF_CSV" ]]; then
    echo "  [error] ref CSV not found for $BASENAME: $THIS_REF_CSV" >&2
    continue
  fi

  # Optional additional-reference JSON (exists for test split in NExT-OE)
  SPLIT_NAME="$(basename "$THIS_REF_CSV" .csv)"   # val | test
  ADD_REF_JSON="$ADD_REF_DIR/add_reference_answer_${SPLIT_NAME}.json"

  extra_args=()
  if [[ -f "$ADD_REF_JSON" ]]; then
    echo "  [info] using add-ref: $ADD_REF_JSON"
    extra_args+=(--add-ref "$ADD_REF_JSON")
  fi

  echo ""
  echo "--- Scoring: $BASENAME ---"
  echo "  ref CSV : $THIS_REF_CSV"
  echo "  output  : $(basename "$WUPS_JSON")"

  "$PYTHON_BIN" scripts/nextqa_oe/score_nextqa_oe.py \
    --pred-path    "$PRED_JSONL" \
    --ref-csv      "$THIS_REF_CSV" \
    --output-path  "$WUPS_JSON" \
    "${extra_args[@]}"

  scored=$((scored + 1))
done

echo ""
echo "=== Done: scored=$scored  skipped=$skipped ==="
