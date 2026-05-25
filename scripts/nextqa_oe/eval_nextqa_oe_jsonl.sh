#!/usr/bin/env bash
# eval_nextqa_oe_jsonl.sh — inference + WUPS scoring for NExT-QA Open-Ended
#
# Usage:
#   MODEL_KIND=baseline SPLIT=val_mini_1000 bash scripts/nextqa_oe/eval_nextqa_oe_jsonl.sh
#
# Key env vars:
#   MODEL_KIND    baseline | diffusion | causal_diffusion  (default: diffusion)
#   MODEL_PATH    override checkpoint path
#   SPLIT         val | val_mini_1000  (default: val_mini_1000)
#   JSONL_PATH    override input JSONL
#   DATA_FOLDER   root directory of NExTVideo clips
#   OUTPUT_DIR    where to write prediction files  (default: results/nextqa_oe)
#   RUN_NAME      tag appended to output filenames  (default: $MODEL_KIND)
#   MAX_FRAMES    frames sampled per video  (default: 100)
#   FPS           frames per second for sampling  (default: 1)
#   MAX_NEW_TOKENS tokens budget for OE generation  (default: 64)
#   LIMIT         run only first N records (for debugging)
#   REF_CSV       reference CSV for scoring  (default: dataset/nextqa/openend/${SPLIT%%_*}.csv)
#   ADD_REF       additional-reference JSON (optional; auto-detected for test split)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

pip install imageio ffmpeg-python moviepy tensorboard

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_KIND="${MODEL_KIND:-diffusion}"
case "$MODEL_KIND" in
  diffusion)
    DEFAULT_MODEL_PATH="work_dirs/nextqa_oe_diffusion"
    ;;
  baseline)
    DEFAULT_MODEL_PATH="work_dirs/nextqa_oe_baseline"
    ;;
  causal_diffusion)
    DEFAULT_MODEL_PATH="work_dirs/nextqa_oe_causal_diffusion"
    ;;
  *)
    # Allow MODEL_KIND to be an arbitrary path
    DEFAULT_MODEL_PATH="$MODEL_KIND"
    ;;
esac

MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
SPLIT="${SPLIT:-val_mini_1000}"
JSONL_PATH="${JSONL_PATH:-data/nextqa_oe/${SPLIT}_sft.jsonl}"
DATA_FOLDER="${DATA_FOLDER:-../Tempo/dataset/nextqa/NExTVideo}"
OUTPUT_DIR="${OUTPUT_DIR:-results/nextqa_oe}"
RUN_NAME="${RUN_NAME:-${MODEL_KIND}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Reference CSV: val_mini_1000 → val, otherwise use SPLIT directly
REF_BASE="${SPLIT%%_mini*}"   # val_mini_1000 → val, val → val
REF_CSV="${REF_CSV:-dataset/nextqa/openend/${REF_BASE}.csv}"

# NExT-OE additional reference (only present for test split)
NEXTOE_DIR="${NEXTOE_DIR:-../NExT-OE}"
ADD_REF_DEFAULT="${NEXTOE_DIR}/dataset/nextqa/add_reference_answer_${REF_BASE}.json"
ADD_REF="${ADD_REF:-$ADD_REF_DEFAULT}"

mkdir -p "$OUTPUT_DIR"

PRED_JSONL="${OUTPUT_PATH:-$OUTPUT_DIR/${SPLIT}_${RUN_NAME}_predictions.jsonl}"

# ── 1. Inference ─────────────────────────────────────────────────────────────
echo "=== [1/2] Inference ==="
echo "  model   : $MODEL_PATH"
echo "  input   : $JSONL_PATH"
echo "  output  : $PRED_JSONL"

extra_infer_args=()
if [[ -n "${LIMIT:-}" ]];               then extra_infer_args+=(--limit "$LIMIT"); fi
if [[ -n "${MAX_VISUAL_TOKENS:-}" ]];   then extra_infer_args+=(--max-visual-tokens "$MAX_VISUAL_TOKENS"); fi
if [[ -n "${ATTN_IMPLEMENTATION:-}" ]]; then extra_infer_args+=(--attn-implementation "$ATTN_IMPLEMENTATION"); fi
if [[ -n "${NUM_CHUNKS:-}" ]];          then extra_infer_args+=(--num-chunks "$NUM_CHUNKS"); fi
if [[ -n "${CHUNK_IDX:-}" ]];           then extra_infer_args+=(--chunk-idx "$CHUNK_IDX"); fi

"$PYTHON_BIN" scripts/nextqa_oe/infer_nextqa_oe_jsonl.py \
  --model-path      "$MODEL_PATH" \
  --jsonl-path      "$JSONL_PATH" \
  --data-folder     "$DATA_FOLDER" \
  --output-path     "$PRED_JSONL" \
  --fps             "${FPS:-1}" \
  --max-frames      "${MAX_FRAMES:-32}" \
  --max-new-tokens  "${MAX_NEW_TOKENS:-64}" \
  --seed            "${SEED:-42}" \
  "${extra_infer_args[@]}"

# ── 2. WUPS Scoring ──────────────────────────────────────────────────────────
echo ""
echo "=== [2/2] WUPS Scoring ==="
echo "  predictions : $PRED_JSONL"
echo "  ref CSV     : $REF_CSV"

extra_score_args=()
if [[ -f "$ADD_REF" ]]; then
  echo "  add-ref     : $ADD_REF"
  extra_score_args+=(--add-ref "$ADD_REF")
fi

"$PYTHON_BIN" scripts/nextqa_oe/score_nextqa_oe.py \
  --pred-path "$PRED_JSONL" \
  --ref-csv   "$REF_CSV" \
  "${extra_score_args[@]}"
