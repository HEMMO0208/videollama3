#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/hmkang/project/videollama3"
cd "$ROOT_DIR"

pip install imageio ffmpeg-python moviepy tensorboard

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Checkpoint is trained on NextQA — point to nextqa work_dirs.
MODEL_KIND="${MODEL_KIND:-causal_diffusion}"
case "$MODEL_KIND" in
  diffusion)
    DEFAULT_MODEL_PATH="work_dirs/nextqa_diffusion"
    ;;
  baseline)
    DEFAULT_MODEL_PATH="work_dirs/nextqa_baseline"
    ;;
  causal_diffusion)
    DEFAULT_MODEL_PATH="work_dirs/nextqa_causal_diffusion"
    ;;
  *)
    DEFAULT_MODEL_PATH="$MODEL_KIND"
    ;;
esac

MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
SPLIT="${SPLIT:-mini}"
JSONL_PATH="${JSONL_PATH:-data/favor/${SPLIT}_sft.jsonl}"
DATA_FOLDER="${DATA_FOLDER:-/home/hmkang/project/videollama3/FAVOR/videos}"
OUTPUT_DIR="${OUTPUT_DIR:-results/favor}"
RUN_NAME="${RUN_NAME:-${MODEL_KIND}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$OUTPUT_DIR"

extra_args=()
if [[ -n "${LIMIT:-}" ]]; then
  extra_args+=(--limit "$LIMIT")
fi
if [[ -n "${MAX_VISUAL_TOKENS:-}" ]]; then
  extra_args+=(--max-visual-tokens "$MAX_VISUAL_TOKENS")
fi
if [[ -n "${ATTN_IMPLEMENTATION:-}" ]]; then
  extra_args+=(--attn-implementation "$ATTN_IMPLEMENTATION")
fi
if [[ -n "${NUM_CHUNKS:-}" ]]; then
  extra_args+=(--num-chunks "$NUM_CHUNKS")
fi
if [[ -n "${CHUNK_IDX:-}" ]]; then
  extra_args+=(--chunk-idx "$CHUNK_IDX")
fi

"$PYTHON_BIN" scripts/favor/infer_favor_jsonl.py \
  --model-path "$MODEL_PATH" \
  --jsonl-path "$JSONL_PATH" \
  --data-folder "$DATA_FOLDER" \
  --output-path "${OUTPUT_PATH:-$OUTPUT_DIR/${SPLIT}_${RUN_NAME}_predictions.jsonl}" \
  --fps "${FPS:-1}" \
  --max-frames "${MAX_FRAMES:-100}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-16}" \
  --seed "${SEED:-42}" \
  "${extra_args[@]}"
