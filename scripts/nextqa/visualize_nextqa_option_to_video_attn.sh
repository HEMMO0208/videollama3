#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/hmkang/project/videollama3"
cd "$ROOT_DIR"

pip install imageio ffmpeg-python moviepy tensorboard

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_KIND="${MODEL_KIND:-diffusion}"
case "$MODEL_KIND" in
  diffusion)
    DEFAULT_MODEL_PATH="work_dirs/nextqa_diffusion"
    ;;
  baseline)
    DEFAULT_MODEL_PATH="work_dirs/nextqa_baseline"
    ;;
  *)
    DEFAULT_MODEL_PATH="$MODEL_KIND"
    ;;
esac

MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
SPLIT="${SPLIT:-val_mini_1000}"
JSONL_PATH="${JSONL_PATH:-data/nextqa/${SPLIT}_sft.jsonl}"
DATA_FOLDER="${DATA_FOLDER:-/home/hmkang/project/tempo/dataset/NExTVideo}"
OUTPUT_DIR="${OUTPUT_DIR:-results/nextqa_option_to_video_attn/${MODEL_KIND}_${SPLIT}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$OUTPUT_DIR"

extra_args=()
if [[ -n "${LAYERS:-}" ]]; then
  extra_args+=(--layers "$LAYERS")
fi

"$PYTHON_BIN" scripts/nextqa/visualize_nextqa_option_to_video_attn.py \
  --model-path "$MODEL_PATH" \
  --jsonl-path "$JSONL_PATH" \
  --data-folder "$DATA_FOLDER" \
  --output-dir "$OUTPUT_DIR" \
  --fps "${FPS:-1}" \
  --max-frames "${MAX_FRAMES:-32}" \
  --limit "${LIMIT:-10}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-eager}" \
  "${extra_args[@]}"
