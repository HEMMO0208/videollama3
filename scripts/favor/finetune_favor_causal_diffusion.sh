#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
ARG_MASTER_PORT=${MASTER_PORT:-16668}
ARG_RANK=${RANK:-0}

WORLD_SIZE=${WORLD_SIZE:-$ARG_WORLD_SIZE}
NPROC_PER_NODE=${NPROC_PER_NODE:-$ARG_NPROC_PER_NODE}
MASTER_ADDR=${MASTER_ADDR:-$ARG_MASTER_ADDR}
MASTER_PORT=${MASTER_PORT:-$ARG_MASTER_PORT}
RANK=${RANK:-$ARG_RANK}

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-$((GLOBAL_BATCH_SIZE/(WORLD_SIZE*NPROC_PER_NODE*LOCAL_BATCH_SIZE)))}

# Fine-tune from the nextqa-trained causal_diffusion checkpoint.
# Diffusion head is already embedded in MODEL_PATH — PRETRAINED_DIFFUSION_HEAD
# defaults to empty so no separate .bin is loaded.
MODEL_PATH=${MODEL_PATH:-work_dirs/nextqa_causal_diffusion}
DATA_FOLDER=${DATA_FOLDER:-/home/hmkang/project/videollama3/FAVOR/videos}
DATA_PATH=${DATA_PATH:-data/favor/train_sft.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-work_dirs/favor_causal_diffusion}
RUN_NAME=${RUN_NAME:-favor_causal_diffusion}
MM_PIXEL_DECODER=${MM_PIXEL_DECODER:?Set MM_PIXEL_DECODER to the ross VAE checkpoint path}
PRETRAINED_DIFFUSION_HEAD=${PRETRAINED_DIFFUSION_HEAD:-}

EXTRA_TRAIN_ARGS=()
if [[ -n "$PRETRAINED_DIFFUSION_HEAD" ]]; then
    EXTRA_TRAIN_ARGS+=(--pretrained_diffusion_head "$PRETRAINED_DIFFUSION_HEAD")
fi

pip install imageio ffmpeg-python moviepy tensorboard

torchrun --nnodes "$WORLD_SIZE" \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    --node_rank "$RANK" \
    videollama3/train.py \
    --deepspeed "$ROOT_DIR/scripts/zero1.json" \
    --model_type videollama3_qwen2 \
    --model_path "$MODEL_PATH" \
    --vision_encoder DAMO-NLP-SG/SigLIP-NaViT \
    --mm_projector_type mlp2x_gelu \
    --data_path "$DATA_PATH" \
    --data_folder "$DATA_FOLDER" \
    --image_merge_size 1 \
    --video_merge_size 2 \
    --fps 1 \
    --max_frames 60 \
    --model_max_length 16384 \
    --mm_max_length 14400 \
    --use_batch_flattening False \
    --use_token_compression True \
    --diffusion_enable True \
    --mm_pixel_decoder "$MM_PIXEL_DECODER" \
    --diffusion_chunk_size 4 \
    --diffusion_vae_image_size 384 \
    --diffusion_target_spatial 12 \
    --diffusion_loss_weight 1.0 \
    --diffusion_query_prob 1.0 \
    --causal_diffusion True \
    "${EXTRA_TRAIN_ARGS[@]}" \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "$LOCAL_BATCH_SIZE" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --llm_lr 1e-5 \
    --mm_projector_lr 1e-5 \
    --vision_encoder_lr 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --report_to tensorboard \
    --run_name "$RUN_NAME"
