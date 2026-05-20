#!/bin/bash
set -e

ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
ARG_MASTER_PORT=${MASTER_PORT:-16667}
ARG_RANK=${RANK:-0}

WORLD_SIZE=${WORLD_SIZE:-$ARG_WORLD_SIZE}
NPROC_PER_NODE=${NPROC_PER_NODE:-$ARG_NPROC_PER_NODE}
MASTER_ADDR=${MASTER_ADDR:-$ARG_MASTER_ADDR}
MASTER_PORT=${MASTER_PORT:-$ARG_MASTER_PORT}
RANK=${RANK:-$ARG_RANK}

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-$((GLOBAL_BATCH_SIZE/(WORLD_SIZE*NPROC_PER_NODE*LOCAL_BATCH_SIZE)))}

MODEL_PATH=${MODEL_PATH:-work_dirs/videollama3_qwen2.5_2b/stage_3}
DATA_FOLDER=${DATA_FOLDER:-../Tempo/dataset/nextqa/NExTVideo}
DATA_PATH=${DATA_PATH:-data/nextqa/train_sft.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-work_dirs/nextqa_baseline}
RUN_NAME=${RUN_NAME:-nextqa_baseline}

pip install imageio ffmpeg-python moviepy

torchrun --nnodes "$WORLD_SIZE" \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    --node_rank "$RANK" \
    videollama3/train.py \
    --deepspeed scripts/zero1.json \
    --model_type videollama3_qwen2 \
    --model_path "$MODEL_PATH" \
    --vision_encoder DAMO-NLP-SG/SigLIP-NaViT \
    --mm_projector_type mlp2x_gelu \
    --data_path "$DATA_PATH" \
    --data_folder "$DATA_FOLDER" \
    --image_merge_size 1 \
    --video_merge_size 2 \
    --fps 1 \
    --max_frames 120 \
    --model_max_length 16384 \
    --mm_max_length 10240 \
    --use_token_compression True \
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
    --save_steps 1000 \
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
