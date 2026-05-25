#!/usr/bin/env bash
#SBATCH --job-name=vllama3-cdiff-nextqa-oe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=32:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=all
#SBATCH --exclude=node10

set -euo pipefail

ROOT_DIR="/home/hmkang/project/videollama3"
cd "$ROOT_DIR"
mkdir -p logs

export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=4
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ── server-specific paths & settings ─────────────────────────────────────────
export NPROC_PER_NODE=4
export MODEL_PATH="${MODEL_PATH:-checkpoints/VideoLLaMA3-2B-Image}"
export DATA_FOLDER="${DATA_FOLDER:-/home/hmkang/project/tempo/dataset/NExTVideo}"
export DATA_PATH="${DATA_PATH:-data/nextqa_oe/train_sft.jsonl}"
export MM_PIXEL_DECODER="${MM_PIXEL_DECODER:-pretrained_vae}"
export PRETRAINED_DIFFUSION_HEAD="${PRETRAINED_DIFFUSION_HEAD:-work_dirs/nextqa_diffusion/diffusion_head.bin}"
export OUTPUT_DIR="${OUTPUT_DIR:-work_dirs/nextqa_oe_causal_diffusion}"
export RUN_NAME="${RUN_NAME:-nextqa_oe_causal_diffusion}"

bash /home/hmkang/init_ross.sh bash "$ROOT_DIR/scripts/nextqa_oe/finetune_nextqa_oe_causal_diffusion.sh"
