#!/usr/bin/env bash
#SBATCH --job-name=vllama3-diff-nextqa-attn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

export MODEL_KIND="${MODEL_KIND:-diffusion}"

bash /home/hmkang/init_ross.sh bash "$ROOT_DIR/scripts/nextqa/visualize_nextqa_video_self_attn.sh"
