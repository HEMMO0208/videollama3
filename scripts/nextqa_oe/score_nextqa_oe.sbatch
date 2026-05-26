#!/usr/bin/env bash
#SBATCH --job-name=vllama3-score-nextqa-oe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=all
#SBATCH --exclude=node10
# NOTE: scoring is CPU-only (WUPS/NLTK); no GPU requested.

set -euo pipefail

ROOT_DIR="/home/hmkang/project/videollama3"
cd "$ROOT_DIR"
mkdir -p logs

export OMP_NUM_THREADS=4
export PYTHONFAULTHANDLER=1
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ── paths & settings ──────────────────────────────────────────────────────────
# Directory holding *_predictions.jsonl files written by infer_nextqa_oe_jsonl
export PRED_DIR="${PRED_DIR:-$ROOT_DIR/results/nextqa_oe}"

# Reference CSV (auto-detected from filename if not set)
# export REF_CSV="/path/to/override.csv"

export REF_BASE_DIR="${REF_BASE_DIR:-$ROOT_DIR/dataset/nextqa/openend}"
export ADD_REF_DIR="${ADD_REF_DIR:-/home/hmkang/project/NExT-OE/dataset/nextqa}"

# Set FORCE=1 to re-score files that already have a .wups.json
export FORCE="${FORCE:-0}"

# ── run ───────────────────────────────────────────────────────────────────────
bash /home/hmkang/init_ross.sh bash "$ROOT_DIR/scripts/nextqa_oe/score_nextqa_oe.sh"
