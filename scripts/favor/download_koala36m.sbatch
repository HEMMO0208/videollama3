#!/usr/bin/env bash
#SBATCH --job-name=favor-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=all
#SBATCH --exclude=node10

set -euo pipefail

ROOT_DIR="/home/hmkang/project/videollama3"
cd "$ROOT_DIR"
mkdir -p logs

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

SFT_JSON="${SFT_JSON:-$ROOT_DIR/dataset/favor/sft.json}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/FAVOR/videos/FAVOR-Train}"
COOKIES="${COOKIES:-$ROOT_DIR/yt_cookies.txt}"
WORKERS="${WORKERS:-4}"
LOG_FILE="logs/download_koala36m_${SLURM_JOB_ID}.log"

echo "=========================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "SFT JSON : $SFT_JSON"
echo "OUT DIR  : $OUT_DIR"
echo "COOKIES  : $COOKIES"
echo "WORKERS  : $WORKERS"
echo "LOG      : $LOG_FILE"
echo "=========================================="

bash /home/hmkang/init_ross.sh bash -c "
    pip install av yt-dlp -q &&
    python '$ROOT_DIR/scripts/favor/download_koala36m.py' \
        --sft_json  '$SFT_JSON' \
        --out_dir   '$OUT_DIR' \
        --cookies   '$COOKIES' \
        --workers   '$WORKERS' \
        --log_file  '$LOG_FILE'
"
