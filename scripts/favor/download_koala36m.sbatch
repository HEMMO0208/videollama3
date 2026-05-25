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

# node 바이너리 탐색 (nvm 경로 포함)
# nvm이 로드되지 않은 SLURM 환경에서도 node를 찾을 수 있도록 명시적 PATH 추가
export PATH="$HOME/.nvm/versions/node/$(ls $HOME/.nvm/versions/node/ 2>/dev/null | sort -V | tail -1)/bin:$HOME/.npm-global/bin:$PATH"
NODE_BIN=$(which node 2>/dev/null || echo "node")

echo "=========================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "SFT JSON : $SFT_JSON"
echo "OUT DIR  : $OUT_DIR"
echo "COOKIES  : $COOKIES"
echo "WORKERS  : $WORKERS"
echo "NODE     : $NODE_BIN ($(node --version 2>/dev/null || echo 'not found'))"
echo "LOG      : $LOG_FILE"
echo "=========================================="

bash /home/hmkang/init_ross.sh bash -c "
    pip install av yt-dlp -q &&
    python '$ROOT_DIR/scripts/favor/download_koala36m.py' \
        --sft_json  '$SFT_JSON' \
        --out_dir   '$OUT_DIR' \
        --cookies   '$COOKIES' \
        --workers   '$WORKERS' \
        --node_bin  '$NODE_BIN' \
        --log_file  '$LOG_FILE'
"
