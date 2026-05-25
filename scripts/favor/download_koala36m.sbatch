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

# init_ross.sh 컨테이너 안에서 Node.js를 설치하고 Python 스크립트를 실행.
# node가 PATH에 없는 환경(SLURM 컨테이너 등)을 위해
# nodeenv(pip)로 pre-built Node.js 바이너리를 /tmp에 설치.
bash /home/hmkang/init_ross.sh bash -c "
    set -e
    pip install av yt-dlp nodeenv -q

    # 컨테이너 내부에서 node 탐색
    NODE_BIN=\$(which node 2>/dev/null || true)
    if [ -z \"\$NODE_BIN\" ]; then
        echo '[sbatch] node not in PATH — installing via nodeenv (pre-built LTS)...'
        NODEENV_DIR=\"/tmp/nodeenv_${SLURM_JOB_ID}\"
        nodeenv --prebuilt --node=lts \"\$NODEENV_DIR\" -q
        NODE_BIN=\"\$NODEENV_DIR/bin/node\"
        echo \"[sbatch] Node.js installed: \$NODE_BIN (\$(\$NODE_BIN --version))\"
    else
        echo \"[sbatch] Node.js found: \$NODE_BIN (\$(\$NODE_BIN --version))\"
    fi

    python '$ROOT_DIR/scripts/favor/download_koala36m.py' \
        --sft_json  '$SFT_JSON' \
        --out_dir   '$OUT_DIR' \
        --cookies   '$COOKIES' \
        --workers   '$WORKERS' \
        --node_bin  \"\$NODE_BIN\" \
        --log_file  '$LOG_FILE'
"
