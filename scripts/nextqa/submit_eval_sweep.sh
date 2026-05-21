#!/usr/bin/env bash
# Submit NextQA eval jobs sweeping over MAX_FRAMES values.
#
# Usage:
#   ./scripts/nextqa/submit_eval_sweep.sh <model_kind> <max_frames...>
#
# Examples:
#   ./scripts/nextqa/submit_eval_sweep.sh baseline 100
#   ./scripts/nextqa/submit_eval_sweep.sh diffusion 8 16 32 64 100
#   ./scripts/nextqa/submit_eval_sweep.sh all 8 16 32 64 100
#
# <model_kind>  : baseline | diffusion | all
# <max_frames>  : one or more integer values

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <model_kind> <max_frames...>"
    echo "  model_kind : baseline | diffusion | all"
    echo "  max_frames : one or more integers"
    exit 1
fi

MODEL_KIND="$1"
shift
MAX_FRAMES_LIST=("$@")

case "$MODEL_KIND" in
    baseline)  KINDS=(baseline) ;;
    diffusion) KINDS=(diffusion) ;;
    all)       KINDS=(baseline diffusion) ;;
    *)
        echo "Unknown model_kind '$MODEL_KIND'. Use baseline, diffusion, or all."
        exit 1
        ;;
esac

for kind in "${KINDS[@]}"; do
    for mf in "${MAX_FRAMES_LIST[@]}"; do
        run_name="${kind}_mf${mf}"
        echo "Submitting: MODEL_KIND=$kind  MAX_FRAMES=$mf  RUN_NAME=$run_name"
        sbatch \
            --export=ALL,MODEL_KIND="$kind",MAX_FRAMES="$mf",RUN_NAME="$run_name" \
            "$SCRIPT_DIR/eval_nextqa_jsonl_${kind}.sbatch"
    done
done
