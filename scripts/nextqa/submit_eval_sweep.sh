#!/usr/bin/env bash
# Submit NextQA eval jobs sweeping over MAX_FRAMES values.
#
# Usage:
#   ./scripts/nextqa/submit_eval_sweep.sh <model_kind> <max_frames...> [-- <sbatch_args...>]
#
# Examples:
#   ./scripts/nextqa/submit_eval_sweep.sh baseline 100
#   ./scripts/nextqa/submit_eval_sweep.sh diffusion 8 16 32 64 100
#   ./scripts/nextqa/submit_eval_sweep.sh causal_diffusion 8 16 32 64 100
#   ./scripts/nextqa/submit_eval_sweep.sh all 8 16 32 64 100
#   ./scripts/nextqa/submit_eval_sweep.sh all 8 16 32 64 100 -- --dependency=afterok:12345
#   ./scripts/nextqa/submit_eval_sweep.sh baseline 32 64 -- --dependency=afterany:111,222 --mail-type=END
#
# <model_kind>  : baseline | diffusion | causal_diffusion | all
# <max_frames>  : one or more integer values
# --            : everything after this is passed verbatim to sbatch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <model_kind> <max_frames...> [-- <sbatch_args...>]"
    echo "  model_kind : baseline | diffusion | causal_diffusion | all"
    echo "  max_frames : one or more integers"
    exit 1
fi

MODEL_KIND="$1"
shift

# Split args on '--'
MAX_FRAMES_LIST=()
EXTRA_SBATCH_ARGS=()
found_sep=0
for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
        found_sep=1
        continue
    fi
    if [[ $found_sep -eq 0 ]]; then
        MAX_FRAMES_LIST+=("$arg")
    else
        EXTRA_SBATCH_ARGS+=("$arg")
    fi
done

if [[ ${#MAX_FRAMES_LIST[@]} -eq 0 ]]; then
    echo "Error: no max_frames values provided."
    exit 1
fi

case "$MODEL_KIND" in
    baseline)  KINDS=(baseline) ;;
    diffusion) KINDS=(diffusion) ;;
    causal_diffusion) KINDS=(causal_diffusion) ;;
    all)       KINDS=(baseline diffusion causal_diffusion) ;;
    *)
        echo "Unknown model_kind '$MODEL_KIND'. Use baseline, diffusion, causal_diffusion, or all."
        exit 1
        ;;
esac

for kind in "${KINDS[@]}"; do
    for mf in "${MAX_FRAMES_LIST[@]}"; do
        run_name="${kind}_mf${mf}"
        echo "Submitting: MODEL_KIND=$kind  MAX_FRAMES=$mf  RUN_NAME=$run_name${EXTRA_SBATCH_ARGS:+  sbatch_args=${EXTRA_SBATCH_ARGS[*]}}"
        sbatch \
            --export=ALL,MODEL_KIND="$kind",MAX_FRAMES="$mf",RUN_NAME="$run_name" \
            "${EXTRA_SBATCH_ARGS[@]}" \
            "$SCRIPT_DIR/eval_nextqa_jsonl_${kind}.sbatch"
    done
done
