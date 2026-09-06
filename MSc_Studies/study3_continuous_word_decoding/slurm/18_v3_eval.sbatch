#!/bin/bash
#SBATCH --job-name=b2q_v3_eval
#SBATCH --partition=k2-gpu-a100        # fallback: k2-gpu-h100 or k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --array=0-2                    # 0: Conformer, 1: BiMamba-2+Gated-MLP, 2: Mamba-3 Stabilized Hybrid
#SBATCH --output=slurm/logs/v3_eval_%A_%a.out

# ============================================================================ #
# Brain2Qwerty V3 Word-Level Evaluation Suite (Study 3, SpanishBCBL 3-Subj)
#
# Array Tasks:
#   0: Conformer Baseline
#   1: BiMamba-2 + Gated MLP
#   2: Mamba-3 Stabilized Hybrid
#
# Submit with:
#   sbatch slurm/18_v3_eval.sbatch
# Or single architecture (e.g. Mamba-3 Hybrid only):
#   sbatch --array=2 slurm/18_v3_eval.sbatch
# ============================================================================ #

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

# Environment activation
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
elif [ -f "$HOME/sharedscratch/conda/envs/b2q/bin/activate" ]; then
    source "$HOME/sharedscratch/conda/envs/b2q/bin/activate"
else
    source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "$HOME/sharedscratch/conda/envs/b2q" 2>/dev/null || true
fi

export BRAIN2QWERTY_STUDIES="${BRAIN2QWERTY_STUDIES:-$HOME/sharedscratch/B2Q/code/SpanishBCBL_3subj}"
[ -d "$BRAIN2QWERTY_STUDIES" ] || export BRAIN2QWERTY_STUDIES="$PWD/SpanishBCBL_3subj"

export BRAIN2QWERTY_CACHE="${BRAIN2QWERTY_CACHE:-$HOME/sharedscratch/B2Q/cache_v1mamba}"
[ -d "$(dirname "$BRAIN2QWERTY_CACHE")" ] || export BRAIN2QWERTY_CACHE="$PWD/.cache/b2q_v1mamba"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
echo "=== Starting Brain2Qwerty V3 Evaluation Task: $TASK_ID on $(hostname) at $(date) ==="

case $TASK_ID in
  0)
    CORE="conformer"
    TAG="v3-conformer-baseline"
    ;;
  1)
    CORE="mamba_mlp"
    TAG="v3-mamba-gated-mlp"
    ;;
  2)
    CORE="mamba3_hybrid_stabilized"
    TAG="v3-mamba3-stabilized-hybrid"
    ;;
  *)
    echo "Unknown task id $TASK_ID"
    exit 1
    ;;
esac

RESULTS_DIR="${BRAIN2QWERTY_CACHE}/results/v3-${CORE}-${TAG}"
CKPT="${RESULTS_DIR}/best_llm.ckpt"

if [ ! -f "$CKPT" ]; then
    echo "Warning: $CKPT not found, checking for best_ctc.ckpt or other checkpoints..."
    if [ -f "${RESULTS_DIR}/best_ctc.ckpt" ]; then
        CKPT="${RESULTS_DIR}/best_ctc.ckpt"
    else
        CKPT=$(find "${RESULTS_DIR}" -name "*.ckpt" 2>/dev/null | head -n 1 || true)
    fi
fi

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "Error: No checkpoint found in ${RESULTS_DIR}"
    exit 1
fi

echo ">>> Evaluating Core: $CORE with Checkpoint: $CKPT"

python -m brain2qwerty_v3.main eval \
    --core "$CORE" \
    --tag "$TAG" \
    --ckpt "$CKPT" \
    --subjects S15 S16 S6 \
    --devices 1

echo ">>> Extracting Predictions and Computing Per-Subject Metrics..."
python -m brain2qwerty_v3.scripts.extract_predictions \
    --input "$RESULTS_DIR" \
    --split test

echo "=== Finished V3 Evaluation Task $TASK_ID at $(date) ==="
