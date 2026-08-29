#!/bin/bash
#SBATCH --job-name=b2q_v3
#SBATCH --partition=k2-gpu-a100        # fallback: k2-gpu-h100 or k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --array=0-2                    # 3 versions: Conformer, BiMamba-Gated-MLP, Mamba3-Stabilized-Hybrid
#SBATCH --output=slurm/logs/v3_%A_%a.out

# ============================================================================ #
# Brain2Qwerty V3 Word-Level End-to-End Decoding Suite (SpanishBCBL 3-Subj)
#
# Array Tasks:
#   0: Version 1 -> Conformer Baseline (V2 reference on SpanishBCBL, CTC + Contrastive + LoRA LLM)
#   1: Version 2 -> BiMamba-2 + Gated Fusion + FFN MLP (Round 3 Champion architecture)
#   2: Version 3 -> Mamba-3 Stabilized Hybrid (BCNorm + RoPE + Softplus Clamping + Attention)
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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
echo "=== Starting Brain2Qwerty V3 Task: $TASK_ID on $(hostname) at $(date) ==="

case $TASK_ID in
  0)
    CORE="conformer"
    TAG="v3-conformer-baseline"
    LR="8e-4"
    ;;
  1)
    CORE="mamba_mlp"
    TAG="v3-mamba-gated-mlp"
    LR="3e-4"
    ;;
  2)
    CORE="mamba3_hybrid_stabilized"
    TAG="v3-mamba3-stabilized-hybrid"
    LR="3e-4"
    ;;
  *)
    echo "Unknown task id $TASK_ID"
    exit 1
    ;;
esac

echo ">>> Running V3 Word-Level Decoder with Core: $CORE (LR=$LR, SpanishBCBL 3-Subj)"

python -m brain2qwerty_v3.main train \
    --core "$CORE" \
    --lr "$LR" \
    --wd 1e-3 \
    --subjects S15 S16 S6 \
    --tag "$TAG" \
    --devices 1

echo "=== Finished V3 Task $TASK_ID at $(date) ==="
