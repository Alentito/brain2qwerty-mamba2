#!/bin/bash
# =========================================================================== #
# Brain2Qwerty V3 Multi-Seed Statistical Evaluation Suite
#
# Runs 3 Seeds (123, 42, 999) across all 3 architectures:
#
# Seed 123:
#   0: Conformer
#   1: BiMamba-2 + Gated MLP
#   2: Mamba-3 Stabilized Hybrid
#
# Seed 42:
#   3: Conformer
#   4: BiMamba-2 + Gated MLP
#   5: Mamba-3 Stabilized Hybrid
#
# Seed 999:
#   6: Conformer
#   7: BiMamba-2 + Gated MLP
#   8: Mamba-3 Stabilized Hybrid
#
# Submit with:
#   sbatch slurm/15_v3_multi_seed_evaluation.sbatch
# Or run specific seeds (e.g. Seed 42 only):
#   sbatch --array=3-5 slurm/15_v3_multi_seed_evaluation.sbatch
# =========================================================================== #
#SBATCH --job-name=b2q_v3_seeds
#SBATCH --output=slurm/logs/v3_seed_%A_%a.out
#SBATCH --error=slurm/logs/v3_seed_%A_%a.out
#SBATCH --partition=k2-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --array=0-8

set -euo pipefail
mkdir -p slurm/logs

echo "=== Starting Brain2Qwerty V3 Multi-Seed Task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date) ==="

# 1. Environment Activation
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || \
source "/opt/conda/etc/profile.d/conda.sh" 2>/dev/null || true

CONDA_ENV="${CONDA_ENV:-$HOME/sharedscratch/conda/envs/b2q}"
if [ -d "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
    echo "Activated Conda environment: $CONDA_ENV"
else
    echo "Warning: Conda env not found at $CONDA_ENV, using default python"
fi

# 2. Environment Variables
export BRAIN2QWERTY_STUDIES="${BRAIN2QWERTY_STUDIES:-$HOME/sharedscratch/B2Q/code/SpanishBCBL_3subj}"
[ -d "$BRAIN2QWERTY_STUDIES" ] || export BRAIN2QWERTY_STUDIES="$PWD/SpanishBCBL_3subj"

export BRAIN2QWERTY_CACHE="${BRAIN2QWERTY_CACHE:-$HOME/sharedscratch/B2Q/cache_v1mamba}"
mkdir -p "$BRAIN2QWERTY_CACHE"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "Using Study Path:  $BRAIN2QWERTY_STUDIES"
echo "Using Cache Path:  $BRAIN2QWERTY_CACHE"
nvidia-smi || true

# 3. Determine Core, Seed, and Learning Rate
case ${SLURM_ARRAY_TASK_ID} in
    # Seed 123
    0) CORE="conformer";                  SEED=123; LR=8e-4 ;;
    1) CORE="mamba_mlp";                  SEED=123; LR=3e-4 ;;
    2) CORE="mamba3_hybrid_stabilized";   SEED=123; LR=3e-4 ;;

    # Seed 42
    3) CORE="conformer";                  SEED=42;  LR=8e-4 ;;
    4) CORE="mamba_mlp";                  SEED=42;  LR=3e-4 ;;
    5) CORE="mamba3_hybrid_stabilized";   SEED=42;  LR=3e-4 ;;

    # Seed 999
    6) CORE="conformer";                  SEED=999; LR=8e-4 ;;
    7) CORE="mamba_mlp";                  SEED=999; LR=3e-4 ;;
    8) CORE="mamba3_hybrid_stabilized";   SEED=999; LR=3e-4 ;;

    *) echo "Invalid task ID: ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac

TAG="seed-${SEED}"
echo ">>> Running Task: ${CORE} | Seed: ${SEED} | LR: ${LR} | Tag: ${TAG}"

python -m brain2qwerty_v3.main train \
    --core "${CORE}" \
    --seed "${SEED}" \
    --lr "${LR}" \
    --wd 1e-3 \
    --tag "${TAG}" \
    --devices 1

echo "=== Finished Task ${SLURM_ARRAY_TASK_ID} (${CORE}, Seed ${SEED}) at $(date) ==="
