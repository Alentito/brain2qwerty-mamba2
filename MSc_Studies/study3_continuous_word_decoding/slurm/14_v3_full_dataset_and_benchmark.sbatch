#!/bin/bash
# =========================================================================== #
# Brain2Qwerty V3 Full Dataset Training & GPU O(N) vs O(N^2) Complexity Profiler
#
# Array Tasks:
#   0: Conformer Baseline (Full Dataset, 1024-dim, 8 Layers)
#   1: Mamba-3 Stabilized Hybrid (Full Dataset, 1024-dim, 8 Layers)
#   2: BiMamba-2 + Gated MLP (Full Dataset, 1024-dim, 8 Layers)
#   3: GPU Computational Complexity & VRAM Profiling Suite (A100)
#
# Submit with:
#   sbatch slurm/14_v3_full_dataset_and_benchmark.sbatch
# Or single task:
#   sbatch --array=1 slurm/14_v3_full_dataset_and_benchmark.sbatch
# =========================================================================== #
#SBATCH --job-name=b2q_v3_full
#SBATCH --output=slurm/logs/v3_full_%A_%a.out
#SBATCH --error=slurm/logs/v3_full_%A_%a.out
#SBATCH --partition=k2-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --array=0-3

set -euo pipefail
mkdir -p slurm/logs

echo "=== Starting Brain2Qwerty V3 Full Dataset Job: Task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date) ==="

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

# 3. Execute Selected Task
case ${SLURM_ARRAY_TASK_ID} in
    0)
        echo ">>> Running Task 0: Conformer Baseline (Full Dataset, LR=8e-4)"
        python -m brain2qwerty_v3.main run \
            --core conformer \
            --lr 8e-4 \
            --wd 1e-3 \
            --tag full-dataset \
            --devices 1
        ;;
    1)
        echo ">>> Running Task 1: Mamba-3 Stabilized Hybrid (Full Dataset, LR=3e-4)"
        python -m brain2qwerty_v3.main run \
            --core mamba3_hybrid_stabilized \
            --lr 3e-4 \
            --wd 1e-3 \
            --tag full-dataset \
            --devices 1
        ;;
    2)
        echo ">>> Running Task 2: BiMamba-2 + Gated MLP (Full Dataset, LR=3e-4)"
        python -m brain2qwerty_v3.main run \
            --core mamba_mlp \
            --lr 3e-4 \
            --wd 1e-3 \
            --tag full-dataset \
            --devices 1
        ;;
    3)
        echo ">>> Running Task 3: GPU Latency, Memory & O(N) vs O(N^2) Complexity Profiling"
        python benchmark_complexity.py --device cuda --full
        ;;
    *)
        echo "Invalid task ID: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

echo "=== Finished V3 Full Dataset Job Task ${SLURM_ARRAY_TASK_ID} at $(date) ==="
