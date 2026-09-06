#!/bin/bash
#SBATCH --job-name=b2q_19subj
#SBATCH --partition=k2-gpu-a100        # fallback: k2-gpu-h100 or k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-2%1                  # 0: Conformer, 1: Mamba-3 Stabilized Hybrid, 2: BiMamba-2 + MLP
                                       # %1 = ONLY ONE task at a time: we have a single A100,
                                       # concurrent tasks would share GPU 0 and OOM together.
#SBATCH --output=slurm/logs/19subj_%A_%a.out

# ============================================================================ #
# Brain2Qwerty V3 Word-Level Training on FULL 19-PARTICIPANT SpanishBCBL
#
# Array Tasks:
#   0: Conformer Baseline (19 Participants, Full Dataset)
#   1: Mamba-3 Stabilized Hybrid (19 Participants, Full Dataset)
#   2: BiMamba-2 + Gated MLP (19 Participants, Full Dataset)
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

# Point to Full 19-Participant Dataset
export BRAIN2QWERTY_STUDIES="${BRAIN2QWERTY_STUDIES:-/mnt/scratch2/users/atito/B2Q/dataset/SpanishBCBL}"
export BRAIN2QWERTY_CACHE="${BRAIN2QWERTY_CACHE:-$HOME/sharedscratch/B2Q/cache_19subj}"
mkdir -p "$BRAIN2QWERTY_CACHE"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# Qwen3.5-0.8B weights: compute nodes often have no internet. Pre-download on
# the login node ONCE before submitting:
#   source ~/sharedscratch/conda/envs/b2q/bin/activate
#   export HF_HOME=$HOME/sharedscratch/B2Q/hf_cache
#   huggingface-cli download Qwen/Qwen3.5-0.8B
# then the job runs fully offline:
export HF_HOME="${HF_HOME:-$HOME/sharedscratch/B2Q/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# STALE CACHE WARNING: word-embedding caches under $BRAIN2QWERTY_CACHE built
# with TinyLlama are invalid now that LLM=Qwen3.5-0.8B. If a run crashes with
# an embedding-dim mismatch, delete the word-embedding cache and rerun.

echo "=== Starting Brain2Qwerty V3 (19-Participant Dataset) Task $SLURM_ARRAY_TASK_ID on $(hostname) ==="
echo "Study Path:  $BRAIN2QWERTY_STUDIES"
echo "Cache Path:  $BRAIN2QWERTY_CACHE"
nvidia-smi || true

case ${SLURM_ARRAY_TASK_ID} in
    0)
        echo ">>> Running Task 0: Conformer Baseline (19 Subjects, Full Dataset)"
        # LR halved (8e-4 -> 4e-4): effective batch dropped 128 -> 64 with the
        # bs16 x accum4 memory fix; 8e-4 collapsed CTC to blank-only output.
        python -m brain2qwerty_v3.main run \
            --core conformer \
            --lr 4e-4 \
            --wd 1e-3 \
            --tag 19subj-full-dataset \
            --devices 1
        ;;
    1)
        echo ">>> Running Task 1: Mamba-3 Stabilized Hybrid (19 Subjects, Full Dataset)"
        # LR halved (3e-4 -> 1.5e-4): same effective-batch 128 -> 64 reason.
        python -m brain2qwerty_v3.main run \
            --core mamba3_hybrid_stabilized \
            --lr 1.5e-4 \
            --wd 1e-3 \
            --tag 19subj-full-dataset \
            --devices 1
        ;;
    2)
        echo ">>> Running Task 2: BiMamba-2 + Gated MLP (19 Subjects, Full Dataset)"
        python -m brain2qwerty_v3.main run \
            --core mamba_mlp \
            --lr 3e-4 \
            --wd 1e-3 \
            --tag 19subj-full-dataset \
            --devices 1
        ;;
    *)
        echo "Unknown task $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

echo "=== Finished Task $SLURM_ARRAY_TASK_ID at $(date) ==="
