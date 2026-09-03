#!/bin/bash
#SBATCH --job-name=b2q_decswap
#SBATCH --partition=k2-gpu-a100        # fallback: k2-gpu-h100 or k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00                # 3-subject swap: ~110 epochs, far under this
#SBATCH --output=slurm/logs/decswap_%j.out

# ============================================================================ #
# Brain2Qwerty V3 DECODER-SWAP: TinyLlama -> Qwen3.5-0.8B (LoRA r=8, alpha=16)
#
# Warm-starts the Mamba-3 Stabilized Hybrid encoder from a donor checkpoint
# (old 3-subject TinyLlama-era run), RESETS llm.* / word_proj_adapter.*, and
# re-trains the decoder side against Spanish-native Qwen3.5-0.8B.
#
# Submit WITHOUT editing this file:
#   sbatch --export=ALL,DONOR_CKPT=/abs/path/to/donor.ckpt \
#          slurm/17_v3_decoder_swap_qwen.sbatch
# Optional overrides (same --export=ALL,... list):
#   SUBJECTS="S15 S16 S6"   (MUST match the donor's training subjects)
#   ENCODER_LR=1e-5  LLM_START=60  MAX_EPOCHS=110  USE_UNSLOTH=1
# ============================================================================ #

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

# --- parameters (env-overridable at submit time) ---------------------------- #
DONOR_CKPT="${DONOR_CKPT:-}"
SUBJECTS="${SUBJECTS:-S15 S16 S6}"
ENCODER_LR="${ENCODER_LR:-1e-5}"
LLM_START="${LLM_START:-60}"
MAX_EPOCHS="${MAX_EPOCHS:-110}"
USE_UNSLOTH="${USE_UNSLOTH:-0}"

if [ -z "$DONOR_CKPT" ] || [ ! -f "$DONOR_CKPT" ]; then
    echo "ERROR: DONOR_CKPT missing or not a file: '$DONOR_CKPT'"
    echo "Submit with: sbatch --export=ALL,DONOR_CKPT=/abs/path/to/donor.ckpt $0"
    echo "Find donors:  find \$HOME/sharedscratch/B2Q -name '*.ckpt' | sort"
    exit 1
fi

# Environment activation
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
elif [ -f "$HOME/sharedscratch/conda/envs/b2q/bin/activate" ]; then
    source "$HOME/sharedscratch/conda/envs/b2q/bin/activate"
else
    source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "$HOME/sharedscratch/conda/envs/b2q" 2>/dev/null || true
fi

# Data + DEDICATED cache dir (TinyLlama-era word-embedding caches are invalid
# under Qwen3.5-0.8B; a fresh dir avoids clobbering the 19-subject cache too)
export BRAIN2QWERTY_STUDIES="${BRAIN2QWERTY_STUDIES:-/mnt/scratch2/users/atito/B2Q/dataset/SpanishBCBL}"
export BRAIN2QWERTY_CACHE="${BRAIN2QWERTY_CACHE:-$HOME/sharedscratch/B2Q/cache_decswap}"
mkdir -p "$BRAIN2QWERTY_CACHE"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# Qwen3.5-0.8B must be in the offline HF cache BEFORE submitting (login node):
#   export HF_HOME=$HOME/sharedscratch/B2Q/hf_cache
#   huggingface-cli download Qwen/Qwen3.5-0.8B
# roberta-large is also needed for the SemER metric at test time:
#   huggingface-cli download roberta-large
export HF_HOME="${HF_HOME:-$HOME/sharedscratch/B2Q/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Decoder-swap (Qwen3.5-0.8B) on $(hostname) ==="
echo "Donor:    $DONOR_CKPT"
echo "Subjects: $SUBJECTS"
echo "Cache:    $BRAIN2QWERTY_CACHE"
nvidia-smi || true

EXTRA_ARGS=""
if [ "$USE_UNSLOTH" = "1" ]; then
    EXTRA_ARGS="--use-unsloth"
fi

python -m brain2qwerty_v3.main run \
    --core mamba3_hybrid_stabilized \
    --subjects $SUBJECTS \
    --init-ckpt "$DONOR_CKPT" \
    --encoder-lr "$ENCODER_LR" \
    --lr 1.5e-4 \
    --wd 1e-3 \
    --contrastive-start 0 \
    --llm-start "$LLM_START" \
    --max-epochs "$MAX_EPOCHS" \
    --tag decoder-swap-qwen \
    --devices 1 \
    $EXTRA_ARGS

echo "=== Finished decoder-swap at $(date) ==="
