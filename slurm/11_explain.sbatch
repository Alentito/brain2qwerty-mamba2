#!/bin/bash
#SBATCH --job-name=b2q_xai
#SBATCH --partition=k2-gpu-a100        # fallback: k2-gpu-h100 or k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/logs/xai_%A.out

# Explainability pass over the test split for one trained checkpoint.
# Produces per-sentence mixing maps, dt profiles, signal saliency (heatmap +
# 2D channel topomap + time course), char-probability heatmaps, and the
# aggregate memory-horizon curve. ~10-15 min for the default 6 sentences.
#
# Usage:  CKPT=<path/to/best.ckpt> CORE=mamba sbatch slurm/11_explain.sbatch
# (CORE must match the checkpoint: mamba | mamba3 | transformer)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; else source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate "$HOME/sharedscratch/conda/envs/b2q"; fi

export BRAIN2QWERTY_STUDIES="${BRAIN2QWERTY_STUDIES:-$HOME/sharedscratch/B2Q/code/SpanishBCBL_3subj}"
[ -d "$BRAIN2QWERTY_STUDIES" ] || export BRAIN2QWERTY_STUDIES="$PWD/SpanishBCBL_3subj"
export BRAIN2QWERTY_CACHE="${BRAIN2QWERTY_CACHE:-$HOME/sharedscratch/B2Q/cache_v1mamba}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CORE="${CORE:-mamba}"
CKPT="${CKPT:?set CKPT=<path/to/best.ckpt>}"
TAG=$(basename "$(dirname "$CKPT")")

python explain_mamba.py --ckpt "$CKPT" --core "$CORE" --small \
    --n-sentences "${N_SENT:-6}" --out "explain_out/$TAG"
