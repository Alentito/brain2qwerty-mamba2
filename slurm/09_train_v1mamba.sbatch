#!/bin/bash
#SBATCH --job-name=b2q_v1mamba
#SBATCH --partition=k2-gpu-a100        # fallback: k2-gpu-h100 or k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1                    # task 0 = mamba, task 1 = transformer
#SBATCH --output=slurm/logs/v1mamba_%A_%a.out

# Trains the two ablation arms (small 512-dim model, subjects S15/S16/S6,
# 200-epoch colab preset). Requires the warmed cache (job 07).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Env: local .venv if present (Mac), otherwise the Kelvin-2 conda env.
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; else source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate "$HOME/sharedscratch/conda/envs/b2q"; fi

# The study path must EXIST even though all features come from the cache
# (the study object validates it at construction).
export BRAIN2QWERTY_STUDIES="${BRAIN2QWERTY_STUDIES:-$HOME/sharedscratch/B2Q/code/SpanishBCBL_3subj}"
[ -d "$BRAIN2QWERTY_STUDIES" ] || export BRAIN2QWERTY_STUDIES="$PWD/SpanishBCBL_3subj"
export BRAIN2QWERTY_CACHE="${BRAIN2QWERTY_CACHE:-$HOME/sharedscratch/B2Q/cache_v1mamba}"
[ -d "$(dirname "$BRAIN2QWERTY_CACHE")" ] || export BRAIN2QWERTY_CACHE="$PWD/.cache/b2q_v1mamba"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CORES=(mamba transformer)
CORE=${CORES[$SLURM_ARRAY_TASK_ID]}

python -m brain2qwerty_v1_mamba.main colab --core "$CORE" --seed 33
echo "[train] arm=$CORE done."
