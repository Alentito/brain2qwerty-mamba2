#!/bin/bash
#SBATCH --job-name=b2q_train
#SBATCH --partition=k2-hipri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/train_%j.out

# Full training: 300 epochs, patience 30, AdamW 5e-5 + OneCycleLR.
# The config targets 8 GPUs but automatically falls back to however many are
# visible (here 1). Checkpoints + per-epoch metrics + per-sentence prediction
# JSON land in $BRAIN2QWERTY_RESULTS.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Env: local .venv if present (Mac), otherwise the Kelvin-2 conda env.
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; else source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate "$HOME/sharedscratch/conda/envs/b2q"; fi

export BRAIN2QWERTY_STUDIES="$PWD/SpanishBCBL"
export BRAIN2QWERTY_CACHE="$PWD/.cache/brain2qwerty"
export BRAIN2QWERTY_RESULTS="$BRAIN2QWERTY_CACHE/results"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -m brain2qwerty_v1.main train --seed 33
echo "[train] done."
