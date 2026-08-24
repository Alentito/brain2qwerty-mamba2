#!/bin/bash
#SBATCH --job-name=b2q_debug
#SBATCH --partition=k2-hipri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=slurm/logs/debug_%j.out

# Smoke test: 1 timeline, 2 epochs, 1 GPU. Pass criteria: loss decreases, no NaN.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export BRAIN2QWERTY_STUDIES="$PWD/SpanishBCBL"
export BRAIN2QWERTY_CACHE="$PWD/.cache/brain2qwerty"
export BRAIN2QWERTY_RESULTS="$PWD/.cache/brain2qwerty/results_debug"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -m brain2qwerty_v1.main debug
echo "[debug] done."
