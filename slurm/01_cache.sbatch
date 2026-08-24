#!/bin/bash
#SBATCH --job-name=b2q_cache
#SBATCH --partition=k2-hipri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=slurm/logs/cache_%j.out

# Pre-warms the CPU-bound MEG feature cache (50 Hz, 0.1-20 Hz filter,
# RobustScaler, keystroke-aligned 500 ms windows). CPU-only, no GPU needed.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export BRAIN2QWERTY_STUDIES="$PWD/SpanishBCBL"
export BRAIN2QWERTY_CACHE="$PWD/.cache/brain2qwerty"
export BRAIN2QWERTY_RESULTS="$BRAIN2QWERTY_CACHE/results"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -m brain2qwerty_v1.main cache
echo "[cache] done."
