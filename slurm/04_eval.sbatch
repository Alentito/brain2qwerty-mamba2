#!/bin/bash
#SBATCH --job-name=b2q_eval
#SBATCH --partition=k2-hipri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --output=slurm/logs/eval_%j.out

# Evaluate best.ckpt on the test split and save the test prediction JSON.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export BRAIN2QWERTY_STUDIES="$PWD/SpanishBCBL"
export BRAIN2QWERTY_CACHE="$PWD/.cache/brain2qwerty"
export BRAIN2QWERTY_RESULTS="$BRAIN2QWERTY_CACHE/results"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CKPT="${1:-$BRAIN2QWERTY_RESULTS/best.ckpt}"
python -m brain2qwerty_v1.main eval --ckpt "$CKPT"
echo "[eval] done."
