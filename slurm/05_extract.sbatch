#!/bin/bash
#SBATCH --job-name=b2q_extract
#SBATCH --partition=k2-hipri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/logs/extract_%j.out

# Turn the prediction JSON into predictions.csv (one row per sentence,
# reconstructed text + CER/WER per subject). CPU-only.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export BRAIN2QWERTY_CACHE="$PWD/.cache/brain2qwerty"
export BRAIN2QWERTY_RESULTS="$BRAIN2QWERTY_CACHE/results"

python -m brain2qwerty_v1.scripts.extract_predictions \
    --input "$BRAIN2QWERTY_RESULTS/callbacks" \
    --split test \
    --output "$BRAIN2QWERTY_RESULTS/predictions.csv"

echo "[extract] done -> $BRAIN2QWERTY_RESULTS/predictions.csv"
