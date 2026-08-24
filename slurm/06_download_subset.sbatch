#!/bin/bash
#SBATCH --job-name=b2q_subset_dl
#SBATCH --partition=k2-hipri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/subset_dl_%j.out

# Downloads ONLY the MEG recordings + logs for subjects S15, S16, S6
# (the gated HF repo requires: hf auth login  — or export HF_TOKEN=...).
# Layout expected by the study loader:
#   SpanishBCBL_3subj/MEG/FIF/15_*/..., 16_*/..., 06_*/...
#   SpanishBCBL_3subj/MEG/logs/**/S{15,16,6}-session*.mat
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Env: local .venv if present (Mac), otherwise the Kelvin-2 conda env.
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; else source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate "$HOME/sharedscratch/conda/envs/b2q"; fi

TARGET="$PWD/SpanishBCBL_3subj"
mkdir -p slurm/logs "$TARGET"

hf download bcbl190626/SpanishBCBL --repo-type dataset --local-dir "$TARGET" \
    --include "MEG/FIF/15_*/*" \
              "MEG/FIF/16_*/*" \
              "MEG/FIF/06_*/*" \
              "MEG/logs/*" \
              "README.md"

echo "[subset] downloaded:"
du -sh "$TARGET"
find "$TARGET/MEG/FIF" -maxdepth 1 -type d
