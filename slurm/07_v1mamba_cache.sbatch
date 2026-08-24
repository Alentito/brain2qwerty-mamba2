#!/bin/bash
#SBATCH --job-name=b2q_v1mamba_cache
#SBATCH --partition=k2-hipri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=slurm/logs/v1mamba_cache_%j.out

# Warms the V1-Mamba feature cache from the 3-subject subset, then packs it
# for upload to Kaggle. After this job, Kaggle only needs the tarball —
# no raw data ever leaves the cluster.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Env: local .venv if present (Mac), otherwise the Kelvin-2 conda env.
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; else source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate "$HOME/sharedscratch/conda/envs/b2q"; fi

# Paths: honor pre-set env vars; default to the Kelvin-2 locations, with a
# repo-local fallback for Mac/local runs.
export BRAIN2QWERTY_STUDIES="${BRAIN2QWERTY_STUDIES:-$HOME/sharedscratch/B2Q/code/SpanishBCBL_3subj}"
[ -d "$BRAIN2QWERTY_STUDIES" ] || export BRAIN2QWERTY_STUDIES="$PWD/SpanishBCBL_3subj"
export BRAIN2QWERTY_CACHE="${BRAIN2QWERTY_CACHE:-$HOME/sharedscratch/B2Q/cache_v1mamba}"
[ -d "$(dirname "$BRAIN2QWERTY_CACHE")" ] || export BRAIN2QWERTY_CACHE="$PWD/.cache/b2q_v1mamba"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p slurm/logs

# 1. smoke-test the data path end to end (1 recording, fast)
python -m brain2qwerty_v1_mamba.main cache --debug

# 2. warm the full 3-subject cache (timeline query skips everything else;
#    NOTE: the timeline index uses long-form ids "Pinet2024Meg/S15")
python -m brain2qwerty_v1_mamba.main cache \
    --subjects S15 S16 S6 \
    --timeline-query "subject in ['Pinet2024Meg/S15','Pinet2024Meg/S16','Pinet2024Meg/S6']"

# 3. pack the cache for upload as a Kaggle Dataset
tar -czf b2q_v1mamba_cache.tar.gz -C "$(dirname "$BRAIN2QWERTY_CACHE")" "$(basename "$BRAIN2QWERTY_CACHE")"
ls -lh b2q_v1mamba_cache.tar.gz
echo "[cache] done -> upload b2q_v1mamba_cache.tar.gz to Kaggle"
