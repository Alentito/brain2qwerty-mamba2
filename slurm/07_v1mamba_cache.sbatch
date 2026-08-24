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
source .venv/bin/activate

export BRAIN2QWERTY_STUDIES="$PWD/SpanishBCBL_3subj"
export BRAIN2QWERTY_CACHE="$PWD/.cache/b2q_v1mamba"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p slurm/logs

# 1. smoke-test the data path end to end (1 recording, fast)
python -m brain2qwerty_v1_mamba.main cache --debug

# 2. warm the full 3-subject cache (timeline query skips everything else)
python -m brain2qwerty_v1_mamba.main cache \
    --subjects S15 S16 S6 \
    --timeline-query "subject in ['S15','S16','S6']"

# 3. pack the cache for upload as a Kaggle Dataset
tar -czf b2q_v1mamba_cache.tar.gz -C "$PWD/.cache" b2q_v1mamba
ls -lh b2q_v1mamba_cache.tar.gz
echo "[cache] done -> upload b2q_v1mamba_cache.tar.gz to Kaggle"
