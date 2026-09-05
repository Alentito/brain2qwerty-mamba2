#!/bin/bash
#SBATCH --job-name=b2q_s4v3
#SBATCH --partition=k2-gpu-a100        # fallback: k2-gpu-h100 or k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --array=0-5%3                  # 6 arms (2 frontends x 3 cores), max 3 concurrent
#SBATCH --output=slurm/logs/s4v3_%A_%a.out

# ============================================================================ #
# Study 4 (V3 continuous pipeline): Frontend x Core grid on subject S16
#
#   frontends: conv (SimpleConv + per-subject Fourier merger) | gnn
#              (GnnContinuousEncoder, per-frame k-NN scalp graph)
#   cores:     conformer (V2 baseline) | mamba3_hybrid_stabilized (Study-3
#              champion) | deltanet (delta-rule linear attention)
#
# Staged schedule (275 epochs): CTC from 0, +contrastive at 150, +LoRA LLM
# at 225. Output: $BRAIN2QWERTY_CACHE/results/v3-[gnn-]<core>-study4/
# Idempotent: skips arms whose training_profile.json exists (written at
# test end). Collect afterwards with:
#   python run_two_studies.py --pipeline v3 --study 1 --study1-subjects S16 \
#       --encoders conv gnn --cores conformer mamba3_hybrid_stabilized deltanet \
#       --collect-only
# ============================================================================ #

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

# Env: local .venv if present (Mac), otherwise the Kelvin-2 conda env.
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; else source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate "$HOME/sharedscratch/conda/envs/b2q"; fi

export BRAIN2QWERTY_STUDIES="${BRAIN2QWERTY_STUDIES:-$HOME/sharedscratch/B2Q/code/SpanishBCBL_3subj}"
[ -d "$BRAIN2QWERTY_STUDIES" ] || export BRAIN2QWERTY_STUDIES="$PWD/SpanishBCBL_3subj"

export BRAIN2QWERTY_CACHE="${BRAIN2QWERTY_CACHE:-$HOME/sharedscratch/B2Q/cache_v1mamba}"
[ -d "$(dirname "$BRAIN2QWERTY_CACHE")" ] || export BRAIN2QWERTY_CACHE="$PWD/.cache/b2q_v1mamba"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
echo "=== Study4-V3 Task $TASK_ID on $(hostname) at $(date) ==="

CORES=(conformer mamba3_hybrid_stabilized deltanet)
ENCS=(conv gnn)
CORE=${CORES[$((TASK_ID % 3))]}
ENC=${ENCS[$((TASK_ID / 3))]}

INFIX=""
[ "$ENC" = "gnn" ] && INFIX="gnn-"
OUT="$BRAIN2QWERTY_CACHE/results/v3-$INFIX$CORE-study4"

if [ -f "$OUT/training_profile.json" ]; then
    echo ">>> [skip] $ENC/$CORE already finished ($OUT)"
    exit 0
fi

echo ">>> Running V3 Study4 arm: frontend=$ENC core=$CORE subject=S16"
python -m brain2qwerty_v3.main train \
    --subjects S16 \
    --core "$CORE" \
    --frontend "$ENC" \
    --wd 1e-3 \
    --tag study4 \
    --devices 1

echo "=== Finished Study4-V3 Task $TASK_ID at $(date) ==="
