#!/bin/bash
#SBATCH --job-name=b2q_s4v1
#SBATCH --partition=k2-gpu-a100        # fallback: k2-gpu-h100 or k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --array=0-5%3                  # 6 arms (2 encoders x 3 cores), max 3 concurrent
#SBATCH --output=slurm/logs/s4v1_%A_%a.out

# ============================================================================ #
# Study 4 (V1 keystroke pipeline): Encoder x Decoder grid on subject S16
#
#   encoders: conv (V1 SimpleConvTimeAgg) | gnn (k-NN scalp-graph encoder)
#   cores:    transformer | mamba (Mamba-2) | deltanet
#
# Output: $BRAIN2QWERTY_CACHE/results/small-[gnn-]<core>-S16-study1/
# Idempotent: skips arms whose test_all_sentences.json exists; resumes
# interrupted arms from last.ckpt. Collect afterwards with:
#   python run_two_studies.py --pipeline v1 --study 1 --study1-subjects S16 \
#       --encoders conv gnn --cores transformer mamba deltanet --collect-only
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
echo "=== Study4-V1 Task $TASK_ID on $(hostname) at $(date) ==="

CORES=(transformer mamba deltanet)
ENCS=(conv gnn)
CORE=${CORES[$((TASK_ID % 3))]}
ENC=${ENCS[$((TASK_ID / 3))]}

PREFIX=""
[ "$ENC" = "gnn" ] && PREFIX="gnn-"
OUT="$BRAIN2QWERTY_CACHE/results/small-$PREFIX$CORE-S16-study1"

if [ -f "$OUT/callbacks/test_all_sentences.json" ]; then
    echo ">>> [skip] $ENC/$CORE already finished ($OUT)"
    exit 0
fi

RESUME=""
if [ -f "$OUT/last.ckpt" ]; then
    echo ">>> Resuming $ENC/$CORE from $OUT/last.ckpt"
    RESUME="--resume $OUT/last.ckpt"
fi

echo ">>> Running V1 Study4 arm: encoder=$ENC core=$CORE subject=S16"
python -m brain2qwerty_v1_mamba.main train \
    --subjects S16 \
    --core "$CORE" \
    --encoder "$ENC" \
    --small \
    --tag study1 \
    $RESUME

echo "=== Finished Study4-V1 Task $TASK_ID at $(date) ==="
