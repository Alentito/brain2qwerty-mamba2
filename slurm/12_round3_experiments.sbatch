#!/bin/bash
#SBATCH --job-name=b2q_r3
#SBATCH --partition=k2-gpu-a100        # fallback: k2-gpu-h100 or k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --array=0-7%3                  # 8 arms, max 3 concurrent
#SBATCH --output=slurm/logs/r3_%A_%a.out

# ============================================================================ #
# Brain2Qwerty Round 3: Advanced Architectures & Continuous Decoding Suite
#
# Array Tasks:
#   0: hybrid           -> Nemotron-H style [M, M, M, A] (Mamba-2 + Attention, lr=3e-4)
#   1: hybrid3          -> Hybrid Mamba-3 (BCNorm + RoPE + Attention [M3, M3, M3, A], lr=3e-4)
#   2: mamba_mlp        -> BiMamba-2 + Learned Gated Fusion + FFN MLP (lr=3e-4)
#   3: mamba3_mlp       -> BiMamba-3 + Learned Gated Fusion + FFN MLP (lr=3e-4)
#   4: hybrid_8l        -> 8-Layer Deep Hybrid [M, M, M, A, M, M, M, A] (lr=3e-4)
#   5: hybrid3_8l       -> 8-Layer Deep Mamba-3 Hybrid (lr=3e-4)
#   6: transformer_deep -> 8-Layer Deep Transformer Baseline (lr=1e-4)
#   7: v3_continuous    -> Brain2Qwerty V3 Continuous Unsegmented CTC Decoder
# ============================================================================ #

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

# Environment activation
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
elif [ -f "$HOME/sharedscratch/conda/envs/b2q/bin/activate" ]; then
    source "$HOME/sharedscratch/conda/envs/b2q/bin/activate"
else
    source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "$HOME/sharedscratch/conda/envs/b2q" 2>/dev/null || true
fi

export BRAIN2QWERTY_STUDIES="${BRAIN2QWERTY_STUDIES:-$HOME/sharedscratch/B2Q/code/SpanishBCBL_3subj}"
[ -d "$BRAIN2QWERTY_STUDIES" ] || export BRAIN2QWERTY_STUDIES="$PWD/SpanishBCBL_3subj"

export BRAIN2QWERTY_CACHE="${BRAIN2QWERTY_CACHE:-$HOME/sharedscratch/B2Q/cache_v1mamba}"
[ -d "$(dirname "$BRAIN2QWERTY_CACHE")" ] || export BRAIN2QWERTY_CACHE="$PWD/.cache/b2q_v1mamba"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
echo "=== Starting Round 3 Experiment Task: $TASK_ID on $(hostname) at $(date) ==="

case $TASK_ID in
  0)
    CORE=hybrid
    TAG=hybrid-lr3e4-wd01-gc1
    OPTS="--lr 3e-4 --wd 0.1 --grad-clip 1.0"
    ;;
  1)
    CORE=hybrid3
    TAG=hybrid3-lr3e4-wd01-gc1
    OPTS="--lr 3e-4 --wd 0.1 --grad-clip 1.0"
    ;;
  2)
    CORE=mamba_mlp
    TAG=mamba-mlp-gated-lr3e4-wd01-gc1
    OPTS="--lr 3e-4 --wd 0.1 --grad-clip 1.0"
    ;;
  3)
    CORE=mamba3_mlp
    TAG=mamba3-mlp-gated-lr3e4-wd01-gc1
    OPTS="--lr 3e-4 --wd 0.1 --grad-clip 1.0"
    ;;
  4)
    CORE=hybrid_8l
    TAG=hybrid-8l-lr3e4-wd01-gc1
    OPTS="--lr 3e-4 --wd 0.1 --grad-clip 1.0"
    ;;
  5)
    CORE=hybrid3_8l
    TAG=hybrid3-8l-lr3e4-wd01-gc1
    OPTS="--lr 3e-4 --wd 0.1 --grad-clip 1.0"
    ;;
  6)
    CORE=transformer_deep
    TAG=transformer-deep8l-lr1e4-wd01-gc1
    OPTS="--lr 1e-4 --wd 0.1 --grad-clip 1.0"
    ;;
  7)
    echo ">>> Task 7: Brain2Qwerty V3 Continuous Unsegmented Sentence CTC Decoder"
    python -m brain2qwerty_v3.main train \
        --devices 1 \
        --lr 3e-4 \
        --wd 0.1 \
        --tag v3-continuous-hybrid-ctc
    echo "=== Finished Task 7 at $(date) ==="
    exit 0
    ;;
  *)
    echo "Unknown task id $TASK_ID"
    exit 1
    ;;
esac

OUT="$BRAIN2QWERTY_CACHE/results/small-$CORE-S15-S16-S6-$TAG"

# Idempotency: skip if already finished
if [ -f "$OUT/callbacks/test_all_sentences.json" ]; then
    echo "[r3] task=$TASK_ID core=$CORE already finished -> skip"
    exit 0
fi

RESUME=""
if [ -f "$OUT/last.ckpt" ]; then
    RESUME="--resume $OUT/last.ckpt"
    echo "[r3] resuming from $OUT/last.ckpt"
fi

echo "[r3] task=$TASK_ID core=$CORE opts=$OPTS $RESUME"
python -m brain2qwerty_v1_mamba.main colab --core "$CORE" --seed 33 \
    --tag "$TAG" $OPTS $RESUME

echo "=== Finished Task $TASK_ID at $(date) ==="
