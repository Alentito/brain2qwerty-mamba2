#!/usr/bin/env bash
#SBATCH --job-name=b2q_r3
#SBATCH --output=slurm_logs/r3_%A_%a.out
#SBATCH --error=slurm_logs/r3_%A_%a.err
#SBATCH --partition=k2-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-7%4

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

mkdir -p slurm_logs
module load cuDNN/8.9.2.26-CUDA-12.1.1 2>/dev/null || true
module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || true

# Activate cluster environment
if [ -f "$HOME/b2q_env/bin/activate" ]; then
    source "$HOME/b2q_env/bin/activate"
elif [ -f "$HOME/.venv/bin/activate" ]; then
    source "$HOME/.venv/bin/activate"
fi

export BRAIN2QWERTY_CACHE="${BRAIN2QWERTY_CACHE:-$HOME/sharedscratch/B2Q/cache_v1mamba}"
export BRAIN2QWERTY_STUDIES="${BRAIN2QWERTY_STUDIES:-$HOME/sharedscratch/B2Q/code/SpanishBCBL_3subj}"
export PYTHONUNBUFFERED=1

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
echo "=== Starting Round 3 Experiment Task: $TASK_ID on $(hostname) at $(date) ==="

case "$TASK_ID" in
    0)
        echo ">>> Task 0: Hybrid Mamba-2 + Global Attention [M, M, M, A] (lr=3e-4)"
        python -m brain2qwerty_v1_mamba.main train \
            --core hybrid \
            --small \
            --lr 3e-4 \
            --wd 0.1 \
            --grad-clip 1.0 \
            --tag hybrid-lr3e4-wd01-gc1
        ;;
    1)
        echo ">>> Task 1: Hybrid Mamba-3 + Global Attention [M3, M3, M3, A] (lr=3e-4)"
        python -m brain2qwerty_v1_mamba.main train \
            --core hybrid3 \
            --small \
            --lr 3e-4 \
            --wd 0.1 \
            --grad-clip 1.0 \
            --tag hybrid3-lr3e4-wd01-gc1
        ;;
    2)
        echo ">>> Task 2: BiMamba-2 + Gated Fusion + FFN MLP (lr=3e-4)"
        python -m brain2qwerty_v1_mamba.main train \
            --core mamba_mlp \
            --small \
            --lr 3e-4 \
            --wd 0.1 \
            --grad-clip 1.0 \
            --tag mamba-mlp-gated-lr3e4-wd01-gc1
        ;;
    3)
        echo ">>> Task 3: BiMamba-3 + Gated Fusion + FFN MLP (lr=3e-4)"
        python -m brain2qwerty_v1_mamba.main train \
            --core mamba3_mlp \
            --small \
            --lr 3e-4 \
            --wd 0.1 \
            --grad-clip 1.0 \
            --tag mamba3-mlp-gated-lr3e4-wd01-gc1
        ;;
    4)
        echo ">>> Task 4: 8-Layer Deep Hybrid [M, M, M, A, M, M, M, A] (lr=3e-4)"
        python -m brain2qwerty_v1_mamba.main train \
            --core hybrid_8l \
            --small \
            --lr 3e-4 \
            --wd 0.1 \
            --grad-clip 1.0 \
            --tag hybrid-8l-lr3e4-wd01-gc1
        ;;
    5)
        echo ">>> Task 5: 8-Layer Deep Mamba-3 Hybrid (lr=3e-4)"
        python -m brain2qwerty_v1_mamba.main train \
            --core hybrid3_8l \
            --small \
            --lr 3e-4 \
            --wd 0.1 \
            --grad-clip 1.0 \
            --tag hybrid3-8l-lr3e4-wd01-gc1
        ;;
    6)
        echo ">>> Task 6: 8-Layer Deep Transformer Baseline (lr=1e-4)"
        python -m brain2qwerty_v1_mamba.main train \
            --core transformer_deep \
            --small \
            --lr 1e-4 \
            --wd 0.1 \
            --grad-clip 1.0 \
            --tag transformer-deep8l-lr1e4-wd01-gc1
        ;;
    7)
        echo ">>> Task 7: Brain2Qwerty V3 Continuous Unsegmented Sentence CTC Decoder"
        python -m brain2qwerty_v3.main train \
            --devices 1 \
            --lr 3e-4 \
            --wd 0.1 \
            --tag v3-continuous-hybrid-ctc
        ;;
    *)
        echo "Unknown task id $TASK_ID"
        exit 1
        ;;
esac

echo "=== Finished Task: $TASK_ID at $(date) ==="
