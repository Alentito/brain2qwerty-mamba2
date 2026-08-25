#!/bin/bash
#SBATCH --job-name=b2q_exp
#SBATCH --partition=k2-gpu-a100        # fallback: k2-gpu-h100 or k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-5                    # 6 arms, see the case block below
#SBATCH --output=slurm/logs/exp_%A_%a.out

# Round-2 experiment grid (single pass):
#   0: mamba  lr=1e-4                       (optimization fix, mild)
#   1: mamba  lr=3e-4                       (optimization fix, aggressive)
#   2: mamba  lr=1e-4 wd=0.1 clip=1.0       (full Mamba best-practice recipe)
#   3: mamba3 lr=1e-4 wd=0.1 clip=1.0       (Mamba-3-style block, mild lr)
#   4: mamba3 lr=3e-4 wd=0.1 clip=1.0       (Mamba-3-style block, aggressive lr)
#   5: transformer lr=1e-4                  (control: is V1's 5e-5 also suboptimal?)
#
# All arms: small 512-dim, subjects S15/S16/S6, colab preset (200 epochs,
# patience 25), seed 33. Outputs are namespaced by --tag under
# $BRAIN2QWERTY_CACHE/results/ so they never clash with round 1.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Env: local .venv if present (Mac), otherwise the Kelvin-2 conda env.
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; else source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate "$HOME/sharedscratch/conda/envs/b2q"; fi

# The study path must EXIST even though all features come from the cache
# (the study object validates it at construction).
export BRAIN2QWERTY_STUDIES="${BRAIN2QWERTY_STUDIES:-$HOME/sharedscratch/B2Q/code/SpanishBCBL_3subj}"
[ -d "$BRAIN2QWERTY_STUDIES" ] || export BRAIN2QWERTY_STUDIES="$PWD/SpanishBCBL_3subj"
export BRAIN2QWERTY_CACHE="${BRAIN2QWERTY_CACHE:-$HOME/sharedscratch/B2Q/cache_v1mamba}"
[ -d "$(dirname "$BRAIN2QWERTY_CACHE")" ] || export BRAIN2QWERTY_CACHE="$PWD/.cache/b2q_v1mamba"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

case $SLURM_ARRAY_TASK_ID in
  0) CORE=mamba;       EXTRA="--lr 1e-4 --tag lr1e4" ;;
  1) CORE=mamba;       EXTRA="--lr 3e-4 --tag lr3e4" ;;
  2) CORE=mamba;       EXTRA="--lr 1e-4 --wd 0.1 --grad-clip 1.0 --tag lr1e4-wd01-gc1" ;;
  3) CORE=mamba3;      EXTRA="--lr 1e-4 --wd 0.1 --grad-clip 1.0 --tag v3lr1e4-wd01-gc1" ;;
  4) CORE=mamba3;      EXTRA="--lr 3e-4 --wd 0.1 --grad-clip 1.0 --tag v3lr3e4-wd01-gc1" ;;
  5) CORE=transformer; EXTRA="--lr 1e-4 --tag lr1e4" ;;
  *) echo "unknown task $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "[exp] task=$SLURM_ARRAY_TASK_ID core=$CORE extra=$EXTRA"
python -m brain2qwerty_v1_mamba.main colab --core "$CORE" --seed 33 $EXTRA
echo "[exp] task=$SLURM_ARRAY_TASK_ID core=$CORE done."
