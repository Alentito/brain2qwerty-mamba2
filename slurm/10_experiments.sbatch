#!/bin/bash
#SBATCH --job-name=b2q_exp
#SBATCH --partition=k2-gpu-a100        # fallback: k2-gpu-h100 or k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --array=0-5%3                  # 6 arms, max 3 concurrent (see case block)
#SBATCH --output=slurm/logs/exp_%A_%a.out

# Round-2 experiment grid (resumable single pass):
#   0: mamba  lr=1e-4                       (optimization fix, mild)
#   1: mamba  lr=3e-4                       (optimization fix, aggressive)
#   2: mamba  lr=1e-4 wd=0.1 clip=1.0       (full Mamba best-practice recipe)
#   3: mamba3 lr=1e-4 wd=0.1 clip=1.0       (Mamba-3-style block, mild lr)
#   4: mamba3 lr=3e-4 wd=0.1 clip=1.0       (Mamba-3-style block, aggressive lr)
#   5: transformer lr=1e-4                  (control: is V1's 5e-5 also suboptimal?)
#
# All arms: small 512-dim, subjects S15/S16/S6, colab preset (200 epochs,
# patience 25), seed 33.
#
# RESUMABLE: if an arm's wall time kills the job, just re-run `sbatch` on this
# same file — finished arms skip themselves, interrupted arms resume from
# last.ckpt with optimizer/scheduler/epoch state intact.
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
  0) CORE=mamba;       TAG=lr1e4;              OPTS="--lr 1e-4" ;;
  1) CORE=mamba;       TAG=lr3e4;              OPTS="--lr 3e-4" ;;
  2) CORE=mamba;       TAG=lr1e4-wd01-gc1;     OPTS="--lr 1e-4 --wd 0.1 --grad-clip 1.0" ;;
  3) CORE=mamba3;      TAG=v3lr1e4-wd01-gc1;   OPTS="--lr 1e-4 --wd 0.1 --grad-clip 1.0" ;;
  4) CORE=mamba3;      TAG=v3lr3e4-wd01-gc1;   OPTS="--lr 3e-4 --wd 0.1 --grad-clip 1.0" ;;
  5) CORE=transformer; TAG=lr1e4;              OPTS="--lr 1e-4" ;;
  *) echo "unknown task $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

OUT="$BRAIN2QWERTY_CACHE/results/small-$CORE-S15-S16-S6-$TAG"

# Idempotency: finished arm -> skip; interrupted arm -> resume from last.ckpt
if [ -f "$OUT/callbacks/test_all_sentences.json" ]; then
  echo "[exp] task=$SLURM_ARRAY_TASK_ID core=$CORE already finished -> skip"
  exit 0
fi
RESUME=""
if [ -f "$OUT/last.ckpt" ]; then
  RESUME="--resume $OUT/last.ckpt"
  echo "[exp] resuming from $OUT/last.ckpt"
fi

echo "[exp] task=$SLURM_ARRAY_TASK_ID core=$CORE opts=$OPTS $RESUME"
python -m brain2qwerty_v1_mamba.main colab --core "$CORE" --seed 33 \
  --tag "$TAG" $OPTS $RESUME
echo "[exp] task=$SLURM_ARRAY_TASK_ID core=$CORE done."
