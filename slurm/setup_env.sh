#!/usr/bin/env bash
# One-time environment setup — run on a LOGIN node:
#   bash slurm/setup_env.sh
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root (Brain2qwerty/)

# Load a Python >= 3.12 and a CUDA toolchain (adjust to your cluster's modules).
# On Kelvin-2 (QUB) e.g.:  module load python/3.12  cuda/12.4
module purge || true
module load python/3.12 || true
module load cuda/12.4  || true

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Exact pinned dependency closure (Linux x86-64 + CUDA 12.4 wheels)
pip install -r requirements.lock
# KenLM is only needed for the optional N-gram LM rescoring step
pip install "kenlm==0.2.0" || echo "[warn] kenlm build failed; N-gram decoding will be skipped"

# Install the repo itself (brain2qwerty_v1, studies, ...)
pip install -e . --no-deps

echo "[setup] done. Python: $(python -V), torch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
