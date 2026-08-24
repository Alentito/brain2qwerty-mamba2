#!/bin/bash
#SBATCH --job-name=b2q_download
#SBATCH --partition=k2-hipri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/download_%j.out

# Downloads the SpanishBCBL dataset (~262 GB!) via the neuralset helper,
# then builds the standardized event dataframe once.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export BRAIN2QWERTY_STUDIES="$PWD/SpanishBCBL"
export BRAIN2QWERTY_CACHE="$PWD/.cache/brain2qwerty"
export BRAIN2QWERTY_RESULTS="$BRAIN2QWERTY_CACHE/results"
mkdir -p slurm/logs "$BRAIN2QWERTY_CACHE"

df -h .   # sanity: you need > 300 GB free

python - <<'PY'
import studies  # registers Pinet2024Meg / Pinet2024Eeg
from neuralset.events import Study

study = Study(name="Pinet2024Meg", path="SpanishBCBL")
study.download()   # fetch recordings + behavioral logs from HF Hub
events = study.build()  # standardized events (cached for all later jobs)
print(events.groupby("type").size())
PY

echo "[download] done."
