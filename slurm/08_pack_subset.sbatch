#!/bin/bash
#SBATCH --job-name=b2q_pack_subset
#SBATCH --partition=k2-hipri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=slurm/logs/pack_subset_%j.out

# Copies ONLY subjects S15, S16, S6 (recordings + logs) out of the full
# SpanishBCBL download into a self-contained subset folder, verifies the
# study loader sees exactly those subjects, and packs it for upload.
#
# Adjust SRC if your full download lives elsewhere.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

SRC="${BRAIN2QWERTY_STUDIES:-$PWD/SpanishBCBL}"     # full download
DST="$PWD/SpanishBCBL_3subj"                         # new subset folder
mkdir -p slurm/logs "$DST/MEG/FIF" "$DST/MEG/logs"

# --- recordings (whole subject dirs: all sessions/blocks) -------------------
for d in "$SRC"/MEG/FIF/15_* "$SRC"/MEG/FIF/16_* "$SRC"/MEG/FIF/06_*; do
    [ -d "$d" ] || { echo "MISSING: $d"; exit 1; }
    cp -r "$d" "$DST/MEG/FIF/"
done

# --- logs (only our subjects' .mat files, keeping the subfolder layout) -----
cd "$SRC/MEG/logs"
find . -type f \( -name "S15-session*.mat" -o -name "S16-session*.mat" -o -name "S6-session*.mat" \) \
    -exec cp --parents {} "$DST/MEG/logs/" \;
cd "$SLURM_SUBMIT_DIR"

# --- any small root-level metadata ------------------------------------------
[ -f "$SRC/README.md" ] && cp "$SRC/README.md" "$DST/"
find "$SRC/MEG" -maxdepth 1 -type f -exec cp {} "$DST/MEG/" \; 2>/dev/null || true

echo "[subset] size:"; du -sh "$DST"
echo "[subset] subject dirs:"; ls "$DST/MEG/FIF"
echo "[subset] log files:"; find "$DST/MEG/logs" -name "*.mat" | wc -l

# --- verify: the study loader must find exactly S6, S15, S16 timelines ------
BRAIN2QWERTY_STUDIES="$DST" python - <<'PY'
import os
import studies  # registers Pinet2024Meg
from neuralset.events import Study

study = Study(name="Pinet2024Meg", path=os.environ["BRAIN2QWERTY_STUDIES"])
recs = study._get_all_files()
print(recs[["subject", "session", "task"]].to_string())
found = sorted(recs.subject.unique())
assert found == ["S15", "S16", "S6"], f"unexpected subjects: {found}"
print("[verify] OK: recordings found for", found)
PY

# --- pack for upload ---------------------------------------------------------
tar -czf SpanishBCBL_3subj.tar.gz -C "$PWD" SpanishBCBL_3subj
ls -lh SpanishBCBL_3subj.tar.gz
echo "[pack] done -> SpanishBCBL_3subj.tar.gz"
