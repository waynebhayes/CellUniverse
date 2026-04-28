#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_BASE_DIR="/Users/wangyiding/CellUniverse/C++/output/🟣HL60_20260419.0638"
BASE_DIR="${1:-$DEFAULT_BASE_DIR}"
FPS="${2:-2}"
LAYOUT="${3:-grid}"

echo "[INFO] Launching napari movie viewer"
echo "[INFO] Base dir : $BASE_DIR"
echo "[INFO] FPS      : $FPS"
echo "[INFO] Layout   : $LAYOUT"

python3 "$SCRIPT_DIR/play_real_and_synth_same_napari.py" \
  "$BASE_DIR" \
  --fps "$FPS" \
  --layout "$LAYOUT"
