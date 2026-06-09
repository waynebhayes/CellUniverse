#!/usr/bin/env zsh
set -euo pipefail

unsetopt bg_nice 2>/dev/null || true

export CELLUNIVERSE_THREADS=8

CPP_ROOT="/Users/wangyiding/CellUniverse/C++"
INPUT_DIR="/Volumes/T9/🦠Cell Universe/💿Data/In_Use/A549 Lung Cancer cells embedded in a Matrigel matrix 263M/01"
OUT_DIR="/Volumes/T9/🦠Cell Universe/🟣Output/A549_LungCancer_Run1"
CONFIG="$CPP_ROOT/config/config_A549_lung_cancer_traditional_celluniverse_run1_20260604.yaml"
INITIAL="$CPP_ROOT/config/initial_A549_000_manual_shapes_scaledZ_for_CellUniverse.csv"
LOG="$OUT_DIR/run_A549_traditional_celluniverse_run1_20260604.log"

mkdir -p "$OUT_DIR"

"$CPP_ROOT/build/celluniverse" \
  0 \
  29 \
  "$INPUT_DIR/t%03d.tif" \
  "$OUT_DIR" \
  "$CONFIG" \
  "$INITIAL" \
  2 \
  "$OUT_DIR" 2>&1 | tee -a "$LOG"
