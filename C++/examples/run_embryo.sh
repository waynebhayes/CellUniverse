#!/bin/bash
set -euo pipefail

# Directory that contains this script: .../C++/examples
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# C++ repo root: .../C++ (parent of examples)
CPP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# -----------------------------------------
# Create run log file (stdout + stderr)
# -----------------------------------------
LOG_FILE="$SCRIPT_DIR/embryo_runLog_$(date +%Y%m%d_%H%M%S).txt"

# Redirect all output (stdout + stderr) to both console and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[INFO] Logging to: $LOG_FILE"

# =========================================
# CellUniverse Embryo Run Script (clean + rebuild + run)
# Location: C++/examples/run_embryo.sh
# Uses: initial_embryo.csv (same folder as this script)
# Output: output_embryo_<timestamp>
# =========================================
# Data/config paths
INPUT_DIR="$CPP_ROOT/examples/input/C.elegans_developing embryo_Fluo-N3DH-CE_Training/01"
CONFIG_FILE="$CPP_ROOT/examples/config.yaml"
INITIAL_FILE="$SCRIPT_DIR/initial_embryo.csv"
OUT_DIR="$CPP_ROOT/examples/output_embryo_$(date +%Y%m%d_%H%M%S)"

# Choose build directory preference
BUILD_DIR="$CPP_ROOT/build"
FALLBACK_BUILD_DIR="$CPP_ROOT/cmake-build-debug"

echo "*******************************************************************************************************"
echo "Cell Universe Embryo Run (clean + rebuild + run)"
echo "*******************************************************************************************************"
echo "CPP Root    : $CPP_ROOT"
echo "Input Dir   : $INPUT_DIR"
echo "Initial CSV : $INITIAL_FILE"
echo "Config File : $CONFIG_FILE"
echo "Output Dir  : $OUT_DIR"
echo "========================================="

# Hard checks (fail fast)
[ -d "$CPP_ROOT" ] || { echo "[FATAL] CPP root not found: $CPP_ROOT"; exit 1; }
[ -d "$INPUT_DIR" ] || { echo "[FATAL] input dir not found: $INPUT_DIR"; exit 1; }
[ -f "$CONFIG_FILE" ] || { echo "[FATAL] config not found: $CONFIG_FILE"; exit 1; }
[ -f "$INITIAL_FILE" ] || { echo "[FATAL] initial csv not found: $INITIAL_FILE"; exit 1; }
[ -f "$CPP_ROOT/CMakeLists.txt" ] || { echo "[FATAL] CMakeLists.txt not found in: $CPP_ROOT"; exit 1; }

# -----------------------------------------
# Clean previous build artifacts
# -----------------------------------------
echo "[STEP] Cleaning previous build artifacts..."

# Always prefer the standard build/ for reproducible rebuilds
if [ -d "$BUILD_DIR" ]; then
  echo "  - Removing: $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi

# Optional: also clean CLion build dir if exists (prevents confusion)
if [ -d "$FALLBACK_BUILD_DIR" ]; then
  echo "  - Removing: $FALLBACK_BUILD_DIR"
  rm -rf "$FALLBACK_BUILD_DIR"
fi

# Recreate build directory
mkdir -p "$BUILD_DIR"

# -----------------------------------------
# Reconfigure + Rebuild
# -----------------------------------------
echo "[STEP] Reconfiguring CMake..."
cmake -S "$CPP_ROOT" -B "$BUILD_DIR"

echo "[STEP] Building..."
cmake --build "$BUILD_DIR" -- -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

# Validate binary
BIN="$BUILD_DIR/celluniverse"
[ -x "$BIN" ] || { echo "[FATAL] build succeeded but binary not found/executable: $BIN"; exit 1; }

# -----------------------------------------
# Detect frame range from t*.tif
# -----------------------------------------
echo "[STEP] Detecting frame range in input directory..."
shopt -s nullglob
FILES=("$INPUT_DIR"/t*.tif)
shopt -u nullglob
[ ${#FILES[@]} -gt 0 ] || { echo "[FATAL] no files like t*.tif in: $INPUT_DIR"; exit 1; }

MIN_FRAME=999999
MAX_FRAME=-1
for f in "${FILES[@]}"; do
  base="$(basename "$f")"     # e.g., t016.tif
  num="${base#t}"             # 016.tif
  num="${num%.tif}"           # 016
  n=$((10#$num))
  if [ "$n" -lt "$MIN_FRAME" ]; then MIN_FRAME="$n"; fi
  if [ "$n" -gt "$MAX_FRAME" ]; then MAX_FRAME="$n"; fi
done

echo "[INFO] Detected frame range: $MIN_FRAME .. $MAX_FRAME"
# -----------------------------------------
# [PATCH] Quick test: force small frame range
# Run only 5 frames starting from the first frame (t001.tif)
# -----------------------------------------
MIN_FRAME=1
MAX_FRAME=5
echo "[INFO] Forced frame range: $MIN_FRAME .. $MAX_FRAME"

# -----------------------------------------
# Run
# -----------------------------------------
echo "[STEP] Running tracker..."
mkdir -p "$OUT_DIR"

"$BIN" \
  "$MIN_FRAME" \
  "$MAX_FRAME" \
  "$INPUT_DIR/t%03d.tif" \
  "$OUT_DIR" \
  "$CONFIG_FILE" \
  "$INITIAL_FILE" 2> >(grep -v "TIFF_Warning TIFFReadDirectory: Unknown field with tag 6500" >&2)

echo "========================================="
echo "Run finished (exit=0)."
echo "Results saved to:"
echo "$OUT_DIR"
echo "========================================="