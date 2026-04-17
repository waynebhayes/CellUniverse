#!/bin/bash
set -euo pipefail

# Directory that contains this script: .../C++/examples
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# C++ repo root: .../C++ (parent of examples)
CPP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_ROOT="$CPP_ROOT/output"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

# -----------------------------------------
# Create run log file (stdout + stderr)
# -----------------------------------------
LOG_FILE="$LOG_DIR/original_data_runLog_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[INFO] Logging to: $LOG_FILE"

# -----------------------------------------
# Paths
# -----------------------------------------
INPUT_DIR="$CPP_ROOT/examples/input/original_data"
INPUT_PATTERN="$INPUT_DIR/frame%03d.tif"   # NOTE: original_data uses frame%03d.tif
CONFIG_FILE="$CPP_ROOT/config/config.yaml"
INITIAL_FILE="$CPP_ROOT/config/initial.csv"     # change if your initial file name differs
OUT_DIR="$OUTPUT_ROOT/output_original_data_$(date +%Y%m%d_%H%M%S)"

BUILD_DIR="$CPP_ROOT/build"
FALLBACK_BUILD_DIR="$CPP_ROOT/cmake-build-debug"
SKIP_CLEAN="${CELLUNIVERSE_SKIP_CLEAN:-0}"
BUILD_TYPE="${CELLUNIVERSE_BUILD_TYPE:-Release}"

echo "======================================================================================================="
echo "Cell Universe Original Data Run (clean + rebuild + run)"
echo "======================================================================================================="
echo "CPP Root        : $CPP_ROOT"
echo "Input Dir       : $INPUT_DIR"
echo "Input Pattern   : $INPUT_PATTERN"
echo "Initial CSV     : $INITIAL_FILE"
echo "Config YAML     : $CONFIG_FILE"
echo "Output Dir      : $OUT_DIR"
echo "Skip Clean      : $SKIP_CLEAN"
echo "Build Type      : $BUILD_TYPE"
echo "======================================================================================================="

# -----------------------------------------
# Hard checks
# -----------------------------------------
[ -d "$CPP_ROOT" ] || { echo "[FATAL] CPP root not found: $CPP_ROOT"; exit 1; }
[ -d "$INPUT_DIR" ] || { echo "[FATAL] input dir not found: $INPUT_DIR"; exit 1; }
[ -f "$CONFIG_FILE" ] || { echo "[FATAL] config not found: $CONFIG_FILE"; exit 1; }
[ -f "$INITIAL_FILE" ] || { echo "[FATAL] initial csv not found: $INITIAL_FILE"; exit 1; }
[ -f "$CPP_ROOT/CMakeLists.txt" ] || { echo "[FATAL] CMakeLists.txt not found in: $CPP_ROOT"; exit 1; }

# -----------------------------------------
# Clean previous build artifacts
# -----------------------------------------
echo "[STEP] Cleaning previous build artifacts..."
if [ "$SKIP_CLEAN" = "1" ]; then
  echo "  - Skipping clean build removal (CELLUNIVERSE_SKIP_CLEAN=1)"
  mkdir -p "$BUILD_DIR"
else
  if [ -d "$BUILD_DIR" ]; then
    echo "  - Removing: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
  fi
  if [ -d "$FALLBACK_BUILD_DIR" ]; then
    echo "  - Removing: $FALLBACK_BUILD_DIR"
    rm -rf "$FALLBACK_BUILD_DIR"
  fi
  mkdir -p "$BUILD_DIR"
fi

# -----------------------------------------
# Reconfigure + Rebuild
# -----------------------------------------
echo "[STEP] Reconfiguring CMake..."
cmake -S "$CPP_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

echo "[STEP] Building..."
cmake --build "$BUILD_DIR" -- -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

BIN="$BUILD_DIR/celluniverse"
[ -x "$BIN" ] || { echo "[FATAL] build succeeded but binary not found/executable: $BIN"; exit 1; }

# -----------------------------------------
# Frame range: force skip frame000-bogus.tif by setting FIRST>=1
# Default: 1..5
# Usage:
#   ./run_original_data.sh        -> 1..5
#   ./run_original_data.sh 10     -> 1..10
#   ./run_original_data.sh 3 8    -> 3..8
# -----------------------------------------
FIRST_FRAME=1
LAST_FRAME=19

if [ "${1:-}" != "" ] && [ "${2:-}" != "" ]; then
  FIRST_FRAME="$1"
  LAST_FRAME="$2"
elif [ "${1:-}" != "" ]; then
  COUNT="$1"
  FIRST_FRAME=1
  LAST_FRAME=$((FIRST_FRAME + COUNT - 1))
fi

# Safety: never allow FIRST_FRAME < 1 for this dataset
if [ "$FIRST_FRAME" -lt 1 ]; then
  FIRST_FRAME=1
fi

echo "[INFO] Running frames: $FIRST_FRAME .. $LAST_FRAME (skipping frame000-bogus.tif)"

# -----------------------------------------
# Run
# -----------------------------------------
echo "[STEP] Running tracker..."
mkdir -p "$OUT_DIR"

"$BIN" \
  "$FIRST_FRAME" \
  "$LAST_FRAME" \
  "$INPUT_PATTERN" \
  "$OUT_DIR" \
  "$CONFIG_FILE" \
  "$INITIAL_FILE" 2> >(grep -Ev "TIFF_Warning TIFFReadDirectory: Unknown field with tag 6500(0|1)?" >&2)

echo "======================================================================================================="
echo "Run finished (exit=0)."
echo "Results saved to:"
echo "$OUT_DIR"
echo "======================================================================================================="
