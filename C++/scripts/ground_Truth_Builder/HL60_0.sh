#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_ROOT="$CPP_ROOT/output"
mkdir -p "$OUTPUT_ROOT"

INPUT_FILE="$CPP_ROOT/examples/input/Simulated_nuclei_HL60 cells_stained_with_Hoechst/01/t000.tif"
CONFIG_FILE="$CPP_ROOT/config/config.yaml"
CSV_FILE_NAME="initial_HL60_0.csv"
OUT_DIR="$OUTPUT_ROOT/output_HL60_0_$(date +%Y%m%d_%H%M%S)"
CSV_OUTPUT="$OUT_DIR/$CSV_FILE_NAME"
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/run_f000_ground_truth.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[INFO] Logging to: $LOG_FILE"

BUILD_DIR="$CPP_ROOT/build"
FALLBACK_BUILD_DIR="$CPP_ROOT/cmake-build-debug"
SKIP_CLEAN="${CELLUNIVERSE_SKIP_CLEAN:-0}"

echo "======================================================================================================="
echo "Cell Universe HL60 Frame 0 Ground Truth Builder"
echo "======================================================================================================="
echo "CPP Root        : $CPP_ROOT"
echo "Input File      : $INPUT_FILE"
echo "Config YAML     : $CONFIG_FILE"
echo "CSV Output      : $CSV_OUTPUT"
echo "Output Dir      : $OUT_DIR"
echo "Run Log         : $LOG_FILE"
echo "Skip Clean      : $SKIP_CLEAN"
echo "======================================================================================================="

[ -d "$CPP_ROOT" ] || { echo "[FATAL] CPP root not found: $CPP_ROOT"; exit 1; }
[ -f "$INPUT_FILE" ] || { echo "[FATAL] input frame not found: $INPUT_FILE"; exit 1; }
[ -f "$CONFIG_FILE" ] || { echo "[FATAL] config not found: $CONFIG_FILE"; exit 1; }
[ -f "$CPP_ROOT/CMakeLists.txt" ] || { echo "[FATAL] CMakeLists.txt not found in: $CPP_ROOT"; exit 1; }

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

echo "[STEP] Reconfiguring CMake..."
cmake -S "$CPP_ROOT" -B "$BUILD_DIR"

echo "[STEP] Building..."
cmake --build "$BUILD_DIR" -- -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

BIN="$BUILD_DIR/celluniverse"
[ -x "$BIN" ] || { echo "[FATAL] build succeeded but binary not found/executable: $BIN"; exit 1; }

echo "[STEP] Running ground-truth builder..."

"$BIN" \
  --build-ground-truth \
  "$INPUT_FILE" \
  "$OUT_DIR" \
  "$CONFIG_FILE" \
  "$CSV_OUTPUT" 2> >(grep -Ev "TIFF_Warning TIFFReadDirectory: Unknown field with tag 6500(0|1)?" >&2)

echo "======================================================================================================="
echo "Run finished (exit=0)."
echo "Results saved to:"
echo "$OUT_DIR"
echo "CSV saved to:"
echo "$CSV_OUTPUT"
echo "======================================================================================================="
