#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_ROOT="$CPP_ROOT/output"
mkdir -p "$OUTPUT_ROOT"

INPUT_DIR="$CPP_ROOT/examples/input/Simulated_nuclei_HL60 cells_stained_with_Hoechst/01"
INPUT_PATTERN="$INPUT_DIR/t%03d.tif"
CONFIG_FILE="$CPP_ROOT/config/config.yaml"
INITIAL_FILE="$CPP_ROOT/config/initial_HL60_0.csv"
OUT_DIR="$OUTPUT_ROOT/output_HL60_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUT_DIR/run_HL60_$(date +%Y%m%d_%H%M%S).txt"

BUILD_DIR="$CPP_ROOT/build"
FALLBACK_BUILD_DIR="$CPP_ROOT/cmake-build-debug"
SKIP_CLEAN="${CELLUNIVERSE_SKIP_CLEAN:-0}"

[ -d "$CPP_ROOT" ] || { echo "[FATAL] CPP root not found: $CPP_ROOT"; exit 1; }
[ -d "$INPUT_DIR" ] || { echo "[FATAL] input dir not found: $INPUT_DIR"; exit 1; }
[ -f "$CONFIG_FILE" ] || { echo "[FATAL] config not found: $CONFIG_FILE"; exit 1; }
[ -f "$INITIAL_FILE" ] || { echo "[FATAL] initial csv not found: $INITIAL_FILE"; exit 1; }
[ -f "$CPP_ROOT/CMakeLists.txt" ] || { echo "[FATAL] CMakeLists.txt not found in: $CPP_ROOT"; exit 1; }

LAST_FRAME_AUTO=$(
  find "$INPUT_DIR" -maxdepth 1 -type f -name 't*.tif' |
  sed -E 's#^.*/t([0-9]+)\.tif$#\1#' |
  sort -n |
  tail -n 1
)
[ -n "$LAST_FRAME_AUTO" ] || { echo "[FATAL] no t*.tif files found in: $INPUT_DIR"; exit 1; }

FIRST_FRAME=0
LAST_FRAME="149"

if [ "${1:-}" != "" ] && [ "${2:-}" != "" ]; then
  FIRST_FRAME="$1"
  LAST_FRAME="$2"
elif [ "${1:-}" != "" ]; then
  LAST_FRAME="$1"
fi

if [ "$FIRST_FRAME" -lt 0 ]; then
  FIRST_FRAME=0
fi

mkdir -p "$OUT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "======================================================================================================="
echo "Cell Universe HL60 Run"
echo "======================================================================================================="
echo "CPP Root        : $CPP_ROOT"
echo "Input Dir       : $INPUT_DIR"
echo "Input Pattern   : $INPUT_PATTERN"
echo "Initial CSV     : $INITIAL_FILE"
echo "Config YAML     : $CONFIG_FILE"
echo "Output Dir      : $OUT_DIR"
echo "Run Log         : $LOG_FILE"
echo "Skip Clean      : $SKIP_CLEAN"
echo "Auto Last Frame : $LAST_FRAME_AUTO"
echo "======================================================================================================="

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

echo "[INFO] Running frames: $FIRST_FRAME .. $LAST_FRAME"

echo "[STEP] Running tracker..."
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
