#!/bin/bash
set -euo pipefail

# Run original dataset (quiet OpenCV TIFF warnings)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$CPP_ROOT/build"

INITIAL_CSV="$CPP_ROOT/examples/initial.csv"
INPUT_PATTERN="$CPP_ROOT/examples/input/original_data/frame%03d.tif"
CONFIG_YAML="$CPP_ROOT/examples/config.yaml"
OUT_DIR="$CPP_ROOT/examples/output_original"

FIRST=1
LAST=19

mkdir -p "$OUT_DIR"

echo "****************************************************************************************************"
echo "Cell Universe Original Run"
echo "*******************************************************************************************************"
echo "Build Dir   : $BUILD_DIR"
echo "Input       : $INPUT_PATTERN"
echo "Initial CSV : $INITIAL_CSV"
echo "Config YAML : $CONFIG_YAML"
echo "Output Dir  : $OUT_DIR"
echo "Frames      : $FIRST .. $LAST"
echo "========================================="

if [ ! -d "$BUILD_DIR" ]; then
  echo "Build dir not found: $BUILD_DIR"
  echo "Compile from C++ root:"
  echo "  cd \"$CPP_ROOT\""
  echo "  mkdir -p build && cd build"
  echo "  cmake -S .. -B ."
  echo "  cmake --build . -j 8"
  exit 1
fi

cd "$BUILD_DIR"

export OPENCV_LOG_LEVEL=SILENT

ARGS=(
  "$FIRST"
  "$LAST"
  "$INITIAL_CSV"
  "$INPUT_PATTERN"
  "$OUT_DIR"
  "$CONFIG_YAML"
)

echo "[CMD] ./celluniverse ${ARGS[*]}"

./celluniverse "${ARGS[@]}" \
  2> >(grep -Ev "TIFF_Warning TIFFReadDirectory: Unknown field with tag 6500(0|1)" >&2)

echo "========================================="
echo "Run finished (exit=0)."
echo "Results saved to:"
echo "$OUT_DIR"
echo "========================================="