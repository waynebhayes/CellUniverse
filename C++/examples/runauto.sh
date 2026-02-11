#!/bin/bash
set -euo pipefail

# Auto Run Script for CellUniverse (get rid of TIFF warnings)
# Location: C++/examples/runauto.sh

# Directory that contains this script: .../C++/examples
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# C++ repo root: .../C++ (parent of examples)
CPP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_DIR="$CPP_ROOT/build"
INPUT_DIR="$CPP_ROOT/examples/input/C.elegans_developing embryo_Fluo-N3DH-CE_Training/01"
CONFIG_FILE="$CPP_ROOT/examples/config.yaml"
INITIAL_FILE="$CPP_ROOT/examples/initial_auto.csv"
OUT_DIR="$CPP_ROOT/examples/output_comprehensive_$(date +%Y%m%d_%H%M%S)"

echo "*******************************************************************************************************"
echo "Cell Universe Auto Run"
echo "*******************************************************************************************************"
echo "CPP Root    : $CPP_ROOT"
echo "Build Dir   : $BUILD_DIR"
echo "Input Dir   : $INPUT_DIR"
echo "Initial CSV : $INITIAL_FILE"
echo "Config File : $CONFIG_FILE"
echo "Output Dir  : $OUT_DIR"
echo "========================================="

# Hard checks (fail fast with clear messages)
[ -d "$BUILD_DIR" ]   || { echo "[FATAL] build dir not found: $BUILD_DIR"; exit 1; }
[ -f "$CONFIG_FILE" ] || { echo "[FATAL] config not found: $CONFIG_FILE"; exit 1; }
[ -f "$INITIAL_FILE" ]|| { echo "[FATAL] initial csv not found: $INITIAL_FILE"; exit 1; }

mkdir -p "$OUT_DIR"

cd "$BUILD_DIR"

# Run & filter noisy TIFF warnings on stderr
./celluniverse \
  0 \
  2 \
  "$INITIAL_FILE" \
  "$INPUT_DIR/t%03d.tif" \
  "$OUT_DIR" \
  "$CONFIG_FILE" 2> >(grep -v "TIFF_Warning TIFFReadDirectory: Unknown field with tag 6500" >&2)

echo "========================================="
echo "Run finished (exit=0)."
echo "Results saved to:"
echo "$OUT_DIR"
echo "========================================="