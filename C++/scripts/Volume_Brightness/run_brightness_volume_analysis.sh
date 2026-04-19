#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_ROOT="$CPP_ROOT/output"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

LOG_FILE="$LOG_DIR/brightness_volume_runLog_$(date +%Y%m%d_%H%M%S).txt"
touch "$LOG_FILE"

log() {
  echo "$@" | tee -a "$LOG_FILE"
}

run_and_log() {
  "$@" 2>&1 | tee -a "$LOG_FILE"
}

log "[INFO] Logging to: $LOG_FILE"

INPUT_DIR="$CPP_ROOT/examples/input/C.elegans_developing embryo_Fluo-N3DH-CE_Training/01"
CONFIG_FILE="$CPP_ROOT/config/config.yaml"
OUTPUT_BASE="$OUTPUT_ROOT"
BUILD_DIR="$CPP_ROOT/build"

FIRST_FRAME=1
LAST_FRAME=10

if [ "${1:-}" != "" ] && [ "${2:-}" != "" ]; then
  FIRST_FRAME="$1"
  LAST_FRAME="$2"
elif [ "${1:-}" != "" ]; then
  LAST_FRAME="$1"
fi

RUN_TAG="brightness_f$(printf "%03d" "$FIRST_FRAME")_to_f$(printf "%03d" "$LAST_FRAME")_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUTPUT_BASE/$RUN_TAG"

log "======================================================================================================="
log "Brightness/Volume Analysis Run"
log "======================================================================================================="
log "CPP Root        : $CPP_ROOT"
log "Input Dir       : $INPUT_DIR"
log "Config YAML     : $CONFIG_FILE"
log "Frame Range     : $FIRST_FRAME .. $LAST_FRAME"
log "Output Base     : $OUTPUT_BASE"
log "Run Output Dir  : $OUT_DIR"
log "======================================================================================================="

[ -d "$CPP_ROOT" ] || { echo "[FATAL] CPP root not found: $CPP_ROOT"; exit 1; }
[ -d "$INPUT_DIR" ] || { echo "[FATAL] input dir not found: $INPUT_DIR"; exit 1; }
[ -f "$CONFIG_FILE" ] || { echo "[FATAL] config not found: $CONFIG_FILE"; exit 1; }
[ -f "$CPP_ROOT/CMakeLists.txt" ] || { echo "[FATAL] CMakeLists.txt not found in: $CPP_ROOT"; exit 1; }

mkdir -p "$BUILD_DIR"
mkdir -p "$OUT_DIR"

log "[STEP] Reconfiguring CMake..."
run_and_log cmake -S "$CPP_ROOT" -B "$BUILD_DIR"

log "[STEP] Building brightness_volume_analyzer..."
run_and_log cmake --build "$BUILD_DIR" --target brightness_volume_analyzer -- -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

BIN="$BUILD_DIR/brightness_volume_analyzer"
[ -x "$BIN" ] || { echo "[FATAL] build succeeded but binary not found/executable: $BIN"; exit 1; }

log "[STEP] Running analysis..."
"$BIN" \
  "$FIRST_FRAME" \
  "$LAST_FRAME" \
  "$INPUT_DIR" \
  "$OUT_DIR" \
  "$CONFIG_FILE" 2>&1 | grep -Ev "TIFF_Warning TIFFReadDirectory: Unknown field with tag 6500(0|1)?" | tee -a "$LOG_FILE"

RESULT_DIR="$OUT_DIR/brightness_volume_analysis"

log "======================================================================================================="
log "Run finished (exit=0)."
log "Result directory:"
log "$RESULT_DIR"
log ""
log "Main outputs:"
log "  - $RESULT_DIR/per_cell_observations.csv"
log "  - $RESULT_DIR/per_cell_summary.csv"
log "  - $RESULT_DIR/split_events.csv"
log "  - $RESULT_DIR/report.txt"
log "  - $RESULT_DIR/volume_vs_mean_intensity_scatter.png"
log "  - $RESULT_DIR/volume_vs_integrated_intensity_scatter.png"
log "  - $RESULT_DIR/per_cell_normalized_time_series.png"
log "======================================================================================================="
