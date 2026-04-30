#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_DIR="$CPP_ROOT/output/✅embryo1~84 1.42G 20250430"
FORMAT="${1:-gif}"

case "$FORMAT" in
  gif|mp4) ;;
  *)
    echo "[FATAL] format must be gif or mp4"
    exit 1
    ;;
esac

OUTPUT_FILE="$BASE_DIR/embryo_demo_1_84_3panel_2fps_oblique.$FORMAT"

echo "[INFO] Base dir   : $BASE_DIR"
echo "[INFO] Format     : $FORMAT"
echo "[INFO] Output     : $OUTPUT_FILE"

python3 "$SCRIPT_DIR/make_embryo_demo_gif_oblique.py" \
  --base-dir "$BASE_DIR" \
  --output "$OUTPUT_FILE" \
  --format "$FORMAT" \
  --fps 2 \
  --x-shift-total 160 \
  --y-shift-total 90
