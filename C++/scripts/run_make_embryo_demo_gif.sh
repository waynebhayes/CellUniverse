#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_DIR="$CPP_ROOT/output/✅embryo1~84 1.42G 20250430"
OUTPUT_GIF="$BASE_DIR/embryo_demo_1_84_3panel_2fps.gif"

echo "[INFO] Base dir   : $BASE_DIR"
echo "[INFO] Output GIF : $OUTPUT_GIF"

python3 "$SCRIPT_DIR/make_embryo_demo_gif.py" \
  --base-dir "$BASE_DIR" \
  --output "$OUTPUT_GIF" \
  --fps 2
