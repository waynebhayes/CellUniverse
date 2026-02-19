#!/bin/bash
set -euo pipefail

# remove_old_outputs.sh <config.ini> <preset>
# Removes all output directories for the given preset, based on:
#   output_base_dir + output_name_rule

if [ -t 1 ]; then
  RED="\033[31m"
  GREEN="\033[32m"
  RESET="\033[0m"
else
  RED=""
  GREEN=""
  RESET=""
fi

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

expand_home() {
  local p="$1"
  if [[ "$p" == "~/"* ]]; then
    printf '%s/%s\n' "$HOME" "${p#~/}"
  else
    printf '%s\n' "$p"
  fi
}

resolve_path() {
  local p="$1"
  local base="$2"
  p="$(expand_home "$p")"
  if [[ "$p" == /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s/%s\n' "$base" "$p"
  fi
}

ini_get() {
  local ini_file="$1"
  local section="$2"
  local key="$3"
  awk -v target_section="$section" -v target_key="$key" '
    BEGIN { in_section=0; }
    /^[[:space:]]*[#;]/ { next; }
    /^[[:space:]]*\[/ {
      sec=$0
      gsub(/^[[:space:]]*\[/, "", sec)
      gsub(/\][[:space:]]*$/, "", sec)
      in_section=(sec==target_section)
      next
    }
    in_section {
      split($0, parts, "=")
      if (length(parts) < 2) next
      k=parts[1]
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", k)
      if (k==target_key) {
        v=substr($0, index($0, "=")+1)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
        print v
        exit
      }
    }
  ' "$ini_file"
}

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <config.ini> <preset>"
  exit 1
fi

INI_FILE="$(expand_home "$1")"
PRESET="$2"

[ -f "$INI_FILE" ] || { echo "[FATAL] config ini not found: $INI_FILE"; exit 1; }

INI_DIR="$(cd "$(dirname "$INI_FILE")" && pwd)"
OUTPUT_BASE_RAW="$(ini_get "$INI_FILE" "$PRESET" "output_base_dir")"
OUTPUT_RULE="$(ini_get "$INI_FILE" "$PRESET" "output_name_rule")"

[ -n "$OUTPUT_BASE_RAW" ] || { echo "[FATAL] missing output_base_dir in preset: $PRESET"; exit 1; }
[ -n "$OUTPUT_RULE" ] || { echo "[FATAL] missing output_name_rule in preset: $PRESET"; exit 1; }

OUTPUT_BASE_DIR="$(resolve_path "$OUTPUT_BASE_RAW" "$INI_DIR")"
[ -d "$OUTPUT_BASE_DIR" ] || { echo "[FATAL] output base dir not found: $OUTPUT_BASE_DIR"; exit 1; }

GLOB_RULE="$OUTPUT_RULE"
GLOB_RULE="${GLOB_RULE//\{preset\}/$PRESET}"
GLOB_RULE="${GLOB_RULE//\{timestamp\}/*}"

echo "Preset          : $PRESET"
echo "Config INI      : $INI_FILE"
echo "Output Base Dir : $OUTPUT_BASE_DIR"
echo "Match Pattern   : $GLOB_RULE"
echo

shopt -s nullglob
matches=( "$OUTPUT_BASE_DIR"/$GLOB_RULE )
shopt -u nullglob

if [ "${#matches[@]}" -eq 0 ]; then
  echo "No matching output directories found."
  exit 0
fi

# Keep the newest directory by modification time.
latest="$(ls -1dt "${matches[@]}" 2>/dev/null | head -n 1)"
[ -n "$latest" ] || { echo "[FATAL] failed to determine latest directory."; exit 1; }

printf "%b[KEEP]%b   %s\n" "$GREEN" "$RESET" "$latest"
removed=0
for p in "${matches[@]}"; do
  [ "$p" = "$latest" ] && continue
  if [ -d "$p" ]; then
    printf "%b[REMOVE]%b %s\n" "$RED" "$RESET" "$p"
    rm -rf "$p"
    removed=$((removed + 1))
  fi
done

if [ "$removed" -eq 0 ]; then
  echo "Only one matching directory exists; nothing removed."
else
  echo "Done. Removed $removed old directory(s), kept latest."
fi
