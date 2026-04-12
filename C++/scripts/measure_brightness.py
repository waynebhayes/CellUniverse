#!/usr/bin/env python3
"""Measure per-cell brightness from frame001.tif and update initial.csv.

Applies the same preprocessing pipeline as the C++ code:
  1. BGR -> Gray -> normalize to [0,1] -> Gaussian blur (sigma=10)
  2. Calibration: measure background in calibration zone
  3. Sigmoid contrast: 1 / (1 + exp(-k * (x - center)))

For each cell, measures mean pixel brightness inside a circular region
at the cell's z-slice, then writes brightness to initial.csv.
"""

import csv
import sys
import os
import cv2
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
TIFF_PATH = os.path.join(PROJECT_DIR, "data", "input", "original_data", "frame001.tif")
CSV_PATH = os.path.join(PROJECT_DIR, "config", "initial.csv")
CONFIG_PATH = os.path.join(PROJECT_DIR, "config", "config.yaml")

# Config defaults (matching config.yaml)
BLUR_SIGMA = 1.5
CAL_X, CAL_Y, CAL_Z = 20, 20, 0
CAL_W, CAL_H = 50, 31
Z_SCALING = 7

def load_and_preprocess(tiff_path):
    """Load multi-page TIFF, preprocess each slice (gray, normalize, blur). No sigmoid."""
    ret, slices = cv2.imreadmulti(tiff_path, flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if not ret or len(slices) == 0:
        print(f"Error: Could not read {tiff_path}")
        sys.exit(1)

    print(f"Loaded {len(slices)} TIFF slices from {tiff_path}")

    processed = []
    for s in slices:
        gray = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
        f32 = gray.astype(np.float32) / 255.0
        blurred = cv2.GaussianBlur(f32, (0, 0), BLUR_SIGMA)
        processed.append(blurred)

    # Measure background brightness
    cal_slice = processed[CAL_Z]
    cal_region = cal_slice[CAL_Y:CAL_Y+CAL_H, CAL_X:CAL_X+CAL_W]
    bg_mean = float(np.mean(cal_region))
    print(f"[Calibration] bgMean={bg_mean:.4f} (this will be background_color)")

    return processed, bg_mean


def measure_cell_brightness(processed_slices, x, y, z_raw, major_r):
    """Measure mean brightness inside a circular region at the cell's z-slice."""
    # z_raw is in raw TIFF space (not scaled)
    z_idx = int(round(z_raw))
    z_idx = max(0, min(z_idx, len(processed_slices) - 1))

    slice_img = processed_slices[z_idx]
    h, w = slice_img.shape

    # Create circular mask centered at (x, y) with radius = majorRadius
    r = int(round(major_r))
    x_c, y_c = int(round(x)), int(round(y))

    # Bounding box
    min_x = max(0, x_c - r)
    max_x = min(w - 1, x_c + r)
    min_y = max(0, y_c - r)
    max_y = min(h - 1, y_c + r)

    pixels = []
    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            dx = px - x
            dy = py - y
            if dx*dx + dy*dy <= major_r * major_r:
                pixels.append(slice_img[py, px])

    if len(pixels) == 0:
        print(f"  Warning: no pixels found for cell at ({x:.1f}, {y:.1f}, z={z_raw})")
        return 0.5  # fallback

    mean_brightness = float(np.mean(pixels))
    return mean_brightness


def main():
    if not os.path.exists(TIFF_PATH):
        print(f"Error: TIFF not found at {TIFF_PATH}")
        sys.exit(1)

    processed_slices, post_bg = load_and_preprocess(TIFF_PATH)

    # Read initial.csv
    rows = []
    with open(CSV_PATH, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)

    print(f"\nMeasuring brightness for {len(rows)} cells:")
    print(f"{'Name':<40} {'x':>8} {'y':>8} {'z':>5} {'majorR':>7} {'brightness':>10}")
    print("-" * 85)

    updated_rows = []
    for row in rows:
        file_name = row[0]
        name = row[1]
        x = float(row[2])
        y = float(row[3])
        z_raw = float(row[4])  # raw z (not scaled)
        major_r = float(row[5])
        minor_r = float(row[6])

        brightness = measure_cell_brightness(processed_slices, x, y, z_raw, major_r)
        print(f"{name:<40} {x:8.2f} {y:8.2f} {z_raw:5.0f} {major_r:7.1f} {brightness:10.4f}")

        updated_rows.append({
            'file': file_name,
            'name': name,
            'x': x,
            'y': y,
            'z': z_raw,
            'majorRadius': major_r,
            'minorRadius': minor_r,
            'brightness': brightness,
        })

    # Write updated CSV
    print(f"\nWriting updated CSV to {CSV_PATH}")
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'name', 'x', 'y', 'z', 'majorRadius', 'minorRadius', 'brightness'])
        for r in updated_rows:
            writer.writerow([
                r['file'], r['name'],
                r['x'], r['y'], r['z'],
                r['majorRadius'], r['minorRadius'],
                f"{r['brightness']:.4f}",
            ])

    print("Done.")


if __name__ == "__main__":
    main()
