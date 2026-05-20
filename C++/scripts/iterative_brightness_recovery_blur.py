#!/usr/bin/env python3
"""Iteratively blur a TIFF stack and rescale bright centers after each blur."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter


# Processing parameters.
SIGMA = 2.5
MAX_ITERATIONS = 15
BRIGHT_CENTER_PERCENTILE = 0.0
EPSILON = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Iteratively Gaussian-blur a TIFF and multiply the whole image after "
            "each iteration to recover the original bright-center mean."
        )
    )
    parser.add_argument("input_tiff", type=Path, help="Path to the input TIFF file.")
    parser.add_argument("output_tiff", type=Path, help="Path for the processed TIFF.")
    return parser.parse_args()


def cast_like_input(image: np.ndarray, original_dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        image = np.rint(image)
        image = np.clip(image, info.min, info.max)
    return image.astype(original_dtype, copy=False)


def bright_center_mask(reference: np.ndarray) -> np.ndarray:
    nonzero = reference > 0
    if not np.any(nonzero):
        return np.ones(reference.shape, dtype=bool)

    threshold = np.percentile(reference[nonzero], BRIGHT_CENTER_PERCENTILE)
    return reference >= threshold


def main() -> None:
    args = parse_args()

    original = tifffile.imread(args.input_tiff)
    original_float = original.astype(np.float32, copy=False)
    processed = original_float.copy()

    mask = bright_center_mask(original_float)
    target_bright_mean = float(np.mean(original_float[mask]))

    print(f"input: {args.input_tiff}")
    print(f"output: {args.output_tiff}")
    print(f"sigma: {SIGMA}")
    print(f"max_iterations: {MAX_ITERATIONS}")
    print(f"bright_center_percentile: {BRIGHT_CENTER_PERCENTILE}")
    print(f"target_bright_center_mean: {target_bright_mean:.6f}")

    for iteration in range(1, MAX_ITERATIONS + 1):
        processed = gaussian_filter(processed, sigma=SIGMA)

        blurred_bright_mean = float(np.mean(processed[mask]))
        lost_fraction = (
            (target_bright_mean - blurred_bright_mean) / target_bright_mean
            if target_bright_mean > EPSILON
            else 0.0
        )
        recovery_factor = (
            target_bright_mean / max(blurred_bright_mean, EPSILON)
            if target_bright_mean > EPSILON
            else 1.0
        )

        processed *= recovery_factor

        recovered_bright_mean = float(np.mean(processed[mask]))
        print(
            f"iteration {iteration}: "
            f"blurred_bright_mean={blurred_bright_mean:.6f}, "
            f"lost={lost_fraction * 100:.3f}%, "
            f"factor={recovery_factor:.6f}, "
            f"recovered_bright_mean={recovered_bright_mean:.6f}"
        )

    output = cast_like_input(processed, original.dtype)
    args.output_tiff.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(args.output_tiff, output)

    print(f"wrote: {args.output_tiff}")
    print(f"shape: {output.shape}")
    print(f"dtype: {output.dtype}")


if __name__ == "__main__":
    main()
