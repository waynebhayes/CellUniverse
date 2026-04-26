from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence

__all__ = [
    "evaluate_sequence_contrast_score",
    "evaluate_sequence_percentile_michelson_contrast",
    "evaluate_sequence_percentile_weber_contrast",
    "load_tiff_as_normalized_sequence",
    "process_prepared_sequence",
    "run_tiff_processing_pipeline",
    "save_normalized_sequence_as_tiff",
]

DEFAULT_INPUT_TIFF_PATH = Path("/home/blue-lobster/p2/UCI/CS295p/images/input/frame001.tif")
DEFAULT_OUTPUT_TIFF_PATH = Path("output_processed_clean.tif")


def load_tiff_as_normalized_sequence(tiff_path: str | Path) -> np.ndarray:
    """Load an ordered TIFF stack and normalize pixel values to 0.0..1.0."""

    tiff_path = Path(tiff_path)
    with Image.open(tiff_path) as image:
        slices = [
            np.asarray(frame, dtype=np.float32) / 255.0
            for frame in ImageSequence.Iterator(image)
        ]

    if not slices:
        raise ValueError(f"No image slices found in TIFF file: {tiff_path}")

    return np.stack(slices, axis=0)


def save_normalized_sequence_as_tiff(
    sequence: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Save a normalized 2D or 3D NumPy array as a TIFF image sequence."""

    output_path = Path(output_path)
    array = np.asarray(sequence, dtype=np.float32)

    if array.ndim < 2:
        raise ValueError("Expected at least a 2D image array.")
    if array.ndim == 2:
        array = array[np.newaxis, ...]

    uint8_array = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    frames = [Image.fromarray(slice_array) for slice_array in uint8_array]

    if not frames:
        raise ValueError("Cannot save an empty image sequence.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_frame, *remaining_frames = frames
    first_frame.save(
        output_path,
        format="TIFF",
        save_all=True,
        append_images=remaining_frames,
    )
    return output_path


def evaluate_sequence_contrast_score(
    sequence: np.ndarray,
    inner_window_size: int = 51,
    outer_window_size: int = 101,
    percentile: float = 0.40,
    structure_threshold: float = 0.02,
    eps: float = 1e-6,
) -> float:
    """Return a local center-surround contrast score for a slice sequence."""

    if inner_window_size < 1 or outer_window_size < 1:
        raise ValueError("Window sizes must be positive integers.")
    if inner_window_size % 2 == 0 or outer_window_size % 2 == 0:
        raise ValueError("Window sizes must be odd integers.")
    if inner_window_size >= outer_window_size:
        raise ValueError("outer_window_size must be larger than inner_window_size.")

    array = _as_3d_float_sequence(sequence)
    slice_scores: list[float] = []

    for slice_image in array:
        # Compare a smaller local average against a larger surrounding average.
        inner_mean = _box_mean(slice_image, inner_window_size)
        outer_mean = _box_mean(slice_image, outer_window_size)
        local_difference = np.abs(inner_mean - outer_mean)
        local_contrast = local_difference / (outer_mean + eps)
        # Ignore nearly flat regions so background does not dominate the score.
        informative_pixels = local_difference >= structure_threshold
        contrast_values = local_contrast[informative_pixels]

        if contrast_values.size == 0:
            slice_scores.append(0.0)
            continue

        slice_scores.append(
            float(np.percentile(contrast_values, _fraction_to_percentile(percentile)))
        )

    return float(np.median(slice_scores))


def evaluate_sequence_percentile_michelson_contrast(
    sequence: np.ndarray,
    low_percentile: float = 0.10,
    high_percentile: float = 0.90,
    eps: float = 1e-6,
) -> float:
    """Return a robust percentile-based Michelson contrast score."""

    if not 0.0 <= low_percentile < high_percentile <= 1.0:
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 1.")

    array = _as_3d_float_sequence(sequence)
    slice_scores: list[float] = []

    for slice_image in array:
        low_value = float(
            np.percentile(slice_image, _fraction_to_percentile(low_percentile))
        )
        high_value = float(
            np.percentile(slice_image, _fraction_to_percentile(high_percentile))
        )
        slice_scores.append((high_value - low_value) / (high_value + low_value + eps))

    return float(np.median(slice_scores))


def evaluate_sequence_percentile_weber_contrast(
    sequence: np.ndarray,
    background_percentile: float = 0.10,
    signal_percentile: float = 0.90,
    background_floor: float = 1.0 / 255.0,
    eps: float = 1e-6,
) -> float:
    """Return a robust percentile-based Weber contrast score."""

    if not 0.0 <= background_percentile < signal_percentile <= 1.0:
        raise ValueError(
            "Percentiles must satisfy 0 <= background_percentile < "
            "signal_percentile <= 1."
        )
    if background_floor < 0.0:
        raise ValueError("background_floor must be non-negative.")

    array = _as_3d_float_sequence(sequence)
    slice_scores: list[float] = []

    for slice_image in array:
        background_value = float(
            np.percentile(slice_image, _fraction_to_percentile(background_percentile))
        )
        signal_value = float(
            np.percentile(slice_image, _fraction_to_percentile(signal_percentile))
        )
        # Keep the denominator away from zero when the background becomes black.
        stable_background = max(background_value, background_floor)
        slice_scores.append((signal_value - stable_background) / (stable_background + eps))

    return float(np.median(slice_scores))


def process_prepared_sequence(sequence: np.ndarray) -> np.ndarray:
    """Run the current iterative enhancement loop on normalized TIFF data."""

    panelty = 0.1
    min_panelty = 0.005
    collapse_backoff = 0.99
    panelty_range = 0.9

    reward_gate = 1.0
    reward_gate_decrement = 0.008
    reward_gate_min = 0.025
    reward = panelty

    score_max = 1.5
    max_count = 300
    no_improvement_patience = 10
    improvement_tolerance = 0.01
    score_drop_stop_threshold = 0.1
    score_percentile = 0.05
    score_percentile_max = 0.90
    score_percentile_increment = 0.025

    post_process_blur_sigma = 2.5

    current = _as_3d_float_sequence(sequence).copy()
    best_sequence = current.copy()
    best_score = float("-inf")
    previous_score: float | None = None

    count = 0
    reward_next_round = True
    current_panelty = panelty
    no_improvement_count = 0
    restore_best_before_reward = False

    while True:
        # Apply the penalty only to values below the configured range ceiling.
        current[current < panelty_range] -= current_panelty
        current[current < 0.0] = 0.0

        if reward_next_round:
            if restore_best_before_reward:
                # Early reward recovery restarts from the best-known image state.
                current = best_sequence.copy()
                restore_best_before_reward = False

            current[current > reward_gate] += reward
            current[current > 1.0] = 1.0
            # Increase the scoring percentile gradually so later rounds favor
            # stronger structures instead of the lower-contrast tail.
            score_percentile = min(
                score_percentile + score_percentile_increment,
                score_percentile_max,
            )
            reward_gate = max(reward_gate_min, reward_gate - reward_gate_decrement)

        # The loop alternates between non-reward and reward rounds.
        reward_next_round = not reward_next_round

        score = evaluate_sequence_contrast_score(
            current,
            percentile=score_percentile,
        )

        if (
            previous_score is not None
            and previous_score - score >= score_drop_stop_threshold
        ):
            # A sharp score drop schedules an immediate recovery reward.
            reward_next_round = True
            restore_best_before_reward = True
            current_panelty = max(min_panelty, current_panelty * collapse_backoff)
            print("round: ", count, " | score: ", score, " | panelty: ", current_panelty)
            previous_score = score
            count += 1
            continue

        if score > best_score + improvement_tolerance:
            best_score = score
            best_sequence = current.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        print("round: ", count, " | score: ", score, " | panelty: ", current_panelty)
        previous_score = score
        count += 1

        if score >= score_max:
            break

        if score == 0.0:
            # Zero score means the current state lost all informative contrast.
            current_panelty = max(min_panelty, current_panelty * collapse_backoff)
            current = best_sequence.copy()
            reward_next_round = True
            continue

        if no_improvement_count >= no_improvement_patience:
            if current_panelty <= min_panelty:
                break

            current_panelty = max(min_panelty, current_panelty * collapse_backoff)
            current = best_sequence.copy()
            no_improvement_count = 0
            reward_next_round = True
            continue

        if count >= max_count:
            break

    print("best round score: ", best_score)

    # Apply a final gentle blur slice-by-slice after the iterative loop.
    best_sequence = np.stack(
        [
            cv2.GaussianBlur(slice_image, (0, 0), post_process_blur_sigma)
            for slice_image in best_sequence
        ],
        axis=0,
    )

    return np.clip(best_sequence, 0.0, 1.0)


def run_tiff_processing_pipeline(
    input_path: str | Path = DEFAULT_INPUT_TIFF_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_TIFF_PATH,
) -> tuple[Path, float, float, float]:
    """Load, process, evaluate, and export a TIFF image sequence."""

    prepared_sequence = load_tiff_as_normalized_sequence(input_path)
    processed_sequence = process_prepared_sequence(prepared_sequence)
    local_score = evaluate_sequence_contrast_score(processed_sequence)
    michelson_score = evaluate_sequence_percentile_michelson_contrast(
        processed_sequence
    )
    weber_score = evaluate_sequence_percentile_weber_contrast(processed_sequence)
    saved_path = save_normalized_sequence_as_tiff(processed_sequence, output_path)
    return saved_path, local_score, michelson_score, weber_score


def _as_3d_float_sequence(sequence: np.ndarray) -> np.ndarray:
    """Convert image input to a clipped float32 sequence with leading slice axis."""

    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim == 2:
        # Promote a single image to a one-slice sequence for uniform handling.
        array = array[np.newaxis, ...]
    elif array.ndim != 3:
        raise ValueError("Expected a 2D image or a 3D slice sequence.")

    return np.clip(array, 0.0, 1.0)


def _fraction_to_percentile(value: float) -> float:
    """Convert a percentile fraction in [0.0, 1.0] to NumPy's 0..100 scale."""

    if not 0.0 <= value <= 1.0:
        raise ValueError("Percentile fractions must be in the range [0.0, 1.0].")
    return value * 100.0


def _box_mean(image: np.ndarray, window_size: int) -> np.ndarray:
    """Return a same-shape box-filter mean using edge padding."""

    radius = window_size // 2
    # Edge padding avoids shrinking the image while keeping border handling simple.
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="edge")
    integral = np.pad(
        padded,
        ((1, 0), (1, 0)),
        mode="constant",
        constant_values=0.0,
    ).cumsum(axis=0).cumsum(axis=1)

    # Integral-image arithmetic gives a box-filter mean without Python loops.
    window_sum = (
        integral[window_size:, window_size:]
        - integral[:-window_size, window_size:]
        - integral[window_size:, :-window_size]
        + integral[:-window_size, :-window_size]
    )
    return window_sum / float(window_size * window_size)


if __name__ == "__main__":
    (
        exported_path,
        contrast_score,
        michelson_contrast_score,
        weber_contrast_score,
    ) = run_tiff_processing_pipeline()
    print(f"Saved processed TIFF to: {exported_path}")
    print(f"Sequence local contrast score: {contrast_score:.6f}")
    print(
        "Sequence percentile Michelson contrast score: "
        f"{michelson_contrast_score:.6f}"
    )
    print(f"Sequence percentile Weber contrast score: {weber_contrast_score:.6f}")
