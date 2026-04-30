#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from make_embryo_demo_gif import (
    DEFAULT_BASE_DIR,
    SEGMENTS,
    build_overlay,
    compute_global_window,
    draw_label,
    iter_frame_specs,
    make_grayscale_rgb,
    natural_pngs,
    normalize_to_uint8,
    resize_rgb,
    viridis_like_rgb,
)


DEFAULT_OUTPUT = DEFAULT_BASE_DIR / "embryo_demo_1_84_3panel_2fps_oblique.gif"


def load_stack(frame_dir: Path) -> np.ndarray:
    pngs = natural_pngs(frame_dir)
    if not pngs:
        raise RuntimeError(f"No PNG slices found in {frame_dir}")
    return np.stack([np.array(Image.open(png), dtype=np.float32) for png in pngs], axis=0)


def oblique_mean_project(
    stack: np.ndarray,
    x_shift_total: int,
    y_shift_total: int,
) -> np.ndarray:
    depth, height, width = stack.shape
    if depth == 1:
        return stack[0]

    z_positions = np.linspace(0.0, 1.0, depth, dtype=np.float32)
    raw_x = np.round(z_positions * x_shift_total).astype(int)
    raw_y = np.round(-z_positions * y_shift_total).astype(int)

    x_min, x_max = int(raw_x.min()), int(raw_x.max())
    y_min, y_max = int(raw_y.min()), int(raw_y.max())

    canvas_width = width + (x_max - x_min)
    canvas_height = height + (y_max - y_min)

    accum = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    count = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    for z in range(depth):
        x0 = int(raw_x[z] - x_min)
        y0 = int(raw_y[z] - y_min)
        accum[y0 : y0 + height, x0 : x0 + width] += stack[z]
        count[y0 : y0 + height, x0 : x0 + width] += 1.0

    count[count == 0.0] = 1.0
    return accum / count


def compose_oblique_canvas(
    real_rgb: np.ndarray,
    synth_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    frame_number: int,
    top_panel_width: int,
    margin: int,
    gap: int,
) -> np.ndarray:
    top_real = resize_rgb(real_rgb, top_panel_width)
    top_synth = resize_rgb(synth_rgb, top_panel_width)
    bottom_width = top_real.shape[1] + top_synth.shape[1] + gap
    bottom_overlay = resize_rgb(overlay_rgb, bottom_width)

    canvas_width = margin * 2 + bottom_overlay.shape[1]
    canvas_height = margin * 3 + gap + top_real.shape[0] + bottom_overlay.shape[0]

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(18, 18, 18))
    canvas.paste(Image.fromarray(top_real), (margin, margin))
    canvas.paste(Image.fromarray(top_synth), (margin + top_real.shape[1] + gap, margin))
    canvas.paste(Image.fromarray(bottom_overlay), (margin, margin * 2 + top_real.shape[0] + gap))

    draw = ImageDraw.Draw(canvas)
    draw_label(draw, (margin + 10, margin + 10), f"Real oblique  frame {frame_number}")
    draw_label(draw, (margin + top_real.shape[1] + gap + 10, margin + 10), "Synth oblique")
    draw_label(
        draw,
        (margin + 10, margin * 2 + top_real.shape[0] + gap + 10),
        "Overlay oblique  synth opacity 0.5  viridis",
    )
    return np.array(canvas)


def resolve_output_format(output: Path, requested_format: str) -> tuple[Path, str]:
    if requested_format == "auto":
        suffix = output.suffix.lower()
        if suffix == ".gif":
            return output, "gif"
        if suffix == ".mp4":
            return output, "mp4"
        raise ValueError(
            "Cannot infer output format from file suffix. "
            "Use --format gif or --format mp4."
        )

    expected_suffix = f".{requested_format}"
    if output.suffix.lower() == expected_suffix:
        return output, requested_format

    adjusted = output.with_suffix(expected_suffix)
    print(f"[INFO] Adjust output suffix to match format: {adjusted}")
    return adjusted, requested_format


def ensure_even_frame_size(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    pad_h = height % 2
    pad_w = width % 2
    if pad_h == 0 and pad_w == 0:
        return frame

    new_height = height + pad_h
    new_width = width + pad_w
    padded = np.zeros((new_height, new_width, 3), dtype=frame.dtype)
    padded[:height, :width] = frame
    if pad_h:
        padded[height:, :width] = frame[height - 1 : height, :width]
    if pad_w:
        padded[:height, width:] = frame[:height, width - 1 : width]
    if pad_h and pad_w:
        padded[height:, width:] = frame[height - 1, width - 1]
    return padded


def write_animation(frames: list[np.ndarray], output: Path, fps: float, fmt: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "gif":
        imageio.mimsave(output, frames, duration=1.0 / fps, loop=0)
        print(f"[DONE] Oblique GIF saved to {output}")
        return

    if fmt != "mp4":
        raise ValueError(f"Unsupported format: {fmt}")

    even_frames = [ensure_even_frame_size(frame) for frame in frames]
    height, width = even_frames[0].shape[:2]
    codec_candidates = ["avc1", "H264", "mp4v"]
    writer = None
    chosen_codec = None
    for codec in codec_candidates:
        candidate = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if candidate.isOpened():
            writer = candidate
            chosen_codec = codec
            print(f"[INFO] MP4 codec selected: {codec}")
            break
        candidate.release()

    if writer is None:
        raise RuntimeError(f"Failed to open MP4 writer for {output}")

    try:
        for frame in even_frames:
            writer.write(cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    print(f"[DONE] Oblique MP4 saved to {output} using codec {chosen_codec}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build embryo oblique-view 3-panel demo animation."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--format",
        choices=("auto", "gif", "mp4"),
        default="auto",
        help="Output format. Default infers from --output suffix.",
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--top-panel-width", type=int, default=420)
    parser.add_argument("--margin", type=int, default=20)
    parser.add_argument("--gap", type=int, default=20)
    parser.add_argument("--x-shift-total", type=int, default=160)
    parser.add_argument("--y-shift-total", type=int, default=90)
    args = parser.parse_args()

    output_path, output_format = resolve_output_format(args.output, args.format)

    frame_specs = list(iter_frame_specs(args.base_dir))
    print(f"[INFO] Building oblique {output_format.upper()} from {len(frame_specs)} frames")
    print(
        f"[INFO] Oblique projection offsets x_total={args.x_shift_total} "
        f"y_total={args.y_shift_total}"
    )

    real_proj: list[np.ndarray] = []
    synth_proj: list[np.ndarray] = []
    frame_numbers: list[int] = []

    for frame_number, real_dir, synth_dir in frame_specs:
        print(f"[LOAD] frame {frame_number}")
        real_stack = load_stack(real_dir)
        synth_stack = load_stack(synth_dir)
        real_proj.append(
            oblique_mean_project(
                real_stack,
                x_shift_total=args.x_shift_total,
                y_shift_total=args.y_shift_total,
            )
        )
        synth_proj.append(
            oblique_mean_project(
                synth_stack,
                x_shift_total=args.x_shift_total,
                y_shift_total=args.y_shift_total,
            )
        )
        frame_numbers.append(frame_number)

    real_low, real_high = compute_global_window(real_proj, 1.0, 99.7)
    synth_low, synth_high = compute_global_window(synth_proj, 1.0, 99.7)
    print(f"[INFO] real window  = [{real_low:.3f}, {real_high:.3f}]")
    print(f"[INFO] synth window = [{synth_low:.3f}, {synth_high:.3f}]")

    gif_frames: list[np.ndarray] = []
    for idx, frame_number in enumerate(frame_numbers):
        real_u8 = normalize_to_uint8(real_proj[idx], real_low, real_high)
        synth_u8 = normalize_to_uint8(synth_proj[idx], synth_low, synth_high)

        real_rgb = make_grayscale_rgb(real_u8)
        synth_rgb = viridis_like_rgb(synth_u8)
        overlay_rgb = build_overlay(real_u8, synth_u8)

        canvas = compose_oblique_canvas(
            real_rgb,
            synth_rgb,
            overlay_rgb,
            frame_number,
            top_panel_width=args.top_panel_width,
            margin=args.margin,
            gap=args.gap,
        )
        gif_frames.append(canvas)

    write_animation(gif_frames, output_path, args.fps, output_format)


if __name__ == "__main__":
    main()
