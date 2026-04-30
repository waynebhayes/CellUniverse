#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


DEFAULT_BASE_DIR = Path(
    "/Users/wangyiding/CellUniverse/C++/output/✅embryo1~84 1.42G 20250430"
)
DEFAULT_OUTPUT = DEFAULT_BASE_DIR / "embryo_demo_1_84_3panel_2fps.gif"


SEGMENTS = [
    (1, 39, "✅embryo1~40_20260414_022211"),
    (40, 52, "✅embryo40~53_20260420.1206"),
    (53, 79, "✅embryo53~79❌105 20260427.0233"),
    (80, 80, "✅embryo80_perfect_bridge_rescue"),
    (81, 84, "✅embryo81~84❌89"),
]


def natural_pngs(frame_dir: Path) -> list[Path]:
    return sorted(frame_dir.glob("*.png"), key=lambda p: int(p.stem))


def iter_frame_specs(base_dir: Path) -> Iterable[tuple[int, Path, Path]]:
    for start, end, stage_name in SEGMENTS:
        stage_dir = base_dir / stage_name
        real_root = stage_dir / "real"
        synth_root = stage_dir / "synth"
        if not real_root.exists() or not synth_root.exists():
            raise FileNotFoundError(f"Missing real/synth directories under {stage_dir}")
        for frame in range(start, end + 1):
            real_dir = real_root / str(frame)
            synth_dir = synth_root / str(frame)
            if not real_dir.exists():
                raise FileNotFoundError(f"Missing real frame directory: {real_dir}")
            if not synth_dir.exists():
                raise FileNotFoundError(f"Missing synth frame directory: {synth_dir}")
            yield frame, real_dir, synth_dir


def mean_project(frame_dir: Path) -> np.ndarray:
    pngs = natural_pngs(frame_dir)
    if not pngs:
        raise RuntimeError(f"No PNG slices found in {frame_dir}")

    accum = None
    for png in pngs:
        arr = np.array(Image.open(png), dtype=np.float32)
        if accum is None:
            accum = arr
        else:
            accum += arr
    assert accum is not None
    return accum / float(len(pngs))


def compute_global_window(images: list[np.ndarray], low_q: float, high_q: float) -> tuple[float, float]:
    flat = np.concatenate([img.ravel() for img in images])
    low = float(np.percentile(flat, low_q))
    high = float(np.percentile(flat, high_q))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        low, high = float(flat.min()), float(flat.max())
    if high <= low:
        high = low + 1.0
    return low, high


def normalize_to_uint8(image: np.ndarray, low: float, high: float) -> np.ndarray:
    clipped = np.clip(image, low, high)
    scaled = (clipped - low) / (high - low)
    return np.round(scaled * 255.0).astype(np.uint8)


def make_grayscale_rgb(gray_u8: np.ndarray) -> np.ndarray:
    return np.repeat(gray_u8[:, :, None], 3, axis=2)


def viridis_like_rgb(gray_u8: np.ndarray) -> np.ndarray:
    anchors = np.array(
        [
            [68, 1, 84],
            [71, 44, 122],
            [59, 81, 139],
            [44, 113, 142],
            [33, 145, 140],
            [39, 173, 129],
            [92, 200, 99],
            [170, 220, 50],
            [253, 231, 37],
        ],
        dtype=np.float32,
    )
    anchor_x = np.linspace(0.0, 1.0, len(anchors), dtype=np.float32)
    values = gray_u8.astype(np.float32) / 255.0

    r = np.interp(values, anchor_x, anchors[:, 0])
    g = np.interp(values, anchor_x, anchors[:, 1])
    b = np.interp(values, anchor_x, anchors[:, 2])
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def resize_rgb(image: np.ndarray, width: int) -> np.ndarray:
    pil = Image.fromarray(image, mode="RGB")
    new_height = int(round(pil.height * (width / pil.width)))
    return np.array(pil.resize((width, new_height), resample=Image.Resampling.BILINEAR))


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text)
    pad = 6
    draw.rounded_rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        radius=10,
        fill=(0, 0, 0),
    )
    draw.text((x, y), text, fill=(255, 255, 255))


def compose_canvas(
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

    bottom_available_width = top_real.shape[1] + top_synth.shape[1] + gap
    bottom_overlay = resize_rgb(overlay_rgb, bottom_available_width)

    canvas_width = margin * 2 + bottom_overlay.shape[1]
    canvas_height = margin * 3 + gap + top_real.shape[0] + bottom_overlay.shape[0]

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(18, 18, 18))
    canvas.paste(Image.fromarray(top_real), (margin, margin))
    canvas.paste(Image.fromarray(top_synth), (margin + top_real.shape[1] + gap, margin))
    canvas.paste(Image.fromarray(bottom_overlay), (margin, margin * 2 + top_real.shape[0] + gap))

    draw = ImageDraw.Draw(canvas)
    draw_label(draw, (margin + 10, margin + 10), f"Real  frame {frame_number}")
    draw_label(draw, (margin + top_real.shape[1] + gap + 10, margin + 10), "Synth")
    draw_label(
        draw,
        (margin + 10, margin * 2 + top_real.shape[0] + gap + 10),
        "Overlay  synth opacity 0.5  viridis",
    )
    return np.array(canvas)


def build_overlay(real_u8: np.ndarray, synth_u8: np.ndarray) -> np.ndarray:
    real_rgb = make_grayscale_rgb(real_u8).astype(np.float32)
    synth_rgb = viridis_like_rgb(synth_u8).astype(np.float32)
    overlay = np.clip(real_rgb * 0.5 + synth_rgb * 0.5, 0, 255)
    return overlay.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build embryo 3-panel demo GIF.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--top-panel-width", type=int, default=420)
    parser.add_argument("--margin", type=int, default=20)
    parser.add_argument("--gap", type=int, default=20)
    args = parser.parse_args()

    frame_specs = list(iter_frame_specs(args.base_dir))
    print(f"[INFO] Building GIF from {len(frame_specs)} frames")

    real_proj: list[np.ndarray] = []
    synth_proj: list[np.ndarray] = []
    frame_numbers: list[int] = []

    for frame_number, real_dir, synth_dir in frame_specs:
        print(f"[LOAD] frame {frame_number}")
        real_proj.append(mean_project(real_dir))
        synth_proj.append(mean_project(synth_dir))
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
        synth_rgb = make_grayscale_rgb(synth_u8)
        overlay_rgb = build_overlay(real_u8, synth_u8)

        canvas = compose_canvas(
            real_rgb,
            synth_rgb,
            overlay_rgb,
            frame_number,
            top_panel_width=args.top_panel_width,
            margin=args.margin,
            gap=args.gap,
        )
        gif_frames.append(canvas)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    duration = 1.0 / args.fps
    imageio.mimsave(args.output, gif_frames, duration=duration, loop=0)
    print(f"[DONE] GIF saved to {args.output}")


if __name__ == "__main__":
    main()
