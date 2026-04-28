#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import imageio.v3 as iio
import napari
import numpy as np
import tifffile as tiff
from qtpy.QtCore import QTimer


def natural_key(value: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(value))]


def collect_frame_dirs(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name))


def collect_recursive_images(folder: Path, patterns: tuple[str, ...]):
    files: list[Path] = []
    for pattern in patterns:
        files.extend(folder.rglob(pattern))
    return sorted(files, key=lambda p: natural_key(p))


def load_volume_from_files(files: list[Path]) -> np.ndarray:
    if not files:
        raise RuntimeError("No files were provided to load_volume_from_files().")

    if files[0].suffix.lower() in {".tif", ".tiff"}:
        slices = [tiff.imread(str(path)) for path in files]
    else:
        slices = [iio.imread(path) for path in files]

    if slices[0].ndim == 2:
        return np.stack(slices, axis=0)

    if slices[0].ndim == 3 and len(slices) == 1:
        return slices[0]

    raise RuntimeError(f"Unsupported slice structure for files under {files[0].parent}")


def load_time_series(folder: Path, label: str) -> np.ndarray:
    frame_dirs = collect_frame_dirs(folder)
    if frame_dirs:
        print(f"[INFO] {label} frame folders found: {len(frame_dirs)}")
        volumes: list[np.ndarray] = []
        expected_shape = None

        for frame_dir in frame_dirs:
            files = sorted(
                list(frame_dir.glob("*.tif")) +
                list(frame_dir.glob("*.tiff")) +
                list(frame_dir.glob("*.png")),
                key=lambda p: natural_key(p.name),
            )
            if not files:
                print(f"[WARN] Skipping empty frame folder: {frame_dir}")
                continue

            volume = load_volume_from_files(files)
            if expected_shape is None:
                expected_shape = volume.shape
            elif volume.shape != expected_shape:
                raise RuntimeError(
                    f"{label} frame shape mismatch at {frame_dir.name}: "
                    f"{volume.shape} != {expected_shape}"
                )

            volumes.append(volume)

        if not volumes:
            raise RuntimeError(f"No valid {label} volumes loaded from frame folders.")

        data = np.stack(volumes, axis=0)
        print(f"[INFO] {label} final 4D shape = {data.shape}")
        return data

    files = collect_recursive_images(folder, ("*.tif", "*.tiff", "*.png"))
    if not files:
        raise RuntimeError(f"No readable image files found under {folder}")

    print(f"[INFO] {label} recursive file count = {len(files)}")
    volumes: list[np.ndarray] = []
    expected_shape = None
    for file_path in files:
        volume = load_volume_from_files([file_path])
        if expected_shape is None:
            expected_shape = volume.shape
        elif volume.shape != expected_shape:
            raise RuntimeError(
                f"{label} volume shape mismatch for {file_path.name}: "
                f"{volume.shape} != {expected_shape}"
            )
        volumes.append(volume)

    data = np.stack(volumes, axis=0)
    print(f"[INFO] {label} final 4D shape = {data.shape}")
    return data


def build_parser():
    parser = argparse.ArgumentParser(
        description="Play real and synth outputs inside one napari window on the same time axis."
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="/Users/wangyiding/CellUniverse/C++/output/🟣HL60_20260419.0638",
        help="Output directory that contains real/ and synth/ subfolders.",
    )
    parser.add_argument("--fps", type=float, default=2.0, help="Playback speed in frames per second.")
    parser.add_argument(
        "--layout",
        choices=("grid", "overlay"),
        default="grid",
        help="Use grid to show real and synth side by side, or overlay to stack them in one view.",
    )
    parser.add_argument("--real-colormap", default="gray", help="Napari colormap for real volume.")
    parser.add_argument("--synth-colormap", default="green", help="Napari colormap for synth volume.")
    return parser


def main():
    args = build_parser().parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    real_dir = base_dir / "real"
    synth_dir = base_dir / "synth"

    if not real_dir.is_dir():
        raise RuntimeError(f"Missing real directory: {real_dir}")
    if not synth_dir.is_dir():
        raise RuntimeError(f"Missing synth directory: {synth_dir}")

    real = load_time_series(real_dir, "real")
    synth = load_time_series(synth_dir, "synth")

    n_frames = min(real.shape[0], synth.shape[0])
    if real.shape[0] != synth.shape[0]:
        print(
            f"[WARN] Different frame counts: real={real.shape[0]}, "
            f"synth={synth.shape[0]}, using {n_frames}"
        )
    real = real[:n_frames]
    synth = synth[:n_frames]

    viewer = napari.Viewer(title=f"Real + Synth: {base_dir.name}")

    layer_real = viewer.add_image(
        real,
        name="real",
        rendering="mip",
        depiction="volume",
        colormap=args.real_colormap,
    )
    layer_synth = viewer.add_image(
        synth,
        name="synth",
        rendering="mip",
        depiction="volume",
        colormap=args.synth_colormap,
    )

    for layer in (layer_real, layer_synth):
        try:
            layer.interpolation = "linear"
        except Exception:
            pass

    viewer.dims.axis_labels = ("t", "z", "y", "x")
    viewer.dims.ndisplay = 3

    if args.layout == "grid":
        viewer.grid.enabled = True
        viewer.grid.stride = 2
    else:
        viewer.grid.enabled = False
        try:
            layer_synth.opacity = 0.55
        except Exception:
            pass

    interval_ms = max(1, int(1000 / max(args.fps, 0.1)))
    timer = QTimer()

    def step():
        t = int(viewer.dims.current_step[0])
        viewer.dims.set_point(0, (t + 1) % n_frames)

    timer.timeout.connect(step)
    timer.start(interval_ms)

    print(f"[INFO] base_dir = {base_dir}")
    print(f"[INFO] real shape  = {real.shape}")
    print(f"[INFO] synth shape = {synth.shape}")
    print(f"[INFO] autoplay = {args.fps} fps, frames = {n_frames}, layout = {args.layout}")

    napari.run()


if __name__ == "__main__":
    main()
