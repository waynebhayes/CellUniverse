#!/usr/bin/env python3
"""Render a four panel embryo showcase MP4 from verified TIFF and lineage data.

This script combines the old 4_Windows_Demo layout with the visual audit style
from napari_embryo_review.py, but renders offscreen with OpenCV. The goal is a
QuickTime safe MP4 without depending on an interactive napari window.
"""

from __future__ import annotations

import argparse
import csv
import colorsys
import math
import re
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import tifffile


DEFAULT_TIF_DIR = Path(
    "/Volumes/T9/🦠Cell Universe/🟣Output/Visualization/ALL_1~171_VISUAL_TIF"
)
DEFAULT_LINEAGE_CSV = Path(
    "/Volumes/T9/🦠Cell Universe/🟣Output/✅C.elegans_developing embryo_CorrectOutput 11.76G/"
    "Yiding_Embryo_1~171_FinalLineageTree.csv"
)
DEFAULT_GT_CSV = Path(
    "/Users/wangyiding/CellUniverse/C++/config/embryo/ground_truth/embryo_FixedGroundTruth.csv"
)
DEFAULT_OUTPUT = Path(
    "/Volumes/T9/🦠Cell Universe/🟣Output/Visualization/demo_videos/"
    "embryo_showcase_f001_f171.mp4"
)

ROOT_COLORS_BGR = [
    (255, 92, 214),  # purple
    (20, 255, 57),   # green
    (59, 212, 255),  # gold
    (255, 245, 0),   # cyan
]

ROOT_LABELS = [
    ("Purple", "P"),
    ("Green", "G"),
    ("Gold", "Y"),
    ("Cyan", "C"),
]


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int


@dataclass
class Node:
    name: str
    parent: str | None
    root: str
    code: str
    depth: int
    children: set[str] = field(default_factory=set)
    angle: float = 0.0
    radius: float = 0.0


@dataclass(frozen=True)
class TreeLayout:
    center: tuple[int, int]
    inner_radius: float
    outer_radius: float
    content_bottom: int
    stats_top: int
    first_frame: int
    last_frame: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a four panel CellUniverse embryo showcase MP4."
    )
    parser.add_argument("--tif-dir", type=Path, default=DEFAULT_TIF_DIR)
    parser.add_argument("--lineage-csv", type=Path, default=DEFAULT_LINEAGE_CSV)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--first-frame", type=int, default=1)
    parser.add_argument("--last-frame", type=int, default=171)
    parser.add_argument(
        "--render-only-frame",
        type=int,
        default=None,
        help="Render only one frame while still using first/last frame for lineage context.",
    )
    parser.add_argument(
        "--display-last-frame",
        type=int,
        default=None,
        help="Frame count shown in panel headers and lineage time rings.",
    )
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--projection", choices=("max", "mean", "middle"), default="max")
    parser.add_argument("--synth-opacity", type=float, default=0.12)
    parser.add_argument(
        "--review-render-scale",
        type=float,
        default=2.0,
        help="Render the large review overlay at a higher internal scale before fitting it into the video.",
    )
    parser.add_argument("--review-ring-opacity", type=float, default=0.62)
    parser.add_argument("--review-ring-scale", type=float, default=1.0)
    parser.add_argument("--review-rings-per-axis", type=int, default=2)
    parser.add_argument("--review-ring-segments", type=int, default=72)
    parser.add_argument("--review-ring-width", type=int, default=1)
    parser.add_argument("--review-center-size", type=int, default=3)
    parser.add_argument("--gt-square-size", type=int, default=5)
    parser.add_argument("--neighbor-distance", type=float, default=42.0)
    parser.add_argument("--green-levels", type=int, default=9)
    parser.add_argument("--max-tree-labels", type=int, default=360)
    parser.add_argument(
        "--tree-min-current-radius-fraction",
        type=float,
        default=0.66,
        help="Zoom early lineage frames so the current live ring fills at least this fraction of the tree radius.",
    )
    parser.add_argument(
        "--tree-max-zoom",
        type=float,
        default=18.0,
        help="Maximum dynamic zoom applied to the lineage tree panel.",
    )
    parser.add_argument(
        "--tree-stats-scale",
        type=float,
        default=0.60,
        help="Scale factor for the root family mitotic frequency box.",
    )
    parser.add_argument(
        "--disable-3d-rotation",
        action="store_true",
        help="Disable synchronized 3D yaw projection for the cell image panels.",
    )
    parser.add_argument(
        "--rotation-degrees",
        type=float,
        default=52.0,
        help="Maximum synchronized yaw angle used by the visual 3D turntable projection.",
    )
    parser.add_argument(
        "--rotation-cycle-frames",
        type=float,
        default=48.0,
        help="Number of frames for one smooth visual rotation cycle.",
    )
    parser.add_argument(
        "--rotation-slice-stride",
        type=int,
        default=1,
        help="Use every Nth z slice for rotated projections. Larger values are faster.",
    )
    parser.add_argument(
        "--rotation-depth-shift",
        type=float,
        default=96.0,
        help="Maximum x parallax in pixels between front and back z slices.",
    )
    parser.add_argument(
        "--rotation-pitch-degrees",
        type=float,
        default=24.0,
        help="Fixed oblique top view angle used to separate z depth vertically.",
    )
    parser.add_argument(
        "--rotation-pitch-shift",
        type=float,
        default=92.0,
        help="Maximum y parallax in pixels between front and back z slices for the oblique view.",
    )
    parser.add_argument("--crf", type=int, default=14)
    parser.add_argument("--preset", type=str, default="slow")
    parser.add_argument("--video-level", type=str, default="5.1")
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Render PNG frames only and skip MP4 encoding.",
    )
    parser.add_argument("--keep-frames", action="store_true", default=True)
    return parser.parse_args()


def natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def frame_number(frame_file: str) -> int | None:
    match = re.search(r"(\d+)", Path(frame_file).stem)
    return int(match.group(1)) if match else None


def lineage_parts(name: str) -> tuple[str, str] | None:
    if "_" in name:
        prefix, code = name.rsplit("_", 1)
        if code and not any(ch.isspace() for ch in code):
            return prefix + "_", code
    return None


def parent_name(name: str) -> str | None:
    parts = lineage_parts(name)
    if parts is not None:
        prefix, code = parts
        if len(code) <= 1 or code[-1] not in "01":
            return None
        return prefix + code[:-1]
    if len(name) > 1 and name[-1] in "01":
        return name[:-1]
    return None


def lineage_code(name: str) -> str:
    parts = lineage_parts(name)
    if parts is not None:
        return parts[1]
    return name


def root_name(name: str) -> str:
    parts = lineage_parts(name)
    if parts is not None:
        prefix, code = parts
        return prefix + code[:1]
    cur = name
    while True:
        parent = parent_name(cur)
        if parent is None:
            return cur
        cur = parent


def lineage_chain(name: str) -> list[str]:
    chain = [name]
    cur = name
    while True:
        parent = parent_name(cur)
        if parent is None:
            break
        chain.append(parent)
        cur = parent
    return chain


def ensure_node(nodes: dict[str, Node], name: str) -> Node:
    if name in nodes:
        return nodes[name]
    parent = parent_name(name)
    root = root_name(name)
    code = lineage_code(name)
    node = Node(name=name, parent=parent, root=root, code=code, depth=max(0, len(code) - 1))
    nodes[name] = node
    if parent is not None:
        parent_node = ensure_node(nodes, parent)
        parent_node.children.add(name)
        node.root = parent_node.root
        node.depth = parent_node.depth + 1
    return node


def build_nodes(names: Iterable[str]) -> dict[str, Node]:
    nodes: dict[str, Node] = {}
    for name in sorted(names, key=natural_key):
        ensure_node(nodes, name)
    return nodes


def sorted_roots(nodes: dict[str, Node]) -> list[str]:
    return sorted([name for name, node in nodes.items() if node.parent is None], key=lambda n: natural_key(lineage_code(n)))


def sorted_children(nodes: dict[str, Node], name: str) -> list[str]:
    return sorted(nodes[name].children, key=lambda n: natural_key(lineage_code(n)))


def collect_leaves(nodes: dict[str, Node], name: str) -> list[str]:
    children = sorted_children(nodes, name)
    if not children:
        return [name]
    leaves: list[str] = []
    for child in children:
        leaves.extend(collect_leaves(nodes, child))
    return leaves


def root_color(nodes: dict[str, Node], root_order: dict[str, int], name: str) -> tuple[int, int, int]:
    return ROOT_COLORS_BGR[root_order.get(nodes[name].root, 0) % len(ROOT_COLORS_BGR)]


def point(center: tuple[int, int], radius: float, angle: float) -> tuple[int, int]:
    return (
        int(round(center[0] + radius * math.cos(angle))),
        int(round(center[1] + radius * math.sin(angle))),
    )


def arc_points(center: tuple[int, int], radius: float, start_angle: float, end_angle: float) -> list[tuple[int, int]]:
    diff = (end_angle - start_angle + math.pi) % (2.0 * math.pi) - math.pi
    steps = max(3, int(abs(diff) * radius / 7.0))
    return [point(center, radius, start_angle + diff * step / steps) for step in range(steps + 1)]


def frame_radius(layout: TreeLayout, frame: int) -> float:
    if layout.last_frame <= layout.first_frame:
        return layout.outer_radius
    t = (frame - layout.first_frame) / (layout.last_frame - layout.first_frame)
    t = max(0.0, min(1.0, t))
    return layout.inner_radius * (1.0 - t) + layout.outer_radius * t


def tree_zoom_scale(
    layout: TreeLayout,
    current_frame: int,
    min_current_radius_fraction: float,
    max_zoom: float,
) -> float:
    raw_current_radius = max(1.0, frame_radius(layout, current_frame))
    target_current_radius = layout.outer_radius * max(
        0.05,
        min(0.95, min_current_radius_fraction),
    )
    requested = target_current_radius / raw_current_radius
    return max(1.0, min(max(1.0, max_zoom), requested))


def display_radius(raw_radius: float, zoom_scale: float) -> float:
    return raw_radius * zoom_scale


def assign_time_layout(nodes: dict[str, Node], birth_frames: dict[str, int], layout: TreeLayout) -> None:
    roots = sorted_roots(nodes)
    if not roots:
        return
    gap = math.radians(8.0)
    start = -math.pi / 2.0
    sector = (2.0 * math.pi) / len(roots)

    def assign_internal_angle(name: str) -> float:
        children = sorted_children(nodes, name)
        if not children:
            return nodes[name].angle
        child_angles = [assign_internal_angle(child) for child in children]
        sx = sum(math.cos(angle) for angle in child_angles)
        sy = sum(math.sin(angle) for angle in child_angles)
        nodes[name].angle = math.atan2(sy, sx)
        return nodes[name].angle

    for root_index, root in enumerate(roots):
        leaves = collect_leaves(nodes, root)
        a0 = start + root_index * sector + gap
        a1 = start + (root_index + 1) * sector - gap
        if len(leaves) == 1:
            nodes[leaves[0]].angle = 0.5 * (a0 + a1)
        else:
            for leaf_index, leaf in enumerate(leaves):
                t = leaf_index / (len(leaves) - 1)
                nodes[leaf].angle = a0 * (1.0 - t) + a1 * t
        assign_internal_angle(root)

    for node in nodes.values():
        node.radius = frame_radius(layout, birth_frames.get(node.name, layout.first_frame))


def read_lineage_csv(path: Path) -> tuple[dict[int, list[dict[str, str]]], dict[int, set[str]], dict[str, int]]:
    rows_by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    names_by_frame: dict[int, set[str]] = defaultdict(set)
    birth_frames: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            frame = frame_number(row.get("file", ""))
            name = row.get("name", "").strip()
            if frame is None or not name:
                continue
            rows_by_frame[frame].append(row)
            names_by_frame[frame].add(name)
            for lineage_name in lineage_chain(name):
                birth_frames.setdefault(lineage_name, frame)
    return dict(rows_by_frame), dict(names_by_frame), birth_frames


def read_gt(path: Path) -> dict[int, list[dict[str, str]]]:
    gt_by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            gt_by_frame[int(float(row["frame"]))].append(row)
    return dict(gt_by_frame)


def read_volume(path: Path) -> np.ndarray:
    # Prefer OpenCV for these generated TIFF stacks. On some local conda
    # installs, tifffile delegates DEFLATE pages to an imagecodecs binary that
    # can be incompatible with the active numpy build.
    ok, pages = cv2.imreadmulti(str(path), [], cv2.IMREAD_UNCHANGED)
    if ok and pages:
        gray_pages: list[np.ndarray] = []
        for page in pages:
            if page.ndim == 3:
                if page.shape[-1] == 4:
                    page = page[..., :3]
                page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
            gray_pages.append(page.astype(np.float32))
        if len(gray_pages) == 1:
            return gray_pages[0]
        return np.stack(gray_pages, axis=0)

    data = tifffile.imread(path)
    if data.ndim == 2:
        return data.astype(np.float32)
    if data.ndim == 3 and data.shape[-1] in (3, 4):
        rgb = data[..., :3].astype(np.float32)
        return cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    if data.ndim == 4 and data.shape[-1] in (3, 4):
        rgb = data[..., :3].astype(np.float32)
        return np.mean(rgb, axis=-1).astype(np.float32)
    return data.astype(np.float32)


def project_volume(volume: np.ndarray, mode: str) -> np.ndarray:
    if volume.ndim == 2:
        return volume.astype(np.float32)
    if mode == "middle":
        return volume[volume.shape[0] // 2].astype(np.float32)
    if mode == "mean":
        return np.mean(volume, axis=0).astype(np.float32)
    return np.max(volume, axis=0).astype(np.float32)


def rotation_angle_for_frame(frame: int, first_frame: int, args: argparse.Namespace) -> float:
    if args.disable_3d_rotation:
        return 0.0
    cycle = max(1.0, float(args.rotation_cycle_frames))
    phase = 2.0 * math.pi * (frame - first_frame) / cycle
    return float(args.rotation_degrees) * math.sin(phase)


def yaw_projection_values(
    volume_shape: tuple[int, ...] | None,
    angle_deg: float,
    args: argparse.Namespace,
) -> tuple[float, float, float, float] | None:
    if volume_shape is None or len(volume_shape) < 3 or args.disable_3d_rotation:
        return None
    yaw = math.radians(angle_deg)
    pitch = math.radians(float(args.rotation_pitch_degrees))
    scale_x = 1.0 - 0.18 * abs(math.sin(yaw))
    scale_y = 1.0 - 0.12 * abs(math.sin(pitch))
    max_shift = float(args.rotation_depth_shift) * math.sin(yaw)
    max_pitch_shift = float(args.rotation_pitch_shift) * math.sin(pitch)
    return scale_x, scale_y, max_shift, max_pitch_shift


def project_xyz_to_rotated_xy(
    points_xyz: np.ndarray,
    volume_shape: tuple[int, ...] | None,
    angle_deg: float,
    args: argparse.Namespace,
) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    xy = points[:, :2].copy()
    values = yaw_projection_values(volume_shape, angle_deg, args)
    if values is None or volume_shape is None or len(volume_shape) < 3:
        return xy
    scale_x, scale_y, max_shift, max_pitch_shift = values
    z_count, height, width = volume_shape[:3]
    center_x = (width - 1.0) * 0.5
    center_y = (height - 1.0) * 0.5
    center_z = (z_count - 1.0) * 0.5
    z_norm = (points[:, 2] - center_z) / max(1.0, center_z)
    xy[:, 0] = center_x + (points[:, 0] - center_x) * scale_x + z_norm * max_shift
    xy[:, 1] = center_y + (points[:, 1] - center_y) * scale_y + z_norm * max_pitch_shift
    return xy


def project_rotated_volume(
    volume: np.ndarray,
    mode: str,
    angle_deg: float,
    args: argparse.Namespace,
) -> np.ndarray:
    if volume.ndim == 2 or args.disable_3d_rotation or abs(angle_deg) < 1e-6:
        return project_volume(volume, mode)
    if mode == "middle":
        return project_volume(volume, mode)

    z_count, height, width = volume.shape[:3]
    values = yaw_projection_values(volume.shape, angle_deg, args)
    if values is None:
        return project_volume(volume, mode)
    scale_x, scale_y, max_shift, max_pitch_shift = values
    center_x = (width - 1.0) * 0.5
    center_y = (height - 1.0) * 0.5
    center_z = (z_count - 1.0) * 0.5
    stride = max(1, int(args.rotation_slice_stride))
    indices = list(range(0, z_count, stride))
    if (z_count - 1) not in indices:
        indices.append(z_count - 1)

    if mode == "mean":
        acc = np.zeros((height, width), dtype=np.float32)
    else:
        acc = np.zeros((height, width), dtype=np.float32)
    count = 0
    for zi in indices:
        z_norm = (zi - center_z) / max(1.0, center_z)
        matrix = np.asarray(
            [
                [scale_x, 0.0, center_x - scale_x * center_x + z_norm * max_shift],
                [0.0, scale_y, center_y - scale_y * center_y + z_norm * max_pitch_shift],
            ],
            dtype=np.float32,
        )
        warped = cv2.warpAffine(
            volume[zi].astype(np.float32),
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if mode == "mean":
            acc += warped
        else:
            np.maximum(acc, warped, out=acc)
        count += 1
    if mode == "mean" and count:
        acc /= float(count)
    return acc.astype(np.float32)


def compute_window(images: list[np.ndarray], low_q: float, high_q: float) -> tuple[float, float]:
    samples = [image.ravel()[::8] for image in images if image.size]
    flat = np.concatenate(samples) if samples else np.asarray([0.0, 1.0])
    low = float(np.percentile(flat, low_q))
    high = float(np.percentile(flat, high_q))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        low = float(flat.min())
        high = float(flat.max())
    if high <= low:
        high = low + 1.0
    return low, high


def normalize_to_u8(image: np.ndarray, low: float, high: float) -> np.ndarray:
    scaled = (np.clip(image, low, high) - low) / (high - low)
    return np.round(scaled * 255.0).astype(np.uint8)


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return b, g, r


def green_palette(color_count: int) -> list[str]:
    values = np.linspace(0.36, 0.78, max(3, color_count))
    colors: list[str] = []
    for value in values:
        red, green, blue = colorsys.hsv_to_rgb(0.35, 1.00, float(value))
        colors.append(f"#{int(255 * red):02x}{int(255 * green):02x}{int(255 * blue):02x}")
    return colors


def split_pair_palette() -> list[str]:
    return ["#5a00d6", "#002499", "#d4b300"]


def float_value(row: dict[str, str], key: str, fallback: float) -> float:
    try:
        return float(row.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def radius_value(row: dict[str, str], key: str, fallback: float) -> float:
    aliases = {
        "aRadius": ("aRadius", "majorRadius"),
        "bRadius": ("bRadius",),
        "cRadius": ("cRadius", "minorRadius"),
    }
    for candidate in aliases.get(key, (key,)):
        if candidate in row and row[candidate] not in ("", None):
            return max(1.0, float_value(row, candidate, fallback))
    return max(1.0, fallback)


def rotation_matrix(row: dict[str, str]) -> np.ndarray:
    tx = float_value(row, "theta_x", 0.0)
    ty = float_value(row, "theta_y", 0.0)
    tz = float_value(row, "theta_z", 0.0)
    cx, sx = math.cos(tx), math.sin(tx)
    cy, sy = math.cos(ty), math.sin(ty)
    cz, sz = math.cos(tz), math.sin(tz)
    return np.asarray(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=float,
    )


def infer_gt_split_groups(gt: list[dict[str, str]], frame: int) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for row in gt:
        parent = row.get("parent_label", "").strip()
        if parent in {"", "0", "-1", "nan", "None"}:
            continue
        try:
            start_frame = int(float(row.get("start_frame", "nan")))
        except ValueError:
            continue
        if start_frame == frame:
            groups.setdefault(parent, set()).add(row["label_id"])
    return groups


def flatten_groups(groups: dict[str, set[str]]) -> set[str]:
    labels: set[str] = set()
    for group_labels in groups.values():
        labels.update(group_labels)
    return labels


def predicted_split_groups_from_gt(
    pred: list[dict[str, str]],
    gt: list[dict[str, str]],
    split_groups: dict[str, set[str]],
    max_distance: float,
) -> dict[str, set[str]]:
    if not split_groups:
        return {}
    pred_points = {
        row["name"]: np.asarray([float(row["x"]), float(row["y"]), float(row["z"])])
        for row in pred
    }
    label_to_group: dict[str, str] = {}
    for group_name, labels in split_groups.items():
        for label in labels:
            label_to_group[label] = group_name
    split_pred: dict[str, set[str]] = {group_name: set() for group_name in split_groups}
    for gt_row in gt:
        group_name = label_to_group.get(gt_row["label_id"])
        if group_name is None:
            continue
        gt_point = np.asarray([float(gt_row["x"]), float(gt_row["y"]), float(gt_row["z_interp"])])
        best_name = ""
        best_dist = float("inf")
        for name, pred_point in pred_points.items():
            dist = float(np.linalg.norm(pred_point - gt_point))
            if dist < best_dist:
                best_name = name
                best_dist = dist
        if best_name and best_dist <= max_distance:
            split_pred[group_name].add(best_name)
    return split_pred


def build_neighbor_palette(rows: list[dict[str, str]], color_count: int, neighbor_distance: float) -> dict[str, int]:
    color_count = max(3, color_count)
    positions = {
        row["name"]: np.asarray([float(row["x"]), float(row["y"]), float(row["z"])])
        for row in rows
    }
    names = [row["name"] for row in rows]
    neighbors: dict[str, list[tuple[str, float]]] = {name: [] for name in names}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            dist = float(np.linalg.norm(positions[left] - positions[right]))
            if dist <= neighbor_distance:
                weight = max(0.05, 1.0 - dist / max(1.0, neighbor_distance))
                neighbors[left].append((right, weight))
                neighbors[right].append((left, weight))
    assignment: dict[str, int] = {}
    color_usage = [0] * color_count
    remaining = set(names)
    while remaining:
        name = max(
            remaining,
            key=lambda n: (
                sum(1 for other, _ in neighbors[n] if other in assignment),
                sum(weight for _, weight in neighbors[n]),
                n,
            ),
        )

        def penalty(idx: int) -> tuple[float, int]:
            value = 0.06 * color_usage[idx]
            for other, weight in neighbors[name]:
                if other in assignment:
                    value += weight * (4.0 if assignment[other] == idx else 0.0)
            return value, idx

        selected = min(range(color_count), key=penalty)
        assignment[name] = selected
        color_usage[selected] += 1
        remaining.remove(name)
    return assignment


def split_group_centroids(
    pred: list[dict[str, str]],
    gt: list[dict[str, str]],
    split_pred_groups: dict[str, set[str]],
    split_gt_groups: dict[str, set[str]],
) -> dict[str, np.ndarray]:
    pred_by_name = {
        row["name"]: np.asarray([float(row["x"]), float(row["y"]), float(row["z"])])
        for row in pred
    }
    gt_by_label = {
        row["label_id"]: np.asarray([float(row["x"]), float(row["y"]), float(row["z_interp"])])
        for row in gt
    }
    centroids: dict[str, np.ndarray] = {}
    for group_name in sorted(set(split_pred_groups) | set(split_gt_groups)):
        points = [
            pred_by_name[name]
            for name in split_pred_groups.get(group_name, set())
            if name in pred_by_name
        ]
        if not points:
            points = [
                gt_by_label[label]
                for label in split_gt_groups.get(group_name, set())
                if label in gt_by_label
            ]
        if points:
            centroids[group_name] = np.mean(np.asarray(points), axis=0)
    return centroids


def split_group_color_map(group_order: list[str], centroids: dict[str, np.ndarray], neighbor_distance: float) -> dict[str, str]:
    palette = split_pair_palette()
    if not group_order:
        return {}
    usage = [0] * len(palette)
    colors: dict[str, str] = {}
    for group in group_order:
        selected = min(range(len(palette)), key=lambda idx: (usage[idx], idx))
        colors[group] = palette[selected]
        usage[selected] += 1
    return colors


def make_wire_rings(
    rows: list[dict[str, str]],
    segments: int,
    radius_scale: float,
    rings_per_axis: int,
) -> dict[str, list[np.ndarray]]:
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=True)
    levels = max(1, rings_per_axis)
    offsets = [0.0] if levels == 1 else ([-0.38, 0.38] if levels == 2 else np.linspace(-0.62, 0.62, levels))
    rings_by_name: dict[str, list[np.ndarray]] = {}
    for row in rows:
        center_xyz = np.asarray([float(row["x"]), float(row["y"]), float(row["z"])])
        rx = radius_value(row, "aRadius", 10.0) * radius_scale
        ry = radius_value(row, "bRadius", rx) * radius_scale
        rz = radius_value(row, "cRadius", ry) * radius_scale
        rot = rotation_matrix(row)
        rings: list[np.ndarray] = []
        for offset in offsets:
            xy_scale = float(np.sqrt(max(0.0, 1.0 - float(offset) * float(offset))))
            rings.append(center_xyz + np.column_stack([
                rx * xy_scale * np.cos(angles),
                ry * xy_scale * np.sin(angles),
                np.full_like(angles, rz * float(offset)),
            ]) @ rot.T)
            xz_scale = float(np.sqrt(max(0.0, 1.0 - float(offset) * float(offset))))
            rings.append(center_xyz + np.column_stack([
                rx * xz_scale * np.cos(angles),
                np.full_like(angles, ry * float(offset)),
                rz * xz_scale * np.sin(angles),
            ]) @ rot.T)
            yz_scale = float(np.sqrt(max(0.0, 1.0 - float(offset) * float(offset))))
            rings.append(center_xyz + np.column_stack([
                np.full_like(angles, rx * float(offset)),
                ry * yz_scale * np.cos(angles),
                rz * yz_scale * np.sin(angles),
            ]) @ rot.T)
        rings_by_name[row["name"]] = rings
    return rings_by_name


def draw_panel_header(panel: np.ndarray, title: str, frame: int, last_frame: int) -> None:
    h, w = panel.shape[:2]
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (7, 7, 7), -1)
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (74, 74, 74), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (1, 1), (w - 2, 38), (18, 18, 18), -1)
    cv2.putText(panel, title, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
    frame_text = f"frame {frame:03d} / {last_frame:03d}"
    text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 1)[0]
    cv2.putText(panel, frame_text, (w - text_size[0] - 14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 1, cv2.LINE_AA)


def paste_fit(panel: np.ndarray, image_bgr: np.ndarray, content_rect: Rect) -> None:
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    src_h, src_w = image_bgr.shape[:2]
    scale = min(content_rect.w / src_w, content_rect.h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    interpolation = cv2.INTER_LANCZOS4 if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=interpolation)
    x = content_rect.x + (content_rect.w - new_w) // 2
    y = content_rect.y + (content_rect.h - new_h) // 2
    panel[y:y + new_h, x:x + new_w] = resized


def render_image_panel(image_bgr: np.ndarray, title: str, frame: int, last_frame: int, width: int, height: int) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    draw_panel_header(panel, title, frame, last_frame)
    paste_fit(panel, image_bgr, Rect(10, 48, width - 20, height - 60))
    return panel


def layout_rects(width: int, height: int) -> dict[str, Rect]:
    margin = 24
    gap = 18
    inner_w = width - margin * 2 - gap
    inner_h = height - margin * 2
    left_w = int(round(inner_w * 0.41))
    right_w = inner_w - left_w
    top_h = int(round(inner_h * 0.30))
    bottom_h = inner_h - top_h - gap
    small_w = (left_w - gap) // 2
    second_small_w = left_w - small_w - gap
    return {
        "real": Rect(margin, margin, small_w, top_h),
        "synth": Rect(margin + small_w + gap, margin, second_small_w, top_h),
        "tree": Rect(margin, margin + top_h + gap, left_w, bottom_h),
        "overlay": Rect(margin + left_w + gap, margin, right_w, inner_h),
    }


def paste_panel(canvas: np.ndarray, panel: np.ndarray, rect: Rect) -> None:
    if panel.shape[1] != rect.w or panel.shape[0] != rect.h:
        panel = cv2.resize(panel, (rect.w, rect.h), interpolation=cv2.INTER_AREA)
    canvas[rect.y:rect.y + rect.h, rect.x:rect.x + rect.w] = panel


def build_tree_layout(
    tree_rect: Rect,
    first_frame: int,
    last_frame: int,
    stats_scale: float,
) -> TreeLayout:
    stats_scale = max(0.35, min(1.0, stats_scale))
    stats_reserved = min(
        max(int(round(132 * stats_scale)), int(round(tree_rect.h * 0.14))),
        tree_rect.h // 4,
    )
    stats_top = tree_rect.h - stats_reserved
    content_top = 46
    content_bottom = max(content_top + 140, stats_top - 8)
    content_h = content_bottom - content_top
    center = (tree_rect.w // 2, content_top + int(round(content_h * 0.50)))
    label_margin = 18
    outer_radius = min((tree_rect.w - 2 * label_margin) * 0.50, (content_h - 2 * label_margin) * 0.50)
    outer_radius = max(80.0, outer_radius)
    inner_radius = max(14.0, outer_radius * 0.06)
    return TreeLayout(center, inner_radius, outer_radius, content_bottom, stats_top, first_frame, last_frame)


def draw_time_rings(
    panel: np.ndarray,
    layout: TreeLayout,
    current_frame: int,
    zoom_scale: float,
) -> None:
    for frame in range(layout.first_frame, layout.last_frame + 1):
        radius = int(round(display_radius(frame_radius(layout, frame), zoom_scale)))
        if radius > layout.outer_radius + 1:
            continue
        color = (48, 48, 48) if frame in {layout.first_frame, layout.last_frame} or frame % 20 == 0 else (19, 19, 19)
        if frame == current_frame:
            color = (68, 68, 68)
        cv2.circle(panel, layout.center, radius, color, 1, cv2.LINE_AA)


def draw_lineage_edge(
    panel: np.ndarray,
    nodes: dict[str, Node],
    root_order: dict[str, int],
    parent: str,
    child: str,
    layout: TreeLayout,
    thickness: int,
    zoom_scale: float,
) -> None:
    parent_node = nodes[parent]
    child_node = nodes[child]
    color = root_color(nodes, root_order, child)
    parent_radius = display_radius(parent_node.radius, zoom_scale)
    child_radius = display_radius(child_node.radius, zoom_scale)
    p0 = point(layout.center, parent_radius, parent_node.angle)
    elbow = point(layout.center, child_radius, parent_node.angle)
    arc = arc_points(layout.center, child_radius, parent_node.angle, child_node.angle)
    pts = np.array([p0, elbow, *arc[1:]], dtype=np.int32)
    cv2.polylines(panel, [pts], False, color, thickness, cv2.LINE_AA)
    cv2.circle(panel, point(layout.center, child_radius, child_node.angle), max(2, thickness), color, -1, cv2.LINE_AA)


def draw_active_life_lines(
    panel: np.ndarray,
    nodes: dict[str, Node],
    root_order: dict[str, int],
    active_names: set[str],
    current_frame: int,
    layout: TreeLayout,
    thickness: int,
    zoom_scale: float,
) -> None:
    current_radius = display_radius(frame_radius(layout, current_frame), zoom_scale)
    for name in sorted(active_names, key=natural_key):
        if name not in nodes:
            continue
        node = nodes[name]
        color = root_color(nodes, root_order, name)
        node_radius = display_radius(node.radius, zoom_scale)
        p0 = point(layout.center, min(node_radius, current_radius), node.angle)
        p1 = point(layout.center, current_radius, node.angle)
        cv2.line(panel, p0, p1, color, thickness, cv2.LINE_AA)
        cv2.circle(panel, p1, max(3, thickness + 1), color, -1, cv2.LINE_AA)


def family_stats_rows(nodes: dict[str, Node], root_order: dict[str, int], birth_frames: dict[str, int], active_names: set[str], current_frame: int, first_frame: int) -> list[tuple[tuple[int, int, int], str]]:
    active_counts: dict[str, int] = defaultdict(int)
    for name in active_names:
        if name in nodes:
            active_counts[nodes[name].root] += 1
    division_counts: dict[str, int] = defaultdict(int)
    for node in nodes.values():
        born_children = [child for child in node.children if birth_frames.get(child, 10**9) <= current_frame]
        if len(born_children) >= 2 and birth_frames.get(node.name, first_frame) <= current_frame:
            division_counts[node.root] += 1
    elapsed_frames = max(1, current_frame - first_frame + 1)
    rows: list[tuple[tuple[int, int, int], str]] = []
    for root in sorted_roots(nodes)[:4]:
        index = root_order[root] % len(ROOT_LABELS)
        full_name, compact_name = ROOT_LABELS[index]
        cells = active_counts.get(root, 0)
        divisions = division_counts.get(root, 0)
        rate = divisions * 10.0 / elapsed_frames
        text = f"{compact_name} {full_name}: cells {cells}  div {divisions}  {rate:.1f}/10f"
        rows.append((ROOT_COLORS_BGR[index], text))
    return rows


def draw_color_stats_box(
    panel: np.ndarray,
    rows: list[tuple[tuple[int, int, int], str]],
    origin: tuple[int, int],
    scale: float,
) -> None:
    if not rows:
        return
    scale = max(0.35, min(1.0, scale))
    font = cv2.FONT_HERSHEY_SIMPLEX
    title = "Root family mitotic frequency"
    title_scale = 0.42 * scale
    row_scale = 0.36 * scale
    row_step = max(13, int(round(22 * scale)))
    title_y = int(round(24 * scale))
    first_row_y = int(round(48 * scale))
    max_text_w = max(cv2.getTextSize(text, font, row_scale, 1)[0][0] for _, text in rows)
    panel_w = min(panel.shape[1] - 24, max(int(round(300 * scale)), max_text_w + int(round(50 * scale))))
    panel_h = int(round(38 * scale)) + row_step * len(rows) + int(round(10 * scale))
    x, y = origin
    cv2.rectangle(panel, (x, y), (x + panel_w, y + panel_h), (6, 6, 6), -1, cv2.LINE_AA)
    cv2.rectangle(panel, (x, y), (x + panel_w, y + panel_h), (72, 72, 72), 1, cv2.LINE_AA)
    cv2.putText(panel, title, (x + 10, y + title_y), font, title_scale, (220, 220, 220), 1, cv2.LINE_AA)
    cursor_y = y + first_row_y
    for color, text in rows:
        cv2.line(
            panel,
            (x + 10, cursor_y - 4),
            (x + 10 + int(round(18 * scale)), cursor_y - 4),
            color,
            max(1, int(round(3 * scale))),
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            text,
            (x + 10 + int(round(26 * scale)), cursor_y),
            font,
            row_scale,
            (224, 224, 224),
            1,
            cv2.LINE_AA,
        )
        cursor_y += row_step


def draw_text_box(panel: np.ndarray, lines: list[str], origin: tuple[int, int], align_right: bool = False, font_scale: float = 0.44) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(line, font, font_scale, 1)[0] for line in lines]
    panel_w = max(width for width, _ in sizes) + 24
    panel_h = sum(height for _, height in sizes) + 16 + 8 * (len(lines) - 1)
    x, y = origin
    if align_right:
        x -= panel_w
    cv2.rectangle(panel, (x, y), (x + panel_w, y + panel_h), (8, 8, 8), -1, cv2.LINE_AA)
    cv2.rectangle(panel, (x, y), (x + panel_w, y + panel_h), (70, 70, 70), 1, cv2.LINE_AA)
    cursor_y = y + 9
    for line, (_, height) in zip(lines, sizes):
        cursor_y += height
        cv2.putText(panel, line, (x + 12, cursor_y), font, font_scale, (225, 225, 225), 1, cv2.LINE_AA)
        cursor_y += 8


def build_label_map(nodes: dict[str, Node], root_order: dict[str, int]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for name, node in nodes.items():
        compact = ROOT_LABELS[root_order.get(node.root, 0) % len(ROOT_LABELS)][1]
        labels[name] = compact + lineage_code(name)
    return labels


def draw_node_labels_on_nodes(
    panel: np.ndarray,
    nodes: dict[str, Node],
    root_order: dict[str, int],
    active_names: set[str],
    birth_frames: dict[str, int],
    label_by_name: dict[str, str],
    current_frame: int,
    layout: TreeLayout,
    max_labels: int,
    zoom_scale: float,
) -> None:
    if max_labels <= 0:
        return
    names = [
        name
        for name in nodes
        if birth_frames.get(name, 10**9) <= current_frame
        and (name in active_names or any(birth_frames.get(child, 10**9) <= current_frame for child in nodes[name].children))
    ]
    if len(names) > max_labels:
        live = [name for name in names if name in active_names]
        branch = [name for name in names if name not in active_names]
        keep_branch_count = max(0, max_labels - len(live))
        step = max(1, math.ceil(len(branch) / max(1, keep_branch_count))) if keep_branch_count else len(branch) + 1
        names = live + branch[::step]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.30 if len(names) < 90 else (0.25 if len(names) < 180 else 0.21)
    thickness = 1
    current_radius = display_radius(frame_radius(layout, current_frame), zoom_scale)
    for name in sorted(names, key=lambda n: (nodes[n].angle, nodes[n].radius, n)):
        node = nodes[name]
        color = root_color(nodes, root_order, name)
        radius = current_radius if name in active_names else display_radius(node.radius, zoom_scale)
        x, y = point(layout.center, radius, node.angle)
        label = label_by_name.get(name, lineage_code(name))
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        pad_x = 3
        pad_y = 2
        x0 = int(round(x - tw / 2 - pad_x))
        y0 = int(round(y - th / 2 - pad_y))
        x0 = max(4, min(panel.shape[1] - tw - 2 * pad_x - 4, x0))
        y0 = max(42, min(layout.stats_top - th - 2 * pad_y - 4, y0))
        x1 = x0 + tw + 2 * pad_x
        y1 = y0 + th + 2 * pad_y + baseline
        cv2.rectangle(panel, (x0, y0), (x1, y1), (5, 5, 5), -1, cv2.LINE_AA)
        cv2.rectangle(panel, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        cv2.putText(panel, label, (x0 + pad_x, y0 + pad_y + th), font, font_scale, (232, 232, 232), thickness, cv2.LINE_AA)


def render_lineage_tree_panel(
    nodes: dict[str, Node],
    birth_frames: dict[str, int],
    active_names: set[str],
    label_by_name: dict[str, str],
    frame: int,
    last_frame: int,
    layout: TreeLayout,
    width: int,
    height: int,
    max_labels: int,
    min_current_radius_fraction: float,
    max_zoom: float,
    stats_scale: float,
) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    draw_panel_header(panel, "Lineage Tree", frame, last_frame)
    root_order = {root: index for index, root in enumerate(sorted_roots(nodes))}
    thickness = max(1, min(width, height) // 360)
    zoom_scale = tree_zoom_scale(
        layout,
        frame,
        min_current_radius_fraction,
        max_zoom,
    )
    draw_time_rings(panel, layout, frame, zoom_scale)
    for name in sorted(nodes, key=lambda n: (birth_frames.get(n, 10**9), natural_key(lineage_code(n)))):
        node = nodes[name]
        if node.parent is None or birth_frames.get(name, 10**9) > frame:
            continue
        draw_lineage_edge(panel, nodes, root_order, node.parent, name, layout, thickness, zoom_scale)
    draw_active_life_lines(panel, nodes, root_order, active_names, frame, layout, thickness, zoom_scale)
    for name, node in nodes.items():
        if birth_frames.get(name, 10**9) > frame:
            continue
        color = root_color(nodes, root_order, name)
        cv2.circle(
            panel,
            point(layout.center, display_radius(node.radius, zoom_scale), node.angle),
            max(2, thickness),
            color,
            -1,
            cv2.LINE_AA,
        )
    draw_text_box(panel, [f"Live cells  {len(active_names)}"], (width - 14, 52), align_right=True)
    rows = family_stats_rows(nodes, root_order, birth_frames, active_names, frame, layout.first_frame)
    stats_h = int(round(38 * max(0.35, min(1.0, stats_scale)))) + max(13, int(round(22 * max(0.35, min(1.0, stats_scale))))) * len(rows) + int(round(10 * max(0.35, min(1.0, stats_scale))))
    draw_color_stats_box(panel, rows, (14, height - stats_h - 12), stats_scale)
    return panel


def make_review_overlay(
    real_u8: np.ndarray,
    synth_u8: np.ndarray,
    pred: list[dict[str, str]],
    gt: list[dict[str, str]],
    frame: int,
    args: argparse.Namespace,
    volume_shape: tuple[int, ...] | None,
    rotation_angle_deg: float,
) -> np.ndarray:
    render_scale = max(1.0, float(args.review_render_scale))
    if render_scale > 1.001:
        source_h, source_w = real_u8.shape[:2]
        render_size = (
            max(1, int(round(source_w * render_scale))),
            max(1, int(round(source_h * render_scale))),
        )
        real_draw = cv2.resize(real_u8, render_size, interpolation=cv2.INTER_LANCZOS4)
        synth_draw = cv2.resize(synth_u8, render_size, interpolation=cv2.INTER_LANCZOS4)
    else:
        real_draw = real_u8
        synth_draw = synth_u8

    overlay = cv2.cvtColor(real_draw, cv2.COLOR_GRAY2BGR).astype(np.float32)
    synth_color = np.zeros_like(overlay)
    synth_color[:, :, 0] = 255.0
    synth_color[:, :, 1] = 255.0
    alpha = np.clip(args.synth_opacity, 0.0, 1.0) * (synth_draw.astype(np.float32) / 255.0)
    overlay = overlay * (1.0 - alpha[:, :, None]) + synth_color * alpha[:, :, None]
    canvas = np.clip(overlay, 0, 255).astype(np.uint8)

    split_groups = infer_gt_split_groups(gt, frame)
    split_labels = flatten_groups(split_groups)
    split_pred_groups = predicted_split_groups_from_gt(pred, gt, split_groups, 22.0)
    split_pred_names = flatten_groups(split_pred_groups)
    greens = green_palette(max(3, args.green_levels))
    green_assignments = build_neighbor_palette(pred, len(greens), max(1.0, args.neighbor_distance))
    split_group_order = sorted(set(split_groups) | set(split_pred_groups))
    split_group_colors = split_group_color_map(
        split_group_order,
        split_group_centroids(pred, gt, split_pred_groups, split_groups),
        max(1.0, args.neighbor_distance * 1.6),
    )
    rings_by_name = make_wire_rings(
        pred,
        max(12, args.review_ring_segments),
        max(0.1, args.review_ring_scale),
        max(1, args.review_rings_per_axis),
    )

    ring_layer = np.zeros_like(canvas)
    ring_alpha = np.zeros(canvas.shape[:2], dtype=np.float32)

    def draw_ring(
        ring_xyz: np.ndarray,
        color: tuple[int, int, int],
        width: int,
        opacity: float,
    ) -> None:
        pts = np.round(
            project_xyz_to_rotated_xy(ring_xyz, volume_shape, rotation_angle_deg, args) * render_scale
        ).astype(np.int32)
        tmp = np.zeros_like(canvas)
        cv2.polylines(tmp, [pts], False, color, max(1, width), cv2.LINE_AA)
        intensity = tmp.max(axis=2).astype(np.float32) / 255.0
        candidate_alpha = intensity * np.clip(opacity, 0.0, 1.0)
        replace = candidate_alpha > ring_alpha
        if np.any(replace):
            ring_alpha[replace] = candidate_alpha[replace]
            ring_layer[replace] = tmp[replace]

    for row in pred:
        name = row["name"]
        if name in split_pred_names:
            continue
        color = hex_to_bgr(greens[green_assignments.get(name, 0)])
        for ring in rings_by_name.get(name, []):
            draw_ring(ring, color, args.review_ring_width, args.review_ring_opacity)

    for group_name, names in split_pred_groups.items():
        color = hex_to_bgr(split_group_colors.get(group_name, "#7a1fff"))
        for name in names:
            for ring in rings_by_name.get(name, []):
                draw_ring(
                    ring,
                    color,
                    args.review_ring_width,
                    min(0.52, args.review_ring_opacity + 0.12),
                )

    if np.any(ring_alpha > 0.0):
        alpha_map = ring_alpha[:, :, None]
        canvas = np.clip(
            canvas.astype(np.float32) * (1.0 - alpha_map)
            + ring_layer.astype(np.float32) * alpha_map,
            0,
            255,
        ).astype(np.uint8)

    for row in pred:
        name = row["name"]
        center_xy = project_xyz_to_rotated_xy(
            np.asarray([float(row["x"]), float(row["y"]), float(row["z"])]),
            volume_shape,
            rotation_angle_deg,
            args,
        )[0] * render_scale
        x = int(round(float(center_xy[0])))
        y = int(round(float(center_xy[1])))
        if not (0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]):
            continue
        if name in split_pred_names:
            group = next((group_name for group_name, names in split_pred_groups.items() if name in names), "")
            color = hex_to_bgr(split_group_colors.get(group, "#7a1fff"))
        else:
            color = hex_to_bgr(greens[green_assignments.get(name, 0)])
        cv2.circle(
            canvas,
            (x, y),
            max(1, int(round(args.review_center_size * render_scale))),
            color,
            -1,
            cv2.LINE_AA,
        )

    gt_square = max(2, int(round(args.gt_square_size * render_scale)))
    for row in gt:
        gt_xy = project_xyz_to_rotated_xy(
            np.asarray([float(row["x"]), float(row["y"]), float(row["z_interp"])]),
            volume_shape,
            rotation_angle_deg,
            args,
        )[0] * render_scale
        x = int(round(float(gt_xy[0])))
        y = int(round(float(gt_xy[1])))
        if not (0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]):
            continue
        half = gt_square // 2
        color = (0, 0, 255)
        if row["label_id"] in split_labels:
            color = (0, 80, 255)
        cv2.rectangle(canvas, (x - half, y - half), (x + half, y + half), color, -1, cv2.LINE_AA)
    return canvas


def make_synth_over_real(real_u8: np.ndarray, synth_u8: np.ndarray, opacity: float) -> np.ndarray:
    real_bgr = cv2.cvtColor(real_u8, cv2.COLOR_GRAY2BGR).astype(np.float32)
    synth_bgr = cv2.applyColorMap(synth_u8, cv2.COLORMAP_VIRIDIS).astype(np.float32)
    alpha = np.clip(opacity, 0.0, 1.0) * (synth_u8.astype(np.float32) / 255.0)
    overlay = real_bgr * (1.0 - alpha[:, :, None]) + synth_bgr * alpha[:, :, None]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def compose_frame(
    real_u8: np.ndarray,
    synth_u8: np.ndarray,
    review_overlay_bgr: np.ndarray,
    nodes: dict[str, Node],
    birth_frames: dict[str, int],
    active_names: set[str],
    label_by_name: dict[str, str],
    frame: int,
    args: argparse.Namespace,
    rects: dict[str, Rect],
    tree_layout: TreeLayout,
) -> np.ndarray:
    canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    canvas[:] = (3, 3, 3)
    display_last_frame = args.display_last_frame or args.last_frame
    real_panel = render_image_panel(cv2.cvtColor(real_u8, cv2.COLOR_GRAY2BGR), "Real Image", frame, display_last_frame, rects["real"].w, rects["real"].h)
    synth_panel = render_image_panel(cv2.applyColorMap(synth_u8, cv2.COLORMAP_VIRIDIS), "Synth viridis", frame, display_last_frame, rects["synth"].w, rects["synth"].h)
    overlay_panel = render_image_panel(review_overlay_bgr, "Review over Real   rings + centers + fixed GT", frame, display_last_frame, rects["overlay"].w, rects["overlay"].h)
    tree_panel = render_lineage_tree_panel(
        nodes,
        birth_frames,
        active_names,
        label_by_name,
        frame,
        display_last_frame,
        tree_layout,
        rects["tree"].w,
        rects["tree"].h,
        args.max_tree_labels,
        args.tree_min_current_radius_fraction,
        args.tree_max_zoom,
        args.tree_stats_scale,
    )
    paste_panel(canvas, real_panel, rects["real"])
    paste_panel(canvas, synth_panel, rects["synth"])
    paste_panel(canvas, overlay_panel, rects["overlay"])
    paste_panel(canvas, tree_panel, rects["tree"])
    return canvas


def encode_h264_mp4(
    frame_dir: Path,
    output_path: Path,
    fps: float,
    first_frame: int,
    crf: int,
    preset: str,
    video_level: str,
) -> None:
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found")
    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        str(first_frame),
        "-i",
        str(frame_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-level",
        str(video_level),
        "-tag:v",
        "avc1",
        "-movflags",
        "+faststart",
        "-crf",
        str(crf),
        "-preset",
        str(preset),
        str(output_path),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg failed")


def check_video_not_green(output_path: Path) -> tuple[bool, str]:
    cap = cv2.VideoCapture(str(output_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False, "opencv could not read first encoded frame"
    channel_mean = frame.reshape(-1, 3).mean(axis=0)
    green_dominance = float(channel_mean[1] - max(channel_mean[0], channel_mean[2]))
    message = "first_frame_bgr_mean=({:.2f},{:.2f},{:.2f}) green_dominance={:.2f}".format(
        channel_mean[0], channel_mean[1], channel_mean[2], green_dominance
    )
    return green_dominance < 45.0, message


def main() -> int:
    args = parse_args()
    if args.last_frame < args.first_frame:
        raise ValueError("--last-frame must be >= --first-frame")
    if args.render_only_frame is not None and not (args.first_frame <= args.render_only_frame <= args.last_frame):
        raise ValueError("--render-only-frame must be inside --first-frame and --last-frame")
    if args.display_last_frame is not None and args.display_last_frame < args.last_frame:
        raise ValueError("--display-last-frame must be >= --last-frame")
    if args.width % 2 or args.height % 2:
        raise ValueError("--width and --height must be even for yuv420p MP4")
    tif_dir = args.tif_dir.expanduser().resolve()
    lineage_csv = args.lineage_csv.expanduser().resolve()
    gt_csv = args.gt.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output_path.parent / f"{output_path.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    render_frames = (
        [args.render_only_frame]
        if args.render_only_frame is not None
        else list(range(args.first_frame, args.last_frame + 1))
    )

    rows_by_frame, names_by_frame, birth_frames = read_lineage_csv(lineage_csv)
    gt_by_frame = read_gt(gt_csv)
    visible_names: set[str] = set()
    for frame, names in names_by_frame.items():
        if args.first_frame <= frame <= args.last_frame:
            visible_names.update(names)
    if not visible_names:
        raise RuntimeError("No lineage rows were found for the requested frame range")
    nodes = build_nodes(visible_names)
    rects = layout_rects(args.width, args.height)
    display_last_frame = args.display_last_frame or args.last_frame
    tree_layout = build_tree_layout(rects["tree"], args.first_frame, display_last_frame, args.tree_stats_scale)
    assign_time_layout(nodes, birth_frames, tree_layout)
    root_order = {root: index for index, root in enumerate(sorted_roots(nodes))}
    label_by_name = build_label_map(nodes, root_order)

    real_proj: dict[int, np.ndarray] = {}
    synth_proj: dict[int, np.ndarray] = {}
    volume_shapes: dict[int, tuple[int, ...]] = {}
    rotation_angles: dict[int, float] = {}
    for frame in render_frames:
        real_path = tif_dir / f"{frame}_real.tif"
        synth_path = tif_dir / f"{frame}_synth.tif"
        if not real_path.is_file() or not synth_path.is_file():
            raise FileNotFoundError(f"Missing TIFF pair for frame {frame}: {real_path} / {synth_path}")
        print(f"[LOAD] frame {frame:03d}")
        angle_deg = rotation_angle_for_frame(frame, args.first_frame, args)
        real_volume = read_volume(real_path)
        synth_volume = read_volume(synth_path)
        volume_shapes[frame] = tuple(real_volume.shape)
        rotation_angles[frame] = angle_deg
        real_proj[frame] = project_rotated_volume(real_volume, args.projection, angle_deg, args)
        synth_proj[frame] = project_rotated_volume(synth_volume, args.projection, angle_deg, args)

    real_low, real_high = compute_window(list(real_proj.values()), 1.0, 99.8)
    synth_low, synth_high = compute_window(list(synth_proj.values()), 1.0, 99.8)
    print(f"[INFO] real window {real_low:.3f} to {real_high:.3f}")
    print(f"[INFO] synth window {synth_low:.3f} to {synth_high:.3f}")

    last_active_names: set[str] = set()
    for frame in render_frames:
        if frame in names_by_frame:
            last_active_names = set(names_by_frame[frame])
        real_u8 = normalize_to_u8(real_proj[frame], real_low, real_high)
        synth_u8 = normalize_to_u8(synth_proj[frame], synth_low, synth_high)
        review_overlay = make_review_overlay(
            real_u8,
            synth_u8,
            rows_by_frame.get(frame, []),
            gt_by_frame.get(frame, []),
            frame,
            args,
            volume_shapes.get(frame),
            rotation_angles.get(frame, 0.0),
        )
        canvas = compose_frame(
            real_u8,
            synth_u8,
            review_overlay,
            nodes,
            birth_frames,
            last_active_names,
            label_by_name,
            frame,
            args,
            rects,
            tree_layout,
        )
        frame_path = frame_dir / f"frame_{frame:04d}.png"
        if not cv2.imwrite(str(frame_path), canvas):
            raise IOError(f"Failed to write {frame_path}")
        print(f"[WRITE] {frame_path}")

    if args.png_only:
        print(f"[DONE] PNG only mode, skipped MP4 encoding for {output_path}")
    else:
        encode_h264_mp4(
            frame_dir,
            output_path,
            args.fps,
            args.first_frame,
            args.crf,
            args.preset,
            args.video_level,
        )
        ok, message = check_video_not_green(output_path)
        print(f"[CHECK] {message}")
        if not ok:
            raise RuntimeError("Encoded MP4 failed the green-frame sanity check")
        print(f"[DONE] {output_path}")
    if args.keep_frames:
        print(f"[INFO] kept PNG frames in {frame_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
