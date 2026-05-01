#!/usr/bin/env python3
"""Create a growing radial lineage-tree GIF/MP4 from CellUniverse cells.csv files."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_CSV_GLOB = "C++/output/*embryo1~84*/*/*.csv"


ROOT_COLORS_BGR = [
    (255, 92, 214),  # bright purple, RGB #D65CFF
    (20, 255, 57),   # bright green,  RGB #39FF14
    (59, 212, 255),  # gold,          RGB #FFD43B
    (255, 245, 0),   # cyan,          RGB #00F5FF
]

ROOT_LABELS = [
    ("Purple", "P"),
    ("Green", "G"),
    ("Gold", "Y"),
    ("Cyan", "C"),
]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a black-background radial lineage-tree animation from CellUniverse cells.csv files."
    )
    parser.add_argument(
        "--csv",
        action="append",
        dest="csv_paths",
        help="Path to a cells.csv file. Repeat for multiple segments. Defaults to scanning C++/output.",
    )
    parser.add_argument("--output-dir", default="C++/output/lineage_tree_demo")
    parser.add_argument("--gif", default="lineage_tree_demo.gif")
    parser.add_argument("--mp4", default="lineage_tree_demo.mp4")
    parser.add_argument("--first-frame", type=int, default=1)
    parser.add_argument("--last-frame", type=int, default=84)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--size", type=int, default=1400)
    parser.add_argument(
        "--transition-frames",
        type=int,
        default=5,
        help="Subframes used to grow newly appearing branches.",
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=4,
        help="Subframes rendered for biological frames with no new lineage nodes.",
    )
    parser.add_argument(
        "--label-style",
        choices=("compact", "full", "none"),
        default="compact",
        help="Tip label format. compact uses P1/G1/Y1/C1; full uses Purple 1, etc.",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=220,
        help="Maximum labels drawn to preserve tree clarity.",
    )
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def discover_default_csvs(repo_root: Path) -> list[Path]:
    csvs = [
        path
        for path in repo_root.glob(DEFAULT_CSV_GLOB)
        if "cell" in path.name.lower()
    ]
    return sorted(csvs, key=lambda p: natural_key(str(p)))


def lineage_parts(name: str) -> tuple[str, str] | None:
    """Return (prefix, lineage_code). For embryo names the code follows the last underscore."""
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


def frame_number(frame_file: str) -> int | None:
    match = re.search(r"(\d+)", Path(frame_file).stem)
    return int(match.group(1)) if match else None


def read_frame_names(csv_paths: Iterable[Path]) -> dict[int, set[str]]:
    frame_names: dict[int, set[str]] = defaultdict(set)
    for csv_path in csv_paths:
        csv_frame_names: dict[int, set[str]] = defaultdict(set)
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                frame = frame_number(row.get("file", ""))
                name = row.get("name", "").strip()
                if frame is None or not name:
                    continue
                csv_frame_names[frame].add(name)
        for frame, names in csv_frame_names.items():
            frame_names[frame] = names
    return dict(frame_names)


def filter_frame_names(
    frame_names: dict[int, set[str]],
    first_frame: int,
    last_frame: int,
) -> dict[int, set[str]]:
    return {
        frame: names
        for frame, names in frame_names.items()
        if first_frame <= frame <= last_frame
    }


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


def build_birth_frames(frame_names: dict[int, set[str]]) -> dict[str, int]:
    birth_frames: dict[str, int] = {}
    for frame in sorted(frame_names):
        for name in frame_names[frame]:
            for lineage_name in lineage_chain(name):
                if lineage_name not in birth_frames:
                    birth_frames[lineage_name] = frame
    return birth_frames


def build_label_map(
    birth_frames: dict[str, int],
    label_style: str,
) -> dict[str, str]:
    if label_style == "none":
        return {}

    roots = sorted({root_name(name) for name in birth_frames}, key=lambda n: natural_key(lineage_code(n)))
    root_index = {root: idx for idx, root in enumerate(roots)}
    names_by_root: dict[str, list[str]] = defaultdict(list)
    for name in birth_frames:
        names_by_root[root_name(name)].append(name)

    labels: dict[str, str] = {}
    for root in roots:
        idx = root_index[root]
        full_name, compact_name = ROOT_LABELS[idx % len(ROOT_LABELS)]
        ordered_names = sorted(
            names_by_root[root],
            key=lambda n: (birth_frames[n], natural_key(lineage_code(n)), n),
        )
        for order, name in enumerate(ordered_names, start=1):
            if label_style == "full":
                labels[name] = f"{full_name} {order}"
            else:
                labels[name] = f"{compact_name}{order}"
    return labels


def ensure_node(nodes: dict[str, Node], name: str) -> Node:
    if name in nodes:
        return nodes[name]

    parent = parent_name(name)
    root = root_name(name)
    code = lineage_code(name)
    depth = max(0, len(code) - 1)
    node = Node(name=name, parent=parent, root=root, code=code, depth=depth)
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
    roots = [name for name, node in nodes.items() if node.parent is None]
    return sorted(roots, key=lambda n: natural_key(lineage_code(n)))


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


def assign_layout(nodes: dict[str, Node], size: int) -> None:
    if not nodes:
        return

    roots = sorted_roots(nodes)
    if not roots:
        return

    max_depth = max(node.depth for node in nodes.values())
    inner_radius = size * 0.085
    outer_radius = size * 0.455
    depth_step = (outer_radius - inner_radius) / max(1, max_depth)

    gap = math.radians(7.0)
    start = -math.pi / 2.0
    sector = (2.0 * math.pi) / len(roots)

    def assign_internal_angle(name: str) -> float:
        children = sorted_children(nodes, name)
        if not children:
            return nodes[name].angle
        child_angles = [assign_internal_angle(child) for child in children]
        sx = sum(math.cos(a) for a in child_angles)
        sy = sum(math.sin(a) for a in child_angles)
        nodes[name].angle = math.atan2(sy, sx)
        return nodes[name].angle

    for root_index, root in enumerate(roots):
        leaves = collect_leaves(nodes, root)
        a0 = start + root_index * sector + gap
        a1 = start + (root_index + 1) * sector - gap
        if len(leaves) == 1:
            nodes[leaves[0]].angle = 0.5 * (a0 + a1)
        else:
            for i, leaf in enumerate(leaves):
                t = i / (len(leaves) - 1)
                nodes[leaf].angle = a0 * (1.0 - t) + a1 * t
        assign_internal_angle(root)

    for node in nodes.values():
        node.radius = inner_radius + depth_step * node.depth


def layout_nodes(visible_names: set[str], size: int) -> dict[str, Node]:
    nodes = build_nodes(visible_names)
    assign_layout(nodes, size)
    return nodes


def interpolate_angle(start: float, end: float, progress: float) -> float:
    diff = (end - start + math.pi) % (2.0 * math.pi) - math.pi
    return start + diff * progress


def interpolate_nodes(
    previous_nodes: dict[str, Node] | None,
    current_nodes: dict[str, Node],
    progress: float,
) -> dict[str, Node]:
    if previous_nodes is None:
        return deepcopy(current_nodes)

    interpolated: dict[str, Node] = {}
    for name, current in current_nodes.items():
        node = deepcopy(current)
        previous = previous_nodes.get(name)
        if previous is not None:
            node.angle = interpolate_angle(previous.angle, current.angle, progress)
            node.radius = previous.radius * (1.0 - progress) + current.radius * progress
        interpolated[name] = node
    return interpolated


def point(center: tuple[int, int], radius: float, angle: float) -> tuple[int, int]:
    return (
        int(round(center[0] + radius * math.cos(angle))),
        int(round(center[1] + radius * math.sin(angle))),
    )


def arc_points(
    center: tuple[int, int], radius: float, start_angle: float, end_angle: float
) -> list[tuple[int, int]]:
    diff = (end_angle - start_angle + math.pi) % (2.0 * math.pi) - math.pi
    steps = max(3, int(abs(diff) * radius / 9.0))
    return [
        point(center, radius, start_angle + diff * i / steps)
        for i in range(steps + 1)
    ]


def edge_points(
    nodes: dict[str, Node],
    parent: str,
    child: str,
    center: tuple[int, int],
) -> list[tuple[int, int]]:
    p = nodes[parent]
    c = nodes[child]
    p0 = point(center, p.radius, p.angle)
    elbow = point(center, c.radius, p.angle)
    arc = arc_points(center, c.radius, p.angle, c.angle)
    return [p0, elbow, *arc[1:]]


def draw_partial_polyline(
    canvas: np.ndarray,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int,
    progress: float,
) -> tuple[int, int] | None:
    progress = max(0.0, min(1.0, progress))
    if len(points) < 2 or progress <= 0.0:
        return None

    segment_lengths: list[float] = []
    total_length = 0.0
    for a, b in zip(points, points[1:]):
        length = math.dist(a, b)
        segment_lengths.append(length)
        total_length += length

    if total_length <= 1e-6:
        return points[-1]

    target_length = total_length * progress
    drawn_points = [points[0]]
    covered = 0.0
    lead = points[0]

    for index, length in enumerate(segment_lengths):
        a = points[index]
        b = points[index + 1]
        if covered + length <= target_length:
            drawn_points.append(b)
            covered += length
            lead = b
            continue

        remaining = max(0.0, target_length - covered)
        t = remaining / max(length, 1e-6)
        lead = (
            int(round(a[0] * (1.0 - t) + b[0] * t)),
            int(round(a[1] * (1.0 - t) + b[1] * t)),
        )
        drawn_points.append(lead)
        break

    if len(drawn_points) >= 2:
        pts = np.array(drawn_points, dtype=np.int32)
        cv2.polylines(canvas, [pts], False, color, thickness, cv2.LINE_AA)
    return lead


def root_color(nodes: dict[str, Node], root_order: dict[str, int], name: str) -> tuple[int, int, int]:
    root = nodes[name].root
    return ROOT_COLORS_BGR[root_order.get(root, 0) % len(ROOT_COLORS_BGR)]


def draw_edge(
    canvas: np.ndarray,
    nodes: dict[str, Node],
    root_order: dict[str, int],
    parent: str,
    child: str,
    center: tuple[int, int],
    thickness: int,
    progress: float,
) -> None:
    color = root_color(nodes, root_order, child)
    points = edge_points(nodes, parent, child, center)
    lead = draw_partial_polyline(canvas, points, color, thickness, progress)
    if lead is not None:
        dot_radius = max(1, thickness // 2)
        cv2.circle(canvas, lead, dot_radius, color, -1, cv2.LINE_AA)


def draw_text_panel(
    canvas: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    align_right: bool = False,
    font_scale: float = 0.58,
    color: tuple[int, int, int] = (220, 220, 220),
) -> None:
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    panel_w = max(w for w, _ in sizes) + 28
    panel_h = sum(h for _, h in sizes) + 16 + 10 * (len(lines) - 1)
    x, y = origin
    if align_right:
        x -= panel_w

    cv2.rectangle(canvas, (x, y), (x + panel_w, y + panel_h), (10, 10, 10), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + panel_w, y + panel_h), (58, 58, 58), 1, cv2.LINE_AA)

    cursor_y = y + 10
    for line, (_, h) in zip(lines, sizes):
        cursor_y += h
        cv2.putText(canvas, line, (x + 14, cursor_y), font, font_scale, color, thickness, cv2.LINE_AA)
        cursor_y += 10


def draw_node_labels(
    canvas: np.ndarray,
    nodes: dict[str, Node],
    active_names: set[str],
    label_by_name: dict[str, str],
    root_order: dict[str, int],
    size: int,
    max_labels: int,
) -> None:
    if not nodes or not label_by_name or max_labels <= 0:
        return

    center = (size // 2, size // 2)
    required_names = {
        name
        for name, node in nodes.items()
        if name in label_by_name and (node.parent is None or bool(node.children))
    }
    live_names = {name for name in active_names if name in nodes and name in label_by_name}
    optional_names = [name for name in nodes if name in label_by_name and name not in required_names and name not in live_names]

    def label_sort_key(name: str) -> tuple[float, float, str]:
        return (nodes[name].angle, nodes[name].radius, label_by_name[name])

    required = sorted(required_names, key=label_sort_key)
    live = sorted(live_names - required_names, key=label_sort_key)
    optional = sorted(optional_names, key=label_sort_key)

    remaining_slots = max(0, max_labels - len(required))
    leaf_candidates = live + optional
    if len(leaf_candidates) > remaining_slots and remaining_slots > 0:
        step = math.ceil(len(leaf_candidates) / remaining_slots)
        leaf_candidates = leaf_candidates[::step]
    elif remaining_slots == 0:
        leaf_candidates = []

    candidates = [(name, True) for name in required] + [(name, False) for name in leaf_candidates]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.36
    thickness = 1
    occupied: list[tuple[int, int, int, int]] = []
    min_gap = 3

    for name, must_draw in candidates:
        node = nodes[name]
        label = label_by_name[name]
        ux = math.cos(node.angle)
        uy = math.sin(node.angle)
        text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_w, text_h = text_size
        pad_x = 4
        pad_y = 2

        best_rect: tuple[int, int, int, int] | None = None
        best_origin: tuple[int, int] | None = None
        radial_offsets = [18, 30, 42, 54, 8, -16]
        tangent_offsets = [0, 12, -12, 24, -24, 36, -36]

        for radial_offset in radial_offsets:
            for tangent_offset in tangent_offsets:
                base_x = center[0] + (node.radius + radial_offset) * ux + tangent_offset * -uy
                base_y = center[1] + (node.radius + radial_offset) * uy + tangent_offset * ux

                if ux >= 0:
                    x0 = int(round(base_x))
                else:
                    x0 = int(round(base_x - text_w - 2 * pad_x))
                y0 = int(round(base_y - text_h // 2 - pad_y))

                if abs(ux) < 0.18:
                    y0 += int(7 * (1 if uy >= 0 else -1))

                x0 = max(8, min(size - text_w - 2 * pad_x - 8, x0))
                y0 = max(78, min(size - text_h - 2 * pad_y - 8, y0))
                rect = (x0, y0, x0 + text_w + 2 * pad_x, y0 + text_h + 2 * pad_y + baseline)

                overlaps = any(
                    not (
                        rect[2] + min_gap < other[0]
                        or rect[0] - min_gap > other[2]
                        or rect[3] + min_gap < other[1]
                        or rect[1] - min_gap > other[3]
                    )
                    for other in occupied
                )
                if not overlaps:
                    best_rect = rect
                    best_origin = (x0, y0)
                    break
            if best_rect is not None:
                break

        if best_rect is None or best_origin is None:
            if not must_draw:
                continue
            base_x = center[0] + (node.radius + 18) * ux
            base_y = center[1] + (node.radius + 18) * uy
            if ux >= 0:
                x0 = int(round(base_x))
            else:
                x0 = int(round(base_x - text_w - 2 * pad_x))
            y0 = int(round(base_y - text_h // 2 - pad_y))
            x0 = max(8, min(size - text_w - 2 * pad_x - 8, x0))
            y0 = max(78, min(size - text_h - 2 * pad_y - 8, y0))
            best_rect = (x0, y0, x0 + text_w + 2 * pad_x, y0 + text_h + 2 * pad_y + baseline)
            best_origin = (x0, y0)

        x0, y0 = best_origin
        rect = best_rect
        color = root_color(nodes, root_order, name)
        fill = (6, 6, 6) if must_draw else (4, 4, 4)
        cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), fill, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), color, 1, cv2.LINE_AA)

        if must_draw:
            node_point = point(center, node.radius, node.angle)
            label_mid = (x0 + text_w // 2 + pad_x, y0 + text_h // 2 + pad_y)
            cv2.line(
                canvas,
                node_point,
                label_mid,
                tuple(int(v * 0.55) for v in color),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            label,
            (x0 + pad_x, y0 + pad_y + text_h),
            font,
            font_scale,
            (232, 232, 232),
            thickness,
            cv2.LINE_AA,
        )
        occupied.append(rect)


def render_tree(
    nodes: dict[str, Node],
    birth_frames: dict[str, int],
    active_names: set[str],
    label_by_name: dict[str, str],
    frame: int,
    size: int,
    total_frames: int,
    growth_progress: float = 1.0,
    max_labels: int = 120,
) -> np.ndarray:
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    roots = sorted_roots(nodes)
    root_order = {root: i for i, root in enumerate(roots)}
    thickness = max(2, size // 700)

    for parent in sorted(nodes, key=lambda n: (nodes[n].depth, natural_key(lineage_code(n)))):
        for child in sorted_children(nodes, parent):
            progress = growth_progress if birth_frames.get(child) == frame else 1.0
            draw_edge(canvas, nodes, root_order, parent, child, center, thickness, progress)

    for root in roots:
        color = root_color(nodes, root_order, root)
        progress = growth_progress if birth_frames.get(root) == frame else 1.0
        radius = max(2, int(round(5 * (0.35 + 0.65 * progress))))
        cv2.circle(canvas, point(center, nodes[root].radius, nodes[root].angle), radius, color, -1, cv2.LINE_AA)

    title = f"Lineage Tree   frame {frame:03d} / {total_frames:03d}"
    cv2.putText(
        canvas,
        title,
        (36, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (232, 232, 232),
        2,
        cv2.LINE_AA,
    )

    draw_text_panel(
        canvas,
        [f"Live cells  {len(active_names)}", f"All cells   {len(nodes)}"],
        (size - 36, 28),
        align_right=True,
        font_scale=0.58,
    )

    draw_node_labels(canvas, nodes, active_names, label_by_name, root_order, size, max_labels)

    legend_x = 38
    legend_y = size - 146
    for root in roots[:4]:
        idx = root_order[root]
        color = ROOT_COLORS_BGR[idx % len(ROOT_COLORS_BGR)]
        full_name, compact_name = ROOT_LABELS[idx % len(ROOT_LABELS)]
        y = legend_y + idx * 28
        cv2.line(canvas, (legend_x, y), (legend_x + 42, y), color, 4, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{compact_name} = {full_name}",
            (legend_x + 56, y + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

    return canvas


def run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg failed")


def write_outputs(frame_dir: Path, out_gif: Path, out_mp4: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg).exists() and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found; cannot encode GIF/MP4")

    input_pattern = str(frame_dir / "frame_%04d.png")
    run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-start_number",
            "0",
            "-i",
            input_pattern,
            "-vf",
            "format=yuv420p",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ]
    )
    run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-start_number",
            "0",
            "-i",
            input_pattern,
            "-vf",
            "split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
            str(out_gif),
        ]
    )


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    csv_paths = [Path(path).expanduser() for path in args.csv_paths or discover_default_csvs(repo_root)]
    csv_paths = [p if p.is_absolute() else repo_root / p for p in csv_paths]
    csv_paths = sorted(csv_paths, key=lambda p: natural_key(str(p)))

    missing = [str(p) for p in csv_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing CSV file(s): " + ", ".join(missing))
    if not csv_paths:
        raise FileNotFoundError(f"No cells.csv files found with default glob {DEFAULT_CSV_GLOB!r}")

    all_frame_names = read_frame_names(csv_paths)
    if not all_frame_names:
        raise ValueError("No cell rows were loaded from the CSV files")

    first_frame = args.first_frame if args.first_frame is not None else min(all_frame_names)
    last_frame = args.last_frame if args.last_frame is not None else max(all_frame_names)
    if last_frame < first_frame:
        raise ValueError("--last-frame must be >= --first-frame")

    frame_names = filter_frame_names(all_frame_names, first_frame, last_frame)
    if not frame_names:
        raise ValueError(f"No cell rows found between frames {first_frame} and {last_frame}")
    missing_frames = [frame for frame in range(first_frame, last_frame + 1) if frame not in frame_names]
    if missing_frames:
        preview = ", ".join(str(frame) for frame in missing_frames[:12])
        if len(missing_frames) > 12:
            preview += ", ..."
        print(f"Warning: missing cell data for {len(missing_frames)} frame(s): {preview}")

    birth_frames = build_birth_frames(frame_names)
    label_by_name = build_label_map(birth_frames, args.label_style)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = output_dir / "frames"
    if frame_dir.exists():
        for old_png in frame_dir.glob("frame_*.png"):
            old_png.unlink()
    else:
        frame_dir.mkdir(parents=True)

    previous_visible_names: set[str] = set()
    previous_layout: dict[str, Node] | None = None
    previous_active_names: set[str] = set()
    rendered_frame_count = 0
    for frame in range(first_frame, last_frame + 1):
        if frame in frame_names:
            previous_active_names = set(frame_names[frame])
        active_names = set(previous_active_names)
        target_visible_names = set(previous_visible_names)
        target_visible_names.update(frame_names.get(frame, set()))
        current_layout = layout_nodes(target_visible_names, args.size)
        added_names = target_visible_names - previous_visible_names
        subframe_count = max(1, args.transition_frames if added_names else args.hold_frames)

        for subframe in range(subframe_count):
            progress = (subframe + 1) / subframe_count
            nodes = interpolate_nodes(previous_layout, current_layout, progress)
            canvas = render_tree(
                nodes,
                birth_frames,
                active_names,
                label_by_name,
                frame,
                args.size,
                last_frame,
                progress,
                args.max_labels,
            )
            frame_path = frame_dir / f"frame_{rendered_frame_count:04d}.png"
            ok = cv2.imwrite(str(frame_path), canvas)
            if not ok:
                raise IOError(f"Failed to write frame image: {frame_path}")
            rendered_frame_count += 1

        previous_visible_names = target_visible_names
        previous_layout = current_layout

    gif_path = Path(args.gif)
    mp4_path = Path(args.mp4)
    if not gif_path.is_absolute():
        gif_path = output_dir / gif_path
    if not mp4_path.is_absolute():
        mp4_path = output_dir / mp4_path

    write_outputs(frame_dir, gif_path, mp4_path, args.fps)

    if not args.keep_frames:
        for png in frame_dir.glob("frame_*.png"):
            png.unlink()
        frame_dir.rmdir()

    print(f"Loaded {len(csv_paths)} CSV file(s)")
    print(f"Rendered {rendered_frame_count} animation frames for biological frames {first_frame}-{last_frame}")
    print(f"GIF: {gif_path}")
    print(f"MP4: {mp4_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
