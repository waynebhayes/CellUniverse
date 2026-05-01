#!/usr/bin/env python3
"""Build a 16:9 four window embryo demo MP4."""

from __future__ import annotations

import argparse
import csv
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


DEFAULT_BASE_DIR = Path(
    "/Users/wangyiding/CellUniverse/C++/output/Yiding_embryo1~84 Verified 2G 20250430"
)
DEFAULT_OUTPUT = Path(
    "/Users/wangyiding/CellUniverse/C++/output/four_windows_demo_16_9/4_windows_demo_16_9.mp4"
)

ROOT_COLORS_BGR = [
    (255, 92, 214),  # bright purple
    (20, 255, 57),   # bright green
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


@dataclass(frozen=True)
class FrameImageSpec:
    frame: int
    real_dir: Path
    synth_dir: Path
    source: str


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
        description="Create a 16:9 four window CellUniverse demo MP4."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument(
        "--csv",
        action="append",
        dest="csv_paths",
        help="Cell CSV path. Repeat this option to pass several files.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--first-frame", type=int, default=1)
    parser.add_argument("--last-frame", type=int, default=84)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--projection", choices=("max", "mean", "middle"), default="max")
    parser.add_argument("--synth-opacity", type=float, default=0.5)
    parser.add_argument(
        "--label-style",
        choices=("compact", "full", "none"),
        default="compact",
    )
    parser.add_argument("--max-labels", type=int, default=140)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument(
        "--napari-preview",
        action="store_true",
        help="Open a napari preview of the projected real and synth stacks after writing the MP4.",
    )
    return parser.parse_args()


def natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def stage_sort_key(path: Path) -> tuple[int, int, list[object]]:
    numbers = [int(value) for value in re.findall(r"\d+", path.name)]
    start = numbers[0] if numbers else 10**9
    end = numbers[1] if len(numbers) > 1 else start
    return start, end, natural_key(path.name)


def frame_number(frame_file: str) -> int | None:
    match = re.search(r"(\d+)", Path(frame_file).stem)
    return int(match.group(1)) if match else None


def discover_base_dir(repo_root: Path, requested: Path) -> Path:
    requested = requested.expanduser()
    if requested.exists():
        return requested.resolve()

    output_dir = repo_root / "C++" / "output"
    candidates = sorted(output_dir.glob("*embryo1~84*"), key=stage_sort_key)
    for candidate in candidates:
        if any((stage / "real").is_dir() and (stage / "synth").is_dir() for stage in candidate.iterdir() if stage.is_dir()):
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot find embryo 1 to 84 output directory: {requested}")


def discover_csvs(base_dir: Path) -> list[Path]:
    csvs = [
        path
        for path in base_dir.glob("*/*.csv")
        if "cell" in path.name.lower()
    ]
    return sorted(csvs, key=lambda p: (stage_sort_key(p.parent), natural_key(p.name)))


def discover_frame_specs(
    base_dir: Path,
    first_frame: int,
    last_frame: int,
) -> dict[int, FrameImageSpec]:
    frame_specs: dict[int, FrameImageSpec] = {}
    stage_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()], key=stage_sort_key)

    for stage_dir in stage_dirs:
        real_root = stage_dir / "real"
        synth_root = stage_dir / "synth"
        if not real_root.is_dir() or not synth_root.is_dir():
            continue

        real_dirs = {
            int(path.name): path
            for path in real_root.iterdir()
            if path.is_dir() and path.name.isdigit()
        }
        synth_dirs = {
            int(path.name): path
            for path in synth_root.iterdir()
            if path.is_dir() and path.name.isdigit()
        }

        for frame in sorted(set(real_dirs) & set(synth_dirs)):
            if first_frame <= frame <= last_frame:
                frame_specs[frame] = FrameImageSpec(
                    frame=frame,
                    real_dir=real_dirs[frame],
                    synth_dir=synth_dirs[frame],
                    source=stage_dir.name,
                )

    return frame_specs


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
        for name in sorted(frame_names[frame], key=natural_key):
            for lineage_name in lineage_chain(name):
                if lineage_name not in birth_frames:
                    birth_frames[lineage_name] = frame
    return birth_frames


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


def frame_radius(layout: TreeLayout, frame: int) -> float:
    if layout.last_frame <= layout.first_frame:
        return layout.outer_radius
    t = (frame - layout.first_frame) / (layout.last_frame - layout.first_frame)
    t = max(0.0, min(1.0, t))
    return layout.inner_radius * (1.0 - t) + layout.outer_radius * t


def assign_time_layout(
    nodes: dict[str, Node],
    birth_frames: dict[str, int],
    layout: TreeLayout,
) -> None:
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


def build_label_map(
    birth_frames: dict[str, int],
    label_style: str,
) -> dict[str, str]:
    if label_style == "none":
        return {}

    roots = sorted({root_name(name) for name in birth_frames}, key=lambda n: natural_key(lineage_code(n)))
    root_index = {root: index for index, root in enumerate(roots)}
    names_by_root: dict[str, list[str]] = defaultdict(list)
    for name in birth_frames:
        names_by_root[root_name(name)].append(name)

    labels: dict[str, str] = {}
    for root in roots:
        index = root_index[root]
        full_name, compact_name = ROOT_LABELS[index % len(ROOT_LABELS)]
        ordered_names = sorted(
            names_by_root[root],
            key=lambda n: (birth_frames[n], natural_key(lineage_code(n)), n),
        )
        for order, name in enumerate(ordered_names, start=1):
            labels[name] = f"{full_name} {order}" if label_style == "full" else f"{compact_name}{order}"
    return labels


def point(center: tuple[int, int], radius: float, angle: float) -> tuple[int, int]:
    return (
        int(round(center[0] + radius * math.cos(angle))),
        int(round(center[1] + radius * math.sin(angle))),
    )


def arc_points(
    center: tuple[int, int],
    radius: float,
    start_angle: float,
    end_angle: float,
) -> list[tuple[int, int]]:
    diff = (end_angle - start_angle + math.pi) % (2.0 * math.pi) - math.pi
    steps = max(3, int(abs(diff) * radius / 7.0))
    return [
        point(center, radius, start_angle + diff * step / steps)
        for step in range(steps + 1)
    ]


def root_color(nodes: dict[str, Node], root_order: dict[str, int], name: str) -> tuple[int, int, int]:
    root = nodes[name].root
    return ROOT_COLORS_BGR[root_order.get(root, 0) % len(ROOT_COLORS_BGR)]


def draw_text_box(
    canvas: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    align_right: bool = False,
    font_scale: float = 0.48,
    color: tuple[int, int, int] = (225, 225, 225),
) -> None:
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    panel_w = max(width for width, _ in sizes) + 24
    panel_h = sum(height for _, height in sizes) + 16 + 8 * (len(lines) - 1)
    x, y = origin
    if align_right:
        x -= panel_w

    cv2.rectangle(canvas, (x, y), (x + panel_w, y + panel_h), (8, 8, 8), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + panel_w, y + panel_h), (70, 70, 70), 1, cv2.LINE_AA)

    cursor_y = y + 9
    for line, (_, height) in zip(lines, sizes):
        cursor_y += height
        cv2.putText(canvas, line, (x + 12, cursor_y), font, font_scale, color, thickness, cv2.LINE_AA)
        cursor_y += 8


def stats_box_style(
    canvas_width: int,
    rows: list[tuple[tuple[int, int, int], str]],
) -> tuple[int, int, float, float]:
    if not rows:
        return 0, 0, 0.0, 0.0

    font = cv2.FONT_HERSHEY_SIMPLEX
    title = "Root family mitotic frequency"
    title_scale = 0.43
    row_scale = 0.39
    thickness = 1
    max_width = max(120, canvas_width - 24)
    title_left = 12
    text_left = 38
    right_pad = 14

    while True:
        title_size = cv2.getTextSize(title, font, title_scale, thickness)[0]
        row_sizes = [cv2.getTextSize(text, font, row_scale, thickness)[0] for _, text in rows]
        panel_w = max(
            [title_left + title_size[0] + right_pad]
            + [text_left + width + right_pad for width, _ in row_sizes]
        )
        if panel_w <= max_width or row_scale <= 0.28:
            break
        title_scale *= 0.94
        row_scale *= 0.94

    panel_w = min(panel_w, max_width)
    panel_h = 28 + 23 * len(rows) + 12
    return panel_w, panel_h, title_scale, row_scale


def draw_color_stats_box(
    canvas: np.ndarray,
    rows: list[tuple[tuple[int, int, int], str]],
    origin: tuple[int, int],
) -> None:
    if not rows:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    title = "Root family mitotic frequency"
    panel_w, panel_h, title_scale, row_scale = stats_box_style(canvas.shape[1], rows)
    thickness = 1
    x, y = origin

    cv2.rectangle(canvas, (x, y), (x + panel_w, y + panel_h), (6, 6, 6), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + panel_w, y + panel_h), (72, 72, 72), 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (x + 12, y + 22), font, title_scale, (220, 220, 220), thickness, cv2.LINE_AA)

    cursor_y = y + 46
    for color, text in rows:
        cv2.line(canvas, (x + 12, cursor_y - 5), (x + 30, cursor_y - 5), color, 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (x + 38, cursor_y), font, row_scale, (224, 224, 224), thickness, cv2.LINE_AA)
        cursor_y += 23


def draw_panel_header(
    panel: np.ndarray,
    title: str,
    frame: int,
    last_frame: int,
) -> None:
    h, w = panel.shape[:2]
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (7, 7, 7), -1)
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (74, 74, 74), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (1, 1), (w - 2, 38), (18, 18, 18), -1)
    cv2.putText(
        panel,
        title,
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    frame_text = f"frame {frame:03d} / {last_frame:03d}"
    text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 1)[0]
    cv2.putText(
        panel,
        frame_text,
        (w - text_size[0] - 14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def paste_fit(
    panel: np.ndarray,
    image_bgr: np.ndarray,
    content_rect: Rect,
) -> None:
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

    src_h, src_w = image_bgr.shape[:2]
    scale = min(content_rect.w / src_w, content_rect.h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x = content_rect.x + (content_rect.w - new_w) // 2
    y = content_rect.y + (content_rect.h - new_h) // 2
    panel[y:y + new_h, x:x + new_w] = resized


def render_image_panel(
    image_bgr: np.ndarray,
    title: str,
    frame: int,
    last_frame: int,
    width: int,
    height: int,
) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    draw_panel_header(panel, title, frame, last_frame)
    paste_fit(panel, image_bgr, Rect(10, 48, width - 20, height - 60))
    return panel


def draw_time_rings(panel: np.ndarray, layout: TreeLayout) -> None:
    for frame in range(layout.first_frame, layout.last_frame + 1):
        radius = int(round(frame_radius(layout, frame)))
        if frame in {layout.first_frame, layout.last_frame} or frame % 10 == 0:
            color = (48, 48, 48)
        else:
            color = (19, 19, 19)
        cv2.circle(panel, layout.center, radius, color, 1, cv2.LINE_AA)

    for frame in [layout.first_frame, 20, 40, 60, layout.last_frame]:
        if layout.first_frame <= frame <= layout.last_frame:
            radius = frame_radius(layout, frame)
            label_point = point(layout.center, radius, math.radians(18))
            cv2.putText(
                panel,
                str(frame),
                (label_point[0] + 3, label_point[1] + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                (105, 105, 105),
                1,
                cv2.LINE_AA,
            )


def draw_lineage_edge(
    panel: np.ndarray,
    nodes: dict[str, Node],
    root_order: dict[str, int],
    parent: str,
    child: str,
    layout: TreeLayout,
    thickness: int,
) -> None:
    parent_node = nodes[parent]
    child_node = nodes[child]
    color = root_color(nodes, root_order, child)
    p0 = point(layout.center, parent_node.radius, parent_node.angle)
    elbow = point(layout.center, child_node.radius, parent_node.angle)
    arc = arc_points(layout.center, child_node.radius, parent_node.angle, child_node.angle)
    pts = np.array([p0, elbow, *arc[1:]], dtype=np.int32)
    cv2.polylines(panel, [pts], False, color, thickness, cv2.LINE_AA)
    cv2.circle(panel, point(layout.center, child_node.radius, child_node.angle), max(2, thickness), color, -1, cv2.LINE_AA)


def draw_active_life_lines(
    panel: np.ndarray,
    nodes: dict[str, Node],
    root_order: dict[str, int],
    active_names: set[str],
    current_frame: int,
    layout: TreeLayout,
    thickness: int,
) -> None:
    current_radius = frame_radius(layout, current_frame)
    for name in sorted(active_names, key=natural_key):
        if name not in nodes:
            continue
        node = nodes[name]
        color = root_color(nodes, root_order, name)
        start_radius = min(node.radius, current_radius)
        p0 = point(layout.center, start_radius, node.angle)
        p1 = point(layout.center, current_radius, node.angle)
        cv2.line(panel, p0, p1, color, thickness, cv2.LINE_AA)
        cv2.circle(panel, p1, max(3, thickness + 1), color, -1, cv2.LINE_AA)


def family_stats_rows(
    nodes: dict[str, Node],
    root_order: dict[str, int],
    birth_frames: dict[str, int],
    active_names: set[str],
    current_frame: int,
    first_frame: int,
) -> list[tuple[tuple[int, int, int], str]]:
    roots = sorted_roots(nodes)
    active_counts: dict[str, int] = defaultdict(int)
    for name in active_names:
        if name in nodes:
            active_counts[nodes[name].root] += 1

    division_counts: dict[str, int] = defaultdict(int)
    for node in nodes.values():
        born_children = [
            child
            for child in node.children
            if birth_frames.get(child, 10**9) <= current_frame
        ]
        if len(born_children) >= 2 and birth_frames.get(node.name, first_frame) <= current_frame:
            division_counts[node.root] += 1

    elapsed_frames = max(1, current_frame - first_frame + 1)
    rows: list[tuple[tuple[int, int, int], str]] = []
    for root in roots[:4]:
        index = root_order[root] % len(ROOT_LABELS)
        full_name, compact_name = ROOT_LABELS[index]
        cells = active_counts.get(root, 0)
        divisions = division_counts.get(root, 0)
        rate_per_10_frames = divisions * 10.0 / elapsed_frames
        text = (
            f"{compact_name} {full_name}: cells {cells}  "
            f"div {divisions}  {rate_per_10_frames:.1f} per 10f"
        )
        rows.append((ROOT_COLORS_BGR[index], text))
    return rows


def draw_node_labels(
    panel: np.ndarray,
    nodes: dict[str, Node],
    root_order: dict[str, int],
    active_names: set[str],
    birth_frames: dict[str, int],
    label_by_name: dict[str, str],
    current_frame: int,
    layout: TreeLayout,
    max_labels: int,
) -> None:
    if not label_by_name or max_labels <= 0:
        return

    current_radius = frame_radius(layout, current_frame)
    required_names = {
        name
        for name, node in nodes.items()
        if name in label_by_name
        and birth_frames.get(name, 10**9) <= current_frame
        and (node.parent is None or any(birth_frames.get(child, 10**9) <= current_frame for child in node.children))
    }
    live_names = {
        name
        for name in active_names
        if name in nodes and name in label_by_name
    }

    def label_sort_key(name: str) -> tuple[float, float, str]:
        node = nodes[name]
        return node.angle, node.radius, label_by_name[name]

    required = sorted(required_names, key=label_sort_key)
    live = sorted(live_names - required_names, key=label_sort_key)
    remaining = max(0, max_labels - len(required))
    if len(live) > remaining and remaining > 0:
        step = math.ceil(len(live) / remaining)
        live = live[::step]
    elif remaining == 0:
        live = []

    candidates = [(name, True) for name in required] + [(name, False) for name in live]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.28
    thickness = 1
    occupied: list[tuple[int, int, int, int]] = []
    min_gap = 2
    h, w = panel.shape[:2]

    for name, must_draw in candidates:
        node = nodes[name]
        color = root_color(nodes, root_order, name)
        label = label_by_name[name]
        label_radius = current_radius if name in active_names else node.radius
        ux = math.cos(node.angle)
        uy = math.sin(node.angle)
        text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_w, text_h = text_size
        pad_x = 3
        pad_y = 2
        best_rect: tuple[int, int, int, int] | None = None
        best_origin: tuple[int, int] | None = None

        for radial_offset in (10, 19, 28, -13, 38):
            for tangent_offset in (0, 9, -9, 18, -18, 27, -27):
                base_x = layout.center[0] + (label_radius + radial_offset) * ux + tangent_offset * -uy
                base_y = layout.center[1] + (label_radius + radial_offset) * uy + tangent_offset * ux
                x0 = int(round(base_x if ux >= 0 else base_x - text_w - 2 * pad_x))
                y0 = int(round(base_y - text_h // 2 - pad_y))
                x0 = max(6, min(w - text_w - 2 * pad_x - 6, x0))
                max_y = min(h - text_h - 2 * pad_y - 6, layout.stats_top - text_h - 2 * pad_y - 8)
                y0 = max(44, min(max_y, y0))
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
            label_point = point(layout.center, label_radius + 10, node.angle)
            x0 = int(round(label_point[0] if ux >= 0 else label_point[0] - text_w - 2 * pad_x))
            y0 = int(round(label_point[1] - text_h // 2 - pad_y))
            x0 = max(6, min(w - text_w - 2 * pad_x - 6, x0))
            max_y = min(h - text_h - 2 * pad_y - 6, layout.stats_top - text_h - 2 * pad_y - 8)
            y0 = max(44, min(max_y, y0))
            best_rect = (x0, y0, x0 + text_w + 2 * pad_x, y0 + text_h + 2 * pad_y + baseline)
            best_origin = (x0, y0)

        x0, y0 = best_origin
        rect = best_rect
        cv2.rectangle(panel, (rect[0], rect[1]), (rect[2], rect[3]), (5, 5, 5), -1, cv2.LINE_AA)
        cv2.rectangle(panel, (rect[0], rect[1]), (rect[2], rect[3]), color, 1, cv2.LINE_AA)
        cv2.putText(
            panel,
            label,
            (x0 + pad_x, y0 + pad_y + text_h),
            font,
            font_scale,
            (232, 232, 232),
            thickness,
            cv2.LINE_AA,
        )
        occupied.append(rect)


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
) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    draw_panel_header(panel, "Lineage Tree", frame, last_frame)
    roots = sorted_roots(nodes)
    root_order = {root: index for index, root in enumerate(roots)}
    thickness = max(1, min(width, height) // 360)

    draw_time_rings(panel, layout)

    for name in sorted(nodes, key=lambda n: (birth_frames.get(n, 10**9), natural_key(lineage_code(n)))):
        node = nodes[name]
        if node.parent is None or birth_frames.get(name, 10**9) > frame:
            continue
        draw_lineage_edge(panel, nodes, root_order, node.parent, name, layout, thickness)

    draw_active_life_lines(panel, nodes, root_order, active_names, frame, layout, thickness)

    for name, node in nodes.items():
        if birth_frames.get(name, 10**9) > frame:
            continue
        color = root_color(nodes, root_order, name)
        radius = max(2, thickness + 1) if node.children else max(1, thickness)
        cv2.circle(panel, point(layout.center, node.radius, node.angle), radius, color, -1, cv2.LINE_AA)

    draw_text_box(
        panel,
        [f"Live cells  {len(active_names)}"],
        (width - 14, 52),
        align_right=True,
        font_scale=0.44,
    )

    rows = family_stats_rows(nodes, root_order, birth_frames, active_names, frame, layout.first_frame)
    stats_w, stats_h, _, _ = stats_box_style(width, rows)
    stats_x = max(10, min(14, width - stats_w - 10))
    stats_y = max(layout.stats_top + 8, height - stats_h - 10)
    draw_color_stats_box(panel, rows, (stats_x, stats_y))

    draw_node_labels(
        panel,
        nodes,
        root_order,
        active_names,
        birth_frames,
        label_by_name,
        frame,
        layout,
        max_labels,
    )
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


def image_files(frame_dir: Path) -> list[Path]:
    files = list(frame_dir.glob("*.png")) + list(frame_dir.glob("*.tif")) + list(frame_dir.glob("*.tiff"))
    return sorted(files, key=lambda p: natural_key(p.name))


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f"Cannot read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def project_volume(frame_dir: Path, mode: str) -> np.ndarray:
    files = image_files(frame_dir)
    if not files:
        raise FileNotFoundError(f"No image slices found in {frame_dir}")

    if mode == "middle":
        return read_gray(files[len(files) // 2]).astype(np.float32)

    result: np.ndarray | None = None
    if mode == "max":
        for file_path in files:
            image = read_gray(file_path)
            if result is None:
                result = image.astype(np.float32)
            else:
                np.maximum(result, image, out=result)
        assert result is not None
        return result

    accum: np.ndarray | None = None
    for file_path in files:
        image = read_gray(file_path).astype(np.float32)
        if accum is None:
            accum = image
        else:
            accum += image
    assert accum is not None
    return accum / float(len(files))


def compute_window(images: list[np.ndarray], low_q: float, high_q: float) -> tuple[float, float]:
    samples = [image.ravel()[::8] for image in images]
    flat = np.concatenate(samples)
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


def make_overlay(real_u8: np.ndarray, synth_u8: np.ndarray, opacity: float) -> np.ndarray:
    real_bgr = cv2.cvtColor(real_u8, cv2.COLOR_GRAY2BGR).astype(np.float32)
    synth_bgr = cv2.applyColorMap(synth_u8, cv2.COLORMAP_VIRIDIS).astype(np.float32)
    alpha = np.clip(opacity, 0.0, 1.0) * (synth_u8.astype(np.float32) / 255.0)
    alpha = alpha[:, :, None]
    overlay = real_bgr * (1.0 - alpha) + synth_bgr * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def build_tree_layout(tree_rect: Rect, first_frame: int, last_frame: int) -> TreeLayout:
    stats_reserved = min(max(132, int(round(tree_rect.h * 0.22))), tree_rect.h // 3)
    stats_top = tree_rect.h - stats_reserved
    content_top = 42
    content_bottom = max(content_top + 120, stats_top - 12)
    content_h = content_bottom - content_top
    center = (tree_rect.w // 2, content_top + int(round(content_h * 0.50)))
    outer_radius = min(tree_rect.w * 0.43, content_h * 0.43)
    inner_radius = max(14.0, outer_radius * 0.06)
    return TreeLayout(
        center=center,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        content_bottom=content_bottom,
        stats_top=stats_top,
        first_frame=first_frame,
        last_frame=last_frame,
    )


def clean_frame_dir(frame_dir: Path) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_png in frame_dir.glob("frame_*.png"):
        old_png.unlink()


def encode_h264_mp4(frame_dir: Path, output_path: Path, fps: float, first_frame: int) -> None:
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg).exists() and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found, so H.264 MP4 cannot be written.")

    input_pattern = str(frame_dir / "frame_%04d.png")
    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        str(first_frame),
        "-i",
        input_pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-level",
        "4.1",
        "-tag:v",
        "avc1",
        "-movflags",
        "+faststart",
        "-crf",
        "18",
        "-preset",
        "medium",
        str(output_path),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg failed to encode MP4")


def compose_frame(
    real_u8: np.ndarray,
    synth_u8: np.ndarray,
    overlay_bgr: np.ndarray,
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

    real_panel = render_image_panel(
        cv2.cvtColor(real_u8, cv2.COLOR_GRAY2BGR),
        "Real Image",
        frame,
        args.last_frame,
        rects["real"].w,
        rects["real"].h,
    )
    synth_panel = render_image_panel(
        cv2.applyColorMap(synth_u8, cv2.COLORMAP_VIRIDIS),
        "Synth viridis",
        frame,
        args.last_frame,
        rects["synth"].w,
        rects["synth"].h,
    )
    overlay_panel = render_image_panel(
        overlay_bgr,
        f"Synth over Real   opacity {args.synth_opacity:.2f}   viridis",
        frame,
        args.last_frame,
        rects["overlay"].w,
        rects["overlay"].h,
    )
    tree_panel = render_lineage_tree_panel(
        nodes,
        birth_frames,
        active_names,
        label_by_name,
        frame,
        args.last_frame,
        tree_layout,
        rects["tree"].w,
        rects["tree"].h,
        args.max_labels,
    )

    paste_panel(canvas, real_panel, rects["real"])
    paste_panel(canvas, synth_panel, rects["synth"])
    paste_panel(canvas, overlay_panel, rects["overlay"])
    paste_panel(canvas, tree_panel, rects["tree"])
    return canvas


def open_napari_preview(real_stack: np.ndarray, synth_stack: np.ndarray, fps: float) -> None:
    import napari

    viewer = napari.Viewer(title="CellUniverse four window demo preview")
    viewer.add_image(real_stack, name="Real image", colormap="gray")
    viewer.add_image(synth_stack, name="Synth image", colormap="viridis", opacity=0.5)
    viewer.dims.axis_labels = ("frame", "y", "x")
    viewer.dims.ndisplay = 2
    print(f"[INFO] Napari preview opened at {fps:.1f} frames per second data timing.")
    napari.run()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    base_dir = discover_base_dir(repo_root, args.base_dir)
    if args.last_frame < args.first_frame:
        raise ValueError("--last-frame must be greater than or equal to --first-frame")
    if args.width % 2 or args.height % 2:
        raise ValueError("--width and --height must be even numbers for yuv420p MP4 output")

    csv_paths = [Path(path).expanduser() for path in args.csv_paths or discover_csvs(base_dir)]
    csv_paths = [path if path.is_absolute() else repo_root / path for path in csv_paths]
    csv_paths = sorted(csv_paths, key=lambda p: (stage_sort_key(p.parent), natural_key(p.name)))
    missing_csvs = [str(path) for path in csv_paths if not path.exists()]
    if missing_csvs:
        raise FileNotFoundError("Missing CSV file(s): " + ", ".join(missing_csvs))
    if not csv_paths:
        raise FileNotFoundError(f"No cell CSV files found under {base_dir}")

    all_frame_names = read_frame_names(csv_paths)
    frame_names = filter_frame_names(all_frame_names, args.first_frame, args.last_frame)
    missing_lineage_frames = [
        frame for frame in range(args.first_frame, args.last_frame + 1)
        if frame not in frame_names
    ]
    if missing_lineage_frames:
        preview = ", ".join(str(frame) for frame in missing_lineage_frames[:12])
        if len(missing_lineage_frames) > 12:
            preview += ", ..."
        print(f"[WARN] Missing lineage CSV data for {len(missing_lineage_frames)} frame(s): {preview}")

    frame_specs = discover_frame_specs(base_dir, args.first_frame, args.last_frame)
    missing_image_frames = [
        frame for frame in range(args.first_frame, args.last_frame + 1)
        if frame not in frame_specs
    ]
    if missing_image_frames:
        preview = ", ".join(str(frame) for frame in missing_image_frames[:12])
        if len(missing_image_frames) > 12:
            preview += ", ..."
        raise FileNotFoundError(f"Missing real or synth image data for frame(s): {preview}")

    birth_frames = build_birth_frames(frame_names)
    all_visible_names: set[str] = set()
    for names in frame_names.values():
        all_visible_names.update(names)
    nodes = build_nodes(all_visible_names)
    label_by_name = build_label_map(birth_frames, args.label_style)

    rects = layout_rects(args.width, args.height)
    tree_layout = build_tree_layout(rects["tree"], args.first_frame, args.last_frame)
    assign_time_layout(nodes, birth_frames, tree_layout)

    print(f"[INFO] base dir: {base_dir}")
    print(f"[INFO] loaded CSV files: {len(csv_paths)}")
    for csv_path in csv_paths:
        print(f"[CSV] {csv_path}")
    print(f"[INFO] image frames: {len(frame_specs)}")
    print(f"[INFO] output size: {args.width} x {args.height}, fps: {args.fps}")

    real_proj: dict[int, np.ndarray] = {}
    synth_proj: dict[int, np.ndarray] = {}
    for index, frame in enumerate(range(args.first_frame, args.last_frame + 1), start=1):
        spec = frame_specs[frame]
        print(f"[LOAD] frame {frame:03d} from {spec.source}")
        real_proj[frame] = project_volume(spec.real_dir, args.projection)
        synth_proj[frame] = project_volume(spec.synth_dir, args.projection)
        if index % 10 == 0:
            print(f"[INFO] loaded {index} projected frame(s)")

    real_low, real_high = compute_window(list(real_proj.values()), 1.0, 99.8)
    synth_low, synth_high = compute_window(list(synth_proj.values()), 1.0, 99.8)
    print(f"[INFO] real contrast window: {real_low:.3f} to {real_high:.3f}")
    print(f"[INFO] synth contrast window: {synth_low:.3f} to {synth_high:.3f}")

    output_path = args.output.expanduser()
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output_path.parent / "frames_16_9"
    clean_frame_dir(frame_dir)

    last_active_names: set[str] = set()
    real_preview_stack: list[np.ndarray] = []
    synth_preview_stack: list[np.ndarray] = []
    for frame in range(args.first_frame, args.last_frame + 1):
        if frame in frame_names:
            last_active_names = set(frame_names[frame])
        active_names = set(last_active_names)

        real_u8 = normalize_to_u8(real_proj[frame], real_low, real_high)
        synth_u8 = normalize_to_u8(synth_proj[frame], synth_low, synth_high)
        overlay_bgr = make_overlay(real_u8, synth_u8, args.synth_opacity)
        canvas = compose_frame(
            real_u8,
            synth_u8,
            overlay_bgr,
            nodes,
            birth_frames,
            active_names,
            label_by_name,
            frame,
            args,
            rects,
            tree_layout,
        )
        frame_path = frame_dir / f"frame_{frame:04d}.png"
        ok = cv2.imwrite(str(frame_path), canvas)
        if not ok:
            raise IOError(f"Failed to write frame image: {frame_path}")
        if args.napari_preview:
            real_preview_stack.append(real_u8)
            synth_preview_stack.append(synth_u8)
        print(f"[WRITE] frame {frame:03d}")

    encode_h264_mp4(frame_dir, output_path, args.fps, args.first_frame)

    if not args.keep_frames:
        for frame_path in frame_dir.glob("frame_*.png"):
            frame_path.unlink()
        try:
            frame_dir.rmdir()
        except OSError:
            pass

    print(f"[DONE] MP4 saved to {output_path}")

    if args.napari_preview:
        open_napari_preview(
            np.stack(real_preview_stack, axis=0),
            np.stack(synth_preview_stack, axis=0),
            args.fps,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
