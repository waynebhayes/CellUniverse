#!/usr/bin/env python3
"""Napari review view for one embryo frame.

This script is only for visual audit. It reads saved prediction output and GT
centers, then overlays small center markers and optional wire rings.
"""

from __future__ import annotations

import argparse
import csv
import colorsys
import math
from pathlib import Path

import napari
import numpy as np
import tifffile


def parse_csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def read_prediction(cells_csv: Path, frame: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with cells_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("file") != f"t{frame:03d}.tif":
                continue
            if row.get("isTrash", "0").lower() in {"1", "true", "yes"}:
                continue
            rows.append(row)
    return rows


def read_gt(gt_csv: Path, frame: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with gt_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if int(row["frame"]) == frame:
                rows.append(row)
    return rows


def frame_tif_path(run: Path, frame: int, kind: str) -> Path:
    flat = run / f"{frame}_{kind}.tif"
    if flat.is_file():
        return flat
    return run / "tiff" / kind / f"{frame}.tif"


def float_value(row: dict[str, str], key: str, fallback: float) -> float:
    try:
        return float(row.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def point_from_prediction(row: dict[str, str]) -> list[float]:
    return [float(row["z"]), float(row["y"]), float(row["x"])]


def point_from_gt(row: dict[str, str]) -> list[float]:
    return [float(row["z_interp"]), float(row["y"]), float(row["x"])]


def radius_value(row: dict[str, str], key: str, fallback: float) -> float:
    return max(1.0, float_value(row, key, fallback))


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


def to_zyx(points_xyz: np.ndarray) -> np.ndarray:
    return points_xyz[:, [2, 1, 0]]


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
        gt_point = np.asarray(
            [float(gt_row["x"]), float(gt_row["y"]), float(gt_row["z_interp"])]
        )
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


def build_neighbor_palette(
    rows: list[dict[str, str]],
    color_count: int,
    neighbor_distance: float,
) -> dict[str, int]:
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

    def similarity(left: int, right: int) -> float:
        if color_count <= 1:
            return 1.0
        diff = abs(left - right) / float(color_count - 1)
        return (1.0 - diff) ** 2

    while remaining:
        name = max(
            remaining,
            key=lambda n: (
                sum(1 for other, _ in neighbors[n] if other in assignment),
                sum(weight for other, weight in neighbors[n] if other in assignment),
                sum(weight for _, weight in neighbors[n]),
                n,
            ),
        )
        target = (len(assignment) * 0.61803398875) % 1.0
        target_idx = int(round(target * (color_count - 1)))

        def candidate_penalty(idx: int) -> tuple[float, int, int]:
            neighbor_penalty = 0.0
            for other, weight in neighbors[name]:
                if other not in assignment:
                    continue
                other_idx = assignment[other]
                neighbor_penalty += weight * (4.0 if idx == other_idx else similarity(idx, other_idx))
            balance_penalty = 0.06 * color_usage[idx]
            target_penalty = 0 if idx == target_idx else abs(idx - target_idx)
            return (neighbor_penalty + balance_penalty, target_penalty, idx)

        selected = min(range(color_count), key=candidate_penalty)
        assignment[name] = selected
        color_usage[selected] += 1
        remaining.remove(name)
    return assignment


def green_palette(color_count: int) -> list[str]:
    values = np.linspace(0.48, 1.0, max(3, color_count))
    colors: list[str] = []
    for value in values:
        red, green, blue = colorsys.hsv_to_rgb(0.36, 0.90, float(value))
        colors.append(
            "#{:02x}{:02x}{:02x}".format(
                int(255 * red),
                int(255 * green),
                int(255 * blue),
            )
        )
    return colors


def split_pair_palette() -> list[str]:
    return ["#7a1fff", "#0033cc", "#ffff00"]


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
        row["label_id"]: np.asarray(
            [float(row["x"]), float(row["y"]), float(row["z_interp"])]
        )
        for row in gt
    }
    centroids: dict[str, np.ndarray] = {}
    for group_name in sorted(set(split_pred_groups) | set(split_gt_groups)):
        points: list[np.ndarray] = [
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


def split_group_color_map(
    group_order: list[str],
    centroids: dict[str, np.ndarray],
    neighbor_distance: float,
) -> dict[str, str]:
    palette = split_pair_palette()
    if not group_order:
        return {}
    neighbors: dict[str, list[tuple[str, float]]] = {group: [] for group in group_order}
    for index, left in enumerate(group_order):
        if left not in centroids:
            continue
        for right in group_order[index + 1:]:
            if right not in centroids:
                continue
            dist = float(np.linalg.norm(centroids[left] - centroids[right]))
            if dist <= neighbor_distance:
                weight = max(0.05, 1.0 - dist / max(1.0, neighbor_distance))
                neighbors[left].append((right, weight))
                neighbors[right].append((left, weight))

    assignment: dict[str, int] = {}
    usage = [0] * len(palette)
    remaining = set(group_order)
    while remaining:
        group = max(
            remaining,
            key=lambda item: (
                sum(1 for other, _ in neighbors[item] if other in assignment),
                sum(weight for _, weight in neighbors[item]),
                item,
            ),
        )
        cycle_target = len(assignment) % len(palette)

        def color_penalty(color_idx: int) -> tuple[float, int, int]:
            near_penalty = 0.0
            for other, weight in neighbors[group]:
                if other not in assignment:
                    continue
                near_penalty += weight * (12.0 if assignment[other] == color_idx else 0.0)
            return (near_penalty + 0.08 * usage[color_idx], color_idx != cycle_target, color_idx)

        selected = min(range(len(palette)), key=color_penalty)
        assignment[group] = selected
        usage[selected] += 1
        remaining.remove(group)
    return {group: palette[assignment[group]] for group in group_order}


def manual_split_groups(split_pred_names: set[str]) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for name in sorted(split_pred_names):
        parent = name[:-1] if name[-1:] in {"0", "1"} else name
        groups.setdefault(parent, set()).add(name)
    return groups


def make_wire_rings(
    rows: list[dict[str, str]],
    segments: int,
    radius_scale: float,
    rings_per_axis: int,
) -> dict[str, list[np.ndarray]]:
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=True)
    rings_by_name: dict[str, list[np.ndarray]] = {}
    levels = max(1, rings_per_axis)
    if levels == 1:
        offsets = [0.0]
    elif levels == 2:
        offsets = [-0.38, 0.38]
    else:
        offsets = np.linspace(-0.62, 0.62, levels)
    for row in rows:
        center_xyz = np.asarray([float(row["x"]), float(row["y"]), float(row["z"])])
        rx = radius_value(row, "aRadius", 10.0) * radius_scale
        ry = radius_value(row, "bRadius", rx) * radius_scale
        rz = radius_value(row, "cRadius", ry) * radius_scale
        rot = rotation_matrix(row)
        rings: list[np.ndarray] = []

        for offset in offsets:
            xy_scale = float(np.sqrt(max(0.0, 1.0 - offset * offset)))
            local_xy = np.column_stack(
                [
                    rx * xy_scale * np.cos(angles),
                    ry * xy_scale * np.sin(angles),
                    np.full_like(angles, rz * offset),
                ]
            )
            rings.append(to_zyx(center_xyz + local_xy @ rot.T))

            xz_scale = float(np.sqrt(max(0.0, 1.0 - offset * offset)))
            local_xz = np.column_stack(
                [
                    rx * xz_scale * np.cos(angles),
                    np.full_like(angles, ry * offset),
                    rz * xz_scale * np.sin(angles),
                ]
            )
            rings.append(to_zyx(center_xyz + local_xz @ rot.T))

            yz_scale = float(np.sqrt(max(0.0, 1.0 - offset * offset)))
            local_yz = np.column_stack(
                [
                    np.full_like(angles, rx * offset),
                    ry * yz_scale * np.cos(angles),
                    rz * yz_scale * np.sin(angles),
                ]
            )
            rings.append(to_zyx(center_xyz + local_yz @ rot.T))
        rings_by_name[row["name"]] = rings
    return rings_by_name


def add_points(
    viewer,
    points: list[list[float]],
    name: str,
    color: str,
    size: float,
    opacity: float,
    symbol: str = "disc",
) -> None:
    if not points:
        return
    try:
        viewer.add_points(
            np.asarray(points, dtype=float),
            name=name,
            size=size,
            symbol=symbol,
            face_color=color,
            opacity=opacity,
            out_of_slice_display=True,
        )
    except TypeError:
        viewer.add_points(
            np.asarray(points, dtype=float),
            name=name,
            size=size,
            face_color=color,
            opacity=opacity,
            out_of_slice_display=True,
        )


def add_points_grouped(
    viewer,
    rows: list[dict[str, str]],
    name: str,
    palette: list[str],
    assignments: dict[str, int],
    size: float,
    opacity: float,
    hidden_names: set[str],
) -> None:
    for color_idx, color in enumerate(palette):
        points = [
            point_from_prediction(row)
            for row in rows
            if row["name"] not in hidden_names and
            assignments.get(row["name"], 0) == color_idx
        ]
        add_points(viewer, points, f"{name} {color_idx + 1}", color, size, opacity)


def add_points_by_group(
    viewer,
    rows: list[dict[str, str]],
    group_names: dict[str, set[str]],
    group_colors: dict[str, str],
    layer_prefix: str,
    size: float,
    opacity: float,
) -> None:
    row_by_name = {row["name"]: row for row in rows}
    for group_name in sorted(group_names):
        points = [
            point_from_prediction(row_by_name[name])
            for name in sorted(group_names[group_name])
            if name in row_by_name
        ]
        add_points(
            viewer,
            points,
            f"{layer_prefix} {group_name}",
            group_colors.get(group_name, "#4b0082"),
            size,
            opacity,
        )


def add_gt_points_by_group(
    viewer,
    gt: list[dict[str, str]],
    group_labels: dict[str, set[str]],
    group_colors: dict[str, str],
    layer_prefix: str,
    size: float,
    opacity: float,
    symbol: str = "disc",
) -> None:
    for group_name in sorted(group_labels):
        points = [
            point_from_gt(row)
            for row in gt
            if row["label_id"] in group_labels[group_name]
        ]
        add_points(
            viewer,
            points,
            f"{layer_prefix} {group_name}",
            group_colors.get(group_name, "#4b0082"),
            size,
            opacity,
            symbol,
        )


def add_ring_layer(
    viewer,
    rings: list[np.ndarray],
    name: str,
    color: str,
    opacity: float,
    edge_width: float,
) -> None:
    if not rings:
        return
    try:
        viewer.add_shapes(
            rings,
            shape_type="path",
            name=name,
            edge_color=color,
            edge_width=edge_width,
            opacity=opacity,
        )
    except TypeError:
        viewer.add_shapes(
            rings,
            shape_type="path",
            name=name,
            edge_color=color,
            opacity=opacity,
        )


def add_ring_layers_grouped(
    viewer,
    rings_by_name: dict[str, list[np.ndarray]],
    palette: list[str],
    assignments: dict[str, int],
    hidden_names: set[str],
    opacity: float,
    edge_width: float,
) -> None:
    for color_idx, color in enumerate(palette):
        rings: list[np.ndarray] = []
        for name, cell_rings in rings_by_name.items():
            if name in hidden_names:
                continue
            if assignments.get(name, 0) == color_idx:
                rings.extend(cell_rings)
        add_ring_layer(
            viewer,
            rings,
            f"Prediction surface rings green {color_idx + 1}",
            color,
            opacity,
            edge_width,
        )


def add_ring_layers_by_group(
    viewer,
    rings_by_name: dict[str, list[np.ndarray]],
    group_names: dict[str, set[str]],
    group_colors: dict[str, str],
    layer_prefix: str,
    opacity: float,
    edge_width: float,
) -> None:
    for group_name in sorted(group_names):
        rings: list[np.ndarray] = []
        for name in group_names[group_name]:
            rings.extend(rings_by_name.get(name, []))
        add_ring_layer(
            viewer,
            rings,
            f"{layer_prefix} {group_name}",
            group_colors.get(group_name, "#4b0082"),
            opacity,
            edge_width,
        )


def add_ring_layers_for_names(
    viewer,
    rings_by_name: dict[str, list[np.ndarray]],
    names: set[str],
    layer_name: str,
    color: str,
    opacity: float,
    edge_width: float,
) -> None:
    rings: list[np.ndarray] = []
    for name in names:
        rings.extend(rings_by_name.get(name, []))
    add_ring_layer(
        viewer,
        rings,
        layer_name,
        color,
        opacity,
        edge_width,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument(
        "--gt",
        type=Path,
        default=Path(
            "/Users/wangyiding/CellUniverse/C++/config/embryo/ground_truth/"
            "Embryo_TRA_all_frames_centroids.csv"
        ),
    )
    parser.add_argument("--miss-labels", default="")
    parser.add_argument("--extra-names", default="")
    parser.add_argument("--related-names", default="")
    parser.add_argument("--gt-split-labels", default="auto")
    parser.add_argument("--split-pred-names", default="")
    parser.add_argument("--split-match-distance", type=float, default=22.0)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--error-size", type=float, default=5.8)
    parser.add_argument("--gt-opacity", type=float, default=0.6)
    parser.add_argument("--pred-opacity", type=float, default=0.72)
    parser.add_argument("--error-opacity", type=float, default=0.6)
    parser.add_argument("--ring-opacity", type=float, default=0.68)
    parser.add_argument("--green-levels", type=int, default=9)
    parser.add_argument("--neighbor-distance", type=float, default=42.0)
    parser.add_argument("--ring-scale", type=float, default=1.0)
    parser.add_argument("--rings-per-axis", type=int, default=2)
    parser.add_argument("--ring-levels", type=int, default=None)
    parser.add_argument("--ring-segments", type=int, default=72)
    parser.add_argument("--ring-edge-width", type=float, default=0.16)
    args = parser.parse_args()

    run = args.run.expanduser().resolve()
    frame = args.frame
    cells_csv = run / "cells.csv"
    real_tif = frame_tif_path(run, frame, "real")
    synth_tif = frame_tif_path(run, frame, "synth")

    pred = read_prediction(cells_csv, frame)
    gt = read_gt(args.gt, frame)
    miss_labels = parse_csv_set(args.miss_labels)
    extra_names = parse_csv_set(args.extra_names)
    related_names = parse_csv_set(args.related_names)
    split_groups = (
        infer_gt_split_groups(gt, frame)
        if args.gt_split_labels == "auto"
        else {"manual": parse_csv_set(args.gt_split_labels)}
    )
    split_labels = flatten_groups(split_groups)
    manual_pred_groups = manual_split_groups(parse_csv_set(args.split_pred_names))
    split_pred_groups = predicted_split_groups_from_gt(
        pred,
        gt,
        split_groups,
        args.split_match_distance,
    )
    for group_name, names in manual_pred_groups.items():
        split_pred_groups.setdefault(group_name, set()).update(names)
    split_pred_names = flatten_groups(split_pred_groups)
    error_pred_names = extra_names | related_names
    hidden_pred_names = split_pred_names | error_pred_names
    green_assignments = build_neighbor_palette(
        pred,
        max(3, args.green_levels),
        max(1.0, args.neighbor_distance),
    )
    greens = green_palette(max(3, args.green_levels))
    split_group_order = sorted(set(split_groups) | set(split_pred_groups))
    split_group_colors = split_group_color_map(
        split_group_order,
        split_group_centroids(pred, gt, split_pred_groups, split_groups),
        max(1.0, args.neighbor_distance * 1.6),
    )

    viewer = napari.Viewer(title=f"f{frame} review: {run.name}", ndisplay=3)
    viewer.add_image(tifffile.imread(real_tif), name="real", colormap="gray", opacity=1.0)
    viewer.add_image(tifffile.imread(synth_tif), name="synth", colormap="cyan", opacity=0.22)

    rings_by_name = make_wire_rings(
        pred,
        max(12, args.ring_segments),
        max(0.1, args.ring_scale),
        max(1, args.rings_per_axis),
    )
    add_ring_layers_grouped(
        viewer,
        rings_by_name,
        greens,
        green_assignments,
        hidden_pred_names,
        args.ring_opacity,
        max(0.02, args.ring_edge_width),
    )
    add_ring_layers_by_group(
        viewer,
        rings_by_name,
        split_pred_groups,
        split_group_colors,
        "Prediction split daughter rings colored",
        min(1.0, args.ring_opacity + 0.12),
        max(0.02, args.ring_edge_width + 0.04),
    )
    add_ring_layers_for_names(
        viewer,
        rings_by_name,
        error_pred_names,
        "Error prediction rings red",
        "#ff0000",
        args.error_opacity,
        max(0.02, args.ring_edge_width + 0.06),
    )
    add_points(
        viewer,
        [point_from_gt(row) for row in gt if row["label_id"] not in split_labels],
        "GT centers black squares",
        "#000000",
        args.point_size,
        args.gt_opacity,
        "square",
    )
    add_gt_points_by_group(
        viewer,
        gt,
        split_groups,
        {group_name: "#000000" for group_name in split_groups},
        "GT newly split centers black squares",
        args.point_size + 0.5,
        0.85,
        "square",
    )
    add_points_grouped(
        viewer,
        pred,
        "Prediction centers green",
        greens,
        green_assignments,
        args.point_size,
        args.pred_opacity,
        hidden_pred_names,
    )
    add_points_by_group(
        viewer,
        pred,
        split_pred_groups,
        split_group_colors,
        "Prediction split daughter centers colored",
        args.point_size + 0.35,
        min(1.0, args.pred_opacity + 0.12),
    )

    missing_gt_points = [
        point_from_gt(row) for row in gt if row["label_id"] in miss_labels
    ]
    add_points(
        viewer,
        missing_gt_points,
        "Missing GT audit red",
        "#ff0000",
        args.error_size,
        args.error_opacity,
    )
    error_prediction_points = [
        point_from_prediction(row)
        for row in pred
        if row["name"] in extra_names or row["name"] in related_names
    ]
    add_points(
        viewer,
        error_prediction_points,
        "Error prediction centers red",
        "#ff0000",
        args.error_size,
        args.error_opacity,
    )

    viewer.dims.ndisplay = 3
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
