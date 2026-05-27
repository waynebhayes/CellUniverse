#!/usr/bin/env python3
"""Validate one CellUniverse embryo frame against GT centers.

This helper is intentionally independent from the algorithm. It only reads the
saved cells.csv and the official GT centroid CSV, then performs one-to-one
center matching within a configurable distance threshold. It can also write the
next-frame initial CSV by reusing the accepted CellUniverse ellipsoids and only
changing the file column.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from pathlib import Path


def read_prediction(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("isTrash", "0")).lower() in {"1", "true", "yes"}:
                continue
            rows.append(
                {
                    "name": row["name"],
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "row": row,
                }
            )
    return rows


def read_gt(path: Path, frame: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["frame"]) != frame:
                continue
            rows.append(
                {
                    "label": row["label_id"],
                    "parent": row["parent_label"],
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z_interp"]),
                }
            )
    return rows


def point_distance(a: dict[str, object], b: dict[str, object]) -> float:
    return math.dist(
        (float(a["x"]), float(a["y"]), float(a["z"])),
        (float(b["x"]), float(b["y"]), float(b["z"])),
    )


def match_centers(
    pred: list[dict[str, object]],
    gt: list[dict[str, object]],
    threshold: float,
) -> tuple[list[int], list[int], int]:
    adjacency = [
        [j for j, gt_row in enumerate(gt) if point_distance(pred_row, gt_row) <= threshold]
        for pred_row in pred
    ]
    pair_pred = [-1] * len(pred)
    pair_gt = [-1] * len(gt)
    dist = [0] * len(pred)

    def bfs() -> bool:
        queue: deque[int] = deque()
        found_free_gt = False
        for pred_index, partner in enumerate(pair_pred):
            if partner == -1:
                dist[pred_index] = 0
                queue.append(pred_index)
            else:
                dist[pred_index] = -1
        while queue:
            pred_index = queue.popleft()
            for gt_index in adjacency[pred_index]:
                next_pred = pair_gt[gt_index]
                if next_pred == -1:
                    found_free_gt = True
                elif dist[next_pred] == -1:
                    dist[next_pred] = dist[pred_index] + 1
                    queue.append(next_pred)
        return found_free_gt

    def dfs(pred_index: int) -> bool:
        for gt_index in adjacency[pred_index]:
            next_pred = pair_gt[gt_index]
            if next_pred == -1 or (dist[next_pred] == dist[pred_index] + 1 and dfs(next_pred)):
                pair_pred[pred_index] = gt_index
                pair_gt[gt_index] = pred_index
                return True
        dist[pred_index] = -1
        return False

    matched = 0
    while bfs():
        for pred_index, partner in enumerate(pair_pred):
            if partner == -1 and dfs(pred_index):
                matched += 1
    return pair_pred, pair_gt, matched


def write_next_initial(pred: list[dict[str, object]], frame: int, path: Path) -> None:
    if not pred:
        raise ValueError("Cannot write initial CSV from an empty prediction")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(pred[0]["row"].keys())  # type: ignore[index, union-attr]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pred_row in pred:
            row = dict(pred_row["row"])  # type: ignore[arg-type]
            row["file"] = f"t{frame + 1:03d}.tif"
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument(
        "--gt",
        type=Path,
        default=Path("C++/config/embryo/ground_truth/Embryo_TRA_all_frames_centroids.csv"),
    )
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--next-initial", type=Path)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()

    pred = read_prediction(args.cells)
    gt = read_gt(args.gt, args.frame)
    pair_pred, pair_gt, matched = match_centers(pred, gt, args.threshold)
    matched_distances = [
        point_distance(pred[pred_index], gt[gt_index])
        for pred_index, gt_index in enumerate(pair_pred)
        if gt_index != -1
    ]
    missing = [index for index, partner in enumerate(pair_gt) if partner == -1]
    extra = [index for index, partner in enumerate(pair_pred) if partner == -1]

    mean_dist = sum(matched_distances) / len(matched_distances) if matched_distances else float("nan")
    max_dist = max(matched_distances) if matched_distances else float("nan")
    print(
        f"frame={args.frame} pred={len(pred)} gt={len(gt)} "
        f"matched<={args.threshold:g}={matched} missing={len(missing)} extra={len(extra)} "
        f"mean={mean_dist:.4f} max={max_dist:.4f}"
    )

    if args.details:
        for gt_index in missing:
            gt_row = gt[gt_index]
            nearest = min(
                (
                    point_distance(pred_row, gt_row),
                    pred_row["name"],
                    pred_row["x"],
                    pred_row["y"],
                    pred_row["z"],
                )
                for pred_row in pred
            )
            print(
                f"MISSING label={gt_row['label']} parent={gt_row['parent']} "
                f"gt=({gt_row['x']:.3f},{gt_row['y']:.3f},{gt_row['z']:.3f}) "
                f"nearest={nearest[1]} d={nearest[0]:.3f} "
                f"pred=({nearest[2]:.3f},{nearest[3]:.3f},{nearest[4]:.3f})"
            )
        for pred_index in extra:
            pred_row = pred[pred_index]
            nearest = min(
                (
                    point_distance(pred_row, gt_row),
                    gt_row["label"],
                    gt_row["parent"],
                    gt_row["x"],
                    gt_row["y"],
                    gt_row["z"],
                )
                for gt_row in gt
            )
            print(
                f"EXTRA name={pred_row['name']} "
                f"pred=({pred_row['x']:.3f},{pred_row['y']:.3f},{pred_row['z']:.3f}) "
                f"nearest_label={nearest[1]} parent={nearest[2]} d={nearest[0]:.3f} "
                f"gt=({nearest[3]:.3f},{nearest[4]:.3f},{nearest[5]:.3f})"
            )

    if args.next_initial is not None and not missing and not extra:
        write_next_initial(pred, args.frame, args.next_initial)
        print(f"next_initial={args.next_initial}")

    return 0 if not missing and not extra else 1


if __name__ == "__main__":
    raise SystemExit(main())
