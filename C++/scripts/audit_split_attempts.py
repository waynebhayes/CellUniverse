#!/usr/bin/env python3
"""Offline split attempt audit for Cell Lumen guided Cell Universe runs.

This script is deliberately outside the tracking path. It may read ground truth
to label true split parents for review, but it never writes tracking outputs or
feeds GT information back into the algorithm.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SCHEDULE_RE = re.compile(r"\[Split Schedule\] frame\s+(\d+)\s+cell=([^\s]+)")
ACCEPT_RE = re.compile(r"\[Split Accepted\]\s+([^\s]+)")


def dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def fnum_from_file(value: str) -> Optional[int]:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


def read_gt_rows(path: Path) -> Dict[int, List[dict]]:
    by_frame: Dict[int, List[dict]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            by_frame.setdefault(frame, []).append(row)
    return by_frame


def gt_point(row: dict) -> Tuple[float, float, float]:
    return (float(row["x"]), float(row["y"]), float(row["z_interp"]))


def read_prediction_rows(cells_csv: Path) -> Dict[int, List[dict]]:
    by_frame: Dict[int, List[dict]] = {}
    with cells_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("isTrash", "0").lower() in {"1", "true", "yes"}:
                continue
            frame = fnum_from_file(row.get("file", row.get("frame", "")))
            if frame is None:
                continue
            by_frame.setdefault(frame, []).append(row)
    return by_frame


def pred_point(row: dict) -> Tuple[float, float, float]:
    return (float(row["x"]), float(row["y"]), float(row["z"]))


def parse_run_log(path: Path, frame: int) -> Tuple[List[str], List[str]]:
    scheduled: List[str] = []
    accepted: List[str] = []
    current_frame: Optional[int] = None
    with path.open(errors="replace") as f:
        for line in f:
            sched = SCHEDULE_RE.search(line)
            if sched:
                current_frame = int(sched.group(1))
                if current_frame == frame:
                    scheduled.append(sched.group(2))
                continue
            if current_frame == frame:
                acc = ACCEPT_RE.search(line)
                if acc:
                    accepted.append(acc.group(1))
    return scheduled, accepted


def read_candidate_graph(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def is_real_candidate(candidate_id: str) -> bool:
    try:
        value = int(candidate_id)
    except ValueError:
        return False
    return 0 <= value < 1_000_000_000


def selected_split_by_parent(rows: Iterable[dict]) -> Dict[str, dict]:
    selected: Dict[str, dict] = {}
    for row in rows:
        if row.get("kind") != "split_pair":
            continue
        if row.get("selected") not in {"1", "true", "True"}:
            continue
        parent = row.get("parent", "")
        if parent and parent not in selected:
            selected[parent] = row
    return selected


def state_points_by_name(rows: Iterable[dict]) -> Dict[str, Tuple[float, float, float]]:
    points: Dict[str, Tuple[float, float, float]] = {}
    for row in rows:
        if row.get("kind") != "continuation":
            continue
        if row.get("source") != "current_cell_state":
            continue
        parent = row.get("parent", "")
        if not parent:
            continue
        points[parent] = (float(row["x1"]), float(row["y1"]), float(row["z1"]))
    return points


def local_density(
    name: str,
    points: Dict[str, Tuple[float, float, float]],
    radius: float,
) -> Tuple[int, float]:
    center = points.get(name)
    if center is None:
        return 0, math.inf
    nearest = math.inf
    count = 0
    for other, point in points.items():
        if other == name:
            continue
        d = dist(center, point)
        nearest = min(nearest, d)
        if d <= radius:
            count += 1
    return count, nearest


def true_split_parent_names(
    gt_by_frame: Dict[int, List[dict]],
    pred_by_frame: Dict[int, List[dict]],
    frame: int,
    match_threshold: float,
) -> Dict[str, dict]:
    daughters_by_parent: Dict[str, List[dict]] = {}
    for row in gt_by_frame.get(frame, []):
        if int(row.get("start_frame", -1)) == frame and int(row.get("parent_label", 0)) != 0:
            daughters_by_parent.setdefault(row["parent_label"], []).append(row)

    prev_gt_by_label = {
        row["label_id"]: row for row in gt_by_frame.get(frame - 1, [])
    }
    prev_pred = pred_by_frame.get(frame - 1, [])
    result: Dict[str, dict] = {}
    for parent_label, daughters in daughters_by_parent.items():
        if len(daughters) < 2 or parent_label not in prev_gt_by_label:
            continue
        parent_center = gt_point(prev_gt_by_label[parent_label])
        best_row = None
        best_dist = math.inf
        for pred in prev_pred:
            d = dist(parent_center, pred_point(pred))
            if d < best_dist:
                best_dist = d
                best_row = pred
        if best_row is not None and best_dist <= match_threshold:
            result[best_row["name"]] = {
                "parent_label": parent_label,
                "match_distance": best_dist,
                "daughter_labels": "|".join(d["label_id"] for d in daughters),
            }
    return result


def float_field(row: Optional[dict], key: str) -> str:
    if not row:
        return ""
    return row.get(key, "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--resume-cells", type=Path, default=None)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--match-threshold", type=float, default=25.0)
    parser.add_argument("--density-radius", type=float, default=45.0)
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args()

    run_log = args.run / "run.log"
    graph_csv = args.run / "candidate_graph" / f"frame_{args.frame}_candidates.csv"
    cells_csv = args.run / "cells.csv"
    if not cells_csv.exists() and args.resume_cells is None:
        raise SystemExit(f"Missing cells.csv and no --resume-cells: {cells_csv}")

    gt_by_frame = read_gt_rows(args.gt)
    pred_by_frame = read_prediction_rows(args.resume_cells or cells_csv)
    true_names = true_split_parent_names(
        gt_by_frame, pred_by_frame, args.frame, args.match_threshold
    )
    scheduled, accepted = parse_run_log(run_log, args.frame)
    graph_rows = read_candidate_graph(graph_csv)
    selected_by_parent = selected_split_by_parent(graph_rows)
    state_points = state_points_by_name(graph_rows)

    accepted_set = set(accepted)
    true_set = set(true_names)
    scheduled_set = set(scheduled)
    print(f"frame={args.frame}")
    print(f"true_split_parents={len(true_set)} scheduled={len(scheduled)} accepted={len(accepted)}")
    print(f"scheduled_true={len(scheduled_set & true_set)} scheduled_false={len(scheduled_set - true_set)}")
    print(f"accepted_true={len(accepted_set & true_set)} accepted_false={len(accepted_set - true_set)}")
    if true_set - scheduled_set:
        print("true_not_scheduled=" + ",".join(sorted(true_set - scheduled_set)))
    if accepted_set - true_set:
        print("false_accepted=" + ",".join(sorted(accepted_set - true_set)))

    out_rows: List[dict] = []
    header = [
        "frame",
        "parent",
        "true_parent",
        "accepted",
        "parent_label",
        "daughter_labels",
        "gt_parent_match_distance",
        "pair_type",
        "score",
        "raw_score",
        "sep",
        "midpoint_dist",
        "parent_shape",
        "parent_dist_balance",
        "parent_dist_near",
        "parent_dist_far",
        "vox_a",
        "vox_b",
        "signal_a",
        "signal_b",
        "window_both",
        "window_missing",
        "window_parent_persists",
        "local_neighbors",
        "nearest_neighbor",
        "note",
    ]
    for parent in scheduled:
        selected = selected_by_parent.get(parent)
        note = selected.get("note", "") if selected else ""
        candidate_a = selected.get("candidate_a", "") if selected else ""
        candidate_b = selected.get("candidate_b", "") if selected else ""
        pair_type = "two_real" if is_real_candidate(candidate_a) and is_real_candidate(candidate_b) else "parent_anchor"
        density_count, nearest = local_density(parent, state_points, args.density_radius)
        truth = true_names.get(parent, {})
        out_rows.append({
            "frame": args.frame,
            "parent": parent,
            "true_parent": int(parent in true_set),
            "accepted": int(parent in accepted_set),
            "parent_label": truth.get("parent_label", ""),
            "daughter_labels": truth.get("daughter_labels", ""),
            "gt_parent_match_distance": truth.get("match_distance", ""),
            "pair_type": pair_type,
            "score": float_field(selected, "score"),
            "raw_score": float_field(selected, "raw_score"),
            "sep": float_field(selected, "sep"),
            "midpoint_dist": float_field(selected, "midpoint_dist"),
            "parent_shape": float_field(selected, "parent_shape"),
            "parent_dist_balance": float_field(selected, "parent_dist_balance"),
            "parent_dist_near": float_field(selected, "parent_dist_near"),
            "parent_dist_far": float_field(selected, "parent_dist_far"),
            "vox_a": float_field(selected, "vox_a"),
            "vox_b": float_field(selected, "vox_b"),
            "signal_a": float_field(selected, "signal_a"),
            "signal_b": float_field(selected, "signal_b"),
            "window_both": note.split("window_both=")[-1].split(";")[0] if "window_both=" in note else "",
            "window_missing": note.split("window_missing=")[-1].split(";")[0] if "window_missing=" in note else "",
            "window_parent_persists": note.split("window_parent_persists=")[-1].split(";")[0] if "window_parent_persists=" in note else "",
            "local_neighbors": density_count,
            "nearest_neighbor": nearest if math.isfinite(nearest) else "",
            "note": note,
        })

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"csv_out={args.csv_out}")
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=header)
        writer.writeheader()
        writer.writerows(out_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
