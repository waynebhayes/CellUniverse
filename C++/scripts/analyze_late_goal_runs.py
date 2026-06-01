#!/usr/bin/env python3
"""Summarize late embryo CellUniverse tuning runs without rerunning C++.

The script reads each run directory under the 170~194 goal output root, compares
saved cells.csv files to GT centers, and extracts high-signal timing and failure
markers from run.log. It is intentionally offline: no images are loaded and no
CellUniverse binary is executed.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict, deque
from pathlib import Path


def read_gt(path: Path, start: int, end: int) -> dict[int, list[dict[str, object]]]:
    gt: dict[int, list[dict[str, object]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if start <= frame <= end:
                gt[frame].append(
                    {
                        "label": row["label_id"],
                        "parent": row["parent_label"],
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "z": float(row["z_interp"]),
                    }
                )
    return dict(gt)


def read_predictions(path: Path, start: int, end: int) -> dict[int, list[dict[str, object]]]:
    by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            file_name = row.get("file", "")
            if not (file_name.startswith("t") and file_name.endswith(".tif")):
                continue
            frame = int(file_name[1:4])
            if not (start <= frame <= end):
                continue
            if str(row.get("isTrash", "0")).lower() in {"1", "true", "yes"}:
                continue
            by_frame[frame].append(
                {
                    "name": row.get("name", ""),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                }
            )
    return dict(by_frame)


def point_distance(a: dict[str, object], b: dict[str, object]) -> float:
    return math.dist(
        (float(a["x"]), float(a["y"]), float(a["z"])),
        (float(b["x"]), float(b["y"]), float(b["z"])),
    )


def match_centers(
    pred: list[dict[str, object]],
    gt: list[dict[str, object]],
    threshold: float,
) -> tuple[list[int], list[int], int, float, float]:
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
            if next_pred == -1 or (
                dist[next_pred] == dist[pred_index] + 1 and dfs(next_pred)
            ):
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

    matched_distances = [
        point_distance(pred[pred_index], gt[gt_index])
        for pred_index, gt_index in enumerate(pair_pred)
        if gt_index != -1
    ]
    mean_dist = (
        sum(matched_distances) / len(matched_distances)
        if matched_distances
        else float("nan")
    )
    max_dist = max(matched_distances) if matched_distances else float("nan")
    return pair_pred, pair_gt, matched, mean_dist, max_dist


STAGE_RE = re.compile(
    r"\[Stage Timing\] frame (?P<frame>\d+) stage=(?P<stage>\S+) seconds=(?P<seconds>[0-9.eE+-]+)"
)
OPT_DONE_RE = re.compile(
    r"\[Optimize Done\] frame (?P<frame>\d+).*?split_attempts=(?P<attempts>\d+).*?"
    r"split_accepted=(?P<accepted>\d+) final_cells=(?P<cells>\d+)"
)
FUSION_RE = re.compile(
    r"\[CellLumen Fusion Result\] frame=t(?P<frame>\d+)\.tif detected_cells=(?P<detected>\d+) "
    r"total_sec=(?P<seconds>[0-9.eE+-]+)"
)


def parse_log(path: Path) -> dict[int, dict[str, object]]:
    info: dict[int, dict[str, object]] = defaultdict(
        lambda: {
            "stages": defaultdict(float),
            "heavy_splits": 0,
            "max_split_sec": 0.0,
            "split_attempts": "",
            "split_accepted": "",
            "final_cells": "",
            "fusion_detected": "",
            "fusion_sec": "",
            "prune_events": [],
            "reject_reasons": defaultdict(int),
        }
    )
    if not path.exists():
        return {}
    with path.open(errors="replace") as handle:
        for line in handle:
            if match := STAGE_RE.search(line):
                frame = int(match.group("frame"))
                stage = match.group("stage")
                seconds = float(match.group("seconds"))
                frame_info = info[frame]
                frame_info["stages"][stage] += seconds  # type: ignore[index]
                if stage == "split_attempt_cell_lumen_prior":
                    if seconds >= 10.0:
                        frame_info["heavy_splits"] = int(frame_info["heavy_splits"]) + 1
                    frame_info["max_split_sec"] = max(
                        float(frame_info["max_split_sec"]), seconds
                    )
                continue
            if match := OPT_DONE_RE.search(line):
                frame_info = info[int(match.group("frame"))]
                frame_info["split_attempts"] = match.group("attempts")
                frame_info["split_accepted"] = match.group("accepted")
                frame_info["final_cells"] = match.group("cells")
                continue
            if match := FUSION_RE.search(line):
                frame_info = info[int(match.group("frame"))]
                frame_info["fusion_detected"] = match.group("detected")
                frame_info["fusion_sec"] = match.group("seconds")
                continue
            if "Stale Continuation" in line and "frame " in line:
                frame_match = re.search(r"frame (?P<frame>\d+)", line)
                if frame_match:
                    event = line.strip()
                    info[int(frame_match.group("frame"))]["prune_events"].append(event)  # type: ignore[index]
                continue
            if "[Split Reject" in line:
                frame_match = re.search(r"frame (?P<frame>\d+)", line)
                if not frame_match:
                    continue
                reason_match = re.search(r"reason=([A-Za-z0-9_\-]+)", line)
                reason = reason_match.group(1) if reason_match else "unknown"
                info[int(frame_match.group("frame"))]["reject_reasons"][reason] += 1  # type: ignore[index]
    return dict(info)


def nearest_summary(
    unmatched: list[int],
    source: list[dict[str, object]],
    target: list[dict[str, object]],
    source_kind: str,
) -> str:
    parts: list[str] = []
    for index in unmatched[:3]:
        row = source[index]
        if not target:
            parts.append(f"{source_kind}={row.get('name') or row.get('label')} nearest=none")
            continue
        best = min(
            (point_distance(row, other), other)
            for other in target
        )
        other = best[1]
        parts.append(
            f"{source_kind}={row.get('name') or row.get('label')} "
            f"nearest={other.get('name') or other.get('label')} d={best[0]:.2f}"
        )
    return "; ".join(parts)


def summarize_run(
    run_dir: Path,
    gt_by_frame: dict[int, list[dict[str, object]]],
    start: int,
    end: int,
    threshold: float,
) -> list[dict[str, object]]:
    pred_by_frame = read_predictions(run_dir / "cells.csv", start, end)
    log_by_frame = parse_log(run_dir / "run.log")
    frames = sorted(set(pred_by_frame) | set(log_by_frame))
    rows: list[dict[str, object]] = []
    if not frames:
        return [
            {
                "run": run_dir.name,
                "frame": "",
                "status": "NO_FRAME_OUTPUT",
                "pred": "",
                "gt": "",
                "matched": "",
                "missing": "",
                "extra": "",
                "mean_dist": "",
                "max_dist": "",
                "fusion_detected": "",
                "fusion_sec": "",
                "optimize_total": "",
                "phase_split_and_perturb": "",
                "pre_pass_image_grounded_pca": "",
                "final_pca_shape": "",
                "heavy_splits": "",
                "max_split_sec": "",
                "split_attempts": "",
                "split_accepted": "",
                "failure_detail": "no cells.csv frames or parsed run.log frames",
                "prune_events": "",
            }
        ]
    for frame in frames:
        pred = pred_by_frame.get(frame, [])
        gt = gt_by_frame.get(frame, [])
        info = log_by_frame.get(frame, {})
        stages = info.get("stages", {}) if info else {}
        if pred and gt:
            pair_pred, pair_gt, matched, mean_dist, max_dist = match_centers(
                pred, gt, threshold
            )
            missing_indices = [i for i, partner in enumerate(pair_gt) if partner == -1]
            extra_indices = [i for i, partner in enumerate(pair_pred) if partner == -1]
            missing = len(missing_indices)
            extra = len(extra_indices)
            status = "PASS" if missing == 0 and extra == 0 else "FAIL"
            failure = ""
            if status == "FAIL":
                failure = " | ".join(
                    part
                    for part in [
                        nearest_summary(missing_indices, gt, pred, "missing_label"),
                        nearest_summary(extra_indices, pred, gt, "extra_cell"),
                    ]
                    if part
                )
        else:
            matched = 0
            missing = len(gt)
            extra = len(pred)
            mean_dist = float("nan")
            max_dist = float("nan")
            status = "NO_GT_OR_PRED"
            failure = "missing predictions or GT"
        prune_events = info.get("prune_events", []) if info else []
        rows.append(
            {
                "run": run_dir.name,
                "frame": frame,
                "status": status,
                "pred": len(pred),
                "gt": len(gt),
                "matched": matched,
                "missing": missing,
                "extra": extra,
                "mean_dist": f"{mean_dist:.4f}",
                "max_dist": f"{max_dist:.4f}",
                "fusion_detected": info.get("fusion_detected", "") if info else "",
                "fusion_sec": info.get("fusion_sec", "") if info else "",
                "optimize_total": f"{float(stages.get('optimize_total', 0.0)):.3f}",
                "phase_split_and_perturb": f"{float(stages.get('phase_split_and_perturb', 0.0)):.3f}",
                "pre_pass_image_grounded_pca": f"{float(stages.get('pre_pass_image_grounded_pca', 0.0)):.3f}",
                "final_pca_shape": f"{float(stages.get('final_pca_shape', 0.0)):.3f}",
                "heavy_splits": info.get("heavy_splits", "") if info else "",
                "max_split_sec": f"{float(info.get('max_split_sec', 0.0)):.3f}" if info else "",
                "split_attempts": info.get("split_attempts", "") if info else "",
                "split_accepted": info.get("split_accepted", "") if info else "",
                "failure_detail": failure,
                "prune_events": " || ".join(prune_events[:8]),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("C++/output/F170-194_Goal0531"),
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=Path("C++/config/embryo/ground_truth/Embryo_TRA_all_frames_centroids.csv"),
    )
    parser.add_argument("--start", type=int, default=170)
    parser.add_argument("--end", type=int, default=194)
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("C++/output/F170-194_Goal0531/_analysis_20260601/late_goal_run_report.tsv"),
    )
    args = parser.parse_args()

    gt_by_frame = read_gt(args.gt, args.start, args.end)
    run_dirs = [
        path
        for path in sorted(args.root.glob("*/*"), key=lambda p: str(p))
        if path.is_dir() and path.name != "_analysis_20260601"
    ]
    all_rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        all_rows.extend(
            summarize_run(run_dir, gt_by_frame, args.start, args.end, args.threshold)
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "frame",
        "status",
        "pred",
        "gt",
        "matched",
        "missing",
        "extra",
        "mean_dist",
        "max_dist",
        "fusion_detected",
        "fusion_sec",
        "optimize_total",
        "phase_split_and_perturb",
        "pre_pass_image_grounded_pca",
        "final_pca_shape",
        "heavy_splits",
        "max_split_sec",
        "split_attempts",
        "split_accepted",
        "failure_detail",
        "prune_events",
    ]
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)

    pass_frames = sorted({int(row["frame"]) for row in all_rows if row["status"] == "PASS"})
    fail_rows = [row for row in all_rows if row["status"] == "FAIL"]
    print(f"report={args.out}")
    print(f"runs={len(run_dirs)} rows={len(all_rows)}")
    print(f"pass_frames={','.join(map(str, pass_frames)) if pass_frames else '-'}")
    print(f"fail_rows={len(fail_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
