#!/usr/bin/env python3
"""Run and audit standalone CellLumen collapse experiments.

This script is deliberately evaluation-only with respect to ground truth.  It
runs CellLumen normally, then compares each completed CSV to GT so collapse
tuning can see where over-merge starts.  GT coordinates are never passed into
CellLumen and are never used to edit runtime outputs.
"""

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_INPUT_DIR = (
    "/Users/wangyiding/CellUniverse/C++/examples/input/"
    "C.elegans_developing embryo_Fluo-N3DH-CE_Training/01"
)
DEFAULT_GT = (
    "/Users/wangyiding/CellUniverse/C++/config/C.elegans developing embryo/"
    "C.elegans_initial/ground_truth/embryo_FixedGroundTruth.csv"
)
DEFAULT_INITIAL_PRIOR = (
    "/Users/wangyiding/CellUniverse/C++/config/C.elegans developing embryo/"
    "C.elegans_initial/initial_files/00_core_start_points/initial_embryo_0.csv"
)
DEFAULT_EXE = "/Users/wangyiding/CellUniverse/C++/build/celluniverse"


COLLAPSE_RE = re.compile(r"\[CellLumen ClusterCentroidCollapse\](?P<body>.*)")


def parse_frame_spec(text):
    frames = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            frames.extend(range(int(start), int(end) + 1))
        else:
            frames.append(int(chunk))
    return sorted(set(frames))


def read_gt(path, frames):
    wanted = set(frames)
    by_frame = {frame: [] for frame in frames}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if frame not in wanted:
                continue
            by_frame[frame].append(
                {
                    "label": row["label_id"],
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z_interp"]),
                }
            )
    return by_frame


def read_predictions(path, pred_z_scale):
    rows = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "name": row["name"],
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]) * pred_z_scale,
                    "major": float(row.get("majorRadius", 0.0) or 0.0),
                    "minor": float(row.get("minorRadius", 0.0) or 0.0),
                }
            )
    return rows


def dist(a, b):
    return math.sqrt(
        (a["x"] - b["x"]) ** 2
        + (a["y"] - b["y"]) ** 2
        + (a["z"] - b["z"]) ** 2
    )


def greedy_match(gt_rows, pred_rows, threshold):
    pairs = []
    for gi, gt in enumerate(gt_rows):
        for pi, pred in enumerate(pred_rows):
            pairs.append((dist(gt, pred), gi, pi))
    pairs.sort(key=lambda item: item[0])

    used_gt = set()
    used_pred = set()
    matches = []
    for distance, gi, pi in pairs:
        if distance > threshold:
            break
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((distance, gt_rows[gi], pred_rows[pi]))

    missing = [gt_rows[i] for i in range(len(gt_rows)) if i not in used_gt]
    extra = [pred_rows[i] for i in range(len(pred_rows)) if i not in used_pred]
    return matches, missing, extra


def nearest_prediction(gt, pred_rows):
    best = None
    for pred in pred_rows:
        distance = dist(gt, pred)
        if best is None or distance < best[0]:
            best = (distance, pred)
    return best


def nearest_gt_neighbor(gt, gt_rows):
    best = None
    for other in gt_rows:
        if other is gt:
            continue
        distance = dist(gt, other)
        if best is None or distance < best[0]:
            best = (distance, other)
    return best


def count_gt_neighbors(gt, gt_rows, radius):
    return sum(1 for other in gt_rows if other is not gt and dist(gt, other) <= radius)


def parse_key_values(text):
    values = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key.strip()] = value.strip().strip('"')
    return values


def read_collapse_stats(log_path):
    latest = {}
    path = Path(log_path)
    if not path.exists():
        return latest
    with path.open(errors="replace") as handle:
        for line in handle:
            match = COLLAPSE_RE.search(line)
            if match:
                latest = parse_key_values(match.group("body"))
    return latest


def write_csv(path, fieldnames, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_cell_lumen(args, frame, frame_dir, csv_path, log_path):
    input_frame = Path(args.input_dir) / f"t{frame:03d}.tif"
    cmd = [
        args.executable,
        "--cell-lumen",
        str(input_frame),
        str(frame_dir),
        args.config,
        str(csv_path),
    ]
    if args.initial_prior:
        cmd.append(args.initial_prior)

    env = os.environ.copy()
    env["CELLUNIVERSE_CELL_LUMEN_SKIP_TIFF"] = "1"
    start = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    return proc.returncode, time.time() - start


def main():
    parser = argparse.ArgumentParser(
        description="Run standalone CellLumen and audit collapse over-merge frames."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--frames", default="0-85")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--gt-csv", default=DEFAULT_GT)
    parser.add_argument("--initial-prior", default=DEFAULT_INITIAL_PRIOR)
    parser.add_argument("--executable", default=DEFAULT_EXE)
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--pred-z-scale", type=float, default=7.0)
    parser.add_argument("--summary-csv", default="")
    parser.add_argument("--missing-audit-csv", default="")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    frames = parse_frame_spec(args.frames)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    gt_by_frame = read_gt(args.gt_csv, frames)
    summary_rows = []
    missing_rows = []

    for frame in frames:
        frame_dir = run_dir / f"f{frame:03d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        csv_path = frame_dir / f"cell_lumen_f{frame:03d}.csv"
        log_path = frame_dir / "run.log"

        elapsed = 0.0
        return_code = 0
        if not (args.reuse_existing and csv_path.exists()):
            return_code, elapsed = run_cell_lumen(args, frame, frame_dir, csv_path, log_path)
        if return_code != 0 or not csv_path.exists():
            print(
                f"f{frame:03d} RUN_FAIL return_code={return_code} csv={csv_path}",
                flush=True,
            )
            summary_rows.append(
                {
                    "frame": frame,
                    "status": "RUN_FAIL",
                    "gt": len(gt_by_frame.get(frame, [])),
                    "pred": 0,
                    "matched": 0,
                    "missing": len(gt_by_frame.get(frame, [])),
                    "extra": 0,
                    "max_distance": "",
                    "mean_distance": "",
                    "seconds": f"{elapsed:.2f}",
                    "collapse_before": "",
                    "collapse_after": "",
                    "collapse_merged_groups": "",
                    "collapse_valley_guarded_pairs": "",
                    "collapse_diameter_guarded_groups": "",
                    "collapse_link_distance": "",
                }
            )
            continue

        gt_rows = gt_by_frame.get(frame, [])
        pred_rows = read_predictions(csv_path, args.pred_z_scale)
        matches, missing, extra = greedy_match(gt_rows, pred_rows, args.threshold)
        max_distance = max((item[0] for item in matches), default=0.0)
        mean_distance = sum(item[0] for item in matches) / len(matches) if matches else 0.0
        status = "PASS" if not missing and not extra else "NO_MISS" if not missing else "MISS"
        collapse = read_collapse_stats(log_path)
        print(
            f"f{frame:03d} {status} GT={len(gt_rows)} pred={len(pred_rows)} "
            f"miss={len(missing)} extra={len(extra)} "
            f"collapse={collapse.get('before', '?')}->{collapse.get('after', '?')} "
            f"valley_guarded={collapse.get('valley_guarded_pairs', '?')} "
            f"diameter_guarded={collapse.get('diameter_guarded_groups', '?')} "
            f"sec={elapsed:.1f}",
            flush=True,
        )

        summary_rows.append(
            {
                "frame": frame,
                "status": status,
                "gt": len(gt_rows),
                "pred": len(pred_rows),
                "matched": len(matches),
                "missing": len(missing),
                "extra": len(extra),
                "max_distance": f"{max_distance:.6f}",
                "mean_distance": f"{mean_distance:.6f}",
                "seconds": f"{elapsed:.2f}",
                "collapse_before": collapse.get("before", ""),
                "collapse_after": collapse.get("after", ""),
                "collapse_merged_groups": collapse.get("merged_groups", ""),
                "collapse_valley_guarded_pairs": collapse.get("valley_guarded_pairs", ""),
                "collapse_diameter_guarded_groups": collapse.get("diameter_guarded_groups", ""),
                "collapse_link_distance": collapse.get("link_distance", ""),
            }
        )

        nearest_pred_by_label = {}
        for gt in missing:
            nearest = nearest_prediction(gt, pred_rows)
            nearest_pred_by_label[gt["label"]] = nearest[1]["name"] if nearest else ""
        shared_nearest_counts = {
            name: list(nearest_pred_by_label.values()).count(name)
            for name in set(nearest_pred_by_label.values())
        }

        for gt in missing:
            nearest_pred = nearest_prediction(gt, pred_rows)
            nearest_gt = nearest_gt_neighbor(gt, gt_rows)
            nearest_name = nearest_pred[1]["name"] if nearest_pred else ""
            missing_rows.append(
                {
                    "frame": frame,
                    "label": gt["label"],
                    "gt_x": f"{gt['x']:.3f}",
                    "gt_y": f"{gt['y']:.3f}",
                    "gt_z": f"{gt['z']:.3f}",
                    "nearest_pred": nearest_name,
                    "nearest_pred_distance": f"{nearest_pred[0]:.3f}" if nearest_pred else "",
                    "nearest_gt_label": nearest_gt[1]["label"] if nearest_gt else "",
                    "nearest_gt_distance": f"{nearest_gt[0]:.3f}" if nearest_gt else "",
                    "gt_neighbors_within_35": count_gt_neighbors(gt, gt_rows, 35.0),
                    "gt_neighbors_within_45": count_gt_neighbors(gt, gt_rows, 45.0),
                    "gt_neighbors_within_60": count_gt_neighbors(gt, gt_rows, 60.0),
                    "missing_share_same_nearest_pred": shared_nearest_counts.get(nearest_name, 0),
                    "collapse_before": collapse.get("before", ""),
                    "collapse_after": collapse.get("after", ""),
                    "collapse_merged_groups": collapse.get("merged_groups", ""),
                    "collapse_valley_guarded_pairs": collapse.get("valley_guarded_pairs", ""),
                    "collapse_diameter_guarded_groups": collapse.get("diameter_guarded_groups", ""),
                    "collapse_link_distance": collapse.get("link_distance", ""),
                }
            )

        if args.summary_csv:
            write_csv(args.summary_csv, list(summary_rows[0].keys()), summary_rows)
        if args.missing_audit_csv and missing_rows:
            write_csv(args.missing_audit_csv, list(missing_rows[0].keys()), missing_rows)

    if args.summary_csv and summary_rows:
        write_csv(args.summary_csv, list(summary_rows[0].keys()), summary_rows)
    if args.missing_audit_csv:
        fieldnames = [
            "frame",
            "label",
            "gt_x",
            "gt_y",
            "gt_z",
            "nearest_pred",
            "nearest_pred_distance",
            "nearest_gt_label",
            "nearest_gt_distance",
            "gt_neighbors_within_35",
            "gt_neighbors_within_45",
            "gt_neighbors_within_60",
            "missing_share_same_nearest_pred",
            "collapse_before",
            "collapse_after",
            "collapse_merged_groups",
            "collapse_valley_guarded_pairs",
            "collapse_diameter_guarded_groups",
            "collapse_link_distance",
        ]
        write_csv(args.missing_audit_csv, fieldnames, missing_rows)

    total_missing = sum(int(row["missing"]) for row in summary_rows)
    total_extra = sum(int(row["extra"]) for row in summary_rows)
    perfect = sum(1 for row in summary_rows if row["missing"] == 0 and row["extra"] == 0)
    no_miss = sum(1 for row in summary_rows if row["missing"] == 0)
    print(
        f"SUMMARY frames={len(summary_rows)} perfect={perfect} no_miss={no_miss} "
        f"total_missing={total_missing} total_extra={total_extra}",
        flush=True,
    )
    return 1 if total_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
