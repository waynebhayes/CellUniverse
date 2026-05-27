#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import cv2
import numpy as np


DEFAULT_DATASET = "/Volumes/T9/🦠Cell Universe/💿Data/In_Use/Developing Drosophila Melanogaster embryo 7.08G"


def read_stack(path):
    ok, slices = cv2.imreadmulti(path, flags=cv2.IMREAD_UNCHANGED)
    if not ok or not slices:
        raise RuntimeError(f"failed to read TIFF stack: {path}")
    return np.stack(slices)


def gt_centroids(dataset_root, sequence, frame):
    path = os.path.join(dataset_root, f"{sequence}_GT", "TRA", f"man_track{frame:03d}.tif")
    stack = read_stack(path)
    flat = stack.reshape(-1)
    mask = flat > 0
    if not np.any(mask):
        return []

    labels = flat[mask].astype(np.int64)
    linear = np.nonzero(mask)[0].astype(np.int64)
    _, height, width = stack.shape
    z = linear // (height * width)
    rem = linear % (height * width)
    y = rem // width
    x = rem % width

    max_label = int(labels.max())
    counts = np.bincount(labels, minlength=max_label + 1)
    sx = np.bincount(labels, weights=x, minlength=max_label + 1)
    sy = np.bincount(labels, weights=y, minlength=max_label + 1)
    sz = np.bincount(labels, weights=z, minlength=max_label + 1)
    centers = []
    for label in np.nonzero(counts)[0]:
        count = int(counts[label])
        if label <= 0 or count <= 0:
            continue
        centers.append({
            "label": int(label),
            "x": float(sx[label] / count),
            "y": float(sy[label] / count),
            "z": float(sz[label] / count),
            "voxels": count,
        })
    return centers


def detections_from_csv(path):
    detections = []
    if not os.path.exists(path):
        return detections
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                detections.append({
                    "name": row["name"],
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "majorRadius": float(row["majorRadius"]),
                    "minorRadius": float(row["minorRadius"]),
                })
            except (KeyError, ValueError) as exc:
                raise RuntimeError(f"bad CellLumen CSV row in {path}: {row}") from exc
    return detections


def scaled_distance(a, b, z_scale):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    dz = (a["z"] - b["z"]) * z_scale
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def greedy_match(gt, detections, z_scale, threshold):
    pairs = []
    for gi, g in enumerate(gt):
        for di, d in enumerate(detections):
            pairs.append((scaled_distance(g, d, z_scale), gi, di))
    pairs.sort(key=lambda item: item[0])

    used_gt = set()
    used_det = set()
    matches = []
    for dist, gi, di in pairs:
        if dist > threshold:
            break
        if gi in used_gt or di in used_det:
            continue
        used_gt.add(gi)
        used_det.add(di)
        matches.append((dist, gt[gi], detections[di]))

    nearest = []
    for g in gt:
        if detections:
            best = min(scaled_distance(g, d, z_scale) for d in detections)
        else:
            best = float("inf")
        nearest.append(best)

    return matches, nearest


def lineage_summary(dataset_root, sequence):
    path = os.path.join(dataset_root, f"{sequence}_GT", "TRA", "man_track.txt")
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            label, start, end, parent = map(int, line.split())
            rows.append({"label": label, "start": start, "end": end, "parent": parent})

    child_counts = defaultdict(int)
    for row in rows:
        if row["parent"] != 0:
            child_counts[row["parent"]] += 1

    division_parents = {parent: count for parent, count in child_counts.items() if count >= 2}
    roots = [row for row in rows if row["parent"] == 0]
    first = min(row["start"] for row in rows)
    last = max(row["end"] for row in rows)
    frame_count = last - first + 1
    explicit_divisions = len(division_parents)
    return {
        "sequence": sequence,
        "track_rows": len(rows),
        "root_tracks": len(roots),
        "first_frame": first,
        "last_frame": last,
        "frame_count": frame_count,
        "nonzero_parent_rows": sum(1 for row in rows if row["parent"] != 0),
        "explicit_division_parents": explicit_divisions,
        "explicit_divisions_per_root": explicit_divisions / len(roots) if roots else 0.0,
        "explicit_divisions_per_root_per_frame": explicit_divisions / (len(roots) * frame_count) if roots and frame_count else 0.0,
    }


def parse_frames(text):
    frames = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = map(int, chunk.split("-", 1))
            frames.extend(range(start, end + 1))
        else:
            frames.append(int(chunk))
    return sorted(set(frames))


def evaluate(args):
    frames = parse_frames(args.frames)
    sequences = [seq.strip() for seq in args.sequences.split(",") if seq.strip()]
    thresholds = [float(value) for value in args.thresholds.split(",")]

    os.makedirs(args.report_dir, exist_ok=True)

    frame_rows = []
    lineage_rows = [lineage_summary(args.dataset_root, seq) for seq in sequences]
    for seq in sequences:
        for frame in frames:
            gt = gt_centroids(args.dataset_root, seq, frame)
            csv_path = os.path.join(args.run_dir, f"seq{seq}", f"frame{frame:03d}.csv")
            detections = detections_from_csv(csv_path)
            for threshold in thresholds:
                matches, nearest = greedy_match(gt, detections, args.z_scale, threshold)
                finite_nearest = [d for d in nearest if math.isfinite(d)]
                frame_rows.append({
                    "sequence": seq,
                    "frame": frame,
                    "threshold_scaled_px": threshold,
                    "gt_count": len(gt),
                    "detected_count": len(detections),
                    "matched_gt": len(matches),
                    "recall_on_labeled_gt": len(matches) / len(gt) if gt else 0.0,
                    "unmatched_gt": len(gt) - len(matches),
                    "median_nearest_scaled_px": float(np.median(finite_nearest)) if finite_nearest else None,
                    "p90_nearest_scaled_px": float(np.percentile(finite_nearest, 90)) if finite_nearest else None,
                })

    frame_csv = os.path.join(args.report_dir, "evaluation_frame_summary.csv")
    with open(frame_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frame_rows)

    lineage_json = os.path.join(args.report_dir, "gt_lineage_summary.json")
    with open(lineage_json, "w") as handle:
        json.dump(lineage_rows, handle, indent=2)

    print(json.dumps({
        "frame_summary_csv": frame_csv,
        "lineage_summary_json": lineage_json,
        "lineage": lineage_rows,
        "rows": frame_rows,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Drosophila CellLumen CSVs against sparse TRA center markers.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--sequences", default="01,02")
    parser.add_argument("--frames", default="0,10,20,30,40,49")
    parser.add_argument("--z-scale", type=float, default=5.0)
    parser.add_argument("--thresholds", default="8,12,16")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
