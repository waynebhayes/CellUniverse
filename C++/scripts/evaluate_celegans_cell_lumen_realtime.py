#!/usr/bin/env python3
"""Real-time CellLumen CSV evaluator for the C. elegans embryo dataset.

This script is intentionally GT-read-only. It never writes or adjusts runtime
outputs. Its job is to make tuning loops faster by printing the miss and extra
status as soon as each standalone CellLumen CSV appears.
"""

import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path


DEFAULT_GT = (
    "/Users/wangyiding/CellUniverse/C++/config/C.elegans developing embryo/"
    "C.elegans_initial/ground_truth/embryo_FixedGroundTruth.csv"
)


CELL_LOG_RE = re.compile(
    r"\[Cell\]\s+name=(?P<name>\S+).*?"
    r"vox=(?P<vox>\d+).*?"
    r"meanI=(?P<mean>[-+0-9.eE]+).*?"
    r"top10MinusShell=(?P<signal>[-+0-9.eE]+)"
)


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


def format_template(template, run_dir, frame):
    frame03 = f"{frame:03d}"
    return Path(
        template.format(
            run_dir=str(run_dir),
            frame=frame,
            frame03=frame03,
        )
    )


def read_gt(gt_csv, frames):
    wanted = set(frames)
    by_frame = {frame: [] for frame in frames}
    with Path(gt_csv).open(newline="") as handle:
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


def read_predictions(csv_path, pred_z_scale):
    rows = []
    with Path(csv_path).open(newline="") as handle:
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


def read_log_stats(log_path):
    stats = {}
    path = Path(log_path)
    if not path.exists():
        return stats
    with path.open(errors="replace") as handle:
        for line in handle:
            match = CELL_LOG_RE.search(line)
            if not match:
                continue
            stats[match.group("name")] = {
                "vox": int(match.group("vox")),
                "mean": float(match.group("mean")),
                "signal": float(match.group("signal")),
            }
    return stats


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


def describe_extra(extra_rows, log_stats, limit):
    parts = []
    for row in extra_rows[:limit]:
        suffix = ""
        if row["name"] in log_stats:
            stat = log_stats[row["name"]]
            suffix = (
                f":vox={stat['vox']},mean={stat['mean']:.1f},"
                f"signal={stat['signal']:.1f}"
            )
        parts.append(f"{row['name']}{suffix}")
    if len(extra_rows) > limit:
        parts.append(f"...+{len(extra_rows) - limit}")
    return ",".join(parts)


def evaluate_frame(frame, gt_rows, csv_path, log_path, args):
    pred_rows = read_predictions(csv_path, args.pred_z_scale)
    matches, missing, extra = greedy_match(gt_rows, pred_rows, args.threshold)
    max_distance = max((item[0] for item in matches), default=0.0)
    mean_distance = (
        sum(item[0] for item in matches) / len(matches) if matches else 0.0
    )
    status = "PASS" if not missing and not extra else "NO_MISS" if not missing else "MISS"
    line = (
        f"f{frame:03d} {status} "
        f"GT={len(gt_rows)} pred={len(pred_rows)} match={len(matches)} "
        f"miss={len(missing)} extra={len(extra)} "
        f"max={max_distance:.2f} mean={mean_distance:.2f}"
    )
    if missing:
        line += " missing=" + ",".join(row["label"] for row in missing)
    if extra:
        log_stats = read_log_stats(log_path)
        line += " extra=" + describe_extra(extra, log_stats, args.extra_detail_limit)
    print(line, flush=True)
    return {
        "frame": frame,
        "gt": len(gt_rows),
        "pred": len(pred_rows),
        "matched": len(matches),
        "missing": len(missing),
        "extra": len(extra),
        "max_distance": max_distance,
        "mean_distance": mean_distance,
        "status": status,
    }


def write_summary(path, rows):
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame",
        "status",
        "gt",
        "pred",
        "matched",
        "missing",
        "extra",
        "max_distance",
        "mean_distance",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate C. elegans CellLumen CSVs as soon as each frame is written."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--frames", default="0-85")
    parser.add_argument("--gt-csv", default=DEFAULT_GT)
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--pred-z-scale", type=float, default=7.0)
    parser.add_argument(
        "--csv-template",
        default="{run_dir}/f{frame03}/cell_lumen_f{frame03}.csv",
    )
    parser.add_argument(
        "--log-template",
        default="{run_dir}/f{frame03}/run.log",
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-sec", type=float, default=1.0)
    parser.add_argument("--timeout-sec", type=float, default=0.0)
    parser.add_argument("--summary-csv", default="")
    parser.add_argument("--extra-detail-limit", type=int, default=8)
    args = parser.parse_args()

    frames = parse_frame_spec(args.frames)
    run_dir = Path(args.run_dir)
    gt_by_frame = read_gt(args.gt_csv, frames)
    started = time.time()
    rows = []

    for frame in frames:
        csv_path = format_template(args.csv_template, run_dir, frame)
        log_path = format_template(args.log_template, run_dir, frame)
        while not csv_path.exists():
            if not args.watch:
                print(f"f{frame:03d} WAIT missing_csv={csv_path}", flush=True)
                break
            if args.timeout_sec > 0 and time.time() - started > args.timeout_sec:
                print(f"timeout waiting for f{frame:03d}: {csv_path}", file=sys.stderr)
                write_summary(args.summary_csv, rows)
                return 2
            time.sleep(args.poll_sec)
        if not csv_path.exists():
            continue
        rows.append(evaluate_frame(frame, gt_by_frame.get(frame, []), csv_path, log_path, args))
        write_summary(args.summary_csv, rows)

    total_missing = sum(row["missing"] for row in rows)
    total_extra = sum(row["extra"] for row in rows)
    perfect = sum(1 for row in rows if row["missing"] == 0 and row["extra"] == 0)
    no_miss = sum(1 for row in rows if row["missing"] == 0)
    print(
        f"SUMMARY frames={len(rows)} perfect={perfect} no_miss={no_miss} "
        f"total_missing={total_missing} total_extra={total_extra}",
        flush=True,
    )
    write_summary(args.summary_csv, rows)
    return 1 if total_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
