#!/usr/bin/env python3
"""Watch a CellUniverse run and validate saved frames against GT centers.

This script is an offline reviewer only. It waits for each checkpoint, reads
the saved cells.csv for that frame, compares centers to the fixed GT CSV, and
prints a compact PASS or FAIL line. It never writes algorithm inputs and never
feeds GT information back into CellUniverse.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from validate_embryo_centers import (  # noqa: E402
    match_centers,
    point_distance,
    read_gt,
    read_prediction,
)


DEFAULT_GT = (
    "/Users/wangyiding/CellUniverse/C++/config/C.elegans developing embryo/"
    "C.elegans_initial/ground_truth/embryo_FixedGroundTruth.csv"
)


def parse_frames(text: str) -> list[int]:
    frames: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            frames.extend(range(int(start), int(end) + 1))
        else:
            frames.append(int(part))
    return sorted(set(frames))


def checkpoint_path(run_dir: Path, frame: int) -> Path:
    return run_dir / "checkpoints" / f"frame_{frame:03d}.txt"


def evaluate_frame(cells_csv: Path, gt_csv: Path, frame: int, threshold: float) -> dict[str, object]:
    pred = read_prediction(cells_csv, frame)
    gt = read_gt(gt_csv, frame)
    pair_pred, pair_gt, matched = match_centers(pred, gt, threshold)
    distances = [
        point_distance(pred[pred_index], gt[gt_index])
        for pred_index, gt_index in enumerate(pair_pred)
        if gt_index != -1
    ]
    missing = [index for index, partner in enumerate(pair_gt) if partner == -1]
    extra = [index for index, partner in enumerate(pair_pred) if partner == -1]
    max_dist = max(distances) if distances else 0.0
    mean_dist = sum(distances) / len(distances) if distances else 0.0
    status = "PASS" if not missing and not extra else "FAIL"
    detail = ""
    if missing:
        labels = ",".join(str(gt[index]["label"]) for index in missing[:8])
        if len(missing) > 8:
            labels += f",...+{len(missing) - 8}"
        detail += f" missing_labels={labels}"
    if extra:
        names = ",".join(str(pred[index]["name"]) for index in extra[:8])
        if len(extra) > 8:
            names += f",...+{len(extra) - 8}"
        detail += f" extra_names={names}"

    print(
        f"f{frame:03d} {status} pred={len(pred)} gt={len(gt)} "
        f"matched={matched} missing={len(missing)} extra={len(extra)} "
        f"mean={mean_dist:.3f} max={max_dist:.3f}{detail}",
        flush=True,
    )
    return {
        "frame": frame,
        "status": status,
        "pred": len(pred),
        "gt": len(gt),
        "matched": matched,
        "missing": len(missing),
        "extra": len(extra),
        "mean_distance": f"{mean_dist:.6f}",
        "max_distance": f"{max_dist:.6f}",
        "missing_labels": ",".join(str(gt[index]["label"]) for index in missing),
        "extra_names": ",".join(str(pred[index]["name"]) for index in extra),
    }


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame",
        "status",
        "pred",
        "gt",
        "matched",
        "missing",
        "extra",
        "mean_distance",
        "max_distance",
        "missing_labels",
        "extra_names",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_failure_audit(run_dir: Path, row: dict[str, object]) -> None:
    """Write an offline audit note for validation failures.

    This records only review evidence. It does not write CellUniverse inputs or
    feed GT data back into the tracker.
    """
    csv_path = run_dir / "failure_audit.csv"
    md_path = run_dir / "failure_audit.md"
    fieldnames = [
        "frame",
        "status",
        "pred",
        "gt",
        "matched",
        "missing",
        "extra",
        "mean_distance",
        "max_distance",
        "missing_labels",
        "extra_names",
        "checkpoint",
        "run_log",
        "search_hint",
    ]
    frame = int(row["frame"])
    checkpoint = checkpoint_path(run_dir, frame)
    run_log = run_dir / "run.log"
    search_terms: list[str] = [f"frame={frame}", f"frame {frame}"]
    if row.get("missing_labels"):
        search_terms.extend(str(row["missing_labels"]).split(","))
    if row.get("extra_names"):
        search_terms.extend(str(row["extra_names"]).split(","))
    audit_row = dict(row)
    audit_row["checkpoint"] = str(checkpoint)
    audit_row["run_log"] = str(run_log)
    audit_row["search_hint"] = "|".join(term for term in search_terms if term)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: audit_row.get(name, "") for name in fieldnames})

    with md_path.open("a") as handle:
        handle.write(f"\n## f{frame:03d} validation failure\n")
        handle.write(
            f"Prediction {row['pred']} vs GT {row['gt']}; "
            f"missing {row['missing']}; extra {row['extra']}; "
            f"mean distance {row['mean_distance']}; "
            f"max distance {row['max_distance']}.\n\n"
        )
        if row.get("missing_labels"):
            handle.write(f"Missing GT labels: {row['missing_labels']}\n\n")
        if row.get("extra_names"):
            handle.write(f"Extra prediction names: {row['extra_names']}\n\n")
        handle.write(f"Checkpoint: `{checkpoint}`\n\n")
        handle.write(f"Run log: `{run_log}`\n\n")
        handle.write(
            "Suggested log search terms: "
            f"`{audit_row['search_hint']}`\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--frames", default="0-85")
    parser.add_argument("--gt", type=Path, default=Path(DEFAULT_GT))
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-sec", type=float, default=2.0)
    parser.add_argument("--timeout-sec", type=float, default=0.0)
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--summary-csv", type=Path)
    args = parser.parse_args()

    frames = parse_frames(args.frames)
    cells_csv = args.run_dir / "cells.csv"
    started = time.time()
    rows: list[dict[str, object]] = []

    for frame in frames:
        checkpoint = checkpoint_path(args.run_dir, frame)
        while args.watch and not checkpoint.exists():
            if args.timeout_sec > 0 and time.time() - started > args.timeout_sec:
                print(f"timeout waiting for f{frame:03d}: {checkpoint}", file=sys.stderr)
                if args.summary_csv:
                    write_summary(args.summary_csv, rows)
                return 2
            time.sleep(args.poll_sec)
        if not checkpoint.exists():
            print(f"f{frame:03d} WAIT missing_checkpoint={checkpoint}", flush=True)
            continue
        row = evaluate_frame(cells_csv, args.gt, frame, args.threshold)
        rows.append(row)
        if args.summary_csv:
            write_summary(args.summary_csv, rows)
        if args.stop_on_fail and row["status"] != "PASS":
            append_failure_audit(args.run_dir, row)
            return 1
        if row["status"] != "PASS":
            append_failure_audit(args.run_dir, row)

    return 0 if all(row["status"] == "PASS" for row in rows) and len(rows) == len(frames) else 1


if __name__ == "__main__":
    raise SystemExit(main())
