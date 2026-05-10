#!/usr/bin/env python3
"""Fair raw-luminosity test for Professor Hayes's constant-luminosity hypothesis.

This script intentionally does not use the thresholded connected-component mask as
its main luminosity measurement. It reads the raw TIFF stacks again, measures raw
light in fixed 3D windows and adaptive ellipsoid masks, subtracts local background
from a surrounding shell, and compares parent luminosity before split against the
sum of daughter luminosities after split.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class RunConfig:
    name: str
    analysis_dir: Path
    input_dir: Path
    config_file: Path


@dataclass
class Measurement:
    bg: float
    voxels: int
    raw_sum: float
    bgsub_luminosity: float
    bgsub_mean: float
    radius_xy: float
    radius_z: float


@dataclass
class SplitEvidence:
    run_name: str
    parent_id: str
    split_frame: int
    child1_id: str
    child2_id: str
    pre_window_count: int
    parent_frame: int
    child_frame: int
    volume_first: float
    volume_last: float
    fixed_mean_first: float
    fixed_mean_last: float
    ellipsoid_mean_first: float
    ellipsoid_mean_last: float
    fixed_pre_cv: float
    ellipsoid_pre_cv: float
    fixed_parent_lum: float
    fixed_daughter_sum: float
    fixed_daughter_parent_ratio: float
    ellipsoid_parent_lum: float
    ellipsoid_daughter_sum: float
    ellipsoid_daughter_parent_ratio: float
    fixed_supports_hypothesis: bool
    ellipsoid_supports_hypothesis: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        help="Run spec: name=analysis_dir. May be passed multiple times.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pre-split-frames", type=int, default=5)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def parse_run_spec(spec: str, input_dir: Path, config_file: Path) -> RunConfig:
    if "=" not in spec:
        raise ValueError(f"Run spec must be name=analysis_dir, got: {spec}")
    name, path = spec.split("=", 1)
    return RunConfig(name=name.strip(), analysis_dir=Path(path).expanduser(), input_dir=input_dir, config_file=config_file)


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_float(row: dict, key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def parse_int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def parse_z_scaling(config_file: Path) -> float:
    if not config_file.exists():
        return 7.0
    text = config_file.read_text(errors="ignore")
    match = re.search(r"^\s*z_scaling\s*:\s*([0-9.]+)\s*$", text, flags=re.MULTILINE)
    if not match:
        return 7.0
    value = float(match.group(1))
    return value if value > 0 else 7.0


def load_raw_volume(path: Path) -> np.ndarray:
    ok, pages = cv2.imreadmulti(str(path), flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if not ok or not pages:
        raise RuntimeError(f"Failed to read TIFF stack: {path}")
    return np.stack([p.astype(np.float32, copy=False) for p in pages], axis=0)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def infer_fixed_radii(observations: List[dict], z_scaling: float) -> Tuple[float, float]:
    diameters = [parse_float(r, "diameter") for r in observations if parse_float(r, "diameter") > 0]
    median_diameter = statistics.median(diameters) if diameters else 76.0
    radius_xy = clamp(0.25 * median_diameter, 10.0, 24.0)
    radius_z = clamp(radius_xy / max(z_scaling, 1.0), 3.0, 8.0)
    return radius_xy, radius_z


def union_ellipsoid_measure(volume: np.ndarray, specs: List[Tuple[float, float, float, float, float]]) -> Measurement:
    """Measure one or more ellipsoid masks as a single non-overlapping signal region.

    The union behavior matters for split events. Two daughter windows may overlap
    immediately after division; measuring the union avoids double-counting raw
    voxels and is closer to the biological question: total light in the daughter
    objects, not total light in two independently sampled windows.
    """
    if not specs:
        return Measurement(0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    z_size, y_size, x_size = volume.shape
    outer_scale = 1.75

    x0 = max(0, int(math.floor(min(x - outer_scale * radius_xy for x, _, _, radius_xy, _ in specs))))
    x1 = min(x_size - 1, int(math.ceil(max(x + outer_scale * radius_xy for x, _, _, radius_xy, _ in specs))))
    y0 = max(0, int(math.floor(min(y - outer_scale * radius_xy for _, y, _, radius_xy, _ in specs))))
    y1 = min(y_size - 1, int(math.ceil(max(y + outer_scale * radius_xy for _, y, _, radius_xy, _ in specs))))
    z0 = max(0, int(math.floor(min(z - outer_scale * radius_z for _, _, z, _, radius_z in specs))))
    z1 = min(z_size - 1, int(math.ceil(max(z + outer_scale * radius_z for _, _, z, _, radius_z in specs))))

    crop = volume[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    zz, yy, xx = np.ogrid[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    signal = np.zeros(crop.shape, dtype=bool)
    shell = np.zeros(crop.shape, dtype=bool)
    for x, y, z, radius_xy, radius_z in specs:
        norm = ((xx - x) / radius_xy) ** 2 + ((yy - y) / radius_xy) ** 2 + ((zz - z) / radius_z) ** 2
        signal |= norm <= 1.0
        shell |= (norm > 1.25 ** 2) & (norm <= outer_scale ** 2)
    shell &= ~signal

    signal_values = crop[signal]
    if signal_values.size == 0:
        radius_xy = max(s[3] for s in specs)
        radius_z = max(s[4] for s in specs)
        return Measurement(0.0, 0, 0.0, 0.0, 0.0, radius_xy, radius_z)

    shell_values = crop[shell]
    if shell_values.size >= 20:
        bg = float(np.median(shell_values))
    else:
        bg = float(np.percentile(crop, 20.0))

    raw_sum = float(np.sum(signal_values))
    bgsub_values = np.maximum(signal_values - bg, 0.0)
    bgsub_luminosity = float(np.sum(bgsub_values))
    bgsub_mean = float(bgsub_luminosity / float(signal_values.size))
    radius_xy = max(s[3] for s in specs)
    radius_z = max(s[4] for s in specs)
    return Measurement(bg, int(signal_values.size), raw_sum, bgsub_luminosity, bgsub_mean, radius_xy, radius_z)


def ellipsoid_measure(volume: np.ndarray, x: float, y: float, z: float, radius_xy: float, radius_z: float) -> Measurement:
    return union_ellipsoid_measure(volume, [(x, y, z, radius_xy, radius_z)])


def coeff_variation(values: List[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return math.nan
    mean = statistics.mean(values)
    if abs(mean) < 1e-12:
        return math.nan
    return statistics.pstdev(values) / abs(mean)


def slope(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return math.nan
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def nearest_observation(rows_by_cell: Dict[str, List[dict]], cell_id: str, frame: int, min_frame: Optional[int] = None, max_frame: Optional[int] = None) -> Optional[dict]:
    candidates = rows_by_cell.get(cell_id, [])
    if min_frame is not None:
        candidates = [r for r in candidates if parse_int(r, "frame_index") >= min_frame]
    if max_frame is not None:
        candidates = [r for r in candidates if parse_int(r, "frame_index") <= max_frame]
    if not candidates:
        return None
    return min(candidates, key=lambda r: abs(parse_int(r, "frame_index") - frame))


def measure_run(run: RunConfig, output_dir: Path, pre_split_frames: int) -> Tuple[List[dict], List[SplitEvidence], dict]:
    observations = read_csv(run.analysis_dir / "per_cell_observations.csv")
    split_events = read_csv(run.analysis_dir / "split_events.csv")
    z_scaling = parse_z_scaling(run.config_file)
    fixed_radius_xy, fixed_radius_z = infer_fixed_radii(observations, z_scaling)

    volume_cache: Dict[str, np.ndarray] = {}

    def raw_volume(frame_name: str) -> np.ndarray:
        if frame_name not in volume_cache:
            volume_cache[frame_name] = load_raw_volume(run.input_dir / frame_name)
        return volume_cache[frame_name]

    measured_rows: List[dict] = []
    measured_by_cell_frame: Dict[Tuple[str, int], dict] = {}
    rows_by_cell: Dict[str, List[dict]] = {}

    for row in observations:
        frame_name = row["frame_name"]
        frame_index = parse_int(row, "frame_index")
        x = parse_float(row, "x")
        y = parse_float(row, "y")
        z = parse_float(row, "z")
        diameter = parse_float(row, "diameter")
        volume = raw_volume(frame_name)

        fixed = ellipsoid_measure(volume, x, y, z, fixed_radius_xy, fixed_radius_z)
        ellipsoid_radius_xy = clamp(0.50 * diameter, fixed_radius_xy, 42.0)
        ellipsoid_radius_z = clamp(ellipsoid_radius_xy / max(z_scaling, 1.0), fixed_radius_z, 10.0)
        ellipsoid = ellipsoid_measure(volume, x, y, z, ellipsoid_radius_xy, ellipsoid_radius_z)

        measured = dict(row)
        measured.update({
            "fixed_radius_xy": f"{fixed.radius_xy:.6f}",
            "fixed_radius_z": f"{fixed.radius_z:.6f}",
            "fixed_bg_raw": f"{fixed.bg:.6f}",
            "fixed_voxels": fixed.voxels,
            "fixed_raw_sum": f"{fixed.raw_sum:.6f}",
            "fixed_bgsub_luminosity": f"{fixed.bgsub_luminosity:.6f}",
            "fixed_bgsub_mean": f"{fixed.bgsub_mean:.6f}",
            "ellipsoid_radius_xy": f"{ellipsoid.radius_xy:.6f}",
            "ellipsoid_radius_z": f"{ellipsoid.radius_z:.6f}",
            "ellipsoid_bg_raw": f"{ellipsoid.bg:.6f}",
            "ellipsoid_voxels": ellipsoid.voxels,
            "ellipsoid_raw_sum": f"{ellipsoid.raw_sum:.6f}",
            "ellipsoid_bgsub_luminosity": f"{ellipsoid.bgsub_luminosity:.6f}",
            "ellipsoid_bgsub_mean": f"{ellipsoid.bgsub_mean:.6f}",
        })
        measured_rows.append(measured)
        measured_by_cell_frame[(row["cell_id"], frame_index)] = measured
        rows_by_cell.setdefault(row["cell_id"], []).append(measured)

    for rows in rows_by_cell.values():
        rows.sort(key=lambda r: parse_int(r, "frame_index"))

    split_rows: List[SplitEvidence] = []
    for event in split_events:
        split_frame = parse_int(event, "frame_index")
        parent_id = event["parent_id"]
        child1_id = event["child1_id"]
        child2_id = event["child2_id"]
        parent_history = [r for r in rows_by_cell.get(parent_id, []) if parse_int(r, "frame_index") < split_frame]
        pre_window = parent_history[-pre_split_frames:]
        if len(pre_window) < 3:
            continue

        parent_obs = pre_window[-1]
        child1_obs = nearest_observation(rows_by_cell, child1_id, split_frame, min_frame=split_frame, max_frame=split_frame + 2)
        child2_obs = nearest_observation(rows_by_cell, child2_id, split_frame, min_frame=split_frame, max_frame=split_frame + 2)
        if child1_obs is None or child2_obs is None:
            continue

        frames = [parse_int(r, "frame_index") for r in pre_window]
        volumes = [parse_float(r, "volume_voxels") for r in pre_window]
        fixed_means = [parse_float(r, "fixed_bgsub_mean") for r in pre_window]
        ellipsoid_means = [parse_float(r, "ellipsoid_bgsub_mean") for r in pre_window]
        fixed_lums = [parse_float(r, "fixed_bgsub_luminosity") for r in pre_window]
        ellipsoid_lums = [parse_float(r, "ellipsoid_bgsub_luminosity") for r in pre_window]

        parent_volume = raw_volume(parent_obs["frame_name"])
        child_frame_name = child1_obs["frame_name"]
        child_volume = raw_volume(child_frame_name)
        child1_diameter = parse_float(child1_obs, "diameter")
        child2_diameter = parse_float(child2_obs, "diameter")
        parent_diameter = parse_float(parent_obs, "diameter")

        parent_fixed_measure = union_ellipsoid_measure(
            parent_volume,
            [(parse_float(parent_obs, "x"), parse_float(parent_obs, "y"), parse_float(parent_obs, "z"), fixed_radius_xy, fixed_radius_z)])
        daughter_fixed_measure = union_ellipsoid_measure(
            child_volume,
            [
                (parse_float(child1_obs, "x"), parse_float(child1_obs, "y"), parse_float(child1_obs, "z"), fixed_radius_xy, fixed_radius_z),
                (parse_float(child2_obs, "x"), parse_float(child2_obs, "y"), parse_float(child2_obs, "z"), fixed_radius_xy, fixed_radius_z),
            ])

        parent_ellipsoid_radius_xy = clamp(0.50 * parent_diameter, fixed_radius_xy, 42.0)
        parent_ellipsoid_radius_z = clamp(parent_ellipsoid_radius_xy / max(z_scaling, 1.0), fixed_radius_z, 10.0)
        child1_ellipsoid_radius_xy = clamp(0.50 * child1_diameter, fixed_radius_xy, 42.0)
        child1_ellipsoid_radius_z = clamp(child1_ellipsoid_radius_xy / max(z_scaling, 1.0), fixed_radius_z, 10.0)
        child2_ellipsoid_radius_xy = clamp(0.50 * child2_diameter, fixed_radius_xy, 42.0)
        child2_ellipsoid_radius_z = clamp(child2_ellipsoid_radius_xy / max(z_scaling, 1.0), fixed_radius_z, 10.0)
        parent_ellipsoid_measure = union_ellipsoid_measure(
            parent_volume,
            [(parse_float(parent_obs, "x"), parse_float(parent_obs, "y"), parse_float(parent_obs, "z"), parent_ellipsoid_radius_xy, parent_ellipsoid_radius_z)])
        daughter_ellipsoid_measure = union_ellipsoid_measure(
            child_volume,
            [
                (parse_float(child1_obs, "x"), parse_float(child1_obs, "y"), parse_float(child1_obs, "z"), child1_ellipsoid_radius_xy, child1_ellipsoid_radius_z),
                (parse_float(child2_obs, "x"), parse_float(child2_obs, "y"), parse_float(child2_obs, "z"), child2_ellipsoid_radius_xy, child2_ellipsoid_radius_z),
            ])

        fixed_parent = parent_fixed_measure.bgsub_luminosity
        fixed_daughter_sum = daughter_fixed_measure.bgsub_luminosity
        ellipsoid_parent = parent_ellipsoid_measure.bgsub_luminosity
        ellipsoid_daughter_sum = daughter_ellipsoid_measure.bgsub_luminosity

        fixed_ratio = fixed_daughter_sum / fixed_parent if fixed_parent > 0 else math.nan
        ellipsoid_ratio = ellipsoid_daughter_sum / ellipsoid_parent if ellipsoid_parent > 0 else math.nan
        fixed_cv = coeff_variation(fixed_lums)
        ellipsoid_cv = coeff_variation(ellipsoid_lums)
        volume_slope = slope(frames, volumes)
        fixed_mean_slope = slope(frames, fixed_means)
        ellipsoid_mean_slope = slope(frames, ellipsoid_means)

        fixed_support = volume_slope < 0 and fixed_mean_slope > 0 and fixed_cv <= 0.25 and 0.75 <= fixed_ratio <= 1.25
        ellipsoid_support = volume_slope < 0 and ellipsoid_mean_slope > 0 and ellipsoid_cv <= 0.25 and 0.75 <= ellipsoid_ratio <= 1.25

        split_rows.append(SplitEvidence(
            run.name,
            parent_id,
            split_frame,
            child1_id,
            child2_id,
            len(pre_window),
            parse_int(parent_obs, "frame_index"),
            min(parse_int(child1_obs, "frame_index"), parse_int(child2_obs, "frame_index")),
            volumes[0],
            volumes[-1],
            fixed_means[0],
            fixed_means[-1],
            ellipsoid_means[0],
            ellipsoid_means[-1],
            fixed_cv,
            ellipsoid_cv,
            fixed_parent,
            fixed_daughter_sum,
            fixed_ratio,
            ellipsoid_parent,
            ellipsoid_daughter_sum,
            ellipsoid_ratio,
            fixed_support,
            ellipsoid_support,
        ))

    measured_fields = list(measured_rows[0].keys()) if measured_rows else []
    write_csv(output_dir / f"{run.name}_fair_luminosity_observations.csv", measured_rows, measured_fields)

    summary = {
        "run_name": run.name,
        "analysis_dir": str(run.analysis_dir),
        "observations": len(observations),
        "split_events": len(split_events),
        "analyzable_split_events": len(split_rows),
        "z_scaling": z_scaling,
        "fixed_radius_xy": fixed_radius_xy,
        "fixed_radius_z": fixed_radius_z,
    }
    return measured_rows, split_rows, summary


def split_evidence_to_row(e: SplitEvidence) -> dict:
    return {
        "run_name": e.run_name,
        "parent_id": e.parent_id,
        "split_frame": e.split_frame,
        "child1_id": e.child1_id,
        "child2_id": e.child2_id,
        "pre_window_count": e.pre_window_count,
        "parent_frame": e.parent_frame,
        "child_frame": e.child_frame,
        "volume_first": f"{e.volume_first:.6f}",
        "volume_last": f"{e.volume_last:.6f}",
        "fixed_mean_first": f"{e.fixed_mean_first:.6f}",
        "fixed_mean_last": f"{e.fixed_mean_last:.6f}",
        "ellipsoid_mean_first": f"{e.ellipsoid_mean_first:.6f}",
        "ellipsoid_mean_last": f"{e.ellipsoid_mean_last:.6f}",
        "fixed_pre_cv": f"{e.fixed_pre_cv:.6f}",
        "ellipsoid_pre_cv": f"{e.ellipsoid_pre_cv:.6f}",
        "fixed_parent_lum": f"{e.fixed_parent_lum:.6f}",
        "fixed_daughter_sum": f"{e.fixed_daughter_sum:.6f}",
        "fixed_daughter_parent_ratio": f"{e.fixed_daughter_parent_ratio:.6f}",
        "ellipsoid_parent_lum": f"{e.ellipsoid_parent_lum:.6f}",
        "ellipsoid_daughter_sum": f"{e.ellipsoid_daughter_sum:.6f}",
        "ellipsoid_daughter_parent_ratio": f"{e.ellipsoid_daughter_parent_ratio:.6f}",
        "fixed_supports_hypothesis": int(e.fixed_supports_hypothesis),
        "ellipsoid_supports_hypothesis": int(e.ellipsoid_supports_hypothesis),
    }


def median(values: List[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return statistics.median(values) if values else math.nan


def count_ratio_near_one(values: List[float], lo: float = 0.75, hi: float = 1.25) -> int:
    return sum(1 for v in values if math.isfinite(v) and lo <= v <= hi)


def write_markdown_report(path: Path, summaries: List[dict], split_rows: List[SplitEvidence], output_dir: Path) -> None:
    by_run: Dict[str, List[SplitEvidence]] = {}
    for row in split_rows:
        by_run.setdefault(row.run_name, []).append(row)

    lines: List[str] = []
    lines.append("# Professor Luminosity Hypothesis Evidence")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append("Professor Hayes's hypothesis: before a nucleus splits, its segmented size may shrink while voxel-level brightness increases, but the total luminosity of the object may remain roughly constant. If true, luminosity could become a useful split cue.")
    lines.append("")
    lines.append("## Measurement Logic")
    lines.append("")
    lines.append("- Raw TIFF stacks were read directly again; this test does not use normalized image values as the final luminosity measurement.")
    lines.append("- Cell centers and split events came from the existing brightness/volume analyzer output.")
    lines.append("- Fixed-window luminosity used the same 3D ellipsoid radius for every cell in a run, inferred from the median tracked diameter and `z_scaling`.")
    lines.append("- Adaptive ellipsoid luminosity used each tracked cell's current diameter, with the same `z_scaling` conversion for z radius.")
    lines.append("- Local background was estimated from a surrounding shell around the cell using median raw intensity.")
    lines.append("- Background-subtracted luminosity was computed as `sum(max(raw_voxel - local_background, 0))` inside the signal mask.")
    lines.append("- For each split, the test compared the parent in the last frame before split with the non-overlapping union of daughter masks at the split frame or within two frames after it.")
    lines.append("- Daughter masks were measured as a union to avoid double-counting overlapping voxels when the two daughters are still close.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    total_splits = sum(s["split_events"] for s in summaries)
    total_analyzable = sum(s["analyzable_split_events"] for s in summaries)
    fixed_support = sum(1 for r in split_rows if r.fixed_supports_hypothesis)
    ellipsoid_support = sum(1 for r in split_rows if r.ellipsoid_supports_hypothesis)
    fixed_shrink_brighten = sum(1 for r in split_rows if r.volume_last < r.volume_first and r.fixed_mean_last > r.fixed_mean_first)
    ellipsoid_shrink_brighten = sum(1 for r in split_rows if r.volume_last < r.volume_first and r.ellipsoid_mean_last > r.ellipsoid_mean_first)
    lines.append(f"- Total split events tested: {total_splits}")
    lines.append(f"- Analyzable split events with parent and both daughters found: {total_analyzable}")
    lines.append(f"- Fixed-window shrink+brighten events before split: {fixed_shrink_brighten}/{total_analyzable}")
    lines.append(f"- Adaptive-ellipsoid shrink+brighten events before split: {ellipsoid_shrink_brighten}/{total_analyzable}")
    lines.append(f"- Fixed-window events supporting the full hypothesis: {fixed_support}/{total_analyzable}")
    lines.append(f"- Adaptive-ellipsoid events supporting the full hypothesis: {ellipsoid_support}/{total_analyzable}")
    lines.append("")

    all_fixed_ratios = [r.fixed_daughter_parent_ratio for r in split_rows]
    all_ellipsoid_ratios = [r.ellipsoid_daughter_parent_ratio for r in split_rows]
    all_fixed_cvs = [r.fixed_pre_cv for r in split_rows]
    all_ellipsoid_cvs = [r.ellipsoid_pre_cv for r in split_rows]
    lines.append("## Aggregate Numerical Evidence")
    lines.append("")
    lines.append(f"- Median fixed-window pre-split luminosity CV: {median(all_fixed_cvs):.4f}")
    lines.append(f"- Median adaptive-ellipsoid pre-split luminosity CV: {median(all_ellipsoid_cvs):.4f}")
    lines.append(f"- Median fixed-window daughter/parent luminosity ratio: {median(all_fixed_ratios):.4f}")
    lines.append(f"- Median adaptive-ellipsoid daughter/parent luminosity ratio: {median(all_ellipsoid_ratios):.4f}")
    lines.append(f"- Fixed-window daughter/parent ratios within [0.75, 1.25]: {count_ratio_near_one(all_fixed_ratios)}/{total_analyzable}")
    lines.append(f"- Adaptive-ellipsoid daughter/parent ratios within [0.75, 1.25]: {count_ratio_near_one(all_ellipsoid_ratios)}/{total_analyzable}")
    lines.append("")

    for summary in summaries:
        rows = by_run.get(summary["run_name"], [])
        lines.append(f"## Run: {summary['run_name']}")
        lines.append("")
        lines.append(f"- Analysis directory: `{summary['analysis_dir']}`")
        lines.append(f"- Observations measured: {summary['observations']}")
        lines.append(f"- Split events: {summary['split_events']}")
        lines.append(f"- Analyzable split events: {summary['analyzable_split_events']}")
        lines.append(f"- Fixed signal radius: xy={summary['fixed_radius_xy']:.3f}, z={summary['fixed_radius_z']:.3f}")
        if rows:
            run_fixed_shrink_brighten = sum(1 for r in rows if r.volume_last < r.volume_first and r.fixed_mean_last > r.fixed_mean_first)
            run_ellipsoid_shrink_brighten = sum(1 for r in rows if r.volume_last < r.volume_first and r.ellipsoid_mean_last > r.ellipsoid_mean_first)
            lines.append(f"- Median fixed-window pre-split CV: {median([r.fixed_pre_cv for r in rows]):.4f}")
            lines.append(f"- Median ellipsoid pre-split CV: {median([r.ellipsoid_pre_cv for r in rows]):.4f}")
            lines.append(f"- Median fixed daughter/parent ratio: {median([r.fixed_daughter_parent_ratio for r in rows]):.4f}")
            lines.append(f"- Median ellipsoid daughter/parent ratio: {median([r.ellipsoid_daughter_parent_ratio for r in rows]):.4f}")
            lines.append(f"- Fixed-window shrink+brighten events: {run_fixed_shrink_brighten}/{len(rows)}")
            lines.append(f"- Adaptive-ellipsoid shrink+brighten events: {run_ellipsoid_shrink_brighten}/{len(rows)}")
            lines.append(f"- Fixed-window full-support events: {sum(r.fixed_supports_hypothesis for r in rows)}/{len(rows)}")
            lines.append(f"- Adaptive-ellipsoid full-support events: {sum(r.ellipsoid_supports_hypothesis for r in rows)}/{len(rows)}")
        lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    if fixed_support == 0 and ellipsoid_support == 0:
        lines.append("Under this more faithful test, the current embryo runs do not support using constant total luminosity as a reliable standalone split rule yet. The shrink+brighten pattern exists in some split parents, but the parent-to-daughter luminosity conservation test is not consistently satisfied across events.")
    else:
        lines.append("The hypothesis has partial support in this test, but not enough to use as a hard rule without additional validation against ground truth.")
    lines.append("")
    lines.append("This does not disprove the biological idea. It means the current measurement pipeline, tracking centers, and mask assumptions do not yet produce a stable enough luminosity signal to safely affect split acceptance. The metric should remain diagnostic until validated on more ground-truth split events.")
    lines.append("")
    lines.append("## Generated Evidence Files")
    lines.append("")
    lines.append(f"- Split-level evidence CSV: `{output_dir / 'split_parent_daughter_luminosity_evidence.csv'}`")
    for summary in summaries:
        lines.append(f"- Observation-level evidence CSV for {summary['run_name']}: `{output_dir / (summary['run_name'] + '_fair_luminosity_observations.csv')}`")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [parse_run_spec(spec, args.input_dir.expanduser(), args.config.expanduser()) for spec in args.run]

    all_split_rows: List[SplitEvidence] = []
    summaries: List[dict] = []
    for run in runs:
        _, split_rows, summary = measure_run(run, output_dir, args.pre_split_frames)
        all_split_rows.extend(split_rows)
        summaries.append(summary)

    split_fieldnames = list(split_evidence_to_row(all_split_rows[0]).keys()) if all_split_rows else [
        "run_name", "parent_id", "split_frame", "fixed_supports_hypothesis", "ellipsoid_supports_hypothesis"
    ]
    write_csv(output_dir / "split_parent_daughter_luminosity_evidence.csv",
              [split_evidence_to_row(row) for row in all_split_rows], split_fieldnames)

    report = args.report.expanduser() if args.report else output_dir / "professor_luminosity_hypothesis_evidence.md"
    write_markdown_report(report, summaries, all_split_rows, output_dir)
    print(f"Wrote report: {report}")
    print(f"Wrote split evidence: {output_dir / 'split_parent_daughter_luminosity_evidence.csv'}")


if __name__ == "__main__":
    main()
