# Verified 1-84 split luminosity hypothesis evidence

## Data source
- Split events were extracted only from the manually verified Cell Universe CSV files for frames 1-84.
- A split event was counted only when the parent existed in frame f-1, disappeared in frame f, and both lineage children `parent0` and `parent1` first appeared in frame f.
- The source CSV file that continues beyond 84 was explicitly truncated at frame 84.
- Raw image intensity was measured from local TIFF stacks, not preprocessed or normalized images.
- CSV z coordinates were divided by z_scaling=7.0 before sampling raw TIFF slices.

## Measurement logic
- Fixed raw 3D window radius: xy=22.930, z=3.101 raw slices.
- Adaptive ellipsoid radius used verified per-cell radii; z radius was converted by z_scaling.
- Local background was estimated from a shell around the signal mask, then subtracted voxel by voxel.
- Daughter luminosity was measured as a non-overlapping union mask to avoid double-counting close daughters.

## Summary
- Verified frames loaded: 1-84 (84 frames).
- Strict verified split events tested: 50.
- Analyzable events: 50.
- Fixed-window shrink+brighten events: 1/50.
- Adaptive-ellipsoid shrink+brighten events: 0/50.
- Fixed-window daughter/parent ratio within [0.75, 1.25]: 4/50.
- Adaptive-ellipsoid daughter/parent ratio within [0.75, 1.25]: 45/50.
- Fixed-window full support: 0/50.
- Adaptive-ellipsoid full support: 0/50.
- Median fixed-window pre-split luminosity CV: 0.1061.
- Median adaptive-ellipsoid pre-split luminosity CV: 0.1546.
- Median fixed-window daughter/parent luminosity ratio: 1.4072.
- Median adaptive-ellipsoid daughter/parent luminosity ratio: 0.9178.

## Interpretation
Using the manually verified lineage events changes the event count from the previous analyzer-derived 26 events to 50 strict verified events. The shrink-and-brighten pattern appears in part of the data, but parent-to-daughter raw luminosity conservation is still not consistently supported. This means the professor hypothesis remains useful as a diagnostic feature, but these verified data do not yet justify using constant total luminosity as a hard split rule.

- Evidence CSV: `/Users/wangyiding/CellUniverse/C++/output/professor_luminosity_hypothesis_verified_1_84_20260506/verified_1_84_split_luminosity_evidence.csv`
