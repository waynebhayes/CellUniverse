# CellLumen early v046 worst frame audit

## Goal

Test whether the early high recall CellLumen output can be cleaned by PCA style local shape evidence and biological spacing rules without using GT at runtime.

The rule tested here is not "choose the brightest candidate". That failed because bright fragments can be off center. The working rule is to collapse a local cluster of over segmented candidates into one centroid when the candidates are closer than the expected early embryo biological spacing.

## Config

YAML:

`C++/config/BackUp_TuningHistory_CellLumen/Celegans_Early000-085_tuning/config_cell_lumen_embryo_early_v046_clusterRecall_clusterCollapse.yaml`

Key switches:

- `finalClusterCentroidRecallRescueEnabled: true`
- `finalClusterCentroidCollapseEnabled: true`
- `finalClusterCentroidCollapseLinkDistance: 34.0`
- `finalClusterCentroidCollapseUseSignalWeights: false`
- `finalDominatedDuplicateFilterEnabled: false`

The collapse uses only runtime CellLumen candidates and raw image signal already measured by CellLumen. GT is used only after the run for scoring.

## Output

Run output:

`C++/output/F000-085_CellLumenEarlyTune0601/v046_clusterRecall_clusterCollapse_worst_frames_noTif_20260601`

No tif files were written.

## GT comparison

GT match threshold: 20 px in scaled 3D coordinates.

| frame | GT | candidates | matched | miss | extra | max match distance | mean match distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 001 | 4 | 7 | 4 | 0 | 3 | 18.434 | 8.677 |
| 002 | 4 | 6 | 4 | 0 | 2 | 16.966 | 9.558 |
| 005 | 6 | 7 | 6 | 0 | 1 | 10.190 | 5.345 |
| 014 | 8 | 8 | 8 | 0 | 0 | 10.163 | 5.793 |
| 035 | 24 | 25 | 24 | 0 | 1 | 18.490 | 3.727 |

Total across these five worst frames:

- GT cells: 46
- candidates after v046: 53
- matched GT cells: 46
- missing GT cells: 0
- extra candidates: 7

## Why NMS was not used

A simple nearest neighbor suppression rule reduced the candidate count, but it deleted real GT supported centers in frames 001, 002, 005, and 014. The reason is that a brighter local fragment was often slightly off center, while the candidate closer to GT had a lower raw score.

Cluster centroid collapse solved that specific failure mode because it does not pick the brightest fragment. It groups nearby fragments and replaces the group with one centroid.

## Log evidence

The C++ run logs show the collapse stage firing:

- frame 001: before 35, after 7
- frame 002: before 24, after 6
- frame 005: before 25, after 7
- frame 014: before 51, after 8
- frame 035: before 43, after 25

## Current interpretation

This is a strong result for the worst early frames. It suggests the early problem was not only "CellLumen cannot see the cells". v045 could see them with high recall, but it produced many local fragments. v046 turns those fragments into one center per local brightness cluster and keeps all GT centers in these five stress frames.

This still needs a broader staged validation before running the full main CellUniverse workflow. The next safe test should be selected early frames around difficult transitions, not the full 0 to 85 sweep yet.
