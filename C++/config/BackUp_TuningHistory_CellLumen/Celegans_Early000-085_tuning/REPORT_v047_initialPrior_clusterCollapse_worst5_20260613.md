# CellLumen early v047 initial prior cluster collapse audit

## Goal

Test a safer code level version of the v046 early cluster collapse idea. The
problem is that early C. elegans frames have a few large bright nuclei, and one
large nucleus can contain several local bright peaks. A pure local peak or
watershed candidate path can therefore split one real cell into many small
candidate centers.

This version keeps the new behavior behind explicit switches. The concentrated
runtime YAML is not modified.

## Config

YAML:

`C++/config/BackUp_TuningHistory_CellLumen/Celegans_Early000-085_tuning/config_cell_lumen_embryo_early_v047_initialPrior_clusterCollapse.yaml`

Key switches:

- `initialPriorClusterCollapseEnabled: true`
- `initialPriorCsvPath: /Users/wangyiding/CellUniverse/C++/config/C.elegans developing embryo/C.elegans_initial/initial_files/00_core_start_points/initial_embryo_0.csv`
- `initialPriorClusterCollapseLinkScale: 0.18`
- `initialPriorClusterCollapseMinLinkDistance: 24.0`
- `initialPriorClusterCollapseMaxLinkDistance: 42.0`

The initial CSV is a legal starting input for Cell Universe. This test uses only
the initial cell spacing scale to estimate the collapse distance. It does not
read current frame GT and does not copy answer coordinates.

## Output

Run output:

`C++/output/CellLumen_EarlyLogicTests_20260613/v047_initialPrior_worst5_noTif`

No tif files were written. The folder contains only small csv and log files.

## GT comparison

GT match threshold: 20 px in scaled 3D coordinates.

| frame | GT | candidates | matched | miss | extra | max match distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 001 | 4 | 7 | 4 | 0 | 3 | 18.43 |
| 002 | 4 | 5 | 4 | 0 | 1 | 10.98 |
| 005 | 6 | 7 | 6 | 0 | 1 | 10.19 |
| 014 | 8 | 8 | 8 | 0 | 0 | 10.16 |
| 035 | 24 | 25 | 24 | 0 | 1 | 18.49 |

Total across these five stress frames:

- GT cells: 46
- candidates after v047: 52
- matched GT cells: 46
- missing GT cells: 0
- extra candidates: 6

## Log evidence

The code read the initial CSV and computed the collapse distance from the early
cell spacing scale:

`median_initial_nn=208.643, link_scale=0.18, link_distance=37.5557`

The collapse stage then ran in `mode=initial_prior`:

- frame 001: before 35, after 7
- frame 002: before 24, after 5
- frame 005: before 25, after 7
- frame 014: before 51, after 8
- frame 035: before 43, after 25

## Current interpretation

This is a low risk improvement over fixed distance cluster collapse. v046 also
kept zero misses on the same stress frames, but v047 removes one extra candidate
by deriving the collapse distance from legal initial geometry instead of using
only a hand picked constant.

The next safe validation step is a broader f000 to f085 no-tif sweep before this
is considered for the density switch YAML.
