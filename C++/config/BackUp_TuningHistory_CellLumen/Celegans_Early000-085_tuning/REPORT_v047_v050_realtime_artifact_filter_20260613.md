# CellLumen early v047 to v050 realtime audit

## Goal

Use realtime GT evaluation to separate three different early CellLumen failure
types:

1. persistent low-density floating artifacts,
2. internal bright fragments inside one large early cell,
3. unsafe cluster collapse that merges two real nearby centers.

GT was used only after each frame CSV was written. It was not used to create or
move candidate centers.

## Realtime evaluator

Script:

`C++/scripts/evaluate_celegans_cell_lumen_realtime.py`

Example command:

```bash
python3 /Users/wangyiding/CellUniverse/C++/scripts/evaluate_celegans_cell_lumen_realtime.py \
  --run-dir "/Users/wangyiding/CellUniverse/C++/output/CellLumen_EarlyLogicTests_20260613/v050b_postCollapseLowDensity_keyframes_noTif" \
  --frames 0-1,3-5,14,18,34-36 \
  --summary-csv "/Users/wangyiding/CellUniverse/C++/output/CellLumen_EarlyLogicTests_20260613/v050b_postCollapseLowDensity_keyframes_noTif/realtime_eval_summary.csv"
```

The script prints each frame as soon as the CSV exists. If a run log is present,
extra candidates are annotated with `vox`, `mean`, and `top10MinusShell`.

## Code change

Added a default-off final low density artifact filter:

- `finalLowDensityArtifactFilterEnabled`
- `finalLowDensityArtifactMaxMeanRatio`
- `finalLowDensityArtifactMaxVoxelRatio`
- `finalLowDensityArtifactMinNearestDistance`

The filter is intentionally not based on radius alone, because floating dust can
look radius-like after fitting. It requires low relative mean intensity, low
relative voxel support, and isolation from the biological candidate field.

The filter runs before and after cluster collapse. The second pass is needed
because cluster recall and collapse can still leave a low-density candidate in
the final output.

## Results

v047 initial prior collapse full f000 to f085:

- perfect frames: 33 / 86
- no-miss frames: 48 / 86
- total missing: 246
- total extra: 124

Interpretation: v047 fixes early internal fragments, but `maxCells=90` is too
broad. It over merges real dense candidates after about f052.

v049 max40 key frames:

- f000 and f002 became perfect after low-density artifact filtering.
- f014 and f035 skipped collapse and exploded to many extras.
- f052 and f055 avoided missing cells but kept many high recall extras.

Interpretation: max40 is too strict for early stress frames.

v050 max60 key frames after post-collapse artifact filtering:

| frame | result |
| --- | --- |
| f000 | PASS, 4/4 |
| f001 | no miss, 2 extra |
| f003 | no miss, 1 extra |
| f004 | miss 1 |
| f005 | no miss, 1 extra |
| f014 | PASS, 8/8 |
| f018 | miss 1 |
| f034 | no miss, 1 extra |
| f035 | PASS, 24/24 |
| f036 | miss 1 |

Interpretation: v050 is the most balanced version from this small test. It
solves the obvious floating dust at f000/f002 and keeps important early collapse
cases like f014/f035, while avoiding v047's later dense over-merge. It is still
not final because f004, f018, and f036 remain missing.

## Next algorithm direction

The remaining misses are not low-density artifacts. In f004, two real GT centers
near each other are collapsed into one midpoint-like candidate. This means the
cluster collapse grouping is too single-linkage-like: fragments can form a
chain between two real centers and cause over-merge.

The next code-level improvement should add a default-off collapse safety guard:
if a cluster has a large internal diameter or two strong separated signal lobes,
do not collapse it into one centroid. Keep the separate candidates instead,
because extra candidates are safer than missing a real cell for Cell Universe.
