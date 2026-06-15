# v076 early Cell Lumen result

Date: 2026-06-13

Config:
`/Users/wangyiding/CellUniverse/C++/config/BackUp_TuningHistory_CellLumen/Celegans_Early000-085_tuning/config_cell_lumen_embryo_early_v076_v043_xy_refine_positive_z.yaml`

Full output:
`/Users/wangyiding/CellUniverse/C++/output/CellLumen_EarlyLogicTests_20260613/v076_v043_xy_refine_positive_z_full_f001_085_noTif`

Evaluation summary:

1. Frames tested: 001 to 085.
2. No miss frames: 85 / 85.
3. Total missing: 0.
4. Total extra: 2123.
5. Mean extra per frame: 24.98.
6. Max extra: 46 at f053.
7. Max matched distance: 19.21 at f005.
8. TIFF output check: no `.tif` or `.tiff` files were present in the output directory.

Main change that worked:

v076 returns to the stable v043 high recall base, then separates local x/y refinement from z refinement. The final local brightness centroid uses a wider radius to pull early sparse centers toward their local bright core, but `finalLocalRefineZBlend: 0.0` prevents that pass from dragging z-separated cells toward the wrong slice. A second z-column pass is still enabled, but `finalZColumnRefinePositiveOnly: true` allows only upward z recovery. This fixed the v043 misses at f001, f002, f005, f014, and f035 while avoiding the v071 regression at f036.

Important comparison:

v068 also reached zero miss on key frames, but its full run failed later with repeated misses around f066 to f082 and had higher extra output. v076 avoided that failure because it does not use the broad z-profile add strategy from v068.

Notes for later concentrated YAML integration:

Do not merge this blindly into the concentrated density switch file yet. The result is now a strong early no-miss candidate, but it remains high recall rather than clean standalone segmentation. It should be integrated as an early low-density profile only after deciding how much extra Cell Universe can safely filter.
