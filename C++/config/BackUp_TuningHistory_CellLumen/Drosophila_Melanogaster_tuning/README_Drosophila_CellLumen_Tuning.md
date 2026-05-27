# Drosophila Melanogaster CellLumen Tuning

This folder stores dataset-specific YAMLs and helper scripts for raw CellLumen tuning only.

Current constraints:

- Do not modify C++ source for this tuning pass without explicit permission.
- Tuning outputs go under `C++/output/CellLum_TuneOutput_Drosophila_Melanogaster`.
- Use `CELLUNIVERSE_CELL_LUMEN_SKIP_TIFF=1` during parameter sweeps to avoid large preview TIFF output.
- Do not set `CELLUNIVERSE_CELL_LUMEN_USE_TRA_MASK=1` during raw detection tests, because that would initialize directly from the GT TRA masks.

Dataset notes from initial inspection:

- Sequences: `01` and `02`, frames `0..49`.
- Raw frame shape: `125 x 603 x 1272`, `uint16`.
- TIFF metadata implies approximate `z_scaling: 5.0`.
- GT TRA labels are sparse center markers, not full cell volumes. Each sampled label has exactly `75` voxels.
- `01_GT/TRA/man_track.txt`: 189 tracks, all `0..49`, all parent `0`.
- `02_GT/TRA/man_track.txt`: 203 tracks, all `0..49`, all parent `0`.
- Therefore explicit GT lineage divisions in the provided TRA files: `0`.
