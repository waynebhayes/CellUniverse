Embryo CellLumen tuning archive
==================================

This folder stores intermediate YAML configs used while tuning the raw
intensity CellLumen path for the embryo dataset. These files are
kept as an experiment record only. The active config remains:

`C++/config/config_embryo.yaml`

Final config source:

The current `config_embryo.yaml` is a combined parameter set. It is not one
single probe YAML copied directly. The best final behavior came from combining
the useful parts of several runs:

- `01_seed_center_tuning/best_seed_distance20_recovered_dense_centers.yaml`
  provided the main marker spacing for dense 130 to 150 frames.
- `01_seed_center_tuning/best_center_seed_blend010_fixed_137_140_147.yaml`
  provided the small seed center blend that fixed z stacked center misses
  without hurting nearby frames.
- `02_voxel_and_low_rescue_tuning/best_dense_minvox1100_recovered_130_150.yaml`
  provided the dense frame minimum voxel relaxation.
- `02_voxel_and_low_rescue_tuning/best_low_rescue_distance28_kept_zero_miss.yaml`
  kept low rescue permissive enough to avoid missing dim cells.
- `03_merge_duplicate_tuning/best_merge5_kept_zero_miss_130_150.yaml`
  showed that a stronger final merge reduced extras but started removing real
  close centers, so merge distance 5 was kept.

The final weak satellite filter was added after those probe YAMLs. It removes
small weak rescued blobs only when they are close to a larger neighbor. The
best verified result at this stage is:

- frames 85 to 100: 1357 GT centers, 1357 matched, 0 missed, 86 extra.
- frames 130 to 150: 3887 GT centers, 3887 matched, 0 missed, 60 extra.

Remaining problem:

The method still produces extra predictions, mostly weak satellite blobs near
real cells. Center recall is currently strong in the tested ranges, but false
positive cleanup still needs work.

Folder guide:

- `01_seed_center_tuning/`: marker seed distance, seed center blend, and seed
  placement experiments.
- `02_voxel_and_low_rescue_tuning/`: minimum voxel, low rescue, and weak or dim
  cell recovery threshold experiments.
- `03_merge_duplicate_tuning/`: final merge, duplicate cleanup, and post refine
  merge distance experiments.
- `04_local_refine_and_z_tuning/`: local center refine and z center correction
  experiments.
- `99_older_grid_runs/`: older broad grid configs that do not fit one clean
  category.
