# C.elegans developing embryo config organization

This folder separates the current C.elegans embryo YAML files by how much
evidence they have.

## Best_until_now

These YAML files are kept here because the correct output run logs point to
them, or because they are required by the verified base config chain.

- `config_embryo_CellLumenFusion_VERIFIED_F085-120_temporalRepair_noTif_20260531.yaml`
  is the main verified 85 to 120 profile. The verified log scan found the
  underlying anti cascade profile in 66 correct output logs.
- `config_embryo_CellLumenFusion_deterministicSplit.yaml` is the verified
  121 to 170 profile. The verified log scan found it in 53 correct output
  logs, with the same file sometimes recorded as a relative path and sometimes
  as an absolute path.
- `config_embryo_CellLumenFusion_F126_refill_futureWindow8_tif_20260602.yaml`
  is kept as the special verified refill for frame 126.
- `config_embryo_CellLumenFusion_F170-194_lateBaseline_noTif_noTemporalRepair_20260531.yaml`
  is the late frame test line used by the verified 170 to 174 outputs.
- `config_cell_lumen_embryo_celluniverse_preprocess_high_recall.yaml` and
  `config_cell_lumen_embryo_standalone_clean.yaml` are kept here because the
  verified fusion path and CellLumen probes depend on them.

The base chain files are also kept here so each best config can still load
after the directory cleanup.

## In Experiment

This folder contains diagnostic profiles, failed center offset attempts,
early density monitor runs, smoke configs, and any candidate that has not been
validated as a replacement for the best profiles.

The file
`config_embryo_CellLumenFusion_UNIFIED_DENSITY_METRICS_CANDIDATE_NOT_VERIFIED_20260609.yaml`
is only a metric collection candidate. It is not a proven single best config.

## Evidence files

- `verified_yaml_usage_from_run_logs_20260609.csv` records which YAML paths
  were found in verified output run logs.
- `cell_density_summary_001_171_20260609.csv` records per frame cell count,
  bounding box density, and nearest neighbor distance from the final lineage
  CSV.
- `cell_density_stage_summary_20260609.csv` summarizes the density boundary
  between early, mid, and late frames.

## Density conclusion

The final lineage CSV shows a real density shift:

- Frames 85 to 120: median nearest neighbor distance is about 49.34 px.
- Frames 121 to 170: median nearest neighbor distance is about 37.37 px.
- Frames 170 to 171: median nearest neighbor distance is about 31.45 px.

This is enough signal to design a density based controller later. The current
C++ code records density metrics, but the split gate values are not yet driven
by density. For that reason the safe state today is still stage specific best
configs plus an experimental metric candidate.
