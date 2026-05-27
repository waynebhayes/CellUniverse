# Cell Lumen Final Config Index

This folder is an organized index of the final Cell Lumen and Cell Lumen plus Cell Universe configuration modes.

The original runtime YAML files are intentionally kept in `C++/config` or their original folder because current run commands and `base_config` paths may refer to those locations directly. The files here are copied and renamed for readability, review, and long term tracking.

## 00_FINAL_ACTIVE_RUNTIME_CONFIGS

These configs are the current best parameter combinations that should be treated as the main usable modes.

- `FINAL_STANDALONE_CLEAN__config_cell_lumen_embryo_standalone_clean.yaml`
  - Best clean Cell Lumen standalone mode.
  - Use when Cell Lumen itself is the final output to inspect.
  - Prioritizes clean centers and avoids carrying every duplicate candidate.

- `FINAL_CELLUNIVERSE_PREPROCESS_HIGH_RECALL__config_cell_lumen_embryo_celluniverse_preprocess_high_recall.yaml`
  - Best Cell Lumen high recall mode for Cell Universe preprocessing.
  - Use when Cell Lumen is only providing raw signal mask or center candidates to Cell Universe.
  - Prioritizes not missing real cells, even if it keeps extra candidates.

- `FINAL_FUSION_DETERMINISTIC_SPLIT__config_embryo_CellLumenFusion_deterministicSplit.yaml`
  - Current Cell Lumen plus Cell Universe fusion mode.
  - Uses raw intensity preserving preprocessing, Cell Lumen center priors, deterministic split proposals, soft scoring, and limited random optimizer work.
  - This is the active path for the frame by frame fusion tests.

## 01_POSITIVE_REFERENCE_CONFIGS

These configs are stable final references or important bases for the final modes, but they are not the primary entry point by themselves.

- `POSITIVE_RAW_NO_PREPROCESS_BASE__config_embryo_NoPreprocess.yaml`
  - Raw information preserving base for embryo Cell Universe experiments.
  - Disables heavy preprocessing while keeping intensity scaling and z interpolation.

- `POSITIVE_CONSERVATIVE_FUSION_BASELINE__config_embryo_CellLumenFusion_conservative.yaml`
  - Earlier conservative Cell Lumen fusion baseline.
  - Useful for comparing the safer center repair path before deterministic split prior became the main route.

- `POSITIVE_LIGHT_PREPROCESS_EMBRYO_REFERENCE__config_embryo_light_preprocess.yaml`
  - Earlier light preprocessing reference for embryo experiments.
  - Useful as a reference for avoiding heavy preprocessing damage to dim cells.

## 02_EXPERIMENTAL_OR_SMOKE_CONFIGS

These are not final parameter sets. They are kept only to reproduce earlier checks.

- `SMOKE_HEAVY_PREPROCESS_CELL_LUMEN_FUSION__config_embryo_heavyPreprocess_CellLumenFusion_smoke.yaml`
  - Local smoke config for probing Cell Lumen fusion with the heavy preprocessing path still enabled.
  - Not recommended for the current fusion direction because heavy preprocessing can erase dim late embryo signal.

## Important Notes

- Do not run from the copied files in this index unless the `base_config` paths are checked first.
- For normal runs, keep using the original config paths in `C++/config`.
- If a copied config becomes the new official runtime file, update the original top level config first, then refresh the copy here.
