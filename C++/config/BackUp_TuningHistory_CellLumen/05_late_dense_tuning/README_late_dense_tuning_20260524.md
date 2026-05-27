# Late dense CellLumen config archive - 2026-05-24

This folder contains YAML configs from the 150 to 194 late dense tuning pass.

Two stable long term entry configs also exist in the top level config folder:

- `C++/config/config_cell_lumen_embryo_standalone_clean.yaml`
  is the clean CellLumen output mode.
- `C++/config/config_cell_lumen_embryo_celluniverse_preprocess_high_recall.yaml`
  is the high recall CellLumen mode for Cell Universe preprocessing.

## Selected configs

- `00_SELECTED_final_configs/FINAL_CLEAN_OUTPUT__config_embryo_late_dense_count100_minvox800_no_weak.yaml`
  Clean final output version. It removes weak satellite candidates and keeps false positives low.

- `00_SELECTED_final_configs/FINAL_HIGH_RECALL_CANDIDATE__config_embryo_late_dense_count100_minvox800_no_final_filters.yaml`
  High recall candidate version. It keeps more near GT duplicate candidates and is more suitable as a candidate source for Cell Universe than as final CellLumen output.

- `00_SELECTED_final_configs/ACTIVE_config_embryo_current_combined_DO_NOT_EDIT_RUNTIME_COPY.yaml`
  Snapshot copy of the active runtime `C++/config/config_embryo.yaml`. The real active file remains in `C++/config/config_embryo.yaml`.

## Supporting configs

- `01_positive_reference_configs_not_final/`: variants that improved part of the behavior but were not selected.
- `02_controls_or_not_selected_configs/`: controls and variants that were useful records but not chosen.
- `99_index/late_dense_config_organization_index.csv`: exact source to organized path mapping.
