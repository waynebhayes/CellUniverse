# CellLumen early 000 to 085 tuning index

This file records the early CellLumen tuning runs so each YAML can be matched
to its validation result. All runs used standalone CellLumen with TIFF output
disabled. Predicted CSV z values were compared after multiplying z by 7.0.
GT matching used a 20 px nearest-neighbor threshold in interpolated 3D space.

YAML root:
`/Users/wangyiding/CellUniverse/C++/config/BackUp_TuningHistory_CellLumen/Celegans_Early000-085_tuning`

Output root:
`/Users/wangyiding/CellUniverse/C++/output/F000-085_CellLumenEarlyTune0601`

Run logs also print the resolved CellLumen config block, but the YAML files in
this directory are the source records for what was changed in each version.

## Summary table

| Version | YAML | Output | Frames | Perfect | No miss | Missing | Extra | Result note |
|---|---|---|---:|---:|---:|---:|---:|---|
| v001 | baseline clean config | v001_baseline_clean_representative_noTif | 18 | 0 | 14 | 6 | 372 | high recall, too many extras |
| v002 | config_cell_lumen_embryo_early_v002_percentileOnly_highThreshold.yaml | v002_percentileOnly_highThreshold_representative_noTif | 18 | 2 | 4 | 89 | 61 | too many misses |
| v003 | config_cell_lumen_embryo_early_v003_percentileOnly_fullSweep_sparseSplit.yaml | v003_percentileOnly_fullSweep_sparseSplit_representative_noTif | 18 | 4 | 7 | 45 | 1484 | severe extra explosion |
| v004 | config_cell_lumen_embryo_early_v004_watershedWideSeeds_noLowRescue.yaml | v004_watershedWideSeeds_noLowRescue_representative_noTif | 16 | 0 | 12 | 6 | 352 | high recall, too many extras |
| v005 | config_cell_lumen_embryo_early_v005_densitySwitch_percentileThenWatershed_noLowRescue.yaml | v005_densitySwitch_percentileThenWatershed_noLowRescue_representative_noTif | 18 | 4 | 10 | 18 | 177 | density switch helped but extras high |
| v006 | config_cell_lumen_embryo_early_v006_densitySwitch_merge28_noLowRescue.yaml | v006_densitySwitch_merge28_noLowRescue_representative_noTif | 18 | 7 | 10 | 18 | 10 | cleaner, still misses |
| v007 | config_cell_lumen_embryo_early_v007_densitySwitch_merge28_scaledZSeedSplit.yaml | v007_densitySwitch_merge28_scaledZSeedSplit_noTif | 18 | 6 | 9 | 20 | 12 | scaled Z seed split did not help |
| v008 | config_cell_lumen_embryo_early_v008_densitySwitch_merge28_sparseGrowthRelax.yaml | v008_densitySwitch_merge28_sparseGrowthRelax_noTif | 18 | 7 | 10 | 16 | 10 | small improvement over v006 |
| v009 | config_cell_lumen_embryo_early_v009_densitySwitch_noFragmentMerge_sparseGrowthRelax.yaml | v009_densitySwitch_noFragmentMerge_sparseGrowthRelax_probe_noTif | 7 | 1 | 4 | 7 | 25 | less merge raised recall but extras rose |
| v010 | config_cell_lumen_embryo_early_v010_densitySwitch_tighterSparseFragmentMerge.yaml | v010_densitySwitch_tighterSparseFragmentMerge_probe_noTif | 7 | 2 | 6 | 5 | 15 | better probe result |
| v011 | config_cell_lumen_embryo_early_v011_densitySwitch_balancedSparseFragmentMerge.yaml | v011_densitySwitch_balancedSparseFragmentMerge_probe_noTif | 7 | 2 | 6 | 5 | 11 | balanced fragment merge |
| v012 | config_cell_lumen_embryo_early_v012_balancedMerge_highPercentilePreference.yaml | v012_balancedMerge_highPercentilePreference_probe_noTif | 7 | 2 | 6 | 5 | 8 | slightly fewer extras |
| v013 | config_cell_lumen_embryo_early_v013_sparseFloorFilter.yaml | v013_sparseFloorFilter_probe_noTif | 7 | 6 | 6 | 5 | 3 | sparse floor filter very useful |
| v014 | config_cell_lumen_embryo_early_v014_sparseFloorFilter_lowRescue.yaml | v014_sparseFloorFilter_lowRescue_probe_noTif | 7 | 0 | 5 | 2 | 47 | low rescue too permissive |
| v015 | config_cell_lumen_embryo_early_v015_sparseFloorFilter_countGatedLowRescue.yaml | v015_sparseFloorFilter_countGatedLowRescue_probe_noTif | 7 | 6 | 6 | 5 | 3 | count gate restored v013 behavior |
| v016 | config_cell_lumen_embryo_early_v016_sparseFloorFilter_countGatedLowRescue4.yaml | v016_sparseFloorFilter_countGatedLowRescue4_probe_noTif | 7 | 6 | 6 | 1 | 2 | better than v015 |
| v017 | config_cell_lumen_embryo_early_v017_sparseFloorFilter_countGatedLowRescue4_weakSignalFloor.yaml | v017_sparseFloorFilter_countGatedLowRescue4_weakSignalFloor_probe_noTif | 7 | 6 | 6 | 1 | 2 | same as v016 |
| v018 | config_cell_lumen_embryo_early_v018_sparseFloorFilter_countGatedLowRescue4_weakSignalFloor_radius186.yaml | v018_sparseFloorFilter_countGatedLowRescue4_weakSignalFloor_radius186_probe_noTif | 7 | 6 | 6 | 1 | 1 | one extra removed |
| v019 | config_cell_lumen_embryo_early_v019_sparseFloorFilter_lowRescue4_weakFloor_zstrong.yaml | v019_sparseFloorFilter_lowRescue4_weakFloor_zstrong_probe_noTif | 7 | 6 | 7 | 0 | 1 | first no-miss probe |
| v020 | config_cell_lumen_embryo_early_v020_sparseFloorFilter_lowRescue4_weakFloor_zstrong_count20to30.yaml | v020_sparseFloorFilter_lowRescue4_weakFloor_zstrong_count20to30_probe_noTif | 7 | 7 | 7 | 0 | 0 | probe perfect |
| v020 full | same as v020 | v020_full_f000_085_noTif | 86 | 52 | 67 | 26 | 33 | best full 000 to 085 run so far |
| v021 | config_cell_lumen_embryo_early_v021_weakSignalFloorOnly_zstrong_count20to30.yaml | v021_weakSignalFloorOnly_zstrong_count20to30_probe_noTif | 14 | 4 | 11 | 6 | 14 | weak floor alone not enough |
| v022 | config_cell_lumen_embryo_early_v022_densityBandedFloor_zstrong_count20to30.yaml | v022_densityBandedFloor_zstrong_count20to30_probe_noTif | 14 | 8 | 11 | 6 | 10 | improved clean probe |
| v023 | config_cell_lumen_embryo_early_v023_densityBandedFloor_scaledZSeedSplit.yaml | v023_densityBandedFloor_scaledZSeedSplit_probe_noTif | 14 | 7 | 11 | 6 | 31 | scaled Z seed split caused extras |
| v024 | config_cell_lumen_embryo_early_v024_densityBandedFloor_scaledZSplitOnly.yaml | v024_densityBandedFloor_scaledZSplitOnly_probe_noTif | 14 | 8 | 11 | 6 | 10 | same as v022 |
| v025 | config_cell_lumen_embryo_early_probe_v025_only9905_densityBandedFloor.yaml | v025_only9905_probe_noTif | 4 | 1 | 2 | 4 | 15 | diagnostic only, helped f028 |
| v026 | config_cell_lumen_embryo_early_v026_densityBandedFloor_lessFragmentMerge.yaml | v026_lessFragmentMerge_probe_noTif | 14 | 3 | 13 | 2 | 43 | high recall, too many extras |
| v027 | config_cell_lumen_embryo_early_v027_densityBandedFloor_weakSignal10to30.yaml | v027_weakSignal10to30_probe_noTif | 14 | 8 | 11 | 6 | 8 | cleanest probe before split tests |
| v028 | config_cell_lumen_embryo_early_v028_weakSignal10to30_zPeakSplit8to14.yaml | v028_weakSignal10to30_zPeakSplit8to14_probe_noTif | 14 | 6 | 11 | 4 | 18 | Z split reduced misses but added extras |
| v029 | config_cell_lumen_embryo_early_v029_zPeakSplit_stricter.yaml | v029_zPeakSplit_stricter_probe_noTif | 14 | 6 | 11 | 4 | 18 | stricter Z split did not improve |
| v030 | config_cell_lumen_embryo_early_v030_zPeakSplit_parentSignalGuard.yaml | v030_zPeakSplit_parentSignalGuard_probe_noTif | 14 | 6 | 11 | 4 | 18 | parent signal guard did not improve |
| v031 | config_cell_lumen_embryo_early_v031_zPeakSplit_noFinalFilters_highRecall.yaml | v031_zPeakSplit_noFinalFilters_highRecall_probe_noTif | 14 | 0 | 10 | 5 | 93 | final filters were not the main cause |
| v032 | config_cell_lumen_embryo_early_v032_watershedPrefer_count10to14_highRecall.yaml | v032_watershedPrefer_count10to14_highRecall_probe_noTif | 14 | 4 | 10 | 14 | 375 | watershed prefer too broad |
| v033 | config_cell_lumen_embryo_early_v033_watershedPrefer_count10to14_closeSeeds.yaml | v033_watershedPrefer_count10to14_closeSeeds_probe_noTif | 7 | 0 | 1 | 58 | 112 | close seeds made watershed collapse |
| v034 | config_cell_lumen_embryo_early_v034_percentile9915_densityBanded.yaml | v034_percentile9915_densityBanded_probe_noTif | 5 | 1 | 2 | 6 | 15 | fixed f024 only, not general |
| v035 | config_cell_lumen_embryo_early_v035_local3DPeakSplit_prefer.yaml | v035_local3DPeakSplit_prefer_probe_noTif | 7 | 0 | 3 | 7 | 19 | local 3D split preferred path regressed hard frames |
| v036 | config_cell_lumen_embryo_early_v036_watershedRescue_highRecall.yaml | v036_watershedRescue_highRecall_probe_noTif | 9 | 4 | 8 | 2 | 10 | best new high recall probe; only f018 still misses |
| v036 full | config_cell_lumen_embryo_early_v036_watershedRescue_highRecall.yaml | v036_full_f000_085_noTif | 86 | 37 | 71 | 19 | 60 | full sweep improves no-miss over v020, but still misses 15 frames |
| v037 | config_cell_lumen_embryo_early_v037_watershedRescue_lessFinalMerge.yaml | v037_watershedRescue_lessFinalMerge_probe_noTif | 9 | 0 | 7 | 2 | 107 | lowering final merge is too broad |
| v038 | config_cell_lumen_embryo_early_v038_watershedRescue_lessMerge_zPeakSplit.yaml | v038_watershedRescue_lessMerge_zPeakSplit_probe_noTif | 9 | 0 | 7 | 2 | 115 | f018 becomes no-miss, but f021 and f035 regress |
| v047 stress | config_cell_lumen_embryo_early_v047_initialPrior_clusterCollapse.yaml | CellLumen_EarlyLogicTests_20260613/v047_initialPrior_worst5_noTif | 5 | 1 | 5 | 0 | 6 | uses legal initial.csv spacing scale to collapse early internal bright fragments; no miss on v046 stress frames |
| v047 full | config_cell_lumen_embryo_early_v047_initialPrior_clusterCollapse.yaml | CellLumen_EarlyLogicTests_20260613/v047_initialPrior_full_f000_085_noTif | 86 | 33 | 48 | 246 | 124 | maxCells 90 is too broad; dense frames after about f052 over merge badly |
| v050 key | config_cell_lumen_embryo_early_v050_initialPrior_lowDensity_max60.yaml | CellLumen_EarlyLogicTests_20260613/v050b_postCollapseLowDensity_keyframes_noTif | 10 | 3 | 7 | 3 | 8 | best balanced key-frame test so far; fixes f000/f002 dust and keeps f014/f035, but f004/f018/f036 still miss |

## Current best choices

For current no-miss early CellLumen output, v107 is the best full run so far:

- YAML: `config_cell_lumen_embryo_early_v107_v105_groupGate25.yaml`
- Output: `/Users/wangyiding/CellUniverse/C++/output/CellLumen_EarlyLogicTests_20260614/v107_groupGate25_full_f000_085_noTif`
- 86 frames checked from f000 to f085
- 86 frames with no GT miss
- total missing = 0
- total extra = 1478
- max extra = 46 at f053
- median extra = 15
- no TIFF files were generated

This is now the early-profile source for
`/Users/wangyiding/CellUniverse/C++/config/C.elegans developing embryo/Concentrated/C_elegans_DensityAuto_Best.yaml`.

For historical low-extra reference, v020 full is still useful:

- 86 frames checked from f000 to f085
- 52 perfect frames
- 67 frames with no GT miss
- 19 frames still have GT misses
- total missing = 26
- total extra = 33

For high recall diagnostics, v026 is useful but not final:

- 14 probe frames checked
- 13 frames with no GT miss
- total missing = 2
- total extra = 43
- this shows weaker fragment merge can recover some missed GT cells, but it
  creates too many extras.

## Full v020 missing frames

The full v020 run still missed GT cells in:

`f018, f021, f024, f028, f033, f036, f045, f051, f053, f058, f059, f060, f061, f064, f065, f078, f080, f083, f085`

## Main tuning lessons so far

1. Early frames f000 to f017 are already no-miss under v020. Only f006 and
   f009 have one extra each.
2. The first hard failure is f018. The missed pair is mostly an XY local split
   problem, not just a Z split problem.
3. Z-only peak split can reduce some misses, but it also splits normal cells.
4. Turning off final filters does not solve f018, f021, or f028.
5. Picking seeded watershed by count 10 to 14 helps some frames but explodes
   extras and hurts later probe frames.
6. The successful late dense tuning pattern was to keep high recall candidate
   output separate from clean final output. Early tuning should follow the same
   idea instead of trying to make one standalone CSV perfect immediately.
7. Local 3D peak split was tested in v035. It is not adopted because it made
   f018, f024, and f028 worse on the probe set.
8. Watershed rescue in v036 is the best new direction. It fixes f021 and f028
   with limited extra cells, but f018 still has two missed GT cells.
   Full 000 to 085 v036 improves no-miss frames from 67 in v020 to 71, while
   increasing extras from 33 to 60.
9. Lowering final duplicate merge in v037 partly helps f018 but produces too
   many extras and introduces a new miss in f035.
10. Combining less merge with Z peak split in v038 proves f018 can be rescued,
    but the same settings are too broad and damage f021 and f035.
11. v047 keeps the v046 cluster collapse direction but computes the collapse
    distance from initial.csv spacing instead of only a fixed constant. On the
    five v046 stress frames it keeps zero missing GT cells and reduces total
    extras from 7 to 6. It still needs a full f000 to f085 no-tif sweep.
12. The full v047 sweep proves the collapse max cell gate cannot stay at 90.
    It starts over merging true dense candidates after about f052.
13. v050 with maxCells 60 plus the default-off low density artifact filter is
    the best current early key-frame candidate, but it still needs a cluster
    diameter or two-lobe safety guard before a full sweep is meaningful.
14. v076 is the current best early high-recall candidate. It returns to the
    stable v043 base, uses wider local x/y brightness refinement, freezes local
    z movement, and allows only upward z-column recovery. The full f001 to f085
    sweep reached 85 / 85 no-miss frames with total_missing = 0 and
    total_extra = 2123. The max extra frame is f053 with 46 extra candidates,
    and no TIFF files were left in the output directory.
