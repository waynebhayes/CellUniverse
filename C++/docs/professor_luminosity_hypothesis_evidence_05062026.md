# Professor Luminosity Hypothesis Evidence

## Question

Professor Hayes's hypothesis: before a nucleus splits, its segmented size may shrink while voxel-level brightness increases, but the total luminosity of the object may remain roughly constant. If true, luminosity could become a useful split cue.

## Measurement Logic

- Raw TIFF stacks were read directly again; this test does not use normalized image values as the final luminosity measurement.
- Cell centers and split events came from the existing brightness/volume analyzer output.
- Fixed-window luminosity used the same 3D ellipsoid radius for every cell in a run, inferred from the median tracked diameter and `z_scaling`.
- Adaptive ellipsoid luminosity used each tracked cell's current diameter, with the same `z_scaling` conversion for z radius.
- Local background was estimated from a surrounding shell around the cell using median raw intensity.
- Background-subtracted luminosity was computed as `sum(max(raw_voxel - local_background, 0))` inside the signal mask.
- For each split, the test compared the parent in the last frame before split with the non-overlapping union of daughter masks at the split frame or within two frames after it.
- Daughter masks were measured as a union to avoid double-counting overlapping voxels when the two daughters are still close.

## Summary

- Total split events tested: 26
- Analyzable split events with parent and both daughters found: 26
- Fixed-window shrink+brighten events before split: 11/26
- Adaptive-ellipsoid shrink+brighten events before split: 3/26
- Fixed-window events supporting the full hypothesis: 0/26
- Adaptive-ellipsoid events supporting the full hypothesis: 0/26

## Aggregate Numerical Evidence

- Median fixed-window pre-split luminosity CV: 0.2025
- Median adaptive-ellipsoid pre-split luminosity CV: 0.1318
- Median fixed-window daughter/parent luminosity ratio: 2.5000
- Median adaptive-ellipsoid daughter/parent luminosity ratio: 0.4850
- Fixed-window daughter/parent ratios within [0.75, 1.25]: 0/26
- Adaptive-ellipsoid daughter/parent ratios within [0.75, 1.25]: 0/26

## Run: early_13_40

- Analysis directory: `/Users/wangyiding/CellUniverse/C++/output/brightness_f013_to_f040_20260506_181423/brightness_volume_analysis`
- Observations measured: 265
- Split events: 8
- Analyzable split events: 8
- Fixed signal radius: xy=19.500, z=3.000
- Median fixed-window pre-split CV: 0.1899
- Median ellipsoid pre-split CV: 0.2869
- Median fixed daughter/parent ratio: 2.2787
- Median ellipsoid daughter/parent ratio: 0.4850
- Fixed-window shrink+brighten events: 5/8
- Adaptive-ellipsoid shrink+brighten events: 0/8
- Fixed-window full-support events: 0/8
- Adaptive-ellipsoid full-support events: 0/8

## Run: mid_53_105

- Analysis directory: `/Users/wangyiding/CellUniverse/C++/output/brightness_f053_to_f105_20260506_181558/brightness_volume_analysis`
- Observations measured: 583
- Split events: 18
- Analyzable split events: 18
- Fixed signal radius: xy=19.000, z=3.000
- Median fixed-window pre-split CV: 0.2184
- Median ellipsoid pre-split CV: 0.1081
- Median fixed daughter/parent ratio: 3.2168
- Median ellipsoid daughter/parent ratio: 0.4806
- Fixed-window shrink+brighten events: 6/18
- Adaptive-ellipsoid shrink+brighten events: 3/18
- Fixed-window full-support events: 0/18
- Adaptive-ellipsoid full-support events: 0/18

## Conclusion

Under this more faithful test, the current embryo runs do not support using constant total luminosity as a reliable standalone split rule yet. The shrink+brighten pattern exists in some split parents, but the parent-to-daughter luminosity conservation test is not consistently satisfied across events.

This does not disprove the biological idea. It means the current measurement pipeline, tracking centers, and mask assumptions do not yet produce a stable enough luminosity signal to safely affect split acceptance. The metric should remain diagnostic until validated on more ground-truth split events.

## Generated Evidence Files

- Split-level evidence CSV: `/Users/wangyiding/CellUniverse/C++/output/professor_luminosity_hypothesis_20260506/split_parent_daughter_luminosity_evidence.csv`
- Observation-level evidence CSV for early_13_40: `/Users/wangyiding/CellUniverse/C++/output/professor_luminosity_hypothesis_20260506/early_13_40_fair_luminosity_observations.csv`
- Observation-level evidence CSV for mid_53_105: `/Users/wangyiding/CellUniverse/C++/output/professor_luminosity_hypothesis_20260506/mid_53_105_fair_luminosity_observations.csv`
