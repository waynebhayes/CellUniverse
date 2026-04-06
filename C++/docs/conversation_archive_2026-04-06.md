# Conversation Archive - 2026-04-06

## Scope

This archive summarizes the follow-up tuning and split-logic changes made after the 2026-04-05 archive. The focus of this session was:

- overlap and size-shrink penalties
- preprocessing simplification and normalization consistency
- split fake-positive / fake-negative guards
- split-axis and daughter-placement constraints
- debug-log analysis of later-frame split behavior

## Overlap Penalty Review

- Traced the overlap penalty implementation in `src/Frame.cpp`.
- Confirmed the penalty is a soft additive cost based on:
  - center distance
  - sum of major radii
  - squared normalized overlap depth
- Clarified:
  - it does not hard-block splits
  - it only discourages overlap through the total cost
  - it can help reject some false splits, but only indirectly

## Perturbation Size-Reduction Penalty

- Added a new perturbation-only soft penalty for radius shrinkage.
- Added YAML-configurable parameter:
  - `prob.size_reduction_penalty_weight`
- Current behavior:
  - if `majorRadius` and/or `minorRadius` decreases during a perturbation proposal
  - add a quadratic penalty proportional to relative shrinkage
- Initial YAML value was set small:
  - `2.0`

## Size Constraints Clarified

- Verified there is no soft growth penalty by default.
- Existing hard constraints remain:
  - `minMajorRadius`
  - `maxMajorRadius`
  - `minMinorRadius`
  - `maxMinorRadius`
- Constructor still clamps radii into those bounds.

## Preprocessing Pipeline Changes

### Removed Post-Sigmoid Blur Subtraction

- Removed the old post-sigmoid blurred subtraction step from the actual pipeline.
- Removed dead config support for:
  - `simulation.post_sigmoid_subtract_sigma`
- The active preprocessing order is now:
  - grayscale
  - normalize to `[0,1]`
  - safe Gaussian blur
  - sigmoid
  - post-sigmoid dim-region subtraction
  - z interpolation

### Stack-Wide Scalar Decisions

- Updated preprocessing so scalar image-stat decisions are computed from the whole stack rather than per-slice or from one calibration plane.
- Specifically:
  - sigmoid-center calibration now uses the calibration ROI over the full stack
  - post-sigmoid dimmest-percentile subtraction now computes one stack-wide cutoff

### Smoothed Dimmest-Percentile Subtraction

- Refined the post-sigmoid subtraction so bright pixels are no longer penalized.
- Current behavior:
  - pixels above the configured cutoff are left unchanged
  - pixels below the cutoff are reduced
  - a transition band near the cutoff smoothly tapers the subtraction
- Added YAML-configurable parameters:
  - `simulation.post_sigmoid_dimmest_transition_width`
  - `simulation.post_sigmoid_dimmest_transition_gradient`

## Outline Rendering

- Updated `drawOutline()` intensity to:
  - `0.25`

## Perturbation Probability Controls

- Split perturbation sign probability into separate increase / decrease controls.
- Added support in `PerturbParams` for:
  - `increase_prob`
  - `decrease_prob`
- Kept legacy `prob` as a backward-compatible fallback.
- Updated current YAML to use the separate sign probabilities.

## Brightness Controls

### Removed Frame-0 Radius Snap-Back

- Removed the logic that restored frame-0 radii back to the initial CSV after optimization.
- The optimizer now keeps the radii it converges to on the first frame.

### Per-Cell Brightness Mean Amplification

- Added a YAML-configurable amplification multiplier for measured per-cell brightness.
- Added parameter:
  - `cell.brightnessMeanAmplification`
- Current behavior:
  - measured mean brightness is multiplied by this factor before blending into cell brightness
- Default / initial value:
  - `1.0`

## Split-Fake Guards Added

### Daughter Overlap-Volume Rejection

- Added a fake-split rejection rule after split burn-in:
  - approximate daughter overlap volume using equivalent-volume spheres
  - reject split if overlap fraction exceeds threshold
- Added YAML-configurable parameter:
  - `prob.split_fake_overlap_volume_fraction_threshold`
- Current tuned YAML value observed in log:
  - `0.15`

### Daughter Radius-Ratio Rejection

- Added another post-burn-in fake-split rejection rule:
  - compute daughter size ratio
  - reject if one daughter is too large relative to the other
- Added YAML-configurable parameter:
  - `prob.split_fake_radius_ratio_threshold`
- Current tuned YAML value observed in log:
  - `1.6`

### Bridge-Brightness Fake-Split Rejection

- Added a new post-burn-in rejection rule based on bridge continuity.
- Current behavior:
  - build a cylinder along the centroid-connection line
  - size it once using daughter 1 volume, once using daughter 2 volume
  - measure real-image brightness inside those cylinders
  - compare bridge brightness against each daughter’s mean brightness
  - reject split if the bridge is still almost as bright as both daughters
- Added YAML-configurable parameter:
  - `prob.split_fake_bridge_brightness_similarity_threshold`
- Initial value:
  - `0.9`

## Split-Axis / Geometry Constraints

### Minor-Axis Alignment Tolerance

- Added a split-direction constraint based on the cell’s local minor axis (local `z` axis).
- The tolerance unit is degrees, not radians.
- Added YAML-configurable parameter:
  - `prob.split_minor_axis_alignment_tolerance_degrees`

### Flatness-Gated Axis Constraint

- Refined the above constraint so it only activates for sufficiently flat cells.
- Added YAML-configurable parameter:
  - `prob.split_minor_axis_alignment_flatness_ratio_threshold`
- Current behavior:
  - only enforce / steer to minor-axis direction when:
    - `minorRadius / majorRadius <= threshold`
- Initial threshold:
  - `0.5`

### Steer Instead of Reject

- Changed the axis constraint behavior:
  - do not reject a split when PCA axis disagrees with the minor axis
  - instead, if the flatness gate is active and the angle exceeds tolerance, force the split axis onto local `z`

### Daughter Centers Forced Onto One Line

- Refined daughter placement so the two daughter centers are projected onto the chosen split line through `pcaCenter`.
- This ensures the daughters are not just orientation-consistent; their centers are also approximately collinear.

## Split Logic Clarifications Captured

- Split and perturb are mutually exclusive per optimizer iteration:
  - random cell
  - maybe split
  - else perturb
- A non-dividing cell can still be selected for a split attempt.
- If a split attempt fails:
  - it reverts to the original parent
  - the cell is blacklisted from more split attempts that frame

## Debug-Log Analysis

### Frame 3 Earlier Run

- Diagnosed a false negative caused by:
  - `split_fake_radius_ratio_threshold`
- Evidence:
  - overlap was acceptable
  - radius-ratio guard alone rejected an otherwise plausible split

### Frame 5 to Frame 6 Run (`output_ubuntu_20260406_004655`)

- Reviewed the user-provided debug log:
  - `/home/blue-lobster/p2/UCI/CS295p/images/outputs/output_ubuntu_20260406_004655/debug_log.txt`
- Found the key pattern:
  - frame 5 ended with 8 cells and no accepted splits
  - frame 6 accepted 3 splits and ended with 11 cells
- Likely cause of false positives:
  - split proposals being driven by weakly elongated cells
  - very selective bright-point PCA using:
    - `selected_fraction=0.05`
  - permissive split gate:
    - `split_elongation_threshold=1.1`
  - burn-in finding lower image cost without enough biological plausibility checks
- Strongest diagnosis:
  - `splitBrightestFraction=0.05` is probably too low and makes PCA hotspot-sensitive
  - `split_elongation_threshold=1.1` is likely too permissive

## Current Config Snapshot Observed in Debug Log

- `sigmoid_k = 60`
- `post_sigmoid_dimmest_percentile = 0.99`
- `post_sigmoid_dimmest_transition_width = 0.2`
- `post_sigmoid_dimmest_transition_gradient = 0.5`
- `split_elongation_threshold = 1.1`
- `split_fake_overlap_volume_fraction_threshold = 0.15`
- `split_fake_radius_ratio_threshold = 1.6`
- `split_minor_axis_alignment_tolerance_degrees = 10`
- `split_minor_axis_alignment_flatness_ratio_threshold = 0.5`
- `splitBrightestFraction` observed in logs as:
  - `0.05`

## Recommended Next Tuning Moves

Highest-priority tuning ideas discussed but not yet applied:

- increase `cell.splitBrightestFraction` from `0.05` to `0.1` or `0.15`
- raise `prob.split_elongation_threshold` from `1.1` to `1.2` or `1.25`
- consider lowering `prob.max_split_probability`
- watch whether the new bridge-brightness guard reduces continuous-blob false positives

## Files Touched During This Session

- `config/config.yaml`
- `includes/ConfigTypes.hpp`
- `includes/Frame.hpp`
- `includes/Spheroid.hpp`
- `includes/CellUniverse.hpp`
- `src/CellUniverse.cpp`
- `src/Frame.cpp`
- `src/Spheroid.cpp`

## Notes for Next Session

- Many split heuristics are now layered:
  - overlap-volume rejection
  - radius-ratio rejection
  - flatness-gated minor-axis steering
  - line-constrained daughter placement
  - bridge-brightness rejection
- The biggest unresolved system-level issue still appears to be split proposal quality, not just split rejection quality.
- The next session should likely focus on:
  - improving PCA point selection stability
  - making split probability / elongation gating less eager
  - validating whether the new bridge-brightness rejection catches the frame-5-to-6 false positives
