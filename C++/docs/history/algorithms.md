# Core Algorithms

> **⚠ 2026-04-14 update:** the authoritative current pipeline is in
> `C++/docs/pipeline.md`. Read that first. Sections below describing
> classification, radius EMA, radius gradient, volume floor, Phase A/B,
> `shape_elongation_classify_threshold`, or the symmetric L2 cost are
> OUTDATED. Current pipeline uses: snap-mask PCA shape fit (no floor),
> asymmetric L2 cost (k=6), classification REMOVED, pre-pass for ALL
> cells, linear P(split) ramp, unified perturb+split loop, bridge gate
> two-tier (vr≥0.95 hard, density+brightness soft).

## Image Preprocessing Pipeline (CellUniverse.cpp: loadFrame)

Active pipeline (as of yp_yd_merge_04072026, following 2026-04-05/06 rework):

1. Load multi-page TIFF via `cv::imreadmulti()`
2. Convert BGR -> Grayscale, normalize to CV_32F [0,1]
3. **Safe/masked Gaussian blur** (`simulation.blur_sigma`) — unlike plain blur, this avoids bleeding zero-valued black borders into valid image content
4. **Calibrate sigmoid center (percentile-based)**: compute a percentile of brightness in the calibration ROI *over the full stack* (not per-slice). `sigmoidCenter = percentileValue` where the percentile is `simulation.sigmoid_center_percentile`. This REPLACED the old `bgMean + sigmoid_center_offset` logic — the `sigmoid_center_offset` field is now parsed but dead (see `C++/src/CellUniverse.cpp:422-448`).
5. **Sigmoid contrast**: `output = 1 / (1 + exp(-k * (input - center)))` with `simulation.sigmoid_k` (was 75, now commonly tuned lower like 60). Cells → ~1.0, background → ~0.0.
6. **Post-sigmoid dim-region subtraction** (added 04-06, replaces the 04-05 blur-subtraction step which was removed):
   - Compute a stack-wide dimmest percentile of the sigmoid-output stack: `simulation.post_sigmoid_dimmest_percentile`
   - Pixels **above** the cutoff are left unchanged
   - Pixels **below** the cutoff are reduced
   - A **transition band** near the cutoff smoothly tapers the subtraction: `simulation.post_sigmoid_dimmest_transition_width` + `simulation.post_sigmoid_dimmest_transition_gradient`
   - Clamp negatives to zero
7. **Z-interpolation**: 33 raw slices → 225 interpolated slices (`simulation.z_scaling`=7). Linear LERP. Formula: `numSynth = z_scaling * (numTiff - 1) + 1`

**Adaptive synthetic background (frame 2+):** After loading frame N, gather real-image voxels NOT occupied by the slightly-expanded previous-frame cells. Compute the mean of the brightest `simulation.adaptive_background_top_fraction` of those voxels, scaled by `(current mean brightness / previous mean brightness)`. Use this to overwrite the synthetic frame's background. Expand factor: `simulation.adaptive_background_expand_factor`.

## Unified Stochastic Loop (CellUniverse.cpp: optimize)

Single loop replaces the former Phase 1 (perturbation) + Phase 2 (split detection). `iterations = num_cells * iterations_per_cell` (e.g., 6*350 = 2100). Each iteration is **either** a perturbation **or** a split attempt — mutually exclusive.

Each iteration:
1. Pick random cell index
2. Compute PCA elongation (cached — one scan per cell per frame)
3. Compute raw P(split) per cell: `rawP = baseSplitProb + max(0, 1 - 1/previousElongation)`
   - Spherical cell (elongation=1.0): raw = 0.03 (base only)
   - Elongated cell (elongation=1.5): raw ≈ 0.36
4. **Rescale all cells' raw P(split) together** so the maximum becomes `max_split_probability`, preserving relative ratios. The old `min(0.5, ...)` cap has been REPLACED with this proportional rescaling.
5. If `rand < P(split)` AND cell not blacklisted: try split via `trySplitCell()`
6. Else: try perturbation via `perturbCell()`
7. Accept if total cost (image L2 + overlap penalty + size-reduction penalty) improved, else revert

**No splits on frame 1.** No time has elapsed for division.

**Split blacklist**: After a failed split attempt, the cell is blacklisted for the rest of the frame (max 1 burn-in per cell per frame). A non-dividing cell can still be selected for a split attempt but will be blacklisted after the first failure.

## Perturbation (Frame.cpp: perturbCell)

1. Save old cell state
2. `getPerturbedCell()` — apply Gaussian offsets to (x, y, z, majorR, minorR, thetaX/Y/Z) each with independent probability
   - `PerturbParams` now supports `increase_prob` / `decrease_prob` split (legacy `prob` still works as fallback)
   - Random brightness perturbation (`cell.brightness.prob`) is **disabled** (set to 0) — brightness is updated per-frame via EMA from the real image, not by random proposals
3. Compute overlap penalty for this cell before and after (O(n) via `computeOverlapForCell`)
4. **Size-reduction soft penalty**: if `majorRadius` or `minorRadius` decreased, add a quadratic penalty proportional to relative shrinkage, weighted by `prob.size_reduction_penalty_weight`. Counters cell collapse-to-minimum. No corresponding growth penalty.
5. Fast re-render only affected z-region via `generateSynthFrameFast()`
6. Compare total cost: `(newImageCost + newOverlap + newShrinkPenalty) - (oldImageCost + oldOverlap + oldShrinkPenalty)`
7. Accept if total cost decreased, else revert

## Per-Frame Per-Cell Brightness Update (CellUniverse.cpp / Spheroid.cpp)

At the start of each frame (after `copyCellsForward` from the previous frame), for every cell:
1. Measure the mean real-image brightness inside the cell's volume — `Spheroid::measureMeanBrightness()` scans the 3D bounding box and averages pixels where the ellipsoid test passes
2. Amplify: `observed *= cell.brightnessMeanAmplification` (default 1.0)
3. Blend into the existing `_brightness` via EMA: `new = old * (1 - blend) + observed * blend` where blend = `cell.brightnessUpdateBlend`
4. The updated `_brightness` is then used by `draw()` for the entire frame's rendering

On frame 1, each cell is seeded with `_brightness = simulation.cell_color`. There is no frame-1 ground-truth radius snap-back anymore (it was removed on 2026-04-06).

## Split Detection (Frame.cpp: trySplitCell + Spheroid.cpp: getSplitCells)

### getSplitCells() — PCA-Based Daughter Placement

1. **Search region**: 3D bounding box centered at cell position, radius = 3 * maxR
2. **Brightness threshold**: mean brightness of pixels inside cell boundary
3. **Neighbor exclusion**: Skip bright pixels closer to any other cell center than to self
4. **Centroid recenter**: If bright-pixel centroid drifts >5px from PCA center, re-collect from true centroid
5. **PCA with isotropic normalization**: Divide all axes by majorRadius (NOT per-axis stddev). Preserves true shape.
6. **Elongation ratio**: lambda1/lambda2. Used to drive adaptive P(split), not as a hard gate.
7. **Centroid-based placement**: Project bright pixels onto split axis, partition into two groups, compute centroid of each group.
8. **Daughter sizing**: `volumeScale = cbrt(0.5) ≈ 0.794`. `daughterR = max(currentR, preOptR) * 0.794`. Daughters inherit parent brightness.

### trySplitCell — Layered Soft Guards + Burn-in + Post-Burn-in Filters

The split pipeline is a **stack of soft filters**. Any of them can reject, none of them hard-block overlap. Order:

1. `valid == false` from `getSplitCells` → skip
2. **Elongation threshold gate**: `lambda1/lambda2 < prob.split_elongation_threshold` (1.5 default, tuned lower like 1.1 in recent runs) → skip burn-in
3. **Pre-burn-in PCA gating** (Yiding's line, added 2026-04-04):
   - `separationOverDaughterMajor < prob.split_pre_burn_in_min_separation_over_major` (~0.35) → reject (collapsed proposal)
   - Z-axis-dominant split with weak transverse separation (`prob.split_pre_burn_in_z_axis_*` thresholds) → reject
4. **Split axis steering** (added 2026-04-06): if the cell is flat enough (`minorRadius/majorRadius <= prob.split_minor_axis_alignment_flatness_ratio_threshold`, default 0.5) AND the PCA axis disagrees with local z by more than `prob.split_minor_axis_alignment_tolerance_degrees` (DEGREES, not radians), the split axis is **steered onto local z** — the split is NOT rejected, just redirected
5. **Daughter placement**: daughters are placed along the chosen split line through `pcaCenter`, with centers forced onto that line (not just orientation-consistent)
6. Replace parent with daughters in cell list
7. **Burn-in**: `prob.split_burn_in_iterations` iterations (default 500 in code, 1000 in current config.yaml — see review risk #2) alternating perturbation on each daughter. Uses overlap penalty (not hard rejection) — same `computeOverlapForCell()` as main loop.
8. **Post-burn-in fake-split guards** (added 2026-04-06):
   - **Overlap-volume guard**: approximate daughter overlap as sphere-equivalent volumes, reject if fraction > `prob.split_fake_overlap_volume_fraction_threshold` (~0.15)
   - **Radius-ratio guard**: reject if `max(d1R,d2R)/min(d1R,d2R)` > `prob.split_fake_radius_ratio_threshold` (~1.6)
   - **Bridge-brightness guard**: build cylinders between daughter centers (sized once per daughter volume using a hardcoded 0.5 factor — see review risk #4), measure real-image brightness inside, reject if bridge brightness is ≥ `prob.split_fake_bridge_brightness_similarity_threshold` (~0.9) × daughters' mean brightness. Catches continuous-blob false positives.
9. **Post-burn-in large-recenter guard** (Yiding's line): if daughters drifted > `prob.split_post_burn_in_large_recenter_min_drift_over_major` (0.85) × majorRadius from their initial PCA placement AND cost improvement < `prob.split_post_burn_in_large_recenter_max_cost_diff` (-40.0), reject — the cell is probably being dragged to absorb a neighbor, not actually splitting.
10. Compare total cost (image L2 + full overlap penalty) before vs after split
11. Accept if `costDiff < -split_cost` (default 20 or 30)

**No hard overlap rejection anywhere.** Daughters near other cells get penalized in cost, not blocked. Every new guard added on 2026-04-06 is a **soft, post-hoc rejection** — it looks at the result of burn-in and decides whether to keep it.

**Split diagnostics**: the `SplitDiagnostics` struct (Yiding's line) logs the values of each guard at every decision point, so you can retrospectively tune thresholds without rerunning with gdb.

## Overlap Penalty (Frame.cpp)

Replaces all hard overlap checks. Continuous penalty added to cost:

```
For each pair (i, j) where dist < combinedR:
    overlapRatio = (combinedR - dist) / combinedR
    penalty += weight * overlapRatio^2
```

- `weight`: configurable `overlap_penalty_weight` (default 1000)
- `computeOverlapPenalty(weight)`: O(n²) for all pairs
- `computeOverlapForCell(cellIdx, weight)`: O(n) for one cell vs all others

## Rendering & Brightness (updated 2026-04-05 / 2026-04-06)

- `draw()` uses **per-cell `_brightness`** — NOT `simulationConfig.cell_color`. This was the central change on 2026-04-05.
- `background_color` = 0.0 — matches post-sigmoid background (~0.0). Unchanged.
- `simulation.cell_color` is now the **initial seed** for each Spheroid's `_brightness` on frame 1, not a universal render color. After frame 1, brightness is driven by the EMA update rule described above.
- **Per-frame EMA update**: `_brightness = _brightness * (1 - blend) + observed * blend * amplification` where `blend = cell.brightnessUpdateBlend` and `amplification = cell.brightnessMeanAmplification` (default 1.0). `observed` is the mean real-image brightness inside the cell volume, measured via `Spheroid::measureMeanBrightness()`.
- Random brightness perturbation in `PerturbParams` is still disabled (`prob=0`) to avoid the brightness-size co-optimization trap from the jh-split-fix branch. The EMA update is driven only by the real image.
- **Outline rendering**: `drawOutline()` draws on all three RGB channels at intensity ~0.25 (was 0.4 earlier in the session).
- See `docs/conversation_archive_2026-04-05.md` and `docs/conversation_archive_2026-04-06.md` for the full evolution. See `docs/plans/backburner.md` for the older brightness experiments that were tried and reverted.

## Cost Function (Frame.cpp: calculateCost + overlap)

```
totalCost = imageCost + overlapPenalty
imageCost = sum over all 225 z-slices of ||real[z] - synth[z]||_L2
overlapPenalty = sum over all overlapping pairs of weight * overlapRatio^2
```

Lower = better fit. Image cost cached in `_currentCost` to avoid redundant 225-slice L2 computations.
