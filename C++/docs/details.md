# CellUniverse C++ — Detailed Technical Documentation

## Overview

CellUniverse tracks 3D cells across time-lapse microscopy frames. For each frame, it fits 3D spheroid models to real image data using Monte Carlo perturbation, then detects cell divisions (splits) using PCA on bright pixels + a stack of soft fake-guards. The pipeline runs frame-by-frame: load + preprocess → update per-cell brightness → optimize cell positions → copy cells forward → save results.

---

## 0.1. 2026-04-11 Update Pointer (read this first)

The split pipeline was rewritten on 2026-04-11 around a **triaxial cell
model** (`a`, `b`, `c` all independent — no longer oblate). All 2026-04-06
fake-guards, Pillar B machinery, and the monolithic 1000-iter burn-in have
been deleted. The new pipeline classifies cells by **fitted shape
elongation** (`max(a,b,c)/min(a,b,c)`) and routes split attempts through
`Frame::trySplitCellPhased`, which evaluates K=5 short candidate burn-ins
and picks the best via a bio check (size ratio, volume fraction, buried
checks, drift-from-seed) + cost gate. Supporting changes:

- `PreviousFrameSnapshot` carries `shapeElongation`, `longAxisDir`,
  `longAxisLength`, and the end-of-frame cell state.
- `Spheroid::shapeElongation()` and `Spheroid::worldLongAxis()` are the
  classification inputs.
- `ProbabilityConfig` fields — new (`P_split_base`, `P_split_max`,
  `shape_elongation_classify_threshold`, `split_candidates_per_attempt`,
  `split_candidate_burn_in_iterations`, `split_candidate_rotation_delta_degrees`,
  `split_candidate_translation_delta_fraction`, `split_direction_agreement_degrees`,
  `bio_daughter_size_ratio_max`, `bio_combined_volume_min_fraction`,
  `bio_combined_volume_max_fraction`, `bio_max_drift_parent_fraction`,
  `bio_max_drift_daughter_fraction`, `split_burn_in_pos_sigma_scale`);
  deleted (all `split_fake_*`, `split_pre_burn_in_*`, `split_post_burn_in_*`,
  `split_minor_axis_alignment_*`, `split_burn_in_iterations`,
  `split_elongation_threshold`).
- `Frame::trySplitCellPhased(cellIndex, snapshot, otherClaimSets,
  useSnapshotDirection, probConfig)` — daughters sized from snapshot parent
  radii, position perturbation sigmas tightened during burn-in, post-burn-in
  drift-from-seed gate rejects daughters that escaped the parent footprint.
- `Frame::bioCheckDaughters(...)` now takes `double refParentVolume`
  instead of `const Spheroid &parent` — volume fraction ratios use snapshot
  parent volume, not live parent.

See `docs/changelogs/changelogv5.md` entries dated 2026-04-11 for the
per-file, per-line before/after. Everything below this section predates
the triaxial rewrite and references APIs and config fields that no longer
exist.

---

## 0. 2026-04-07 Update Pointer (read this first)

This document's lower sections predate the **2026-04-05 / 2026-04-06 brightness + split-guard rework** and the **2026-04-07 merge** (`yp_yd_merge_04072026`). Sections 1–3 below have been updated. **Sections 6 (Split Detection) and below still describe the older, simpler burn-in logic** and are missing most of the new guards.

For the authoritative current state, read in this order:

1. `docs/conversation_archive_2026-04-05.md` — percentile-based sigmoid, per-cell brightness EMA, safe blur, post-sigmoid dim subtraction, P(split) proportional rescale, adaptive synthetic background
2. `docs/conversation_archive_2026-04-06.md` — split fake-guards (overlap-volume, radius-ratio, bridge-brightness), flatness-gated minor-axis steering, size-reduction perturbation penalty, sign-split perturb probabilities
3. `docs/plans/2026-04-07-yp-yd-merge-review-and-next-steps.md` — merge review (goods/bads, dead code, risks) + phased implementation plan for cleanup / brightness-unification / lineage / tests
4. `.claude/rules/algorithms.md` — updated 2026-04-07 with the full preprocessing pipeline and layered split guards
5. `.claude/rules/gotchas.md` — updated 2026-04-07 — brightness is now LIVE, not disabled; sigmoid center is auto-calibrated (not background_color); split fake-guards enumerated
6. `.claude/rules/config.md` — updated 2026-04-07 with the full config field inventory including 04-05/06 additions

### Known stale sections in THIS file

- **Section 6 (Split Detection)** describes only the original burn-in with overlap penalty. It is **missing** all post-burn-in fake-guards (overlap-volume, radius-ratio, bridge-brightness, large-recenter), pre-burn-in PCA gates (separation, z-axis collapse), and the flatness-gated minor-axis steering. See `algorithms.md` for the current layered guard stack (~11 steps).
- **Section 7 (PCA Split Detection)** is mostly still correct but missing the flatness-gated minor-axis steering added on 2026-04-06 (forces split axis onto local z when `minorR/majorR ≤ flatness_threshold` AND PCA axis disagrees with z by more than `tolerance_degrees`).
- Any reference to **"brightness perturbation disabled"** or **"draw() uses cell_color"** in lower sections is OUTDATED — see 3.2 above for the corrected per-cell brightness EMA path.
- Any reference to `sigmoid_center_offset` as an active parameter is OUTDATED — it is parsed but unread. Use `sigmoid_center_percentile` instead.
- The `P(split) = min(max_split_probability, ...)` formula mentioned in the algorithms/config sections is OUTDATED — the current formula computes raw P(split) per cell and then proportionally rescales all cells so max = `max_split_probability`.

### Config field inventory added 2026-04-05/06 (see rules/config.md for full table)

**Cell-level:** `brightnessUpdateBlend`, `brightnessMeanAmplification`, `splitBrightestFraction`, `increase_prob`/`decrease_prob` per-parameter split.

**Simulation-level:** `sigmoid_center_percentile` (replaces `sigmoid_center_offset`), `post_sigmoid_dimmest_percentile`, `post_sigmoid_dimmest_transition_width`, `post_sigmoid_dimmest_transition_gradient`, `adaptive_background_expand_factor`, `adaptive_background_top_fraction`.

**Prob-level (split guards):** `split_fake_overlap_volume_fraction_threshold`, `split_fake_radius_ratio_threshold`, `split_fake_bridge_brightness_similarity_threshold`, `split_minor_axis_alignment_tolerance_degrees` (DEGREES, not radians), `split_minor_axis_alignment_flatness_ratio_threshold`, `split_pre_burn_in_min_separation_over_major`, `split_pre_burn_in_z_axis_*`, `split_post_burn_in_large_recenter_*`, `size_reduction_penalty_weight`.

---

## 1. Startup & Initialization

### 1.1 Entry Point — `main()` in `C++/src/main.cpp`

**Arguments** (parsed at lines 151–176):
| Arg | Meaning | Example |
|-----|---------|---------|
| `argv[1]` | First frame number | `1` |
| `argv[2]` | Last frame number | `10` |
| `argv[3]` | Input path (directory, pattern, or file) | `input/original_data` |
| `argv[4]` | Output directory | `output_run01` |
| `argv[5]` | YAML config file | `config.yaml` |
| `argv[6]` | Initial cell CSV | `initial.csv` |

### 1.2 Config Loading — `BaseConfig::explodeConfig()` in `C++/includes/ConfigTypes.hpp`

The YAML config (e.g., `config.yaml`) is parsed into three sub-configs:

- **SimulationConfig** (lines 10–43): `iterations_per_cell`, `background_color`, `cell_color`, `z_scaling`, `blur_sigma`, `z_slices`
- **ProbabilityConfig** (lines 45–73): `perturbation`, `split`, `split_cost`, `split_elongation_threshold`
- **SpheroidConfig** (lines 130–172): Per-parameter perturbation settings (`x`, `y`, `z`, `majorRadius`, `minorRadius`, `thetaX`, `thetaY`, `thetaZ`) each with `prob`, `mu`, `sigma` plus global constraints (`minMajorRadius`, `maxMajorRadius`, `minMinorRadius`, `maxMinorRadius`)

Each perturbation parameter has a `PerturbParams` struct (lines 81–104) with:
- `prob`: Probability of applying a random offset (vs. using mean)
- `mu`: Mean offset
- `sigma`: Standard deviation
- `getPerturbOffset()`: Returns `N(mu, sigma)` with probability `prob`, else returns `mu`

### 1.3 Image Loading — `getImageFilePaths()` in `main.cpp` (lines 46–138)

Supports three input modes:
1. **Printf pattern** (e.g., `frame%03d.tif`): Generates paths for the frame range
2. **Directory**: Auto-detects `.tif`/`.tiff` files, sorts alphabetically, selects frame range
3. **Single file**: Returns one path

Calls `updateTiffConfigIfNeeded()` (lines 27–43) to auto-detect z-slice count from the first TIFF file using `cv::imreadmulti()`.

### 1.4 Cell Creation — `CellFactory::createCells()` in `C++/src/CellFactory.cpp` (lines 14–117)

Takes `firstFrameFile` as a parameter (passed from `main.cpp`). Reads the initial CSV in one of two formats:

**Case A — 7-column format** (lines 45–70):
Columns: `file, name, x, y, z, majorRadius, minorRadius` (no brightness column — brightness defaults to 0.5).

**Case B — 4-column Napari format** (lines 78–104):
Columns: `cell_type, z, y, x`. Uses `firstFrameFile` as the frame key and assigns default radii (10.0).

For each row:
1. Parse fields, trim whitespace
2. Scale z-coordinate: `z *= z_scaling` (line 61/88)
3. Construct `Spheroid` with clamped radii
4. Group by filename: `map<filename, vector<Spheroid>>`

### 1.5 CellUniverse Construction — `CellUniverse()` in `C++/src/CellUniverse.cpp` (lines 180–208)

For each image path:
1. Call `loadFrame()` to load and preprocess the TIFF
2. Create a `Frame` object with the initial cells (first frame) or empty cells (later frames)

---

## 2. Frame Loading & Z-Interpolation

### 2.1 `loadFrame()` in `C++/src/CellUniverse.cpp` (lines 64–177)

Each TIFF frame is a multi-page file where each page is one z-slice (e.g., 33 slices from a confocal microscope).

**Processing steps:**
1. Load all TIFF pages with `cv::imreadmulti()` (line 74)
2. For each slice:
   - Convert BGR → Grayscale (line 93)
   - Apply Gaussian blur with configurable sigma (line 57–59 via `processImage`)
   - Convert to CV_32F normalized to [0, 1] (line 54)
3. **Sigmoid contrast enhancement** (CellUniverse.cpp:422–448, updated 2026-04-05):
   - **Calibration (percentile-based, active path)**: Compute a percentile of brightness values in the calibration ROI *over the full stack* (not per-slice). Sigmoid center = `simulation.sigmoid_center_percentile` percentile of those values. This REPLACED the earlier `bgMean + sigmoid_center_offset` logic — the `sigmoid_center_offset` field is now parsed but **unread** (scheduled for deletion per `docs/plans/2026-04-07-yp-yd-merge-review-and-next-steps.md`).
   - **Safe/masked blur** is applied before the sigmoid (avoids bleeding zero-valued borders into valid content).
   - **Apply sigmoid**: `output = 1 / (1 + exp(-k * (input - center)))` with `simulation.sigmoid_k` (was 75, commonly tuned to ~60).
   - **Post-sigmoid dim-region subtraction** (added 2026-04-06): compute a stack-wide dimmest percentile (`post_sigmoid_dimmest_percentile`, ~0.99). Pixels above cutoff are kept unchanged; pixels below are reduced; a transition band (`post_sigmoid_dimmest_transition_width`, `post_sigmoid_dimmest_transition_gradient`) smoothly tapers the subtraction. Clamp negatives to zero.
   - **Result**: Cells → ~1.0, background → ~0.0. This gives the L2 cost function clear gradient signal at cell boundaries.
4. **Z-interpolation** (lines 130–147): Expand 33 slices to 225 using linear interpolation:
   - Original slices placed at indices: `0, z_scaling, 2×z_scaling, ...` (where z_scaling=7)
   - Between each pair, insert `z_scaling - 1 = 6` interpolated slices
   - Formula: `interpolated[k] = (1 - t) × slice[i] + t × slice[i+1]` where `t = k / z_scaling`
   - Result: 33 × 7 - 6 = **225 total z-slices**
5. Validate: throws error if result != expected slice count (line 150)

---

## 3. How Cells Are Rendered (Synthetic Image)

### 3.1 Full Render — `Frame::generateSynthFrame()` in `C++/src/Frame.cpp` (lines 56–89)

Creates the entire synthetic image stack from scratch:

```
For each z-slice (0 to 224):
    Create blank image filled with background_color (0.0)
    For each cell:
        cell.draw(image, simulationConfig, z)
```

### 3.2 `Spheroid::draw()` in `C++/src/Spheroid.cpp` (lines 191–211)

Renders one spheroid into one z-slice. This is a **per-pixel analytical test** — no voxel matrix is stored.

`draw()` uses **per-cell `_brightness`** (updated 2026-04-05). It does NOT use `simulationConfig.cell_color` except as the frame-1 seed for each cell's `_brightness`. After frame 1, `_brightness` is updated every frame via EMA blend from the real image mean inside the cell volume, multiplied by `cell.brightnessMeanAmplification`. Combined with `background_color=0.0`, this matches the post-sigmoid real image where background is ~0.0 and cells track their measured brightness.

**Per-frame brightness update** (see `CellUniverse::updateCellBrightnessFromReal()` or equivalent):
1. Measure mean real-image brightness inside cell volume via `Spheroid::measureMeanBrightness()`
2. `observed *= cell.brightnessMeanAmplification` (default 1.0)
3. EMA blend: `_brightness = _brightness * (1 - blend) + observed * blend` where `blend = cell.brightnessUpdateBlend`
4. Clamp to `[minBrightness, maxBrightness]`

**Steps:**
1. **Early exit**: Skip if z-slice is beyond `maxR = max(majorR, majorR, minorR)` from cell center
2. **Bounding box**: Compute 2D box as `center +/- maxR` in x and y
3. **Per-pixel test**:
   ```
   For each pixel (x, y) in bounding box:
       dx = x - center.x
       dy = y - center.y
       dz = z - center.z

       // Transform to cell's local coordinate frame
       (lx, ly, lz) = inverseRotate(dx, dy, dz)

       // Ellipsoid equation test
       val = (lx/a)^2 + (ly/b)^2 + (lz/c)^2

       if val <= 1.0:
           pixel = simulationConfig.cell_color  (1.0)
   ```

The **inverse rotation** (`inverseRotatePoint()`, lines 19–42) applies:
`R_total^T = Rx(-θx) × Ry(-θy) × Rz(-θz)` to transform world coordinates into the spheroid's local frame where it is axis-aligned.

The spheroid shape is **oblate**: `a = b = majorRadius`, `c = minorRadius`. So:
- `a = b`: Equal equatorial radii
- `c`: Polar radius (shorter for oblate/pancake shape)

### 3.3 Fast Re-render — `Frame::generateSynthFrameFast()` in `Frame.cpp` (lines 78–117)

When only one cell changes (during perturbation), re-renders only the affected region:
1. Compute bounding box enclosing both old and new cell positions
2. Copy unchanged z-slices from existing synthetic frame
3. Re-render only z-slices within the bounding box

### 3.4 Cell Outlines — `Spheroid::drawOutline()` in `Spheroid.cpp` (lines 214–255)

Used when saving output images. Same as `draw()` but only marks **surface pixels** where `0.95 <= val <= 1.05` in green.

---

## 4. Cost Function

### 4.1 `Frame::calculateCost()` in `Frame.cpp` (lines 63–76)

Measures how well the synthetic image matches the real image:

```
totalCost = 0
For each z-slice i:
    totalCost += ||real[i] - synth[i]||_L2
```

Where `||...||_L2` is the L2 norm (`cv::NORM_L2`) over all 225 z-slices. Lower cost = better fit.

An overlap penalty (`computeOverlapPenalty()` / `computeOverlapForCell()`, lines 213–258) is added to the cost during perturbation and split evaluation. This penalizes cells that physically overlap, weighted by `config.prob.overlap_penalty_weight`.

---

## 5. Unified Stochastic Optimization Loop

### 5.1 `CellUniverse::optimize()` in `CellUniverse.cpp` (lines 209–378)

Runs `iterations_per_cell * num_cells` total iterations in a **unified stochastic loop** where each iteration is either a perturbation or a split attempt.

**Key features:**
- **P(split) driven by PREVIOUS frame's PCA elongation** (lines 248–265): If a cell looked elongated last frame, it is more likely to be dividing now, so it gets a higher P(split). Frame 1 has no previous data, so all cells use the base rate. The current frame's PCA (via `trySplitCell` -> `getSplitCells`) is used only for the split AXIS, not probability.
- **Pre-opt shapes saved** (lines 238–243): Before any iterations, save each cell's position and radii. Phase 1 perturbations can collapse a cell toward one blob; pre-opt state feeds PCA center and daughter sizing.
- **Split blacklist** (line 246): Cells that fail a burn-in are blacklisted from further split attempts this frame, preventing redundant 500-iteration burn-ins.

**Each iteration** (lines 273–360):
1. **Pick random cell**: uniform random from current cell list
2. **Decide split or perturb**: roll `uniform01` against `pSplit` for that cell
3. **If split** (lines 294–331): Call `frame.trySplitCell()` with pre-opt shapes. Accept if `costDiff < -split_cost`. On failure, blacklist the cell.
4. **If perturb** (lines 332–346): Call `frame.perturbCell()` which applies random offsets, overlap penalty, fast re-render, accept if cost decreased.

**End of frame** (lines 362–377): Compute PCA elongation for each cell on THIS frame's image and store in `previousElongations` for the next frame's P(split).

### 5.2 `Spheroid::getPerturbedCell()` in `Spheroid.cpp`

Creates a copy with random offsets applied to every parameter:
- **Position**: `x ± N(mu_x, sigma_x)`, same for y, z
- **Radii**: `majorR ± N(mu_major, sigma_major)`, same for minorR
- **Rotation**: `θx ± N(mu_θx, sigma_θx)`, same for θy, θz

Each offset uses `PerturbParams::getPerturbOffset()`: with probability `prob`, sample from `N(mu, sigma)`; otherwise use `mu` (typically 0).

The new `Spheroid` constructor clamps radii to `[min, max]` bounds and enforces `minorR ≤ majorR`.

### 5.3 Overlap Penalty — `Frame::computeOverlapPenalty()` / `computeOverlapForCell()` in `Frame.cpp` (lines 213–258)

Overlap is now handled via a continuous penalty in the cost function rather than a hard rejection gate:
```
For all pairs of cells (i, j):
    dist = ||center[i] - center[j]||
    combinedR = majorR[i] + majorR[j]

    if dist < combinedR:
        overlapRatio = (combinedR - dist) / combinedR
        penalty += weight * overlapRatio^2
```

The penalty is weighted by `config.prob.overlap_penalty_weight` and added to the L2 image cost during both perturbation and split evaluation. This allows cells to be near each other but penalizes physical overlap proportionally.

---

## 6. Split Detection (within unified loop)

### 6.1 Split Triggering — `CellUniverse::optimize()` in `CellUniverse.cpp` (lines 294–331)

Splits are triggered stochastically within the unified optimization loop (Section 5). When a cell is selected for a split attempt:
1. Look up pre-opt shapes for the cell
2. Call `frame.trySplitCell()` with pre-opt position, radii, elongation threshold, and overlap weight
3. Accept if `costDiff < -split_cost`; otherwise blacklist the cell from further attempts this frame

### 6.2 `Frame::trySplitCell()` in `Frame.cpp` (lines 292–418)

The main split evaluation function. Takes a cell index and returns `{costDiff, callback}`.

#### Step A: Get Split Geometry (lines 292–317)

```cpp
// Collect neighbor cell centers (excluding the cell being split)
neighborCenters = [center of every other cell]

// Call PCA-based split detection with pre-opt shapes
(child1, child2, valid, elongationRatio) = oldCell.getSplitCells(
    realFrame, z_scaling, neighborCenters, preOptMajorR, preOptMinorR, preOptX, preOptY, preOptZ)
```

If `valid == false` (too few bright pixels or constraint failure), logs `[Split Skip]` and returns.

#### Step A.5: Elongation Ratio Filter (lines 319–324)

Before the expensive burn-in, check whether the PCA elongation ratio indicates a likely split:

```
if splitElongationThreshold > 0.0 AND elongationRatio < splitElongationThreshold:
    Log "[Split Skip] <name> elongation_ratio=X < threshold=Y"
    Return {0.0, no-op}
```

- `elongationRatio ~ 1.0`: Spherical bright pixels -> no split signal -> skip
- `elongationRatio > threshold`: Elongated -> proceed to burn-in
- Setting `split_elongation_threshold: 0` in config disables the filter

#### Step B: Overlap Handling (lines 326–352)

Remove parent, add both daughters to the cell list. No hard overlap rejection — overlap penalty in cost handles it. Daughters near other cells get penalized proportionally, not blocked.

#### Step C: Burn-in Optimization (lines 357–388)

500 iterations of alternating perturbation on daughter 1 and daughter 2, using overlap penalty in cost:

```
For iter = 0 to 499:
    dIdx = (iter even) ? daughter1 : daughter2

    saved = cells[dIdx]
    cells[dIdx] = cells[dIdx].getPerturbedCell()

    // Compute overlap penalty for this cell before and after perturbation
    oldCellOverlap = computeOverlapForCell(dIdx, overlapWeight)
    newCellOverlap = computeOverlapForCell(dIdx, overlapWeight)

    // Evaluate cost (image L2 + overlap delta)
    trialFrame = generateSynthFrameFast(saved, cells[dIdx])
    trialImageCost = calculateCost(trialFrame)
    improvement = (trialImageCost + newCellOverlap) - (bestImageCost + oldCellOverlap)

    if improvement < 0:
        Accept perturbation
        accepted++
    else:
        Revert to saved
```

Logs: `[Split Burn-in] <name> burn_in_accepted=X/500 oldCost=Y newCost=Z diff=W`

#### Step D: Return Result (lines 402–418)

```
costDiff = bestTotalCost - oldTotalCost
callback = function(accept):
    if accept: keep daughters, update synth frame
    if reject: remove daughters, restore parent
return {costDiff, callback}
```

---

## 7. PCA Split Detection (getSplitCells)

### 7.1 `Spheroid::getSplitCells()` in `Spheroid.cpp` (lines 249–499)

This is the core algorithm that determines WHERE to place daughter cells. Returns `tuple<Spheroid, Spheroid, bool, float>` (daughters, validity, elongation ratio).

#### Step 1: Define Search Area (lines 253–265)

```cpp
maxR = max(majorRadius, majorRadius, minorRadius)  // = majorRadius for oblate
splitSearchRadius = maxR × 3.0                     // Generous bounding box

// PCA center: use pre-opt position when available (Phase 1 may shift
// the cell toward one blob, making PCA look spherical from that center)
pcaCenter = (preOptMajorR > 0) ? preOptPosition : currentPosition

// 3D bounding box centered at pcaCenter
minX = max(0, floor(pcaCenter.x - splitSearchRadius))
maxX = min(cols-1, ceil(pcaCenter.x + splitSearchRadius))
// Same for Y, Z
```

#### Step 2: Compute Mean Brightness (lines 276–295)

First pass: iterate over all pixels within the cell's own boundary (`val ≤ 1.0` using the ellipsoid equation) and compute the average brightness. This threshold separates "cell tissue" from "background."

```
For each pixel (x,y,z) in bounding box:
    Transform to local frame: (lx,ly,lz) = inverseRotate(dx,dy,dz)
    val = (lx/a)² + (ly/b)² + (lz/c)²
    if val ≤ 1.0:
        brightnessSum += real_image[z][y][x]
        count++

meanBrightness = brightnessSum / count
```

#### Step 3: Collect Bright Pixels (lines 297–344)

Second pass: collect all bright pixels within the bounding box. These are the input to PCA. No ellipsoidal boundary is applied — the bounding box (3×maxR) and neighbor exclusion naturally limit the area.

```
For each pixel (x,y,z) in bounding box:
    if pixel_brightness ≤ meanBrightness: skip (background)

    // Neighbor exclusion: skip if closer to another cell than to pcaCenter
    // (uses pre-opt position so distance is from original midpoint)
    distToSelf = (x-pcaCenter.x)² + (y-pcaCenter.y)² + (z-pcaCenter.z)²
    for each neighbor center:
        distToNeighbor = (x-nc.x)² + (y-nc.y)² + (z-nc.z)²
        if distToNeighbor < distToSelf: skip (belongs to neighbor)

    rawPoints.add(x, y, z)  // Store in image coordinates
```

**Why this matters:**
- The **brightness threshold** filters out dark background pixels, keeping only cell tissue
- The **bounding box** (3×maxR) sets the computational limit using pre-opt effective radii
- The **neighbor exclusion** prevents bright pixels from adjacent cells from contaminating the PCA

#### Step 4: PCA with Isotropic Normalization (by majorRadius)

PCA (Principal Component Analysis) finds the direction of maximum spread in the point cloud. If the cell has split, the bright pixels form two clusters and PCA finds the axis connecting them.

```
// Center the data: subtract pcaCenter (pre-opt position when available)
For each point:
    data[i] = (point.x - pcaCenter.x, point.y - pcaCenter.y, point.z - pcaCenter.z)

// Isotropic normalization: divide ALL axes by majorRadius
// This preserves true shape. Per-axis stddev normalization was tested
// and rejected — it suppresses Z-direction splits because small
// minorRadius inflates Z eigenvalues.
data[i].x /= majorRadius
data[i].y /= majorRadius
data[i].z /= majorRadius

// Run PCA on normalized data
pca = cv::PCA(data)
eigenvalues: L1 >= L2 >= L3
eigenvector1 = direction of maximum variance (in normalized space)

// Transform eigenvector back to image space
ev_image = (ev_norm.x * majorRadius, ev_norm.y * majorRadius, ev_norm.z * majorRadius)
split_axis = normalize(ev_image)

elongation_ratio = L1 / L2
```

**Interpreting the results:**
- `elongation_ratio ~ 1.0`: Point cloud is spherical -> no clear split direction
- `elongation_ratio > 1.5`: Point cloud is elongated -> likely two clusters (split)
- `split_axis`: The direction connecting the two clusters

#### Step 5: Centroid-Based Daughter Placement (lines 428–487)

Split the bright pixels into two groups and compute their centroids:

```
// Volume conservation: each daughter has half the parent's volume
volumeScale = ∛(0.5) ≈ 0.794
effMajorR = max(currentMajorR, preOptMajorR)  // Use pre-Phase-1 size if larger
effMinorR = max(currentMinorR, preOptMinorR)
daughterMajorR = effMajorR × 0.794
daughterMinorR = effMinorR × 0.794

// Partition pixels by projection onto split axis
For each bright pixel point:
    projection = (point - center) · split_axis

    if projection ≥ 0:  → group1 (positive side)
    if projection < 0:  → group2 (negative side)

centroid1 = mean(group1 points)  // 3D average
centroid2 = mean(group2 points)  // 3D average
```

**Fallback**: If all pixels fall on one side, use fixed offset placement:
`centroid = center ± split_axis × (daughterMajorR × 0.5)`

#### Step 6: Create Daughter Spheroids (lines 490–498)

```
daughter0 = Spheroid(
    name = parent_name + "0",
    position = centroid1,
    majorRadius = daughterMajorR,
    minorRadius = daughterMinorR,
    rotation = parent rotation (θx, θy, θz)  // Inherited
)

daughter1 = Spheroid(
    name = parent_name + "1",
    position = centroid2,
    ...same radii and rotation...
)
```

Both daughters must pass `checkConstraints()` (radii within config bounds). If either fails, `valid = false`.

Logs: `[Split Placement] centroid-based: c1=(x,y,z) c2=(x,y,z) sep=D`

---

## 8. Post-Split & Frame Transition

### 8.1 Applying Splits — within `CellUniverse::optimize()` (lines 294–331)

Splits are accepted or rejected inline during the unified loop (Section 5). When `costDiff < -split_cost`, the callback is invoked with `true`, which keeps the daughters and updates the synthetic frame. No separate "apply all splits" phase exists.

On rejection, the callback reverts to the parent cell and the cell is added to the split blacklist.

### 8.2 Copy Forward — `CellUniverse::copyCellsForward()` in `CellUniverse.cpp` (lines 464–472)

After optimizing frame N, copy all cells to frame N+1 as the initial guess:
```
frames[N+1].cells = frames[N].cells  // Deep copy
```

The next frame's optimization loop will then optimize these positions for the new image.

### 8.3 Save Outputs — `CellUniverse::saveImages()` (lines 380–417) & `saveCells()` (lines 419–462)

**Images**: For each z-slice (0-224):
- `output/real/{frame}/0.png` ... `224.png` -- Real image with green cell outlines
- `output/synth/{frame}/0.png` ... `224.png` -- Synthetic rendering

**Cells**: Append to `output/cells.csv`:
```
file, name, x, y, z, majorRadius, minorRadius, theta_x, theta_y, theta_z
```

---

## 9. Complete Execution Flow

```
1. STARTUP
   main.cpp
   ├── Parse 6 command-line arguments
   ├── Load YAML config -> BaseConfig
   ├── Auto-detect z-slices from TIFF metadata
   ├── Load image file paths for frame range
   ├── Create cells from initial CSV (CellFactory, takes firstFrameFile)
   └── Initialize CellUniverse with cells + images

2. FOR EACH FRAME (e.g., frames 1-10):

   2a. LOAD IMAGE (during CellUniverse construction)
       loadFrame() [CellUniverse.cpp:64-177]
       ├── Read multi-page TIFF (33 z-slices)
       ├── Convert to grayscale, blur, normalize to [0,1]
       ├── Calibrate sigmoid center from background zone
       ├── Apply sigmoid: cells -> ~1.0, background -> ~0.0
       └── Z-interpolate: 33 -> 225 slices (linear, z_scaling=7)

   2b. UNIFIED STOCHASTIC OPTIMIZATION LOOP
       CellUniverse::optimize() [CellUniverse.cpp:209-378]
       │
       │   iterations = num_cells * iterations_per_cell
       │   P(split) driven by PREVIOUS frame's PCA elongation
       │   Pre-opt shapes saved before any iterations
       │   Split blacklist prevents redundant burn-ins
       │
       └── For each iteration: pick random cell, then either:

           PERTURBATION (most iterations):
           Frame::perturbCell() [Frame.cpp:178-211]
           ├── Apply random perturbation (position, radii, rotation)
           │   Spheroid::getPerturbedCell()
           ├── Compute overlap penalty for this cell
           │   Frame::computeOverlapForCell() [Frame.cpp:237-258]
           ├── Render changed region only
           │   Frame::generateSynthFrameFast() [Frame.cpp:78-117]
           │   └── Spheroid::draw() [Spheroid.cpp:191-211]
           ├── Compute cost (L2 norm + overlap penalty)
           │   Frame::calculateCost() [Frame.cpp:63-76]
           └── Accept if total cost decreased, else revert

           SPLIT ATTEMPT (stochastic, P(split) from prev frame):
           Frame::trySplitCell() [Frame.cpp:292-418]
           │
           ├── PCA SPLIT DETECTION
           │   Spheroid::getSplitCells() [Spheroid.cpp]
           │   ├── Compute brightness threshold in search region
           │   ├── Collect bright pixels in bounding box
           │   │   (3*maxR radius, brightness + neighbor filtering)
           │   ├── PCA with isotropic normalization (by majorRadius)
           │   │   -> split_axis, elongation_ratio
           │   ├── Partition pixels by split axis -> two groups
           │   ├── Compute 3D centroids -> daughter positions
           │   └── Create daughter spheroids (0.794* parent radii)
           │
           ├── OVERLAP: continuous penalty (no hard rejection)
           │
           ├── BURN-IN (500 iterations)
           │   ├── Alternate perturbing daughter 1 and daughter 2
           │   ├── Cost = L2 image cost + overlap penalty delta
           │   ├── Accept only cost-reducing moves
           │   └── Track best configuration
           │
           ├── EVALUATE: costDiff = bestTotalCost - oldTotalCost
           │   if costDiff < -split_cost -> accept inline
           │
           └── On rejection: blacklist cell from further attempts

       End of frame: compute PCA elongation for each cell,
       store in previousElongations for next frame's P(split)

   2c. SAVE & ADVANCE
       ├── Copy cells to next frame
       │   CellUniverse::copyCellsForward() [CellUniverse.cpp:464-472]
       ├── Save real+synth images as PNGs
       │   CellUniverse::saveImages() [CellUniverse.cpp:380-417]
       └── Append cell states to CSV
           CellUniverse::saveCells() [CellUniverse.cpp:419-462]
```

---

## 10. Key Constants & Thresholds

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `split_burn_in_iterations` | 1000 (config) / 500 (code default) | config.yaml / ConfigTypes.hpp | Perturbations per daughter after split placement — **default mismatch noted in 2026-04-07 review** |
| `z_scaling` | 7 | config/config.yaml | Z-interpolation factor (33 -> 225 slices) |
| `sigmoid_k` | 75 (commonly tuned to ~60) | config/config.yaml | Sigmoid steepness for contrast enhancement |
| `sigmoid_center_percentile` | tuned | config/config.yaml | **Active**: percentile of calibration ROI used as sigmoid center |
| `background_color` | 0.0 | `Frame::_backgroundValue` | **NO LONGER CONFIG** — runtime-mutable Frame member, updated by adaptive-bg path |
| `cell_color` | 1.0 | `CellFactory::CellFactory()` literal | **NO LONGER CONFIG** — frame-1 seed for per-cell `_brightness` only |
| `brightnessUpdateBlend` | tuned | config/config.yaml | EMA factor for per-frame per-cell brightness update |
| `brightnessMeanAmplification` | 1.0 | config/config.yaml | Multiplier applied to measured mean brightness before EMA blend |
| `splitBrightestFraction` | 0.055 | config/config.yaml | Top fraction of bright in-cell pixels kept for split PCA |
| Daughter volume scale | cbrt(0.5) ~ 0.794 | Spheroid.cpp | Each daughter has ~half the parent volume |
| Bridge cylinder volume factor | 0.5 (hardcoded) | Frame.cpp:668-669 | **Stray hardcoded value — scheduled for configurization** |
| Overlap penalty | continuous | Frame.cpp:213-258 | Proportional penalty replaces hard overlap gate |
| `size_reduction_penalty_weight` | 2.0 | config/config.yaml | Soft quadratic penalty on radius shrinkage during perturbation. No growth penalty. |
| Split search radius | 3*maxR (effective) | Spheroid.cpp | PCA pixel collection bounding box (uses pre-opt radii) |
| Split elongation threshold | 1.5 (often tuned to 1.1) | config/config.yaml | PCA elongation ratio below which burn-in is skipped |
| Split fake-guard thresholds | configurable | config/config.yaml | `split_fake_overlap_volume_fraction_threshold` (~0.15), `split_fake_radius_ratio_threshold` (~1.6), `split_fake_bridge_brightness_similarity_threshold` (~0.9) |
| Minor-axis steering | DEGREES | config/config.yaml | `split_minor_axis_alignment_tolerance_degrees` — note degree units, not radians |
| Post-burn-in large-recenter gate | 0.85 / -40.0 | config/config.yaml | `split_post_burn_in_large_recenter_min_drift_over_major` × majorR drift threshold + `split_post_burn_in_large_recenter_max_cost_diff` |
| Surface outline intensity | 0.25 (all RGB) | Spheroid.cpp | Outline now drawn on all channels at 0.25 (was 0.4 earlier) |
| Surface outline band | 0.95-1.05 | Spheroid.cpp:234 | Pixels drawn as cell outline |

---

## 11. File Reference

| File | Lines | Role |
|------|-------|------|
| `C++/src/main.cpp` | ~236 | Entry point, argument parsing, image loading |
| `C++/src/CellUniverse.cpp` | ~498 | Frame loading, preprocessing (sigmoid), unified optimization loop, I/O |
| `C++/src/Frame.cpp` | ~423 | Synthetic rendering, cost function, perturbation, split evaluation |
| `C++/src/Spheroid.cpp` | ~658 | Cell geometry, drawing, PCA split detection, perturbation |
| `C++/src/CellFactory.cpp` | ~117 | CSV parsing (7-col and 4-col formats), cell initialization |
| `C++/includes/CellUniverse.hpp` | | CellUniverse class declaration |
| `C++/includes/Frame.hpp` | | Frame class declaration |
| `C++/includes/Spheroid.hpp` | | Spheroid class + SpheroidParams declaration |
| `C++/includes/ConfigTypes.hpp` | | All config structs, YAML parsing |
| `C++/includes/types.hpp` | | Type aliases (Cost, CostCallbackPair, etc.) |
| `C++/config/config.yaml` | | Runtime configuration |
| `C++/config/initial.csv` | | Initial cell positions and shapes |
