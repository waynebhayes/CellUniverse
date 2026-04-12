# PCA Input Fix, Structural Split Detection, and Frame-Anchor Drift Penalty

**Date:** 2026-04-09
**Branch:** `yp_yd_merge_04072026`
**Status:** Proposed — awaiting user approval
**Authors:** Planning agent (with diagnostic and measurement context from user session)

---

## 1. Motivation

### 1.1 The triggering incident

In run `output_jihang_20260408_164644`, frame 19 contained a false split: cell
`1f89abf484c94c498a23cad71ebee0cb1` was split into two daughters, and daughter
`...11` ended up small (r = 21.2), placed at (197, 272, 120) — **58 units away
from where it had been at the end of frame 18** and ~2.0 × majorRadius from the
parent's start-of-frame anchor. By frame 20 it had drifted further to
(211, 300, 78), landing in noise / inter-cell background. The lineage was
corrupted from that point forward.

### 1.2 Why the existing 7-layer guard stack did not catch it

Tracing the debug log against `Frame::trySplitCell`
(`C++/src/Frame.cpp:529-770`) and the two guard helpers
`shouldRejectSplitPreBurnIn` (`Frame.cpp:172-210`) and
`shouldRejectSplitPostBurnIn` (`Frame.cpp:212-234`):

- **Pre-burn-in z-axis collapse guard** barely missed the cutoff:
  `driftOverParentMajor ≈ 0.352`, threshold
  `split_pre_burn_in_z_axis_min_drift_over_major = 0.40`. The guard only fires
  when drift *exceeds* the threshold alongside a near-z axis, so this
  proposal passed.

- **Post-burn-in large-recenter guard** checks
  `splitDiagnostics.driftOverParentMajor` — but that field is sourced from
  `recenterDrift` inside `Spheroid::getSplitCells`
  (`Spheroid.cpp:829-831`), which is the **pre-burn-in one-shot PCA-center
  recenter**, not the drift that accumulated during the burn-in or after
  acceptance. So the guard literally cannot see the actual problem.

- **Cost-improvement threshold** was `costDiff ≈ -399`, well below the
  `split_cost = 80` acceptance threshold — the fake split was cost-profitable
  because the daughter successfully fitted a neighboring bright cluster.

- **Bridge-brightness / overlap-volume / volume-ratio guards** all passed
  because the two daughters were genuinely placed at two different bright
  clusters in the real image — one real cell, one neighboring cell. The
  "continuous blob" fake-guard is built for false positives *within one
  cell*, not for neighbor capture.

### 1.3 Motion baseline

We measured frame-to-frame cell motion across frames 1-18 of the same run
(the "good" range). Legitimate per-frame motion:
- Mean: ~12 units, max: ~29 units.
- `drift / majorRadius` ratio range: **0.35 – 0.96** (almost all < 1.0).

The false split's drift was 58 units ≈ **2.0 × majorRadius**, well outside
the distribution of real motion. The signal is clean enough that a 1.2× ratio
cap would never fire on legitimate motion but reliably catch this failure
class.

### 1.4 The triaxial red herring

We initially suspected that cells might genuinely be triaxial and that the
oblate model was mis-fitting them, and that the false split was a symptom
of the mis-fit. We measured:

- **Bbox-PCA (current code's input)**: 57% of cell-frame measurements had
  r12 = λ1/λ2 ≥ 1.30 (apparent equatorial asymmetry). Median r12 ≈ 1.40.
- **Strict-interior, brightness-weighted PCA** (Python tool using `tifffile`
  to load raw TIFFs; voxels masked to be strictly inside each cell's
  analytic ellipsoid; PCA weighted by intensity rather than hard-thresholded):
  **Median r12 = 1.02, p90 = 1.06, max = 1.83.** 94% of cells classified as
  oblate-or-spherical.

The cells whose bbox-PCA showed the strongest "elongation" (r12 ∈ 2.0–3.0)
all had strict-interior r12 ≈ 1.02. **The triaxial signal was neighbor
leakage from the 2-3×maxR bounding box, not a real cell-shape property.**
See §8 for the formal rejection of the triaxial generalization.

### 1.5 The central insight

The input to the PCA is wrong, not the PCA itself and not the cell model.
`Spheroid::getSplitCells` collects bright pixels from a bounding box that
routinely overlaps neighboring cells (`insideExpandedGate` at 2.0 × the
effective radii in `Spheroid.cpp:395-404`). When a neighbor's mass is
included in the cloud, the PCA eigenvectors point partly at the neighbor.
This:

1. Inflates `elongationRatio`, which bumps up `P(split)` for the next frame
   (`CellUniverse.cpp:632-663`), making this same cell keep rolling for
   splits every frame until one stochastic attempt happens to clear all the
   guards.
2. Gives a split-axis proposal that already aims at the neighbor before
   burn-in, so burn-in "successfully" pulls a daughter onto the neighbor's
   cluster and the cost-improvement gate is happy to accept it.

The strict-interior, brightness-weighted PCA would have reported
`elongationRatio ≈ 1.02` for this cell, `P(split)` would have stayed at the
baseline 0.03, and the split attempt would have been astronomically
unlikely to even be *tried*, let alone accepted.

---

## 2. Goals and Non-Goals

### Goals

1. **Stop neighbor leakage** from corrupting the split-elongation trigger
   and the split-axis computation.
2. **Add a structural split signal** (local-maximum finder) that is
   complementary to PCA and will validate any PCA-proposed split.
3. **Prevent cells and freshly-split daughters from drifting** more than
   ~1.2× majorRadius from their start-of-frame pose during a single frame.
4. **Simplify the guard stack** by removing guards whose only purpose was
   to patch around neighbor leakage once the underlying PCA input is fixed.
5. **Produce validation artifacts** (side-by-side shadow-mode logs) so the
   simplification is evidence-backed, not blind.

### Non-Goals

- Adding triaxial (three-radius) cell modeling — explicitly rejected. See §8.
- Replacing the unified stochastic loop or the cost-function architecture.
- Adding lineage-tree output (separate planned work).
- Changing the sigmoid preprocessing, per-cell brightness EMA, or overlap
  penalty formulation.
- Introducing hard overlap rejection anywhere.

---

## 3. Overview and Dependency Graph

Three pillars, each independently rollbackable:

```
Pillar A (strict-interior PCA)
  ├── enables: deletion / loosening of 7+ split-guard config fields
  └── unblocks cleaner elongation trigger → fewer false P(split) boosts

Pillar B (local-max structural detector)
  ├── independent of A — can be added before, after, or in parallel
  └── provides a second-opinion gate: require both PCA and local-max agree

Pillar C (frame anchor + drift penalty)
  ├── independent of A and B
  └── catches anything A and B miss; also bounds perturbation drift
```

Recommended implementation order: **C, then A, then B**. Rationale:
- C is the smallest and most mechanical change, lowest regression risk,
  catches the broadest failure class, and gives us a safety net to fall
  back on while we land A.
- A is the biggest change (touches `getSplitCells` internals, affects
  both consumers of PCA) but delivers the largest correctness win.
- B is pure addition: a new method and a new gate. Can be landed last
  and tuned against the observations from A.

---

## 4. Pillar A — Strict-Interior Brightness-Weighted PCA

### 4.1 Summary

Replace the bright-pixel cloud used by `Spheroid::getSplitCells`'s PCA with
voxels sampled **strictly inside** (or slightly expanded around) the cell's
analytic ellipsoid, weighted by pixel intensity rather than hard-thresholded
by a top-fraction cutoff.

Two calling contexts, two sampling modes:

1. **Elongation trigger (`Frame::computeElongationForCell`,
   `Frame::computeElongationRatios`)** — used end-of-frame to compute
   `previousElongations`, which drives next-frame P(split). This should
   see the cleanest possible signal: strict interior (1.0 × ellipsoid).

2. **Split axis / daughter placement (`Frame::trySplitCell`)** — still
   needs to find the daughter clusters when the cell really is dividing,
   and dividing daughters DO push mass just outside the parent's strict
   boundary. This should use a modest expansion (~1.3 × ellipsoid radii),
   retaining neighbor exclusion for safety.

Neighbor exclusion becomes mostly a no-op for mode 1 (strict-interior pixels
are by definition closer to self than any neighbor), but keep it for both
modes to be safe in cluster configurations where a neighbor's boundary
grazes the self-ellipsoid.

> **Note from code reading:** the *current* `Spheroid::getSplitCells`
> already uses an `insideExpandedGate` at 2.0 × `effA/effB/effC`
> (`Spheroid.cpp:395-404`) — not 3 ×. The 3 × is only the search bbox
> that bounds voxel iteration, not the gate that decides which voxels
> contribute to the PCA. So Pillar A's "1.3 × expanded" is a tightening
> of the existing 2.0 × gate, not a relaxation. Direction unchanged.
>
> **Also from code reading:** the current code's filtering is **percentile-based**
> via `splitBrightestFraction` (default 0.055 = top 5.5%) at
> `Spheroid.cpp:448-466`, not `mean+stddev`. Mean and stddev are computed
> and logged but not used as the threshold. The fix is the same — use all
> interior pixels weighted by intensity rather than hard-threshold the top ~5%.

### 4.2 Code changes

#### 4.2.1 Add a sampling mode enum and parameters to `Spheroid::getSplitCells`

**File:** `C++/includes/Spheroid.hpp`

Above the `Spheroid` class declaration, add:

```cpp
enum class SplitSampleMode {
    StrictInteriorWeighted,   // 1.0 × ellipsoid, brightness-weighted PCA
    ExpandedInteriorWeighted  // ~1.3 × ellipsoid, brightness-weighted PCA
};
```

Extend `getSplitCells` signature (keeping defaults so existing callers keep
working mid-refactor):

```cpp
std::tuple<Spheroid, Spheroid, bool, float, SplitDiagnostics> getSplitCells(
    const std::vector<cv::Mat> &image, float z_scaling,
    float backgroundColor,
    const std::vector<cv::Point3f> &neighborCenters = {},
    float preOptMajorR = 0.0f, float preOptMinorR = 0.0f,
    float preOptX = 0.0f, float preOptY = 0.0f, float preOptZ = 0.0f,
    float splitSearchRadiusMultiplier = 3.0f,
    float splitMinorAxisAlignmentToleranceDegrees = 180.0f,
    float splitMinorAxisAlignmentFlatnessRatioThreshold = 0.5f,
    float splitMinorAxisAlignmentMinRadiusDisableThreshold = 0.0f,
    SplitSampleMode sampleMode = SplitSampleMode::ExpandedInteriorWeighted,
    float sampleExpansionFactor = 1.3f) const;
```

#### 4.2.2 Replace the pixel collection + PCA core in `Spheroid::getSplitCells`

**File:** `C++/src/Spheroid.cpp`, function `getSplitCells`
(lines ~332-865).

The current flow (see `Spheroid.cpp:395-664`) is:

1. `insideExpandedGate` at 2.0 × `effA/effB/effC`.
2. Compute `insideBrightnessValues` inside the cell ellipsoid, use
   `splitBrightestFraction` percentile as `pixelThreshold`.
3. Collect `rawPoints` where `pixel >= pixelThreshold` AND
   `insideExpandedGate` AND not closer to a neighbor.
4. Recenter pass if the centroid drifts > 5 px.
5. PCA on `rawPoints` centered at `pcaCenter`, isotropic normalization by
   `majorRadius` (`normR = effA`).

New flow (both modes):

1. Compute an ellipsoid gate at
   `sampleExpansionFactor × (effA, effB, effC)`:
   - For `StrictInteriorWeighted`, `sampleExpansionFactor = 1.0f`.
   - For `ExpandedInteriorWeighted`, `sampleExpansionFactor = 1.3f` (knob).
2. Walk the tight bounding box that encloses that expanded ellipsoid
   (not the 3 × maxR search bbox). Reuse `scanSpheroidVolume` but with
   expanded `invA2/invB2/invC2` (reciprocals of
   `(sampleExpansionFactor * effA)^2`, etc.) and a fresh gate test inside
   the visitor: only include voxels where the gate test returns `val <= 1.0`.
3. For every included voxel, compute a weight:
   `w = max(0, pixel - backgroundColor)` (clamped so background contributes
   nothing; avoids dragging the centroid toward the dim edge).
4. Skip any voxel that is closer to a neighbor center than to `pcaCenter`
   (existing logic, preserved).
5. Accumulate:
   - Total weight `W = Σ w`.
   - Weighted centroid `c = Σ (w * p) / W`.
   - Weighted covariance `C = Σ w * (p - c)(p - c)^T / W`.
6. Run a 3×3 symmetric eigendecomposition on `C` (OpenCV has
   `cv::eigen(cv::Mat)` for symmetric matrices; we already depend on
   `cv::PCA` so this adds no new dep).
7. Derive `elongationRatio = sqrt(λ1) / sqrt(λ2)` **in raw image units**,
   NOT normalized by `majorRadius`. The isotropic normalization in the
   current code (`Spheroid.cpp:622-627`) was there because the hard-threshold
   cloud was unbalanced; a brightness-weighted covariance of the strict
   interior is already balanced by construction. This also means the
   elongation number is directly comparable to what the Python strict-interior
   measurement reported (median ≈ 1.02 for oblate cells).
   - Preserve a log line that ALSO reports the old isotropic-normalized
     ratio for backward compatibility with existing debug dumps, so nothing
     is silently regressed. Mark it clearly, e.g., `elongation_ratio_legacy`.
8. First eigenvector is the split axis (no renormalization needed since
   we're in raw image units).
9. The recenter pass is **deleted** for the strict-interior mode — strict
   interior has no recenter pathology. For expanded-interior, keep the
   recenter pass but bump the drift trigger from 5 px to
   `0.25 * effMajorR` (so the gate is cell-size-relative).
10. Skip the "one-sided fallback" (`Spheroid.cpp:796-802`): with
    brightness-weighted collection there are always samples on both sides
    unless the cell is genuinely at the edge of the image, in which case
    we'd rather propagate `valid=false` than propose a fabricated daughter.

The daughter-placement code (Steps 4 in the current comments,
`Spheroid.cpp:726-803`) remains — it projects voxels onto `split_axis`,
partitions into two groups, takes the centroid of each. With the
expanded-interior cloud this still works.

Update `SplitDiagnostics`:
- New field `float elongationRatioLegacy = 1.0f;` (the old isotropic-
  normalized ratio, for shadow-mode comparison).
- New field `int sampleCount = 0;` (number of voxels included in the
  weighted covariance — replaces the existing `rawPointsFinal` log).
- New field `float sampleWeightSum = 0.0f;` (Σ w — a proxy for the
  integrated brightness that went into the PCA).
- Keep `insideCount` as-is (inside-ellipsoid bright voxels, still used
  by `split_min_inside_count` gate).

#### 4.2.3 Wire the elongation-trigger callsite to strict-interior

**File:** `C++/src/Frame.cpp`, functions `computeElongationRatios`
(lines 497-512) and `computeElongationForCell` (lines 514-527).

Both already call `getSplitCells` and only use the returned `elongation`.
Change them to pass
`SplitSampleMode::StrictInteriorWeighted, 1.0f`, so the end-of-frame
elongation write into `previousElongations` uses the clean signal.

#### 4.2.4 Wire the split-attempt callsite to expanded-interior

**File:** `C++/src/Frame.cpp`, function `trySplitCell`
(line 529 onward; the `getSplitCells` call is at lines 563-571).

Pass `SplitSampleMode::ExpandedInteriorWeighted, sampleExpansionFactor`,
where `sampleExpansionFactor` is a new config field (see §4.4).

#### 4.2.5 Optional: add a second getSplitCells call for sanity cross-check

Inside `trySplitCell`, after the expanded-interior `getSplitCells` call
succeeds, optionally call `getSplitCells` again with
`StrictInteriorWeighted` and log both `elongationRatio`s:

```
[Split PCA Both] <name>
    strict_interior_elongation=<x>
    expanded_interior_elongation=<y>
```

This is the shadow-mode logging required for guard deletion in §4.5. It
costs one extra PCA per split attempt (bounded by split probability and
number of cells; negligible in practice). Gate behind a config flag
`split_log_shadow_pca` (default `true` during validation, `false` after).

### 4.3 Updated debug log lines

Add or modify these log lines in `Spheroid::getSplitCells`:

```
[Split Sampling] <name>
    mode=<StrictInterior|ExpandedInterior>
    expansion=<1.00|1.30>
    ellipsoid_bbox=(...)
    sample_count=<N>
    sample_weight_sum=<W>
    neighbors_skipped=<K>

[PCA Split] <name>
    elongation_ratio=<x>                  // NEW: raw weighted-covariance ratio
    elongation_ratio_legacy=<y>           // OLD: isotropic-normalized ratio
    eigenvalues=(l1, l2, l3)
    split_axis=(ax, ay, az)
    sample_count=<N>
    sample_mode=<StrictInterior|ExpandedInterior>
```

### 4.4 Config fields added / removed / loosened

**Added:**

| Field | Default | Purpose |
|-------|---------|---------|
| `split_pca_sample_expansion_factor` | 1.3 | Multiplier on `(effA, effB, effC)` used as the strict-interior ellipsoid gate for the split-attempt PCA. 1.0 = strict, 1.3 = modest expansion to catch early-division daughters. |
| `split_pca_elongation_sample_mode` | `"strict"` | Sampling mode for the end-of-frame elongation trigger. `"strict"` (1.0 × ellipsoid) or `"expanded"` (use `split_pca_sample_expansion_factor`). Keep default `"strict"` — expanded is only for the split-attempt. |
| `split_log_shadow_pca` | `true` | When true, log both strict-interior and expanded-interior elongation ratios in `trySplitCell` for validation. Set to `false` once guard deletion is complete. |

**Loosened or Removed Over Time:**

See §4.5.

### 4.5 Guards that become removable (with deletion criteria)

Once Pillar A is in and shadow-mode logging shows strict-interior PCA is
clean, the following guards are candidates for loosening or deletion. None
should be deleted in the same PR as Pillar A — each requires evidence from
at least one shadow-mode run over frames 1-21.

| Config field | File | Disposition | Deletion criterion |
|---|---|---|---|
| `split_pre_burn_in_z_axis_max_abs` | ConfigTypes.hpp:83 | **Remove** | Shadow-mode log shows no `[Split HeuristicReject] reason=z_axis_internal_structure` firing on any legit split. |
| `split_pre_burn_in_z_axis_max_separation_over_major` | ConfigTypes.hpp:84 | **Remove** | Same as above — both are companions. |
| `split_pre_burn_in_z_axis_min_drift_over_major` | ConfigTypes.hpp:85 | **Remove** | Same as above. Note: this is the guard that **almost** caught the frame-19 false split (0.352 vs 0.40). With strict-interior PCA it should never need to fire. |
| `split_minor_axis_alignment_tolerance_degrees` | ConfigTypes.hpp:75 | **Keep as safety net, loosen** | Only fires for flat cells where PCA picks a near-XY axis. Strict-interior PCA makes bad axis picks unlikely, but the steer-onto-z behavior is harmless. Loosen `tolerance_degrees` from 150 back toward 20 once shadow-mode shows clean axes. |
| `split_minor_axis_alignment_flatness_ratio_threshold` | ConfigTypes.hpp:76 | **Keep** | Gates minor-axis alignment; no action needed. |
| `split_minor_axis_alignment_min_radius_disable_threshold` | ConfigTypes.hpp:77 | **Keep** | Protects small cells from steering; no action needed. |
| `split_fake_overlap_volume_fraction_threshold` | ConfigTypes.hpp:73 | **Keep as safety net** | Detects daughters placed inside each other; unchanged by PCA input fix. |
| `split_fake_volume_ratio_threshold` | ConfigTypes.hpp:74 | **Keep as safety net** | Detects lopsided daughters (one absorbs most of the cluster). Pillar A should make these rarer but this guard is cheap and orthogonal. |
| `split_fake_bridge_brightness_similarity_threshold` | ConfigTypes.hpp:79 | **Keep as safety net, loosen** | Detects continuous blobs (one real cell treated as two). With strict-interior PCA, genuine continuous blobs should not produce a split trigger at all — so this guard mostly becomes silent. Loosen from 0.99 to 0.95. |
| `split_post_burn_in_large_recenter_min_drift_over_major` | ConfigTypes.hpp:86 | **Remove** | Checks pre-burn-in recenter drift, not post-burn-in drift, as documented above. It's looking at the wrong thing. Pillar C's frame anchor is the correct replacement. |
| `split_post_burn_in_large_recenter_max_cost_diff` | ConfigTypes.hpp:87 | **Remove** | Companion to the above. |
| `split_pre_burn_in_min_separation_over_major` | ConfigTypes.hpp:82 | **Keep** | Rejects collapsed (overlapping) daughter proposals. Orthogonal to PCA input. |
| `split_min_inside_count` | ConfigTypes.hpp:92 | **Keep** | Guards against boundary cells whose bbox is clipped. Orthogonal to PCA input. |

Each removal must be accompanied by a changelog entry (per CLAUDE.md
"Changelog Requirement") citing the shadow-mode log evidence, a config.yaml
update, and removal from the `ProbabilityConfig::explodeConfig` /
`trySplitCell` parameter list / `Frame::trySplitCell` signature. Because of
the long signature (`Frame.hpp:40-58`), each removal can be done in a
single reviewable mechanical PR.

### 4.6 Validation requirements

Before any guard is deleted:

1. **Shadow-mode run over frames 1-21 of the known-good dataset**
   (`data/input/original_data`). Enable `split_log_shadow_pca = true` and
   keep existing guards active. Collect the full `[Split PCA Both]`,
   `[Split Sampling]`, `[Split HeuristicReject]`, and `[Split Rejected
   Fake]` log lines.

2. **Side-by-side PCA comparison**: grep the log for
   `[Split PCA Both]` lines and assert that for every cell where
   `elongation_ratio_legacy > 1.30` (old bbox), the strict-interior
   `elongation_ratio <= 1.10`. The measurement in §1.4 implies this should
   hold for ~94% of cells.

3. **Split-decision regression**: compare the list of accepted splits
   (`[Split Accepted]` lines) between the strict-interior run and the
   baseline `output_jihang_20260408_164644` run. Criteria:
   - Every split accepted in the baseline over frames 1-18 (the "good"
     range) is **also accepted** in the new run (or rejected for an
     explicitly-logged reason).
   - The frame-19 false split on
     `1f89abf484c94c498a23cad71ebee0cb1` is **not** triggered (P(split)
     stays at baseline because `elongationRatio ≈ 1.02`) or, if
     triggered, is rejected by cost, anchor penalty, or local-max
     disagreement.

4. **Elongation-distribution histogram**: for every frame, log
   every cell's strict-interior `elongationRatio`. Expect the distribution
   centered near 1.02 with a long right tail for genuine pre-divisional
   cells (1.3-2.0). If the distribution shifts meaningfully from the
   Python measurement baseline, investigate before deleting guards.

---

## 5. Pillar B — Local-Maximum Structural Split Detector

### 5.1 Summary

Add a second, structurally-motivated split proposer that finds splits by
locating **two distinct bright peaks** in the real image around a cell.
This catches the failure mode where a cell has truly started dividing and
the daughter clusters are just outside the parent's current ellipsoid —
PCA (even strict-interior with 1.3 × expansion) may miss them, but a peak
finder sees two clean blobs.

The two detectors are used together as an AND gate: a split is attempted
only when both the (strict- or expanded-interior) PCA elongation AND the
local-max detector agree there is a two-cluster structure. This is the
robustness dividend: each detector has its own failure modes, and an AND
gate uses whichever is cleaner for a given cell.

### 5.2 Algorithm

Input: cell (from `Frame::cells[cellIdx]`), real frame stack,
and the cell's neighbor centers.

1. **Region of interest**: axis-aligned bounding box around the cell at
   `1.8 × effMajorR` on x/y and `1.8 × effMajorR` on z (so it extends a
   bit past the parent boundary but not into the next cell's territory).
   The factor is a config knob.

2. **Smoothing**: apply a small 3D Gaussian (`cv::GaussianBlur` on each
   z-slice first, then linearly interpolate; OpenCV does not ship a 3D
   Gaussian primitive but a separable sigma in each dimension is
   sufficient). Sigma = `local_max_smoothing_sigma` (default 2.5 voxels).

3. **Peak finding**: scan every interior voxel (skipping the 1-voxel
   border of the ROI to permit 26-neighbor comparison). A voxel `v` is a
   local maximum if:
   - It is `>=` all 26 26-connected neighbors.
   - Its intensity is at least
     `meanBrightness + local_max_min_prominence_over_stddev * stddevBrightness`
     where `meanBrightness`, `stddevBrightness` come from
     `Spheroid::measureBrightnessStats` (already exists at
     `Spheroid.cpp:233-275`).

4. **Non-maximum suppression**: sort candidate peaks by intensity
   descending. Iterate and keep each peak only if no higher-intensity
   kept peak is within
   `local_max_suppression_radius_factor × daughterMajorR`
   (default factor = 0.9; `daughterMajorR = cbrt(0.5) * effMajorR`).

5. **Neighbor rejection**: drop any candidate peak that is closer to a
   neighbor cell's center than to the parent's own center.

6. **Decision by count**:
   - **0 peaks** → no candidate; return `{valid=false}`.
   - **1 peak** → no split signal; return `{valid=false}`.
   - **≥ 2 peaks** → take the two highest-intensity peaks
     (p1, p2) and validate with a valley check:
     - Sample 7 points along the line segment p1→p2 (excluding the
       endpoints).
     - The minimum real-image brightness along the line must be
       `<= local_max_valley_max_relative_depth * min(pixel(p1), pixel(p2))`
       (default 0.85).
     - If the valley check passes, return `{valid=true, p1, p2}`.
     - Else return `{valid=false}`.

7. **Return value**: a new struct
   ```cpp
   struct LocalMaxSplitProposal {
       bool valid = false;
       cv::Point3f daughter1;
       cv::Point3f daughter2;
       float peakBrightness1 = 0.0f;
       float peakBrightness2 = 0.0f;
       float valleyBrightness = 0.0f;
       int numCandidatePeaksBeforeNMS = 0;
       int numCandidatePeaksAfterNMS = 0;
   };
   ```

### 5.3 Where it lives

Free function inside the anonymous namespace in `C++/src/Frame.cpp`, near
the other helpers (`computeSphereIntersectionVolume`, etc.). Declaration
not exposed in `Frame.hpp`.

Signature:

```cpp
LocalMaxSplitProposal proposeSplitByLocalMaxima(
    const Spheroid &cell,
    const std::vector<cv::Mat> &realFrame,
    const std::vector<cv::Point3f> &neighborCenters,
    float roiExpansionFactor,
    float smoothingSigma,
    float minProminenceOverStddev,
    float suppressionRadiusFactor,
    float valleyMaxRelativeDepth);
```

Called from `Frame::trySplitCell` after the existing `getSplitCells` call
(so `SplitDiagnostics` is already populated). Add a new diagnostic block:

```cpp
const auto peakProposal = proposeSplitByLocalMaxima(
    oldCell, _realFrame, neighborCenters,
    localMaxRoiExpansion, localMaxSmoothingSigma,
    localMaxMinProminenceOverStddev,
    localMaxSuppressionRadiusFactor,
    localMaxValleyMaxRelativeDepth);

std::cout << "[Split LocalMax] " << oldCell.getName()
          << " valid=" << peakProposal.valid
          << " peaks_pre_nms=" << peakProposal.numCandidatePeaksBeforeNMS
          << " peaks_post_nms=" << peakProposal.numCandidatePeaksAfterNMS
          << " p1=(" << peakProposal.daughter1.x << "," ... << ")"
          << " p2=(" << peakProposal.daughter2.x << "," ... << ")"
          << " valley=" << peakProposal.valleyBrightness
          << std::endl;
```

### 5.4 Decision protocol vs PCA

**Recommended: AND gate, PCA primary, local-max as confirmation.**

```
// Inside trySplitCell, after getSplitCells + pre-burn-in guards:
if (!peakProposal.valid) {
    std::cout << "[Split LocalMaxReject] " << oldCell.getName()
              << " reason=local_max_does_not_agree" << std::endl;
    return {0.0, [](bool){}};
}
```

Rationale:
- PCA (with pillar A fixing its input) is the cheap primary driver. It
  runs once per cell per frame via `computeElongationForCell` and feeds
  `previousElongations` → next-frame P(split). This keeps the existing
  probabilistic machinery intact.
- Local-max runs only when a split is actually attempted (bounded by
  `num_cells × max_split_probability` per frame ≈ 6 × 0.2 = 1.2 calls per
  frame on average — negligible cost).
- Requiring BOTH to agree gives a high-confidence gate without slowing
  down the unified loop.

Alternative protocols considered and rejected:

- **Replace PCA entirely with local-max**: rejected because local-max
  gives no elongation scalar to feed P(split) with, and the PCA-elongation
  feedback loop is structurally important for catching slow pre-divisional
  cells over multiple frames.
- **OR gate (either detector can trigger)**: rejected because local-max
  alone will fire on any two bright points (even two touching different
  cells); PCA elongation is the cell-geometry constraint.
- **PCA as advisory only, local-max as primary**: rejected because
  local-max has no natural "how elongated is this cell" scalar and P(split)
  would need a new driver.

### 5.5 Config fields added

| Field | Default | Purpose |
|-------|---------|---------|
| `local_max_enabled` | `true` | Master switch for the local-max detector. Set `false` to fall back to PCA-only gating. |
| `local_max_roi_expansion_factor` | 1.8 | ROI bbox radius = this × effMajorR. |
| `local_max_smoothing_sigma` | 2.5 | Gaussian sigma for pre-peak smoothing (voxels). |
| `local_max_min_prominence_over_stddev` | 1.5 | Peak threshold = `mean + this × stddev`. |
| `local_max_suppression_radius_factor` | 0.9 | NMS radius = this × daughterMajorR. |
| `local_max_valley_max_relative_depth` | 0.85 | Valley < this × min(peak1, peak2) required. |

All live under `prob:` in `config.yaml` and are parsed by
`ProbabilityConfig::explodeConfig` (ConfigTypes.hpp:105). Each gets a
default in the ProbabilityConfig constructor matching the above.

### 5.6 Performance estimate

ROI voxel count: ~`(2 * 1.8 * majorR)^3`. For majorR ≈ 30, that is
~250k voxels per call. Per frame:
- `max_split_probability = 0.2`, `num_cells ≈ 6`, so ~1.2 calls/frame.
- Per call: one separable Gaussian (3 × 250k multiplies) + one 26-neighbor
  sweep + sort of O(hundreds) peaks. ~few ms per call.
- **Total per frame: <10 ms.** Negligible vs the ~minutes the optimizer
  spends on burn-in.

### 5.7 Local-max validation

- On the baseline good run (frames 1-18), every existing `[Split Accepted]`
  should have `peakProposal.valid == true`. Check by grep.
- On frame 19's false split, `peakProposal` for
  `1f89abf484c94c498a23cad71ebee0cb1` should report 1 or 0 peaks (the cell
  is undivided; the second "peak" is the neighbor's).

---

## 6. Pillar C — Frame Anchor and Drift Penalty

### 6.1 Summary

Each cell records its start-of-frame pose in private fields. During any
cost evaluation (perturbation cost or burn-in cost), add a quadratic
penalty on the distance between the cell's current position and its frame
anchor, once that distance exceeds `allowedRatio × anchorMajorR`.

This is the smallest, most mechanical, most broadly-useful pillar. It
prevents:
- Perturbation-driven drift (a cell sliding off its start-of-frame center
  just because doing so lowers local L2).
- Freshly-split daughter drift (a daughter moving to absorb a neighbor
  post-acceptance; see §1.1).

### 6.2 Spheroid changes

**File:** `C++/includes/Spheroid.hpp`

Add private members:

```cpp
private:
    cv::Point3f _frameAnchor{0.0f, 0.0f, 0.0f};
    float _frameAnchorMajorR = 0.0f;
    float _frameAnchorMinorR = 0.0f;
    bool _frameAnchorSet = false;
```

Add public methods:

```cpp
public:
    void setFrameAnchor();               // capture current pose
    void setFrameAnchor(const cv::Point3f &pos, float majorR, float minorR);
    void clearFrameAnchor();
    bool hasFrameAnchor() const { return _frameAnchorSet; }
    cv::Point3f getFrameAnchor() const { return _frameAnchor; }
    float getFrameAnchorMajorR() const { return _frameAnchorMajorR; }
    float getFrameAnchorMinorR() const { return _frameAnchorMinorR; }
```

**File:** `C++/src/Spheroid.cpp`

Implementations are trivial one-liners. The no-arg `setFrameAnchor()`
copies `_position`, `_major_radius`, `_minor_radius`.

The constructor must NOT set `_frameAnchorSet = true` — a brand-new
`Spheroid` has no anchor until something explicitly captures one. The
`getPerturbedCell()` helper creates a new Spheroid from `SpheroidParams`
and must not copy the anchor (which lives only on the long-lived cell, not
on throwaway proposals). To preserve this cleanly, add a post-construction
"copy anchor" step in the callsites that need it (below).

### 6.3 Anchor capture callsites

Three callsites set the anchor:

1. **`CellFactory::createCells`** (`C++/src/CellFactory.cpp:63-65, 98-101`)
   — every Spheroid created from the initial CSV gets
   `setFrameAnchor()` called before it is pushed into `initialCells`.

2. **`CellUniverse::copyCellsForward`** (`C++/src/CellUniverse.cpp:1101-1109`)
   — after `frames[to].cells = frames[to - 1].cells;`, loop over the new
   frame's cells and call `setFrameAnchor()` on each. This is the primary
   per-frame capture point.

3. **`Frame::trySplitCell`** (`C++/src/Frame.cpp:529-770`) — after the
   burn-in loop completes and BEFORE the fake-split guards, call
   `setFrameAnchor()` on each daughter. This freezes the daughter's
   post-burn-in placement so that any subsequent perturbation of that
   daughter is penalized if it drifts. Freshly-split daughters thus start
   the remaining iterations of the current frame with their anchor already
   set to their post-burn-in pose — they can move within ±1.2 × their own
   majorR but no more.

   Important: the anchor is set on cells that may still be rejected by
   the fake-split guards or cost threshold. On rejection, the daughter
   is discarded entirely (the cells are popped and the parent
   reinserted), so stale anchors are not a concern.

### 6.4 Drift penalty helpers

**File:** `C++/includes/Frame.hpp` and `C++/src/Frame.cpp`

Add a private static helper:

```cpp
// In Frame (private or anonymous namespace in Frame.cpp):
double computeDriftPenalty(const Spheroid &cell,
                           float weight,
                           float allowedRatio);
```

Implementation:

```cpp
double Frame::computeDriftPenalty(const Spheroid &cell, float weight, float allowedRatio)
{
    if (weight <= 0.0f || !cell.hasFrameAnchor()) return 0.0;
    const float r = cell.getFrameAnchorMajorR();
    if (r <= 1e-6f) return 0.0;
    const cv::Point3f a = cell.getFrameAnchor();
    const float dx = cell.getX() - a.x;
    const float dy = cell.getY() - a.y;
    const float dz = cell.getZ() - a.z;
    const double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    const double normDrift = dist / static_cast<double>(r);
    const double excess = std::max(0.0, normDrift - static_cast<double>(allowedRatio));
    return static_cast<double>(weight) * excess * excess;
}
```

### 6.5 Integration into `perturbCell`

**File:** `C++/src/Frame.cpp:413-448`

Current signature:
```cpp
CostCallbackPair Frame::perturbCell(size_t index, float overlapWeight, float sizeReductionWeight);
```

New signature:
```cpp
CostCallbackPair Frame::perturbCell(size_t index,
                                    float overlapWeight,
                                    float sizeReductionWeight,
                                    float anchorDriftWeight,
                                    float anchorDriftAllowedRatio);
```

Inside the function, compute `oldAnchorDrift` and `newAnchorDrift` before
and after the perturbation (using `computeDriftPenalty` on `oldCell` and
`cells[index]` respectively) and fold them into `costDiff`:

```cpp
double oldAnchorDrift = computeDriftPenalty(oldCell, anchorDriftWeight, anchorDriftAllowedRatio);
// ... perturb ...
double newAnchorDrift = computeDriftPenalty(cells[index], anchorDriftWeight, anchorDriftAllowedRatio);

double costDiff = (newImageCost + newOverlapCell + sizeReductionPenalty + newAnchorDrift)
                - (oldImageCost + oldOverlapCell + oldAnchorDrift);
```

Callsite update: `CellUniverse::optimize` at `CellUniverse.cpp:762` passes
the two new config values.

### 6.6 Integration into `trySplitCell` burn-in

**File:** `C++/src/Frame.cpp:636-665` (the burn-in inner loop).

Currently the burn-in improvement formula is:
```cpp
double improvement = (trialImageCost + newCellOverlap)
                   - (bestImageCost + oldCellOverlap);
```

Change to:
```cpp
double oldDriftPenalty = computeDriftPenalty(saved, anchorDriftWeight, anchorDriftAllowedRatio);
double newDriftPenalty = computeDriftPenalty(cells[dIdx], anchorDriftWeight, anchorDriftAllowedRatio);
double improvement = (trialImageCost + newCellOverlap + newDriftPenalty)
                   - (bestImageCost + oldCellOverlap + oldDriftPenalty);
```

Note: during the initial part of burn-in, the daughters do NOT yet have
an anchor (because we set it AFTER the burn-in completes, per §6.3). So
`computeDriftPenalty` returns 0 during burn-in for the daughters. This is
by design: the burn-in is allowed to move the daughters freely to find a
good post-split position; the anchor catches what happens afterward.

**Alternative considered**: set the daughter anchor to its initial
post-split position (before burn-in). Rejected because burn-in legitimately
moves daughters by up to 0.85 × majorR (the existing
large-recenter threshold), which would trip the drift penalty. Post-burn-in
anchor is the right abstraction: it freezes the accepted placement.

### 6.7 Config fields added

| Field | Default | Purpose |
|-------|---------|---------|
| `frame_anchor_drift_penalty_weight` | 500.0 | Weight on `(excess)^2` drift penalty. Tune so that a drift of 2.0 × majorR gives a penalty that dominates a typical per-frame L2 improvement (observed legit improvements: tens to hundreds; a 2.0 × drift should be ≥500). |
| `frame_anchor_allowed_drift_ratio` | 1.2 | No penalty for `drift <= 1.2 × majorR`. Based on the motion measurements in §1.3 (max legit ratio 0.96). |

Both live under `prob:` and are parsed by `ProbabilityConfig::explodeConfig`.

### 6.8 Validation

1. **Unit-style check**: on the baseline good run, assert that no
   `[Perturb]` log line records a drift > 1.2 × majorR (since legit motion
   stays ≤ 0.96 × majorR, the penalty should never fire on clean data).
2. **Regression check**: frame 19 false-split daughter drift of 2.0 ×
   majorR should produce a penalty of
   `500 * (2.0 - 1.2)^2 = 500 * 0.64 = 320`, which combined with the
   other cost terms should flip the split-acceptance decision.
3. **Anchor-reset correctness**: verify that `previousElongations` and the
   frame-anchor capture happen in the correct order in
   `copyCellsForward` vs `optimize`. Specifically, the anchor should be
   set BEFORE the optimize loop starts for frame N.

---

## 7. Implementation order and dependencies

Recommended order of merge (each PR independently reviewable):

1. **PR-1: Pillar C (frame anchor)** — minimal, mechanical, no behavioral
   regression risk when `frame_anchor_drift_penalty_weight = 0` (default
   kept at 0 on merge). Files touched:
   `Spheroid.hpp`, `Spheroid.cpp`, `CellFactory.cpp`, `CellUniverse.cpp`,
   `Frame.hpp`, `Frame.cpp`, `ConfigTypes.hpp`, `config.yaml`. Ship with
   weight 0, flip to 500 after one validation run.

2. **PR-2: Pillar A — step 1 (shadow mode)** — add `SplitSampleMode`
   enum, add the strict-interior sampling path alongside the existing
   one, wire `computeElongationForCell` + `computeElongationRatios` to
   log both ratios, do NOT change the split-attempt pathway yet.
   Ship with `split_log_shadow_pca = true` and collect one full run.

3. **PR-3: Pillar A — step 2 (wire the elongation trigger)** — point
   the end-of-frame elongation consumer at strict-interior mode. This is
   the smallest behavioral change that delivers the primary P(split)
   correctness win. Still leave split-attempt on the old expanded-bbox
   path.

4. **PR-4: Pillar A — step 3 (wire the split-attempt PCA)** — point
   `trySplitCell` at `ExpandedInteriorWeighted` mode with
   `split_pca_sample_expansion_factor = 1.3`. Review carefully against
   shadow-mode logs from PR-2 / PR-3.

5. **PR-5: Pillar B (local-max detector)** — add the helper, wire it as
   an AND gate in `trySplitCell`. Can be landed in parallel with PR-3 or
   PR-4 since it doesn't touch PCA internals.

6. **PR-6: Guard deletion** — sequential sub-PRs, one guard per PR, each
   referencing shadow-mode log evidence. See §4.5 for the disposition
   table. Can start once PR-4 is landed and one clean run is done.

---

## 8. Triaxial Generalization — Considered and Rejected

### 8.1 What was considered

Generalize `Spheroid` from oblate (a = b, c) to triaxial (a ≠ b ≠ c), so
that cells whose bbox-PCA showed `r12 = λ1/λ2 > 1.3` in the equatorial
plane could be fit with two independent equatorial radii instead of one.
This would in principle reduce elongation-signal false positives by
absorbing the equatorial asymmetry into the cell model itself.

Downstream changes would have required:
- A new perturbation dimension (`minorRadius2` or similar).
- Update to `Spheroid::draw` (already general for 3 distinct radii; see
  `Spheroid.cpp:206-226` — `invA2, invB2, invC2` are independent).
- Update to `measureMeanBrightness`, `getSplitCells`, all cost functions.
- New SpheroidParams field, CSV parsing update.
- Re-tuning of every split guard that uses `effA, effB, effC`.

### 8.2 Why it was tested

Bbox-PCA measurement over frames 1-21 showed **57% of cell measurements
had r12 ≥ 1.30**. If this were a real cell-shape signal, the oblate
approximation would be genuinely wrong and any split guard driven by PCA
would have a ~57% false-positive elongation floor.

### 8.3 Why it was rejected

A strict-interior, brightness-weighted PCA (Python script using
`tifffile` to load raw TIFFs, voxels masked strictly inside each cell's
analytic ellipsoid, cov matrix weighted by pixel intensity) was run over
the same frames, and the distribution collapsed:

| Statistic | Bbox PCA (current code) | Strict-interior, weighted |
|---|---|---|
| Median r12 | ~1.40 | **1.02** |
| p90 r12 | ~2.1 | **1.06** |
| Max r12 | ~3.0 | **1.83** |
| % r12 ≥ 1.30 | **57%** | **~6%** |

Every cell flagged as "strongly elongated" by the bbox-PCA
(`r12 ∈ [2.0, 3.0]`) had a strict-interior r12 of ~1.02. The apparent
elongation was **pixel contamination from neighboring cells within the
2-3 × maxR bounding box**, not any geometric cell-shape signal.

Since 94% of cells are oblate-or-spherical under the correct measurement,
the cost/risk of triaxial generalization is not justified: the simpler
oblate model is correct as long as the PCA input is fixed.

### 8.4 When to reconsider

If a future dataset shows:
- Strict-interior brightness-weighted r12 distribution median > 1.15
  across multiple frames for stable, non-dividing cells; OR
- Visual inspection of cell shapes that are qualitatively triaxial (e.g.
  long-axis-in-XY rods, not disks); OR
- A tracker regression that can't be explained by PCA-input or anchor
  drift issues.

Then revisit triaxial. Until then: **do not pursue**, and do not re-run
this analysis. The bbox vs. strict-interior measurement is the decisive
result.

---

## 9. Validation Plan (Summary)

| Pillar | What to measure | Success criterion |
|---|---|---|
| A | `[Split PCA Both]` shadow log over frames 1-21 | For every cell where `elongation_ratio_legacy > 1.3`, `elongation_ratio (strict) ≤ 1.10`. |
| A | List of accepted splits over frames 1-18 | All baseline-accepted splits still accepted. |
| A | Frame-19 false split | Not triggered (`P(split)` stays at baseline) or rejected by cost/anchor. |
| A | End-of-frame elongation histogram | Median ≈ 1.02, p90 ≈ 1.06, consistent with the Python measurement in §1.4. |
| B | `[Split LocalMax]` log on baseline-accepted splits | `peakProposal.valid == true` for all. |
| B | `[Split LocalMax]` log on frame-19 false split | `valid == false` (0 or 1 peak). |
| B | Per-frame wall-clock with local-max | Adds < 50 ms/frame. |
| C | `[Perturb]` log on clean baseline | Drift penalty never positive (drift stays ≤ 1.2 × majorR). |
| C | Frame-19 false split with anchor penalty | Penalty contributes ~300-500 to costDiff, flipping the decision. |

Every validation run must be archived under `output/` with the timestamp
directory pattern matching existing runs (e.g., `output_jihang_YYYYMMDD_HHMMSS`),
with the console log captured to `runtime_log/`.

---

## 10. Risks and Rollback

### 10.1 Risks

- **R1: Strict-interior PCA under-reports elongation for genuinely
  pre-divisional cells.** A cell about to split has mass slightly
  outside its current ellipsoid; strict interior (1.0 ×) may miss
  that. Mitigation: the split-attempt pathway uses
  `ExpandedInteriorWeighted` at 1.3 ×, so the actual daughter
  placement sees the expanded cloud. The elongation trigger uses strict
  interior, but P(split) has a floor of `baseSplitProb = 0.03` for
  every cell every frame, so a cell that strict-interior reports as
  spherical still gets 3%-per-iteration chances to be proposed for
  splitting. If this proves insufficient, raise `baseSplitProb` or
  relax the elongation trigger mode back to expanded.

- **R2: Local-max AND-gate blocks legit splits.** If local-max is
  over-tuned, some real splits might report `valid=false`. Mitigation:
  `local_max_enabled = false` is a one-line rollback flag. Shadow-mode
  log lines in the baseline run will catch this before flipping.

- **R3: Frame anchor penalty blocks legit cell motion during burn-in.**
  Daughter post-burn-in placement can legitimately be > 1.2 × majorR
  from the parent anchor. Mitigation: the daughter anchor is set AFTER
  burn-in, so this cannot fire during burn-in. The parent anchor is
  cleared (parent is erased) when the split is accepted.

- **R4: Strict-interior centroid pulls onto dimmer voxels.** Brightness
  weighting with `w = max(0, pixel - bg)` uses the post-sigmoid signal,
  which is ~1.0 inside cells and ~0.0 outside — so the weight is
  approximately binary and dim-voxel drift is minimal. Still, the
  `[Split Sampling]` log line dumps `sample_weight_sum` for sanity
  checking.

- **R5: Guard deletion removes a guard that was catching something
  we hadn't diagnosed.** Mitigation: §4.5's deletion criterion requires
  a shadow-mode run showing the guard never fires, so deletion is
  always evidence-backed. Any guard can be re-added from the git
  history if regressions appear.

### 10.2 Rollback

- **Pillar C**: set `frame_anchor_drift_penalty_weight: 0.0`. The code
  stays in place but the penalty is inert. No code revert needed.
- **Pillar A**: set `split_pca_elongation_sample_mode: "expanded"` and
  `split_pca_sample_expansion_factor: 2.0` to re-create the legacy
  behavior. Or revert PR-3 and PR-4 independently.
- **Pillar B**: set `local_max_enabled: false`. The helper runs but the
  gate is bypassed.
- **Guard deletion**: each deletion is a separate PR per §7 step 6 and
  can be git-reverted in isolation.

---

## 11. Open Questions for User Review

1. **Pillar A mode default for the split-attempt callsite.** 1.3 ×
   expansion is a proposed default. Should the validation plan try 1.2
   and 1.5 too? Or start conservative at 1.5 and tighten later once
   local-max is catching structurally-real splits?

2. **Elongation metric definition.** The current code logs
   `elongation_ratio = lambda1/lambda2` (ratio of normalized eigenvalues).
   The Python measurement and this plan propose `sqrt(lambda1)/sqrt(lambda2)`
   (ratio of principal axes in raw units). This changes the numerical
   scale of `previousElongations` and therefore the P(split) formula in
   `CellUniverse.cpp:639-641`. Options:
   - Keep `lambda1/lambda2` (current). Then the strict-interior median
     will be ~1.04, not ~1.02 as the Python script measured (since the
     Python script reports sqrt ratio).
   - Switch to sqrt ratio. Then `baseSplitProb + max(0, 1 - 1/prevElong)`
     needs re-tuning — the threshold at which P(split) becomes
     interesting shifts. E.g., with sqrt ratio 1.5, the formula gives
     `0.03 + 0.333 = 0.363`; with eigenvalue ratio 2.25 it gives
     `0.03 + 0.556 = 0.586`.

   Recommendation: keep `lambda1/lambda2` for the struct field to avoid
   touching the P(split) formula, but ALSO log the sqrt ratio in
   `[PCA Split]` for human interpretability. User decision needed.

3. **Daughter anchor timing.** The plan sets the daughter anchor
   **after** burn-in (§6.3). An alternative is setting it **before**
   burn-in (at the initial post-split position) but with a relaxed
   `allowedRatio` of, say, 2.0 during burn-in. This would penalize
   runaway drift during burn-in itself. Worth doing? User decision.

4. **Config-field tuning defaults.** `frame_anchor_drift_penalty_weight =
   500.0` and `frame_anchor_allowed_drift_ratio = 1.2` are based on the
   §1.3 measurement but have not been A/B-tested. Are these good enough
   to land as defaults, or should they go in with weight 0 on first
   merge?

5. **Local-max smoothing implementation.** OpenCV's `cv::GaussianBlur`
   is 2D. For 3D we can either:
   - Smooth each z-slice in XY then smooth along z as a 1D pass. Fast.
   - Use a precomputed 3D kernel. Slower, more accurate.

   Recommendation: separable XY + Z, since our z-resolution is already
   7× interpolated and effectively smooth. User decision if this is
   unacceptable.

6. **Guard deletion scope.** §4.5 marks 4 guards as removable (z-axis
   triple + post-burn-in large-recenter pair). Are you comfortable with
   physically deleting code and config for these, or would you prefer to
   leave them in place with their thresholds loosened to no-op values
   (simpler rollback, slightly noisier config)?

---

## 12. File Reference (for implementation)

### Files that must be modified

| File | Why |
|---|---|
| `C++/includes/Spheroid.hpp` | Add `SplitSampleMode` enum, `_frameAnchor*` members, anchor accessors, extended `getSplitCells` signature. |
| `C++/src/Spheroid.cpp` | New sampling-mode branches in `getSplitCells`, brightness-weighted covariance, updated logging, anchor method implementations. |
| `C++/includes/Frame.hpp` | Updated `trySplitCell` / `perturbCell` signatures (anchor drift params). |
| `C++/src/Frame.cpp` | `computeDriftPenalty`, `proposeSplitByLocalMaxima`, wire both into `perturbCell` and `trySplitCell`, pass new params through. |
| `C++/src/CellUniverse.cpp` | Set frame anchor in `copyCellsForward`; pass new config fields into `perturbCell` / `trySplitCell`. |
| `C++/src/CellFactory.cpp` | Set frame anchor on initial cells (frame 1). |
| `C++/includes/ConfigTypes.hpp` | New config fields on `ProbabilityConfig` (§4.4, §5.5, §6.7); updated `explodeConfig` / `printConfig`. |
| `C++/config/config.yaml` | Add the new fields with proposed defaults. |
| `C++/docs/changelogv3.md` | Changelog entries per CLAUDE.md requirement. |
| `C++/docs/details.md` | Architecture update (new pillar sections). |
| `.claude/rules/algorithms.md` | Update §"Split Detection" to reflect the two-mode PCA and local-max gate. |
| `.claude/rules/config.md` | Document new config fields. |
| `.claude/rules/gotchas.md` | Add entry for frame anchor: "anchor is set in copyCellsForward, must not be copied by getPerturbedCell". |

### Files read but not modified

- `C++/src/main.cpp` (confirmed `copyCellsForward` call site at line 232)
- `C++/src/LineageViewer.cpp`
- `C++/src/BrightnessVolumeAnalyzer.cpp`, `C++/src/EmbryoBrightTracker.cpp`
  (standalone tools, not part of main binary)

---

## 13. Deferred / Future Work

### 13.1 Preprocessing blur is 2D-only and inflates apparent cell xy size (added 2026-04-09)

**Discovered during:** Frame-8 fit analysis on run `output_jihang_20260409_044216`
(the cache + caps + shadow-log run). User observation: cells look distinctly
larger in the preprocessed real frames than in the raw TIFF data.

**Root cause:** `CellUniverse.cpp::applySafeGaussianBlur` (line ~191) applies
`cv::GaussianBlur` with `sigma = simulation.blur_sigma` (currently 10.0) to
each z-slice **independently** — there is no blur along the z axis. The
pipeline then does linear LERP z-interpolation (σ_z = 0 effectively). So:

- xy gets smeared by σ=10 → apparent cell boundary extends ~5-15 pixels past
  the true boundary depending on cell brightness vs background
- z keeps its true extent

This creates a **structural bias toward apparently oblate cells** regardless
of the real cell shape. A sphere of radius 20 in the raw data becomes a
disk of roughly (radius 25-30 in xy, radius 20 in z) in the preprocessed
image. The optimizer fits an oblate spheroid with `majR ≈ 25-30, minR ≈ 20`
— exactly the flat-cell pattern observed across all recent runs.

**Numerical evidence** from `[Preprocess]` log lines in run `20260409_044216`
frame 1:

```
stage=post_blur_pre_sigmoid  min=0.409  max=0.582  mean=0.427  stddev=0.0084
[Calibration]  sigmoidCenter=0.419  sigmoid_k=75
stage=post_sigmoid           min=0    max=1.0    mean=0.0205
```

The post-blur dynamic range is collapsed to a 0.17-wide window. The sigmoid
center (0.419) is 0.008 above the mean (0.427) — essentially at background
level. Combined with `sigmoid_k=75`, the sigmoid operates as a near-step
function centered just above background, which means **anything the blur
lifts above background by more than a tiny amount becomes a "cell"**.

**Symptoms this explains across the 2026-04-05 → 2026-04-09 debugging:**

1. Cells running to the `maxMajorRadius` cap (they're chasing the
   blur-inflated boundary).
2. "Shrinkage ratchet" on `minorRadius` when real brightness dips — the
   xy size is blur-anchored, z size isn't, so the L2 gradient drives only
   z down.
3. "Cells flat on the wrong axis" visual complaint — all cells are
   forced oblate in xy by the 2D blur bias, even cells whose real
   orientation is different.
4. Why raising `maxMajorRadius` from 40 to 50 allowed `1f2ed10d` to grow
   to (44, 44) — the blur inflation was always pushing it to grow, the
   cap was just clamping it.

**Proposed experiments** (ordered smallest → largest change, all deferred
until after Pillar A is fully landed):

1. **Reduce `blur_sigma: 10.0 → 5.0`.** Single config knob. Expected to
   halve the xy inflation. Risk: more noise in the cost function; monitor
   `avg_abs_resid` per frame and the per-frame `perturb_accepted` counter.
   If accept rate drops sharply (e.g., <40 at frames 7-8), the noise is
   hurting convergence — back off to 7.0.

2. **Raise `sigmoid_center_percentile: 0.45 → 0.55`.** Moves the sigmoid
   cutoff from "background level" to "modestly above background". Tightens
   the effective cell boundary without changing the blur. Independent knob,
   can be combined with (1). Risk: genuinely dim cells may stop being
   detected.

3. **Apply blur in 3D instead of 2D.** Root-cause fix. Modify
   `applySafeGaussianBlur` to accept a separate σ_z parameter, and after
   the 2D per-slice blur, run a 1D Gaussian along z. Or use OpenCV's
   `cv::sepFilter2D` in a z-stack loop. New config field:
   `simulation.blur_sigma_z` (default equal to `blur_sigma` for isotropic,
   or smaller like 3.0 to account for z-interpolation). Eliminates the
   structural oblate bias entirely.

**Why this is deferred and not part of Pillar A:**

- Pillar A is about fixing the split-detection PCA input. Preprocessing
  affects *what the cost function sees*, which is upstream and orthogonal.
- Changing `blur_sigma` will shift every per-cell L2 cost, every perturbation
  accept/reject, and every apparent cell size. It's a bigger behavioral
  change than PR-3 / PR-4 / PR-5 combined.
- Doing preprocessing rework at the same time as Pillar A would make it
  impossible to attribute run-over-run differences to the right cause.

**Acceptance criteria when this work is actually done:**

- Side-by-side preprocessed stack vs raw TIFF for frame 8 — cells should
  visually be the same xy size (±2 px).
- `[Preprocess] stage=post_blur_pre_sigmoid stddev` should be substantially
  higher (at least 0.02, up from 0.008) — the blur is no longer smearing
  all variance away.
- Cell trajectories in `cells.csv`: `1f2ed10d` should settle at a smaller
  steady-state majR than 44 (probably ~30-35, matching its true biological
  size). `1f89abf` should not collapse in minR across frames 5-7.
- Frame 8 visual synth should have cells whose rendered boundary closely
  matches the raw-TIFF cell boundary.

**Cross-reference:** this issue interacts with the brightness EMA shrinkage
ratchet diagnosed earlier in this session (see frame-8 discussion in the
conversation archive). The blur-induced size inflation is the UPSTREAM
cause of the downstream shrinkage pressure — fixing preprocessing should
reduce how hard cells have to work to maintain a plausible size.

### 13.1.1 Update 2026-04-09: the blur-damping hypothesis is weaker than it looked

After the `output_jihang_20260409_062322` run, the blur hypothesis was
tested directly by running a Python brightness-weighted PCA diagnostic
on the raw TIF (`C++/data/input/original_data/frame011.tif`), bypassing
the blur + sigmoid + dim-subtract pipeline entirely. Code archived at
`/tmp/raw_elongation_all.py` and `/tmp/frame11_maxproj.py` for
re-running.

**Key raw-data findings (frame 11):**

| statistic | value |
|---|---|
| raw stack dtype | uint16 |
| raw intensity min/max | 0 / 294 (out of 65535 — **uses < 0.5% of dynamic range**) |
| raw p01 / p50 / p99 | 0 / 108 / 131 |
| cell-vs-background contrast | ~1.2× |
| max asp12 across all 11 cells | **1.145** (a tiny degenerate cell) |
| 1f2ed10d raw asp12 | **1.009** (perfectly round) |
| 1f89abf daughter cb0 raw asp12 | 1.024 |
| e9077 daughter a50 raw asp12 | 1.009 |

**Conclusion:** the raw data at frame 11 *also* shows no brightness-weighted
elongation for any cell the ground truth says should split there. The
preprocessing isn't hiding shape information that's present in the raw
data — the shape information isn't there to begin with, at least not in
the form PCA can capture.

This changes the plan's priorities:

1. **Preprocessing rework is downgraded from "probable root cause" to
   "nice-to-have optimization."** Reducing `blur_sigma` or moving to
   3D blur may still help marginally (raw asp12 of 1.18 vs preprocessed
   strict asp12 of ~1.15 for 1f2ed10d's max), but the effect is small.
   The experiments in §13.1 are still valid but no longer high priority.

2. **Pillar B (local-max structural detector, PR-5) is promoted to
   highest-priority remaining fix.** It's the only detector that can
   catch cells whose division manifests as two-peak structure without
   brightness-weighted asymmetry — which the raw-data measurement shows
   is exactly the regime for the f11/f19/f20 340/a51 misses.

3. **Pillar A (PR-3 + PR-4 elongation gate) is sufficient for the
   cells it was designed to catch.** f3 e9077 (raw strict would measure
   ~1.6, matches shadow log), f8 1f89abf (raw shows clear pre-division),
   and 12345 f3 are all caught. The remaining misses are structurally
   outside PCA's reach and need a different detector.

4. **The blur is actually appropriate for the data's SNR.** With a cell
   contrast of only 1.2× above background in raw uint16 values, pixel-level
   noise would dominate the cost function without heavy smoothing. The
   `blur_sigma=10` value is defensible for this dataset. Reducing it
   would re-introduce noise into the L2 cost and might destabilize the
   perturbation accept rate.

**Deferred but not removed from scope:** the 2D-only blur bias (cells
forced oblate) is still a real artifact, but it's a secondary concern
compared to Pillar B. If PR-5 lands and misses persist, revisit §13.1
experiments.
