# Changelog v5 — Dual Pipeline, 3-Tier Simplification, Pillar B Boost

Opened 2026-04-10. Covers the dual-pipeline preprocessing, 3-tier split simplification,
Pillar B elongation boost, daughter separation increase, and bright peak staleness check.

---

## 2026-04-10 — Dual preprocessing pipeline (Option C) — ACTIVE

### Problem/Motivation

The single preprocessing pipeline (blur → sigmoid → dim-subtract → z-interp) rounds
cell boundaries via the heavy Gaussian blur (sigma=5). Close daughters that have a slight
dip between them get the dip smoothed away, then sigmoid pushes everything to ~1.0. Both
PCA and Pillar B see one round blob instead of two. This is the root cause of:

- 12345-341 never splitting (Pillar B finds 0 peaks at f19, PCA says elongation=1.11)
- PCA elongation being systematically suppressed for pre-divisional cells
- The halo making cells look spherical to all analysis tools

### Solution

Two separate image stacks per frame:

| Pipeline                 | Order                                                       | Purpose                                                 |
| ------------------------ | ----------------------------------------------------------- | ------------------------------------------------------- |
| **Cost frame**     | raw → blur(sigma=5) → sigmoid → dim-subtract → z-interp | L2 cost for perturbation (smooth landscape)             |
| **Analysis frame** | raw → sigmoid → blur(sigma=1) → dim-subtract → z-interp | PCA, Pillar B, elongation measurement (preserves shape) |

The analysis pipeline applies sigmoid FIRST, creating sharp 0/1 boundaries, then a very
light blur (sigma=1) to remove pixel noise. The sigmoid step function survives light
smoothing much better than the raw brightness profile.

### Files Changed

**File:** `C++/includes/CellUniverse.hpp`

Added `LoadedFrame` struct with `costFrame` and `analysisFrame` members.
Changed `loadFrame` return type from `std::vector<cv::Mat>` to `LoadedFrame`.

**File:** `C++/includes/Frame.hpp`

Added dual-pipeline Frame constructor taking both `realFrame` and `analysisFrame`.
Added `_analysisFrame` private member.
Added `getAnalysisFrame()` accessor (falls back to `_realFrame` if analysis frame empty).

**File:** `C++/src/Frame.cpp`

Added dual-pipeline Frame constructor.
Routed all PCA calls (`computeStrictInteriorWeightedElongation`), all Pillar B calls
(`proposeSplitByLocalMaxima`, `proposeSplitByLocalMaximaNeighborMasked`,
`runPillarBNeighborMasked`), and `getSplitCells` to use `analysisFrame` instead of
`_realFrame`. L2 cost computation, `measureMeanBrightness`, and bridge brightness
measurement remain on `_realFrame`.

**File:** `C++/src/CellUniverse.cpp`

`loadFrame` now builds both pipelines in parallel from the same raw TIFF slices.
Analysis pipeline: grayscale → normalize → sigmoid(same k, same center) → light
blur(sigma=1) → post-sigmoid dim-subtract → z-interp. Constructor updated to use
`LoadedFrame` and pass both stacks to Frame.

In-frame snapshot PCA sampling and end-of-frame PCA both switched from
`frame.getRealFrame()` to `frame.getAnalysisFrame()`.

### Effect

- PCA and Pillar B now see the true cell shape without heavy blur rounding
- L2 cost landscape unchanged (perturbation stability preserved)
- Close daughters that were invisible to PCA/Pillar B under sigma=5 blur
  become detectable under the sigmoid-first + sigma=1 analysis pipeline

---

## 2026-04-10 — 3-Tier simplification: remove all post-burn-in guards except min-radius + cost — ACTIVE

### Problem/Motivation

The post-burn-in guard stack in `trySplitCellFromSnapshot` had 8 constraints:
radius floor (1.5× minR), position drift, brightness floor, overlap fraction,
volume ratio, bridge brightness similarity, plus the cost threshold. Each was
added to fix a specific edge case, but together they over-constrained the system:

- `daughter_at_radius_floor` (1.5× minR) killed splits where one daughter was
  small but viable (e.g. majorR=13.8, floor=15)
- `bridge_brightness_similarity` killed premature-but-recoverable split attempts
  (12345-340 at f19: bridge was continuous because the real split hadn't manifested)
- `overlap_fraction` killed dense-cluster splits where daughters legitimately overlap

### Solution

Simplified to 3 tiers:

**Tier 1 — Trigger**: snapshot elongation ≥ 1.20 OR Pillar B 2-peak boost to 1.25

**Tier 2 — Placement**: bright peaks → current-frame Pillar B → snapshot axis

**Tier 3 — Accept/reject**: only two checks:

1. Daughter at absolute minimum radius (majorR ≤ minMajorRadius or minorR ≤ minMinorRadius)
2. Cost threshold (costDiff < -split_cost, checked by caller)

Removed from `trySplitCellFromSnapshot`:

- `sanityMinRadiusFactor` parameter (was 1.5× — now uses raw minR)
- `sanityMaxDriftFactor` parameter (position drift check)
- `sanityMinBrightnessFactor` parameter (brightness floor check)
- `fakeSplitOverlapVolumeFractionThreshold` parameter (overlap guard)
- `fakeSplitVolumeRatioThreshold` parameter (volume ratio guard)
- `fakeSplitBridgeBrightnessSimilarityThreshold` parameter (bridge guard)

### Files Changed

**File:** `C++/includes/Frame.hpp`

Removed 6 parameters from `trySplitCellFromSnapshot` signature:
`fakeSplitOverlapVolumeFractionThreshold`, `fakeSplitVolumeRatioThreshold`,
`fakeSplitBridgeBrightnessSimilarityThreshold`, `sanityMinRadiusFactor`,
`sanityMaxDriftFactor`, `sanityMinBrightnessFactor`.

**File:** `C++/src/Frame.cpp`

Replaced ~80 lines of post-burn-in guards with ~15 lines: just the min-radius
check at absolute floor + cost diff return. Removed all bridge cylinder
computation, overlap fraction computation, volume ratio computation, position
drift check, and brightness floor check from the snapshot path.

**File:** `C++/src/CellUniverse.cpp`

Simplified `trySplitCellFromSnapshot` call site: removed the 6 guard parameters.

### Effect

- Splits that produce strong cost improvement (costDiff < -80) are accepted
  regardless of daughter overlap, volume ratio, or bridge brightness
- Only truly degenerate splits (daughter at absolute min radius) are blocked
- The L2 cost function becomes the sole arbiter of split quality

---

## 2026-04-10 — Daughter separation increase from 1.0× to 1.5× parentMajorR — ACTIVE

### Problem/Motivation

Second-gen daughters placed at 1.0× separation start nearly overlapping.
During burn-in one absorbs the bright region while the other collapses.

### Solution

Changed `halfSep` from `0.5f * parentMajorR` to `0.75f * parentMajorR`
(total sep = 1.5×) in both the snapshot-axis and Pillar B fallback paths.

### Files Changed

**File:** `C++/src/Frame.cpp` — two `halfSep` assignments changed.

---

## 2026-04-10 — Pillar B elongation boost for halo-suppressed cells — ACTIVE

### Problem/Motivation

12345-341 has P(split)=0 at its GT split frame because the sigmoid halo
makes the cell look spherical to PCA (elongation=1.11), while Pillar B
can detect two distinct bright peaks in the real image.

### Solution

At end of frame, run Pillar B for ALL cells (removed the elongation ≥ 1.10
gate). When Pillar B finds 2 valid peaks but PCA elongation is below 1.25,
boost the snapshot elongation to 1.25. This guarantees a split attempt next
frame with bright-peak placement (placement=0).

### Files Changed

**File:** `C++/src/CellUniverse.cpp` — removed elongation gate on Pillar B,
added boost block with `kPillarBElongBoostMin = 1.25f`.

---

## 2026-04-10 — Bright peak staleness check — ACTIVE

### Problem/Motivation

Snapshot bright peaks from end of previous frame can become stale if the cell
drifts significantly during the current frame's optimization. A stale peak
may point to a neighbor's bright blob, placing one daughter in the wrong region.

### Solution

Before using snapshot bright peaks (placement=0), verify both peaks are within
`1.5 × parentMajorR` of the current cell center. If either is too far, fall
back to axis-based placement. Logged as `[Split PeakTooFar]`.

### Files Changed

**File:** `C++/src/Frame.cpp` — added distance check in the bright-peaks
placement branch of `trySplitCellFromSnapshot`.

---

## 2026-04-10 — Legacy code removal (~773 lines) — ACTIVE

### Removed

1. **`Frame::trySplitCell`** method (~391 lines) — replaced entirely by
   `trySplitCellFromSnapshot`. Declaration removed from Frame.hpp,
   implementation removed from Frame.cpp.
2. **`shouldRejectSplitPreBurnIn`** function (~42 lines) — pre-burn-in
   collapsed-overlap and z-axis guards, no longer called.
3. **`shouldRejectSplitPostBurnIn`** function (~24 lines) — large-recenter
   guard, no longer called.
4. **`proposeSplitByLocalMaxima`** original function (~283 lines) — replaced
   by `proposeSplitByLocalMaximaNeighborMasked`. The sphere distance test
   variant is deleted; neighbor masking is the sole implementation.
5. **Legacy else-branch** in `CellUniverse::optimize` (~30 lines) — the
   `trySplitCell` call site with 27 parameters. Replaced by a simple
   `continue` when no snapshot exists (frame 1 has splits disabled).

### Config fields preserved

The following config fields in `ProbabilityConfig` are still parsed from YAML
but no longer read by the snapshot-driven path. They remain for backward
compatibility of the YAML parser:

- `split_elongation_threshold`
- `split_fake_overlap_volume_fraction_threshold`
- `split_fake_volume_ratio_threshold`
- `split_fake_bridge_brightness_similarity_threshold`
- `split_search_radius_multiplier`
- `split_minor_axis_alignment_*` (3 fields)
- `split_pre_burn_in_*` (4 fields)
- `split_post_burn_in_*` (2 fields)
- `split_min_inside_count`
- `local_max_enabled`

These can be removed from both `ConfigTypes.hpp` and `config.yaml` in a
future cleanup pass when the YAML schema is stabilized.

---

## 2026-04-10 — Additional dead code removal (~890 lines) — ACTIVE

### Removed from Frame.cpp (~200 lines)

Anonymous namespace helper functions that were only called by the deleted
`trySplitCell`:

- `computeEquivalentSphereRadius` (8 lines)
- `computeSphereIntersectionVolume` (17 lines)
- `computeOverlapVolumeFractionApprox` (18 lines)
- `computeSpheroidVolume` (9 lines)
- `computeDaughterVolumeRatio` (11 lines)
- `computeCylinderMeanBrightnessAlongSegment` (68 lines)
- `computeBridgeCylinderRadius` (11 lines)

Also removed `Frame::computeElongationForCell` (53 lines) — called
`getSplitCells` (expensive bbox PCA) just for a diagnostic log line, while
the actual snapshot elongation was already computed separately via
`computeStrictInteriorWeightedElongation`. End-of-frame now uses only
the strict-interior PCA result directly.

### Removed from CellUniverse.cpp (~10 lines)

`preOptShapes` map — was populated per frame but never read after the legacy
`trySplitCell` removal. The snapshot-driven path uses `snapshot.majorRadius`
instead of pre-optimization shapes.

### Removed from Spheroid.cpp (~534 lines)

`Spheroid::getSplitCells` — the entire bbox-PCA-based daughter placement
method. No remaining callers after `computeElongationForCell` was removed.

### Removed from Spheroid.hpp (~20 lines)

- `SplitDiagnostics` struct — only used by `getSplitCells`
- `getSplitCells` declaration

### Net effect

Total dead code removed in this session: ~1660 lines across Frame.cpp,
CellUniverse.cpp, Spheroid.cpp, Spheroid.hpp. The remaining active code:

- Frame.cpp: 1076 lines (was ~2000+)
- Spheroid.cpp: 668 lines (was ~1200+)
- CellUniverse.cpp: 1356 lines

---

## 2026-04-10 — Biological daughter sanity checks + raise split_cost — ACTIVE

### Problem

Run 042955 showed cascading false-positive splits at f2: e3d034 (non-splitter)
split with costDiff=-91, e9077 split too early with one daughter collapsed to
R=(10.6, 5.2). Both passed the simplified tier-3 check because costDiff < -80
and the collapsed daughter was just above the absolute min-radius floor.

### Solution

1. **Biological checks** in `trySplitCellFromSnapshot` (replace the min-radius check):

   - **Size asymmetry**: reject if `max(d1R, d2R) / min(d1R, d2R) > 2.5`
     (real daughters are roughly equal)
   - **Fraction of parent**: reject if either daughter's majorR < 40% of
     parent majorR (daughters should be ~80% of parent, not tiny ghosts)
2. **Raise split_cost** from 80 to 200. Real splits produce costDiff of -250
   to -800; the false positives at f2 had costDiff of -91 and -106.

### Files Changed

**File:** `C++/src/Frame.cpp` — replaced absolute min-radius check with
two biological checks (size ratio + parent fraction).

**File:** `C++/config/config.yaml` — `split_cost: 80` → `split_cost: 200`.

Data-driven thresholds (from 7 TP splits in run 230653):

- Size ratio: max 1.5× (observed range: 1.01–1.26)
- Parent fraction: 0.50–1.10 (observed range: 0.56–1.05)

---

## 2026-04-10 — Analysis pipeline: raw + light blur (no sigmoid) — ACTIVE

### Problem

The dual pipeline (sigmoid-first + light blur) still produced the halo
problem because the sigmoid itself is the halo generator — it amplifies dim
peripheral pixels around cells to ~1.0, making cells look rounder than they
are. The analysis frame was nearly identical to the cost frame.

### Solution

Analysis pipeline changed from `raw → sigmoid → blur(1)` to `raw → blur(1)`
only. No sigmoid, no dim-subtract. PCA and Pillar B now see the actual
brightness profile — cells appear as their true shape without artificial
rounding.

A separate `analysisBackgroundValue` is computed from the calibration ROI on
the raw analysis frame (~0.43) instead of using the post-sigmoid background
(~0.0). This is passed to `computeStrictInteriorWeightedElongation` as the
noise floor for brightness-weighted PCA.

### Files Changed

**File:** `C++/src/CellUniverse.cpp` — removed sigmoid + dim-subtract from
analysis pipeline; added `analysisBackground` computation from calibration
ROI; set on Frame via `setAnalysisBackgroundValue`.

**File:** `C++/includes/CellUniverse.hpp` — added `analysisBackground` field
to `LoadedFrame`.

**File:** `C++/includes/Frame.hpp` — added `_analysisBackgroundValue` member,
`setAnalysisBackgroundValue`/`getAnalysisBackgroundValue` accessors.

**File:** `C++/src/CellUniverse.cpp` — PCA calls now use
`frame.getAnalysisBackgroundValue()` instead of `frame.getBackgroundValue()`.

---

## 2026-04-11 — Triaxial split pipeline rewrite — ACTIVE

### Problem/Motivation

The old split pipeline classified cells by image-PCA elongation on interior
pixels. Oblate fits (`a == b`) collapse the PCA spectrum, so splitting cells
registered at ~1.15 elongation — below the 1.20 cutoff — and frame-3 of the
reference dataset never split e9077 or 12345. No amount of threshold tuning
fixed it because the classification signal itself was wrong.

The rewrite replaces that with a triaxial cell model (`a`, `b`, `c` all
independent) and classifies on **fitted shape elongation**
(`max(a,b,c)/min(a,b,c)`) from each cell's Spheroid rather than from image
pixels. Cells above the threshold are **pre_classified** and get guaranteed
split attempts in Phase B; the rest stay **non_classified** and run
perturbations in Phase A with PCA-fallback split attempts as a missed-split
recovery path. The monolithic 1000-iter burn-in is replaced with K=5 short
candidate burn-ins followed by a bio check. ~2000 lines of oblate-era guards
were deleted (Pillar B machinery, strict-interior PCA helpers, fake-split
guards, burn-in overlap hard rejections, z-axis pre-gates).

### Files Changed

**File:** `C++/includes/ConfigTypes.hpp`

`ProbabilityConfig` replaced. New fields (defaults shown):

```cpp
float P_split_base = 0.03f;
float P_split_max = 0.5f;
float shape_elongation_classify_threshold = 1.20f;
float overlap_penalty_weight = 500.0f;
float size_reduction_penalty_weight = 30.0f;
float split_cost = 80.0f;
float split_direction_agreement_degrees = 20.0f;
int   expected_daughter_pre_pass_iterations = 1;
int   split_candidates_per_attempt = 5;
int   split_candidate_burn_in_iterations = 20;
float split_candidate_rotation_delta_degrees = 8.0f;
float split_candidate_translation_delta_fraction = 0.2f;
float bio_daughter_size_ratio_max = 1.5f;
float bio_combined_volume_min_fraction = 0.6f;
float bio_combined_volume_max_fraction = 1.3f;
```

Legacy aliases `split` and `max_split_probability` still parsed for backward
compatibility. All `split_fake_*`, `split_pre_burn_in_*`,
`split_post_burn_in_*`, `split_minor_axis_alignment_*`, `split_burn_in_iterations`,
`split_elongation_threshold`, `split_search_radius_multiplier`, and
`split_min_inside_count` fields removed.

**File:** `C++/includes/Spheroid.hpp`

`PreviousFrameSnapshot` rewritten — now carries `shapeElongation`,
`longAxisDir`, `longAxisLength`, plus the end-of-frame cell state
(`position`, `majorRadius`, `bRadius`, `minorRadius`, theta, brightness).
Legacy `StrictInteriorPcaResult` and `principalAxis` fields removed.

New inline `Spheroid::shapeElongation()` returns `max(a,b,c)/min(a,b,c)`.
New `Spheroid::worldLongAxis(dir, length)` returns the world-space direction
and length of the longest fitted semi-axis.

**File:** `C++/src/Spheroid.cpp`

`worldLongAxis()` implemented. `computeStrictInteriorWeightedElongation()`
and its PCA helper deleted (~275 LOC).

**File:** `C++/includes/Frame.hpp`

Declared `trySplitCellPhased(cellIndex, snapshot, otherClaimSets, useSnapshotDirection, probConfig)` and `ClaimSet` type alias
(`std::map<std::string, std::vector<cv::Point3f>>`).

Removed: `trySplitCell`, `trySplitCellFromSnapshot`, `runPillarBNeighborMasked`,
`PillarBResult`, `LocalMaxSplitProposal`, and all Pillar B / local-maxima
fields.

**File:** `C++/src/Frame.cpp`

New `trySplitCellPhased` (~300 LOC) with the full pipeline:

1. Gather bright pixels in a snapshot-centered 3R bounding box, Voronoi-filtered against the other cells' claim sets.
2. Compute PCA direction + centroids.
3. Generate K candidates (primary direction + rotation variants + translation variants). For pre-classified cells with a trusted snapshot direction, use it; if PCA disagrees by more than `split_direction_agreement_degrees`, both become primaries.
4. Evaluate each candidate via a short burn-in (`split_candidate_burn_in_iterations` default 20) on alternating daughters, pick the one with the lowest post-burn-in total cost.
5. Bio check: `bioCheckDaughters` (size ratio, combined volume fraction, daughter-buried-in-other, daughter-buried-in-sibling).
6. Cost check: accept if `bestTotal - baselineTotal < -split_cost`.

New helpers: `gatherBrightPixelsVoronoi`, `pca3DWithCentroids`,
`bioCheckDaughters`, `buildDaughter`, `orthonormalFrame`, `rotateAroundAxis`.

Deleted: `trySplitCell` (legacy monolithic path), `runPillarBNeighborMasked`,
`proposeSplitByLocalMaxima`, all fake-split guards. Total ~700 LOC removed.

**File:** `C++/src/CellUniverse.cpp`

`optimize()` rewritten. Flow:

1. Compute `rawP_i = P_split_base + max(0, 1 - 1/snapshot.shapeElongation_i)` per cell, rescale so `max = P_split_max`.
2. Classify: `pre_classified` if `snapshot.shapeElongation > shape_elongation_classify_threshold`, else `non_classified`.
3. Phase A: iterate non_classified cells (perturb + missed-split recovery via `trySplitCellPhased` with `useSnapshotDirection=false`).
4. Phase B: iterate pre_classified cells (perturb + snapshot-driven split attempts with `useSnapshotDirection=true`).
5. End-of-frame: build new `PreviousFrameSnapshot` for every cell from its fitted Spheroid state, using `shapeElongation()` and `worldLongAxis()`.

The frame-start expected-daughter pre-pass is currently a **stub** — it
iterates the pre_classified cells but doesn't run PCA to image-ground the
seed positions. Daughter seeds fall back to raw snapshot extrapolation.

**File:** `C++/config/config.yaml`

`prob:` section rewritten to use the new field names. Legacy config fields
deleted (around 25 fields for fake-split guards, burn-in, local-maxima, etc).

### Effect

- Frame 3 of the reference dataset now pre-classifies 12345 (shapeElong=1.46)
  and e9077 pre was 1.25 at f1 but 1.15 at f2. e9077 gets split via the
  non_classified fallback path instead.
- Architecture is cleaner — one split function, three bio gates, one cost
  gate, no stacked pre/post burn-in guards.
- Pipeline is unbuilt at the time of this entry — see next section for the
  first-run diagnostics and follow-up fixes.

---

## 2026-04-11 — Snapshot-sized daughters, burn-in sigma cap, drift-from-seed gate — ACTIVE

### Problem/Motivation

First validation run of the triaxial pipeline
(`outputs/output_jihang_20260411_061106/`) showed three failure modes:

1. **12345 at f3 rejected on cost** (`diff=-13.85` vs threshold -80) even
   though it was correctly pre-classified. Its bR shrank from 33.4 (f2
   snapshot) to 21.8 during Phase B perturbations before the split attempt
   resolved; the split daughters sized from the shrunken live parent were
   too small to clear the cost threshold.
2. **1f2ed false split at f4** (`costDiff=-120.9`). Never pre-classified
   (shapeElong 1.14–1.18), accepted via non-classified fallback. Final
   daughter positions: d0=(362, 295, 139), d1=(372, 300, 88). z-separation
   of 51 voxels in a parent with minorR=37 — d1 at z=88 has its own
   footprint at z=63–113, well outside the parent's z=92–168 range. The
   daughter escaped the parent during the 20-iter burn-in (z sigma=8,
   prob=0.35 allows ~25 voxels of drift).
3. **e3d03 at f5 false split** — same pattern, smaller scale (parent at
   z=82 with minorR=18, daughters at z=69 and z=94, 25-voxel spread).

None of the four existing bio gates caught the false splits:

- size ratio 31/25 = 1.22 < 1.5 ✓
- volume fraction (25600+16588)/45582 = 0.93 ∈ [0.6, 1.3] ✓
- center-based buried check: 51-voxel separation > 31-voxel radius ✓
- neighbor-buried: ✓

Only one `[Split Reject bio]` fired in the entire run (8cbdf at f2, size
ratio). Cost was doing all the filtering and letting false splits through.

### Solution

Three complementary fixes — one structural, one mechanical, one post-hoc:

**(1) Snapshot-sized daughters.** In `trySplitCellPhased`, compute daughter
radii from the parent's `PreviousFrameSnapshot` values when valid, not from
the live (possibly Phase-B-perturbed) parent. Also use snapshot-based
reference volume for the bio volume-fraction check so the ratio isn't
skewed by shrunken live parents.

**(2) Tight burn-in position sigmas.** Before the candidate evaluation
loop, scale `Spheroid::cellConfig.{x,y,z}.sigma` by
`split_burn_in_pos_sigma_scale` (0.4 default → sigmas become 2, 2, 3.2
instead of 5, 5, 8). Restore after the loop. Global static mutation is
safe — single-threaded optimizer, all exit paths restore before return.

**(3) Drift-from-seed bio gate.** Track the initial candidate seed
positions of the best candidate. After burn-in, measure the Euclidean
distance each daughter has wandered from its seed and reject if either
exceeds `max(bio_max_drift_parent_fraction * parent_maxR, bio_max_drift_daughter_fraction * daughter_maxR)` (defaults 0.4 and 0.8).

### Files Changed

**File:** `C++/includes/ConfigTypes.hpp`

Lines 129–150 (before):

```cpp
float bio_daughter_size_ratio_max = 1.5f;
float bio_combined_volume_min_fraction = 0.6f;
float bio_combined_volume_max_fraction = 1.3f;

ProbabilityConfig() = default;
```

Lines 129–170 (after): added three new fields with comments, and parse lines
in `explodeConfig` (`bio_max_drift_parent_fraction`,
`bio_max_drift_daughter_fraction`, `split_burn_in_pos_sigma_scale`).

**File:** `C++/config/config.yaml`

Lines 182–196 (after): added the three new fields under `prob:` with
defaults `bio_max_drift_parent_fraction: 0.4`,
`bio_max_drift_daughter_fraction: 0.8`,
`split_burn_in_pos_sigma_scale: 0.4`.

**File:** `C++/src/Frame.cpp`

**Lines 656–665 (before):**

```cpp
bool bioCheckDaughters(
    const Spheroid &daughter1,
    const Spheroid &daughter2,
    const Spheroid &parent,
    ...)
```

**Lines 656–665 (after):**

```cpp
bool bioCheckDaughters(
    const Spheroid &daughter1,
    const Spheroid &daughter2,
    double refParentVolume,
    ...)
```

The `parent` parameter is gone; the volume check now uses `refParentVolume`
supplied by the caller. All other checks unchanged.

**Lines 742–766 (before):**

```cpp
Spheroid buildDaughter(
    const std::string &name,
    const cv::Point3f &center,
    const Spheroid &parent,
    float volumeScale)
{
    const auto &cfg = Spheroid::cellConfig;
    const float dMajor = std::clamp(parent.getMajorRadius() * volumeScale, ...);
    const float dB     = std::clamp(parent.getBRadius() * volumeScale, ...);
    const float dMinor = std::clamp(parent.getMinorRadius() * volumeScale, ...);
```

**Lines 742–776 (after):**

```cpp
Spheroid buildDaughter(
    const std::string &name,
    const cv::Point3f &center,
    const Spheroid &parent,
    float volumeScale,
    float srcMajor,
    float srcB,
    float srcMinor)
{
    const auto &cfg = Spheroid::cellConfig;
    const float dMajor = std::clamp(srcMajor * volumeScale, ...);
    const float dB     = std::clamp(srcB     * volumeScale, ...);
    const float dMinor = std::clamp(srcMinor * volumeScale, ...);
```

`parent` is still used for `theta_x/y/z` and `getBrightness()` passthrough —
only the radii come from the explicit `src*` arguments.

**Lines ~795–820 (new) in `trySplitCellPhased`:**

```cpp
const bool snapshotValid = snapshot.valid &&
    snapshot.majorRadius > 1e-3f &&
    snapshot.minorRadius > 1e-3f;
const float srcMajor = snapshotValid ? snapshot.majorRadius : parent.getMajorRadius();
const float srcB     = (snapshotValid && snapshot.bRadius > 1e-3f)
    ? snapshot.bRadius : parent.getBRadius();
const float srcMinor = snapshotValid ? snapshot.minorRadius : parent.getMinorRadius();

const double refParentVolume =
    static_cast<double>(srcMajor) *
    static_cast<double>(srcB) *
    static_cast<double>(srcMinor);
const float srcMaxR = std::max({srcMajor, srcB, srcMinor});
```

**Lines ~985–1000 (new) — burn-in sigma swap:**

```cpp
const float posScale = std::max(0.0f, probConfig.split_burn_in_pos_sigma_scale);
PerturbParams savedPerturbX = Spheroid::cellConfig.x;
PerturbParams savedPerturbY = Spheroid::cellConfig.y;
PerturbParams savedPerturbZ = Spheroid::cellConfig.z;
Spheroid::cellConfig.x.sigma = savedPerturbX.sigma * posScale;
Spheroid::cellConfig.y.sigma = savedPerturbY.sigma * posScale;
Spheroid::cellConfig.z.sigma = savedPerturbZ.sigma * posScale;
```

Restored at lines ~1051–1053 after the candidate loop.

**Lines ~1002–1005 (after) — daughter construction:**

```cpp
Spheroid child1 = buildDaughter(parentName + "0", cand.d1Pos, parent,
                                 volumeScale, srcMajor, srcB, srcMinor);
Spheroid child2 = buildDaughter(parentName + "1", cand.d2Pos, parent,
                                 volumeScale, srcMajor, srcB, srcMinor);
```

**Lines ~1020–1030 (new) — record best seed positions:**

```cpp
if (candTotal < bestTotal) {
    bestTotal = candTotal;
    ...
    bestSeedD1 = cand.d1Pos;
    bestSeedD2 = cand.d2Pos;
}
```

**Lines ~1068–1093 (new) — drift-from-seed bio gate before the size/volume check:**

```cpp
const float daughterMaxR = std::max(bestD1MaxR, bestD2MaxR);
const float driftLimit = std::max(
    probConfig.bio_max_drift_parent_fraction * srcMaxR,
    probConfig.bio_max_drift_daughter_fraction * daughterMaxR);
const float drift1 = static_cast<float>(cv::norm(
    cv::Point3f(bestD1.getX(), bestD1.getY(), bestD1.getZ()) - bestSeedD1));
const float drift2 = static_cast<float>(cv::norm(
    cv::Point3f(bestD2.getX(), bestD2.getY(), bestD2.getZ()) - bestSeedD2));
if (drift1 > driftLimit || drift2 > driftLimit) {
    std::cout << "[Split Reject bio] " << parentName
              << " reason=drift_from_seed d1=" << drift1
              << " d2=" << drift2
              << " limit=" << driftLimit << std::endl;
    return {0.0, noop};
}
```

The `bioCheckDaughters` call at line ~1095 now passes `refParentVolume`
instead of `parent`.

### Effect

- **12345 unblock (expected):** daughters sized from snapshot bR=33.4 will
  be ~26.5 each instead of ~17, large enough to cover the real brightness
  and clear the -80 cost threshold.
- **False split suppression (expected):** the 1f2ed f4 daughter at z=88
  would have drifted ~23 voxels from its seed at ~z=111; cap is
  `max(0.4*37.8, 0.8*31) = 24.8`, borderline pass — the tighter burn-in
  sigmas (z sigma becomes 3.2 instead of 8) should prevent it from ever
  reaching that far. If it still does, the drift gate catches it.
- **No change to real-split e9077:** drift from seed was ~15 voxels, well
  within `max(0.4*29.2, 0.8*19) = 15.2` (borderline) and inside the
  tighter `max(0.8*19) = 15.2` daughter-scaled limit.

Post-first-build validation pending.

---

## 2026-04-11 — Final refine burn-in + extensive split diagnostics — ACTIVE

### Problem/Motivation

Second run after the snapshot-sized daughters / drift-gate fix
(`outputs/output_jihang_20260411_064520/`) showed that:

1. Real splits are improving cost modestly (e9077 f3 diff=-39, e3d03 f4
   diff=-61, 12345 f4 diff=-76) but still not hitting the -80 threshold.
2. 12345 at f3 on the pre-classified path went to **+13.55** (cost got
   worse) — snapshot-sized daughters with the snapshot direction are not
   settling well enough in 20 burn-in iters to clear the baseline.
3. The 1f2ed false split at f4 slipped through again at diff=-91.89.
4. We have no visibility into the actual PCA centroids, Voronoi keep
   counts, or per-candidate trajectories — the rejection pattern is hard
   to diagnose without seeing what the split pipeline is actually doing
   internally.

### Solution

Two changes:

**(1) Final refine burn-in on the winning candidate.** After the K=5
candidate loop picks a winner, run `split_final_refine_iterations`
(default 30) extra perturb iterations on just the winning pair of
daughters before the bio/cost gates fire. Uses the same tight burn-in
sigmas. Gives chosen daughters a chance to settle before gating.

**(2) Extensive split diagnostic logging** — trace every stage of the
pipeline so the next run can be diagnosed without re-instrumenting:

- `[Split Attempt]` — parent name, snapshot validity, src vs live radii, snapshot shapeElong, long axis length, parent live position.
- `[Split Seeds]` — D1_seed / D2_seed (snapshot extrapolation), longAxisDir, boxRadius.
- `[Voronoi In]` — other-cell claim-set summary (cellCount, pointCount, per-cell breakdown).
- `[Voronoi Out]` — box voxels, voxels in sphere, voxels above brightness cutoff, voronoi-rejected count, final kept count.
- `[Split PCA]` — dirPca, D1_exp / D2_exp (PCA centroids), centroid separation.
- `[Split Dirs]` — mode (snapshot_only / snapshot+pca / pca_only), snapshot direction, PCA direction, angle in degrees, nPrimaries, nPixels.
- `[Split Baseline]` — pre-split imageCost / overlap / total / threshold / nCandidates / burnIters.
- `[Split Cand]` (per candidate) — seed1, final1, drift1, seed2, final2, drift2, total cost, image / overlap breakdown.
- `[Split Refine]` — preTotal, postTotal, delta, refineDrift1, refineDrift2, final positions, accept count over the refine pass.
- `[Split Accepted]` — costDiff, seed + final + radii + drift for each daughter.
- `[Split Reject cost]` — diff, threshold, best candidate idx, daughter positions, radii, drifts.
- `[Split Reject bio]` — reason, daughter positions, radii, refParentVolume.
- `[Pre-Pass]` (in CellUniverse.cpp) — stub status, shapeElong, longAxisLen, snapCenter, seed1, seed2 for each pre-classified cell.

Direction-agreement logic matches the spec and is unchanged:

- `angle < split_direction_agreement_degrees` → primary = dir_snap only
- `angle >= split_direction_agreement_degrees` → primaries = {dir_snap, dir_pca}
- `useSnapshotDirection == false` → primaries = {dir_pca}

### Files Changed

**File:** `C++/includes/ConfigTypes.hpp` — added
`split_final_refine_iterations = 30` in `ProbabilityConfig`, with
parse in `explodeConfig`.

**File:** `C++/config/config.yaml` — added
`split_final_refine_iterations: 30` under `prob:`.

**File:** `C++/src/Frame.cpp`

- `gatherBrightPixelsVoronoi` now takes an optional `GatherStats*` out-param that receives `boxVoxels`, `inSphere`, `aboveBrightness`, `voronoiRejected`, `voronoiKept` counts.
- `trySplitCellPhased` instrumented with all the log lines above.
- New refine-burn-in block after candidate loop: reinstalls winning state, runs `split_final_refine_iterations` perturbs on the daughters, re-captures as new best, restores pre-split state for gating.

**File:** `C++/src/CellUniverse.cpp` — pre-pass stub now logs
`[Pre-Pass]` header + per-cell seed positions so the stub-vs-real
transition is visible in logs.

### Effect

- 12345 at f3 should benefit from the extra 30 refine iters — the snapshot direction is stale but the centroid placement was OK; longer refinement in the same direction should let daughters settle to better-costing positions.
- Every accepted or rejected split now prints a full trace — next run's log will show exact primary directions, candidate drifts, voronoi keep counts, and refine deltas.
- No change to any bio / cost thresholds.

---

## 2026-04-11 — Pre-pass PCA grounding + compensatory perturbation — ACTIVE

### Problem/Motivation

Cross-referencing the triaxial pipeline plan
(`docs/plans/2026-04-10-triaxial-pipeline-redesign.md`) against the current
code revealed two missing procedures:

1. **Pre-pass was a stub.** Plan lines 67–91 call for running PCA on each
   pre-classified cell's snapshot-centered bounding box to overwrite
   `D1_exp/D2_exp` with image-grounded centroids. The stub left the seeds
   at raw snapshot extrapolation. This is almost certainly why 12345 at f3
   had a positive costDiff (+13.55) in run `064520` — the self-claim and
   direction were both based on stale f2 snapshot data.
2. **No compensatory perturbation after split failure.** Plan lines
   144–155 and 227–238 specify that both bio-fail and cost-fail branches
   should `revert; blacklist i; perturb i once`. The "perturb once" step
   was missing — rejected splits just blacklisted the cell without using
   the remaining iteration budget on a compensatory move.

### Solution

**(1) Real pre-pass.** New `Frame::imageGroundExpectedDaughters(...)`
public method — cheap PCA-only path that reuses
`gatherBrightPixelsVoronoi` + `pca3DWithCentroids` (the same helpers
`trySplitCellPhased` uses for its candidate generation). Returns the two
image-grounded PCA centroids via out-params plus an optional kept-pixel
count for logging. Handles the "PCA failed / too few pixels" case by
returning false, which the caller treats as "keep the snapshot seeds".

`CellUniverse::optimize` pre-pass loop rewritten to:

- Build the per-cell Voronoi claim-set for each pre_classified cell (same
  matrix as before — non_classified neighbors contribute `snapshot.center`,
  other pre_classified contribute current `{D1_exp, D2_exp}` which is
  seed on round 0 and image-grounded on round k+1).
- Call `imageGroundExpectedDaughters` and overwrite `expectedDaughters[name]`
  on success.
- Log `[Pre-Pass]` per cell with seed positions, grounded positions, and
  per-axis shifts so the re-grounding is visible in logs.
- Iterate `expected_daughter_pre_pass_iterations` rounds (default 1) —
  round k+1 uses round k's grounded positions for the neighbor claim sets.

**(2) Compensatory perturbation.** In `runPhase`, the split failure branch
now looks up the parent by name (after the `callback(false)` revert) and
runs a single `perturbCell` as the "perturb i once" compensation. Accepted
compensatory perturbations increment `perturbAccepted` and feed the running
residual accounting the same way main-loop perturbations do.

### Files Changed

**File:** `C++/includes/Frame.hpp`

Added public `Frame::imageGroundExpectedDaughters(cellIndex, snapshot, otherCellsClaimSets, outD1, outD2, outKeptPixels = nullptr)` declaration.

**File:** `C++/src/Frame.cpp`

Implemented `imageGroundExpectedDaughters` right after the anonymous
namespace close (line ~803). Uses `gatherBrightPixelsVoronoi`
(with `GatherStats` out-param for the kept count) and `pca3DWithCentroids`.
Box radius is `3 * max(snapshot.majorR, bR, minorR)` — same as the split
path's box so the same pixels are considered. Returns false when the pixel
count is below 20 or PCA fails.

**File:** `C++/src/CellUniverse.cpp`

Replaced the pre-pass stub with the real call to
`frame.imageGroundExpectedDaughters`. On success, `expectedDaughters[name]`
is updated to the grounded positions. `[Pre-Pass]` log lines now report
`kept=` pixels, `shift1=`, `shift2=` (distances from the snapshot seed to
the grounded centroid), and the full before/after positions.

Added the compensatory perturbation block inside `runPhase` after the
split rejection branch — `std::find_if` by cell name, call `perturbCell`
once with the normal main-loop perturbation sigmas (not the tight split
burn-in scales), and feed the accept result back into the residual
counters.

Also, inside `runPhase` before calling `trySplitCellPhased`, copy the
snapshot into a local `splitSnapshot` and — if the cell is pre-classified
and `expectedDaughters` has an entry — overwrite
`splitSnapshot.position`, `longAxisDir`, and `longAxisLength` from the
grounded `{D1_exp, D2_exp}` midpoint + delta. Without this override,
`trySplitCellPhased` rebuilds its self-claim from the raw snapshot
long-axis fields and misses the grounded positions the pre-pass produced.
The radii fields (`majorRadius`/`bRadius`/`minorRadius`) are left
untouched so daughter sizing still uses the snapshot dimensions.

### Effect

- **Pre-pass now image-grounds expected daughters.** 12345 at f3 will
  enter Phase B with `D1_exp/D2_exp` reflecting the actual f3 bright
  regions — if the real cell has moved or rotated since f2, the pre-pass
  catches it. Both the self-claim Voronoi filter and the direction
  agreement check will see updated data.
- **Rejected splits no longer waste their iteration.** The parent gets
  one compensatory perturbation at the normal main-loop sigmas — useful
  because rejected splits typically happen on cells that still need to
  refine their live fit.
- **Pipeline now matches the plan in full** (per the same doc's Pipeline
  Flow chart). The only remaining "stub" is
  `expected_daughter_pre_pass_iterations` being configurable: currently
  runs N rounds but most runs only need one.

Post-first-build validation pending.

---

## 2026-04-11 — Midpoint-near-parent + single-daughter-volume + lowered cost threshold — ACTIVE

### Problem/Motivation

Run `outputs/output_jihang_20260411_072416/` confirmed pre-pass + snapshot-
sized daughters are working, but exposed two specific failure modes:

1. **Real splits barely missing the cost threshold.** 12345 f3 at -76.54,
   e9077 f3 at -71.65, 12345 f4 at -66.69 — all within 15 of the -80
   threshold. These are real biological divisions that the system has
   identified correctly, but the -80 floor is too strict.
2. **Two false splits still passing with big cost deltas.** 1f2ed f4
   at -138.99 and e3d03 f5 at -158.86 — both accepted. Neither is a
   biological division.

Cross-referencing the trajectories against the log:

- **1f2ed f4**: parent stable at maxR ≈ 37 since f1 (oversized relative
  to the real cell). At f4, PCA finds a bright region 38 voxels west of
  parent center. Daughters land at (365, 296, 97) and (371, 297, 131) —
  **daughter midpoint is at (365, 296, 109), 38 vox from parent at
  (394, 284, 131)** — a full parent-diameter away. d2 volume = 0.74 *
  parent volume — one daughter is mimicking the parent itself.
- **e3d03 f5**: parent started at x=112 (initial CSV), drifted to x=131
  over 4 frames. At f5, PCA finds the real cell content back at x ≈ 104.
  Daughter midpoint at (104, 235, 85), **27 vox from parent at (131,
  233, 81)** — also past a parent-diameter.

Verified via `[Voronoi Out]` logs that neighbor exclusion is working
correctly in both cases (no neighbors within 60+ voxels of the daughter
positions). The PCA is latching onto real image bright content that's
simply not where the parent currently is.

**The separator that works**: `||daughter_midpoint - snapshot.position|| / parent.maxR`:

| Split    | Ratio          | Real?    |
| -------- | -------------- | -------- |
| 12345 f3 | **0.40** | real ✓  |
| e9077 f3 | **0.69** | real ✓  |
| 1f2ed f4 | **1.04** | false ✗ |
| e3d03 f3 | **1.28** | false ✗ |
| e3d03 f5 | **1.36** | false ✗ |

Threshold 0.95 cleanly separates. Snapshot position (not live) is the
correct reference: Phase A perturbation on a non-classified parent
frequently drifts the live parent onto one of the actual image daughters,
which would corrupt a live-reference check. The snapshot center is
frozen at frame start and reflects where the cell WAS before the image
started showing two daughters.

For asymmetric "mimicking parent" splits where one daughter inherits
most of the parent volume, a direct volume ratio check catches it:
`max(vol_d1, vol_d2) / refParentVolume > 0.65` → reject. 1f2ed f4 d2
vol = 0.74 * parent fails; 12345 f3 max daughter at 0.56, e9077 f3 at
0.41 both pass.

### Solution

Three complementary changes:

**(1) Midpoint-near-snapshot-parent bio gate.** New
`bio_max_midpoint_parent_fraction = 0.95`. Rejects when the daughter
midpoint is more than 0.95 × srcMaxR from the snapshot parent center.
Inline in `trySplitCellPhased` between the drift gate and the existing
`bioCheckDaughters` call, because it needs access to `snapshot.position`
which isn't in the bio-check signature.

**(2) Single-daughter volume bio gate.** New
`bio_max_single_daughter_volume_fraction = 0.65`. Added to
`bioCheckDaughters` as check 2b, between the combined-volume fraction
check and the buried checks.

**(3) Lower `split_cost` from 80 to 50.** With the two new gates
backstopping false positives, the cost threshold can be relaxed to let
real splits at -66 through -77 clear.

### Files Changed

**File:** `C++/includes/ConfigTypes.hpp` — added
`bio_max_midpoint_parent_fraction = 0.95f` and
`bio_max_single_daughter_volume_fraction = 0.65f` to `ProbabilityConfig`,
with parse entries in `explodeConfig`.

**File:** `C++/config/config.yaml` — added both new fields under
`prob:`. Lowered `split_cost` from 80 to 50.

**File:** `C++/src/Frame.cpp`

In `bioCheckDaughters`, split check 2 into 2 (combined volume) + 2b
(single-daughter volume):

```cpp
const double d1Vol = cellVolume(daughter1);
const double d2Vol = cellVolume(daughter2);
const double combinedVol = d1Vol + d2Vol;
// ... combined volume check unchanged ...

// 2b. Single-daughter volume gate.
const double maxDaughterFraction =
    std::max(d1Vol, d2Vol) / refParentVolume;
if (maxDaughterFraction > probConfig.bio_max_single_daughter_volume_fraction) {
    reasonOut = "single_daughter_volume_" + std::to_string(maxDaughterFraction);
    return false;
}
```

In `trySplitCellPhased` between the drift gate (5a) and the bio check
call (5b), added the midpoint-near-parent gate (5a'):

```cpp
const cv::Point3f daughterMidpoint(
    0.5f * (bestD1.getX() + bestD2.getX()),
    0.5f * (bestD1.getY() + bestD2.getY()),
    0.5f * (bestD1.getZ() + bestD2.getZ()));
const float midpointDist = static_cast<float>(cv::norm(
    daughterMidpoint - snapshot.position));
const float midpointLimit =
    probConfig.bio_max_midpoint_parent_fraction * srcMaxR;
if (midpointDist > midpointLimit) {
    std::cout << "[Split Reject bio] " << parentName
              << " reason=midpoint_from_parent dist=" << midpointDist
              << " limit=" << midpointLimit << ... << std::endl;
    return {0.0, noop};
}
```

### Effect

- **12345 f3 (real, diff=-76.5)**: midpoint ratio 0.40 ✓, single daughter frac 0.56 ✓, cost -76 < -50 ✓ → **will accept**
- **e9077 f3 (real, diff=-71.6)**: midpoint ratio 0.69 ✓, single daughter frac 0.41 ✓, cost -71 < -50 ✓ → **will accept**
- **12345 f4 (real, diff=-66.7)**: cost -66 < -50 ✓ → **will accept**
- **1f2ed f4 (false, diff=-138.99)**: midpoint ratio 1.04 ✗ → **will reject on midpoint**
- **e3d03 f3 (false, diff=-77.86)**: midpoint ratio 1.28 ✗ → **will reject on midpoint**
- **e3d03 f5 (false, diff=-158.86)**: midpoint ratio 1.36 ✗ → **will reject on midpoint**

Post-first-build validation pending.

---

## 2026-04-11 — Bridge brightness gate (pixel-projection version) — ACTIVE

### Problem/Motivation

User flagged a false-split pattern where the previous gates all pass:
two equal-size daughters placed symmetrically at the parent center,
covering one continuous bright cell in the image. Every existing gate
says "plausible":

- Midpoint = parent center → passes midpoint-vs-snapshot gate
- Both daughters equal size → passes size ratio gate
- Each daughter ~0.5 * parent vol → passes single-daughter volume gate
- Combined vol ≈ parent vol → passes combined volume gate
- Drift from seed small → passes drift gate

Yet visually (and biologically) it's one cell, not two. The only
distinguishing feature is **brightness profile along the split axis**:
a real division has a dim dividing groove between daughter centers, a
fake split has continuous high brightness.

### Solution — Bridge brightness gate using the PCA pixel set

Two-signal check on the Voronoi-filtered bright pixels that
`gatherBrightPixelsVoronoi` already produced for the split attempt.
Projects every pixel onto the daughter split axis and normalizes:
`t ∈ [-1, +1]` where `t=-1` is bestD1 center and `t=+1` is bestD2 center.

Two independent "is there a valley between daughters?" signals:

1. **Gap density**: fraction of in-range pixels (`|t| < 1.5`) that
   fall in the central band `|t| < 0.3` (middle 30% of the axis). Real
   division: the dividing groove empties the gap → low density. Fake
   continuous cell: uniform density → high.
2. **Valley ratio**: mean brightness of gap pixels (`|t| < 0.3`) divided
   by mean brightness of edge pixels (`0.6 < |t| < 1.1`). Real division:
   gap is dimmer than peaks → low ratio. Fake: brightness is flat →
   ratio ~1.0.

Gate rejects **only when BOTH signals indicate flat profile**
(`gap_density > 0.18` AND `valley_ratio > 0.85`). Requiring both to
fire is deliberately conservative — real divisions with partial
grooves (daughters not fully separated yet) still pass.

### Why reusing the PCA pixel set (not raw point samples)

- **All pixels, not 5 points.** Typically 100k–400k Voronoi-filtered
  pixels per split attempt, vs. 5 line samples in an earlier proposal.
- **Neighbor exclusion already applied.** These pixels are provably
  "this cell's territory" — neighbor brightness can't contaminate the
  profile.
- **Brightness threshold already applied.** Only pixels above the
  background cutoff are considered, matching the PCA input exactly.
- **No extra image sampling.** The cost is one linear pass over the
  pixel vector — ~1% of the gather cost.

### Files Changed

**File:** `C++/includes/ConfigTypes.hpp` — added `bio_bridge_max_gap_density = 0.18f` and `bio_bridge_max_valley_ratio = 0.85f` to `ProbabilityConfig`,
with parse entries.

**File:** `C++/config/config.yaml` — added both new fields under `prob:`.

**File:** `C++/src/Frame.cpp`

In `trySplitCellPhased`, between the midpoint-near-parent gate (5a') and
the `bioCheckDaughters` call (5b), added a new block "5a''. Bridge
brightness gate":

```cpp
const cv::Point3f axisVec = bestD2Pos - bestD1Pos;
const float axisLen = cv::norm(axisVec);
if (axisLen > 1e-3f && pixels.size() >= 1000) {
    const cv::Point3f axisDir = axisVec / axisLen;
    const float halfLen = 0.5f * axisLen;

    int gapCount = 0, edgeCount = 0, totalInRange = 0;
    double gapBrightSum = 0, edgeBrightSum = 0;

    for (const auto &bp : pixels) {
        const float t = dot(bp.pos - midpoint, axisDir) / halfLen;
        const float absT = std::abs(t);
        if (absT > 1.5f) continue;
        ++totalInRange;
        if (absT < 0.3f) { ++gapCount; gapBrightSum += bp.weight; }
        else if (absT > 0.6f && absT < 1.1f) { ++edgeCount; edgeBrightSum += bp.weight; }
    }

    const float gapDensity = (float)gapCount / totalInRange;
    const float valleyRatio = (gapBrightSum / gapCount) / (edgeBrightSum / edgeCount);

    if (gapDensity > bio_bridge_max_gap_density &&
        valleyRatio > bio_bridge_max_valley_ratio) {
        // reject
    }
}
```

Logs `[Split Bridge]` with `totalInRange`, `gapCount`, `edgeCount`,
`gapDensity`, `gapBright`, `edgeBright`, `valleyRatio` for every split
attempt so the thresholds can be tuned from real data.

Rejection log tag: `[Split Reject bio] ... reason=bridge_flat gapDensity=X valleyRatio=Y`.

### Effect

- Catches the hardest "two equal-size daughters at parent center over
  one continuous cell" pattern that passes every other gate.
- Logs the metrics for every split attempt even when the gate doesn't
  fire, so real vs. fake thresholds can be confirmed from real data in
  the next run.
- Pass-through path for real divisions: even a partial groove (gap
  density OR valley ratio shows dimming) keeps the split alive.

Post-first-build validation pending.

---

## 2026-04-11 — Burn-in radius sigma lock + midpoint gate removal — ACTIVE

### Problem/Motivation

Run `outputs/output_jihang_20260411_082245/` showed one false split
slipping past every gate: **e3d03 at f3**, `costDiff=-52.49`, accepted.
Log inspection showed:

- e3d03 got a triaxial fit this run (`shapeElong=1.289`, vs 1.011 in
  the prior run), so it was pre-classified.
- Daughters built at `0.794 * src = (18.14, 14.07, 18.14)`.
- After 20 burn-in + 30 refine iterations, **d2 had collapsed to
  r=(14.55, 10.00, 5.68)** — minorR hit near the radius floor of 5.
- d2 volume dropped from ~4600 to 826; single-daughter volume check
  passed (d1 at 0.634 < 0.65 threshold, barely).
- d2 moved to a distant spot (x=143, y=200, z=39), 40+ voxels from d1.
- Bridge brightness check passed because there's real empty space
  between d1 and d2 in the image (two bright regions with dark gap)
  → valleyRatio=0.426, gate requires BOTH signals flat, valley wasn't.

Root cause: the earlier `split_burn_in_pos_sigma_scale=0.4` only
constrained x/y/z position sigmas during burn-in. Radius sigmas ran
at full config value (~2 per axis), so over 50 iterations of decrease-
biased burn-in acceptance, d2's minorR drifted from 18 → 5.68, a
12-voxel collapse to the floor. Once d2 is a thin sliver, its volume
is tiny and it can "hide" at a distant image feature.

Separately, examining the midpoint-near-snapshot-parent gate showed
it has become **effectively dormant** due to an interaction with the
pre-pass: the pre-pass grounds `snapshot.position` to the image
bright region, which pulls the reference toward wherever the daughters
will land. For e3d03 f3 the raw snapshot was (113.36, 205.92, 92.15)
but the grounded snapshot was (128.23, 211.74, 76.41) — midpoint
distance dropped from 37.7 (raw) to 17.0 (grounded), well under the
21.7 limit. The gate is measuring "daughters are near where the
pre-pass said they are", which is trivially true by construction.

The 7 midpoint rejects in run 082245 were all on cells that shouldn't
have been split anyway (daughters of the false e3d03 f3 split or
non-dividing cells). With the radius lock fix below, those cascading
false splits disappear, and the midpoint gate loses its only
contribution.

### Solution

**(1) Radius sigma scale for burn-in.** New config field
`split_burn_in_radius_sigma_scale = 0.1`. Applied to
`majorRadius/bRadius/minorRadius` perturbation sigmas during the
candidate burn-in and the final refine pass, restored afterwards on
every exit path. At 0.1 with sigma=2, effective sigma during burn-in
is 0.2 per iteration — daughters can adjust by ~1-2 voxels total
across 50 iterations, nowhere near enough to collapse.

**(2) Remove the midpoint-near-snapshot-parent gate.** The bridge
brightness gate catches false splits via image content (no gap →
fake), which is a direct measurement rather than a positional proxy.
Removing the midpoint gate eliminates double-counting and the
false-positive risk on real migrating cells. The `daughterMidpoint`
computation is retained because the bridge gate uses it as the origin
of the axis projection.

The `bio_max_midpoint_parent_fraction` config field is left in place
(parsed but unused) so the gate can be re-enabled quickly if future
diagnostics show bridge missing something.

### Files Changed

**File:** `C++/includes/ConfigTypes.hpp` — added
`split_burn_in_radius_sigma_scale = 0.1f` in `ProbabilityConfig` with
parse entry. Documented the "collapsed sliver" pathology in the
comment.

**File:** `C++/config/config.yaml` — added
`split_burn_in_radius_sigma_scale: 0.1` under `prob:`.

**File:** `C++/src/Frame.cpp`

In `trySplitCellPhased`:

- Extended the burn-in sigma save/install/restore blocks to also
  save, scale, and restore `majorRadius/bRadius/minorRadius` sigmas
  alongside `x/y/z`. Applied at three locations: before the candidate
  loop (install), at the early-return path for `bestIdx < 0` (restore),
  and before the gate sequence (restore).
- Removed the midpoint-near-snapshot-parent gate (5a'). Left a comment
  explaining why (pre-pass interaction). The `daughterMidpoint`
  variable is still computed because the bridge gate projects pixels
  onto the axis that passes through it.

### Effect

- **e3d03 f3 (false, previously accepted)**: daughters will stay at
  their built-in radii (18.14, 14.07, 18.14) throughout burn-in. d2
  can no longer collapse to a sliver. With full-sized daughters, one
  at the parent location and one at a distant faint-content spot, the
  L2 cost either fails to improve or the single-daughter-volume gate
  fires on the larger d1 (which will be closer to 0.5 * parent when
  both daughters are built from the same src radii).
- **Midpoint rejects (run 082245 had 7)**: all were downstream of the
  e3d03 f3 false split. With e3d03 f3 no longer accepting, those
  false daughter-of-false-split attempts disappear entirely.
- **No change to real splits**: real splits (12345 f3, e9077 f3,
  12345 f4) weren't getting rejected by the radius collapse issue
  and their daughters stay near-parent-sized. The midpoint gate
  never rejected them, so removing it changes nothing.

Post-first-build validation pending.

---

## 2026-04-11 — Pre-pass calibration + snapshot-state parent for splits — ACTIVE

### Problem/Motivation

Run `outputs/output_jihang_20260411_160820/` (22 frames) accepted only
3 splits total: `1f2ed@f11` (real, correct), `e3d03@f14` (false), and
`12345@f20` (real but **17 frames late** — should be f3). Six expected
real splits never happened.

Log inspection showed a common pattern: by the time each pre-classified
cell's split attempt fires in Phase B, the live parent has been
perturbed significantly. For `e9077@f3`:

```
src=(28.17, 30.56, 25.17)    ← f2 snapshot
live=(23.94, 30.56, 23.94)   ← LIVE parent at split attempt time
pos=(315.83, 177.99, 129.48) ← live position
snap position=(297.34, 164.69, 120.79)
```

- majorR shrunk **28.17 → 23.94** (15% reduction)
- minorR shrunk **25.17 → 23.94** (5% reduction)
- Position drifted **(+18.5, +13.3, +8.7) voxels**

Phase B's main-loop perturbations use full sigmas (x/y=5, z=8, radius=2)
— tight burn-in sigmas only apply INSIDE `trySplitCellPhased` for the
candidate burn-in. Before the first split attempt fires, each Phase B
cell gets ~3 accepted perturbations (P_split≈0.3, blacklist prevents
multiple attempts) which is enough to drift 15-20 voxels if the bright
content pulls in one direction. The live parent collapses onto one
incipient daughter, so when the split attempt finally fires, the cost
delta `(collapsed_parent - daughters)` is small — real splits miss the
threshold by 5-15 cost units.

Additionally, `e9077@f3` was rejected by the drift gate with
`d1=19.57 > limit=19.34` — **by 0.23 voxels**. The PCA seed was placed
at z=123 but the real bright content was at z=101 (22 voxels away),
and the burn-in couldn't refine fully within the drift limit.

User image diagnostics confirmed:

- **Frame 7**: a visibly elongated real cell with a **round** fitted
  ellipsoid (shapeElongation ~1.0) — Phase A/B collapsed the fit.
- **Frame 7**: fitted cell drifted from the real bright mass into
  adjacent empty space — Phase A/B's position perturbation didn't
  return to the true cell.

### Solution

Two complementary changes:

**(1) Pre-pass position calibration.** New config field
`split_calibration_iterations_per_cell = 50`. Between the pre-pass and
Phase A/B, each cell gets 50 `perturbCell` iterations with:

- Tight position sigmas (reusing `split_burn_in_pos_sigma_scale = 0.4`
  → effective sigma 2 voxels per axis)
- **Zero radius sigmas** (`majorRadius/bRadius/minorRadius` sigmas
  forced to 0) — radii are completely frozen during calibration

This refines the parent's center without allowing collapse. The
calibrated live parent stays at snapshot-sized radii but finds the
best position for a fixed-size ellipsoid. For a dividing cell, this
is the center-of-mass of both daughters. For a non-dividing cell, it's
the true cell center.

**(2) Snapshot-state parent as split baseline.** In
`trySplitCellPhased`, at function entry after reading `liveParent`:

- Construct `snapshotParent` from `snapshot.position`, `snapshot.{major,b,minor}Radius`, `snapshot.theta{X,Y,Z}`, `snapshot.brightness`.
- Install `snapshotParent` at `cells[cellIndex]` and update
  `_synthFrame` + `_currentCost` via `generateSynthFrameFast` (patches
  only the affected z-slices).
- The baseline cost is now computed against the snapshot-state parent,
  not the Phase B-drifted live parent.
- `parent = cells[cellIndex]` captures the snapshotParent for downstream
  use (candidate generation, daughter building inherits snapshot theta
  via `buildDaughter`'s use of `parent.getCellParams().theta_*`).

All 7 rejection paths (cellIndex check, pixels<20, pca_failed, bestIdx<0,
drift_from_seed, bridge_flat, bioCheck, cost) now call a new
`restoreLiveParent()` lambda that:

- Swaps `cells[cellIndex]` back from snapshotParent to liveParent
- Calls `generateSynthFrameFast(snapshotParent, liveParent, ...)` to
  re-render the affected z-range
- Updates `_synthFrame`, `_currentCost`, `_currentCostPerSlice`

The acceptance callback's reject branch does the same (captures
`liveParent`, `snapshotParent`, `cellIndex`, `snapshotValid` by value).
The accept branch installs `bestCells` (with daughters) as before — the
live parent is permanently replaced.

### Files Changed

**File:** `C++/includes/ConfigTypes.hpp` — added
`split_calibration_iterations_per_cell = 50` in `ProbabilityConfig` +
parse entry.

**File:** `C++/config/config.yaml` — added
`split_calibration_iterations_per_cell: 50` under `prob:` with explanatory
comment.

**File:** `C++/src/CellUniverse.cpp` — added calibration loop after the
pre-pass block. Iterates every cell with tight position sigmas and
frozen radius sigmas, logs `[Calibration]` header + per-cell accept
counts and pre/post position drift.

**File:** `C++/src/Frame.cpp`

In `trySplitCellPhased`:

- Lines ~891–950: new block 0 — build snapshotParent from snapshot data,
  install at `cells[cellIndex]`, update synth/cost via
  `generateSynthFrameFast`. Logs `[Split Snapshot Parent]` with live vs
  snapshot positions/radii/thetas and the cost delta from the swap.
- Lines ~980–1000: new `restoreLiveParent` lambda that reverses the
  swap on rejection paths.
- Lines 1078, 1087, 1362, 1489, 1607, 1624, 1641: added
  `restoreLiveParent();` call before each existing `return {0.0, noop};`.
- Lines ~1665–1720: callback now captures `liveParentCopy`,
  `snapshotParentCopy`, `cellIndexCopy`, `snapshotValidCopy`. The reject
  branch installs `savedCellsCopy`, then swaps
  `cells[cellIndexCopy]` back to `liveParentCopy` and re-renders via
  `generateSynthFrameFast` so Phase B's live state is not lost after a
  rejected split attempt.
- `[Split Attempt]` log line now shows both `liveR`/`livePos` and
  `parentNow`/`parentPos` so the swap is visible in diagnostics.

### Effect

- Pre-pass calibration prevents radius collapse, giving each cell a
  fair "whole cell" baseline before Phase A/B starts.
- Splits compare daughters against a fully-reconstructed snapshot-state
  parent (position + radii + rotation from f_prev end). Real splits
  should show cost deltas closer to the theoretical maximum
  `(whole_divided_cell_synth) - (two_daughter_synth)` instead of
  `(collapsed_parent) - (two_daughters)`.
- Rejection is fully reversible: the live parent state is preserved and
  Phase B can continue to perturb normally after a failed split attempt.

Post-first-build validation pending. Watch for `[Calibration]` and
`[Split Snapshot Parent]` log lines, and specifically check
`costDiff` values for `12345@f3`, `e9077@f3`, and `1f89abf@f8` — they
should be more negative now that the baseline is an honest whole-cell
fit.

---

## 2026-04-11 — Centroid position calibration (snapshot vs PCA midpoint) — ACTIVE

### Problem/Motivation

The previous calibration pass only did perturbation refinement — starting
from the snapshot-inherited position and letting small local moves adjust
it. But for a cell that has drifted significantly between frames or is
already dividing, the snapshot position may be 15–30 voxels off from the
true center-of-mass. 50 perturbation iterations with tight sigmas can't
close that gap reliably.

The **weighted mean of Voronoi-filtered bright pixels** is an analytic
estimate of where the cell's mass is centered — which is exactly the
midpoint between daughters for a symmetric division, or the true center
for a non-dividing cell. This is the same quantity `pca3DWithCentroids`
computes internally before splitting on the eigenvector. We can use it
as a candidate position and compare to the snapshot, picking whichever
gives lower L2 cost.

### Solution

New `Frame::calibrateCellPositionViaCentroid(cellIndex, snapshot, claimSets)`:

1. Gathers Voronoi-filtered bright pixels from a snapshot-centered box
   (same box / same claim rules as the split path).
2. Computes the weighted mean position from the gathered pixels.
3. Builds a candidate Spheroid at the centroid position (radii,
   rotation, brightness inherited from the current cell — POSITION ONLY
   calibration).
4. Renders incrementally via `generateSynthFrameFast`, computes new cost.
5. If new cost < current cost, installs the candidate and returns true.
6. Otherwise reverts and returns false.

**Skipped** when the centroid is within 0.5 voxels of the current
position (no meaningful change) or when the bright pixel set is too
small (<20 pixels).

The calibration pass in `CellUniverse::optimize` now runs **two steps
per cell**:

- **Step 1 (analytic):** build the claim set, call
  `calibrateCellPositionViaCentroid`. This is a one-shot move, cheap,
  tries to jump to the analytically-correct position if it helps.
- **Step 2 (perturbation):** run 50 perturb iterations with tight
  position sigmas and frozen radii, starting from whatever position
  step 1 chose. Handles local refinement the centroid estimate may
  not reach perfectly.

Step 1 gets the cell to the right neighborhood in one move; step 2
polishes the position within that neighborhood.

### Files Changed

**File:** `C++/includes/Frame.hpp` — declared
`Frame::calibrateCellPositionViaCentroid(cellIndex, snapshot, otherCellsClaimSets)`, non-const, returns bool.

**File:** `C++/src/Frame.cpp` — implemented the method right after
`imageGroundExpectedDaughters`. Logs `[Centroid Calibration]` with
`ACCEPTED` / `REJECTED` / `no_move` / `skipped (...)` status plus
positions, move distance, cost delta, and kept-pixel count.

**File:** `C++/src/CellUniverse.cpp`

- Added a `buildCalibrationClaimSet` local lambda that builds the
  frame-start claim set for a given self-cell (pre-classified others
  contribute expected-daughter pairs, non-classified contribute snapshot
  center).
- Calibration loop now calls `calibrateCellPositionViaCentroid` before
  the perturbation refinement, passing the per-cell claim set.
- `[Calibration]` log line extended with `postCentroid`,
  `centroidShift`, `perturbDrift`, `totalDrift` so the analytic step
  and the perturbation step are separately traceable.

### Effect

- Cells that are far from their snapshot position at frame start get
  moved to their analytic center in one step, before perturbation wastes
  iterations walking there.
- Cells that are already at the right position (or where the centroid
  estimate is worse than the current fit) stay put — the cost comparison
  guarantees the calibration never makes things worse.
- Perturbation refinement still runs, so local L2 optimization still
  happens after the centroid jump.

Post-first-build validation pending. Watch for `[Centroid Calibration] ACCEPTED` lines on cells that should be dividing — especially `e9077@f3`
and `12345@f3`, where the snapshot position may be significantly off from
the true midpoint.

---

## 2026-04-11 — Dual-midpoint split candidate generation — ACTIVE

### Problem/Motivation

Split candidate generation always placed daughters around the PCA
midpoint `0.5 * (d1Pca + d2Pca)` — the weighted mean of the
Voronoi-filtered bright pixels. When the parent has drifted between
frames or when the PCA midpoint is slightly off due to asymmetric bright
distribution, this is the wrong starting point. The snapshot position
(where the cell was at the end of the previous frame) may give a
cleaner baseline — especially for cells that divide without moving
much.

Rather than picking one midpoint, generate candidates for BOTH and let
burn-in + cost comparison decide which one wins.

### Solution

In `trySplitCellPhased` candidate generation, replace the single-midpoint
loop with a dual-midpoint loop:

```cpp
std::vector<MidpointOption> midpoints;
midpoints.push_back({pcaMidpoint, pcaSep, "pca_mid"});
if (snapshotValid && distance(pca, snapshot) > 0.5) {
    midpoints.push_back({snapshotPos, snapLongLen, "snap_mid"});
}

for (direction in primaryDirs) {
    for (midpoint in midpoints) {
        generate primary candidate at midpoint + direction
        generate 2 rotation variants
        generate 2 translation variants
    }
}
```

When PCA and snapshot midpoints are within 0.5 voxels of each other,
only the PCA midpoint is used (duplicate candidates would be wasteful).
Otherwise both are tried.

Each candidate carries a label (`pca_mid_primary`, `pca_mid_rot+`,
`snap_mid_primary`, etc.). The best-performing candidate's label is
stored in `bestLabel` and logged in `[Split Accepted]` and
`[Split Reject cost]` lines so you can see which midpoint consistently
wins.

### Files Changed

**File:** `C++/includes/ConfigTypes.hpp` — raised
`split_candidates_per_attempt` default from 5 to 10 (two midpoints × 5
variants fits in 10 per primary direction).

**File:** `C++/config/config.yaml` — updated value and comment.

**File:** `C++/src/Frame.cpp`

In `trySplitCellPhased`:

- New `Candidate::label` field — carries the midpoint/variant identity.
- New `MidpointOption` struct listing (center, separation, label).
  Always includes `pca_mid`. Adds `snap_mid` when snapshot is valid and
  the two midpoints differ by more than 0.5 voxels.
- Candidate generation nested loop now iterates over (direction,
  midpoint) pairs instead of just directions.
- New `[Split Midpoints]` log line reports both midpoints and the
  chosen count.
- `bestLabel` tracked alongside `bestCells` / `bestSeed*` during the
  candidate loop.
- `[Split Cand]`, `[Split Accepted]`, `[Split Reject cost]` log lines
  now include the candidate label.

### Effect

- Split attempts now try BOTH the image-derived midpoint (PCA) and the
  historical midpoint (snapshot) as starting points for daughter
  placement. Burn-in + cost comparison picks the winner.
- For cells that drift between frames, the snapshot midpoint may no
  longer match the image — PCA midpoint wins. For cells that divide
  in place, the snapshot midpoint is the true center — it may give a
  lower-cost split. Either way, the better option is discovered
  automatically.
- `[Split Accepted] bestLabel=...` lines in the next run's log will
  tell us empirically which midpoint is winning for real splits — if
  it's consistently `snap_mid`, the snapshot midpoint is the better
  anchor; if `pca_mid`, the image centroid is.

Post-first-build validation pending.

## 2026-04-12 — Shape EMA, config tuning, drift gate disable

### Shape EMA (volume-preserving PCA-driven radius blending) — **ACTIVE**

**Problem:** Cell 1f89ab stays spherical (majorR≈minorR≈30.5, shapeElong=1.09) across
all frames even though the real image shows clear pre-division elongation at f7+. The
perturbation optimizer has no gradient pushing radii apart because a round sphere covers
both bright lobes well enough in L2 cost. This causes: low P(split), no snapshot
direction, poor PCA seed placement, and missed splits.

**Solution:** Shape EMA — analogous to the existing brightness EMA. Each frame, after
optimization, measure the cell's PCA eigenvalue ratios from real-image pixels inside
the cell volume, then blend the fitted radii toward the observed shape while preserving
total volume (cbrt(a*b*c) stays constant).

**Files changed:**

- `C++/includes/ConfigTypes.hpp` — added `shapeUpdateBlend` field + YAML parsing
- `C++/includes/Spheroid.hpp` — added `PCAShapeResult` struct, `measurePCAShape()`, `setRadii()`
- `C++/src/Spheroid.cpp` — implemented `measurePCAShape()` (PCA on weighted bright pixels
  with isotropic normalization) and `setRadii()` (clamped radius setter)
- `C++/src/CellUniverse.cpp` — shape EMA block after brightness EMA: measure PCA, compute
  volume-preserving target radii, blend sorted axes, apply via `setRadii()`
- `C++/config/config.yaml` — added `shapeUpdateBlend: 0.3`

**Effect:** Cells whose real-image shape diverges from the fitted model will gradually
adapt their radii to match, making shapeElongation track real pre-division elongation.

### Config tuning — **ACTIVE**

- `split_cost: 20 → 15` — real splits [-90, -18] vs false splits [+5, +57], 23+ unit gap
- `bio_max_drift_parent_fraction: 0.4 → 999.0` — drift gate disabled; bridge+cost sufficient
- `bio_max_drift_daughter_fraction: 0.85 → 999.0` — same rationale
- `brightnessMeanAmplification: 1.1 → 1.0` — removed 10% amplification causing cell shrinking

### Always-dual-direction candidate generation — **ACTIVE**

**Problem:** At f10, cell 1f2ed's snapshot had the correct y-dominant split direction
`(0.23, -0.89, 0.40)` but it was discarded because `snapElong=1.19 < threshold 1.20`.
PCA gave a wrong z-dominant axis `(0.19, 0.13, 0.97)`. Daughters placed along z failed
the bridge gate. By f12 the directions corrected, but the split was 1 frame late.

**Solution:** Remove the direction-selection heuristic entirely. Always try BOTH PCA
and snapshot directions (when they disagree by >10°), generating 20 candidates instead
of 10. Cost picks the winner. When directions agree (<10°), only 10 candidates are
generated (no duplication waste).

**File:** `C++/src/Frame.cpp` lines ~1272-1295

**Before:** `useSnapshotDirection` flag gated by `shape_elongation_classify_threshold`.
If `snapElong < threshold`, snapshot direction was discarded and only PCA used.

**After:** PCA direction always included. Snapshot direction added as second primary
whenever it exists and differs from PCA by >10°. No classify threshold for direction
selection. The threshold still controls P(split) classification but no longer gates
the direction.

**Effect:** At f10, both z-direction (PCA) and y-direction (snapshot) candidates are
tried. The y-direction candidates should win on cost, enabling the split 1-2 frames
earlier.

## 2026-04-12 (evening) — Ellipsoid rename, local-frame PCA, shortest-axis split

### Cherry-pick: Spheroid → Ellipsoid rename (Vincent's 8c0975a) — **ACTIVE**

Renamed Spheroid to Ellipsoid across all files. `majorRadius` → `aRadius`,
`minorRadius` → `cRadius`. `abRatio` perturbation parameter removed — a, b, c axes
now perturb independently (sigma=5 each, up from sigma=2). Rotation sigma 0.05→0.125.

### Split axis: shortest fitted axis — **ACTIVE**

Changed `Ellipsoid::worldSplitAxis()` (renamed from `worldLongAxis`) to return the
**shortest** of (a, b, c) instead of the longest. Daughters separate along the thin
dimension of the cell. For a pancake cell lying flat, this places daughters stacked
through z rather than side by side in xy.

All naming updated: `longAxisDir` → `splitAxisDir`, `longAxisLength` → `splitAxisLength`,
`worldLongAxis()` → `worldSplitAxis()` across Ellipsoid.hpp, Ellipsoid.cpp,
CellUniverse.cpp, and Frame.cpp.

### Local-frame PCA — **ACTIVE**

**Problem:** PCA in world frame is dominated by the cell's overall shape (flat cells
produce z-dominant eigenvectors from their thinness, not from actual daughter separation).
For cells with a=c, rotation drift makes the snapshot direction unreliable. World-frame
PCA cannot distinguish cell shape from brightness asymmetry.

**Solution:** Transform bright pixels into the cell's local coordinate frame before PCA:

1. `localPos = R_inverse * (worldPos - cellCenter)`
2. `localPos /= (a, b, c)` (normalize to unit sphere)
3. PCA on normalized local positions
4. Un-normalize eigenvector (scale by a, b, c) and rotate back to world

**Files changed:**

- `Frame.cpp: pca3DWithCentroids()` — new optional parameters `cellCenter`, `invRotation`,
  `radiusA/B/C`. When provided, transforms to local frame before covariance computation.
  Eigenvector un-normalized and rotated back to world. D1/D2 centroids in world space.
- `Frame.cpp: imageGroundExpectedDaughters()` — passes cell rotation + radii to PCA
- `Frame.cpp: trySplitCellPhased()` — passes parent rotation + radii to PCA
- `Ellipsoid.hpp` — `generateInverseRotationMatrix()` moved to public

### Always-dual-direction (no angle threshold) — **ACTIVE**

Removed the 10-degree angle threshold for adding snapshot direction. Both PCA and
snapshot directions are ALWAYS tried as candidate primaries (10 candidates each = 20
total). This covers a=c cells where snapshot direction drifts due to unconstrained
rotation, and cells where PCA and snapshot happen to agree on the wrong axis.

### Shape EMA — **REMOVED**

Added and removed in the same session. PCA inside a single undivided cell sees symmetric
brightness distribution — cannot detect pre-division elongation. The chicken-and-egg
problem is unsolvable with this approach. Vincent's independent-axis perturbation
(sigma=5, abRatio removed) addresses shaping through the optimizer instead.

### Drift-from-seed gate — **REMOVED** (code + config)

Fully removed from Frame.cpp, ConfigTypes.hpp, and config.yaml. drift1/drift2 retained
as diagnostic-only values in log output. All false splits caught by drift were also
caught by cost gate. The gate blocked real splits for spherical cells where PCA seed
placement was poor (20-28 voxel drift needed to reach the actual daughter blob).

### Config changes

- `split_cost: 50 → 15` (tuned through 20→15 based on iterative validation)
- `shape_elongation_classify_threshold: 1.20 → 1.15`
- `split_candidate_burn_in_iterations: 20 → 50`
- Radius sigmas: 2 → 5 (all three axes independent, abRatio removed)
- `brightnessUpdateBlend: 0.9 → 0.95` (Vincent's value)
- `brightnessMeanAmplification: 1.1` (Vincent's value, kept after testing)
- `brightnessMeasurementTopPercentile: 0.3` (new, Vincent's feature)

---

## 2026-04-13 — Local-axis split directions, adaptive bridge gate, anti-shrink penalty, age gate — ACTIVE

### 1. Replace PCA direction with local-axis brute-force (Frame.cpp)

**Problem:** Both world-frame and local-frame PCA produce z-dominant split directions for flat cells. The brightness gradient in z (concentrated fluorescent signal in central z-slices) dominates the eigenvector regardless of normalization. Cell 1f2ed (a=c=34.9, b=27.4) was blocked by bridge_flat for 10+ frames because PCA always found z as the principal direction.

**Solution:** Remove PCA for direction-finding entirely. Instead, extract all three cell-local axes (a, b, c) from the rotation matrix and try each as a candidate split direction. Cost picks the winner.

**Files changed:**

**File:** `C++/src/Frame.cpp`

**Lines 759-822 (new helper):**

```cpp
bool centroidsAlongAxis(
    const std::vector<BrightPixel> &points,
    const cv::Point3f &axis,
    cv::Point3f &d1Out,
    cv::Point3f &d2Out)
```

Projects bright pixels onto a given direction, splits at the median, returns weighted centroids of each half. Gives data-driven midpoint and separation for each candidate axis.

**Lines 1408-1458 (replaced PCA direction logic):**

**Before:**

```cpp
cv::Point3f dirPca, d1Pca, d2Pca;
if (!pca3DWithCentroids(pixels, dirPca, d1Pca, d2Pca, ...))
    ...
std::vector<cv::Point3f> primaryDirs;
primaryDirs.push_back(dirPca);
if (snapshotValid) primaryDirs.push_back(snapshot.splitAxisDir);
```

**After:**

```cpp
auto extractWorldAxis = [&](int axisIdx) -> cv::Point3f { ... };
const cv::Point3f axisA = extractWorldAxis(0);
const cv::Point3f axisB = extractWorldAxis(1);
const cv::Point3f axisC = extractWorldAxis(2);
std::vector<cv::Point3f> primaryDirs = {axisA, axisB, axisC};
```

**Lines 1467-1510 (data-driven separation per axis):**

For each axis, `centroidsAlongAxis` projects pixels onto that direction and computes the centroid of each half. The distance between centroids = data-driven separation for that axis. Fallback to axis radius if centroid computation fails.

```cpp
struct AxisPlacement { cv::Point3f d1, d2, midpoint; float separation; bool valid; };
AxisPlacement axisPlace[3];
for (int i = 0; i < 3; ++i) {
    axisPlace[i].valid = centroidsAlongAxis(pixels, primaryDirs[i], ...);
    axisPlace[i].separation = cv::norm(axisPlace[i].d1 - axisPlace[i].d2);
}
```

**Lines 1514-1576 (candidate generation with axis labels):**

Candidates now labeled `data_axA_primary`, `snap_axB_rot+`, etc. Two midpoints per axis (data-driven centroid + snapshot center if different).

**Effect:** 1f2ed splits at f8 instead of f12 (4-frame improvement). The a-axis in world frame correctly captures xy-separation that PCA missed. No dependency on PCA direction quality. Log shows `[Split Dirs] mode=local_axes` with all three axis directions + radii.

---

### 2. Adaptive bridge gate — surface-based gap measurement (Frame.cpp)

**Problem:** The old bridge gate used fixed fractional zones (|t|<0.3 = gap, |t| 0.6-1.1 = edges) along the daughter axis. For close daughters (sep=40, radius≈24), the "gap" zone was inside both daughters' volumes — no valley possible. Bridge_flat rejected 100+ true splits across 3 runs.

**Solution:** Compute each daughter's actual ellipsoid radius along the split axis direction. Gap = space between daughter surfaces. Skip bridge entirely when daughters overlap (gapWidth ≤ 2).

**File:** `C++/src/Frame.cpp`

**Lines 1897-2023 (replaced bridge gate):**

**Before:**

```cpp
const float t = signedProj / halfLen;
if (absT < 0.3f) { gapCount++; ... }
else if (absT > 0.6f && absT < 1.1f) { edgeCount++; ... }
```

**After:**

```cpp
auto ellipsoidRadiusAlongDir = [](const Ellipsoid &e, const cv::Point3f &dir) -> float {
    // ||diag(a,b,c) * R^T * dir||
    ...
};
const float r1Along = ellipsoidRadiusAlongDir(bestD1, axisDir);
const float r2Along = ellipsoidRadiusAlongDir(bestD2, axisDir);
const float gapWidth = axisLen - r1Along - r2Along;
const float gapLo = -halfLen + r1Along;  // d1 inner surface
const float gapHi =  halfLen - r2Along;  // d2 inner surface
// Gap zone: between surfaces. Edge zones: inside each daughter.
if (gapWidth > 2.0f) { /* apply bridge gate */ }
// else: skip bridge, cost decides
```

**Effect:** Bridge_flat rejections dropped from 100+ to 1 in the test run. True splits with overlapping daughters (gapWidth < 0) are no longer blocked. Log now shows `axisLen`, `r1Along`, `r2Along`, `gapWidth`.

---

### 3. Brightness-scaled size reduction penalty (Frame.cpp)

**Problem:** Dim cells (brightness 0.12-0.17) shrink because their weak L2 signal (~0.02 per-pixel error at edges) can't hold cell size against the `size_reduction_penalty_weight`. Bright cells (0.7+, ~0.49 per-pixel) hold size perfectly. 8cbdf shrank from vol=6222 to vol=1801 (71% loss). Correlated with brightness: shrinking cells had brightness 0.12-0.17, stable cells had 0.55+.

**Solution:** Scale the shrink penalty by inverse brightness: `penalty *= 1.0 / max(brightness, 0.15)`. Dim cells get ~6.7× stronger penalty, bright cells ~1.4×.

**File:** `C++/src/Frame.cpp`

**Lines 6-28 (modified `computeSizeReductionPenalty`):**

**Before:**

```cpp
const auto accumulateReductionPenalty = [weight](float oldRadius, float newRadius) {
    ...
    return static_cast<double>(weight) * reductionRatio * reductionRatio;
};
```

**After:**

```cpp
const float brightness = oldCell.getBrightness();
const double brightnessScale = 1.0 / static_cast<double>(std::max(brightness, 0.15f));
const auto accumulateReductionPenalty = [weight, brightnessScale](...) {
    ...
    return static_cast<double>(weight) * brightnessScale * reductionRatio * reductionRatio;
};
```

**Effect:** At brightness=0.15 → effective weight = 50 × 6.67 = 333. At brightness=0.7 → effective weight = 50 × 1.43 = 71. Dim cells get much stronger protection against shrinkage.

---

### 4. Minimum split age gate (Ellipsoid.hpp, ConfigTypes.hpp, CellUniverse.cpp)

**Problem:** With the adaptive bridge gate, premature splits occurred: 12345..0 split at f5 (2 frames after birth at f3), 1f89ab1 split at f8 (1 frame after birth at f7). Biologically impossible — cells need multiple frames to grow and divide.

**Solution:** Track each cell's birth frame. Don't attempt splits until `min_frames_before_split` frames have passed since birth. Initial cells (from CSV) have birthFrame=-100 so they're always eligible.

**File:** `C++/includes/Ellipsoid.hpp`

**Line 91 (new private field):**

```cpp
int _birthFrame = -100; // frame when this cell was created (-100 = initial/always eligible)
```

**Lines 132-133 (new accessors):**

```cpp
int getBirthFrame() const { return _birthFrame; }
void setBirthFrame(int frame) { _birthFrame = frame; }
```

**File:** `C++/includes/ConfigTypes.hpp`

**Line 245 (new config field):**

```cpp
int min_frames_before_split = 8;
```

**Line 273 (YAML parsing):**

```cpp
if (node["min_frames_before_split"]) min_frames_before_split = node["min_frames_before_split"].as<int>();
```

**File:** `C++/src/CellUniverse.cpp`

**Lines 730-737 (age check before split):**

```cpp
const int cellAge = frameIndex - frame.cells[cellIdx].getBirthFrame();
const int minAge = config.prob.min_frames_before_split;
const bool canSplit = pSplit > 0.0f
                   && splitBlacklist.count(cellName) == 0
                   && cellAge >= minAge;
```

**Lines 798-806 (tag daughters on accepted split):**

```cpp
for (auto &cell : frame.cells) {
    if (cell.getBirthFrame() < 0 &&
        cell.getName().size() > cellName.size() &&
        cell.getName().substr(0, cellName.size()) == cellName) {
        cell.setBirthFrame(frameIndex);
    }
}
```

**Effect:** Daughter born at f3 can first split at f8 (5 frames later per config). Blocks the 1-2 frame premature splits while allowing legitimate 2nd-gen splits at f19-20 (16+ frames after birth).

---

### 5. Frozen burn-in radii (config.yaml)

**Problem:** During burn-in (50 iterations), daughter c-radius collapsed from built=12.5 to 5.0-5.6 even with `split_burn_in_radius_sigma_scale=0.1` (sigma=0.5). This caused the volume_fraction gate to reject e9077's true split (combined fraction dropped to 0.44-0.58, below the 0.6 threshold).

**Solution:** Set `split_burn_in_radius_sigma_scale: 0.0` — freeze radii entirely during burn-in. Daughters keep their built size (0.794 × parent), burn-in only refines position and orientation.

**Effect:** Daughters maintain correct volume through burn-in, preventing volume_fraction false rejections on real splits.

---

### Config changes

- `split_candidates_per_attempt: 10 → 30` (1-3 short axes × 2 midpoints × 5 variants)
- `split_burn_in_radius_sigma_scale: 0.1 → 0.0` (freeze radii during burn-in)
- `min_frames_before_split: 5` (new — minimum cell age before split eligible)

---

## 2026-04-13 (cont.) — Volume floor, short-axis filter, minimum-gap bridge — ACTIVE

### 6. Volume floor — cells can reshape but not shrink (Ellipsoid.hpp, Ellipsoid.cpp, Frame.cpp)

**Problem:** 8cbdf shrinks from vol=6269 to vol=2921 (53% loss) over 13 frames. The brightness-scaled shrink penalty is insufficient. Meanwhile, e3d03 pancakes naturally (a grows 32%, c shrinks 19%, volume +1%) — legitimate shape changes that must be allowed.

**Solution:** Track birth volume (a×b×c at construction). Reject any perturbation that would push total volume below 85% of birth volume. Allows arbitrary axis reshaping (pancaking, elongation) as long as total volume is conserved.

**File:** `C++/includes/Ellipsoid.hpp`

**Line 92 (new field):**

```cpp
double _birthVolume = 0.0; // a*b*c at creation; volume floor = 85% of this
```

**Lines 135-140 (new accessors):**

```cpp
double getBirthVolume() const { return _birthVolume; }
void setBirthVolume(double vol) { _birthVolume = vol; }
double getVolume() const {
    return static_cast<double>(getARadius()) *
           static_cast<double>(getBRadius()) *
           static_cast<double>(getCRadius());
}
```

**File:** `C++/src/Ellipsoid.cpp`

**Line 230 (constructor — set birth volume after clamping):**

```cpp
_birthVolume = a * b * c;
```

**Lines 559-562 (getPerturbedCell — preserve birth volume + birth frame):**

```cpp
perturbedCell._birthVolume = _birthVolume;
perturbedCell._birthFrame = _birthFrame;
```

**File:** `C++/src/Frame.cpp`

**Lines 334-345 (perturbCell — volume floor check after min-radius check):**

```cpp
{
    const double birthVol = cells[index].getBirthVolume();
    if (birthVol > 0.0) {
        const double newVol = cells[index].getVolume();
        if (newVol < 0.85 * birthVol) {
            cells[index] = oldCell;
            return {0.0, [](bool) {}};
        }
    }
}
```

**Effect:** Validated against best-run data: e3d03 pancaking (vol +1% over 45 frames) would never hit the 85% floor. 8cbdf shrinkage (vol -29% by f10) would be caught at ~f5.

---

### 7. Short-axis filter for split directions (Frame.cpp)

**Problem:** Trying the longest axis as a split direction creates false splits where daughters placed end-to-end match the z-brightness gradient (e3d03 false split at f2 via axA/axC with sep=54.9, costDiff=-39 to -69).

**Solution:** Only try axes whose radius is within 20% of the shortest radius. For elongated cells (e9077: 31.2/24.3/16.6), only the shortest axis (axC) qualifies → 10 candidates. For nearly-spherical cells (1f2ed: 32.7/31.8/32.7), all three qualify → 30 candidates.

**File:** `C++/src/Frame.cpp`

**Lines 1448-1483 (replaced "exclude unique longest" with "include within 20% of shortest"):**

```cpp
const float minR = std::min({rA, rB, rC});
const float shortAxisThreshold = minR * 1.2f;
for (int i = 0; i < 3; ++i) {
    if (radii3[i] <= shortAxisThreshold) {
        primaryDirs.push_back(axes3[i]);
    }
}
```

**Effect:** e9077 and 12345 correctly get 1 axis (10 candidates). 1f2ed/1f89ab keep all 3 (30 candidates). e3d03 (21.8/20.7/21.8) still gets all 3 because radii are too close — this remains an open issue.

---

### 8. Minimum-gap bridge gate — always measures central region (Frame.cpp)

**
    Problem:** The adaptive bridge gate skipped entirely when daughters overlapped (gapWidth ≤ 0), removing false-split protection for close/overlapping daughters. Premature splits and false splits passed unchecked.

**Solution:** Enforce a minimum gap half-width of 15% of the daughter axis half-length. The bridge gate always samples the central region between daughters and always fires. No more skip condition.

**File:** `C++/src/Frame.cpp`

**Lines 1967-1975 (minimum gap zone):**

```cpp
const float minGapHalf = 0.15f * halfLen;
const float surfaceGapHalf = 0.5f * gapWidth;
const float effectiveGapHalf = std::max(minGapHalf, surfaceGapHalf);
const float gapLo = -effectiveGapHalf;
const float gapHi =  effectiveGapHalf;
```

**Lines 2006-2010 (bridge always fires):**

```cpp
// No more gapWidth > 2.0 skip. Bridge always fires.
const bool densityFlat = gapDensity > bio_bridge_max_gap_density;
const bool brightnessFlat = valleyRatio > bio_bridge_max_valley_ratio;
```

**Effect:** bridge_flat rejections went from 0-1 (when skipping) to 13 in this run. True splits with real valleys (e9077 vr=0.47-0.52, 12345 vr=0.55-0.60) pass correctly. But e3d03 still passes bridge because gapDensity stays low (0.05-0.07) for the narrow minimum gap zone.

---

### 9. New diagnostic logs (Frame.cpp, CellUniverse.cpp)

- `[Split Winner]` — after candidate loop, shows winning candidate idx, label, pre-refine costDiff, seed positions
- `[Split Age Block]` — when age gate prevents a split attempt
- `[Split Dirs]` — enhanced with `minR`, `thresh`, `selected=[axB,axC]`
- `[Split Bridge]` — enhanced with `effGapHalf`

---

## 2026-04-13 (evening): Per-frame PCA shape update replaces symmetric gradient shrink

**Status:** ACTIVE

**Problem / Motivation:** Thin/elongated real cells were being modeled as round cells. The radius-gradient update in `Ellipsoid::measureRadiusGradient` samples at 2 poles per axis; when the cell's rotation is misaligned with the real bright streak, every pole lands in dark regions and all 3 axes shrink symmetrically — the cell stays round, just smaller. Rotation perturbation (sigma=0.1 rad) is too weak to find the correct orientation via L2 cost alone, especially for dim cells. The gradient has no mechanism to compare a-axis vs b-axis and drive an aspect-ratio change.

**Fix:** Per-frame 3-eigenvector PCA on Voronoi-filtered bright pixels. Eigenvectors become the new cell axes (rotation); radii scale with sqrt of eigenvalues. Greedy-matches PCA axes to current (a,b,c) identity to prevent axis swaps between frames. Results blended with current state (EMA) for stability. Volume floor preserved.

**Files changed:**

1. `C++/includes/Ellipsoid.hpp` — added `setRotation(tx, ty, tz)` and `getThetaX/Y/Z()` accessors.
2. `C++/src/Ellipsoid.cpp` — implemented `setRotation`.
3. `C++/includes/Frame.hpp` — declared `calibrateCellShapeViaPca`.
4. `C++/src/Frame.cpp` — implemented `calibrateCellShapeViaPca` plus static helpers `rotationMatrixToEulerZYX`, `wrapAngle`, `blendAngle`.
5. `C++/includes/ConfigTypes.hpp` — added `pcaShapeBlend`, `pcaShapeRadiusScale`, `pcaShapeMinPixels`, `pcaShapeScanFactor` to `EllipsoidConfig`.
6. `C++/config/config.yaml` — disabled radius gradient, enabled PCA shape update with `blend=0.3`, `scale=2.236`, `minPixels=50`, `scanFactor=1.5`.
7. `C++/src/CellUniverse.cpp` — new PCA shape calibration pass that runs after position calibration and before pre-pass.

### Detail: `Ellipsoid::setRotation` and theta getters

**File:** `C++/includes/Ellipsoid.hpp` (after `setRadii` declaration):

```cpp
void setRotation(float thetaX, float thetaY, float thetaZ);
float getThetaX() const { return static_cast<float>(_theta_x); }
float getThetaY() const { return static_cast<float>(_theta_y); }
float getThetaZ() const { return static_cast<float>(_theta_z); }
```

**File:** `C++/src/Ellipsoid.cpp` (after `setRadii`):

```cpp
void Ellipsoid::setRotation(float thetaX, float thetaY, float thetaZ)
{
    _theta_x = static_cast<double>(thetaX);
    _theta_y = static_cast<double>(thetaY);
    _theta_z = static_cast<double>(thetaZ);
}
```

### Detail: `Frame::calibrateCellShapeViaPca` (new method)

**File:** `C++/src/Frame.cpp` (inserted after `calibrateCellPositionViaCentroid`):

Core algorithm:

1. Gather Voronoi-filtered bright pixels in a box of radius `scanFactor * max(a,b,c)` around the cell's current center.
2. Skip if pixel count < `minPixels`.
3. Compute weighted centroid and 3x3 covariance in world frame.
4. `cv::eigen(cov, eigvals, eigvecs)` — rows sorted by descending eigenvalue.
5. Greedy-match PCA axes to current (a,b,c) axes by largest |dot product|. Flip matched axis signs so dot > 0.
6. Build rotation matrix from matched axes as columns; flip 3rd axis if det < 0 (enforce proper rotation).
7. Decompose matrix to Euler angles via convention R = Rz*Ry*Rx:
   - `ty = asin(-R[2][0])`, else gimbal lock branch
   - `tx = atan2(R[2][1], R[2][2])`, `tz = atan2(R[1][0], R[0][0])`
8. Target radius per axis = `radiusScale * sqrt(eigenvalue)`.
9. Blend: `newR = (1-blend)*oldR + blend*targetR` per axis; angles blended via shortest circular path.
10. Apply volume floor: reject if `newVol < 0.85 * birthVol`.
11. `setRadii(newA, newB, newC)` + `setRotation(newTx, newTy, newTz)`.

### Detail: config fields (`EllipsoidConfig`)

**File:** `C++/includes/ConfigTypes.hpp` (new fields after `radiusGradientIterations`):

```cpp
float pcaShapeBlend{0.0f};
float pcaShapeRadiusScale{2.236f};
int   pcaShapeMinPixels{50};
float pcaShapeScanFactor{1.5f};
```

### Detail: YAML config

**File:** `C++/config/config.yaml`:

```yaml
radiusGradientStepSize: 0.0      # disabled — PCA replaces gradient
radiusGradientIterations: 0

pcaShapeBlend: 0.3
pcaShapeRadiusScale: 2.236       # sqrt(5) for uniform ellipsoid variance
pcaShapeMinPixels: 50
pcaShapeScanFactor: 1.5
```

### Detail: new pass in `CellUniverse::optimize`

**File:** `C++/src/CellUniverse.cpp` (inserted after position-calibration block, before pre-pass):

For each cell, build a ClaimSet where each OTHER cell contributes its current (calibrated) position as a single claim point, then call `calibrateCellShapeViaPca`. Regenerate synth frame afterwards since rotations and radii changed.

### Effect

- Rotation now follows the bright pixel cloud directly, independent of L2 cost. Thin diagonal cells get their a-axis aligned with the streak in one frame.
- Radii scale with eigenvalues, so aspect ratio adapts: a>>b≈c for elongated streaks, a≈b≈c for spherical cells.
- EMA blend keeps updates stable — single-frame noise can't snap the cell shape.
- Replaces `measureRadiusGradient` loop (disabled via config). Gradient code remains in place but unused; can be deleted later.

### Diagnostic logs

- `[PCA Shape] frame N cells=M blend=B scale=S minPixels=P scanFactor=F` — pass entry
- `[PCA Shape] cell=... kept=K tgtR=(a,b,c) curR=(a,b,c) newR=(a,b,c) tgtT=(x,y,z) newT=(x,y,z)` — per-cell update
- `[PCA Shape] cell=... skipped kept=K min=P` — when pixel count too low
- `[PCA Shape] cell=... floor_blocked newVol=V birthVol=B` — volume floor prevented update

### Validation targets

- Thin/elongated real cells should converge to elongated ellipsoids (aspect ratio > 2) within 3-5 frames instead of staying round.
- `tgtR` should differ significantly across the three axes for elongated cells (check `[PCA Shape]` logs).
- Non-splitting cells (e3d03, 8cbdf) should still remain rejected for splitting — elongation eigenvalue ratio alone is NOT the split gate, so true blob shapes staying blob-shaped is expected and fine.

---

## 2026-04-13 (late evening): Iterative PCA shape fit + cleanup of dead paths

**Status:** ACTIVE

**Problem:** Three shape-tracking code paths were stacked (stochastic radius perturbation, radius EMA, radius gradient) and all failed. The single-shot PCA added earlier that day used an EMA blend of 0.3 that converged too slowly; cells in the screenshot of thin diagonal streaks stayed round for many frames. Additional dead code (age gate, birthVolume ratchet, size-reduction penalty, min_frames_before_split) was stacking up with no consumer.

**Fix:** Replace the single-shot PCA with an iterate-to-convergence fit, and delete every dead approach.

### Deleted code paths

**Removed from `C++/includes/ConfigTypes.hpp` (EllipsoidConfig):**
- `radiusUpdateBlend`, `radiusMeasurementBrightnessThreshold`  (radius EMA)
- `radiusGradientStepSize`, `radiusGradientIterations`  (radius gradient)
- `pcaShapeBlend`, `pcaShapeScanFactor`  (superseded by iterative config)

**Removed from `C++/includes/ConfigTypes.hpp` (ProbabilityConfig):**
- `size_reduction_penalty_weight` (failed per prior session notes — froze cells)
- `min_frames_before_split` (declared, parsed, never read)

**Removed from `C++/includes/Ellipsoid.hpp`:**
- `_birthFrame`, `getBirthFrame`, `setBirthFrame` (age gate never triggered)
- `_birthVolume`, `getBirthVolume`, `setBirthVolume` (ratchet prevented legitimate shrinkage)
- `measureRadiiFromImage` declaration
- `measureRadiusGradient` declaration

**Removed from `C++/src/Ellipsoid.cpp`:**
- Full bodies of `measureRadiiFromImage` and `measureRadiusGradient`
- `_birthVolume = a*b*c` ctor line
- `perturbedCell._birthVolume = _birthVolume; perturbedCell._birthFrame = _birthFrame;`

**Removed from `C++/src/Frame.cpp`:**
- `computeSizeReductionPenalty` helper and its two call sites
- Volume floor block in `perturbCell` (used `_birthVolume`, no longer exists)
- `sizeReductionWeight` parameter on `perturbCell`

**Removed from `C++/src/CellUniverse.cpp`:**
- Radius EMA loop (~45 lines, lines 956–998 of prior version)
- Radius gradient loop (~38 lines, lines 1004–1037 of prior version)
- `sizeReductionWeight` local and all 3 callsites

**Removed from YAML (`C++/config/config.yaml`):**
- aRadius/bRadius/cRadius perturbation blocks (radii are PCA-driven now)
- brightness perturbation block (brightness is EMA-driven from real image)
- `radiusUpdateBlend`, `radiusMeasurementBrightnessThreshold`
- `radiusGradientStepSize`, `radiusGradientIterations`
- `pcaShapeBlend`, `pcaShapeScanFactor`
- `size_reduction_penalty_weight`
- `min_frames_before_split`
- `aRadiusProbabilityStep/Trust`, `bRadius…`, `cRadius…` (adaptive-perturb constants, unused with PCA shape fit)

### Iterative PCA shape fit (NEW)

**File:** `C++/src/Frame.cpp` — `Frame::calibrateCellShapeViaPca` rewritten from single-shot to iterative.

**Signature (Frame.hpp):**
```cpp
bool calibrateCellShapeViaPca(
    size_t cellIndex,
    const ClaimSet &otherCellsClaimSets,
    int   maxIters,
    float radiusScale,
    int   minPixels,
    float maskScale,
    float convergeRadius,
    float convergeAngleDeg,
    bool  updatePosition,
    float maxPosShiftFraction);
```

**Algorithm per iteration:**
1. Gather bright pixels in sphere `maskScale * maxR` around current cell center, Voronoi-filtered against other cells.
2. Tighten to `maskScale * current ellipsoid` (local-frame ellipsoid test). This rejects halo pixels off the cell's current belief.
3. Stop if pixel count < `minPixels`.
4. Weighted centroid + 3×3 covariance in world frame.
5. `cv::eigen(cov, eigvals, eigvecs)` — rows sorted descending.
6. Detect degeneracy: if `maxVar/minVar < 1.1`, the cell is near-spherical; skip rotation update (axes arbitrary).
7. Greedy-match PCA axes to current a/b/c axes by `|dot product|`. Flip matched signs to preserve axis identity. Compute max axis rotation angle.
8. Build R from matched axes as columns; enforce det=+1 by flipping 3rd axis if negative. Decompose to Euler (R = Rz·Ry·Rx).
9. Target radii = `radiusScale * sqrt(eigenvalue)` in matched order.
10. Position update: shift toward centroid capped at `maxPosShiftFraction * maxR`.
11. Apply directly (no EMA). Reconstruct cell via `EllipsoidParams` to carry the new position.
12. Convergence: stop when max radius change < `convergeRadius` AND max axis angle < `convergeAngleDeg`.

**YAML config (defaults):**
```yaml
pcaShapeMaxIters: 15
pcaShapeRadiusScale: 2.236       # sqrt(5): uniform-ellipsoid variance
pcaShapeMinPixels: 50
pcaShapeMaskScale: 1.3
pcaShapeConvergeRadius: 0.3
pcaShapeConvergeAngleDeg: 2.0
pcaShapeUpdatePosition: true
pcaShapeMaxPosShiftFraction: 0.5
```

### CellUniverse::optimize integration

The shape calibration pass replaces the removed EMA/gradient loops. It runs after position calibration and before the pre-pass, once per cell:

```cpp
frame.calibrateCellShapeViaPca(ci, others,
    pcaMaxIters, pcaScale, pcaMin,
    maskScale, convR, convAng,
    updatePos, posShiftCap);
```

### Diagnostic logs

- `[PCA Shape] frame N cells=M maxIters=K scale=S minPixels=P maskScale=X updatePos=B`
- `[PCA Shape] cell=... iter=i n=N degen=0/1 R=(a,b,c) dR=Δ axisAng=deg posShift=S`
- `[PCA Shape] cell=... iter=i stop_too_few pixels=N min=P`

### Effect

- Shape + rotation + position converge per frame; cells cannot drift or inflate because PCA reads actual bright-pixel variance.
- Thin diagonal streaks now elongate to match within ~3 iters instead of ~10 frames.
- Near-spherical cells skip rotation update (no axis-ID jitter).
- Source tree ~300 lines lighter; config file dropped 8 dead keys + 4 dead perturb blocks.
