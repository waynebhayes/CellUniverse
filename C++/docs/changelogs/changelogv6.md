# Changelog v6

Opened 2026-04-13 (late evening). Covers the iterative PCA shape-fit work after v5 closed.

---

## 2026-04-13 (late evening patch 2): Fix PCA oscillation — rank assignment

**Status:** ACTIVE

**Problem:** On the first 3-frame run after the iterative PCA rewrite, `[PCA Shape]` logs showed period-3 oscillation through all 15 iters:

```
e3d03 iter 2: R=(19.15, 16.25, 24.78)
e3d03 iter 3: R=(23.58, 19.20, 16.42)
e3d03 iter 4: R=(19.42, 16.61, 25.75)
```

Symptoms: `dR` never dropped below the convergence threshold; `axisAng` stayed 35-50° iter after iter; eigenvalues cycled through slot labels.

**Root cause:** greedy axis-matching by `|current_axis · pca_axis|` chose the largest-absolute-dot PCA axis for each slot in slot order (a → b → c). For cells where the current orientation was close to 45° off the true axes, small perturbations in the mask changed dot-product magnitudes just enough to reassign pca axis ids between slots each iteration. Radii followed the reassignment, so slot a could hold the largest eigenvalue one iter and the smallest the next.

**Fix:** `C++/src/Frame.cpp` (inside `Frame::calibrateCellShapeViaPca`) — replaced greedy matching with strict eigenvalue-rank assignment.

**Before:**
```cpp
// Greedy-match PCA axes to current a/b/c by largest |dot|.
int pcaForCur[3] = {-1, -1, -1};
bool pcaUsed[3] = {false, false, false};
for (int ci = 0; ci < 3; ++ci) {
    int bestIdx = -1;
    double bestAbs = -1.0;
    for (int pj = 0; pj < 3; ++pj) {
        if (pcaUsed[pj]) continue;
        const double d = curAxis[ci].x * pcaAxis[pj].x +
                         curAxis[ci].y * pcaAxis[pj].y +
                         curAxis[ci].z * pcaAxis[pj].z;
        if (std::abs(d) > bestAbs) { bestAbs = std::abs(d); bestIdx = pj; }
    }
    pcaForCur[ci] = bestIdx;
    pcaUsed[bestIdx] = true;
}

cv::Point3f matchedAxis[3];
double matchedVariance[3];
double maxAxisAngle = 0.0;
for (int ci = 0; ci < 3; ++ci) {
    const int pj = pcaForCur[ci];
    cv::Point3f v = pcaAxis[pj];
    const double dot = curAxis[ci].x * v.x + curAxis[ci].y * v.y + curAxis[ci].z * v.z;
    if (dot < 0.0) { v.x = -v.x; v.y = -v.y; v.z = -v.z; }
    matchedAxis[ci] = v;
    matchedVariance[ci] = pcaVariance[pj];
    const double a = std::acos(std::clamp(std::abs(dot), 0.0, 1.0));
    if (a > maxAxisAngle) maxAxisAngle = a;
}
```

**After:**
```cpp
// Strict eigenvalue-rank assignment: a = largest variance, b = middle,
// c = smallest. Greedy |dot| matching oscillated when eigenvalue
// ordering differed from current slot labeling, because matched
// variances cycled between slots each iteration. Rank assignment is
// stable — physical variance ranks are invariant to the label rotation.
cv::Point3f matchedAxis[3];
double matchedVariance[3];
double maxAxisAngle = 0.0;
for (int ci = 0; ci < 3; ++ci) {
    cv::Point3f v = pcaAxis[ci];
    // Sign-align with current slot direction for rotation continuity.
    const double dot = curAxis[ci].x * v.x + curAxis[ci].y * v.y + curAxis[ci].z * v.z;
    if (dot < 0.0) { v.x = -v.x; v.y = -v.y; v.z = -v.z; }
    matchedAxis[ci] = v;
    matchedVariance[ci] = pcaVariance[ci];
    const double ang = std::acos(std::clamp(std::abs(dot), 0.0, 1.0));
    if (ang > maxAxisAngle) maxAxisAngle = ang;
}
```

**Effect:**
- Iteration should now converge in 3-6 iters as the radii stabilize at the rank-ordered eigenvalues.
- The `axisAng` metric still measures rotation disagreement and drives the angle convergence gate.
- Slot identity (a/b/c) is determined solely by eigenvalue rank — physical tracking of a cell across frames is via rotation + radii, not slot labels.

**Open observations from the same 3-frame run** (may be downstream of oscillation; re-run after this fix):
- 1 false split: e3d03 accepted at f2 with `data_axA_primary`, costDiff=-40.28.
- 1 missed split: 12345 rejected at f3, costDiff=+4.36 (split made cost slightly worse).
- Both cells were showing chaotic eigenvalue shuffling, so stable shape may remove the spurious split geometries and give 12345 a cleaner split signal.

---

## 2026-04-14: Skip PCA shape fit for pre-classified cells (Fix A)

**Status:** ACTIVE

**Problem:** At frame 8, cell 1f89ab's ground-truth split was missed. Root cause traced to the iterative PCA shape fit collapsing onto one of the two emerging daughters:

- End of f7 (perfect fit):  `R=(50.0, 39.9, 18.7)` on a tilted plate.
- Start of f8 PCA fit:  `R=(48.9, 31.7, 15.7)` → iter 14: `R=(42.7, 30.1, 12.25)`.
- Pixel mask `n` dropped from 186k (iter 0) to 122k (iter 14) — the fit concentrated on one brighter daughter lobe instead of both.
- `liveCost` = 3056 became artificially low (one ellipsoid fits one daughter almost perfectly in L2).
- Split attempt with correctly-sized daughters (built from `snapshot.aRadius/bRadius/cRadius`, not live) produced total cost 3162. Cost diff = +106 vs live baseline → rejected.

The collapse happens when a cell has two bright lobes (two emerging daughters). Weighted centroid + ellipsoid-mask feedback latches onto the brighter lobe.

**Fix:** In `CellUniverse::optimize`, skip `calibrateCellShapeViaPca` for cells whose name is in `preClassifiedNames`. The snapshot shape from end-of-previous-frame is the authoritative parent state for splitting cells — it must not be re-fit, because the bright region now contains two daughters and any single-ellipsoid fit is wrong.

**File:** `C++/src/CellUniverse.cpp`

**Before** (inside the PCA shape-fit loop):
```cpp
for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
    const std::string sname = frame.cells[ci].getName();
    Frame::ClaimSet others = buildShapeClaimSet(sname);
    frame.calibrateCellShapeViaPca(ci, others,
                                   pcaMaxIters, pcaScale, pcaMin,
                                   maskScale, convR, convAng,
                                   updatePos, posShiftCap);
}
frame.regenerateSynthFrame();
```

**After:**
```cpp
int skippedPre = 0;
for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
    const std::string sname = frame.cells[ci].getName();
    if (preClassifiedNames.count(sname)) {
        std::cout << "  [PCA Shape] cell=" << sname
                  << " skipped pre_classified" << std::endl;
        ++skippedPre;
        continue;
    }
    Frame::ClaimSet others = buildShapeClaimSet(sname);
    frame.calibrateCellShapeViaPca(ci, others,
                                   pcaMaxIters, pcaScale, pcaMin,
                                   maskScale, convR, convAng,
                                   updatePos, posShiftCap);
}
if (skippedPre > 0) {
    std::cout << "[PCA Shape] frame " << displayFrame
              << " skipped_pre_classified=" << skippedPre << std::endl;
}
frame.regenerateSynthFrame();
```

### Elongation data for classification threshold

Collected from the 22-frame run (snapshot elongation at end of frame preceding each GT split):

| Cell | GT split | Snapshot elong |
|---|---|---|
| e9077 | f3 | 1.28 |
| 12345 | f3 | 1.36 |
| 1f89ab | f8 | 3.49 |
| 1f2ed | f11 | 3.11 |
| e9077a50 | f19 | 2.84 |
| 12345..0 | f20 | 1.50 |
| 12345..1 | f20 | 2.37 |
| e9077a51 | f20 | 2.19 |
| e3d03 (never splits) | max observed | 2.08 |
| 8cbdf (never splits) | max observed | 1.49 |

No threshold cleanly separates splitting from non-splitting cells. Current config value `shape_elongation_classify_threshold = 1.3` (bumped from 1.15) catches everything except e9077 at f3 (elong 1.28). e9077's first-generation division is a small spherical cell with weak pre-division elongation — trade-off accepted.

### Expected effect

1. For 1f89ab at f8: snapshot shape preserved as `R=(50, 39.9, 18.7)`, tilted-plate orientation intact. Pre-pass PCA on bright pixels, normalized by the correct `(1/a, 1/b, 1/c)`, should yield a `splitAxisDir` closer to perpendicular-to-plate (short axis) rather than the current pure-Z direction biased by the too-small `c`. Daughters placed in their correct positions; split should pass the cost gate.
2. For 1f2ed, e9077a50, 12345..*, e9077a51: same mechanism — preserved snapshot shape should give the split attempt a fair baseline.
3. Non-splitting cells (8cbdf, e3d03) continue to get the full PCA shape fit.
4. e9077 at f3 (elong 1.28 < 1.3) falls into `non_classified` — gets PCA shape fit. For first-gen sphere-to-split cells this is less risky because the pre-division shape is near-spherical (no single-dominant-daughter lobe yet).

### Known limitation

If the threshold is too strict (e.g., 1.5), first-generation splits (e9077, 12345) will be mis-classified as non-splitting and get shape-fit → same 1f89ab-style collapse. 1.3 is a compromise. Better predictors than elongation (e.g., bright-valley detection inside the parent) would enable cleaner classification.

---

## 2026-04-14: Revert Fix A, add snap-shrink floor (Fix C), collapse Phase A/B

**Status:** ACTIVE. Supersedes Fix A.

**Why Fix A had to be reverted:** new 4-frame run showed e9077's f3 split (previously accepted in run 222850 with costDiff=-59.8) now rejected by `asymmetric_edges`. Root cause: skipping the shape fit preserved a near-isotropic snap shape (`R≈(38.7, 30.3, 27.8)`), which muddled the pre-pass local-frame PCA normalization and placed daughter seeds incorrectly. The old run had accidentally benefited from PCA elongating `a=50`, which made local-frame normalization strongly anisotropic and produced correct split seeds.

### Change 1 — Revert Fix A

**File:** `C++/src/CellUniverse.cpp`

Removed the `preClassifiedNames.count(sname)` skip branch from the PCA shape-fit loop. All cells now run shape fit again.

### Change 2 — Snap-shrink floor (Fix C)

**Files:** `C++/src/Frame.cpp`, `C++/includes/Frame.hpp`, `C++/src/CellUniverse.cpp`

Added three optional parameters to `Frame::calibrateCellShapeViaPca` — `snapMinRA`, `snapMinRB`, `snapMinRC`. When each is > 0, the per-iteration `targetR_i` is floored at `snapMinR_i`. This prevents the mask-feedback collapse that was shrinking cells onto one emerging daughter (1f89ab f8 pattern).

`CellUniverse::optimize` passes `0.9 × snapshot.aRadius/bRadius/cRadius` as the floor. A cell can still grow, tilt, and refine shape — it just can't drop below 90% of its previous-frame radii along any axis. The `[PCA Shape]` log now includes a `floored=ABC` triple showing which axes hit the floor.

**Excerpt (`Frame.cpp`):**
```cpp
float targetA = radiusScale * std::sqrt((float)matchedVariance[0]);
float targetB = radiusScale * std::sqrt((float)matchedVariance[1]);
float targetC = radiusScale * std::sqrt((float)matchedVariance[2]);

bool flooredA = false, flooredB = false, flooredC = false;
if (snapMinRA > 0.0f && targetA < snapMinRA) { targetA = snapMinRA; flooredA = true; }
if (snapMinRB > 0.0f && targetB < snapMinRB) { targetB = snapMinRB; flooredB = true; }
if (snapMinRC > 0.0f && targetC < snapMinRC) { targetC = snapMinRC; flooredC = true; }
```

**Call site (`CellUniverse.cpp`):**
```cpp
constexpr float kSnapShrinkFloor = 0.9f;
float snapMinA = 0.0f, snapMinB = 0.0f, snapMinC = 0.0f;
auto snapIt = previousSnapshots.find(sname);
if (snapIt != previousSnapshots.end() && snapIt->second.valid) {
    snapMinA = kSnapShrinkFloor * snapIt->second.aRadius;
    snapMinB = kSnapShrinkFloor * (snapIt->second.bRadius > 1e-3f
                                   ? snapIt->second.bRadius
                                   : snapIt->second.aRadius);
    snapMinC = kSnapShrinkFloor * snapIt->second.cRadius;
}
frame.calibrateCellShapeViaPca(ci, others,
                               pcaMaxIters, pcaScale, pcaMin,
                               maskScale, convR, convAng,
                               updatePos, posShiftCap,
                               snapMinA, snapMinB, snapMinC);
```

### Change 3 — Collapse Phase A / Phase B into a single unified loop

**File:** `C++/src/CellUniverse.cpp`

Validation of the 222850 run showed **no split-acceptance difference between the first and second halves of a frame's attempts** (early 91/3 ≈ 3.3% vs late 96/3 ≈ 3.1%). Phase A/B ordering was bought no benefit. Position calibration already settles all cells upfront.

Replaced the two-call sequence:
```cpp
runPhase(nonClassifiedNames, /*phaseB*/ false, ...);
runPhase(preClassifiedNames, /*phaseB*/ true, ...);
```
with a single unified call over the union of both sets, using `phaseB=true` claim-set semantics (live positions, expected-daughters for pre-classified neighbors). The log stanza now prints `Frame N iter=i ...` without the A/B label.

### Validation target for the next run

- `[PCA Shape]` logs should show `floored=` triples engaging on cells like 1f89ab at f8 (where the fit previously collapsed). Verify `c` stays at floor (≥ 0.9 × snap.cRadius), not below.
- Confirm `e9077` and `12345` at f3 still split (regressed by Fix A).
- Confirm 1f89ab at f8, 1f2ed at f11, e9077a50 at f19, 12345..{0,1} + e9077a51 at f20 now split, or at minimum produce cleaner bridge/cost signals.
- `[Split Bridge] valleyRatio` becomes the key diagnostic for whether Fix D (image-evidence-based acceptance) should be pursued next.

---

## 2026-04-14: Asymmetric L2 cost (Fix E)

**Status:** ACTIVE

**Problem:** Symmetric L2 cost can't reliably decide between "one big parent covering two daughters + dark valley" and "two daughters covering only bright blobs." The parent's fit to the valley is penalized as (synth_bright − real_dim)² per voxel, which squares small differences down — so the valley penalty is often modest relative to the daughter-placement-error penalty when daughters are slightly misplaced. Result: the split cost gate is sensitive to parent fit quality and daughter seed accuracy, both of which we can't guarantee.

**Fix:** Per-voxel squared error gets multiplied by `k` (default 3.0) when `synth > real` (cell overshoots into a dim image region). When `synth < real` (cell undershoots a bright region), the unweighted squared error is used. This amplifies the "parent covers dark valley" penalty linearly while leaving "daughter misses bright region" unchanged, so splits that cover only bright regions reliably win the cost comparison.

**Files changed:**

### `C++/src/Frame.cpp`

Added static helper:
```cpp
static double asymmetricL2Slice(const cv::Mat &real, const cv::Mat &synth, float k)
{
    if (k <= 1.0f + 1e-6f) {
        return cv::norm(real, synth, cv::NORM_L2);
    }
    CV_Assert(real.type() == CV_32F && synth.type() == CV_32F);
    const int rows = real.rows;
    const int cols = real.cols;
    double sumSq = 0.0;
    for (int y = 0; y < rows; ++y) {
        const float *rR = real.ptr<float>(y);
        const float *sR = synth.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            const float diff = sR[x] - rR[x];
            const float sq = diff * diff;
            sumSq += (diff > 0.0f) ? (k * sq) : sq;
        }
    }
    return std::sqrt(sumSq);
}
```

Replaced every `cv::norm(real, synth, cv::NORM_L2)` call in the cost path with `asymmetricL2Slice(real, synth, simulationConfig.asymmetric_cost_weight)`:
- `Frame::refreshFullCostCache` (full-frame cache rebuild)
- `Frame::calculateIncrementalCost` (incremental per-slice recompute)
- `Frame::calculateCost` (non-cached fallback)

With `k=1.0` the helper short-circuits to `cv::norm`, preserving bit-identical behavior when disabled.

### `C++/includes/ConfigTypes.hpp`

Added `SimulationConfig::asymmetric_cost_weight` (default 1.0) + YAML parse in `explodeConfig`.

### `C++/config/config.yaml`

```yaml
  asymmetric_cost_weight: 3.0
```

Under the `simulation:` block. Documented inline.

### Effect expected

- For a cell where PCA shape fit overshoots into dark space: shape fit pays more → converges to a tighter, more accurate size.
- For split attempts: the single-parent-ellipsoid baseline pays extra for covering the valley between two daughters, so the two-daughter candidate consistently wins the cost comparison by a larger margin. `[Split Reject cost]` should fire less often for real splits.
- For wrong daughter placement (one daughter in empty space): still penalized correctly — a daughter ellipsoid covering dim real pixels overshoots, so it pays the same `k×` penalty as a parent would. Fix doesn't create false-positive splits.

### Interaction with Fix C (snap-shrink floor)

- Fix C keeps radii from collapsing inward (mask-feedback prevention).
- Fix E keeps radii from inflating outward into dark regions (overshoot penalty).
- Together: the shape fit converges to a ring around the actual bright extent in both directions.

### Tuning

- `k = 3.0` starting point. If shape fit shrinks cells too aggressively, lower to 2.0. If parent-covers-valley still wins split comparisons, raise to 4.0.
- Log diagnostic: split-attempt `costDiff` magnitude should grow (more negative for real splits, more positive for false splits).

### Validation targets

- `e9077` and `12345` at f3 split with negative costDiff.
- `1f89ab` at f8, `1f2ed` at f11, `e9077a50` at f19, `12345..{0,1}` + `e9077a51` at f20 all split.
- `e3d03` and `8cbdf` remain rejected throughout 45 frames.
- No new false splits.

---

## 2026-04-14: Config cleanup — remove dead fields

**Status:** ACTIVE

**Removed (0 non-definition references in src/):**
- `split_direction_agreement_degrees` (ProbabilityConfig + YAML). Was part of the old dual-direction-gating logic; now both PCA and snapshot directions are always tried, so the agreement angle is never read.
- `bio_max_midpoint_parent_fraction` (ProbabilityConfig + YAML). The midpoint-near-parent gate was removed when pre-pass grounding took over; config field was stale.

**Files changed:**
- `C++/includes/ConfigTypes.hpp` — deleted field declarations and `explodeConfig` parse calls.
- `C++/config/config.yaml` — deleted the two YAML keys.

---

## 2026-04-14: Fix E perf — SIMD-ify asymmetric cost helper

**Status:** ACTIVE

**Problem:** First run with `asymmetric_cost_weight=3.0` took ~4 hours for 22 frames vs the old run's ~1 hour. Root cause: `asymmetricL2Slice` was a scalar per-pixel loop. `cv::norm(..., NORM_L2)` uses SIMD (AVX/NEON) for 4-8× speedup; the scalar replacement lost that.

**Fix:** rewrote using SIMD-optimized OpenCV mat ops. Identity:
```
sum(w*diff²) where w = k if diff>0 else 1
= sum(diff²) + (k-1) * sum(diff² where diff>0)
```

**File:** `C++/src/Frame.cpp` — `asymmetricL2Slice` body replaced:

```cpp
cv::Mat diff;
cv::subtract(synth, real, diff);
cv::Mat diffSq;
cv::multiply(diff, diff, diffSq);
const double sumSq = cv::sum(diffSq)[0];

cv::Mat posMask;
cv::compare(diff, 0.0f, posMask, cv::CMP_GT);   // 8U mask

cv::Mat posSq = cv::Mat::zeros(diffSq.size(), diffSq.type());
diffSq.copyTo(posSq, posMask);                  // only overshoot pixels
const double posSumSq = cv::sum(posSq)[0];

const double asymSumSq = sumSq + (double)(k - 1.0f) * posSumSq;
return std::sqrt(std::max(0.0, asymSumSq));
```

Behavior identical to the scalar version at float precision; runtime should drop back to ~1× cv::norm.

---

## 2026-04-14 (late): Loop perf + daughter-shape fixes

**Status:** ACTIVE

### Change 1 — Unified loop optimizations (`CellUniverse.cpp`)

Several per-iteration redundancies removed from `runPhase`:

1. **Eligibility rebuild cached.** Was scanning `frame.cells` and doing `phaseNames.count()` on each iteration (3000 iters/frame × 8 cells × O(log n) set lookup = ~80k set ops/frame for nothing). Now `eligible` vector is built once per frame and only rebuilt after a split accept (cells list grew). Rejected splits leave cells in place, so no maintenance needed.
2. **Removed dead empty `if (splitBlacklist.count(cname) > 0) { }` block.**
3. **String copies → const refs.** `const std::string cname = ...` was copying every iteration; now `const std::string &cname`.
4. **Removed `std::find_if` after rejected split.** Was scanning cells to re-find the parent by name even though `cellIdx` was still valid (trySplitCellPhased reverts in place on rejection). Direct `perturbCell(cellIdx, ...)` now.

Estimated savings: small per-iter but multiplied by 3000 iters × 22 frames.

### Change 2 — Fix C revised: per-axis floor → volume floor (`Frame.cpp`, `Frame.hpp`, `CellUniverse.cpp`)

**Problem:** Screenshot after the Fix C+E run showed misshapen daughters — elongated rods, pancakes inherited from parent — that should have been near-spherical. Root cause: daughters inherit `cbrt(0.5) × parent_radii` per axis (preserving parent's aspect ratio), and the per-axis Fix C floor at `0.9 × snap_radius` prevented the next-frame PCA shape fit from ever reshaping the aspect ratio. Axes could only grow, not shrink.

**Fix:** replace the per-axis floor with a single volume floor. If `a × b × c < 0.9 × snap_volume`, rescale all three radii uniformly with `cbrt(snap_volume / vol)` so the total volume reaches the floor. Aspect ratio is unconstrained — a pancake daughter can reshape into a sphere, a sphere into a pancake, anything in between, as long as total volume stays within 10% of snap.

This still blocks the mask-feedback collapse (total volume can't shrink below 0.9 × snap) but doesn't pin aspect ratio.

**Signature change:**
```cpp
// Before
bool calibrateCellShapeViaPca(... float snapMinRA, float snapMinRB, float snapMinRC);

// After
bool calibrateCellShapeViaPca(... float snapMinVolume);
```

**Implementation:**
```cpp
bool volFloored = false;
if (snapMinVolume > 0.0f) {
    const float vol = targetA * targetB * targetC;
    if (vol > 0.0f && vol < snapMinVolume) {
        const float s = std::cbrt(snapMinVolume / vol);
        targetA *= s; targetB *= s; targetC *= s;
        volFloored = true;
    }
}
```

**Call-site (CellUniverse.cpp):**
```cpp
constexpr float kSnapVolumeFloor = 0.9f;
float snapMinVolume = kSnapVolumeFloor * sA * sB * sC;   // from snapshot
frame.calibrateCellShapeViaPca(ci, others, ..., snapMinVolume);
```

Log `[PCA Shape]` now includes `volFlr=0/1` instead of the per-axis `floored=ABC`.

### Expected effect

- Daughter cells can reshape to match actual image geometry on the frame AFTER birth — no more pancake daughters locked in parent's aspect.
- Pre-pass PCA (which uses a, b, c for local-frame normalization) will get correct shape → correct split axis direction.
- The "2 xy + 1 z" expected pattern for f20 splits should materialize (currently got 1 xy + 1 z + 2 missed).
- 12345..0 and 12345..1 at f20 missed splits should recover.

---

## 2026-04-14: Replace volume floor with fixed snap-mask (Fix C v3)

**Status:** ACTIVE. Supersedes per-axis floor AND volume floor.

**Problem with prior floors:** both per-axis and volume variants blocked legitimate shrinkage. Run 033245 logs showed 1f89ab at f7 fitted to R=(48.9, 36.0, 29.2) with b,c hitting the per-axis floor. The cell was pre-splitting (divides at f8) — its real extent may be smaller than the 0.9×snap floor allows. Floor created oversized appearance; removing the floor brings back the collapse-onto-one-daughter behavior.

**Diagnosis:** the collapse was caused by the mask tightening with shrinking fitted radii. Each iteration computed the sphere + ellipsoid mask from the *live* (current-iter) radii, so smaller fit → smaller mask → fewer bright pixels visible → PCA sees only the near region → smaller fit. Runaway feedback loop.

**Fix:** compute the mask from **snapshot radii** (previous-frame fit, fixed for the whole frame) instead of live. The mask stays constant throughout iteration; PCA always sees the full bright cloud. Radii are free to converge to whatever the image actually says — no floor needed.

### Files changed

- `C++/includes/Frame.hpp` — `calibrateCellShapeViaPca` signature: `snapMinVolume` → `maskA, maskB, maskC`.
- `C++/src/Frame.cpp` — mask computed once per call from `effMask{A,B,C}` (snap values, or live as fallback); volume floor block removed; log `volFlr=` dropped.
- `C++/src/CellUniverse.cpp` — call site passes `snap.{aRadius,bRadius,cRadius}`.

### Implementation detail

```cpp
const float effMaskA = (maskA > 0.0f) ? maskA : cell.getARadius();
const float effMaskB = (maskB > 0.0f) ? maskB : cell.getBRadius();
const float effMaskC = (maskC > 0.0f) ? maskC : cell.getCRadius();
const float maskMaxR = std::max({effMaskA, effMaskB, effMaskC});
const float sphereR = maskScale * maskMaxR;          // fixed sphere radius
const double invA2Fixed = 1.0/(maskScale*effMaskA*maskScale*effMaskA);
// ... same for B, C

for (int iter = 0; iter < maxIters; ++iter) {
    // gather pixels with FIXED sphereR, invA2Fixed, etc.
    // ...
}
```

### Why this works

- Pre-splitting cell (e.g., 1f89ab f8): mask sized for snapshot (covers both emerging daughters and valley). PCA sees the full elongated bright cloud → correctly elongated radii; doesn't collapse onto one daughter.
- Non-splitting cell with real shrinkage: mask sized for snap. If PCA says radii should be smaller, radii shrink freely — no floor blocking. Mask stays bigger than necessary but that just means extra safe margin; PCA result is the same.
- First-frame cells (no snap): mask falls back to live radii.

### Expected visual effect

Both the collapse-onto-daughter pattern AND the oversized-by-floor pattern should disappear.

### Remaining concerns

- If a cell drifts far between frames, snap-mask may no longer cover the real bright region. Position calibration already runs before shape fit to handle this — should be fine.
- If cell's real shape changes dramatically (unlikely between frames), mask may be oriented wrong. Rotation still uses live orientation, so partial adaptation; full adaptation across 1-2 frames.

---

## 2026-04-14: Bridge gate two-tier + asymmetric cost bumped to 6

**Status:** ACTIVE

**Problem from run 144944:**
- False split on 8cbdf at f3 accepted with costDiff = −22.6, despite bridge showing **valleyRatio = 0.998** (gap brightness ≈ edge brightness — unambiguous "no valley" signal). Reason: bridge rejection rule was `densityFlat AND brightnessFlat`; gapDensity = 0.176 missed the 0.18 threshold by 0.004, so AND was false. Flat valley alone couldn't fire.
- Missed split on e9077 at f3 with costDiff = +5 to +15, despite `valleyRatio = 0.556` (textbook real valley). Asymmetric cost `k=3` didn't amplify parent's valley-coverage penalty enough to flip the sign.

**Root causes:**
1. Bridge AND rule too permissive for pure no-valley cases.
2. Asymmetric cost `k=3` only adds ~30-60 extra penalty for valley coverage over ~5000 voxels — close to flipping but not enough when parent shape is correct (with snap-mask, baseline is tight).

### Change 1 — Bridge gate: two-tier reject

**File:** `C++/src/Frame.cpp`

Added a strong-signal tier: if `valleyRatio ≥ 0.95` alone, reject — no need to wait for density to co-fire. Keeps the existing AND tier for subtler cases.

**Before:**
```cpp
if (densityFlat && brightnessFlat && edgeCount > 0) reject;
```

**After:**
```cpp
constexpr float kNoValleyHardThreshold = 0.95f;
const bool noValleyAtAll = (valleyRatio >= kNoValleyHardThreshold);
if (edgeCount > 0 && (noValleyAtAll || (densityFlat && brightnessFlat))) reject;
```

Log adds `noValleyTier=0/1` so it's visible which tier fired.

### Change 2 — Raise `asymmetric_cost_weight` 3.0 → 6.0

**File:** `C++/config/config.yaml`

Doubles the per-voxel amplification on parent's valley coverage. For e9077's ~5000 valley voxels at diff ≈ 0.25: expected extra penalty grows from ~30-60 to ~100-150 — enough to flip e9077's costDiff from +5..+15 to negative.

### Expected effects on next run

| Cell | Frame | Current behavior | Expected after fix |
|---|---|---|---|
| 8cbdf | f3 | ❌ false accept (costDiff=−22, vr=0.998) | ✅ rejected by tier-1 bridge (vr ≥ 0.95) |
| 8cbdf…0 | f4 | ❌ cascading false (vr=0.814) | still possible — tier-1 doesn't fire here, relies on cost |
| e9077 | f3 | ❌ missed (costDiff=+5..+15, vr=0.556) | ✅ cost should flip negative with k=6 |
| 12345 | f3 | ✅ accepted | ✅ still accepted (stronger negative) |
| 1f89ab | f8 | ❌ missed | likely ✅ (vr=0.54 in prior run; k=6 amplifies further) |
| 1f2ed | f11 | ✅ accepted (old runs) | ✅ |

### Tuning notes

- If k=6 still leaves true splits borderline, try k=8 or k=10.
- If k=6 produces new false splits (too aggressive valley amplification), drop to k=5.
- Bridge tier-1 threshold 0.95 is conservative — only fires when valley is brighter than 95% of edge brightness. Very rare for real splits (typical real vr is 0.3-0.7). Safe.

---

## 2026-04-14: Pre-pass for ALL cells + image-PCA direction as extra candidate axis

**Status:** ACTIVE

**Problem from run 155618:**
- e9077 at f3 snap elong = 1.23 → below threshold 1.3 → **non-classified** → no pre-pass ran → split-axis direction came from snap's `worldSplitAxis()` which is arbitrary for near-round snaps.
- Split direction logged as `splitAxisDir=(-0.73, 0.17, 0.66)` (mostly X), whereas old run 222850 with pre-pass got `(0.09, 0.20, 0.97)` (nearly pure Z, which is correct — real daughters separate in Z).
- Wrong direction → seeds placed in wrong direction → daughters drifted 29-36 voxels during refinement → high cost → rejected.

**User diagnosis (correct):** At f3 the cell has already started dividing, so the live shape fit is confused between "one elongated blob" and "one daughter". The snapshot (from end of f2, when cell was still one unit) gives correct parent position + volume. But snap *direction* can be unreliable for cells with low historical elongation. The PCA of current-frame bright pixels is the only way to find the midpoint and direction of the two emerging blobs. Previously classification-gated — needs to run for every cell.

### Change 1 — Pre-pass runs for every cell

**File:** `C++/src/CellUniverse.cpp`

`expectedDaughters` now seeded for every cell with a valid snapshot (was: only `preClassifiedNames`). Pre-pass iteration loop also iterates all cells (via `frame.cells` instead of `preClassifiedNames`). Log tag updated to `allCells=M` instead of `preClassified=M`.

### Change 2 — Split attempt gets image-PCA direction + length from snap override

**File:** `C++/src/CellUniverse.cpp`, split attempt block.

```cpp
// Before: isPre gated the override.
// After: always override splitSnapshot.splitAxisDir and .splitAxisLength
//   with pre-pass D1/D2 values. POSITION preserved as true snap center
//   so trySplitCellPhased's candidate generator can try both midpoints.
```

`useSnapDir` flag hardcoded to `true` (it's only a log-tag anyway).

### Change 3 — Add imgPca direction as extra primary axis in trySplitCellPhased

**File:** `C++/src/Frame.cpp`

After the existing parent-rotation axis extraction (axA/axB/axC filtered to those within 20% of minR), if `useSnapshotDirection && snapshotValid && snapshot.splitAxisDir` is non-degenerate AND differs from existing axes (|dot|<0.95), add it to `primaryDirs` with label `imgPca`.

Candidate generation then tries this direction alongside the parent-rotation axes. Each primary direction gets two midpoint variants (`data_*` = pixel-projection centroid, `snap_*` = snapshot.position = true snap center). So e9077 f3 would get candidates like:
- `data_axC_primary` — snap's axC direction + its pixel-centroid midpoint
- `data_imgPca_primary` — pre-pass direction + its pixel-centroid midpoint (the key new one)
- `snap_imgPca_primary` — pre-pass direction + true snap center midpoint
- Plus rot/trans variants of each

Cost picks the winner.

### Why this matches user's original design

User described the ideal fix: "try both snapshot midpoint and PCA frame midpoint, and choose whichever works better, with snapshot parent volume and size for daughter sizing."

Matches:
- Snap midpoint option: `snap_*` candidates use `splitSnapshot.position` = true snap center
- PCA midpoint option: `data_imgPca_*` candidates use pixel-projection midpoint on pre-pass direction (equivalent to pre-pass midpoint)
- Snap direction option: `*_axA/B/C_*` candidates use parent-rotation axes
- PCA direction option: `*_imgPca_*` candidates use pre-pass direction
- Daughter sizing: unchanged — uses `snapshot.aRadius/bRadius/cRadius × cbrt(0.5)`

### Expected effect

- e9077 f3: `data_imgPca_primary` should give Z-dominant direction (matching old run 222850 behavior), seeds at correct image-grounded positions, cost flips negative, split accepted.
- Similar gain for other cells with borderline snap elongation.
- 8cbdf false split still blocked by bridge tier-1 (vr=0.998).
- Redundant candidates kept — cost is the arbiter.

### Follow-up: classification is now almost unused

After this change, `preClassifiedNames` is used only to log membership (step [Classify]). The behavioral difference between pre-classified and non-classified cells is gone. Can remove in a cleanup pass if desired, but no rush — the bookkeeping is cheap.

---

## 2026-04-14: Remove classification + linear P(split) ramp

**Status:** ACTIVE

**User directive:** "Remove the classification completely and scale the split chance based on margin from low to high, 0.03 to 0.5."

### Change 1 — Delete classification

**Files:** `C++/src/CellUniverse.cpp`, `C++/includes/ConfigTypes.hpp`, `C++/config/config.yaml`

Removed:
- `preClassifiedNames`, `nonClassifiedNames` std::set<std::string>
- `[Classify] frame N pre=X non=Y T=Z` log line
- `T_classify` variable and `shape_elongation_classify_threshold` config field + YAML key
- All `isPre`, `isOtherPre` branches in claim-set construction
- Distinction between "pre_classified" (snapshot D1/D2 claim) and "non_classified" (live position) neighbors

The `buildOtherClaimSet` lambda simplified: every neighbor contributes its pre-pass D1/D2 seed if available, falls back to live position otherwise. Cells that split-accepted this frame contribute their daughters' live positions; rejected-split cells contribute their live parent position.

### Change 2 — Linear P(split) ramp on snapshot elongation

**File:** `C++/src/CellUniverse.cpp`

Old formula (proportional rescale):
```cpp
rawP = P_split_base + max(0, 1 - 1/snapshot.shapeElongation);
probabilityScale = P_split_max / max(rawP across all cells);
P(split) = rawP * probabilityScale;
```

New formula (linear ramp):
```cpp
const float t = clamp((snapElong - 1.0f) / (2.0f - 1.0f), 0.0f, 1.0f);
P(split) = P_split_base + t * (P_split_max - P_split_base);
```

- Elong 1.0 → P = 0.03 (P_split_base)
- Elong 1.5 → P = 0.265
- Elong 2.0 → P = 0.5 (P_split_max)
- Elong > 2.0 → P = 0.5 (clamped)

Advantages:
- **Every cell** gets at least `P_split_base` chance, so first-gen splits with low-elong snaps still get attempted regularly.
- No classification threshold ambiguity — smooth ramp with elongation.
- No dependency on other cells' elongations (previous rescale coupled them).

### Why this works given the cost + bridge + bio gates

With split attempts happening more uniformly across cells, **rejection must rely on image evidence, not pre-filtering by elongation**:
- Bridge gate (tier-1: valleyRatio ≥ 0.95 alone) rejects "no valley" cells like 8cbdf.
- Bridge gate (tier-2: density AND brightness flat) rejects subtler flat profiles.
- Bio gates (volume fraction, daughter size ratio, buried check) reject geometrically invalid splits.
- Cost gate (with Fix E k=6 asymmetric) rejects splits where parent fits better than two daughters.

A cell that isn't actually splitting will fail at least one of these gates regardless of P(split) being rolled. A cell that IS splitting gets the full machinery with image-PCA direction available even at low snap elong.

### Split attempt integration unchanged from prior step

The pre-pass runs for every cell (already in place from 2026-04-14 earlier edit). `splitSnapshot.splitAxisDir/Length` overridden with pre-pass values; `splitSnapshot.position` preserved as true snap center; `imgPca` added as extra primary axis inside `trySplitCellPhased`. `useSnapDir = true` always.

### Expected behavior change

- e9077 f3: P(split) now ≈ 0.06 (elong 1.23, near base rate). Split attempts still fire a few times across 500×8 iterations. The pre-pass now produces correct Z direction → `data_imgPca_primary` candidate places daughters correctly → cost should flip negative → accept.
- 12345 f3: P(split) ≈ 0.08 (elong 1.26). Same path.
- 1f89ab f8: snap elong from end of f7 was ~3.5 → P(split) = 0.5 (clamped). Many attempts, pre-pass direction likely correct.
- 8cbdf (never splits): P(split) ≈ 0.06 per iter. Still attempts occasionally, but bridge tier-1 rejects on vr ≥ 0.95 and bio/cost gates block other variants.
- e3d03 (never splits): similar to 8cbdf.

### Follow-up cleanup opportunities

- `expected_daughter_pre_pass_iterations` still parsed from YAML; only 1 round is useful now that pre-pass runs for all cells. Can delete or keep as-is.
- `buildOtherClaimSet` still takes `bool phaseB` parameter; now ignored. Remove in next cleanup pass.
- `useSnapshotDirection` parameter of `trySplitCellPhased` always true. Can drop.

## 2026-04-14 (evening): Per-daughter valley gate + absolute edge-brightness gate, strip dead frozen-radii code

**Status:** ACTIVE

### Problem — false splits pass the bridge gate

Analysis of run `output_jihang_20260414_170618`:

| Split | edge1 | edge2 | gap | pooled valleyRatio | edgeAsym | Got | Should |
|---|---|---|---|---|---|---|---|
| 12345 f3 | 0.113 | 0.084 | 0.067 | 0.68 | 0.74 | accept | accept ✓ |
| e9077 f3 | 0.089 | 0.083 | 0.050 | 0.58 | 0.93 | accept | accept ✓ |
| a51 f5 (FALSE) | 0.059 | **0.029** | 0.037 | 0.72 | 0.49 | accept | reject |
| a511 f7 (FALSE, cascade) | 0.040 | **0.028** | 0.027 | 0.82 | 0.70 | accept | reject |

For a51 f5, gap (0.037) is **brighter** than edge2 (0.029) — daughter2 sits in near-background. The pooled `valleyRatio = gap / avg(edge1, edge2) = 0.72` hides this because the bright edge1 drags the average up. The `edgeAsymmetry < 0.4` gate misses it too (0.49 squeaks past).

The fundamental issue: averaging the two edges lets a phantom daughter on the dim side pass whenever the real daughter on the bright side is bright enough to mask it.

### Solution — per-daughter valley ratio + absolute minimum edge brightness

**Per-daughter valley ratio:** compute `valleyRatio1 = gap/edge1`, `valleyRatio2 = gap/edge2`, and gate on `max(valleyRatio1, valleyRatio2)`. A real split has valleys on BOTH sides (max < ~0.85). A phantom daughter has no valley on its side (max ≥ 1.0).

**Absolute edge brightness:** reject if `min(edge1Bright, edge2Bright) < 0.05`. Independent of ratios — a daughter centered near background (~0.0) is phantom regardless of how the other edge looks.

Applied to the 4 splits above:

| Split | valleyRatio1 | valleyRatio2 | worst | min(edge) | Verdict |
|---|---|---|---|---|---|
| 12345 f3 | 0.59 | 0.79 | 0.79 | 0.084 | accept ✓ |
| e9077 f3 | 0.56 | 0.60 | 0.60 | 0.083 | accept ✓ |
| a51 f5 | 0.62 | **1.26** | **1.26** | **0.029** | reject ✓ (both gates fire) |
| a511 f7 | 0.68 | **0.97** | **0.97** | **0.028** | reject ✓ (both gates fire) |

Clean separation: worst ratio and min edge brightness both cleanly discriminate real from phantom.

### Files changed

**File:** `C++/src/Frame.cpp`

**Lines 2327-2421 (after) — bridge gate section:**
```cpp
// Per-daughter valley ratios. Pooling edges (gap/edgeAvg)
// hides asymmetry — a bright real daughter averages with a
// dim phantom daughter and the gap still looks like a valley.
// Checking gap against EACH edge independently catches the
// phantom case: if gap >= edge on one side, that daughter is
// in near-empty space, not a real cell body.
const float valleyRatio1 = (edge1Bright > 1e-6f)
    ? (gapBright / edge1Bright)
    : 1.0f;
const float valleyRatio2 = (edge2Bright > 1e-6f)
    ? (gapBright / edge2Bright)
    : 1.0f;
const float worstValleyRatio = std::max(valleyRatio1, valleyRatio2);
// Pooled ratio kept for logging only (diagnostic continuity).
const float valleyRatio = (edgeBright > 1e-6f)
    ? (gapBright / edgeBright)
    : 0.0f;

std::cout << "  [Split Bridge] " << parentName
          << ...
          << " valleyRatio1=" << valleyRatio1
          << " valleyRatio2=" << valleyRatio2
          << " worstValleyRatio=" << worstValleyRatio
          << " valleyRatioPooled=" << valleyRatio
          << std::endl;

// Absolute edge-brightness gate
constexpr float kMinEdgeBrightAbsolute = 0.05f;
if (edge1Count > 0 && edge2Count > 0 &&
    std::min(edge1Bright, edge2Bright) < kMinEdgeBrightAbsolute) {
    std::cout << "[Split Reject bio] " << parentName
              << " reason=edge_too_dim" << ... << std::endl;
    restoreLiveParent();
    return {0.0, noop};
}

// Per-daughter valley gate
const bool densityFlat = gapDensity > probConfig.bio_bridge_max_gap_density;
const bool brightnessFlat = worstValleyRatio > probConfig.bio_bridge_max_valley_ratio;
constexpr float kNoValleyHardThreshold = 0.95f;
const bool noValleyAtAll = (worstValleyRatio >= kNoValleyHardThreshold);
if (edgeCount > 0 && (noValleyAtAll || (densityFlat && brightnessFlat))) {
    std::cout << "[Split Reject bio] " << parentName
              << " reason=bridge_flat" << ... << std::endl;
    restoreLiveParent();
    return {0.0, noop};
}
```

**Removed:** the separate `edgeAsymmetry < 0.4` rejection block. Its role is subsumed by the per-daughter valley gate (when one edge is much dimmer, its valleyRatio explodes).

### Dead code stripped — frozen radii plumbing

Radii are not perturbed anywhere in the pipeline:
- `config.yaml` has no `aRadius/bRadius/cRadius` entries (comment at top: "radius perturbation params are not read here")
- `PerturbParams{}` default-initializes sigma=0, prob=0
- Prior run logs confirm: `majorSigma=0->0 bSigma=0->0 minorSigma=0->0`
- Radii only change via PCA shape fit (explicit) and `buildDaughter` (seeded from parent)

The `split_burn_in_radius_sigma_scale` config + save/restore of radius params in Frame.cpp and CellUniverse.cpp were no-ops (savedSigma=0 → 0*scale=0). Removed to reduce noise.

**File:** `C++/src/Frame.cpp`

- Removed `radiusScale` constant (reading deleted config field).
- Removed `savedPerturbMajor/B/Minor` save/restore (3 sites: install + 2 restore exits).
- Removed `majorSigma/bSigma/minorSigma/radiusScale` fields from `[Split Sigmas]` log.
- Updated comment on `builtR` diagnostic to reflect that radii are never perturbed (not "frozen by sigma scale").

**File:** `C++/src/CellUniverse.cpp`

- Removed `savedCalMajor/B/Minor` save/restore in position calibration block (lines ~379–465 old numbering).
- Removed the three `aRadius/bRadius/cRadius.sigma = 0.0f` writes.
- Removed `radiusSigma=0 (frozen)` from the `[Calibration]` log.

**File:** `C++/includes/ConfigTypes.hpp`

- Deleted field `float split_burn_in_radius_sigma_scale = 0.1f;`
- Deleted the matching `explodeConfig` YAML parse line.
- Deleted the preceding comment block describing the field.

**File:** `C++/config/config.yaml`

- Deleted line `split_burn_in_radius_sigma_scale: 0.0`

**File:** `C++/docs/pipeline.md`

- Deleted the same line from the config example block.

### Effect

1. False splits caught by image evidence, not squeaked through on cost-diff alone. Expected to eliminate f5 a51 and f7 a511 cascades in next run.
2. `split_cost` threshold and asymmetric-cost weight unchanged — gates do the work, cost stays sensitive for real splits.
3. Dead code removed; future readers won't wonder why radius sigmas are being scaled to zero repeatedly.

### Unchanged / not yet done

- `bio_bridge_max_valley_ratio = 0.85` config value unchanged; it is now the threshold on `worstValleyRatio` instead of pooled valleyRatio.
- `bio_bridge_max_gap_density = 0.18` unchanged.
- `kMinEdgeBrightAbsolute = 0.05f` currently hardcoded in `Frame.cpp`; promote to config if tuning needed.
- `split_cost`, `asymmetric_cost_weight` unchanged. If false splits still leak after this, consider raising asym weight 6 → 8 or split_cost 15 → 20.

## 2026-04-14 (evening): Widen cost gap — asymmetric weight 6→8 (B1) + per-daughter PCA radius refit (A1)

**Status:** ACTIVE

### Problem — true vs false split cost diffs overlap

From run `output_jihang_20260414_170618`:

| Split | costDiff | margin vs `-split_cost=-15` |
|---|---|---|
| 12345 f3 ✓ | -23.4 | -8.4 |
| e9077 f3 ✓ | -18.4 | -3.4 |
| e9077a51 f4 ✓ | -15.6 | -0.6 |
| a51 f5 ✗ | -15.5 | -0.5 |

True and false splits separated by only 0.1 in costDiff — cost gate cannot reliably discriminate. Two levers applied in this change to widen the gap:

- **B1 (reward side):** raise `asymmetric_cost_weight` 6.0 → 8.0. Per-voxel L2 multiplied by k when `synth > real` (overshoot into dim). A true-split parent must overshoot the valley between daughters → 33% more penalty on parent. A false-split phantom daughter sits in near-background → also more penalty. Both mechanisms push costDiff more negative for true splits and less negative (or positive) for phantoms.
- **A1 (reward side):** per-daughter PCA radius refit in the split refine phase. Daughters inherit parent-based built radii (`0.794 × parent.src`) and never adjust. Real daughters are usually smaller / differently shaped; refitting each daughter to its Voronoi-claimed pixel cloud tightens the synth → lower image cost. Phantom daughters in empty space hit the `split_daughter_refit_min_radius_fraction * built` floor and gain nothing.

### B1 — asymmetric cost weight 6 → 8

**File:** `C++/config/config.yaml`

**Line 127 (before):**
```yaml
  asymmetric_cost_weight: 6.0
```

**Line 131 (after):**
```yaml
  asymmetric_cost_weight: 8.0
```

Comment block above updated with the rationale.

### A1 — per-daughter PCA radius refit

**File:** `C++/includes/ConfigTypes.hpp`

Added to `ProbabilityConfig`:

```cpp
// Per-daughter PCA radius refit in the split refine phase (A1).
int split_daughter_refit_iterations = 3;
float split_daughter_refit_min_radius_fraction = 0.6f;
```

Matching `explodeConfig` entries added.

**File:** `C++/config/config.yaml`

```yaml
  split_daughter_refit_iterations: 3
  split_daughter_refit_min_radius_fraction: 0.6
```

**File:** `C++/src/Frame.cpp`

Inserted between positional refine loop and best-state re-capture (≈ lines 2110–2180). After the 30-iter positional refine settles each daughter's center, for each daughter:

1. Build a `ClaimSet` that treats every OTHER cell (including the sibling daughter) as a Voronoi claimant — prevents sibling pixels from contaminating this daughter's PCA.
2. Call `calibrateCellShapeViaPca` with:
   - `maskA/B/C = daughter built radii` (fixed mask prevents feedback collapse, same pattern as the per-frame snap-mask).
   - `updatePosition = false` (positions are fixed from refine).
   - `maxIters = split_daughter_refit_iterations` (3 by default — tight budget).
   - All other knobs from the global `Ellipsoid::cellConfig.pcaShape*` settings.
3. Clamp fitted radii to `max(fitted, split_daughter_refit_min_radius_fraction * built)` per axis. Prevents the collapsed-sliver regression that drove the original frozen-radii design.
4. `setRadii(fitA, fitB, fitC)` on the daughter.

After both daughters refit, call `generateSynthFrame()` + `refreshFullCostCache()` so the cost gate uses synth rendered at the new radii.

Diagnostic line `[Split Daughter Refit]` logs built / floor / pre / post radii per daughter.

### Expected behavior change

- **True split (e.g. e9077 f3):** each daughter PCA-refits to its own compact blob, typically smaller than built (the 0.794 × parent starting size was generous). Synth matches real tighter → image cost drops further. k=8 asymmetric also hurts parent baseline more on the valley region. costDiff shifts from -18 to, expected, -25 to -35.
- **False split (e.g. a51 f5):** phantom daughter's PCA finds few bright pixels in its Voronoi territory (near background). Fitted radii collapse toward the floor (0.6 × built). Daughter covers less volume → less cost reduction from "covering" the phantom region. k=8 asymmetric hurts the phantom daughter's own synth-on-dark overshoot more. costDiff shifts from -15.5 toward -10 or positive → reject.
- **Borderline good split (e9077a51 f4 at -15.6):** expected to become clearly negative (< -20) with tight daughters + k=8.

### Risk & mitigation

- Radius collapse regression: guarded by `min_radius_fraction = 0.6`. A daughter can't shrink below 60% of built size on any axis. Tuneable.
- False tightening on noisy edges: `split_daughter_refit_iterations = 3` is a tight budget. Snap-mask (fixed built-radii mask) prevents mask-feedback loops.
- Cost threshold: `split_cost = 15` unchanged. With the new diff values, true splits land comfortably below threshold; false splits don't. If real splits start getting rejected after these changes, lower threshold to 10. If false still pass, raise to 20.

### Files changed

- `C++/config/config.yaml` — `asymmetric_cost_weight` 6.0 → 8.0, added two `split_daughter_refit_*` entries.
- `C++/includes/ConfigTypes.hpp` — added two fields + `explodeConfig` reads.
- `C++/src/Frame.cpp` — inserted per-daughter refit block inside `trySplitCellPhased` refine phase; updated the `[Split Refine]` diagnostic comment to describe post-refit radii semantics.

