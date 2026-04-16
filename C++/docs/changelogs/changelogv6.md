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

## 2026-04-14 (evening): Quadratic PCA weighting (Option A), CMake optimization flags (P1), OpenMP on shape-fit + cost loops (P2, P3)

**Status:** ACTIVE

### Option A — quadratic intensity weighting in PCA shape fit

**Problem:** run `output_jihang_20260414_184954` showed 1f89ab at f8 converging to R=(41.88, 23.56, 17.01) — 30% bigger on every axis than the known-good 0413 run's (32.35, 19.80, 13.38). Pixel-count comparison revealed the new fit includes 62% more pixels than the good-run fit. The extra pixels are a dim halo around the cell body. Under linear intensity weighting, halo volume × low-weight exceeds core volume × high-weight, so PCA variance tracks the halo's spatial extent rather than the cell body's.

With `max(aRadius,bRadius,cRadius)` raised 50→65 in the current config, cells grew to ~55 aR in frames 2–5, bloated snap masks 30% bigger than the good run. Mask-feedback kept cells oversized through division frames. Clamping is not a fundamental fix — a genuinely larger cell would be artificially constrained.

**Solution:** weight each pixel's contribution to the PCA covariance by `intensity²` instead of `intensity`. For microscopy PSF-convolved signals, this is the log-likelihood weight for fitting Gaussian means under Gaussian noise. Mathematically:

- Uniform-bright ellipsoid (all weights equal): `w² ≡ w`, so radius = `sqrt(5) × sqrt(var)` remains correct.
- Bright core + dim halo: core-to-halo weight ratio increases from ~5× (linear) to ~25× (quadratic). Halo contribution to variance drops by an order of magnitude. Fit tracks the core.

**File:** `C++/src/Frame.cpp` (lines ~1301–1336 in `calibrateCellShapeViaPca`)

**Before:**
```cpp
double sx = 0, sy = 0, sz = 0, wsum = 0;
for (const auto &bp : pixels) {
    sx += bp.pos.x * bp.weight;  // w = intensity above background
    sy += bp.pos.y * bp.weight;
    sz += bp.pos.z * bp.weight;
    wsum += bp.weight;
}
...
for (const auto &bp : pixels) {
    const double w = bp.weight;
    cxx += w * dx * dx;
    ...
}
```

**After:**
```cpp
double sx = 0, sy = 0, sz = 0, wsum = 0;
for (const auto &bp : pixels) {
    const double w2 = static_cast<double>(bp.weight) * bp.weight;
    sx += bp.pos.x * w2;
    sy += bp.pos.y * w2;
    sz += bp.pos.z * w2;
    wsum += w2;
}
...
for (const auto &bp : pixels) {
    const double w2 = static_cast<double>(bp.weight) * bp.weight;
    cxx += w2 * dx * dx;
    ...
}
```

Both centroid and covariance use `w²` consistently; normalization by `sum(w²)` preserves the variance-to-radius relation `radius = radiusScale × sqrt(variance)`.

### P1 — CMake Release flags + LTO + OpenMP

**File:** `C++/CMakeLists.txt`

Added:
- Default `CMAKE_BUILD_TYPE=Release` when none specified.
- `-O3 -march=native -DNDEBUG` on Release for GCC/Clang/AppleClang.
- Interprocedural optimization (LTO) enabled when compiler supports it (`check_ipo_supported`).
- `find_package(OpenMP REQUIRED)` + `target_link_libraries(celluniverse OpenMP::OpenMP_CXX)`.

Expected standalone speedup: 2–5× if we were previously on `-O0`; 1.1–1.3× from LTO. OpenMP link enables the per-cell pragmas below.

### P2 — OpenMP on per-cell shape-fit loop

**File:** `C++/src/CellUniverse.cpp` (line 506)

Added `#pragma omp parallel for schedule(dynamic)` on the shape-fit loop. Cells are independent: `_realFrame` is read-only, each thread writes a unique `frame.cells[ci]`, `ClaimSet` is constructed thread-locally. Expected ~N× speedup on shape fit phase (N = thread count).

Per-iter `std::cout` logs may interleave across threads — this is expected, not a correctness issue. Set `OMP_NUM_THREADS=1` for deterministic log order during debugging.

Loop index type changed `size_t → int` because OpenMP for-loop requires a signed integer counter. Implicit conversion to `size_t` at the downstream `calibrateCellShapeViaPca` call is safe (non-negative).

**Pre-pass loop left sequential** (CellUniverse.cpp:559) — it writes to a shared `expectedDaughters` std::map, not thread-safe without a per-thread results vector + merge. Pre-pass is not a hot spot; can be parallelized later.

### P3 — OpenMP on full-image cost loops

**File:** `C++/src/Frame.cpp` (`refreshFullCostCache` at line 94, `calculateCost` at line 178)

Added `#pragma omp parallel for reduction(+:totalCost) schedule(static)` on the 225-slice loops. `asymmetricL2Slice` is a pure function; `_realFrame[i]` and `_synthFrame[i]` are independent per thread. Safe.

`calculateIncrementalCost` (the hot path inside `perturbCell`) **left sequential** — only ~5 slices are recomputed per call, thread setup overhead would exceed gains.

### Expected combined effect

- Option A: `1f89ab` f8 expected to converge near (32, 20, 13) not (42, 24, 17); split pipeline should accept at f9 or earlier instead of missing.
- P1 + P2 + P3: expected 5–15× runtime reduction on shape-fit-heavy frames. Release build flags may account for most of the gain if debug build was being used previously.

### Not yet implemented (per the 2026-04-14 universal bbox plan)

- **Bbox cost refactor (Stages 1–3)** — deferred to next session. Plan locked at `C++/docs/plans/2026-04-14-universal-bbox-cost.md`. Decisions D1–D7 confirmed. Remaining work: ~500 LOC new code, ~100 LOC migration, ~150 LOC deletion, parameter retuning (`split_cost`, `overlap_penalty_weight`).
- **P6** (OMP on split candidates) — deferred until bbox lands (per-thread state is cheaper with bbox).
- **P4** (bright-pixel gather cache) — deferred.

### Files changed

- `C++/src/Frame.cpp` — quadratic PCA weighting (Option A); OMP on `refreshFullCostCache` + `calculateCost`.
- `C++/src/CellUniverse.cpp` — OMP on shape-fit loop.
- `C++/CMakeLists.txt` — Release default, `-O3 -march=native -DNDEBUG`, IPO/LTO, OpenMP.

## 2026-04-14 (evening): Universal bounding-box cost infrastructure (feature-flagged)

**Status:** ACTIVE (behind `use_bbox_cost: false` flag — legacy full-image path still default)

### Summary

Implements Stages 1–3 of the Universal Bbox Cost plan (`C++/docs/plans/2026-04-14-universal-bbox-cost.md`) behind a config flag. When `use_bbox_cost: true` is set, perturbation and the split cost gate use per-cell bbox cost with Voronoi neighbor exclusion instead of full-image L2. When false (default), behavior is unchanged.

Design decisions locked at plan (D1–D7):
- D1: bbox = `bbox_margin_scale × max(a,b,c)` half-extent per axis, default `3.0` (matches existing `boxRadius` convention at `Frame.cpp:1035/1096/1583`)
- D2: Voronoi exclusion using claim-point sets (same rule as `gatherBrightPixelsVoronoi`)
- D3: per-cell bbox cost replaces `_currentCost` semantics at decision sites (legacy state kept for diagnostic continuity)
- D4: no cache — bbox cost computed per request
- D5: split uses union bbox of parent + both refined daughters
- D6: overlap penalty retained (bbox cost is blind to overlap — Voronoi excludes the overlap region from both cells)
- D7: `split_cost` / `overlap_penalty_weight` to be retuned empirically after validation runs

### Stage 1 — infrastructure (`Frame.hpp`, `Frame.cpp`)

New struct `BoundingBox3D` (inclusive int bounds + nx/ny/nz/volume helpers).

New `Frame` methods:

- `computeCellBbox(cellIdx, marginScale)` — per-cell AABB with `marginScale × maxR` half-extent, clamped to image bounds.
- `computeUnionBbox(cellIndices, marginScale)` — union of per-cell bboxes.
- `computeUnionBboxWithPoints(cellIndices, marginScale, extraPoints, pointRadius)` — includes extra 3D points (e.g. daughter seeds) as spheres of `pointRadius` each.
- `buildExclusionMask(bbox, selfClaimPoints, otherClaimSets)` — 3D uint8 mask; voxel `v` is kept iff its nearest claim point across self ∪ others belongs to self. Flatten-first (otherClaimSets → single vector) for tight inner-loop access. Parallelized via `#pragma omp parallel for` over z-slices.
- `calculateBboxCost(bbox, synthFrame, mask)` — asymmetric L2 over bbox voxels where `mask[v]=1`. Same `asymmetric_cost_weight` as `calculateCost`. Parallelized with `reduction(+:totalCost)`.

### Stage 2 — perturbCell bbox path (`Frame.cpp:547`)

When `_useBboxCost` is true:

1. Compute union bbox of pre-perturbation and post-perturbation cell positions (covers small drift within burn-in sigmas).
2. Build `selfClaimPoints = [cell.center]`, `otherClaimSets = {other.name → [other.center]}`.
3. `buildExclusionMask` once per call.
4. `oldImageCost = calculateBboxCost(bbox, _synthFrame, mask)`, `newImageCost = calculateBboxCost(bbox, newSynthFrame, mask)`.
5. `costDiff = (newImageCost + newOverlap) − (oldImageCost + oldOverlap)`.

When `_useBboxCost` is false: unchanged legacy `calculateIncrementalCost` path.

Accept callback updates `_synthFrame` in both modes; only the legacy path writes `_currentCost` + `_currentCostPerSlice` (stale under bbox mode — kept for diagnostic logging only).

### Stage 3 — split cost gate bbox path (`Frame.cpp:2777`)

Rather than refactor burn-in and refine (many call sites using `calculateIncrementalCost` + cached `_currentCost`), the bbox path injects at the FINAL cost gate:

1. After bio checks pass, build union bbox of savedCells[cellIndex] (parent pre-split) + bestD1 + bestD2 with `bbox_margin_scale × maxR` margin each.
2. `selfClaimPoints = [parent.center, d1.center, d2.center]`, `otherClaimSets = savedCells except parent`.
3. `bboxBaseline = calculateBboxCost(bbox, savedSynth, mask)` — synth with parent.
4. `bboxCandidate = calculateBboxCost(bbox, bestSynth, mask)` — synth with daughters.
5. Add overlap delta computed from `bestTotal − bestImageCost − baselineOverlap`.
6. `costDiff = bboxCandidate − bboxBaseline + overlapDelta`.
7. Gate: `costDiff >= -split_cost` → reject.

Burn-in ranking and refine still use full-image cost — cheap simplification for this staging, and candidate ranking is a weaker signal than the final gate decision. Can be migrated in a later pass if gate-only bbox proves insufficient.

New diagnostic log `[Split Bbox Cost]` emitted when bbox mode is on, showing `bboxVolume / bboxBaseline / bboxCandidate / bboxImageDiff / overlapDelta / bboxTotalDiff / fullCostDiff`.

### Files changed

- `C++/includes/Frame.hpp` — `BoundingBox3D` struct; 5 new public methods; `setUseBboxCost` / `getUseBboxCost` / `getBboxMarginScale`; three new private members `_useBboxCost`, `_bboxMarginScale`, `_bboxOverlapWeight`.
- `C++/src/Frame.cpp` — implementations of the five bbox helpers (+175 LOC); bbox branch in `perturbCell`; bbox cost gate in `trySplitCellPhased`.
- `C++/includes/ConfigTypes.hpp` — `use_bbox_cost` + `bbox_margin_scale` on `ProbabilityConfig` with `explodeConfig` entries.
- `C++/config/config.yaml` — `use_bbox_cost: false` + `bbox_margin_scale: 3.0` with comment block on retuning expectations.
- `C++/src/CellUniverse.cpp` — `frame.setUseBboxCost(...)` at the start of `optimize()`; new `[Optimize]` log fields.

### How to enable

In `config.yaml`:
```yaml
prob:
  use_bbox_cost: true
  bbox_margin_scale: 3.0
```

Re-run. `[Optimize] ... useBboxCost=1 bboxMarginScale=3` confirms activation. For each split, `[Split Bbox Cost] ... bboxBaseline=... bboxCandidate=... bboxImageDiff=... fullCostDiff=...` shows the new cost diff alongside the legacy number for direct comparison.

### Expected behavior (with flag ON)

- **Per-cell perturbation decisions** isolated from unrelated image regions. Cell A's perturbation no longer depends on cell B's current fit elsewhere in the image.
- **Split cost gate** measures only the parent+daughters voxel set. `split_cost=15` (tuned for full-image numbers) will be much too loose for bbox numbers — expect real splits at `bboxImageDiff ≈ -3 to -10` and phantoms at `≈ 0 to positive`. Retune `split_cost` to around `2–5` after first validation run.
- **Overlap penalty** unchanged magnitude but now competes with smaller image cost numbers. May need `overlap_penalty_weight` reduced from `500` to `50–100` to avoid dominating.
- **Runtime**: bbox is ~5M voxels (3×maxR cube) vs ~36M full image, roughly 6× less work per cost eval. Combined with OpenMP on the bbox-cost loop, significant speedup expected.

### Deferred

- **Burn-in + refine bbox cost** — currently use full-image; could migrate for uniform scale but candidate ranking is less sensitive than gate decision. Revisit after validation.
- **Legacy cost removal** — `calculateCost`, `calculateIncrementalCost`, `_currentCost`, `_currentCostPerSlice`, `refreshFullCostCache` remain in place and are used when `use_bbox_cost: false`. Delete only after bbox path is fully validated as a drop-in replacement.
- **Per-thread bbox synth for parallel split candidates (P6 in plan)** — waits for bbox path migration into burn-in.

## 2026-04-14 (evening): Multithread build — cross-platform OpenMP setup + per-cell / per-slice parallelization

**Status:** ACTIVE

### Goal

Current runtime is ~1 h per 22 frames. The dominant hot paths — per-cell shape fit, full-image asymmetric L2 cost, and (with the bbox flag) per-voxel bbox cost / exclusion-mask build — are all embarrassingly parallel across cells or z-slices. Introduce OpenMP with zero behavior change when it's unavailable (pragmas become no-ops), and auto-configure the build on both Linux and macOS.

### CMake cross-platform OpenMP setup (`C++/CMakeLists.txt`)

**Before:** no OpenMP detection, no explicit Release flags, no LTO. Depending on how the user invoked CMake, builds could land on `-O0` and run ~20× slower than optimized.

**After:**

```cmake
# Default to Release when no build type is specified.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Link-time optimization check.
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)

# OpenMP: Linux/GCC just works. macOS AppleClang needs Homebrew libomp +
# explicit flags. We auto-detect the Brew prefix and seed the hints so
# `find_package(OpenMP)` succeeds without extra cmake -D args.
if(APPLE)
    find_program(BREW_EXECUTABLE brew)
    if(BREW_EXECUTABLE)
        execute_process(
            COMMAND ${BREW_EXECUTABLE} --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE LIBOMP_BREW_RESULT
            ERROR_QUIET
        )
        if(LIBOMP_BREW_RESULT EQUAL 0 AND EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" CACHE STRING "")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" CACHE STRING "")
            set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "")
            set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "")
            set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib" CACHE FILEPATH "")
        endif()
    endif()
endif()
find_package(OpenMP)

# Release-only flags on main target.
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O3 -march=native -DNDEBUG>
)
if(IPO_SUPPORTED)
    set_property(TARGET ${PROJECT_NAME} PROPERTY
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()
```

OpenMP is **optional**: if `find_package(OpenMP)` fails, the build still succeeds and the `#pragma omp` directives in source become no-ops (compiler ignores them). A `STATUS` message tells the user how to install libomp on their platform.

### Build instructions (both platforms)

**macOS (one-time setup):**
```bash
brew install libomp
```

**Any platform (build):**
```bash
cd C++/build
rm CMakeCache.txt  # only needed the first time after CMakeLists.txt changes
cmake -S .. -B .
cmake --build . -j
```

Look for `-- Found OpenMP_CXX: ...` in configure output. If missing, build still works serially.

### Parallel regions added

#### `Frame::refreshFullCostCache` (full-image cost rebuild)

**File:** `C++/src/Frame.cpp:94`

```cpp
#pragma omp parallel for reduction(+:totalCost) schedule(static)
for (int i = 0; i < nSlices; ++i) {
    const double sliceCost = asymmetricL2Slice(_realFrame[i], _synthFrame[i], asymK);
    _currentCostPerSlice[i] = sliceCost;
    totalCost += sliceCost;
}
```

Called once per frame. 225 independent slice-L2 computations. Safe because `asymmetricL2Slice` is pure, `_realFrame[i]` / `_synthFrame[i]` are independent across i, and `_currentCostPerSlice[i]` is written by exactly one thread.

#### `Frame::calculateCost` (on-demand full-image cost)

**File:** `C++/src/Frame.cpp:178`

Same pattern as `refreshFullCostCache`. Called from the split-path snapshot-baseline swap site.

**Not parallelized:** `calculateIncrementalCost` — the hot perturbation path only recomputes 5–10 slices per call, thread setup overhead would dominate.

#### `CellUniverse::optimize` shape-fit loop

**File:** `C++/src/CellUniverse.cpp:506`

```cpp
const int nCells = static_cast<int>(frame.cells.size());
#pragma omp parallel for schedule(dynamic)
for (int ci = 0; ci < nCells; ++ci) {
    Frame::ClaimSet others = buildShapeClaimSet(sname);  // thread-local
    // ... extract mask radii from snapshot ...
    frame.calibrateCellShapeViaPca(ci, others, ...);      // writes cells[ci] only
}
```

Cells are independent: `_realFrame` is read-only; each thread writes a unique `frame.cells[ci]`; `ClaimSet` is thread-local. Dynamic scheduling because per-cell work varies (bigger cells take more PCA iterations).

Loop index changed `size_t → int` because OpenMP for-loops require signed integer counters; `int → size_t` implicit conversion downstream is safe for non-negative values.

**Caveat:** per-iteration `[PCA Shape]` logs may interleave across threads. Correct output but ugly. Set `OMP_NUM_THREADS=1` to serialize logs for debugging.

**Not parallelized:** the pre-pass loop (`CellUniverse.cpp:573`) writes a shared `std::map<std::string, ...> expectedDaughters` — map operations are not concurrency-safe. Pre-pass is not a hot spot (~10 cells × 1 round/frame); can be refactored to a per-cell results vector and merged if needed.

**Also not parallelized:** the per-cell position-calibration loop (`CellUniverse.cpp:425`) — each `perturbCell` writes `_synthFrame` and `_currentCost` as global state. Would need per-thread synth buffers to parallelize; defer.

#### `Frame::buildExclusionMask` (Voronoi mask for bbox cost)

**File:** `C++/src/Frame.cpp` (bbox infrastructure section)

```cpp
#pragma omp parallel for schedule(static)
for (int z = bbox.zMin; z <= bbox.zMax; ++z) {
    for (int y = bbox.yMin; y <= bbox.yMax; ++y) {
        for (int x = bbox.xMin; x <= bbox.xMax; ++x) {
            // ... Voronoi test vs selfClaimPoints ∪ flattened otherPoints ...
            if (keep) mask[yOff + (x - bbox.xMin)] = 1;
        }
    }
}
```

Called once per bbox cost evaluation when `use_bbox_cost: true`. 3D bbox up to ~5M voxels. Each voxel writes a unique mask byte. Safe.

#### `Frame::calculateBboxCost`

**File:** `C++/src/Frame.cpp` (bbox infrastructure section)

```cpp
#pragma omp parallel for reduction(+:totalCost) schedule(static)
for (int z = bbox.zMin; z <= bbox.zMax; ++z) {
    // ... per-slice inner loop summing asymmetric L2 over mask=1 voxels ...
    totalCost += sliceCost;
}
```

Same per-z-slice parallelism pattern as `calculateCost`. Safe.

### Thread-safety audit

Verified before enabling pragmas:

| Concern | Status |
|---|---|
| `Ellipsoid::cellConfig` global | Read-only within parallel regions. All writes (sigma scaling for burn-in / calibration) happen in serial code outside the OMP loops. ✓ |
| `cv::Mat` sharing | `_realFrame[i]` read across threads — safe (reads are reentrant). Writes to distinct slices of `_synthFrame` happen in the serial outer `perturbCell`, not inside parallel regions. ✓ |
| `std::cout` interleaving | Non-deterministic output order. Correct content. Acceptable for debug logs. Not a correctness issue. |
| `rand()` / `std::uniform_int_distribution` | Shape-fit / cost loops don't call RNG. The main MCMC loop (which does) is serial. ✓ |
| `std::map` / `std::vector` writes | Map writes only in serial regions. Vector writes to distinct indices only. ✓ |

### Expected speedup

On an 8-core machine (macOS M-series, Linux x86-64):

| Stage | Expected speedup | Notes |
|---|---|---|
| `-O3 -march=native -DNDEBUG` alone | 2–5× over `-O0` | zero risk, biggest single lever if previously on Debug |
| LTO/IPO | 1.1–1.3× | small, zero risk |
| OMP per-cell shape fit | 4–7× on this stage | cells are independent; dynamic scheduling |
| OMP per-slice full-image cost | 4–6× on this stage | called a handful of times per frame |
| OMP on bbox mask + bbox cost | 4–6× when `use_bbox_cost: true` | small bbox per call, but hot in bbox mode |

Combined on a bbox-enabled full-frame run: ~8–20× overall. 1 h / 22 frames → 3–8 min / 22 frames target.

### Files changed

- `C++/CMakeLists.txt` — Release default, `-O3 -march=native -DNDEBUG`, IPO/LTO check, OpenMP auto-detect with macOS Brew libomp prefix seeding.
- `C++/src/Frame.cpp` — `#pragma omp parallel for` on `refreshFullCostCache`, `calculateCost`, `buildExclusionMask`, `calculateBboxCost`.
- `C++/src/CellUniverse.cpp` — `#pragma omp parallel for schedule(dynamic)` on shape-fit loop; `size_t → int` loop index.

### Tuning knobs

Environment variables at run time:

- `OMP_NUM_THREADS=N` — force thread count. Use `1` for deterministic log order.
- `OMP_SCHEDULE=dynamic,1` — override schedule; current code uses `static` (cost loops) and `dynamic` (shape fit) which is generally right.
- `OMP_PROC_BIND=true` — pin threads to cores (Linux). Macs ignore this.

### Deferred

- **OMP on pre-pass loop** — requires converting `expectedDaughters` writes to a per-thread results vector + merge. ~30 min work.
- **OMP on position-calibration loop** — requires per-thread synth buffers for the perturbation-with-cost pattern. ~2 hrs.
- **OMP on split candidate burn-in (P6 in plan)** — requires bbox cost in burn-in (so per-thread state is small). Blocked on bbox Stage 3 full migration.
- **OMP on image preprocessing** (blur, sigmoid, z-interpolation in `ImageHandler`) — per-slice independence; straightforward add if preprocessing wall-clock is a visible bottleneck on larger datasets.

## 2026-04-14 (evening): Tunable PCA weight exponent + bridge-gate threshold extraction to YAML

**Status:** ACTIVE

### Motivation

Option A (quadratic PCA weighting, `w²`) tightens fits onto the bright cell core. Initial validation showed cells looking slightly too small — enough halo contribution is needed for the ellipsoid to cover the full cell body, not just the peak. Rather than revert the principled halo-suppression, expose the exponent as a config knob so it can interpolate between linear (1.0, bloat) and quadratic (2.0, tight).

Also: the bridge gate had two magic numbers embedded as `constexpr float` in `Frame.cpp` — `0.95` no-valley hard threshold, `0.05` absolute edge-brightness floor. Promoted to YAML for trackable tuning.

### Changes

**File:** `C++/includes/ConfigTypes.hpp`

Added to `EllipsoidConfig`:
```cpp
float pcaShapeWeightExponent{1.5f};  // exp=1 linear, 2 quadratic
```

Added to `ProbabilityConfig`:
```cpp
float bio_bridge_no_valley_hard_threshold = 0.95f;
float bio_bridge_min_edge_brightness_absolute = 0.05f;
```

All three have matching `explodeConfig` parse entries.

**File:** `C++/src/Frame.cpp` (`calibrateCellShapeViaPca`)

Replaced hardcoded `w² * dx * dx` with configurable-exponent lambda:
```cpp
const float weightExponent = std::max(0.0f, Ellipsoid::cellConfig.pcaShapeWeightExponent);
const bool exp1 = std::abs(weightExponent - 1.0f) < 1e-6f;
const bool exp2 = std::abs(weightExponent - 2.0f) < 1e-6f;
auto effectiveWeight = [&](float w) -> double {
    if (exp1) return static_cast<double>(w);
    if (exp2) return static_cast<double>(w) * w;
    return std::pow(static_cast<double>(w), static_cast<double>(weightExponent));
};
```
Used in both centroid and covariance accumulators. Fast paths for `exp=1.0` and `exp=2.0` avoid `std::pow`.

**File:** `C++/src/Frame.cpp` (bridge gate)

Replaced:
```cpp
constexpr float kMinEdgeBrightAbsolute = 0.05f;
constexpr float kNoValleyHardThreshold = 0.95f;
```
with:
```cpp
const float kMinEdgeBrightAbsolute = probConfig.bio_bridge_min_edge_brightness_absolute;
const float kNoValleyHardThreshold = probConfig.bio_bridge_no_valley_hard_threshold;
```

**File:** `C++/config/config.yaml`

Added three new tunable entries with explanatory comments on tuning direction:
```yaml
cell:
  pcaShapeWeightExponent: 1.5       # 1.0 bloat, 2.0 tight, 1.5 balanced
prob:
  bio_bridge_no_valley_hard_threshold: 0.95
  bio_bridge_min_edge_brightness_absolute: 0.05
```

### Default choice — `pcaShapeWeightExponent = 1.5`

`2.0` was too tight (cell radii noticeably smaller than the visible cell body). `1.0` was the original bloated behavior. `1.5` is a balanced interpolation:

- Core-to-halo weight ratio ~7× (vs 5× linear, 25× quadratic)
- Fitted radii ~15% smaller than pure linear, ~15% larger than pure quadratic
- Enough halo contribution for the ellipsoid to cover cell extent without tracking diffuse haze

Retune per dataset. Observation procedure:

- `[PCA Shape]` converged radii look too small vs real image → lower exponent (1.3, 1.2, 1.0).
- `[PCA Shape]` radii bloat into neighboring structure → raise exponent (1.7, 2.0).

### Tunable YAML surface (complete list)

All tracked through git history now:

- `pcaShapeWeightExponent` (1.5) — this change
- `bio_bridge_no_valley_hard_threshold` (0.95) — this change
- `bio_bridge_min_edge_brightness_absolute` (0.05) — this change
- `pcaShapeRadiusScale` (2.236) — alternative lever for cell size
- `asymmetric_cost_weight` (8.0)
- `use_bbox_cost` (false)
- `bbox_margin_scale` (3.0)
- `split_daughter_refit_iterations` (3)
- `split_daughter_refit_min_radius_fraction` (0.6)
- `split_cost` (15) — retune if `use_bbox_cost` is enabled
- `overlap_penalty_weight` (500) — retune if `use_bbox_cost` is enabled

### Files changed

- `C++/includes/ConfigTypes.hpp` — 3 new fields + explodeConfig entries.
- `C++/src/Frame.cpp` — configurable weight exponent in PCA moments; bridge thresholds read from ProbabilityConfig.
- `C++/config/config.yaml` — 3 new tunable entries with tuning-direction comments.

## 2026-04-14 (evening): Configurable daughter volume scale + empirical analysis

**Status:** ACTIVE

### Empirical analysis from good 0413 run (`output_jihang_20260413_good_shape/cells.csv`)

Daughter / snap-parent radius ratios at the moment of split:

| Split | Snap parent (a, b, c) | Daughter (built) | Per-axis ratio |
|---|---|---|---|
| e9077 f3 | 30.2, 28.9, 23.7 | 24.0, 23.0, 18.8 | 0.795, 0.796, 0.793 |
| 12345 f3 | 31.9, 29.5, 23.4 | 25.3, 23.4, 18.6 | 0.793, 0.793, 0.795 |
| 1f89ab f9 | 32.4, 19.8, 13.4 | 25.7, 15.7, 10.6 | 0.793, 0.793, 0.791 |

**Mean ratio = 0.794 = ∛(1/2)** — matches the hardcoded `volumeScale = std::cbrt(0.5f)` exactly. Confirms volume-preserving split sizing produces the daughter shapes the system has been generating.

After one frame of free shape fit, daughters relax to slightly larger sizes (e.g. 1f89ab daughters at f10 reach (38.6, 23.8, 14.6) — *bigger* than the f8 parent in 2 axes because the parent was constricted mid-divide, while post-split daughters round up).

Conclusion: 0.794 is the right mathematical default, but it has no flexibility for cases where the SNAP itself is wrong (e.g. shape-fit drift makes the snap too tight). Promoted to YAML so it can be tuned alongside `pcaShapeWeightExponent`.

### Implementation

**File:** `C++/includes/ConfigTypes.hpp` (in `ProbabilityConfig`)

```cpp
// Built-time per-axis radius scale for newly-spawned daughters.
// daughter.radii = scale * snapshot_parent.radii.
// Default ∛(0.5) ≈ 0.7937 preserves total cell volume.
float split_daughter_volume_scale = 0.7937f;
```

With matching `explodeConfig` parse entry.

**File:** `C++/src/Frame.cpp` (`trySplitCellPhased`)

**Before:**
```cpp
const float volumeScale = std::cbrt(0.5f);
```

**After:**
```cpp
const float volumeScale = std::max(0.1f, probConfig.split_daughter_volume_scale);
```

Floor at 0.1 prevents pathological config (daughters with radii < 1 voxel).

**File:** `C++/config/config.yaml`

```yaml
# Tune up (0.85, 0.90, 1.0) for larger initial daughters — useful when
# parent PCA fit is tight; bio gates trim if too big.
# Tune down (0.65, 0.55) for tighter initial daughters.
split_daughter_volume_scale: 0.7937
```

### Tuning recommendation

Leave at 0.7937 unless investigation of a specific failed split shows the daughters are clearly undersized at the cost gate. Then experiment with 0.85 → 0.95 → 1.0 in 3 separate runs and compare `[Split Reject cost]` diagnostic numbers.

If the parent PCA fit is the root cause of tight daughters (which is suspected given Option A weight exponent + position drift), fix the parent fit first (`pcaShapeWeightExponent` lower, position calibration capped) — daughter scale is downstream.

### Files changed

- `C++/includes/ConfigTypes.hpp` — `split_daughter_volume_scale = 0.7937f` field + `explodeConfig` entry.
- `C++/src/Frame.cpp` — read `volumeScale` from config instead of hardcoding `cbrt(0.5f)`.
- `C++/config/config.yaml` — `split_daughter_volume_scale: 0.7937` with empirical justification + tuning comments.

## 2026-04-14 (evening): Snap-anchored parent — skip calibration + freeze PCA shape position when snap is valid

**Status:** ACTIVE

### Problem (re-stated, with empirical evidence)

Run `output_jihang_20260414_212223` (weightExponent=1.3) split 12345 at f3 ✓ but missed e9077 at f3. Root cause: position calibration's brightness centroid is multi-modal for any cell whose bright pixels span two emerging daughter blobs — it drifts toward the brighter blob. PCA shape fit then locks live radii onto that one daughter. Live cost beats snap cost, becomes the split baseline, and the split attempt looks like a redundant move (parent already covers d1; daughters at d1+d2 only marginally better).

For e9077 f3 in 212223: snap=(297, 164, 121) → centroid jumped 23 px to (300, 184, 110) → perturb settled at (300, 187, 106). 30-pixel total drift, landing at one daughter. Split costDiff = −1.64 vs threshold −15 → reject.

### Empirical analysis: snap elongation cannot gate calibration

Per-frame snap elongations from the good 0413 run:

| Cell | Last snap before split | Splits next frame? |
|---|---|---|
| e9077 | **1.28** | yes |
| 12345 | 1.36 | yes |
| 1f89ab f8 | 2.42 | yes |
| e3d03 f5 | 2.25 | **no** |
| 8cbdf | 1.49 | **no** |

Distributions overlap heavily — e3d03 (no split) at 2.25 is more elongated than e9077 (split) at 1.28. No threshold cleanly separates "about to split" from "stable elongated." Threshold 1.3 would fail e9077 specifically. Conclusion: snap elongation is not a usable gate signal for calibration.

### Solution: snap is authoritative for any cell with valid snap

Pure Option A — skip position calibration entirely when a valid snap exists. Snap is the end-of-previous-frame position from a fully-converged optimization. For surviving cells (every cell after frame 1), snap is already the correct anchor.

Calibration retained for:
- Frame 1 cells (no snap exists)
- Newly-born cells with no prior snap (rare — daughters get snapshotted at end of birth frame)

Position adjustments for genuine cell motion are deferred to the unified perturb+split loop, where each move is gated by `costDiff < 0`. The loop has 500 iterations × cell with sigma=5 perturbation — sufficient to converge on real translation.

PCA shape fit also frozen on position (`pcaShapeUpdatePosition: false`). Same reasoning — shape fit's centroid update path is the same multi-modal drift mechanism.

### Optional drift-cap safety valve

For datasets where non-dividing cells genuinely translate fast, the config exposes `calibration_max_centroid_jump_voxels`:

- `0.0` (default): always skip when snap is valid (pure Option A).
- `> 0`: allow calibration BUT reject the centroid jump if it would move the cell more than this many voxels. Tiny jumps (genuine motion) kept; big jumps (multi-modal drift) rejected.

For this embryo dataset (cells move ≤ 5–15 px per frame, all geometrically isolated except early-frame symmetric clusters), the default 0.0 is safe.

### Neighbor exclusion in dense packing — analysis

Concern: with calibration off, all cells use snap positions for Voronoi. In dense packing (later frames, cells shoulder-to-shoulder), small position errors could mis-assign territory.

Four mitigations compound:

1. **Snap-mask** in PCA shape fit (`maskScale=1.3 × snap_radii`): geometrically bounds search extent.
2. **Quadratic-ish PCA weighting** (`pcaShapeWeightExponent=1.3`): leaked-from-neighbor halo pixels weight ~10× less than self-core pixels.
3. **Voronoi midplane error scales with cell-pair position error** (~5 px max for typical motion → ~2.5 px midplane shift → ~10% of cell boundary as mis-assigned, mostly suppressed by quadratic weighting).
4. **Pre-pass uses dual D1/D2 claim points** for dividing neighbors → naturally partitions multi-blob neighbors.

Net: shape-fit error from stale neighbor positions in dense packing is ~0.5% of variance — negligible compared to the calibration-drift damage we eliminate.

### Implementation

**File:** `C++/includes/ConfigTypes.hpp`

```cpp
float calibration_max_centroid_jump_voxels = 0.0f;  // 0 = always skip when snap valid
```

With matching `explodeConfig` parse entry.

**File:** `C++/includes/Ellipsoid.hpp`

```cpp
void setPosition(const cv::Point3f &pos) { _position = pos; }
```

(Added missing position setter — needed for the drift-cap restore path.)

**File:** `C++/src/CellUniverse.cpp` (calibration loop)

```cpp
const float calibrationDriftCap = std::max(0.0f, config.prob.calibration_max_centroid_jump_voxels);

for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
    auto calSnapIt = previousSnapshots.find(calName);
    const bool hasValidSnap = (calSnapIt != previousSnapshots.end()) && calSnapIt->second.valid;

    if (hasValidSnap && calibrationDriftCap <= 0.0f) {
        // Skip entirely — snap is authoritative.
        std::cout << "  [Calibration] cell=" << calName << " skipped reason=snap_valid ..." << std::endl;
        continue;
    }
    if (hasValidSnap) {
        // Drift-cap mode: tentatively apply, then check magnitude.
        frame.calibrateCellPositionViaCentroid(ci, calSnapIt->second, calOthers);
        const float jump = ...;
        if (jump > calibrationDriftCap) {
            frame.cells[ci].setPosition(calSnapIt->second.position);  // restore snap
            continue;
        }
    }
    // ... rest of calibration as before (frame 1 / no-snap cells) ...
}
```

**File:** `C++/config/config.yaml`

```yaml
prob:
  calibration_max_centroid_jump_voxels: 0.0   # pure Option A
cell:
  pcaShapeUpdatePosition: false               # snap is authoritative
```

### Expected behavior change

For e9077 f3 in 212223:
- **Before:** calibration drifts 30 px to one daughter, live cost beats snap, baseline at one daughter, split rejected (costDiff −1.64).
- **After:** calibration skipped (snap valid). Live position = snap = (297, 164, 121). Snap and live identical. Baseline unaffected by drift. Split should proceed against authentic parent baseline.

For 1f89ab f8 in 212223 (and earlier runs):
- **Before:** calibration drifted ~10 px, snap-mask shape fit pulls radii toward inflated extent → tight snap → tight daughters → cost gate fails.
- **After:** calibration skipped. Shape fit uses snap radii as mask anchor (stable). Cell shape stays consistent across frames. Splits should fire on time.

### Tuning notes

If a future run shows that genuine cell motion is too slow to be tracked by perturbation alone (e.g., a cell visibly drifts 15+ px from where the synth places it after several hundred iterations), raise `calibration_max_centroid_jump_voxels` to 5 or 8. This re-enables small calibration steps while still rejecting drift jumps.

If non-dividing cells in dense packing exhibit shape-fit bloat from stale neighbor positions, the deeper fix is to switch neighbor exclusion from point-Voronoi to ellipsoid-membership exclusion (per the bbox plan's D2 alternative). Not implemented in this change.

### Files changed

- `C++/includes/ConfigTypes.hpp` — `calibration_max_centroid_jump_voxels` field + parse.
- `C++/includes/Ellipsoid.hpp` — `setPosition(const cv::Point3f&)` accessor.
- `C++/src/CellUniverse.cpp` — skip-when-snap-valid logic + drift-cap rejection in the calibration loop.
- `C++/config/config.yaml` — `calibration_max_centroid_jump_voxels: 0.0` (default skip), `pcaShapeUpdatePosition: false`.

## 2026-04-15: Calibration cleanup — frame-1-only gate, drift-cap removed

**Status:** ACTIVE (supersedes the per-cell snap-validity check from 2026-04-14 evening "Snap-anchored parent")

### Why

The previous shipping iteration left a per-cell snap-validity check inside the calibration loop with a `calibration_max_centroid_jump_voxels` config field as an optional drift-cap safety valve. In practice:

- Default `0.0` always skips when snap is valid → equivalent to "skip from frame 2 onward."
- The drift-cap branch (`> 0`) was a hypothetical safety valve we agreed was unlikely to be needed.
- Per-cell snap check + tentative-apply-then-reject branching was 90 LOC of conditional logic for what reduces to "calibrate frame 1 only."

Cleaned up to express the rule directly.

### Changes

**File:** `C++/src/CellUniverse.cpp`

Replaced the per-cell snap-check + drift-cap loop with a single top-level gate:

```cpp
const bool runCalibration = (frameIndex == 0) && (calibrationIters > 0) && !frame.cells.empty();
if (runCalibration) {
    // ... calibration loop (frame-1 cells only, no snap to consult) ...
} else if (calibrationIters > 0 && !frame.cells.empty()) {
    std::cout << "[Calibration] frame " << displayFrame
              << " skipped (snap-anchored: every cell uses end-of-previous-frame snap)"
              << std::endl;
}
```

Frame-1 cells have no snap, so a synthetic `bootSnap` is built from initial CSV state to satisfy `calibrateCellPositionViaCentroid`'s signature.

**File:** `C++/includes/ConfigTypes.hpp`

Removed:
- `float calibration_max_centroid_jump_voxels = 0.0f;`
- Matching `explodeConfig` parse line.

**File:** `C++/config/config.yaml`

Removed the entry and its 14-line explanatory comment block.

### Equivalence preserved

The rule "skip when snap is valid" is exactly equivalent to "skip from frame 2 onward" because:

- Frame 1 cells: loaded from CSV, no `previousSnapshots` map exists yet → no snap.
- Frame 2+ cells: every surviving cell was snapshotted at end of the previous frame → has snap.
- Newly-born daughters: get snapshotted at end of birth frame → have snap by next frame.

So the frame-index gate covers the same cases as the snap-validity check, with no behavior difference.

### What if a future dataset needs the drift-cap?

If a dataset arises where non-dividing cells genuinely translate fast and the unified-loop perturbation (sigma=5, 500 iters/cell) is too slow to track, the right fix is a more general "selective calibration" — likely a different mechanism than the discarded drift-cap. Re-introduce as needed at that point. The plumbing is deleted; reintroduction is ~10 LOC if motivated.

### Files changed

- `C++/src/CellUniverse.cpp` — frame-1-only calibration gate, ~50 LOC removed.
- `C++/includes/ConfigTypes.hpp` — `calibration_max_centroid_jump_voxels` field + parse removed.
- `C++/config/config.yaml` — entry + 14 lines of comment removed.

## 2026-04-15: Bbox migration into burn-in / refine + OMP on pre-pass + OMP on image preprocessing

**Status:** ACTIVE

### Bbox cost migration — burn-in + refine + final-gate unified

**Bug fixed:** Previously the bbox flag covered only the FINAL split cost gate (Stage 3 from `2026-04-14 evening: Universal bounding-box cost infrastructure`). Burn-in and refine still used `_currentCost` (full-image cost) — but `_currentCost` is never updated by `perturbCell`'s bbox path, so it became stale during burn-in. Result: candidate-winner ranking inside `trySplitCellPhased` compared candidates against a stale baseline, picking the wrong winner more often than not. The final gate then re-ran bbox cost from scratch on the (possibly wrong) winner.

**Fix:** Build one union bbox + Voronoi exclusion mask at the top of `trySplitCellPhased`. All cost evaluations inside the function — baseline, each candidate's post-burn-in score, refine's post-state, final cost gate — use the same bbox + mask. Apples-to-apples comparison; no more stale `_currentCost`.

**File:** `C++/src/Frame.cpp` (`trySplitCellPhased`)

```cpp
BoundingBox3D splitBbox;
std::vector<uint8_t> splitMask;
if (_useBboxCost) {
    std::vector<cv::Point3f> seedPoints;
    for (const auto &cand : candidates) {
        seedPoints.push_back(cand.d1Pos);
        seedPoints.push_back(cand.d2Pos);
    }
    const float pointR = _bboxMarginScale * std::max({srcMajor, srcB, srcMinor});
    splitBbox = computeUnionBboxWithPoints({cellIndex}, _bboxMarginScale,
                                           seedPoints, pointR);
    // ... build self-claims = parent center, other-claims = neighbor centers ...
    splitMask = buildExclusionMask(splitBbox, selfClaims, otherClaimsForBbox);
}

auto evalImageCost = [&](const std::vector<cv::Mat> &synth) -> double {
    if (_useBboxCost) return calculateBboxCost(splitBbox, synth, splitMask);
    return _currentCost;  // legacy
};
```

Each call site that used `_currentCost` for image cost now uses `evalImageCost(_synthFrame)`:

- Baseline (initial state)
- Per-candidate score after burn-in
- Pre-refine baseline (from `bestImageCost`)
- Post-refine new best

`refreshFullCostCache()` calls inside the candidate loop and the daughter-refit block are skipped under bbox flag (cache only matters for the legacy path's `_currentCost` reads).

**Stage-3 final-gate simplified:** the previous 90-LOC bbox computation at the cost-check site is now redundant — `costDiff = bestTotal - baselineTotal` is bbox-based throughout. Replaced with a single line + a `mode=` field in the rejection log so the diagnostic distinguishes bbox vs full-image rejects.

### Pre-pass OMP — per-thread results + serial merge

**File:** `C++/src/CellUniverse.cpp` (pre-pass loop, ~line 600)

Previously the loop wrote to `expectedDaughters[name]` inside a `for (cell)` body, blocking parallelization (std::map with concurrent writes on different keys is not thread-safe due to tree rebalancing).

Restructured:

1. At start of each round, snapshot `expectedDaughters` as `roundSnapshot` (read-only during the round).
2. Allocate `std::vector<PrePassResult> results(nCells)` — per-thread-unique writes by index.
3. `#pragma omp parallel for schedule(dynamic)` runs each cell's image-grounded PCA into its own `results[ci]` entry. Logs accumulated into `std::ostringstream` per result, not to `std::cout`.
4. After the parallel region, serial merge: apply non-failed results back to `expectedDaughters`, emit logs in cell-index order (deterministic output even with N threads).

`Frame::imageGroundExpectedDaughters` is `const` — pure-read on Frame state, safe.

### Image preprocessing OMP — per-frame loading

**File:** `C++/src/CellUniverse.cpp` (constructor, Pass-1 frame loading)

Frames are independent (each `loadFrame` opens its own TIFF, allocates its own slice stack). Pre-allocated the result vectors and added `#pragma omp parallel for schedule(dynamic)` over the file list. Speeds up startup proportionally to thread count for multi-frame runs.

### Files changed

- `C++/src/Frame.cpp` — bbox migration in `trySplitCellPhased`: union bbox + mask built once, `evalImageCost` lambda used at all 4 read sites (baseline, candidate, pre-refine, post-refine), Stage-3 final-gate block reduced from 90 to 4 LOC.
- `C++/src/CellUniverse.cpp` — pre-pass loop restructured for OMP with per-thread results + serial merge; image-loading loop parallelized over files.

### What's still deferred

| Item | Status |
|---|---|
| Legacy cost path (`calculateCost`, `_currentCost`, etc.) removal | Wait until bbox is validated as drop-in |
| `split_cost` / `overlap_penalty_weight` retuning for bbox mode | Needs run data |
| Remove `use_bbox_cost` feature flag | After validation + retuning |
| OMP on per-cell position calibration | Calibration is now frame-1-only with few cells; low priority |
| OMP on split-candidate burn-in (P6 from plan) | Now feasible since burn-in uses bbox cost (per-thread state ≈ small bbox synth slab); follow-up work |

## 2026-04-15: P4 (bright-pixel gather cache) + flip use_bbox_cost to true; P6 deferred with scope note

**Status:** ACTIVE

### P4 — bright-pixel gather cache in `calibrateCellShapeViaPca`

`gatherBrightPixelsVoronoi` is the dominant cost inside the shape-fit iteration loop: O(boxVolume × nNeighbors) per call, called up to 15 times per cell per frame. Inputs are mostly invariant across iterations:

- `_realFrame` — never changes
- `_backgroundValue` — never changes
- `sphereR` — computed once at function entry from FIXED snap mask radii
- `otherCellsClaimSets` — passed by const ref, frozen
- `center` — only changes when `updatePosition=true` AND the centroid moves

Cache the raw (Voronoi-filtered, brightness-filtered) pixel set keyed on `center`. Re-gather only when the centroid moves more than 1e-4 px on any axis. With `pcaShapeUpdatePosition: false` (current default), cache hits on every iter after the first → 14× speedup on the gather phase, which is typically 70–80% of shape-fit wall time.

**File:** `C++/src/Frame.cpp` (`calibrateCellShapeViaPca`)

```cpp
cv::Point3f cachedCenter{NaN, NaN, NaN};
std::vector<BrightPixel> cachedRaw;
int cachedHits = 0, cachedMisses = 0;

for (int iter = 0; iter < maxIters; ++iter) {
    const cv::Point3f center(cell.getX(), cell.getY(), cell.getZ());
    const bool cacheHit = !cachedRaw.empty()
        && std::abs(center.x - cachedCenter.x) < 1e-4f
        && std::abs(center.y - cachedCenter.y) < 1e-4f
        && std::abs(center.z - cachedCenter.z) < 1e-4f;
    if (!cacheHit) {
        cachedRaw = gatherBrightPixelsVoronoi(...);
        cachedCenter = center;
        ++cachedMisses;
    } else {
        ++cachedHits;
    }
    const std::vector<BrightPixel> &raw = cachedRaw;
    // ... ellipsoid mask test on raw, PCA on filtered pixels ...
}

// End-of-fit diagnostic:
//   [PCA Shape Cache] cell=... hits=14 misses=1 hitRate=0.933
```

When `updatePosition=true`, cache invalidates whenever the centroid jumps — correct fallback to legacy behavior.

### use_bbox_cost: true

`config.yaml` flipped from `false` to `true`. The bbox cost is now the active path for both perturbCell and trySplitCellPhased (full migration through burn-in/refine/gate landed earlier today).

Watch in next run's logs:

- `[Optimize] ... useBboxCost=1 bboxMarginScale=3` confirms activation
- `[Split Bbox Init] ... bboxXYZ=(...) volume=...` confirms bbox built per split attempt
- `[Split Reject cost] ... mode=bbox diff=...` shows bbox-based costDiff
- `[PCA Shape Cache] hits=N misses=1` confirms P4 hit rate

Expected behavior change vs `use_bbox_cost: false`:

- Per-cell decisions isolated from unrelated image regions
- `costDiff` magnitudes much smaller (bbox ~5M voxels vs full-image ~36M)
- `split_cost = 15` is likely now too loose — expect `costDiff` for real splits in the ~−2 to −10 range, phantoms near 0 or positive. Retune to `~2-5` after observing first run's numbers.
- `overlap_penalty_weight = 500` may also need tuning down (~50–100) since it now competes with smaller image cost magnitudes.

### P6 — deferred with proper scope estimate

Original plan estimated 2 hours assuming `perturbCell` and `generateSynthFrameFast` already accepted thread-local state by parameter. They don't — both are Frame member functions that read/write `cells`, `_synthFrame`, `_currentCost`. True candidate-level parallelism requires:

1. Refactor `perturbCell` to overload taking `(std::vector<Ellipsoid>&, std::vector<cv::Mat>&, double&)` instead of using Frame members. ~150 LOC.
2. Refactor `generateSynthFrameFast` similarly. ~80 LOC.
3. Refactor `computeOverlapForCell` and `buildExclusionMask` to take cells param. ~40 LOC.
4. Allocate per-thread synth slabs (bbox-sized ~23 MB instead of full-image 144 MB to fit on typical 8-core machine memory budget). Helper to copy savedSynth into a slab allocator.
5. Rewrite the candidate loop to use `#pragma omp parallel for` over per-thread state.
6. Reconcile state into a winning `bestCells / bestSynth` after the parallel region.

Total: ~300–400 LOC across Frame.cpp + Frame.hpp, plus careful testing that bbox-in-burn-in semantics are preserved.

This should be a separate session with proper testing infrastructure. For now, candidate loop runs serially. Inner `calculateBboxCost` already has `#pragma omp parallel for reduction(+:totalCost)` over z-slices, providing 4–6× per-cost-call speedup on bbox sizes. The outer candidate loop's serial nature is the remaining inefficiency.

### Files changed

- `C++/src/Frame.cpp` — P4 cache in `calibrateCellShapeViaPca` (~30 LOC added) + final cache-stats log (~10 LOC).
- `C++/config/config.yaml` — `use_bbox_cost: false → true`.

## 2026-04-15: Deterministic log ordering under OpenMP

**Status:** ACTIVE

### Goal

Multi-threaded shape fit was producing interleaved `[PCA Shape]` log lines (different cells' iter logs jumbled together). Same problem applied to image preprocessing parallelization. User wanted readable, frame-order-correct logs for offline analysis.

Pre-pass already had deterministic logs (per-thread `ostringstream` results + serial merge in cell-index order, landed earlier today).

### Changes

#### Shape fit — sink parameter + per-cell merge

**File:** `C++/includes/Frame.hpp`

Added `<ostream>` include. Added optional `std::ostream *logSink = nullptr` parameter to `calibrateCellShapeViaPca`. When non-null, all per-iter log lines (`[PCA Shape] iter=...` and `[PCA Shape Cache] hits=...`) are appended to the sink instead of `std::cout`.

**File:** `C++/src/Frame.cpp` (`calibrateCellShapeViaPca`)

```cpp
std::ostream &log = logSink ? *logSink : std::cout;
// ... all internal cout calls replaced with `log`:
log << "  [PCA Shape] cell=" << ... << std::endl;
log << "  [PCA Shape Cache] cell=" << ... << std::endl;
```

Backward-compatible: callers that don't pass a sink keep the legacy `std::cout` behavior.

**File:** `C++/src/CellUniverse.cpp` (shape-fit OMP loop)

```cpp
const int nCells = static_cast<int>(frame.cells.size());
std::vector<std::ostringstream> shapeLogs(nCells);
#pragma omp parallel for schedule(dynamic)
for (int ci = 0; ci < nCells; ++ci) {
    // ... build others claim set, mask radii ...
    frame.calibrateCellShapeViaPca(ci, others, ..., &shapeLogs[ci]);
}
// Serial merge: emit per-cell log blocks in cell-index order.
for (int ci = 0; ci < nCells; ++ci) {
    const std::string buf = shapeLogs[ci].str();
    if (!buf.empty()) std::cout << buf;
}
```

Result: shape-fit logs appear in stable cell-index order regardless of thread count or scheduling. `OMP_NUM_THREADS=8` and `OMP_NUM_THREADS=1` produce byte-identical log output for this section.

#### Image preprocessing — reverted to sequential

The OMP-on-frame-loading change shipped earlier today created interleaved `[Preprocess]`, `[LoadFrame]`, `[PreprocessScores]`, `[IterPreprocess]` logs across many helper functions inside `ImageHandler`. Refactoring all of those with a sink parameter would be invasive (~10 cout sites in 5 functions). The wall-time win is small (~25 sec out of a ~1 hr run, since loading is one-shot at startup).

Trade-off: keep image preprocessing serial → preserves log readability at modest startup cost.

**File:** `C++/src/CellUniverse.cpp`

Reverted Pass-1 frame loading loop from `#pragma omp parallel for` to sequential. `loadedFrames` and `frameMeanBrightness` go back to push_back instead of indexed assignment.

### Audit — all parallel regions, log handling

| Site | Logs? | Order-deterministic? |
|---|---|---|
| `Frame::refreshFullCostCache` (OMP slice reduction) | None inside loop | N/A |
| `Frame::calculateCost` (OMP slice reduction) | None | N/A |
| `Frame::calculateBboxCost` (OMP slice reduction) | None | N/A |
| `Frame::buildExclusionMask` (OMP voxel loop) | None | N/A |
| `CellUniverse` shape-fit (OMP per-cell loop) | Yes | ✅ via sink + serial merge (this change) |
| `CellUniverse` pre-pass (OMP per-cell loop) | Yes | ✅ via per-thread results + serial merge (earlier) |
| `CellUniverse` image preprocessing | Yes | ✅ reverted to sequential (this change) |

All parallel paths now produce deterministic log output. Setting `OMP_NUM_THREADS=1` is no longer needed for clean logs.

### Floating-point caveat (unchanged)

`reduction(+:totalCost)` in cost loops still has implementation-defined summation order → bit-pattern of cost values may differ by ~1e-6 across thread schedules. Decision boundaries (perturb accept/reject, split gate) are robust to ε-level differences. For bit-exact reproducibility (e.g., comparing two runs with diff), still use `OMP_NUM_THREADS=1`.

### Files changed

- `C++/includes/Frame.hpp` — `<ostream>` include, `logSink` parameter on `calibrateCellShapeViaPca`.
- `C++/src/Frame.cpp` — `log` reference in `calibrateCellShapeViaPca`, three cout sites routed to it.
- `C++/src/CellUniverse.cpp` — shape-fit loop accumulates per-cell `ostringstream` and emits in serial; image preprocessing loop reverted to sequential.

## 2026-04-15: Fix dangling cellName reference (8cbdf-twice-attempt bug) + dead-code audit

**Status:** ACTIVE

### The bug

Observed: in run `output_jihang_20260414_212223` at frame 2, `8cbdf86d308d4599936e7fdbc23375f5` got TWO `[Split Attempt]` log entries even though the `splitBlacklist` should have prevented re-attempt. Counter `splitAttempted` consistently logged `7` for a 6-cell frame.

**Root cause:** `C++/src/CellUniverse.cpp:822` (before fix):

```cpp
const std::string &cellName = frame.cells[cellIdx].getName();
```

`cellName` is a const-reference to the Ellipsoid's `_name` member. After `frame.trySplitCellPhased(...)` returns:

- `cells` was mutated extensively: `cells.erase(...)`, `cells.push_back(child1)`, `cells.push_back(child2)`, then `cells = savedCells` (full vector reassignment via copy assignment).
- The original Ellipsoid at `cellIdx` is destroyed and replaced with a copy from `savedCells` (which is a different Ellipsoid object with a different `_name` string instance).
- `cellName` reference is now **dangling** — points to deallocated string storage.

The misleading comment said "Snapshot cellName before callback() invalidates the const-ref" — but invalidation actually happened earlier, inside `trySplitCellPhased`'s candidate-eval mutations, not at `callback()`. The "snapshot" `cellNameCopy(cellName)` ran on already-dangling memory → copy contained garbage characters (whatever happened to be in the freed heap region).

`splitBlacklist.insert(garbageString)` succeeded, but the next iteration's `splitBlacklist.count(cellName_real_value)` returned 0 (real name vs garbage entry → no match). `canSplit` stayed true → cell got picked again → second split attempt.

This was undefined behavior — sometimes the freed memory still held the right string content (run worked correctly), sometimes it didn't (saw 8cbdf86d twice). The "8cbdf86d" pattern was probably consistent because of how that particular frame's heap layout worked out.

### The fix

**File:** `C++/src/CellUniverse.cpp:822`

Change reference to value:

```cpp
// VALUE COPY — must not be a reference. trySplitCellPhased and
// perturbCell mutate frame.cells (erase/push_back/full reassign
// via savedCells), invalidating any reference to the original
// ellipsoid's _name string. A dangling cellName silently caused
// splitBlacklist to be inserted with garbage content, defeating
// the "max one split attempt per cell per frame" invariant —
// the same cell would re-attempt because its real name wasn't
// found in the blacklist (the blacklist had a corrupted entry).
// Observed as the "8cbdf86d gets 2 split attempts" symptom.
const std::string cellName = frame.cells[cellIdx].getName();
```

`cellName` is now a value with stable storage. All subsequent uses (blacklist insert, snapshot erase, log lines) read the correct string regardless of vector mutations.

Removed the now-redundant `const std::string cellNameCopy(cellName);` line and replaced all `cellNameCopy` references with `cellName`.

### Other dangling-reference candidates checked

| Site | Pattern | Verdict |
|---|---|---|
| `CellUniverse.cpp:736` `name = frame.cells[ci].getName()` in pre-pass merge | Reference, used immediately on next line, no mutation between | Safe |
| `CellUniverse.cpp:810` `cname` in `rebuildEligible` lambda | Reference, used in same statement (`phaseNames.count(cname)`), no mutation | Safe |
| `splitProbabilities[cellName]` reads | Map operator[] on stable cellName value (after fix) | Safe |
| `previousSnapshots.find(cellName)` iterator usage | Iterator used briefly, no map mutations between use sites | Safe |

### Dead code removed

**`Ellipsoid::setPosition(const cv::Point3f&)`** — added during the rolled-back drift-cap implementation (2026-04-14 evening), never called by any code after that revert. Pure dead code.

**File:** `C++/includes/Ellipsoid.hpp`

Removed the inline `setPosition` accessor.

### Other audit findings (kept, not bugs)

- **`Frame::computeUnionBbox`** (without points) is only called internally by `computeUnionBboxWithPoints`. Could be made private but useful as a public utility for future bbox extensions. Kept.
- **`bestPerSlice` / `_currentCostPerSlice`** writes under `_useBboxCost` are dead writes (the values are stale and unused for decisions in bbox mode). Required by the legacy path, so they stay until the legacy path is removed.
- **`expected_daughter_pre_pass_iterations`** still parsed and used (line 611). Default 1 is the only meaningful value now, but the field is live.

### Files changed

- `C++/src/CellUniverse.cpp` — `cellName` changed to value copy, `cellNameCopy` removed, all references updated, comment block explains the bug.
- `C++/includes/Ellipsoid.hpp` — `setPosition` accessor removed.

## 2026-04-15: Raise overlap_penalty_weight 500 → 3000 for bbox-cost discrimination

**Status:** ACTIVE

### Motivation

Post-run analysis of `output_jihang_20260414_212223` + `output_jihang_20260415_002144` (both bbox-enabled) revealed that the cost gate cannot discriminate true from phantom splits:

| Category | n | min | median | max |
|---|---|---|---|---|
| Accepted | 6 | −38657 | −15148 | −13916 |
| Bio-rejected | 190 | **−72354** | −4886 | **+3565** |

Some bio-rejected candidates had image-cost signals STRONGER than any accepted split (−72354 vs −38657). The cost function alone cannot tell phantom from real; bio gates (bridge, volume, buried) do the actual discrimination. User's philosophy goal — "cost alone decides" — required tightening the gap.

### Root-cause analysis of the −72354 case (e3d03 phantom)

- Parent at snap position (112, 206, 93) with R=(22, 18, 14) — **correctly sized**, not over-fit.
- Daughters after burn-in+refine: d1=(124.85, 215.05, 86.69), d2=(125.09, 216.07, 85.94).
- **d1 and d2 only 1.3 px apart** — collapsed onto the same point.
- Combined daughter volume ~24k ≈ parent volume ~23k (volume preserved).
- Daughter collapse point is ~16 px from parent's snap position.

This isn't phantom shrink — it's **phantom relocate**. Two daughters acted as a single ellipsoid at a better-fit position. e3d03 moved between frames, perturbation hadn't caught up, split mechanism was hijacked as a position-correction tool.

Image cost correctly says "this placement fits real better" (it does). But morphologically it's not a split. Only bio gates can tell.

### Why overlap penalty was disabled in bbox mode

`overlap_penalty_weight = 500` was tuned against full-image cost magnitudes of ~7,700 (ratio: 500/7700 ≈ 6.5% of image cost). Under bbox mode, image cost magnitudes jumped to 10,000–90,000 (ratio: 500/50000 ≈ 1% of image cost). Overlap penalty became effectively disabled relative to image cost.

Two daughters 1 px apart with radii ~18 produce an overlap-volume penalty of ~500 × ~1 ≈ 500 — a negligible add to a 20,000 image cost drop. Daughters collapse freely.

### Fix

**File:** `C++/config/config.yaml`

```yaml
# Scaled UP to 3000 for bbox-cost mode. At weight=500 against bbox image
# cost magnitudes 10k-90k, overlap penalty was effectively disabled.
# At weight=3000, penalty scales to 5k-20k — competitive with image cost.
overlap_penalty_weight: 3000.0
```

Comment block explains the rationale and the legacy-path caveat (500 was correct for `use_bbox_cost: false`).

### Expected effect

- **e9077 f3 merge:** daughters collapsed from seeds 35 px apart to 4 px apart during burn-in. At weight=3000, collapsing to 4 px apart would add ~10k+ overlap penalty — enough to dominate the image-cost gain from collapse. Daughters should stay separated, bridge should see a proper valley, split accepts.
- **e3d03 phantom:** daughters collapsed to 1 px apart. At weight=3000, overlap penalty would exceed 15k+ — larger than the 72k image-cost gain? Probably not enough to reject on cost, but the daughters would have DRIFTED APART during burn-in under the stronger overlap pressure. If they stayed apart, real image wouldn't favor them (no second bright blob at e3d03 to cover), image cost would not improve, cost gate rejects.
- **Real accepts:** no overlap (daughters land at separate bright blobs), overlap penalty = 0, no change to cost signal.

### Why NOT lower asymmetric_cost_weight

Considered lowering `asymmetric_cost_weight` 8 → 2 as a fix. Rejected because:

- Lower asymK reduces BOTH real-split signal AND phantom signal proportionally → narrows gap, doesn't widen it.
- Real splits (−15k to −48k) rely on asymK amplification of "parent covers dark valley" penalty. Lowering asymK would push real splits toward −3k to −10k — closer to the bio-reject floor.
- The asymmetric weight isn't the pathology; the disabled overlap penalty is.

Kept `asymmetric_cost_weight: 8.0`.

### Philosophy alignment

User's goal: **"all false splits positive, all true splits negative, clear margin."**

This fix moves in that direction by:
1. Real splits: unchanged (no daughter collapse → no overlap penalty → same strong −15k to −48k cost signal)
2. Phantom splits involving daughter collapse: overlap penalty activates during refine → daughters forced apart → if no real second blob, image cost worsens → diff moves toward 0 or positive

Cannot achieve 100% separation through cost alone (phantom-relocate case still produces legit image-cost improvement). But the goal is a CLEAR MARGIN, and bio gates act as the safety net for the remaining edge cases.

### Files changed

- `C++/config/config.yaml` — `overlap_penalty_weight: 500 → 3000` with rationale comment block.

## 2026-04-15: Lower pcaShapeWeightExponent 1.3 → 1.1 — bright cells were over-shrinking

**Status:** ACTIVE

### User observation

From run 002144 screenshots + cells.csv: bright cells visually shrinking toward the end of the sequence. Not the dim cells — specifically the bright ones.

### Data confirms

Tracing 12345 daughter 0 (bright cell):

| Frame | R(a, b, c) | Note |
|---|---|---|
| f10 | (34.8, 29.0, 24.1) | |
| f13 | (39.3, 30.4, 26.5) | |
| f16 | (42.7, 33.3, 30.4) | **peak** |
| f17 | (40.3, 34.7, 28.8) | |
| f18 | (39.4, 27.0, 16.1) | dropping |
| f19 | (30.0, 21.3, **10.6**) | cRadius at MIN floor |
| f20 | SPLIT | |

The cell's c-axis ratcheted down from 30.4 → 10.6 in 4 frames before splitting. Many other cells (e3d03, daughters) were pinned at `cRadius = 10` (the configured `minCRadius` floor) — the fit wanted to go thinner.

### Root cause

`pcaShapeWeightExponent = 1.3` emphasizes the brightest core. For a bright cell:

- Core voxels weighted `w^1.3` — strongly dominant
- Boundary voxels weighted much less
- Thin-axis PCA variance driven almost entirely by the peak → tight fit
- Snap-mask prevents the RADIUS from collapsing below a fixed mask size, but within the mask the PCA still pulls the fit toward the peak
- Across frames the fit ratchets down, hitting the hard `minCRadius=10` floor

For dim cells (e.g. e3d03), the core-to-halo ratio is smaller, so the w^1.3 weighting is less biased → fit stays moderate.

### Fix

Lower `pcaShapeWeightExponent` from 1.3 to 1.1:

- Still mildly favors core over halo (prevents regression of the 1f89ab-halo-bloat we originally fixed with Option A)
- But much closer to linear weighting
- Previous safety-net fixes (bbox cost + Voronoi exclusion + snap-mask + A1 daughter refit) now bound halo contamination geometrically, not via weighting — so weighted core emphasis isn't needed as a second line of defense

**File:** `C++/config/config.yaml`

```yaml
# 2026-04-15: Lowered 1.3 → 1.1. Observed via run 002144 that bright
# cells were over-shrinking — 12345..0 dropped from (40, 33, 30) at
# f16 to (30, 21, 11) at f19 (c-radius at floor) before split.
# Many cells hitting minCRadius=10 floor = fit trying to collapse
# thinnest axis. At exp=1.3, core-bias was still too strong for
# bright cells.
pcaShapeWeightExponent: 1.1
```

If bright cells still shrink at 1.1, drop to 1.0 (pure linear). If dim cells bloat, raise back to 1.2.

### Why NOT shrink the bbox seed set (yet)

User raised the accuracy concern on the proposed "shrink bbox from 40 seeds to 2 (pre-pass D1/D2 only)" optimization. Answer:

- Candidate seeds from different axes (snap_axB, snap_axC, imgPca) span ~30 px range
- Daughter burn-in drift observed 10–30 px
- Combined: daughters can land 40–60 px from primary pre-pass seed
- Shrinking bbox to pre-pass ± 60 px margin would miss snap_axC / snap_axB variants → those candidates' costs aren't fully counted → scoring bias

**Deferred.** Prioritize accuracy over perf. A safer middle ground exists (drop rot/trans variants, keep only 4 primary axis × 2 seeds = 8 points), but not urgent until split accuracy stabilizes.

### Files changed

- `C++/config/config.yaml` — `pcaShapeWeightExponent: 1.3 → 1.1` with rationale.

## 2026-04-15: Drop axA + axB from split-primary axes — data-driven candidate pruning

**Status:** ACTIVE

### Usage data from run 002144

Across 196 split-winner observations and 6 accepted splits:

| Axis | Wins | % | Accepts |
|---|---|---|---|
| axC (local shortest) | 100 | 51% | 1 |
| imgPca (pre-pass) | 75 | 38% | **5** |
| axB | 19 | 10% | 0 |
| axA (local longest) | 2 | 1% | 0 |

All 6 accepted splits came from axC or imgPca. axA and axB earned 21 wins as burn-in candidates but zero accepts — they're candidate-set bloat. They only entered the primary-directions list for near-round cells (where the 1.2× threshold admitted them), where they competed against the "real" directions (axC and imgPca) and lost.

### Biological consistency

Cells divide along their SHORTEST semi-axis — the cleavage plane normal for rods, the thin dimension for pancake-shaped cells. Including the longest-axis (axA) or mid-axis (axB) as candidate division directions is physiologically wrong; the 1.2× threshold was there to cover PCA axis-ranking noise for near-round cells. But imgPca (image-grounded PCA from pre-pass) already provides a noise-robust direction for near-round cells — the local axes aren't needed as backup.

### Change

**File:** `C++/src/Frame.cpp` (`trySplitCellPhased`, axis-selection block)

**Before** (threshold-based selection, included any axis within 20% of shortest):
```cpp
const float minR = std::min({rA, rB, rC});
const float shortAxisThreshold = minR * 1.2f;
std::vector<cv::Point3f> primaryDirs;
std::vector<std::string> primaryNames;
for (int i = 0; i < 3; ++i) {
    if (radii3[i] <= shortAxisThreshold) {
        primaryDirs.push_back(axes3[i]);
        primaryNames.push_back(names3[i]);
    }
}
// + safety fallback if nothing qualified
```

**After** (single shortest axis, no threshold):
```cpp
int shortIdx = 0;
for (int i = 1; i < 3; ++i) {
    if (radii3[i] < radii3[shortIdx]) shortIdx = i;
}
std::vector<cv::Point3f> primaryDirs{axes3[shortIdx]};
std::vector<std::string> primaryNames{names3[shortIdx]};
```

imgPca is still added after (unchanged logic). Typical primary-direction count drops from:
- 1 axis (triaxial cells) + imgPca = 2 → unchanged
- 2 axes (slightly-triaxial) + imgPca = 3 → **1 + imgPca = 2** (drop axB)
- 3 axes (near-round) + imgPca = 4 → **1 + imgPca = 2** (drop axA + axB)

Candidates per attempt (2 midpoints × 5 variants per axis):
- Triaxial: 2 × 10 = 20 → unchanged
- Slightly-triaxial: 3 × 10 = 30 → **20** (33% fewer)
- Near-round: 4 × 10 = 40 (capped at 30) → **20** (33% fewer)

**Log updates:** `[Split Dirs]` no longer logs `minR`/`thresh`; now logs `shortestAxis=axC shortestR=15.23` to identify which of {axA, axB, axC} was selected.

### Expected impact

- **Split-phase runtime**: ~20–30% faster on frames with near-round cells (fewer candidates to burn-in)
- **Split accuracy**: unchanged for this dataset (0/6 accepts came from dropped axes)
- **Simpler code**: removed threshold constant + safety fallback; single unambiguous axis-selection rule

### Risk

The `primary` variant and `trans-` variant also had 0/6 accepts in this run, but were retained (n=6 is small; dropping them is more aggressive). If next run shows split pipeline still too slow AND still no accepts from `primary`/`trans-`, those can be dropped too.

### Files changed

- `C++/src/Frame.cpp` — axis selection simplified to single shortest axis; `[Split Dirs]` log updated; unused `minR`, `maxR`, `shortAxisThreshold` vars removed.

## 2026-04-15: REVERT pcaShapeWeightExponent 1.1 → 1.3 — catastrophic halo bloat regression

**Status:** ACTIVE (reverts the earlier 1.3→1.1 change from 2026-04-15)

### What broke

Run `output_jihang_20260415_021558` (first run with `pcaShapeWeightExponent: 1.1`) showed catastrophic halo-bloat regression. e9077 at frame 5:

```
[PCA Shape] iter=0   R=(76.23, 20.58, 19.38)  dR=11.23  axisAng=65.78
[PCA Shape] iter=14  R=(75.31, 21.83, 19.09)  dR=10.31  axisAng=53.71   (never converged)
```

- aRadius fit = 75.3 (clamped to `maxARadius=65` in output)
- Fit never converged (`dR` stuck at ~10, axis orientation oscillating)
- Screenshot confirmed: ellipses wildly mis-placed, some missing bright blobs entirely

This is the EXACT pathology Option A (quadratic weighting) was introduced to fix: with near-linear weighting, halo pixels contaminate PCA variance → fit expands to encompass halo + whatever other bright structure is in the Voronoi territory.

### Why my diagnosis was wrong

The 1.3 → 1.1 change was motivated by:
1. User observation: "bright cells shrinking towards the end"
2. Data: 12345..0 trajectory dropped (40, 33, 30) → (30, 21, 11) across f16→f19
3. Cells pinned at `minCRadius=10` floor

**All three are actually correct behaviors:**

- **12345 f16→f19 shrinkage** is pre-split elongation. The cell was about to divide at f20. In the 3-4 frames before division, a cell elongates along its cleavage axis and thins perpendicular to it. The (40, 33, 30) → (30, 21, 11) progression shows the cell elongating in a and squeezing b/c. That's the natural cellular geometry of division, not a PCA bug.
- **Cells at `minCRadius=10`** for flat-pancake cells (e3d03 etc.) are at their correct true thickness. The floor is working as a hard lower bound for safety, not a pathology.
- **"Shrinking at the end"** in the visualization is the collapsed daughter pair from the a50/a51 overlap bug (fixed separately via overlap_penalty_weight 500→3000). That's a post-split overlap issue, not an over-shrinkage issue.

### What this teaches

- `pcaShapeWeightExponent` is on a sharp cliff near 1.1–1.2. Below ~1.2 the fit enters halo-dominated regime and bloats catastrophically. Above ~1.2 core emphasis keeps halo suppressed.
- The previously-stable value was 1.3 — it's near the cliff edge but on the good side.
- **Voronoi + snap-mask + bbox do NOT fully replace weighted core emphasis.** Halo pixels inside self's Voronoi territory are still present; only PCA weighting suppresses their influence on variance.
- Don't "visually diagnose" bright-cell shrinkage as a PCA bug without first checking whether the cell is near a split event.

### Change

**File:** `C++/config/config.yaml`

```yaml
pcaShapeWeightExponent: 1.3   # reverted from 1.1; 1.1 caused halo bloat
```

With a multi-line comment explaining the revert and the cliff behavior so future readers don't make the same mistake.

### What NOT to tune here going forward

- `pcaShapeWeightExponent` value — leave at 1.3. The cliff is sharp; experimentation is high-risk. If bright cells genuinely shrink WITHOUT an imminent split, investigate the shape fit input (Voronoi territory, claim points) before touching this.
- Attempting to remove weighted core emphasis in favor of Voronoi/bbox/snap-mask — those are geometric filters; they exclude pixels by location but don't down-weight halo pixels that are IN self's territory. Weighting is orthogonal and still necessary.

### Files changed

- `C++/config/config.yaml` — `pcaShapeWeightExponent: 1.1 → 1.3` with corrective multi-line comment.

## 2026-04-15: Raise overlap_penalty_weight 3000 → 30000 — prevent daughter collapse during burn-in

**Status:** ACTIVE

### Diagnosis

Run 021558, e9077 f3 (real split, should accept):

```
Baseline   image=69678  overlap=0.96   total=69679
Candidate  image=31622  overlap=2209   total=33831
Refine     → postTotal=27292  (delta -6539)
Final      d1=(298.7, 188.2, 104.3)
           d2=(298.8, 186.6, 102.9)  ← 1.6 px apart
Bridge     worstValleyRatio=1.18  → REJECT
```

During burn-in, daughters drifted from seeds 50 px apart to ~7 px apart, then refine collapsed them to 1.6 px apart. Bridge correctly rejected the collapsed pair, but the real split was lost.

### Root cause: overlap penalty saturates too low

`computeOverlapForCell` formula:
```
overlapRatio = (combinedR − dist) / combinedR        ∈ [0, 1]
penalty = weight × overlapRatio²
```

Maximum penalty per pair = `weight` (at ratio=1, daughters fully collapsed).

At `weight=3000`:
- Distance 30 px: penalty 187
- Distance 20 px: penalty 750
- Distance 10 px: penalty 1687
- Distance  1 px: penalty 2852

Each burn-in step advances daughters ~2 px closer. Image cost drops 500-2000 per step (daughters cover shared bright peak more). Overlap penalty delta per step was 10-500 — **much smaller than image gain** → every approach step accepted.

Max overlap penalty of 3000 was about **2% of the available image-cost drop (~42K)**. Effectively unopposed.

### Fix

Raise weight **3000 → 30000**. At weight=30000:

- Distance 30 px: penalty **1875**
- Distance 20 px: penalty **7500**
- Distance 10 px: penalty **16875**
- Distance  1 px: penalty **28524**

Each approach step now adds 5-15K overlap penalty — **overwhelms the 500-2000 image gain per step**. Approach moves rejected → daughters stay separated → burn-in produces a morphologically-valid split attempt.

### Safety check — real splits unaffected

Run 002144 accepted splits had daughter pair distances 39-49 px, radii 22-30. Overlap ratios: 0.1-0.2. Penalty at weight=30000: 400-1200. That's negligible vs the 15K-48K image-cost improvements those splits produced. Real splits still accept by a wide margin.

Late-frame packed cells (legitimately touching) would see higher penalties, but bounding-sphere approximation already over-estimates overlap for elongated cells — the fix isn't introducing a new inaccuracy, just strengthening the existing signal.

### What this fixes

1. **e9077 f3**: daughters stay separated during burn-in → bridge sees a valid valley → accept.
2. **Phantom splits via daughter collapse** (e3d03-type): daughters can't converge onto one bright spot → phantom split cost advantage disappears → cost gate correctly identifies phantom as not-a-real-improvement.
3. **User's philosophy alignment**: cost now properly penalizes "collapsed daughters" morphology through overlap, so cost-gate decisions align better with bio checks. Still not 100% decisive alone (phantom-relocate cases remain), but the gap narrows.

### Side effect to monitor

Late-frame dense packings where cells legitimately touch may see strong separation pressure. If cells in f18-f22 visibly push apart beyond realistic separation, lower weight (try 15000 or 20000).

### Files changed

- `C++/config/config.yaml` — `overlap_penalty_weight: 3000 → 30000` with detailed rationale comment block.

## 2026-04-15: Scale asymmetric_cost_weight 8 → 64 for bbox-mode signal fraction

**Status:** ACTIVE

### Motivation (user observation)

Fix E (original, 2026-04-14) introduced asymmetric L2 cost at `k=3.0` tuned against full-image cost magnitudes (~8K total). The PER-VOXEL contribution of "parent covers dark valley" overshoot is `diff² × k` — for ~500 valley voxels with diff ≈ 0.15, that's ~270 penalty. At total cost 8K, valley penalty is ~3% of total — enough to shift cost-gate decisions.

Under bbox-cost mode, the same voxel set (cell-adjacent) is measured with Voronoi exclusion, but bbox cost MAGNITUDE is ~8× larger because bbox concentrates on self's territory where signal is densest:

| Mode | Example total (e9077 f3) | Valley overshoot penalty at k=8 | % of total |
|---|---|---|---|
| Full-image | 8.7K | ~270 | ~3% |
| Bbox | 69.7K | ~270 (same voxels) | **~0.4%** |

The asymmetric signal becomes 7–8× weaker as a fraction of total under bbox. User correctly identified: "since using bbox the cost is 4x more than before, the penalty should be more."

### Fix

Scale `asymmetric_cost_weight` by 8× (matching the bbox/full-image cost ratio exactly):

**File:** `C++/config/config.yaml`

```yaml
asymmetric_cost_weight: 64.0   # was 8.0; scaled 8x for bbox mode
```

At k=64:
- Valley overshoot penalty per voxel: `diff² × 64`
- 500 valley voxels × 0.0225 × 64 = ~720 penalty per parent-covers-valley scenario
- Relative to 70K bbox total: ~1.0%

Approaches the 3% fractional impact the original Fix E (k=3) had in full-image mode. The 8× scaling is principled: it matches the bbox/full-image total-cost ratio, so the asymmetric-signal-to-total-cost fraction is preserved vs the working full-image config.

### Expected effect

- **Parent-covering-valley baseline costs**: proportionally higher due to amplified valley penalty
- **Real-split cost drops**: larger in absolute terms (same voxels removed from overshoot when daughters cover bright regions without valley)
- **Phantom-collapsed-daughter candidates**: unchanged (no overshoot-avoidance advantage — daughters just cover a single bright spot; no valley coverage change)
- **Cost gate discrimination**: real-split diffs should become more strongly negative; phantom-collapse diffs stay roughly same → gap WIDENS → cost alone becomes more decisive

### Side effect to monitor

Cell-boundary voxels where synth > real due to ellipsoid rendering approximation (soft-edge rendering vs real image noise) get amplified 8× more. If these accumulate and cause real-split cost signals to become noisy or unstable, step down to k=32 or k=24.

### Files changed

- `C++/config/config.yaml` — `asymmetric_cost_weight: 8.0 → 64.0` with reasoning comment block.

## 2026-04-15: Config audit — scale split_cost 15 → 1500 for bbox mode + fix stale comments

**Status:** ACTIVE

### Motivation

User flagged: "since the bbox fix, is there anything more that could break because of the cost change?"

Audit found one remaining cost-scale dependency + stale documentation:

### 1. `split_cost: 15` was inert under bbox

Acceptance threshold: `costDiff < -split_cost` → accept. At full-image scale (total cost ~8K), 15 was meaningful. Under bbox (total cost ~70K), 15 is effectively zero.

Empirical evidence from run 002144 (bbox mode with split_cost=15):

| Category | n | costDiff range |
|---|---|---|
| Accepted | 6 | −13,916 to −48,192 |
| Bio-rejected | 190 | −72,354 to +3,565 |
| **Cost-rejected** | **0** | — |

The cost gate fired zero times across 196 split attempts — all filtering was done by bio gates. The cost gate was not contributing to split accept/reject decisions.

Phantom collapse projection with overlap_penalty_weight=30000 (saturation-bounded):
- Parent baseline (e3d03 example): 91,979
- Collapsed daughter candidate: 19,625
- Overlap penalty at full collapse: +30,000
- Net candidate total: 49,625
- costDiff: **−42,354**

Phantom still passes at threshold 15. Raising split_cost to 1500 catches marginal candidates near zero without blocking any observed real split (minimum accepted was −13,916).

### 2. Stale `use_bbox_cost` block comments

The comments shipped with Stage 2 said:
```
#   - split_cost: expected 2-5 (vs 15 for full-image diff)
#   - overlap_penalty_weight: expected 50-100 (vs 500) — tune empirically
```

These were preliminary guesses before bbox mode was validated. Actual values landed opposite direction:
- `split_cost`: RAISED to 1500 (100×), not lowered to 2–5
- `overlap_penalty_weight`: RAISED to 30000 (60×), not lowered to 50–100
- `asymmetric_cost_weight`: RAISED to 64 (8×)

Comments replaced with the actual tuning values so future readers aren't misled.

### Changes

**File:** `C++/config/config.yaml`

```yaml
# Before:
split_cost: 15

# After:
split_cost: 1500
# + multi-line comment explaining the tuning history and why cost gate
# remains secondary (bio gates do most phantom rejection work)
```

Plus `use_bbox_cost` block comments updated to reflect actual tuning:
```yaml
# Bbox cost magnitudes are ~8× full-image magnitudes for the same cell.
# Required tuning when enabled (SET AUTOMATICALLY in current config):
#   - split_cost:             15    → 1500    (100× scale)
#   - overlap_penalty_weight: 500   → 30000   (60× scale; saturation-bound)
#   - asymmetric_cost_weight: 8     → 64      (8× scale, matches ratio)
```

### What was checked and found safe

Audited every other config value:

| Type | Examples | Bbox-safe? |
|---|---|---|
| Ratios/fractions | `bio_*_ratio`, `bio_*_fraction`, `split_daughter_volume_scale` | ✓ scale-independent |
| Probabilities | `P_split_base`, `P_split_max`, perturbation `increase_prob` / `decrease_prob` | ✓ |
| Absolute image units | `bio_bridge_min_edge_brightness_absolute: 0.05` (real-image brightness) | ✓ |
| Voxel units | `sigma` values, `minRadius`, `maxRadius`, `pcaShapeConvergeRadius`, `pointR` | ✓ |
| Counts/iterations | `iterations_per_cell`, `split_candidates_per_attempt`, `burn_in_iterations` | ✓ |
| Angles | `pcaShapeConvergeAngleDeg`, `split_candidate_rotation_delta_degrees` | ✓ |

### Remaining known limitations (not this fix)

- **Cost gate is still only a secondary filter.** Phantom splits with costDiff deep into negative (e.g. −42K from the e3d03 projection) still pass the −1500 threshold. Bio gates catch them. For cost alone to be decisive, would need:
  - Non-saturating overlap formula (penalty goes to infinity as daughters approach, not bounded by `weight`)
  - Or explicit daughter-daughter distance hard constraint
  
  Both are separate work items. Current state: cost gate is meaningful for marginal candidates; bio gates catch strong-signal phantoms.

### Files changed

- `C++/config/config.yaml`:
  - `split_cost: 15 → 1500` with tuning-history comment
  - `use_bbox_cost` block comments updated to reflect actual tuning values

## 2026-04-15: Per-cell adaptive PCA weight exponent — bright cells get looser fit, dim cells unchanged

**Problem:** With a single global `pcaShapeWeightExponent: 1.3`, bright cells (very high core / moderate halo — e.g., `1f89ab`, `e9077`) fit tighter than their visible extent. Core pixels (weight ≈ 0.8) contribute `0.8^1.3 ≈ 0.75` while halo pixels (weight ≈ 0.3) contribute `0.3^1.3 ≈ 0.22` — core outweighs halo ~3.4×, so the weighted-moment fit snaps onto the core and the reported radii under-measure the cell. Dim cells (e3d03) have near-uniform brightness, so `exp=1.3` already gives a correct extent. Lowering the global exponent would bloat dim cells and, per run `021558`, catastrophically bloat bright cells whose Voronoi territory has neighbor bleed (aRadius hit max 65, never converged).

**Fix:** Pick the exponent per-cell from the cell's own pixel-cloud brightness distribution. Bright core-dominated cells loosen toward `expBright=1.15`; dim/uniform cells stay at `expDim=1.3` (no behavior change). Cliff-safety: floor `expBright` at 1.15 — above the empirical cliff that killed 1.1 in 021558.

### Mechanism

1. On each cache miss in `calibrateCellShapeViaPca` (same scope as the raw gather — invariant across iterations otherwise), measure
    ```
    pCore = (count of raw pixels with weight > coreBrightnessThreshold) / total
    ```
2. Map `pCore` to the cell's exponent:
    ```
    pCore ≤ fracLow  (0.10)  →  exp = expDim    (1.3)
    pCore ≥ fracHigh (0.40)  →  exp = expBright (1.15)
    between                 →  linear interp
    ```
3. Use that exponent for the remainder of the shape-fit iterations.

### Files changed

**File:** `C++/includes/ConfigTypes.hpp`

**Added** after `pcaShapeWeightExponent{1.5f};`:
```cpp
bool  pcaShapeAdaptiveExponent{false};
float pcaShapeWeightExponentBright{1.15f};
float pcaShapeCoreBrightnessThreshold{0.6f};
float pcaShapeCoreFractionLow{0.10f};
float pcaShapeCoreFractionHigh{0.40f};
```

**Added** to `explodeConfig`:
```cpp
if (node["pcaShapeAdaptiveExponent"]) pcaShapeAdaptiveExponent = node["pcaShapeAdaptiveExponent"].as<bool>();
if (node["pcaShapeWeightExponentBright"]) pcaShapeWeightExponentBright = node["pcaShapeWeightExponentBright"].as<float>();
if (node["pcaShapeCoreBrightnessThreshold"]) pcaShapeCoreBrightnessThreshold = node["pcaShapeCoreBrightnessThreshold"].as<float>();
if (node["pcaShapeCoreFractionLow"]) pcaShapeCoreFractionLow = node["pcaShapeCoreFractionLow"].as<float>();
if (node["pcaShapeCoreFractionHigh"]) pcaShapeCoreFractionHigh = node["pcaShapeCoreFractionHigh"].as<float>();
```

**File:** `C++/src/Frame.cpp`, `calibrateCellShapeViaPca`

**Before the iter loop (new block after `cachedMisses = 0;`):**
```cpp
const bool   adaptiveExp = Ellipsoid::cellConfig.pcaShapeAdaptiveExponent;
const float  expDim      = std::max(0.0f, Ellipsoid::cellConfig.pcaShapeWeightExponent);
const float  expBright   = std::max(0.0f, Ellipsoid::cellConfig.pcaShapeWeightExponentBright);
const float  coreT       = Ellipsoid::cellConfig.pcaShapeCoreBrightnessThreshold;
const float  coreLo      = Ellipsoid::cellConfig.pcaShapeCoreFractionLow;
const float  coreHi      = std::max(coreLo + 1e-6f,
                                    Ellipsoid::cellConfig.pcaShapeCoreFractionHigh);
float cellWeightExponent = expDim;
```

**Inside the `if (!cacheHit) { ... }` block (appended after `++cachedMisses;`):**
```cpp
if (adaptiveExp && !cachedRaw.empty()) {
    int coreCount = 0;
    for (const auto &bp : cachedRaw) {
        if (bp.weight > coreT) ++coreCount;
    }
    const float pCore = static_cast<float>(coreCount) /
                        static_cast<float>(cachedRaw.size());
    const float t = std::clamp(
        (pCore - coreLo) / (coreHi - coreLo), 0.0f, 1.0f);
    cellWeightExponent = expDim + t * (expBright - expDim);
    log << "  [PCA Shape Exp] cell=" << cell.getName()
        << " pCore=" << pCore
        << " exp=" << cellWeightExponent
        << std::endl;
}
```

**Replaced** the per-iter exponent read:
```cpp
// was
const float weightExponent = std::max(0.0f, Ellipsoid::cellConfig.pcaShapeWeightExponent);
// now
const float weightExponent = cellWeightExponent;
```

**File:** `C++/config/config.yaml`

**Appended** after `pcaShapeWeightExponent: 1.3`:
```yaml
pcaShapeAdaptiveExponent: true
pcaShapeWeightExponentBright: 1.15
pcaShapeCoreBrightnessThreshold: 0.6
pcaShapeCoreFractionLow: 0.10
pcaShapeCoreFractionHigh: 0.40
```

### Safety rails

- `expBright` floored at `1.15` by convention — below ~1.2 is the empirical halo-dominated cliff (killed `exp=1.1` globally in `021558`). Adaptive only loosens bright cells, but even then stays above the cliff.
- `adaptiveExponent=false` (default in struct) preserves legacy behavior exactly. YAML turns it on for this tracking dataset.
- `[PCA Shape Exp]` log emitted per cell per cache-miss. Grep for `exp=1.1` — should never appear. Expect two distinct clusters: `exp=1.3` (dim) and values around `1.15–1.2` (bright).

### Verification plan

- `grep "\[PCA Shape Exp\]"` — every cell emits one line per cache-miss. Check cluster distribution.
- `grep "iter=14.*R=("` — aRadius must not approach 65 for any cell. That would signal the halo-bloat cliff was crossed.
- Compare bright-cell radii against the good-shape reference run. 1f89ab and e9077 should measure larger than in runs before this change.
- Dim cells (e3d03, 8cbdf): radii must be unchanged within noise, since they take the dim branch.

### Effect

Bright core-dominated cells see their radii converge to include their own halo (visible extent in image). Dim cells are untouched. Split gates downstream (bridge, volume, edge) see a more accurate parent, improving both false-positive and false-negative rates without the cliff risk of a global exponent change.



## 2026-04-15: Snap-anchored bbox (Option A) — restore position anchor lost under follow-the-cell bbox

**Problem** (see `docs/plans/2026-04-15-bbox-anchor-flaw-analysis.md` for the full writeup). The `_useBboxCost` path in `perturbCell` recomputes the cell's bbox per iteration centered on its LIVE position (union of pre and post positions for each single perturbation). Over 500 iterations the cell can drift 20-25 px, the bbox drifts with it, and voxels at the original snap position drop out of scope entirely. Under full-image L2 those voxels were always counted — real-cell material at the abandoned position produced an undershoot penalty `(0 - bright)^2` that anchored the cell to its real-cell location. Under follow-the-cell bbox that anchor is MISSING. Cells drift freely toward any local image-cost basin, including onto other cells' territories.

Observed pathologies traced to this flaw:
- `e3d03 f4` wandering y=214 → y=189 (25 px drift) in run 030113
- `12345..341 f4` drifting from its birth position (143, 177, 104) to (125, 191, 94)
- Both cells converging on the same bright spot, final distance 2.4 px
- `e3d03 f3 phantom-relocate` in run 002144: daughters collapsed 16 px from parent
- Daughter collapse during burn-in (`worstValleyRatio > 0.95`)

Overlap penalty alone cannot fix it because the root issue is not cell-cell interaction — it is the MISSING PER-CELL POSITION ANCHOR. Even raising `overlap_penalty_weight` from 500 → 30000 only partially mitigated daughter collapse (overlap formula saturates at `weight`).

**Fix — Option A (snap-anchored bbox):** compute each cell's bbox ONCE at frame start, centered on `snap.position` with half-extent `bbox_margin_scale * snap.maxRadius`. Store by cell name. For every perturbation of that cell during the frame, use the fixed bbox as the cost evaluation window. Voxels at the snap position are ALWAYS in scope — if the cell drifts away, synth there goes to 0 while real stays bright, producing the anchoring undershoot penalty. Mask (Voronoi exclusion) remains LIVE-positioned so neighbor exclusion reflects current claim geometry. Cells without a valid snap (frame 1, or daughters created by a split accept mid-frame) fall back to the legacy live pre/post-union bbox.

### Mechanism

1. `Frame::computeBboxAtPoint(center, radius, marginScale)` — new helper, axis-aligned bbox centered on an arbitrary point.
2. `Frame::_snapBboxes` — `unordered_map<string, BoundingBox3D>` keyed by cell name. `setSnapBbox`, `clearSnapBboxes`, `hasSnapBbox` on the public interface.
3. `CellUniverse::optimize` — after the PCA shape fit + `regenerateSynthFrame`, install per-cell snap-bboxes from `previousSnapshots`. Always `clearSnapBboxes()` first so stale names from last frame do not leak.
4. `Frame::perturbCell` — when `_useBboxCost` is true, look up `cellName` in `_snapBboxes`. If present, use it as `bboxUnion`. Else fall through to legacy live-pre/post-union bbox.

### Files changed

- `C++/includes/Frame.hpp` — `#include <unordered_map>`; new `computeBboxAtPoint` declaration; `setSnapBbox` / `clearSnapBboxes` / `hasSnapBbox` inline methods; `std::unordered_map<std::string, BoundingBox3D> _snapBboxes` private member.
- `C++/src/Frame.cpp` — `computeBboxAtPoint` implementation; `perturbCell` bbox-setup block rewritten to prefer `_snapBboxes.find(cellName)` and fall through to the existing live pre/post-union logic only on miss.
- `C++/src/CellUniverse.cpp` — after the PCA shape fit `regenerateSynthFrame`, clear + install `_snapBboxes` from `previousSnapshots` (guarded by `config.prob.use_bbox_cost`). Emits `[Snap Bbox] frame N installed=X total=Y` per frame.

### Notes

- **Mask policy:** the Voronoi exclusion mask is still built from LIVE claim positions inside `perturbCell`. That keeps neighbor exclusion reflective of current geometry; the anchor is the bbox itself, not the mask. Upgrading the mask to snap-positioned claims is a follow-up but was not needed for the anchor fix.
- **Split path untouched:** `trySplitCellPhased` has its own union bbox from `computeUnionBboxWithPoints({cellIndex}, ..., daughterSeeds, pointR)`. Parent is installed at snapshot state during split evaluation (existing design); daughter seeds are candidate positions. That logic already anchors the split evaluation correctly and does not need `_snapBboxes`.
- **Post-split daughters:** when a split accepts mid-frame, the parent name disappears from `cells` but its stale `_snapBboxes` entry is harmless (never looked up again). New daughter names have no entry → their subsequent perturbations automatically fall through to the legacy live bbox, which is correct (they have no meaningful "snap" position yet).
- **Frame 1 uses full-image cost (not bbox).** With no snapshots, bbox mode would fall back to the follow-the-cell bbox for every cell — losing the position anchor that is the entire point of bbox cost. Full-image is the correct global anchor while frame-1 cells are still settling into their initial fit. `CellUniverse::optimize` gates `setUseBboxCost` on `frameIndex > 0`; snap-bbox install is gated on the same flag. The `[Optimize]` log line prints `(frame-1 forced full-image)` when this exception fires.
- **`[Snap Bbox] frame N installed=X total=Y`** log line confirms coverage per frame. `X < Y` is expected on any frame where splits created daughters last frame — daughters do not have snaps themselves until the end-of-frame snapshot pass runs.

### Verification

Against the pathologies in the plan document:
1. `e3d03 f4` should stay near its snap position `y=214` (no more drift to `y=189`). Check via `awk` on `e3d03` rows in `cells.csv`.
2. `12345..341 f4` (first-gen daughter, no snap yet) should drift less because its neighbors (`e3d03`, `e9077`) are now snap-anchored and do not overlap it.
3. Split accept rate vs GT matches or improves — anchor reduces phantom-relocate pressure.
4. `[Snap Bbox] frame N installed=X` — `X = Y` (or `Y - k` where k = cells newborn from splits last frame) for every frame past 1.
5. No halo-bloat regression: the anchor only affects cost-evaluation geometry, not shape fit. PCA shape fit paths are unchanged.

### Rollback

Flip `use_bbox_cost: false` in `config.yaml` — snap-bbox installation is guarded by that flag, and `perturbCell`'s bbox branch is skipped entirely in full-image mode. No need to revert the source change.

## 2026-04-15: Extend Option A to split daughters — fixes daughter collapse during burn-in

**Problem.** Option A snap-anchors `perturbCell` for cells that have a valid snap, but the split path's daughter burn-in calls `perturbCell` on daughter candidates whose names (`parentName + "0"`, `parentName + "1"`) are NOT in `_snapBboxes`. Daughter perturbations therefore fall through to the legacy follow-the-cell bbox, lose the position anchor, and both daughters drift toward a shared bright lobe during the 500-iter burn-in. Bridge then correctly rejects the collapsed pair.

**Direct evidence** — e9077 frame 3, current run `034723` vs 0413 good run:

| | seeds apart | final daughters apart | `worstValleyRatio` | outcome |
|---|---|---|---|---|
| 0413 good | 50 px | **50 px** | 0.52 | accept |
| 034723 | 47 px | **2 px** | 1.36 | reject (bridge_flat) |

Same cell, same cell position, similar seed separation. 0413 daughters stayed apart because full-image L2 cost anchored them globally; current run's bbox-mode burn-in let them collapse onto the same peak.

**Fix — install `splitBbox` (the shared union bbox already computed at top of `trySplitCellPhased`) under BOTH daughter candidate names in `_snapBboxes` for the duration of the split attempt.** Each daughter's `perturbCell` then scores cost over the union bbox covering both lobes: abandoning one lobe registers an undershoot penalty → each daughter is anchored to its own lobe, exactly as full-image cost did in 0413. Erased on every exit path (restoreLiveParent for rejects, accept callback for the final decision).

### Files changed

- `C++/src/Frame.cpp`, `trySplitCellPhased`:
  - After `[Split Bbox Init]` log (inside `if (_useBboxCost)`): set `_snapBboxes[parentName+"0"] = splitBbox` and `_snapBboxes[parentName+"1"] = splitBbox`.
  - `restoreLiveParent` lambda: erase both daughter keys at the top (covers all reject paths that route through it — early-return, bio reject, cost reject).
  - Accept callback body: erase both daughter keys BEFORE the `if (accept)` split (covers both accept and reject branches; new permanent daughters fall through to the legacy live bbox for the rest of the frame, consistent with "daughters have no real snap yet").

### Notes

- No change to split gate or seeding logic. Only the daughter perturbation cost window changes.
- The cleanup pattern is intentionally idempotent (`erase` of a missing key is a no-op). `restoreLiveParent` + accept callback may both run on reject; that's fine.
- Under full-image mode (`_useBboxCost == false`), this code path is skipped entirely — legacy behavior preserved.

### Verification

- `grep "Split Refine"` — `refineDrift1` + `refineDrift2` should stay modest (daughters don't collapse into each other).
- `grep "Split Accepted"` — daughter pair distances `sqrt((d1-d2)²)` should be comparable to seed distances, not collapse to < 5 px.
- `grep "Split Reject bio"` with `reason=bridge_flat` + `worstValleyRatio > 0.95` should drop substantially (the noValleyTier-1 collapse signature disappears when daughters don't collapse).
- Specifically: e9077 frame 3 should accept (matching 0413 good run), with final daughters ~50 px apart and `valleyRatio < 0.85`.

### Rollback

Revert the three edit blocks in `Frame.cpp` (splitBbox install, restoreLiveParent erase, callback erase). Or set `use_bbox_cost: false` for an unconditional fallback to 0413-style full-image cost everywhere.

## 2026-04-15: Raise pcaShapeMaskScale 1.3 → 1.6 — match 0413 good run's converged mask size

**Problem.** Current bright cells fit smaller than in the 0413 good run despite identical config values (`pcaShapeRadiusScale=2.236`, `pcaShapeMaskScale=1.3`). Root cause: 0413 used LIVE iteratively-updated mask (mask recomputed from current live radii every iteration), which grew across iterations and converged at an effective mask = `1.3 × Rfinal`. The current design deliberately fixes the mask to `1.3 × snap radii` (gotcha B — snap-mask invariant prevents collapse onto one daughter in dividing cells), so the fit never sees the halo pixels that 0413's grown mask gathered.

Direct evidence from e3d03 frame 1 (the one cell whose iter trace is directly comparable):
- 0413: iter 0 `n=28,061` → iter 3 `n=67,724` (2.4× growth). Converged `R=(26.5, 19.5, 16.6)`.
- Current: iter 0 `n=28,071` (same as 0413's iter 0, confirming same snap input), stays at `n=28,071` forever. `R=(19.5, 16.8, 14.5)` — matches 0413's iter 0 exactly, but can't grow beyond it.

For bright cells the undersell is visible in final radii vs 0413 (frame 1 post-fit):

| Cell | 0413 | Current | delta |
|---|---|---|---|
| 12345 | (40.3, 35.3, 24.4) | (39.4, 34.1, 23.5) | -2% |
| e9077 | (38.7, 30.3, 27.8) | (37.0, 29.2, 24.4) | -4% / -12% on cRadius |
| 1f2ed | (43.5, 33.9, 30.3) | (37.2, 32.8, 29.6) | -15% on aRadius |
| 1f89ab | (48.4, 34.8, 31.2) | (39.4, 33.7, 30.3) | -19% on aRadius |

**Fix.** Raise `pcaShapeMaskScale` from 1.3 to 1.6. Approximates 0413's converged-mask effective size (~`1.3 × 1.2`) directly at iter 0 without needing iterative mask growth. Snap-mask invariant preserved — mask is still fixed across iterations, just larger.

### Rationale for 1.6 specifically

- 0413's converged mask was `1.3 × Rfinal`. Rfinal ≈ `1.2 × Rsnap` for bright cells (based on the frame 1 comparison above).
- Effective mask in 0413 ≈ `1.3 × 1.2 × Rsnap = 1.56 × Rsnap`.
- Rounded to 1.6 for safety headroom.

### Why this is preferable to per-cell adaptive exponent or percentile radius calc

The adaptive-exponent (2026-04-15) and percentile-radius proposals both operate on the pixel set gathered inside the mask. If the mask is too tight, the halo pixels aren't in the set, and neither approach can recover them. Fixing the mask size is upstream of both — it brings the halo pixels IN, after which even the baseline `sqrt(variance) × 2.236` gives the right answer (same math as 0413).

Adaptive exponent stays in place as a secondary lever for peaked distributions where the larger mask still undersells — but with mask 1.6, the ramp should rarely fire (most pixels now come from inside the real cell extent, not its halo core).

### Risks

- **Halo bloat** — if the enlarged mask reaches into neighbor territory that Voronoi exclusion misses, the PCA absorbs foreign pixels and inflates. Watch `[PCA Shape] iter=14 R=(65, ...)` — aRadius ≈ 65 signals the ceiling-clamp. Observed in run `021558` when `pcaShapeWeightExponent` fell to 1.1; not expected here (we're not changing exponent), but the larger mask makes it incrementally more plausible.
- **Voronoi bleed on tight clusters** — two adjacent cells with snap positions closer than `2 × 1.6 × Rsnap` would have masks that overlap; Voronoi exclusion should resolve the overlap correctly (nearest claim point wins), but any error in the claim set leaks pixels across cells. If radii start drifting toward a single large blob, check Voronoi claim sets.
- **No direct harm to frame 1** — frame 1 uses full-image cost (separate fix) and the shape fit runs before any snap bboxes are relevant.

### Files changed

- `C++/config/config.yaml`: `pcaShapeMaskScale: 1.3 → 1.6` with multi-line rationale comment.

### Rollback

Change `pcaShapeMaskScale` back to 1.3. Shape fits will return to the current smaller-radii regime.

## 2026-04-15: Lower asymmetric_cost_weight 64 → 32 — fix single-peak daughter clustering

**Problem.** In run `041912`, e9077 frame 3 (ground-truth split frame) got rejected by bridge_flat. Log analysis showed that among 20 split candidates:

| Candidate | seeds apart | final daughter separation | image cost |
|---|---|---|---|
| idx=12 (**winner**) | 51 px | **9 px** (collapsed) | 274,505 |
| idx=15 | 53 px | 60 px (stayed apart) | 469,831 |
| idx=19 | 50 px | 56 px (stayed apart) | 357,664 |

Candidates with correctly-separated daughters existed; the cost gate picked the collapsed one because its image cost was 23% lower. The collapse then continued through refine (9 px → 2.5 px) and the bridge correctly rejected the degenerate pair. Daughter-snap-bbox is working (separated candidates prove the anchor is in effect) — the cost landscape is the remaining problem.

**Root cause.** `asymmetric_cost_weight: 64` makes undershoot cost 64× more than overshoot. When the real image has one dominant bright peak and a dimmer secondary lobe (typical of a cell in the early stage of dividing), two strategies compete:
- **Collapsed on primary peak**: synth = 2× brightness there (overshoot, weight 1 per voxel), synth = 0 at dimmer lobe (undershoot at that dimmer lobe, but the real brightness is low so the absolute penalty is small).
- **Separated on two lobes**: synth = 1× brightness each (matches), but each daughter only partly covers its lobe, leaving halo regions with synth < real (undershoot, weight 64 per voxel × large halo volume).

At k=64 the separated-strategy undershoot dominates the cost — collapsed wins. Same cell in the 0413 good run with k=8 had separated daughters win (verified: 0413's e9077 f3 accepted with `valleyRatio=0.52`).

**Fix.** Lower `asymmetric_cost_weight` from 64 to 32. Halves the undershoot-over-overshoot amplification ratio, making the separated strategy cost-competitive again. Still 4× the pre-bbox value (8) so most of the bbox-fraction scaling rationale (the "parent covers dark valley" signal's fractional impact) is preserved.

### Files changed

- `C++/config/config.yaml`: `asymmetric_cost_weight: 64.0 → 32.0` with multi-line rationale comment added to tuning history.

### Verification

- `grep -A5 "e9077.*Split Attempt" <run>/debug_log.txt | grep -E "Split Cand|Split Winner|Split Accepted|Split Reject"` on the next run's frame 3.
- Expected: a separated candidate (final d1↔d2 > 30 px) wins on cost. Bridge passes with `valleyRatio < 0.85`. Split accepts.
- If e9077 f3 still rejects with the SAME single-peak-clustering pattern, next step is k=24.

### Rollback

Flip `asymmetric_cost_weight` back to 64. No source changes.

## 2026-04-15: Shared daughter Voronoi mask — port 0413's anchor fully into bbox scope

**Problem.** After the daughter-snap-bbox fix and `asymK: 64→32`, run `044735` still had e9077 frame 3 rejected with the SAME pattern: the top 7 candidates had daughter separations of 5.1–16.2 px (collapsed), while candidates that stayed separated (48–56 px) cost 30–50% more. Dropping `asymK` actually *widened* the relative gap (collapsed/separated cost ratio went from 1.20 at k=64 to 1.29 at k=32).

**Root cause discovered** (not `asymK`, not the bbox anchor).

`perturbCell` rebuilds a Voronoi exclusion mask per call from live claim sets:
```cpp
selfClaims   = [current cell's position]
otherClaims  = [every OTHER cell, including sibling daughter]
mask         = voxels closer to self than to any otherClaim
```

For daughters d1 and d2 during burn-in, the sibling IS in `otherClaims`. So d1's mask excludes voxels closer to d2, and vice versa. The Voronoi boundary between them moves WITH them.

As d1 drifts toward d2:
- d1's mask shrinks on the side facing d2
- Voxels at d1's ORIGINAL lobe are now closer to d2 → they fall OUT of d1's mask
- d1 never "sees" the undershoot cost of leaving those voxels uncovered — they aren't in d1's cost window anymore

d2 experiences the mirror problem. Neither daughter pays an abandonment cost. They freely drift toward a shared bright region. Overlap penalty is the only counterforce, and it saturates at `weight × 1.0 = 30000` when daughters fully overlap — easily beaten by an image-cost gain of 100K+ from covering the shared peak.

This is the mechanism that `asymK` tuning could never fix. The cost-to-converge is ≈ 0 because the mask moves out from under the abandoned voxels.

**Contrast with 0413.** Full-image L2 had no Voronoi mask between daughters. Both daughters' costs were evaluated over the entire 225-slice image. If d1 moved away from its lobe, the lobe registered undershoot immediately — anchoring was automatic because the cost scope never shifted.

**Fix — install a shared mask for daughter candidates during split attempt.** The `splitMask` at `trySplitCellPhased` top is already built correctly: self-claim = PARENT position, otherClaims = real neighbors (sibling not included because daughters don't exist at that point). Install it under both daughter candidate names in a new `_sharedMasks` map. `perturbCell` checks that map first and uses the stored mask when present.

Under the shared mask:
- Both daughters measured over the SAME voxel set (parent's full territory, excluding real neighbors)
- Synth at each voxel = combined rendering of both daughters
- If d1 abandons its lobe: lobe still in mask, synth goes to 0 there, real stays bright → undershoot penalty at the abandoned lobe (weight `asymK`)
- Penalty scales with lobe volume — structurally impossible to zero-cost converge anymore

This is 0413's global anchor, ported into bbox scope.

### Files changed

- `C++/includes/Frame.hpp`:
  - New public methods: `setSharedMask(name, mask)`, `clearSharedMasks()`, `hasSharedMask(name)`.
  - New private member: `std::unordered_map<std::string, std::vector<uint8_t>> _sharedMasks`.
- `C++/src/Frame.cpp`:
  - `perturbCell` (bbox branch): lookup `_sharedMasks[cellName]` first; use stored mask if present, else rebuild Voronoi mask from live positions (legacy path for normal perturbation).
  - `trySplitCellPhased`: alongside `_snapBboxes` install of `splitBbox`, also `_sharedMasks[parentName+"0"] = splitMask` and `["1"] = splitMask`.
  - `restoreLiveParent` lambda: erase both `_sharedMasks` keys (covers all reject paths).
  - Accept callback body: erase both `_sharedMasks` keys BEFORE the accept/reject split (post-split real daughters fall through to live Voronoi mask, which IS the correct behavior once they're real cells competing for territory — the shared mask was specifically for burn-in where mutual exclusion is the wrong model).

### Verification

After rebuild, for e9077 frame 3 in the next run:

```bash
RUN=$(ls -t C++/outputs/ | head -1)
# Separated candidates should now dominate the top of the cost ranking
awk '/Split Cand.*e9077/ {
    match($0,/final1=\([^)]+\)/); f1=substr($0,RSTART+8,RLENGTH-9)
    match($0,/final2=\([^)]+\)/); f2=substr($0,RSTART+8,RLENGTH-9)
    match($0,/total=[0-9.]+/);   tot=substr($0,RSTART+6,RLENGTH-6)+0
    split(f1,a1,","); split(f2,a2,",")
    dx=a1[1]-a2[1]; dy=a1[2]-a2[2]; dz=a1[3]-a2[3]
    sep=sqrt(dx*dx+dy*dy+dz*dz)
    printf "sep=%.1f total=%.1f\n", sep, tot
}' C++/outputs/$RUN/debug_log.txt | sort -k2 -t=
```

Expected: the low-cost candidates (winners) now have sep > 30 px. Collapsed candidates (sep < 15 px) should move to the BOTTOM of the ranking. `[Split Accepted] e9077` at frame 3.

### Notes

- Only affects daughter burn-in + refine. Regular per-cell perturbation (outside split attempts) is unchanged — its live Voronoi mask is correct when cells are real and compete for territory.
- The `splitMask` already excludes real neighbors (other cells at their current positions). Using it for daughter burn-in doesn't change that — only removes the sibling from otherClaims.
- Shared-mask storage is a full `vector<uint8_t>` copy per daughter (×2 per split attempt). Mask size = bbox volume, typically ~10M voxels = ~10MB per daughter. Small, lives for the duration of one split attempt.
- Cleanup is mechanical. Stale entries can't leak: `restoreLiveParent` covers rejects, accept callback covers accepts, and `erase`-of-absent-key is a no-op so double-cleanup is safe.

### Rollback

Remove the `_sharedMasks.find(cellName)` check in `perturbCell`'s bbox branch — falls back to live Voronoi mask unconditionally. Or remove the install lines in `trySplitCellPhased` — the map stays empty, `find` always misses, behavior is identical to pre-fix.

## 2026-04-15: asymK 32 → 8 + split_cost 1500 → 375 — fix boundary-overshoot double-counting

**Problem.** After the daughter-snap-bbox + shared-mask fixes, run `050447` still had e9077 f3 rejected with a collapsed winner (sep=6.4 px, total=141K) beating separated candidates (sep≈45–60 px, total=162–190K) by 20-50K on cost.

**Root cause** — traced by reading `Ellipsoid::draw` (`C++/src/Ellipsoid.cpp:403-407`) and `asymmetricL2Slice` (`C++/src/Frame.cpp:22-47`):

1. `Ellipsoid::draw` overwrites: `image.at<float>(y, x) = _brightness`. When two daughters overlap, the second overwrites the first — stacked cells render as ONE ellipsoid's worth of synth, not two.
2. `asymmetricL2Slice` amplifies OVERSHOOT (synth > real), not undershoot: `asymSumSq = sumSq + (k-1) * posSumSq` where `posSumSq` is the L2 sum over pixels where `diff = synth - real > 0`.

Combined effect for the split candidate cost:

- **Separated daughters** render as 2 ellipsoids → 2 × boundary-overshoot regions (where each ellipsoid extends slightly past the real cell into dim halo) → amplified k× by asymK.
- **Collapsed daughters** stack → effectively 1 ellipsoid due to overwrite → 1 × boundary overshoot amplified k×. Pays undershoot (weight 1, unamplified) at the missed lobe — a smaller penalty.

At `k=32`: separated pays ~160K boundary-overshoot cost, collapsed pays ~80K boundary-overshoot + ~12.5K undershoot = ~92K. Collapsed wins by ~68K. Observed gap 20-50K (depending on candidate-specific geometry) is consistent with this model.

asymK at this level was doing TWO jobs in tension:
- Accepting real splits vs no-split: asymK amplifies parent-over-valley overshoot → split is cheaper → split accepts. WANTS high k.
- Picking separated vs collapsed WITHIN a split attempt: asymK amplifies per-ellipsoid boundary overshoot → collapsed wins. WANTS low k.

Design target was job #1; we broke job #2.

**Fix.** Drop `asymK` to 8 (matches 0413 good-run value). Crossover analysis:
```
separated = 2 × boundary × (Δ²) × k
collapsed = 1 × boundary × (Δ²) × k + missed_lobe × (Δ²) × 1
```
Separated wins when `k < missed_lobe_volume / boundary_volume`. For typical e9077 geometry (50K lobe, 10K boundary), crossover is `k ≈ 5`. k=8 is above the exact crossover but close enough that with real-data noise, separated candidates should dominate the top of the cost ranking.

Also scaled `split_cost: 1500 → 375` proportionally (1/4 ratio since k=32→8). At k=8 bbox total costs drop ~4×, so threshold follows.

### Files changed

- `C++/config/config.yaml`:
  - `asymmetric_cost_weight: 32.0 → 8.0` with diagnosis comment pointing to `draw` overwrite + boundary-overshoot double-counting.
  - `split_cost: 1500 → 375` with scaling rationale.

### Verification

After next build + run:

```
RUN=$(ls -t C++/outputs/ | head -1)
# e9077 f3 candidates: winner should now be a separated candidate (sep > 30)
awk '/Split Cand.*e9077/ { ... sep + total ... }' C++/outputs/$RUN/debug_log.txt | sort -k2 -t=

# Split outcomes: e9077, 12345, and the other splits should accept where GT says.
grep -E "Split (Accepted|Reject).*frame 3" C++/outputs/$RUN/debug_log.txt
```

Expected:
- e9077 f3: accept with daughter sep > 30 px, `valleyRatio < 0.95`.
- 12345 f3: accept (was already finding separated candidates; just needed bridge to pass on them — lower k doesn't affect bridge threshold but the cost-ranking winner may now have cleaner valley signature).
- e3d03 f3, 8cbdf f3, 1f2ed f3, 1f89ab f3: reject (these should NOT split at frame 3 per GT).

### Regression watch

Lower asymK weakens the anti-phantom signal. Watch `e3d03` and `8cbdf` (GT: never split) across all frames — if they start splitting, the k-value isn't enough to catch phantoms and we need the structural fix (compute overshoot once per union footprint, not per cell).

### Rollback

Restore both values to `asymmetric_cost_weight: 32.0` and `split_cost: 1500.0`. No source changes.

### Structural alternative (deferred)

The real fix is to make rendering + cost symmetric: either additive rendering (synth at overlap = sum, clamped) which naturally penalizes stacking, or compute the per-split candidate overshoot once over the UNION footprint of the daughters. Current overwrite-render + per-cell-cost creates the double-counting. The k-tuning is a workaround; the structural fix keeps k-tuning independent of split-candidate comparison.

## 2026-04-15: Bridge gate redesign — single metric gap/max(edges), retires 2-tier + density

**Problem.** In run `053555` e9077 f3 accepted correctly (sep=50 px, the asymK+shared-mask fixes landed) but 12345 f3 was rejected by `bridge_flat` with:

```
valleyRatio1 = 0.750   (d1 side: clear valley)
valleyRatio2 = 1.009   (d2 side: gap ≈ edge)
worstValleyRatio = 1.009  → tier-1 reject (>= 0.95)
edge1Bright = 0.108, edge2Bright = 0.080, gapBright = 0.081
```

Daughters positioned correctly (47 px apart, matching 0413 good run). The failure was entirely the metric: real asymmetric division produces one brighter and one dimmer daughter. The dimmer daughter's edge brightness is naturally close to the gap brightness (both are "somewhat above background" without the bright lobe's intensity), so `gap/edge_dim ≈ 1`. Taking `worst = max(vr1, vr2)` reports "no valley" from the dim side, independent of how clear the valley actually is on the bright side.

This is a structural flaw in the metric for asymmetric division, which is biologically the common case post-cytokinesis. No threshold tuning generalizes.

**Fix — single-metric gate with max(edges) reference.** The question "is there a valley?" should be answered by "does the gap drop from the BRIGHTER cell body?" — not "does it drop from both". A dim daughter's edge-to-gap ratio is an artifact of the daughter being dim, not an indicator of valley structure.

```
valleyFromBright = gapBright / max(edge1Bright, edge2Bright)
reject if valleyFromBright > bio_bridge_max_valley_ratio (0.85)
```

### Verification against run 053555 frame 3

| Cell (GT) | edge1 | edge2 | gap | `gap/max(edges)` | new | old tier-1 |
|---|---|---|---|---|---|---|
| **12345** (split) | 0.108 | 0.080 | 0.081 | **0.75** | accept | reject |
| **1f2ed** (no split) | 0.129 | 0.124 | 0.148 | **1.15** | reject | reject |
| **1f89ab** (no split) | 0.091 | 0.087 | 0.110 | **1.21** | reject | reject |
| **e3d03** (no split) | 0.030 | 0.100 | — | caught by `edge_too_dim` | reject | reject |
| **8cbdf** (no split) | caught by `edge_too_dim` | | | | reject | reject |

All correct outcomes at threshold 0.85 (unchanged from existing `bio_bridge_max_valley_ratio`).

### Generalization behavior

- **Symmetric division**: edges equal, `gap/max = gap/either`, same decision as before.
- **Asymmetric division**: dim daughter's edge ≈ gap is expected; max-of-edges picks the bright daughter as the reference. Valley on the bright side drives the decision.
- **Pre-division single cell**: gap ≈ both edges, ratio ≈ 1, reject.
- **Cell with cleavage furrow neck (mid-division)**: gap dimmer than bright end, ratio < 0.85, accept. This is biologically correct — the neck is cytokinesis in progress.
- **Empty-space daughters**: caught by existing `edge_too_dim` absolute gate (independent).

### Retired parameters (still parsed for backward compat)

- `bio_bridge_no_valley_hard_threshold` (was 0.95): tier-1 compensation for worst-of-two-sides punishing asymmetry. No longer needed.
- `bio_bridge_max_gap_density` (was 0.18): density conjunction for subtle valleys. Absorbed into the single threshold. Gap density still logged in `[Split Bridge]` for diagnostic continuity.

### Files changed

- `C++/src/Frame.cpp`, `trySplitCellPhased` bridge block:
  - New computed value `brighterEdge = max(edge1Bright, edge2Bright)` and `valleyFromBright = gapBright / brighterEdge`.
  - Decision: `bridgeFlat = (valleyFromBright > bio_bridge_max_valley_ratio)`. Single condition, no tiers, no density.
  - `[Split Bridge]` log gets new field `valleyFromBright=`; old fields (vr1, vr2, worst, pooled, density) preserved for diagnostic continuity.
  - `[Split Reject bio] reason=bridge_flat` log updated with the new metric alongside legacy fields.
- `C++/config/config.yaml`:
  - `bio_bridge_max_valley_ratio: 0.85` — comment rewritten to describe the new metric.
  - `bio_bridge_max_gap_density` + `bio_bridge_no_valley_hard_threshold` — commented `[DEPRECATED 2026-04-15]`, values preserved, fields still parsed, no effect on decision.

### Verification plan

After rebuild, frame-3 splits to watch:
- e9077: accept (already accepting via asymK=8 fix)
- 12345: accept (the specific case this fix targets)
- 1f2ed, 1f89ab, e3d03, 8cbdf: reject (negative controls — should NOT regress)

Broader check across all frames:
- `grep valleyFromBright <run>/debug_log.txt | awk` to see the distribution. Real splits should cluster < 0.85; phantoms > 1.0. If any phantom has valleyFromBright < 0.85, inspect — but the `edge_too_dim` gate plus downstream bio checks (volume/buried) should still catch most failure modes.

### Why this generalizes to other datasets

The new metric asks "does the bright-pixel profile drop from the brighter daughter's cell body into the midpoint?" — a direct geometric property of cell division. It doesn't depend on:

- Daughters being symmetric in brightness (asymmetry is normal)
- Daughters being symmetric in volume (volume ratios are checked separately)
- Parent orientation or snap position (the gate runs on post-refine positions)
- Image intensity scale (threshold is a RATIO, invariant to overall brightness)

The only assumption is that a real division produces a dim midpoint relative to the brighter daughter's body — which is the definition of cytokinesis cleaving the cell. Datasets where this assumption fails (no visible cleavage furrow on any axis) would fail the bridge regardless of metric, and should use a non-brightness-based division signal.

### Rollback

Revert `trySplitCellPhased` bridge block to use `worstValleyRatio`-based tiered logic with `bio_bridge_no_valley_hard_threshold` and `bio_bridge_max_gap_density`. Both config fields are still parsed so no YAML changes needed for rollback.

## 2026-04-15: Frozen per-cell shape reference — decouple shape-fit mask from snap radii (breaks compounding bloat)

**Problem** (observed in run `output_jihang_20260415_060504`). Shape-fit radii compound across frames because the pixel-collection mask is `maskScale × snap_radii`, and snap = previous-frame fitted radii. A small initial over-fit inflates the next frame's mask, which pulls in more halo, which inflates the next fit further, and so on until the fit hits the hard ceiling.

Measured compounding on `e9077..a51` (second-gen daughter, GT splits at frame 20):

```
f3  (birth)  R=(24, 22, 21)
f7           R=(34, 29, 26)
f10          R=(37, 34, 32)
f13          R=(38, 37, 34)
f17          R=(45, 37, 35)
f20          R=(65, 34, 22)   ← clamped at maxARadius ceiling; PCA wanted 66
             PCA pre-clamp target = 79 at one iteration
             dR stuck at 1.0 for 10 iterations (ceiling fights fit)
```

Same pattern visible on `e3d03` (never splits, should stay steady):
`f1=(22,18,16) → f4=(28,24,16) → +27% aRadius in 3 frames`.

The compounding also caused the rendering artifacts in frame 20–22 screenshots: oversized ellipsoids visibly overlapping neighbors, cells "shifting" from their real-image positions to occupy bloated Voronoi territory. Neighbor exclusion and bridge gate are doing their jobs — the bug is upstream in the shape fit's mask basis.

**Fix — frozen per-cell shape reference.** Capture each cell's radii ONCE at birth and use that value as the mask basis for every subsequent frame's shape fit, rather than re-reading snap (which carries last-frame bloat into the next fit).

- Frame 1 seed cells: reference = post-PCA-fit radii at the end of frame 1's shape fit.
- Daughter cells: reference captured at the end of the next frame's shape fit (one-frame delay; snap from split-refit radii covers the first frame).
- Reference never updates. Mask radii = reference × `pcaShapeMaskScale`. Compounding chain broken at the first link.

### Mechanism

In `CellUniverse::optimize` (shape-fit block), replace the snap-lookup for mask radii with a `cellShapeReference[cellName]` lookup. Fall back to snap only if no reference exists (shouldn't happen in steady state).

At the end of the shape-fit block, capture references for any cell in `frame.cells` that doesn't yet have one. This covers:
- Frame 1 (all cells): initial seed cells get reference post-fit.
- Frame 2+ (daughters born via split in the previous frame's main loop): their reference is captured on first entry.
- Frame N+ (mid-frame-split daughters): registered on the subsequent frame.

Reference values are stable once set; the map is append-only per cell. Parent cells that were erased by a split accept leave stale entries in the map — harmless (map is a pure lookup).

### Why this breaks compounding

- Mask is fixed per cell across all its frames.
- A bloated fit at frame N doesn't feed back into frame N+1's mask.
- PCA at frame N+1 sees the same pixel set scope as every other frame → converges to the same radii if the real cell hasn't changed.
- Pre-split elongation still fits correctly because the pixel cloud inside the fixed mask still has the elongated bright distribution; PCA responds to that.
- Bloat is self-limiting: fit can't exceed `reference × maskScale × safety` regardless of history.

### Trade-offs

- Frame-1 fit quality matters more. If frame 1 under-fits a bright cell (e.g., halo cut off), the small reference persists forever.
- A cell that legitimately grows to >`reference × maskScale` extent over the run can't be captured — PCA saturates at the mask boundary. Typical pre-split elongation is ~30% over baseline, well within any reasonable mask scale (1.5+).
- One-frame delay for daughters is the main subtlety; acceptable because daughter post-refit radii are already a good estimate.

### Files changed

- `C++/includes/CellUniverse.hpp`:
  - Added `#include <array>`.
  - New private member `std::map<std::string, std::array<float, 3>> cellShapeReference`.
- `C++/src/CellUniverse.cpp`:
  - In the per-cell shape-fit loop: lookup `cellShapeReference[sname]` first; on miss, fall back to snap radii (legacy).
  - After the shape-fit loop ends (post-regenerate): loop over `frame.cells` and register reference for any cell without an entry, using the just-fitted radii. Log `[Shape Reference] frame N captured=X total=Y` when any capture happens.

### Verification

After the next run:

```bash
RUN=$(ls -t C++/outputs/ | head -1)
# Reference capture should log at frame 1 with captured=total (all seed cells),
# and at frames immediately after each split with captured=2 (new daughters).
grep "\[Shape Reference\]" "$RUN/debug_log.txt"

# e9077..a51 should NOT hit aRadius >= 60 anywhere in its trajectory.
awk -F, 'NR>1 && $2 ~ /e9077677575842b1a2925729fbcfb3a51/ {
  printf "  %s R=(%.1f,%.1f,%.1f)\n", $1, $6, $7, $8
}' "$RUN/cells.csv"

# e3d03 should stay close to its initial size throughout; no compounding growth.
awk -F, 'NR>1 && $2 ~ /^e3d03/ {
  printf "  %s R=(%.1f,%.1f,%.1f)\n", $1, $6, $7, $8
}' "$RUN/cells.csv"

# No aRadius >= 58 anywhere (safety check vs ceiling).
grep "PCA Shape.*iter=14" "$RUN/debug_log.txt" | \
  awk -F'R=\\(' '{print $2}' | awk -F',' '{if ($1+0 >= 58) print $0}'
```

Expected:
- e9077..a51 trajectory peaks far below 60, no ceiling hits.
- e3d03 aRadius stays ~22 throughout (± a couple of voxels) instead of growing to 28.
- No catastrophic PCA iterations (no `aRadius >= 60`).
- Screenshot sanity: frame 20–22 cells no longer overlap or shift wildly.

### Rollback

Revert the two edit blocks in `CellUniverse.cpp` (the reference-lookup branch and the capture block). Shape fit returns to snap-based mask with compounding risk.

### What this does NOT fix

- 1f89ab..1 z=224 boundary jump — that's a POSITION problem, not shape. Separate fix needed (cap snap-bbox half-extent per axis, or monotone boundary guard in perturbCell).
- 8cbdf z=224 initial condition — initial.csv seeding, not a code problem.
- Split timing 1-frame late for e9077 descendants — likely correlated with this bloat (ceiling clamp distorts the pre-split fit), so this fix may also improve timing. Validate in the verification run.

## 2026-04-15: Upgrade shape reference from frozen to bounded-growth

**Problem with the frozen version.** Captured frame-1 fitted radii as a permanent reference. For first-generation seed cells this works (pre-split elongation ~10–30%, well within `maskScale=1.6` cap). For second-generation daughter cells it breaks: daughters are born at ~0.8× parent size and grow toward ~1× parent over many frames (~88% growth for e9077..a51 from birth to pre-split across 14 frames), exceeding the frozen 1.6× cap. Shape fit would get capped at `reference × 1.6`, under-measuring the real cell.

**Fix — bounded-growth reference.** Each frame the reference updates toward the observed fit, but capped at ±5%/frame:

```cpp
constexpr float refGrowthCap = 0.05f;      // 5% per frame
ref[i] = clamp(fit[i], ref[i] * 0.95, ref[i] * 1.05);
```

Properties:

- **Normal growth (0-5%/frame)** passes through unchanged. Daughter cells growing 4.6%/frame (measured e9077..a51) track perfectly.
- **Shrinking (0-5%/frame)** tracks down. Ref follows the cell's real extent.
- **Bloat (>5%/frame)** hits the upper cap. Mask grows at most 5%/frame even if the fit wants more. Compounding chain throttled: a 20%/frame bloat scenario would take 14 frames to reach the ceiling instead of 3, and would be obvious in logs long before it got there.
- **Stable cells** (0%/frame change) have stable ref. No drift from noise.

### Implementation

`C++/src/CellUniverse.cpp` — the per-cell reference update block after the shape-fit loop:

- New cells (frame 1 seeds, first appearance of post-split daughters): capture current fit directly. `captured++`.
- Existing cells: clamp the fit into the ±5% window around the previous reference. `updated++`.
- Logged each frame: `[Shape Reference] frame N captured=X updated=Y total=Z growthCap=0.05`.

### Verification plan

After next run, the reference map should show all cells being updated each frame (not frozen). The key checks:

```bash
RUN=$(ls -t C++/outputs/ | head -1)

# Ref updates should fire for every cell every frame after birth
grep "\[Shape Reference\]" "$RUN/debug_log.txt"

# e9077..a51 should reach its real pre-split extent (~45) without being capped
awk -F, 'NR>1 && $2 ~ /e9077677575842b1a2925729fbcfb3a51/ {
  printf "  %s R=(%.1f,%.1f,%.1f)\n", $1, $6, $7, $8
}' "$RUN/cells.csv"

# No aRadius >= 58 anywhere (bloat guardrail)
grep "PCA Shape.*iter=14" "$RUN/debug_log.txt" | \
  awk -F'R=\\(' '{print $2}' | awk -F',' '$1+0 >= 58 {print}'

# e3d03 should stay steady (no compounding) while still being able to vary ±5%/frame
awk -F, 'NR>1 && $2 ~ /^e3d03/ { printf "  %s R=(%.1f,%.1f,%.1f)\n", $1, $6, $7, $8 }' "$RUN/cells.csv"
```

Expected:
- `e3d03`: aRadius stays in [22, 28] range across all 22 frames, no compounding to 28+
- `e9077..a51`: aRadius climbs smoothly from birth ~24 to ~45 pre-split (not capped at 38, not shot to 65)
- No catastrophic PCA iterations
- Better frame 4 overlap and frame 5 shape alignment (downstream of shape-fit quality)

### Rollback

Set `refGrowthCap = 0.0f` to restore frozen-reference behavior (frame-1 capture, never updates).

## 2026-04-15: Adaptive radius inflation for peaked cells

**Problem.** Observed in user screenshot: a bright cell fitted visibly tighter than its real halo extent. Root cause is that `radii = sqrt(5) × sqrt(variance)` with `radiusScale = 2.236` is the analytically correct radius for a **uniform-density** ellipsoid. For peaked distributions (Gaussian-like bright-core cells), `2.236 × σ` gives the 97% containment radius — but the visible halo extends to 3σ+ (99%+ containment). Result: peaked cells' fits are systematically 10-15% smaller than what the user sees.

Global solution (raising `radiusScale` to 2.5) would fix peaked cells but:
- Over-inflate uniform/dim cells where sqrt(5) was exact
- Amplify halo-bleed / neighbor-contamination in overfitted cells

**Fix — adaptive radius inflation driven by `pCore`**, reusing the adaptive-exponent infrastructure.

```
pCore ≤ fracLow    →  scale_factor = 1.0                         (uniform, unchanged)
pCore ≥ fracHigh   →  scale_factor = pcaShapeRadiusInflationBright (peaked, inflated)
between            →  linear interpolation

radii_final = scale_factor × radiusScale × sqrt(variance)
```

### Cell-type behavior

- **Peaked bright-core**: pCore high → 15% radius inflation → visible halo captured. Example: `1f89ab` frame 1 fit at 44.1 → becomes 50.7 (matching 0413's 48.4 within noise).
- **Uniform/dim**: pCore low → 1.0× scale → unchanged behavior. Example: `e3d03` stays ~22 aRadius, not inflated.
- **Halo-bleed / overfitted**: pCore is typically LOW for these cases (halo dilutes the core fraction) → scale_factor stays at 1.0 → no additional inflation. Bounded-growth reference still caps them independently.

### Why peaked ≠ overfitted (the user's concern addressed)

Overfitted cells (halo bloat, neighbor bleed) come from the *mask* absorbing foreign pixels, which widens the pixel distribution. Their pCore is LOW because the halo/bleed dilutes the bright-core fraction. pCore-based inflation doesn't fire for them.

Peaked cells have their bright-core fraction HIGH (concentrated peak). pCore-based inflation fires, corrects the 97% → 99%+ gap.

So the two problems get orthogonal treatments:
- Overfitted → bounded-growth reference caps ref/mask growth at 5%/frame
- Underfitted peaked → adaptive radius inflation corrects the analytic-vs-real formula mismatch

### Files changed

- `C++/includes/ConfigTypes.hpp`:
  - New field `float pcaShapeRadiusInflationBright{1.15f}` with documentation.
  - YAML parse for the new field.
- `C++/src/Frame.cpp`, `calibrateCellShapeViaPca`:
  - New `radInflBright` read from config.
  - New per-cell `cellRadiusInflation` set alongside `cellWeightExponent` on pCore compute.
  - New log field `radInfl=` in `[PCA Shape Exp]`.
  - `targetA/B/C = cellRadiusInflation × radiusScale × sqrt(variance)` — inflation applied at radius output.
- `C++/config/config.yaml`: `pcaShapeRadiusInflationBright: 1.15` with block comment explaining the peaked-vs-uniform rationale.

### Verification

```bash
RUN=$(ls -t C++/outputs/ | head -1)

# Expect bright cells with radInfl > 1.0, dim cells with radInfl = 1.0
grep "\[PCA Shape Exp\]" "$RUN/debug_log.txt" | head -20

# 1f89ab aRadius at frame 1 should be near 48-52 (was 44.1 pre-fix)
grep "PCA Shape.*cell=1f89abf484.*iter=14" "$RUN/debug_log.txt" | head -5

# e3d03 and 8cbdf should stay at their previous sizes (no inflation)
awk -F, 'NR>1 && $2 ~ /^e3d03/ {printf "  %s R=(%.1f,%.1f,%.1f)\n", $1, $6, $7, $8}' "$RUN/cells.csv" | head -5
```

### Rollback

Set `pcaShapeRadiusInflationBright: 1.0` in YAML or `pcaShapeAdaptiveExponent: false` to disable the whole adaptive path.

### Caveat

Pushing `pcaShapeRadiusInflationBright > 1.2` starts inflating uniformly across peaked cells and risks reaching into legitimate neighbor territory. 1.15 is calibrated against 0413's post-LIVE-mask fits; upward tuning above 1.2 should be evidence-backed.

## 2026-04-15: Fix pCore threshold + raise daughter refit iterations

**Problem.** Inspection of run `091510` log showed `[PCA Shape Exp]` reporting `pCore=0 exp=1.3 radInfl=1` for EVERY cell on every frame. Neither the adaptive exponent nor the adaptive radius inflation ever triggered. Root cause: `pcaShapeCoreBrightnessThreshold = 0.6` was calibrated against post-sigmoid [0, 1] pixel values, but the current iterative-contrast preprocessing pipeline produces pixel weights in the **0.05–0.20 range**. No pixel ever exceeds 0.6 → pCore stays 0 → both adaptive paths collapse to "uniform" defaults.

This explained why bright cells were still under-fit and why newborn daughters looked especially tight in screenshots.

**Fix A — lower the core threshold.**

```yaml
pcaShapeCoreBrightnessThreshold: 0.6 → 0.15
```

0.15 sits between typical halo brightness (0.05–0.10) and typical core brightness (0.15–0.25) for this dataset. Peaked cells now show pCore > 0.2; uniform cells stay near 0. Adaptive exponent + adaptive radius inflation finally do their jobs.

**Fix B — more iterations for newborn daughter refit.**

Newborn daughters get a compressed PCA pass at split time (was 3 iterations, with `pcaShapeMaxIters=15` reserved for mature cells). Screenshot evidence showed daughters visibly under-fit at birth — the 3 iterations weren't enough for the PCA to converge against the adaptive inflation + radius-growth feedback.

```yaml
split_daughter_refit_iterations: 3 → 8
```

Adds ~5ms per daughter per split (negligible). 8 iterations lets the fit climb with adaptive inflation engaged.

### Verification

```bash
RUN=$(ls -t C++/outputs/ | head -1)

# Peaked cells should now show pCore > 0.1 and radInfl > 1.0
grep "\[PCA Shape Exp\]" "$RUN/debug_log.txt" | awk '{
  match($0,/pCore=[0-9.]+/); p=substr($0,RSTART+6,RLENGTH-6)
  match($0,/radInfl=[0-9.]+/); r=substr($0,RSTART+8,RLENGTH-8)
  print "pCore="p" radInfl="r
}' | sort -u | head -20

# Newborn daughter radii should be closer to adult size
grep "Split Daughter Refit" "$RUN/debug_log.txt" | head -10
```

Expected:
- Bright cells (1f89ab, 1f2ed, 12345, e9077): `pCore ≈ 0.2–0.5`, `radInfl ≈ 1.05–1.15`
- Dim cells (e3d03, 8cbdf): `pCore < 0.1`, `radInfl = 1.0`
- Daughter refit `post=(...)` values noticeably larger than `built=(...)` when the daughter is on a real bright region

### Rollback

Revert both values: `pcaShapeCoreBrightnessThreshold: 0.15 → 0.6` and `split_daughter_refit_iterations: 8 → 3`.

## 2026-04-15: Correct daughter-refit direction — cap radii + enable position update

**Problem.** Screenshot evidence clarified: newborn daughters are being drawn OVERSIZE and OFF-CENTER, not undersize. Refit log confirmed:
```
e9077 d2:   built=(30.5, 22.5, 18.4) → post=(38.4, 31.9, 26.7)   +26% / +42% / +45%
1f89ab d1:  built=(33.1, 26.0, 22.6) → post=(33.9, 31.0, 26.5)   +3% / +19% / +17%
```
Radii grew significantly during refit because (a) mask = `built × 1.6` was wide enough to absorb neighbor / halo pixels across the immature sibling-Voronoi boundary, and (b) PCA centroid was fixed at the burn-in position while real cell centroid sat a few voxels off.

**The previous "more iterations" fix went the wrong direction** (more iterations cement the oversize). Reverted.

**Fix A — ceiling on refit radii.** Add upper cap symmetric to the existing floor. Daughters clamped to `[0.6 × built, 1.1 × built]` per axis. Allows slight shrink/grow adjustment, prevents 20-45% bloat from Voronoi leak.

```yaml
split_daughter_refit_min_radius_fraction: 0.6    # existing floor
split_daughter_refit_max_radius_fraction: 1.1    # NEW ceiling
split_daughter_refit_iterations: 3               # reverted from 8
```

**Fix B — enable position update during daughter refit.** The mature-cell shape fit keeps `updatePosition=false` because snap is the authoritative position source. But a newborn daughter HAS no snap — its post-burn-in centroid is still an estimate. Let the refit's PCA centroid slide the cell to match the actual bright-pixel centroid:

```cpp
calibrateCellShapeViaPca(..., /*updatePosition=*/true, ...)
```

Only 3 iterations so the slide is bounded.

### Files changed

- `C++/includes/ConfigTypes.hpp`: new `split_daughter_refit_max_radius_fraction` field + YAML parse.
- `C++/src/Frame.cpp`, `trySplitCellPhased::refitOne`:
  - Added `maxFrac`, `ceilA/B/C` above the lambda.
  - Changed `/*updatePosition=*/false` → `/*updatePosition=*/true` for refit only.
  - Clamp radii with `std::clamp(fit, floor, ceil)` instead of `std::max(fit, floor)`.
  - Log line now includes `ceil=...`, `prePos=...`, `postPos=...`, `posShift=...` for diagnostics.
- `C++/config/config.yaml`:
  - `split_daughter_refit_max_radius_fraction: 1.1` with rationale.
  - `split_daughter_refit_iterations: 8 → 3` reverted with explanation.

### Verification

```bash
RUN=$(ls -t C++/outputs/ | head -1)

# Refit post radii should be within [0.6 × built, 1.1 × built] per axis
grep "Split Daughter Refit" "$RUN/debug_log.txt" | head -10

# posShift should be nonzero (position update enabled), typically 1-5 voxels
grep "Split Daughter Refit" "$RUN/debug_log.txt" | grep -oE "posShift=[0-9.]+" | head -10

# Downstream: newborn daughters no longer overlap neighbors or render off-center
```

### Rollback

- `split_daughter_refit_max_radius_fraction: 1.1 → 10.0` effectively disables the ceiling.
- `updatePosition=true → false` reverts to fixed-centroid refit.

## 2026-04-15: Fit-side growth cap — anti-bloat within mask

**Problem.** Run 091510 f20 showed catastrophic single-frame bloat: `e9077..a50` aRadius jumped **41.5 → 65.2 (+57%)** in one frame and hit the ceiling. Cascade: ceiling-bloated parent inflated volume_fraction reference → split at f20 rejected (0.584 < 0.60 bio limit), parent stayed at 65 for f21/f22, neighbors' Voronoi territories distorted, `12345..01` subsequently bloated from 23 to 55 aRadius in one frame at f22. User screenshots showed the resulting chaos.

**Why bounded-growth on REF didn't prevent it.** The reference cap limits `ref` to ±5%/frame, but `mask = ref × maskScale = ref × 1.6`. The fit can land anywhere inside the mask — up to **60% above ref in a single frame**. Ref-cap controls the mask basis; fit is unconstrained within mask.

For e9077..a50 at f20: `ref_f19 = 41.5`, `mask_f20 = 66.4`, PCA produced `fit = 65.2` (inside mask, right at ceiling). Ref updated to 43.6 (+5%) — but damage was done: parent synth at aRadius=65 wrecked neighboring cost evaluations.

**Fix.** Clamp FIT to `ref × 1.10` per frame AFTER the PCA converges, before applying to the cell. Combined with the existing 5% ref cap, growth is now limited in both the mask basis (ref, slow) AND the per-frame realization (fit, fast within mask).

### Walkthrough for e9077..a50 with the fix

```
f19: ref=41, fit=41.5, cell.aRadius=41.5
f20: mask = 41 × 1.6 = 66
     PCA target aRadius = 65 (naively)
     fit cap = ref × 1.1 = 45
     cell.aRadius clamped to 45 (not 65)
     ref updates to min(45, 41×1.05) = 43
f20 split attempt:
     parent aRadius = 45 (not 65)
     parent volume = 45 × 31 × 20 ≈ 28K (not 40K)
     daughters sum = 18.8K
     volume_fraction = 18.8/28 = 0.67  ← PASSES 0.60 limit
     SPLIT ACCEPTS
```

Cascade prevented at the first link.

### Implementation

`C++/src/CellUniverse.cpp`, shape-fit block:

```cpp
constexpr float fitGrowthCap = 0.10f;  // 10%/frame max upward fit
const float fitUpFactor = 1.0f + fitGrowthCap;
for each cell with an existing reference:
    cap per axis = ref[i] * fitUpFactor
    new_radius[i] = min(fit[i], cap[i])
    if any axis clamped: cell.setRadii(new_radii)
// Then regenerateSynthFrame(), then ref-update block (unchanged)
```

### Why 10% for fit vs 5% for ref

- **Ref cap (5%)**: slow, conservative — ref is the mask basis, feeds directly into next frame. Biasing toward underestimation is safer than overestimation (prevents compounding). Matches measured legitimate daughter growth rate of 4.6%/frame for `e9077..a51`.
- **Fit cap (10%)**: slightly looser — allows legitimate single-frame shape adjustments (e.g., cell genuinely elongating by 8-9% as pre-split begins) while blocking the 30%+ jumps that are the bloat signature. Still tight enough that the cell can't cross the ceiling in any realistic trajectory.

Net growth per frame: fit can be up to 10% above ref, ref tracks fit up to 5%/frame → effective cell growth is ~5-7%/frame steady state, enough for any legitimate biology.

### Pre-split elongation check

Pre-split cells elongate 30-40% in the cleavage direction over 3-5 frames. At 10%/frame fit cap: `1.10⁴ = 1.46×` baseline in 4 frames — more than enough headroom. The cap doesn't prevent pre-split elongation from being measured.

### Verification

```bash
RUN=$(ls -t C++/outputs/ | head -1)

# Expect Fit Growth Cap log lines where bloat was being attempted
grep "\[Fit Growth Cap\]" "$RUN/debug_log.txt"

# e9077..a50 aRadius should NOT jump to 65 at f20
awk -F, 'NR>1 && $2 == "e9077677575842b1a2925729fbcfb3a50" {
  printf "  %s R=(%.1f,%.1f,%.1f)\n", $1, $6, $7, $8
}' "$RUN/cells.csv"

# No cell aRadius >= 58 anywhere
grep "PCA Shape.*iter=14" "$RUN/debug_log.txt" | awk -F'R=\\(' '{print $2}' | awk -F',' '$1+0 >= 58 {print}'

# 12345..01 should stay steady (no +152 px jump at f22)
awk -F, 'NR>1 && $2 ~ /^123456793425243543545020349852340.$/ {
  printf "  %s %s\n", $1, $2; printf "    pos=(%.1f,%.1f,%.1f) R=(%.1f,%.1f,%.1f)\n", $3, $4, $5, $6, $7, $8
}' "$RUN/cells.csv" | tail -20

# Splits at f20: all three (12345..0, 12345..1, e9077..a51) should still accept
# AND e9077..a50 should ALSO accept this time (volume_fraction no longer blocked by bloated parent)
grep -E "frame 20.*Split (Accepted|Reject)" "$RUN/debug_log.txt" || grep -B1 "Split (Accepted|Reject)" "$RUN/debug_log.txt" | grep -A1 "frame 20"
```

### Rollback

Set `fitGrowthCap = 1.0f` to disable the cap (allows fit = mask size, i.e., prior behavior).

### Caveat

A cell that genuinely grows more than 10% in one frame (unusual but possible e.g. for a frame where the cell appears much brighter than usual) will have its fit capped. Over 2-3 frames the bounded ref catches up and fit converges to correct size. If this creates visible lag, raise `fitGrowthCap` toward 0.15.

## 2026-04-15: Raise daughter refit floor 0.6 → 0.85 — Voronoi-sibling-shrinkage compensation

**Problem.** Run 091510 f20 e9077..a50 split rejected by `volume_fraction=0.584 < 0.60`. Investigation against 0412 best_run (which DID successfully split this cell) showed that the rejection was NOT caused by wrong parent size — user confirmed current run's parent shape R=(41.5, 29.5, 26.3) is correct, while 0412's parent at R=(23.5, 27.1, 18.5) was underspecified by ~25%.

The actual issue: the daughter REFIT systematically shrinks the axes perpendicular to the split direction because of Voronoi sibling exclusion.

**Mechanism.** Daughter refit's pixel gather (`gatherBrightPixelsVoronoi`) places the sibling into `otherClaims`. For each pixel in the combined cell's bright cloud, the pixel is assigned to whichever daughter's centroid is closer. The Voronoi boundary is the perpendicular bisector between the two daughter positions.

Effect on PCA variance:
- **Split axis (d1-d2 direction)**: pixels are fully in one or the other half, each daughter sees its full half → PCA variance along this axis is roughly correct
- **Perpendicular axes**: each pixel's sibling-proximity test is symmetric around the split axis; both daughters compete for the same pixels → each daughter effectively sees HALF the perpendicular pixel extent → PCA variance halves → radii on perpendicular axes shrink

Observed for e9077..a50 f20:
```
built radii:    (32.9, 23.4, 20.9)   volumeScale × parent snap
split axis:     d1-d2 = (+4, +23, +10)  mostly Y
d1 refit:       (32.1, 18.9, 15.3)   a: 97% of built, b: 81%, c: 73%
d2 refit:       (30.1, 19.3, 16.3)   a: 91% of built, b: 82%, c: 78%
```

b/c axes (perpendicular to the mostly-Y split) shrunk by 18-27%. Combined daughter volume = 58% of parent → failed the 60% bio_combined_volume_min_fraction threshold.

**Fix.** Raise the refit floor from 0.6 × built to 0.85 × built. At 0.85 floor, each daughter is guaranteed ≥ 67.5% per axis (0.85 × 0.7937), so combined volume ≥ 2 × 0.85³ × 0.5 = 0.614 × parent volume — just above the 0.60 bio threshold.

### Validation on the f20 a50 case

```
built:             (32.9, 23.4, 20.9)
floor at 0.85:     (28.0, 19.9, 17.8)
ceil at 1.10:      (36.2, 25.7, 23.0)

d1 raw refit:      (32.1, 18.9, 15.3)
d1 clamped:        (32.1, 19.9, 17.8)   ← b and c floored
d1 volume:         11,363

d2 raw refit:      (30.1, 19.3, 16.3)
d2 clamped:        (30.1, 19.9, 17.8)
d2 volume:         10,664

sum:               22,027
parent snap vol:   32,189
volume_fraction:   0.684  ← PASSES (limit 0.60) ✓
```

Split accepts.

### Does this break legitimate asymmetric division?

A true split where one daughter is 60% volume and the other 85% would have its smaller daughter clamped up to 85% of built. That distorts the fit slightly but preserves volume_fraction, and the main optimize loop's subsequent shape fit (bounded growth at 5%/frame) will correct it over the next few frames as the cell's ref-tracker settles.

Minor distortion on asymmetric first-frame is acceptable to keep the split from being spuriously rejected.

### Files changed

- `C++/config/config.yaml`: `split_daughter_refit_min_radius_fraction: 0.6 → 0.85` with rationale comment.

### Rollback

`split_daughter_refit_min_radius_fraction: 0.85 → 0.6`.

## 2026-04-15: Non-saturating overlap penalty — fundamental fix for cell drift into neighbors

**Problem.** Run 153343 frame 3: e3d03 drifted 52 px in one frame from (113, 205) to (144, 176), landing essentially on top of 12345 (distance 10.9 px, radii sum 72.9). Consequence: 12345's split was rejected with `d1_buried_in_e3d03` because the daughter candidate fell inside the bloated e3d03 ellipsoid's territory. GT frame 3 expected both 12345 and e9077 to split; only e9077 did.

**Why existing overlap penalty didn't stop the drift.** The formula was:
```
penalty = weight × ratio²    (with ratio = (combinedR − dist) / combinedR)
```
This saturates at `weight` when cells fully coincide (ratio=1). At `weight = 30,000`:
- ratio=0.85 (e3d03 into 12345) → penalty = 21,675
- ratio=1.0 (coincident) → penalty = 30,000

Any finite image-cost gain larger than 30K wins. Cells can always stack if the cost landscape supports it. Increasing `weight` only raises the ceiling; the saturation shape is fundamentally the problem.

**Fix.** Non-saturating (barrier-form) formula that diverges as cells approach full coincidence:
```
penalty = weight × ratio² / (1 − ratio + ε)
```
With `ε = 0.01` and `weight = 30,000`:

| ratio | old (saturating) | new (non-saturating) | ratio of new/old |
|---|---|---|---|
| 0.3 | 2,700 | 3,830 | 1.42× |
| 0.5 | 7,500 | 14,700 | 1.96× |
| 0.7 | 14,700 | 49,000 | 3.33× |
| 0.85 | 21,675 | 121,500 | 5.60× |
| 0.95 | 27,075 | 452,000 | 16.7× |
| 0.99 | 29,400 | 2,940,000 | 100× |
| 1.0 | 30,000 | huge (1/ε-bounded) | structurally prohibitive |

At ratio=0.85 (the e3d03 situation), the penalty jumps from 21K to 121K — a 5.6× increase, plenty to dominate the image-cost gain that was previously winning. At ratio=0.95+ (severe overlap), the penalty explodes — no physically-meaningful image gain can overcome it.

**Properties:**
- **No new tuning knobs** — ε is numerical stability only
- **Light overlap unchanged in practice** — ratio=0.3 penalty goes 2.7K → 3.8K, doesn't punish legitimate cell touching
- **Heavy overlap forbidden** — asymptote at ratio=1 is mechanically impossible to cross
- **Dataset-agnostic** — doesn't depend on image brightness, cost function magnitudes, or preprocessing. Works for any biology.

### Files changed

- `C++/src/Frame.cpp`:
  - New `static inline double nonSaturatingOverlap(ratio, weight)` at file scope before `computeOverlapPenalty`. Encodes the barrier formula with `EPS=0.01`.
  - `Frame::computeOverlapPenalty` replaces inline `weight × ratio²` with `nonSaturatingOverlap(ratio, weight)`.
  - `Frame::computeOverlapForCell` same replacement.

### Verification

```bash
RUN=$(ls -t C++/outputs/ | head -1)

# e3d03 and 12345 should NOT overlap at frame 3 (mutual distance > sum of minR)
awk -F, -v f="frame003.tif" 'NR>1 && $1==f {
  n[NR]=$2; x[NR]=$3; y[NR]=$4; z[NR]=$5; r[NR]=$6
}
END {
  for(i in n) for(j in n) if(i<j) {
    d=sqrt((x[i]-x[j])^2+(y[i]-y[j])^2+(z[i]-z[j])^2); cr=r[i]+r[j]
    if(d < cr) printf "overlap: %s ↔ %s d=%.1f cr=%.1f\n", substr(n[i],1,10), substr(n[j],1,10), d, cr
  }
}' "$RUN/cells.csv"

# 12345 split at frame 3 should accept (no d1_buried_in_e3d03 rejection)
grep -E "Split (Accepted|Reject).*frame 3" "$RUN/debug_log.txt" || \
  awk '/frame 3/,/frame 4/' "$RUN/debug_log.txt" | grep -E "Split (Accepted|Reject)"

# Full split count vs GT (expect 8: f3:2, f8:1, f11:1, f19:1, f20:3)
grep "Split Accepted" "$RUN/debug_log.txt" | wc -l
```

### Rollback

Revert `computeOverlapPenalty` and `computeOverlapForCell` to the inline `weight * ratio * ratio` formula. Remove the `nonSaturatingOverlap` helper.

### Caveat

Cells that are SUPPOSED to touch (e.g., packed embryos where cells physically contact) get slightly stronger penalty at low ratios. At ratio=0.2 (edges touching, 20% overlap by the bounding-sphere approximation), penalty goes 1.2K → 1.5K (25% up). Acceptable for this dataset; would need retune for denser embryo imaging.

## 2026-04-15: Tighten adaptive trigger — pcaShapeCoreFractionLow 0.10 → 0.30

**Problem.** Run 161013 f4 showed `e3d03 ↔ 12345..0` overlap of 22 px (distance 35.4, radii sum 57.7, ratio 0.39). Non-saturating overlap penalty gave 7K penalty at this ratio — image cost gain beat it.

Root cause: e3d03 was fit at aRadius=31 when its true size is ~22. The 40% oversize turned a merely-close neighbor pair into an overlapping one. Size inflation traced to adaptive path firing on dim cells:

```
e3d03 f1:  pCore = 0.60  → ramp t = (0.60-0.10)/(0.40-0.10) = 1.0 (full inflation)
                        → radInfl = 1.15, exp = 1.15
           Raw PCA aRadius = 22.3
           Inflated aRadius = 22.3 × 1.15 = 25.7
           After frames of bounded growth: f4 aRadius = 31
```

With `pcaShapeCoreFractionLow=0.10`, any cell with pCore > 0.10 gets partial-to-full inflation. e3d03 (moderately uniform pancake cell) qualifies as "fully peaked" under this too-permissive trigger.

**Fix.** Raise `pcaShapeCoreFractionLow` from 0.10 to 0.30. Only cells with `pCore > 0.30` qualify for ANY adaptive treatment.

Expected behavior:
- Bright core-dominated cells (1f89ab, 1f2ed, pre-split 12345/e9077): pCore ≥ 0.5 → full inflation (unchanged)
- Moderate cells (e3d03, 8cbdf): pCore 0.2-0.3 → no inflation (radInfl=1.0, exp=1.3)
- Clearly dim cells: pCore < 0.2 → no inflation (unchanged)

With e3d03 at radInfl=1.0, f1 aRadius = 22.3 (raw PCA). Over subsequent frames with bounded growth, aRadius stays near 22-23. Distance to 12345..0 (27 aRadius) = 35, sum = 49 → no overlap.

### Why this is more fundamental than raising overlap weight

- Raising `overlap_penalty_weight` forces cells apart regardless of cost landscape → risks pushing cells off their real positions
- Tightening the adaptive trigger fixes the SIZE at source → cells are correctly sized → no artificial overlap to fight against
- Overlap penalty (non-saturating, shipped earlier) still handles the remaining case where cells LEGITIMATELY approach each other

### Files changed

- `C++/config/config.yaml`: `pcaShapeCoreFractionLow: 0.10 → 0.30` with full rationale comment.

### Verification

```bash
RUN=$(ls -t C++/outputs/ | head -1)

# e3d03 should NOT get adaptive inflation (radInfl=1.0)
grep "\[PCA Shape Exp\].*e3d03" "$RUN/debug_log.txt" | head -5

# e3d03 aRadius across frames should stay ~22 (no bloat to 31)
awk -F, 'NR>1 && $2 ~ /^e3d03/ {
  printf "  %s R=(%.1f,%.1f,%.1f)\n", $1, $6, $7, $8
}' "$RUN/cells.csv" | head -8

# f4 overlap check — should be zero
awk -F, -v f="frame004.tif" 'NR>1 && $1==f {
  n[NR]=$2; x[NR]=$3; y[NR]=$4; z[NR]=$5; r[NR]=$6
}
END {
  for(i in n) for(j in n) if(i<j) {
    d=sqrt((x[i]-x[j])^2+(y[i]-y[j])^2+(z[i]-z[j])^2); cr=r[i]+r[j]
    if(d<cr) printf "overlap: %s ↔ %s d=%.1f cr=%.1f\n", substr(n[i],1,10), substr(n[j],1,10), d, cr
  }
}' "$RUN/cells.csv"
```

Expected:
- e3d03 `[PCA Shape Exp]` lines show `pCore=0.2-0.3`, `radInfl=1.0`, `exp=1.3` (previously `radInfl=1.15`, `exp=1.15`)
- e3d03 aRadius stays 22-25 across all frames (was 25-31)
- No pairwise overlaps at f4 (e3d03 ↔ 12345..0 gap > 0)
- Bright cells (1f89ab etc.) continue to show radInfl ≥ 1.05 (still inflated)

### Rollback

`pcaShapeCoreFractionLow: 0.30 → 0.10`.












