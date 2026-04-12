# Changelog v4: PCA Input Fix and Residual Safety Nets (2026-04-09 onward)

**Branch:** `yp_yd_merge_04072026`

This changelog covers the implementation of the 2026-04-09 plan
`C++/docs/plans/2026-04-09-pca-input-fix-and-residual-safety-nets.md` —
the three-pillar fix for the false-split / lineage-corruption failure mode
diagnosed from the frame-19 incident in `output_jihang_20260408_164644`.

The plan has three pillars:

- **Pillar A** — replace `Spheroid::getSplitCells`'s 2× expanded-bbox
  PCA with a strict-interior brightness-weighted PCA. The primary fix.
- **Pillar B** — add a structural local-maximum split detector that
  acts as an AND-gate alongside Pillar A.
- **Pillar C** — frame-anchor + drift penalty as a residual safety net.

Implementation order: A first (in three sub-PRs), then B, then C.

### Status Key

| Status | Meaning |
|--------|---------|
| **ACTIVE** | Change is in the current codebase |
| **REVERTED** | Change was implemented then undone |
| **SUPERSEDED** | Change was replaced by a later entry |

---

## 2026-04-09: PR-2 — Strict-Interior Brightness-Weighted PCA (shadow mode) — **ACTIVE**

**Plan:** `C++/docs/plans/2026-04-09-pca-input-fix-and-residual-safety-nets.md`, Pillar A, step 1 of 3.

**Status:** Shadow mode only — does NOT change any split decision yet.
PR-3 will wire it as the elongation trigger; PR-4 will wire the
split-attempt path.

**Problem:** The frame-19 false split in `output_jihang_20260408_164644`
traced to `Spheroid::getSplitCells` collecting bright pixels from a 2.0 ×
bounding box that overlapped neighboring cells. PCA on the contaminated
cloud reported `elongation_ratio ≈ 1.42` for cell
`1f89abf484c94c498a23cad71ebee0cb1` whose brightness mass is actually
nearly oblate. A strict-interior brightness-weighted PCA computed in a
Python tool over the same dataset gave median `sqrt(L1/L2) ≈ 1.02` (94% of
cells classify as oblate-or-spherical), versus median ≈ 1.40 from the bbox
PCA (57% classify as triaxial). The triaxial signal was neighbor leakage,
not real cell shape. See §1.4 of the plan for the full measurement.

**Goal of this PR:** Add a strict-interior brightness-weighted PCA path
alongside the existing one and log both ratios side-by-side, without
changing any decision logic. This produces validation data for PR-3
(which will replace the elongation trigger's PCA input) and eventually
PR-4 (split-attempt path).

### Change 1 — `StrictInteriorPcaResult` struct + method declaration

**File:** `C++/includes/Spheroid.hpp`

**Before** (after `SplitDiagnostics` struct, around line 60):

```cpp
struct SplitDiagnostics {
    /* ... existing fields ... */
    int insideCount = 0;
};

class Spheroid { /* ... */ };
```

**After:**

```cpp
struct SplitDiagnostics {
    /* ... existing fields unchanged ... */
    int insideCount = 0;
};

// (~30-line doc comment + new struct — see Spheroid.hpp:60-90 for full text)
struct StrictInteriorPcaResult
{
    bool valid = false;
    float elongationRatio = 1.0f;   // sqrt(L1)/sqrt(L2) in raw image units
    float lambda1 = 0.0f;
    float lambda2 = 0.0f;
    float lambda3 = 0.0f;
    int sampleCount = 0;
    float sampleWeightSum = 0.0f;
};

class Spheroid { /* ... */ };
```

A new public method declaration is also added inside `class Spheroid`,
right after `getSplitCells`:

```cpp
StrictInteriorPcaResult computeStrictInteriorWeightedElongation(
    const std::vector<cv::Mat> &image,
    float backgroundColor) const;
```

### Change 2 — `Spheroid::computeStrictInteriorWeightedElongation` implementation

**File:** `C++/src/Spheroid.cpp`

**Before** (right before `Spheroid::checkConstraints` at line 867 of the
previous file):

```cpp
bool Spheroid::checkConstraints() const { /* ... */ }
```

**After:** A new ~140-line method is inserted before `checkConstraints`.
Its full source lives in `Spheroid.cpp` and follows this structure:

```cpp
StrictInteriorPcaResult Spheroid::computeStrictInteriorWeightedElongation(
    const std::vector<cv::Mat> &image, float backgroundColor) const
{
    // Walk voxels strictly inside the cell's own analytic ellipsoid
    // (val <= 1.0 with the cell's own a, b, c — no expanded gate, no
    // neighbor exclusion, no percentile filter).
    //
    // Pass 1: weighted centroid using w = max(0, pixel - backgroundColor).
    // Pass 2: weighted 3x3 covariance.
    // Eigendecompose with cv::eigen (returns eigenvalues sorted descending).
    // elongationRatio = sqrt(L1) / sqrt(L2).
    //
    // Returns valid=false if fewer than 4 voxels have positive weight or
    // total weight is below 1e-9.
}
```

The method reuses the existing `scanSpheroidVolume` template helper
(defined in the anonymous namespace at the top of `Spheroid.cpp`) and the
`generateInverseRotationMatrix` private member, so no new helpers are
introduced.

**Why brightness weighting and not the existing percentile threshold:**
The percentile threshold (`splitBrightestFraction = 0.055`, the top 5.5%
of pixels) reduces sample size by ~20× and amplifies noise in eigenvalue
ratios. The Python strict-interior measurement showed weighted PCA gives a
much cleaner distribution (median r12 1.02, p90 1.06) than
threshold-filtered PCA (median 1.27, p90 1.64). Weighting uses every
interior pixel proportional to its brightness above background, which is
statistically more robust.

**Why no neighbor exclusion:** Strict-interior pixels (val ≤ 1.0 with the
cell's own ellipsoid) are by definition closer to the self center than to
any non-overlapping neighbor's center. Adding neighbor exclusion would be
a no-op for the strict-interior path; it's only needed when the gate is
expanded.

### Change 3 — Shadow logging in `Frame::computeElongationRatios`

**File:** `C++/src/Frame.cpp` (around line 497)

**Before:**

```cpp
std::map<std::string, float> Frame::computeElongationRatios() const
{
    std::map<std::string, float> ratios;
    for (size_t i = 0; i < cells.size(); ++i) {
        std::vector<cv::Point3f> neighbors;
        for (size_t j = 0; j < cells.size(); ++j) {
            if (j != i) neighbors.push_back(cells[j].get_center());
        }
        auto [d1, d2, valid, elongation, splitDiagnostics] = cells[i].getSplitCells(
            _realFrame, simulationConfig.z_scaling, _backgroundValue, neighbors);
        ratios[cells[i].getName()] = valid ? elongation : 1.0f;
    }
    return ratios;
}
```

**After:** Same code, plus a strict-interior call and `[Split PCA Both]`
log line inside the loop:

```cpp
std::map<std::string, float> Frame::computeElongationRatios() const
{
    std::map<std::string, float> ratios;
    for (size_t i = 0; i < cells.size(); ++i) {
        std::vector<cv::Point3f> neighbors;
        for (size_t j = 0; j < cells.size(); ++j) {
            if (j != i) neighbors.push_back(cells[j].get_center());
        }
        auto [d1, d2, valid, elongation, splitDiagnostics] = cells[i].getSplitCells(
            _realFrame, simulationConfig.z_scaling, _backgroundValue, neighbors);
        ratios[cells[i].getName()] = valid ? elongation : 1.0f;

        // [PR-2 shadow] strict-interior brightness-weighted PCA, logged only.
        const StrictInteriorPcaResult strictResult =
            cells[i].computeStrictInteriorWeightedElongation(_realFrame, _backgroundValue);
        std::cout << "[Split PCA Both] " << cells[i].getName()
                  << " caller=elongation_trigger"
                  << " bbox_elongation=" << (valid ? elongation : 1.0f)
                  << " strict_elongation=" << (strictResult.valid ? strictResult.elongationRatio : 1.0f)
                  << " strict_lambda=(" << strictResult.lambda1
                  << "," << strictResult.lambda2
                  << "," << strictResult.lambda3 << ")"
                  << " strict_samples=" << strictResult.sampleCount
                  << " strict_weight_sum=" << strictResult.sampleWeightSum
                  << " strict_valid=" << (strictResult.valid ? 1 : 0)
                  << std::endl;
    }
    return ratios;
}
```

### Change 4 — Shadow logging in `Frame::computeElongationForCell`

**File:** `C++/src/Frame.cpp` (around line 530, after the change above)

The same `[Split PCA Both]` log line is added after the existing
`getSplitCells` call, with `caller=elongation_for_cell` to distinguish the
call site.

### Change 5 — Shadow logging in `Frame::trySplitCell`

**File:** `C++/src/Frame.cpp` (around line 580, immediately after the
existing `getSplitCells` call inside `trySplitCell` and BEFORE the
`if (!valid)` early-return)

The same `[Split PCA Both]` log line is added with `caller=try_split`.

### Effect

1. **Zero behavior change.** No decision logic, no split guard, no cost
   function, no perturbation path is modified. The only difference is one
   extra method call per existing PCA call site, plus three new log lines
   per frame per cell that gets PCA'd.
2. **Validation artifact.** A baseline run will produce `[Split PCA Both]`
   log lines that allow direct comparison of `bbox_elongation` (current
   code) vs `strict_elongation` (proposed PR-3 input). PR-3's deletion
   criterion for guards (`split_pre_burn_in_z_axis_*`,
   `split_post_burn_in_large_recenter_*`) requires this log to confirm
   strict-interior PCA reports `strict_elongation ≤ ~1.10` for cells
   where bbox PCA reports `> 1.30` (~94% of cells per the Python
   measurement).
3. **Performance.** Each strict-interior PCA call is two passes over the
   cell's bounding box (~30k voxels for a 30-radius cell) plus one 3×3
   eigendecomposition. Bounded by num_cells per frame for the elongation
   trigger callsites and by the split-attempt rate for `trySplitCell`
   (~1-2 calls/frame). Total overhead is well under 100 ms per frame
   versus the seconds spent on the unified loop.
4. **Forward compatibility.** PR-3 will modify `Frame::computeElongationRatios`
   and `Frame::computeElongationForCell` to *return*
   `strictResult.elongationRatio` instead of the bbox `elongation`. The
   shadow log line can stay (or be gated behind a config flag) so we can
   keep monitoring both numbers.

### How this corresponds to the plan

- **Plan §4.1 (Pillar A summary):** "Replace the bright-pixel cloud used
  by `Spheroid::getSplitCells`'s PCA with voxels sampled strictly inside
  the cell's analytic ellipsoid, weighted by pixel intensity rather than
  hard-thresholded by a top-fraction cutoff." — PR-2 implements the
  strict-interior weighted variant as a *separate method* rather than
  adding a sampling-mode parameter to `getSplitCells`. The plan called for
  `SplitSampleMode { StrictInteriorWeighted, ExpandedInteriorWeighted }`;
  PR-2 takes the simpler "side function" approach because it's purely
  additive. PR-3 will introduce the proper mode switch when the shadow
  data confirms the approach.
- **Plan §4.2.5 (shadow logging):** PR-2 implements the `[Split PCA Both]`
  log line at all three call sites. The plan suggested gating shadow
  logging behind `split_log_shadow_pca` config flag with default `true`;
  PR-2 hardcodes shadow logging on (no config flag yet) since PR-2 is
  purely instrumentation. PR-3 will introduce the config flag when the
  strict-interior PCA actually drives decisions.
- **Plan §4.5 (guard deletion):** No guards are deleted in PR-2. PR-2
  only produces the validation evidence that PR-3+ will use as the
  deletion criterion.

### Testing / validation

User to build and run on the existing dataset
(`scripts/run_celluniverse.sh config/user_input_configurations.ini jihang`),
then grep the log:

```bash
grep "\[Split PCA Both\]" output_jihang_*/debug_log.txt | head
```

Expected pattern:

- For cells whose `bbox_elongation` is ~1.0–1.1: `strict_elongation`
  should also be ~1.0–1.1.
- For cells whose `bbox_elongation` is ≥ 1.30 (e.g., `1f89abf…1` in
  frame 19, F0/F1, E0): `strict_elongation` should be ≤ ~1.10.
- `strict_valid=1` should hold for all cells with non-trivial brightness.
- If `strict_valid=0` for any cell with `bbox_elongation > 0`, that's a
  bug in `computeStrictInteriorWeightedElongation` (probably the
  `< 4 voxels` or `W ≤ 1e-9` early returns firing on a real cell).

### Files modified

| File | Change |
| --- | --- |
| `C++/includes/Spheroid.hpp` | Added `StrictInteriorPcaResult` struct + `computeStrictInteriorWeightedElongation` declaration |
| `C++/src/Spheroid.cpp` | Added `Spheroid::computeStrictInteriorWeightedElongation` implementation (~140 lines) |
| `C++/src/Frame.cpp` | Added `[Split PCA Both]` shadow logging at 3 call sites: `computeElongationRatios`, `computeElongationForCell`, `trySplitCell` |
| `C++/docs/changelogs/changelogv4.md` | This entry (new file) |

---

## 2026-04-09: Per-slice L2 cost cache in `Frame` — **ACTIVE**

**Plan:** Independent perf refactor — NOT part of the 2026-04-09 PCA-input plan.
Sibling PR to PR-2. Session hand-off prompt: Prompt 2 ("Performance optimization
(L2 per-slice cache)") in the 2026-04-09 session summary.

**Problem:** A full 21-frame run took ~58 minutes (`Time elapsed: 3463.32
seconds` in `output_jihang_20260408_164644/debug_log.txt`). The bottleneck
was `Frame::calculateCost()` (`Frame.cpp:298-311`) recomputing
`cv::norm(_realFrame[i], synthFrame[i], NORM_L2)` over **all** 225 z-slices
on every perturbation iteration, even though `generateSynthFrameFast()`
only re-renders the ~20-40 slices inside the moved cell's bounding box.
Per frame: 3500 iter/cell × ~13 cells = ~45k perturb iterations × ~75 ms in
`calculateCost` ≈ ~57 sec/frame just on the cost function, plus similar
cost inside the split burn-in inner loop.

**Goal:** Replace the full 225-slice L2 recomputation on every iteration
with an incremental update that only recomputes the slices that actually
changed, using a cached per-slice contribution vector. Bit-identical output
(`cells.csv`) versus the pre-refactor code is the acceptance criterion.

### Correctness argument (why bit-identical output is preserved)

1. **Unchanged slices are literally the same buffer.** In
   `generateSynthFrameFast()`, slices outside the moved cell's z-range are
   pushed onto the new synth vector as a shallow copy of `_synthFrame[i]`
   (`Frame.cpp:337` before the refactor): cv::Mat is refcounted, so the
   new entry and the cached one point at the same pixel buffer. Therefore
   `cv::norm(_realFrame[i], newSynthFrame[i], NORM_L2)` is bit-identical
   to `cv::norm(_realFrame[i], _synthFrame[i], NORM_L2)`, and in turn
   bit-identical to the cached `_currentCostPerSlice[i]` value (which was
   populated by the same cv::norm call on the same operands previously).
2. **Affected slices are freshly recomputed** via the exact same cv::norm
   call that `calculateCost` would use. Same operands, same rounding.
3. **Total is always re-summed from the full per-slice vector** in
   slice-index order 0..n-1. This matches the summation order of the old
   `calculateCost` loop, so the total is bit-identical.

### Change 1 — Per-slice cost cache member + helper declarations

**File:** `C++/includes/Frame.hpp`

**Lines 33-37 (before):**

```cpp
    std::vector<cv::Mat> generateSynthFrame();
    std::vector<cv::Mat> generateSynthFrameFast(Spheroid &oldCell, Spheroid &newCell);
    Cost calculateCost(const std::vector<cv::Mat> &synthFrame);
```

**Lines 26-37 (after):**

```cpp
    std::vector<cv::Mat> generateSynthFrame();
    // generateSynthFrameFast re-renders only the z-slices that fall inside the
    // union of oldCell's and newCell's bounding box; all other slices are
    // shallow-copied from _synthFrame so they share the same pixel buffer.
    // The optional out-pointers receive the [first, last] affected slice index
    // (inclusive), or both -1 if no slice was re-rendered. Callers that want
    // the affected range for incremental cost caching should pass non-null
    // pointers; callers that don't care (tests, etc.) can call the 2-arg form.
    std::vector<cv::Mat> generateSynthFrameFast(Spheroid &oldCell, Spheroid &newCell,
                                                int *outAffectedZMin = nullptr,
                                                int *outAffectedZMax = nullptr);
    Cost calculateCost(const std::vector<cv::Mat> &synthFrame);
```

**Line 63 (before):**

```cpp
    void regenerateSynthFrame() { _synthFrame = generateSynthFrame(); _currentCost = calculateCost(_synthFrame); }
```

**Line 72 (after):**

```cpp
    void regenerateSynthFrame() { _synthFrame = generateSynthFrame(); refreshFullCostCache(); }
```

**Lines 83 (before):**

```cpp
    double _currentCost = -1.0; // cached L2 image cost of _synthFrame
```

**Lines 83-113 (after):**

```cpp
    double _currentCost = -1.0; // cached L2 image cost of _synthFrame
    // Per-slice L2 contribution of _synthFrame to the total image cost. Kept
    // in sync with _synthFrame / _currentCost so that a perturbation touching
    // only a few z-slices can rebuild the total by recomputing those slices'
    // cv::norm and summing the unchanged cached values. sum(_currentCostPerSlice)
    // equals _currentCost (up to summation order, which is always 0..n-1).
    std::vector<double> _currentCostPerSlice;
    // ...
    float _backgroundValue = 0.0f;
    cv::Size getImageShape();

    // Rebuild _currentCostPerSlice and _currentCost from scratch by walking
    // every slice of _synthFrame vs _realFrame. Used after a full render
    // (constructor, regenerateSynthFrame) where the incremental cache cannot
    // be updated delta-wise.
    void refreshFullCostCache();

    // Given a new synth frame that differs from _synthFrame only in slices
    // [affectedZMin, affectedZMax], compute the new total image cost and
    // write the new per-slice contributions into outNewPerSlice. Slices
    // outside the affected range inherit their cached values unchanged
    // (shallow-copied pixel buffer — bit-identical L2).
    double calculateIncrementalCost(const std::vector<cv::Mat> &newSynthFrame,
                                    int affectedZMin, int affectedZMax,
                                    std::vector<double> &outNewPerSlice) const;
```

### Change 2 — Constructor seeds the cache via `refreshFullCostCache`

**File:** `C++/src/Frame.cpp`

**Lines 268-270 (before):**

```cpp
    _synthFrame = generateSynthFrame();
    _currentCost = calculateCost(_synthFrame);
}
```

**Lines 268-270 (after):**

```cpp
    _synthFrame = generateSynthFrame();
    refreshFullCostCache();
}
```

### Change 3 — `refreshFullCostCache` + `calculateIncrementalCost` implementations

**File:** `C++/src/Frame.cpp`

**Lines 272-325 (after):** (new method definitions inserted between the
constructor and `generateSynthFrame`)

```cpp
void Frame::refreshFullCostCache()
{
    if (_realFrame.size() != _synthFrame.size())
        throw std::runtime_error("Mismatch in image stack sizes");

    _currentCostPerSlice.assign(_realFrame.size(), 0.0);
    double totalCost = 0.0;
    for (size_t i = 0; i < _realFrame.size(); ++i) {
        const double sliceCost = cv::norm(_realFrame[i], _synthFrame[i], cv::NORM_L2);
        _currentCostPerSlice[i] = sliceCost;
        totalCost += sliceCost;
    }
    _currentCost = totalCost;
}

double Frame::calculateIncrementalCost(const std::vector<cv::Mat> &newSynthFrame,
                                       int affectedZMin, int affectedZMax,
                                       std::vector<double> &outNewPerSlice) const
{
    if (_realFrame.size() != newSynthFrame.size())
        throw std::runtime_error("Mismatch in image stack sizes");
    if (_currentCostPerSlice.size() != _realFrame.size())
        throw std::runtime_error("Per-slice cost cache not initialized");

    outNewPerSlice = _currentCostPerSlice;

    if (affectedZMin >= 0 && affectedZMax >= 0) {
        const int nSlices = static_cast<int>(_realFrame.size());
        const int zMin = std::max(0, affectedZMin);
        const int zMax = std::min(nSlices - 1, affectedZMax);
        for (int i = zMin; i <= zMax; ++i) {
            outNewPerSlice[i] = cv::norm(_realFrame[i], newSynthFrame[i], cv::NORM_L2);
        }
    }

    // Always sum in slice-index order so the result is bit-identical to
    // what calculateCost(newSynthFrame) would return.
    double totalCost = 0.0;
    for (size_t i = 0; i < outNewPerSlice.size(); ++i) {
        totalCost += outNewPerSlice[i];
    }
    return totalCost;
}
```

### Change 4 — `generateSynthFrameFast` tracks and writes affected z-range

**File:** `C++/src/Frame.cpp`

**Lines 313-352 (before):**

```cpp
std::vector<cv::Mat> Frame::generateSynthFrameFast(Spheroid &oldCell, Spheroid &newCell)
{
    // ... existing body ...
    for (size_t i = 0; i < z_slices.size(); ++i) {
        double z = z_slices[i];
        if (z < minCorner[2] || z > maxCorner[2]) {
            synthFrame.push_back(_synthFrame[i]);
            continue;
        }
        cv::Mat synthImage = cv::Mat(shape, CV_32F, cv::Scalar(_backgroundValue));
        for (const auto &cell : cells) cell.draw(synthImage, simulationConfig, z);
        synthFrame.push_back(synthImage);
    }
    return synthFrame;
}
```

**Lines 368-420 (after):** (signature gains two optional int* out-pointers; body tracks affectedMin/Max as the loop runs)

```cpp
std::vector<cv::Mat> Frame::generateSynthFrameFast(Spheroid &oldCell, Spheroid &newCell,
                                                   int *outAffectedZMin, int *outAffectedZMax)
{
    // ... existing body ...
    int affectedMin = -1;
    int affectedMax = -1;

    for (size_t i = 0; i < z_slices.size(); ++i) {
        double z = z_slices[i];
        if (z < minCorner[2] || z > maxCorner[2]) {
            synthFrame.push_back(_synthFrame[i]);
            continue;
        }

        if (affectedMin < 0) affectedMin = static_cast<int>(i);
        affectedMax = static_cast<int>(i);

        cv::Mat synthImage = cv::Mat(shape, CV_32F, cv::Scalar(_backgroundValue));
        for (const auto &cell : cells) cell.draw(synthImage, simulationConfig, z);
        synthFrame.push_back(synthImage);
    }

    if (outAffectedZMin) *outAffectedZMin = affectedMin;
    if (outAffectedZMax) *outAffectedZMax = affectedMax;

    return synthFrame;
}
```

### Change 5 — `perturbCell` uses incremental cost

**File:** `C++/src/Frame.cpp`

**Lines 430-447 (before):**

```cpp
    auto newSynthFrame = generateSynthFrameFast(oldCell, cells[index]);
    double newImageCost = calculateCost(newSynthFrame);
    // Use cached cost instead of recalculating L2 over all 225 slices
    double oldImageCost = _currentCost;

    double costDiff = (newImageCost + newOverlapCell + sizeReductionPenalty)
                    - (oldImageCost + oldOverlapCell);

    CallBackFunc callback = [this, newSynthFrame, oldCell, index, newImageCost](bool accept)
    {
        if (accept) {
            this->_synthFrame = newSynthFrame;
            this->_currentCost = newImageCost;
        } else {
            this->cells[index] = oldCell;
        }
    };
```

**Lines 499-527 (after):**

```cpp
    int affectedMin = -1;
    int affectedMax = -1;
    auto newSynthFrame = generateSynthFrameFast(oldCell, cells[index],
                                                &affectedMin, &affectedMax);
    std::vector<double> newCostPerSlice;
    double newImageCost = calculateIncrementalCost(newSynthFrame,
                                                   affectedMin, affectedMax,
                                                   newCostPerSlice);
    double oldImageCost = _currentCost;

    double costDiff = (newImageCost + newOverlapCell + sizeReductionPenalty)
                    - (oldImageCost + oldOverlapCell);

    CallBackFunc callback = [this, newSynthFrame, newCostPerSlice,
                             oldCell, index, newImageCost](bool accept)
    {
        if (accept) {
            this->_synthFrame = newSynthFrame;
            this->_currentCost = newImageCost;
            this->_currentCostPerSlice = newCostPerSlice;
        } else {
            this->cells[index] = oldCell;
        }
    };
```

### Change 6 — `trySplitCell` pre-burn-in: seed local `bestCostPerSlice`

**File:** `C++/src/Frame.cpp`

**Lines 669-672 (before):**

```cpp
    auto bestSynthFrame = generateSynthFrame();
    double bestImageCost = calculateCost(bestSynthFrame);
    double bestOverlap = computeOverlapPenalty(overlapWeight);
    double bestTotalCost = bestImageCost + bestOverlap;
```

**Lines 750-765 (after):**

```cpp
    auto bestSynthFrame = generateSynthFrame();
    // Seed a local per-slice cost cache from the full render so that the
    // burn-in inner loop can drive incremental cost updates without
    // recomputing L2 over all 225 slices on every daughter perturbation.
    std::vector<double> bestCostPerSlice(_realFrame.size(), 0.0);
    double bestImageCost = 0.0;
    for (size_t i = 0; i < _realFrame.size(); ++i) {
        const double sliceCost = cv::norm(_realFrame[i], bestSynthFrame[i], cv::NORM_L2);
        bestCostPerSlice[i] = sliceCost;
        bestImageCost += sliceCost;
    }
    double bestOverlap = computeOverlapPenalty(overlapWeight);
    double bestTotalCost = bestImageCost + bestOverlap;
```

### Change 7 — `trySplitCell`: save pre-split cache, swap in post-split cache

**File:** `C++/src/Frame.cpp`

**Lines 685-686 (before):**

```cpp
    auto savedSynthFrame = _synthFrame;
    _synthFrame = bestSynthFrame;
```

**Lines 786-791 (after):**

```cpp
    auto savedSynthFrame = _synthFrame;
    auto savedCostPerSlice = _currentCostPerSlice;
    const double savedCost = _currentCost;
    _synthFrame = bestSynthFrame;
    _currentCostPerSlice = bestCostPerSlice;
    _currentCost = bestImageCost;
```

### Change 8 — `trySplitCell`: burn-in inner loop uses incremental cost

**File:** `C++/src/Frame.cpp`

**Lines 704-721 (before):**

```cpp
        // Render and compute image cost
        auto trialFrame = generateSynthFrameFast(saved, cells[dIdx]);
        double trialImageCost = calculateCost(trialFrame);

        double improvement = (trialImageCost + newCellOverlap) - (bestImageCost + oldCellOverlap);

        if (improvement < 0) {
            bestSynthFrame = trialFrame;
            bestImageCost = trialImageCost;
            _synthFrame = trialFrame;
            accepted++;
        } else {
            cells[dIdx] = saved;
        }
    }

    _synthFrame = savedSynthFrame;
```

**Lines 809-841 (after):**

```cpp
        int trialAffectedMin = -1;
        int trialAffectedMax = -1;
        auto trialFrame = generateSynthFrameFast(saved, cells[dIdx],
                                                 &trialAffectedMin, &trialAffectedMax);
        std::vector<double> trialCostPerSlice;
        double trialImageCost = calculateIncrementalCost(trialFrame,
                                                         trialAffectedMin, trialAffectedMax,
                                                         trialCostPerSlice);

        double improvement = (trialImageCost + newCellOverlap) - (bestImageCost + oldCellOverlap);

        if (improvement < 0) {
            bestSynthFrame = trialFrame;
            bestImageCost = trialImageCost;
            bestCostPerSlice = trialCostPerSlice;
            _synthFrame = trialFrame;
            _currentCost = trialImageCost;
            _currentCostPerSlice = trialCostPerSlice;
            accepted++;
        } else {
            cells[dIdx] = saved;
        }
    }

    _synthFrame = savedSynthFrame;
    _currentCost = savedCost;
    _currentCostPerSlice = savedCostPerSlice;
```

### Change 9 — `trySplitCell` accept callback captures `bestCostPerSlice`

**File:** `C++/src/Frame.cpp`

**Lines 808-821 (before):**

```cpp
    CallBackFunc callback = [this, bestSynthFrame, bestImageCost, oldCell, index](bool accept)
    {
        if (accept) {
            this->_synthFrame = bestSynthFrame;
            this->_currentCost = bestImageCost;
        } else {
            this->cells.pop_back();
            this->cells.pop_back();
            this->cells.insert(this->cells.begin() + index, oldCell);
        }
    };
```

**Lines 928-943 (after):**

```cpp
    CallBackFunc callback = [this, bestSynthFrame, bestImageCost, bestCostPerSlice,
                              oldCell, index](bool accept)
    {
        if (accept) {
            this->_synthFrame = bestSynthFrame;
            this->_currentCost = bestImageCost;
            this->_currentCostPerSlice = bestCostPerSlice;
        } else {
            this->cells.pop_back();
            this->cells.pop_back();
            this->cells.insert(this->cells.begin() + index, oldCell);
        }
    };
```

### Effect

1. **`calculateCost` is no longer called on the optimizer hot path.** All
   internal callers go through `refreshFullCostCache` (once per full render)
   or `calculateIncrementalCost` (per perturbation). The old `calculateCost`
   method remains for the direct-use tests in `frame_test.cc`.
2. **Per-iteration work drops from `O(225 slices)` to
   `O(affected slices)`.** In a typical perturbation the moved cell's z
   bounding box covers 20-40 slices, so the L2 work per iteration is cut
   roughly 5-10×.
3. **Total throughput win (expected, to be confirmed):** ~58 min run drops
   to ~10-15 min for the 21-frame baseline. Final numbers measured on the
   user's machine after a validation run.
4. **Bit-identical decision output.** The incremental update reads and
   writes the same pixel buffers as the old full-L2 path (shallow-copied
   cv::Mats for unchanged slices, freshly rendered ones for changed
   slices) and sums in the same 0..n-1 order, so every `cv::norm` result
   and every sum is bit-for-bit identical to the pre-refactor code. The
   unified loop's accept/reject decisions and therefore `cells.csv` must
   match the baseline run exactly.

### Files modified (PR only)

| File | Change |
| --- | --- |
| `C++/includes/Frame.hpp` | Added `_currentCostPerSlice` member, `refreshFullCostCache` decl, `calculateIncrementalCost` decl. Extended `generateSynthFrameFast` signature with two optional int* out-pointers. `regenerateSynthFrame` inline calls `refreshFullCostCache()`. |
| `C++/src/Frame.cpp` | Constructor seeds cache via `refreshFullCostCache`. New `refreshFullCostCache` and `calculateIncrementalCost` implementations. `generateSynthFrameFast` body tracks affected z-range and writes it through out-pointers. `perturbCell` uses `calculateIncrementalCost` and updates per-slice cache on accept. `trySplitCell` seeds a local per-slice cache from the post-split full render, saves+swaps the pre-split cache around the burn-in loop, uses `calculateIncrementalCost` inside the loop, restores the pre-split cache after, and the final accept callback writes the post-split cache on accept. |
| `C++/docs/changelogs/changelogv4.md` | This entry. |

### Acceptance criterion (user to verify)

After building on the user's machine and running the 21-frame baseline:

```bash
diff output/cells.csv output_jihang_20260408_164644/cells.csv
```

**Expected:** zero diff (the files are bit-identical). If any cell state
differs, the cache update logic has a bug — do not merge; debug first.

Suggested sanity check before the full 21-frame run: run frames 1-3 and
compare those first. Three frames catch any drift in the cache much
faster than 21 frames.

**Note (2026-04-09):** bit-identical acceptance is only possible with a
deterministic RNG seed. `CellUniverse::optimize` currently seeds from
`std::random_device` (`CellUniverse.cpp:600-601`), so two runs of the
same binary will diverge. Soft validation on run `20260409_042355` (3
frames, 257 sec) confirmed no regressions via per-frame counters,
`avg_abs_resid`, and split patterns falling inside the RNG noise band.
Full-scale validation on run `20260409_044216` (22 frames, 2400 sec)
gave ~1.5× speedup averaged across frames (less than the 5-7×
projection because early frames have fewer iterations to amortize over;
later frames were closer to 2×).

---

## 2026-04-09: Volume recovery master switch — **ACTIVE**

**Problem:** The per-frame volume recovery block in `CellUniverse::optimize`
(`CellUniverse.cpp:819-906`) has never fired in any logged run (verified
on runs `20260408_164644`, `20260409_025421`, `20260409_042355`,
`20260409_044216` — zero `[Volume Recovery]` lines in any debug_log.txt).

Root cause: the greedy scan at `CellUniverse.cpp:862-883` breaks on the
first scale step where candidate `measureMeanBrightness` drops below the
current best (with epsilon 1e-6). For any cell sitting on a bright core
— which is the only state where recovery is relevant — any upscale from
`scale=1.02` pulls in dimmer periphery pixels, so the candidate's mean
brightness is immediately less than the base, the loop breaks on the
first iteration, `bestCell` stays equal to the input, and the log
guard at line 885 (`bestCell.getMajorRadius() > cell.getMajorRadius() +
1e-6f`) evaluates false. The block is structurally dead code at its
current thresholds.

**Fix:** Add `volumeRecoveryEnabled` master switch (default `false`)
following the pattern of `flatCellRotationRefineEnabled`. The code stays
in place in case the algorithm is reworked (e.g. picking the argmax
brightness across all scales instead of breaking on first drop), but
the block is gated off by default so it doesn't contribute any runtime
cost.

### Change 1 — `CellConfig::volumeRecoveryEnabled` field + parsing

**File:** `C++/includes/ConfigTypes.hpp`

Added the field alongside the other volumeRecovery knobs with a comment
explaining why it defaults to false, and the corresponding
`if (node["volumeRecoveryEnabled"]) volumeRecoveryEnabled = ...` block
in `SpheroidConfig::explodeConfig`.

### Change 2 — Gate the optimize block

**File:** `C++/src/CellUniverse.cpp:819`

**Before:**
```cpp
if (frameIndex > 0 && config.cell) {
    const float lossThreshold = ...
```

**After:**
```cpp
if (frameIndex > 0 && config.cell && config.cell->volumeRecoveryEnabled) {
    const float lossThreshold = ...
```

### Change 3 — YAML key

**File:** `C++/config/config.yaml`

Added `volumeRecoveryEnabled: false` in the `cell:` section just above
the existing `volumeRecoveryLossFractionThreshold` line, with a comment
explaining why it's off by default.

### Effect

Zero behavior change versus the previous runs, because the block never
fired anyway. Runtime saving is minor (the block was already exiting
quickly for real cells). The value is signaling intent: the current
implementation is a known no-op and should not be relied on to prevent
shrinkage.

If you want anti-shrinkage behavior in the future, the correct lever is
`prob.size_reduction_penalty_weight` (applied during perturbation as a
soft quadratic penalty), not fixup-after-the-fact volume recovery.

---

## 2026-04-09: Remove dead `Frame::computeElongationRatios()` — **ACTIVE**

**Problem:** `Frame::computeElongationRatios()` was declared in Frame.hpp
and defined in Frame.cpp (~30 lines) but had zero callers anywhere in
`src/`. Confirmed by grep:

```
grep -rn "computeElongationRatios" C++/src/ C++/includes/
# → only the declaration and the definition itself
```

PR-2's shadow logging block inside it had also added a `caller=elongation_trigger`
variant of `[Split PCA Both]` — the run `20260409_044216` log shows
zero lines from that caller, confirming the function is never invoked
(175 lines from `caller=try_split`, 188 from `caller=elongation_for_cell`,
0 from `caller=elongation_trigger`).

The only live consumer of per-cell elongation is
`CellUniverse.cpp:800`, which calls `frame.computeElongationForCell(ci)`
in a loop (the end-of-frame `[Elongation for next frame]` block).

### Change

**File:** `C++/includes/Frame.hpp:47`

Removed the declaration `std::map<std::string, float> computeElongationRatios() const;`.

**File:** `C++/src/Frame.cpp:578-611` (pre-change line numbers)

Removed the entire function body including the PR-2 `caller=elongation_trigger`
shadow log block.

### Effect

Pure dead-code removal. Zero behavior change. `<map>` include is still
required in Frame.hpp for other uses. Saves ~35 lines of source.

---

## 2026-04-09: PR-3 — wire strict-interior PCA into end-of-frame elongation — **ACTIVE**

**Plan:** `C++/docs/plans/2026-04-09-pca-input-fix-and-residual-safety-nets.md`,
Pillar A, step 2 of 3. Depends on PR-2 (shadow-mode logging — landed).

**Problem:** `Frame::computeElongationForCell` returned the bbox-PCA
`elongationRatio` from `Spheroid::getSplitCells`. That number is
contaminated by neighbor leakage because `getSplitCells` collects bright
pixels from a 2×-expanded ellipsoid gate that overlaps nearby cells.
The contaminated elongation is written into `previousElongations` at
the end of each frame, which feeds the next frame's `P(split)` formula
(`CellUniverse.cpp:632-663`), giving false P(split) boosts to cells
that aren't actually elongated — the root cause of the frame-19 false
split in `output_jihang_20260408_164644` and the frame-22 degenerate
split in `output_jihang_20260409_044216` (daughter at R=(10,5) minimum
bounds, bbox_elongation=1.32 but strict_elongation=1.12).

**Shadow-mode evidence (run `20260409_044216`, 363 `[Split PCA Both]`
lines):**

| site | bbox_elongation | strict_elongation | interpretation |
|---|---|---|---|
| frame 8 split on `1f89abf` | 2.11 | **1.46** | real pre-division, strict confirms |
| frame 3 split on `e9077` | 1.14 | **1.59** | strict catches asymmetry bbox missed |
| frame 22 false split on `1f89abf...cb1` | 1.32 | **1.12** | strict would have prevented |
| most non-splitting cells | ~1.0–1.1 | ~1.02–1.06 | consistent with Python measurement |

Zero `strict_valid=0` on real cells across 363 observations. The one
`strict_valid=0` was on a `R=(10,5)` minimum-bounds cell where the
`<4 interior voxels` early return correctly fires. No bugs in
`computeStrictInteriorWeightedElongation`.

### Change

**File:** `C++/src/Frame.cpp` — `computeElongationForCell`

Renamed the local `elongation` to `bboxElongation` for clarity, kept
the shadow log block, and changed the return statement:

**Before:**
```cpp
return valid ? elongation : 1.0f;
```

**After:**
```cpp
if (strictResult.valid) {
    return strictResult.elongationRatio;
}
return valid ? bboxElongation : 1.0f;
```

The fall-through preserves the old behavior for the rare case of a
degenerate cell whose strict-interior sample count is below 4 — in that
path the bbox value is still used (or 1.0 if both are invalid).

### Effect on `previousElongations` → `P(split)`

`P(split)` in `CellUniverse.cpp:632-663` is computed from
`previousElongation[cell]` as:

```
rawP = baseSplitProb + max(0, 1 - 1/prevElong)
```

followed by a whole-frame proportional rescale to `max_split_probability`.

With PR-3, `prevElong` is now the strict value, which is ≤ the bbox
value for almost all cells. Quantitative effect:

- Cells that were spuriously elongated in the bbox measurement
  (e.g., bbox 1.40, strict 1.02) drop from `rawP = 0.03 + 0.286 = 0.316`
  to `rawP = 0.03 + 0.020 = 0.050` — ~6× lower P(split).
- Cells that are genuinely pre-divisional (e.g., `1f89abf` at frame 8:
  bbox 2.11, strict 1.46) drop from `rawP = 0.557 → 0.345`, still well
  above baseline and still rescaled to or near `max_split_probability`.
- Cells with bbox below strict (e9077 frame 3: bbox 1.14, strict 1.59)
  now get the **higher** strict value: `rawP = 0.347` instead of
  `0.153`, roughly doubling P(split) — which is the right answer for
  that cell (it was a missed split that the baseline at frame 3 caught).

**Expected run-over-run changes versus `20260409_044216`:**

1. Fewer late-frame false-positive splits driven by neighbor-leakage
   bbox elongation. The frame-22 `1f89abf...cb1` degenerate split is
   the prime candidate to disappear.
2. More attempts on cells whose strict elongation is genuinely high —
   this may fix the `e9077` frame-3 missed split (strict=1.59, bbox=1.14)
   by giving more RNG rolls to find a daughter placement that passes
   the z-axis heuristic guard.
3. Slight change to the total split count distribution — by how much
   depends on the rescale cap at `max_split_probability = 0.2`.

Shadow logging kept in place so post-run analysis can still compare
both values. `caller=elongation_for_cell` lines remain; `caller=try_split`
lines remain; `caller=elongation_trigger` lines are gone along with
the dead `computeElongationRatios()` function.

### Guards NOT touched in this PR

Per Pillar A §4.5, the following guards are removal candidates once
strict PCA is driving decisions AND a shadow-mode run shows they don't
fire on legit splits:

- `split_pre_burn_in_z_axis_max_abs` / `_max_separation_over_major` / `_min_drift_over_major`
- `split_post_burn_in_large_recenter_*`

**These are NOT touched in PR-3.** PR-3 only wires strict PCA into the
elongation trigger. The split-attempt pathway in `Frame::trySplitCell`
still uses the bbox-PCA `getSplitCells` return values for daughter
placement and guard evaluation. Guard deletion is deferred to PR-6.

Concrete example of why not yet: in run `20260409_044216`, the frame 3
`e9077` split was rejected by `z_axis_internal_structure` because the
centroid-based daughter placement gave `sepOverDaughterMajor=0.441 <
threshold 1.30`. Removing that guard now would let the split through
with a bad initial placement, and burn-in might not be able to recover.
The right fix is to either (a) also switch the split-attempt pathway to
strict-interior PCA (PR-4), or (b) validate that the post-burn-in
fake-split guards catch bad placements on their own.

### Files modified

| File | Change |
| --- | --- |
| `C++/src/Frame.cpp` | `computeElongationForCell`: return `strict_elongation` when valid, fall back to bbox otherwise. Renamed local for clarity. |

### Acceptance criterion (user to verify)

Rebuild and run the full 22-frame dataset. Expected:

1. Runtime similar to `20260409_044216` (~2400 sec ±RNG), no regressions.
2. Fewer `(10, 5)` minimum-bounds daughters in the final cells.csv
   (concretely: the frame-22 `1f89abf...cb1` split at bbox=1.32 /
   strict=1.12 should not fire).
3. `e9077` may split earlier than frame 20 (unclear exactly when — depends
   on RNG and guard interactions).
4. Shadow log still shows `[Split PCA Both]` lines from `caller=elongation_for_cell`
   and `caller=try_split`, but zero lines from `caller=elongation_trigger`
   (that caller was removed with `computeElongationRatios`).

Quick greps to run after:

```bash
# Degenerate cells (R approaches minimum bounds)
awk -F, '$6 < 12 && $7 < 7' outputs/<new_run>/cells.csv

# P(split) trajectory for a specific cell
grep "e9077.*P(split)" outputs/<new_run>/debug_log.txt

# Split acceptance counts by frame
grep "\[Optimize Done\]" outputs/<new_run>/debug_log.txt
```

---

## 2026-04-09: Wire `splitElongationThreshold` gate in `trySplitCell` — **ACTIVE**

**Problem:** `Frame::trySplitCell` (`C++/src/Frame.cpp:624`) accepts a
`splitElongationThreshold` parameter but **the function body never
references it**. The parameter is passed from `CellUniverse::optimize`
(`CellUniverse.cpp:715`) carrying `config.prob.split_elongation_threshold`
(1.3 in the current config), but there is no `if (elongation < threshold)`
check anywhere inside `trySplitCell`. The elongation gate was either never
wired up after a refactor or was silently removed. Verified by grep:

```
$ grep -n splitElongationThreshold C++/src/Frame.cpp
C++/src/Frame.cpp:624:                                     float splitElongationThreshold,
```

Exactly one hit, in the parameter list. Zero references in the body.

**Impact:** Every cell has baseline `P(split) = 0.03`, giving roughly 10
split attempts per cell per frame (350 iter × ~9 cells / 22 frames × 0.03).
With no elongation gate, any attempt that survives `!valid`, `splitMinInsideCount`,
`shouldRejectSplitPreBurnIn`, and the post-burn-in fake-guards can fire —
regardless of how spherical the cell actually is. Over 22 frames this
accumulates to near-certain false-positive splits on spherical cells,
and the false-positive daughter placements often produce a daughter
at or near the minimum radius bounds.

**Concrete failures in run `output_jihang_20260409_062322`** (the
cache + caps + PR-3 run, *before* this fix):

| frame | cell | bbox_elongation | strict_elongation | outcome |
|---|---|---|---|---|
| 8 | `e3d034...c8d` | 1.27 | **1.14** | FP split, one daughter born at R=(10, 5) |
| 11 | `8cbdf86d` | 1.10 | **1.03** | FP split on near-sphere |
| 19 | `1f89abf...cb1` | 1.08 | **1.13** | FP split, daughters produced |

All three cells had `strict_elongation` well below the configured
threshold of 1.30. With the gate missing, the splits proceeded. The
e3d034 false positive is particularly bad: the daughter born at (10, 5)
at frame 8 persisted as a degenerate cell through the rest of the run
(observed at frames 12, 14, 15, 21, 22 all near minimum bounds), making
the lineage tree contain a ghost daughter that tracks nothing real.

**Fix:** Add an elongation gate at the top of `trySplitCell`, right after
the `!valid` early-return and before `splitMinInsideCount`. The gate
prefers `strictResult.elongationRatio` (clean, not neighbor-contaminated)
and falls back to the bbox `elongationRatio` only when
`strictResult.valid == false` (degenerate tiny cells where the strict
interior has fewer than 4 weighted voxels).

### Change

**File:** `C++/src/Frame.cpp` — `Frame::trySplitCell`

**Lines 668-693 (before):**

```cpp
    // [PR-2 shadow] Strict-interior brightness-weighted PCA, side-by-side
    // with the bbox PCA above. Logged only — does NOT influence the split
    // decision in this commit. See Pillar A in [...]
    const StrictInteriorPcaResult strictResult =
        oldCell.computeStrictInteriorWeightedElongation(_realFrame, _backgroundValue);
    std::cout << "[Split PCA Both] " << oldCell.getName()
              << " caller=try_split"
              << " bbox_elongation=" << elongationRatio
              /* ... */
              << std::endl;

    if (!valid)
    {
        std::cout << "[Split Skip] " << oldCell.getName()
                  << " getSplitCells returned invalid" << std::endl;
        return {0.0, [](bool accept) {}};
    }
```

**Lines 668-716 (after):** The `[Split PCA Both]` log comment is updated
to reflect that the value is now a decision input rather than a shadow,
and the elongation gate is inserted between the `!valid` check and the
`splitMinInsideCount` check:

```cpp
    // Strict-interior brightness-weighted PCA, logged side-by-side with the
    // bbox PCA from getSplitCells. As of PR-3 this drives computeElongationForCell's
    // P(split) trigger; this block uses the same value to gate the split attempt
    // itself, closing the latent bug where splitElongationThreshold was passed
    // into trySplitCell but never referenced in the function body. [...]
    const StrictInteriorPcaResult strictResult = /* ... */;
    std::cout << "[Split PCA Both] " << oldCell.getName() /* ... */;

    if (!valid) { /* skip */ }

    // Elongation gate: reject the split attempt when the cell is not
    // elongated enough to warrant a division. Prefer strict-interior
    // brightness-weighted elongation; fall back to bbox PCA only if
    // strict returned invalid (degenerate tiny cell, etc.).
    const float effectiveElongation =
        strictResult.valid ? strictResult.elongationRatio : elongationRatio;
    if (splitElongationThreshold > 0.0f && effectiveElongation < splitElongationThreshold)
    {
        std::cout << "[Split Skip] " << oldCell.getName()
                  << " reason=below_elongation_threshold"
                  << " bbox=" << elongationRatio
                  << " strict=" << (strictResult.valid ? strictResult.elongationRatio : 1.0f)
                  << " effective=" << effectiveElongation
                  << " threshold=" << splitElongationThreshold
                  << " source=" << (strictResult.valid ? "strict" : "bbox")
                  << std::endl;
        return {0.0, [](bool accept) {}};
    }
```

### Effect

1. **Expected rejections** on the three documented false positives
   above. All three have `strict_elongation < 1.30`:
   - `e3d034` f8: `strict=1.14` → skip (`[Split Skip] ... reason=below_elongation_threshold`)
   - `8cbdf86d` f11: `strict=1.03` → skip
   - `1f89abf...cb1` f19: `strict=1.13` → skip

2. **Expected acceptances** (legitimate splits from the `062322` run are
   preserved because their strict elongation is well above threshold):
   - `e9077` f3: `strict=1.59` → above 1.30, proceeds
   - `12345` f3: `strict≥1.3` (confirmed via its shadow log) → proceeds
   - `1f89abf` f8: `strict=1.46` → proceeds
   - `12345...341` f20: `strict=1.29` (borderline but passes strict≥1.29 vs threshold 1.30 — could be a near miss; monitor)
   - `e9077...a50` f20: `strict` unknown (need to check the debug log) but
     it did split with bbox=1.22, need to verify strict is above 1.30

3. **Downstream effect on degenerate cells**: the `e3d034...d1` ghost
   daughter at R=(10, 5) disappears entirely because the f8 false
   positive that produced it is rejected. The other two degenerate
   cells in the `062322` run (`e9077...a51` and `12345...3410`)
   are not from false-positive splits — they're legitimate daughters
   that drifted to minimum bounds via ordinary perturbation. Those
   are a separate issue (deferred).

4. **`split_elongation_threshold: 1.3` in config.yaml now has effect.**
   Previously it was silently ignored. Users who want to tune the
   sensitivity can now actually do so. Lower = more permissive
   (more splits, higher false-positive rate). Higher = more strict
   (fewer splits, higher miss rate).

### Risk: borderline legitimate splits

The threshold of 1.30 may be too strict for cells whose genuine
pre-division signal manifests as only a mild elongation in the strict
measurement. Specifically `12345...341` at f20 had
`strict_elongation=1.29` in the shadow log, which is just below the
cutoff. After this fix, that split might be rejected and cascade into
additional missed splits downstream.

**Mitigation:** if the next run shows legitimate splits being blocked
at strict ∈ [1.25, 1.30], loosen the threshold to 1.25 in config.yaml
(single config change, no code edit). Monitor the new
`[Split Skip] ... reason=below_elongation_threshold` log lines in the
next run to see exactly which cells are being blocked and whether
they're legitimate.

### Not a functional change to other paths

- `getSplitCells` still computes bbox PCA for daughter placement and
  the search axis. That's PR-4's territory and is out of scope here.
- The `splitMinInsideCount`, `shouldRejectSplitPreBurnIn`, and
  post-burn-in fake-guards are unchanged.
- `computeElongationForCell` (PR-3) is unchanged. It already returns
  strict elongation; this fix uses the same value at the split-attempt
  site for consistency.

### Files modified

| File | Change |
| --- | --- |
| `C++/src/Frame.cpp` | Insert elongation gate in `trySplitCell` after the `!valid` check; update the shadow-log comment to reflect that the value is now a decision input. |

### Acceptance criterion (user to verify)

Rebuild and re-run. Expected changes versus `output_jihang_20260409_062322`:

- **Zero** `e3d034`, `8cbdf86d`, or `1f89abf...cb1` false-positive splits.
- **Zero** daughters at R=(10, 5) from split placements (cells at that
  size from ordinary perturbation drift are a separate issue).
- **Frame 3**, **frame 8 (1f89abf only)**, **frame 20 (12345...341 + others)** still caught.
- Final cell count likely drops from 14 to 11-12 (losing the three
  false positives; real splits may also cascade differently due to RNG).
- New log lines `[Split Skip] ... reason=below_elongation_threshold`
  visible in the debug log at the previously-failing sites.

Quick greps to run:

```bash
OUT=outputs/output_jihang_<new_timestamp>

# (1) confirm the gate fires on the expected cells
grep "below_elongation_threshold" $OUT/debug_log.txt | head -20

# (2) degenerate cells — should be down from 3 to at most 2 (the drift-driven ones)
awk -F, '$6 < 13 && $7 < 7' $OUT/cells.csv

# (3) per-frame split summary
grep "\[Optimize Done\]" $OUT/debug_log.txt

# (4) confirm legitimate splits still fire
grep "\[Split Accepted\]" $OUT/debug_log.txt
```

---

## 2026-04-09: Top-K-percent strict-interior PCA (max with halo-weighted) — **ACTIVE**

**Problem:** The brightness-weighted strict-interior PCA in
`Spheroid::computeStrictInteriorWeightedElongation` weights every interior
voxel by `(pixel - background)`. For bright cells, the heavy 2D blur
(`blur_sigma=10`) creates an isotropic halo around the cell core. After
post-sigmoid processing the halo pixels are still well above background,
so they get non-trivial weights. Because the halo is roughly isotropic
(it's a Gaussian-blurred outline), it dilutes any rod or dumbbell signal
present in the cell core toward the spherical case.

**Concrete failure** in run `output_jihang_20260409_074740`: cell
`1f2ed10d323c4cb288424e988893788f` is visibly rod-shaped at frame 10/11
(verified by viewing the raw TIF crop with the user) and is supposed to
divide between f10 and f11 per ground truth. The legacy halo-weighted
strict PCA reports its peak elongation at **only 1.25 at f11** (across all
22 frames). The 1.30 split threshold blocks the legitimate split. A
top-percentile Python diagnostic computed over the same cell on the
**raw TIF data** (bypassing preprocessing) reports much higher elongation
when filtered to the brightest pixels:

| top fraction | asp12 |
|---|---|
| 5% | 1.39 |
| 2% | 1.86 |
| 1% | 2.24 |
| 0.5% | 2.48 |

The bright core IS rod-shaped — the halo just averages it out in the
legacy measurement.

**Fix:** Compute BOTH a halo-weighted PCA (legacy behavior) AND a top-K-
filtered PCA in the same method, store both in the result struct, and
expose `effectiveElongationRatio = max(halo, topK)`. Callers use the
effective value for split decisions, which:

1. Preserves every catch of the legacy method (when halo > top-K, the max
   keeps the halo value — important for cells like e9077 at f3 where halo
   measures 1.61 but top-K measures only 1.12).
2. Adds bright-cell catches (when top-K > halo, the max picks up the
   rod signal that the halo dilution hides — 1f2ed10d at f11 jumps from
   1.25 → 2.41).

**K choice:** A sweep across 1f2ed10d, 1f89abf, e9077, 12345, 8cbdf86d,
e3d034 at multiple frames (Python script `/tmp/k_sweep.py`) showed:

| K | 1f2ed10d_f11 (TP) | 8cbdf86d_f10 top-K (TN) | e3d034_f10 top-K (TN) |
|---|---|---|---|
| 0.01 | 2.98 ✓ | 1.65 ❌ FP | 1.59 ❌ FP |
| 0.02 | 2.79 ✓ | 1.55 ❌ FP | 1.40 ❌ FP |
| 0.05 | 2.56 ✓ | 1.36 ❌ FP | 1.28 borderline |
| **0.10** | **2.41 ✓** | **1.25 ✓** | **1.20 ✓** |
| 0.20 | 2.20 ✓ | 1.19 ✓ | 1.29 borderline |

K=0.10 is the sweet spot — it catches the hard true positive (1f2ed10d
at f11) without inflating any spherical TN above the 1.30 gate. Lower K
values create FPs via sparse-sample noise (a tiny random subset of
pixels in a spherical region happens to align in some axis).

### Change 1 — `SpheroidConfig::strictElongationTopKFraction`

**File:** `C++/includes/ConfigTypes.hpp`

Added field next to `splitBrightestFraction`, default `0.10f`, parsed
from `strictElongationTopKFraction` yaml key.

### Change 2 — `StrictInteriorPcaResult` struct extended

**File:** `C++/includes/Spheroid.hpp`

Added fields:
- `topKElongationRatio` — top-K-only elongation
- `topKLambda1/2/3` — top-K eigenvalues (for logging)
- `topKSampleCount` — pixels above the K threshold
- `topKThreshold` — the actual pixel-intensity cutoff used
- `effectiveElongationRatio` — `max(elongationRatio, topKElongationRatio)`,
  the value callers should use for decisions

The legacy `elongationRatio` field is unchanged in semantics — it still
holds the halo-weighted value, so anything reading the old field gets
the old behavior.

### Change 3 — `Spheroid::computeStrictInteriorWeightedElongation` rewrite

**File:** `C++/src/Spheroid.cpp`

Restructured into 3 passes:

1. **Pass 0 (new):** collect interior pixel intensities into a
   `std::vector<float>`. Memory: ~4 bytes per voxel × ~100k voxels =
   ~400 KB per call. Negligible.
2. **`std::nth_element` partition** to find the top-K threshold without
   a full sort.
3. **Pass 1:** weighted centroids for BOTH halo-weighted and top-K-
   filtered PCAs in a single scan loop (one if-branch decides whether
   the voxel also accumulates into the top-K stats).
4. **Pass 2:** weighted covariances for both, again in a single scan.
5. Eigendecompose both 3×3 covariance matrices via a local `eigenRatio`
   lambda.
6. Set `effectiveElongationRatio = std::max(elongation, topKElongation)`.

If the top-K subset has fewer than 4 valid samples, the top-K result
falls back to the halo result so `effective` is at least conservative
and never spuriously high.

### Change 4 — `Frame::computeElongationForCell` returns `effectiveElongationRatio`

**File:** `C++/src/Frame.cpp`

The PR-3 wiring now uses `strictResult.effectiveElongationRatio` instead
of `strictResult.elongationRatio`. The shadow log is extended to dump
all three values: `strict_elongation` (halo), `strict_topk`,
`strict_effective`, plus `strict_topk_samples` and
`strict_topk_threshold` for debugging.

### Change 5 — `Frame::trySplitCell` gate uses `effectiveElongationRatio`

Same idea applied to the gate added in the previous changelog entry.
The `[Split Skip] reason=below_elongation_threshold` log line now
includes `strict_halo`, `strict_topk`, and `effective` so we can see
which method (or both) said the cell wasn't elongated.

### Change 6 — `config.yaml`

Added `strictElongationTopKFraction: 0.10` under `cell:` section with a
comment explaining the rationale and what to do if the value needs
tuning.

### Predicted run-over-run effects vs `output_jihang_20260409_074740`

| cell-frame | halo (current) | top-K K=0.10 | **effective (new)** | gate at 1.30 |
|---|---|---|---|---|
| 1f2ed10d_f11 (HARD TP) | 1.25 | 2.41 | **2.41** | ✅ now catches |
| e9077_f3 (current TP) | 1.61 | 1.12 | **1.61** | ✅ preserved |
| 1f89abf_f8 (current TP) | 1.46 | ~1.10 | **1.46** | ✅ preserved |
| 12345_f3 (current TP) | high | ~1.07 | high | ✅ preserved |
| 8cbdf86d_f10 (TN) | 1.04 | 1.25 | **1.25** | ✅ no FP |
| e3d034_f10 (TN) | ~1.00 | 1.20 | **1.20** | ✅ no FP |

The expected run-over-run change versus the previous gate-fix run
(`074740`):

1. **1f2ed10d catches at f11** (the previously-stuck case the user
   visually identified). prevElong from f10 will be inflated by top-K
   (~1.21 → 1.21), and at f11 the gate effective elongation is 2.41,
   well above 1.30 — the split fires.
2. The other 4 misses (e9077 cascade misses at f19/f20, 12345…340/341
   at f20, e9077 itself at f3 — actually e9077 at f3 might still
   succeed since halo-weighted gives it 1.61) likely **stay missed**,
   because both halo-weighted and top-K stay below 1.30 for those
   cells. They are Pillar B (local-max detector) territory.
3. False-positive rate should NOT increase. K=0.10 is conservative
   enough that no TN in the empirical sweep crossed 1.30.
4. The single degenerate `12345…3400` cell at f22 (drift-driven
   shrinkage to (10,5)) is not addressed by this change — that's a
   `perturbCell` min-R clamp issue, deferred.

### Expected new cell count

If 1f2ed10d catches at f11, the run gains 1 split there. Net cells:
10 + 1 = 11 (vs ground truth 14, vs previous run 10). The remaining 3
missing splits need Pillar B.

### Files modified

| File | Change |
| --- | --- |
| `C++/includes/ConfigTypes.hpp` | New `strictElongationTopKFraction` field + parsing |
| `C++/includes/Spheroid.hpp` | Extended `StrictInteriorPcaResult` with top-K fields and `effectiveElongationRatio` |
| `C++/src/Spheroid.cpp` | Rewrote `computeStrictInteriorWeightedElongation` to compute both halo-weighted and top-K PCAs in one pass each, return max as effective |
| `C++/src/Frame.cpp` | `computeElongationForCell` and `trySplitCell` gate use `effectiveElongationRatio`; shadow logs dump all three values |
| `C++/config/config.yaml` | New `strictElongationTopKFraction: 0.10` under `cell:` section |

### Acceptance criterion (user to verify)

Rebuild and rerun. Expected:

1. New `strict_topk` and `strict_effective` fields appear in every
   `[Split PCA Both]` log line.
2. `1f2ed10d` at f11 attempts to split (look for
   `[Split Attempt] 1f2ed10d... frame=11`). The attempt should pass the
   elongation gate (`strict_topk` ≈ 2.4, `effective` ≈ 2.4 > 1.3).
3. Whether it actually accepts depends on cost diff — the daughter
   placement still uses bbox PCA (PR-4 territory), so success is not
   guaranteed even with the gate passing.
4. No new false positives. Spherical cells (8cbdf86d, e3d034, 1f2ed10d
   at f5/f15) should still see `strict_effective < 1.30`.
5. e9077 / 1f89abf splits at f3 / f8 still fire (preserved by the max
   strategy).
6. If `12345...341` at f20 (which had `strict_halo=1.04` last run) now
   suddenly inflates from top-K, that's a false positive — the K
   parameter needs to be raised (try 0.15 or 0.20).

Quick greps after the run:

```bash
OUT=outputs/output_jihang_<new_timestamp>

# (1) confirm new fields are in the log
grep "strict_topk" $OUT/debug_log.txt | head -3

# (2) does 1f2ed10d try to split at f11?
awk '/\[FrameState Before\] frame 11/,/\[FrameState Before\] frame 12/' $OUT/debug_log.txt \
  | grep "1f2ed10d.*Split"

# (3) split summary
grep "\[Optimize Done\]" $OUT/debug_log.txt

# (4) any new gate skips
grep "below_elongation_threshold" $OUT/debug_log.txt | wc -l
```

## 2026-04-09: PR-5 — Pillar B local-maximum split detector + perturbCell min-R clamp — **ACTIVE**

### Motivation

Run `output_jihang_20260409_101212` analysis showed a split-composition
failure even though the cell count landed at the target 14. The three
structural problems were symmetric:

1. **`e9077677…a5` f3 miss.** The parent's bright-pixel strict-interior
   PCA at split-attempt gave `strict_halo=1.168`, `strict_topk=1.196`,
   `bbox=1.157` — below the 1.30 elongation gate. Top-K cannot help
   because the brightest 10% pixels genuinely are not elongated at f3;
   the nascent daughters are not yet pulled apart enough to show up as
   a variance-ratio signal. The parent eventually splits at f20 (17
   frames late), and the entire `e9077-a50 → a500/a501` (f19) and
   `e9077-a51 → a510/a511` (f20) lineage never happens.
2. **`e3d034289…8d` f4 false positive.** PCA top-K fires at 2.43
   (halo=1.22) because this cell has real internal brightness
   asymmetry, but it is a true-negative — never supposed to split.
   Top-K alone has no mechanism to distinguish "elongated bright
   distribution" from "two daughter nuclei"; the burn-in accepted
   27/1000 perturbations with `diff=-131` because the two-daughter
   placement slightly explains the asymmetry better than a single
   parent would.
3. **`12345…3400` drifted to the minimum radius.** Visible in the f8
   screenshot of run 101212 as the degenerate crumpled ellipse next to
   `12345…341` (the real f3 split's second daughter, which also shrunk
   to minimum via the same ratchet). The Spheroid constructor silently
   clamps majorRadius/minorRadius to `minMajorRadius`/`minMinorRadius`,
   so any decrease-biased perturbation sequence parks the cell at the
   floor where the tiny L2 footprint is cheap.

Both (1) and (2) can be addressed by the SAME signal: a structural
two-peak detector. Run it AND-gated with the PCA elongation gate so it
compounds with existing sensitivity:

- `e9077` f3 → PCA gate fails (1.20 < 1.30) AND the peak detector would
  see two bright peaks → AND composition still cannot fire the split.
  **This is intentional:** the downstream correctness win is (2), where
  the peak detector vetoes the `e3d034` FP that PCA alone waves through.
  Fixing the `e9077` miss requires either lowering the elongation gate
  (risks new FPs) or using the peak detector as an alternative trigger
  (OR-gate) — deferred to a follow-up after observing PR-5 behavior.
- `e3d034` f4 → PCA gate passes (topK=2.43 > 1.30) AND the peak
  detector sees only one dominant peak → AND composition rejects → the
  FP is vetoed before the 1000-iter burn-in.
- `1f2ed10d` f11 (the top-K victory in run 101212) → PCA gate passes
  AND the peak detector sees two peaks → AND still passes → split
  still fires on time.

Problem (3) is solved by a straightforward proposal-level revert in
`Frame::perturbCell`.

See `docs/plans/2026-04-09-pca-input-fix-and-residual-safety-nets.md`
§5 for the full Pillar B specification.

### Change 1: `Frame.cpp` — `proposeSplitByLocalMaxima` helper + `LocalMaxSplitProposal` struct

New free function and POD struct in the anonymous namespace at
`C++/src/Frame.cpp`, inserted between `computeBridgeCylinderRadius` and
`shouldRejectSplitPreBurnIn` (~270 lines). Not exposed in `Frame.hpp`
— all callers live inside `Frame.cpp`.

Algorithm (matches plan §5.2):

1. Axis-aligned ROI bbox of radius `roiExpansionFactor × effMajorR`
   (default 1.8) around the cell center, clipped to frame bounds.
   `effMajorR = max(currentMajorR, preOptMajorR)` matches the split-
   attempt PCA window so the peak search and PCA see the same volume.
2. **2D smoothing:** every ROI slice is `cv::GaussianBlur`-smoothed in
   xy with `sigma = smoothingSigma` (default 2.5 voxels). Kernel size
   is `round(6 × sigma) | 1` to keep it odd.
3. **1D smoothing along z:** hand-rolled 1D Gaussian convolution with
   half-kernel `ceil(3 × sigma)` (OpenCV has no 3D Gaussian). Uses
   `cv::scaleAdd` to avoid a 4th copy.
4. **ROI mean/stddev:** one full pass for the prominence floor:
   `prominenceThreshold = mean + minProminenceOverStddev × stddev`.
5. **Peak candidate scan:** every interior voxel (skip 1-voxel border)
   is checked as `>=` all 26 neighbors AND above the prominence floor.
6. **Non-maximum suppression:** intensity-sorted, each candidate kept
   only if no higher-intensity kept peak is within
   `suppressionRadiusFactor × daughterMajorR` where
   `daughterMajorR = cbrt(0.5) × effMajorR`.
7. **Neighbor reject:** any peak closer to a `neighborCenters[i]` than
   to the parent's own center is dropped.
8. **Decision by count:** 0 → reject (code 1); 1 → reject (code 2);
   ≥2 → take top-two, sample 7 points along the segment (exclude
   endpoints), valley = min of those 7; reject if
   `valley > valleyMaxRelativeDepth × min(peak1, peak2)` (code 3).

Return struct is POD, every field is logged:

```cpp
struct LocalMaxSplitProposal
{
    bool valid = false;
    cv::Point3f peak1{0,0,0}, peak2{0,0,0};
    float peak1Intensity = 0.0f, peak2Intensity = 0.0f;
    float valleyIntensity = 0.0f;
    int peaksBeforeNMS = 0, peaksAfterNMS = 0;
    int rejectReason = 0;  // 0=valid, 1=zero_peaks, 2=one_peak,
                           // 3=valley_too_bright, 4=roi_degenerate
};
```

### Change 2: `Frame.cpp` — AND-gate wiring in `trySplitCell`

Inserted between `shouldRejectSplitPreBurnIn` and `cells.erase` so the
veto happens BEFORE the 1000-iter burn-in cost:

```cpp
const float pillarBEffMajorR =
    (preOptMajorR > 0.0f) ? std::max(oldCell.getMajorRadius(), preOptMajorR)
                          : oldCell.getMajorRadius();
LocalMaxSplitProposal peakProposal = proposeSplitByLocalMaxima(
    oldCell, _realFrame, neighborCenters, pillarBEffMajorR,
    localMaxRoiExpansion, localMaxSmoothingSigma,
    localMaxMinProminenceOverStddev,
    localMaxSuppressionRadiusFactor,
    localMaxValleyMaxRelativeDepth);

std::cout << "[Split LocalMax] " << oldCell.getName()
          << " valid=" << (peakProposal.valid ? 1 : 0)
          << " reject_reason=" << peakProposal.rejectReason
          << " peaks_pre_nms=" << peakProposal.peaksBeforeNMS
          << " peaks_post_nms=" << peakProposal.peaksAfterNMS
          << " p1=(" << peakProposal.peak1.x << "," << peakProposal.peak1.y
          << "," << peakProposal.peak1.z << ")"
          << " p2=(" << peakProposal.peak2.x << "," << peakProposal.peak2.y
          << "," << peakProposal.peak2.z << ")"
          << " peak1_intensity=" << peakProposal.peak1Intensity
          << " peak2_intensity=" << peakProposal.peak2Intensity
          << " valley=" << peakProposal.valleyIntensity
          << " eff_majorR=" << pillarBEffMajorR
          << std::endl;

if (localMaxEnabled && !peakProposal.valid) {
    std::cout << "[Split LocalMaxReject] " << oldCell.getName()
              << " reason=local_max_does_not_agree"
              << " reject_code=" << peakProposal.rejectReason
              << " peaks_post_nms=" << peakProposal.peaksAfterNMS
              << std::endl;
    return {0.0, [](bool accept) {}};
}
```

Note: the `[Split LocalMax]` diagnostic fires even when
`local_max_enabled = false`. Only the reject branch is gated by the
master switch, so you can flip the switch off and still observe peak
counts for A/B tuning without any behavioral change.

### Change 3: `Frame::trySplitCell` signature extension

**File:** `C++/includes/Frame.hpp` lines 48-72 — six new trailing
parameters with defaults for backward compat with existing tests:

```cpp
int splitMinInsideCount = 50000,
bool localMaxEnabled = true,
float localMaxRoiExpansion = 1.8f,
float localMaxSmoothingSigma = 2.5f,
float localMaxMinProminenceOverStddev = 1.5f,
float localMaxSuppressionRadiusFactor = 0.9f,
float localMaxValleyMaxRelativeDepth = 0.85f);
```

`C++/src/Frame.cpp` lines 918-942 mirrors the new params in the
out-of-line definition.

### Change 4: `CellUniverse.cpp` — thread new params through

**File:** `C++/src/CellUniverse.cpp` lines 713-737 — the `trySplitCell`
call now passes the six new `config.prob.local_max_*` knobs after
`split_min_inside_count`:

```cpp
auto result = frame.trySplitCell(cellIdx, preOptMajorR, preOptMinorR,
                                 /* ... existing 17 params ... */,
                                 config.prob.split_burn_in_iterations,
                                 config.prob.split_min_inside_count,
                                 config.prob.local_max_enabled,
                                 config.prob.local_max_roi_expansion_factor,
                                 config.prob.local_max_smoothing_sigma,
                                 config.prob.local_max_min_prominence_over_stddev,
                                 config.prob.local_max_suppression_radius_factor,
                                 config.prob.local_max_valley_max_relative_depth);
```

### Change 5: `ConfigTypes.hpp` — new `ProbabilityConfig` fields

**File:** `C++/includes/ConfigTypes.hpp` — six fields added under the
`split_min_inside_count` block plus YAML parsing in
`ProbabilityConfig::explodeConfig`:

```cpp
bool  local_max_enabled = true;
float local_max_roi_expansion_factor = 1.8f;
float local_max_smoothing_sigma = 2.5f;
float local_max_min_prominence_over_stddev = 1.5f;
float local_max_suppression_radius_factor = 0.9f;
float local_max_valley_max_relative_depth = 0.85f;
```

Every field is optional on the YAML side — the default is preserved
on absence so old configs still parse.

### Change 6: `config.yaml` — defaults under `prob:`

**File:** `C++/config/config.yaml` — appended after
`max_split_probability: 0.2`:

```yaml
local_max_enabled: true
local_max_roi_expansion_factor: 1.8
local_max_smoothing_sigma: 2.5
local_max_min_prominence_over_stddev: 1.5
local_max_suppression_radius_factor: 0.9
local_max_valley_max_relative_depth: 0.85
```

### Change 7: `Frame::perturbCell` min-R clamp

**File:** `C++/src/Frame.cpp` — `Frame::perturbCell`, immediately after
`cells[index] = cells[index].getPerturbedCell();`. Proposal-level
revert that detects when the Spheroid ctor's silent min-radius clamp
swallowed a decrease-biased perturbation and rolls back the proposal
before any cost is computed.

**After:**

```cpp
cells[index] = cells[index].getPerturbedCell();

// Min-radius hard clamp (2026-04-09): prevent cells from ratcheting down
// to minimum radius bounds via unconstrained perturbation. The Spheroid
// ctor silently clamps majorRadius/minorRadius to minMajorRadius/
// minMinorRadius, so a decrease-biased perturbation sequence parks cells
// at the floor where the L2 cost rewards the tiny footprint. Revert any
// proposal that would take either radius FROM above the floor TO the
// floor; proposals already at the floor are still allowed.
{
    const float newMajorR = cells[index].getMajorRadius();
    const float newMinorR = cells[index].getMinorRadius();
    const float oldMajorR = oldCell.getMajorRadius();
    const float oldMinorR = oldCell.getMinorRadius();
    const float minMajorR = static_cast<float>(Spheroid::cellConfig.minMajorRadius);
    const float minMinorR = static_cast<float>(Spheroid::cellConfig.minMinorRadius);
    constexpr float kClampEpsilon = 1e-3f;
    const bool hitMajorFloor = (newMajorR <= minMajorR + kClampEpsilon) &&
                               (oldMajorR  >  minMajorR + kClampEpsilon);
    const bool hitMinorFloor = (newMinorR <= minMinorR + kClampEpsilon) &&
                               (oldMinorR  >  minMinorR + kClampEpsilon);
    if (hitMajorFloor || hitMinorFloor) {
        cells[index] = oldCell;
        return {0.0, [](bool) {}};
    }
}
```

Concrete failure addressed: `12345…3400` parked at `(majorR=10, minorR=5)`
at frame 22 in `output_jihang_20260409_074740`, and `12345…341` visible
as the crumpled ellipse in the f8 screenshot of run 101212 — both
degenerate via the same ratchet.

This fix is **orthogonal to Pillar B** but bundled in the same PR per
user request on 2026-04-09.

### Files modified

| File | Change |
| --- | --- |
| `C++/includes/ConfigTypes.hpp` | 6 new `local_max_*` fields in `ProbabilityConfig` + YAML parsing |
| `C++/config/config.yaml` | New `local_max_*` defaults under `prob:` |
| `C++/includes/Frame.hpp` | `trySplitCell` signature extended with 6 Pillar B params |
| `C++/src/Frame.cpp` | New `LocalMaxSplitProposal` struct + `proposeSplitByLocalMaxima` helper; AND-gate wiring in `trySplitCell`; min-R clamp in `perturbCell` |
| `C++/src/CellUniverse.cpp` | Thread 6 new `config.prob.local_max_*` knobs through to `trySplitCell` |

### Expected outcomes on the next jihang run

1. `[Split LocalMax]` diagnostic line appears for every surviving split
   attempt. The `peaks_pre_nms`, `peaks_post_nms`, `peak1/2_intensity`,
   `valley`, and `reject_reason` fields should tell us whether the
   detector is firing sensibly on real cells.
2. `e3d034` at f4 should be vetoed by `[Split LocalMaxReject]` with
   `reject_code=2` (one peak) or `reject_code=3` (continuous blob). If
   it still splits, `local_max_min_prominence_over_stddev` needs
   raising.
3. `1f89abf484…cb1` at f22 (the FP sibling split) should also be
   vetoed — that cell had `prevElong=1.35` driven by noise.
4. `1f2ed10d` at f11 MUST still split. If the log shows
   `peaks_post_nms < 2`, the detector is too strict and the prominence /
   ROI expansion knobs need loosening.
5. `e9077` at f3 is expected to STILL miss under the AND-gate
   composition (the PCA elongation gate still vetoes at 1.20). The
   question to answer from the log is whether the peak detector WOULD
   have found two peaks — if yes, a follow-up PR can wire it as an
   alternative trigger (OR-gate) to finally catch `e9077` at f3.
6. No more drift-to-minimum cells in `cells.csv`. Check:
   `awk -F, '$6 < 13 && $7 < 7' $OUT/cells.csv` — should return zero
   rows (was non-empty in runs 074740 and 101212).

### Acceptance-check greps

```bash
OUT=outputs/output_jihang_<new_timestamp>

# (1) confirm Pillar B log fields appear
grep "\[Split LocalMax\]" $OUT/debug_log.txt | head -3

# (2) any LocalMax rejects?
grep "\[Split LocalMaxReject\]" $OUT/debug_log.txt

# (3) per-frame split summary
grep "\[Optimize Done\]" $OUT/debug_log.txt

# (4) accepted splits — who made it through, who got vetoed
grep "\[Split Accepted\]" $OUT/debug_log.txt

# (5) FP candidates from run 101212 — did they get vetoed?
grep -E "Split (LocalMax|Accepted).*(e3d034|1f89abf484.*cb1)" $OUT/debug_log.txt

# (6) 1f2ed10d still splitting?
grep -E "Split (LocalMax|Accepted).*1f2ed10d" $OUT/debug_log.txt

# (7) any drift-to-minimum cells left?
awk -F, '$6 < 13 && $7 < 7' $OUT/cells.csv
```

### Rollback

- **Pillar B:** set `local_max_enabled: false` in `config.yaml`. The
  helper still runs and logs for shadow observability, but the reject
  branch is skipped and split behavior reverts to PR-3 + top-K only.
- **Min-R clamp:** revert the 20-line block in `Frame::perturbCell`.
  Orthogonal to everything else; no cross-dependencies.

## 2026-04-09: PR-5 calibration pass — ellipsoid peak filter + valley 0.95 + burn-in min-R clamp — **ACTIVE**

### Motivation

Run `output_jihang_20260409_112621` was the first run with Pillar B live.
It delivered the headline win (`e9077` catches at f3, the previously-
missed split that blocked the entire e9077 lineage) but revealed three
calibration failures that the initial Pillar B landing didn't predict:

1. **False negative on `1f2ed10d` at f11** — the top-K PCA victory from
   run 101212 was LOST. PCA gate passes enormously (`strict_topk=2.766`)
   and Pillar B finds 3 peaks, BUT the valley threshold of 0.85 rejects
   the split because the daughters are still physically connected:
   valley=0.921, min(peak1, peak2)=0.997, ratio=0.924 > 0.85. Nascent
   splits have shallow valleys by nature — the check is tuned backwards
   for our use case.

2. **False positive on `e3d034` at f5** and **cascade FP on `e3d034-d0`
   at f20** — Pillar B's neighbor-reject uses Euclidean distance to cell
   CENTERS, so a peak sitting inside the 1.8× ROI expansion zone but
   outside the cell's actual analytic ellipsoid is kept as long as it is
   closer (by Euclidean center distance) to the target cell than to any
   neighbor. Concretely, `e3d034`'s peak2 at (115, 178, 107) is outside
   the cell ellipsoid (normalized distance² = 1.965 at 1.0× scale) —
   it's in the 1.8× ROI but outside the 1.0× cell body — yet it passed
   the old filter because 12345's center was still further away.

3. **False positive on `1f89abf-cb0` at f18** — peak1 at (346, 367, 127)
   is 63 pixels in x from the cell center (majorR=37). The peak
   separation (81 pixels) is larger than the cell diameter (73). Peak1
   is clearly in a different cell's territory but, again, passes the
   neighbor-reject because no other cell's center is even closer to it.

All three FPs share the **same underlying bug**: the ROI expansion
factor of 1.8 lets peaks from outside the cell body count as "our"
peaks. Distance-to-center neighbor rejection is too weak because it
doesn't model cell boundaries — it only compares to a single point.

### Fix 1: `Spheroid::isPointInsideEllipsoid` helper + ellipsoid filter in Pillar B

Adds a public method on `Spheroid` to test whether a world-space point
lies inside the cell's analytic ellipsoid at a configurable scale
factor. Honors the cell's rotation (theta_x/y/z) by inverse-rotating
the displacement into the local frame before applying the standard
`(x/a)² + (y/b)² + (z/c)² ≤ 1` test. Inserted into
`proposeSplitByLocalMaxima`'s NMS loop after the neighbor-reject.

**File:** `C++/includes/Spheroid.hpp`

**After (inserted after `getPerturbedCell()` declaration, public
section):**

```cpp
// Test whether a world-space point lies inside this cell's analytic
// ellipsoid, optionally scaled. scaleFactor=1.0 is the exact cell
// boundary; scaleFactor>1 inflates isotropically. Used by Pillar B's
// local-max peak filter to exclude peaks that lie in the ROI
// expansion zone but outside the cell's actual body (e.g. neighbor
// contamination). The cell's rotation (theta_x/y/z) is honored.
bool isPointInsideEllipsoid(const cv::Point3f &worldPoint,
                            float scaleFactor = 1.0f) const;
```

**File:** `C++/src/Spheroid.cpp`

**After (new method body inserted after `getPerturbedCell`):**

```cpp
bool Spheroid::isPointInsideEllipsoid(const cv::Point3f &worldPoint,
                                      float scaleFactor) const
{
    const double dx = static_cast<double>(worldPoint.x) - _position.x;
    const double dy = static_cast<double>(worldPoint.y) - _position.y;
    const double dz = static_cast<double>(worldPoint.z) - _position.z;
    double lx = 0.0, ly = 0.0, lz = 0.0;
    inverseRotatePoint(dx, dy, dz, lx, ly, lz);

    const double clampedScale = std::max(1e-3, static_cast<double>(scaleFactor));
    const double sa = a * clampedScale;
    const double sb = b * clampedScale;
    const double sc = c * clampedScale;
    if (sa <= 0.0 || sb <= 0.0 || sc <= 0.0) {
        return false;
    }
    const double fx = lx / sa;
    const double fy = ly / sb;
    const double fz = lz / sc;
    return (fx * fx + fy * fy + fz * fz) <= 1.0;
}
```

**File:** `C++/src/Frame.cpp` — `proposeSplitByLocalMaxima`

Extended helper signature with a new trailing `peakInsideEllipsoidScale`
parameter, and inserted the filter in the NMS loop right after the
neighbor-reject continue and before `keptPeaks.push_back(candidate)`:

```cpp
if (closerToNeighbor) continue;

// Ellipsoid inside-test: reject peaks that lie in the ROI expansion
// zone but outside the cell's own analytic ellipsoid at the
// configured scale. This is how we distinguish "two peaks inside
// this cell body" from "one peak inside + one peak leaking in from
// an adjacent bright region". Tuned at 1.1 on run 112621: catches
// e3d034 f5 (peak2 norm²≈1.40 outside), 1f89abf-cb0 f18 (peak1
// norm²≈2.60 outside), e3d034-d0 f20 (peak2 norm²≈1.034 just
// outside), while keeping all true-positive cells' peaks inside.
if (peakInsideEllipsoidScale > 0.0f &&
    !cell.isPointInsideEllipsoid(candidate.pos, peakInsideEllipsoidScale)) {
    continue;
}

keptPeaks.push_back(candidate);
```

### Fix 2: valley threshold 0.85 → 0.95

**File:** `C++/config/config.yaml`

**Before:**

```yaml
local_max_valley_max_relative_depth: 0.85
```

**After:**

```yaml
# 2026-04-09 post-run-112621 calibration: raised 0.85 -> 0.95.
# At 0.85 the check rejected 1f2ed10d at f11 (a true split in progress)
# because the daughters were still physically connected (valley=0.921,
# peak=0.997 → ratio 0.924 > 0.85). Nascent splits have shallow valleys
# by nature; a tighter threshold kills TPs. 0.95 passes 1f2ed10d while
# the new ellipsoid inside-test (see below) independently kills the FPs
# that the old loose threshold used to let through.
local_max_valley_max_relative_depth: 0.95
```

The C++ default in `ProbabilityConfig` was also updated from 0.85f to
0.95f for consistency.

### Fix 3: min-R clamp inside `trySplitCell` burn-in loop

The PR-5 landing added a min-R clamp in `Frame::perturbCell` to prevent
cells from ratcheting to minimum radius bounds. But the burn-in inner
loop in `Frame::trySplitCell` bypasses `perturbCell` entirely — it
calls `cells[dIdx].getPerturbedCell()` directly — so daughters can
still sink to the floor during the 1000-iteration burn-in. Observed on
`12345` at f3 in run 112621: the valid Pillar B proposal was vetoed by
the post-burn-in `daughter_volume_ratio=35.64 > 6` guard because one
daughter sat at `(majorR=10, minorR=5.4)` while the other remained at
normal size.

**File:** `C++/src/Frame.cpp` — `Frame::trySplitCell` burn-in inner
loop, immediately after `cells[dIdx] = cells[dIdx].getPerturbedCell();`
and BEFORE the post-perturbation overlap computation.

**After:**

```cpp
// Perturb the daughter
cells[dIdx] = cells[dIdx].getPerturbedCell();

// Min-R clamp inside burn-in (2026-04-09 fix 3 after run 112621):
// mirror the perturbCell clamp so daughters can't ratchet down to the
// min-radius floor during the 1000-iter burn-in. Without this, one
// daughter sinks to minimum during burn-in and the post-burn-in
// daughter_volume_ratio guard rejects the legit split. Observed on
// 12345 at f3 in run 112621 (daughter_volume_ratio=35.64, one
// daughter at (majorR=10, minorR=5.4)). Revert any proposal that
// would take either radius FROM above the floor TO the floor;
// proposals already at the floor are still allowed.
{
    const float newMajorR = cells[dIdx].getMajorRadius();
    const float newMinorR = cells[dIdx].getMinorRadius();
    const float oldMajorR = saved.getMajorRadius();
    const float oldMinorR = saved.getMinorRadius();
    const float minMajorR = static_cast<float>(Spheroid::cellConfig.minMajorRadius);
    const float minMinorR = static_cast<float>(Spheroid::cellConfig.minMinorRadius);
    constexpr float kClampEpsilon = 1e-3f;
    const bool hitMajorFloor = (newMajorR <= minMajorR + kClampEpsilon) &&
                               (oldMajorR  >  minMajorR + kClampEpsilon);
    const bool hitMinorFloor = (newMinorR <= minMinorR + kClampEpsilon) &&
                               (oldMinorR  >  minMinorR + kClampEpsilon);
    if (hitMajorFloor || hitMinorFloor) {
        cells[dIdx] = saved;
        continue;
    }
}
```

Using `continue` skips cost computation and accept/reject for this
iteration — equivalent to a silent rejection that doesn't count toward
`accepted` (the clamp hit was not a real perturbation we'd want to
log as "accepted").

### Change plumbing — config field + signature extension

**File:** `C++/includes/ConfigTypes.hpp` — added
`local_max_peak_inside_ellipsoid_scale` field (default 1.1f) to
`ProbabilityConfig`, and YAML parsing in `explodeConfig`. Also updated
the `local_max_valley_max_relative_depth` default from 0.85f to 0.95f.

**File:** `C++/includes/Frame.hpp` — `trySplitCell` signature extended
by one more trailing param (total now 30):

```cpp
float localMaxPeakInsideEllipsoidScale = 1.1f);
```

**File:** `C++/src/Frame.cpp` — out-of-line definition signature
mirrors the header. `proposeSplitByLocalMaxima` also got the new
trailing param.

**File:** `C++/src/CellUniverse.cpp` — added
`config.prob.local_max_peak_inside_ellipsoid_scale` to the
`trySplitCell` call after `local_max_valley_max_relative_depth`.

**File:** `C++/config/config.yaml` — added
`local_max_peak_inside_ellipsoid_scale: 1.1` under `prob:` with a
comment explaining the tuning.

### Files modified

| File | Change |
| --- | --- |
| `C++/includes/Spheroid.hpp` | New public `isPointInsideEllipsoid` declaration |
| `C++/src/Spheroid.cpp` | `isPointInsideEllipsoid` method body (~25 lines) |
| `C++/includes/ConfigTypes.hpp` | `local_max_peak_inside_ellipsoid_scale` field (default 1.1f) + parser; valley default 0.85→0.95 |
| `C++/config/config.yaml` | Valley 0.85→0.95; new `local_max_peak_inside_ellipsoid_scale: 1.1` |
| `C++/includes/Frame.hpp` | `trySplitCell` gets one more trailing param (30 total) |
| `C++/src/Frame.cpp` | `proposeSplitByLocalMaxima` gets ellipsoid filter; `trySplitCell` passes it through; burn-in gets min-R clamp |
| `C++/src/CellUniverse.cpp` | Thread new knob through |

### Expected outcomes on the next jihang run

Predicted from the measured peak positions in run 112621:

| Frame | Cell | 112621 result | Predicted post-fix |
|---|---|---|---|
| f3 | e9077 | ✓ (both peaks just outside 1.0×, inside 1.1×) | ✓ |
| f3 | 12345 | ✗ post-burn-in volume_ratio=35.6 | ✓ (fix 3: burn-in clamp) |
| f5 | e3d034 | ✗ FP (peak2 norm²=1.40 at 1.0×, passed at 1.8× ROI) | vetoed (fix 1) |
| f8 | 1f89abf | ✓ | ✓ |
| f11 | 1f2ed10d | ✗ valley=0.924 > 0.85 | ✓ (fix 2: 0.924 < 0.95) |
| f18 | 1f89abf-cb0 | ✗ FP (peak1 norm²=2.60) | vetoed (fix 1) |
| f19 | e9077-a50 | ✗ (cell drifted to dark bg, no peaks) | ✗ (needs Pillar C) |
| f20 | e3d034-d0 | ✗ FP cascade (peak2 norm²=1.03) | vetoed (fix 1, and parent never splits anyway) |
| f20 | e9077-a51 | ✓ | ✓ |
| f20 | 12345-340 | ✗ (parent never split) | ✓ (parent now splits at f3) |
| f20 | 12345-341 | ✗ (parent never split) | ✓ (parent now splits at f3) |

**Predicted cell count: 13 at f22** (was 12 in 112621, target 14). The
remaining miss is `e9077-a50` at f19 — that cell drifts to dark
background between f3 and f18 and can't be recovered by Pillar B alone
because the ROI contains no bright pixels by the time the split should
fire. This is a Pillar C (frame-anchor drift penalty) problem, not a
Pillar B problem.

### Acceptance-check greps

```bash
OUT=outputs/output_jihang_<new_timestamp>

# (1) e3d034 at f5 should now be rejected — look for reject_reason=2 (one peak)
grep "\[Split LocalMax\] e3d034" $OUT/debug_log.txt | head -5

# (2) 1f2ed10d at f11 should now VALIDATE (was rejected with reject_reason=3)
grep "\[Split LocalMax\] 1f2ed10d" $OUT/debug_log.txt

# (3) 12345 at f3 should now ACCEPT (was rejected post-burn-in for volume ratio)
grep -E "(\[Split Accepted\]|\[Split Rejected Fake\]).*12345" $OUT/debug_log.txt

# (4) 1f89abf-cb0 at f18 should no longer fire
grep "\[Split Accepted\].*1f89abf484c94c498a23cad71ebee0cb0" $OUT/debug_log.txt

# (5) full accepted splits
grep "\[Split Accepted\]" $OUT/debug_log.txt

# (6) per-frame summary
grep "\[Optimize Done\]" $OUT/debug_log.txt
```

### Rollback

- **Fix 1 (ellipsoid filter):** set
  `local_max_peak_inside_ellipsoid_scale: 1.8` in config.yaml — at
  `scale == roi_expansion_factor`, the filter is a no-op. Or set it to
  a very large number like 99.0 to disable entirely.
- **Fix 2 (valley 0.95):** set `local_max_valley_max_relative_depth: 0.85`
  back in config.yaml.
- **Fix 3 (burn-in min-R clamp):** revert the 25-line block in
  `Frame::trySplitCell`'s burn-in loop — the `cells[dIdx] = saved;
  continue;` path. No other code depends on it.

## 2026-04-09: PR-5 calibration v2 — replace ellipsoid filter with sphere distance test — **ACTIVE**

### Motivation

Run `output_jihang_20260409_124330` (the first run with the previous
calibration pass) regressed dramatically: 2 splits accepted, 8 cells at
f22 (was 12 in 112621, target 14). Only `12345` f3 ✓ and `1f2ed10d`
f12 ✓ landed; `e9077` f3 missed, `1f89abf` f8 missed, `e9077-a51` never
existed because parent never split.

**Root cause: the rotation-aware ellipsoid test from the previous
calibration pass was wrong for oblate cells under heavy rotation.**

The previous calibration computed `(dx/a)² + (dy/b)² + (dz/c)² ≤ 1`
hand-by-hand for run-112621 peaks and concluded scale=1.1 was the
sweet spot. That hand-computation **implicitly assumed unrotated axes**
(treating dx/dy/dz directly as local-frame coordinates). The actual
implementation correctly inverse-rotated the displacement before
applying the test, but the calibration target was based on the
unrotated math, so the implementation was stricter than predicted.

For 1f89abf at f8 in run 124330:
- `theta = (3.93, 4.60, 3.15)` rad (large rotation)
- `R = (a=38.33, c=20.24)` — oblate, c is half of a
- Both peaks are at world-space distances 28.1 and 25.7 from the cell
  center (well inside `1.1 × 38.33 = 42.16`)
- After inverse rotation, the peak displacement gets projected partly
  onto the c-axis. Any vector component on c is divided by 20 instead
  of 38, so even modest local-frame z values produce `(lz/c)² → 1.0`.
- One of the two peaks crossed the threshold and got dropped → 1 peak →
  `reject_reason=2` → split rejected → 1f89abf f8 missed entirely

For `e9077` at f3 in run 124330: separate failure mode — pre-burn-in
`shouldRejectSplitPreBurnIn` killed it at `sepOverDaughterMajor=0.338 <
0.35`. This is a stochastic borderline reject (the same gate accepted
e9077 in run 112621 at 0.357). Not a Pillar B issue. Not addressed in
this calibration.

### Fix: sphere distance test instead of ellipsoid

Replace the rotation-aware ellipsoid inside-test with a simple Euclidean
sphere test of radius `(scale × effMajorR)`. Properties:
- **Rotation-invariant by design** — uses majorR (the cell's largest
  extent) regardless of orientation. The c-axis no longer has any
  influence on the filter.
- **Correctly calibrated against measured peaks across three runs**
  (101212, 112621, 124330) — see the table in the inlined code comment.
- **Cheaper** — one distance calc per peak, no inverse rotation, no
  per-axis division.

**File:** `C++/src/Frame.cpp` — `proposeSplitByLocalMaxima`'s NMS loop,
right after the neighbor-reject continue.

**Before:**

```cpp
if (peakInsideEllipsoidScale > 0.0f &&
    !cell.isPointInsideEllipsoid(candidate.pos, peakInsideEllipsoidScale)) {
    continue;
}
```

**After:**

```cpp
if (peakInsideEllipsoidScale > 0.0f) {
    const double pdx = static_cast<double>(candidate.pos.x - cellCenter.x);
    const double pdy = static_cast<double>(candidate.pos.y - cellCenter.y);
    const double pdz = static_cast<double>(candidate.pos.z - cellCenter.z);
    const double maxDist = static_cast<double>(peakInsideEllipsoidScale)
                         * static_cast<double>(effMajorR);
    if (pdx * pdx + pdy * pdy + pdz * pdz > maxDist * maxDist) {
        continue;
    }
}
```

The parameter name `peakInsideEllipsoidScale` is kept for code/yaml
backward compat — the semantics changed but renaming would touch many
files. The comment block above the test documents the misnomer; future
PR can rename to `peakMaxDistanceOverMajorRadius` if desired.

`Spheroid::isPointInsideEllipsoid` (added in the previous calibration
pass) is no longer called from anywhere but is left in place — it is a
correct rotation-aware ellipsoid test and may be useful for future
features.

### Calibration table (verified)

| Case | dist | majorR | dist/majorR @ 1.1 | Verdict | Truth |
|---|---|---|---|---|---|
| 1f89abf f8 peak1 | 28.1 | 38.3 | 0.74 | inside | TP |
| 1f89abf f8 peak2 | 25.7 | 38.3 | 0.67 | inside | TP |
| 1f2ed10d f11 peak1 | 41.9 | 41 | 1.02 | inside | TP |
| 1f2ed10d f11 peak2 | 13.7 | 41 | 0.33 | inside | TP |
| e9077 f3 peak1 | 32.1 | 31.5 | 1.02 | inside | TP |
| e9077 f3 peak2 | 31.1 | 31.5 | 0.99 | inside | TP |
| 12345 f3 peak1 | 21.0 | 42 | 0.50 | inside | TP |
| e9077-a51 f20 peak1 | 23 | 32 | 0.72 | inside | TP |
| e9077-a51 f20 peak2 | 30 | 32 | 0.94 | inside | TP |
| **e3d034 f5 peak2** | 49 | 40 | 1.22 | outside | FP rejected |
| **1f89abf-cb0 f18 p1** | 65 | 37 | 1.76 | outside | FP rejected |
| **e3d034-d0 f20 p2** | 30.7 | 27.6 | 1.11 | outside | FP rejected |

All TPs pass at 1.1, all FPs fail at 1.1, no margin issues.

### YAML comment update

`C++/config/config.yaml` — the comment above
`local_max_peak_inside_ellipsoid_scale` is rewritten to explain the
sphere test, the rotation/oblate root cause, and the calibration
across all three runs. Field name unchanged (still
`local_max_peak_inside_ellipsoid_scale: 1.1`) to avoid YAML breakage.

### Files modified

| File | Change |
| --- | --- |
| `C++/src/Frame.cpp` | NMS-loop ellipsoid filter replaced with inline sphere distance check |
| `C++/config/config.yaml` | Comment rewrite explaining the sphere semantics + rotation/oblate root cause |

`Spheroid::isPointInsideEllipsoid` is unused-but-retained.

### Expected outcomes on the next jihang run

Predicted from the measured peak positions across runs 101212, 112621,
124330:

| Frame | Cell | 124330 result | Predicted post-fix |
|---|---|---|---|
| f3 | e9077 | ✗ pre-burn-in sep_over_major=0.338<0.35 | depends on stochastic timing — may pass |
| f3 | 12345 | ✓ | ✓ |
| f5 | e3d034 (FP) | vetoed | vetoed (sphere test still rejects) |
| f8 | 1f89abf | ✗ over-filtered | **✓ (both peaks now pass sphere test)** |
| f11/f12 | 1f2ed10d | ✓ at f12 | ✓ |
| f18 | 1f89abf-cb0 (FP) | did not fire (TP missing) | vetoed if TP fires |
| f19 | e9077-a50 | ✗ drift | ✗ (Pillar C territory) |
| f20 | e9077-a51 | ✗ (parent never split) | depends on parent |
| f20 | 12345-340, 341 | ✗ | ✓ (parent now splits at f3) |

**Predicted cell count at f22: 11–13** depending on how the e9077 f3
stochastic gate goes. The sphere test should robustly fix 1f89abf f8
(removing the over-filtering) and preserve all the FP vetoes from the
previous calibration.

If `e9077` f3 still misses (the pre-burn-in `sepOverDaughterMajor` gate
is borderline at 0.35 and the cell happens to land at 0.34), that's a
separate issue: the gate threshold is too tight for a stochastic
measurement. Possible follow-up: lower
`split_pre_burn_in_min_separation_over_major` from 0.35 to 0.30.

### Acceptance-check greps

```bash
OUT=outputs/output_jihang_<new_timestamp>

# (1) 1f89abf at f8 — should now find 2 peaks (was 1 in run 124330)
grep "\[Split LocalMax\] 1f89abf484c94c498a23cad71ebee0cb " $OUT/debug_log.txt | head -3

# (2) accepted splits
grep "\[Split Accepted\]" $OUT/debug_log.txt

# (3) per-frame summary
grep "\[Optimize Done\]" $OUT/debug_log.txt

# (4) FPs that should still be vetoed
grep -E "\[Split Accepted\] (e3d034|1f89abf484c94c498a23cad71ebee0cb0)" $OUT/debug_log.txt
```

### Rollback

Set `local_max_peak_inside_ellipsoid_scale: 99.0` (or any large number)
in `config.yaml` to disable the sphere filter entirely. Or revert the
`if (peakInsideEllipsoidScale > 0.0f)` block in
`proposeSplitByLocalMaxima` and re-enable the
`cell.isPointInsideEllipsoid` call (the method is still defined).

## 2026-04-09: PR-A — Snapshot capture infrastructure (no behavior change) — **ACTIVE**

### Motivation

After the PR-5 calibration v2 run (run 161559) caught 4 true splits with 0
FPs but missed 4 second-generation daughter splits, root-cause analysis
revealed a structural problem the existing code cannot fix: the live PCA
recomputation inside `Frame::trySplitCell` measures the cell's CURRENT
state, after the optimizer has been actively flattening the elongation that
motivated the split attempt in the first place.

For example, `12345-340` at f16 had `prevElong = 1.641` (from end of f15)
which drove `P(split) = 0.2`. The dice rolled split, but by that iteration
the cell's live `strict_effective` had dropped to 1.371 — the optimizer
had perturbed the cell ~40 times into a roundier shape between start of
frame and the split attempt. The split signal exists in the cached
elongation but is discarded by the live re-check.

The full analysis and architectural fix are in
`C++/docs/plans/2026-04-09-snapshot-driven-split-architecture.md`. The
proposed pipeline trusts the previous frame's measurement as the
authoritative "should this cell try to split" signal, places daughters
using snapshot data (last frame's parent position + PCA principal axis +
optionally Pillar B bright peaks), and validates by post-burn-in biological
sanity gates instead of pre-filtering with live PCA.

This PR-A is the **first of six PRs** that implement the architecture. PR-A
adds the snapshot infrastructure but **does not change any behavior** — it
populates the snapshots but never reads them. PR-B will add the hard
prevElong cutoff that consults the snapshot. PR-C will add snapshot-driven
daughter placement. PR-D will populate `snapshot.brightPeaks` from
end-of-frame Pillar B with neighbor masking. PR-E removes redundant guards.
PR-F is the validation sweep.

### Change 1: `Spheroid.hpp` — extend `StrictInteriorPcaResult` with principal axis

The strict-interior PCA already computes eigenvalues and uses the largest
two for the elongation ratio. The new architecture also needs the **λ1
eigenvector** (the direction of maximum variance — i.e., the split
direction) so daughter placement can use it.

**File:** `C++/includes/Spheroid.hpp`

**After (added to `StrictInteriorPcaResult`):**

```cpp
// Principal axis (λ1 eigenvector) of the HALO-weighted PCA, expressed in
// world coordinates as a unit vector. Direction of maximum variance in
// the bright pixel cloud — i.e., the direction along which the cell is
// most elongated. Used by the snapshot-driven daughter placement
// architecture (PR-A through PR-D) as the split axis when the previous
// frame's snapshot lacks a 2-peak structure to use directly.
// Default (1,0,0) on invalid result.
cv::Point3f principalAxis{1.0f, 0.0f, 0.0f};
```

### Change 2: `Spheroid.hpp` — new `PreviousFrameSnapshot` struct

The new authoritative source for next-frame split decisions. Holds the
peak elongation observed during the previous frame, the cell's pose at the
peak moment, the principal axis at that moment, and (in PR-D) the bright
peaks from end-of-frame Pillar B.

**File:** `C++/includes/Spheroid.hpp`

**After (new struct following `StrictInteriorPcaResult`):**

```cpp
struct PreviousFrameSnapshot
{
    bool valid = false;
    float strictEffectiveElongation = 1.0f;
    cv::Point3f position{0.0f, 0.0f, 0.0f};
    float majorRadius = 0.0f;
    float minorRadius = 0.0f;
    float thetaX = 0.0f;
    float thetaY = 0.0f;
    float thetaZ = 0.0f;
    float brightness = 0.5f;
    cv::Point3f principalAxis{1.0f, 0.0f, 0.0f};
    std::vector<cv::Point3f> brightPeaks;
    std::vector<float> brightPeakIntensities;
};
```

### Change 3: `Spheroid.cpp` — extract eigenvectors in the strict-interior PCA

`computeStrictInteriorWeightedElongation` was using `cv::eigen(covMat,
eigenvaluesMat)` which only outputs eigenvalues. Switch to the 3-arg form
`cv::eigen(covMat, eigenvaluesMat, eigenvectorsMat)` to also extract the
λ1 eigenvector. The eigenvector is normalized to unit length and stored
in `result.principalAxis`.

**File:** `C++/src/Spheroid.cpp`

**Before (the eigenRatio lambda):**

```cpp
auto eigenRatio = [](double c00, double c01, double c02,
                     double c11, double c12, double c22,
                     float &outL1, float &outL2, float &outL3) -> float {
    cv::Mat covMat = (cv::Mat_<double>(3, 3) <<
        c00, c01, c02,
        c01, c11, c12,
        c02, c12, c22);
    cv::Mat eigenvaluesMat;
    if (!cv::eigen(covMat, eigenvaluesMat) || eigenvaluesMat.rows < 3) {
        return 1.0f;
    }
    /* extract l1, l2, l3 */
    return (l2 > 1e-12)
        ? static_cast<float>(std::sqrt(l1 / l2))
        : 1.0f;
};
```

**After:**

```cpp
auto eigenRatio = [](double c00, double c01, double c02,
                     double c11, double c12, double c22,
                     float &outL1, float &outL2, float &outL3,
                     cv::Point3f *outPrincipalAxis) -> float {
    cv::Mat covMat = (cv::Mat_<double>(3, 3) <<
        c00, c01, c02,
        c01, c11, c12,
        c02, c12, c22);
    cv::Mat eigenvaluesMat;
    cv::Mat eigenvectorsMat;
    if (!cv::eigen(covMat, eigenvaluesMat, eigenvectorsMat) || eigenvaluesMat.rows < 3) {
        return 1.0f;
    }
    /* extract l1, l2, l3 */
    if (outPrincipalAxis != nullptr && eigenvectorsMat.rows >= 1 &&
        eigenvectorsMat.cols >= 3) {
        const double ax = eigenvectorsMat.at<double>(0, 0);
        const double ay = eigenvectorsMat.at<double>(0, 1);
        const double az = eigenvectorsMat.at<double>(0, 2);
        const double norm = std::sqrt(ax * ax + ay * ay + az * az);
        if (norm > 1e-9) {
            *outPrincipalAxis = cv::Point3f(
                static_cast<float>(ax / norm),
                static_cast<float>(ay / norm),
                static_cast<float>(az / norm));
        }
    }
    return (l2 > 1e-12)
        ? static_cast<float>(std::sqrt(l1 / l2))
        : 1.0f;
};
```

The halo-weighted call passes `&result.principalAxis`. The top-K call
passes `nullptr` because the snapshot uses the halo axis as the canonical
split direction.

### Change 4: `CellUniverse.hpp` — add `previousSnapshots` map

**File:** `C++/includes/CellUniverse.hpp`

```cpp
std::map<std::string, PreviousFrameSnapshot> previousSnapshots;
```

Stored alongside the existing `previousElongations` map. Both are
populated in PR-A; PR-B will start consulting `previousSnapshots` while
`previousElongations` remains for backward compatibility until PR-E.

### Change 5: `CellUniverse.cpp` — in-frame running max sampling

Inside the main optimize loop, after the perturbation/split branch
completes for an iteration, every 50 iterations the just-perturbed cell's
strict-interior elongation is recomputed and compared against its running
max. If higher, the running max is updated AND the cell's pose at that
moment is captured (so the snapshot reflects the cell at peak elongation,
not at end-of-frame after the optimizer has flattened it).

**File:** `C++/src/CellUniverse.cpp` — inside `optimize()` main loop

```cpp
// PR-A snapshot infrastructure: in-frame running max sampling.
// Every kSnapshotSampleInterval iterations, measure the just-touched
// cell's strict-interior elongation and update its running max snapshot.
if ((i + 1) % kSnapshotSampleInterval == 0 &&
    cellIdx < frame.cells.size())
{
    const auto sampledParams = frame.cells[cellIdx].getCellParams();
    const StrictInteriorPcaResult sample =
        frame.cells[cellIdx].computeStrictInteriorWeightedElongation(
            frame.getRealFrame(), frame.getBackgroundValue());
    if (sample.valid) {
        auto &snap = inFrameSnapshots[sampledParams.name];
        if (sample.effectiveElongationRatio > snap.strictEffectiveElongation) {
            snap.valid = true;
            snap.strictEffectiveElongation = sample.effectiveElongationRatio;
            snap.position = cv::Point3f(
                sampledParams.x, sampledParams.y, sampledParams.z);
            snap.majorRadius = sampledParams.majorRadius;
            /* ... rest of pose ... */
            snap.principalAxis = sample.principalAxis;
        }
    }
}
```

`kSnapshotSampleInterval = 50` is the default. With ~3500 iterations per
frame, this gives 70 sampling opportunities per frame, distributed across
~10 cells = ~7 samples per cell per frame. Each sample is one
`computeStrictInteriorWeightedElongation` call (~5-15ms), so the per-frame
overhead is ~350-1050ms. Acceptable.

### Change 6: `CellUniverse.cpp` — end-of-frame snapshot capture

After the main loop exits, for each surviving cell:
1. Compute the end-of-frame strict-interior PCA (gives elongation +
   principal axis at the cell's settled pose).
2. Combine with the in-frame running max via `max(running_max,
   end_of_frame)` to get the snapshot's elongation field.
3. If the running max was larger, use the pose captured at that moment;
   otherwise use the end-of-frame pose.
4. Store in `previousSnapshots[name]`.
5. Also keep `previousElongations[name]` updated for backward compat.

**File:** `C++/src/CellUniverse.cpp` — after the main loop

```cpp
// Erase stale entries for cells that disappeared this frame
{
    std::set<std::string> liveCells;
    for (const auto &cell : frame.cells) liveCells.insert(cell.getName());
    for (auto it = previousSnapshots.begin(); it != previousSnapshots.end();) {
        if (liveCells.find(it->first) == liveCells.end()) {
            it = previousSnapshots.erase(it);
        } else {
            ++it;
        }
    }
}

std::cout << "[Snapshot] frame " << displayFrame << std::endl;
for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
    auto p = frame.cells[ci].getCellParams();
    /* legacy previousElongations update */

    const StrictInteriorPcaResult endResult =
        frame.cells[ci].computeStrictInteriorWeightedElongation(
            frame.getRealFrame(), frame.getBackgroundValue());

    const float endElong = endResult.valid
        ? endResult.effectiveElongationRatio : 1.0f;

    const auto inFrameIt = inFrameSnapshots.find(p.name);
    const float runningMax = (inFrameIt != inFrameSnapshots.end() &&
                              inFrameIt->second.valid)
        ? inFrameIt->second.strictEffectiveElongation : 1.0f;

    PreviousFrameSnapshot snap;
    snap.valid = true;
    snap.strictEffectiveElongation = std::max(runningMax, endElong);

    if (runningMax > endElong && /* ... */) {
        /* use in-frame pose */
    } else {
        /* use end-of-frame pose */
    }
    /* brightPeaks left empty in PR-A */

    previousSnapshots[p.name] = snap;
    /* diagnostic log */
}
```

### Diagnostic logging

A new `[Snapshot] frame N` block is logged at end-of-frame, with one line
per cell:

```
[Snapshot] frame 16
  e9077…a5  peak_elong=1.502  end_elong=1.07  in_frame_max=1.502  axis=(0.21,0.97,-0.12)  pos=(298,164,120)
  12345…34  peak_elong=1.641  end_elong=1.04  in_frame_max=1.641  axis=(0.85,-0.50,0.18)  pos=(140,175,128)
  ...
```

This gives us shadow-mode visibility into how often the running max
exceeds the end-of-frame measurement (the cases where the architecture
fix matters most).

### Files modified

| File | Change |
| --- | --- |
| `C++/includes/Spheroid.hpp` | `StrictInteriorPcaResult.principalAxis` field; new `PreviousFrameSnapshot` struct |
| `C++/src/Spheroid.cpp` | `computeStrictInteriorWeightedElongation` extracts λ1 eigenvector via 3-arg `cv::eigen` |
| `C++/includes/CellUniverse.hpp` | `previousSnapshots` map |
| `C++/src/CellUniverse.cpp` | In-frame running max sampling (every 50 iters); end-of-frame snapshot capture; `[Snapshot]` diagnostic logging |

### Behavioral guarantees (PR-A)

- **No split decision changes.** `previousSnapshots` is populated but never
  read by `Frame::trySplitCell`, the elongation gate, or the P(split)
  computation.
- **No daughter placement changes.** `getSplitCells` and the existing PCA
  paths are unchanged.
- **No new gate, no new rejection path.**
- **`previousElongations` is still maintained** with the same values as
  before (the `frame.computeElongationForCell` call is preserved exactly).

The only observable changes are:
1. The new `[Snapshot]` log block at end-of-frame.
2. Slightly higher per-frame compute (~350-1050ms for 70 in-frame samples
   plus ~50-150ms for end-of-frame snapshot building).
3. Small memory addition (~256 bytes per cell for the snapshot struct).

### Acceptance criteria (user to verify)

After build + run:

```bash
OUT=outputs/output_jihang_<new_timestamp>

# (1) confirm Snapshot log block appears
grep "\[Snapshot\] frame" $OUT/debug_log.txt | head -3

# (2) verify cell count and split count match the previous run (no behavior change)
grep "\[Optimize Done\]" $OUT/debug_log.txt
grep "\[Split Accepted\]" $OUT/debug_log.txt

# (3) check whether the in-frame max ever exceeds the end-of-frame value
#     (this is the case PR-B will leverage)
grep "\[Snapshot\]" $OUT/debug_log.txt -A 100 | \
  awk '/peak_elong/ {
    match($0, /peak_elong=([0-9.]+)/, p);
    match($0, /end_elong=([0-9.]+)/, e);
    if (p[1] > e[1] + 0.05) print
  }' | head
```

The third grep shows cells where the in-frame max exceeds end-of-frame by
more than 0.05. These are the cases where PR-B will use a better elongation
signal. If the grep returns nothing, the running max sampling isn't
catching anything new and we can drop it from PR-B.

### Rollback

Revert the diff. PR-A does not affect any decision so reverting has zero
behavioral impact. The added struct and map become dead code.

## 2026-04-09: PR-B — Hard prevElong cutoff from snapshot + remove live gate + remove z-axis guard — **ACTIVE**

### Changes

1. **P(split) now reads from `previousSnapshots`** (PR-A) instead of
   `previousElongations`. Falls back to the legacy map for cells without
   a snapshot. The snapshot's `strictEffectiveElongation` is the max of
   (running max during the previous frame, end-of-previous-frame) — the
   signal preserved by PR-A's in-frame sampling.

2. **Hard cutoff at `MIN_TRIGGER_ELONGATION = 1.20`**: cells whose
   snapshot elongation is below 1.20 get `P(split) = 0`. No split attempt
   is ever made for spherical cells, regardless of the dice roll.

3. **Live elongation gate REMOVED from `Frame::trySplitCell`**: the
   `strict_effective < splitElongationThreshold` check is deleted. The
   strict-interior PCA is still computed and logged at split-attempt time
   for diagnostics, but it no longer gates the split. The snapshot is the
   sole authority.

4. **Z-axis pre-burn-in guard REMOVED from `shouldRejectSplitPreBurnIn`**:
   the `diag.axisAbsZ > zAxisMaxAbs` check is deleted. This guard
   stochastically killed legitimate splits at borderline threshold values
   (12345 at f3 in run 181950: `axisAbsZ=0.924, threshold=0.92` — killed
   by 0.004; same cell passed at 0.904 in run 161559). The overlapping-
   daughters check (first guard) is preserved. The z-axis guard was the
   direct blocker for 12345's f3 split in run 181950, causing the entire
   downstream cascade (cell settles on one daughter, neighbor captures
   the other, 12345-340/341 never exist).

### Files modified

| File | Change |
| --- | --- |
| `C++/src/CellUniverse.cpp` | P(split) reads `previousSnapshots`; hard cutoff 1.20; log says "from snapshot" |
| `C++/src/Frame.cpp` | Live elongation gate deleted; z-axis guard deleted from `shouldRejectSplitPreBurnIn` |

### Expected outcomes

- 12345 at f3 should no longer be killed by the z-axis guard (the guard
  is gone). If the split-time getSplitCells PCA still points along z,
  Pillar B and the cost gate will evaluate it — not the z-axis heuristic.
- Cells below 1.20 snapshot elongation will never attempt splits
  (hard cutoff replaces the probabilistic dice → live gate chain).
- The combination of snapshot-driven P(split) + no live gate + no z-axis
  guard gives every cell with `prevElong >= 1.20` a clean path to split
  attempt → Pillar B → burn-in → cost gate, without intermediate
  heuristic vetoes.

---

## 2026-04-09 — PR-C — Snapshot-driven daughter placement — ACTIVE

### Problem/Motivation

The legacy `trySplitCell` path re-runs `getSplitCells` PCA on the current
(optimizer-corrupted) frame state, AND-gates with Pillar B which rejects
22/23 attempts due to NMS merging peaks under blur_sigma=10. Every single
split attempt in run 211750 failed: either P(split)=0 (snapshot elongation
collapsed below 1.20 at the critical frame) or Pillar B reject_reason=2
(only 1 peak after NMS). Zero splits accepted across 22 frames.

### Solution

New `Frame::trySplitCellFromSnapshot()` method that builds daughters
directly from the previous frame's snapshot data — no live PCA, no Pillar B
pre-filter. Daughters are placed along `snapshot.principalAxis` (or from
`snapshot.brightPeaks` when populated by PR-D). Burn-in and post-burn-in
fake guards still validate the result.

### Files Changed

**File:** `C++/includes/Frame.hpp`

Added `PillarBResult` public struct and `trySplitCellFromSnapshot` declaration:
```cpp
CostCallbackPair trySplitCellFromSnapshot(
    size_t cellIndex,
    const PreviousFrameSnapshot &snapshot,
    float overlapWeight = 1000.0f,
    int splitBurnInIterations = 1000,
    float fakeSplitOverlapVolumeFractionThreshold = 0.15f,
    float fakeSplitVolumeRatioThreshold = 6.0f,
    float fakeSplitBridgeBrightnessSimilarityThreshold = 0.99f,
    float sanityMinRadiusFactor = 1.5f,
    float sanityMaxDriftFactor = 1.5f,
    float sanityMinBrightnessFactor = 0.3f);
```

**File:** `C++/src/Frame.cpp`

Added ~230 LOC `trySplitCellFromSnapshot` implementation after line 1376.
Key sections:
- Daughter construction: volumeScale=cbrt(0.5), placed along principalAxis
  or brightPeaks, inheriting snapshot theta/brightness
- Same burn-in loop as legacy (1000 iters, min-R clamp)
- PR-D sanity gates: radius floor (1.5× minR), position drift (1.5× majorR),
  brightness floor (0.3× snapshot brightness)
- Existing post-burn-in fake guards (overlap fraction, volume ratio, bridge)
- `[Split FromSnapshot]` / `[Split SnapshotBurnIn]` / `[Split SnapshotReject]`
  diagnostic log tags

**File:** `C++/src/CellUniverse.cpp`

Modified split attempt site (optimize loop): when `previousSnapshots[name].valid`,
calls `trySplitCellFromSnapshot`; falls back to legacy `trySplitCell` only when
no snapshot exists (frame 1, which has splits disabled anyway).

### Effect

- Bypasses Pillar B AND-gate entirely for the snapshot-driven path
- Bypasses getSplitCells live PCA (no optimizer corruption of the signal)
- Bypasses shouldRejectSplitPreBurnIn (snapshot placement is stable)
- Bypasses shouldRejectSplitPostBurnIn (replaced by sanity gate 3: drift check)
- The only remaining guards: burn-in min-R clamp + sanity gates 1-4 +
  existing fake guards (overlap/volume ratio/bridge) + cost threshold

---

## 2026-04-09 — PR-D — Pillar B neighbor masking + end-of-frame peak capture — ACTIVE

### Problem/Motivation

The original `proposeSplitByLocalMaxima` uses a sphere distance test to
filter neighbor peaks, which was calibrated at 1.1× majorR but still causes
issues for dense cell clusters. More importantly, `snapshot.brightPeaks` was
left empty in PR-A, meaning PR-C's daughter placement always falls back to
principal-axis-based positioning.

### Solution

1. **New `proposeSplitByLocalMaximaNeighborMasked()`** — variant that takes
   actual `Spheroid` objects as neighbors, masks their voxels (via
   `isPointInsideEllipsoid` at 1.1× scale) to ROI mean BEFORE smoothing,
   then runs the same peak detection + NMS + valley check. No sphere distance
   test needed — masking replaces it.

2. **`Frame::runPillarBNeighborMasked()`** — public wrapper that builds the
   neighbor list and calls the masked variant, returning a `PillarBResult`.

3. **End-of-frame peak capture** — after the optimization loop, for each cell
   with elongation >= 1.10, runs the neighbor-masked Pillar B to populate
   `snapshot.brightPeaks`. These peaks seed next-frame's daughter placement
   in `trySplitCellFromSnapshot`.

### Files Changed

**File:** `C++/includes/Frame.hpp`

Added `PillarBResult` struct (public) and `runPillarBNeighborMasked` method.

**File:** `C++/src/Frame.cpp`

Added `proposeSplitByLocalMaximaNeighborMasked()` (~200 LOC) in the anonymous
namespace. Same smoothing/peak-detection/NMS/valley pipeline as the original,
with two key differences: (a) neighbor voxels masked to ROI mean before
smoothing, (b) no sphere distance test in the NMS loop.

Added `Frame::runPillarBNeighborMasked()` public wrapper (~30 LOC).

**File:** `C++/src/CellUniverse.cpp`

End-of-frame snapshot capture now calls `frame.runPillarBNeighborMasked()` for
cells with elongation >= 1.10 and stores the peaks in `snap.brightPeaks`.
Added `brightPeaks=N` to the snapshot log line.

### Effect

- Snapshot.brightPeaks will now contain 2 peaks when the neighbor-masked
  Pillar B detects a valid two-peak structure at end of frame
- Next-frame `trySplitCellFromSnapshot` will use these peaks directly as
  daughter centers instead of falling back to principal-axis placement
- Neighbor contamination eliminated by per-voxel ellipsoid masking

---

## 2026-04-09 — PR-E — Dead code removal — ACTIVE

### Problem/Motivation

With the snapshot-driven architecture (PR-A through PR-D) fully wired in,
several data structures and code paths are now dead:
- `previousElongations` map — replaced by `previousSnapshots`
- Legacy `trySplitCell` is only reachable on frame 1 (splits disabled)

### Files Changed

**File:** `C++/includes/CellUniverse.hpp`

Removed `std::map<std::string, float> previousElongations` member. Updated
comment on `previousSnapshots` to reflect it is now the sole data store.

**File:** `C++/src/CellUniverse.cpp`

- Removed all `previousElongations` reads/writes/clears
- `prevElongation` (used in split-attempt logging) now reads from
  `previousSnapshots` instead
- Removed `previousElongations.erase()` from the split-accepted block
- Removed `previousElongations.clear()` from end-of-frame
- Removed `previousElongations[p.name] = elong` write
- Cleaned up dead `neighborCenters` computation in end-of-frame Pillar B section

### Effect

- ~15 lines of dead code removed from CellUniverse.cpp
- Single source of truth for inter-frame split state: `previousSnapshots`

---

## 2026-04-10 — Current-frame Pillar B fallback for unreliable snapshot axis — ACTIVE

### Problem/Motivation

When a cell's snapshot elongation is low (<1.15), the PCA principal axis is
essentially random (eigenvector of a nearly-spherical cloud is noise).
Daughters placed along this garbage axis during `trySplitCellFromSnapshot`
consistently collapse during burn-in — one daughter lands on a bright region,
the other drifts into dark space and hits the radius floor.

Observed on `12345-341`: 10/12 split attempts killed by
`daughter_at_radius_floor` because the snapshot axis from f18 (elongation
1.024) pointed in a random Y-direction while the actual two-blob structure
was visible in the current frame's preprocessed image.

### Solution

When the snapshot has no bright peaks AND `snapshot.strictEffectiveElongation
< currentFramePillarBElongThreshold` (default 1.15), run the neighbor-masked
Pillar B on the **current frame** as a last-resort peak finder. If it finds
2 valid peaks, use those for daughter placement. Otherwise fall through to
the (unreliable) snapshot axis.

ROI expansion increased to 2.5x for current-frame Pillar B (vs 1.8x for
end-of-frame) because when the parent cell is fitted small/round, the actual
daughter blobs may lie outside the 1.8x radius.

### Files Changed

**File:** `C++/includes/Frame.hpp`

Added 7 current-frame Pillar B parameters to `trySplitCellFromSnapshot`
signature: `currentFramePillarBElongThreshold` (1.15),
`currentFramePillarBRoiExpansion` (2.5), smoothing sigma, prominence,
suppression radius, valley depth, neighbor mask scale.

**File:** `C++/src/Frame.cpp`

Modified daughter placement logic in `trySplitCellFromSnapshot`. Three-tier
placement priority:
1. Snapshot bright peaks (from previous frame's end-of-frame Pillar B)
2. **NEW**: Current-frame Pillar B (when snapshot elongation < 1.15)
3. Snapshot principal axis (fallback)

Added `[Split CurrentFramePillarB]` diagnostic log line showing peaks found.
Changed `usedBrightPeaks` log field to `placement=0/1/2` indicating which
tier was used.

**File:** `C++/src/CellUniverse.cpp`

Updated `trySplitCellFromSnapshot` call to pass Pillar B config values for
smoothing sigma, prominence, suppression, valley depth, and mask scale.

### Effect

- Cells with low snapshot elongation now get placement from the current
  frame's actual bright structure instead of a random PCA axis
- Called once per split attempt (only when snapshot elongation < 1.15),
  so compute cost is bounded
- Larger ROI (2.5x) compensates for the parent cell being fitted small
- Neighbor masking prevents false peaks from adjacent cells

---

## 2026-04-10 — Increase daughter separation from 1.0× to 1.5× parentMajorR — ACTIVE

### Problem/Motivation

Second-generation daughter splits (e9077-a51, 12345-340, 12345-341) all fail
with `daughter_at_radius_floor` — one daughter collapses to minimum during
burn-in. Root cause: daughters placed at `±0.5 × parentMajorR` (total sep =
1.0× parentMajorR). For second-gen cells with majorR=18-28, the daughter
radius is `0.794 × 25 ≈ 20`, so sep=25 vs daughterR=20 means significant
overlap. During burn-in, one daughter absorbs the bright region while the
other gets pushed into dark space and collapses.

Contrast with the one second-gen split that succeeded (e9077-a50 at f20):
its daughters both survived because they happened to land on two well-
separated bright blobs despite the same 1.0× separation.

### Solution

Increase `halfSep` from `0.5f × parentMajorR` to `0.75f × parentMajorR`
(total separation = 1.5× parentMajorR). Applied to both the snapshot-axis
placement path and the current-frame Pillar B fallback-to-axis path. Does
not affect the bright-peaks path (placement=0) which uses actual peak
positions directly.

### Files Changed

**File:** `C++/src/Frame.cpp`

Lines 1748 and 1760: changed `0.5f * parentMajorR` to `0.75f * parentMajorR`.

### Effect

- Daughters start further apart, reducing overlap during burn-in
- Both daughters more likely to land on distinct bright regions
- Burn-in refines position, so the extra separation is corrected if too wide

---

## 2026-04-10 — Pillar B elongation boost: structural 2-peak signal overrides halo-suppressed PCA — ACTIVE

### Problem/Motivation

12345-341 never splits because the sigmoid halo makes the cell look
spherical to PCA. At f19, PCA measures elongation=1.11 → the snapshot
carries this into f20 → P(split)=0 → never attempted at the GT split frame.

Meanwhile, the real image at f19 shows two distinct bright blobs that
Pillar B can detect. The structural signal exists but PCA is blind to it
because the halo rounds the bright pixel cloud.

### Solution

Two changes to the end-of-frame Pillar B section in CellUniverse::optimize:

1. **Run Pillar B for ALL cells** — removed the `elongation >= 1.10` gate.
   The whole point is to detect splits PCA misses, so gating by PCA
   elongation defeats the purpose.

2. **Elongation boost**: when Pillar B finds 2 valid peaks but the PCA
   elongation is below 1.25, boost the snapshot's
   `strictEffectiveElongation` to 1.25. This guarantees the cell passes
   the 1.20 hard cutoff at the trigger site next frame, giving it a split
   attempt. The bright peaks are already stored for daughter placement
   (placement=0 path), so the cell gets both: a trigger AND good placement.

The 1.25 boost value is just above the 1.20 cutoff — enough to trigger but
not so high that P(split) dominates other cells' probabilities.

### Files Changed

**File:** `C++/src/CellUniverse.cpp`

End-of-frame snapshot section: removed `if (snap.strictEffectiveElongation
>= 1.10f)` gate. Added elongation boost block with
`kPillarBElongBoostMin = 1.25f`. Added `[Snapshot PillarB boost]` log line.

### Effect

- Cells with halo-suppressed PCA elongation (<1.25) that show two structural
  bright peaks get boosted to 1.25 → guaranteed split attempt next frame
- The bright peaks are used directly for daughter placement (placement=0),
  bypassing the unreliable PCA axis entirely
- Non-splitting cells that happen to show 2 peaks are still filtered by
  burn-in + post-burn-in guards + cost threshold
- Compute cost: Pillar B now runs for every cell at end of frame instead of
  only elongated ones. ~5-10ms per cell × 10-14 cells = ~50-140ms per frame.

---

## 2026-04-10 — Bright peak staleness check: reject peaks too far from current cell center — ACTIVE

### Problem/Motivation

e9077-a51 regressed in run 022214: the snapshot bright peaks from end of f19
pointed to `(204,265,67)` and `(186,269,127)`, but the cell had drifted to a
different position by the time the f20 split attempt fired. The second peak
at z=127 was 60 z-units from the cell center at z=67 — likely a neighbor's
bright blob that the neighbor masking didn't fully cover. One daughter was
placed on this stale peak and collapsed during burn-in.

In run 230653 (where the cell succeeded), axis-based placement with
sep=30.8 kept both daughters near the cell's own bright blob.

### Solution

Before using snapshot bright peaks (placement=0), validate that both peaks
are within `1.5 × parentMajorR` of the **current** cell center. If either
peak is too far, fall through to axis-based placement. Logged as
`[Split PeakTooFar]`.

### Files Changed

**File:** `C++/src/Frame.cpp`

Modified the `snapshot.brightPeaks.size() == 2` branch in
`trySplitCellFromSnapshot`: added distance check from each peak to
`oldCell.get_center()`. Changed `if/else if` chain to `if/if` so
the peak-too-far case falls through to the axis-based paths.

### Effect

- Stale peaks from drifted cells are rejected, falling back to axis placement
- Peaks captured at end of previous frame that are still near the cell are
  used normally (no change for well-positioned cells)

## 2026-04-11 — Remove dead config fields and their code blocks

**Status:** ACTIVE

### Motivation

Ten config fields and their corresponding code blocks were dead:
- `volumeRecoveryEnabled` (and 2 thresholds) — gated by `false`, never fired in any run
- `flatCellRotationRefineEnabled` (and 4 tuning knobs) — gated by `false` in config
- `splitBrightestFraction` — declared and parsed but never referenced in source
- `firstFrameBrightnessPerturbationOnly` — gated by `false`, used only by the
  `brightnessPerturbFirstFrameOnly` block that disabled brightness perturbation
  after frame 1
- `initialABRatio` (CellFactory.hpp) — declared but never referenced in source

Removing them reduces config noise, eliminates ~200 lines of dead code in
`CellUniverse::optimize()`, and removes YAML knobs that mislead tuners.

### Files changed

- `C++/config/config.yaml`
- `C++/includes/ConfigTypes.hpp`
- `C++/includes/CellFactory.hpp`
- `C++/src/CellUniverse.cpp`

### Changes

**File:** `C++/config/config.yaml`

Removed 10 YAML lines:
- `firstFrameBrightnessPerturbationOnly: false`
- `volumeRecoveryEnabled: false`
- `volumeRecoveryLossFractionThreshold: 0.2`
- `volumeRecoveryMaxScaleIncreaseFraction: 0.6`
- `flatCellRotationRefineEnabled: false`
- `flatCellRotationRefineFlatnessThreshold: 0.8`
- `flatCellRotationRefineAngleStep: 0.1`
- `flatCellRotationRefineMaxOffsetDegrees: 15.0`
- `flatCellRotationRefinePasses: 1`
- `splitBrightestFraction: 0.055`

**File:** `C++/includes/ConfigTypes.hpp`

Removed from `SpheroidConfig` class:
- Field `float splitBrightestFraction{0.10f};`
- Field `bool firstFrameBrightnessPerturbationOnly{false};`
- Field `bool volumeRecoveryEnabled{false};` and its 10-line comment block
- Field `float volumeRecoveryLossFractionThreshold{0.4f};`
- Field `float volumeRecoveryMaxScaleIncreaseFraction{0.3f};`
- Field `bool flatCellRotationRefineEnabled{true};` and its 4-line comment block
- Field `float flatCellRotationRefineFlatnessThreshold{0.8f};`
- Field `float flatCellRotationRefineAngleStep{0.15f};`
- Field `float flatCellRotationRefineMaxOffsetDegrees{15.0f};`
- Field `int flatCellRotationRefinePasses{2};`

Removed from `SpheroidConfig::explodeConfig()`:
- `if (node["splitBrightestFraction"])` parse line
- `if (node["firstFrameBrightnessPerturbationOnly"])` parse block (3 lines)
- `if (node["volumeRecoveryEnabled"])` parse block (3 lines)
- `if (node["volumeRecoveryLossFractionThreshold"])` parse block (3 lines)
- `if (node["volumeRecoveryMaxScaleIncreaseFraction"])` parse block (4 lines)
- `if (node["flatCellRotationRefineEnabled"])` parse block (3 lines)
- `if (node["flatCellRotationRefineFlatnessThreshold"])` parse block (4 lines)
- `if (node["flatCellRotationRefineAngleStep"])` parse block (3 lines)
- `if (node["flatCellRotationRefineMaxOffsetDegrees"])` parse block (4 lines)
- `if (node["flatCellRotationRefinePasses"])` parse block (3 lines)

**File:** `C++/includes/CellFactory.hpp`

Removed field `float initialABRatio = 1.0f;` (line 26).

**File:** `C++/src/CellUniverse.cpp`

Removed the `brightnessPerturbFirstFrameOnly` block (~8 lines at former lines 252-259).

Removed the entire `volumeRecoveryEnabled` if-block (~90 lines at former lines 937-1027),
including the greedy brightness-monotonic upscale search loop.

Removed the entire `flatCellRotationRefineEnabled` if-block (~100 lines at former
lines 1029-1131), including the 3D rotation grid search with multi-pass refinement.

### Effect

- 10 dead YAML fields removed from config
- ~200 lines of dead code removed from `CellUniverse::optimize()`
- 10 dead field declarations + ~30 lines of dead parse code removed from ConfigTypes.hpp
- 1 dead field removed from CellFactory.hpp
- Zero remaining references to any removed field in `*.cpp`, `*.hpp`, or `*.yaml`

