# Changelog v7 — Performance, correctness, and structural fixes (2026-04-16)

Opened 2026-04-16. Previous: `changelogv6.md` (closed).

Branch: `jl_bbox_multithread_04152026`

---

## 2026-04-16: Performance + correctness fixes — bbox ordering, stale cost, move semantics, PCA weight cache

Six changes shipped together. No logic changes — only performance optimizations and correctness fixes for silent bugs.

### Change 1: Fix bbox cost ordering — skip wasted `refreshFullCostCache` on bbox frames

**Problem.** `CellUniverse::optimize` called `frame.regenerateSynthFrame()` BEFORE `frame.setUseBboxCost()`. Since `_useBboxCost` defaults to `false` on each Frame object, `regenerateSynthFrame()` always ran `refreshFullCostCache()` — a full 225-slice asymmetric L2 computation (~32M pixel ops) — then immediately discarded the result when `setUseBboxCost(true)` was called 13 lines later. Wasted ~32M pixel ops per frame for every bbox frame (frames 2+).

**Fix.** Moved `setUseBboxCost()` before `regenerateSynthFrame()` so the bbox flag is set when `regenerateSynthFrame` checks `if (!_useBboxCost) refreshFullCostCache()`.

**File:** `C++/src/CellUniverse.cpp`

**Lines 275-291 (before):**
```cpp
frame.regenerateSynthFrame();

// ... comment block ...
const bool bboxActiveThisFrame = config.prob.use_bbox_cost && (frameIndex > 0);
frame.setUseBboxCost(bboxActiveThisFrame,
                     config.prob.bbox_margin_scale,
                     config.prob.overlap_penalty_weight);
```

**Lines 275-293 (after):**
```cpp
const bool bboxActiveThisFrame = config.prob.use_bbox_cost && (frameIndex > 0);
frame.setUseBboxCost(bboxActiveThisFrame,
                     config.prob.bbox_margin_scale,
                     config.prob.overlap_penalty_weight);

frame.regenerateSynthFrame();
```

**Effect.** Saves ~32M pixel ops per frame on frames 2+. Frame 1 (`bboxActiveThisFrame=false`) still runs `refreshFullCostCache` as before.

### Change 2: Fix stale `_currentCost` in split snapshot-vs-live baseline comparison

**Problem.** In bbox mode, `perturbCell`'s accept callback does NOT update `_currentCost` or `_currentCostPerSlice` (intentionally — bbox cost is computed fresh each time). By the time a split attempt fires, `_currentCost` holds the initial value from `refreshFullCostCache` (which no longer even runs for bbox frames after Change 1). The snapshot-vs-live comparison at `trySplitCellPhased` line 1976:
```cpp
const double liveCostBeforeSwap = _currentCost;  // STALE
```
compared the swap result against the initial synth cost, not the current live synth. Could pick the wrong baseline (snap vs live) in edge cases.

**Fix.** When `_useBboxCost` is true, compute a fresh bbox cost over a temporary union bbox covering both live and snap positions. When false (legacy), use the existing `calculateIncrementalCost` path unchanged. Also gated `_currentCost`/`_currentCostPerSlice` updates on `!_useBboxCost` in the accept branch.

**File:** `C++/src/Frame.cpp`

**Lines 1976-2013 (before):**
```cpp
const double liveCostBeforeSwap = _currentCost;
// ... swap in snapshot, generateSynthFrameFast, calculateIncrementalCost ...
const bool useSnapshotBaseline = (swappedImageCost <= liveCostBeforeSwap);
if (useSnapshotBaseline) {
    _synthFrame = swappedSynth;
    _currentCost = swappedImageCost;
    _currentCostPerSlice = swappedPerSlice;
}
```

**Lines 1976-2053 (after):**
```cpp
// ... swap in snapshot, generateSynthFrameFast ...
if (_useBboxCost) {
    // Fresh bbox cost over union of live+snap positions
    BoundingBox3D cmpBbox = union(liveBbox, snapBbox);
    liveCostForComparison = calculateBboxCost(cmpBbox, _synthFrame, noMask);
    snapCostForComparison = calculateBboxCost(cmpBbox, swappedSynth, noMask);
} else {
    // Legacy: calculateIncrementalCost (correct — _currentCost maintained)
    liveCostForComparison = _currentCost;
    snapCostForComparison = calculateIncrementalCost(...);
}
const bool useSnapshotBaseline = (snapCostForComparison <= liveCostForComparison);
if (useSnapshotBaseline) {
    _synthFrame = swappedSynth;
    if (!_useBboxCost) { _currentCost = ...; _currentCostPerSlice = ...; }
}
```

Log line updated to show `(bbox)` or `(full)` suffix.

**Effect.** Split baseline comparison is now correct in bbox mode. Both live and snap costs are computed fresh over the same bbox region.

### Change 3: Cache PCA pixel weights — eliminate redundant `pow()` calls

**Problem.** `calibrateCellShapeViaPca` computed `effectiveWeight(bp.weight)` (which calls `std::pow()` for non-integer exponents) twice per pixel per PCA iteration: once in the centroid loop and once in the covariance loop. With N=500-5000 pixels × 15 iterations × 6-14 cells, this was 90K-2.1M redundant `pow()` calls per frame.

**Fix.** Cache per-pixel effective weights in `std::vector<double> pixelWeights` during the centroid loop. Reuse cached values in the covariance loop.

**File:** `C++/src/Frame.cpp`

**Lines 1734-1754 (before):**
```cpp
double sx = 0, sy = 0, sz = 0, wsum = 0;
for (const auto &bp : pixels) {
    const double we = effectiveWeight(bp.weight);
    sx += bp.pos.x * we; // ...
}
// ...
for (const auto &bp : pixels) {
    // ...
    const double we = effectiveWeight(bp.weight);  // REDUNDANT pow()
    cxx += we * dx * dx; // ...
}
```

**Lines 1734-1758 (after):**
```cpp
std::vector<double> pixelWeights(pixels.size());
double sx = 0, sy = 0, sz = 0, wsum = 0;
for (size_t pi = 0; pi < pixels.size(); ++pi) {
    const double we = effectiveWeight(pixels[pi].weight);
    pixelWeights[pi] = we;
    sx += pixels[pi].pos.x * we; // ...
}
// ...
for (size_t pi = 0; pi < pixels.size(); ++pi) {
    // ...
    const double we = pixelWeights[pi];  // cached — no pow()
    cxx += we * dx * dx; // ...
}
```

**Effect.** Eliminates N×iters `pow()` calls per cell per frame. For typical workloads (6 cells × 2000 pixels × 15 iters), saves ~180K pow() calls per frame.

### Change 4: Move semantics in `perturbCell` lambda capture

**Problem.** `perturbCell` returned a `CostCallbackPair` containing a lambda that captured `newSynthFrame` (225 cv::Mat) and `newCostPerSlice` (225 doubles) by copy. Called ~3000 times per frame (cells × iterations_per_cell), this produced ~3000 full vector copies per frame. The callback is called immediately and discarded — the copy is wasted.

**Fix.** Capture by `std::move` with `mutable` lambda. Inside the accept branch, assign with `std::move` too.

**File:** `C++/src/Frame.cpp`

**Lines 653-668 (before):**
```cpp
CallBackFunc callback = [this, newSynthFrame, newCostPerSlice, ...](bool accept) {
    // ...
    this->_synthFrame = newSynthFrame;
    this->_currentCostPerSlice = newCostPerSlice;
};
```

**Lines 653-671 (after):**
```cpp
CallBackFunc callback = [this,
                         newSynthFrame = std::move(newSynthFrame),
                         newCostPerSlice = std::move(newCostPerSlice),
                         ...](bool accept) mutable {
    // ...
    this->_synthFrame = std::move(newSynthFrame);
    this->_currentCostPerSlice = std::move(newCostPerSlice);
};
```

**Effect.** Saves ~3000 vector copies per frame. Each copy was 225 cv::Mat (shallow copy, so ~225 ref-count bumps + vector alloc) and 225 doubles (1.8 KB alloc+copy).

### Change 5: Move semantics in split candidate loop + callback closure

**Problem.** Inside `trySplitCellPhased`:
- The candidate loop captured `bestCells = cells`, `bestSynth = _synthFrame`, `bestPerSlice = _currentCostPerSlice` by copy on each new best candidate — but `cells`/`_synthFrame`/`_currentCostPerSlice` were immediately overwritten from `savedCells`/`savedSynth`/`savedPerSlice` on the next line.
- The refine re-capture did the same pattern.
- `savedCells`/`savedSynth`/`savedPerSlice` were declared `const`, preventing moves into callback copies at the end. The callback closure captured 6 large vectors by copy.

**Fix.**
- Candidate loop: `bestCells = std::move(cells)` etc. (source overwritten immediately after).
- Refine re-capture: same pattern, with diagnostic reads BEFORE the move.
- `savedCells`/`savedSynth`/`savedPerSlice`: removed `const` qualifier.
- End-of-function: `std::move` into `savedCellsCopy`/`savedSynthCopy`/`savedPerSliceCopy`.
- Callback closure: capture by `std::move`. Accept/reject branches assign by `std::move`.

**Files changed:** `C++/src/Frame.cpp` — candidate loop (~line 2764-2780), refine re-capture (~line 2970-2990), saved state declarations (~line 2506-2509), callback copies (~line 3356-3362), callback closure (~line 3379-3394), accept/reject branches (~line 3407-3431).

**Effect.** Saves ~K+6 vector copies per split attempt (K = number of candidates, typically 20). Each copy involves full `std::vector<Ellipsoid>` or `std::vector<cv::Mat>` allocations.

### Change 6: Fix use-after-move in refine diagnostic (Bug #1 from code scan)

**Problem.** Change 5 introduced a use-after-move: after `bestCells = std::move(cells)`, the diagnostic block read `cells[d1IdxRefine]` and `cells[d2IdxRefine]` — accessing a moved-from (empty) vector. Undefined behavior: crash on debug builds, garbage output on release builds.

Additionally, `const Ellipsoid &refinedD1 = cells[d1IdxRefine]` created a reference into `cells` that dangled after the move, then was read in the `std::cout` statement.

**Fix.** Moved the entire diagnostic block (position reads, radii reads, `std::cout`) BEFORE the `std::move` calls. Changed `const Ellipsoid &refinedD1` from reference to copy (`const Ellipsoid refinedD1`) — the copy is a single Ellipsoid, negligible cost.

**File:** `C++/src/Frame.cpp`

**Lines 2970-3007 (after fix):**
```cpp
bestImageCost = evalImageCost(_synthFrame);
bestTotal = bestImageCost + computeOverlapPenalty(...);

// Diagnostic reads BEFORE moving cells
const cv::Point3f postRefineD1(cells[d1IdxRefine].getX(), ...);
const cv::Point3f postRefineD2(cells[d2IdxRefine].getX(), ...);
const Ellipsoid refinedD1 = cells[d1IdxRefine];  // copy, not ref
const Ellipsoid refinedD2 = cells[d2IdxRefine];
// ... std::cout ...

// Move AFTER all reads complete
bestCells = std::move(cells);
bestSynth = std::move(_synthFrame);
bestPerSlice = std::move(_currentCostPerSlice);
```

**Effect.** Eliminates undefined behavior on every split attempt that reaches the refine path.

---

## 2026-04-16: Bug fixes — data race in parallel PCA, daughters excluded from perturbation

### Change 7: Fix data race in parallel PCA shape fit (Bug #2)

**Problem.** The PCA shape fit runs inside `#pragma omp parallel for` (CellUniverse.cpp line 570). Each thread:
- **Writes** to `frame.cells[ci]` — radii, rotation, and optionally position via `calibrateCellShapeViaPca`.
- **Reads** all other cells' positions via `buildShapeClaimSet` at line 561, which calls `other.getX()`, `other.getY()`, `other.getZ()`, `other.getName()` for every cell in the frame.

Thread A writing `cells[B]` while thread C reads `cells[B]` is a data race — undefined behavior under the C++ memory model. With `pcaShapeUpdatePosition=false` (current config), the written fields (radii/rotation) differ from the read fields (position/name), making it benign on x86. With `pcaShapeUpdatePosition=true`, the position IS written, and another thread reading it sees a torn value — real data race with real consequences (wrong Voronoi claims → wrong PCA pixel sets → wrong shape fits).

**Fix.** Snapshot all cell positions into a `std::vector<CellPosSnapshot>` BEFORE the parallel region. `buildShapeClaimSet` reads from the snapshot (immutable) instead of from `frame.cells`. The parallel region now only writes to `frame.cells[ci]` (each thread to its own index) and reads from the snapshot (shared read-only). No data race.

**File:** `C++/src/CellUniverse.cpp`

**Lines 533-555 (before):**
```cpp
auto buildShapeClaimSet = [&](const std::string &selfName) -> Frame::ClaimSet {
    Frame::ClaimSet others;
    for (const auto &other : frame.cells) {            // DATA RACE: reads live cells
        const std::string otherName = other.getName();
        if (otherName == selfName) continue;
        others[otherName].push_back(cv::Point3f(
            other.getX(), other.getY(), other.getZ()));
    }
    return others;
};
// ... parallel for ...
```

**Lines 533-564 (after):**
```cpp
struct CellPosSnapshot { std::string name; float x, y, z; };
std::vector<CellPosSnapshot> cellPosSnap;
cellPosSnap.reserve(frame.cells.size());
for (const auto &c : frame.cells) {
    cellPosSnap.push_back({c.getName(), c.getX(), c.getY(), c.getZ()});
}

auto buildShapeClaimSet = [&](const std::string &selfName) -> Frame::ClaimSet {
    Frame::ClaimSet others;
    for (const auto &snap : cellPosSnap) {             // reads immutable snapshot
        if (snap.name == selfName) continue;
        others[snap.name].push_back(cv::Point3f(snap.x, snap.y, snap.z));
    }
    return others;
};
// ... parallel for ...
```

**Effect.** Eliminates data race in OMP parallel PCA shape fit. Safe regardless of `pcaShapeUpdatePosition` setting.

### Change 8: Allow daughters to receive perturbation after mid-frame split (Bug #3)

**Problem.** `runPhase` builds `phaseNames` from `frame.cells` at frame start (line 1118-1121). When a split is accepted mid-frame, two daughter cells (`parentName + "0"`, `parentName + "1"`) are added to `frame.cells`. `rebuildEligible()` at line 1076 rebuilds the eligible index list by checking `phaseNames.count(cname)`. Since daughter names were not in `phaseNames` (they didn't exist at frame start), they are excluded and receive zero perturbation iterations for the rest of the frame.

For early-frame splits (e.g., iteration 200 of 3000), daughters miss ~2800 perturbation iterations. They rely entirely on the split burn-in + refine positioning quality.

**Fix.** After a split is accepted, insert the daughter names into `phaseNames` before calling `rebuildEligible()`. Changed `phaseNames` from `const std::set<std::string>&` to `std::set<std::string>&` (non-const) to allow insertion.

Daughters get `pSplit = 0.0f` (not in `splitProbabilities` map — `operator[]` default-inserts 0.0f), so they are never eligible for split attempts in the same frame — only perturbation. This is correct (daughters shouldn't split the frame they were born).

**File:** `C++/src/CellUniverse.cpp`

**Line 960 (before):**
```cpp
auto runPhase = [&](const std::set<std::string> &phaseNames,
```

**Line 960 (after):**
```cpp
auto runPhase = [&](std::set<std::string> &phaseNames,
```

**Lines 1058-1064 (before):**
```cpp
if (accept) {
    splitAccepted++;
    splitAcceptedInPhase.insert(cellName);
    previousSnapshots.erase(cellName);
    // Rebuild eligible: the parent cell at cellIdx was replaced
    // by two daughters appended to the cells vector.
    rebuildEligible();
```

**Lines 1058-1072 (after):**
```cpp
if (accept) {
    splitAccepted++;
    splitAcceptedInPhase.insert(cellName);
    previousSnapshots.erase(cellName);
    // Add daughter names to phaseNames so they become eligible
    // for perturbation during the rest of this frame.
    phaseNames.insert(cellName + "0");
    phaseNames.insert(cellName + "1");
    // Rebuild eligible: the parent cell at cellIdx was replaced
    // by two daughters appended to the cells vector.
    rebuildEligible();
```

**Effect.** Newly-split daughters now receive perturbation iterations for the remainder of the frame. For a 6-cell frame with 500 iters/cell (3000 total), a split accepted at iteration 500 gives each daughter ~2500/8 ≈ 312 perturbation iterations to refine position — vs zero before.

---

## 2026-04-16: Lower daughter refit ceiling 1.1 → 1.05 — resolve structural conflict with single-daughter bio gate

**Problem.** The `bio_max_single_daughter_volume_fraction` gate (0.65) is structurally incompatible with the refit ceiling `split_daughter_refit_max_radius_fraction` (1.1).

The maximum possible single-daughter volume ratio after refit = `(ceil × volumeScale)³ = (1.1 × 0.7937)³ = 0.665`. This EXCEEDS the 0.65 gate threshold. Any daughter that refits to ceiling on all 3 axes is rejected — this catches all phantom splits (expected) but also real splits where the daughter's pixel cloud is slightly oversized.

Data across 3 reference runs:

| Category | sdv range | n |
|---|---|---|
| Accepted real splits | 0.307 – 0.627 | 35 |
| 12345 f3 (real, rejected) | 0.651 | 1 |
| Phantom rejections | 0.651 – 0.665 | 25 |

The gate sits in a 2.4% gap between real (max 0.627) and phantom (min 0.651). 12345 at f3 — a GT-correct split — refitted its daughter to sdv=0.651, falling into the phantom zone. This caused the entire cascade: 12345 unsplit → 31 px overlap with e3d03 → non-saturating overlap penalty pushed e3d03 155 px at f5 → shape collapse → false positive split at f18.

**Fix.** Lower `split_daughter_refit_max_radius_fraction` from 1.1 to 1.05.

At ceil=1.05:
- Max phantom sdv = `(1.05 × 0.7937)³ = 0.579`
- Gate (0.65) margin = +12.3% (vs -2.4% at ceil=1.1)
- All 35 accepted real splits (max 0.627) remain comfortably below gate
- 12345 f3 daughter would cap at ~0.58, passing the gate

The 5% ceiling still allows daughters to grow slightly from their volume-preserving birth size. Subsequent frames grow naturally via bounded-growth reference (±5%/frame) in the main shape fit.

**File:** `C++/config/config.yaml`

**Before:** `split_daughter_refit_max_radius_fraction: 1.1`
**After:** `split_daughter_refit_max_radius_fraction: 1.05`

**Effect.** Resolves the structural conflict. Real splits that refit near ceiling now pass the bio gate. Phantom splits still rejected (sdv ≤ 0.579 < 0.65).

### Rollback

`split_daughter_refit_max_radius_fraction: 1.05 → 1.1`.

---

## 2026-04-16: Config validation — data-driven tuning + configurable percentile

Four changes based on cross-run data validation against the best22 (8/8 GT) reference.

### Change 9: Make percentile radius configurable + lower 0.95 → 0.90

**Problem.** Percentile-based radii (introduced in v6) were hardcoded at 95th percentile. Comparison against best22 (variance-based, 8/8 GT) showed radii 20-73% larger:
- e3d03 c-axis: 13.7 → 23.7 (+73%)
- 1f2ed c-axis: 17.6 → 27.5 (+56%)
- 1f89a a-axis: 35.4 → 44.8 (+27%)

Root cause: the `percentileRadius` lambda uses unweighted pixel projections. Dim halo pixels at the mask boundary (maskScale=1.8) get equal vote with bright core pixels. The 95th percentile captures halo extent, not cell extent. Fit growth cap fires on 2-8 cells EVERY frame — chronic overestimation.

**Fix.** Added `pcaShapeRadiusPercentile` to `EllipsoidConfig` (default 0.90, YAML-tunable). Changed the hardcoded `0.95` in `calibrateCellShapeViaPca` to read from `Ellipsoid::cellConfig.pcaShapeRadiusPercentile`. Set config to 0.90 — cuts top 10% instead of 5%, reducing halo contamination. For a uniform [0,R] distribution, 90th percentile = 0.9R (10% underestimate; fit growth cap tracks up over a few frames).

**Files changed:**
- `C++/includes/ConfigTypes.hpp`: new field `float pcaShapeRadiusPercentile{0.90f}` + YAML parser
- `C++/src/Frame.cpp`: `calibrateCellShapeViaPca` reads `Ellipsoid::cellConfig.pcaShapeRadiusPercentile` instead of hardcoded `0.95`
- `C++/config/config.yaml`: `pcaShapeRadiusPercentile: 0.90`

### Change 10: Tighten adaptive inflation trigger — pcaShapeCoreFractionLow 0.15 → 0.25

**Problem.** With `pcaShapeCoreFractionLow=0.15`, ALL cells got partial-to-full adaptive inflation because all had avgPcore > 0.15 (range 0.162–0.227). Uniform cells like 1f89a (pCore=0.162) and e9077 (pCore=0.189) got 3-4% radius inflation they didn't need. The adaptive path was supposed to only help bright peaked cells.

**Fix.** `pcaShapeCoreFractionLow: 0.15 → 0.25`. Only cells with pCore > 0.25 (8cbdf=0.226, 1f2ed=0.227) get inflation. Uniform cells stay at base exponent.

**File:** `C++/config/config.yaml`

### Change 11: Raise split_cost threshold — 375 → 2000

**Problem.** Accepted splits had costDiff -5154 to -24333. Cost-rejected had +11054 to +46828. The threshold at -375 sat 13.7x below the weakest real split, leaving a 16K dead zone where marginal phantoms could pass.

**Fix.** `split_cost: 375 → 2000`. Still 2.6x below the weakest real split (-5154). Adds meaningful cost barrier for candidates between -375 and -2000.

**File:** `C++/config/config.yaml`

### Change 12: Widen bbox margin — 2.0 → 2.5

**Problem.** Run 030254 showed 5/6 unique cells exceeding their bbox half-extent. e3d03 drifted 169 px with bboxHalf=43. The overlap penalty overwhelmed the 1x undershoot anchor — cells rationally abandoned their position to escape overlap. At 2.0×, the window was too tight for cells under strong overlap pressure.

**Fix.** `bbox_margin_scale: 2.0 → 2.5`. Window at maxR=25: 62.5 px (was 50 px). 25% more headroom before the cell exits the bbox and loses the position anchor entirely.

**File:** `C++/config/config.yaml`

### Rollback

Config: revert the four values. Code: remove `pcaShapeRadiusPercentile` field from `ConfigTypes.hpp`, restore `0.95` hardcoded in `Frame.cpp`.

---

## 2026-04-16: Auto-calibrating configs — proportional sigma, overlap, bbox floor, split cost

Four changes that make critical parameters scale with cell state instead of being fixed constants.

### Change 13: Radius-proportional perturbation sigma

**Problem.** Position perturbation sigma is fixed (`x/y sigma=5, z sigma=8`). Large cells (R=50) need 10 steps to cross themselves; small daughters (R=15) need 3. Large cells under-explore; small cells over-explore and waste iterations.

**Fix.** Added `positionScale` parameter to `Ellipsoid::getPerturbedCell()` that scales position offsets. In `perturbCell`, compute `scale = maxR / perturbSigmaReferenceRadius`. A cell at the reference radius (25 px) uses the base sigma unchanged; larger cells take proportionally bigger steps.

**Files changed:**
- `C++/includes/Ellipsoid.hpp`: new `positionScale` parameter (default 1.0)
- `C++/src/Ellipsoid.cpp`: scale x/y/z offsets by `positionScale`
- `C++/src/Frame.cpp`: compute `posScale = maxR / refR` in `perturbCell`
- `C++/includes/ConfigTypes.hpp`: new `perturbSigmaReferenceRadius{0.0f}`
- `C++/config/config.yaml`: `perturbSigmaReferenceRadius: 25.0`

### Change 14: Brightness-proportional overlap penalty

**Problem.** Overlap penalty weight (30000) is fixed regardless of cell brightness. Dim cells have low image cost but the same overlap penalty → overlap dominates → cell flees its position. Root cause of the e3d03 155 px drift cascade.

**Fix.** In `perturbCell`, scale the overlap weight by `(cellBrightness / meanBrightness)²`. Dim cells get lower overlap weight proportional to their lower image-cost contribution. Mean brightness computed once per frame in `CellUniverse::optimize`, passed via `Frame::setMeanCellBrightness()`.

**Files changed:**
- `C++/includes/Frame.hpp`: new `_meanCellBrightness` member + setter
- `C++/src/Frame.cpp`: `perturbCell` computes `effectiveOverlapWeight` using brightness ratio
- `C++/src/CellUniverse.cpp`: computes mean brightness at frame start

### Change 15: Bbox minimum half-extent floor

**Problem.** Small cells (R=10) get a bbox half-extent of only 25 px (2.5 × 10). The position anchor has few voxels to work with, and the cell can easily drift out.

**Fix.** Added `kMinBboxHalfExtent = 40.0f` floor in `computeBboxAtPoint`. `r = max(40, marginScale × radius)`. Small cells always get at least 40 px of bbox.

**File:** `C++/src/Frame.cpp` — `computeBboxAtPoint`

### Change 16: Adaptive split cost threshold (proportional to baseline)

**Problem.** Fixed `split_cost=2000` applies the same absolute threshold to all cells. A dim cell with baseline cost 10K needs only -2000 improvement (20%); a bright cell with baseline 100K needs the same -2000 (2%). The fractional requirement is uneven.

**Fix.** `effective_threshold = max(split_cost, split_cost_fraction × baselineImageCost)`. At `fraction=0.03`, every cell must improve by at least 3% of its own baseline OR the fixed 2000, whichever is larger.

**Files changed:**
- `C++/includes/ConfigTypes.hpp`: new `split_cost_fraction{0.0f}`
- `C++/src/Frame.cpp`: `trySplitCellPhased` computes `adaptiveThreshold`
- `C++/config/config.yaml`: `split_cost_fraction: 0.03`

### Rollback

Revert all code changes. Config: set `perturbSigmaReferenceRadius: 0`, remove `split_cost_fraction` line, restore `bbox_margin_scale: 2.5` (no floor in code).

---

## 2026-04-16: Weighted percentile radii — restore elongation signal

### Change 17: Replace unweighted percentile with brightness-weighted percentile

**Problem.** Unweighted percentile radii destroyed the elongation signal that drives split detection. Cross-run comparison at 12345 f2:

| Run | Method | Radii (a,b,c) | Elongation | f3 split? |
|---|---|---|---|---|
| best22 | Variance | (33.3, 30.8, 23.3) | 1.43 | YES |
| 030254 | Percentile (unweighted) | (31.4, 29.3, 30.8) | 1.07 | NO |
| 042424 | Percentile (unweighted) | (31.3, 29.3, 30.7) | 1.07 | NO |

Root cause: unweighted percentile gives every pixel equal vote. The pixel-collection mask is roughly spherical (birth × 1.8), so dim halo pixels extend equally in all directions. On the thin c-axis, the cell has fewer bright pixels but the same amount of halo → percentile reports similar extent → cell appears round.

Variance-based radii worked because the weight exponent (1.3) de-emphasized halo. The weighted variance along the thin axis was genuinely smaller because bright pixels concentrated near the center.

This was the root cause of ALL the problems in runs 030254 and 042424: 12345 appeared round → low P(split) → wrong split direction → missed f3 split → overlap cascade → e3d03 drift → false positives.

**Fix.** Changed `percentileRadius` to use weighted percentile: sort pixels by |projection|, accumulate brightness weights, stop when cumulative weight reaches `percentile × totalWeight`. Uses the already-cached `pixelWeights[]` array (from Change 3). Bright core pixels dominate the percentile; dim halo pixels contribute proportionally less.

Expected effect: thin axes get smaller radii (fewer bright pixels at large projections), recovering the elongation signal. For 12345 f2, the c-axis should be noticeably thinner than a/b, giving elongation ~1.3-1.4 instead of 1.07.

**File:** `C++/src/Frame.cpp` — `percentileRadius` lambda inside `calibrateCellShapeViaPca`

**Before (unweighted):**
```cpp
std::vector<float> proj;
for (const auto &bp : pixels) {
    proj.push_back(std::abs(projection));  // all pixels equal vote
}
std::nth_element(proj.begin(), proj.begin() + idx, proj.end());
return proj[idx];
```

**After (weighted):**
```cpp
struct ProjWeight { float proj; double weight; };
std::vector<ProjWeight> pw;
for (size_t pi = 0; pi < pixels.size(); ++pi) {
    pw.push_back({std::abs(projection), pixelWeights[pi]});
}
std::sort(pw.begin(), pw.end(), by_proj);
double cumWeight = 0.0;
for (const auto &p : pw) {
    cumWeight += p.weight;
    if (cumWeight >= percentile * totalWeight) return p.proj;
}
```

**Performance note:** `std::sort` is O(N log N) vs `std::nth_element` O(N). For N=2000 pixels × 3 axes × 15 iters, sort adds ~270K comparisons per cell per frame. Acceptable — sort is cache-friendly and the total overhead is <1ms per cell. If profiling shows this matters, a weighted nth-element can be implemented later.

### Rollback

Restore the unweighted `std::vector<float> proj` + `std::nth_element` path. Remove `pixelWeights` usage in the lambda.

---

## 2026-04-16: Revert to variance-based radii — fundamental fix for undersizing

### Change 18: Replace weighted percentile with variance-based radii

**SUPERSEDES Change 17** (weighted percentile). Weighted percentile produced correct elongation (1.81) and fixed the 12345 f3 split, but radii were 40-50% undersized — cells hit the min radius floor (10) on b/c axes. The weight exponent (1.3) that helps PCA rotation makes the weighted percentile too aggressive for radius measurement.

**Root cause analysis.** The weight exponent serves two purposes:
1. PCA rotation/shape: exp=1.3 correctly de-emphasizes halo → accurate axis direction
2. Radius measurement: exp=1.3 crushes boundary pixels → radius underestimated

Variance-based `radiusScale × sqrt(variance)` solves this because `radiusScale = sqrt(5)` is analytically calibrated to compensate for the weighting's effect on the variance. The percentile has no such calibration — it measures the Nth weight-quantile directly, with no compensation factor.

**Evidence.** Run 054437 (weighted percentile, 0.90):
- e3d03: R=(15.4, 11.1, 10.0) — 40-50% undersized, b/c at floor
- 12345_0: R=(18.4, 15.6, 12.7) — 49% undersized
- e9077..a50 missed at f19 — too small to generate meaningful cost signal

Best22 (variance-based, radiusScale=2.236):
- e3d03: R=(26.2, 20.9, 13.7) — correct
- 12345_0: R=(36.3, 28.0, 22.4) — correct
- 8/8 GT splits, correct elongation (1.43)

**Fix.** Reverted `percentileRadius` lambda to the original variance formula:
```cpp
targetA = cellRadiusInflation * radiusScale * sqrt(matchedVariance[0]);
targetB = cellRadiusInflation * radiusScale * sqrt(matchedVariance[1]);
targetC = cellRadiusInflation * radiusScale * sqrt(matchedVariance[2]);
```

This uses the SAME `matchedVariance` eigenvalues from the PCA covariance (weighted with exp=1.3) and the calibrated `radiusScale = sqrt(5)` from config.

Birth-based mask (Change 2 of v6) prevents the mask-feedback loops that motivated the original percentile experiment. The percentile is no longer needed.

The `pcaShapeRadiusPercentile` config field and the percentile lambda are removed from the active code path. `pcaShapeRadiusScale` (2.236) is the active radius parameter again.

**File:** `C++/src/Frame.cpp` — `calibrateCellShapeViaPca`, percentile lambda replaced with variance formula

**Effect.** Radii should match best22 levels. Elongation signal preserved (variance along thin axis is genuinely smaller). The `cellRadiusInflation` adaptive path still applies for peaked cells. Combined with all other session fixes (birth mask, bbox cost, move semantics, bug fixes, auto-calibration), this should produce correct radii + correct splits + fast runtime.

### Rollback

Restore the weighted-percentile lambda from Change 17.

---

## 2026-04-16: Three-stage PCA pipeline — position refinement + adaptive mask

### Change 19: Per-cell position refinement + adaptive mask sizing

**Problem.** Cells were drifting across frames (100-200 px shifts observed in run 134156). The static birth mask + fixed `maskScale` couldn't adapt: undersized cells had weak position anchors, and mispositioned cells had their PCA fit applied to the wrong blob.

Two compounding issues:
1. **Wrong position**: snap from previous frame may be drifted. Fixed mask on wrong center → PCA fits the wrong region.
2. **Wrong mask size**: one `maskScale` for all cells can't handle size variance. 1.8 was too big (runtime), 1.4 was too tight (undersized radii).

**Fix.** Added two pre-stages to `calibrateCellShapeViaPca`:

**Stage 0a: Position refinement.** Before PCA, compute weighted peak centroid (using `weight^3` — so the brightest peak dominates over dim gaps). Apply a capped shift (max 10% of birth maxR per frame) toward the peak. Pulls drifted cells back gradually.

Using `weight^3` instead of linear weighting is critical for pre-split cells: a linear centroid falls in the dim gap between two emerging daughters, which would destroy split detection. `weight^3` makes the brighter peak dominate, pulling the cell to ONE daughter blob — the pre-pass still finds both via `imageGroundExpectedDaughters`.

**Stage 0b: Adaptive mask sizing.** With correct position, iteratively grow/shrink the mask based on bright-pixel density in the outer shell:
- Density > 0.6 in shell [0.85R, R] → mask too small, expand 15%
- Density < 0.2 → mask too big, shrink 10%
- Otherwise → right-sized, stop
- Clamped to [0.8, 2.2] × birth radii, max 5 iterations

This replaces the fixed `maskScale` config value. Each cell gets its own mask, adapted to actual image footprint. Cells in bright halo expand; cells in sparse regions shrink.

**Files changed:**
- `C++/src/Frame.cpp`: `calibrateCellShapeViaPca` now starts with position refinement and adaptive mask computation before the PCA iteration loop
- `C++/config/config.yaml`: `pcaShapeMaskScale: 1.4 → 1.2` (now the STARTING scale for adaptive search, not a fixed value)

**Expected effect:**
- Drifted cells pull back to correct position over 2-3 frames (10% cap)
- Undersized cells expand their mask → PCA sees full extent → correct radii
- Cells in empty regions shrink their mask → less wasted computation
- Pre-split cells stay on one blob (exponent=3 avoids gap centroid)

**Logs:**
- `[Pos Refine] cell=... peak=... shift=... capped=... newPos=...`
- `[Adaptive Mask] cell=... scale=... density=... iters=...`

### Rollback

Remove the Stage 0a and Stage 0b blocks. Restore `const float effMaskA = (maskA > 0.0f) ? maskA : cell.getARadius();` with the fixed `maskScale` multiplier.

---

## 2026-04-16: Gradient-based adaptive mask + tighter soft Voronoi band

### Change 20: Adaptive mask uses inner/outer density GRADIENT

**Problem.** Run 155004 showed the absolute-density-based adaptive mask never expanded: density in the outer shell is naturally 0.2-0.5 because real cells have fuzzy boundaries (many outer pixels below brightness cutoff). The expand threshold (density > 0.6) was never reached. All cells stayed at the starting scale, radii remained 20-40% smaller than best22.

**Fix.** Replace absolute density with GRADIENT between inner and outer shells:
- Inner shell: [0.70R, 0.85R]
- Outer shell: [0.85R, R]
- ratio = outerDensity / innerDensity
- ratio > 0.50 → expand (outer shell still has >50% of inner's brightness → cell extends further)
- ratio < 0.15 → shrink (outer shell is <15% of inner → mask overshoots)
- Between → converged

This measures whether brightness is TRAILING OFF (cell ended) or STILL PRESENT (cell extends). The ratio is robust to the absolute density floor caused by brightness-filtering dim boundary pixels.

Also raised `kScaleMax: 2.2 → 2.5` to allow more expansion for cells that need it.

**File:** `C++/src/Frame.cpp` — Stage 0b inside `calibrateCellShapeViaPca`

Log updated: `[Adaptive Mask] cell=... scale=... innerN=... outerN=... ratio=... iters=...`

### Change 21: Tiered soft Voronoi — full weight for isolated pixels

**Problem.** The original soft Voronoi applied ownership weighting to ALL pixels, including isolated boundary pixels where no neighbor is anywhere nearby. Those pixels got ownership < 1.0 just because `otherBest` was finite (e.g., ownership = 0.8 for pixels where nearest neighbor is 40+ px away). This shrank the weighted variance across the entire cell, contributing to the undersized radii.

**Fix.** Three-tier ownership:
- `otherBest < 0.25 × selfBest` → reject (clearly other's territory)
- `0.25 × selfBest ≤ otherBest < selfBest` → soft ownership (contested zone)
- `otherBest ≥ selfBest` → **full weight** (self is closer than any other cell)

Only genuinely contested pixels — where another cell is actually closer than self — get down-weighted. Isolated boundary pixels (where self is naturally closest) contribute their full brightness to the PCA variance, preserving radius accuracy.

**File:** `C++/src/Frame.cpp` — `gatherBrightPixelsVoronoi`

**Effect.** Cells in isolation (most of the interior + boundary of well-separated cells) get full-weight PCA. Only cells genuinely competing for pixels at the boundary see soft weighting. Radii should recover toward best22 levels while still preventing Voronoi starvation when cells are actually close.

### Rollback

Revert both changes: restore single-density threshold (0.2/0.6) for mask, restore two-tier Voronoi (reject OR soft always).

---

## 2026-04-16: FUNDAMENTAL FIX — temporal priors (hard radius cap + position prior)

Previous fixes all tried to make PCA behave through guards around it (adaptive mask, soft Voronoi, position refinement). But PCA has no temporal memory — it fits whatever pixels you give it. If the pixel cloud contains neighbors, halo, or drift, PCA bloats. The real fix is to add temporal priors that constrain the cell's state across frames.

### Change 22: Hard ±5% radius cap (replaces 10% upper-only cap)

**Problem.** Run 162856 f4 showed 12345..1 PCA output R=(78.8, 33.5, 18.6) when birth radii were (26.7, 22.9, 15.8). The one-sided 10% fit cap reduced to ref × 1.10 = ~28.6 upper bound, but even 10% per frame compounds unboundedly.

**Fix.** Replace single-sided 10% cap with **two-sided ±5% cap**:
```cpp
constexpr float fitGrowthCap = 0.05f;
newA = std::clamp(fA, ref[0] * 0.95f, ref[0] * 1.05f);
```

Biology: cells change size <5%/frame. This is a hard physical prior. No image evidence can make a cell double or halve in one frame — the code simply won't allow it.

For newly-born cells (no ref yet), the cap doesn't apply — they establish their initial radii via daughter refit (which has its own clamps).

**File:** `C++/src/CellUniverse.cpp` — fit growth cap section

### Change 23: Quadratic position prior in perturbCell

**Problem.** The snap-anchored bbox provides an image-based position anchor (voxels at snap always in scope). But this can be overwhelmed by:
- Overlap penalty pushing cell away from a crowded neighbor
- Bright halo from neighbors attracting the cell
- Soft Voronoi down-weighting boundary pixels → weakens bbox anchor

The result: cells drift 100-200 px across frames despite the snap bbox. Biologically, cells move <5 px per frame — 100+ px drift is pathological.

**Fix.** Add a TEMPORAL prior to perturbCell: quadratic penalty on distance from snap position.

```cpp
penalty = position_prior_weight × ||cell.pos - snap.pos||²
costDiff = (newImageCost + newOverlap + newPrior) - (oldImageCost + oldOverlap + oldPrior)
```

At `weight=50`:
- 5 px drift → prior = 1250 (minor)
- 20 px drift → prior = 20,000 (significant, comparable to bbox image cost)
- 50 px drift → prior = 125,000 (overwhelms any gain)

This is independent of image evidence. The cell CAN'T drift 100 px even if the image "wants" it to. The prior is the biological constraint: cells don't teleport.

**Files:**
- `C++/includes/Frame.hpp`: new `_snapPositions` map + `_positionPriorWeight` + setters
- `C++/includes/ConfigTypes.hpp`: new `position_prior_weight` field
- `C++/src/Frame.cpp`: `perturbCell` computes `oldPositionPrior`, `newPositionPrior`, includes in costDiff
- `C++/src/CellUniverse.cpp`: calls `setSnapPosition` alongside `setSnapBbox`; calls `setPositionPriorWeight`
- `C++/config/config.yaml`: `position_prior_weight: 50.0`

### Why this is the fundamental fix

Previous fixes were reactive — adaptive mask, soft Voronoi, position refinement all tried to make PCA see the "right" pixels. But:
- Cells don't magically get the right mask (adaptive mask can still fail)
- Pixels are pixels; weighting doesn't change biology

The temporal priors work differently:
- Radii: constrained at code level to ±5%/frame change. No PCA output can override.
- Position: quadratic penalty in cost. Drift costs more than any image gain.

Together they create a Kalman-filter-like behavior: the cell's state at frame N is the state at N-1 plus small deformations allowed by image evidence within biology constraints.

**Expected impact:**
- No more runaway radius bloat (R=78 from birth R=27 impossible)
- No more 100+ px drift (quadratic prior dominates cost)
- Daughters can't immediately split (stuck within 5% of birth radii for several frames — fit growth blocked even if PCA tries to grow them)
- e3d03 and similar non-splitters stay at their real size and position

### Rollback

Revert `fitGrowthCap: 0.05 → 0.10` (with one-sided upper logic). Set `position_prior_weight: 0` to disable prior. Remove snap position storage from Frame.

---

## 2026-04-16 evening: Asymmetric radius cap — allow pre-split pinching

### Change 24: Radius cap asymmetric — tight upper (+10%), loose lower (-50%)

**Problem.** Change 22's symmetric ±5% cap was too tight. Run 170909 showed 1f89ab f8 split FAILED because the cell couldn't pinch fast enough. Compare radii evolution:

| Frame | Best22 c-axis | Run 170909 c-axis | Delta/frame |
|---|---|---|---|
| f5 | 36.4 | 34.2 | — |
| f6 | **22.6** (-38%) | 32.5 (-5%, capped) | 7x slower |
| f7 | 18.5 (-18%) | 30.9 (-5%, capped) | — |
| f8 | SPLITS | stays round | — |

Biology: cells pinch dramatically (30-40% per frame) along the division axis during pre-split. The ±5% cap prevented this, keeping cells round, so the split cost gate rejected (daughters don't fit much better than a round parent).

**Fix.** Asymmetric cap:
- **Upper bound: +10% per frame** (tight — prevents PCA bloat when it sees halo/neighbors)
- **Lower bound: -50% per frame** (loose — allows pre-split pinching observed in best22)

Rationale: growth by 40% in one frame is always pathological (bloat). Shrinkage by 40% is normal biology (division pinching). The asymmetry reflects this.

```cpp
constexpr float fitUpCap   = 0.10f;  // +10%/frame max growth
constexpr float fitDownCap = 0.50f;  // -50%/frame max shrink (pinching)
newA = clamp(fA, ref[0] × 0.50, ref[0] × 1.10);
```

**File:** `C++/src/CellUniverse.cpp` — fit growth cap section

**Expected effect:**
- Pre-split cells can pinch → develop bimodal shape → split cost gate fires correctly
- PCA can't bloat from wrong pixels (upper cap still tight)
- 1f89ab f8 split should work (matches best22 timing)

Log updated: `[Fit Growth Cap] frame N clamped=X up=0.1 down=0.5`

### Rollback

`fitUpCap: 0.10, fitDownCap: 0.05` to restore symmetric-tight behavior.

---

## 2026-04-16 evening: COMPLETE ROLLBACK — remove all speculative constraint additions

### Change 25: Roll back adaptive mask, soft Voronoi, position refinement, position prior, asymmetric cap

**Problem.** Over the course of this session, 5 constraint-adding features were layered onto the best22 baseline:

1. Adaptive mask (gradient-based expansion/shrinkage per cell)
2. Soft Voronoi weighting (proportional ownership)
3. Position refinement (weight³ peak centroid shift)
4. Position prior (quadratic drift penalty, weight=50)
5. Asymmetric radius cap (+10% upper, -50% lower)

Each fix addressed a real symptom, but the combination is WORSE than best22. Run 170909 showed:
- Only 3 splits in 22 frames (GT: 8)
- 1f89ab f8 missed — cell couldn't pinch (f6 c-axis only went 34.2 → 32.5 vs best22's 36.4 → 22.6)
- Frame 2 cells had wrong orientations
- Overlaps still present

The speculative additions were interacting destructively. We've been chasing symptoms rather than fixing root causes.

**Fix.** Complete rollback to the best22-equivalent baseline plus only the proven bug/perf fixes from this session.

### KEPT (proven fixes)

- **Bug fixes**: bbox ordering, stale cost in split baseline, data race in parallel PCA, daughter eligibility in phaseNames, use-after-move fix
- **Perf fixes**: PCA weight cache, move semantics, z-skip render (all 4 sites)
- **Bio gate**: refit ceiling 1.1 → 1.05 (resolved structural conflict with single-daughter gate)
- **Cost gate**: neighbor-bridging rejection (catches stretched splits)
- **Cost formula**: asymK threshold 0.03 (eliminates double-boundary bias)
- **Bbox floor**: min half-extent 40 px
- **Variance-based radii**: restored from percentile experiment

### REMOVED (speculative)

- **Adaptive mask** (Stage 0b) → fixed `maskScale × birthR` (maskScale=1.6)
- **Soft Voronoi ownership** → hard keep/reject based on nearest claim
- **Position refinement** (Stage 0a, weight³ peak centroid) → removed entirely
- **Position prior** (quadratic drift penalty) → `position_prior_weight: 0.0`
- **Asymmetric radius cap** → restored one-sided +10% upper only, downward unconstrained

### Files changed

- `C++/src/Frame.cpp`:
  - Removed Stage 0a (position refinement) and Stage 0b (adaptive mask) from `calibrateCellShapeViaPca`
  - Restored hard Voronoi in `gatherBrightPixelsVoronoi`
  - Removed position prior computation in `perturbCell`
- `C++/src/CellUniverse.cpp`:
  - Restored +10% one-sided fit cap (downward unconstrained)
  - Removed `setSnapPosition`/`setPositionPriorWeight` calls
- `C++/includes/Frame.hpp`:
  - Removed `_snapPositions`, `_positionPriorWeight` members
  - Removed `setSnapPosition`, `clearSnapPositions`, `setPositionPriorWeight` accessors
- `C++/config/config.yaml`:
  - `pcaShapeMaskScale: 1.2 → 1.6` (best22-validated)
  - `position_prior_weight: 50.0 → 0.0` (disabled)

### Rationale

The best22 run (8/8 GT splits, all correct timing) proves the baseline system WORKS with:
- Fixed birth mask at 1.6
- Hard Voronoi
- No position prior or refinement
- One-sided +10% growth cap

All the subsequent issues (drift, undersized cells, missed splits in 170909) were caused by the speculative additions, not the baseline. Rolling back restores a proven configuration. Any future issues should be addressed one at a time with verification, not layered.

### Rollback

Re-apply the git diff for these fixes. Not recommended.
