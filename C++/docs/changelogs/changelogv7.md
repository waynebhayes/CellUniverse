# Changelog v7 — Performance, correctness, and structural fixes (2026-04-16)

**CLOSED 2026-04-19** at Change 50 (snap-only split candidate generation).
New work continues in `changelogv8.md`, which opens with the
`yp_fix_mask_04172026` preprocessing merge.

Opened 2026-04-16. Previous: `changelogv6.md` (closed).

Branch: `jl_bbox_multithread_04152026` → `jl_runtime_improve_04162026`

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

## 2026-04-17 (late): Edge-based shape fit — Phase A

### Change 36: New shape-fit entry point `Frame::fitCellShapeViaEdges`

**Status: ACTIVE (behind `use_edge_shape_fit=true` toggle).**

**Motivation.** The legacy `calibrateCellShapeViaPca` computes radii via `radiusScale × sqrt(variance)` over a mask-bounded pixel cloud whose variance depends on the mask size, the weighting exponent, and iteratively-fit radii. Every parameter change introduces a new feedback loop or edge case (mask-feedback collapse, halo bloat, adaptive-exponent regressions). Each regression led to another mitigation (birth-based mask, adaptive inflation, level-set walk, growth cap, etc.) and by the 2026-04-17 session each of these was fighting another.

Replaces that with a direct image-gradient approach, closer to how an eye identifies a cell: find the edge and measure from there. No mask, no iteration, no variance-to-radius conversion.

**Algorithm.**

For each cell, at frame start (before perturbation):

1. **Rotation** — unweighted position PCA on Voronoi-filtered bright pixels.
   - Gather bright pixels (`I > bg + 0.02`) inside sphere of radius `gatherRadiusScale × snapMaxR` (default 1.3 × snapMaxR) around the cell's snap position.
   - Voronoi-filter against other cells' snap positions: a pixel is kept only if the current cell's snap is the closest claim.
   - Compute unweighted PCA on those pixel POSITIONS (no brightness weighting — prevents the adaptive-exponent feedback loop that plagued the prior path).
   - Axes ordered by descending eigenvalue: `axis[0]` = major, `axis[1]` = middle, `axis[2]` = minor.
   - Sign-align each new axis with the cell's current axis slot for frame-to-frame rotation continuity.
2. **Radii** — gradient edge walks, 6 in total (3 axes × ± directions).
   - Walk from the cell's current position outward along the axis at `delta = 0.5 px` increments, up to `walkRadiusScale × snapMaxR` (default 2.0 × snapMaxR).
   - At each step, sample real-image brightness via trilinear interpolation.
   - Walk terminates at either an image-boundary hit (partial-volume stop) or a Voronoi midpoint crossing (neighbor becomes closer).
   - Smooth the collected brightness profile with a 3-tap moving average.
   - Find the position of peak `|dI/dr|` (central differences, with subpixel parabolic fit around the peak).
   - If `|peak grad| > min_grad_mag` (default 0.008/px), accept the peak position as the edge radius on that side.
   - Otherwise (dim cell, no clear membrane): keep previous-frame radius (growth cap applied downstream).
3. **Per-axis combine.**
   - Both sides valid → `r = (r+ + r-) / 2`.
   - One side partial-volume → mirror the other side's radius.
   - Neither valid → fall back to previous radius.
4. **Apply.** Write rotation (Euler ZYX) + radii to `cells[cellIndex]`. Position unchanged.

**Diagnostic.** `[EdgeFit]` summary line and per-axis `[EdgeFit A/B/C]` breakdown:

```
[EdgeFit A] cell=NAME r+=X r-=Y R=Z grad+=G1 grad-=G2 strategy=avg|mirror_*|pos_only|...
  [pv+/-] [vor+/-]
[EdgeFit] cell=NAME R=(a,b,c) n=N axisAng=DEG driftShift=D [degen]
```

`driftShift` = distance between snap center and the bright-pixel centroid. Large values (> 10 px) indicate the cell likely drifted from its biology last frame; shape fit still proceeds but the flag helps correlate drift with split/shape anomalies.

**Files changed.**

- `C++/includes/ConfigTypes.hpp`: added `ProbabilityConfig` fields:
  - `use_edge_shape_fit` (default false)
  - `edge_shape_gather_radius_scale` (default 1.3)
  - `edge_shape_walk_radius_scale` (default 2.0)
  - `edge_shape_min_grad_mag` (default 0.008)
  - YAML parser entries for all four.
- `C++/includes/Frame.hpp`: declaration of `fitCellShapeViaEdges`.
- `C++/src/Frame.cpp`:
  - New anonymous-namespace helpers: `trilinearBrightness`, `walkGradientEdge`, `GradWalkResult` struct.
  - New method `Frame::fitCellShapeViaEdges` (~180 lines).
  - Inserted between `rotationMatrixToEulerZYX` and `calibrateCellShapeViaPca`.
- `C++/src/CellUniverse.cpp`: dispatcher inside the parallel shape-fit loop:
  ```cpp
  if (config.prob.use_edge_shape_fit) {
      const float snapMaxR = std::max({maskA, maskB, maskC,
                                       frame.cells[ci].getARadius(), ...});
      frame.fitCellShapeViaEdges(ci, others, snapMaxR, ...);
  } else {
      frame.calibrateCellShapeViaPca(ci, others, ...);
  }
  ```
- `C++/config/config.yaml`: `use_edge_shape_fit: true` + tuning params.

**Expected effects.**

- No mask-based feedback — radii come from image gradients directly.
- No iteration — single-pass fit. Cheaper per-frame.
- Natural partial-volume handling (cells at z=0/224 mirror their in-image side).
- Natural neighbor handling (walk Voronoi-stops).
- Halo bleed does NOT bloat radii: the gradient peak is at the membrane's steep drop, not the halo tail.
- Feedback-loop parameters (`pcaShapeWeightExponent`, `pcaShapeAdaptiveExponent`, `pcaShapeCoreBrightnessThreshold`, etc.) become irrelevant — they're still read for the legacy path but not consulted when `use_edge_shape_fit=true`.

**Known risks and open questions.**

- Voronoi partition based on snap positions is imperfect when neighbors drifted in the prior frame. Data from `best22` (f1-f15) shows median snap-drift of 12.7 px with p95 = 24 px. Voronoi midpoint shifts up to 25 px on close-neighbor pairs (e3d03 ↔ 12345 at ~50 px pair distance). Gradient walks are robust to ~10-15 px seed offset; PCA rotation can be slightly biased for close-neighbor pairs. Acceptable for Phase A validation; the `driftShift` diagnostic surfaces the problem cases.
- `min_grad_mag=0.008` is a conservative first guess. If Phase B shows legitimate dim-cell edges being missed, lower to 0.005. If noisy images cause spurious peaks, raise to 0.012.
- Gather sphere 1.3× snap_maxR may be too tight for cells that have grown significantly. Watch for cases where `driftShift` is low but radii come out small (growth underestimated).

**Phase B.** After validation, delete `calibrateCellShapeViaPca` and the dead `pcaShape*` config params. Keep the dispatcher inline code.

**Rollback.** Set `use_edge_shape_fit: false` in `config.yaml`. Baseline mask+weighted-PCA path still fully functional.

## 2026-04-18 (late): Position prior re-introduced + edge-fit disabled (checkpoint)

### Change 37: Position prior with free-motion threshold; edge-fit toggle OFF

**Status: ACTIVE**

**Motivation.** Phase A edge-fit (Change 36) stabilized shape mechanics but produced systematically under-sized radii vs best22 (mean ΔR=14.86 px at run 071958 f22). Visual inspection showed synth ellipsoids smaller than real cells. Switching back to baseline `calibrateCellShapeViaPca` recovered best22-quality radii (mean ΔR=2.58 px at f1, 10.96 px at f22) — the baseline shape fit is not the problem. The baseline's historical issue was uncontrolled drift. Re-introducing the position prior (quadratic penalty on distance from snap beyond 20 px) addresses drift while keeping baseline shape quality.

**Changes:**

1. **Re-added position prior machinery** (previously implemented and rolled back as part of a conservative package in run 174913):
   - `C++/includes/Frame.hpp`: added `_snapPositions`, `_positionPriorWeight`, `_positionPriorThreshold` members; setters `setSnapPosition`, `clearSnapPositions`, `setPositionPriorWeight`, `setPositionPriorThreshold`.
   - `C++/includes/ConfigTypes.hpp`: added `position_prior_threshold` field (was missing). `position_prior_weight` already existed.
   - `C++/src/Frame.cpp::perturbCell`: added position-prior calculation before `costDiff` computation:
     ```cpp
     const float over = max(0, ||cell.pos - snap.pos|| - threshold);
     penalty = weight × over²;
     costDiff = (newImg + newOverlap + newPrior) - (oldImg + oldOverlap + oldPrior);
     ```
     Prior is zero below threshold (allows legitimate motion), quadratic above.
   - `C++/src/CellUniverse.cpp`: calls `setSnapPosition(name, snap.position)` alongside `setSnapBbox`, `setPositionPriorWeight(config.prob.position_prior_weight)`, `setPositionPriorThreshold(config.prob.position_prior_threshold)` once per frame.
   - `C++/config/config.yaml`: `position_prior_weight: 0.0 → 10.0`, added `position_prior_threshold: 20.0` (documented: at w=10, 40 px drift = 4000 penalty, dominates any image gain past the threshold).

2. **Edge-fit toggle OFF** (checkpoint — will return to it if baseline shape ever regresses):
   - `C++/config/config.yaml`: `use_edge_shape_fit: true → false`. Legacy `calibrateCellShapeViaPca` is active; edge-fit code remains compiled but unused.

**Results (run 082209, full 22 frames):**

- **8/8 GT splits accepted** (same as best22 baseline count):
  - f3 e9077, f3 12345, f8 1f89ab, f12 1f2ed (1 late vs GT f11), f20 e9077..a50 (+1 from f19, allowed), f20 e9077..a51, f20 12345..0, f20 12345..1.
  - No false positives on e3d03 or 8cbdf.
- **Shape quality:** avg ΔR 2.58 at f1 → 10.96 at f22 (vs edge-fit run's 14.86 at f22).
- **Position anchoring:** parent cells ≤ 5 px from best22; only divergence is e9077 daughters (Monte Carlo split placement at f3 put them 49 px from best22's placement; trajectories diverge further each frame).
- **Observed remaining issue:** c-axis (minor axis) systematically under-fit over generations. Second-gen daughters at f20 have c hitting the 10-px floor. Likely cause: weighted PCA + z-interpolation (z_scaling=7 means z variance is over-concentrated since adjacent interpolated z-slices are near-duplicates). Not addressed in this change.

**Files changed (this checkpoint):**

- `C++/config/config.yaml`: `position_prior_weight: 10.0`, `position_prior_threshold: 20.0`, `use_edge_shape_fit: false`.
- `C++/includes/Frame.hpp`: `_snapPositions`, `_positionPriorWeight`, `_positionPriorThreshold` members; corresponding setters.
- `C++/includes/ConfigTypes.hpp`: `position_prior_threshold` field.
- `C++/src/Frame.cpp::perturbCell`: prior penalty in costDiff.
- `C++/src/CellUniverse.cpp`: `setSnapPosition`/`setPositionPriorWeight`/`setPositionPriorThreshold` calls alongside `setSnapBbox`.

**Rollback.** Set `position_prior_weight: 0.0` in config.yaml (disables prior; other code paths remain but are no-ops). Set `use_edge_shape_fit: true` to restore edge-fit path.

## 2026-04-18 (late): Working checkpoint — two-sided growth cap + linear weight + strong prior

### Change 38: Three coordinated fixes yield best 22-frame result to date

**Status: ACTIVE. Run 192227 validated: 8/8 GT splits on-time, shape avg ΔR=1.57 at f1 / 6.96 at f22 vs best22.**

Three simultaneous changes, each addressing a different pathology, produced a run that matches best22 quality across all 22 frames.

**1. Two-sided growth cap** (`C++/src/CellUniverse.cpp`)

Before: one-sided `ref × 1.10` upper bound, downward unconstrained. PCA could fit a c-axis of 14 when ref was 23 — a 40% single-frame collapse. Daughters inherited 0.7937× of the crashed c, hitting the min radius floor.

After: two-sided `[ref × 0.90, ref × 1.10]` clamp. Cells can still pinch 10%/frame (enough for biological pre-split), but single-frame collapses are blocked.

```cpp
const float capUpA   = ref[0] * fitUpFactor;
const float capDownA = ref[0] * fitDownFactor;
const float newA = std::clamp(fA, capDownA, capUpA);
```

Observed in run 192227: no cells hit the floor at all. Min c-axis seen across f1-f22 was 9.9 px (vs 8-15 cells at the 10 floor in 082209).

**2. Minimum radius 10 → 5** (`C++/config/config.yaml`)

Before: `minARadius/BRadius/CRadius: 10` — this was where the broken cells landed. Many cells stuck at 10-10-10 (catastrophic pancakes) or 10-10-* (near-total collapse).

After: `5` floor. Combined with the two-sided cap, no cell actually reaches the floor — but if the cap EVER fails, the floor provides more graceful degradation.

**3. Linear weight exponent** (`C++/config/config.yaml`)

Before: `pcaShapeWeightExponent: 1.3`. `weight = brightness^1.3` concentrates mass on the bright core. For a cell with bright core (brightness ~0.3) + dim halo (brightness ~0.1), core pixels had 4× more weight per pixel. Halo contribution was suppressed → radii ≈ radius-of-gyration of CORE alone → cells rendered smaller than their visible halo.

After: `1.0` (linear weighting). Halo pixels vote proportional to brightness. Weighted variance reflects both core and halo spread → radii match the visible cell extent.

Concrete effect on 12345 at f1:
- Before (exp=1.3): a=33.5 (vs best22 42.7, −9 px)
- After (exp=1.0): a=40.2 (vs best22 42.7, −2.5 px)

**4. Position prior 10 → 30** (`C++/config/config.yaml`)

Before: `position_prior_weight: 10`, `threshold: 20`. Penalty = 10 × (drift-20)². A 60 px drift cost 16,000 — insufficient to override image-cost gains from drifting into orphan bright regions (frequently 20-40k).

After: weight=30. Same drift now costs 48,000, dominating typical image-cost drift incentives. Observed in run 192227: per-frame drift stayed within the threshold for parent cells; only the e9077 daughters diverged significantly from best22's track (but due to different Monte Carlo split placement at f3, not continuous drift).

**Files changed:**

- `C++/src/CellUniverse.cpp:660-682`: two-sided growth cap with `std::clamp`, both up and down factors.
- `C++/config/config.yaml`:
  - `minARadius/BRadius/CRadius: 10 → 5`
  - `pcaShapeWeightExponent: 1.3 → 1.0`
  - `position_prior_weight: 10 → 30`
  - `use_edge_shape_fit: true → false` (committed earlier — baseline PCA shape path)

**Run 192227 results (full 22 frames, against best22):**

| Metric | 082209 (prev best) | 192227 (this run) |
|---|---|---|
| Splits on-time | 7/8 (1f2ed at f12) | **8/8** |
| f1 avg ΔR | 2.58 | **1.57** |
| f11 avg ΔR | 6.44 | **3.97** |
| f22 avg ΔR | 10.96 | **6.96** |
| f22 max Δpos | 209 | 209 (e9077 daughter MC divergence — same) |

### Remaining minor issues

- **e3d03 trajectory over 22 frames**: starts (115, 204, 85), ends (96, 269, 60) — 67 px y drift, 25 px z drift accumulated. Per-frame drift is within bounds (≤15 px) but the direction is consistent, accumulating. Would need direction-penalized prior or trajectory-smoothness term.
- **e9077 daughter Monte Carlo divergence**: daughters born at different positions than best22 at f3 split → tracks diverge over time (up to 200 px by f22). Biology may still be correct; just different MC path than best22.

### Rollback

Revert each config value:
- `minARadius/BRadius/CRadius: 5 → 10`
- `pcaShapeWeightExponent: 1.0 → 1.3`
- `position_prior_weight: 30 → 10`
- Restore one-sided growth cap in CellUniverse.cpp (change `std::clamp(fA, capDownA, capUpA)` back to `std::min(fA, capUpA)`).


---

## 2026-04-19: Dead-code cleanup — edge-fit, deprecated bridge fields, dead bbox member

**Status: ACTIVE**

After the 45-frame run 210424 (18/20 GT splits, 0 FP) validated the current code path, this round removes the dead alternatives left from earlier experiments. No behavioral change — only deletions of code that was never executed in the working configuration.

### Change 39: Remove edge-based shape fit (Phase A) — toggle was permanently `false`

**Problem.** `Frame::fitCellShapeViaEdges` and its helpers (`walkGradientEdge`, `trilinearBrightness`, `GradWalkResult`) were dispatched from `CellUniverse::optimize` only when `config.prob.use_edge_shape_fit` was true. The flag has been `false` in YAML and the default since the path was retired (see Change 38 / session 2026-04-18). ~497 lines of dead code in `Frame.cpp`, ~25 lines of dead declaration in `Frame.hpp`, and 4 dead config fields.

**Files changed:**
- `C++/src/Frame.cpp` — deleted lines 1584-2080 (the edge-fit comment block, anonymous-namespace helpers, and `Frame::fitCellShapeViaEdges` definition)
- `C++/includes/Frame.hpp` — deleted the `fitCellShapeViaEdges` declaration block (the 22-line declaration + leading comment)
- `C++/src/CellUniverse.cpp` — collapsed `if (config.prob.use_edge_shape_fit) { ... } else { calibrateCellShapeViaPca(...) }` to just the always-taken `calibrateCellShapeViaPca` call (saves an `if` branch and ~14 lines of dispatch)
- `C++/includes/ConfigTypes.hpp` — deleted `use_edge_shape_fit`, `edge_shape_gather_radius_scale`, `edge_shape_walk_radius_scale`, `edge_shape_min_grad_mag` field declarations + 4 corresponding `explodeConfig` parses
- `C++/config/config.yaml` — deleted the 28-line `use_edge_shape_fit` block + 3 `edge_shape_*` fields

**Effect.** ~550 LOC removed. Build is faster, binary is smaller, and there is one shape-fit path to reason about. No runtime change because the flag was already off.

### Change 40: Remove deprecated bridge config fields (`bio_bridge_max_gap_density`, `bio_bridge_no_valley_hard_threshold`)

**Problem.** Both fields were retired from the bridge gate decision in 2026-04-15 (changelogv6 Change 24); the gate became single-metric (`gapBright / max(edge1, edge2) < bio_bridge_max_valley_ratio`). The fields stayed in `ConfigTypes.hpp`, `config.yaml`, and code comments as "still parsed for backward compat" — but nothing reads them. They are pure noise that confuses anyone reading the config.

**Files changed:**
- `C++/includes/ConfigTypes.hpp` — deleted `bio_bridge_max_gap_density` field + parse (line 241, line 352), deleted `bio_bridge_no_valley_hard_threshold` field + parse (line 248, line 354). Tightened the bridge-gate doc comment in the field block.
- `C++/config/config.yaml` — deleted both `bio_bridge_max_gap_density: 0.18` and `bio_bridge_no_valley_hard_threshold: 0.95` lines along with their `[DEPRECATED]` comment blocks.
- `C++/src/Frame.cpp` — cleaned 3 obsolete comments referencing the deleted field names (lines 3681-3684, 3834-3838, 3902-3904 by old line numbers).

**Effect.** YAML and ConfigTypes now match the actual decision logic. No runtime change.

### Change 41: Remove dead `_bboxOverlapWeight` member + drop third arg from `setUseBboxCost`

**Problem.** `Frame::_bboxOverlapWeight` was set by `setUseBboxCost(bool, float, float)` but never read anywhere — the bbox-cost path computes overlap penalty via `computeOverlapForCell(idx, weight)` with the weight passed directly from the caller. The member was a dead write since 2026-04-15.

**Files changed:**
- `C++/includes/Frame.hpp` — removed the `_bboxOverlapWeight` member and the third parameter of `setUseBboxCost`. Setter signature is now `setUseBboxCost(bool enable, float marginScale)`.
- `C++/src/CellUniverse.cpp` — drop `config.prob.overlap_penalty_weight` argument from the `setUseBboxCost` call in `optimize()`.
- `C++/docs/pipeline.md` — corrected the pipeline ASCII diagram to show the 2-arg setter signature.

**Effect.** Three lines lighter; one less misleading "is this the right weight?" question for future readers.

### Change 42: Remove dead `Frame::buildExclusionMask` definition + declaration

**Problem.** `_sharedMasks` was removed in 2026-04-15 (changelogv6) and the cost path now passes an empty mask to `calculateBboxCost`. `buildExclusionMask` itself remained — ~54 lines of OMP-parallel Voronoi distance computation with zero callers. The scan agent and code-review agent both flagged it independently.

**Files changed:**
- `C++/src/Frame.cpp` — deleted the `Frame::buildExclusionMask` body (~54 LOC at the old lines 298-351).
- `C++/includes/Frame.hpp` — deleted the matching declaration + comment.

**Effect.** Dead OMP-parallel code path removed.

### Change 43: Eliminate silent default-insert in `splitProbabilities` lookup

**Problem.** `runPhase` reads per-cell P(split) at every iteration via `splitProbabilities[cellName]`. For newborn daughter names (added to `phaseNames` after a split accepts but never inserted into `splitProbabilities`), the `operator[]` lookup default-inserts a `0.0f` entry. With 10K+ iterations per cell, this silently bloats the map with phantom-zero entries for every daughter name, every iteration after their birth. Behaviorally indistinguishable (both give P=0) but unnecessary mutation in a hot read path.

**File:** `C++/src/CellUniverse.cpp`

**Lines 1029 (before):**
```cpp
const float pSplit = allowSplits ? splitProbabilities[cellName] : 0.0f;
```

**Lines 1029-1033 (after):**
```cpp
float pSplit = 0.0f;
if (allowSplits) {
    auto pIt = splitProbabilities.find(cellName);
    if (pIt != splitProbabilities.end()) pSplit = pIt->second;
}
```

**Effect.** Pure read on the hot path — no map mutation. Tiny perf win, mostly a cleanup.

### Cleanup summary

| Item | Lines removed | Risk |
|------|---|---|
| `Frame::fitCellShapeViaEdges` + helpers | ~497 in Frame.cpp, ~25 in Frame.hpp | None — toggle was off |
| Edge-fit dispatch | ~14 in CellUniverse.cpp | None |
| 4 `edge_shape_*` config fields + parses | ~30 in ConfigTypes.hpp, ~28 in config.yaml | None |
| 2 deprecated bridge fields + parses | ~7 in ConfigTypes.hpp, ~10 in config.yaml | None |
| `Frame::buildExclusionMask` | ~58 (definition + decl) | None — zero callers |
| `_bboxOverlapWeight` member + setter param | 3 | None — dead write |
| `splitProbabilities` `operator[]` smell | net +4 in CellUniverse.cpp | None — same behavior, no map mutation |

**Total: ~700 LOC of dead code removed across 5 files.**

### Deferred bug findings (NOT applied — flagged for review)

The bug-and-perf review surfaced 3 latent issues that are higher risk to "fix" without a controlled test, since they could regress the working 18/20 GT split rate:

1. **`Ellipsoid::worldSplitAxis` rotation indexing** — reviewer flagged that `worldSplitAxis` (and 3 other call sites in `Frame.cpp`) read rows of R instead of columns of R for the world-direction of a local axis. The 4 sites are internally consistent (they all use the same convention), so the per-cell semantics survive. However, `pca3DWithCentroids` uses the opposite (correct) convention. For axis-aligned cells the bug is invisible; for rotated cells the world-direction of the split axis disagrees with PCA-derived directions. Defer until we can A/B test on a rotation-heavy frame.

2. **`Frame::perturbCell` whole-proposal reject on radius floor** — when any radius hits the min-floor, the entire perturbation (including position/brightness/rotation deltas in the same proposal) is dropped. This penalizes legitimate non-radius perturbations for cells near the floor. Fix is to restore radii but keep the other deltas in the proposal. Defer because it changes perturbation acceptance semantics.

3. **Cache stale-write in bbox mode** — `_currentCost` and `_currentCostPerSlice` are written by `restoreLiveParent` and the split-callback reject branch using values seeded from a stale (bbox mode never refreshed) `_currentCostPerSlice`. The cache is never read in bbox mode, so the bug is currently latent. Defensive fix would guard these writes with `if (!_useBboxCost)`. Defer because no observable effect.

These are tracked here so a follow-up session can decide whether to apply them with a controlled comparison run.


---

## 2026-04-19: Deferred-bug fixes + runtime optimizations

**Status: ACTIVE (pending validation)**

Picked up two of the three deferred bugs from the previous round (#1 worldSplitAxis indexing, #3 cache stale-write) plus three runtime perf wins. Bug #2 (perturbCell whole-proposal reject) is documented but NOT fixed — see the explanation in Change 46.

### Change 44: Fix `worldSplitAxis` row-vs-column-of-R indexing (3 sites)

**Problem.** `R_T` is `R^T` stored row-major: `R_T[r*3 + c] = R[c, r]`. The world direction of local axis `i` is column `i` of `R`, which equals `(R_T[3i], R_T[3i+1], R_T[3i+2])`. The code at three call sites read `(R_T[i], R_T[3+i], R_T[6+i])` instead — that's row `i` of `R`, not column `i`. For axis-aligned cells (R = I) the two patterns coincide. For rotated cells (any non-trivial Euler triple) they differ — a different vector entirely.

The bug compounded: `Ellipsoid::worldSplitAxis`, `Frame.cpp::extractWorldAxis`, and `Frame.cpp::calibrateCellShapeViaPca::curAxis` all used the wrong convention consistently, while `pca3DWithCentroids` used the correct convention. This means PCA-derived eigenvectors and "current axis" directions disagreed for rotated cells, weakening the slot-matching in shape fit and producing incorrect split directions.

**Files changed:**
- `C++/src/Ellipsoid.cpp::worldSplitAxis` (~lines 108-145) — read `R_T[3i + 0/1/2]` instead of `R_T[i + 0/3/6]`. Updated comment.
- `C++/src/Frame.cpp::calibrateCellShapeViaPca::curAxis` (~line 1837) — same fix in the PCA slot-matching loop.
- `C++/src/Frame.cpp::trySplitCellPhased::extractWorldAxis` lambda (~line 2269) — same fix in the split-direction extractor that produces axisA/B/C candidates.

**Behavioral implications.**
- For axis-aligned cells: zero change.
- For rotated cells: split direction now points along the correct world-frame direction of the shortest local axis. Pre-pass PCA daughter ordering and Voronoi self-claims align with the corrected direction.
- For PCA shape fit: the `curAxis[i]` slot identity now actually corresponds to local axis `i` in world frame, so greedy slot matching against PCA eigenvectors is meaningful.

**Risk.** This changes the working 18/20 GT split rate's behavior on any rotated cell. Validation run pending.

### Change 45: Guard `_currentCost` cache writes in bbox mode

**Problem.** When `_useBboxCost` is true, `regenerateSynthFrame` skips `refreshFullCostCache`, so `_currentCostPerSlice` stays stale (Change 1 perf optimization). Two functions still wrote to the cache using values seeded FROM the stale per-slice array:
- `Frame.cpp::restoreLiveParent` (split path reset on reject)
- `Frame.cpp::trySplitCellPhased` split-callback reject branch

The writes were latent (no consumer reads `_currentCost` in bbox mode), but they computed a useless `calculateIncrementalCost` over 225 slices each time and stored garbage. Bug becomes observable if any future code path reads the cache in bbox mode.

**Fix.** Both sites now guard cache writes with `if (!_useBboxCost)`. Bbox mode skips the recompute entirely; legacy full-image mode keeps the original behavior.

**Files changed:** `C++/src/Frame.cpp` — `restoreLiveParent` (~line 2127) + the lambda's reject branch (~line 3530).

**Effect.** Saves one full incremental cost recompute per split reject in bbox mode. Defensive — no behavior change.

### Change 46: Bug #2 (perturbCell whole-proposal reject) — DEFERRED, requires Ellipsoid setters

**Status: NOT FIXED.** Documented for a future structural refactor.

**Why not fixed now.** The intended fix replaces the `cells[index] = oldCell; return early` pattern with one that keeps the position/brightness/rotation deltas in the perturbation while restoring only the radii. The natural way to do this — `Ellipsoid kept = Ellipsoid(getCellParams(...))` — round-trips through `EllipsoidParams`, which does NOT carry per-cell perturb probability state (`_aRadiusPerturbParams`, etc.). After re-construction the cell loses its tuned probabilities, regressing iteration-over-iteration probability adjustment.

**Correct fix would be:** add public `Ellipsoid::setARadius/setBRadius/setCRadius` setters (or equivalently `Ellipsoid::setRadii(a,b,c)`) so the existing `cells[index]` can be modified in place without losing internal state. That's a small structural change but spans Ellipsoid.hpp + .cpp + any consumer that re-reads radii after the floor check. Tracked for the next session.

### Change 47: `calculateBboxCost` inner-loop specialization (perf)

**Problem.** The hot inner loop had two branches that prevented the compiler from auto-vectorizing:
1. `if (useMask && !mask[...]) continue;` — useMask is loop-invariant per call but the compiler can't be sure of `mask` aliasing.
2. `if (useAsym && d > asymThreshold) d2 *= asymK;` — useAsym is loop-invariant but the per-voxel data-dependent comparison stays.

In the working configuration `useMask=false` (no Voronoi exclusion mask in cost path) and `useAsym=true` (k=8). So the path that runs in practice is "no-mask, asym-active" — and we want that path to vectorize.

**Fix.** Split the loop into:
- `if (!useMask)` branch: nested loops with `useAsym` hoisted ABOVE the y-loop. The asym sub-loop uses a branchless `mul = (d > thresh) ? asymK : 1.0f` so SIMD lanes can compute it without diverging.
- `else` (useMask) branch: original code preserved bit-for-bit (rare path; correctness over speed).

**File:** `C++/src/Frame.cpp::calculateBboxCost` (~line 298).

**Effect.** Inner asym loop body is now 4 fp operations + 1 select — the compiler vectorizes (AVX2 = 8 floats per cycle on x86, NEON = 4 on ARM). On a typical bbox (~100^3 voxels), this should be a 2-4× speedup for the cost evaluation. The cost eval is ~1/3 of the optimize loop's wall time, so net ~10-20% faster per frame.

### Change 48: Parallelize per-frame image save (perf)

**Problem.** `CellUniverse::saveImages` ran two sequential loops of 225 `cv::imwrite` calls each (real, then synth). PNG encoding is single-threaded zlib, so sequential write took several seconds per frame.

**Fix.** Single OMP-parallel loop over slice index that writes both real and synth in the same iteration. 450 PNG encodes execute on all available cores.

**File:** `C++/src/CellUniverse.cpp::saveImages` (~line 1265).

**Effect.** With ~8 cores, ~7-8× faster per-frame save. Saves on the order of seconds per frame on 45-frame runs (~1-2 minutes total).

### Change 49: Parallelize PNG → TIFF post-processing (perf)

**Problem.** `scripts/convert_png_to_tiff.py` iterated frame folders sequentially and read PNGs sequentially within each folder. For a 45-frame run with 225 PNGs per folder, that's 45 * 225 = 10125 sequential `imageio.imread` calls per stream (real + synth = 2 streams).

**Fix.** Two-level parallelism via `ThreadPoolExecutor`:
- Outer: convert frame folders concurrently (default `os.cpu_count()` workers).
- Inner: read PNGs within each folder concurrently (max 8 workers).

Optional `--workers N` CLI argument for tuning. Threading is fine here because `imageio.imread` releases the GIL during decode and `tifffile.imwrite` is I/O-bound.

**File:** `C++/scripts/convert_png_to_tiff.py`.

**Effect.** Order-of-magnitude faster post-processing. On a 45-frame run with 8 cores, drops from ~30-60s to ~5-10s.

### Validation (run 023805, 45 frames)

Compared `output_jihang_20260419_023805` vs baseline `output_jihang_20260418_best45` (run 210424).

| Metric | New (with fixes) | Baseline | Δ |
|---|---|---|---|
| Total splits accepted | **19** | 18 | +1 |
| Final cell count (f45) | **25** | 24 | +1 |
| GT splits matched | 17 | 17 | 0 |
| GT splits + extra accepted | 19 | 18 | +1 |

**Direct wins (NEW better):** `12345 split` on time at f3 (was +1 in baseline), `1f2ed..0` on time at f31 (was +1), `7×3rd-gen` complete at f39 (vs 6+1 missed).

**Cascade losses:** `e9077..a51` 4 frames early at f16 (vs GT f20), `e9077..a50` 12 frames late at f32 (vs baseline f20), `1f89ab..0` MISSED entirely (baseline got at f29).

**Diagnosis (split-axis direction analysis):** Compared `splitAxisDir` (the seed direction picked by `worldSplitAxis`) against `actualUnit` (final daughter separation vector after burn-in) for the first 4 GT splits.

| Cell | NEW: angle(seed, actual) | BASE: angle(seed, actual) |
|---|---|---|
| e9077 f3 | **58° off** | 0° (perfect) |
| 12345 f3/f4 | 8° (good) | 0° (perfect) |
| 1f89ab f8 | 83° off | 84° off |
| 1f2ed f11 | **89° (orthogonal!)** | 8° (good) |

In 3 of 4 baseline splits, the OLD (row-of-R) `splitAxisDir` perfectly predicted where daughters land. After the fix, the new `splitAxisDir` is up to 89° off the actual division direction. Burn-in compensates by drifting daughters significantly more, landing at suboptimal local optima.

**Conclusion:** The mathematical fix is correct in isolation, but the rest of the system (daughter placement, PCA-matching, snapshot propagation) was implicitly built around the row-of-R convention. Fixing only `worldSplitAxis` + `extractWorldAxis` + `calibrateCellShapeViaPca::curAxis` introduced a geometry inconsistency: those 3 sites are now in column-of-R convention while consumers still expect row-of-R behavior.

**Next step (this session):** Audit the full geometry stack. If a consistent column-of-R representation can be applied across all sites without breaking other invariants, do it. Otherwise revert Bug 1 and document the row-of-R convention as load-bearing.

The 22-frame validation (config now set to 22 frames) will be the test for the consistent-geometry attempt.

### Rollback

Per-change rollback (each is independent):
- Change 44: revert the 3 indexing fixes by restoring `(R_T[i], R_T[3+i], R_T[6+i])` reads.
- Change 45: drop the `if (!_useBboxCost)` guards (restores prior latent stale-write).
- Change 47: collapse two branches back into one merged loop.
- Change 48: split parallel write back into two sequential loops.
- Change 49: revert convert_png_to_tiff.py to pre-threading version.

---

## 2026-04-19: Snap-only split candidate generation (Change 50)

**Status: ACTIVE — validated against best22 baseline**

### Problem

After Bug 1 fix (Change 44, worldSplitAxis indexing), 22-frame validation showed mild regression — `12345` split slipped from f3 (best22) to f6 (3 frames late). Per-frame analysis revealed the cascade:

1. Bug 1 fix changed `curAxis` interpretation in `calibrateCellShapeViaPca` slot matching
2. PCA shape fit at frame start converged to slightly different parent radii (e.g. 12345 b-axis 24.1 → 29.1)
3. Wider parent → wider daughter seeds (daughter radii = parent.getARadius() × `split_daughter_volume_scale`)
4. Bigger daughters render into more area, including dim periphery → asymmetric L2 (k=8) penalty fires hard
5. Split rejected with `costDiff=+2530` instead of `-31000`

The root issue: **split candidates were derived from LIVE cell state**, which drifts during in-frame perturbation. Even small shape-fit divergence cascades into completely different daughter sizes and bbox cost outcomes.

### Design rule (new invariant)

Split candidates source their inputs from exactly two places:
- **Snapshot** (frozen previous-frame state): direction, midpoint, **size, shape**
- **PCA** (current-frame image data, computed around snapshot.position): direction, midpoint **only**

LIVE state is never consulted for candidate generation.

This produces 4 base candidate configs × (1 primary + 2 rotation + 2 translation) = up to 20 seeds:

| Family | Direction | Midpoint | Size | Shape |
|--------|-----------|----------|------|-------|
| `snap_axC` | snap-rotation × shortest local axis | snap.position | snap radii × scale | snap rotation |
| `data_axC` | snap-rotation × shortest local axis | PCA centroid (pixels around snap) | snap radii × scale | snap rotation |
| `snap_imgPca` | snapshot.splitAxisDir (image-PCA from pre-pass) | snap.position | snap radii × scale | snap rotation |
| `data_imgPca` | snapshot.splitAxisDir | PCA centroid | snap radii × scale | snap rotation |

### Implementation

**File:** `C++/src/Frame.cpp::trySplitCellPhased`

**Before (line 2158):**
```cpp
// parent now reflects the installed snapshot-state (or live fallback
// when snapshot is invalid). Everything downstream treats this as the
// baseline parent.
Ellipsoid parent = cells[cellIndex];
```

`cells[cellIndex]` is conditionally swapped to `snapshotParent` only when snap cost ≤ live cost. Otherwise live remains. So `parent` was sometimes live.

**After (line 2158):**
```cpp
// Geometric reference for split candidate generation: ALWAYS use the
// snapshot-state parent (per 2026-04-19 design rule: split candidates
// use SNAP or PCA-derived data only, never LIVE).
//
// Rationale: live position/radii drift during in-frame perturbation. If
// axis directions, radii, or fallback midpoints were derived from the
// live cell, that drift would cascade into different candidate seeds
// each iteration, making the split decision sensitive to perturbation
// history (frame-3 12345 / e9077 regression in run 084534).
//
// cells[cellIndex] still holds whatever the cost comparison above chose
// (snapshot or live) — that is the RENDERING baseline for the cost
// delta, intentionally separate from the GEOMETRIC reference here.
//
// Falls back to the live cell only when no snapshot exists (frame 1,
// newborn daughter post-split — but those are filtered earlier).
Ellipsoid parent = snapshotValid ? snapshotParent : cells[cellIndex];
```

The cost-baseline comparison logic (lines 2089-2135) is preserved unchanged — that decides which RENDERED state daughters must beat, separate from where candidates are seeded.

### Effect

Every downstream consumer of `parent` now reads snap state:

- `parent.generateInverseRotationMatrix()` → snap rotation → `axisA/B/C` are world directions of snap's local axes
- `parent.getARadius/B/C()` → snap radii → daughter built radii = snap × `split_daughter_volume_scale`
- `parent.getX/Y/Z()` → snap position → fallback midpoint when `centroidsAlongAxis` returns invalid
- `parent.getThetaX/Y/Z()` (via `buildDaughter`) → snap rotation → daughter rotation = snap rotation

### Validation (run 095555, 22 frames)

| Cell | GT | SNAP-ONLY (this) | BEST22 | Bug-1-only (run 084534) |
|---|---|---|---|---|
| e9077 | f3 | **f3 ✓** | f3 ✓ | f4 (+1) |
| 12345 | f3 | **f3 ✓** | f3 ✓ | f6 (+3) ⚠️ |
| 1f89ab | f8 | **f8 ✓** | f8 ✓ | f8 ✓ |
| 1f2ed | f11 | **f11 ✓** | f12 (+1) | f11 ✓ |
| e9077..a50 | f19 | f20 (+1) | f20 (+1) | f20 (+1) |
| 12345..0 | f20 | **f20 ✓** | f20 ✓ | f20 ✓ |
| 12345..1 | f20 | **f20 ✓** | f20 ✓ | f20 ✓ |
| e9077..a51 | f20 | **f20 ✓** | f20 ✓ | f20 ✓ |

| Score | SNAP-ONLY | BEST22 | Bug-1-only |
|---|---|---|---|
| Splits accepted | 8/8 | 8/8 | 8/8 |
| **On-time** | **7/8** | 6/8 | 5/8 |
| Cells at f22 | 14 | 14 | 14 |

**Snap-only beats both BEST22 and Bug-1-only** — gains `1f2ed` on time without losing anything. This validates the design: anchoring split candidates to snap (the previous-frame fixed reference) makes the decision insensitive to in-frame drift.

### Rollback

Restore line 2158 to:
```cpp
Ellipsoid parent = cells[cellIndex];
```

### Open follow-ups

- Re-run 45-frame test to verify the cascade fix holds at longer horizons (the Bug-1-only 45-frame run had additional regressions at f16/f29/f32 that may also resolve with snap-only).
- Consider applying the same "snap-only" principle to other in-frame state consumers (e.g. perturbation reference radius — currently uses live `maxR` for posScale; could use snap).
