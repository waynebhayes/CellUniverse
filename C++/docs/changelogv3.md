# Changelog v3: Unified Loop, Overlap Penalty, Calibration, Performance (2026-03-27)

**Branch:** `jl-restructure-cpp-03272026`

This changelog covers all non-restructure changes made on 2026-03-27/28:
- Part A: Dynamic split probability + overlap penalty + per-cell brightness
- Part B: Background auto-calibration + performance optimization + brightness init
- Part C: Revert to unified stochastic loop + lazy PCA elongation cache + adaptive P(split)

### Status Key

| Status | Meaning |
|--------|---------|
| **ACTIVE** | Change is in the current codebase |
| **REVERTED** | Change was implemented then undone |
| **SUPERSEDED** | Change was replaced by a later entry |

---

# Part A: Dynamic Split, Overlap Penalty, Per-Cell Brightness

## A1. Unified Perturbation/Split Loop — **ACTIVE**

**Problem:** Phase 1 (perturbation) and Phase 2 (split detection) ran sequentially. The team wants each iteration to decide per-cell whether to try a split or perturbation.

**File:** `C++/src/CellUniverse.cpp` — `optimize()`

**Before:** Two-phase approach — Phase 1 ran all perturbations, then Phase 2 evaluated every cell for splits independently.

**After:** Single unified loop. Each iteration:
1. Pick random cell
2. With probability `P(split)`: try split via `randomSplitCell()`, accept if cost improves
3. Else: try perturbation via `perturbCell()`, accept if cost improves

---

## A2. Overlap Penalty in Cost Function — **ACTIVE**

**Problem:** Overlapping cells were hard-rejected (`checkIfCellsValid`). This binary gate doesn't give the optimizer gradient information.

**File:** `C++/src/Frame.cpp` — `perturbCell()` (new), `computeOverlapPenalty()` (new)

**Penalty formula:** For each overlapping pair, `weight * ((combinedR - dist) / combinedR)^2`. Quadratic scaling: slight overlaps mildly penalized, deep overlaps heavily penalized.

```cpp
double Frame::computeOverlapPenalty(float weight) const
{
    double totalPenalty = 0.0;
    for (size_t i = 0; i < cells.size(); ++i) {
        auto pi = cells[i].getCellParams();
        float ri = pi.majorRadius;
        for (size_t j = i + 1; j < cells.size(); ++j) {
            auto pj = cells[j].getCellParams();
            float rj = pj.majorRadius;
            float dist = sqrt(dx*dx + dy*dy + dz*dz);
            float combinedR = ri + rj;
            if (dist < combinedR) {
                float overlapRatio = (combinedR - dist) / combinedR;
                totalPenalty += weight * overlapRatio * overlapRatio;
            }
        }
    }
    return totalPenalty;
}
```

**ConfigTypes.hpp:** Added `overlap_penalty_weight` field (default 1000.0) with YAML parsing.

**config.yaml:** Added `overlap_penalty_weight: 1000.0`.

---

## A3. Bugfix: Segfault in computeOverlapPenalty — **ACTIVE**

**Problem:** `computeOverlapPenalty()` called `cells[i].major_magnitude()` which accesses `_x_vec[0]`, but cells created from `(x,y,z,majorR,minorR)` have empty `_x_vec`. Segfault on first iteration.

**Fix:** Changed to use `getCellParams().majorRadius` instead of `major_magnitude()`.

---

## A4. PCA-Guided Split (randomSplitCell) — **ACTIVE**

**File:** `C++/src/Frame.cpp` — `randomSplitCell()` (new)

Calls `getSplitCells()` for PCA-guided daughter placement (axis + centroids). If PCA fails (too few bright pixels), falls back to random axis. No burn-in, no elongation gate — cost function + overlap penalty are the only judges.

1. Remove parent, add two daughters (half volume each: `cbrt(0.5) * parentR`)
2. Full render + cost comparison
3. Accept if `costDiff < 0`, else revert

---

## A5. Per-Cell Brightness as Perturbable Parameter — **ACTIVE**

**Problem:** All cells rendered at fixed `cell_color=0.6`. The optimizer inflates cells to maxRadius because bigger = more area covered = lower L2 cost when brightness is uniform.

**Solution:** Added `_brightness` as a per-cell perturbable parameter [0.15, 0.95].

**Files changed:**
- `ConfigTypes.hpp` — Added `PerturbParams brightness`, `minBrightness`, `maxBrightness` to SpheroidConfig
- `Spheroid.hpp` — Added `float brightness` to SpheroidParams, `float _brightness` to Spheroid
- `Spheroid.cpp` — Constructor clamps brightness. `getPerturbedCell()` perturbs brightness. `draw()` uses `_brightness` instead of `cell_color`. `getSplitCells()` daughters inherit parent brightness.
- `CellUniverse.cpp` — `saveCells()` CSV header and output include brightness
- `Frame.cpp` — `randomSplitCell()` daughters inherit parent brightness

**config.yaml:**

```yaml
cell:
  brightness:
    prob: 0.15
    mu: 0
    sigma: 0.03
  minBrightness: 0.15
  maxBrightness: 0.95
```

**Reduced perturbation sigma** (less aggressive):

```yaml
  x/y sigma: 20 -> 10
  z sigma: 12 -> 8
  majorRadius/minorRadius sigma: 4 -> 2
```

---

## A6. Suppressed TIFF Warnings — **ACTIVE**

**File:** `C++/src/main.cpp` (line 181)

```cpp
cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
```

---

# Part B: Background Calibration, Performance, Brightness Init

## B1. Root Cause of Oversized Cells

The config had `background_color: 0.2` but the real image background is ~0.42. The optimizer grew cells to cover background because `|real_bg - synth_cell|^2 = |0.42 - 0.50|^2 = 0.006` was less than `|real_bg - synth_bg|^2 = |0.42 - 0.20|^2 = 0.048`. Covering background with cells *reduced* L2 cost.

---

## B2. Add Calibration Fields to SimulationConfig — **ACTIVE** (sigmoid_k parsed but sigmoid itself reverted)

**File:** `C++/includes/ConfigTypes.hpp`

**Added fields:**

```cpp
    float sigmoid_k = 75.0f;
    float sigmoid_center_offset = 0.047f;
    int calibration_x = 20;
    int calibration_y = 20;
    int calibration_z = 0;
    int calibration_width = 50;
    int calibration_height = 31;
```

**Added parsing in `explodeConfig`:**

```cpp
    if (node["sigmoid_k"]) sigmoid_k = node["sigmoid_k"].as<float>();
    if (node["sigmoid_center_offset"]) sigmoid_center_offset = node["sigmoid_center_offset"].as<float>();
    if (node["calibration_x"]) calibration_x = node["calibration_x"].as<int>();
    if (node["calibration_y"]) calibration_y = node["calibration_y"].as<int>();
    if (node["calibration_z"]) calibration_z = node["calibration_z"].as<int>();
    if (node["calibration_width"]) calibration_width = node["calibration_width"].as<int>();
    if (node["calibration_height"]) calibration_height = node["calibration_height"].as<int>();
```

Note: `sigmoid_k` is parsed but sigmoid preprocessing was implemented then reverted (too harsh). Only the calibration fields are actively used.

---

## B3. Fix Blur Sigma (hardcoded -> config) — **ACTIVE**

**File:** `C++/src/CellUniverse.cpp` — `processImage()`

**Before:**

```cpp
cv::GaussianBlur(processedImage, processedImage, cv::Size(0, 0), 1.5);
```

**After:**

```cpp
cv::GaussianBlur(processedImage, processedImage, cv::Size(0, 0), config.simulation.blur_sigma);
```

Config value stays at 1.5, but now tunable.

---

## B4. Auto-Calibrate background_color from Real Image — **ACTIVE** (sigmoid part REVERTED, calibration-only kept)

**File:** `C++/src/CellUniverse.cpp` — `loadFrame()`, `C++/includes/CellUniverse.hpp`

**Signature change:** `const BaseConfig &config` -> `BaseConfig &config` (non-const to allow update)

**Added before z-interpolation:**

```cpp
        if (!processedZSlices.empty()) {
            int calZ = std::clamp(config.simulation.calibration_z, 0, (int)numTiffSlices - 1);
            int calX = config.simulation.calibration_x;
            int calY = config.simulation.calibration_y;
            int calW = std::min(config.simulation.calibration_width,
                               processedZSlices[calZ].cols - calX);
            int calH = std::min(config.simulation.calibration_height,
                               processedZSlices[calZ].rows - calY);

            if (calW > 0 && calH > 0) {
                cv::Rect roi(calX, calY, calW, calH);
                float bgMean = (float)cv::mean(processedZSlices[calZ](roi))[0];
                config.simulation.background_color = bgMean;
                std::cout << "[Calibration] bgMean=" << bgMean
                          << " (updated background_color)" << std::endl;
            }
        }
```

**Effect:** Measures real background (~0.42) in a cell-free calibration zone and sets `background_color` to match. Eliminates the L2 mismatch that drove cells to grow.

**Note:** Sigmoid preprocessing was implemented first (calibration + sigmoid contrast `1/(1+exp(-k*(x-center)))`) but reverted because it made the image near-binary and hard to interpret. The auto-calibration alone fixes the root cause.

---

## B5. Cache L2 Cost in Frame — **ACTIVE**

**File:** `C++/includes/Frame.hpp`, `C++/src/Frame.cpp`

**Added field:**

```cpp
double _currentCost = -1.0; // cached L2 image cost of _synthFrame
```

**Updated `regenerateSynthFrame`:**

```cpp
void regenerateSynthFrame() { _synthFrame = generateSynthFrame(); _currentCost = calculateCost(_synthFrame); }
```

**Constructor:** `_currentCost = calculateCost(_synthFrame);` after generating synth frame.

**perturbCell, randomSplitCell, trySplitCell:** All use `_currentCost` instead of `calculateCost(_synthFrame)` and update cache in callbacks on accept.

**Effect:** Eliminates one full L2 computation (225 z-slices) per iteration. For 2100 iterations per frame, saves ~2100 full L2 computations.

---

## B6. O(n) Per-Cell Overlap — **ACTIVE**

**File:** `C++/includes/Frame.hpp`, `C++/src/Frame.cpp`

**Added:**

```cpp
double Frame::computeOverlapForCell(size_t cellIdx, float weight) const
{
    double penalty = 0.0;
    auto pi = cells[cellIdx].getCellParams();
    float ri = pi.majorRadius;
    for (size_t j = 0; j < cells.size(); ++j) {
        if (j == cellIdx) continue;
        // ... same distance/overlap check as computeOverlapPenalty ...
    }
    return penalty;
}
```

**perturbCell rewritten to use it:**

Before: `computeOverlapPenalty()` O(n^2) x 2 + revert/restore cell

After: `computeOverlapForCell()` O(n) x 2, no revert needed

---

## B7. Remove End-of-Frame PCA Elongation Scan — **SUPERSEDED by C2/C3** (end-of-frame PCA removed, replaced by lazy per-cell PCA cache)

**File:** `C++/src/CellUniverse.cpp`, `C++/includes/CellUniverse.hpp`

**Removed:** `computeElongationRatios()` call at end of each frame (6 full 3D PCA volume scans) and `previousElongations` member.

**P(split):** Changed from elongation-driven (`min(0.5, base + 1-1/elongation)`, up to 50%) to fixed `config.prob.split` (3%).

---

## B8. perturbCell Takes overlapWeight Parameter — **ACTIVE**

**File:** `C++/includes/Frame.hpp`

**Before:** `CostCallbackPair perturbCell(size_t index);`

**After:** `CostCallbackPair perturbCell(size_t index, float overlapWeight = 1000.0f);`

`optimize()` now passes `config.prob.overlap_penalty_weight` instead of hardcoding.

---

## B9. printCellInfo Shows Brightness — **ACTIVE**

**File:** `C++/includes/Spheroid.hpp`

Added `<< " brightness: " << _brightness` to `printCellInfo()` output.

---

## B10. CellFactory Reads Brightness from CSV — **ACTIVE**

**File:** `C++/src/CellFactory.cpp`

**Before:** Created cells with default brightness 0.5.

**After:** Reads optional 8th column as brightness. Falls back to 0.5 if missing or "None". Backward compatible.

```cpp
float brightness = 0.5f;
if (tokens.size() >= 8 && tokens[7] != "None" && !tokens[7].empty()) {
    brightness = std::stof(tokens[7]);
}
initialCells[filePath].push_back(
    Spheroid(SpheroidParams(cellName, x, y, z, majorRadius, minorRadius,
                           0.0f, 0.0f, 0.0f, brightness))
);
```

---

## B11. Measured Cell Brightness in initial.csv — **ACTIVE**

**File:** `C++/config/initial.csv`

**Before columns:** `file,name,x,y,z,majorRadius,minorRadius,z_scaling,split_alpha,opacity`

**After columns:** `file,name,x,y,z,majorRadius,minorRadius,brightness`

**Measured values** (normalize + blur 1.5, no sigmoid):

| Cell | Brightness |
|------|-----------|
| e3d0... | 0.6238 |
| 1234... | 0.4805 |
| e907... | 0.5002 |
| 1f2e... | 0.4770 |
| 1f89... | 0.4684 |
| 8cbd... | 0.4912 |

**Script:** `C++/scripts/measure_brightness.py` — loads frame001.tif, applies same preprocessing as C++ (normalize + blur 1.5), measures mean pixel brightness inside each cell's circular boundary at its z-slice.

---

## B12. Config Changes — **ACTIVE**

**File:** `C++/config/config.yaml`

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `maxMajorRadius` | 40 | 35 | Largest initial cell ~31; 35 gives ~15% growth |
| `maxMinorRadius` | 35 | 30 | Tighter constraint matching initial sizes |
| `calibration_*` | (absent) | Added | Calibration zone for auto background_color |
| `sigmoid_k` | (absent) | 75.0 | Parsed but not used (sigmoid reverted) |

---

# Performance Summary

| Operation | Before (per iter) | After (per iter) | Savings |
|-----------|-------------------|-------------------|---------|
| Old image cost | 225-slice L2 norm | Cached lookup | ~50% of total time |
| Overlap penalty | O(n^2) x 2 + revert | O(n) x 2 | ~6x faster for 6 cells |
| End-of-frame PCA | 6 full 3D scans at end | 6 lazy scans amortized (C2) | Same cost, better timing |
| P(split) | Fixed 3% (B7) | Adaptive 3-50% via PCA (C3) | Elongated cells get more attempts |

# All Files Changed

| File | Changes |
|------|---------|
| `includes/ConfigTypes.hpp` | overlap_penalty_weight, calibration fields, sigmoid fields, YAML parsing |
| `includes/CellUniverse.hpp` | non-const loadFrame, removed previousElongations |
| `includes/Frame.hpp` | perturbCell signature, computeOverlapForCell, _currentCost, randomSplitCell |
| `includes/Spheroid.hpp` | brightness in SpheroidParams/Spheroid, printCellInfo, const-correctness |
| `src/CellUniverse.cpp` | unified optimize loop, blur from config, auto-calibrate background_color, simplified P(split) |
| `src/Frame.cpp` | perturbCell with overlap+caching, computeOverlapPenalty, computeOverlapForCell, randomSplitCell, cost caching in trySplitCell |
| `src/Spheroid.cpp` | brightness in constructor/getPerturbedCell/draw/getSplitCells/getCellParams |
| `src/CellFactory.cpp` | reads brightness from CSV column 8 |
| `src/main.cpp` | suppress TIFF warnings |
| `config/config.yaml` | overlap_penalty_weight, brightness params, calibration params, reduced max radii |
| `config/initial.csv` | 8-column format with measured brightness |
| `scripts/measure_brightness.py` | new script to measure cell brightness from TIFF |

---

# Part C: Revert to Unified Stochastic Loop + Lazy PCA Elongation Cache + Adaptive P(split)

## C1. Reverted Two-Phase Optimize Back to Unified Stochastic Loop — **ACTIVE** (two-phase was REVERTED)

**Problem:** Part B (B7) removed the end-of-frame PCA elongation scan and set P(split) to a fixed 3%. Before that, the optimize loop had been briefly restructured into Phase 1 (perturbation only) + Phase 2 (evaluate all cells for splits) + Phase 3 (post-split perturbation). The original CellUniverse paper specifies that P(split) should be stochastic per-iteration and can depend on cell parameters, not a separate sequential phase.

**File:** `C++/src/CellUniverse.cpp` — `optimize()`

**Before (two-phase approach):**

Phase 1 ran all `totalIterations` as perturbation-only. Phase 2 then evaluated every cell for splits independently. Phase 3 ran post-split perturbation iterations.

**After (unified stochastic loop, lines 230-301):**

Each of the `totalIterations` iterations picks a random cell, then with probability `pSplit` tries a split via `randomSplitCell()`, else tries a perturbation via `perturbCell()`. Accept if `costDiff < 0`, else revert.

```cpp
for (size_t i = 0; i < totalIterations; ++i) {
    if (frame.cells.empty()) break;

    // Pick random cell
    std::uniform_int_distribution<size_t> cellDist(0, frame.cells.size() - 1);
    size_t cellIdx = cellDist(gen);
    auto params = frame.cells[cellIdx].getCellParams();

    // ... compute pSplit from cached elongation (see C3) ...

    if (uniform01(gen) < pSplit) {
        // --- Try split ---
        splitAttempted++;
        auto result = frame.randomSplitCell(cellIdx, overlapWeight);
        double costDiff = result.first;
        auto callback = result.second;

        if (costDiff < 0) {
            callback(true);
            splitAccepted++;
            elongationCache.erase(params.name);
            // ... logging ...
        } else {
            callback(false);
        }
    } else {
        // --- Try perturbation ---
        auto result = frame.perturbCell(cellIdx, overlapWeight);
        double costDiff = result.first;
        auto callback = result.second;

        if (costDiff < 0) {
            callback(true);
            perturbAccepted++;
            residSum += costDiff;
            residCount++;
        } else {
            callback(false);
        }
    }
}
```

**Effect:** Restores the paper's stochastic perturbation/split decision per iteration. Splits can now occur at any point during optimization, not just at the end.

---

## C2. Added Lazy PCA Elongation Caching — **ACTIVE**

**Problem:** B7 removed the end-of-frame `computeElongationRatios()` call (which did 6 full 3D PCA volume scans). But to drive adaptive P(split) (C3), we need elongation values. Computing PCA every iteration is prohibitively expensive.

**Solution:** Each cell's PCA elongation is computed lazily on first encounter during the iteration loop and cached in a `std::map<std::string, float>` for the rest of the frame. Only 1 PCA scan per cell per frame total, amortized across all iterations.

**File:** `C++/src/CellUniverse.cpp` — `optimize()`, lines 218-246

**Added (cache declaration, line 222):**

```cpp
// Lazy PCA elongation cache — computed once per cell per frame.
// When a cell is first picked, we compute its PCA elongation and cache it.
// Higher elongation = cell looks bimodal = more likely to split.
// Only 1 PCA scan per cell (not per iteration), so cost is O(num_cells).
std::map<std::string, float> elongationCache;
```

**Added (cache lookup per iteration, lines 239-246):**

```cpp
// Get or compute PCA elongation (cached per cell per frame)
float elongation = 1.0f;
auto it = elongationCache.find(params.name);
if (it != elongationCache.end()) {
    elongation = it->second;
} else {
    elongation = frame.computeElongationForCell(cellIdx);
    elongationCache[params.name] = elongation;
}
```

**Effect:** Each cell pays for exactly 1 PCA scan the first time it is randomly selected. Subsequent picks of the same cell reuse the cached value. For 6 cells and 2100 iterations, this is 6 PCA scans total instead of 0 (B7's approach) or 2100 (per-iteration approach).

---

## C3. Adaptive P(split) Driven by Cached PCA Elongation — **ACTIVE**

**Problem:** B7 set P(split) to a fixed `config.prob.split` (3%) for all cells. This means spherical cells waste split attempts, and elongated cells (about to divide) don't get enough attempts.

**Solution:** P(split) scales with the cell's PCA elongation ratio using the formula:

```
P(split) = min(0.5, baseSplitProb + max(0, 1 - 1/elongation))
```

| Elongation | P(split) | Split attempts per 350 iters |
|-----------|---------|------------------------------|
| 1.0 (spherical) | 0.03 (base) | ~10 |
| 1.2 | 0.20 | ~70 |
| 1.5 | 0.36 | ~126 |
| 2.0 | 0.50 (cap) | ~175 |

Elongated cells get 10-15x more split attempts than spherical cells.

**File:** `C++/src/CellUniverse.cpp` — `optimize()`, lines 248-253

**Before (B7, fixed probability):**

```cpp
float pSplit = baseSplitProb; // fixed 0.03
```

**After (adaptive, lines 248-253):**

```cpp
// Adaptive P(split) driven by PCA elongation:
//   elongation=1.0 (spherical): P = base (0.03)
//   elongation=1.5: P = base + 0.33 = 0.36
//   elongation=2.0: P = base + 0.50 = 0.50 (cap)
// Capped at 0.5 so perturbation always dominates.
float pSplit = std::min(0.5f, baseSplitProb + std::max(0.0f, 1.0f - 1.0f / elongation));
```

**Effect:** Spherical cells (elongation ~1.0) spend 97% of iterations on perturbation. Elongated cells (elongation ~1.5+) get 33-50% split attempts. The 0.5 cap ensures perturbation always gets at least half the iterations even for very elongated cells.

---

## C4. Added Frame::computeElongationForCell(size_t cellIdx) — **ACTIVE**

**Problem:** `computeElongationRatios()` (existing) computes PCA elongation for ALL cells at once. The lazy cache (C2) needs to compute for a single cell on demand.

**File:** `C++/src/Frame.cpp` — lines 458-471

**New method:**

```cpp
float Frame::computeElongationForCell(size_t cellIdx) const
{
    if (cellIdx >= cells.size()) return 1.0f;

    std::vector<cv::Point3f> neighbors;
    for (size_t j = 0; j < cells.size(); ++j) {
        if (j != cellIdx) neighbors.push_back(cells[j].get_center());
    }

    auto [d1, d2, valid, elongation] = cells[cellIdx].getSplitCells(
        _realFrame, simulationConfig.z_scaling, neighbors);

    return valid ? elongation : 1.0f;
}
```

**File:** `C++/includes/Frame.hpp` — line 41

**Added declaration:**

```cpp
float computeElongationForCell(size_t cellIdx) const;
```

**Effect:** Wraps `getSplitCells()` for a single cell, collecting neighbor centers and returning just the elongation ratio. Returns 1.0 (spherical) if PCA fails (too few bright pixels). Used by the lazy cache in `optimize()`.

---

## C5. Cache Invalidation on Split Accept — **ACTIVE**

**Problem:** When a split is accepted, the parent cell is replaced by two daughters. The parent's cached elongation is stale. Daughters are new cells with no cache entry.

**File:** `C++/src/CellUniverse.cpp` — `optimize()`, line 266

**Added after split acceptance:**

```cpp
// Invalidate cache — parent is gone, daughters are new
elongationCache.erase(params.name);
```

**Effect:** The parent's cache entry is removed. Daughters will get fresh PCA scans on their first encounter (they have new names like `parentName + "0"` and `parentName + "1"`, so they won't have cache entries). This ensures the adaptive P(split) reflects the current cell geometry, not the pre-split parent.

---

# Part C Performance Summary

| Aspect | Part B (B7) | Part C |
|--------|-------------|--------|
| P(split) | Fixed 3% for all cells | 3%-50% adaptive based on PCA elongation |
| PCA scans per frame | 0 (removed) | 6 (one per cell, lazy) |
| Split attempts per frame (6 cells, 2100 iters) | ~63 (3% of 2100) | ~63 for spherical cells, up to ~630 for elongated cells |
| Optimize loop structure | Unified stochastic | Unified stochastic (same, matching paper) |

# Part C Files Changed

| File | Changes |
|------|---------|
| `C++/src/CellUniverse.cpp` | Reverted to unified stochastic loop, added elongationCache, adaptive P(split), cache invalidation on split |
| `C++/src/Frame.cpp` | Added `computeElongationForCell()` method |
| `C++/includes/Frame.hpp` | Added `computeElongationForCell()` declaration |

---

# Part D: Split Acceptance Threshold Fix

## D1. Require split_cost Threshold in randomSplitCell Acceptance — **ACTIVE**

**Problem:** `randomSplitCell` accepted any split with `costDiff < 0` (any improvement, even -0.01). This caused cascading false splits -- daughters at 0.794x parent radius look "elongated" to PCA, get high P(split), split again, and so on. By frame 5: 6 cells exploded to 153 cells, 120 at minimum radius, with names 27 generations deep.

**Root cause:** The unified stochastic loop (C1) used `costDiff < 0` for split acceptance, while the older `trySplitCell` path already required `costDiff < -split_cost`. The two code paths had inconsistent thresholds.

**File:** `C++/src/CellUniverse.cpp` — `optimize()`, line 262

**Before (line 262):**

```cpp
            if (costDiff < 0) {
```

**After (line 262):**

```cpp
            if (costDiff < -config.prob.split_cost) {
```

**Config value:** `split_cost: 20` (from `config.yaml`), so splits now require `costDiff < -20`.

**Log output updated (lines 267-271):**

The `[Split Accepted]` log line already includes the threshold for diagnostics:

```cpp
                std::cout << "[Split Accepted] " << params.name << " frame=" << displayFrame
                          << " iter=" << i << " diff=" << costDiff
                          << " threshold=" << -config.prob.split_cost
                          << " elongation=" << elongation
                          << " P(split)=" << pSplit << std::endl;
```

**Effect:** Only splits with substantial cost improvement (at least -20 L2 cost reduction) are accepted. This matches the threshold used by `trySplitCell` and prevents the cascading false-split chain where tiny cost improvements compound into runaway cell proliferation.

---

## D2. Switch from randomSplitCell to trySplitCell for Split Attempts — **ACTIVE**

**Problem:** After D1 added the `split_cost: 20` threshold, no splits happened at all. `randomSplitCell` has no burn-in -- it places raw PCA-guided daughters and checks cost immediately. Without refinement, the cost improvement is tiny (-0.01 to -5), never reaching the -20 threshold.

**File:** `C++/src/CellUniverse.cpp` — `optimize()`, line 258

**Before (line 258):**

```cpp
auto result = frame.randomSplitCell(cellIdx, overlapWeight);
```

**After (line 258):**

```cpp
auto result = frame.trySplitCell(cellIdx, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                 config.prob.split_elongation_threshold);
```

**Why trySplitCell works:** `trySplitCell` includes the full split evaluation pipeline:
1. PCA-guided daughter placement via `getSplitCells()`
2. Elongation threshold check (skip if ratio < 1.5, avoiding burn-in cost on spherical cells)
3. 500-iteration burn-in to refine daughter positions, radii, and orientations
4. Cost comparison after burn-in

With burn-in, legitimate splits produce -20 to -500 cost improvement, easily passing the `split_cost: 20` threshold. Without burn-in (`randomSplitCell`), raw unrefined daughters rarely improve cost by more than -5.

**Effect:** Restores functional split detection within the unified stochastic loop. The combination of D1 (threshold gate) and D2 (burn-in via `trySplitCell`) ensures that only well-refined, high-confidence splits are accepted.

---

## D3. Removed dead randomSplitCell — **ACTIVE**

Removed `randomSplitCell()` declaration from `Frame.hpp` and ~80-line implementation from `Frame.cpp`. Superseded by `trySplitCell()` in D2. No callers remain.

---

## D4. Skip splits on frame 1 — **ACTIVE**

**File:** `C++/src/CellUniverse.cpp` — `optimize()`

Added `bool allowSplits = (frameIndex > 0);` and set `pSplit = 0.0f` when `!allowSplits`. No cells can split in the initial frame — no time has elapsed for division.

---

## D5. Reset cached elongation after failed burn-in — **ACTIVE**

**File:** `C++/src/CellUniverse.cpp` — `optimize()`

After a failed split attempt (costDiff >= -split_cost), set `elongationCache[params.name] = 1.0f`. This drops the cell's P(split) from up to 50% back to the base 3% for the rest of the frame.

**Problem:** Cell `8cbdf86d` had elongation=8.2 (P=50%), ran ~15 burn-ins of 500 iterations each (7,500 wasted iterations) in frame 1 — best improvement was only -0.3 vs threshold -20. The cached elongation never changed so it kept retrying.

**Fix:** After one failed burn-in, elongation resets to 1.0. Max 1 burn-in per cell per frame for cells that won't split.

---

## D6. Log P(split) for every cell at start of each frame — **ACTIVE**

**File:** `C++/src/CellUniverse.cpp` — `optimize()`

Added a loop before the main iteration loop that computes and prints PCA elongation and P(split) for every cell. This also pre-seeds the elongation cache so no lazy PCA computation is needed during the iteration loop.

**Output format:**

```
[P(split)] frame 2
  e3d0... elongation=1.68 P(split)=0.43
  8cbd... elongation=8.20 P(split)=0.50
```

---

## D7. Blacklist cells after failed split attempt — **ACTIVE** (supersedes D5 cache reset)

**File:** `C++/src/CellUniverse.cpp` — `optimize()`

**Problem:** D5's cache reset (elongation -> 1.0) dropped P(split) to 3%, but at 3% the cell still got ~10 split attempts per frame. Each attempt called `trySplitCell` which recomputes PCA internally (ignoring the outer cache), gets the real elongation (e.g., 1.56), passes the 1.5 threshold, and runs 500-iter burn-in again. Cell `e3d0` ran ~5 burn-ins in frame 2, wasting 2,500 iterations.

**Fix:** Added `std::set<std::string> splitBlacklist`. After any failed split attempt (burn-in or elongation skip), the cell's name is added to the blacklist. The split decision check now requires `splitBlacklist.find(params.name) == splitBlacklist.end()`. Each cell gets at most 1 split attempt per frame.

**Effect:** Frame 2 goes from 65 split attempts to <=6 (one per cell). Saves thousands of wasted burn-in iterations.

---

## D8. Replaced all hard overlap constraints in trySplitCell with overlap penalty — **ACTIVE**

**Problem:** `trySplitCell()` had two hard overlap constraints that blocked splits for cells in the dense center (exactly the cells that SHOULD split):

1. **Pre-burn-in check (lines 362-400):** Daughters that overlapped ANY existing cell at 0.95x combined radius were instantly rejected — split never attempted. Dense-center cells `e907`, `1234`, `1f89` (the ones that should split at frames 3 and 8) were most affected.

2. **Burn-in check (lines 420-437):** During burn-in, any daughter perturbation that overlapped an existing cell at 0.8x combined radius was rejected. Acceptance rate dropped to 5-15% for dense cells, producing tiny improvements (-1 to -3) while isolated cells like `e3d0` got -9.5.

3. **Burn-in cost (line 442):** Used pure image L2 cost, not image + overlap penalty. Different objective than the main loop which uses `imageCost + overlapPenalty`.

**Result:** The cells biologically dividing at frames 3 and 8 (`e907`, `1234`, `1f89`) could never pass the `split_cost: 20` threshold because their burn-in was crippled by hard rejection. Meanwhile isolated cells that shouldn't split (`e3d0`) showed the best improvements.

**Fix:** Removed both hard overlap checks. Burn-in now uses the same overlap penalty as the main loop:

```cpp
// Before each perturbation: measure overlap for this daughter
double oldCellOverlap = computeOverlapForCell(dIdx, overlapWeight);

// Perturb
cells[dIdx] = cells[dIdx].getPerturbedCell();

// Measure new overlap
double newCellOverlap = computeOverlapForCell(dIdx, overlapWeight);

// Render and compare total cost (image + overlap)
double improvement = (trialImageCost + newCellOverlap) - (bestImageCost + oldCellOverlap);
if (improvement < 0) accept; else revert;
```

Final cost diff also includes full overlap penalty for fair comparison with the pre-split state.

**Files changed:**
- `C++/src/Frame.cpp` — `trySplitCell()`: removed pre-burn-in hard overlap (lines 362-400), removed burn-in hard overlap (lines 420-437), burn-in now uses `computeOverlapForCell()` + image cost

**All hard overlap constraints are now removed from the codebase.** Only `computeOverlapPenalty()` and `computeOverlapForCell()` (continuous penalty) remain.

---

# Part E: Code Review Fixes + Dead Code Cleanup (2026-03-28)

## E1. PerturbParams default initialization + thread_local RNG — **ACTIVE**

**Problem:** `PerturbParams::prob/mu/sigma` were uninitialized — garbage values if YAML node missing. `getPerturbOffset()` created new `std::random_device` + `std::mt19937` per call (~18,900 engine constructions/frame).

**File:** `C++/includes/ConfigTypes.hpp`

**Fix:** Added `= 0.0f` defaults. Made RNG `thread_local static`:
```cpp
float prob  = 0.0f;
float mu    = 0.0f;
float sigma = 0.0f;

[[nodiscard]] float getPerturbOffset() const {
    thread_local std::mt19937 gen{std::random_device{}()};
    // ...
}
```

## E2. BaseConfig raw pointer → unique_ptr — **ACTIVE**

**Problem:** `SpheroidConfig* cell` with raw `new`/`delete`, missing move operations (Rule of Five violation). Double `explodeConfig()` leaked memory.

**File:** `C++/includes/ConfigTypes.hpp`

**Fix:** Replaced with `std::unique_ptr<SpheroidConfig> cell`. Added explicit move operations via `= default`. Deep copy via `std::make_unique`. Added `#include <memory>`.

## E3. trySplitCell overlap weight from config — **ACTIVE**

**Problem:** `trySplitCell` hardcoded `overlapWeight = 1000.0f` while `perturbCell` used config value. Inconsistent cost comparison if user changed `overlap_penalty_weight`.

**File:** `C++/includes/Frame.hpp`, `C++/src/Frame.cpp`, `C++/src/CellUniverse.cpp`

**Fix:** Added `float overlapWeight = 1000.0f` parameter to `trySplitCell` signature. Removed hardcoded local. Caller in `optimize()` passes `config.prob.overlap_penalty_weight`.

## E4. Dead code in trySplitCell removed — **ACTIVE**

**File:** `C++/src/Frame.cpp`

Removed dead `oldTotalCost` computation (line 376) that was immediately overwritten on line 382.

## E5. Broken test fixed — **ACTIVE**

**File:** `C++/tests/spheroid_test.cc`

Test `DrawColorsCenterPixelAndLeavesFarPixelUnchanged` expected `cell_color` (0.9f) but `draw()` now uses `_brightness` (default 0.5f). Updated test to expect 0.5f.

## E6. Dead code removal (Steps 1-8) — **ACTIVE**

### Removed dead config fields (ConfigTypes.hpp)
- `cell_color` from SimulationConfig — replaced by per-cell `_brightness`
- `padding` from SimulationConfig — never referenced
- `perturbation` from ProbabilityConfig — never read by code

### Removed dead classes (ConfigTypes.hpp)
- `CellConfig` abstract base class — vestigial, SpheroidConfig no longer inherits it
- `SphereConfig` class — Sphere is deprecated
- `SpheroidConfig::explodeConfig` no longer marked `override`

### Removed dead methods (Spheroid.hpp/cpp)
- `getParameterizedCell()` — never called, replaced by `getPerturbedCell()`
- `performPCA()` — never called, PCA done inline in `getSplitCells()`

### Removed dead member variable (Frame.hpp/cpp)
- `_realFrameCopy` — deep-cloned 225 z-slices in constructor but never read anywhere. Wasted memory every frame.

### Removed dead type aliases (types.hpp)
- `ParamImageMap` — never used
- `ParamValMap` — never used

### Removed dead variables and commented code
- `unsigned int x` counter in `Frame::generateSynthFrame()` — declared, incremented, never read
- Commented-out interpolation code block in `Frame::generateSynthFrame()`
- Commented-out duplicate `length()` definition in Frame.cpp
- Commented-out assert in CellUniverse.cpp
- Commented-out debug logging in Spheroid.cpp

### Replaced std::endl with '\n'
- All 75 occurrences across all source files replaced. Eliminates unnecessary buffer flush in tight optimization loops.

### Added config fields (ConfigTypes.hpp + config.yaml)
- `split_burn_in_iterations: 500` — was hardcoded in Frame.cpp
- `max_split_probability: 0.5` — was hardcoded in CellUniverse.cpp

### Config YAML cleanup (config.yaml)
- Removed `cell_color: 0.6`, `padding: 0`, `perturbation: 0.97`
- Added `split_burn_in_iterations: 500`, `max_split_probability: 0.5`

### Test fixes
- `spheroid_test.cc`: Updated to expect `_brightness` (0.5f) instead of `cell_color`
- `spheroid_rotation_test.cc`: Updated to use `kTestBrightness` constant, removed `cell_color` references
- `frame_test.cc`: Removed `cell_color` and `padding` references

### printConfig made const
- `SimulationConfig::printConfig()`, `ProbabilityConfig::printConfig()`, `BaseConfig::printConfig()` all marked `const`

---

## E7. Blur synthetic image to match real image — **ACTIVE**

**Problem:** Real image is preprocessed with Gaussian blur (sigma=1.5), giving soft cell boundaries. Synthetic image has hard analytical edges. This mismatch causes cells to inflate — covering the real image's blurred halo around cells reduces L2 cost, so the optimizer always prefers bigger cells.

**File:** `C++/src/Frame.cpp` — `generateSynthFrame()` and `generateSynthFrameFast()`

**Fix:** Apply same Gaussian blur to synthetic slices after drawing cells:

```cpp
if (simulationConfig.blur_sigma > 0) {
    cv::GaussianBlur(synthImage, synthImage, cv::Size(0, 0), simulationConfig.blur_sigma);
}
```

Added in both `generateSynthFrame()` (full render) and `generateSynthFrameFast()` (fast partial render).

**Effect:** Synthetic cell boundaries now blur identically to real image. A too-big cell's blurred edge extends into background, increasing L2 error. The optimizer gets a clear gradient signal at the correct cell boundary instead of a bias toward inflation.

## E8. PCA threshold: mean → mean + 0.5*stddev — **ACTIVE**

**File:** `C++/src/Spheroid.cpp` — `getSplitCells()`

Computes stddev of brightness inside cell boundary. Threshold changed from `meanBrightness` to `meanBrightness + 0.5 * stddevBrightness`. Filters out near-background pixels for cleaner PCA signal.

## E9. Config tuning — **ACTIVE**

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `split_cost` | 20 | 3 | Burns-in produce -3 to -10 improvements |
| `x/y sigma` | 10 | 5 | Less aggressive position jumps |
| `brightness prob` | 0.15 | 0.2 | More frequent brightness exploration |
| `brightness sigma` | 0.03 | 0.05 | Larger steps to find correct brightness |
| `overlap_penalty_weight` | 1000 | 5000 | Stronger anti-inflation pressure |
| `maxMajorRadius` | 35 | 32 | Tighter cap (largest initial cell is 31) |
| `maxMinorRadius` | 30 | 25 | Forces oblate shape |
| `sigmoid_k/center/offset` | present | **removed** | Dead — sigmoid reverted, never used |
| `background_color` | 0.2 | 0 | Just a placeholder, auto-calibrated |

### std::endl restored for progress lines
Key progress/status lines (`[Optimize]`, `[Split Accepted]`, `[Calibration]`, etc.) restored to `std::endl` for immediate visibility through pipes. Inner-loop lines keep `'\n'` for performance.

---

## E10. Minimal preprocessing experiment — **ACTIVE**

**Philosophy:** Per project advisor — minimal data preprocessing, minimal constraints. The algorithm's power comes from randomness + cost function (per the published paper). Heavy preprocessing (blur, sigmoid) distorts the boundary signal. Hard constraints (max radius, volume cap) prevent the optimizer from exploring valid configurations. Let Monte Carlo do the work.

**Hypothesis:** Without blur, real image has sharp cell boundaries. Synthetic image also has sharp boundaries. L2 cost becomes very sensitive to cell size — a cell 2 pixels too big gets penalized immediately because those pixels are at cell brightness while real image drops to background sharply. No blur halo to hide the mismatch. Cells should find correct size without needing tight max radius bounds.

### Config changes

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `blur_sigma` | 5.0 | **0** | No blur — sharp boundaries guide optimizer |
| `maxMajorRadius` | 32 | **60** | Remove tight cap — let optimizer find size |
| `maxMinorRadius` | 25 | **45** | Same |
| `minMajorRadius` | 10 | **5** | Allow very small cells |
| `minMinorRadius` | 5 | **3** | Same |
| `minBrightness` | 0.15 | **0.05** | Wider brightness range |
| `overlap_penalty_weight` | 5000 | **500** | Reduced — cost function should handle overlap |

### Code changes

**CellUniverse.cpp: processImage()** — added `blur_sigma > 0` guard so blur is skipped when sigma=0:
```cpp
if (config.simulation.blur_sigma > 0) {
    cv::GaussianBlur(processedImage, processedImage, cv::Size(0, 0), config.simulation.blur_sigma);
}
```

**Frame.cpp: generateSynthFrame() and generateSynthFrameFast()** — already had `blur_sigma > 0` guards from E7.

### Expected outcome
- Cells find correct size from sharp L2 boundaries (no inflation)
- Per-cell brightness differentiates cells naturally
- Splits only accepted when they genuinely improve the sharp-boundary fit
- If this works, it validates the minimal-constraint philosophy and simplifies the entire system

---

## E11. L1 cost function replaces L2 — **ACTIVE**

**Problem:** L2 cost (Euclidean norm) allows a brightness-size co-optimization trap. The optimizer simultaneously grows cells AND lowers brightness toward background. Covering background pixels at low brightness costs very little under L2 (small² = tiny). Cells inflate while brightness converges to ~0.45 regardless of true cell brightness.

**Root cause:** L2 = sqrt(sum((a-b)²)) overweights large errors. The optimizer focuses on covering bright cell centers (big errors if missed) while ignoring boundary accuracy (small errors either way). This biases toward bigger cells.

**Fix:** Switch from L2 to L1 norm: `sum(|a-b|)`. L1 treats all pixel errors proportionally — covering a background pixel costs the same per-pixel as missing a cell pixel at equal brightness difference. No quadratic bias toward large errors.

**File:** `C++/src/Frame.cpp` line 79

**Before:**
```cpp
totalCost += cv::norm(_realFrame[i], synthFrame[i], cv::NORM_L2);
```

**After:**
```cpp
totalCost += cv::norm(_realFrame[i], synthFrame[i], cv::NORM_L1);
```

**Config:** `blur_sigma` set to 1.0 (minimal noise reduction). Max radii remain at 60/45 (no tight constraints).

**Note:** L1 produces much larger cost values than L2. The `split_cost: 3` threshold was tuned for L2 scale (~10,000 total). L1 scale will be different — threshold may need adjusting based on results.

## E12. Binary cost function (paper's approach) — **ACTIVE** (supersedes E11 L1)

**Problem:** Both L2 and L1 suffer from the brightness-size co-optimization trap. The optimizer lowers brightness toward background while growing cells. All cells converge to ~0.46 brightness regardless of true cell brightness. Neither L2 nor L1 fixes this because both use continuous pixel values where brightness gaming is possible.

**Solution:** Binary cost function — threshold both real and synthetic images at the auto-calibrated `background_color`, then count mismatched pixels (XOR). This is exactly the published paper's approach.

**File:** `C++/src/Frame.cpp` — `calculateCost()`

```cpp
const float threshold = simulationConfig.background_color;
for (size_t i = 0; i < _realFrame.size(); ++i) {
    cv::Mat realBin, synthBin;
    cv::threshold(_realFrame[i], realBin, threshold, 1.0f, cv::THRESH_BINARY);
    cv::threshold(synthFrame[i], synthBin, threshold, 1.0f, cv::THRESH_BINARY);
    totalCost += cv::norm(realBin, synthBin, cv::NORM_L1);
}
```

**Why this works:**
- Covering a background pixel costs exactly 1 (synth=1, real=0)
- Missing a cell pixel costs exactly 1 (synth=0, real=1)
- Perfectly symmetric — no brightness gaming possible
- Cell brightness is irrelevant — only position and size affect cost
- A cell at the right size produces the right binary mask

**Threshold:** Uses `simulationConfig.background_color` which is auto-calibrated from the real image's cell-free calibration zone. Pixels above this = cell tissue, below = background.

**Note:** `split_cost` needs re-tuning for binary cost scale. Binary cost counts mismatched pixels — total will be in the tens of thousands range.

---

# Part D: Sigmoid-Pipeline Invariants Cleanup (2026-04-08)

**Branch:** `yp_yd_merge_04072026`

Scope: 5 former `SimulationConfig` fields that were dead or sigmoid-pipeline invariants masquerading as config have been removed from YAML and the struct. See `docs/plans/2026-04-07-yp-yd-merge-review-and-next-steps.md` for the original review, and `docs/conversation_archive_2026-04-05.md` / `2026-04-06.md` for the history of the brightness + sigmoid rework these fields outlived.

## D1. Delete `sigmoid_center_offset` (already dead) — **ACTIVE**

**Problem:** Field was parsed in `ConfigTypes.hpp:24,49` and exposed in `config.yaml:87`, but had **zero reads** anywhere in `C++/src/`. The active sigmoid calibration uses `sigmoid_center_percentile` instead. Users reading YAML would falsely assume tuning this field would change behavior.

**Files:** `C++/includes/ConfigTypes.hpp`, `C++/config/config.yaml`, `C++/scripts/config.yaml`

**Before** (`ConfigTypes.hpp`):
```cpp
float sigmoid_center = 0.445f;
float sigmoid_center_percentile = 0.4f;
float sigmoid_center_offset = 0.047f;
```
...plus a matching `if (node["sigmoid_center_offset"]) ...` parse line and YAML entry.

**After:** All three locations removed. YAML no longer lists the field.

**Effect:** Removes a confusing inert knob. No behavior change.

## D2. Delete `cell_color` — **ACTIVE**

**Problem:** As of 2026-04-05, `draw()` uses per-cell `_brightness` rather than `simulationConfig.cell_color`. The only remaining read was `CellFactory.cpp:5` which used it as the frame-1 seed for `initialBrightness`. This is a sigmoid-pipeline invariant (post-sigmoid cells ≈ 1.0), not a tunable knob.

**Files:** `C++/includes/ConfigTypes.hpp`, `C++/src/CellFactory.cpp`, `C++/config/config.yaml`, `C++/scripts/config.yaml`

**Before** (`CellFactory.cpp:5`):
```cpp
initialBrightness = config.simulation.cell_color;
```

**After:**
```cpp
// Frame-1 seed for per-cell _brightness. Post-sigmoid cells are ~1.0.
// After frame 1, the per-cell EMA update (measureMeanBrightness *
// brightnessMeanAmplification blended via brightnessUpdateBlend) takes over.
initialBrightness = 1.0f;
```

`SimulationConfig::cell_color` declaration, constructor initializer, and `explodeConfig` parse line removed. YAML entry removed.

**Effect:** No behavior change. Removes the illusion that tuning `cell_color` affects rendering after frame 1 — it doesn't, and hasn't since 2026-04-05.

## D3. Move `background_color` from `SimulationConfig` to `Frame::_backgroundValue` — **ACTIVE**

**Problem:** `background_color` was not truly config — it was runtime state. The adaptive-background path at `CellUniverse.cpp:565` mutates it per-frame via `frame.setBackgroundColor(updatedBackground)`, and the value is then passed to `Spheroid::getSplitCells` as the PCA noise floor. Storing it inside `SimulationConfig` as if it were a fixed tunable obscured the runtime mutation.

**Files:** `C++/includes/Frame.hpp`, `C++/src/Frame.cpp`, `C++/src/CellUniverse.cpp`, `C++/includes/ConfigTypes.hpp`, `C++/config/config.yaml`, `C++/scripts/config.yaml`

**Before** (`Frame.hpp:60`):
```cpp
void setBackgroundColor(float backgroundColor) { simulationConfig.background_color = backgroundColor; }
```

**After** (`Frame.hpp`):
```cpp
void setBackgroundColor(float backgroundColor) { _backgroundValue = backgroundColor; }
float getBackgroundValue() const { return _backgroundValue; }
...
private:
    ...
    // Runtime-mutable synth frame background and PCA noise floor. Starts at 0.0 (post-sigmoid
    // background invariant). Updated per-frame by the adaptive background path in
    // CellUniverse::optimize via setBackgroundColor().
    float _backgroundValue = 0.0f;
```

All read sites (`Frame.cpp:279, 341, 508, 524, 564`) updated from `simulationConfig.background_color` to `_backgroundValue`. The `CellUniverse.cpp:246, 282` fallbacks (inside `estimateAdaptiveBackgroundFromFrame`) updated to return literal `0.0f` with a comment.

`SimulationConfig::background_color` declaration, constructor initializer, `explodeConfig` parse line, and `printConfig` line all removed. YAML entry removed.

**Effect:** No behavior change. Preserves the PCA noise floor ↔ adaptive-bg coupling (the parameter on `Spheroid::getSplitCells` stays). Makes the runtime-mutable nature of the value explicit in its location (Frame, not config).

## D4. Delete `sigmoid_center` (defensive default, always overwritten) — **ACTIVE**

**Problem:** `sigmoid_center` was read exactly once at `CellUniverse.cpp:422` as `float sigmoidCenter = config.simulation.sigmoid_center; // default from config`, then immediately overwritten at line 437 by the percentile calibration in every realistic code path (unless `processedZSlices.empty()` or `calW<=0||calH<=0`, both defensive-programming guards).

**Files:** `C++/includes/ConfigTypes.hpp`, `C++/src/CellUniverse.cpp`, `C++/config/config.yaml`, `C++/scripts/config.yaml`

**Before** (`CellUniverse.cpp:422`):
```cpp
float sigmoidCenter = config.simulation.sigmoid_center; // default from config
```

**After:**
```cpp
// Defensive fallback; overwritten in every realistic path by the percentile calibration below.
float sigmoidCenter = 0.445f;
```

`SimulationConfig::sigmoid_center` declaration and parse line removed. YAML entry removed.

**Effect:** No behavior change. The defensive guard still works — it just uses a literal instead of reading an inert config field.

## D5. Delete `calibration_z` (zero reads) — **ACTIVE**

**Problem:** Field was declared at `ConfigTypes.hpp:31`, parsed at line 55, exposed in both YAML files, but had **zero reads** anywhere in source. The sigmoid calibration uses only `calibration_x/y/width/height`.

**Files:** `C++/includes/ConfigTypes.hpp`, `C++/config/config.yaml`, `C++/scripts/config.yaml`

**Before** (`ConfigTypes.hpp`):
```cpp
int calibration_z = 0;
...
if (node["calibration_z"]) calibration_z = node["calibration_z"].as<int>();
```

**After:** Both lines removed. YAML entry removed.

**Effect:** Removes a pure dead field. No behavior change.

## Summary

5 fields removed. Net diff across the pass:
- `ConfigTypes.hpp`: shrinks `SimulationConfig` by 5 declarations, 5 parse lines, 1 printConfig line, 2 constructor initializers
- `Frame.hpp`: gains `_backgroundValue` private member + `getBackgroundValue()` accessor; `setBackgroundColor()` retargeted
- `Frame.cpp`: 5 read sites swapped from `simulationConfig.background_color` to `_backgroundValue`
- `CellUniverse.cpp`: 2 fallbacks use `0.0f` literal, 1 `sigmoidCenter` default uses `0.445f` literal
- `CellFactory.cpp`: `initialBrightness = 1.0f` literal with a comment explaining the frame-1 seed semantics
- `config/config.yaml` + `scripts/config.yaml`: 4 entries deleted per file, header comment added explaining why
- Rules docs (`config.md`, `gotchas.md`) + `details.md` updated to reflect the new state

**No behavior change at runtime.** This is a pure "stop pretending runtime state is config and stop parsing dead fields" refactor.

---

# Part E: Z-Clamp + Tighter Split Gates (2026-04-08)

**Branch:** `yp_yd_merge_04072026`
**Motivation:** Run `output_jihang_20260408_161444` exposed two failure modes:
1. A cell drifted to `z=266.777` (42 units past the top of the z-stack, which has max valid z=224). The cost function has no z-boundary penalty, so cells escaping the image volume reduce L2 cost (they stop contributing any pixels) and get rewarded by the optimizer. Verified at `[FrameState Before] frame 4` line 492 with `inside_count=0` at split attempt time.
2. A false split was accepted in frame 3 on `8cbdf86d308d4599936e7fdbc23375f5`: `prevElong=1.05` (essentially spherical), at `z=221` (near edge), cost diff = -26.99 (just past the -20 threshold), `burn_in_accepted=15/1000`, daughters stacked 8.59 px apart at the same z. The split was invisible in the output images because both daughters rendered on top of each other at the same slice.

## E1. Z-clamp in `Spheroid` constructor — **ACTIVE**

**Problem:** `perturbCell` can propose `z` offsets of ~8 per iteration. With no boundary check, a cell can drift off the z-stack. Outside `[0, z_slices-1]`, the cell's volume doesn't intersect any real image slice → zero pixel contribution → lower L2 cost → optimizer rewards the drift.

**Files:** `C++/includes/ConfigTypes.hpp`, `C++/src/Spheroid.cpp`, `C++/src/CellUniverse.cpp`

**Change 1** — add `maxZ` to `SpheroidConfig` (static default, runtime-updated):

```cpp
// ConfigTypes.hpp (SpheroidConfig)
// Maximum valid z position (interpolated z-space). Used to clamp Spheroid
// center z in the constructor, preventing cells from drifting off the z-stack.
// Default 224 = (z_slices=225) - 1. Runtime-updated by CellUniverse::loadFrame
// to the actual interpolated slice count - 1. Not parsed from YAML.
float maxZ{224.0f};
```

**Change 2** — add z clamp to `Spheroid::Spheroid` constructor alongside the existing radius clamps (matching the pattern):

```cpp
// Spheroid.cpp — after the major/minor radius clamps
_position.z = std::fmax(_position.z, 0.0f);
_position.z = std::fmin(_position.z, cellConfig.maxZ);
```

**Change 3** — update `Spheroid::cellConfig.maxZ` in `CellUniverse::CellUniverse` after `loadFrame` finishes interpolation (which is when we know the actual slice count):

```cpp
// CellUniverse.cpp — in the constructor loop over imagePaths
config.simulation.z_slices = real_frame.size();
Spheroid::cellConfig.maxZ = static_cast<float>(real_frame.size()) - 1.0f;
```

**Effect:** Cells can no longer exist outside `[0, z_slices-1]` in interpolated z-space. Any perturbation that would push the center past the boundary is silently clamped at construction. This fixes the `z=266` escape seen in the 16:14 run. Also makes `getPerturbedCell()` behave correctly — it constructs a new `Spheroid` via `SpheroidParams`, which passes through the constructor clamp.

## E2. Raise `split_cost` from 20 → 80 — **ACTIVE**

**Problem:** Legit splits in the 16:14 run had `diff = -305` and `diff = -609`. The false split had `diff = -26.99`. Any threshold between -50 and -100 cleanly separates them.

**Files:** `C++/config/config.yaml`, `C++/scripts/config.yaml`

```yaml
prob:
  split_cost: 80   # was 20
```

**Effect:** Marginal cost improvements no longer qualify as splits. Legit splits are unaffected (they clear this threshold by 3-7x).

## E3. Raise `split_elongation_threshold` from 1.1 → 1.3 — **ACTIVE**

**Problem:** The false-split cell had `prevElong=1.05022` — essentially spherical. At `threshold=1.1`, even 1.05 would still reach burn-in if the base split probability fired. At `threshold=1.3`, cells must have a real shape signal to attempt splits.

**Files:** `C++/config/config.yaml`, `C++/scripts/config.yaml`

```yaml
prob:
  split_elongation_threshold: 1.3   # was 1.1
```

**Effect:** Cells with aspect ratio < 1.3 no longer attempt splits. Combined with E2, marginal borderline-spherical cells are filtered at two stages.

## E4. New `split_min_inside_count` gate — **ACTIVE**

**Problem:** The false-split cell had `inside_count=32134` (total bright voxels inside its ellipsoid volume). Legit splits had `inside_count` from 113k to 226k. Small cells — either naturally tiny or with clipped bounding boxes at the z-boundary — don't give reliable PCA and produce degenerate daughters.

**Files:** `C++/includes/Spheroid.hpp`, `C++/src/Spheroid.cpp`, `C++/includes/ConfigTypes.hpp`, `C++/includes/Frame.hpp`, `C++/src/Frame.cpp`, `C++/src/CellUniverse.cpp`, both YAML files.

**Change 1** — Add `insideCount` to `SplitDiagnostics`:

```cpp
// Spheroid.hpp
struct SplitDiagnostics
{
    // ... existing fields ...
    // Total bright voxels inside the cell's ellipsoid volume used for split PCA.
    int insideCount = 0;
};
```

**Change 2** — populate it in `Spheroid::getSplitCells`:

```cpp
// Spheroid.cpp — after the existing diagnostics field assignments
diagnostics.insideCount = candidateVoxelCount;
```

**Change 3** — new `ProbabilityConfig` field:

```cpp
// ConfigTypes.hpp
int split_min_inside_count = 50000;
// + explodeConfig parse line
```

**Change 4** — new `Frame::trySplitCell` parameter:

```cpp
// Frame.hpp signature (last param)
int splitMinInsideCount = 50000
```

**Change 5** — gate in `Frame::trySplitCell`, immediately after the `valid == false` check:

```cpp
if (splitMinInsideCount > 0 && splitDiagnostics.insideCount < splitMinInsideCount)
{
    std::cout << "[Split Skip] " << oldCell.getName()
              << " reason=too_small_for_split"
              << " inside_count=" << splitDiagnostics.insideCount
              << " threshold=" << splitMinInsideCount
              << std::endl;
    return {0.0, [](bool accept) {}};
}
```

**Change 6** — `CellUniverse::optimize` passes the config field through:

```cpp
// last arg to the trySplitCell call
config.prob.split_min_inside_count
```

**Change 7** — YAML:

```yaml
prob:
  split_min_inside_count: 50000
```

**Effect:** Cells below the threshold are rejected pre-burn-in with a `reason=too_small_for_split` log line, before any expensive daughter optimization. Catches boundary cells and natural-small cells.

## Summary

| Change | Type | Immediate effect |
|---|---|---|
| E1 (z-clamp) | Source + runtime state | Cells physically cannot exist outside `[0, z_slices-1]`. Fixes the `z=266` escape. |
| E2 (`split_cost: 20→80`) | Config only | Marginal splits rejected. Legit splits unaffected. |
| E3 (`split_elongation_threshold: 1.1→1.3`) | Config only | Near-spherical cells don't attempt splits. |
| E4 (`split_min_inside_count: 50000`) | Source + config | Tiny/boundary cells rejected pre-burn-in with a clear log line. |

**Expected runtime change:** the frame-3 `8cbdf86d` false split and the frame-4 daughter re-split should both be filtered. Legit splits (`12345679...`, `e9077677...`) should still land. The `z=266` escape on the `...f50` daughter is prevented by construction.

**Tuning knobs exposed:** `split_cost`, `split_elongation_threshold`, `split_min_inside_count` are all live config fields — no code changes needed for future adjustment.

## E5. Master switch for flat-cell rotation refine grid search — **ACTIVE**

**Problem:** `CellUniverse::optimize` runs a nested 3D rotation grid search (x/y/z offsets, multiple passes) over every flat cell every frame at lines 908-1008. For each candidate rotation it calls `Spheroid::measureBrightnessStats`, which scans the entire cell volume. With multiple passes and a fine angle step, this can be hundreds of brightness scans per flat cell per frame — a significant runtime cost. The per-cell brightness EMA update already handles brightness tracking, so the rotation refine is arguably redundant.

**Files:** `C++/includes/ConfigTypes.hpp`, `C++/src/CellUniverse.cpp`, `C++/config/config.yaml`, `C++/scripts/config.yaml`

**Change 1** — new `SpheroidConfig` field with YAML parsing:

```cpp
// ConfigTypes.hpp
bool flatCellRotationRefineEnabled{true};  // default true for backward compat

// explodeConfig
if (node["flatCellRotationRefineEnabled"]) {
    flatCellRotationRefineEnabled = node["flatCellRotationRefineEnabled"].as<bool>();
}
```

**Change 2** — gate the entire block in `CellUniverse::optimize`:

```cpp
// CellUniverse.cpp
if (config.cell && config.cell->flatCellRotationRefineEnabled) {
    // ... existing 100-line block unchanged ...
}
```

**Change 3** — both YAML files default the switch to `false`:

```yaml
flatCellRotationRefineEnabled: false
```

**Effect:** When disabled (the new default), the entire rotation grid search is skipped — no brightness scans, no candidate Spheroid constructions, no `[Flat Rotation Refine]` log lines. The other `flatCellRotationRefine*` knobs are preserved but ignored. When re-enabled via YAML, old behavior is restored with no code change.
