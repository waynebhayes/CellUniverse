# Dead Code Removal & Code Cleanup Plan

**Date:** 2026-03-28
**Branch:** `jl-restructure-cpp-03272026`
**Status:** Plan only — do not implement until approved

---

## Part 1: Dead Code Removal

### 1.1 Dead Config Fields (ConfigTypes.hpp)

| Field | Location | Why Dead | Action |
|-------|----------|----------|--------|
| `cell_color` | SimulationConfig | Replaced by per-cell `_brightness` — no code reads it | Remove field, parsing, print |
| `padding` | SimulationConfig | Never referenced by any code | Remove field, parsing, print |
| `perturbation` | ProbabilityConfig | Never read — P(split) is computed from PCA elongation | Remove field, parsing, print |
| `SphereConfig` class | Lines 139-162 | Sphere is deprecated, never instantiated | Remove entire class |
| `CellConfig` abstract base | Lines 131-137 | Only `SphereConfig` inherited it; `SpheroidConfig` can drop it | Remove class, change SpheroidConfig to not inherit |

### 1.2 Dead Type Aliases (types.hpp)

| Alias | Why Dead | Action |
|-------|----------|--------|
| `ParamImageMap` | Never used anywhere | Remove |
| `ParamValMap` | Never used anywhere | Remove |

### 1.3 Dead Methods

| Method | File | Why Dead | Action |
|--------|------|----------|--------|
| `Spheroid::getParameterizedCell()` | Spheroid.cpp:278-312 | Never called — replaced by `getPerturbedCell()` | Remove from .hpp and .cpp |
| `Spheroid::performPCA()` | Spheroid.cpp:655-691 | Never called — PCA is done inline in `getSplitCells()` | Remove from .hpp and .cpp |
| `Frame::computeOverlapPenalty()` (full O(n²)) | Frame.cpp | Check if still called — `computeOverlapForCell()` replaced it for perturbation | Remove if no callers remain |

### 1.4 Dead Member Variables

| Variable | File | Why Dead | Action |
|----------|------|----------|--------|
| `Frame::_realFrameCopy` | Frame.hpp:53 | Deep-cloned in constructor but **never read** anywhere | Remove declaration + constructor clone logic (~6 lines) |

### 1.5 Commented-Out Code

| Location | What | Action |
|----------|------|--------|
| Frame.hpp:31 | `// DataFrame getCellsAsParams();` | Remove |
| Frame.cpp:65-73 | Commented interpolation logic | Remove |
| Frame.cpp:199-202 | Duplicate `length()` definition | Remove |
| CellUniverse.cpp:68 | `// assert(numTiffSlices == 33);` | Remove |
| CellUniverse.cpp:49,120,182 | Commented debug logging | Remove |
| Spheroid.cpp:174-178,197-201 | Commented debug logging | Remove |

### 1.6 Unused Variables

| Variable | File:Line | Action |
|----------|-----------|--------|
| `x` (counter) | Frame.cpp:54 in `generateSynthFrame()` | Remove |
| `simConfig` | CellUniverse.cpp:48 in `processImage()` | Remove, use `config.simulation` directly |

---

## Part 2: Code Cleanup — Function Decomposition

### 2.1 `getSplitCells()` — 339 lines → 5 functions

**Current:** One 339-line monolith doing search region setup, bright pixel collection, centroid refinement, PCA, daughter placement.

**Proposed decomposition:**

```
getSplitCells()                    → ~40 lines (orchestrator)
├── prepareSplitSearchRegion()     → ~30 lines (bounding box, maxR, pcaCenter)
├── collectBrightPixels()          → ~60 lines (threshold, neighbor exclusion, pixel scan)
├── refinePCACenter()              → ~40 lines (centroid drift check, re-collection)
├── computeSplitAxis()             → ~50 lines (PCA normalization, eigenvalues, axis)
└── placeDaughterCells()           → ~40 lines (centroid projection, sizing, SpheroidParams)
```

Shared helper:
```cpp
bool isCloserToNeighbor(float x, float y, float z, cv::Point3f center,
                        const std::vector<cv::Point3f>& neighbors);
```
Eliminates the duplicated neighbor distance logic (currently ~25 lines x 2).

### 2.2 `optimize()` — 144 lines → 3 functions

**Current:** One loop combining RNG setup, PCA caching, split/perturb decision, progress logging.

**Proposed decomposition:**

```
optimize()                         → ~50 lines (loop + dispatch)
├── precomputeElongations()        → ~30 lines (PCA cache for all cells, logging)
└── attemptCellAction()            → ~40 lines (split vs perturb decision + execution)
```

### 2.3 `trySplitCell()` — 132 lines → 2 functions

**Current:** PCA retrieval + validation + burn-in + cost comparison all in one.

**Proposed decomposition:**

```
trySplitCell()                     → ~40 lines (orchestrator + callback)
└── runBurnIn()                    → ~40 lines (the 500-iteration loop)
```

---

## Part 3: Code Cleanup — Naming & Style

### 3.1 Cryptic Variable Names

| Current | Proposed | File | Reason |
|---------|----------|------|--------|
| `a`, `b`, `c` | Keep as internal, document | Spheroid.hpp | Standard ellipsoid notation (a=b=major, c=minor) — add comment |

### 3.2 Magic Numbers → Named Constants or Config

| Magic Number | Location | Proposed |
|-------------|----------|----------|
| `3.0f` (search radius multiplier) | Spheroid.cpp:329 | Config: `split_search_radius_multiplier` or const |
| `25.0f` (drift threshold squared) | Spheroid.cpp:457 | `const float PCA_DRIFT_THRESHOLD_SQ = 25.0f;` |
| `500` (burn-in iterations) | Frame.cpp:391 | Config: `split_burn_in_iterations` |
| `1000.0f` (overlap weight in burn-in) | Frame.cpp:370 | Pass from config (already done in main loop, not in burn-in) |
| `0.5f` (max P(split) cap) | CellUniverse.cpp | Config: `max_split_probability` |
| `0.794` (cbrt(0.5)) | Spheroid.cpp | `const float HALF_VOLUME_SCALE = std::cbrt(0.5f);` |

### 3.3 Large Parameter Lists → Structs

```cpp
// Replace 5 pre-opt parameters with:
struct PreOptState {
    float majorR = 0.0f;
    float minorR = 0.0f;
    float x = 0.0f, y = 0.0f, z = 0.0f;
};

// getSplitCells signature: 8 params → 4
getSplitCells(const ImageStack& image, float z_scaling,
              const std::vector<cv::Point3f>& neighbors,
              const PreOptState& preOpt = {});

// trySplitCell signature: 7 params → 3
trySplitCell(size_t index, const PreOptState& preOpt = {},
             float elongationThreshold = 1.5f);
```

---

## Part 4: Config Cleanup

### 4.1 Remove Dead YAML Fields

These fields are parsed by ConfigTypes.hpp but never used on the current branch:

```yaml
# REMOVE from config.yaml:
cell_color: 0.6        # replaced by per-cell brightness
padding: 0              # never used
perturbation: 0.97      # never read by code
```

### 4.2 Move Hardcoded Values to Config

```yaml
# ADD to config.yaml prob section:
split_burn_in_iterations: 500
max_split_probability: 0.5
split_search_radius_multiplier: 3.0
```

---

## Execution Order

| Step | What | Risk | Dependencies |
|------|------|------|-------------|
| 1 | Remove dead variables + commented code | None | — |
| 2 | Remove dead config fields + YAML cleanup | Low | Step 1 |
| 3 | Remove dead methods + classes | Low | Step 2 |
| 4 | Extract magic numbers to config | Low | Step 2 |
| 5 | Decompose `getSplitCells()` | **Medium** | Step 3 |
| 6 | Decompose `optimize()` | **Medium** | Step 3 |
| 7 | Decompose `trySplitCell()` | **Medium** | Step 3 |
| 8 | Replace parameter lists with structs | **Medium** | Steps 5-7 |

**Build verification required after each step.** User must rebuild to confirm compilation.

---

## Part 5: Additional Items from Code Reviews (cpp-review + Core Guidelines)

### Already Fixed (top 5 priority)

| # | Issue | Fix Applied |
|---|-------|-------------|
| 1 | `PerturbParams` uninitialized `prob/mu/sigma` | Added `= 0.0f` defaults |
| 2 | `PerturbParams::getPerturbOffset()` creates RNG per call (~19K/frame) | Made `thread_local static` |
| 3 | `BaseConfig` raw `new SpheroidConfig*` — no move ops | Replaced with `unique_ptr`, added move ops |
| 4 | `trySplitCell` hardcodes `overlapWeight=1000` | Added parameter, removed hardcoded value |
| 5 | Test expects `cell_color` but `draw()` uses `_brightness` | Updated test to expect 0.5f |

### Remaining Items (to be done in future passes)

#### Code Quality

| # | Issue | Priority | File |
|---|-------|----------|------|
| 9 | `getCellParams()` returns by value in tight loops (25K string allocs/frame) | SHOULD FIX | Spheroid.cpp |
| 10 | C-style casts `(float)a` — use `static_cast<float>()` | SHOULD FIX | Spheroid.cpp |
| 11 | `rand()` used for random axis fallback — use `<random>` | SHOULD FIX | Spheroid.cpp:547,564 |
| 12 | `setenv`/`getenv` hack between main.cpp and CellFactory | SHOULD FIX | main.cpp, CellFactory.cpp |
| 13 | `copyCellsForward(int to)` — int vs size_t comparison | SHOULD FIX | CellUniverse.cpp:428 |
| 14 | Pre-opt params always zero — PCA uses post-collapse state | SHOULD FIX | CellUniverse.cpp:282 |
| 15 | Uninitialized `valid`/`elongationRatio` — use structured bindings | SHOULD FIX | Frame.cpp:339 |
| 16 | `SimulationConfig::explodeConfig` takes `const YAML::Node` by value not reference | SHOULD FIX | ConfigTypes.hpp:31 |
| 17 | PCA threshold uses `mean` not `mean + 0.5*stddev` per docs | CONSIDER | Spheroid.cpp:409 |
| 18 | `typedef` in types.hpp — use `using` aliases | CONSIDER | types.hpp |

#### Function Decomposition (deferred — Part 2 of original plan)

| Function | Lines | Status |
|----------|-------|--------|
| `getSplitCells()` | 339 | Deferred — break into 5 helpers |
| `optimize()` | 144 | Deferred — extract precompute + loop body |
| `trySplitCell()` | 132 | Deferred — extract burn-in loop |
