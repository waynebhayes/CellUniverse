# Universal Bounding-Box Cost Refactor

**Date:** 2026-04-14 (evening)
**Status:** DESIGN — decisions locked (D1–D7), implementation pending
**Motivation:** decouple per-cell cost decisions from unrelated image regions; concentrate signal; speed up every cost evaluation.

---

## 1. Problem

Every cost computation today — perturbation, split burn-in, split refine, split cost gate — measures asymmetric L2 over the **entire image stack** (225 slices × full xy). This has three problems:

1. **Signal dilution.** A split changes ~10k voxels; the other ~36M voxels contribute equal cost to both baseline and candidate but dominate the absolute numbers. `costDiff` between true and false splits sits at 0.1–8.0 on top of a ~7700 baseline (< 0.1% change). Small-fraction discrimination is fragile.
2. **Cross-cell contamination.** When cell A is perturbed, its cost change is measured against the full image cost — which includes cell B's fit, cell C's fit, etc. If B or C drift or get mis-rendered elsewhere, A's local decision is polluted.
3. **Cost.** Full-slice L2 over 225 slices per perturbation × thousands of perturbations per frame is the dominant runtime. Most of those voxels are unaffected by the perturbation and contribute no information.

## 2. Proposal — universal bbox cost

Replace full-image L2 with **per-cell bbox cost with neighbor exclusion**, applied uniformly at every cost-evaluation site.

**For a given cell `c`:**

- **bbox(c)** = axis-aligned box centered on `c.position` with half-extent `bboxMarginScale * c.maxRadius` on each side (clamped to image bounds).
- **exclusion mask** over bbox(c): a 3D binary mask where `mask[v] = 0` if voxel `v` is claimed by any OTHER cell (Voronoi: closer to another cell's center than to `c`), `mask[v] = 1` otherwise.
- **cost(c)** = sum over voxels in bbox(c) where `mask[v] = 1` of `asymmetric_L2(real[v], synth[v], k)`.

Every perturbation / split decision compares `cost(c)` before vs after, over the **same bbox and same exclusion mask** — apples-to-apples.

## 3. Key design decisions — RESOLVED

### D1. Bbox sizing — **CONFIRMED: `3 * maxRadius` half-extent**

Matches the existing convention in the codebase:

- `Frame.cpp:1035` (pre-pass PCA): `boxRadius = 3.0f * max(srcMajor, srcB, srcMinor)`
- `Frame.cpp:1096` (shape fit gather): same
- `Frame.cpp:1583` (split attempt pixel gather): same

All three existing call sites use `3 × max(a, b, c)`. Universal bbox cost adopts the same formula. A 30-voxel radius cell gives a 180³ = 5.8M voxel bbox — ~6× smaller than the full image, comfortable daughter-drift margin.

Bbox formula:

```cpp
const float boxRadius = 3.0f * std::max({cell.aRadius, cell.bRadius, cell.cRadius});
bbox.xMin = clamp(floor(cell.x - boxRadius), 0, cols-1);
bbox.xMax = clamp(ceil (cell.x + boxRadius), 0, cols-1);
// ... y, z similarly
```

For split union bbox:
```cpp
bbox(parent) ∪ bbox(d1) ∪ bbox(d2), each with 3×maxR half-extent
```

### D2. Neighbor exclusion — **CONFIRMED: match `gatherBrightPixelsVoronoi`**

Use the same Voronoi-with-claim-point-sets exclusion already in use at `Frame.cpp:479–568`. Keep voxel iff its nearest claim point across ALL cells belongs to self.

```cpp
bool keep = true;
float selfBest = infinity;
for (const auto &sp : selfClaimPoints)
    selfBest = min(selfBest, distSq(v, sp));
for (const auto &kv : otherClaimSets)
    for (const auto &op : kv.second)
        if (distSq(v, op) < selfBest) { keep = false; break; }
```

- `selfClaimPoints` = the cell(s) we're computing cost for (single cell for perturb, {parent, d1, d2} for split).
- `otherClaimSets` = map<name→vector<Point3f>> for every non-self cell — the existing claim-set pattern used throughout `CellUniverse::optimize` (see `buildShapeClaimSet`, `buildCalibrationClaimSet`).

Benefit: consistent with shape-fit / pre-pass / split paths; no new exclusion semantics introduced. If pre-pass seeded expected-daughter positions into a claim set, the cost computation honors those seeds too.

### D3. No global cost — **CONFIRMED**

- Replace scalar `_currentCost` with per-cell bbox-cost bookkeeping.
- `_currentCostPerSlice` (225-double cache) deleted.
- Diagnostics that logged `_currentCost` either (a) log the sum of per-cell bbox costs or (b) removed.
- `refreshFullCostCache` deleted; replaced with per-cell cost-on-demand.
- Overlap penalty continues to be computed per-cell via `computeOverlapForCell` (already per-cell); added to bbox cost at decision time.

### D4. No cost cache initially — **CONFIRMED**

**Clarification on the question "would cache affect how cells split or are computed?":**

A *correct* cache does not change decisions — it's a pure perf optimization (the current `_currentCostPerSlice` is bit-exact because `cv::Mat` shallow aliasing means unchanged slice buffers are the same pointer, so cached L2 is literally the same number). Cache correctness is a bug surface, not a design lever.

An *incorrect* cache would change decisions, and usually for the worse.

Plan: **start with no cache.** Each `perturbCell` or split cost evaluation computes bbox cost from scratch. Measure runtime. If it's the bottleneck, add caching with invalidation rules:

1. Cache `bboxCost[cellIdx]`.
2. On perturbCell accept for cell `i`: invalidate `i`'s entry AND every cell `j` whose bbox(j) intersects bbox(i).
3. On split accept: invalidate all cells (simplest) or just neighbors of parent.

Caching is optional Stage 5 work. Stage 2–3 uses no cache.

### D5. Split union bbox — **CONFIRMED**

`splitBbox = bbox(parent) ∪ bbox(d1) ∪ bbox(d2)` with 3×maxR half-extent each. Sized to cover the WORST-CASE candidate (max daughter offset across all candidates) so the same voxel set is used for:

- baseline (parent rendered, daughters absent)
- every candidate in burn-in (d1 + d2 rendered, parent absent)
- refine
- cost gate

Exclusion set for split: all cells except parent and the current d1/d2. Apples-to-apples across all comparisons.

### D6. Overlap penalty — **KEEP unchanged**

User confirmation: "We do also need an overlapping check, so that the daughters don't get placed or perturbed on each other."

Keep `computeOverlapForCell` / `computeOverlapPenalty` as the geometric overlap term. With Voronoi exclusion, the overlap region is excluded from BOTH cells' bbox image costs, so bbox cost alone is blind to overlap — the penalty is the only signal preventing daughters from being placed on each other or on other cells. Added to bbox image cost at decision time exactly as today.

### D7. Parameter retuning — **CONFIRMED: test, then tune**

Expected magnitude shift:

| Quantity | Full-image (today) | Bbox (expected) |
|---|---|---|
| Per-cell baseline cost | n/a (global ~7700) | 50–200 |
| Split costDiff (true split) | −18 to −23 | expected ~−3 to −10 |
| Split costDiff (false split) | −15.5 | expected near 0 or positive |
| Overlap penalty contribution | 0 to ~1000 | unchanged magnitude |

Parameters to retune after Stage 3:

| Parameter | Current | Direction |
|---|---|---|
| `split_cost` | 15 | Lower (~2–5) — threshold on bbox diff not full diff. |
| `overlap_penalty_weight` | 500 | May drop (~50–100) — it was scaled to dominate full-image cost swings; now competes with bbox-scale L2. Too high → overlap becomes impossible and cells can't pack densely. Too low → daughters sit on each other. Tune empirically. |
| `asymmetric_cost_weight` | 8 | Unchanged (per-voxel multiplier, scale-invariant). |

Process: land Stage 3, run frames 1–8, log `[Split Winner] bboxCostDiff=X` for every accepted + rejected candidate. Set `split_cost` to split real from phantom. Same for overlap.

## 4. Implementation staging

### Stage 1 — infrastructure (1 file, ~300 LOC)

- `BoundingBox3D` struct: `{int xMin, xMax, yMin, yMax, zMin, zMax}` + helpers.
- `Frame::computeCellBbox(size_t cellIdx, float marginScale)` → BoundingBox3D, clipped to image bounds.
- `Frame::computeUnionBbox(std::vector<size_t> cellIndices, float marginScale)` → for split.
- `Frame::buildExclusionMask(const BoundingBox3D &bbox, const std::vector<size_t> &excludeIndices)` → `std::vector<uint8_t>` of bbox volume, 1=include, 0=exclude.
- `Frame::calculateBboxCost(const BoundingBox3D &bbox, const std::vector<uint8_t> &mask)` → asymmetric L2 on included voxels.

### Stage 2 — perturbation switch (1 file, ~50 LOC)

- `Frame::perturbCell` currently uses `calculateIncrementalCost`. Replace with:
  - Compute bbox(cellIdx) (union of pre and post positions to cover drift).
  - Build exclusion mask (all other cells).
  - costBefore = calculateBboxCost(bbox, mask) on current synth.
  - Apply perturbation, regenerate synth inside bbox.
  - costAfter = calculateBboxCost(bbox, mask).
  - delta = costAfter - costBefore + overlapDelta.
- `calculateIncrementalCost` stays as dead code for now (remove after split is migrated).

### Stage 3 — split switch (1 file, ~100 LOC)

- In `trySplitCellPhased`:
  - Baseline: compute unionBbox + exclusion mask (exclude all but parent); cost = bboxCost with parent rendered.
  - Each candidate: unionBbox may differ (daughter positions differ); recompute per candidate. cost = bboxCost with d1+d2 rendered over its own bbox.
  - Wait — if bbox differs per candidate, we can't compare directly. FIX: use a SINGLE unionBbox for the whole attempt, sized to cover all candidate daughter positions (add extra margin for candidate variation). This way baseline and all candidates use the same voxel set.
- Refine: same unionBbox as ranking.
- Cost gate: same.

### Stage 4 — parameter retuning + cleanup

- Remove old `_currentCost` state.
- Remove `calculateCost` and `calculateIncrementalCost` or demote them to diagnostics-only.
- Update `split_cost`, `overlap_penalty_weight` in config.
- Run known-good test (frames 1–8), measure costDiffs, retune.

## 5. Risks

1. **Per-cell cost doesn't reflect global image fit.** A cell that minimizes its local bbox cost might drift into a bad global configuration. Voronoi exclusion mitigates this (cells can't "steal" neighbor territory because neighbor territory is excluded from this cell's cost). But corner cases possible — e.g., if a cell grows beyond its Voronoi region, its bbox includes voxels outside its claim territory that are also claimed by no one, so still counted. Should be fine.
2. **Claim-set drift during burn-in.** During split burn-in, daughter positions move. The exclusion mask built pre-burn-in uses parent position; as daughters separate, their own territory changes. For consistency I'll use the FINAL daughter positions (post-burn-in) when building the mask for the cost gate. For burn-in ranking itself, the mask is built at candidate start and stays fixed — a simplification that costs some precision but avoids per-iter mask rebuild.
3. **Speed of exclusion mask build.** For a 180³ bbox with 10 neighbors, naive build is 180³ × 10 ≈ 58M distance comparisons per mask build — ~0.5s. Need to be smart:
   - Build mask once per perturbCell call (not per candidate in burn-in).
   - Cache mask across consecutive perturbations of the same cell when neighbors haven't moved.
4. **OpenCV SIMD.** Asymmetric L2 per slice uses `cv::subtract`/`cv::multiply`/`cv::sum` on full `cv::Mat` slices. For bbox cost I'll do per-voxel scalar sum inside the bbox loop (with exclusion mask) — slower than SIMD per slice but over much smaller volume. Can optimize later with ROI + slice mask if needed.
5. **Overlap penalty interaction.** Unchanged by bbox. No new risk.

## 6. Rollout plan

- Land Stage 1 infrastructure + unit tests (can exist alongside old code).
- Land Stage 2 perturbation switch with a feature flag (`use_bbox_cost: true/false` in config) — run side-by-side with old and compare.
- Land Stage 3 split switch behind same flag.
- Run a full frame-1-to-45 test under both configs.
- If bbox config produces equal or better split accuracy with comparable runtime:
  - Remove feature flag, make bbox the only path.
  - Delete old `calculateCost`/`calculateIncrementalCost` or keep as diagnostic-only.
  - Retune `split_cost`, `overlap_penalty_weight`.
- If bbox is worse: diagnose per Stage 4 numbers, adjust design, retry.

## 7. Estimated scope

- New code: ~500 LOC across 1 source file, 1 header.
- Changed code: ~100 LOC (perturbCell + trySplitCellPhased migration, config.yaml).
- Deleted code (post-rollout): `calculateCost`, `calculateIncrementalCost`, `_currentCost`, `_currentCostPerSlice`, `refreshFullCostCache` — ~150 LOC.
- Config changes: 2 new fields (`bbox_margin_scale`, `use_bbox_cost` flag during rollout).
- Doc changes: pipeline.md, gotchas.md, this plan file.

Work estimate: 1–2 sessions for Stages 1–3, 1 session for Stage 4 (tune + cleanup).

---

## 8. Performance & parallelization plan

Current runtime: ~1h per 22 frames. Target: 5–15 min per 22 frames (4–12× speedup). This section covers both standalone perf work and how it interacts with the bbox refactor.

### 8.1 Current hot spots (measured + inferred)

Inferred from call counts per frame (assuming ~10 cells, 3000 main-loop iters/frame):

| Site | Calls/frame | Work per call | Total relative cost |
|---|---|---|---|
| `perturbCell` → `calculateIncrementalCost` | ~3000 | 225-slice cost, ~5 slices actually recomputed | **high** (cost loop dominates) |
| Split burn-in `perturbCell` | ~30 cand × 50 iter × ~0.3 split prob × ~10 cells = 4500 | same as above, but 10 slices affected | **high** |
| `generateSynthFrameFast` (inside perturb) | same count | writes affected z-range only | **medium** — draw() per cell per affected slice |
| `calibrateCellShapeViaPca` (shape fit) | 10 cells × 15 iter = 150 | 3D bbox scan + Voronoi + PCA eigen | **high** (pixel gather dominates) |
| Pre-pass PCA | 10 cells × 1 round | same | **medium** |
| Split attempt (per accepted split) | ~2/frame | 30 candidates × burn-in + refine + gates | **spike** |
| Full-image cost cache refresh | 1/frame at start | 225-slice L2 | **low** (once per frame) |

Dominant: `asymmetricL2Slice` (already SIMD-ified) called inside cost evaluations. Cell-independent work inside a single frame = huge parallelism opportunity.

### 8.2 OpenMP — per-cell parallelization (biggest win)

The cleanest parallelization is **per-cell loops** where cells are independent:

#### 8.2.1 Shape fit loop (Frame.cpp – `CellUniverse.cpp:499–517`)

```cpp
#pragma omp parallel for schedule(dynamic)
for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
    Frame::ClaimSet others = buildShapeClaimSet(sname);  // read-only input
    frame.calibrateCellShapeViaPca(ci, others, ...);     // writes cells[ci] only
}
frame.regenerateSynthFrame();  // after all cells done
```

**Safety:** `_realFrame` is read-only. `cells[ci]` writes are unique per thread. `claimSet` is thread-local. The Ellipsoid PCA work is self-contained. Safe.

**Expected speedup:** ~N× on an N-core machine for this stage. On an 8-core machine, shape fit phase goes from ~150ms to ~20ms per frame.

#### 8.2.2 Pre-pass loop (CellUniverse.cpp – same pattern)

Same safety argument. Same parallelization.

#### 8.2.3 Position calibration loop (CellUniverse.cpp:413–449)

This one's trickier — each `perturbCell` writes to `_synthFrame` and `_currentCost`. Cannot parallelize safely without per-thread synth buffers.

**Option:** restructure calibration to compute per-cell local metrics, accept/reject without touching global synth. Defer the synth update to a single-threaded post-loop pass. Requires redesign; park for later.

### 8.3 OpenMP — per-slice parallelization (cost loops)

For `calculateCost` / `calculateIncrementalCost` / `refreshFullCostCache`, 225 independent slice-L2 calls:

```cpp
double totalCost = 0.0;
#pragma omp parallel for reduction(+:totalCost) schedule(static)
for (int i = 0; i < numSlices; ++i) {
    totalCost += asymmetricL2Slice(_realFrame[i], _synthFrame[i], asymK);
}
```

**Safety:** `asymmetricL2Slice` is pure (reads two `cv::Mat`s, returns a double). `_realFrame[i]` and `_synthFrame[i]` are independent. Safe.

**Caveat:** `perturbCell` currently recomputes only ~5 slices per call (the cell's affected z-range). Parallelizing 5 slices over N threads has high overhead. Only worth it for `refreshFullCostCache` (225 slices at frame start). For incremental cost, don't parallelize — the inner loop is too short.

**Expected speedup:** full-cost refresh ~N× (once per frame, not a major bottleneck). Split baseline cost eval (full 225 slices) also benefits when called.

### 8.4 OpenMP — split candidate parallelization (medium effort)

Split burn-in runs 30 candidates × 50 iters each, serially. Candidates are INDEPENDENT in principle (each starts from the same baseline state), but they share `_synthFrame` and `cells`.

To parallelize candidates: each thread needs its own snapshot of `(cells, _synthFrame, _currentCost, _currentCostPerSlice)`. Synth frame is ~225 × 400 × 400 × 4 bytes = 144 MB per snapshot. On a 16-thread machine: 2.3 GB. Memory cost substantial but feasible.

**Recommendation:** defer until after bbox refactor. With bbox cost, per-thread state is just a bbox-sized synth slab (~180³ × 4 = 23 MB, 16× smaller). Parallelizing then is cheap.

### 8.5 Compiler / build flags (zero-risk free speedup)

Current `CMakeLists.txt` has **no explicit optimization flags** and no default build type. Depending on how CMake is invoked, we may be running `-O0` (debug builds are ~20× slower than `-O3`).

Add to CMakeLists.txt:

```cmake
# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Release optimization flags
target_compile_options(celluniverse PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
)

# Link-time optimization (IPO / LTO)
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
if(IPO_SUPPORTED)
    set_property(TARGET celluniverse PROPERTY
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(celluniverse OpenMP::OpenMP_CXX)
```

**Expected speedup:** 2–5× from `-O3 -march=native` alone (if we were on `-O0`), 1.1–1.3× from LTO, no speedup from just adding the OpenMP link (need `#pragma` directives too).

### 8.6 Algorithmic reductions

Orthogonal to parallelization:

#### 8.6.1 Bright-pixel gather cache

`gatherBrightPixelsVoronoi` scans a 3D box and does Voronoi exclusion. Called from:

- Shape fit (15 iters × 10 cells = 150 calls/frame)
- Pre-pass (10 cells × 1 round)
- Split attempt (per cell, once)

The RAW pixel scan (all voxels in box, above brightness cutoff) is independent of rotation/radii — only changes when the cell's position or claim sets change. Split into two stages:

1. `gatherCandidatePixels(center, radius, claimSet)` — returns `vector<BrightPixel>` of all bright pixels in box respecting Voronoi. Result depends on `(center, radius, claimSet)` only.
2. `filterByEllipsoidMask(pixels, rotation, radii)` — applies the ellipsoid-mask test in local frame.

Cache stage 1 per (cell, frame) — rotations/radii change each PCA iter but center + claimSet don't. Saves 14/15 of the gather work during shape fit.

**Expected speedup:** 5–10× on shape fit. Combined with per-cell OMP → 40–80× on this stage.

### 8.7 Interaction with bbox refactor (Section 3)

The bbox refactor is itself a major perf win because every cost eval scans a much smaller voxel set:

- Full-image cost: 225 × 400 × 400 × k ops = 36M × k
- Bbox cost (single cell, 3×maxR): 180³ × k = 5.8M × k ≈ **6× faster baseline**
- Bbox with Voronoi exclusion: maybe 3–4M × k ≈ **10× faster**

Combined with OpenMP per-cell:

- Baseline cost eval: 10× (bbox) × 8× (OMP) = **80× faster** on per-cell decisions.
- Split attempt: 10× (smaller union bbox) × 8× (OMP candidates with smaller per-thread state) = **80× faster**.

The bbox refactor also makes OMP on split candidates feasible (per-thread state is tiny). Doing them together is a natural pairing.

### 8.8 Thread-safety audit (prerequisite for OMP)

Before enabling `#pragma omp parallel for`, verify:

- `Ellipsoid::cellConfig` is static global — read-only during parallel section is OK, but writes (e.g. `cellConfig.x.sigma = ...` during calibration) must happen OUTSIDE the parallel region.
- `srand()` / `rand()` / `std::uniform_int_distribution` — per-thread RNG state needed. OpenMP doesn't automatically provide this. Options: (a) `thread_local` RNG, (b) pass an RNG into each function (cleaner but invasive), (c) use `#pragma omp critical` around RNG calls (slow, defeats the purpose).
- `std::cout` logging — thread-interleaved output is OK for diagnostic logs (each thread's `<<` chain completes atomically with `std::cout::sync_with_stdio` default). For clean per-cell logging, consider a per-thread `std::ostringstream` flushed at sync points.
- `cv::Mat` clone / subtract / sum are thread-safe for independent operands. But `cv::Mat` COPY of the same matrix across threads is fine (reference-counted). Writing different `cells[i]` entries is fine because `std::vector` elements don't share storage.

Action: audit `calibrateCellShapeViaPca` and `perturbCell` for hidden global writes before marking them OMP-safe. One known issue: `Ellipsoid::cellConfig` mutations. These happen pre/post OMP regions already, so fine if we're careful.

### 8.9 Staged performance work

| Stage | Effort | Expected speedup | Risk |
|---|---|---|---|
| **P1. CMake flags** (`-O3 -march=native -DNDEBUG` + LTO) | 30 min | 2–5× (if we were on -O0) | zero |
| **P2. OMP on shape fit + pre-pass loops** | 2 hrs + thread audit | 4–8× on those stages | low (cells are independent, just need RNG guard) |
| **P3. OMP on cost refresh / full-image eval** | 1 hr | 8× on cost evals that hit full image | zero |
| **P4. Bright-pixel gather cache** | 3 hrs | 5–10× on shape fit | medium (cache invalidation correctness) |
| **P5. Bbox cost refactor** (Sections 1–7 above) | 1–2 sessions | 6–10× on per-cell cost | medium (behavior change, requires retune) |
| **P6. OMP on split candidates** (post-bbox) | 2 hrs | 8× on split attempt phase | low (bbox makes per-thread state cheap) |

**Combined expected speedup (P1 + P2 + P3 + P5 + P6):** ~20–30× overall. 1h/22 frames → ~2-3 min/22 frames.

### 8.10 Rollout order

1. **Ship P1 first** (CMake flags — zero risk, free speedup, verify whether we're already on `-O3`).
2. **Ship P2 + P3** (OMP per-cell loops + cost refresh) after a quick thread-safety audit.
3. **Decide:** A1+B1 validation first. If split accuracy is fixed, **skip P5 (bbox refactor)** entirely unless runtime still unacceptable. The bbox refactor is only worth doing if we also need it for correctness.
4. If bbox refactor lands, then P6 + P4 are easy follow-ups.

Don't block A1+B1 validation on any of this. Perf work is additive.
