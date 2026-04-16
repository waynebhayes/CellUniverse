# CellUniverse Pipeline (Current)

**Last updated:** 2026-04-15 — bbox cost + snap-anchored bbox + daughter shared-mask + single-metric bridge + adaptive exponent/inflation + bounded-growth shape reference + fit-side growth cap + OMP multithreading

This is the authoritative end-to-end pipeline for a single frame. It supersedes
the per-frame flow in `docs/plans/2026-04-10-triaxial-pipeline-redesign.md`
and the 2026-04-14 version of this document. Per-fix rationale lives in
`docs/changelogs/changelogv6.md`.

---

## Frame-level flow

```text
================================================================
                        FRAME N START
================================================================
                             |
                             v
      Load raw image, preprocess (ImageHandler, iterative contrast)
                             |
                             v
      Compute adaptive background from frame N-1 cells
                             |
                             v
      Set bbox cost mode:
        frame 1  → full-image L2 (no snapshots available)
        frame 2+ → bbox cost ON (per-cell snap-anchored bbox, see below)
                             |
                             v
      Read snapshot for every cell (end-of-frame N-1):
        { center, a, b, c, rotation,
          splitAxisDir, splitAxisLength,
          shapeElongation = max(a,b,c) / min(a,b,c) }
                             |
                             v
      Precompute P(split)_i linear ramp:
        t = clamp((snapElong - 1.0) / (2.0 - 1.0), 0, 1)
        P(split) = P_split_base + t * (P_split_max - P_split_base)
      (No classification threshold. Every cell has a baseline rate;
       elongated ones scale up toward P_split_max.)
                             |
                             v
      Seed expectedDaughters[name] for every cell with a snap:
        D1_seed = snap.center - 0.5 * splitAxisLength * splitAxisDir
        D2_seed = snap.center + 0.5 * splitAxisLength * splitAxisDir
                             |
                             v
================================================================
      STAGE 1 — POSITION CALIBRATION  (frame 1 only)
================================================================
                             |
                             v
      Runs only when frameIndex == 0. Centroid jump + perturbCell
      refinement for each seed cell. Frame 2+ skipped because every
      cell uses its end-of-previous-frame snap as authoritative
      position. The `[Calibration] frame N skipped` log was removed
      — skip is silent in the normal path.
                             |
                             v
================================================================
      STAGE 2 — PCA SHAPE FIT  (per cell, all cells — snap-masked)
                             [OpenMP-parallelized across cells]
================================================================
                             |
                             v
      For each cell, up to pcaShapeMaxIters (=15) passes:
        1. Gather bright pixels inside FIXED mask sized from the
           cellShapeReference[name] (NOT from snap radii):
              sphere r = maskScale * max(refA, refB, refC)
              ellipsoid mask radii = maskScale * (refA, refB, refC)
           Mask stays constant across iterations — prevents
           mask-feedback collapse onto one daughter.
           See "Shape reference (bounded growth)" below for how ref
           is populated and updated.
           Pixel gather is Voronoi-filtered against the current
           positions of all other cells (claim set).
        2. If |mask| < pcaShapeMinPixels: stop.
        3. Compute pCore = fraction(bright pixel weight > threshold).
           Adaptive exponent:
             exp_used = lerp(expDim=1.3, expBright=1.15, pCore ramp)
           Adaptive radius inflation:
             radius_scale_factor = lerp(1.0, radInflBright=1.15, same ramp)
        4. Weighted centroid mu, covariance C using weights w^exp_used.
        5. cv::eigen(C) -> lambda0>=lambda1>=lambda2, e0,e1,e2.
        6. If lambda0/lambda2 < 1.1: skip rotation (degenerate).
        7. RANK ASSIGNMENT (a <- e0, b <- e1, c <- e2) with sign-align
           to current slot direction.
        8. r_i = radius_scale_factor * pcaShapeRadiusScale * sqrt(lambda_i)
           NO floor; radii free to shrink or grow.
        9. R = [e0|e1|e2], force det=+1, decompose to Euler.
        10. Shift center toward mu, capped at
            pcaShapeMaxPosShiftFraction * maxR per iter.
            (pcaShapeUpdatePosition=false — snap is authoritative;
            enabled only inside split daughter refit.)
        11. Converged when max|dR| < pcaShapeConvergeRadius AND max
            axis rotation < pcaShapeConvergeAngleDeg.
                             |
                             v
      After all cells fit:
        FIT-SIDE GROWTH CAP (10%/frame):
          for each cell with an existing ref:
            cell.radii[i] = min(fitted[i], ref[i] * 1.10)
        Prevents single-frame bloat jumps within a mask that is
        ref * maskScale (up to 1.6x ref).

        REFERENCE UPDATE (5%/frame bounded growth):
          new cells: cellShapeReference[name] = current fit
          existing cells:
            ref[i] = clamp(fit[i], ref[i]*0.95, ref[i]*1.05)
        Ref is the mask basis for frame N+1. Ratchets slowly up
        or down so the shape fit can neither bloat compound nor
        noise-drift collapse.
                             |
                             v
================================================================
      STAGE 3 — SNAP-ANCHORED BBOX INSTALL  (frame 2+ only)
================================================================
                             |
                             v
      For each cell with a valid snapshot:
        bbox = axis-aligned box centered at snap.position with
               half-extent = bbox_margin_scale * snap.maxRadius
        frame.setSnapBbox(name, bbox)
      Cells without snap (daughters born mid-frame) fall through to
      a live pre/post-union bbox in perturbCell.
      Bbox is used as the cost-evaluation window for ALL of this
      cell's perturbations this frame — voxels at the snap position
      are always in scope, so drifting away from snap registers an
      undershoot penalty (position anchoring).
                             |
                             v
================================================================
      STAGE 4 — IMAGE-GROUNDED DAUGHTER PRE-PASS  (ALL cells)
                             [OpenMP-parallelized across cells]
================================================================
                             |
                             v
      For every cell with a valid snapshot:
        Gather bright pixels in bounding box (3 x snap.maxR),
        Voronoi-filter vs neighbors.
        LOCAL-FRAME PCA on pixels:
          localPos = R_inv * (worldPos - center)
          localPos /= (a, b, c)
          Principal eigenvector -> un-normalize -> rotate to world.
        D1_exp, D2_exp = world-space centroids of two halves
        (projections of pixels onto eigenvector, split at median).

      Pre-pass ALWAYS runs — used to find the midpoint + direction of
      the two emerging daughter blobs in the current frame, independent
      of snapshot's (possibly stale) split axis.
                             |
                             v
================================================================
      STAGE 5 — UNIFIED PERTURB + SPLIT LOOP  (all cells)
================================================================
                             |
                             v
      Loop (cells.size() * iterations_per_cell):
        Pick random eligible cell.
        if rand < P(split) & not blacklisted:
            attempt split (see Split attempt section below)
        else:
            perturbCell (position + rotation ONLY; radii fixed by
                         PCA shape fit above).

      perturbCell cost evaluation:
        frame 1 or no snap-bbox: legacy full-image L2.
        frame 2+ with snap-bbox: asymmetric L2 over bbox voxels,
          mask built from Voronoi claim points (self + others).
          If a shared mask was installed (split daughter burn-in)
          for this cell name, use it directly — no sibling mutual
          exclusion.
                             |
                             v
================================================================
      END OF FRAME N
================================================================
                             |
                             v
      Brightness EMA update (per-cell, from real image)
                             |
                             v
      Freeze snapshot for frame N+1:
        shapeElongation = max(a,b,c) / min(a,b,c)  (from fitted shape)
        splitAxisDir = cell.worldSplitAxis()  (shortest axis, world)
        splitAxisLength = shortest radius
                             |
                             v
      Save cells.csv + output PNG images
```

---

## Split attempt (inside Frame::trySplitCellPhased)

For a cell that rolled a split attempt:

1. **Install snapshot-state parent** — swap in an Ellipsoid at snap position, snap radii, snap rotation. Pick `baseline = min(liveCost, snapCost)`.

2. **Gather bright pixels** — Voronoi-filtered in 3×snap.maxR box. `selfClaim` from overridden `splitSnapshot.splitAxisDir * splitAxisLength` (pre-pass values).

3. **Build primary axes** — shortest snapshot local axis only + imgPca direction (dropped axA and axB based on run data showing 0/6 accepts from them). Typical: **20 candidates** (2 primary × 2 midpoints × 5 variants).

4. **Build candidate matrix** — for each primary direction:
   - `data_<axis>_<variant>` midpoint = pixel-projection centroid on that direction
   - `snap_<axis>_<variant>` midpoint = `snapshot.position`
   - 5 variants each: primary, rot-, rot+, trans-, trans+

5. **Split bbox + shared mask install (bbox mode)**:
   - Compute splitBbox = union of parent + all daughter seeds × margin
   - Build splitMask (parent Voronoi, excludes real neighbors — sibling NOT included because daughters don't exist yet)
   - Install `_snapBboxes[parentName+"0"] = splitBbox; ["1"] = splitBbox`
   - Install `_sharedMasks[parentName+"0"] = splitMask; ["1"] = splitMask`
   - Inside perturbCell, daughters use these (skipping the live-Voronoi mask rebuild that would exclude the sibling). This ports 0413's global anchor into bbox scope.

6. **Burn-in** — each candidate runs `split_candidate_burn_in_iterations` perturbCell passes on both daughters with tight sigmas. Tracks total cost.

7. **Winner** — smallest total cost across candidates.

8. **Refine** — `split_final_refine_iterations` extra perturb iters on winner with same tight sigmas.

9. **Daughter refit (per daughter)** — short PCA shape fit with `updatePosition=true`:
   - floor = `split_daughter_refit_min_radius_fraction (0.85) * built_radii`
   - ceil  = `split_daughter_refit_max_radius_fraction (1.1) * built_radii`
   - Clamp final radii into `[floor, ceil]` per axis
   - Position-update lets centroid slide to real bright-pixel centroid (only during daughter refit, not mature-cell shape fit)

10. **Bridge gate** — sample real-image brightness along `(d1 → d2)` axis:
    - **edge1** zone: inside d1, far from gap
    - **gap** zone: middle, expanded to 30% of halfLen minimum
    - **edge2** zone: inside d2, far from gap
    - `valleyFromBright = gapBright / max(edge1Bright, edge2Bright)`
    - **Reject edge_too_dim** if `min(edge1, edge2) < bio_bridge_min_edge_brightness_absolute (0.05)` — one daughter in empty space.
    - **Reject bridge_flat** if `valleyFromBright > bio_bridge_max_valley_ratio (0.85)` — no valley between the brighter daughter and midpoint.
    - Single-metric (not two-tier). The previous tier-1 `worstValleyRatio` hard-0.95 check was retired because it punished legitimate asymmetric division (dim daughter's edge ≈ gap inflates its vr independent of valley quality).

11. **Bio gate**:
    - `max(r1, r2) / min(r1, r2) > bio_daughter_size_ratio_max (1.5)` → reject
    - `(v1 + v2) / refParentVolume ∉ [0.6, 1.3]` → reject
    - `max(v1, v2) / refParentVolume > 0.65` → reject
    - Either daughter buried in non-sibling → reject
    - Either daughter buried in sibling → reject

12. **Cost gate** — accept if `costDiff < -split_cost`. Else reject.

13. **Cleanup on any exit path** — erase `_snapBboxes` and `_sharedMasks` entries for daughter names via `restoreLiveParent` (reject paths) and accept callback (both branches).

---

## Bbox cost (universal)

For frame 2+, perturbCell and split cost both use bbox cost with Voronoi exclusion:

```
cost = asymmetric_L2(real, synth) summed over voxels inside bbox
       where voxel's nearest Voronoi claim belongs to self
```

- **bbox**: snap-anchored per cell (fixed for whole frame, centered at snap.position). Falls back to live pre/post-union bbox for cells without snap (frame 1, newborn daughters).
- **mask**: Voronoi exclusion built from live claim positions at each perturbation, EXCEPT when `_sharedMasks[cellName]` is installed (split daughter burn-in) — then uses pre-built mask without sibling exclusion.
- **asymK**: amplifies overshoot (synth > real) with `asymmetric_cost_weight` factor.

Option A (snap-anchored bbox) ports 0413's global position anchor into local-bbox scope: the snap-position voxels are always in the cost window, so a cell drifting away from snap pays an undershoot penalty.

---

## Shape reference (bounded growth)

Each cell tracks a persistent `(a, b, c)` reference that drives the shape-fit mask:

- **Birth**: captured as the cell's current fitted radii (first appearance in frame.cells).
- **Per-frame update** (after shape fit, before reference-consuming stages):
  ```
  ref_new[i] = clamp(fit[i], ref[i] * 0.95, ref[i] * 1.05)
  ```
  ±5%/frame max change. Allows daughter cells to grow toward adult size at ~5%/frame (matches measured biology). Blocks noise-driven slow drift and halo compounding.
- **Use**: mask half-extent = `maskScale × ref`. Decouples the mask from snap (which reflects last-frame fit, carrying any bloat into the next frame).

Combined with the fit-side cap (+10%/frame), the shape fit is bounded against both compounding drift and instant bloat jumps.

---

## Shape fit details

**Key design**: mask built from the persistent shape reference. Mask stays fixed across iterations. Radii are free to roam within the mask. No volume floor, no per-axis floor. Rank-assignment keeps slot identity stable (a = largest eigenvalue, b = middle, c = smallest).

**If reference missing** (very first frame or edge case): mask falls back to snap radii, then to current live radii.

### Biology — split along short axis

Cells pinch through their thinnest dimension:
- Rod (prolate): short axis direction = cleavage plane normal. Daughters end up stacked along the LONG axis, but the `splitAxisDir` in snapshot is the SHORT axis direction.
- Pancake (oblate): thin dimension is c; daughters stack through z.

`Ellipsoid::worldSplitAxis` returns the shortest semi-axis direction. Do NOT change this — it is biologically correct.

---

## Asymmetric L2 cost

Per-voxel squared error multiplied by `k = asymmetric_cost_weight (=8)` when `synth > real` (cell covers darker real region). Unweighted when `synth <= real`.

```cpp
sumSq = sum(diff²)                              // all pixels
posSumSq = sum(diff² where diff > 0)            // overshoot only
asymSumSq = sumSq + (k - 1) * posSumSq
return sqrt(asymSumSq)
```

SIMD-optimized via `cv::subtract`, `cv::multiply`, `cv::compare`, `cv::sum`.

**Why it matters:** parent ellipsoid covering a dark valley between two daughters pays amplified penalty. Two correctly-placed daughters avoid the valley. Makes the cost comparison reliable when the shape fit is correct.

**Direction of amplification:** this penalizes OVERSHOOT (cell covers dark image). It does NOT amplify undershoot. Higher k pushes candidate selection toward candidates where each cell tightly covers only its real-image territory.

---

## Adaptive exponent + radius inflation

Each cell computes `pCore = fraction(raw pixel weights above pcaShapeCoreBrightnessThreshold)` once per shape-fit pass. Two outputs ramped on the same `pCore`:

- **Weighted PCA exponent**: `expDim (1.3)` for uniform/dim cells, ramping down to `expBright (1.15)` for peaked cells. Looser halo weighting for peaked cells lets halo contribute to the fit.
- **Radius inflation**: `1.0×` for uniform (sqrt(5) is analytically exact), ramping up to `radInflBright (1.15×)` for peaked (compensates PCA's 97%-containment underestimation of Gaussian-like distributions).

Overfitted cells (halo bleed, neighbor contamination) typically have LOW pCore because halo dilutes the core fraction — the adaptive path doesn't fire for them, so inflation doesn't amplify bloat. Bounded-growth reference handles their cap separately.

Threshold `pcaShapeCoreBrightnessThreshold` must match the dataset's pixel-weight range — 0.15 for this dataset (iterative-contrast preprocessing produces weights in 0.05–0.20). Set too high (e.g., 0.6 for pre-sigmoid-era) and pCore is always 0, adaptive paths collapse to defaults.

---

## P(split) linear ramp

```cpp
t = clamp((snapElongation - 1.0) / (2.0 - 1.0), 0, 1)
P(split) = P_split_base + t * (P_split_max - P_split_base)
```

- `snapElong = 1.0` → P(split) = 0.03 (P_split_base)
- `snapElong = 1.5` → P(split) = 0.265
- `snapElong = 2.0` → P(split) = 0.50 (P_split_max, clamped above)

All cells get at least the base rate, so first-gen splits with near-round snaps still attempt.

---

## Multithreading (OpenMP)

Per-cell work that is cell-independent runs in parallel with deterministic log ordering:

- **PCA shape fit** (Stage 2) — each cell fits in its own thread; per-cell `ostringstream` log sinks merged serially in cell-index order after the parallel region.
- **Pre-pass** (Stage 4) — same pattern. `imageGroundExpectedDaughters` is const-safe.
- **Bbox cost inner loop** — `#pragma omp parallel for reduction(+:totalCost)` over z-slices inside `calculateBboxCost`.
- **Full-cost cache refresh** — slice-parallel.

Main optimize loop and split candidate burn-in remain serial (shared mutable state in `Frame`; parallelizing requires a separate thread-local-state design).

Build: CMake auto-detects OpenMP on macOS (Homebrew libomp auto-prefix) and Linux (system libgomp). Falls back to sequential build if absent.

---

## Config knobs (config.yaml — current)

```yaml
cell:
  pcaShapeMaxIters: 15
  pcaShapeRadiusScale: 2.236          # sqrt(5), uniform-ellipsoid variance
  pcaShapeMinPixels: 50
  pcaShapeMaskScale: 1.6              # raised from 1.3 to match 0413 effective converged mask
  pcaShapeConvergeRadius: 0.3
  pcaShapeConvergeAngleDeg: 2.0
  pcaShapeUpdatePosition: false        # snap is authoritative for mature cells
  pcaShapeMaxPosShiftFraction: 0.5
  pcaShapeWeightExponent: 1.3          # baseline (dim) exponent

  # Adaptive exponent + radius inflation (driven by pCore)
  pcaShapeAdaptiveExponent: true
  pcaShapeWeightExponentBright: 1.15
  pcaShapeCoreBrightnessThreshold: 0.15
  pcaShapeCoreFractionLow: 0.10
  pcaShapeCoreFractionHigh: 0.40
  pcaShapeRadiusInflationBright: 1.15

simulation:
  asymmetric_cost_weight: 8.0          # overshoot amplification factor

prob:
  # P(split) ramp
  P_split_base: 0.03
  P_split_max: 0.5

  # Split thresholds
  split_cost: 375                      # min cost improvement to accept
  overlap_penalty_weight: 30000.0      # scales with bbox cost magnitudes

  # Bbox cost
  use_bbox_cost: true                   # frame 2+; frame 1 forced full-image
  bbox_margin_scale: 3.0

  # Split candidate generation
  split_candidates_per_attempt: 30
  split_candidate_burn_in_iterations: 50
  split_final_refine_iterations: 30
  split_calibration_iterations_per_cell: 50  # frame 1 only
  split_candidate_rotation_delta_degrees: 8
  split_candidate_translation_delta_fraction: 0.2

  # Daughter refit (per-daughter PCA after split refine)
  split_daughter_refit_iterations: 3
  split_daughter_refit_min_radius_fraction: 0.85
  split_daughter_refit_max_radius_fraction: 1.1
  split_daughter_volume_scale: 0.7937   # cbrt(1/2) volume-preserving

  # Bio gates
  bio_daughter_size_ratio_max: 1.5
  bio_combined_volume_min_fraction: 0.6
  bio_combined_volume_max_fraction: 1.3
  bio_max_single_daughter_volume_fraction: 0.65
  bio_bridge_max_valley_ratio: 0.85    # single metric: gap / max(edges)
  bio_bridge_min_edge_brightness_absolute: 0.05

  # DEPRECATED (parsed for backward compat, no effect)
  bio_bridge_max_gap_density: 0.18
  bio_bridge_no_valley_hard_threshold: 0.95

  split_burn_in_pos_sigma_scale: 0.4
```

Growth caps (hardcoded as constants in `CellUniverse.cpp` shape-fit block):
- `refGrowthCap = 0.05` — reference updates ±5%/frame
- `fitGrowthCap = 0.10` — fit clamped at ref × 1.10 per frame

---

## Deleted / deprecated code (do not resurrect without reason)

| Item | Why retired |
|---|---|
| `shape_elongation_classify_threshold` + classification | Linear ramp handles discrimination |
| `measureRadiusGradient` + radius gradient loop | Symmetric-shrink when rotation wrong |
| `measureRadiiFromImage` + radius EMA | Percentile-based, biased by sampling |
| `_birthFrame` + age gate | Never fired |
| `_birthVolume` ratchet + volume floor | Blocked legitimate shrinkage |
| Per-axis snap-shrink floor (Fix C v1) | Blocked aspect-ratio change |
| `size_reduction_penalty_weight` | Froze cells solid |
| `min_frames_before_split` | Never read |
| `bio_max_midpoint_parent_fraction` | Pre-pass grounding replaced it |
| `split_direction_agreement_degrees` | Superseded by always-try-both |
| `pcaShapeBlend` | Iteration replaced EMA blending |
| Phase A / Phase B ordering | No measurable effect |
| Bridge tier-1 `worstValleyRatio >= 0.95` | Punished legitimate asymmetric division |
| Bridge `gapDensity` conjunction | Single valleyFromBright metric subsumes it |
| `bio_bridge_no_valley_hard_threshold` | Retired with tier-1 |
| `bio_bridge_max_gap_density` | Retired with tier-2 |
| `split_burn_in_radius_sigma_scale` | Radii never perturbed during burn-in |
| `calibration_max_centroid_jump_voxels` | Calibration is frame-1-only now |
| Frame-N>=2 `[Calibration] frame N skipped` log | Silent skip, log noise |

---

## Diagnostic log tags

- `[P(split)]` — per-frame P(split) per cell
- `[Calibration]` — per-cell position calibration result (frame 1 only)
- `[PCA Shape] cell=... iter=i` — shape fit iteration progress
- `[PCA Shape Exp] cell=... pCore=... exp=... radInfl=...` — adaptive exponent + inflation per cell
- `[PCA Shape Cache]` — bright-pixel gather cache hit/miss
- `[Fit Growth Cap] frame N clamped=X` — fit-side cap fired this frame
- `[Shape Reference] frame N captured=X updated=Y total=Z growthCap=0.05` — reference-update accounting
- `[Snap Bbox] frame N installed=X total=Y` — snap-bbox install coverage
- `[Pre-Pass] frame N allCells=M` — pre-pass invocation
- `[Pre-Pass Claims]` — Voronoi claim set per cell
- `[Pre-Pass] round=0 cell=...` — per-cell pre-pass result with shift from seed
- `[Split Snapshot Parent]` — snapshot-state parent install
- `[Split Attempt]` — attempt header with snap/live state
- `[Split Seeds]` — seeds from snap axis
- `[Split Dirs] selected=[axC, imgPca]` — primary axes used for candidate generation
- `[Split AxisPlace]` — per-axis data midpoint computation
- `[Split Sigmas]` — burn-in sigma values
- `[Split Bbox Init]` — split union bbox + mask seed points
- `[Split Cand]` — per-candidate burn-in result (seed/final/drift/cost)
- `[Split Winner]` — winning candidate after burn-in
- `[Split Daughter Refit] d1/d2 ... built=... floor=... ceil=... pre=... post=... prePos=... postPos=... posShift=...` — per-daughter refit
- `[Split Refine]` — refine-iter result
- `[Split Bridge]` — bridge gate measurements (valleyFromBright, edges, gap, density for continuity)
- `[Split Reject bio] reason=...` — bio/bridge rejection (bridge_flat / edge_too_dim / volume_fraction / size_ratio / buried / sibling_buried)
- `[Split Reject cost]` — cost gate rejection
- `[Split Accepted]` — final split acceptance with costDiff and daughter geometry
- `[Optimize Done] frame N perturb_accepted=... split_attempts=... split_accepted=... final_cells=...` — per-frame summary

---

## Related docs

- `docs/changelogs/changelogv6.md` — per-fix change log with before/after code
- `docs/plans/2026-04-14-universal-bbox-cost.md` — bbox cost migration design record
- `docs/plans/2026-04-15-bbox-anchor-flaw-analysis.md` — Option A analysis
- `docs/plans/2026-04-10-triaxial-pipeline-redesign.md` — older design record (OUTDATED, references classification)
- `.claude/rules/gotchas.md` — critical invariants
- `.claude/rules/codebase.md` — file-by-file roles
