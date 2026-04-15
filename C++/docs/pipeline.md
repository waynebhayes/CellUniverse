# CellUniverse Pipeline (Current)

**Last updated:** 2026-04-14 (evening) — classification removed, pre-pass for all cells, snap-mask shape fit, asymmetric L2 cost, bridge two-tier

This is the authoritative end-to-end pipeline for a single frame. It supersedes
the per-frame flow in `docs/plans/2026-04-10-triaxial-pipeline-redesign.md`,
which is preserved as the (outdated) design record.

---

## Frame-level flow

```text
================================================================
                        FRAME N START
================================================================
                             |
                             v
      Load raw image, preprocess (ImageHandler)
                             |
                             v
      Compute adaptive background from frame N-1 cells
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
      STAGE 1 — POSITION CALIBRATION  (per cell, all cells)
================================================================
                             |
                             v
      Step 1: centroid jump (Voronoi-filtered bright pixels)
      Step 2: 50 perturbCell iters (tight position sigma, rotation free)
      (Overlaps with Stage 2 centroid update; candidate for reduction.)
                             |
                             v
================================================================
      STAGE 2 — PCA SHAPE FIT  (per cell, all cells — snap-masked)
================================================================
                             |
                             v
      For each cell, up to pcaShapeMaxIters (=15) passes:
        1. Gather bright pixels inside FIXED mask sized from SNAPSHOT
           radii (maskScale * snap.{a,b,c}), Voronoi-filtered.
           Mask stays constant across iterations — prevents
           mask-feedback collapse onto one daughter.
        2. If |mask| < pcaShapeMinPixels: stop.
        3. Weighted centroid mu, covariance C.
        4. cv::eigen(C) -> lambda0>=lambda1>=lambda2, e0,e1,e2.
        5. If lambda0/lambda2 < 1.1: skip rotation (degenerate).
        6. RANK ASSIGNMENT (a <- e0, b <- e1, c <- e2) with sign-align
           to current slot direction.
        7. r_i = pcaShapeRadiusScale * sqrt(lambda_i) — NO floor, radii
           free to shrink or grow to match real image extent.
        8. R = [e0|e1|e2], force det=+1, decompose to Euler.
        9. Shift center toward mu, capped at
           pcaShapeMaxPosShiftFraction * maxR per iter.
        10. Converged when max|dR| < pcaShapeConvergeRadius AND max
            axis rotation < pcaShapeConvergeAngleDeg.
                             |
                             v
================================================================
      STAGE 3 — IMAGE-GROUNDED DAUGHTER PRE-PASS  (ALL cells)
================================================================
                             |
                             v
      For every cell with a valid snapshot:
        Gather bright pixels in bounding box (3 x snap.maxR),
        Voronoi-filter vs neighbors (using D1/D2 seeds if available,
        else live positions).
        LOCAL-FRAME PCA on pixels:
          localPos = R_inv * (worldPos - center)
          localPos /= (a, b, c)
          Principal eigenvector -> un-normalize -> rotate to world.
        D1_exp, D2_exp = world-space centroids of two halves
        (projections of pixels onto eigenvector, split at median).

      Pre-pass ALWAYS runs — used to find the midpoint + direction of
      the two emerging daughter blobs in the current frame, independent
      of snapshot's (possibly stale / arbitrary) split axis.
                             |
                             v
================================================================
      STAGE 4 — UNIFIED PERTURB + SPLIT LOOP  (all cells)
================================================================
                             |
                             v
      Loop (cells.size() * iterations_per_cell):
        Pick random eligible cell.
        if rand < P(split) & not blacklisted:
            attempt split (see STAGE 5)
        else:
            perturbCell (position + rotation ONLY; radii are PCA-driven).

      No Phase A/B distinction. Position calibration and shape fit
      already settled all cells upfront.
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

## Stage 5 — Split attempt (inside Frame::trySplitCellPhased)

For a cell that rolled a split attempt:

1. **Install snapshot-state parent** — swap in an Ellipsoid at snap position, snap radii, snap rotation. Pick `baseline = min(liveCost, snapCost)` (snap chosen when live has drifted).

2. **Gather bright pixels** — Voronoi-filtered in 3×snap.maxR box. `selfClaim` from overridden `splitSnapshot.splitAxisDir * splitAxisLength` (pre-pass values).

3. **Build primary axes** — snapshot parent's rotation axes (axA, axB, axC) filtered to those within 20% of `min(rA,rB,rC)`, PLUS `imgPca` direction (from the overridden `splitSnapshot.splitAxisDir` = pre-pass PCA direction), added if non-degenerate vs existing.

4. **Build candidate matrix** — for each primary direction:
   - `data_<axis>_<variant>` midpoint = pixel-projection centroid on that direction
   - `snap_<axis>_<variant>` midpoint = `snapshot.position` (true snap center)
   - 5 variants each: primary, rot-, rot+, trans-, trans+

   Typical total: **20-30 candidates**. Cap at `split_candidates_per_attempt`.

   Combinations include:
   - Snap direction × data midpoint (snap direction with pixel-projection centroid)
   - Snap direction × snap midpoint (pure snap)
   - PCA direction × data midpoint (pure PCA)
   - PCA direction × snap midpoint (cross-validation)

5. **Burn-in** — each candidate runs 50-iter perturbCell on both daughters with tight sigmas. Tracks total cost.

6. **Winner** — smallest total cost across candidates.

7. **Refine** — 30 extra perturb iters on winner with same tight sigmas.

8. **Bridge gate** — sample real-image brightness along `(d1 → d2)` axis:
   - **edge1** zone: inside d1, far from gap (`[-halfLen-r1Along, gapLo]`)
   - **gap** zone: middle (`[gapLo, gapHi]`, expanded to 30% of halfLen minimum)
   - **edge2** zone: inside d2, far from gap
   - `valleyRatio = gapBright / edgeBright`
   - `edgeAsymmetry = min(edge1, edge2) / max(edge1, edge2)`
   - **Reject asymmetric_edges** if `edgeAsymmetry < 0.4` (one daughter in dim/empty).
   - **Reject tier-1** if `valleyRatio >= 0.95` (no valley at all — one continuous blob).
   - **Reject tier-2** if `gapDensity > bio_bridge_max_gap_density (0.18) AND valleyRatio > bio_bridge_max_valley_ratio (0.85)`.

9. **Bio gate**:
   - `max(r1, r2) / min(r1, r2) > bio_daughter_size_ratio_max (1.5)` → reject
   - `(v1 + v2) / refParentVolume ∉ [0.6, 1.3]` → reject
   - `max(v1, v2) / refParentVolume > 0.65` → reject (one daughter mimics parent)
   - Either daughter buried in non-sibling → reject
   - Either daughter buried in sibling → reject

10. **Cost gate** — accept if `costDiff < -split_cost (=15)`. Else reject.

---

## Shape fit details

**Key design**: mask built from SNAPSHOT radii, not live. Mask stays fixed across iterations. Radii are free to roam. No volume floor, no per-axis floor. Rank-assignment keeps slot identity stable (a = largest eigenvalue, b = middle, c = smallest).

**If snap missing** (first frame, just-split daughter): mask falls back to current live radii.

### Biology — split along short axis

Cells pinch through their thinnest dimension:
- Rod (prolate): short axis direction = cleavage plane normal. Daughters end up stacked along the LONG axis, but the `splitAxisDir` in snapshot is the SHORT axis direction.
- Pancake (oblate): thin dimension is c; daughters stack through z.

`Ellipsoid::worldSplitAxis` returns the shortest semi-axis direction. Do NOT change this — it is biologically correct.

---

## Asymmetric L2 cost (Fix E)

Per-voxel squared error multiplied by `k = asymmetric_cost_weight (=6)` when `synth > real` (cell covers darker real region). Unweighted when `synth <= real`.

```cpp
sumSq = sum(diff²)                              // all pixels
posSumSq = sum(diff² where diff > 0)            // overshoot only
asymSumSq = sumSq + (k - 1) * posSumSq
return sqrt(asymSumSq)
```

SIMD-optimized via `cv::subtract`, `cv::multiply`, `cv::compare`, `cv::sum`.

**Why it matters:** parent ellipsoid covering a dark valley between two daughters pays amplified penalty. Two correctly-placed daughters avoid the valley (or their individual overshoots cancel). Makes the cost comparison reliable when the shape fit is correct.

---

## P(split) linear ramp

```cpp
t = clamp((snapElongation - 1.0) / (2.0 - 1.0), 0, 1)
P(split) = P_split_base + t * (P_split_max - P_split_base)
```

- `snapElong = 1.0` → P(split) = 0.03 (P_split_base)
- `snapElong = 1.5` → P(split) = 0.265
- `snapElong = 2.0` → P(split) = 0.50 (P_split_max, clamped above)

All cells get at least the base rate, so first-gen splits with near-round snaps still attempt. The unified loop's gates (bridge, bio, cost) handle rejection of bad attempts.

---

## Config knobs (config.yaml — current)

```yaml
cell:
  # Iterative PCA shape fit (snap-mask)
  pcaShapeMaxIters: 15
  pcaShapeRadiusScale: 2.236       # sqrt(5), uniform-ellipsoid variance
  pcaShapeMinPixels: 50
  pcaShapeMaskScale: 1.3
  pcaShapeConvergeRadius: 0.3
  pcaShapeConvergeAngleDeg: 2.0
  pcaShapeUpdatePosition: true
  pcaShapeMaxPosShiftFraction: 0.5

simulation:
  asymmetric_cost_weight: 6.0      # Fix E

prob:
  P_split_base: 0.03               # linear ramp base rate
  P_split_max: 0.5                 # linear ramp top (snapElong >= 2)

  split_cost: 15                   # min cost improvement to accept
  overlap_penalty_weight: 500.0

  split_candidates_per_attempt: 30
  split_candidate_burn_in_iterations: 50
  split_final_refine_iterations: 30
  split_calibration_iterations_per_cell: 50
  split_candidate_rotation_delta_degrees: 8
  split_candidate_translation_delta_fraction: 0.2

  bio_daughter_size_ratio_max: 1.5
  bio_combined_volume_min_fraction: 0.6
  bio_combined_volume_max_fraction: 1.3
  bio_max_single_daughter_volume_fraction: 0.65
  bio_bridge_max_gap_density: 0.18
  bio_bridge_max_valley_ratio: 0.85
  # Bridge tier-1 hardcoded at 0.95 in Frame.cpp

  split_burn_in_pos_sigma_scale: 0.4
```

---

## Deleted code / config (do not resurrect without reason)

| Item | Why deleted |
|---|---|
| `shape_elongation_classify_threshold` + classification | Thresholds couldn't cleanly separate; now linear ramp handles discrimination |
| `measureRadiusGradient` + radius gradient loop | Symmetric-shrink when rotation wrong; no mechanism to elongate |
| `measureRadiiFromImage` + radius EMA | Percentile-based, biased by 3D sampling distribution |
| `_birthFrame` + age gate | Never fired in any run |
| `_birthVolume` ratchet + volume floor | Blocked legitimate shrinkage |
| Per-axis snap-shrink floor (Fix C v1) | Same problem as volume floor — blocked aspect-ratio change |
| `size_reduction_penalty_weight` | Froze cells solid |
| `min_frames_before_split` | Declared but never read |
| `bio_max_midpoint_parent_fraction` | Pre-pass grounding replaced it |
| `split_direction_agreement_degrees` | Old dual-direction gate, now always-try-both |
| `pcaShapeBlend` | Iteration replaced EMA blending |
| Phase A / Phase B ordering | No measurable effect on split acceptance (3.3% vs 3.1% early/late) |

---

## Diagnostic log tags

- `[P(split)]` — per-frame P(split) per cell
- `[Calibration]` — per-cell position calibration result
- `[PCA Shape] cell=... iter=i` — shape fit iteration progress
- `[Pre-Pass] frame N allCells=M` — pre-pass invocation
- `[Pre-Pass Claims]` — Voronoi claim set per cell
- `[Pre-Pass] round=0 cell=...` — per-cell pre-pass result with shift from seed
- `[Split Winner]` — winning candidate after burn-in
- `[Split Refine]` — refine-iter result
- `[Split Bridge]` — bridge gate measurements (edge1, edge2, gap, vr, asym)
- `[Split Reject bio] reason=...` — bio/bridge rejection reason
- `[Split Reject cost]` — cost gate rejection
- `[Split Accepted]` — final split acceptance
- `[Split Dirs] selected=[axB, axC, imgPca]` — primary axes used for candidate generation
- `[Split AxisPlace]` — per-axis data midpoint computation
- `[Split Sigmas]` — burn-in sigma values

---

## Related docs

- `docs/changelogs/changelogv6.md` — per-fix change log with before/after code
- `docs/plans/2026-04-10-triaxial-pipeline-redesign.md` — design record (OUTDATED, references classification)
- `.claude/rules/algorithms.md` — (OUTDATED in places, see session notes)
- `.claude/rules/config.md` — (OUTDATED, references removed fields)
- `.claude/rules/gotchas.md` — read this for critical invariants
