# Triaxial Split Pipeline — Current Design

**Created:** 2026-04-10
**Last updated:** 2026-04-14 (shape-fit update)
**Branch:** `jl_triaxial_pipeline_yp_preprocess_merge_04112026` (shape work on a sub-branch)

> **2026-04-14 update:** Added the **iterative PCA shape fit** pass, which now owns shape
> (rotation + 3 radii + center). It runs between the position calibration and the pre-pass.
> Deleted code paths: radius EMA, radius gradient, size-reduction penalty, `_birthFrame`
> (age gate), `_birthVolume` ratchet, `min_frames_before_split`. See section
> [Iterative PCA Shape Fit](#iterative-pca-shape-fit-2026-04-14) below and
> `docs/pipeline.md` for the authoritative current pipeline.

---

## Summary

Triaxial ellipsoid model (`a != b != c`) with a phased split-detection pipeline driven by snapshot state from the previous frame. The split axis is the cell's **shortest fitted axis** (daughters separate through the thin dimension). PCA runs in the **cell's local frame** (inverse rotation + normalization by a, b, c) so the analysis is invariant to cell orientation and shape.

Key invariants:
- Snapshot (frozen end-of-previous-frame state) is the authority for split decisions.
- Both PCA and snapshot directions are **always tried** (10 candidates each, 20 total). Cost picks the winner.
- PCA operates in the cell's local coordinate frame, not world frame.
- Split axis = shortest fitted axis in local frame, rotated to world.
- Neighbor exclusion is phased: non-split cells settle in Phase A before split attempts in Phase B.
- Bio gates AND cost gate must both pass for a split to be accepted.
- **Per-frame shape is fit by iterative PCA BEFORE any perturbation or split attempt** (2026-04-14). Stochastic perturbation no longer touches radii.

---

## Pipeline Flow

```text
=================================================================
                        FRAME N START
=================================================================
                             |
                             v
        Load raw image, preprocess (ImageHandler pipeline)
                             |
                             v
        Compute adaptive background from frame N-1 cells
                             |
                             v
        Read snapshots for every cell from end of frame N-1:
          { center, a, b, c, rotation,
            splitAxisDir (shortest axis in world),
            splitAxisLength (shortest radius),
            shapeElongation = max(a,b,c) / min(a,b,c) }
        Snapshots are frozen for the entire frame.
                             |
                             v
        Classify cells using T_classify (1.15):
          pre_classified  = { cells with snapshot.shapeElong > T }
          non_classified  = everything else
                             |
                             v
        Precompute per-cell P(split):
          rawP_i     = P_split_base + max(0, 1 - 1/prevElong_i)
          P(split)_i = rawP_i * (P_split_max / max_i rawP_i)
                             |
                             v
        Seed expected daughter positions for pre_classified cells:
          D1_seed = snapshot.center - 0.5 * splitAxisLen * splitAxisDir
          D2_seed = snapshot.center + 0.5 * splitAxisLen * splitAxisDir

=================================================================
         CALIBRATION — per-cell position refinement
=================================================================
                             |
                             v
        For EVERY cell:
          Step 1 — centroid jump:
            Voronoi-filtered bright pixel weighted mean
            Accept if cost improves (position only)
          Step 2 — perturbation refinement:
            Tight position sigmas (0.4x), frozen radii
            50 iterations, accept if cost improves
            Restore main-loop sigmas

=================================================================
         ITERATIVE PCA SHAPE FIT  (added 2026-04-14)
=================================================================
                             |
                             v
        For EVERY cell, run up to pcaShapeMaxIters passes:
          1. Collect bright pixels in sphere(maskScale * maxR),
             Voronoi-filter vs other cell centers,
             tighten to (maskScale * current ellipsoid).
          2. If |pixels| < pcaShapeMinPixels: stop.
          3. Weighted centroid + 3x3 world-frame covariance.
          4. cv::eigen -> lambda0 >= lambda1 >= lambda2, e0, e1, e2.
          5. Degeneracy: if lambda0/lambda2 < 1.1 skip rotation update.
          6. RANK ASSIGNMENT:
                slot a <- e0, r_a = scale*sqrt(lambda0)
                slot b <- e1, r_b = scale*sqrt(lambda1)
                slot c <- e2, r_c = scale*sqrt(lambda2)
             Sign-align each e_i with current slot direction.
          7. Compose R = [e0|e1|e2], force det=+1, decompose to
             Euler angles (R = Rz*Ry*Rx).
          8. Shift center toward centroid, cap per-iter at
             pcaShapeMaxPosShiftFraction * maxR.
          9. Apply directly (no EMA). Check convergence:
             max|dR| < pcaShapeConvergeRadius AND
             max axis angle < pcaShapeConvergeAngleDeg.

        Output: rotation + 3 radii + center fit to the bright
        cloud before any perturbation / split attempt runs.

=================================================================
         PRE-PASS — ground expected daughters in the image
=================================================================
                             |
                             v
        For each pre_classified cell:
          Gather bright pixels in bounding box (3 x maxR)
          Voronoi exclude using claim-sets:
            non_classified  -> { snapshot.center }
            pre_classified  -> { D1_seed, D2_seed }
          LOCAL-FRAME PCA on surviving pixels:
            transform pixels by inverse rotation matrix
            normalize by (1/a, 1/b, 1/c)
            PCA eigenvector rotated back to world
          D1_exp, D2_exp = world-space centroids of two PCA halves
          (overwrites seeds with image-grounded estimates)

=================================================================
         PHASE A — non_classified cells
=================================================================
                             |
                             v
        Loop (non_classified count x iterations_per_cell):
          Pick random non_classified cell
          if rand < P(split): SPLIT ATTEMPT
          else:               PERTURBATION (standard)

=================================================================
         PHASE B — pre_classified cells
=================================================================
                             |
                             v
        Loop (pre_classified count x iterations_per_cell):
          Pick random pre_classified cell
          if rand < P(split): SPLIT ATTEMPT
          else:               PERTURBATION (standard)

=================================================================
         SPLIT ATTEMPT (same logic for Phase A and B)
=================================================================
                             |
                             v
        Save liveParent = cells[i]
        Install snapshotParent (snapshot state)
        Compute baseline = min(liveCost, snapCost)
                             |
                             v
        Gather bright pixels, Voronoi exclusion:
          Phase A neighbors: { snapshot.center }
          Phase B neighbors: { live_position } (settled)
                             |
                             v
        LOCAL-FRAME PCA on pixels:
          Transform pixels: localPos = R_inv * (worldPos - center)
          Normalize: localPos /= (a, b, c)
          PCA on normalized local positions
          Eigenvector un-normalized (scale by a,b,c) and
          rotated back to world space
          -> dirPca, D1_exp, D2_exp (world-space)
                             |
                             v
        ALWAYS-DUAL-DIRECTION CANDIDATES:
          Direction 1: dirPca (from local-frame PCA)
          Direction 2: snapshot.splitAxisDir (shortest axis)
          Both always tried, no angle threshold.
          10 candidates per direction = 20 total.
                             |
                             v
        Two midpoint options per direction:
          pca_mid  = 0.5 * (D1_exp + D2_exp)
          snap_mid = snapshot.position
          (skip snap_mid if |pca_mid - snap_mid| < 0.5)
                             |
                             v
        Generate candidates:
          For each (direction x midpoint):
            primary + 2 rotations + 2 translations = 5
          Total: up to 20 candidates (2 dirs x 2 mids x 5)
                             |
                             v
        Per-candidate burn-in (50 iterations)
          Tight sigmas: position 0.4x, radius 0.1x
        Select best by total cost
                             |
                             v
        Final refine (30 iterations) on winner
                             |
                             v
        BRIDGE BRIGHTNESS GATE:
          Project Voronoi pixels onto daughter separation axis
          gap = middle 30%, edges = 60-110%
          gap_density = gapCount / total
          valley_ratio = gap_bright / edge_bright
          REJECT only if BOTH:
            gap_density > 0.18 AND valley_ratio > 0.85
                             |
                             v
        BIO CHECK:
          Size ratio, combined volume,
          single-daughter volume (max/parent < 0.65),
          buried-in-other, buried-in-sibling
                             |
                   +-- PASS --+-- FAIL --+
                                          |
                                     restoreLiveParent
                                     blacklist cell
                                     compensatory perturb
                             |
                             v
        COST CHECK:
          costDiff < -split_cost (-15)?
                             |
                   +-- PASS --+-- FAIL --+
                   |                     |
                   v                     v
                ACCEPT              restoreLiveParent
                (daughters          blacklist cell
                 installed)         compensatory perturb

=================================================================
                     END OF FRAME N
=================================================================
                             |
                             v
        Brightness EMA update (per-cell, from real image)
                             |
                             v
        Freeze snapshot for frame N+1
                             |
                             v
        Save cells.csv + output images
```

---

## Split Axis: Shortest Fitted Axis

The snapshot split axis is the **shortest** of the three fitted radii (a, b, c), identified in the cell's local frame and rotated to world space via `Ellipsoid::worldSplitAxis()`. Daughters separate along this axis.

For a pancake cell (a ~ b >> c): split axis = c direction (daughters stacked through thin dimension).
For a prolate cell (a >> b ~ c): split axis = b or c direction (daughters stacked perpendicular to elongation).

Note: when two radii are equal (e.g., a=c), the cell is rotationally symmetric around the b-axis. The rotation around b is unconstrained and drifts freely, making the snapshot direction unreliable. This is why PCA (in local frame) is always tried as an independent direction source.

---

## Local-Frame PCA

All PCA in the split pipeline operates in the cell's local coordinate frame:

1. **Transform**: `localPos = R_inverse * (worldPos - cellCenter)`
2. **Normalize**: `localPos.x /= a, localPos.y /= b, localPos.z /= c`
3. **PCA**: compute weighted covariance and eigenvectors in normalized local space
4. **Un-normalize**: scale eigenvector by (a, b, c) to restore spatial proportions
5. **Rotate back**: apply forward rotation to get world-space direction
6. **Centroids**: D1/D2 computed in world space from the original pixel positions

This makes PCA invariant to cell orientation and shape. A flat cell lying in xy no longer produces a z-dominant eigenvector from its thinness — all axes are normalized to unit scale before PCA runs.

---

## Direction Selection

**Always try both directions.** No angle threshold. 10 candidates per direction = 20 total.

| Direction | Source | Candidates |
|-----------|--------|------------|
| PCA | Local-frame PCA eigenvector (from current image) | 10 |
| Snapshot | Shortest fitted axis from previous frame | 10 |

This covers all cases: cells where PCA and snapshot agree, cells where they disagree, and cells where one source is unreliable (e.g., a=c rotation drift).

---

## Neighbor Exclusion Matrix

| Splitting cell | Neighbor type | Exclusion source |
|---------------|--------------|-----------------|
| Phase A | non_classified | `snapshot.center` |
| Phase A | pre_classified | `{D1_exp, D2_exp}` |
| Phase B | non_classified | `live_position` (settled in A) |
| Phase B | pre_classified, not yet processed | `{D1_exp, D2_exp}` |
| Phase B | pre_classified, split accepted | `live_position` of daughters |
| Phase B | pre_classified, split rejected | `live_position` of parent |

---

## Config Fields (current values)

### Split classification and probability

| Field | Value | Purpose |
|-------|-------|---------|
| `P_split_base` | 0.03 | Base per-iteration split probability |
| `P_split_max` | 0.5 | Ceiling after proportional rescale |
| `shape_elongation_classify_threshold` | 1.15 | Phase A/B classification threshold |

### Candidate generation

| Field | Value | Purpose |
|-------|-------|---------|
| `split_candidate_burn_in_iterations` | 50 | Burn-in iterations per candidate |
| `split_candidate_rotation_delta_degrees` | 8 | Rotation variant offset |
| `split_candidate_translation_delta_fraction` | 0.2 | Translation variant as fraction of daughterR |
| `split_final_refine_iterations` | 30 | Refine iterations on winning candidate |

### Perturbation

| Field | Value | Purpose |
|-------|-------|---------|
| Position sigma (x/y/z) | 5/5/8 | Gaussian offset per perturbation step |
| Rotation sigma | 0.05 | Radians per step |

Radius perturbation is no longer used — radii are owned by the iterative PCA shape fit.

### Bio gates

| Field | Value | Purpose |
|-------|-------|---------|
| `bio_daughter_size_ratio_max` | 1.5 | Reject if one daughter >> other |
| `bio_combined_volume_min_fraction` | 0.6 | Reject if daughters too small combined |
| `bio_combined_volume_max_fraction` | 1.3 | Reject if daughters too large combined |
| `bio_max_single_daughter_volume_fraction` | 0.65 | Reject if one daughter mimics parent |
| `bio_bridge_max_gap_density` | 0.18 | Bridge gate gap density limit |
| `bio_bridge_max_valley_ratio` | 0.85 | Bridge gate valley ratio limit |

### Cost and penalties

| Field | Value | Purpose |
|-------|-------|---------|
| `split_cost` | 15 | Minimum cost improvement to accept split |
| `overlap_penalty_weight` | 500 | Quadratic overlap penalty weight |

### Iterative PCA shape fit (added 2026-04-14)

| Field | Value | Purpose |
|-------|-------|---------|
| `pcaShapeMaxIters` | 15 | Maximum per-frame, per-cell iterations |
| `pcaShapeRadiusScale` | 2.236 | `r = scale * sqrt(eigenvalue)` (sqrt(5) uniform-ellipsoid) |
| `pcaShapeMinPixels` | 50 | Stop iter if mask below this count |
| `pcaShapeMaskScale` | 1.3 | Ellipsoid mask scale-up (× current radii) |
| `pcaShapeConvergeRadius` | 0.3 | Per-iter max radius delta for convergence (voxels) |
| `pcaShapeConvergeAngleDeg` | 2.0 | Per-iter max axis-rotation for convergence (degrees) |
| `pcaShapeUpdatePosition` | true | Use PCA centroid to drive center (capped) |
| `pcaShapeMaxPosShiftFraction` | 0.5 | Per-iter center-shift cap (× maxR) |

### Brightness

| Field | Value | Purpose |
|-------|-------|---------|
| `brightnessUpdateBlend` | 0.95 | EMA blend factor (per-frame) |
| `brightnessMeanAmplification` | 1.1 | Multiplier on measured brightness |
| `brightnessMeasurementTopPercentile` | 0.3 | Use top 30% brightest in-cell voxels for EMA |

---

## Removed Gates & Code Paths

| Item | Reason for removal |
|------|-------------------|
| **Drift-from-seed** | All false splits it caught were also caught by cost gate. Blocked real splits for spherical cells with poor PCA seed placement. Code and config fully removed. |
| **Shape EMA** | PCA inside a single undivided cell sees symmetric pixels — can't detect pre-division elongation. Replaced by iterative PCA shape fit (2026-04-14). |
| **Radius gradient** (2-pole sampling per axis) | Symmetric-shrink when rotation was wrong; no mechanism to elongate. Deleted 2026-04-14. |
| **Radius EMA** (percentile-based observed-radius blend) | Biased by 3D sampling distribution. Deleted 2026-04-14. |
| **`size_reduction_penalty_weight`** | Dim cells froze solid. Deleted 2026-04-14. |
| **`_birthFrame` / age gate** | Never fired. Deleted 2026-04-14. |
| **`_birthVolume` ratchet** | One-way growth lock prevented legitimate shrinkage. Deleted 2026-04-14. |
| **`min_frames_before_split`** | Declared, parsed, never read. Deleted 2026-04-14. |
| **Midpoint-near-parent** | Pre-pass grounding undermines it. |
| **10-degree direction threshold** | Replaced by always-dual-direction. |

---

## Iterative PCA Shape Fit (2026-04-14)

Shape (rotation + 3 radii + optional center) is now fit directly from the bright-pixel cloud each frame, before any perturbation or split attempt. Replaces radius EMA and radius gradient (both deleted).

**Why:** stochastic perturbation and gradient-based updates cannot reliably align a thin/elongated bright streak inside a round-initialized ellipsoid. L2 cost provides weak signal for dim cells. PCA reads shape directly from the variance of bright pixels.

**Per-iteration algorithm** (see `Frame::calibrateCellShapeViaPca` in `src/Frame.cpp`):

1. Collect pixels in a sphere of radius `maskScale * max(a,b,c)` around cell center; Voronoi-filter against all other cells' centers; tighten to `maskScale * current ellipsoid` via local-frame test.
2. Stop if fewer than `pcaShapeMinPixels` remain.
3. Compute weighted centroid and 3×3 covariance (weight = brightness − background).
4. `cv::eigen` returns eigenvalues `λ₀ ≥ λ₁ ≥ λ₂` and eigenvectors `e₀, e₁, e₂`.
5. **Degeneracy check:** if `λ₀/λ₂ < 1.1` the cell is near-spherical; skip rotation update.
6. **Rank assignment (stable):** `a ← e₀`, `b ← e₁`, `c ← e₂` (no greedy matching). Sign-flip each `eᵢ` so its dot with current slot-i direction is ≥ 0.
7. `rᵢ = pcaShapeRadiusScale · √λᵢ`.
8. Compose `R = [e₀ | e₁ | e₂]`, force `det = +1`, decompose to Euler `(θx, θy, θz)` under `R = Rz·Ry·Rx`.
9. Shift center toward centroid; cap shift at `pcaShapeMaxPosShiftFraction · maxR`.
10. Apply directly (no EMA). Converged when `max|Δr| < pcaShapeConvergeRadius` AND max axis angle < `pcaShapeConvergeAngleDeg`.

**Rank assignment vs greedy match:** the first iteration of the rewrite used greedy `|dot|` matching between PCA axes and current slot axes. On near-degenerate cells the label assignment flipped each iter, producing period-3 oscillation (15 iters without convergence). Rank assignment is invariant: the largest eigenvalue always lands in slot a. Physical tracking across frames is via rotation + radii, not slot labels — splits still target `axC` (shortest) correctly.

**Biology:** cells divide along the short axis = smallest eigenvalue direction = slot c. Shape fit and split geometry share this convention.

**Diagnostic logs:**

- `[PCA Shape] frame N cells=M maxIters=K scale=S minPixels=P maskScale=X updatePos=B`
- `[PCA Shape] cell=... iter=i n=N degen=0/1 R=(a,b,c) dR=Δ axisAng=deg posShift=S`
- `[PCA Shape] cell=... iter=i stop_too_few pixels=N min=P`

**Effect:**

- Cells cannot inflate past their real bright extent — PCA reads actual variance.
- Thin streaks in round-initialized cells converge to elongated ellipsoids within ~3–6 iters instead of ~10 frames of stochastic drift.
- Rotation locks onto bright axis direction in the first iteration for non-degenerate cells.

---

## Backburner

**Drift+bridge baseline fallback** — when `parentDrift > parentMaxR` AND `bridge_vr < 0.60`, use snapshot baseline instead of `min(liveCost, snapCost)`. Designed to fix the 1f89ab0 f28 case where the live parent drifts onto one daughter blob, making its cost artificially good. Analysis showed this would accept the real split (diff=-18.9) without accepting false splits (8cbdf bridge_vr > 0.65 at high drift). Not yet implemented.

---

## Known Limitations

1. **1-frame split delays** — stochastic burn-in sometimes produces poor candidates. Affects ~30% of splits. Not fixable by gate tuning without risking false splits.
2. **Parent drift onto daughter blob** — live parent drifts to sit on one daughter, making its cost artificially good. See backburner item above.
3. **a=c rotation drift** — when two radii are equal, rotation around the unique axis is unconstrained. Snapshot direction drifts to arbitrary world direction. Mitigated by always-dual-direction (PCA provides the correct direction from image).
4. **Second-generation cells harder to split** — smaller cells in denser neighborhoods produce weaker cost signals and noisier bridge profiles.
