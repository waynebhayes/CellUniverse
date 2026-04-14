# CellUniverse Pipeline (Current)

**Last updated:** 2026-04-14

This is the authoritative end-to-end pipeline for a single frame. It supersedes
the per-frame flow in `docs/plans/2026-04-10-triaxial-pipeline-redesign.md`,
which is preserved as the design record.

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
      Classify (T_classify = 1.15):
        pre_classified = cells with shapeElong > T
        non_classified = everything else
                             |
                             v
      Precompute P(split)_i (rank-rescaled by P_split_max)
                             |
                             v
      Seed expected daughters D1/D2 for pre_classified cells
                             |
                             v
================================================================
      STAGE 1 — POSITION CALIBRATION  (per cell)
================================================================
                             |
                             v
      Step 1: centroid jump (Voronoi-filtered bright pixels)
      Step 2: 50 perturbCell iterations (tight position sigma,
              radii frozen, rotation free)
                             |
                             v
================================================================
      STAGE 2 — ITERATIVE PCA SHAPE FIT  (per cell)
================================================================
                             |
                             v
      For each cell, up to pcaShapeMaxIters (=15) passes:
        1. Gather bright pixels inside maskScale*maxR sphere,
           Voronoi-filter, tighten to maskScale*current ellipsoid.
        2. If |mask| < pcaShapeMinPixels: stop.
        3. Weighted centroid mu, covariance C.
        4. cv::eigen(C) -> lambda0>=lambda1>=lambda2, e0,e1,e2.
        5. If lambda0/lambda2 < 1.1: skip rotation (degenerate).
        6. RANK ASSIGNMENT:
              a <- e0, b <- e1, c <- e2
              r_i = scale * sqrt(lambda_i)
              sign-align each e_i to current slot direction.
        7. R = [e0|e1|e2], force det=+1, decompose to Euler.
        8. Shift center toward mu, capped by
           pcaShapeMaxPosShiftFraction * maxR.
        9. Apply (no EMA). Converged when
           max|dR| < pcaShapeConvergeRadius AND
           max axis angle < pcaShapeConvergeAngleDeg.
                             |
                             v
================================================================
      STAGE 3 — IMAGE-GROUNDED DAUGHTER PRE-PASS
                (pre_classified cells only)
================================================================
                             |
                             v
      For each pre_classified cell:
        Gather bright pixels in bounding box (3 x maxR),
        Voronoi-filter vs all other claim sets.
        LOCAL-FRAME PCA on surviving pixels:
          localPos = R_inv * (worldPos - center)
          localPos /= (a, b, c)
          eigenvector -> un-normalize -> rotate to world.
        D1_exp, D2_exp = world-space centroids of the two halves.
                             |
                             v
================================================================
      STAGE 4 — PHASE A (non_classified cells)
================================================================
                             |
                             v
      Loop (non_classified * iterations_per_cell):
        Pick random non_classified cell.
        if rand < P(split): SPLIT ATTEMPT
        else:               perturbCell  (position + rotation)
                             |
                             v
================================================================
      STAGE 5 — PHASE B (pre_classified cells)
================================================================
                             |
                             v
      Loop (pre_classified * iterations_per_cell):
        Same split/perturb branch as Phase A.
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
      Freeze snapshot for frame N+1
                             |
                             v
      Save cells.csv + output PNG images
```

---

## Stage 2 — Iterative PCA Shape Fit

**Owner of shape.** Rotation + 3 radii + (optional) centroid are fit per frame
directly from the image. Replaces the old radius EMA + radius gradient paths,
both of which were deleted on 2026-04-14. Stochastic perturbation no longer
touches radii.

### Key design choices

| Choice | Reason |
|---|---|
| Iterate to convergence | Single-shot PCA with EMA blend converged at 30%/frame; thin cells stayed round for ~10 frames. Iteration converges in ~3–6 passes. |
| Ellipsoid mask (1.3 × current) | Sphere masks absorb neighbor halo after Voronoi. Ellipsoid mask uses the cell's own belief as a tighter prior. |
| No EMA blend | Iteration IS the stabilization; EMA only slowed it down. |
| Rank-based axis assignment | Greedy `\|dot\|` matching oscillated period-3. Rank (a=λ₀, b=λ₁, c=λ₂) is invariant. |
| Update position (centroid) | Position misalignment biases the mask → biased PCA. Capped shift keeps it stable. |
| Skip rotation when degenerate | Near-spherical cells have arbitrary eigvecs; don't rotate by noise. |

### Biology

Cells divide along the short axis (smallest variance direction = slot c).
The split-attempt path reads `axC` as the primary candidate, which is the
same axis identity this shape fit assigns. Consistent across both stages.

### Config knobs (`config.yaml` → `cell:`)

| Key | Default | Purpose |
|---|---|---|
| `pcaShapeMaxIters` | 15 | Per-frame, per-cell iteration cap |
| `pcaShapeRadiusScale` | 2.236 | `r = scale · sqrt(eigenvalue)` (√5 uniform-ellipsoid) |
| `pcaShapeMinPixels` | 50 | Stop iter if fewer than this in mask |
| `pcaShapeMaskScale` | 1.3 | Ellipsoid mask scale-up |
| `pcaShapeConvergeRadius` | 0.3 | Convergence threshold on max |Δr| (voxels) |
| `pcaShapeConvergeAngleDeg` | 2.0 | Convergence threshold on max axis rotation |
| `pcaShapeUpdatePosition` | true | Drive centroid from PCA |
| `pcaShapeMaxPosShiftFraction` | 0.5 | Per-iter cap: shift ≤ fraction · maxR |

### Logs

- `[PCA Shape] frame N cells=M maxIters=K scale=S minPixels=P maskScale=X updatePos=B`
- `[PCA Shape] cell=... iter=i n=N degen=0/1 R=(a,b,c) dR=Δ axisAng=deg posShift=S`
- `[PCA Shape] cell=... iter=i stop_too_few pixels=N min=P`

### Source

- `Frame::calibrateCellShapeViaPca` in `C++/src/Frame.cpp`
- Invoked from `CellUniverse::optimize` in `C++/src/CellUniverse.cpp`, after
  the position-calibration block, before the pre-pass.

---

## What stochastic perturbation still handles

`perturbCell` (stochastic Monte Carlo) now only touches:

- Position (x, y, z) — tight sigma during calibration, main-loop sigma in the
  unified Phase A / Phase B loops.
- Rotation (θx, θy, θz) — small refinement on top of the PCA-locked orientation.

Radii and brightness are driven by observation-based updates (PCA for shape,
EMA for brightness). The size-reduction penalty, radius EMA, radius gradient,
age gate, and birthVolume ratchet have all been deleted — see
`docs/changelogs/changelogv5.md` and `changelogv6.md` for the per-file removal
log.

---

## Deleted code paths (2026-04-14 cleanup)

| Item | Replaced by |
|---|---|
| `Ellipsoid::measureRadiiFromImage` + radius EMA loop | Iterative PCA shape fit |
| `Ellipsoid::measureRadiusGradient` + gradient loop | Iterative PCA shape fit |
| `ProbabilityConfig::size_reduction_penalty_weight` + `computeSizeReductionPenalty` | Nothing needed — PCA fits shape from the image |
| `Ellipsoid::_birthFrame` + age gate | Nothing needed — age gate never fired |
| `Ellipsoid::_birthVolume` ratchet | Nothing — ratchet blocked legitimate shrinkage |
| `ProbabilityConfig::min_frames_before_split` | Nothing — field was never read |

YAML dropped: `aRadius`/`bRadius`/`cRadius`/`brightness` perturbation blocks,
`radiusUpdateBlend`, `radiusMeasurementBrightnessThreshold`,
`radiusGradientStepSize`/`Iterations`, `pcaShapeBlend`, `pcaShapeScanFactor`,
`size_reduction_penalty_weight`, `min_frames_before_split`,
`aRadiusProbabilityStep/Trust` + b/c variants.

---

## Related docs

- `docs/plans/2026-04-10-triaxial-pipeline-redesign.md` — design record for the
  triaxial split pipeline, updated with the shape-fit section.
- `docs/changelogs/changelogv5.md` — through 2026-04-13 evening (radius EMA,
  radius gradient, first PCA rewrite).
- `docs/changelogs/changelogv6.md` — rank-assignment fix and any subsequent
  shape-fit evolution.
- `.claude/rules/algorithms.md`, `.claude/rules/config.md`,
  `.claude/rules/gotchas.md` — still describe parts of the pre-shape-fit
  world; defer to this doc when they disagree.
