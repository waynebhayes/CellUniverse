# Triaxial Pipeline Redesign

**Date:** 2026-04-10
**Status:** Design locked, not yet implemented
**Branch target:** `yp_yd_merge_04072026`

---

## Summary

Replace the oblate spheroid model (`a = b`, `c`) with a triaxial ellipsoid (`a ≠ b ≠ c`) and restructure the split-detection pipeline around a snapshot-driven, phased loop.

The driver for this redesign is a structural blind spot in the current pipeline: the oblate fit cannot represent pancaking cells, so neither PCA nor Pillar B can reliably detect pre-split shape at end of frame. With a triaxial model, the fitted shape itself carries the elongation signal (`max(a,b,c) / min(a,b,c)`), and the split-detection code simplifies dramatically.

Key invariants this design preserves:

- Randomness lives in per-iteration Monte Carlo, not in split classification.
- The snapshot (frozen end-of-previous-frame state) is the authority for split decisions during the frame, decoupling detection from the parent's live drift.
- Every split attempt uses image-based PCA for daughter position; pre-classified splits additionally trust the snapshot's fitted direction.
- Neighbor exclusion is phased so that live positions are trustworthy when splitting cells need them.
- Both bio check AND cost check must pass for a split to be accepted. Bio check runs first (cheap, geometric sanity); cost check runs second (expensive, only if bio passes).

---

## Pipeline Flow

```text
=================================================================
                        FRAME N START
=================================================================
                             |
                             v
        Load raw image, preprocess (single pipeline —
        assume shape is cleanly visible)
                             |
                             v
        Compute adaptive background from frame N-1 cells
                             |
                             v
        Read snapshots for every cell from end of frame N-1:
          { center, a, b, c, rotation,
            longAxisDir, longAxisLength,
            shapeElongation = max(a,b,c) / min(a,b,c) }
        Snapshots are frozen for the entire frame.
                             |
                             v
        Classify cells using T_classify:
          pre_classified  = { cells with snapshot.shapeElong > T_classify }
          non_classified  = everything else
                             |
                             v
        Precompute per-cell P(split):
          rawP_i     = P_split_base + max(0, 1 - 1 / prevElong_i)
          P(split)_i = rawP_i * (P_split_max / max_i rawP_i)
          P_split_base and P_split_max are config fields.
                             |
                             v
        Precompute SEED expected daughter positions for every
        pre_classified cell (will be refined in the next step):
          D1_seed = snapshot.center - 0.5 * longAxisLength * longAxisDir
          D2_seed = snapshot.center + 0.5 * longAxisLength * longAxisDir

                             |
                             v

=================================================================
         PRE-PASS — ground expected daughters in the image
=================================================================
                             |
                             v
        For each pre_classified cell i:
          bounding box at snapshot.center of i
          gather bright pixels in the box from current image
          Voronoi exclude using SEED claim-sets for other cells:
            non_classified other  -> { snapshot.center }
            pre_classified other  -> { D1_seed, D2_seed }
          PCA on the surviving pixels
          project onto PCA eigenvector 1, split at median
          D1_exp_i = centroid of group 1
          D2_exp_i = centroid of group 2
          (overwrites the seed with an image-grounded estimate)

        Optional: iterate the pre-pass using D*_exp from round k
        to build the claim-sets for round k+1. One pass is usually
        enough; only iterate if diagnostics show estimates moving
        significantly between rounds.

        After the pre-pass, every pre_classified cell has an
        image-grounded { D1_exp, D2_exp } pair that reflects the
        actual bright masses in the current frame, not a
        snapshot extrapolation.

=================================================================
         CALIBRATION — per-cell position refinement
=================================================================
                             |
                             v
        For EVERY cell (pre_classified AND non_classified):

          Step 1 — analytic centroid jump:
            build claim-set for this cell (same rules as pre-pass
              but frame-start state):
                pre_classified other -> { D1_exp, D2_exp }
                non_classified other -> { snapshot.center }
            gather Voronoi-filtered bright pixels
            centroid = weighted_mean(pixels)
            (this is the analytic midpoint — same as the mean
             pca3DWithCentroids computes internally, which is the
             midpoint between daughters for a symmetric division)
            construct candidate Spheroid at centroid position
              (radii, rotation, brightness inherited from current cell)
            compute L2 cost at centroid position incrementally
            if cost(centroid) < cost(current):
              install candidate (POSITION ONLY changes)
            else:
              revert, keep current position

          Step 2 — perturbation refinement:
            install TIGHT position sigmas (posScale ~0.4)
            FREEZE radius sigmas (radiusScale = 0)
            loop M iterations (split_calibration_iterations_per_cell):
              perturbCell(i) with tight sigmas, accept if cost improves
            restore main-loop sigmas

        After calibration:
          - Each cell sits at (or near) its best single-ellipsoid
            fit position at snapshot-size radii
          - For a dividing cell, this is the center-of-mass of both
            incipient daughters (the snapshot-size parent cannot
            collapse onto one daughter)
          - For a non-dividing cell, this is the true cell center
          - Phase A / Phase B split attempts now see a correctly
            centered, correctly sized baseline parent instead of a
            half-collapsed drifted one

=================================================================
         PHASE A — iterate non_classified cells
=================================================================
                             |
                             v
        Loop (len(non_classified) * iterations_per_cell) times:
          pick a random non_classified cell (skip blacklisted)
          roll rand ~ U(0,1)
          if rand < P(split)_i:   -> SPLIT ATTEMPT (snapshot-only)
          else:                    -> PERTURBATION

        +--------------------+--------------------+
        |                                         |
        v                                         v
  PERTURBATION                            SPLIT ATTEMPT (A)
  perturbCell(i)                          Save liveParent = cells[i]
  overlap + size-reduction                Build snapshotParent from
  accept/revert                             snapshot.{pos, a, b, c,
                                              theta, brightness}
                                          cells[i] = snapshotParent
                                          Re-render synth via
                                            generateSynthFrameFast
                                            (live -> snapshot)
                                          Baseline cost now reflects
                                            snapshot-state parent
                                                |
                                                v
                                          Bounding box at
                                            snapshot.center of i
                                          Gather bright pixels in
                                            box from current image
                                                |
                                                v
                                          Voronoi exclusion —
                                          EVERY other cell uses
                                          SNAPSHOT data:
                                            non_classified:
                                              { snapshot.center }
                                            pre_classified:
                                              { D1_exp, D2_exp }
                                                |
                                                v
                                          direction_0 = PCA eigvec 1
                                            (no snapshot direction —
                                             this cell was not
                                             elongated last frame)
                                          D1_0, D2_0 = PCA centroids
                                                |
                                                v
                                          Candidate midpoints (two
                                          options tried in parallel):
                                            pca_mid  = 0.5*(D1_0+D2_0)
                                            snap_mid = snapshot.center
                                          (skip snap_mid if
                                           |pca_mid - snap_mid| < 0.5)
                                                |
                                                v
                                          Generate K candidates:
                                            for each (midpoint, dir0):
                                              primary + 2 rotations
                                              + 2 translations
                                          Per-candidate burn-in with
                                            tight position sigmas AND
                                            tight radius sigmas (0.1x)
                                          Select best by post-burn-in
                                          cost (label: pca_mid / snap_mid)
                                                |
                                                v
                                          FINAL REFINE on winner:
                                            M more iterations of
                                            perturbCell on the two
                                            daughters (tight sigmas)
                                                |
                                                v
                                          Drift-from-seed gate
                                            reject if |final - seed|
                                              > max(0.4*srcMaxR,
                                                    0.8*daughterMaxR)
                                                |
                                                v
                                          Bridge brightness gate:
                                            project Voronoi pixels
                                            onto split axis,
                                            bin into gap (|t|<0.3)
                                            and edges (0.6<|t|<1.1)
                                            gap_density  = gapCount / total
                                            valley_ratio = gap_bright /
                                                           edge_bright
                                            REJECT only if BOTH
                                              gap_density > 0.18 AND
                                              valley_ratio > 0.85
                                            (catches "flat profile"
                                             false splits that have
                                             no dividing groove)
                                                |
                                                v
                                          Bio check
                                          (size ratio,
                                           combined volume,
                                           single-daughter volume
                                             max / parent < 0.65
                                             (catches "one daughter
                                              mimics parent" pattern),
                                           buried-in-other,
                                           buried-in-sibling)
                                                |
                                     +- PASS -+- FAIL -+
                                                        |
                                                        v
                                                  restoreLiveParent();
                                                  blacklist i;
                                                  perturb i once
                                          Cost check:
                                            costDiff < -split_cost?
                                                |
                                     +- PASS -+- FAIL -+
                                     |                 |
                                     v                 v
                                  ACCEPT           restoreLiveParent();
                                  (cells[i] gone,  blacklist i;
                                   daughters       perturb i once
                                   installed)

        At the end of Phase A, every non_classified cell has
        consumed its per-cell iteration budget. Their live
        positions are fresh and trustworthy.

=================================================================
         PHASE B — iterate pre_classified cells
=================================================================
                             |
                             v
        Loop (len(pre_classified) * iterations_per_cell) times:
          pick a random pre_classified cell (skip blacklisted)
          roll rand ~ U(0,1)
          if rand < P(split)_i:   -> SPLIT ATTEMPT (pre_classified)
          else:                    -> PERTURBATION

        +--------------------+--------------------+
        |                                         |
        v                                         v
  PERTURBATION                            SPLIT ATTEMPT (B)
  perturbCell(i)                          Save liveParent = cells[i]
  overlap + size-reduction                Build snapshotParent from
  accept/revert                             snapshot.{pos, a, b, c,
                                              theta, brightness}
                                          cells[i] = snapshotParent
                                          Re-render synth via
                                            generateSynthFrameFast
                                            (live -> snapshot)
                                          Baseline cost now reflects
                                            snapshot-state parent
                                                |
                                                v
                                          Bounding box at
                                            snapshot.center of i
                                          Gather bright pixels in
                                            box from current image
                                                |
                                                v
                                          Voronoi exclusion —
                                          other cells use:
                                            non_classified:
                                              { live_position }
                                              (settled in Phase A)
                                            pre_classified, not yet
                                            processed in B:
                                              { D1_exp, D2_exp }
                                            pre_classified, split
                                            already accepted:
                                              { live_position } of
                                              each daughter
                                            pre_classified, split
                                            already rejected:
                                              { live_position } of
                                              the parent
                                                |
                                                v
                                          Compute two direction seeds:
                                            dir_snap = snapshot
                                                       longAxisDir
                                            dir_pca  = PCA eigvec 1
                                          Compare angles:
                                            < T_dir_agree:
                                              primary = dir_snap only
                                            >= T_dir_agree:
                                              primaries = {dir_snap,
                                                           dir_pca}
                                            (try both directions,
                                             let best-cost decide)
                                                |
                                                v
                                          Candidate midpoints (two
                                          options tried in parallel):
                                            pca_mid  = 0.5*(D1_exp+D2_exp)
                                            snap_mid = snapshot.center
                                          (skip snap_mid if
                                           |pca_mid - snap_mid| < 0.5)
                                                |
                                                v
                                          Generate K candidate
                                          placements for each
                                          (midpoint, direction) pair:
                                            primary + 2 rotations
                                            + 2 translations
                                          Per-candidate burn-in with
                                            tight position sigmas AND
                                            tight radius sigmas (0.1x)
                                          Select best by post-burn-in
                                          cost (label: pca_mid / snap_mid)
                                                |
                                                v
                                          FINAL REFINE on winner:
                                            M more iterations of
                                            perturbCell on the two
                                            daughters (tight sigmas)
                                                |
                                                v
                                          Drift-from-seed gate
                                            reject if |final - seed|
                                              > max(0.4*srcMaxR,
                                                    0.8*daughterMaxR)
                                                |
                                                v
                                          Bridge brightness gate:
                                            project Voronoi pixels
                                            onto split axis,
                                            bin into gap (|t|<0.3)
                                            and edges (0.6<|t|<1.1)
                                            gap_density  = gapCount / total
                                            valley_ratio = gap_bright /
                                                           edge_bright
                                            REJECT only if BOTH
                                              gap_density > 0.18 AND
                                              valley_ratio > 0.85
                                            (catches "flat profile"
                                             false splits that have
                                             no dividing groove)
                                                |
                                                v
                                          Bio check
                                          (size ratio,
                                           combined volume,
                                           single-daughter volume
                                             max / parent < 0.65
                                             (catches "one daughter
                                              mimics parent" pattern),
                                           buried-in-other,
                                           buried-in-sibling)
                                                |
                                     +- PASS -+- FAIL -+
                                                        |
                                                        v
                                                  restoreLiveParent();
                                                  blacklist i;
                                                  perturb i once
                                          Cost check:
                                            costDiff < -split_cost?
                                                |
                                     +- PASS -+- FAIL -+
                                     |                 |
                                     v                 v
                                  ACCEPT           restoreLiveParent();
                                  (cells[i] gone,  blacklist i;
                                   daughters       perturb i once
                                   installed)

=================================================================
                     END OF FRAME N
=================================================================
                             |
                             v
        For every cell, compute and freeze the snapshot for
        frame N+1:
          { center, a, b, c, rotation,
            longAxisDir, longAxisLength, shapeElongation }
                             |
                             v
        Save cells.csv + output images (real overlay + synth)
                             |
                             v
                     FRAME N+1 START
```

---

## Neighbor Exclusion Matrix

Voronoi claim-set assignment for each cell-pair combination:

| Splitting cell (self)    | Neighbor (other)                   | Exclusion source for neighbor                |
| ------------------------ | ---------------------------------- | -------------------------------------------- |
| Phase A — non_classified | non_classified                     | `snapshot.center`                            |
| Phase A — non_classified | pre_classified                     | `{D1_exp, D2_exp}` (image-grounded pre-pass) |
| Phase B — pre_classified | non_classified                     | **`live_position`** (settled in Phase A)     |
| Phase B — pre_classified | pre_classified, not yet processed  | `{D1_exp, D2_exp}` (image-grounded pre-pass) |
| Phase B — pre_classified | pre_classified, split accepted     | `live_position` of each daughter             |
| Phase B — pre_classified | pre_classified, split rejected     | `live_position` of parent                    |

A pixel belongs to the splitting cell iff its nearest claim-set point across all cells is one of that cell's own points. Pixels claimed by any other cell are dropped from the PCA input.

---

## Direction Source by Cell Category

| Cell category                                             | Direction primary                       | PCA as additional candidate?                            |
| --------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------- |
| non_classified (snapshot round, no meaningful axis)       | PCA eigenvector 1                       | N/A — PCA is already the primary                        |
| pre_classified, `angle(dir_snap, dir_pca) < T_dir_agree`  | snapshot `longAxisDir`                  | No, directions agree                                    |
| pre_classified, `angle(dir_snap, dir_pca) >= T_dir_agree` | snapshot `longAxisDir` AND PCA eigvec 1 | Yes — both are primary candidates, burn-in picks winner |

---

## Midpoint Source for Candidate Generation

Each split attempt generates candidates around TWO midpoint options and lets burn-in + cost comparison pick the winner. The two options are tried as separate "primary placements" within the candidate loop:

| Midpoint option | Center                                         | Separation                     | Label used in logs |
| --------------- | ---------------------------------------------- | ------------------------------ | ------------------ |
| `pca_mid`       | `0.5 * (D1_exp + D2_exp)` (image centroid)     | `\|D1_exp - D2_exp\|` (PCA sep)  | `pca_mid_primary`, `pca_mid_rot±`, `pca_mid_trans±`    |
| `snap_mid`      | `snapshot.position` (last-frame cell center)   | `snapshot.longAxisLength`       | `snap_mid_primary`, `snap_mid_rot±`, `snap_mid_trans±` |

`snap_mid` is **skipped** when `|pca_mid - snap_mid| < 0.5` voxels (the two are essentially the same point). Otherwise both are tried, giving up to `2 midpoints × N directions × 5 variants` candidates per split attempt — capped at `split_candidates_per_attempt` (default 10).

The winning candidate's label is stored in `bestLabel` and printed in the `[Split Accepted]` / `[Split Reject cost]` log lines so the diagnostic logs show which midpoint type consistently wins for real splits.

---

## Config Fields

### Split classification and probability

| Field                                 | Purpose                                                                            | Starting value |
| ------------------------------------- | ---------------------------------------------------------------------------------- | -------------- |
| `P_split_base`                        | Base per-iteration split probability for any cell, before elongation scaling       | `0.03`         |
| `P_split_max`                         | Ceiling after proportional rescale across all cells                                | `0.5`          |
| `shape_elongation_classify_threshold` | Snapshot `max(a,b,c)/min(a,b,c)` above which a cell is pre-classified as splitting | `~1.20`        |

### Direction handling

| Field                               | Purpose                                                                         | Starting value |
| ----------------------------------- | ------------------------------------------------------------------------------- | -------------- |
| `split_direction_agreement_degrees` | Above this angle, PCA direction is added as an extra candidate primary          | `~20°`         |

### Expected-daughter pre-pass

| Field                                   | Purpose                                                                     | Starting value |
| --------------------------------------- | --------------------------------------------------------------------------- | -------------- |
| `expected_daughter_pre_pass_iterations` | Refinement rounds for the frame-start pre-pass (1 = no iteration)           | `1`            |

### Candidate placement refinement

| Field                                         | Purpose                                                     | Starting value |
| --------------------------------------------- | ----------------------------------------------------------- | -------------- |
| `split_candidates_per_attempt`                | K, number of candidate placements per split attempt         | `~5`           |
| `split_candidate_burn_in_iterations`          | M, short burn-in iterations per candidate                   | `~20`          |
| `split_candidate_rotation_deltas_degrees`     | Small rotation offsets for candidate generation             | `±5, ±10`      |
| `split_candidate_translation_deltas_fraction` | Small translation offsets as fraction of daughterR          | `±0.2`         |

### Bio check (generic sphere-like, first gate — must pass before cost check)

| Field                              | Purpose                                                                                                                          | Starting value  |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `bio_daughter_size_ratio_max`      | Reject if `max(d1R, d2R) / min(d1R, d2R) > this`                                                                                 | `1.5`           |
| `bio_combined_volume_min_fraction` | Reject if combined daughter volume `< this * parent volume`                                                                      | `0.6`           |
| `bio_combined_volume_max_fraction` | Reject if combined daughter volume `> this * parent volume`                                                                      | `1.3`           |
| `bio_daughter_buried_threshold`    | Reject if either daughter's center is inside another cell's ellipsoid — **must include the sibling daughter as "another cell"**  | (boolean check) |

### Preserved from existing pipeline

| Field                                     | Purpose                                                                 |
| ----------------------------------------- | ----------------------------------------------------------------------- |
| `iterations_per_cell`                     | Per-cell iteration budget within each phase                             |
| `split_cost`                              | Minimum cost improvement to accept a split (second gate, after bio)     |
| `overlap_penalty_weight`                  | Weight on quadratic overlap penalty                                     |
| `size_reduction_penalty_weight`           | Weight on shrinkage penalty during perturbation                         |
| `minMajorRadius` / `maxMajorRadius`       | Hard bounds on `a` axis                                                 |
| `minMinorRadius` / `maxMinorRadius`       | Hard bounds on `c` axis                                                 |
| (new) `minBAxisRadius` / `maxBAxisRadius` | Hard bounds on `b` axis — needed because triaxial breaks `a = b`        |

### Removed from existing pipeline

These exist today to work around the oblate model's blindness and become dead under the triaxial design:

- `split_elongation_threshold` (replaced by `shape_elongation_classify_threshold`)
- `split_fake_overlap_volume_fraction_threshold`
- `split_fake_radius_ratio_threshold` (subsumed by `bio_daughter_size_ratio_max`)
- `split_fake_bridge_brightness_similarity_threshold`
- `split_minor_axis_alignment_tolerance_degrees`
- `split_minor_axis_alignment_flatness_ratio_threshold`
- `split_pre_burn_in_min_separation_over_major`
- `split_pre_burn_in_z_axis_max_abs`
- `split_pre_burn_in_z_axis_max_separation_over_major`
- `split_pre_burn_in_z_axis_min_drift_over_major`
- `split_post_burn_in_large_recenter_min_drift_over_major`
- `split_post_burn_in_large_recenter_max_cost_diff`
- `split_burn_in_iterations` (superseded by `split_candidate_burn_in_iterations` × K)
- `post_sigmoid_dimmest_percentile` and all related post-sigmoid dim-subtract fields (assuming the new preprocessing pipeline no longer needs them; verify during implementation)

---

## Edge Cases

Cases the design handles correctly, cases it does not, and cases it does not try to handle.

### Handled correctly

**Parent drifts onto one daughter during Phase B perturbations.**
Split detection uses `snapshot.center` for the bounding box and trusts the snapshot direction for pre-classified cells. The live parent's drifted state is irrelevant to where the daughters will be placed. PCA on image pixels inside the snapshot-centered box sees the actual bright masses, not the drifted fit.

**Non-splitting neighbor drifts into the bounding box near but not at a daughter.**
Phase A settles non-classified neighbors first, so by Phase B their live positions are trustworthy. The Voronoi claim-set uses each neighbor's live position. A pixel near the drifted neighbor is closer to the neighbor than to either of the splitting cell's expected daughter centers, so it is excluded from the PCA input. The split placement sees only the legitimate daughter masses.

**Splitting neighbor's bright mass overlaps the bounding box.**
Other pre-classified cells contribute both `{D1_exp, D2_exp}` to the claim-set, not their single snapshot center. A pixel in a region where that neighbor's daughters will be is correctly claimed by the neighbor, not by the splitting cell under inspection. Order effects within Phase B exist but are noise-level.

**Missed split from long camera interval — cell was round last frame but has physically split this frame.**
Base `P_split_base = 0.03` guarantees every cell gets roughly `iterations_per_cell × 0.03` split attempts per frame even if its snapshot elongation is below `T_classify`. The attempt runs in Phase A with snapshot-only data (no live positions yet). Direction comes from PCA only. If the two masses exist in the image, PCA finds them and the cost + bio check accept the split. If only one mass exists, PCA places both daughters on top of each other, bio check rejects, cell is blacklisted for the rest of the frame.

**Cost baseline timing bias — early-iteration vs late-iteration attempts.**
A split attempt at iteration 5 compares against a barely-settled parent; an attempt at iteration 300 compares against a settled parent. For a real split, no single triaxial ellipsoid can cover two separated daughter masses well, so `P_cost_best > D_cost` is structurally guaranteed. The delta shrinks as the parent settles but never inverts. Early attempts and late attempts both clear the `split_cost` threshold for real splits. For non-splits, PCA on a single mass produces degenerate daughter placement that the bio check catches before the cost check ever runs.

**Pre-classified cell where snapshot direction and current-image PCA disagree.**
`split_direction_agreement_degrees` triggers a dual-primary candidate set: both `dir_snap` and `dir_pca` are seeded, each with rotation and translation variants. Per-candidate burn-in + best-cost selection chooses whichever direction was actually right. We do not have to hard-commit to one source.

**Degenerate false split from asymmetric bright noise on a single mass.**
A blob with a bright hotspot on one side can produce two PCA centroids that are just separated enough to dodge the bio check's size and volume ratios. This is rare and usually caught by the cost check in combination with the size-reduction penalty, because the asymmetric daughters tend to shrink along one axis. If diagnostics show it happening in practice, the mitigation is to log the split placements per attempt and raise `bio_daughter_size_ratio_max` closer to 1.3.

**Expected-daughter positions drift from actual daughter positions.**
The seed `D1_seed/D2_seed` computed from the snapshot long axis is an extrapolation and can be 5–15 voxels off per coordinate. Using those seeds directly in the Voronoi claim-set during other cells' split attempts would introduce noise at the exclusion boundary. The frame-start pre-pass fixes this by re-running PCA on the current image inside each pre_classified cell's snapshot bounding box and overwriting `D1_exp/D2_exp` with image-grounded centroids. Residual error after one pre-pass is sub-voxel for well-separated daughter masses and bounded by neighbor-exclusion noise (since the pre-pass itself uses the seeds for its own Voronoi). Iterating the pre-pass closes the gap further; `expected_daughter_pre_pass_iterations` controls how many rounds. PCA centroid computation averages over hundreds of pixels, so even the unrefined seed case shifts centroids by sub-pixel amounts from a few dozen boundary misclassifications — acceptable in practice, but the pre-pass removes most of the systematic error for free.

### Handled with acceptable degradation

**Splitting neighbor drifts along the split axis into the expected daughter region.**
Voronoi exclusion treats the neighbor's `{D1_exp, D2_exp}` as claim points. A pixel wedged between the splitting cell's D2 and the neighbor's D1 can be claimed by either side depending on which exact point is closest. A few misclassified pixels at the boundary do not meaningfully shift the PCA centroids. The split still produces approximately correct daughter positions.

**Two adjacent pre_classified cells, both splitting.**
Phase B processes cells in random order. Whichever fires first uses the other's `{D1_exp, D2_exp}` for exclusion; whichever fires second uses live positions (if the first accepted) or live position (if the first rejected). Order-dependent, but the shift is small and attributable to the same randomness present everywhere else in the Monte Carlo loop.

### Known failure modes (not fixed in this design)

**Splitting neighbor sits physically between a splitting cell's parent center and one of its daughters, on the split axis.**
The Voronoi boundary runs through where the daughter should be. Pixels near the daughter get claimed by the neighbor, and the splitting cell's PCA sees only one mass (its other daughter). PCA fails to find two well-separated centroids, bio check rejects, split is abandoned. The biologically correct answer — "there is no room for the daughter because the neighbor is wedged in" — is enforced implicitly, but the real issue (crowded geometry) is not reasoned about. A future fix would need a secondary signal like frame-to-frame motion continuity.

**Missed split where the optimizer in Phase A drifts the non-classified cell onto one mass instead of stretching to cover both.**
If the triaxial optimizer sits the cell on one daughter and leaves the other as high residual, the live elongation does not jump. Base `P_split_base` still gives the cell a handful of split attempts during Phase A, but those attempts run with snapshot-only data (not live data) because Phase A is still running. If the snapshot was round and the optimizer biased toward one mass, PCA on the current image inside the snapshot bounding box should still find both masses (because we center on the snapshot, not the drifted position). The case where this fails is when the snapshot bounding box is tight enough to miss one of the emerging masses. Mitigation: keep the bounding box radius generous (`3 × maxR` of the snapshot).

**Compound case — missed split plus physically overlapping neighbor.**
If a cell missed its previous-frame split classification AND has a neighbor physically overlapping it, Phase A may fit the cell onto the neighbor's mass. The snapshot bounding box at the cell's previous position does not cover the neighbor's region, so PCA sees only whatever legitimate mass is at the snapshot position. If that is one emerging daughter, split attempts will fail the bio check. If that is nothing, the cell will be flagged as poorly fit but never split. This is a genuine blind spot that requires a motion-continuity signal to fix — a bigger architectural change, explicitly out of scope.

**Cell that divides into three daughters in a single frame interval.**
The entire design assumes binary division. A three-way division would produce PCA centroids that average to a two-way split in the wrong place. The bio check may or may not catch it depending on geometry. This is biologically rare and deferred.

---

## Addresses Prior Failure Modes

Three false-split modes were observed in the current oblate pipeline and patched with brittle guards. Each is either structurally eliminated by the triaxial model or caught by a simpler, more principled check in the new pipeline. This section documents how.

### Daughter mimicking parent

**The old failure:** A split is accepted but the two daughters end up occupying the parent's location — either both piled on top of each other or one grown to parent size — so the "split" does not actually represent two cells. In the old pipeline this was patched with the bridge-brightness guard, which worked by checking whether the cylinder between daughter centers was still bright; brittle and hard to tune.

**How the new pipeline catches it:** Three overlapping mechanisms, no single brittle check.

1. **Overlap penalty inside per-candidate cost.** The quadratic overlap penalty shoots up when daughters overlap. Any candidate with piled-up daughters has higher post-burn-in cost than alternatives that spread them along the split axis. The best-cost selection drops the degenerate candidate.
2. **Combined-volume bio check.** `bio_combined_volume_min_fraction = 0.6` and `bio_combined_volume_max_fraction = 1.3`. If one daughter grows to parent-sized while the other stays daughter-sized, combined volume exceeds 1.3 × parent and the bio check rejects.
3. **Final cost gate.** Two daughters covering the same bright mass fit the image no better than one parent covering it, and the overlap penalty makes the total worse. Cost check rejects.

**Residual risk:** The specific corner case of "both daughters at half-size, exactly piled on each other" passes the volume check (combined ~1.0 × parent) and passes the size-ratio check (both same size). It relies entirely on the overlap penalty and the `bio_daughter_buried_threshold` check to catch it. The buried check must be implemented so that the sibling daughter counts as "another cell" for the burial test — if the implementation only checks against neighbors and excludes the sibling, this corner case slips through. Flag for implementation review.

### Z-split — two stacked spheres mimicking a pancake (or the inverse)

**The old failure:** A flat-wide-XY pancaked cell would sometimes be split along the Z axis as if it were two round cells stacked in Z. The oblate model (`a = b`) cannot represent a flat-wide-XY shape — it fits the bright mass as *either* a pancake with `a = b, small c` *or* two round spheres stacked in Z, and the two interpretations are unstable because the model lacks the degree of freedom to distinguish them. The old pipeline tried to patch this with `split_minor_axis_alignment_tolerance_degrees` (steer the split axis onto local z when flat) and the `split_pre_burn_in_z_axis_*` gates. Brittle and only fixed one direction of the confusion.

**How the new pipeline catches it:** Structurally eliminated at the model level, not by a check. The triaxial model (`a ≠ b ≠ c`) has the degree of freedom the oblate model lacks.

- A pancake fits as `a > b >> c` with the long axis in its actual XY orientation. Elongation signal is `a / c`, which is high, so the cell is pre_classified. Split direction comes from the fitted long axis, which correctly points along the pancake's XY major axis.
- Two Z-stacked spheres fit as `c > a ≈ b`, with the long axis along Z. Elongation signal is `c / a`, also high. Split direction correctly points along Z.
- Both scenarios produce high elongation and correct split directions. The triaxial fit distinguishes them because one has `a > b` and small `c`, the other has `a ≈ b` and large `c`. The minor-axis alignment guards and z-axis pre-burn-in gates exist *only* to patch the oblate model's blindness and are on the deletion list.

**Residual risk:** Optimizer convergence. If the initial placement for a pancake happens to be two Z-stacked spheres (or vice versa), the optimizer may get stuck in the wrong local minimum even though the correct triaxial fit is globally better. This is an optimization issue, not a model issue. Mitigation comes from initial placement via PCA centroids on the actual bright mass (which point at the correct long axis) and from the per-frame perturbation budget being large enough to escape local minima. No additional guard is proposed; monitor during diagnostics if it appears.

### Daughter shrink-grow — one collapses to min, other grows to parent size

**The old failure:** During burn-in, one daughter shrinks toward the minimum radius while the other grows toward parent size. End state: effectively a single cell at the larger daughter's position with a tiny ghost daughter contributing almost nothing. The image cost is good (the large daughter covers the bright mass) but the cell count is wrong. The old pipeline patched this with `split_fake_radius_ratio_threshold = 1.6` post-burn-in.

**How the new pipeline catches it:** Three mechanisms working together, the first being the strongest.

1. **`bio_daughter_size_ratio_max = 1.5` bio check.** If one daughter is more than 1.5× the other in any radius, reject. A shrunk daughter at `10` and a grown daughter at `30` is a 3.0× ratio. Caught immediately at the bio gate, before the cost check even runs. This is the primary defense, slightly tighter than the old 1.6 threshold.
2. **`size_reduction_penalty_weight` during per-candidate burn-in.** Discourages either daughter from shrinking during burn-in. Candidates that drift asymmetric accumulate shrinkage penalty and lose the per-candidate best-cost contest to candidates that stay symmetric.
3. **Candidate seeding symmetry.** All K candidates start from symmetric placements (`cbrt(0.5) × parent_radii` on both daughters). The burn-in has a limited iteration budget per candidate, so there's less time for the asymmetric mode to take hold even if it would eventually win.

**Residual risk:** Purely a tuning issue. If `size_reduction_penalty_weight` is set too low and `bio_daughter_size_ratio_max` is set too loose (above 1.5), this mode can still happen. Both are config-adjustable. The starting values match the tuning that is known to work in the current codebase, so this is a knob to guard, not a design hole.

### Confidence Ordering

In order from most to least structurally solved:

1. **Z-split / pancake confusion** — solved at the model level. Cannot come back unless the triaxial model is reverted.
2. **Shrink-grow** — caught by an explicit bio check plus a burn-in penalty. Robust as long as config tuning stays within sane bounds.
3. **Daughter mimicking parent** — caught by three redundant mechanisms with one corner case that depends on the `bio_daughter_buried_threshold` implementation treating the sibling daughter as "another cell." Flag for implementation review.

---

## Previously Used Split Gates (Historical Reference)

The current bio check is intentionally minimal: daughter size ratio, combined volume fraction, daughter-buried-in-neighbor, and daughter-buried-in-sibling. During the oblate-era implementation many more gates were layered on top of the split path to patch specific failure modes. They are all removed in this redesign because the triaxial model + candidate refinement burn-in + PCA-on-image-centroid placement catches those cases structurally.

This section archives the removed gates so the rationale is available if any of these failure modes resurfaces in practice. Each entry describes what the gate checked, which failure mode it targeted, why the new pipeline no longer needs it, and under what conditions it might be worth reintroducing.

### Pre-burn-in gates (applied before daughter burn-in started)

**`split_elongation_threshold`**
PCA-image-elongation cutoff (~1.3). Below this, the burn-in was skipped and the cell was blacklisted for the rest of the frame.
- Failure mode caught: round cells wasting burn-in iterations on hopeless splits.
- Why removed: replaced by `shape_elongation_classify_threshold` applied to the *fitted* shape, computed once per cell at frame start. No per-attempt gate needed.
- Reintroduce if: diagnostics show the new Phase B is burning candidate iterations on cells the fit considers elongated but PCA-on-image considers round (rare — usually means the fit is wrong).

**`split_min_inside_count`**
Minimum bright-voxel count inside the cell's ellipsoid (~50000). Below this, the split was skipped.
- Failure mode caught: edge cells clipped by the z-stack having unreliable PCA from too few pixels.
- Why removed: the new `trySplitCellPhased` rejects attempts when `gatherBrightPixelsVoronoi` returns fewer than 20 pixels. The threshold is the same in spirit, just applied against the actual post-exclusion bright-pixel set rather than the full interior volume.
- Reintroduce if: Phase B is accepting splits on cells near the z-stack boundary whose PCA is noise.

**`split_search_radius_multiplier`**
Expansion factor for the PCA bounding box around the cell (3.0×). Too small and the PCA missed peripheral bright mass; too large and it absorbed neighbors.
- Failure mode caught: bounding-box tuning tradeoff.
- Why removed: the new code hardcodes `3 × max(a,b,c)` as the box radius, which is the same default. Voronoi exclusion handles the neighbor-absorption problem separately.
- Reintroduce if: a dataset with very different cell sizes needs per-dataset tuning. Easy to expose as config again.

**`split_pre_burn_in_min_separation_over_major`** (~0.35)
Rejected pre-burn-in if daughter center separation was less than this fraction of the parent's `majorRadius`. Catches "collapsed" splits where the two PCA centroids are essentially on top of each other.
- Failure mode caught: PCA on a round blob produces two centroids at the same place, the naïve placement code uses them anyway, burn-in accepts the degenerate split.
- Why removed: the bio check (`bio_daughter_size_ratio_max` + sibling-buried check) rejects the same pathological case post-burn-in. The pre-burn-in check was purely an efficiency optimization — skip the burn-in when we already know it's bad.
- Reintroduce if: per-candidate burn-in is too expensive for the collapsed-placement case. Cheap to re-add as a pre-filter on each candidate before running its burn-in loop.

**`split_pre_burn_in_z_axis_max_abs` / `split_pre_burn_in_z_axis_max_separation_over_major` / `split_pre_burn_in_z_axis_min_drift_over_major`**
Trio of gates rejecting split attempts whose PCA axis was z-dominant (`|dir.z| > 0.92`) with weak transverse separation and small mid-burn-in drift.
- Failure mode caught: pancaking cells where the oblate fit was so wrong that image PCA placed the split axis in Z (splitting a flat disc "vertically"). These were always false positives in the oblate era.
- Why removed: the triaxial fit represents the pancake correctly (`a > b >> c`), so the snapshot long axis points along the pancake's XY major axis. PCA on the image agrees. The z-axis failure mode no longer exists at the model level.
- Reintroduce if: a dataset has genuinely z-stacked cells (vertical division axis). None of our datasets do, but the check would actually be harmful there — it would kill legitimate splits.

### Post-burn-in gates (applied after daughter burn-in completed)

**`split_fake_overlap_volume_fraction_threshold`** (~0.15)
Rejected if the daughter pair's sphere-equivalent overlap volume exceeded 15% of their combined volume.
- Failure mode caught: daughter pair collapses back into each other after burn-in.
- Why removed: replaced by the generic overlap penalty in the cost function. The burn-in's cost function already penalizes overlap via `overlap_penalty_weight × overlapRatio²`, so candidates that collapse lose the best-cost contest.
- Reintroduce if: the overlap penalty weight is too low to push daughters apart. The old fraction check is strictly post-hoc; the cost penalty is integrated over the burn-in.

**`split_fake_volume_ratio_threshold`** / **`split_fake_radius_ratio_threshold`** (~1.6 ratio, ~6.0 volume)
Rejected if one daughter was more than ~1.6× the other's radius or ~6× the volume.
- Failure mode caught: shrink-grow degenerate splits (one daughter collapses to min, the other grows to parent size).
- Why removed: replaced by `bio_daughter_size_ratio_max = 1.5` (tighter than old 1.6). The new version is applied in the bio check at the selected candidate, not post-burn-in on the final state — same effect, simpler code path.
- Reintroduce if: cells with legitimately asymmetric daughters (biologically rare) need to be allowed. Relaxing `bio_daughter_size_ratio_max` to 1.6–2.0 handles it without re-adding the separate check.

**`split_fake_bridge_brightness_similarity_threshold`** (~0.9)
Built a cylinder between the daughter centers in the raw image, measured mean brightness, and rejected if the bridge was ≥ 0.9× of the daughters' mean brightness (i.e., the "gap" between daughters was as bright as the daughters themselves → they're still one continuous blob).
- Failure mode caught: fake split where the underlying mass is a single continuous blob fit by two overlapping daughters.
- Why removed: two reasons. (1) The new preprocessing doesn't have sharp 0/1 boundaries anymore, so the bridge-brightness ratio is noisy and drifts. (2) The new pipeline's PCA centroid placement on Voronoi-filtered pixels inherently requires two distinct bright masses — if there's no gap, the two centroids collapse and the size-ratio or buried-sibling checks reject.
- Reintroduce if: false positives appear where two daughters are placed across a continuous bright blob. The bridge check was tuned for sigmoid output and would need re-tuning for the new preprocessing (different thresholds, probably `>= 0.95` similarity to match the less sharp boundaries).

**`split_minor_axis_alignment_tolerance_degrees` / `split_minor_axis_alignment_flatness_ratio_threshold` / `split_minor_axis_alignment_min_radius_disable_threshold`**
If the cell was flat (`minorRadius / majorRadius <= 0.5`) AND the PCA axis disagreed with local z by more than the tolerance (in degrees), the split axis was **forcibly steered** onto local z.
- Failure mode caught: pancake cells where the oblate fit was blind to the XY long axis, so PCA latched onto noise in the z direction. The steering hack forced the split to go along z — wrong, but less wrong than a random XY direction.
- Why removed: triaxial model represents pancakes correctly and PCA on the pancake's bright mass naturally points along the long XY axis. No steering needed; steering would in fact be harmful now because it would fight the correct fit.
- Reintroduce if: never, under the triaxial design. Specifically anti-triaxial.

**`split_post_burn_in_large_recenter_min_drift_over_major` / `split_post_burn_in_large_recenter_max_cost_diff`**
Rejected splits where daughters drifted more than 0.85× parent `majorRadius` from their initial placement AND cost improvement was weaker than −40. Signal: the cell is being dragged to absorb a neighbor rather than actually splitting.
- Failure mode caught: split attempts that survive burn-in by claiming a neighbor's bright mass, not by genuinely dividing.
- Why removed: replaced by the Voronoi exclusion in `gatherBrightPixelsVoronoi`. The new placement code can't see pixels that belong to neighbors, so it can't place daughters on them in the first place. The post-hoc recenter check was a symptom detector; the Voronoi exclusion is a cause fix.
- Reintroduce if: Voronoi exclusion is disabled or the claim-sets are stale (e.g., aggressive neighbor drift between frames). The check is a cheap backup diagnostic.

### Image-PCA gates on the parent cell

**`computeStrictInteriorWeightedElongation` (Pillar A PCA)**
Brightness-weighted PCA on the bright pixels strictly inside the cell's fitted ellipsoid. Returned both a halo-weighted elongation and a top-K-percent-brightest elongation, with effective = max of the two.
- Failure mode caught: detecting pre-split elongation from image pixels rather than the fit.
- Why removed: the triaxial model's `shapeElongation()` method returns `max(a,b,c)/min(a,b,c)` from the fit directly. With the new model, the fit itself captures the elongation signal — image PCA is redundant (and was blind anyway when the fit was round).
- Reintroduce if: the triaxial optimizer fails to find `b ≠ a` fits for pancaking cells, leaving `shapeElongation()` stuck near 1. Then an image-based elongation signal would be a useful cross-check. Also useful as a debug/diagnostic signal even when not driving decisions.

**`strictElongationTopKFraction`**
Top-K-percent brightest-pixels PCA used inside `computeStrictInteriorWeightedElongation`.
- Why removed: the function using it is removed.

### Pillar B — structural local-maximum detector

**`runPillarBNeighborMasked` / `proposeSplitByLocalMaximaNeighborMasked` / `PillarBResult`**
Structural local-maximum split detector. For each cell, smoothed the region around its bounding box with a Gaussian, masked neighbor voxels to ROI mean, and hunted for two distinct local brightness maxima with a dim valley between them. If found, the cell was marked as having "bright peaks" and a split attempt was placed directly between those peaks.

Used in two roles:
- **Role A** (end-of-frame) — populated `snapshot.brightPeaks` for the next frame's split placement, bypassing PCA's structural blindness to pancaking.
- **Role B** (per-attempt) — validated the placement before committing.

Associated config: `local_max_roi_expansion_factor`, `local_max_smoothing_sigma`, `local_max_min_prominence_over_stddev`, `local_max_suppression_radius_factor`, `local_max_valley_max_relative_depth`, `local_max_peak_inside_ellipsoid_scale`, `local_max_enabled`.

- Failure modes caught: pre-split pancaking cells where oblate PCA returned `~1.15` (below the split threshold) but Pillar B structurally saw two distinct peaks; conversely, rejecting PCA elongation that passed threshold when Pillar B saw only one peak (single-hotspot false positives).
- Why removed: both failure modes are structurally addressed by the triaxial model + candidate refinement. Pancaking is captured by `shapeElongation` from the fit; single-hotspot false positives are rejected by the bio check because PCA centroids on a single mass collapse together (caught by `bio_daughter_size_ratio_max` / sibling-buried check).
- Reintroduce if: the triaxial fit is unreliable for a dataset (e.g., cells that genuinely resist being fit as triaxial ellipsoids), OR the preprocessing output has too much noise for PCA on raw bright pixels to give a clean direction. Pillar B was tuned for sigmoid-boundary output and would need retuning for the new preprocessing (`smoothing_sigma`, `min_prominence_over_stddev` in particular).

### Pillar B elongation boost

**`kPillarBElongBoostMin = 1.25`**
If Pillar B found two distinct peaks but the image PCA elongation was below 1.25, the snapshot elongation was boosted to 1.25 so the cell would get a split attempt the next frame regardless of PCA.
- Why removed: Pillar B is removed; the boost has nothing to boost.
- Reintroduce if: Pillar B is reintroduced.

### Burn-in budget

**`split_burn_in_iterations` (~1000)**
Monolithic burn-in loop on a single candidate placement. Alternated daughter perturbations for ~1000 iterations before the cost check.
- Why removed: replaced by `split_candidates_per_attempt × split_candidate_burn_in_iterations` (~5 × 20 = 100). The new approach explores placement space with K short burn-ins instead of one long one. Total budget is lower but more robust to a bad initial placement.
- Reintroduce if: diagnostics show the short per-candidate burn-in isn't settling enough for accurate post-burn-in cost measurement. Easy to raise `split_candidate_burn_in_iterations` without switching back to the monolithic approach.

---

## Bio Check Scope (Current)

Keeping the active bio check minimal and well-defined. Only these four checks run, in this order:

1. **Daughter size ratio.** `max(d1R, d2R) / min(d1R, d2R) > bio_daughter_size_ratio_max` → reject. Catches shrink-grow and degenerate radii.
2. **Combined volume fraction.** `(V_d1 + V_d2) / V_parent ∉ [bio_combined_volume_min_fraction, bio_combined_volume_max_fraction]` → reject. Catches mass conservation violations.
3. **Daughter buried in another cell.** For each neighbor cell other than the two daughters: if either daughter center is inside the neighbor's ellipsoid, reject. Catches daughters colliding with pre-existing cells.
4. **Sibling buried in sibling.** Each daughter's center must NOT be inside its sibling's ellipsoid. Catches daughters collapsing to the same point.

If any previously-used gate from the section above starts showing up as a false positive or false negative in diagnostics, add it back to this list **explicitly** (with its config field and the failure mode it targets) before re-enabling it in code. Do not silently re-enable anything — the goal of the cleanup is to keep the active bio check short and each check justified.

---

## Design Decisions Locked In

- Triaxial model (`a ≠ b ≠ c`) replaces oblate spheroid model.
- Elongation signal is `max(a, b, c) / min(a, b, c)` from the fitted shape, not from image PCA.
- Snapshot (frozen end-of-previous-frame state) is the authority for split detection during a frame. Live parent drift does not affect split placement.
- Neighbor exclusion is phased: non-classified cells settle first, then pre-classified cells use live positions for non-classified neighbors.
- Direction source is asymmetric: PCA for non-classified cells, snapshot fit for pre-classified cells (with PCA as a fallback candidate when directions disagree).
- Position source is always PCA centroids of the two halves of bright pixels inside the bounding box.
- Expected daughter positions (`D1_exp`, `D2_exp`) for pre_classified cells are image-grounded via a frame-start pre-pass, not raw snapshot extrapolation. Seed with snapshot long axis, then overwrite with PCA centroids from the current image.
- Split attempts are Monte Carlo per-iteration with configurable base probability and proportional rescaling by previous-frame elongation.
- Blacklist remains: one failed split attempt blacklists the cell for the rest of the frame, and the subsequent iteration becomes a parent perturbation.
- Candidate refinement burn-in: K placements per split attempt, each with a short burn-in, pick best by post-burn-in cost. No long single-placement burn-in.
- Both bio check AND cost check must pass for a split to be accepted. Bio check is the first gate (cheap, geometric sanity); cost check is the second (expensive, only if bio passes). Either failing reverts the parent and blacklists the cell.
- No mid-frame live-elongation reclassification.
- No frame-start settling pass before the split loop (bypassed by the structural cost argument).
- Bio check is dataset-agnostic: radii bounds, daughter size ratio, combined volume ratio, buried-in-neighbor check. All other oblate-era guards are removed.

---

## Scope of Implementation Work (Rough)

To be expanded into a per-file plan before coding begins.

1. **Triaxial model migration** — `SpheroidParams`, `Spheroid::draw`, `Spheroid::getPerturbedCell`, overlap computation, rotation handling, `CellFactory`, `initial.csv` schema. Load-bearing for everything else. Largest chunk.
2. **Pipeline restructuring** — Phase A / Phase B ordering in `CellUniverse::optimize`, classification at frame start, per-cell claim-set bookkeeping in Phase B, config additions.
3. **Split attempt rewrite** — new `getSplitCandidates` that builds K placements with asymmetric direction handling, candidate-level burn-in loop, Voronoi claim-set exclusion, generic bio check.
4. **Dead code removal** — oblate-specific guards, dual-image pipeline, Pillar B peak finding, elongation boost, staleness check, the removed config fields listed above. Full inventory in the next section.

A per-file diff summary and LOC estimate will be written as a separate implementation plan document once this design is approved.

---

## Dead Code Inventory

Audit of the current codebase (branch `yp_yd_merge_04072026`) produced from a read-only exploration on 2026-04-10. Line ranges are approximate — accurate enough to locate with grep, not exact after subsequent edits. Each item explains what makes it dead under the new pipeline.

### 1. Functions and methods to delete entirely

- **`C++/includes/Frame.hpp` (~lines 34-35)** — dual-pipeline `Frame` constructor declaration. New pipeline uses a single preprocessed image.
- **`C++/src/Frame.cpp` (~lines 340-348)** — `Frame::Frame(realFrame, analysisFrame, ...)` implementation. Same reason.
- **`C++/src/Frame.cpp` (~lines 49-299)** — `proposeSplitByLocalMaximaNeighborMasked()`. Pillar B peak finder is structurally unnecessary; new pipeline uses PCA for position and snapshot long axis for direction.
- **`C++/src/Frame.cpp` (~lines 1098-1132)** — `Frame::runPillarBNeighborMasked()`. Same reason.
- **`C++/src/Frame.cpp`** — `trySplitCellFromSnapshot()` (replaces with a new `trySplitCellCandidates()` that implements the K-candidate refinement flow). Gut the whole function, keep the name freed up for the rewrite.
- **`C++/src/Spheroid.cpp`** — `getSplitCells()` (the oblate PCA split detector). Replaced by the frame-start pre-pass + per-attempt candidate generator inside the new split path.
- **`C++/src/Frame.cpp`** — helpers that exist solely for the old `trySplitCell` / `getSplitCells` pair: anything named `checkSplit*`, `pre_burn_in_*`, `post_burn_in_*`, `splitFake*`. Grep for each and verify no perturbation-path caller before deleting.

### 2. Oblate-specific split guards

- **`C++/src/Frame.cpp` (~lines 743-823, inside the old snapshot split path)** — Pillar B fallback placement + "peak too far" rejection + snapshot-axis fallback block. All of these are workarounds for the oblate model being blind to pancaking; the new triaxial + pre-pass pipeline has no need for them.
- **`C++/src/Frame.cpp` (~line 716)** — `volumeScale = cbrt(0.5)` daughter sizing from snapshot oblate radii. Replaced by triaxial daughter sizing based on `(a, b, c)` inheritance.
- **Pre-burn-in gates** (inside `trySplitCellFromSnapshot`): separation-over-major, z-axis-dominant, z-axis-max-separation, z-axis-min-drift checks. All soft filters added for the oblate model's false-positive modes.
- **Post-burn-in fake-split guards** (inside `trySplitCellFromSnapshot`): overlap-volume fraction, radius ratio, bridge brightness, large-recenter checks. Same reason.
- **Minor-axis alignment steering block** (inside `trySplitCellFromSnapshot`): the flatness-gated axis-snap-to-local-z logic. Triaxial model carries the direction in the fit itself.

### 3. Dual-image pipeline scaffolding

- **`C++/includes/Frame.hpp` (~lines 79-85)** — `_analysisFrame` member, `getAnalysisFrame()`, `_analysisBackgroundValue`, `setAnalysisBackgroundValue()`.
- **`C++/includes/Frame.hpp` (~lines 96, 109)** — `_analysisFrame` and `_analysisBackgroundValue` declarations (paired with the accessors above).
- **`C++/src/CellUniverse.cpp` (~lines 23-27)** — `LoadedFrame` struct with `costFrame`, `analysisFrame`, `analysisBackground`. Collapse to a single-frame return type.
- **`C++/src/CellUniverse.cpp` (the two-path `loadFrame` logic)** — the branch that computes an analysis frame separately from the cost frame. Keep only the cost-frame path.
- **Post-sigmoid dim-region subtraction** (`simulation.post_sigmoid_dimmest_*` fields and code path in `loadFrame`) — flag for removal, but **verify during implementation**. If the new preprocessing still benefits from dim-region subtraction, keep it; if it was added to prop up the analysis frame specifically, delete.

### 4. Pillar B, peak finding, elongation boost, staleness check

- **`C++/includes/Frame.hpp` (~lines 16-27)** — `PillarBResult` struct.
- **`C++/src/Frame.cpp` (~lines 25-36)** — `LocalMaxSplitProposal` internal struct.
- **`C++/src/CellUniverse.cpp` (~lines 1006-1012)** — `kPillarBElongBoostMin` constant and the "raise elongation to 1.25 if Pillar B finds 2 peaks" logic. No mid-frame reclassification in the new design.
- **`C++/src/CellUniverse.cpp` (~lines 974-1020)** — end-of-frame Pillar B snapshot population block (the one that writes `brightPeaks` / peak positions into the snapshot). Snapshot now only stores `(center, a, b, c, rotation, longAxisDir, longAxisLength, shapeElongation)`.
- **Staleness check** (wherever it lives in `CellUniverse.cpp` — search for `stale` / `staleness` near split gating). The pre-pass handles stale snapshot extrapolation directly.
- **`SplitDiagnostics` struct** (in `Frame.hpp` or `Frame.cpp`) — if every field is about the removed guards (pre-burn-in, post-burn-in, bridge, etc.), delete the whole struct. If a few fields still make sense for the new flow, keep those and delete the rest.

### 5. Config fields to delete

**`C++/includes/ConfigTypes.hpp` — `ProbabilityConfig` struct:**

- `split_elongation_threshold` — replaced by `shape_elongation_classify_threshold`.
- `split_fake_overlap_volume_fraction_threshold`
- `split_fake_radius_ratio_threshold` (and legacy `split_fake_volume_ratio_threshold` if still present)
- `split_fake_bridge_brightness_similarity_threshold`
- `split_minor_axis_alignment_tolerance_degrees`
- `split_minor_axis_alignment_flatness_ratio_threshold`
- `split_minor_axis_alignment_min_radius_disable_threshold`
- `split_pre_burn_in_min_separation_over_major`
- `split_pre_burn_in_z_axis_max_abs`
- `split_pre_burn_in_z_axis_max_separation_over_major`
- `split_pre_burn_in_z_axis_min_drift_over_major`
- `split_post_burn_in_large_recenter_min_drift_over_major`
- `split_post_burn_in_large_recenter_max_cost_diff`
- `split_burn_in_iterations` — replaced by `split_candidate_burn_in_iterations` × K

For each field: delete from the struct, from `BaseConfig::explodeConfig` YAML parsing, and from `C++/config/config.yaml`.

**`C++/includes/ConfigTypes.hpp` — `SimulationConfig` struct:**

- `post_sigmoid_dimmest_percentile`
- `post_sigmoid_dimmest_transition_width`
- `post_sigmoid_dimmest_transition_gradient`

These are flagged for removal but **verify during implementation** — if the new preprocessing still uses dim-region subtraction, keep them.

**Already-dead fields to sweep while we're here:**

- `sigmoid_center_offset` — noted in the existing rules as "parsed but dead."

**`C++/config/config.yaml`:**

Every removed field above should also be stripped from the YAML. Run a grep for each deleted struct member before finalizing.

### 6. Diagnostic and logging structures that only served the old guards

- **`C++/src/Frame.cpp`** — log lines with tags `[Split PeakTooFar]`, `[Split CurrentFramePillarB]`, `[Split FromSnapshot]`, `[Split SnapshotBurnIn]`, `[Split SnapshotReject]`. Most of these reference guards that are going away. Some may still be reusable under a generic `[Split Reject]` / `[Split Accepted]` scheme — when rewriting `trySplitCellCandidates`, keep the two generic tags and delete everything else.
- **`SplitDiagnostics`** (see section 4) — verify no remaining call site depends on the old fields before removing the struct entirely.

---

## Deliberately NOT Deleted

These were considered during the audit and kept. Listed here so they do not get swept up accidentally in a later cleanup pass.

- **`Frame::perturbCell()`** (`C++/src/Frame.cpp`) — perturbation path stays; orthogonal to split detection.
- **`Spheroid::computeStrictInteriorWeightedElongation()`** — still used to compute snapshot `shapeElongation` at frame end; core to classification threshold.
- **`Spheroid::measureMeanBrightness()`** — used by per-cell brightness EMA update.
- **`computeOverlapForCell()` and `computeOverlapPenalty()`** — used by perturbation and by the new per-candidate burn-in; not split-specific.
- **`Frame::_realFrame`** — canonical cost frame for L2 optimization. Only the dual `_analysisFrame` is going away.
- **Brightness-related config** — `brightnessUpdateBlend`, `brightnessMeanAmplification`, adaptive background fields. Per-cell brightness EMA stays.
- **Radius bounds** — `minMajorRadius`, `maxMajorRadius`, `minMinorRadius`, `maxMinorRadius`. Still enforced. A new `minBAxisRadius`/`maxBAxisRadius` pair joins them for the triaxial `b` axis.
- **`split_cost`** — repurposed as the cost-gate threshold in the new flow. Same field, same meaning.
- **`overlap_penalty_weight`, `size_reduction_penalty_weight`** — used during perturbation and per-candidate burn-in.
- **`StrictInteriorPcaResult` struct**, **`PreviousFrameSnapshot` struct** — core to the snapshot mechanism the new pipeline relies on.

---

## Rewrite vs Delete Note

Several items in sections 1 and 2 are marked for deletion but their names (`trySplitCell*`, `getSplitCells`) will be reused for the new implementation. Treat these as "rewrite, not delete" — the header declarations can stay as stubs during the migration so the rest of the codebase keeps compiling, and the function bodies get replaced wholesale. This matters for the order of implementation: the dead-code sweep should happen *after* the triaxial model migration and the new split-attempt implementation are in place, not before.
