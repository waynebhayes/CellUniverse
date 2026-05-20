# Split Candidate Logic Current Code

This note describes the current split-candidate generation path in
`Frame::trySplitCellPhased`.

## Entry Conditions

| Step | Current behavior | Notes |
|---|---|---|
| Split probability | `CellUniverse::optimize` computes `P(split)` from previous-frame snapshot elongation. | Trash cells get `P(split)=0`; normal cells get a linear ramp from `P_split_base` to `P_split_max`. |
| Snapshot requirement | A split attempt needs a valid `PreviousFrameSnapshot`. | If missing or invalid, the cell is blacklisted for split attempts this frame. |
| Baseline parent | The split path builds a snapshot-state parent and compares live-parent cost vs snapshot-parent cost. | It uses the lower-cost baseline for the split cost comparison, but candidate geometry is still based on snapshot/PCA data, not in-frame drift. |
| Bright-pixel input | Bright pixels are gathered around `snapshot.position` in a sphere of radius `3 * max(parent radii)`, with Voronoi claim filtering. | The claim set uses expected daughter positions for dividing neighbors when available. |

## Direction Types

| Direction type | Label prefix | How it is built | Purpose |
|---|---|---|---|
| Shortest local axis | `axA`, `axB`, or `axC` | Extracts the parent ellipsoid local axes in world space and keeps only the axis with the smallest radius. | Biological default: cells are expected to divide along their thinnest axis. |
| Image-grounded direction | `imgPca` | Comes from the pre-pass expected daughter positions: `expectedD2 - expectedD1`, supplied through `snapshot.splitAxisDir`. It is added only if it is not nearly duplicate of the shortest local axis (`abs(dot) <= 0.95`). | Handles near-round cells where fitted local axes are arbitrary and the current image gives a better daughter-lobe direction. |

Current code does not try all three local axes. It only tries the single
shortest local axis plus optional `imgPca`.

## Midpoint Types

| Midpoint type | Label prefix | How it is built | Purpose |
|---|---|---|---|
| Data midpoint | `data_<axis>` | Projects gathered bright pixels onto the candidate direction, splits them at median projection, computes weighted centroids for both halves, and uses their midpoint. | Lets the current image decide where the two bright lobes are centered. |
| Snapshot midpoint | `snap_<axis>` | Uses `snapshot.position` as the midpoint, while keeping the separation measured from the data projection. It is added only if it differs from the data midpoint by more than `0.5` voxels. | Keeps one conservative candidate anchored at the previous-frame tracked parent center. |

## Placement Variants

For each `(direction, midpoint)` pair, five candidate placements are generated.

| Variant | Label suffix | Formula / behavior | Purpose |
|---|---|---|---|
| Primary | `_primary` | Places daughters symmetrically around the midpoint along the selected direction. | Baseline placement. |
| Negative rotation | `_rot-` | Rotates the direction by `-split_candidate_rotation_delta_degrees` around a perpendicular axis. | Tests a slightly tilted split axis. |
| Positive rotation | `_rot+` | Rotates the direction by `+split_candidate_rotation_delta_degrees`. | Tests the opposite tilt. |
| Negative translation | `_trans-` | Shifts both daughter seeds along the selected direction by `-transDelta`. | Tests an axial offset one way. |
| Positive translation | `_trans+` | Shifts both daughter seeds along the selected direction by `+transDelta`. | Tests an axial offset the other way. |

Current config values:

| Parameter | Current value | Meaning |
|---|---:|---|
| `split_candidate_rotation_delta_degrees` | `8` | Rotation variant angle. |
| `split_candidate_translation_delta_fraction` | `0.2` | `transDelta = 0.2 * daughterR`. |
| `split_candidates_per_attempt` | `30` | Hard cap after candidate generation. |

Typical candidate count:

| Available directions | Midpoints per direction | Variants per midpoint | Total before cap |
|---:|---:|---:|---:|
| 1 | 1 | 5 | 5 |
| 1 | 2 | 5 | 10 |
| 2 | 1 | 5 | 10 |
| 2 | 2 | 5 | 20 |

With the current generator, the `30` candidate cap usually does not truncate.

## Example Labels

| Example label | Meaning |
|---|---|
| `data_axC_primary` | Shortest local axis is `axC`; midpoint from bright-pixel data; no variant perturbation. |
| `data_axC_rot-` | Same, but direction rotated by negative configured angle. |
| `snap_axC_trans+` | Shortest local axis; midpoint fixed at snapshot position; both daughters shifted in positive axis direction. |
| `data_imgPca_primary` | Image-grounded direction; midpoint from current bright-pixel data. |
| `snap_imgPca_rot+` | Image-grounded direction; midpoint fixed at snapshot position; positive rotation variant. |

## Candidate Evaluation

| Stage | What happens |
|---|---|
| Build daughters | Daughters inherit snapshot parent radii scaled by `split_daughter_volume_scale`. Current value is `0.7937`. |
| Candidate burn-in | Parent is replaced by two daughters, then daughters alternately run `split_candidate_burn_in_iterations` perturb steps. Current value is `50`. |
| Cost score | Candidate total is image cost plus overlap penalty. In bbox mode, all candidates and baseline use the same split union bbox. |
| Per-candidate pre-filter | Candidate must have both daughters locally bright enough and show a valley between daughters. Failed candidates cannot win the cost ranking. |
| Winner refine | Best passing candidate gets `split_final_refine_iterations` more alternating daughter perturb steps. Current value is `30`. |
| Daughter PCA refit | Each daughter gets a short PCA shape refit, clamped by configured min/max fractions. |
| Final gates | Geometry, daughter short-axis alignment, daughter overlap, bridge valley, size/volume/buried checks, and final cost threshold decide acceptance. |

## Snapshot Midpoint Meaning

The `snap_*` midpoint is based on `snapshot.position`, not the cell's current
in-frame moved position. The split path may use the moved live cell as the
lower-cost rendering baseline, but candidate geometry intentionally stays tied
to snapshot/PCA-derived data.
