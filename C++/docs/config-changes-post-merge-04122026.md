# Config Changes Since Vincent Merge (2026-04-12)

All changes in `C++/config/config.yaml` since the merge of `yp_change_preprocessing_method_04102026`.

## split_cost: 50 → 15

**Why:** Real splits produce costDiff in [-90, -18]. False splits that pass all bio gates produce [+5, +57]. The original threshold of 50 blocked every real split. Lowered through 20 → 15 based on iterative validation:
- At 20: two real splits missed by <1 unit (1f89ab f8: -19.37, e9077a51 f20: -19.74)
- At 15: all 8 expected splits accepted, zero false splits. Gap to nearest false split: 23+ units.

## bio_max_drift_parent_fraction: 0.4 → 999.0 (disabled)
## bio_max_drift_daughter_fraction: 0.8 → 999.0 (disabled)

**Why:** The drift-from-seed gate was the primary blocker for 1f89ab (a nearly-spherical cell with poor PCA seed placement). PCA overestimates separation for spherical cells (picks up z-taper → z-dominant split axis with inflated separation), placing seeds too far from the real daughter blobs. Daughters need 20-28 voxels of drift to reach the correct positions, exceeding the 19-23 voxel limit.

Analysis of all drift-rejected cases showed every false split also has positive costDiff (+6 to +57), so bridge + cost gates are sufficient. With drift disabled, 1f89ab splits at f8 as expected.

**Reversibility:** Set back to 0.4 / 0.85 if false splits appear in future datasets.

## brightnessMeanAmplification: 1.1 → 1.0

**Why:** Vincent's merge set this to 1.1, pushing synthetic cell brightness 10% above the measured mean. This triggers the brightness-size co-optimization trap: brighter synthetic → optimizer shrinks cells to reduce the "too bright" edge area → cells end up smaller than they should be. Pre-merge value was 1.0. Restored.

## brightness.increase_prob: 0.7 → 0.0
## brightness.decrease_prob: 0.8 → 0.0

**Why:** Vincent's merge enabled adaptive probability tracking for brightness perturbation. But `brightness.sigma` was already 0.0 (perturbation offset always zero), so these probabilities tracked a no-op — wasted computation with no effect. Pre-merge values were 0.0. Restored to match. Brightness is driven solely by the EMA (`brightnessUpdateBlend: 0.9`), not by random perturbation.

## shapeUpdateBlend: (new) 0.1

**Why:** NEW feature — Shape EMA. Cell 1f89ab stays spherical (majorR≈minorR≈30.5, shapeElong=1.09) across all frames despite visible pre-division elongation in the real image at f7+. The perturbation optimizer has no gradient to push radii apart because a round sphere covers both bright lobes acceptably in L2 cost.

Shape EMA blends each cell's radii toward PCA-observed eigenvalue ratios from the real image, preserving total volume (cbrt(a×b×c) constant). This nudges cells toward their actual observed shape over multiple frames.

Started at 0.3 (too aggressive — cells shrank), reduced to 0.1 for gentler blending.

**Requires code change** (not config-only): `Spheroid::measurePCAShape()`, `Spheroid::setRadii()`, shape EMA block in `CellUniverse::optimize()`.

## shape_elongation_classify_threshold: 1.20 → 1.15

**Why:** At f10, cell 1f2ed had snapElong=1.19 — just below the 1.20 threshold. It was
classified as non-split (perturbed in Phase A) instead of pre-split (held for Phase B
with settled neighbors). Lowering to 1.15 correctly classifies borderline cells as
pre-split, giving them better Voronoi neighbor exclusion during split attempts.

Not lowered further to 1.10 to avoid classifying nearly-spherical cells as pre-split
unnecessarily (adds wasted split attempts). False-split cells that reach 1.15+ elongation
are still handled by cost/bridge gates.

Note: the classify threshold no longer gates direction selection (see dual-direction
code change below). It only controls Phase A/B ordering and P(split) classification.

## split_candidate_burn_in_iterations: 20 → 50

**Why:** At f3, cell 12345 had daughters positioned correctly (within 3-4 voxels of PCA
centroids) but costDiff was only -3.09 (needed -15). With only 20 burn-in iterations per
candidate (tight sigmas: position=2px, radius=0.2), daughters get ~3-4 accepted moves —
not enough to optimize into the best configuration. Increasing to 50 gives more room
to settle, potentially pushing marginal splits past the cost threshold.

With dual-direction candidates (code change), this means up to 20 candidates × 50
iterations = 1000 burn-in iterations per split attempt. Split attempts are infrequent
relative to the main optimization loop, so performance impact is modest.

---

## Code changes (require rebuild)

### Always-dual-direction candidate generation

**File:** `C++/src/Frame.cpp` — direction selection in `trySplitCellPhased`

**Why:** At f10, 1f2ed's snapshot had the correct y-dominant split direction
`(0.23, -0.89, 0.40)` but PCA gave a wrong z-dominant axis `(0.19, 0.13, 0.97)`.
The old code discarded the snapshot direction because `snapElong=1.19 < threshold 1.20`.

**Change:** Always try PCA direction. When a valid snapshot direction exists and differs
from PCA by >10°, add it as a second primary direction. This doubles the candidates
(10→20) when directions disagree. Cost picks the winner — no heuristic threshold for
direction selection. When directions agree (<10°), only 10 candidates are generated.

### Shape EMA

**Files:** `ConfigTypes.hpp`, `Spheroid.hpp`, `Spheroid.cpp`, `CellUniverse.cpp`

New `measurePCAShape()` method and per-frame volume-preserving radius blend toward
PCA-observed shape. See changelog entry above.

---

## Unchanged from merge (confirmed correct)

| Field | Value | Notes |
|-------|-------|-------|
| brightnessUpdateBlend | 0.9 | Same pre- and post-merge. Aggressive EMA, works well. |
| brightness.sigma | 0.0 | Perturbation disabled. Brightness is EMA-only. |
| initialBrightness | 0.22 | Vincent's value, slightly above pre-merge 0.2. Fine. |
| backgroundColor | 0.05 | Vincent's preprocessing background level. |
| overlap_penalty_weight | 500.0 | Unchanged. |
| size_reduction_penalty_weight | 30 | Unchanged. |
| All perturbation sigmas | unchanged | x:5, y:5, z:8, majorR:2, bR:2, minorR:2, theta:0.05 |
