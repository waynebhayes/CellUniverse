# Changelog v8 — yp_fix_mask preprocessing merge + post-merge work (2026-04-19+)

Opened 2026-04-19. Previous: `changelogv7.md` (closed at Change 50).

This file marks the start of a new stage: integration of the `yp_fix_mask_04172026` preprocessing pipeline (parallel raw load + global percentile normalization + iterative contrast-scoring preprocessing). Our snap-only split logic from v7 (Change 50) is preserved alongside the new preprocessing.

Branch (merge done on): `jl_yp_preprocessing_merge_04192026`

---

## 2026-04-19: Merge yp_fix_mask preprocessing pipeline (Change 1)

**Status: ACTIVE — validated against snap-only baseline**

### Cherry-picked from `origin/yp_fix_mask_04172026`

| Commit | Content |
|---|---|
| `9b3fcfd` | CMakeLists.txt arm/x86 portability |
| `6ca6331` | Make pre-process parallel |
| `87ee430` | Simplified preprocessing logic, much faster + better contrast |
| `25c5923` | Preprocessing improvement: split `loadFrame` into `loadRawFrame` + `preprocessLoadedFrame`; added contrast/brightness scoring fields and 4-pass constructor |
| `b575246` | Adaptive cube pooling code (kept disabled in our config) |

**Skipped:** `a9b19cb` (their shape config tuning), `ffc1917` (signal-strength guided perturbation), `f42ccc2`/`82055e3`/`f56e67e`/`c1e1dd2`/`6661c96` (their tuning rounds), `cbc0714`/`14afa6a` (debug toggles).

### Files changed

- **`C++/CMakeLists.txt`** — copied yp wholesale (we had no local edits)
- **`C++/src/ImageHandler.cpp`** — copied yp wholesale (~900 LOC: 4-pass preprocessing pipeline, percentile scoring, adaptive cube pooling implementation, signal-center localization)
- **`C++/includes/ImageHandler.hpp`** — copied yp wholesale (new public API: `loadRawFrame`, `preprocessLoadedFrame`, `evaluateSequenceContrastScore`, plus `loadFrame` with optional `logSink`)
- **`C++/includes/ConfigTypes.hpp`** — copied yp wholesale, added back `position_prior_threshold` field + parse (ours; yp doesn't have it)
- **`C++/includes/Frame.hpp`** — added minimal `SignalCenter` struct (public nested) so yp's `ImageHandler::localizeSignalCentersInStack` compiles; no member or accessors added (signal-guided perturbation feature is unused in our pipeline)
- **`C++/src/CellUniverse.cpp`** — added 3 helpers (`computeStackPercentile`, `computeStackMax`, `normalizeStackToSharedScale`); replaced constructor body with yp's 4-pass pipeline (parallel raw load → global percentile normalization → parallel preprocess → frame construction). Kept everything else (`optimize`, snap-only split logic, all our shape/split work) unchanged.
- **`C++/config/config.yaml`** — hybrid: yp's preprocessing block wholesale, our shape/split tuning preserved. Pooling disabled per user instruction.

### Why the constructor port was necessary

First merge attempt copied only `ImageHandler.cpp` + config preprocessing fields; preprocessing produced inverted images (mean=0.95 vs expected ~0.05). Diagnosis: yp's `loadFrame` runs an iterative contrast-scoring loop that assumes inputs have been globally normalized to a shared low/high reference. Our prior single-pass `loadFrame` per frame skipped that normalization. Porting yp's 4-pass constructor (which computes global percentiles across all loaded frames before per-frame preprocessing) restored correct image polarity.

### Validation (run 113836, 22 frames)

Compared HYBRID (this merge) vs SNAP-ONLY (run 095555, last v7 baseline) vs BEST22.

| Cell | GT | HYBRID | SNAP-ONLY | BEST22 |
|------|----|---|---|---|
| e9077 | f3 | **f3 ✓** | f3 ✓ | f3 ✓ |
| 12345 | f3 | **f3 ✓** | f3 ✓ | f3 ✓ |
| 1f89ab | f8 | **f8 ✓** | f8 ✓ | f8 ✓ |
| 1f2ed | f11 | **f11 ✓** | f11 ✓ | f12 (+1) |
| e9077..a50 | f19 | f20 (+1) | f20 (+1) | f20 (+1) |
| 12345..0 | f20 | **f20 ✓** | f20 ✓ | f20 ✓ |
| 12345..1 | f20 | **f20 ✓** | f20 ✓ | f20 ✓ |
| e9077..a51 | f20 | **f20 ✓** | f20 ✓ | f20 ✓ |

| Score | HYBRID | SNAP-ONLY | BEST22 |
|---|---|---|---|
| Splits accepted | 8/8 | 8/8 | 8/8 |
| **On-time** | **7/8** | **7/8** | 6/8 |
| Cells at f22 | 14 | 14 | 14 |

**Hybrid matches snap-only on timing.** Notable difference: split cost diffs are 2-3× larger (e.g. e9077 f3: -114k vs -38k snap-only). Stronger preprocessing contrast → bigger split signal → more confident accepts and more headroom for marginal cases.

### Rollback

Per-file rollback (each independent):
- `git checkout HEAD~1 -- C++/CMakeLists.txt`
- `git checkout HEAD~1 -- C++/src/ImageHandler.cpp C++/includes/ImageHandler.hpp`
- `git checkout HEAD~1 -- C++/src/CellUniverse.cpp` (loses constructor port; `ImageHandler.cpp` will then need rebuild against old signature — partial rollback unstable, prefer full rollback)
- `git checkout HEAD~1 -- C++/includes/ConfigTypes.hpp C++/includes/Frame.hpp C++/config/config.yaml`

For full rollback: `git revert <merge-commit>` (clean, takes everything back together).

### Open follow-ups

- **Backburner — halo coverage**: synth cells don't cover the bright-cell HALO (only the core). Levers when we tackle this: `pcaShapeMaskScale` (1.6 → 1.8-2.0), `pcaShapeRadiusPercentile` (0.90 → 0.95-0.97), or direct `pcaShapeRadiusScale` bump.
- **45-frame validation pending** — confirm hybrid behavior holds at longer horizons (the snap-only 45-frame run from v7 had cascade effects at f16/f29/f32 which may resolve differently with yp preprocessing's stronger contrast).
- **Pooling evaluation** — `adaptive_cube_pooling_enabled` is currently `false`. Run a separate validation with it enabled once baseline is stable.
- **45-frame run vs `e9077..a50` at f19** — still missed (1-frame late) across all three runs (ours, snap-only, hybrid). Possibly the cleanest remaining +1-late target.

### Notes for future entries

- Active branch `jl_yp_preprocessing_merge_04192026` is split off from `jl_runtime_improve_04162026` after the snap-only commit (`3d08daf`). Future cleanup work should branch off here.
- All future preprocessing/cleanup/optimization changes go in this v8 file. v7 is closed at Change 50 (snap-only).

---

## 2026-04-19: Config audit — magnitude-dependent gate retuning (Change 2)

**Status: ACTIVE — validated 8/8 splits 7/8 on time 0 FP at 22-frame horizon**

### Why

Yp preprocessing produces image cost magnitudes 2-3× larger than ours. Real-split cost diffs grew similarly (was -8k to -30k snap-only, now -56k to -177k hybrid). Existing gates (split_cost=2000 fixed, split_cost_fraction=0.03) were calibrated for the OLD magnitudes — too lenient at the new scale. The 45-frame hybrid run (130101) confirmed: **e3d03 false-positive split at f4** with diff=-58k against baseline 195k (30% improvement) cleanly cleared the 0.03 (~5860) threshold.

### Fields changed in `C++/config/config.yaml`

| Field | Was | Is | Why |
|---|---|---|---|
| `split_cost` | 2000 | **7000** | Fixed split-cost floor; ~3.5× to track cost magnitude growth |
| `split_cost_fraction` | 0.03 | **0.20** | Real splits show 100%+ improvements; FPs show ~30%. 20% blocks FPs cleanly with huge margin for real splits |
| `overlap_penalty_weight` | 30000 | **75000** | 2.5× to keep relative overlap pressure constant |
| `position_prior_weight` | 30 | **75** | 2.5× so position prior is not diluted by larger image costs |
| `bio_bridge_max_valley_ratio` | 0.85 | **0.75** | Tighter bridge gate; FP slipped past 0.85 by spanning genuine empty space between distant daughters |
| `bio_bridge_min_edge_brightness_absolute` | 0.04 | **0.07** | Yp preprocessing brightens halos → phantom edges sit on halo. Tightening blocks halo-on-halo phantom splits |
| `pcaShapeMaskScale` | 1.6 | **1.3** | Smaller mask = less halo absorption = more compact fits |
| `pcaShapeWeightExponent` | 1.0 | **1.3** | Suppress halo influence in PCA centroid (snap-only baseline value) |
| `pcaShapeRadiusPercentile` | 0.90 | **0.85** | Tighter radii — top 10% under yp may include halo |
| `adaptive_background_top_fraction` | 0.4 | **0.2** | Top 40% of bg candidates may include cell halo; top 20% targets cleaner bg |

### Validation (run 135708, 22 frames)

| Cell | GT | Result |
|---|---|---|
| e9077 | f3 | f3 ✓ |
| 12345 | f3 | f3 ✓ |
| 1f89ab | f8 | f8 ✓ |
| 1f2ed | f11 | f11 ✓ |
| e9077..a50 | f19 | f20 (+1) |
| 12345..0 | f20 | f20 ✓ |
| 12345..1 | f20 | f20 ✓ |
| e9077..a51 | f20 | f20 ✓ |

8/8 splits, 7/8 on time, 0 FP. Identical timing to pre-audit hybrid run. Tightened gates blocked the f4 FP without losing any real splits.

---

## 2026-04-19: Signal-guided perturbation port + iter tune (Change 3)

**Status: ACTIVE — validated 8/8 splits 7/8 on time 0 FP**

Ported yp's `ffc1917` signal-guided perturbation feature in full (frame-level toggle design, not per-iteration).

### Files changed

- **`C++/includes/Ellipsoid.hpp`** — added `setPosition(float x, float y, float z)` setter (used by signal-guided teleport).
- **`C++/includes/Frame.hpp`** — `_signalCenters` member, `setSignalCenters`/`getSignalCenters`, `useSignalGuidance` param on `perturbCell`.
- **`C++/src/Frame.cpp::perturbCell`** — added signal-guidance teleport block (~40 LOC) after `getPerturbedCell`. When `useSignalGuidance` is true, finds the nearest signal center to oldPos and resamples cell position from a normal distribution around that center (sigma scaled by per-cluster `sigmaScale × signal_guided_sigma_range_multiplier`).
- **`C++/src/CellUniverse.cpp`** — added `BrightBox` struct, `chooseNearestDivisorSize`, `localizeSignalCentersForFrame` (~170 LOC). In `optimize()`, decides per-frame whether to use signal-guided based on:
  1. `signal_guided_position_enabled` is true
  2. Centers detected ≥ previous frame cell count (otherwise fallback to random mode)
- **Iteration count split**: `signal_guided_iterations_per_cell` for guided frames, `random_iterations_per_cell` for random frames. Falls back to `iterations_per_cell` when either is < 0.
- **`C++/config/config.yaml`** —
  - `iterations_per_cell: 500 → 150`
  - `signal_guided_position_enabled: true`
  - `signal_guided_iterations_per_cell: 100`
  - `random_iterations_per_cell: 150`
  - `signal_guided_box_size_scale: 0.4`, `signal_guided_min_box_brightness_delta: 0.4`, `signal_guided_min_sigma_scale: 0.35`, `signal_guided_sigma_range_multiplier: 10.0` (yp tuned defaults)

### Why frame-level (not per-iteration)

First port attempt (run 100630) used signal-guidance for ALL perturbations: regression to 5/8 splits, perturb-accept rate dropped to 5-20/frame. Signal-guided teleport is destructive when applied unconditionally — it overwrites good positions ~80% of the time. Yp's frame-level design (run 100631) uses signal-guided only when conditions favor it: enough centers were detected to plausibly cover all cells.

### Validation (run 100631, 22 frames)

Per-frame mode breakdown:
- f1: signal_guided (no prev-frame fallback condition; 0 perturbs but no splits expected)
- f3: signal_guided (centers=6 = prevCells=6) → 2 GT splits accepted ✓
- All other frames: random fallback (centers usually 4-9, well under cell count)

Result: **8/8 splits, 7/8 on time, 0 FP**. Same as audit baseline.

### Open follow-ups (still applicable from Change 1)

- 45-frame validation
- Halo coverage tuning
- Pooling evaluation
- Fix `e9077..a50` +1 late at f19 (consistent miss across all hybrid runs)


---

## 2026-04-19: PCA shape revert + signal-guided disabled for isolation (Change 4)

**Status: ACTIVE — validated 8/8 splits 7/8 on time 0 FP at 22-frame, daughter radii restored**

### Why

The Change 2 audit tightened three PCA shape fields defensively (`pcaShapeMaskScale: 1.6→1.3`, `pcaShapeWeightExponent: 1.0→1.3`, `pcaShapeRadiusPercentile: 0.90→0.85`) alongside the cost/bio gate tuning. Inspection of the resulting 22-frame run showed daughter c-axis shrinking from 15 px (baseline) to 7 px by f20 — cells compacted too aggressively. The audit's cost/bio gate changes (`split_cost_fraction: 0.03→0.20`, tightened bridge gate) were already blocking the e3d03 f4 FP directly, so the PCA shape tightening was redundant and harmful: it suppressed legitimate cell extent without adding FP protection.

### Fields reverted in `C++/config/config.yaml`

| Field | Change 2 (audit) | Change 4 (reverted) | Baseline value |
|---|---|---|---|
| `pcaShapeMaskScale` | 1.3 | **1.6** | 1.6 (best22 validated) |
| `pcaShapeWeightExponent` | 1.3 | **1.0** | 1.0 (linear weighting) |
| `pcaShapeRadiusPercentile` | 0.85 | **0.90** | 0.90 (top 10% trim) |

All three back to their v7 snap-only baseline values. The cost and bio gates from Change 2 (`split_cost: 7000`, `split_cost_fraction: 0.20`, `overlap_penalty_weight: 75000`, `position_prior_weight: 75`, `bio_bridge_max_valley_ratio: 0.75`, `bio_bridge_min_edge_brightness_absolute: 0.07`, `adaptive_background_top_fraction: 0.2`) are KEPT — they are the actual FP-blocking levers.

### Signal-guided disabled for isolation

`signal_guided_position_enabled: true → false`. Change 3 (signal-guided port) is kept in the codebase and config; only the master toggle is off so the 22-frame validation isolates the PCA shape revert's effect. Per-frame iter buckets still apply (`random_iterations_per_cell: 150` is used on all frames when the toggle is off). Re-enabling is a separate evaluation after the 45-frame baseline lands.

### Validation (run 162148, 22 frames)

| Cell | GT | Result |
|---|---|---|
| e9077 | f3 | f3 ✓ |
| 12345 | f3 | f3 ✓ |
| 1f89ab | f8 | f8 ✓ |
| 1f2ed | f11 | f11 ✓ |
| e9077..a50 | f19 | f20 (+1) |
| 12345..0 | f20 | f20 ✓ |
| 12345..1 | f20 | f20 ✓ |
| e9077..a51 | f20 | f20 ✓ |

8/8 splits, 7/8 on time, 0 FP. Same timeline as Changes 2 and 3. **Daughter radii restored to baseline sizes** (c-axis ~15 px at f20 vs 7 px with the audit PCA settings).

### Conclusion

Cost-gate audit, not PCA shape tightening, blocks FPs. PCA shape fields stay at baseline values (1.6 / 1.0 / 0.90). Ready for 45-frame validation (the critical next test — confirms no FP regression at longer horizon AND that daughter/grand-daughter radii stay visible).

### Open follow-ups

- 45-frame validation of current config (Change 2 gates + Change 4 PCA revert + signal-guided OFF)
- Re-enable `signal_guided_position_enabled: true` once 45-frame baseline is clean, to see if guidance helps anywhere when not active by default
- Halo coverage tuning (now easier to assess with baseline PCA values)
- Pooling evaluation (`adaptive_cube_pooling_enabled: false` still untested)
- `e9077..a50` +1 late at f19 (hard case, consistent across all hybrid runs)


---

## 2026-04-19: Shape-fix — pooling + radiusScale + cost gate revert (Change 5)

**Status: ACTIVE — validated 8/8 splits 7/8 on time 0 FP at 22-frame; cell shapes restored to visually match real bright extents**

### Why

User flagged after Change 4 validation (run 162148): cell size/shape "very incorrect vs the two runs before the merge" — synth ellipsoids fit the bright cell cores but not the full cell extent. Frame-by-frame comparison vs pre-merge baseline (095555) confirmed:

- **Frame-1 PCA radii systematically undershot baseline**: 12345 c-axis 16.7 vs 22.7 (−26%), e9077 c-axis 20.0 vs 25.5 (−22%), 1f89ab a-axis 35.4 vs 43.3 (−18%).
- **Mask pixel count** (`n`) at f1 was ~40% lower than baseline (e.g. 12345: 95044 vs 154863), while **meanW** was ~2× higher — yp preprocessing creates brighter cores with compressed halos, collapsing weighted variance along thin axes.
- **Cells kept shrinking frame-over-frame** in the Change 4 baseline: 1f89ab a-axis drifted from −11% at f1 to −40% at f10 to −32% at f22. The fit growth cap couldn't overcome per-frame PCA undershoot.

### Investigation path (what we tried and why each failed before landing on the final combo)

1. **`pcaShapeRadiusPercentile: 0.90 → 0.95`** — no effect. Pipeline doc + code confirm percentile path is DEAD since v7 Change 50; active code uses `pcaShapeRadiusScale × sqrt(weightedVariance)`. Reverted.
2. **`pcaShapeMaskScale: 1.6 → 1.8`** — small positive effect (~+1-2% on c-axis for some cells). Kept.
3. **`pcaShapeRadiusScale: 2.236 → 2.5`** (run 175740) — inflated cells uniformly by +12%, but inflated baseline image costs too (1f89ab f8 baseline 145k → 226k). Real split improvements as a fraction of baseline dropped from 31% (Change 4) to 16%, **failing the audit's `split_cost_fraction: 0.20` gate**. 1f89ab f8 rejected.
4. **`split_cost_fraction: 0.20 → 0.12`** on top of rs=2.5 (run 181007) — still not enough. Baseline cost rose further (225k → 368k), real ratio dropped to 9.8%. Chasing the fraction was a losing game.
5. **`pcaShapeRadiusScale` reverted to 2.236 + `adaptive_cube_pooling_enabled: true`** (run 182118) — pooling spread core brightness into cube-neighborhoods, mask pixel count recovered +26% at f1, c-axis mostly recovered (12345 c=16.7→18.5, e9077 c=20.0→23.2). But user observation: cells still "almost uniform brightness" AND still undersized by 20-40% by mid-run because pooling's `majority_threshold: 0.7` erodes the boundary, and weighted variance still concentrates fit near core.
6. **Final combo**: pooling ON + `pcaShapeRadiusScale: 2.5` (re-applied) + **`split_cost` and `split_cost_fraction` reverted to pre-merge values** (2000 / 0.03). With cost gate relaxed, bigger baseline costs no longer block real splits. Bio gates (tightened in Change 2: valley 0.75, edge 0.07) + re-enabled position prior (w=75) carry FP protection.

### Fields changed (all in `C++/config/config.yaml`)

| Field | Change 4 value | Change 5 value | Active lever? |
|---|---|---|---|
| `pcaShapeMaskScale` | 1.6 | **1.8** | Yes (small c-axis gain) |
| `pcaShapeRadiusScale` | 2.236 | **2.5** | Yes (primary fit-size lever) |
| `pcaShapeRadiusPercentile` | 0.95→0.90 | 0.90 | **Dead code** (see comment) |
| `adaptive_cube_pooling_enabled` | false | **true** | Yes (halo preservation) |
| `split_cost` | 7000 | **2000** | Yes (pre-merge value restored) |
| `split_cost_fraction` | 0.20 | **0.03** | Yes (pre-merge value restored) |

Kept from Change 2: `overlap_penalty_weight: 75000`, `position_prior_weight: 75`, `bio_bridge_max_valley_ratio: 0.75`, `bio_bridge_min_edge_brightness_absolute: 0.07`, `adaptive_background_top_fraction: 0.2`.

### Validation (run 193101, 22 frames)

| Cell | GT | Result | Split diff |
|---|---|---|---|
| e9077 | f3 | f3 ✓ | −122910 |
| 12345 | f3 | f3 ✓ | −71028 |
| 1f89ab | f8 | f8 ✓ | −56957 |
| 1f2ed | f11 | f11 ✓ | −123776 |
| e9077..a50 | f19 | f20 (+1 late) | −147848 |
| e9077..a51 | f20 | f20 ✓ | −57553 |
| 12345..0 | f20 | f20 ✓ | −121329 |
| 12345..1 | f20 | f20 ✓ | −101780 |

**8/8 splits, 7/8 on time, 0 FP.** e3d03 at f4 was rejected by bio gate (`d1_bridging_to_12345`) — exactly the historical regression, now blocked even with relaxed 0.03 cost fraction. Multiple bridging-type FPs at f18/f20/f21/f22 also all blocked by bio gate. **Bio gate is fully load-bearing for FP control; cost gate is now purely a floor.**

### Shape improvement (Δ vs baseline 095555, f1)

| Cell | Change 4 (162148) | Change 5 (193101) |
|---|---|---|
| 1f89ab a-axis | −11% | **−0.2%** ✓ |
| 1f89ab b-axis | −5% | +6% |
| 1f89ab c-axis | −7% | +4% |
| 12345 a-axis | +2% | +13% |
| 12345 c-axis | −26% | −9% |
| e9077 c-axis | −22% | +1% ✓ |
| 1f2ed c-axis | −7% | +11% |
| e3d03 (all axes) | +9%/+8%/+10% | +21%/+21%/+23% (overshoot) |

Most cells now within ±15% of baseline. User visual confirmation: "now it looks better, some cells do still falls a little bit short, but overall looks good."

### Known remaining issues

- **e3d03 over-inflated** by ~20-25% — smallest/dimmest cell, gets relatively more PCA-driven inflation under rs=2.5. Not blocking but visually slightly rounder than real.
- **Some e9077 late-frame daughters shrink aggressively** (e9077..a50 at f22 c-axis = 7px, −45% vs baseline). Fit growth cap may need re-examination for small grand-daughter cells.
- **`e9077..a50` consistent +1 late at f19** — unchanged across all hybrid runs and all our tuning. Not a cost/shape gate issue; needs targeted investigation.

### Rollback

Single-file revert: `git checkout HEAD~1 -- C++/config/config.yaml` reverts all 6 field changes together. Incremental rollback: see the `2026-04-19 (final/size-fix/...)` comment blocks in config.yaml.

### Open follow-ups

- **45-frame validation** — next major test. Confirms FP blocking holds at longer horizon with 0.03 cost fraction + tightened bio gates.
- **e3d03 overshoot** — if bothersome, small cells could get a per-cell radiusScale cap (code change) or `pcaShapeAdaptiveExponent` could be retuned to de-emphasize rs bump for dim cells.
- **e9077..a50 f19 miss** — dedicated investigation: fit quality at f19 vs split criteria.
- **Pooling threshold tuning** — if visual shape is still unsatisfying, `adaptive_cube_pooling_majority_threshold: 0.7 → 0.4` would preserve edge cubes (tried mentally, not empirically this session).

