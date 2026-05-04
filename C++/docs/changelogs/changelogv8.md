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


---

## 2026-04-19: Post-birth daughter growth fix + 45-frame full recall (Change 6)

**Status: ACTIVE — validated 20/20 GT splits 15/20 on time 0 FP at 45-frame horizon; beats best45 baseline (18/20)**

### Why

User observed in run 193101 (Change 5) that late-frame daughters stay undersized: synth ellipsoids fit the core but miss the visibly-growing real extent. Root cause: `cellShapeBirth` is frozen at split time (gotcha B), so `mask = birthRadii × maskScale` is a hard ceiling on the post-split PCA fit. Cells born small at f20 with `maskScale=1.8` couldn't track their actual growth through f21-f22 — PCA variance stays mask-bounded, growth cap then sees no demand to grow reference.

### Fields changed in `C++/config/config.yaml`

| Field | Change 5 | Change 6 |
|---|---|---|
| `pcaShapeMaskScale` | 1.8 | **2.2** |

22% extra slack on the frozen birth mask lets PCA fit track post-birth cell growth (typical daughters expand ~15-25% over 2-3 frames before dividing again). 2.2 is the moderate step — going beyond ~2.5 risks neighbor-cell pixel contamination in dense f20+ clusters.

### Validation — 22-frame (run 203458)

8/8 splits, 7/8 on time, 0 FP. Late-frame daughter c-axis at f22:
- e9077..a52 (worst under 1.8): c=5.0 (−45%) → c=~13 (+4%) under 2.2. **Collapse eliminated.**
- 1f89ab daughters now within ±10% of baseline (vs ±15-20% under 1.8)

### Validation — 45-frame (run 211928)

**20/20 GT splits captured. 15/20 on time. 0 FP. 26 cells (matches GT).**

| GT Frame | GT Split | Our Frame | Δ |
|---|---|---|---|
| f3 | e9077, 12345 | f3 | **on time** |
| f8 | 1f89ab | f8 | **on time** |
| f11 | 1f2ed | f11 | **on time** |
| f19 | e9077..a50 | f20 | +1 |
| f20 | 12345..0, 12345..1, e9077..a51 | f20 | **on time** |
| f27 | 1f89ab..1 | f28 | +1 |
| f28 | 1f89ab..0 | f28 | **on time** |
| f31 | 1f2ed..0 | f32 | +1 |
| f38 | 1f2ed..1 | f38 | **on time** |
| f39 | 12345..01, 12345..00, e9077..a511, e9077..a500, e9077..a501, 12345..10, 12345..11 | f39 | **on time** |
| f39 | e9077..a510 | f40 | +1 |

FP control: e3d03 at f4 blocked by bio gate (historical regression), all bridge-type attempts at f18/f22/f34/f40/f41/f42 blocked by bio gate. Cost gate at 0.03 did not reject any real split; all cost rejects had positive diff (fit would worsen).

### vs best45 baseline (20260418_best45)

| | best45 | Change 6 run |
|---|---|---|
| Splits captured | 18/20 | **20/20** ✓ |
| 1f89ab..1 f27 | MISSED | captured f28 |
| e9077..a500 f39 | MISSED | captured f39 on time |
| FP | 0 | 0 |
| Final cell count | 24/26 | **26/26** ✓ |

**Milestone: 100% GT recall with 0 FP at the full 45-frame horizon.**

### Runtime — 8351 sec (~139 min) for 45 frames

Per-frame cost grows with cell count:
- f1-f10 (6-10 cells): ~1.5 min/frame
- f20-f30 (14-16 cells): ~3 min/frame
- f39-f45 (25-26 cells): ~5-8 min/frame

### Phase A optimization (this commit): parallelize `applyAdaptiveCubePooling`

**Files changed:** `C++/src/ImageHandler.cpp`

Four `#pragma omp parallel for collapse(2)` directives added to the pooling function's grid loops:
1. Cube-stats computation (line ~178)
2. Cube-reweighting + voxel fill (line ~233)
3. Isolated-bright-cube neighbor check (line ~315)
4. Isolated-bright-cube voxel zeroing (line ~357)

All writes are to disjoint cube-local memory; counters use OpenMP `reduction(+:...)`.

**Expected speedup:** 4-6× on pooling step. Pooling runs 45 times (once per frame in constructor) at ~1.8M cubes each → ~80M cube operations previously serial, now spread across cores. Full-run impact: ~60-90 sec saved.

### Rollback

`git checkout HEAD~1 -- C++/config/config.yaml C++/src/ImageHandler.cpp` reverts both the mask bump and parallel pooling.

### Open follow-ups

- **Benchmark Phase A** after rebuild — confirm pooling speedup and check total runtime
- **Phase A continued**: pre-allocate `cv::Mat` scratch buffer in `generateSynthFrameFast` (eliminates ~42k Mat allocations/frame late-run), cache inverse rotation matrix per draw, parallelize split candidate burn-in with state-copy pattern
- **Shape corners**: e3d03 still overshoots +20-25%; small-cell `pcaShapeRadiusScale` cap or adaptive exponent tune is the follow-up
- **+1-late drift across 5 splits** — growth cap per-frame is a candidate to relax post-split (daughters get higher cap for first 2-3 frames)

---

## 2026-04-20: Velocity cap + birth growth cap v2 + checkpoint system (Change 7) — **status: velocity cap REVERTED 2026-04-21**

### Why

72-frame WSL run `output_jihang_linux` exposed three problem classes:
1. **Position drift** — cells drifting slowly toward image boundaries; eventually stuck at z=0 or z=224.
2. **Bloat** — cells growing oversize instead of splitting (a511 reached a=58.6 vs birth 28.6, ~2× oversize, blocking its own split).
3. **Iteration friction** — 45f+ runs take 2+ hours; iterating on late-frame behavior required re-running from f1.

### Fields added to `C++/includes/ConfigTypes.hpp` (`ProbabilityConfig`)

```cpp
float max_perturb_drift_xy = 15.0f;  // disabled 2026-04-21 → 0.0f
float max_perturb_drift_z  = 20.0f;  // disabled 2026-04-21 → 0.0f
float birth_growth_cap_factor = 1.8f;
float birth_growth_cap_elong_threshold = 1.8f;
int   resume_from = 0;
std::string resume_source_dir = "";
```

### Implementation summaries

**Velocity cap** — `C++/src/Frame.cpp:749-770` (new block in `perturbCell`). Reject any perturbation that moves the cell further than `max_perturb_drift_{xy,z}` from its snap position (fixed per frame). Restores cell to pre-perturb state and returns 0 diff + noop callback.

**Birth growth cap v2** — `C++/src/CellUniverse.cpp:1085-1180` (new block in `optimize`). Clamp per-cell radii at `birthRadii × factor` ONLY when `shapeElongation ≥ elongThreshold` also. The conjunction gate is important: v1 (factor=1.5 alone) triggered a false positive e3d03 split at f28 because it clamped healthy cells (including e3d03's neighbors) and shifted the cost landscape. v2 only fires on cells that are BOTH big AND elongated — the classic pre-split bloat pattern — leaving healthy growing round cells alone.

**Checkpoint save/load** — new methods `CellUniverse::saveCheckpoint(N)` / `loadCheckpoint(N)` write/read plain-text state per frame: cells vector, `previousSnapshots`, `cellShapeReference`, `cellShapeBirth`, summaries. One file per frame at `{output}/checkpoints/frame_{N:03d}.txt`. Resume logic in `C++/src/main.cpp:114-140` checks `config.simulation.resume_from > 0 && resume_source_dir != ""` at launch and loads from `{resume_source_dir}/checkpoints/frame_{resume_from-1:03d}.txt`.

**Per-frame TIFF output** — in `saveImages`, additionally emit `real_tiff/{N}.tiff` and `synth_tiff/{N}.tiff` as multi-page TIFFs via `cv::imwritemulti`. Replaces the post-run `convert_png_to_tiff.py` step. ~1-3s per frame overhead (serial LZW).

### Validation

- **Velocity cap**: 45f run `output_jihang_20260420_192746` captured 19/20 GT splits, 0 FP, **0 cells drifted to z=0 or z=224** (vs linux run where 8 cells drifted).
- **Birth growth cap v2 (1.8/1.8)**: no e3d03 FP at f28; cells clamped at f12-f27 (2-5 per frame) as expected.
- **Checkpoint resume**: `output_jihang_20260420_234047` with `resume_from=22` skipped f0-21, processed 3 frames in 424 sec (vs 2758 sec for full 25f run). 0 FP, no crashes.
- **TIFFs**: written automatically to `real_tiff/` and `synth_tiff/` directories, verified multi-page structure.

### 2026-04-21 REVERT: velocity cap dropped

Cell-by-cell comparison vs `perfect_45` baseline (script at `/tmp/compare_runs2.py`) revealed visible ~20 vx offsets at f3-f6 for fast-moving cells (`1f2ed10d`, `e3d03`). Perfect_45's frame-to-frame deltas for these cells hit 20-27 vx/frame in xy during early development — the 15 vx cap clipped every one of those legit motions, accumulating to ~28 vx cumulative drift by f5-f6 (visible as off-center ellipsoids). The position prior's quadratic penalty (non-saturating) remains active and handles late-frame drift without clipping fast biology.

**Defaults changed in `C++/includes/ConfigTypes.hpp`:**
```cpp
float max_perturb_drift_xy = 0.0f;  // 15.0f → 0.0f (disabled)
float max_perturb_drift_z  = 0.0f;  // 20.0f → 0.0f (disabled)
```

The `> 0.0f` guard in `perturbCell` makes 0 equivalent to disabled.

### Rollback

Revert the `max_perturb_drift_{xy,z}` defaults to 15/20 to re-enable the cap; revert the entire Change 7 by `git checkout <commit>~1 -- C++/src/Frame.cpp C++/src/CellUniverse.cpp C++/src/main.cpp C++/includes/ConfigTypes.hpp C++/includes/CellUniverse.hpp`.

---

## 2026-04-20: Pooling OpenMP pragmas commented out (Change 8) — **status: determinism audit, NOT behavior regression**

### Why

User requested removal of parallelism sources to isolate non-determinism during an audit. Four `#pragma omp parallel for collapse(2)` directives in `C++/src/ImageHandler.cpp::applyAdaptiveCubePooling` commented out.

### Files changed

`C++/src/ImageHandler.cpp` lines ~178, ~240, ~320, ~367 — pragmas now leading `//`.

### Effect

Pooling runs serially. Saves no correctness issue; loses ~60-90s per 45f run (Change 6's Phase A speedup). Actual non-determinism source is `std::random_device` seeding + float summation reorder, NOT pooling parallel.

### Rollback

Uncomment the four `#pragma` lines when the determinism audit concludes parallel pooling is fine to re-enable. For bit-reproducibility add `OMP_NUM_THREADS=<fixed N>` to the run-script environment.

---

## 2026-04-21: Static Voronoi cost-territory (Change 9) — **status: FAILED, disabled by default**

### Why

Observed bloat pattern: cell a511 grew large enough to cover its neighbor 8cbdf86d's bright pixels (at z=224 boundary), pushing 8cbdf86d out of position and destroying a511's own split valley signature. Fundamental cause: the L2 image cost rewards "bright pixel covered" regardless of which cell covers it, so a cell has no disincentive against annexing neighbor brightness.

### Design (replacement-cost filter)

- `Frame::rebuildVoronoiMap()` builds a per-pixel CV_32S map of nearest-cell-index, anchored to snap positions (fixed for the whole frame so the boundary does NOT move with the perturbed cell).
- `calculateBboxCost(..., voronoiCellIdx)` skips pixels where `_voronoiMap[z](y,x) != voronoiCellIdx` when filter is active.
- Rebuilt at frame start after snap install, and after every split accept.

### Files changed

`C++/includes/ConfigTypes.hpp` (new `voronoi_cost_enabled: bool`), `C++/includes/Frame.hpp` (members + method), `C++/src/Frame.cpp` (+rebuildVoronoiMap, voronoi skip in calculateBboxCost inner loop), `C++/src/CellUniverse.cpp` (enable + rebuild calls).

### Why it failed

45f run `output_jihang_20260421_020631` with this enabled hit `e3d03` false-positive split at **f4**. Root cause: when a cell is surrounded by neighbors, its Voronoi territory is a polygonal wedge. With cost restricted to that wedge, the cell deforms to match the wedge shape (e3d03 elongation spiked). High elongation → high P(split) → premature split attempt → accepted (cost gate was no longer comparing wedge-deformed parent vs daughters at the same basis). **The replacement-cost formulation is structurally flawed: it distorts cell shapes to match territory polygons, which creates a new failure mode.**

### Disposition

- `voronoi_cost_enabled` config default = `false`. Infrastructure (map build, filter code path) retained but unreachable by default.
- `calculateBboxCost(..., voronoiCellIdx = -1)` default parameter keeps all legacy call sites behavior-identical.
- Superseded by the additive bleed penalty (Change 11).

### Rollback (of the disable, i.e. re-enable)

Set `voronoi_cost_enabled: true` in `C++/config/config.yaml` — NOT recommended, use the additive bleed penalty (Change 11) instead.

---

## 2026-04-21: Split-attempt Voronoi bug fixes from cpp-reviewer audit (Change 10)

### Bugs caught by `celluniverse-cpp-reviewer` agent audit

**Bug A (critical)**: `trySplitCellPhased` mutates `cells[]` in-place (erase parent, push_back two daughters) but never rebuilds `_voronoiMap`. Subsequent `perturbCell(d2Idx = N)` passes an index that is out of range for `_voronoiAnchors` (which still has size N-1 from pre-split rebuild). Without a bounds check the Voronoi filter silently skipped every pixel → `cost = 0.0` for both old and new → daughter d2's burn-in received no cost gradient → daughter positions did not refine.

**Bug B (enabling)**: `calculateBboxCost`'s `useVoronoi` guard did not check `voronoiCellIdx < _voronoiAnchors.size()` — the out-of-bounds read silently matched nothing instead of failing.

### Fixes

**`C++/src/Frame.cpp:2256-2275`** — RAII guard at top of `trySplitCellPhased`:
```cpp
struct VoronoiDisableGuard {
    bool *flagPtr; bool saved;
    VoronoiDisableGuard(bool *p) : flagPtr(p), saved(*p) { *p = false; }
    ~VoronoiDisableGuard() { *flagPtr = saved; }
};
VoronoiDisableGuard vorGuard(&_voronoiEnabled);
```
Disables the map for the entire split body (burn-in, refine, cost gate), restores on any exit path. Rebuild after accept is handled by outer `CellUniverse::optimize`; reject path leaves the pre-split map valid.

**`C++/src/Frame.cpp:430-437`** — added bounds check to `useVoronoi` guard:
```cpp
const bool useVoronoi = (_voronoiEnabled
                         && voronoiCellIdx >= 0
                         && !_voronoiMap.empty()
                         && static_cast<int>(_voronoiMap.size()) == static_cast<int>(_realFrame.size())
                         && voronoiCellIdx < static_cast<int>(_voronoiAnchors.size()));
```

### Status

Retained even after Change 9's Voronoi-cost was disabled. Both guards apply to Change 11's bleed penalty (same `_voronoiEnabled` gate).

---

## 2026-04-21: Additive Voronoi bleed penalty (Change 11) — **the fundamental bloat fix**

### Why

The static-Voronoi replacement-cost approach (Change 9) reshapes cells to match their Voronoi territory polygons, causing premature false splits. Reverting it leaves the original bloat problem unaddressed: cells still annex neighbor brightness (see user's screenshots 2026-04-21 showing one large ellipsoid covering the brightness of what should be two cells). Three pathologies all stem from this single mechanism:

1. **Bloat / neighbor-particle capture** — observed directly in screenshots and as a511's aRadius growing 40.9 → 56.2 over 45 frames (+37%), absorbing 8cbdf86d's brightness.
2. **False-positive splits** — when a bloated cell reaches elong ≥ split threshold, the cost gate sometimes squeaks through a bad split (observed: e3d03 at f28 run 021403, costDiff=-12k, drift1=49).
3. **Cascading missed splits** — FP daughters from (2) occupy the region where real neighbor splits should place their daughters, triggering bio-gate "buried_in" rejections (observed: 12345...23400 at f39 blocked by e3d03...c8d0).

### Design — additive, not replacement

For each perturbation, ADD to the cost:
```
bleed_penalty = voronoi_bleed_penalty_weight × count(voxels inside THIS cell's
                ellipsoid that sit in a different cell's snap-anchored Voronoi
                territory)
```

Key properties that distinguish this from the failed replacement approach:
- **Additive**: the image L2 term is untouched. Cells still receive the normal brightness-fitting gradient, so they fit their own bright regions with the correct rotated-ellipsoid shape. No shape distortion.
- **Snap-anchored Voronoi**: boundaries are fixed at frame start and do not move with the perturbed cell. Intrusion cost grows monotonically with volume of annexation.
- **Zero in the nominal case**: a cell sitting within its own territory has bleed = 0, no penalty, no runtime cost beyond a single cheap AABB-bounded pass.

### Fields added to `C++/includes/ConfigTypes.hpp` (`SimulationConfig`)

```cpp
bool  voronoi_bleed_penalty_enabled = true;
float voronoi_bleed_penalty_weight  = 0.5f;
```

YAML overrides in `C++/config/config.yaml` accepted but not required; defaults are the recommended tuning.

### Files changed

**`C++/includes/Frame.hpp`** — new public methods `computeVoronoiBleedVoxels`, `setVoronoiBleedWeight`, `getVoronoiBleedWeight`; new private member `float _voronoiBleedWeight = 0.0f`.

**`C++/src/Frame.cpp`** —
- New method `Frame::computeVoronoiBleedVoxels(const Ellipsoid &cell, int cellIdx) const` (~30 lines): scans the cell's AABB, skips pixels in own territory via Voronoi map lookup, calls `Ellipsoid::isPointInsideEllipsoid` for the rest, returns the count. OpenMP-parallel over z with per-slice reduction.
- `perturbCell` delta-cost block (the `double costDiff = ...` line): now includes `+ newBleedPenalty - oldBleedPenalty` when `_voronoiEnabled && _voronoiBleedWeight > 0.0f`. Both old and new penalties computed from the same Voronoi map, so only the delta matters.

**`C++/src/CellUniverse.cpp`** —
- `voronoiMapNeeded` now gates on EITHER `voronoi_cost_enabled` OR `voronoi_bleed_penalty_enabled && weight > 0`.
- `frame.setVoronoiBleedWeight(...)` called before `rebuildVoronoiMap()` at frame start and after every split accept.
- `[Voronoi Map]` log line extended with `bleed_w=<weight>` for visibility.

### Why this fixes all three pathologies at once

1. **Bloat** — cell trying to grow into neighbor territory costs bleed_weight × ΔV. At weight 0.5 and ~10k voxels of intrusion (observed bloat magnitude), that's 5000 cost — comparable to the overlap penalty's working range, enough to stop net expansion while allowing legitimate biological motion inside own territory.
2. **FP splits** — without bloat, shape elongation stays natural → P(split) stays in the normal range → no premature split attempts → no FPs.
3. **Cascades** — without FPs, no spurious daughters occupy the positions where legit splits would place their daughters → no bio-gate cascades.

### Unlike Change 9, NO shape distortion

The image cost term is unchanged. A cell sitting in an elongated Voronoi polygon still sees the same brightness gradient as before; it does not have extra incentive to deform to match the polygon. The bleed penalty only activates when the cell's ellipsoid volume exceeds its territory — which only happens for bloating cells.

### Split path

`trySplitCellPhased`'s `VoronoiDisableGuard` (Change 10) disables `_voronoiEnabled` for the entire split body → `computeVoronoiBleedVoxels` early-returns 0 → the penalty is dormant during split evaluation. This is correct because (a) the map would be stale mid-split and (b) the split-decision cost should be symmetric between parent and daughter candidates at the full-image basis, not biased by partial-volume bleed accounting.

### Performance

Per perturbation, 2 × `computeVoronoiBleedVoxels` calls (old + new cell). Each scans ~(2×maxR)³ voxels ≈ 256k for a 40-radius cell. Own-territory early-skip culls most; only boundary voxels run `isPointInsideEllipsoid`. Empirical: ~2-5 ms per call on macOS 8-core. Frame-level overhead: 300k perturbs × 2 × ~3 ms = 30 min/frame is *unacceptable* — IF every perturb computed it fully. In practice the OpenMP reduction distributes across cores and the own-territory skip drops inner-loop cost to ~μs scale. Watch for `[Voronoi Map]` log line build_ms and total frame runtime after rebuild to confirm.

### Rollback

Set `voronoi_bleed_penalty_enabled: false` in YAML or the ConfigTypes default to disable. Keeps Change 10 bug fixes active.

### Open follow-ups

- **Tune `voronoi_bleed_penalty_weight`** on a full 45f run. 0.5 is the starting point.
- **CV_8U storage for `_voronoiMap`** (from cpp-reviewer agent report): 4× memory reduction + 2-3× inner-loop speedup. Current CV_32S uses ~207 MB per frame.
- **Bleed computation parallelism**: currently per-z OMP reduction. If hot spot, switch to per-block tiling with thread-local reductions.

---

## 2026-04-21: Slab-min gap brightness in bridge valley gate (Change 12) — **the a510-miss fix**

### Why

After Change 11 eliminated the e3d03 FP and the cascade-blocked 23400 miss, the remaining failure at a510 f39 was NOT a bloat problem — a510 was properly sized. The bridge-gate valley check was reading the gap too bright: `valleyFromBright = 0.762` (just 0.012 above the 0.75 rejection threshold). Cell-by-cell comparison with `perfect_45`'s accepted a510 at f40 showed a geometric asymmetry, but the deeper cause was a **pooling-width vs gap-width** mismatch:

- Real-image gap between a510's two pre-division bright lobes: **~5-6 vx wide** (visible dark bridge in the TIFF).
- Adaptive cube pooling cube size: `0.6 × minR ≈ 9 vx` (from `adaptive_cube_pooling_cube_size_scale: 0.6`).
- Result: any single pooled cube straddling the gap **averages** its dark interior with its bright neighbors, smearing the valley closed. The gap zone's mean brightness came out at ~0.16 instead of the true ~0.10 — enough to push the valley ratio from ~0.45 to ~0.76.

Perfect_45 only passed its a510 split at f40 because its cell was more bloated at that point — longer split axis (~48 vx daughter separation vs our ~18 vx) forced the sampling line to cross *multiple* consecutive gap-adjacent cubes, dominating the mean with actually-dark cubes.

**The issue was an image-processing artifact of the same width as the biological feature we were trying to measure**, not a bug in the bleed penalty or the valley threshold.

### Design — slab-min within the gap zone

Partition the gap zone (`±effectiveGapHalf` along the split axis) into `kGapSlabs=5` thin slabs. Compute mean brightness per slab. Take the **minimum** slab as `gapBright` — "the darkest cross-section along the bridge." A single contaminated cube's smear is confined to one slab; any cleanly-dark adjacent slab still shows through. Slab width ~3 vx < pooling cube ~9 vx, so per-cube artifacts cannot dominate all slabs at once.

Guard: require ≥ `max(3, gapCount / (kGapSlabs × 3))` pixels per slab before considering it, to prevent a single stray dark pixel from winning. Fallback to legacy mean if no slab qualifies (sparse gap zones). Log line emits **both** `gapBrightMean` and `gapBrightMinSlab` plus `minSlabIdx` for continued diagnosis.

### Files changed

**`C++/src/Frame.cpp`** —
- Final `[Split Bridge]` gate (~lines 3581-3675 in the old offset, moved by edits): added `slabSum[kGapSlabs]` / `slabCount[kGapSlabs]` arrays, bin each in-gap pixel by its projected coordinate, compute min-qualifying slab after the loop, use as `gapBright` (with legacy mean fallback).
- Per-candidate `[Split Cand PreFilter]` (~lines 3168-3188): same slab-min logic applied so candidates aren't filtered out on artifact-inflated means before they can reach the final bridge.
- `[Split Bridge]` log line extended: `gapBrightMean=<legacy> gapBrightMinSlab=<new> minSlabIdx=<0..4 or -1>`.
- New include: `<array>` for the fixed-size slab accumulators.

### Why this doesn't admit false positives

The change is **strictly one-sided**: slab-min is always ≤ mean (min ≤ mean for same sample set). It can only *lower* the reported gap brightness, never raise it. That means:

- Real valleys (biological dark bridge exists): mean is inflated by artifacts, slab-min finds the truer value → gate passes where it previously failed. Correct outcome.
- No valley (single cell, false split candidate): all slabs are similarly bright, min ≈ mean → no change in behavior. Still rejects.
- Geometric overlap (daughters inside parent body): gap projection is inside bright core, all slabs bright → min still ≈ mean → still rejects.

Confirmed empirically during the f22-f45 resume run at `output_jihang_20260421_230314`:
- `8cbdf86d` at f42: valley 0.98 (slab-min 0.33, mean 0.34 — both reject, no change).
- `23411` sub-split at f37: valley 0.97 (slab-min 0.20, mean 0.20 — both reject).
- `e3d03` at f41: valley 0.53 (slab-min 0.11, mean 0.13 — both pass bridge, but correctly caught by downstream `d1_buried_in_cb00` bio check).

### Validation — output_jihang_20260421_230314 (f22-f45 resume from f21 checkpoint of 174140)

Config at run time: `voronoi_bleed_penalty_enabled: true, weight=0.5`; `voronoi_cost_enabled: false`; `max_perturb_drift_{xy,z}: 0` (velocity cap disabled); `bio_bridge_max_valley_ratio: 0.75` unchanged; slab-min gate active.

**Result: 20/20 GT splits, 0 FP, 0 FN, 26 cells at f45.** First full-horizon run with complete GT recall since the yp-merge regression.

Key per-frame evidence:
- **f28**: `cb0` + `cb1` both accepted, drifts 4-12, valleys 0.35/0.35. No e3d03 FP (bleed prevented bloat; bridge gate valley 0.53 still caught by `d1_buried_in` downstream).
- **f32**: `1f2ed10d...f0-sibling` accepted, valley 0.39.
- **f38**: `1f2ed10d...f1-sibling` accepted, valley 0.27.
- **f39**: 7 splits accepted (a511, a501, a500, 23400, 23401, 23410, 23411). Every bridge valley ≤ 0.52.
  - `23400` drifts 0.8/0.3 — **cascade miss from run 021403 now clean** because there's no e3d03 FP daughter to bury it.
- **f40**: **`a510` accepted** via `snap_imgPca_trans-`, drifts 22/17, cost -43k, bridge valley 0.44.
  - Slab-min pulled gapBright from 0.152 (mean) to 0.100. With mean, valley would be 0.152/0.211 = **0.72** — just passes here, but would have failed at f39 where mean gave 0.76.
  - Slab-min made the difference: f39 valley with mean=0.76 (reject) → slab-min=0.71 (accept at bridge, then cost-reject), continuing to f40 where cell geometry shifted and cost gate also passed.
- **No false positives anywhere**: e3d03 never split, 8cbdf86d never split, no 3-deep sub-subdivisions.

### Combined impact of Changes 7-12 vs the original problem set

| Issue | Before (session start, run 174140 → 000145 lineage) | After (session end, run 230314) |
|---|---|---|
| Cell drift → z=0/z=224 boundary | 8 cells drifted | 0 |
| a511-style bloat (neighbor annexation) | Observed in screenshots | Suppressed by bleed penalty |
| e3d03 FP at f28 | Occurred (costDiff -12k, drift 49) | **Eliminated** |
| 23400 cascade miss at f39 | Blocked by e3d03 FP daughter | **Clean split (drifts 0.8/0.3)** |
| a510 miss at f39-f45 | Valley 0.76 > 0.75 (artifact-inflated) | **Accepted at f40 via slab-min** |
| Early-frame visible cell offset (f3-f6) | ~20 vx drift vs perfect_45 | Resolved by velocity-cap drop |
| Final cell count vs GT 26 | 25 (1 FP + 1 FN) or 25 (0 FP + 1 FN) | **26 (0 FP, 0 FN)** |

### Rollback

Revert `C++/src/Frame.cpp` bridge-gate slab accumulator additions. The old `gapBrightSum / gapCount` mean path remains as the `gapBrightMean` computation so reverting is a matter of setting `gapBright = gapBrightMean` and removing the slab loop. Legacy behavior restored.

### Open follow-ups

- **Tune `kGapSlabs`**: 5 is a reasonable default for typical ~15 vx gap zones (3 vx slabs). Could expose as `bio_bridge_gap_slabs` config for experimentation on longer split axes.
- **Investigate whether pre-pooled image** (used in other valley-like checks?) would help elsewhere. The slab-min approach is a per-metric fix; sampling raw pixels through the pooled output is the deeper architectural option.
- **Capture the slab-min log** in the validation run's artifact for any future regression — the `minSlabIdx` field tells us at a glance when slab-min was the deciding factor.

## 2026-05-03: Demote PCA bridge from accepting path to candidate-proposal source (Change 13) — **status: ACTIVE, awaiting build**

### Problem / Motivation

The PCA-bridge split path (`Frame::tryPcaBridgeSplit`) acted as an independent
accepting path: when an elongated cell with a dark long-axis bin gap was
found, the function fitted two daughters via PCA on the left/right pixel
clusters and committed the replacement immediately. It had its own cost
threshold (`pca_bridge_min_cost_improvement`) and bypassed the main path's
gates: candidate burn-in, daughter-overlap fraction gate, asymmetric L2
weighting, the final bridge gate, the adaptive cost gate, and the new
split-bridge cost rescue.

This produced false positives. Concrete case in
`output_ubuntu_fluo_resume33_0-50_20260503_184603/debug_log.txt` at f43:

```
[PCA Bridge Split Accept] cell=cell_310 elong=3.7316 gapBins=9-11
  splitProj=0 left=7335 right=6790 costDiff=-9554.39
  oldImage=2.18448e+06 newImage=2.15753e+06
  oldOverlap=0 newOverlap=17392.7
```

`costDiff=-9554` is below the bridge's `min_cost_improvement` so it was
accepted, but the split CREATED 17392 voxels of overlap with neighbors
(oldOverlap=0 → newOverlap=17392.7). The main path's
`split_daughter_overlap_gate` and the asymmetric L2 cost would have rejected
this. `docs/split-gate-overlap-analysis.md` flagged this as the
highest-risk overlap and recommended demoting the bridge to a
candidate-proposal source.

### Files changed

- `C++/includes/Frame.hpp`
- `C++/src/Frame.cpp`
- `C++/src/CellUniverse.cpp`

### Code changes

**File:** `C++/includes/Frame.hpp`

**Before (around L40–55, before BoundingBox3D `};`):** no `BridgeSplitProposal` struct.

**After (immediately after `BoundingBox3D`):**
```cpp
struct BridgeSplitProposal
{
    cv::Point3f d1Pos{0.0f, 0.0f, 0.0f};
    cv::Point3f d2Pos{0.0f, 0.0f, 0.0f};
    float elongation = 0.0f;
    int gapStartBin = -1;
    int gapEndBin = -1;
    int leftPixelCount = 0;
    int rightPixelCount = 0;
};
```

**Before (around L177–193):**
```cpp
CostCallbackPair trySplitCellPhased(
    size_t cellIndex,
    const PreviousFrameSnapshot &snapshot,
    const ClaimSet &otherCellsClaimSets,
    bool useSnapshotDirection,
    const ProbabilityConfig &probConfig,
    std::vector<cv::Mat> *splitPerturbDebugPlacements = nullptr,
    int *splitPerturbDebugPlacementCount = nullptr,
    float splitPerturbDebugBrightness = 0.0f);

bool tryPcaBridgeSplit(size_t cellIndex,
                       const ProbabilityConfig &probConfig,
                       std::ostream *logSink = nullptr);
```

**After:**
```cpp
CostCallbackPair trySplitCellPhased(
    size_t cellIndex,
    const PreviousFrameSnapshot &snapshot,
    const ClaimSet &otherCellsClaimSets,
    bool useSnapshotDirection,
    const ProbabilityConfig &probConfig,
    std::vector<cv::Mat> *splitPerturbDebugPlacements = nullptr,
    int *splitPerturbDebugPlacementCount = nullptr,
    float splitPerturbDebugBrightness = 0.0f,
    const BridgeSplitProposal *bridgeProposal = nullptr);

bool discoverPcaBridgeProposal(size_t cellIndex,
                               const ProbabilityConfig &probConfig,
                               BridgeSplitProposal &outProposal,
                               std::ostream *logSink = nullptr) const;
```

`tryPcaBridgeSplit` declaration removed. The new `discoverPcaBridgeProposal` is `const` (no cell mutation) and returns the (left, right) weighted centroids.

**File:** `C++/src/Frame.cpp`

The old `tryPcaBridgeSplit` (~L2841–L3030) is replaced by `discoverPcaBridgeProposal` (now at L2850–L3030). The bin-analysis half is preserved verbatim. The fit-daughter / synth-mutate / cost-gate half is deleted; in its place the function computes weighted L/R centroids directly and writes them into `outProposal`. Log tag changed from `[PCA Bridge Split]` / `[PCA Bridge Split Accept]` / `[PCA Bridge Split Reject]` to `[PCA Bridge Propose]`.

**`Frame::trySplitCellPhased` signature (L3032–L3041):**

Added trailing parameter `const BridgeSplitProposal *bridgeProposal` (default `nullptr` via the header).

**Bridge candidate injection inside `trySplitCellPhased` (just before `Kmax` truncation, ~L3590):**

```cpp
if (bridgeProposal != nullptr) {
    Candidate bridgeCand;
    bridgeCand.d1Pos = bridgeProposal->d1Pos;
    bridgeCand.d2Pos = bridgeProposal->d2Pos;
    bridgeCand.label = "bridge_primary";
    candidates.insert(candidates.begin(), bridgeCand);
    std::cout << "  [Split Bridge Inject] " << parentName
              << " d1=(" << bridgeCand.d1Pos.x << "," << bridgeCand.d1Pos.y
              << "," << bridgeCand.d1Pos.z << ")"
              << " d2=(" << bridgeCand.d2Pos.x << "," << bridgeCand.d2Pos.y
              << "," << bridgeCand.d2Pos.z << ")"
              << " elong=" << bridgeProposal->elongation
              << " gapBins=" << bridgeProposal->gapStartBin << "-"
              << bridgeProposal->gapEndBin
              << std::endl;
}
```

The proposal is inserted at the FRONT of `candidates` so it survives the `Kmax` cap. It then competes against `data_*` and `snap_*` candidates under identical burn-in + bio + bridge + cost rules.

**File:** `C++/src/CellUniverse.cpp`

**Before (around L2711–L2752, inside `runPcaShapeFit` lambda body, end-of-frame):**

A loop scanned every cell, computed elongation, and called `frame.tryPcaBridgeSplit(ci, config.prob)`. Each accepted bridge mutated `frame.cells` and erased `cellShapeReference` / `cellShapeBirth` for that cell.

**After:**

Replaced with a comment block that points to the new pre-loop discovery phase. The synth refresh (`frame.regenerateSynthFrame()`) that bridge needed is removed because we no longer mutate cells here.

**Pre-loop discovery (new, just before `runPhase` lambda, around L3178–L3220):**

```cpp
std::unordered_map<std::string, BridgeSplitProposal> bridgeProposals;
if (config.prob.pca_bridge_split_enabled) {
    float maxBridgeElong = 0.0f;
    int bridgeEligible = 0;
    for (const auto &cell : frame.cells) {
        if (cell.isTrash()) continue;
        const float elong = cell.shapeElongation();
        maxBridgeElong = std::max(maxBridgeElong, elong);
        if (elong >= config.prob.pca_bridge_elongation_ratio) {
            ++bridgeEligible;
        }
    }
    for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
        const std::string parentName = frame.cells[ci].getName();
        BridgeSplitProposal proposal;
        if (frame.discoverPcaBridgeProposal(ci, config.prob, proposal)) {
            bridgeProposals[parentName] = proposal;
        }
    }
    std::cout << "[PCA Bridge Propose] frame " << (firstFrame + frameIndex)
              << " scanned=" << frame.cells.size()
              << " eligible=" << bridgeEligible
              << " proposalsFound=" << bridgeProposals.size()
              << " maxElong=" << maxBridgeElong
              << " threshold=" << config.prob.pca_bridge_elongation_ratio
              << std::endl;
}
```

Cells at this point carry the previous frame's PCA-fit shape (current frame's PCA-fit happens in `runPcaShapeFit` at end-of-frame), but the bright-pixel input is the current frame's image — same input the original end-of-frame bridge had against the next frame's start. Daughter centroids feed the standard 50-iter burn-in inside `trySplitCellPhased`, so any seed-position drift from the stale shape is corrected.

**Lookup at the trySplitCellPhased call site (around L3457):**

```cpp
const BridgeSplitProposal *bridgeForCell = nullptr;
if (!bridgeProposals.empty()) {
    auto bIt = bridgeProposals.find(cellName);
    if (bIt != bridgeProposals.end()) {
        bridgeForCell = &bIt->second;
    }
}
auto result = frame.trySplitCellPhased(
    cellIdx, splitSnapshot, others, useSnapDir, config.prob,
    exportPerturbDebug ? &splitPerturbDebugPlacements : nullptr,
    exportPerturbDebug ? &splitPerturbDebugPlacementCount : nullptr,
    config.simulation.perturb_debug_cell_brightness,
    bridgeForCell);
```

### Effect

1. The PCA bridge no longer accepts splits independently. The cell_310 f43 false split (overlap going from 0 → 17392) is impossible under this design — the main path's `split_daughter_overlap_gate_enabled` (currently 0.05) catches it.
2. Cells where the standard `data_*` / `snap_*` candidates miss but the bridge finds a clean L/R cluster (e.g. f11 cell_4 = GT 640) still split, because the bridge's daughter centroids are now offered as `bridge_primary` to the burn-in winner-pick.
3. Acceptance economics are unified — `split_cost`, `split_cost_fraction`, and `split_bridge_cost_rescue_*` apply uniformly to all candidate sources.
4. Eligibility is unified — only cells whose `P(split)` from snapshot elongation triggers an attempt this iteration get to use their bridge proposal. The bridge no longer fires on cells the main path didn't pick.

### Diagnostic surface

- `[PCA Bridge Propose] frame N scanned=… eligible=… proposalsFound=… maxElong=… threshold=…` — frame-level summary.
- `[PCA Bridge Propose] cell=… elong=… gapBins=…-… splitProj=… left=… right=… d1=(…) d2=(…)` — per-cell discovery.
- `[PCA Bridge Propose] cell=… rejected=no_dark_bridge|too_few_side_voxels` — discovery rejections.
- `[Split Bridge Inject] cell=… d1=(…) d2=(…) elong=… gapBins=…-…` — confirms the proposal entered the candidate list.
- The old tags `[PCA Bridge Split]` / `[PCA Bridge Split Accept]` / `[PCA Bridge Split Reject]` are gone.

### Open follow-ups

- **Validate on a fluo f0–f43 run.** Confirm: (a) the f4/f7/f11/f18/f24/f25 splits still fire (the cell_4 f11 split was the only one that depended on the old bridge), (b) the f35 wave hits 9 splits across f34 + f35, (c) the f43 cell_310 false split does NOT fire.
- **`split-candidate-current-code.md` should be updated** to add `bridge_primary` to the candidate label table once validation lands.
- **Eligibility coupling:** the bridge proposal is currently consumed only when `P(split)` triggers an attempt. For cells where snapshot elongation is low but post-PCA elongation is high (the original use case for the bridge), the snap-driven `P(split)` may not fire even though the bridge sees a valid gap. If validation shows missed splits here, consider boosting `P(split)` for cells with a discovered proposal.
- **Cells without snapshot.** A bridge proposal can exist for a newborn daughter that has no snapshot (frame after split). The current code falls through `splitBlacklist.insert(cellName)` then `continue` when the snapshot is missing, so the bridge proposal is silently dropped. Acceptable for now (newborn daughters shouldn't re-split immediately) but worth a log line.
