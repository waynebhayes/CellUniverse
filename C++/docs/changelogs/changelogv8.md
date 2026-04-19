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
