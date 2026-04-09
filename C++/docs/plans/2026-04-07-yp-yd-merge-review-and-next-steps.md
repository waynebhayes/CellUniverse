# YP/YD Merge Review & Next Steps Plan

**Date:** 2026-04-07
**Branch:** `yp_yd_merge_04072026` (30 commits ahead of `master`, 0 behind)
**Merge commit:** `611dc7d` ("shitty merge")
**Post-merge cleanup:** `803479f` (revert sigmoid constants), `41d21ca` (extract hardcoded values)

---

# Part 1 — Merge Review

## 1.1 What the merge does

The merge commit `611dc7d` combines two independent efforts that both attacked **false cell splits** — the known failure mode where two daughter cells achieve a *lower* residual than the parent even though the split is wrong. Neither branch alone was sufficient; they attack the problem from different angles.

### YP's line (percentile preprocessing + per-cell controls)
Branch: `yp_fix_point_selection_04052026`. Commits `1f0a8bc` … `c2c0f3d` (13 commits).

| Commit | What it changed |
|---|---|
| `1f0a8bc` | **PCA point selection**: bright-pixel collection for split PCA uses **percentile threshold** instead of the mean-intensity threshold. A small number of noisy bright outliers no longer dominate the PCA. |
| `fd86e4d` | **Percentile-based preprocessing** of the real image (in addition to sigmoid). Debugging false positives from frame 5 onward. |
| `8c6cad9` | **Elongation factor** for adaptive P(split) computed from radius ratio directly, not from PCA eigenvalues. |
| `a95a0c6` | **Background calibration** auto-measured from a cell-free zone (wired up through `CellUniverse::loadFrame`). |
| `ca519b7` | **Per-cell brightness amplification factor** added to make the rendering selection stable. |
| `6433e3d` | First "guard for false split" — shape constraints on daughter placement. |
| `823af8a` | **Volume-based split ratio**: daughter sizing uses `volumeScale = cbrt(0.5) ≈ 0.794` instead of ad-hoc radius factors. Mathematically principled — conserves volume. |
| `b42add9`, `99111b8`, `c2c0f3d` | Tuning and false-split handling. |

### Yiding's line (PCA gating + analyzers)
Branch: `Yiding_Eureka_preciseFalseSplitControl_validatedThroughF21_20260405`. 5 commits.

| Commit | What it changed |
|---|---|
| `aee0a91` | **Brightness Volume Analyzer**: standalone tool, separate executable. `BrightnessVolumeAnalyzer.cpp` (1641 LOC) + `EmbryoBrightTracker.cpp/hpp` (1737 LOC). Measures cell brightness/volume over time for offline analysis. |
| `7150f04` | **Expanded split PCA gating**: pre-burn-in rejection of split proposals whose daughter separation is too small relative to daughter radius, or whose split axis is Z-dominated. |
| `f01447b` | **Tightened post-burn-in filter**: reject splits where daughters drift far from initial PCA placement but only achieve a marginal cost improvement. Adds verbose split diagnostics. |

### Post-merge cleanup

| Commit | What it did | Verdict |
|---|---|---|
| `611dc7d` | The merge itself. Author: `Yuancen Pu`. Commit message is literally "shitty merge". 3967 insertions across 15 files. **Config was broken** — set `background_color=0.23`, `cell_color=0.995`, `splitBrightestFraction=0.065`. | Broken window |
| `803479f` "revert path configuration" | Reverts `background_color → 0.0`, `cell_color → 1.0`, `splitBrightestFraction → 0.055`. Commit message is misleading — it's a rendering-constant revert, not a path revert. | Fixes a critical regression |
| `41d21ca` "make hard coded values comfigurable" | Extracts 10 split-gating thresholds from code into config + ConfigTypes: `split_pre_burn_in_min_separation_over_major`, `split_pre_burn_in_z_axis_max_abs`, `split_post_burn_in_large_recenter_min_drift_over_major`, `split_post_burn_in_large_recenter_max_cost_diff`, and others. 182 insertions, 93 deletions. | Genuine cleanup |

## 1.2 Goods

### G1. Soft split-gating is architecturally sound
`Frame.cpp:172-225` gets two new soft rejection gates around the burn-in:
- **Pre-burn-in**: reject if `separationOverDaughterMajor < 0.35` (collapsed proposal) or Z-axis-dominant with weak transverse separation
- **Post-burn-in**: reject if center-drift > 0.85 × parent major radius AND cost improvement barely squeaks past threshold (cost_diff > -40)

These are **gates on candidate proposals**, not hard constraints on cell placement — so they comply with the "no hard overlap rejection" rule from `.claude/rules/gotchas.md` #5. They catch the exact failure mode where a false split drifts daughters around until L2 improves marginally.

### G2. Volume-based daughter sizing
`volumeScale = cbrt(0.5) ≈ 0.794` is mathematically correct (conservation of volume across the split). Cleaner than the old ad-hoc radius ratio.

### G3. Separate analyzer executable, not pollution
`BrightnessVolumeAnalyzer` and `EmbryoBrightTracker` each have their own `main()` and are built into a separate `brightness_volume_analyzer` target in `CMakeLists.txt`. They do NOT link into the main `celluniverse` binary. This is correct — they're offline diagnostic tools, and the core optimizer stays clean.

### G4. `41d21ca` actually extracted the knobs
The 10 new config fields in `ConfigTypes.hpp:163-186` are wired all the way through: `Frame::trySplitCell()` signature was updated, `CellUniverse::optimize()` passes them in, `config.yaml:108-113` exposes them. No dangling references.

### G5. Split diagnostics
`SplitDiagnostics` struct + verbose logging at decision points means you can retrospectively understand *why* a split was accepted/rejected without rerunning with gdb. Useful for tuning the new thresholds.

## 1.3 Bads

### B1. CRITICAL (now fixed but fragile): the merge shipped with wrong sigmoid constants
The merge commit `611dc7d` had `background_color=0.23` and `cell_color=0.995`. These must be `0.0` and `1.0` to match the post-sigmoid image (see `.claude/rules/gotchas.md` #10). If anyone tested between `611dc7d` and `803479f` (~30 minutes), their L2 cost was biased and the optimizer was fighting a wrong target. The revert fixes it but the config now lacks a comment explaining *why* 0.0 and 1.0 are sacred — next time this is easy to re-break.

### B2. Default mismatch for `split_burn_in_iterations`
- `ConfigTypes.hpp:91` → default `500`
- `config.yaml:114` → `1000`

If the YAML field is missing or parse fails silently, burn-in drops from 1000 to 500, which weakens daughter optimization and could re-introduce false splits. No parse-time assertion.

### B3. Duplicated brightness-measurement logic (3-way)
`BrightnessVolumeAnalyzer.cpp` and `EmbryoBrightTracker.cpp` both measure cell brightness/volume independently, and `Spheroid::measureMeanBrightness()` does it a third way. ~3000 LOC of overlap. If a bug is found in one, it must be fixed in three places.

### B4. Stray hardcoded `0.5` in Frame.cpp:668-669
```cpp
const double daughter1BridgeVolume = 0.5 * computeSpheroidVolume(cells[d1Idx]);
```
This is the "bridge cylinder" volume fraction between daughters, used in split validation. `41d21ca` was the "make hardcoded values configurable" commit, yet this one survived.

### B5. Post-burn-in gate rationale is undocumented
The thresholds `0.85` (drift ratio) and `-40` (cost diff) are in `ConfigTypes.hpp:97-98` but **not explained anywhere** — not in `CLAUDE.md`, not in `algorithms.md`, not in the changelog. Future tuning will be guessing.

### B6. `EmbryoBrightTracker.cpp` + `BrightnessVolumeAnalyzer.cpp` are a 3K-LOC cliff without headers/tests
`BrightnessVolumeAnalyzer.cpp` (1641 LOC) doesn't have a corresponding `.hpp` at all — it's all internal-linkage. No unit tests. No documentation of its CLI contract.

## 1.4 Dead code findings

### D1. `SimulationConfig.sigmoid_center_offset` — parsed but never read (VERIFIED)
- Declared `C++/includes/ConfigTypes.hpp:24` (default `0.047f`)
- Parsed `C++/includes/ConfigTypes.hpp:49`
- Exposed in `C++/config/config.yaml:87` and `C++/scripts/config.yaml:80`
- **Zero reads** in `C++/src/`. The active sigmoid calibration at `C++/src/CellUniverse.cpp:422-448` uses `sigmoid_center_percentile`, not `sigmoid_center_offset`.
- `C++/docs/details.md:83,571` still documents it as active — stale doc.
- `C++/docs/conversation_archive_2026-04-05.md:164` confirms: *"sigmoid_center_offset is currently parsed but not used in the active percentile-based calibration path"*.

### D2. No other dead code in the active path
- **0** functions defined-but-uncalled in `CellUniverse.cpp`, `Frame.cpp`, `Spheroid.cpp`, `CellFactory.cpp`, `main.cpp`
- **0** `if (false)` / `#if 0` blocks
- **0** commented-out code blocks > 5 lines
- **All** shell scripts in `C++/scripts/` point to live binaries
- The new analyzers (`BrightnessVolumeAnalyzer.cpp`, `EmbryoBrightTracker.cpp`) are intentionally in a separate executable, not dead.

## 1.5 Risk ranking

| # | Risk | If wrong, what breaks | Fix |
|---|---|---|---|
| 1 | `background_color`/`cell_color` revert is fragile — no comment anchors the correct values | L2 cost gradients bias, cells inflate or shrink, false splits re-appear | Add warning comment in `config.yaml:75-76` pointing to gotchas.md #10 |
| 2 | `split_burn_in_iterations` default mismatch (500 vs 1000) | Silent 2× reduction in burn-in quality if config field disappears → false splits re-appear | Match defaults or require explicit YAML field |
| 3 | 3-way duplicated brightness measurement | Bug fixes don't propagate; diagnostic numbers drift from optimizer numbers | Extract shared `BrightnessMeasurement.hpp` |
| 4 | `sigmoid_center_offset` is parsed-but-dead — users who read config.yaml will think tuning it changes behavior | Wasted tuning effort, confused users, stale docs | Delete the field and update `details.md` |
| 5 | 3K LOC of new analyzer code has zero unit tests and one file has no header | Silent bugs in diagnostics lead to wrong conclusions about split behavior | Add a minimal smoke test + pull shared struct definitions into a header |

## 1.6 Verdict

The merge works and is architecturally correct, but it was committed in a rush — the commit message is "shitty merge", the cleanup took two more commits, and the cleanup itself missed `sigmoid_center_offset` (dead), the hardcoded `0.5` in `Frame.cpp:668`, and the undocumented thresholds. The soft-gate design (pre-/post-burn-in rejection based on geometric properties) is the right approach and respects the project's "no hard overlap rejection" rule.

---

# Part 2 — Next Steps Plan

## 2.1 Plan at a glance

| # | Phase | Scope | Size |
|---|---|---|---|
| 1 | Cleanup pass | Kill `sigmoid_center_offset`, add warning comments on locked sigmoid constants, unify default-vs-yaml drift, configurize `0.5` bridge factor, document post-burn-in gates | Small |
| 2 | Brightness measurement unification | Extract one canonical brightness/volume API from `Spheroid`, port analyzers to it, delete duplicated scan code | Large |
| 3 | Lineage mapping | Record parent on split, emit `lineage.csv`, extend `cells.csv` with `parent_name`, Graphviz dump | Medium |
| 4 | Regression-guarding tests | Sigmoid preprocessing snapshot + brightness-measurement parity + lineage round-trip | Small |

Phase 1 is a single "cleanup pass" that lands the five quick wins from the review in one sweep. Phase 2 is the larger refactor of the 3 duplicate brightness code paths. Phase 3 is the headline lineage feature. Phase 4 is the safety net.

## 2.2 Phase-by-phase steps

### Phase 1: Cleanup pass (single commit)

**1.1 Delete `sigmoid_center_offset` dead code**
- Files: `C++/includes/ConfigTypes.hpp:24,49,67-72` area; `C++/config/config.yaml:87`; `C++/scripts/config.yaml:80`; `C++/docs/details.md:83,571`
- Change: remove the field declaration, remove the parse branch, remove the YAML line in both configs, and update `details.md` to describe `sigmoid_center_percentile` as the authoritative method.
- Acceptance: `Grep "sigmoid_center_offset" C++/src C++/includes C++/config` returns no hits.
- Complexity: Trivial
- Deps: none

**1.2 Add warning comment on locked sigmoid constants**
- File: `C++/config/config.yaml:75-76`
- Change: insert a multi-line comment above `background_color: 0.0` explaining: these values are tied to the sigmoid preprocessing pipeline (cells → ~1.0, background → ~0.0 after `applySigmoid`). Reference `.claude/rules/gotchas.md` #10 and `C++/src/CellUniverse.cpp:446-462`. Say explicitly: "do NOT change unless you are also changing `sigmoid_k`, `sigmoid_center_percentile`, or the sigmoid formula itself." Mirror in `C++/scripts/config.yaml`.
- Acceptance: the warning block appears above both `background_color` lines.
- Complexity: Trivial
- Deps: none

**1.3 Sync `split_burn_in_iterations` default with config**
- Files: `C++/includes/ConfigTypes.hpp:91`; `C++/config/config.yaml:114`; `C++/scripts/config.yaml`
- Change: flip the C++ default from `500` to `1000` so the code-level fallback matches the YAML. Add a comment: "Must stay in sync with config.yaml default — 500 is known to produce weaker burn-in / more false splits."
- Acceptance: all three locations at 1000.
- Complexity: Trivial
- Deps: none

**1.4 Configurize the `0.5` bridge-volume factor**
- Files: `C++/src/Frame.cpp:668-669`; `C++/includes/ConfigTypes.hpp` (ProbabilityConfig block); `C++/config/config.yaml`; `C++/scripts/config.yaml`
- Change: add `split_bridge_volume_fraction` to `ProbabilityConfig` with default `0.5f`, parse it in `explodeConfig()`, print in `printConfig()`. Update `trySplitCell` signature in `Frame.cpp` and `Frame.hpp` to accept the new parameter, replace `0.5 *` with the passed value. Plumb through from `CellUniverse.cpp::optimize()`.
- Acceptance: `Grep "0\.5 \* computeSpheroidVolume" C++/src/Frame.cpp` returns no hits.
- Complexity: Small
- Deps: none
- Risk: signature change to `trySplitCell` — must touch the caller in `CellUniverse.cpp`.

**1.5 Document post-burn-in recenter gate**
- Files: `.claude/rules/algorithms.md` (in the `trySplitCell` section); `C++/docs/details.md`; `C++/docs/changelogv3.md`
- Change: add a "Post-burn-in large-recenter rejection" paragraph explaining:
  - `split_post_burn_in_large_recenter_min_drift_over_major=0.85`: how far the center of mass must move from pre-burn-in position (as fraction of `majorRadius`) to count as "large recenter"
  - `split_post_burn_in_large_recenter_max_cost_diff=-40.0`: if drift is large AND cost improvement < this, reject — the cell is being dragged to absorb a neighbor, not actually splitting
  - Code reference: `Frame.cpp:727-735` and `shouldRejectSplitPostBurnIn()`
- Acceptance: future engineer can tune these from `algorithms.md` alone without grepping source.
- Complexity: Trivial (docs only)
- Deps: none

**1.6 Single changelog entry for the cleanup pass**
- File: `C++/docs/changelogv3.md`
- Change: append one `## 2026-04-07` block listing 1.1 through 1.5 with before/after code per `.claude/rules/changelog.md`. Status tag `**ACTIVE**`.
- Complexity: Trivial
- Deps: 1.1-1.5

### Phase 2: Brightness measurement unification

Three independent brightness/volume measurement code paths exist today (review gap B3). This phase collapses them.

**2.1 Survey the three measurement APIs**
- Read-only: `C++/src/Spheroid.cpp` (find `measureMeanBrightness`); `C++/src/BrightnessVolumeAnalyzer.cpp`; `C++/src/EmbryoBrightTracker.cpp` + `EmbryoBrightTracker.hpp`
- Write `C++/docs/plans/2026-04-07-brightness-unification-notes.md` listing for each: scan shape (3D bounding box vs per-slice), brightness formula (mean/percentile/weighted), volume formula (voxel count vs analytical). Identify canonical behavior — `Spheroid::measureMeanBrightness` is the hot-path reference.
- Complexity: Small
- Deps: none

**2.2 Extract `BrightnessMeasurement` free functions**
- New files: `C++/includes/BrightnessMeasurement.hpp`, `C++/src/BrightnessMeasurement.cpp`
- Change: define free functions `measureMeanBrightness(const SpheroidParams&, const std::vector<cv::Mat>& stack)`, `measureVolume(const SpheroidParams&)`, `measureBrightnessPercentile(...)`. Move body of `Spheroid::measureMeanBrightness` into the free function; the member method delegates. Keep the existing Spheroid method signature so every caller keeps compiling.
- Complexity: Medium
- Deps: 2.1

**2.3 Port `BrightnessVolumeAnalyzer.cpp` to use the new API**
- File: `C++/src/BrightnessVolumeAnalyzer.cpp` (scan loops)
- Change: replace internal scan with calls to `BrightnessMeasurement::`. Keep `main()` at line 1406. Update `CMakeLists.txt` to link `BrightnessMeasurement.cpp` into `brightness_volume_analyzer`.
- Complexity: Medium
- Deps: 2.2

**2.4 Port `EmbryoBrightTracker.cpp` to use the new API**
- Files: `C++/src/EmbryoBrightTracker.cpp`, `C++/includes/EmbryoBrightTracker.hpp`
- Change: same as 2.3 — remove per-cell scan, call into `BrightnessMeasurement::`. Do not touch higher-level filtering.
- Complexity: Medium
- Deps: 2.2

**2.5 Changelog + details.md update**
- Files: `C++/docs/changelogv3.md`, `C++/docs/details.md`
- Change: new changelog entry naming the three files collapsed; add a "Brightness measurement: single source of truth" subsection in `details.md`.
- Complexity: Trivial
- Deps: 2.3, 2.4

### Phase 3: Lineage mapping (the headline feature)

Goal: produce a complete parent-child lineage across all frames as a machine-readable CSV + Graphviz `.dot`.

**3.1 Decide record shape**
- Options:
  - (a) Extend `cells.csv` with a `parent_name` column
  - (b) Emit a separate `lineage.csv` with `frame, parent_name, daughter_name, split_cost_diff, elongation`
  - **Recommended: do both.** (a) is self-contained for all cells.csv consumers (including `LineageViewer`). (b) gives a compact split-event log for downstream analysis.

**3.2 Add parent tracking in `trySplitCell`**
- File: `C++/src/Frame.cpp:586-588` (daughter push) and acceptance callback at `Frame.cpp:737-750`
- Change: when `accept == true`, append a `LineageEvent { frameIndex, parentName, d1Name, d2Name, costDiff }` to a new `Frame::_lineageEvents` vector (new field on `Frame.hpp`). Do NOT record on rejection.
- Risk: verify naming uniqueness in `Spheroid::getSplitCells` so `parent + "0"/"1"` can't collide.
- Complexity: Small
- Deps: none

**3.3 Pass frame index into the split call site**
- File: `C++/src/CellUniverse.cpp` (find `trySplitCell` call in `optimize()`)
- Change: store frame index on Frame via `Frame::setCurrentFrameIndex(int)` at the top of the optimize loop (setter approach preferred over signature bloat).
- Complexity: Trivial
- Deps: 3.2

**3.4 Extend `cells.csv` with `parent_name` column**
- File: `C++/src/CellUniverse.cpp:1069-1088`
- Change: update header at 1069 to append `,parent_name`. For each row at 1075-1088, append `LineageUtils::parentNameOf(params.name)` (extract the logic at `LineageViewer.cpp:24-28` into a new free helper in `C++/includes/LineageUtils.hpp`). Empty string for root cells.
- Acceptance: `cells.csv` has 11 columns; existing columns unchanged.
- Complexity: Trivial
- Deps: none
- Note: this is the backward-compatible slice — ship it first. Delivers lineage-via-inference even without 3.2.

**3.5 Emit `lineage.csv` from recorded events**
- File: `C++/src/CellUniverse.cpp` (new method `saveLineage(int frameIndex)` mirroring `saveCells`)
- Change: append events from `frames[frameIndex]._lineageEvents` to `outputPath + "/lineage.csv"`. Header: `frame,parent_name,daughter0_name,daughter1_name,cost_diff,elongation`. Call after `saveCells(i)` in the frame loop.
- Complexity: Small
- Deps: 3.2, 3.3

**3.6 Graphviz `.dot` export**
- New file: `C++/src/LineageDotExport.cpp` (~60 lines)
- New method: `CellUniverse::writeLineageDot(const std::string& path)` called once at end of run
- Change: iterate all cells keyed by name; for each with a parent (via `LineageUtils::parentNameOf`), emit `"parent" -> "daughter";`. Wrap in `digraph Lineage { ... }`.
- Acceptance: `dot -Tpng lineage.dot -o lineage.png` produces a viewable tree.
- Complexity: Small
- Deps: 3.4

**3.7 LineageViewer compatibility**
- File: `C++/src/LineageViewer.cpp:24`
- Change: **no code change required** — viewer already infers parents from naming. Add comment: "Canonical parent-name inference — also used by `LineageUtils::parentNameOf` and the `parent_name` column in `cells.csv`. If naming convention changes, update all three."
- Complexity: Trivial
- Deps: 3.4

**3.8 Changelog + details.md + rules update**
- Files: `C++/docs/changelogv3.md`, `C++/docs/details.md`, `.claude/rules/gotchas.md` (remove item #2 from "Known Limitations" — lineage is no longer incomplete)
- Complexity: Trivial
- Deps: 3.4-3.6

### Phase 4: Regression-guarding tests

**4.1 Sigmoid preprocessing snapshot test**
- New: `C++/tests/test_sigmoid_preprocessing.cpp`
- Build synthetic 2-slice 100x100 image (cell rectangle at 0.6, background at 0.2), call sigmoid path, assert cells > 0.9 and background < 0.1. Locks "cells → ~1.0, background → ~0.0" invariant.
- Complexity: Small
- Deps: none

**4.2 Brightness-measurement parity test**
- New: `C++/tests/test_brightness_measurement_parity.cpp`
- After Phase 2 lands, assert all three APIs (now delegating to one impl) return identical floats (within `1e-6`) on the same input.
- Complexity: Small
- Deps: Phase 2

**4.3 Lineage round-trip test**
- New: `C++/tests/test_lineage_roundtrip.cpp`
- Build minimal `CellUniverse` with 2 frames + 1 hand-scripted split; assert daughters with `parent + "0"/"1"` naming are written to `lineage.csv` with correct `parent_name`. Verifies 3.2-3.5 end-to-end.
- Complexity: Small
- Deps: Phase 3

**4.4 Register tests in CMake**
- File: `C++/CMakeLists.txt` (the `core_unit_test` target)
- Add the three new `.cpp` files to the test executable sources.
- Complexity: Trivial
- Deps: 4.1-4.3

## 2.3 Phase sequencing

**Order: Phase 1 → 4.1 → Phase 3 → Phase 2 → 4.2+4.3 → 4.4**

Reasoning:
- **Phase 1 first.** Independent, low-risk cleanups. One commit.
- **4.1 immediately after Phase 1**, before any refactor. Cheapest insurance against silently re-breaking sigmoid constants (the `803479f` revert reason).
- **Phase 3 (lineage) before Phase 2 (brightness unify).** Three reasons:
  1. Lineage is the **headline feature** per `CLAUDE.md`'s project goal — user cares more about shipping it than about refactoring internal dupes.
  2. Phase 3 only touches `Frame.cpp` split callback + `CellUniverse.cpp` I/O + a new header. Does not interact with Phase 2's brightness paths.
  3. Phase 3 starts with 3.4, which is **backward-compatible and isolated** — even if the user pauses lineage work after 3.4, they already have inferred `parent_name` in every row.
- **Phase 2 after Phase 3** so lineage CSV can be visually spot-checked before the refactor disturbs brightness measurements used by fake-split detection.
- **4.2 and 4.3** after their respective phases. **4.4** last (mechanical).

**Gate:** do not start Phase 2 until Phase 1 is merged and 4.1 has caught at least one accidental edit.

## 2.4 Risks and open questions

**Open questions to decide before execution:**
- Phase 3.1: lineage record shape — recommend both `parent_name` column + separate `lineage.csv`, but user may prefer only one.
- Phase 2: link `BrightnessMeasurement.cpp` into three separate executables, or make it a small static lib? Static lib cleaner but adds CMake complexity.
- Phase 3.5: `lineage.csv` append-per-frame (mirroring `saveCells`) or write-once at end? Append-per-frame is safer if run crashes mid-way.
- Phase 1.4: is hardcoded `0.5` at `Frame.cpp:668-669` tied to `computeDaughterVolumeRatio` fake-split gate, or independent? If tied, share a constant instead of two parameters.

**Regression risks for the false-split fix:**
- Phase 1.3: if user currently relies on YAML override and code default has never kicked in, change is cosmetic. But if ANY code path constructs `ProbabilityConfig` without YAML parse (test harness?), this change makes burn-in twice as long. Grep for `ProbabilityConfig()` constructors.
- Phase 1.4: must verify the new config key is read in `optimize()` AND passed into `trySplitCell`. Missing a caller means hardcoded value is lost.
- Phase 2: `Frame.cpp:656-657` uses `measureMeanBrightness` in fake-split gating hot path. Numerical drift in extracted free function could shift `bridgeSimilarity` enough to flip accept/reject. This is why 4.2 is non-optional.
- Phase 3.2: record-on-accept inside callback is critical — logging before caller rejection would pollute lineage with rejected splits.

**Extra tests a careful engineer would add:**
- Dense-cluster split burn-in smoke test with fixed random seed, asserts same split accept count (locks in the current false-split fix).
- Post-burn-in large-recenter test feeding `shouldRejectSplitPostBurnIn` synthetic diagnostics, verifies 0.85/-40 gate fires at boundary.

## 2.5 Handoff notes

**For the next Claude session picking this up cold:**

- **Branch:** `yp_yd_merge_04072026` (30 commits ahead of master). Merge `611dc7d` combined YP's percentile/volume line and YD's gating line. Follow-up commits `803479f` and `41d21ca` cleaned part of the gap.
- **Where to re-read first:** `.claude/CLAUDE.md`, `.claude/rules/algorithms.md`, `.claude/rules/gotchas.md` (items #5-10), `.claude/rules/codebase.md`, this plan file, `C++/docs/conversation_archive_2026-04-05.md`, `C++/docs/conversation_archive_2026-04-06.md`.
- **Key files:** `C++/src/Frame.cpp` (`trySplitCell` 529-753, bridge logic 652-710), `C++/src/CellUniverse.cpp` (`loadFrame` 400-494 sigmoid pipeline, `saveCells` 1050-1093), `C++/includes/ConfigTypes.hpp` (91-98 post-burn-in gate thresholds), `C++/config/config.yaml`, `C++/src/LineageViewer.cpp`.
- **What's done at hand-off:** cleanup gaps enumerated (1-5), lineage plan specced (Phase 3), brightness duplication plan specced (Phase 2). NOTHING is coded yet.
- **What's next:** start Phase 1 (one commit, 5 steps, Trivial/Small). Write changelog per `.claude/rules/changelog.md`.
- **Do NOT:** build or run the project, add hard overlap rejection, re-enable `brightness` perturbation, run git commands on user's behalf.
