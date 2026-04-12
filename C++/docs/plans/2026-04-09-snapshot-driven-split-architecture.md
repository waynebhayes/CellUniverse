# Snapshot-Driven Split Architecture

**Date:** 2026-04-09
**Branch:** `yp_yd_merge_04072026`
**Status:** Proposed — awaiting user approval
**Authors:** Planning session (with diagnostic context from runs 101212, 112621, 124330, 161559)

---

## 1. Motivation

### 1.1 The four missed splits in run 161559

After landing Pillar B (sphere distance test calibration v2) in run 161559, the
overall split-detection rate stabilized at 4 true positives, 0 false positives,
10 cells at f22 (target 14). The four missing splits are all
**second-generation daughter divisions**:

| Frame | Cell | Ground truth split | Status in 161559 |
|---|---|---|---|
| f19 | `e9077…a50` | parent → a500 / a501 | not detected |
| f20 | `e9077…a51` | parent → a510 / a511 | not detected |
| f20 | `12345…340` | parent → 3400 / 3401 | not detected |
| f20 | `12345…341` | parent → 3410 / 3411 | not detected |

These four cells share a **common failure mode** that the current PR-5
calibration cannot fix.

### 1.2 The timing pathology

Tracing each cell's split attempts across the run reveals a consistent pattern.
Take `12345-340` as the canonical example:

| Frame | `prevElong` (cached) | split-time `strict_effective` | Outcome |
|---|---|---|---|
| f15 | 1.285 | 1.281 | gate skip |
| **f16** | **1.641** | 1.371 (passes 1.30 gate) | reached Pillar B → 1 peak → veto |
| f17 | 1.388 | 1.403 | reached Pillar B → 1 peak → veto |
| f18 | 1.383 | 1.139 | gate skip |
| f19 | 1.079 | 1.039 | gate skip |
| f20 | 1.012 | 1.041 | gate skip |

The cached `prevElong` shows a clear elongation peak at f16-f17 (the cell was
visibly elongated at end of f15). But by the time the actual split *attempt*
fires within f16, the live `strict_effective` measurement has dropped from 1.64
to 1.37 — the optimizer has perturbed the cell ~40 times into a roundier shape
that fits the L2 cost slightly better. By f19, the cell is fully fitted as
spherical and PCA reports 1.04, well below the 1.30 elongation gate.

The split signal **exists** in the cached `previousElongations` map. It is
discarded by the live PCA recomputation in `Frame::trySplitCell`, which
re-checks the cell's *current* state and finds it round.

### 1.3 The strict-interior under-sampling (Pillar A regression)

Compounding the timing issue, the strict-interior PCA (Pillar A) **under-samples
the bright pixel cloud when the cell has been fitted as a small sphere**. For
`12345-340` at f19:

| Measurement | Value |
|---|---|
| `bbox_elongation` (2× expanded box) | 1.462 |
| `strict_halo` (1.0× cell ellipsoid) | 1.039 |
| `strict_topk` (top 10% inside 1.0×) | 1.009 |
| `strict_effective` | 1.039 |

The bbox PCA correctly catches an elongated bright cloud (1.46) — that cloud is
the actual two-blob structure of the dividing cell, lying just outside the
cell's currently-fitted small spherical body. The strict-interior PCA samples
only pixels inside the 1.0× ellipsoid, where the brightness is uniform, and
reports 1.04. **Pillar A's strict-interior was designed to suppress neighbor
leakage but it also suppresses legitimate division signal that lies just
outside the parent cell's fitted body.**

### 1.4 Why Pillar B alone cannot rescue these cases

`e9077-a51` at f18 is a hard counterexample to the "Pillar B will catch what
PCA misses" hypothesis. The cell has `strict_effective = 1.79` (huge), but
Pillar B's raw scanner finds **only one local maximum** in the smoothed ROI
(`peaks_pre_nms = 1`). The cell is shape-elongated by PCA but uniformly bright;
the daughters at this stage are too close together for the sigma-2.5 Gaussian
smoothing to resolve as two distinct peaks.

`e9077-a50` at f18 has the opposite problem: `strict_effective = 1.38` (passes
gate), `peaks_pre_nms = 2`, but `peaks_post_nms = 1` — the post-NMS sphere
distance filter dropped one of the candidates. The user's diagnosis: the
daughters are physically very close, and the sigmoid blends them so only one
post-smoothing local maximum survives the prominence floor.

### 1.5 The central insight

**The current architecture re-decides every split attempt from scratch using
the cell's current (mid-frame, optimizer-fitted) state.** This throws away the
cleaner signal that exists at the end of the previous frame, and replaces it
with a measurement taken AFTER the optimizer has been actively flattening the
elongation that motivated the attempt in the first place.

**The fix is an architectural inversion:** trust the previous frame's
measurement as the authoritative "should this cell try to split?" signal.
Use it for both the trigger (P(split) probability) AND the decision gate.
Place daughters using the previous frame's PCA axis and bright peak structure.
Use the current frame only for two purposes: (a) burn-in refinement of
daughter pose, and (b) post-hoc sanity validation of the result.

---

## 2. Goals and Non-Goals

### Goals

1. **Honor the previous frame's elongation signal as authoritative** for the
   "should this cell attempt a split" decision. Stop re-checking with a live
   PCA on the perturbation-corrupted current state.
2. **Place daughters using snapshot data** (last frame's parent position, PCA
   principal axis, optionally bright peak positions) instead of the
   current-frame `getSplitCells` PCA.
3. **Validate by outcome, not by pre-filter**: post-burn-in checks confirm the
   split makes biological sense, with cost residual + sanity gates.
4. **Allow multi-frame "wait and try"**: a cell that stays elongated for 2-3
   frames should keep getting split attempts each frame, with the snapshot
   updating each time. Soft rejection, not hard blacklisting.
5. **Reduce the guard stack**: with snapshot-based placement, several
   pre-burn-in and post-burn-in heuristic guards become redundant and can be
   deleted.
6. **Subtract neighbor pixels from Pillar B's ROI** to eliminate the neighbor
   leakage failure mode without resorting to coarse Euclidean distance tests.

### Non-Goals

1. **Preprocessing rework.** The blur sigma + sigmoid pipeline is known to
   over-round cells, but fixing it is a separate, larger plan. This plan
   makes the downstream architecture robust enough that preprocessing changes
   become a refinement rather than a precondition.
2. **Frame-anchor drift penalty (Pillar C from the previous plan).** That plan
   was deferred because per-frame anchor at any reasonable threshold either
   over-constrains real motion or under-constrains pathological drift. The
   snapshot-based approach achieves a similar goal for the split path
   (placement uses the previous frame's position) without affecting general
   perturbation motion.
3. **Multi-attempt-per-frame splits.** The current blacklist (one attempt per
   cell per frame) is preserved. Multi-attempt would require larger restructure
   and may not be needed once the architecture honors the cached signal.

---

## 3. Architectural Inversion

### 3.1 Current pipeline

```
Start of frame:
  cache previousElongations[name] from end of last frame
  compute splitProbabilities[name] from previousElongations

Main loop (3500 iterations):
  pick random cell
  if dice rolls split:
    [Frame::trySplitCell]
      getSplitCells(current state, current frame)        ← LIVE PCA #1
      computeStrictInteriorWeightedElongation()          ← LIVE PCA #2
      gate: strict_effective < threshold ? skip          ← KILLS attempts
      gate: insideCount < min ? skip
      pre-burn-in guards
      Pillar B (proposeSplitByLocalMaxima)
      gate: !valid ? skip
      cells.erase(parent), push(daughters)
      burn-in (1000 iters)
      post-burn-in fake-guards
      post-burn-in large-recenter guard
      gate: costDiff < -split_cost ? accept
  else:
    perturbCell

End of frame:
  for each cell: previousElongations[name] = strict_effective at end-of-frame
```

### 3.2 New pipeline

```
Start of frame:
  load previousSnapshots[name] (struct with elongation, axis, position, peaks)
  compute splitProbabilities[name] from snapshot.elongation
  HARD CUT: if snapshot.elongation < threshold, set splitProb[name] = 0

Main loop (3500 iterations):
  pick random cell
  if dice rolls split AND snapshot.elongation >= threshold:
    [Frame::trySplitCellFromSnapshot]
      no live elongation gate
      no live strict PCA recomputation
      build daughters from snapshot.position, snapshot.principalAxis,
        and snapshot.brightPeaks (if available)
      cells.erase(parent), push(daughters)
      burn-in (1000 iters) — refines daughter positions to current frame
      post-burn-in biological sanity gates:
        - daughter sizes above floor + epsilon
        - daughter positions within reasonable distance of snapshot.position
        - daughter brightness above threshold
        - PILLAR B POST-HOC: each daughter's body contains a bright peak
          in the smoothed real image (NOT in dark space)
        - existing daughter overlap / volume ratio gates (still useful)
      gate: costDiff < -split_cost ? accept
      soft reject: if any sanity gate fires, revert WITHOUT blacklist
        (cell may try again next frame if still elongated)
  else:
    perturbCell

End of frame:
  for each cell:
    snapshot.elongation = max strict_effective seen this frame (Loop-A style)
    snapshot.position = cell's end-of-frame position
    snapshot.principalAxis = end-of-frame PCA principal eigenvector
    snapshot.majorRadius/minorRadius = end-of-frame values
    snapshot.brightPeaks = result of Pillar B run with neighbor masking
                           (next frame's split attempt will use these as
                            daughter placement candidates)
```

### 3.3 The key inversions

| Aspect | Current | New |
|---|---|---|
| **Authority** for "should this cell split" | Live state mid-frame | End-of-previous-frame snapshot |
| **Daughter placement** input | Current-frame `getSplitCells` PCA | Snapshot's PCA axis + bright peaks |
| **Pillar B's role** | Pre-filter on parent ROI before placement | (a) End-of-frame peak finder for next frame's snapshot, AND (b) post-hoc per-daughter validator |
| **Rejection model** | Pre-filter heavy, post-hoc light | Pre-filter minimal, post-hoc gates |
| **Multi-frame retry** | Implicit (each frame independent) | Explicit (snapshot updates each frame, soft reject doesn't blacklist) |

---

## 4. Detailed Components

### 4.1 The `PreviousFrameSnapshot` data structure

New struct in `C++/includes/Spheroid.hpp` (or a new header):

```cpp
struct PreviousFrameSnapshot
{
    // Elongation signal
    float strictEffectiveElongation = 1.0f;  // peak observed during last frame

    // Pose (end-of-frame, after optimization stabilized)
    cv::Point3f position{0, 0, 0};
    float majorRadius = 0.0f;
    float minorRadius = 0.0f;
    cv::Point3f principalAxis{1, 0, 0};   // λ1 eigenvector — split direction
    float thetaX = 0.0f, thetaY = 0.0f, thetaZ = 0.0f;
    float brightness = 0.5f;

    // Bright peaks from end-of-frame Pillar B run with neighbor masking.
    // If exactly 2 peaks, used directly as daughter centers in the next
    // frame's split attempt. If 0 or 1, fall back to PCA-axis placement.
    std::vector<cv::Point3f> brightPeaks;
    std::vector<float> brightPeakIntensities;

    // Parent bbox at end-of-frame, used as the ROI for next frame's
    // post-hoc Pillar B daughter validation (so the validation samples
    // the same volume the snapshot was built from).
    int bboxMinX = 0, bboxMaxX = 0;
    int bboxMinY = 0, bboxMaxY = 0;
    int bboxMinZ = 0, bboxMaxZ = 0;

    bool valid = false;  // false on frame 1 (no previous frame)
};
```

Stored in `CellUniverse` as `std::map<std::string, PreviousFrameSnapshot>
previousSnapshots`. Replaces the existing `std::map<std::string, float>
previousElongations`.

### 4.2 Snapshot capture timing

**Recommendation: hybrid capture** combining the cheap and the comprehensive.

- **Running max during the frame**: every time `computeElongationForCell` is
  called (which happens periodically during the main loop), check if the new
  `strict_effective` exceeds the running max for this cell. If yes, snapshot
  the cell's full state at that moment. This preserves the peak elongation
  signal even when the optimizer rounds the cell out by end of frame.
- **End-of-frame finalization**: after the loop exits, run one final
  `computeElongationForCell` for every cell, plus a Pillar B peak-finder pass
  with neighbor masking. The `principalAxis`, `position`, `majorRadius`,
  `minorRadius`, `theta_x/y/z`, `bbox*`, and `brightPeaks` fields are
  finalized from this end-of-frame state. The `strictEffectiveElongation`
  field uses `max(running_max, end_of_frame)`.

This way the elongation signal captures the strongest moment during the frame
(important for detecting transient elongation that fades), while the geometric
fields use the end-of-frame state (important for placement consistency with
the next frame's burn-in starting point).

### 4.3 Hard `prevElong` gate at the trigger site

In `CellUniverse::optimize`, replace the soft probability formula with a hard
cutoff:

```cpp
const float MIN_TRIGGER_ELONGATION = 1.20f;  // configurable

for (const auto &cell : frame.cells) {
    auto p = cell.getCellParams();
    auto it = previousSnapshots.find(p.name);
    const float prevElong = (it != previousSnapshots.end() && it->second.valid)
        ? it->second.strictEffectiveElongation : 1.0f;

    if (!allowSplits || prevElong < MIN_TRIGGER_ELONGATION) {
        rawSplitProbabilities[p.name] = 0.0f;  // never attempt split
        continue;
    }

    rawSplitProbabilities[p.name] =
        baseSplitProb + std::max(0.0f, 1.0f - 1.0f / prevElong);
}
```

The threshold (1.20) is below the current 1.30 elongation gate because the
snapshot value is the *peak* observed during the previous frame, which is more
optimistic than the live measurement. With Loop-A style max-tracking, even
cells that briefly elongate during a frame will get carried forward.

### 4.4 New `Frame::trySplitCellFromSnapshot` method

Replaces the existing `trySplitCell` for the snapshot-driven path. New
signature:

```cpp
CostCallbackPair trySplitCellFromSnapshot(
    size_t cellIndex,
    const PreviousFrameSnapshot &snapshot,
    /* + sanity-check thresholds */);
```

Body, in order:

1. **No live PCA recomputation, no live elongation gate.** Trust the snapshot.
2. **Build daughter pair from snapshot:**
   - Daughter major radius: `cbrt(0.5) × snapshot.majorRadius` (volume conserved)
   - Daughter minor radius: `cbrt(0.5) × snapshot.minorRadius`
   - **Placement priority**:
     - If `snapshot.brightPeaks.size() == 2`: use those as daughter centers
       directly (bright peaks from the previous frame's Pillar B run encode
       the actual two-blob structure)
     - Else: place along `snapshot.principalAxis`, separated by
       `1.0 × snapshot.majorRadius`, centered on `snapshot.position`
   - Daughter rotation: inherit `snapshot.thetaX/Y/Z` minus an angular offset
     so each daughter's local "long axis" still points roughly along the split
     direction (TODO: pick a sensible offset, possibly zero for now)
   - Daughter brightness: inherit `snapshot.brightness`
3. **Replace parent with daughters in the `cells` vector** (existing logic).
4. **Save current frame state** (synth, cost cache) so we can revert.
5. **Burn-in (1000 iterations)** — refines daughter positions and shapes to
   the current frame's image. Same loop as today, including the burn-in min-R
   clamp from PR-5 calibration.
6. **Post-burn-in biological sanity gates** (see §4.5).
7. **Cost residual gate**: `costDiff < -split_cost` to accept; otherwise
   **soft reject**.
8. **Soft reject** = revert daughters to parent, BUT do not blacklist the
   cell. The cell can be tried again later in this frame (if the dice rolls
   split again — unlikely with the blacklist still in place per current
   semantics) or, more importantly, in the *next* frame if it's still
   elongated.

### 4.5 Post-burn-in biological sanity gates

Every gate is a SOFT reject that does not blacklist the cell.

| # | Gate | Threshold | Reasoning |
|---|---|---|---|
| 1 | Each daughter `majorRadius > 1.5 × minMajorRadius` | tunable | Reject if a daughter is at the floor — degenerate placement |
| 2 | Each daughter `minorRadius > 1.5 × minMinorRadius` | tunable | Same for minor |
| 3 | Each daughter position drift `< 1.5 × snapshot.majorRadius` from snapshot center | tunable | Daughters refine but don't fly off |
| 4 | Daughter brightness `> 0.5 × snapshot.brightness` | tunable | Daughter has signal, not floating in noise |
| 5 | **Pillar B post-hoc per daughter** — see §4.6 | bright peak in body | Daughter lands on a bright peak, not dark space |
| 6 | Daughter overlap volume fraction (existing fake-guard) | unchanged | Already tested |
| 7 | Daughter volume ratio (existing fake-guard) | unchanged | Already tested |
| 8 | Bridge brightness similarity (existing fake-guard) | maybe relax | The relaxed valley check from PR-5 calibration |

Gates 1-5 are new (added by this plan). Gates 6-8 already exist and are
preserved.

### 4.6 Pillar B post-hoc per-daughter check (the new gate 5)

Pillar B currently runs as a pre-filter on the parent's ROI. In the new
architecture, it has TWO distinct roles:

#### Role A: end-of-frame snapshot peak finder

After every frame's main loop completes, for each surviving cell, run Pillar
B with **neighbor pixel masking** (see §4.7) on the cell's current bbox. Store
the resulting peaks in `snapshot.brightPeaks`. These peaks become the daughter
center candidates for the NEXT frame's split attempt on this cell.

#### Role B: post-hoc per-daughter validator (gate 5)

After burn-in completes, for each daughter independently:

1. Build a small ROI around the daughter (`1.0 × daughter majorR`).
2. Run a simplified peak finder on the smoothed real image inside that ROI
   (same Gaussian smoothing as Pillar B, prominence floor `mean + σ`).
3. **Fail if** no local maximum exists inside the daughter's analytic ellipsoid
   AND the mean brightness inside the daughter's ellipsoid is below
   `snapshot.brightness × 0.5`.

This is a binary check: "is the daughter sitting on something bright?" If
both daughters fail, the split is rejecting the cell into dark space and we
revert. If only one fails, that's also a reject (we don't want one good
daughter and one ghost daughter).

This gate is **structurally simpler** than the current Pillar B because it
doesn't need 2-peak detection or valley validation — it just needs to confirm
each daughter is positioned on a bright region. The 2-peak detection happens
upstream (in Role A, end of previous frame).

### 4.7 Neighbor pixel masking before Pillar B

When Pillar B (Role A or B) scans an ROI, it currently has to deal with
brightness from neighboring cells leaking into the ROI. The current sphere
distance test on each peak is a coarse approximation; replace it with explicit
masking.

Algorithm for `proposeSplitByLocalMaximaWithNeighborMasking`:

1. Build the ROI bbox as before (`1.8 × effMajorR` around the cell center).
2. For each neighbor cell whose bbox intersects this ROI:
   - For each voxel in the ROI, test if it lies inside the neighbor's analytic
     ellipsoid using `Spheroid::isPointInsideEllipsoid` at `1.0 × scale` (or
     slightly larger, e.g. `1.1×`).
   - If yes, set that voxel's mask to "excluded" — the smoothed image at that
     voxel is replaced with the ROI mean, so it cannot register as a peak.
3. Smooth the masked ROI as before.
4. Run peak detection on the masked smoothed image.
5. Apply the existing prominence + NMS + valley checks (no need for the
   sphere distance test — it's replaced by the mask).

This addresses the e3d034 / 1f89abf-cb0 false-positive cases from earlier
runs without the calibration headaches of the sphere distance test. The mask
is exact (it knows which voxels belong to which cell) instead of distance-
based-approximate.

The existing `Spheroid::isPointInsideEllipsoid` method (added in PR-5
calibration but currently unused) becomes the foundation for this masking.

### 4.8 Soft rejection vs blacklist

Current semantics: any failed split attempt blacklists the cell for the rest
of the frame, preventing further split attempts.

New semantics: distinguish two failure types.

1. **Hard reject** (blacklist for the frame): the cell is structurally not a
   split candidate. Trigger conditions:
   - Snapshot is invalid (frame 1, or new daughter cell)
   - Hard biological sanity violation (e.g., daughter would render zero
     pixels, daughter at min radius)
2. **Soft reject** (allow another attempt this frame OR next frame): the
   placement was reasonable but the result wasn't good enough. Trigger
   conditions:
   - Cost residual not improved enough (`costDiff > -split_cost`)
   - Daughters land on weak brightness regions
   - Daughters drift too far during burn-in

For the user's "cell stays pancaking 2-3 frames before splitting" case: a
soft-rejected cell carries its high `prevElong` into the next frame's
snapshot, and the next frame tries again with a fresh refinement. Over
multiple frames, the optimizer's chance of finding the right placement
compounds.

Within a single frame, soft reject still effectively blacklists (the dice is
unlikely to roll split again given how P(split) is computed once per frame),
but it doesn't prevent the conceptual retry across frames.

---

## 5. Code Removal Candidates

The new architecture makes several existing pieces of code redundant or
weaker than the new mechanisms. Each is a candidate for deletion in PR-E.

### 5.1 Live elongation gate in `Frame::trySplitCell`

```cpp
// Frame.cpp:718-732
if (splitElongationThreshold > 0.0f && effectiveElongation < splitElongationThreshold)
{
    // [Split Skip] reason=below_elongation_threshold
    return {0.0, [](bool accept) {}};
}
```

**Status: DELETE.** Replaced by the hard `prevElong` cutoff at the trigger
site. The trigger never even calls `trySplitCellFromSnapshot` for cells whose
snapshot elongation is below threshold.

### 5.2 Pre-burn-in `shouldRejectSplitPreBurnIn` guards

```cpp
// Frame.cpp:172-210 — collapsed-overlap + z-axis-dominant checks
shouldRejectSplitPreBurnIn(...)
```

**Status: PROBABLY DELETE.** These guards were designed to filter
pathological PCA outputs from `getSplitCells` (e.g., daughters placed
on top of each other due to bright-pixel centroid drift). With
snapshot-based placement, these pathologies don't occur — the daughter
centers come from the previous frame's PCA axis or bright peaks, both of
which are stable.

Validation: run with the guards still in place but log when they would have
fired in the new architecture. If they never fire, delete in PR-E. If they
catch real edge cases, keep but tune.

### 5.3 Post-burn-in large-recenter guard

```cpp
// Frame.cpp:212-234
shouldRejectSplitPostBurnIn(...)
```

**Status: PROBABLY DELETE.** Detects daughters that drifted significantly
from their initial PCA-derived placement during burn-in, with weak cost
improvement. The new architecture replaces this with the more direct gate 3
(daughter position drift from snapshot < 1.5 × snapshot.majorRadius), which
checks the same thing more cleanly.

### 5.4 Sphere distance test inside Pillar B

```cpp
// Frame.cpp inside proposeSplitByLocalMaxima NMS loop
if (peakInsideEllipsoidScale > 0.0f) {
    /* sphere distance check */
}
```

**Status: DELETE.** Replaced by the explicit neighbor pixel masking before
the smoothing step. The mask knows which voxels belong to which cell, so it
is more precise than the distance-based approximation.

### 5.5 Top-K elongation in strict-interior PCA

```cpp
// Spheroid.cpp computeStrictInteriorWeightedElongation
// — both halo-weighted and top-K paths
```

**Status: KEEP, but its role changes.** The strict-interior PCA still drives
the snapshot's elongation value. The principal axis (eigenvector) becomes
explicitly used for daughter placement, which it currently isn't. The top-K
branch may still help — keep both halo and top-K, take the max as before.

### 5.6 `Spheroid::getSplitCells` PCA-driven daughter placement

```cpp
// Spheroid.cpp ~330-650
getSplitCells(realFrame, ..., neighborCenters, preOpt*)
```

**Status: KEEP for end-of-frame Pillar B / snapshot building, but not
called by `trySplitCellFromSnapshot`.** The new path uses the snapshot
directly. The existing method becomes a helper for building snapshots and
running diagnostics.

Long term we may want to delete it entirely once the new path is stable, but
for the initial PR landing, keep it for safety and rollback.

### 5.7 `splitElongationThreshold` parameter

**Status: DELETE the parameter from `trySplitCell` signature, REPLACE with
new `MIN_TRIGGER_ELONGATION` config in `CellUniverse::optimize`.** Same
threshold value, different consumer.

### 5.8 `previousElongations` map

**Status: REPLACE with `previousSnapshots` map.** The scalar map is too thin
to support snapshot-based placement.

---

## 6. Implementation Phasing

Six PRs, each independently reviewable. PR-A through PR-D implement the
architecture; PR-E removes the now-redundant code; PR-F is the validation
sweep.

### PR-A: Snapshot capture infrastructure

**Files:** `Spheroid.hpp`, `Spheroid.cpp`, `CellUniverse.hpp`,
`CellUniverse.cpp`, `Frame.hpp`, `Frame.cpp`, `ConfigTypes.hpp`, `config.yaml`.

**Scope:**
- Add `PreviousFrameSnapshot` struct.
- Add `Spheroid::computePrincipalAxis()` helper that returns the λ1
  eigenvector from the strict-interior PCA.
- Add `previousSnapshots` map to `CellUniverse`, retain `previousElongations`
  in parallel for transition (delete in PR-E).
- At end of each frame's optimize loop, populate the snapshot for every cell.
- Track running max strict_effective during the frame (Loop-A from earlier
  discussion) and use `max(running_max, end_of_frame)` as the snapshot's
  elongation field.
- No behavioral change yet — the snapshot is captured but not used for
  decisions.

**LOC estimate:** ~120

### PR-B: Hard `prevElong` cutoff + remove live gate

**Files:** `CellUniverse.cpp`, `Frame.cpp`, `Frame.hpp`, `ConfigTypes.hpp`,
`config.yaml`.

**Scope:**
- Add `MIN_TRIGGER_ELONGATION` config field (default 1.20).
- In `CellUniverse::optimize`, set `splitProbabilities[name] = 0` whenever
  `previousSnapshots[name].strictEffectiveElongation < MIN_TRIGGER_ELONGATION`.
- Delete the live elongation gate inside `Frame::trySplitCell`.
- Delete `splitElongationThreshold` parameter from `trySplitCell` signature.
- The placement still uses `getSplitCells` (current behavior) — only the
  trigger logic changes. This isolates the impact.

**LOC estimate:** ~50

### PR-C: Snapshot-driven daughter placement

**Files:** `Frame.hpp`, `Frame.cpp`, `Spheroid.hpp`, `Spheroid.cpp`,
`CellUniverse.cpp`.

**Scope:**
- New method `Frame::trySplitCellFromSnapshot(cellIdx, snapshot, ...)`.
- New helper `Spheroid::buildDaughterPairFromSnapshot(snapshot)` that
  constructs the daughter pair from snapshot data without any current-frame
  PCA.
- `CellUniverse::optimize` switches to calling the new method when
  `snapshot.valid`; falls back to legacy `trySplitCell` for frame-1 splits
  (which can't have a snapshot).
- Burn-in unchanged.
- Existing post-burn-in fake-guards unchanged.

**LOC estimate:** ~200

### PR-D: Pillar B with neighbor masking + post-hoc gates

**Files:** `Frame.cpp`, `ConfigTypes.hpp`, `config.yaml`.

**Scope:**
- New variant of `proposeSplitByLocalMaxima` that takes a list of neighbor
  `Spheroid` objects (not just centers) and uses
  `Spheroid::isPointInsideEllipsoid` to mask out neighbor voxels before
  smoothing.
- Pillar B Role A: called at end-of-frame for every cell to populate
  `snapshot.brightPeaks`.
- Pillar B Role B: called per-daughter after burn-in as the new gate 5.
- New post-burn-in sanity gates 1-4 (size, position, brightness).
- Soft-reject mechanism (no blacklist on biological sanity failures).

**LOC estimate:** ~250

### PR-E: Code removal pass

**Files:** `Frame.cpp`, `Frame.hpp`, `Spheroid.cpp`, `Spheroid.hpp`,
`CellUniverse.cpp`, `ConfigTypes.hpp`, `config.yaml`.

**Scope:**
- Delete live elongation gate (already done in PR-B but final cleanup of
  parameters and config).
- Delete `shouldRejectSplitPreBurnIn` if validation in PR-A through PR-D
  shows it never fires on the new path.
- Delete `shouldRejectSplitPostBurnIn` (large-recenter guard) — replaced by
  the snapshot-based drift check (gate 3).
- Delete the sphere distance test from `proposeSplitByLocalMaxima` —
  replaced by neighbor masking in PR-D.
- Delete `previousElongations` map — replaced by `previousSnapshots`.
- Delete legacy `trySplitCell` and `getSplitCells` if the new
  `trySplitCellFromSnapshot` covers all cases (validate on multiple runs
  first; this deletion can be deferred to a later PR if needed).

**LOC estimate:** ~150 deletions

### PR-F: Validation sweep

**Scope:**
- Run the test data with all PRs landed.
- Verify all 4 missed second-gen splits are detected (e9077-a50/a51,
  12345-340/341).
- Verify no FP regressions (e3d034, 1f89abf-cb0).
- Verify cell count at f22 reaches 14 (the ground truth target).
- Compare timing — burn-in cost should be similar to current; the only added
  cost is end-of-frame Pillar B per cell (~5-10ms × num_cells).
- If a clean run lands 14 cells with 0 FPs and acceptable timing, PR-F
  documents the result and the architecture is complete.

---

## 7. Open Questions and Risks

### 7.1 Snapshot timing — running max OR end-of-frame?

The plan above uses both: `max(running_max, end_of_frame)`. This requires
calling `computeElongationForCell` periodically during the main loop, which
adds compute. Possible compromises:

- Call it every 50 iterations
- Call it only when a cell is selected for perturbation
- Skip the running max entirely; use only end-of-frame

The cheapest option is end-of-frame only, but that loses the peak signal that
we know exists (12345-340 at f16 had peak elongation 1.64 mid-frame, dropped
to 1.40 by end of frame). Recommendation: every 50 iterations as a starting
point, tune later.

### 7.2 What happens to a cell that never elongates above threshold?

In the new architecture, a cell with `prevElong < 1.20` for every frame of
its life never gets a split attempt. If a real division goes through a cell
that for whatever reason never registers as elongated above threshold, the
division will be missed entirely. This is a tradeoff: the threshold prevents
false positives but limits sensitivity.

Mitigation: lower the threshold to 1.15 or 1.10 if validation shows missed
splits have peak elongation in that range.

### 7.3 Snapshot staleness for fast-moving cells

The snapshot is captured at the end of frame N. The split attempt happens at
some iteration of frame N+1, by which time the cell may have moved. The
snapshot's `position` field becomes stale. For cells that move slowly
(<10 px/frame), this is fine. For cells that move 20+ px/frame, the daughter
placement based on `snapshot.position` may land 20+ px from where the daughters
should actually be.

Mitigation: in `trySplitCellFromSnapshot`, after computing the snapshot-based
initial placement, run a brief 1-iteration "centroid recenter" — move the
daughter pair so its midpoint matches the cell's current center. This
preserves the snapshot's PCA axis and daughter separation but re-anchors to
the current cell position.

### 7.4 Pillar B at end-of-frame compute cost

Per cell, Pillar B with neighbor masking on a `1.8 × majorR` ROI is
~5-10ms based on the current implementation profile. For 10 cells, that's
50-100ms per frame, added to a frame budget of ~10-20s. ~0.5% overhead.
Negligible.

### 7.5 The first-frame bootstrap problem

Frame 1 has no previous snapshot, so no cell can split. This is already true
in the current code (`allowSplits = (frameIndex > 0)`), so no regression.

### 7.6 Daughter cells inherit snapshots from where?

When a cell `parent` splits into `parent0` and `parent1`, neither daughter has
a previous snapshot. They cannot split for the first frame after their
creation. This is fine — biology doesn't have cells that immediately divide
again. The first frame after a split, the daughters get new snapshots from
their own state at end-of-frame.

### 7.7 Cells stay pancaking for 2-3 frames

Confirmed by user observation: a cell may stay elongated for 2-3 frames
before the actual division happens. The new architecture supports this
naturally:

- Frame N: cell elongated, split attempted, soft-reject (cost not enough)
- Frame N+1: cell still elongated, snapshot updates, split attempted again
- Frame N+2: cell still elongated, snapshot updates, split attempted, succeeds

The blacklist within a frame still applies (one attempt per frame), but
across frames the cell keeps trying as long as its snapshot remains
elongated. This is the right behavior — burn-in noise on any single frame
shouldn't kill a real split.

### 7.8 Preprocessing complementary fix

The architecture is robust to preprocessing degradation, but the most
noticeable improvement would come from preserving more elongation in the
preprocessed image. Two complementary paths (deferred to a separate plan):

- **Lower `blur_sigma`** from 10 to 5 or 3. Risk: noise in cost gradient.
- **Dual pipelines**: heavy-blurred for L2 cost, light-blurred for PCA and
  Pillar B. Risk: 2× memory.

Defer until after this plan ships and we measure how much of the gap closes
from architecture alone.

---

## 8. Validation Plan

Each PR has its own go/no-go criteria:

### PR-A (snapshot capture)

- All cells have a snapshot at end of every frame.
- Snapshot elongation values match or exceed `previousElongations` for all
  cells.
- No behavioral change in split decisions (because nothing reads the new
  snapshot yet).
- Confirmed by: rerun on test data, compare cells.csv to a baseline run.

### PR-B (hard cutoff + live gate removal)

- Splits that previously fired at `strict_effective ≥ 1.30` should still
  fire (because their `prevElong` is also ≥ 1.30 in those cases, just
  measured one frame earlier).
- Splits that previously failed at the live gate (12345-340 f17, e9077-a50
  f18) may now proceed past the gate but get caught by other downstream
  gates — this is fine, we'll fix those in PR-C/D.
- Confirmed by: cell count should not decrease vs run 161559 (still ≥ 10).

### PR-C (snapshot-driven placement)

- The 4 missed second-gen splits should now have a chance to fire because
  their snapshots have high elongation (12345-340 has 1.64 at f16, 12345-341
  has 1.61, etc.).
- Confirmed by: at least 2 of the 4 missed splits should be accepted in the
  test run.

### PR-D (Pillar B masking + post-hoc gates)

- All 4 missed second-gen splits should now fire.
- No new FPs introduced (e3d034 cascade, 1f89abf-cb0 should remain rejected).
- Cell count at f22 should reach 14.
- Confirmed by: full test run.

### PR-E (code removal)

- No regression from PR-D run.
- Reduced LOC, cleaner control flow.

### PR-F (sweep)

- 14 cells at f22, 0 false positives, all 8 ground-truth splits detected at
  the right frames (within ±1 frame).

---

## 9. Risks and Rollback

### 9.1 Risks

- **Snapshot-based placement may generate worse initial daughter positions**
  than the current PCA-based placement for cells whose snapshot is stale.
  Mitigation: §7.3 centroid recenter.
- **The hard `prevElong` cutoff at 1.20 may be too aggressive** and miss
  splits on cells whose elongation never exceeds this. Mitigation: tune
  threshold from validation runs.
- **Removing pre-burn-in guards may reactivate old false-positive
  pathologies.** Mitigation: shadow-mode logging in PR-A through PR-D, only
  delete in PR-E after confirming guards never fire on the new path.
- **End-of-frame Pillar B cost** may be higher than estimated if neighbor
  masking is implemented inefficiently. Mitigation: profile and optimize the
  hot path.

### 9.2 Rollback per PR

- **PR-A:** revert. Snapshots are not used for decisions, so reverting has
  zero behavioral effect.
- **PR-B:** revert the cutoff and re-add the live gate. Exact restoration of
  current behavior.
- **PR-C:** call the legacy `trySplitCell` instead of the new method via a
  config flag (`use_snapshot_placement: false`).
- **PR-D:** disable post-hoc gates with config flags. Pillar B Role B becomes
  shadow-mode logging only.
- **PR-E:** the deletions are replaceable from git history. Each deleted
  block can be restored individually.

---

## 10. Summary

| | Current | New |
|---|---|---|
| Trigger signal | Live `strict_effective` per attempt | Cached `snapshot.strictEffectiveElongation` (peak observed last frame) |
| Daughter placement | `getSplitCells` PCA on current frame | `snapshot.brightPeaks` or `snapshot.principalAxis` from last frame |
| Pre-burn-in guards | Multiple heuristic guards | Minimal — trust the snapshot |
| Burn-in | Same | Same (still refines daughters to current frame) |
| Post-burn-in guards | Cost + fake-guards | Cost + biological sanity (5 new gates + existing fake-guards) |
| Pillar B role | Pre-filter on parent ROI before split | Both: end-of-frame snapshot peaks AND post-hoc daughter validator |
| Neighbor leakage | Sphere distance test (approximate) | Pixel masking (exact) |
| Multi-frame retry | Implicit | Explicit (soft reject does not blacklist) |
| Lines of code added | n/a | ~620 |
| Lines of code deleted | n/a | ~150 |
| Net | n/a | ~470 LOC |

The new architecture is fundamentally simpler at the *decision* level
(trust the snapshot), while pushing complexity into the snapshot capture
and validation. The total LOC increase is modest given the scope, and
several existing layers of heuristic guards become deletable once the
snapshot path is trusted.

---

## 11. Action items before implementation

1. **Review this plan** and confirm the architectural inversion is the
   right call. Specific points for discussion:
   - Is the hard `MIN_TRIGGER_ELONGATION = 1.20` threshold acceptable, or
     should it be lower (1.15 / 1.10)?
   - Soft-reject vs hard-reject semantics — agree on which post-burn-in
     gates are soft and which are hard.
   - Snapshot timing — running max every 50 iters, or end-of-frame only?
   - Neighbor masking — at `1.0×` neighbor ellipsoid scale, or slightly
     larger to account for halo bleed?
2. **PR-A first** — small, no behavioral change, infrastructure only.
3. **PR-B + C in sequence** — these together implement the architectural
   inversion. Ship as separate PRs to keep diffs reviewable.
4. **PR-D** is the largest single PR. Consider splitting Role A and Role B
   if size becomes unmanageable.
5. **PR-E and PR-F** are cleanup and validation.

Once approved, start with PR-A.
