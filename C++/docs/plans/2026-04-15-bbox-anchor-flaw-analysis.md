# Bbox Cost — Fundamental Anchoring Flaw Analysis

**Date:** 2026-04-15 (late night)
**Status:** DESIGN DOC — no code changes yet, pending user decision
**Context:** Third/fourth round of debugging bbox-mode issues. User asked why bbox mode keeps breaking when full-image mode worked. Goal of this doc: stop and understand the problem before implementing another patch.

---

## TL;DR

Bbox cost is a **self-centered** evaluation — the bounding box follows the cell as it moves. This means when a cell drifts to a new position, the cost evaluation no longer includes the voxels the cell LEFT BEHIND. Under full-image L2, those left-behind voxels were always counted (creating an "undershoot" penalty that anchored cells to their real-cell locations). Under bbox, that anchor is missing. Cells can drift freely toward any local image-cost basin, including onto other cells' territories. Overlap penalty alone cannot fix this because the root issue isn't cell-cell interaction — it's the MISSING PER-CELL POSITION ANCHOR.

---

## 1. Observed symptoms (what's broken)

Across runs 002144, 021558, 030113 — all bbox-enabled — we've seen repeated patterns that DIDN'T occur in full-image runs:

### 1a. Daughters collapsing during burn-in (e9077 f3 pattern)

- Seeds start 50 px apart
- During burn-in each daughter independently wanders toward shared bright peak
- Final distance 1–4 px apart
- Bridge rejects (`worstValleyRatio > 0.95`)

Addressed partially by raising `overlap_penalty_weight` 500 → 30000. But overlap saturates at `weight` when fully overlapping → strong image-cost-gain collapses still win.

### 1b. Phantom "relocate" — daughters collapsed at a different position from parent (e3d03 phantom in 002144)

- Parent at snap position
- Both daughters wander ~16 px away from parent
- Daughters end up 1–2 px apart at a new location that's a better image fit
- Cost signal strongly negative (−72K); morphology clearly wrong (bio gate catches it)

Not fixable via overlap alone because both daughters move TOGETHER to a position that's image-cost optimal.

### 1c. Cell-cell overlap between unrelated cells (screenshot f4, 030113)

- e3d03 starts at snap position y=214 (its real-cell location)
- 12345..341 just spawned at (143, 177, 104) from f3 split
- During f4 unified loop:
  - e3d03 wanders y=214 → y=189
  - 12345..341 wanders to y=190
- Both converge on same (125, 190, 94) region, 2.4 px apart
- Real image has only one cell there

This is the case that triggered this analysis. It's NOT a split-time issue, it's a regular-perturbation issue. Overlap penalty at weight=30000 didn't prevent it because:

- Each small approach step during 500 iterations added modest overlap penalty
- Each step also gained image cost (cells found better local fits)
- Net slightly negative → accept → drift accumulates
- Final state: local minimum. Both cells locally optimal but globally wrong.

### 1d. Bright cell under-fit (e3d03 shape fit)

- Less clearly caused by bbox — primarily caused by `pcaShapeWeightExponent=1.3`
- But bbox makes it worse: with Voronoi exclusion, other cells' pixel contamination is gone but halo pixels WITHIN self's Voronoi territory still get 10× less weight than core → tight core fit

Independent issue, not addressed by this design.

### Common thread

**Cells drift from their real-cell positions to other attractive regions.** This doesn't happen under full-image cost. Why?

---

## 2. Root cause — bbox loses the "leave-behind cost"

### 2a. Full-image cost behavior

Cost is summed over EVERY voxel in the 225-slice image. When cell A at position P1 perturbs toward cell B at position P2:

- At P1: A was covering real cell material. After A moves, A's synth at P1 is empty (background). Real at P1 is still bright.
  - Contribution to cost: `(synth_P1 - real_P1)² = (0 - bright)² = bright²`  (undershoot, weight 1)
  - This cost is COUNTED because P1 is a voxel in the image, always summed.
- At P2: A arrives. B is also there. Synth may double-count or saturate (depending on render). Cost either goes up (overshoot) or stays same.
- **Net**: moving A away from its real-cell location ALWAYS creates a large new cost (undershoot at P1). Cell is ANCHORED to P1 by this cost.

### 2b. Bbox cost behavior

Cost is summed over a BOX CENTERED ON the cell's CURRENT position, with Voronoi neighbor exclusion.

When A at P1 perturbs to new position P1' (slightly closer to P2):

- Before move: bbox centered at P1. Cost = sum over P1-area voxels.
- After move: bbox centered at P1'. Cost = sum over P1'-area voxels.
- The voxels at P1 (now empty of A's synth, but real still bright) are NOT IN THE NEW BBOX if A moved more than `bbox_margin × maxR` away from P1.
- Per-iteration move is small (sigma ~5 px), so pre/post bboxes overlap heavily. The abandonment cost is partially counted for small moves but LOST as the cell drifts far.

**Over 500 iterations, cell drifts 20–25 px. Bbox drifts with it. Original P1 is no longer in the bbox.** Abandonment cost invisible.

Current code DOES compute union of pre/post bboxes in perturbCell. But that union is only of TWO consecutive positions (before/after a single perturbation), not the ORIGINAL snap position. The snap position is lost after the first few iterations.

### 2c. Verification from observed data (e3d03 at f4 in run 030113)

- Snap pos at f4 start: (124.73, 214.05, 87.97)
- After 500 iters: (124.71, 188.97, 95.55) — moved 25 px
- At y=214 (original): real cell is there? In the good run, yes — e3d03 in good 0413 run moved y=216 → y=232 (downward). So real image has e3d03 material at y=216-232, not y=189.
- But in new run, e3d03 drifted UP to y=189 where the cell apparently found a better cost fit
- The abandoned y=214 area had real cell material uncovered. Under full-image cost, that would have cost 1000+ in undershoot. Under bbox, invisible.

### 2d. Why both cells converge to same spot

Each cell's perturbation is cost-driven toward its own bbox local minimum. If two cells' bboxes overlap on a shared bright region, BOTH cells find the shared region cost-favorable. Neither cell "knows" the other is heading there. By the time they're close, the overlap penalty fires but they're already near a joint local minimum where no unilateral move improves.

---

## 3. Why this worked under full-image

Full-image cost is GLOBAL: it sees the entire frame always. If cell A leaves P1, the full-image cost goes UP by `real_bright(P1)²` at each leaving step. If cell A approaches cell B's P2, double coverage creates overshoot → cost goes UP. Both directions penalized.

Cells are effectively anchored to their real-cell positions because ANY deviation creates new cost somewhere. The perturbation is a balance between:

- Improving local fit at current position (pulls cell toward real signal)
- Avoiding undershoot at abandoned position (pulls cell BACK to real signal it was covering)
- Avoiding overshoot at occupied positions (pulls cell AWAY from other cells' territories)

All three constraints are present simultaneously. Cells find equilibrium at their real-cell positions.

Under bbox: only the first constraint (improve local fit) is present. The other two are geometrically clipped out of the evaluation.

---

## 4. Option analysis

### Option A — SNAP-ANCHORED bbox

Compute each cell's bbox ONCE at frame start, centered on `snap.position`. Use this FIXED bbox for all of that cell's perturbations during the frame.

**Code sketch:**

```cpp
// In CellUniverse::optimize at frame start, after shape fit:
for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
    const auto &snapIt = previousSnapshots.find(frame.cells[ci].getName());
    if (snapIt != previousSnapshots.end() && snapIt->second.valid) {
        frame.setCellBbox(ci, computeBboxAt(snapIt->second.position, snapIt->second.maxRadius, marginScale));
    } else {
        // Frame 1 or new daughter — use current live position
        frame.setCellBbox(ci, computeCellBbox(ci, marginScale));
    }
}

// perturbCell:
BoundingBox3D bbox = _cellBboxes[index];   // fixed for the frame
oldImageCost = calculateBboxCost(bbox, _synthFrame, mask);
// apply perturbation
newImageCost = calculateBboxCost(bbox, newSynthFrame, mask);
// Voxels outside bbox are not counted either way — apples-to-apples
```

**Pros:**
- Restores position anchoring: cell can move within bbox, but leaving bbox loses cost-relevance (synth outside bbox doesn't count, so moving out actually HURTS cost).
- Preserves most bbox perf benefit (still much smaller than full-image).
- Matches existing "snap is authoritative" design philosophy.
- Clean semantic: one cell, one bbox per frame. No drift tracking.

**Cons:**
- If cell genuinely moved > `marginScale × maxR / 2` = ~3 × maxR / 2 = 1.5 × maxR between frames (e.g., 30 px for a 20-px-radius cell), the real cell may be near the bbox edge. Cell can still reach it, but with less margin.
- Voronoi mask is also fixed to snap positions. If a neighbor genuinely moved, its claim point is stale. Small accuracy loss on cell-boundary voxels.
- ~50 LOC of new code (per-frame bbox storage, snap-to-bbox computation, wire into perturbCell).

**Risk:** moderate. Changes cost semantics. May improve or hurt split accuracy — unclear until tested.

### Option B — Explicit "leave-behind" cost

Keep bbox following the cell (current behavior), but ADD a cost term for the snap position if the cell has moved away from it.

**Code sketch:**

```cpp
// In perturbCell:
BoundingBox3D localBbox = computeCellBbox(index, marginScale);  // follows cell
double localCost = calculateBboxCost(localBbox, synth, mask);

// If cell has valid snap AND has moved away from it:
double leaveCost = 0.0;
if (snapValid && cv::norm(cell.pos - snap.pos) > marginScale * snap.maxR) {
    // Cell has moved outside its "natural" snap region
    BoundingBox3D snapBbox = computeBboxAt(snap.pos, snap.maxR, marginScale);
    leaveCost = calculateBboxCost(snapBbox, synth, mask);  // un-masked (cell is gone from here)
}

totalCost = localCost + leaveCost;
```

**Pros:**
- Least invasive to existing bbox code.
- Still benefits from bbox (most cost eval is local).
- Explicitly encodes the anchoring principle.

**Cons:**
- Doubles cost computation for cells that have drifted (~2× perf penalty on those).
- More complex: need to decide WHEN leave-behind cost applies (drift threshold), HOW much to weight it.
- Still has the cell-cell overlap issue because both cells would happily drift if each gains more than it loses at leave-behind.
- The "outside bbox" threshold is a tuning parameter without clear scientific basis.

**Risk:** medium-high. Introduces new tuning parameter, has edge cases.

### Option C — Revert to full-image cost

Set `use_bbox_cost: false` and revert all cost-related config values to full-image tuning:

```yaml
use_bbox_cost: false
asymmetric_cost_weight: 8.0
overlap_penalty_weight: 500.0
split_cost: 15
```

**Pros:**
- Known-working behavior from all pre-bbox runs.
- Zero code changes.
- Zero risk.

**Cons:**
- Loses perf benefit (~6–8× slower per cost eval).
- Loses the cross-cell isolation benefit.
- "Throws away" all bbox work from this session + last week.

**Risk:** zero. Known regression to prior state.

### Option D — Hybrid: full-image cost with Voronoi exclusion

Keep the Voronoi exclusion idea but apply it to the FULL image, not a bbox. Cost = asymmetric L2 over all voxels where voxel's nearest claim is the cell being perturbed.

This captures:
- Per-cell isolation (Voronoi gives each cell its own territory) ✓
- Anchoring (full image means snap-position voxels still counted) ✓

**Pros:**
- Combines benefits of both bbox AND full-image.
- Each cell's cost is over its own full territory (no bbox cutoff), so anchoring works.
- Cross-cell isolation works (Voronoi excludes neighbor pixels).

**Cons:**
- Building a FULL-IMAGE Voronoi mask per cell is expensive — ~36M voxels × ~10 neighbors = 360M distance comparisons. Much slower than bbox's 5M × 10 = 50M.
- Still ~10× slower than current full-image (which doesn't do Voronoi per cell).
- Changes cost meaning AGAIN. More rounds of tuning.

**Risk:** high. Perf prohibitive. Not obvious it works better than Option A.

---

## 5. Recommendation

**Option A (snap-anchored bbox)** is the principled fix that preserves bbox benefits while restoring position anchoring.

- It directly addresses the root cause (bbox drift losing anchor).
- It's consistent with the existing "snap is authoritative" design philosophy.
- The perf benefit of bbox is preserved (cost computed on a fixed bbox per cell, still much smaller than full-image).
- Voronoi exclusion still works (fixed on snap positions).
- Implementation is ~50 LOC.

**If we want to be fully safe, Option C is zero-risk** — revert to full-image cost entirely. We lose ~6–8× perf but get known-working behavior. All tuning (asymK, overlap, split_cost) reverts to pre-bbox values.

Do NOT recommend Option B (explicit leave-behind cost) — doubles cost evaluation for drifted cells, introduces new tuning parameter, still has cross-cell overlap issue.

Do NOT recommend Option D (Voronoi-full-image) — perf prohibitive.

## 6. What to do next

User decides among:

1. **Implement Option A** (~50 LOC, medium risk, preserves perf)
2. **Revert to Option C** (zero code, zero risk, loses perf)
3. **Do more investigation** before committing — e.g., run a minimal test where we ARTIFICIALLY anchor a cell to its snap and see if the wandering stops.

If (1) chosen: implementation plan:

- Add `std::vector<BoundingBox3D> _cellBboxes` member to `Frame`
- Add `setCellBbox(size_t idx, BoundingBox3D)` setter
- At frame start in `CellUniverse::optimize`, after shape fit, compute per-cell snap-bboxes and install via `setCellBbox`
- In `perturbCell`, read the stored snap-bbox instead of computing `computeCellBbox(index, marginScale)` each time
- For split attempts (`trySplitCellPhased`), the union bbox logic is unchanged — still computes a fresh union from candidate seeds
- For frame 1 or newly-born cells with no snap: fall back to live-position bbox

### Validation criteria

After Option A is implemented, the following should be true:

1. e3d03 f4 stays near its snap position y=214 (not wandering to y=189)
2. 12345..341 f4 stays near its birth position (143, 177, 104) (not drifting to 125, 191, 94)
3. Cells don't overlap each other unless they genuinely need to (real image has two cells in same area)
4. Split accept rate vs GT matches or improves compared to current runs

## 7. Lessons learned

- **Bbox cost is not a drop-in replacement for full-image cost.** It changes the semantic of what "cost" anchors to.
- The original plan for bbox (2026-04-14-universal-bbox-cost.md) didn't fully analyze the anchoring implications. We discovered them empirically through ~6 failed runs.
- When adopting a new cost formulation, explicitly audit what invariants the old cost provided (e.g., position anchoring via global summation) that the new cost may break.
- "Snap is authoritative" was established as a design invariant but bbox implementation broke it (cells drift from snap freely). Need to re-establish the invariant at the COST FUNCTION level.
