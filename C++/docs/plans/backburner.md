# Backburner — Deferred Experiments and Ideas

Ideas that were explored, partially implemented, then reverted or shelved. Code changes have been undone but the learnings are preserved here for future reference.

---

## 1. Per-cell brightness as perturbable parameter

**What:** Add `_brightness` field to each Spheroid, perturbed alongside position/radius. `draw()` uses `_brightness` instead of global `cell_color`. Initial brightness measured from real image and stored in CSV.

**Status:** Code exists in Spheroid.hpp/cpp (`_brightness` member, `getPerturbedCell` perturbs it, `getCellParams` returns it). Currently disabled via config `brightness.prob: 0`. draw() reverted to use `cell_color`.

**Why shelved:** Brightness-size co-optimization trap. The optimizer simultaneously grows cells AND lowers brightness toward background. All cells converge to ~0.46 brightness regardless of true cell brightness. Bigger cells at lower brightness always reduce L2 cost. Tried under both L2 and L1 — same trap under both.

**To re-enable:** Set `brightness.prob > 0` in config.yaml, change draw() to use `_brightness`, add brightness column to CSV.

**What would make it work:** A cost function that is invariant to brightness (binary cost, normalized cross-correlation, or per-cell normalized cost).

---

## 2. L1 cost function (sum of absolute differences)

**What:** Replace `cv::NORM_L2` with `cv::NORM_L1` in `calculateCost()`.

**Why tried:** L2 overweights large errors, causing optimizer to focus on covering bright cell centers while ignoring boundary accuracy. L1 treats all errors proportionally.

**Why shelved:** Still allows brightness-size co-optimization. Cells inflate under L1 same as L2 (just with different cost magnitude ~1.6M vs ~10K). Also requires re-tuning all thresholds (split_cost, overlap_penalty_weight) for the new scale.

**Learning:** The cost metric (L1 vs L2) doesn't fix the fundamental issue — as long as brightness is a degree of freedom, the optimizer will game it.

---

## 3. Binary cost function (paper's approach)

**What:** Threshold both real and synthetic images at auto-calibrated `background_color`. Count mismatched pixels (XOR). Symmetric cost: covering bg pixel = missing cell pixel.

**Why tried:** Eliminates brightness from cost equation entirely. This is what the published paper uses with binary images.

**Why shelved:** Creates a flat cost landscape. Small perturbations (move 1px, change radius by 0.1) don't flip any binary pixels → zero cost change → all perturbations rejected. No gradient for the optimizer to follow.

**What would make it work:** Combine with a continuous component — e.g., binary cost for large-scale structure + small L2 regularization for fine tuning. Or use soft thresholding (sigmoid instead of hard threshold).

---

## 4. Semi-binary cost (threshold real only, keep synth continuous)

**What:** Threshold the real image to binary (cell=1, bg=0), keep synthetic image continuous. Cost = `sum(|binary_real - continuous_synth|)`.

**Why tried:** Smooth gradient from continuous synth while preventing brightness gaming on the real side.

**Why shelved:** Opposite bias — optimizer makes cells as bright as possible (closer to binary 1) and as big as possible (cover all 1-pixels). Rewards bright+big instead of dim+big.

---

## 5. Blur-matched synthetic rendering

**What:** Apply same Gaussian blur to synthetic image that was applied to real image preprocessing. Both images have soft boundaries → L2 is sensitive to correct cell size.

**Why tried:** Without blur matching, real image has soft cell boundaries (from preprocessing blur) while synthetic has hard edges. Too-big synthetic cell covers the blurred halo, reducing L2 cost.

**Status:** REVERTED. Code removed from Frame.cpp generateSynthFrame() and generateSynthFrameFast(). When active, it caused cells to appear tiny and dim — the double blur (real preprocessing + synthetic rendering) made cells shrink to compensate.

**Learning:** Helps somewhat but doesn't fix inflation alone. Cells still grow to max bounds.

---

## 6. Minimal preprocessing (blur_sigma=0, relaxed constraints)

**What:** No blur on real or synthetic images. maxMajorRadius=60, maxMinorRadius=45. Let sharp boundaries guide optimizer.

**Why tried:** Professor's philosophy — minimal preprocessing, let randomness work.

**Why shelved:** Cells inflate even more without blur. Sharp boundaries help in theory but the cost function's brightness-size trap dominates.

---

## 7. Dynamic split probability from current frame PCA

**What:** Compute PCA elongation at the start of each frame and use it for P(split) in the same frame.

**Why replaced:** Changed to use PREVIOUS frame's elongation for P(split) probability. Current frame's PCA is still used for the split AXIS (via getSplitCells). Rationale: Phase 1 perturbation can collapse an elongated cell into a sphere, hiding the elongation signal in the current frame. Previous frame preserves the pre-collapse elongation.

---

## Key insight from all experiments

The fundamental challenge is that **any continuous cost function where brightness is a degree of freedom allows the optimizer to game brightness-vs-size**. The published paper avoided this by using binary images. For grayscale 3D microscopy, the solution is one of:

1. Fix brightness (don't perturb it) — simple but fragile for dim cells
2. Binary/thresholded cost — needs soft thresholding to maintain gradient
3. Brightness-invariant cost (NCC, SSIM) — most principled but expensive
4. Per-cell normalized cost — normalize each cell's contribution by its volume

---

## 8. Auto-calibrate background_color from real image

**What:** Measure mean brightness in a cell-free calibration zone and set `background_color` to match (~0.42).

**Status:** REVERTED. Code still measures and logs `bgMean` but no longer overrides `config.simulation.background_color`. Uses config value (0.2) instead.

**Why shelved:** With bg=0.42 and cell_color=0.6, the synthetic cell brightness (0.6) is too bright relative to real preprocessed cells (~0.48). The optimizer shrinks cells to minimum because covering pixels at 0.6 where the real image is 0.48 costs more than leaving them as background 0.42. The original bg=0.2 with cell_color=0.6 provides the high contrast the optimizer needs.
