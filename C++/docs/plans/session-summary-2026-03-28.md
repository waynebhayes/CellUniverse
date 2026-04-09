# Session Summary: 2026-03-28

## What Works

1. **Restructured directory** — clean layout: src/, config/, scripts/, data/, deprecated/, docs/
2. **Dead code removed** — Cell.hpp, Sphere, Bacilli, mathhelper, args, getParameterizedCell, performPCA, _realFrameCopy, dead config fields, dead type aliases
3. **Code quality improved** — unique_ptr for BaseConfig, thread_local RNG, static_cast, direct accessors, structured bindings, using aliases, YAML::Node by reference, firstFrameFile as parameter
4. **Unified stochastic loop** — single loop replaces Phase 1/Phase 2, P(split) driven by previous frame's PCA elongation
5. **Overlap penalty** — continuous penalty replaces all hard rejection gates in both perturbCell and trySplitCell burn-in
6. **Pre-opt shapes** — saved before optimization, passed to trySplitCell for PCA centering and daughter sizing
7. **Cross-frame PCA** — previous frame's elongation drives P(split), current frame's PCA used for split axis
8. **Split blacklist** — max 1 burn-in per cell per frame, daughters blacklisted same frame
9. **Daughter size validation** — each daughter must be 15-85% parent volume, daughters within 3x of each other
10. **Sigmoid preprocessing** — auto-calibrated center from background zone, k=75
11. **Volume cap** — reference volumes from initial CSV (not post-optimization), max 1.5x growth

## What Doesn't Work (Open Issues)

### 1. Cell inflation
**Root cause:** Hard-edged synthetic ellipsoid cannot match soft-boundary real image. L2 cost always biases toward bigger because covering boundary gradient pixels is cheaper than leaving them.

**Tried and failed:**
- Per-cell brightness (brightness-size co-optimization trap)
- L1 cost (same trap)
- Binary cost (flat gradient — no optimization possible)
- Semi-binary cost (opposite bias — cells grow to max brightness)
- Asymmetric cost (didn't prevent inflation, just changed magnitudes)
- Blur matching synth to real (double-blur caused shrinking)
- No blur (even worse inflation)
- Lowering cell_color (helps find equilibrium but doesn't fix root cause)

**Current mitigation:** Volume cap (1.5x initial volume). Works but locks in initial sizes — if initial sizes are wrong, cap preserves the error.

### 2. False splits (daughters mimicking parent)
**Root cause:** Two smaller cells are always more flexible than one large cell for covering the same area. Any cell that's not perfectly fitted benefits from splitting. The cost improvement from splitting is indistinguishable from legitimate cell division.

**Tried and failed:**
- Various split_cost thresholds (3, 20, 40, 50, 200, 300) — false and real splits have overlapping cost improvements
- Elongation threshold gate — false splits can still have high elongation
- P(split) only boost above elongation threshold — base rate (3%) still allows ~63 attempts
- Daughter size validation — validates at split time but daughters shrink after

### 3. Post-split daughter shrinkage
**Root cause:** After splitting, some daughters are placed in areas with weak real-image signal. Without brightness recovery to stabilize them, they collapse to minRadius over subsequent frames.

**What the working 0303 run had that we don't:**
- Brightness recovery (stabilizes dim cells)
- Per-cell brightness tracking (EMA updates)
- Brightness floor (prevents death spiral)
- Multiple split gates (relative gain, strong split override)

### 4. e907 missed split at frame 3
**Root cause:** e907's PCA elongation from frame 2 was below threshold (~1.02), so P(split) was base rate (0.03). At base rate, split was attempted but burn-in didn't produce enough cost improvement to pass the threshold.

## Key Insights

1. **The L2 cost function inherently rewards bigger cells in 3D** — the integral over 225 z-slices biases toward larger spheroids because peripheral slices add favorable marginal cost.

2. **No cost function variant fixes this** — L1, L2, binary, asymmetric all have the same fundamental issue. The rendering model (hard-edged analytical ellipsoid) doesn't match the real image profile (soft boundaries from blur + sigmoid).

3. **The sigmoid k=75 with blur_sigma=3 produces good contrast** — cells are near-white, background is near-black. But the transition zone is still 5-10 pixels wide, and the synthetic cell has a 1-pixel transition.

4. **Volume cap is necessary but not sufficient** — it prevents inflation but doesn't help with false splits or shrinkage.

5. **The split-fix branch's approach (brightness recovery + volume cap + multiple gates) worked because it addressed all three issues simultaneously.** The minimal-constraint approach can't replicate this with just cost function tuning.

## Recommended Next Steps

1. **Accept the volume cap as a necessary constraint** — it's biologically motivated (cells don't instantly double)

2. **Add brightness recovery back** — but simplified. Just measure brightness from real image each frame and set it. No EMA, no floor, no recovery loop. One measurement per cell per frame.

3. **Add relative split gain gate** — require split to improve cost by a percentage of the cell's individual contribution, not just an absolute threshold. This separates "cell is genuinely splitting" (large relative improvement) from "cell is just poorly fitted" (small relative improvement).

4. **Consider blurring the synthetic image** — but with sigmoid-matched parameters, not the raw blur. The synth blur should match the post-sigmoid gradient profile. This is the most principled fix for the rendering mismatch.

## Current Config (config.yaml)

```yaml
cellType: spheroid
cell: x/y sigma=5, z sigma=8, majorR/minorR sigma=2, brightness prob=0
  maxMajorRadius: 40, maxMinorRadius: 35
simulation: bg=0.0, cell=0.95, blur=3.0, sigmoid_k=75, sigmoid_center=0.445
prob: split=0.03, split_cost=300, elongation_threshold=1.5, overlap=1000, burn_in=500, max_P=0.5
```

## Files Modified This Session

- CellUniverse.cpp — sigmoid preprocessing, volume cap, cross-frame elongation, pre-opt shapes, daughter blacklist, P(split) threshold gate
- Frame.cpp — L2 cost (back from L1/binary/asymmetric), direct accessors in overlap, structured bindings, overlap penalty in burn-in
- Spheroid.cpp — static_cast, thread_local RNG, PCA threshold mean+0.5*stddev
- ConfigTypes.hpp — unique_ptr BaseConfig, sigmoid_center field, PerturbParams defaults, cell_color restored
- CellUniverse.hpp — previousElongations, referenceVolumes
- CellFactory.cpp/hpp — firstFrameFile parameter
- main.cpp — unsigned frame loop, no setenv
- types.hpp — using aliases
- All tests updated
- config.yaml — many iterations of tuning
- initial.csv — reverted to 7-column no brightness
