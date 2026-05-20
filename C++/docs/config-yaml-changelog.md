# Config YAML Changelog Notes

Historical tuning comments extracted from `config/config.yaml` to keep the active configuration readable.

## Original Lines 66-76

```yaml
  # 2026-04-19 (size-fix attempt): tried 2.236 → 2.5 to compensate for
  # compressed c-axis variance after yp preprocessing. Worked per-axis but
  # inflated baseline image costs (225k → 368k at f8 1f89ab), making real
  # split cost-fraction improvements too small to clear gates. REVERTED to
  # 2.236; trying pooling instead (preprocessing-side fix, not PCA-side).
  # 2026-04-19 (final size-fix): 2.236 → 2.5 (re-applied). Pooling run at
  # 2.236 showed persistent 20-40% undershoot on cell extent — synth
  # ellipsoids don't cover the visible bright region. With cost_fraction
  # now at 0.03, the cost gate is no longer load-bearing, so bigger baseline
  # image costs from larger cells don't block splits. Bumping radiusScale
  # lets PCA fit reach past the core into the bright halo gradient.
```

## Original Lines 80-122

```yaml
  # Pixel-collection mask half-extent multiplier: the PCA fit gathers bright
  # pixels inside (maskScale * snap radii). Mask is FIXED across iterations
  # (gotcha B — snap-mask invariant prevents collapse onto one daughter in
  # dividing cells), so this is the SINGLE mask the fit sees.
  # 2026-04-15: raised 1.3 → 1.6 after validating against the 0413 "good
  # shape" run. 0413 used LIVE iteratively-updated mask and converged at an
  # effective mask = ~1.3 × converged radii (i.e. much larger than 1.3 × snap
  # because the fit grew each iter). Current design blocks mask-growth on
  # purpose, so a bigger FIXED multiplier is needed to recover the same
  # halo-inclusive pixel set on iteration 0. Empirically 0413's bright cells
  # reached pixel counts 2.4× the iter-0 count; scaling the mask linearly to
  # capture that volume up-front gives ~1.6. Watch for halo-bloat (aRadius
  # hitting 65 ceiling) or Voronoi bleed signatures; if either appears, drop
  # back to 1.5 or tighten Voronoi exclusion.
  # With percentile-based radii, the mask is a pixel-gathering WINDOW
  # whose exact size matters much less — the 95th percentile of
  # |projection| is robust to extra pixels at the mask boundary (they
  # fall in the top 5% that the percentile ignores). Birth-based mask
  # prevents feedback loops. 1.8 is a moderate value: generous enough
  # to see the full cell + some halo, small enough to not pull in
  # vast amounts of empty space. The tuning sensitivity is gone because
  # radius computation no longer depends on variance (which WAS
  # sensitive to every pixel's distance from centroid).
  # Fixed mask scale: mask half-extent = maskScale × birth radii.
  # 1.6 is the best22 validated value (8/8 GT splits).
  # Stays constant across PCA iterations AND across frames — no
  # feedback loops (no adaptive mask, no collapse onto one daughter).
  # 2026-04-19 (later): reverted 1.3 → 1.6. The audit shrunk cell radii too
  # aggressively when combined with the other two PCA-shape tightenings —
  # daughter c-axis dropped from 15 px to 7 px by f20. Cost/bio gate audit
  # blocks FPs directly, so this defensive shrinkage is redundant + harmful.
  # 2026-04-19 (size-fix): 1.6 → 1.8. Compared 162148 vs 095555 PCA Shape
  # logs — yp preprocessing gives ~40% fewer mask pixels but ~2× meanW
  # (brighter cores, compressed halo). c-axis collapsed 20-40% because
  # 1.6×birthR mask no longer reaches the now-sharpened halo tail. Larger
  # mask lets halo pixels vote, restoring minor-axis extent.
  # 2026-04-19 (post-merge daughter growth): 1.8 → 2.2. Run 193101 showed
  # late-frame daughters (e9077 f20 grand-daughters) stay undersized —
  # birth mask is frozen at split-time radii, so as real cells grow post-
  # birth, PCA hits the mask ceiling and reference stops growing. 22% extra
  # slack lets daughters follow their real growth through the ~2-5 frames
  # before the next mitosis. 2.2 is a moderate bump — too much risks
  # neighbor-cell contamination in dense f20+ clusters.
```

## Original Lines 129-157

```yaml
  # Per-pixel intensity exponent in the PCA weighted moments.
  # Higher  → stronger core emphasis, smaller fitted radii (tight fit).
  # Lower   → halo pixels contribute more, larger fitted radii (loose fit).
  #   1.0 = linear (pre-2026-04-14 behavior; bloat)
  #   1.5 = balanced (current default)
  #   2.0 = quadratic (tight, core-only)
  #   3.0 = very tight core
  # Tune down (e.g. 1.3) if cells look too small after Option A; tune up
  # (e.g. 1.8) if halo is still bloating the fit.
  # 2026-04-15 (morning): tried 1.3 → 1.1 based on apparent "bright cells
  # shrinking" observation. Caused CATASTROPHIC halo bloat — e9077 at f5
  # fit aRadius=75 (clamped to max 65), fit never converged. REVERTED to
  # 1.3. The "shrinking" observation actually reflected correct pre-split
  # elongation (cells thin along cleavage axis before dividing, e.g.
  # 12345 f16→f19 progression before its f20 split). Not a PCA bug.
  # exp=1.1 is in the linear-weighting regime where halo pixels
  # contaminate PCA variance; Voronoi + snap-mask + bbox alone are NOT
  # enough to control halo — we need weighted core emphasis too.
  # Lesson: weight exponent 1.3 is near the edge of a sharp cliff.
  # Don't go below without very strong evidence.
  # 2026-04-18: lowered 1.3 → 1.0 (linear weighting). Weight 1.3
  # concentrates mass on the bright core, giving a radius ≈ radius-of-
  # gyration of the CORE alone. The halo (which makes cells look round
  # visually) was suppressed. Linear weight lets halo pixels vote
  # proportional to brightness, producing fuller radii that match the
  # visible cell extent.
  # 2026-04-19 (later): reverted 1.3 → 1.0. See pcaShapeMaskScale comment.
  # Linear weighting lets halo contribute fully — needed for proper cell
  # extent rendering. Cost/bio gates handle FP control independently.
```

## Original Lines 160-174

```yaml
  # Per-cell adaptive exponent. When true, each cell picks its own exponent
  # based on how core-dominated its Voronoi-filtered bright-pixel cloud is.
  # Bright core-dominated cells ramp toward `pcaShapeWeightExponentBright`
  # (looser fit, halo gets a fairer vote). Dim/uniform cells stay at
  # `pcaShapeWeightExponent` above (no behavior change). Metric:
  #   pCore = fraction(raw pixel weight > pcaShapeCoreBrightnessThreshold)
  # Ramp:
  #   pCore <= pcaShapeCoreFractionLow   → expDim
  #   pCore >= pcaShapeCoreFractionHigh  → expBright
  #   between                            → linear interpolation
  # FLOOR on expBright is 1.15 by convention — going below ~1.2 enters the
  # empirical halo-dominated cliff (see 2026-04-15 021558 disaster). The
  # adaptive path only loosens bright cells; dim cells were never the
  # problem. Disabled by default; enable when you see bright cells fitting
  # smaller than they visibly are.
```

## Original Lines 177-186

```yaml
  # Core brightness threshold for pCore computation.
  # 2026-04-15 (late morning): lowered 0.6 → 0.15 after discovering
  # pCore=0 for every cell in run 091510 — the 0.6 threshold was
  # calibrated for post-sigmoid [0,1] pixel values, but the new
  # iterative-contrast preprocessing produces pixel weights in the
  # 0.05-0.20 range. Nothing ever exceeded 0.6, so adaptive exponent
  # and adaptive radius inflation both collapsed to "uniform"
  # defaults. 0.15 sits between typical halo (~0.05-0.10) and core
  # (~0.15-0.25), so peaked cells get the adaptive treatment and
  # uniform cells stay put.
```

## Original Lines 188-197

```yaml
  # pCore is now RELATIVE (fraction of pixels > 1.5 × cell's own mean
  # weight), so the ramp thresholds are calibrated for that metric:
  #   uniform cell: pCore ≈ 0.05-0.15 (few pixels above 1.5×mean)
  #   peaked cell:  pCore ≈ 0.20-0.40 (bright core above 1.5×mean)
  # fracLow=0.15 means cells with < 15% above relative threshold get
  # no inflation; fracHigh=0.35 means cells with > 35% get full inflation.
  # 2026-04-16: raised 0.15 → 0.25. At 0.15, ALL cells (pCore 0.16-0.23)
  # got partial inflation. Only 8cbdf (0.23) and 1f2ed (0.23) are legitimately
  # peaked. At 0.25, uniform cells (1f89a pCore=0.16, e9077 pCore=0.19)
  # stay at base exponent — no unneeded 3-6% radius inflation.
```

## Original Lines 209-221

```yaml
  # Percentile for radius computation. The Nth percentile of |projection
  # onto each PCA axis| becomes the radius for that axis.
  # 0.95 = original (captures 95% of bright pixel extent; oversized by
  #         20-73% vs variance-based radii in the best22 reference run
  #         because halo/boundary pixels inflate the tail).
  # 0.90 = tighter (cuts top 10%; ~5% underestimate for uniform, but
  #         fit growth cap tracks up over subsequent frames).
  # 0.85 = core-only (aggressive; may need higher fitGrowthCap).
  # 2026-04-19 (later): reverted 0.85 → 0.90. See pcaShapeMaskScale comment.
  # Daughter c-axis was halving each generation; needed proper extent.
  # NOTE: this field is parsed but UNUSED in the main fit path since v7
  # Change 50 (variance-based radii reinstated). Active lever is
  # pcaShapeRadiusScale above. Kept at 0.90 for consistency only.
```

## Original Lines 264-268

```yaml
  # 2026-04-19: 500 → 150. With yp's stronger preprocessing + signal-guided
  # perturbation, fewer iterations are needed to converge. Halves run time.
  # NOTE: iterations_per_cell is the FALLBACK when both iter buckets below
  # are < 0. With signal-guided design (yp ffc1917), per-frame iter count
  # is signal_guided_iterations_per_cell or random_iterations_per_cell.
```

## Original Lines 270-273

```yaml
  # Signal-guided perturbation iter buckets (yp ffc1917). Each frame uses
  # ONE of these depending on whether signal-guided was activated for that
  # frame (centers >= prevCellCount). Set to -1 to fall back to
  # iterations_per_cell. yp default 100 each.
```

## Original Lines 279-285

```yaml
  # Preprocessing pipeline tuning. Hybrid strategy (2026-04-19, post-FP at f4):
  # - Iterative loop fields = OUR snap-only values (run 095555). yp's values
  #   produced larger cost magnitudes that snuck a false split past the cost
  #   gate at f4 e3d03 (45-frame run 130101). Our values keep cost magnitudes
  #   in the range our gates were calibrated for.
  # - Post-process intensity stretch fields below = YP values (per user
  #   direction): yp pushes contrast harder, which is the win we want.
```

## Original Lines 363-364

```yaml
  # Number of radius trials to run in parallel during each frame's contrast
  # preprocessing radius search.
```

## Original Lines 385-388

```yaml
  # When true, skip the upfront "preprocess all frames into memory" pass and
  # run prepareFrame(frame) immediately before optimize(frame). This reduces
  # peak memory by keeping the pipeline frame-by-frame. Ignored when
  # quit_after_preprocessing is true, because that mode exits before analysis.
```

## Original Lines 410-414

```yaml
  # Checkpoint resume: moved to run_celluniverse.sh INI preset on 2026-04-22.
  # Set `resume_from` and `resume_source_dir` in user_input_configurations.ini
  # under the preset. Override via CLI positional args 7 and 8 also possible.
  # This YAML no longer controls resume — the INI preset is the source of
  # truth so resume decisions stay alongside frame-range and I/O config.
```

## Original Lines 430-431

```yaml
  # 2026-04-19 audit: 0.4 → 0.2. Yp preprocessing brightens halos, so top 40%
  # of "background candidates" may include halo. Top 20% targets cleaner bg.
```

## Original Lines 434-480

```yaml
  # Asymmetric L2 cost weight (Fix E). Per-voxel squared error is multiplied
  # by this factor when synth > real (cell covers dark image region).
  # 1.0 = symmetric L2 (disabled). Makes "parent covering dark valley
  # between two daughters" reliably cost more than "two daughters covering
  # bright blobs."
  #
  # Tuning history:
  #   3.0   — original Fix E, full-image cost mode, total cost ~8K
  #   6.0   — 2026-04-14 (morning): raised for stronger split signal
  #   8.0   — 2026-04-14 (evening): raised again, full-image cost
  #  64.0   — 2026-04-15 (evening): scaled 8x to match bbox/full ratio
  #
  # Why 64 for bbox: bbox cost magnitudes are ~8x full-image (e9077 f3:
  # full-image=8.7K, bbox=69.7K). The "parent covers dark valley"
  # per-voxel overshoot penalty is the SAME voxel set in absolute terms
  # (same diff² × k), but as a FRACTION of total cost under bbox it's
  # proportionally weaker (valley penalty in full-image was ~3% of
  # total; under bbox at k=8 it drops to ~0.4%). Scaling k by the full
  # 8x ratio (k=64) restores the fractional impact of the asymmetric
  # signal to what it was in full-image mode.
  #
  # Side effect: ~8x more pressure on cell-boundary voxels where
  # synth>real due to ellipsoid rendering approximation. Not a real
  # overshoot, but gets amplified. Monitor: if real splits' costDiffs
  # become noisy or unstable, step down to 32 or 24.
  #
  #  32.0   — 2026-04-15 (late night): dropped from 64 after run 041912
  #           showed e9077 f3 split rejected by bridge_flat. Collapsed
  #           candidates kept winning the cost race. Diagnosis via
  #           Ellipsoid::draw (overwrite rendering, not additive):
  #           stacked daughters render as one ellipsoid's worth of
  #           synth → one ellipsoid's boundary overshoot. Separated
  #           daughters render as TWO ellipsoids → two boundaries of
  #           overshoot. asymK amplifies overshoot 32×, so the extra
  #           boundary cost dominates the benefit of covering both
  #           lobes. At k=32 collapsed still won by ~20-30K.
  #
  #   8.0    — 2026-04-15 (very late): matched to 0413 good run's value.
  #           Back-of-envelope crossover analysis suggested separated
  #           beats collapsed when k is below ~6. k=8 is just above the
  #           crossover but closes the 32K boundary-overshoot gap by 4×.
  #           split_cost rescaled proportionally (1500 → 375) so the
  #           accept threshold stays meaningful at the new magnitudes.
  #           If e3d03 / 8cbdf regress to false positives (split signal
  #           too weak against dim valleys), next step is a structural
  #           fix: compute overshoot once per union footprint rather
  #           than per-cell (fixes the double-counting at the root).
```

## Original Lines 492-499

```yaml
  # Signal-guided perturbation (yp ffc1917). Detects bright clusters in the
  # real image and uses them as teleport targets in perturbCell — helps cells
  # stuck on local optima jump to the right bright signal. Sigma scales with
  # cluster brightness via signal_guided_min_sigma_scale +
  # signal_guided_sigma_range_multiplier.
  # 2026-04-19: temporarily disabled to isolate PCA shape revert effect.
  # When false, useSignalGuidanceThisFrame is always false → all frames use
  # random_iterations_per_cell + pure-random perturbation.
```

## Original Lines 504-505

```yaml
  # Minimum brightness (above background) for a box to be considered. Filters
  # out background noise. yp default 0.4.
```

## Original Lines 511-512

```yaml
  # Sigma range multiplier on top of cellConfig.x/y/z.sigma. yp tuned 10.0
  # which gives wide-jump teleports.
```

## Original Lines 534-564

```yaml
  # Split acceptance threshold: `costDiff < -split_cost` → accept.
  # Tuning history:
  #   15     — full-image cost mode. Full-image total cost ~8K; 15 was
  #            meaningful (~0.2% of total).
  #   1500   — 2026-04-15 (evening): scaled ~100× for bbox mode. Bbox
  #            total cost ~70K; real splits accept at costDiff -13916
  #            to -48192 (all well past -1500). Phantom splits under
  #            overlap_penalty_weight=30000 have costDiff floor around
  #            -40K (overlap cap leaves a large phantom signal). 1500
  #            threshold doesn't block -40K phantoms but catches
  #            marginal candidates near zero.
  #
  # Note: cost gate is still only a secondary filter; bio gates
  # (bridge + volume + buried + edge_brightness) do most of the
  # phantom-rejection work. Stronger cost-only rejection of phantoms
  # would require a non-saturating overlap formula or a hard
  # daughter-daughter distance constraint, both separate work.
  #   375    — 2026-04-15 (very late): scaled proportionally with
  #            asymK 32 → 8 (1/4 of 1500). At k=8 bbox total costs
  #            drop ~4× so the threshold follows.
  #   2000   — 2026-04-16: raised from 375. Run 030254 showed 16K gap
  #            between weakest accepted (-5154) and weakest cost-reject
  #            (+11054). 2000 adds safety against marginal phantoms
  #            without risking any real splits (all accepted > 5K).
  # 2026-04-19 audit: 2000 → 7000. Yp preprocessing scales image cost
  # magnitudes 2-3× larger; raise the fixed split-cost floor proportionally.
  # 2026-04-19 (size-fix followup): 7000 → 2000. Reverted to pre-merge value.
  # Analysis of splits across runs: pooling regime real split ratios are
  # 25-30%+ (stronger signal than pre-merge); the 1f89ab f8 low-ratio case
  # is due to its skinny per-frame fit, not a cost-gate calibration issue.
  # 2000 matches pre-merge (095555) which split 1f89ab f8 at 5% ratio safely.
```

## Original Lines 566-584

```yaml
  # Proportional split cost: effective threshold = max(split_cost,
  # split_cost_fraction × baselineImageCost). Ensures split must improve
  # cost by at least this fraction of the cell's own baseline cost.
  # 0.0 = disabled (fixed threshold only).
  # 0.03 = 3% improvement required. In run 030254, accepted splits had
  # costDiff/baseline ratios of 7-35%, so 3% is safe.
  # 2026-04-19 audit: 0.03 → 0.20. The 45-frame run (130101) had a FP at
  # e3d03 f4 with diff/baseline = 30%. Real splits in hybrid runs show 100%+
  # improvements (e.g. e9077 f3: -114886 vs baseline 17332 = 660%).
  # Bumping fraction to 20% blocks the 30% FP cleanly while leaving huge
  # margin for real splits.
  # 2026-04-19 (size-fix followup): 0.20 → 0.12. Reverted back to 0.20 once
  # radiusScale went back to 2.236 (pooling handles c-axis instead).
  # 2026-04-19 (final): 0.20 → 0.03. Reverted to pre-merge value. Rationale:
  # (1) pooling regime real splits show 25-50% ratios on most cells — huge
  # margin over 0.03; (2) cells with skinny per-frame fits (e.g. 1f89ab f8
  # at 5-16%) cleared 0.03 in pre-merge baseline; (3) FPs blocked by tightened
  # bio gates (valley 0.75, edge 0.07) and re-enabled position prior (w=75)
  # — cost gate is no longer load-bearing for FP control.
```

## Original Lines 587-605

```yaml
  # Position prior (re-introduced 2026-04-18 after Phase A edge-fit
  # stabilized shape). Quadratic penalty on distance from snap beyond
  # position_prior_threshold:
  #   penalty = position_prior_weight × max(0, ||cell.pos - snap.pos|| - threshold)²
  # Addresses drift escape-hatch where snap-bbox undershoot saturates
  # once the cell fully exits its bbox. Weight 10 gives:
  #   25 px drift (5 px past threshold): penalty = 250
  #   40 px drift (20 px past):          penalty = 4000
  #   80 px drift (60 px past):          penalty = 36000 (dominant)
  # Legitimate biological motion (<20 px/frame) pays zero penalty.
  # 2026-04-18: raised 10 → 30 after run 093414 showed false-split
  # cascade (12345 missed at f3, cells drifted 60-70 px at f4, false
  # splits at f5/f6). At w=10 the prior penalty for 60 px drift
  # (16000) was not enough to override image cost gains from drifting
  # into orphan regions. At w=30 penalty scales 3× → 48000, which
  # should overwhelm typical image-cost drift incentives.
  # 2026-04-19 audit: 30 → 75. Yp's larger image cost magnitudes dilute the
  # position prior's relative weight. Scale 2.5× to keep the same relative
  # anchoring strength as the snap-only baseline.
```

## Original Lines 608-633

```yaml
  # Overlap penalty weight. Formula in computeOverlapForCell:
  #   penalty = weight × ((combinedR − dist) / combinedR)²
  # This saturates at `weight` when cells fully overlap (dist=0).
  #
  # History:
  #   500    (full-image legacy) — OK with image cost magnitudes ~7700.
  #   3000   (bbox mode first try) — still too low; daughters collapsed.
  #   30000  (2026-04-15 evening) — prevents incremental approach.
  #
  # Why 30000: run 021558 showed e9077 f3 daughters drift from 50 → 1.6
  # px apart during burn-in + refine because each approach step gained
  # ~500-2000 image cost while adding only ~10-500 overlap penalty at
  # weight=3000. Bridge then rejected (collapsed daughters ≠ real split)
  # but the cost gate had already been fooled. At weight=30000, each
  # approach step adds 2000-15000 overlap penalty — overwhelms the
  # image-cost-driven collapse gradient.
  #
  # Safety on real splits: accepted splits in run 002144 had daughter
  # pair distances 39-49 px with radii ~22-30. Overlap ratios 0.1-0.2 →
  # penalty 400-1200 at weight=30000. Negligible vs 15k-48k image gains.
  #
  # Legacy full-image path (use_bbox_cost=false): revert to 500 if that
  # mode ever re-enabled. Image cost magnitudes differ by ~10x.
  # 2026-04-19 audit: 30000 → 75000. Yp's image costs are 2-3× larger; scale
  # overlap weight proportionally so daughters can't game an overlap-light
  # local optimum that the existing weight no longer prevents.
```

## Original Lines 678-697

```yaml
  # Primary bridge threshold. The bright-pixel profile along the split axis
  # must show a drop from the brighter daughter's edge into the midpoint
  # gap by at least this factor: gap / max(edge1, edge2) < this limit.
  # Uses max-of-edges so asymmetric division (one daughter inherits more
  # cytoplasm, renders dimmer) is judged by the brighter side's valley
  # signal only; the dimmer daughter's edge ≈ gap is expected and doesn't
  # invalidate the split.
  #
  # Default 0.85: real splits typically drop to 0.3-0.7 of brighter edge;
  # phantom splits have gap >= both edges (ratio >= 1.0).
  # 2026-04-19 audit: 0.85 → 0.75. The e3d03 f4 FP slipped past the 0.85
  # gate by placing daughters far apart in y, where the genuine empty space
  # between them dropped the valley ratio. Tighter ratio (0.75) requires
  # a deeper valley to accept a split.
  # 2026-04-27: 0.75 → 0.90 for the current preprocessing/perturb setup.
  # Run output_ubuntu_1-6_20260427_094145 rejected all f3 candidates before
  # final gates: expected e9077 candidates had valley≈0.81-0.84 and expected
  # 12345 candidates had valley≈0.86-0.89, while obvious continuous blobs
  # in that frame stayed near 0.96-1.03. This restores those real f3 splits
  # without accepting the clear no-valley candidates.
```

## Original Lines 699-711

```yaml
  # Absolute minimum edge brightness (real-image units above background).
  # Rejects splits where min(edge1Bright, edge2Bright) < threshold — one
  # daughter sits in near-background empty space. Typical cell cores
  # are 0.10-0.30 above background; 0.05 is ~3× bg noise. Raise if
  # background noise is higher; lower if real daughters are very dim.
  # 2026-04-15 (night): lowered 0.05 → 0.04 after 1f2ed f11 split
  # was rejected with edge2Bright=0.047 (just 0.003 below 0.05).
  # The real daughter was visible in the image but its brightness
  # measurement happened to be slightly below threshold. 0.04 is
  # still well above background noise (~0.01-0.02).
  # 2026-04-19 audit: 0.04 → 0.07. Yp preprocessing amplifies halos, so
  # phantom-daughter edges sitting on bright halo can pass the 0.04 threshold.
  # Tightening blocks halo-on-halo phantom splits in bright-region cells.
```

## Original Lines 716-728

```yaml
  # Universal bbox cost (Universal Bbox Plan, 2026-04-14).
  # When true, perturbation and split cost evaluation both use a per-cell
  # 3D bbox with Voronoi neighbor exclusion instead of full-image L2.
  # Concentrates signal on the cell's own territory, eliminates cross-cell
  # contamination, and runs ~6× faster per cost eval.
  #
  # Bbox cost magnitudes are ~8× full-image magnitudes for the same cell.
  # Required tuning when enabled (SET AUTOMATICALLY in current config):
  #   - split_cost:             15    → 1500    (100× scale)
  #   - overlap_penalty_weight: 500   → 30000   (60× scale; saturation-bound)
  #   - asymmetric_cost_weight: 8     → 64      (8× scale, matches ratio)
  # If switching back to use_bbox_cost=false, revert all three to the
  # full-image values noted above.
```

## Original Lines 730-736

```yaml
  # 2026-04-16 (early): lowered 3.0 → 2.0 to tighten position anchor.
  # 2026-04-16 (late): raised 2.0 → 2.5. Run 030254 showed 5/6 cells
  # exceeding their bbox half-extent. At 2.0 × maxR=25, window is 50 px
  # — overlap-driven drift (155 px for e3d03) easily escapes. At 2.5,
  # window is 62.5 px — more headroom before the cell exits the bbox
  # and loses the anchor. The anchor is still effective because the
  # snap-position undershoot is always included in the bbox.
```

## Original Lines 739-752

```yaml
  # A1: after positional refine, re-fit each daughter's radii via PCA on
  # its Voronoi-claimed pixel cloud. Tightens real daughters (lower cost)
  # while phantom daughters in empty space hit the floor and gain nothing.
  # Clamps to [min * built_radii, max * built_radii] to prevent both
  # collapsed-sliver regressions (floor) and bloat from immature Voronoi
  # split absorbing neighbor/halo pixels (ceiling).
  # Set iterations=0 to disable.
  #
  # 2026-04-15 (late morning): tried 3 → 8 iterations to fix suspected
  # under-fit. Inverted after screenshot showed daughters were actually
  # OVER-fit (e9077 d2 built=(30.5,22.5,18.4) → post=(38.4,31.9,26.7),
  # +26%/+42%/+45%). Reverted to 3. The overfit is from immature
  # Voronoi split at birth absorbing too-wide pixel set; added a
  # ceiling to prevent it (split_daughter_refit_max_radius_fraction).
```

## Original Lines 754-762

```yaml
  # 2026-04-15 (afternoon): raised 0.6 → 0.85 after validating against
  # 0412 best_run e9077..a50 split miss. Voronoi sibling exclusion during
  # daughter refit systematically shrinks daughters' b/c radii (axes
  # perpendicular to split direction lose half their pixels, PCA variance
  # halves). At floor=0.6, daughters shrunk so much that volume_fraction
  # check (0.6 bio limit) failed at 0.58, blocking the legitimate split.
  # Floor=0.85 gives each daughter ≥67.5% of parent extent per axis →
  # combined volume ≥ 61% of parent, above the bio threshold. Ceiling
  # at 1.1 (below) still prevents bloat-at-birth.
```

## Original Lines 764-778

```yaml
  # Upper cap: post-refit radii clamped to this × built_radii per axis.
  # Prevents newborn daughters from bloating past built when the sibling
  # Voronoi boundary hasn't settled. Subsequent frames grow naturally
  # via bounded-growth reference in the main shape fit.
  #
  # 2026-04-16: lowered 1.1 → 1.05 to resolve structural conflict with
  # bio_max_single_daughter_volume_fraction (0.65). At ceil=1.1, the
  # max possible single-daughter volume = (1.1 × 0.7937)³ = 0.665,
  # which EXCEEDS the 0.65 gate. Any daughter refitting to ceiling on
  # all 3 axes is rejected — catching all phantoms but also real splits
  # whose daughter pixel cloud is slightly oversized (12345 f3 rejected
  # at sdv=0.651 in run 030254, causing 155 px drift cascade on e3d03).
  # At ceil=1.05: max sdv = (1.05 × 0.7937)³ = 0.579, giving 12.3%
  # margin below the 0.65 gate. All 35 accepted real splits across 3
  # reference runs had sdv ≤ 0.627 — comfortably within 1.05 ceiling.
```

## Original Lines 781-793

```yaml
  # Built-time per-axis radius scale for newly-spawned daughters.
  # daughter.radii = scale * snapshot_parent.radii.
  #
  # Empirical analysis (good 0413 run, e9077/12345/1f89ab splits):
  # actual daughter / snap-parent ratio = 0.793-0.796 in every case →
  # 0.7937 = cube root of 1/2 is the volume-preserving choice and matches
  # observed reality. Daughters then re-fit to image extent at the next
  # frame's PCA shape pass.
  #
  # Tune up (e.g. 0.85, 0.90, 1.0) if parent PCA fit is too tight and you
  # want daughters to cover more cell material on the first cost eval —
  # bio gates and per-daughter refit will trim if too big.
  # Tune down (e.g. 0.65, 0.55) for tighter initial daughters.
```
