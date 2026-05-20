# Configuration Reference (config.yaml)

> **⚠ 2026-04-14 update:** the authoritative current config reference is in
> `C++/docs/pipeline.md` (§ Config knobs). Sections below referencing
> `shape_elongation_classify_threshold`, `size_reduction_penalty_weight`,
> `min_frames_before_split`, `radiusGradient*`, `radiusUpdateBlend`,
> `pcaShapeBlend`, `pcaShapeScanFactor`, `bio_max_midpoint_parent_fraction`,
> or `split_direction_agreement_degrees` are OUTDATED — those fields have
> been DELETED. New authoritative fields: `asymmetric_cost_weight` (under
> `simulation:`), `pcaShapeMaskScale` / `pcaShapeUpdatePosition` /
> `pcaShapeMaxPosShiftFraction` / `pcaShapeConvergeRadius` /
> `pcaShapeConvergeAngleDeg` (under `cell:`). P(split) is now a linear
> ramp from `P_split_base` at snap elong 1.0 to `P_split_max` at 2.0.

Three sections: `cell`, `simulation`, `prob`. All parsed in `ConfigTypes.hpp: BaseConfig::explodeConfig()`.

## Cell Section — Perturbation Parameters

Each perturbable parameter (x, y, z, majorRadius, minorRadius, thetaX, thetaY, thetaZ) has:
- `prob`: Legacy probability of applying random offset per iteration (0.0 - 1.0)
- `increase_prob` / `decrease_prob`: Optional sign-split probabilities (added 2026-04-06). If set, override `prob`. Use these to bias a parameter toward growth or shrinkage.
- `mu`: Mean of Gaussian offset (usually 0)
- `sigma`: Standard deviation of Gaussian offset

| Parameter | prob | sigma | Notes |
|-----------|------|-------|-------|
| x | 0.45 | 5 | Pixel units in XY plane |
| y | 0.45 | 5 | Pixel units in XY plane |
| z | 0.35 | 8 | Interpolated z-units (1 unit = 1/7 of a raw slice) |
| majorRadius | 0.2 | 2 | Controls XY size. Shrinking triggers `size_reduction_penalty_weight`. |
| minorRadius | 0.2 | 2 | Controls Z thickness. Shrinking triggers `size_reduction_penalty_weight`. |
| thetaX | 0.3 | 0.1 | Radians |
| thetaY | 0.3 | 0.1 | Radians |
| thetaZ | 0.3 | 0.1 | Radians |
| brightness | 0 | 0 | **Random perturbation disabled.** Per-cell `_brightness` is updated every frame via EMA from the real image, not by random proposals. |

**Radius bounds:**
- `minMajorRadius: 10`, `maxMajorRadius: 40`
- `minMinorRadius: 5`, `maxMinorRadius: 35`

**Brightness bounds:**
- `minBrightness: 0.15`, `maxBrightness: 0.95` (clamping still applies to the EMA-updated `_brightness`)

**Cell-level brightness controls (added 2026-04-05/06):**

| Field | Default | Purpose |
|-------|---------|---------|
| `brightnessUpdateBlend` | (tune) | EMA blend factor for per-frame brightness update: `new = old * (1 - blend) + observed * blend` |
| `brightnessMeanAmplification` | 1.0 | Multiplier applied to measured mean brightness before blending. Raises per-cell brightness above the measured mean. |
| `splitBrightestFraction` | 0.055 | Top fraction of bright in-cell pixels to keep for PCA split-point selection. Tune: 0.05 is too hotspot-sensitive per 04-06 diagnosis; try 0.1–0.15. |

## Simulation Section — Image Processing & Rendering

| Field | Default | Purpose |
|-------|---------|---------|
| `iterations_per_cell` | 350 | Iterations per cell per frame in unified loop |
| `z_scaling` | 7 | Z-interpolation factor (33 raw → 225 interpolated) |
| `blur_sigma` | 10.0 | **Safe/masked** Gaussian blur sigma before sigmoid (does not bleed zero-borders) |
| `sigmoid_k` | 75.0 (tuned ~60) | Sigmoid steepness — higher = sharper contrast |
| `sigmoid_center_percentile` | 0.45 | Percentile of brightness in calibration ROI over the full stack, used as sigmoid center |
| `calibration_x/y` | 20/20 | Calibration zone origin (cell-free ROI used for sigmoid center percentile) |
| `calibration_width/height` | 50/31 | Calibration zone size |
| `post_sigmoid_dimmest_percentile` | 0.99 | Dimmest percentile cutoff for post-sigmoid subtraction — pixels above this are kept, below reduced |
| `post_sigmoid_dimmest_transition_width` | 0.2 | Width of smooth transition band around the cutoff |
| `post_sigmoid_dimmest_transition_gradient` | 0.5 | Gradient strength of the transition |
| `adaptive_background_expand_factor` | (tune) | Expansion factor around previous-frame cells when computing adaptive synth background |
| `adaptive_background_top_fraction` | (tune) | Fraction of brightest non-cell voxels averaged for adaptive bg |

**Active sigmoid pipeline:** grayscale → normalize [0,1] → safe Gaussian blur(sigma=blur_sigma) → **percentile** of calibration ROI → sigmoid center = that percentile → apply `1/(1+exp(-k*(x-center)))` → **post-sigmoid dim-region subtraction** → z-interpolation. Cells → ~1.0, background → ~0.0.

**Removed as of 2026-04-08 (the "sigmoid pipeline invariants" cleanup):**
- `background_color` — now a runtime-mutable `Frame::_backgroundValue` (default 0.0), updated per-frame by the adaptive background path. No longer in YAML.
- `cell_color` — now a literal `1.0f` in `CellFactory::CellFactory` for the frame-1 brightness seed. After frame 1 the per-cell EMA update drives `_brightness`.
- `sigmoid_center` — defensive default replaced with literal `0.445f` in `CellUniverse::loadFrame`; always overwritten by the percentile calibration in realistic runs.
- `sigmoid_center_offset` — already dead before this cleanup.
- `calibration_z` — never read anywhere.

Do NOT re-add any of these to YAML. If you see a tuning guide referencing them, it predates the cleanup.

## Prob Section — Split & Overlap

| Field | Default | Purpose |
|-------|---------|---------|
| `split` | 0.03 | Base split probability before PCA elongation boost |
| `split_cost` | 20-30 | Minimum cost improvement to accept a split after burn-in |
| `split_elongation_threshold` | 1.5 (tuned 1.1) | PCA elongation below this = skip burn-in. 1.1 observed as too permissive per 04-06 debug log. |
| `overlap_penalty_weight` | 1000.0 | Weight for overlap penalty: `weight * overlapRatio^2` |
| `split_burn_in_iterations` | **1000** in config, 500 in code default (risk #2 — sync them) | Iterations in split burn-in loop |
| `max_split_probability` | 0.5 | Max P(split) after proportional rescale across all cells |
| `size_reduction_penalty_weight` | 2.0 | Quadratic soft penalty on perturbation proposals that shrink majorRadius or minorRadius. No corresponding growth penalty. |

**Split fake-guards (added 2026-04-06):**

| Field | Default | Purpose |
|-------|---------|---------|
| `split_fake_overlap_volume_fraction_threshold` | 0.15 | Reject post-burn-in if daughters' sphere-equivalent overlap fraction exceeds this |
| `split_fake_radius_ratio_threshold` | 1.6 | Reject if one daughter's radius is more than this fraction larger than the other |
| `split_fake_bridge_brightness_similarity_threshold` | 0.9 | Reject if the bridge cylinder between daughters is still ≥ this × the daughters' mean brightness (continuous blob detected) |
| `split_minor_axis_alignment_tolerance_degrees` | 10 | Tolerance (**DEGREES not radians**) between PCA split axis and local z axis, below which the axis is steered onto z |
| `split_minor_axis_alignment_flatness_ratio_threshold` | 0.5 | Axis alignment only enforced when `minorRadius/majorRadius ≤ this` |

**Split PCA-gating guards (Yiding's line, added 2026-04-04, extracted to config on 2026-04-07):**

| Field | Default | Purpose |
|-------|---------|---------|
| `split_pre_burn_in_min_separation_over_major` | 0.35 | Reject pre-burn-in if daughter center separation is less than this × daughterMajor |
| `split_pre_burn_in_z_axis_max_abs` | 0.92 | Reject if split axis z component exceeds this AND transverse separation is weak |
| `split_pre_burn_in_z_axis_max_separation_over_major` | 1.30 | Companion threshold to the above |
| `split_pre_burn_in_z_axis_min_drift_over_major` | 0.40 | Minimum drift required for z-dominant splits to be accepted |
| `split_post_burn_in_large_recenter_min_drift_over_major` | 0.85 | Daughters drifted more than this × majorRadius during burn-in = "large recenter" |
| `split_post_burn_in_large_recenter_max_cost_diff` | -40.0 | If large-recenter AND cost improvement weaker than this, reject (being dragged to absorb a neighbor, not splitting) |

**P(split) formula (updated 2026-04-05):**
1. For each cell, compute raw: `rawP = split + max(0, 1 - 1/previousElongation)`
2. Rescale all cells proportionally so `max(rawP) = max_split_probability`
3. Preserve relative ratios between cells

Previously this used `min(max_split_probability, ...)` per-cell; the current formula does a whole-frame rescale instead.
