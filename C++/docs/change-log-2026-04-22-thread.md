# Change Log: 2026-04-22 Thread

This document records the code and behavior changes made during the 2026-04-22 working thread.

Unless another document is explicitly requested later in this same conversation, future "document the changes" requests should append to this file.

## OpenMP Adaptive Cube Pooling Fix And Refinement

### Problem

The build failed in `src/ImageHandler.cpp` with GCC/OpenMP errors like:

```text
error: not enough for loops to collapse
error: 'gz' was not declared in this scope
```

The failing code used `#pragma omp parallel for collapse(2)` above loops where statements such as:

```cpp
const int z0 = gz * cubeSize;
const int z1 = std::min(depth, z0 + cubeSize);
```

appeared between the `gz` loop and the nested `gy` loop. OpenMP `collapse(n)` requires the collapsed loops to be perfectly/directly nested, with no ordinary statements between the loop headers.

### Initial Fix

The illegal `collapse(2)` clauses were temporarily removed from the affected pragmas so GCC could compile the code. This confirmed the issue was the OpenMP loop structure, not the parallel safety of the work itself.

### Legal `collapse(2)` Trial

The per-`gz` setup variables were moved inside the `gy` loop body. That made the `gz` and `gy` loops directly nested and restored legal `collapse(2)` behavior.

### Final `collapse(3)` Trial

The setup variables were moved one level farther inward so the structure became:

```cpp
for (int gz = 0; gz < gridZ; ++gz)
{
    for (int gy = 0; gy < gridY; ++gy)
    {
        for (int gx = 0; gx < gridX; ++gx)
        {
            const int z0 = gz * cubeSize;
            const int z1 = std::min(depth, z0 + cubeSize);
            const int y0 = gy * cubeSize;
            const int y1 = std::min(rows, y0 + cubeSize);
            const int x0 = gx * cubeSize;
            const int x1 = std::min(cols, x0 + cubeSize);
            ...
        }
    }
}
```

The affected adaptive cube pooling loops now use legal `collapse(3)` forms:

- cube statistics computation
- cube reweighting and voxel fill
- isolated bright cube neighbor-check pass
- isolated bright cube zeroing pass

### Behavioral Assessment

Moving `z0/z1/y0/y1` inward does not change the mathematical logic. The same cube bounds are computed for each `(gz, gy, gx)` tuple. The values are recomputed more often, but the resulting voxel ranges are unchanged.

Parallel safety remains stable because each cube writes to its own `cubeStats`, `pooledCubeValues`, `clearCube`, or disjoint voxel region in `pooled`. The only reductions involved are integer counters, so there is no floating-point reduction nondeterminism introduced by this change.

### Verification

The project was rebuilt successfully with:

```bash
cmake --build build -j20
```

## Additional Parallelization Review

The rest of the OpenMP usage was inspected for possible similar refinements.

### Good Future Candidate

`src/Frame.cpp` contains bounding-box cost loops that currently parallelize only over `z`. These loops may benefit from `collapse(2)` over `(z, y)` because a cell bounding box can have too few z-slices to fully occupy many threads.

Recommended future shape:

```cpp
#pragma omp parallel for collapse(2) reduction(+:totalCost) schedule(static)
for (int z = bbox.zMin; z <= bbox.zMax; ++z) {
    for (int y = bbox.yMin; y <= bbox.yMax; ++y) {
        ...
        for (int x = bbox.xMin; x <= bbox.xMax; ++x) {
            ...
        }
    }
}
```

`collapse(3)` was not recommended as the first option there because keeping the inner `x` loop serial preserves contiguous row access and SIMD/vectorization opportunities.

### Larger Future Candidate

The bright-box scan in `src/ImageHandler.cpp` around the signal-guided center detection also has a 3D grid structure, but it pushes into a shared `boxes` vector. Parallelizing that safely would require thread-local vectors or a preallocated result array followed by compaction.

## Frame Export Pipeline Switches

### Goal

Add YAML-controlled export format switches so output frames can be exported as PNG slices, TIFF stacks, or both.

### Config Fields

Two booleans were added under `simulation`:

```yaml
export_frame_png: true
export_frame_tiff: false
```

Defaults preserve the prior behavior by keeping PNG enabled and TIFF disabled.

### Output Layout

PNG output is written under:

```text
output/png/real/<frame>/<slice>.png
output/png/synth/<frame>/<slice>.png
```

TIFF stack output is written under:

```text
output/tiff/real/<frame>.tif
output/tiff/synth/<frame>.tif
```

If both switches are enabled, both export branches run. If only TIFF is enabled, only stack files are written. If only PNG is enabled, only per-slice PNGs are written.

### `.tif` Extension

The TIFF stack export was changed from `.tiff` to `.tif` to match the requested extension and the project’s common TIFF input naming.

## Napari-Friendly TIFF Export

### Motivation

Napari can generally read TIFF, but OpenCV’s default `imwritemulti` output may not be ideal for all napari reader/plugin combinations. The initial TIFF writer used OpenCV defaults and ignored the write result.

### Export Improvements

The TIFF stack branch was made more reader-friendly by:

- converting pages to single-channel grayscale
- converting output to `uint8`
- validating that all slices have the same size
- writing uncompressed TIFF pages instead of relying on OpenCV’s default LZW compression
- checking `cv::imwritemulti(...)` return values
- throwing a runtime error if TIFF export fails

The exported TIFF intensity mapping is:

```text
stored_uint8 = clamp(internal_float * 255, 0, 255)
```

So exported PNG/TIFF files should be interpreted as 8-bit intensity images in the `0..255` range.

### Remaining Limitation

OpenCV does not write rich OME-TIFF/ImageJ metadata. If napari still has trouble reading the stacks, the next step would be an OME-TIFF export path using a TIFF writer that supports axis metadata.

## Initial CSV Parsing Robustness

### Problem

The original `CellFactory::createCells` reader skipped the first line as a header but parsed all data rows by fixed column position. This made it unsafe to reorder columns or insert new columns before expected fields.

### Header-Aware Parsing

The initial CSV parsing was made header-aware. It now recognizes columns by normalized header names and aliases rather than relying only on fixed positions.

Supported regular initial-cell aliases include:

```text
filePath / file / frame / frameName / image / imageFile / imagePath
cellName / name / id / cellId / label
x
y
z
aRadius / majorRadius / radiusA / radius / r
bRadius / radiusB / middleRadius / intermediateRadius
cRadius / minorRadius / radiusC / zRadius
```

Supported Napari-style aliases include:

```text
cell_type / cellType / type
z
y
x
```

### Fallback Compatibility

The parser still supports the older positional formats:

```text
filePath, cellName, x, y, z, aRadius, cRadius
filePath, cellName, x, y, z, aRadius, bRadius, cRadius
cell_type, z, y, x
```

Bad rows are skipped with warnings instead of crashing the whole program on parse errors.

### `bRadius` Compatibility

`bRadius` support was made explicit:

- if a `bRadius`/alias column exists and has a valid numeric value, it is used
- if the column is missing, empty, `None`, `null`, or `nan`, `bRadius` falls back to `aRadius`

This preserves oblate-compatible behavior while allowing triaxial initialization.

## Initial CSV File Reformatting

### `config/initial_auto.csv`

This file previously had no header, so the first data row would be consumed as the header and skipped. It also contained trailing `None` fields.

It was reformatted to:

```csv
file,name,x,y,z,aRadius,bRadius,cRadius
```

and each row now fills `bRadius` with the same value as `aRadius`.

### `config/initial_embryo.csv`

The Napari-style header was changed from:

```csv
cell_type,z,y,x
```

to:

```csv
cellType,z,y,x
```

Both spellings are supported by the parser, but `cellType` matches one of the explicit aliases cleanly.

## CSV Pipeline Extraction

### Motivation

After adding header aliases, robust field lookup, `bRadius` fallback, Napari support, and bad-row handling, CSV parsing became too large for `CellFactory.cpp`.

### New Handler

A new handler was introduced:

```text
includes/CsvHandler.hpp
src/CsvHandler.cpp
```

It follows the same architectural idea as `ImageHandler`: keep file-format parsing and preprocessing out of the factory/model logic.

### New Data Structure

`CsvHandler` returns `InitialCellRecord` objects:

```cpp
struct InitialCellRecord
{
    std::string filePath;
    std::string cellName;
    float x;
    float y;
    float z;
    float aRadius;
    float bRadius;
    float cRadius;
};
```

### Current Responsibility Split

`CsvHandler` owns:

- CSV splitting and trimming
- simple quoted field handling
- header normalization
- alias lookup
- named regular initial CSV parsing
- named Napari CSV parsing
- old positional regular CSV fallback
- old positional Napari fallback
- bad-row warnings
- `bRadius` fallback behavior

`CellFactory` now owns:

- reading `InitialCellRecord` objects from `CsvHandler`
- converting those records into `EllipsoidParams`
- constructing `Ellipsoid` instances
- grouping cells by frame path

### Build Integration

`src/CsvHandler.cpp` was added to the main `celluniverse` executable in `CMakeLists.txt`.

### Verification

The refactor was verified with:

```bash
cmake --build build -j20
```

The build completed successfully.

## Branch-Only Preprocessing, Pooling, And Cleanup Updates

This section documents the accepted changes that exist on this branch after the preprocessing/debugging work that followed the original 2026-04-22 thread. These notes are branch-local and should not be read as historical behavior for older branches.

### Iterative Preprocessing Random Step Policy

The iterative contrast preprocessing was changed from a one-pass directional gate loop into a randomized reward/penalty step search.

Current behavior:

- reward gates are generated from `iterative_reward_gate` down to `iterative_reward_gate_min` using `iterative_reward_gate_step`
- each gate contributes two possible random operations:
  - `Penalty`
  - `Reward`
- all step probabilities start uniform
- after each trial, the selected step probability is updated by `iterative_reward_gate_learning_rate`
- probabilities are normalized after every update and clamped by `iterative_reward_gate_min_probability`
- accepted image edits are cumulative inside a radius trial
- each radius trial returns its historical best accepted stack, not necessarily the final current stack

The old name `iterative_reward_gate_decrement` was removed because the value now means gate spacing, not a directional decrement. The active name is:

```yaml
iterative_reward_gate_step
```

The old penalty shrink/backoff logic was removed. Penalty operations now use fixed:

```yaml
iterative_penalty
```

The removed knobs are:

```yaml
iterative_min_penalty
iterative_collapse_backoff
```

### Contrast Score And Radius Search

Contrast scoring now evaluates candidate preprocessing results across multiple radius windows. Radius trials are processed in bounded batches rather than averaging the result of a batch.

Relevant controls:

```yaml
contrast_window_radius_step
contrast_penalty_min_radius_scale
contrast_reward_weight
contrast_penalty_weight
preprocess_radius_batch_size
iterative_score_max
iterative_score_target_tolerance
iterative_score_percentile
```

The final radius trial is selected by target-score quality rather than blindly choosing the largest score. Scores inside the configured target window are preferred. This prevents overly aggressive contrast amplification from winning simply because its score is numerically larger.

### Frame Intensity Normalization

The old shared/global intensity normalization was removed. Normalization is now per-frame only.

Current controls:

```yaml
frame_intensity_normalization_enabled
frame_intensity_percentile_exclude_zeros
frame_intensity_scale_low_percentile
frame_intensity_scale_high_percentile
frame_intensity_hard_max
```

When enabled, each frame computes its own low/high percentile references. The previous cross-frame reference mode and the old `global_intensity_*` naming were removed.

### Removed Post-Process Intensity Adjustment

The old post-process hard black threshold and middle-band amplification controlled by these parameters was removed:

```yaml
post_process_intensity_adjustment_enabled
post_process_amplification
post_process_black_percentile
post_process_white_percentile
```

The corresponding C++ loop in `ImageHandler::processPreparedSequence` and the old helper-script copy in `scripts/image_processor_clean.py` were removed. Remaining post-process blur/blend controls are still active:

```yaml
post_process_blur_sigma
post_process_final_blur_sigma
post_process_final_direct_weight
post_process_final_direct_amplification
post_process_final_blurred_amplification
```

These happen before post-alignment blackoff.

### Post-Alignment Blackoff

After preprocessing, the post-alignment cleanup sequence is:

```text
post_alignment_black_threshold
post_alignment_black_percentile
adaptive chunk blackoff
tiny isolated particle removal
export/load final preprocessed frame
```

`post_alignment_black_threshold` is an absolute cutoff: pixels below the normalized threshold are set to zero.

`post_alignment_black_percentile` is a per-frame relative cutoff. It samples only finite nonzero pixels, computes the configured percentile cutoff, and zeros pixels at or below that cutoff.

When brightness alignment is disabled, this cleanup happens immediately after each frame is preprocessed. When brightness alignment is enabled, all frames are preprocessed first, aligned to the minimum sampled edge-background value, and then cleaned/exported.

### Edge Brightness Alignment

Edge brightness alignment remains optional:

```yaml
edge_brightness_alignment_enabled
edge_brightness_alignment_xy_margin
edge_brightness_alignment_left_offset
edge_brightness_alignment_right_offset
edge_brightness_alignment_top_offset
edge_brightness_alignment_bottom_offset
edge_brightness_alignment_max_shift
```

The edge sample region uses a shared XY margin with configurable offsets from the real image edges. The top/bottom/left/right definitions are relative to image coordinates:

- left/right: x-axis image columns
- top/bottom: y-axis image rows

When enabled, alignment happens after all frame preprocessing is complete and before final cleanup/export.

### Adaptive Chunk Blackoff

A 3D chunk detector was added after the initial percentile blackoff.

Definition:

- foreground: voxel value `> 0`
- background: voxel value `== 0`
- connectivity: 26-neighbor 3D connectivity
- counted chunk: connected component whose z/y/x bounding-box dimensions are each within the configured size range

Controls:

```yaml
post_alignment_chunk_blackoff_enabled
post_alignment_chunk_target_count
post_alignment_chunk_min_size
post_alignment_chunk_max_size
post_alignment_chunk_percentile_step
post_alignment_chunk_max_percentile
post_alignment_chunk_detector_threads
```

Behavior:

```text
copy post-threshold, pre-percentile stack as baseline
apply initial post_alignment_black_percentile
count chunks
if count > target:
    restore baseline
    raise percentile by post_alignment_chunk_percentile_step
    reapply blackoff
    count again
    repeat until count <= target or max percentile is reached
```

The adaptive retries now restore the unblackoffed baseline before each raised-percentile attempt. This avoids cumulative blackoff from repeatedly computing percentiles on an already-zeroed stack.

The detector has a bounded parallel foreground-mask prepass controlled by:

```yaml
post_alignment_chunk_detector_threads
```

The exact 26-neighbor connected-component flood fill remains serial to avoid racy overcounts. If the detector is called inside an existing OpenMP region, the prepass falls back to one thread to avoid nested oversubscription.

### Tiny Isolated Particle Removal

A post-blackoff cleanup step was added to remove isolated nonzero connected components that are smaller than the configured minimum cell size.

Control:

```yaml
post_alignment_tiny_particle_removal_enabled
```

Default:

```yaml
post_alignment_tiny_particle_removal_enabled: false
```

When enabled, it runs after adaptive chunk blackoff. It uses the same 26-neighbor component traversal and removes a component if its bounding-box dimensions are all smaller than the configured minimum cell diameters:

```text
x dimension < 2 * minARadius
y dimension < 2 * minBRadius, or 2 * minARadius if B is not configured
z dimension < 2 * minCRadius
```

The step logs removed component and voxel counts under:

```text
[PostAlignmentTinyParticleRemoval]
```

### Cube Pooling

The previous adaptive cube pooling naming was simplified to fixed cube pooling:

```yaml
cube_pooling_enabled
cube_pooling_cost_comparison_enabled
cube_pooling_cube_size
cube_pooling_mode
```

Supported modes are now:

```yaml
mean
max
min
median
top_percentile
low_percentile
```

Additional controls:

```yaml
cube_pooling_top_fraction
cube_pooling_low_fraction
```

Mode behavior:

- `mean`: fill each cube with the arithmetic mean
- `max`: fill each cube with the maximum voxel value
- `min`: fill each cube with the minimum voxel value
- `median`: fill each cube with the median voxel value
- `top_percentile`: fill each cube with the mean of the brightest configured fraction
- `low_percentile`: fill each cube with the mean of the dimmest configured fraction

The earlier `lower_percentile` wording was renamed to `low_percentile`, and `cube_pooling_lower_fraction` was renamed to `cube_pooling_low_fraction`.

### Export Behavior

Preprocessed debug export now writes only completed, final-ready preprocessed images after all enabled post-preprocessing cleanup has run.

Old unconditional duplicate exports were removed. Runtime image export is controlled by the same format switches:

```yaml
export_frame_png
export_frame_tiff
export_preprocessed_images
export_post_localization_images
```

When preprocessing debug export is enabled, final preprocessed stacks are written under:

```text
output/preprocessed/<frame>.tif
```

The old duplicate `real_tiff`/`synth_tiff` style unconditional export path was removed.
