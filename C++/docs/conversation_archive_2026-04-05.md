# Conversation Archive - 2026-04-05

## Environment and Build

- Investigated initial CMake failure due to missing OpenCV package discovery.
- Clarified installation guidance for:
  - `libopencv-dev`
  - `libyaml-cpp-dev`
- Added clearer OpenCV-not-found error messaging in `CMakeLists.txt`.
- Fixed OpenCV logging compile issue by explicitly including:
  - `opencv2/core/utils/logger.hpp`

## VS Code / Tooling

- Verified the project exports `build/compile_commands.json`.
- Explained that compilation succeeded because CMake had correct include paths while VS Code IntelliSense did not.
- Recommended pointing VS Code to:
  - `${workspaceFolder}/build/compile_commands.json`

## Image Input Handling

- Investigated TIFF load crash from files like `._frame001.tif`.
- Determined these were macOS sidecar metadata files.
- Updated directory input discovery to ignore hidden / sidecar files such as:
  - `._*`
  - dotfiles

## Split Point Selection

- Replaced the old mean/stddev threshold for PCA split-point selection with percentile-style selection.
- Added YAML-configurable parameter:
  - `cell.splitBrightestFraction`
- Current behavior:
  - collect in-cell brightness values
  - keep the top configured fraction for PCA candidate points

## Preprocessing Changes

### Safe Gaussian Blur

- Replaced plain Gaussian blur with a masked blur that avoids bleeding zero-valued black borders into valid image content.

### Sigmoid Center

- Changed sigmoid center calibration from mean-plus-offset logic to a configurable percentile of the calibration ROI.
- Added YAML-configurable parameter:
  - `simulation.sigmoid_center_percentile`

### Post-Sigmoid Background Subtraction

- Added post-sigmoid subtraction using a heavily blurred image.
- Later changed the subtraction source from blurred raw normalized image to blurred processed pre-sigmoid image.
- Added / used YAML parameter:
  - `simulation.post_sigmoid_subtract_sigma`

### Mean-Capped Blur Subtraction

- Removed a temporary multiplicative subtraction factor.
- Current subtraction behavior:
  - blur the processed pre-sigmoid image
  - cap blurred values above the blurred mean down to the mean
  - subtract this capped image from the post-sigmoid image
  - clamp negatives to zero

### Dimmest-Percentile Floor Subtraction

- Added a second subtraction step after blurred-image subtraction:
  - compute a dimmest percentile over the corrected slice
  - subtract that scalar from the whole slice
  - clamp negatives to zero
- Added YAML-configurable parameter:
  - `simulation.post_sigmoid_dimmest_percentile`

## Cell Rendering and Brightness

- Verified that each `Spheroid` object had its own `_brightness` member, but the renderer was previously using only `simulation.cell_color`.
- Updated rendering so each cell is now drawn using its own `_brightness`.
- Changed first-frame cell initialization so brightness starts from the configured `simulation.cell_color`.
- Added per-frame brightness update rule:
  - measure mean real-image brightness inside the cell volume
  - update with a weighted blend:
    - `new = old * (1 - blend) + observed * blend`
- Added YAML-configurable parameter:
  - `cell.brightnessUpdateBlend`

## Frame 1 Ground-Truth Size Restoration

- Added first-frame-only correction after optimization:
  - compare tracked cell radii against initial ground-truth radii by cell name
  - if different, restore major/minor radius to ground truth
  - recompute brightness using the corrected size
  - regenerate the synth frame

## Synthetic Background Adaptation

- Added frame-to-frame adaptive synthetic background logic for frames after the first:
  - exclude regions occupied by slightly expanded previous-frame cells
  - gather remaining real-frame voxels
  - compute the mean of the brightest configured fraction of those remaining voxels
  - scale by current-frame mean brightness divided by previous-frame mean brightness
  - overwrite current synth background color with the scaled value
- Added YAML-configurable parameters:
  - `simulation.adaptive_background_expand_factor`
  - `simulation.adaptive_background_top_fraction`

## Residual Logging

- Verified the code originally logged only signed average residual.
- Added logging of average absolute residual:
  - `avg_abs_resid`

## Split Probability

- Reviewed split probability formula based on previous-frame elongation.
- Reworked probability normalization:
  - compute raw split probabilities for all cells in a frame
  - rescale all of them together so the maximum becomes `max_split_probability`
  - preserve relative ratios among cells

## Split Elongation Logic

- Temporarily changed split elongation logic from PCA variance ratio to `a / c`.
- Observed this caused strong false positives for oblate but single cells.
- Reverted back to original PCA-based elongation:
  - `lambda1 / lambda2`

## PCA Point Filtering

- Added an additional rule to PCA point selection:
  - exclude points with brightness lower than current `background_color`
- Applied both to:
  - the in-cell values used to derive the percentile threshold
  - the final raw points passed to PCA

## Draw Outline

- Changed `drawOutline()` so RGB outputs are no longer green-only.
- Outline is now drawn on all three channels at approximately:
  - `0.4`

## Split Behavior Analysis

- Investigated multiple false positives and false negatives from logged runs.
- Identified recurring causes:
  - permissive split threshold
  - cost improvements from intensity structure rather than true biological division
  - z-stacked pseudo-splits
  - aggressive preprocessing altering the cost landscape
- Confirmed:
  - current split probability uses previous-frame PCA elongation
  - current split validation uses current-frame PCA elongation
  - daughter cells are still created with approximately equal volume

## Verified Code Behaviors Discussed

- First frame looks better largely because:
  - it starts from initialized CSV geometry
  - splits are disabled on frame 1
- Later frames inherit cells via `copyCellsForward()`
- Radius is then changed by perturbation-based optimization
- Synthetic image cost is based on stackwise L2 norm against the processed real frame
- Overlap penalty is applied as a soft additive term to the cost
- `sigmoid_k` is still active
- `sigmoid_center_offset` is currently parsed but not used in the active percentile-based calibration path
- `a` and `b` are always equal in the current oblate spheroid model

## Files Touched During Session

- `CMakeLists.txt`
- `README.md` discussion only, no final README patch committed here
- `src/main.cpp`
- `src/CellFactory.cpp`
- `src/CellUniverse.cpp`
- `src/Spheroid.cpp`
- `src/Frame.cpp`
- `includes/CellUniverse.hpp`
- `includes/CellFactory.hpp`
- `includes/ConfigTypes.hpp`
- `includes/Spheroid.hpp`
- `includes/Frame.hpp`
- `config/config.yaml`

## Notes

- Several intermediate experiments were added, evaluated, and in some cases later reverted.
- This archive reflects the final state of the conversation and the main reasoning/changes made during the session.
