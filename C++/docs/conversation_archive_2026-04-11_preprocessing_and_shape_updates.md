# Conversation Archive: 2026-04-11 Preprocessing And Shape Updates

This document archives the code changes, debugging, and design analysis completed during the conversation on April 11, 2026 in the `C++` CellUniverse project.

## Scope

The work in this session covered:

- extracting image preprocessing and image I/O into a dedicated `ImageHandler` class
- porting the Python preprocessing pipeline from `scripts/image_processor_clean.py` into C++
- exposing preprocessing controls in `config/config.yaml`
- debugging mismatches between the Python and C++ preprocessing behavior
- reducing preprocessing logs
- adding frame-to-frame brightness alignment after preprocessing
- adding YAML-controlled export of preprocessed images
- adding a YAML-controlled preprocess-only debug exit
- making initial cell brightness configurable
- making base background color configurable
- changing contrast-score logic several times
- enabling in-plane anisotropy so spheroid `a` and `b` radii can differ
- making anisotropy initialization and perturbation configurable in YAML
- adjusting real-image output to remain single-channel
- changing outline intensity to follow background instead of using a fixed white value
- discussing possible multithreading opportunities in preprocessing

## 1. `ImageHandler` Class Extraction

### Goal

Create a dedicated C++ class named `ImageHandler` in separate files and move image preprocessing and image I/O responsibilities into it.

### Changes

- Added:
  - `includes/ImageHandler.hpp`
  - `src/ImageHandler.cpp`
- Moved image-related responsibilities out of `main.cpp` and `CellUniverse.cpp`:
  - image path discovery
  - TIFF stack loading
  - per-slice normalization
  - preprocessing pipeline execution
  - interpolation orchestration
  - image-stack statistics and preprocessing score logging

### Integration updates

- `main.cpp` now calls `ImageHandler::getImageFilePaths(...)`
- `CellUniverse.cpp` now calls `ImageHandler::loadFrame(...)`
- `CellUniverse.hpp` no longer declares the old free functions
- `CMakeLists.txt` and test CMake config were updated to compile `src/ImageHandler.cpp`
- `tests/process_image_test.cc` was updated to reference `ImageHandler`

## 2. Python Preprocessing Pipeline Port

### Source of truth used

The preprocessing logic was ported from:

- `scripts/image_processor_clean.py`

### Main C++ ported features

The port introduced the following logic into `ImageHandler`:

- TIFF loading and normalization using `/ 255.0`
- iterative penalty/reward enhancement loop
- local center-surround contrast scoring
- percentile-based Michelson contrast scoring
- percentile-based Weber contrast scoring
- post-process black thresholding
- post-process band amplification
- final per-slice Gaussian blur

### Important implementation note

The Python code intentionally divides TIFF values by `255.0` even when the TIFF slices are `uint16`. The C++ version was updated to match this exact behavior rather than introducing a different normalization strategy.

## 3. Preprocessing Config Plumbing Added To YAML

### Added and/or used under `simulation:`

The following knobs were added or wired through `SimulationConfig` so the preprocessing behavior could be configured from `config/config.yaml`:

- `iterative_penalty`
- `iterative_min_penalty`
- `iterative_collapse_backoff`
- `iterative_penalty_range`
- `iterative_reward_gate`
- `iterative_reward_gate_decrement`
- `iterative_reward_gate_min`
- `iterative_reward`
- `iterative_score_max`
- `iterative_max_count`
- `iterative_no_improvement_patience`
- `iterative_improvement_tolerance`
- `iterative_score_drop_stop_threshold`
- `iterative_score_percentile`
- `iterative_score_percentile_max`
- `iterative_score_percentile_increment`
- `contrast_structure_threshold`
- `contrast_eps`
- `post_process_blur_sigma`
- `post_process_amplification`
- `post_process_black_percentile`
- `post_process_white_percentile`
- `michelson_low_percentile`
- `michelson_high_percentile`
- `michelson_eps`
- `weber_background_percentile`
- `weber_signal_percentile`
- `weber_background_floor`
- `weber_eps`
- `export_preprocessed_images`
- `quit_after_preprocessing`

### Later cleanup

The YAML file was later cleaned to remove stale preprocessing keys that were no longer active in the runtime logic:

- `contrast_inner_window_size`
- `contrast_outer_window_size`
- `calibration_x`
- `calibration_y`
- `calibration_width`
- `calibration_height`

These keys still existed in `ConfigTypes.hpp` at the time of archiving, but were removed from YAML because they were not actively used by the current preprocessing path.

## 4. Debugging Python/C++ Preprocessing Mismatch

The first C++ port did not behave like the Python version. The main mismatches found were:

### 4.1 Premature clipping to `[0, 1]`

Problem:

- The C++ port clipped the normalized sequence too early.
- The Python version does not clip before the iterative reward/penalty loop.
- Because TIFF values were scaled by `/255.0`, some values were greater than `1.0`, and the reward logic depended on those values remaining intact.

Fix:

- Removed the early clamp from `processImage(...)`
- Kept clipping only at the end of the iterative pipeline

### 4.2 Shallow copy bug with `std::vector<cv::Mat>`

Problem:

- The C++ version used normal assignment for `std::vector<cv::Mat>`
- `cv::Mat` copies are shallow by default
- `bestSequence = current` caused the saved best state to alias the current mutable state
- Later destructive updates corrupted the supposed "best" sequence

Fix:

- Added explicit deep stack cloning
- Used cloned copies whenever restoring or preserving `bestSequence`

### Validation

After these fixes, the iterative preprocessing round trace matched the Python reference closely enough to confirm the ported behavior.

## 5. Iterative Preprocessing Logging Changes

### Original issue

The iterative preprocessing printed a line every round, which made debugging noisy and long-running runs hard to scan.

### Change made

- Progress logging was reduced to once every 50 rounds
- Final best score was still always printed

Result:

- intermediate log volume was reduced substantially
- final score visibility was preserved

## 6. Brightness Alignment Across Frames After Preprocessing

### Request

After preprocessing, align all frames together by multiplying each frame by:

`(average mean brightness of all frames) / (current frame mean brightness)`

### Implementation

In the `CellUniverse` constructor:

1. all requested frames are preprocessed first
2. per-frame mean brightness is measured
3. a global mean brightness across frames is computed
4. each preprocessed frame stack is scaled by:
   - `global_mean / current_frame_mean`
5. values are clamped back into `[0, 1]`

### Effect

The actual image stacks used by the optimizer are now brightness-aligned across the selected frame set.

## 7. Preprocessed Image Export

### Goal

Export the preprocessed image stacks directly to disk for debugging.

### Added switch

Under `simulation:`:

- `export_preprocessed_images`

### Behavior

When enabled, the program exports the exact post-preprocessing, post-brightness-alignment stacks to:

- `<output>/preprocessed/<frame_stem>/`

## 8. Per-Cell Brightness Perturbation Probabilities

### Problem identified

The brightness perturbation adaptation logic had been wired through `Spheroid::cellConfig.brightness`, which is shared globally by all cells. That meant:

- one cell's accepted or rejected brightness move changed the probabilities for every cell
- frame-to-frame brightness perturbation behavior was not truly cell-specific
- the intended learning signal from one cell could leak into unrelated cells

### Requested behavior

The requested model was:

- each cell must maintain its own brightness increase and decrease probabilities
- on the very first frame, initialize every cell from the existing YAML brightness perturbation values
- before each later frame, initialize each cell with:
  - `yaml_value * (1 - trust) + previous_frame_final_value * trust`
- make `trust` configurable in `config/config.yaml`

### Implementation

The following changes were made:

- added per-cell runtime storage of brightness perturb probabilities to `Spheroid`
- changed brightness perturb sampling to use the cell's own local probabilities instead of the shared static config
- changed perturbation accept/reject updates so only the perturbed cell's probability pair is adjusted
- preserved split consistency by copying the parent's brightness probability state into both daughter cells during `getSplitCells(...)`
- updated `CellUniverse::copyCellsForward(...)` so newly propagated cells blend their carried-forward probability state with the YAML base using a new trust parameter

### New YAML knob

Under `cell:`:

- `brightnessProbabilityTrust: 0.9`

### Formula used

For each cell and for each of the brightness increase/decrease probabilities on frames after the first:

- `blended = yaml_base * (1 - trust) + previous_frame_probability * trust`

### Compatibility note

The existing `firstFrameBrightnessPerturbationOnly` switch originally disabled later-frame brightness perturbation by zeroing the shared global brightness perturb config. After moving brightness probabilities into each `Spheroid`, that global toggle no longer affected the active sampling path.

To preserve prior semantics, the optimization step now explicitly zeros each cell's local brightness increase/decrease probabilities on frames after the first whenever `firstFrameBrightnessPerturbationOnly: true`.

Each z-slice is saved as a `.png`.

## 8. Preprocess-Only Debug Exit

### Goal

Add a YAML switch to stop execution after preprocessing for debugging.

### Added switch

Under `simulation:`:

- `quit_after_preprocessing`

### Default behavior

The code default was set to:

- `false`

Meaning:

- unless explicitly set to `true` in YAML, the program continues normally to the end

### Final semantics after refinement

When enabled:

- requested frames are still preprocessed
- brightness alignment still runs
- preprocessed image export still runs if enabled
- but the program stops before optimization
- cell loading and full runtime-frame construction were also skipped in the debug path to make the mode more lightweight

Important note:

- this mode means "preprocess all requested frames, then quit"
- it does not mean "quit after the very first frame is preprocessed"

## 9. Initial Brightness Configuration

### Request

Make initial cell brightness configurable in YAML and set the initial value to `0.2`.

### Added config

Under `cell:`:

- `initialBrightness: 0.2`

### Effect

`CellFactory` now seeds new cells from `config.cell->initialBrightness` instead of a hardcoded value.

## 10. Background Color Configuration

### Request

Make the synthetic background color configurable in YAML and place it near `initialBrightness`.

### Added config

Under `cell:`:

- `backgroundColor`

### Effect

- frame objects now start with a configurable synthetic background level
- the existing adaptive-background logic can still update later frames during optimization

## 11. Contrast Score Design Iterations

Several iterations of the contrast score were discussed and implemented:

### 11.1 Single fixed window

Earlier form:

- one fixed `inner` / `outer` contrast window pair

### 11.2 Five-window scale sweep

Changed to:

- generate 5 scales spanning configured cell size range
- compute a score at each
- initially average top 2

### 11.3 Ten-window scale sweep

Changed to:

- generate 10 scales
- initially average top 5
- then changed to select top 1 only

### 11.4 Reverted from 10 back to 5

Due to performance concerns:

- reverted 10 scales back to 5
- temporarily kept top-1 selection

### 11.5 Final current design from this conversation

The final request removed the 5-window sweep entirely.

Current behavior at the end of this session:

- build one contrast window pair from `cell.maxMajorRadius`
- build one contrast window pair from `cell.maxMinorRadius`
- compute one score for each
- return the average of those two scores

This is the active contrast score behavior at the end of the conversation.

## 12. Outline Rendering Changes

### 12.1 Outline intensity tied to background

The outline value was changed from a fixed high/white intensity to:

- `110%` of the current frame background
- capped at `1.0`

This means outlines now scale relative to the current background level instead of always appearing white.

### 12.2 Real image output changed to single-channel

Originally:

- real outputs were converted to 3-channel images before outlines were drawn

Changed to:

- keep real image outputs as single-channel
- draw outlines directly into grayscale images
- then convert to `CV_8U` for saving

## 13. Spheroid `a != b` In-Plane Anisotropy

### Request

Allow `a` and `b` radii to differ, make the difference perturbable like other factors, and make the feature YAML-configurable.

### Chosen model

The implementation kept the existing primary radii convention and introduced a configurable in-plane aspect ratio:

- `a = majorRadius`
- `b = majorRadius / equatorialAspectRatio`
- `c = minorRadius`

Where:

- `equatorialAspectRatio >= 1.0`
- it represents `longer_in_plane_axis : shorter_in_plane_axis`

### New config added

Under `cell:`:

- `abRatio:` perturbation block
  - `increase_prob`
  - `decrease_prob`
  - `mu`
  - `sigma`
- `initialABRatio`
- `maxABRatio`

### Implementation details

- `SpheroidParams` now carries `equatorialAspectRatio`
- `Spheroid` stores `_equatorial_aspect_ratio`
- constructor clamps the ratio to `[1.0, maxABRatio]`
- `b` is derived from `a / ratio`
- perturbation proposals now include ratio perturbation
- daughter cells inherit the parent ratio
- rotation-refine candidate cells preserve the ratio
- saved cell records now include `equatorialAspectRatio`

### Updated affected logic

To keep heuristics more consistent with the new geometry:

- some volume calculations were updated from `a*a*c` to `a*b*c`

### Important behavioral interpretation

At the end of the conversation:

- `majorRadius` still means the longer in-plane radius
- `minorRadius` still means the third-axis radius `c`
- the ratio controls the difference between the two in-plane radii
- `theta_z` rotates the in-plane ellipse around the local `z` axis

## 14. Perturbation Behavior Analysis

The conversation also included analysis of how perturbation currently works:

- perturbation proposals are not action-selected from a list
- each enabled parameter rolls independently
- a single perturbation proposal may change multiple parameters at once
- all sampled parameter changes are bundled into one candidate cell
- the whole candidate is then accepted or rejected as one unit

This was explained using the following concepts:

- `getPerturbedCell()` samples all parameter offsets independently
- each parameter uses its own `PerturbParams`
- there is no exhaustive search over all actions
- perturbation is stochastic and proposal-based, not deterministic or exhaustive

## 15. Multithreading Analysis For Preprocessing

Multithreading for the iterative preprocessing phase was analyzed but not implemented.

### Conclusion reached

Yes, it is feasible, especially:

- per-slice penalty pass
- per-slice reward pass
- per-slice final blur
- per-slice contrast score calculation

### Key constraint

The outer iterative loop must remain sequential because:

- each round depends on the previous round’s state

### Main risks discussed

- shallow `cv::Mat` copies
- oversubscription if OpenCV also parallelizes internally
- shared result containers needing careful thread-local handling

## 16. Files Most Significantly Touched During This Session

The conversation resulted in repeated updates to these files:

- `src/ImageHandler.cpp`
- `includes/ImageHandler.hpp`
- `src/CellUniverse.cpp`
- `src/main.cpp`
- `includes/ConfigTypes.hpp`
- `config/config.yaml`
- `src/Frame.cpp`
- `src/Spheroid.cpp`
- `includes/Spheroid.hpp`
- `src/CellFactory.cpp`
- `includes/CellFactory.hpp`

## 17. Notes About Current Repository State

At the time of this archive:

- `config/config.yaml` had been actively edited multiple times during the conversation
- some legacy config fields still remained in `ConfigTypes.hpp` and logging, even though they were removed from YAML
- the `quit_after_preprocessing` switch was enabled in the working YAML during debugging at one point
- multiple preprocessing and scoring behaviors were revised several times in sequence

This file is intended to preserve the reasoning and evolution of those changes, not just the final code state.

## 18. Verification Summary

Throughout the conversation, the project was rebuilt multiple times using:

```bash
cmake --build build
```

Build verification was repeatedly used after each cross-cutting change set, especially after:

- `ImageHandler` extraction
- Python preprocessing port
- preprocessing bug fixes
- anisotropy model changes
- output/export path changes
- config/plumbing additions

The build was successful after the final changes archived here.
