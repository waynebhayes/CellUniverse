# Changelog

## Voxel-to-Analytical Rotation Rewrite

The spheroid rendering was completely rewritten from a **voxel-matrix approach**
(precompute a 3D grid of 0/1 values, rotate it, look up pixels) to an
**analytical inverse-rotation approach** (for each pixel, inverse-rotate back
to local coords and evaluate the spheroid equation). This eliminates large
memory allocations, fixes rotation artifacts from integer discretization, and
significantly improves performance by only scanning a bounding box instead of
the full image.

### Files Changed

#### `C++/includes/Spheroid.hpp`

##### SpheroidParams struct

- **Line 16** (added): `#ifndef M_PI` / `#endif` preprocessor guard for
  portability.
- **Lines 31-33** (improved): Rotation angle comments changed from
  `// x fixed`, `// y fixed`, `// z fixed` to
  `// rotation about x-axis (radians)`, etc.
- **Line 35** (changed): Default constructor now explicitly initializes
  `theta_x`, `theta_y`, `theta_z` to `0`.
- **Line 38** (added): New 9-parameter constructor
  `SpheroidParams(name, x, y, z, majorRadius, minorRadius, theta_x, theta_y, theta_z)`
  to allow specifying rotation angles at construction.
- **Lines 47-49** (fixed): Angle calculation changed from `std::atan(z/y)` to
  `std::atan2(_z_vec[1], _z_vec[2])` (and similar for y, z). Fixes undefined
  behavior when divisor is zero and produces correct angle signs.

##### Spheroid class

- **Lines 80-82** (added): `double _theta_x, _theta_y, _theta_z` private
  members to store rotation state.
- **Lines 86-87** (added): `inverseRotatePoint(dx, dy, dz, &lx, &ly, &lz)` —
  analytically transforms world-space displacements back to the spheroid's
  local (upright) frame using inverse rotation matrices.
- **Line 95** (changed): Default Spheroid constructor initializes `_theta_x`,
  `_theta_y`, `_theta_z` to 0.
- **Lines 97-99** (changed): `printCellInfo()` now prints `"Spheroid name:"`
  (was `"Sphere name:"`) and includes `theta_x`, `theta_y`, `theta_z` in
  output.
- **Removed**: `matrix` and `rotated_matrix` members
  (`std::vector<std::vector<std::vector<int>>>`). The spheroid is no longer
  rasterized into a 3D voxel grid.
- **Removed**: `_is_spheroid()`, `_rotate_point()`, `_rotate_matrix()`.
- **Removed**: `paintPixelUpright()` and `paintPixelRotated()`. Drawing is now
  done analytically instead of matrix lookup.
- **Removed**: Leftover commented-out dead code and trailing whitespace.

#### `C++/src/Cells/Spheroid.cpp`

##### Rotation system rewrite (major change)

- **Lines 19-42** (added): `inverseRotatePoint()` — performs inverse rotation
  (R_total^T = Rx^T * Ry^T * Rz^T) to transform world-space displacements
  back to the spheroid's local coordinate frame. This is the core of the new
  analytical drawing approach.
- **Removed**: `_is_spheroid()` — checked if a voxel coordinate falls inside
  the spheroid equation.
- **Removed**: `_rotate_point()` — applied forward rotation using manual matrix
  multiplication with embedded debug prints.
- **Removed**: `_rotate_matrix()` — built a doubled-size 3D voxel grid and
  rotated every point onto it.

##### Constructor changes

- **Line 48** (bug fix): Minor radius was initialized from
  `init_props.majorRadius` (both radii set to major). Now correctly uses
  `init_props.minorRadius`.
- **Lines 46-76** (changed): `_theta_x`, `_theta_y`, `_theta_z` are now
  initialized from `init_props`. Added radius clamping to enforce config
  bounds. Added check that `_minor_radius` cannot exceed `_major_radius`.
  Added `throw std::invalid_argument` if any radius (a, b, c) is <= 0.
  Removed entire triple-nested loop that built the 3D voxel grid.

##### `print()` rewrite

- **Lines 90-96** (rewritten): Old version printed raw voxel matrix contents
  (triple-nested loop of 0s and 1s). New version prints a single summary line
  with position, radii, and rotation angles.

##### `get_matrix_size()` changed

- **Lines 98-101** (changed): Old version returned `matrix.size()`. New version
  returns `ceil(2 * max(a, b, c))` as an approximate diameter for backward
  compatibility.

##### `getShapeAt()` fix

- **Lines 103-108** (fixed): Old version returned
  `{_major_radius, _minor_radius, paramClass.x, paramClass.y}` — used
  incorrect `paramClass` member. New version returns
  `{_major_radius, _minor_radius, _position.x, _position.y}` — uses the
  cell's actual position.

##### `draw()` rewrite (major change)

- **Lines 113-157** (rewritten):
  - Old: Iterated over every pixel in the entire image, called
    `paintPixelUpright()` to check against the precomputed voxel matrix.
  - New: Computes conservative bounding box using `maxR = max(a, b, c)`.
    Early-returns if the z-slice is outside the bounding sphere. For each pixel
    in the bounding box, computes displacement from center, applies
    `inverseRotatePoint()` to get local coordinates, then checks the analytic
    spheroid equation `x^2/a^2 + y^2/b^2 + z^2/c^2 <= 1`.
  - Performance: Only iterates over the bounding box instead of the full image.

##### `drawOutline()` rewrite (major change)

- **Lines 161-196** (rewritten):
  - Old: Contained entirely commented-out code (non-functional).
  - New: Same bounding-box and inverse-rotation logic as `draw()`. Draws pixels
    where the spheroid equation value falls in `[0.95, 1.05]` (near-surface
    band) at **line 186**. Supports both single-channel (`float`) and
    3-channel (`Vec3f`, green channel) images. Respects the `dormant` flag.

##### `calculateCorners()` updated

- **Lines 404-417** (changed): Old version used `_major_radius` for all three
  bounding box dimensions. New version uses `maxR = max(a, b, c)` as a
  conservative bounding box that accounts for any rotation orientation
  (**line 406**).

##### Removed

- `paintPixelUpright()` and `paintPixelRotated()` — bounds-checked lookups
  into the voxel matrix. No longer needed.
- ~60 lines of commented-out debug prints and dead code.

#### `C++/src/Frame.cpp`

- **Line 172** (changed): `drawOutline` color parameter changed from `0` to
  `1.0` (green channel in BGR).
- **Line 32** (changed): Z-value calculation changed from
  `double zValue = simulationConfig.z_scaling * (i - simulationConfig.z_slices / 2);`
  to `double zValue = i;` — uses raw slice index directly (scaling/centering
  handled elsewhere now).

#### `C++/src/CellFactory.cpp`

- **Line 44**: `z *= z_scaling;` remains, applying z-scaling to parsed
  coordinates.
- **Removed**: `z -= z_offset;` — the line that subtracted `z_offset` from the
  parsed z-coordinate. Consistent with the `Frame.cpp` change: the coordinate
  system no longer centers z-values around the middle slice.

### Summary

| File             | Scope of Change                                                                        |
| ---------------- | -------------------------------------------------------------------------------------- |
| `Spheroid.hpp`   | Structural: new rotation members, removed matrix storage, new API                      |
| `Spheroid.cpp`   | Major rewrite: voxel matrix -> analytical rotation, bug fixes, performance improvement |
| `Frame.cpp`      | One-line change: z-scaling removed from z-value calculation                             |
| `CellFactory.cpp`| One-line removal: z-offset subtraction removed                                         |

---

## 2026-02-17 — Add Angle Perturbation to Spheroid Optimizer

Previously the optimizer could only perturb position (x, y, z) and radii
(majorRadius, minorRadius). The three rotation angles (theta_x, theta_y,
theta_z) were frozen, preventing the optimizer from exploring different
cell orientations when fitting to real microscopy data.

This change adds angle perturbation using the same `PerturbParams` pattern
already used for all other parameters.

### Files Changed

#### `C++/includes/ConfigTypes.hpp`

- **Lines 138-140** (added): Three new `PerturbParams` fields on `SpheroidConfig`:
  ```cpp
  PerturbParams thetaX{};
  PerturbParams thetaY{};
  PerturbParams thetaZ{};
  ```
- **Lines 154-156** (added): YAML parsing for the new fields in `explodeConfig()`:
  ```cpp
  thetaX.explodeParams(node["thetaX"]);
  thetaY.explodeParams(node["thetaY"]);
  thetaZ.explodeParams(node["thetaZ"]);
  ```

#### `C++/src/Cells/Spheroid.cpp`

- **Lines 206-208** (`getPerturbedCell`): Replaced frozen angle pass-through with
  perturbed offsets:
  ```cpp
  // Before:
  _theta_x,  // TODO: add angle perturbation when config supports it
  _theta_y,
  _theta_z

  // After:
  _theta_x + cellConfig.thetaX.getPerturbOffset(),
  _theta_y + cellConfig.thetaY.getPerturbOffset(),
  _theta_z + cellConfig.thetaZ.getPerturbOffset()
  ```

- **Lines 218-220** (`getParameterizedCell`): Added angle offset extraction from
  the params map:
  ```cpp
  float thetaXOffset = params["thetaX"];
  float thetaYOffset = params["thetaY"];
  float thetaZOffset = params["thetaZ"];
  ```

- **Lines 229-231** (`getParameterizedCell`): Added fallback perturbation for the
  `params.empty()` branch:
  ```cpp
  thetaXOffset = Spheroid::cellConfig.thetaX.getPerturbOffset();
  thetaYOffset = Spheroid::cellConfig.thetaY.getPerturbOffset();
  thetaZOffset = Spheroid::cellConfig.thetaZ.getPerturbOffset();
  ```

- **Lines 243-245** (`getParameterizedCell`): Applied angle offsets when
  constructing the new SpheroidParams:
  ```cpp
  // Before:
  _theta_x,  // preserve rotation angles
  _theta_y,
  _theta_z

  // After:
  _theta_x + thetaXOffset,
  _theta_y + thetaYOffset,
  _theta_z + thetaZOffset
  ```

#### `C++/examples/config.yaml`

- **Lines 28-39** (added): Angle perturbation configuration under `cell:`:
  ```yaml
  thetaX:
    prob: 0.3
    mu: 0
    sigma: 0.1
  thetaY:
    prob: 0.3
    mu: 0
    sigma: 0.1
  thetaZ:
    prob: 0.3
    mu: 0
    sigma: 0.1
  ```
  Sigma of 0.1 radians (~5.7 degrees) provides small rotational steps per
  iteration. Adjust `prob` and `sigma` to control how aggressively the
  optimizer explores orientations.

---

## 2026-02-17 — Fix PCA-Based Cell Splitting

`getSplitCells()` was supposed to use PCA on real image data to find the
cell's long axis and split along it. It never worked due to four bugs, all
now fixed.

### Bugs Fixed

1. **PCA ran on every bounding-box grid point** — not just cell pixels. A
   uniform rectangular grid produces meaningless eigenvalues (debug.log showed
   385.3, 385.2, 385.2 — nearly equal, reflecting a cube, not a cell).
2. **PCA result was completely ignored** — after computing PCA, the code
   generated a random split axis with `rand()`. The eigenvectors were never
   used for the actual split direction.
3. **No z_scaling** — z coordinates were raw slice indices (0-30) while x/y
   were pixel coordinates (0-400). With `z_scaling=7`, PCA was heavily biased
   toward the x-y plane because z had artificially low variance.
4. **Daughter cells lost parent rotation** — the 6-arg `SpheroidParams`
   constructor was used, which sets `theta_x/y/z = 0`.

### Files Changed

#### `C++/includes/Spheroid.hpp`

- **Line 111** (changed): Added `float z_scaling` parameter to `getSplitCells`:
  ```cpp
  // Before:
  std::tuple<Spheroid, Spheroid, bool> getSplitCells(const std::vector<cv::Mat> &image) const;

  // After:
  std::tuple<Spheroid, Spheroid, bool> getSplitCells(const std::vector<cv::Mat> &image, float z_scaling) const;
  ```

#### `C++/src/Frame.cpp`

- **Line 291** (changed): Pass `simulationConfig.z_scaling` at the call site
  in `Frame::split()`:
  ```cpp
  // Before:
  std::tie(child1, child2, valid) = oldCell.getSplitCells(_realFrame);

  // After:
  std::tie(child1, child2, valid) = oldCell.getSplitCells(_realFrame, simulationConfig.z_scaling);
  ```

#### `C++/src/Cells/Spheroid.cpp`

Complete rewrite of `getSplitCells` (was lines 249-311, now lines 249-355):

- **Lines 253-258** (fixed): Bounding box clamping now uses `std::floor`/
  `std::ceil` and `image.size() - 1` to prevent off-by-one out-of-bounds
  access (the loop uses `<=`).

- **Lines 260-276** (new): Adaptive brightness threshold. Computes the mean
  brightness across all pixels in the bounding box. Cell pixels are brighter
  than background, so pixels above the mean are classified as cell pixels.
  Returns `false` (no split) if the bounding box contains zero pixels.

- **Lines 278-291** (fixed — Bug 1 & 3): Point collection now filters by
  brightness (`image[z].at<float>(y, x) > meanBrightness`) and scales z
  coordinates by `z_scaling` so PCA operates in world-space:
  ```cpp
  // Before: added ALL grid points, no filtering, no z-scaling
  points.emplace_back(x, y, z);

  // After: only bright pixels, z scaled to world-space
  if (image[z].at<float>(y, x) > meanBrightness) {
      points.emplace_back(
          static_cast<float>(x),
          static_cast<float>(y),
          static_cast<float>(z) * z_scaling);
  }
  ```

- **Lines 293-335** (fixed — Bug 2): PCA result is now used as the split axis.
  The first eigenvector (direction of maximum variance = cell's long axis) is
  extracted, normalized, and used as the split direction. Falls back to random
  only if fewer than 3 bright pixels or a degenerate (zero-norm) eigenvector.
  Debug output prints elongation ratio, split axis, eigenvalues, and bright
  pixel count:
  ```cpp
  // Before: PCA computed and printed but IGNORED; random axis used
  double theta = ((double)rand() / RAND_MAX) * 2 * M_PI;
  double phi = ((double)rand() / RAND_MAX) * M_PI;
  cv::Point3f split_axis(sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi));

  // After: PCA first eigenvector used as split axis
  cv::Vec3f ev = eigenPairs[0].second;
  split_axis = cv::Point3f(ev[0], ev[1], ev[2]);
  // ... normalize, with random fallback only for degenerate cases
  ```

- **Lines 345-351** (fixed — Bug 4): Daughter cells now use the 9-arg
  `SpheroidParams` constructor to inherit the parent's rotation angles:
  ```cpp
  // Before: 6-arg constructor, angles reset to 0
  Spheroid cell1(SpheroidParams(_name + "0", ..., halfMajorRadius, halfMinorRadius));

  // After: 9-arg constructor, angles preserved
  Spheroid cell1(SpheroidParams(
      _name + "0", ...,
      halfMajorRadius, halfMinorRadius, _theta_x, _theta_y, _theta_z));
  ```

---

## 2026-02-17 — Fix and Enable saveCells() for Lineage Tracking

`saveCells()` was commented out and non-functional. The old implementation
referenced `CellParams.file` (which doesn't exist) and only wrote `file,name`
columns. Additionally, `getCellParams()` returned a 6-arg `SpheroidParams` that
lost the rotation angles. All fixed so that cell identities, positions, sizes,
and rotation angles are saved to `output/cells.csv` after each frame.

Cell splits are visible in the output via the naming convention: when cell
`abc123` splits, daughters are named `abc1230` and `abc1231`.

### Files Changed

#### `C++/includes/Frame.hpp`

- **Line 40** (added): Public accessor for the private `imageName` field:
  ```cpp
  std::string getImageName() const { return imageName; }
  ```
  Needed by `saveCells()` to write the frame/file column in the CSV.

#### `C++/includes/Spheroid.hpp`

- **Line 117** (fixed): Changed `getCellParams()` return type from `CellParams`
  to `SpheroidParams`. `Spheroid` does not inherit from `Cell`, so there is no
  virtual override constraint:
  ```cpp
  // Before:
  CellParams getCellParams() const;

  // After:
  SpheroidParams getCellParams() const;
  ```

#### `C++/src/Cells/Spheroid.cpp`

- **Line 400** (fixed): Changed return type in the definition to match the
  header (`SpheroidParams` instead of `CellParams`).

- **Line 401** (fixed): `getCellParams()` now uses the 9-arg constructor
  to include rotation angles:
  ```cpp
  // Before:
  CellParams Spheroid::getCellParams() const {
      return SpheroidParams(_name, ..., _major_radius, _minor_radius);
  }

  // After:
  SpheroidParams Spheroid::getCellParams() const {
      return SpheroidParams(_name, ..., _major_radius, _minor_radius, _theta_x, _theta_y, _theta_z);
  }
  ```

#### `C++/src/Lineage.cpp`

- **Line 314** (fixed): Removed `static_cast<SpheroidParams>` since
  `getCellParams()` now returns `SpheroidParams` directly:
  ```cpp
  // Before (compile error — can't downcast base to derived):
  SpheroidParams params = static_cast<SpheroidParams>(cell.getCellParams());

  // After:
  SpheroidParams params = cell.getCellParams();
  ```

- **Lines 288-329** (rewritten): Replaced the commented-out `saveCells()` with
  a working implementation that:
  - Writes to `output/cells.csv` in append mode (one frame at a time)
  - Writes header on frame 0 (truncates for fresh runs):
    `file,name,x,y,z,majorRadius,minorRadius,theta_x,theta_y,theta_z`
  - Iterates over `frame.cells`, calls `getCellParams()` on each, and writes
    all parameters as a CSV row
  - Prints cell info to stdout via `printCellInfo()` for debugging
  - Logs the number of cells saved per frame

#### `C++/src/main.cpp`

- **Line 126** (uncommented): Enabled the `saveCells()` call in the main loop:
  ```cpp
  // Before:
  // lineage.saveCells(frame); // TODO: Fix this

  // After:
  lineage.saveCells(frame);
  ```

---

## 2026-02-17 — Fix Daughter Cell Radii Causing All Splits to Be Rejected

Cell splits were silently rejected every time because `getSplitCells()` halved
the parent's radii (`_major_radius / 2.0`), producing daughter radii below
`minMajorRadius`. For example, a parent with `majorRadius=28` produced
daughters with radius 14, which fails the `minMajorRadius: 15` constraint in
`checkConstraints()`. Cell `e3d034...` (radius 14.5) would produce daughters
with radius 7.25 — far below the floor.

The fix scales radii by `cbrt(0.5) ≈ 0.794` instead of `0.5`. This correctly
halves each daughter's *volume* (V = 4/3 * pi * a * b * c) while keeping radii
at ~80% of the parent, well above the minimum constraints.

### Files Changed

#### `C++/src/Cells/Spheroid.cpp`

- **Lines 342-347** (fixed): Replaced radius halving with volume-based scaling:
  ```cpp
  // Before:
  double halfMajorRadius = _major_radius / 2.0;
  double halfMinorRadius = _minor_radius / 2.0;

  // After:
  double volumeScale = std::cbrt(0.5);
  double daughterMajorRadius = _major_radius * volumeScale;
  double daughterMinorRadius = _minor_radius * volumeScale;
  ```

- **Lines 350-355** (updated): Daughter cell construction now uses
  `daughterMajorRadius` / `daughterMinorRadius` instead of the old
  `halfMajorRadius` / `halfMinorRadius` variable names.

---

## 2026-02-17 — Fix PCA Collecting Pixels Outside the Cell

PCA was collecting all bright pixels in the bounding box, not just pixels
belonging to the cell. With z-scaling=7 the bounding box spanned ~60 z-slices,
capturing 140,000+ bright pixels from background and neighboring cells. PCA
on this data always found z as the dominant axis (eigenvalue ratios of ~57x),
so daughters were placed purely above/below each other — a useless split that
the cost function correctly rejected every time.

The fix uses the same `inverseRotatePoint` + spheroid equation test that
`draw()` uses, so PCA only sees voxels that are actually inside this cell's
analytical boundary.

### Files Changed

#### `C++/src/Cells/Spheroid.cpp`

- **Lines 260-287** (rewritten): Replaced the brightness-threshold approach
  with the inverse-rotation spheroid membership test:
  ```cpp
  // Before: collected ALL pixels brighter than the mean in the bounding box
  if (image[z].at<float>(y, x) > meanBrightness) {
      points.emplace_back(x, y, z * z_scaling);
  }

  // After: only collect pixels inside this cell's spheroid boundary
  double dx = (double)x - _position.x;
  double dy = (double)y - _position.y;
  double dz = (double)z - _position.z;
  double lx, ly, lz;
  inverseRotatePoint(dx, dy, dz, lx, ly, lz);
  double val = (lx*lx)/(a*a) + (ly*ly)/(b*b) + (lz*lz)/(c*c);
  if (val <= 1.0) {
      points.emplace_back(x, y, z * z_scaling);
  }
  ```
  This also removes the mean-brightness computation loop (old lines 260-276)
  since it is no longer needed.

---

## 2026-02-17 — Remove z_scaling from PCA Points

Even after switching to the spheroid-boundary test, PCA still found z as the
dominant axis because z_scaling=7 made the oblate spheroid (flat in local z)
appear as a tall column in world-space. The eigenvalue ratio was ~48x in z,
so every split axis was `(~0, ~0, ~1)` — stacking daughters vertically, which
never reduced cost.

PCA should see the cell's shape as it appears in image-slice coordinates, not
world-space. Removing `* z_scaling` from the z coordinate lets PCA find the
actual x/y elongation of the cell.

### Files Changed

#### `C++/src/Cells/Spheroid.cpp`

- **Lines 278-285** (changed): Removed `* z_scaling` from the z coordinate
  when adding points for PCA:
  ```cpp
  // Before:
  points.emplace_back(
      static_cast<float>(x),
      static_cast<float>(y),
      static_cast<float>(z) * z_scaling);

  // After:
  points.emplace_back(
      static_cast<float>(x),
      static_cast<float>(y),
      static_cast<float>(z));
  ```

---

## 2026-02-17 — Fix checkConstraints() Using Default-Constructed Config

`checkConstraints()` created a local `SpheroidConfig config;` with all-zero
fields (`minMajorRadius=0`, `maxMajorRadius=0`, etc.). The check
`_major_radius <= config.maxMajorRadius` evaluated to `22 <= 0` — always
false. This meant **every single split was rejected** at the constraint check,
regardless of whether daughter radii were valid.

### Files Changed

#### `C++/src/Cells/Spheroid.cpp`

- **Lines 397-399** (fixed): Use the static `cellConfig` instead of a
  default-constructed local:
  ```cpp
  // Before:
  bool Spheroid::checkConstraints() const {
      SpheroidConfig config;
      return (config.minMajorRadius <= _major_radius) && ...
  }

  // After:
  bool Spheroid::checkConstraints() const {
      return (cellConfig.minMajorRadius <= _major_radius) && ...
  }
  ```

---

## 2026-02-17 — Add Post-Split Burn-in to Fix Splits Being Rejected by Cost Function

Cell splits were mechanically working (PCA found correct axes, constraints
passed, daughters were created) but **always rejected** by the hill-climbing
cost comparison. Root cause: `Frame::split()` compared the cost of two
freshly-placed, unoptimized daughter cells against a well-optimized parent.
The raw daughters never match the real data as well as the parent that had
thousands of perturbation iterations to settle — so `costDiff < 0` was never
satisfied and every split was reverted.

The fix adds a **40-iteration burn-in** phase after the split: the two
daughters are alternately perturbed (position, size, angles) using the same
`getPerturbedCell()` logic as normal optimization. Only perturbations that
reduce cost are kept. After burn-in, the settled daughters' cost is compared
against the parent's cost. This gives splits a fair chance.

Additionally, split probability was increased from 1% to 5% so that more
split attempts occur during the optimization loop.

### Files Changed

#### `C++/src/Frame.cpp`

- **Lines 310-373** (rewritten): Replaced the immediate cost comparison in
  `Frame::split()` with a post-split burn-in loop:
  ```cpp
  // Before (lines 310-328): immediate cost comparison, daughters unoptimized
  auto newSynthFrame = generateSynthFrame();
  double newCost = calculateCost(newSynthFrame);
  double oldCost = calculateCost(_synthFrame);
  // ... callback returns {newCost - oldCost, callback}

  // After (lines 310-373): 40-iteration burn-in before cost comparison
  auto bestSynthFrame = generateSynthFrame();
  double bestCost = calculateCost(bestSynthFrame);
  double oldCost = calculateCost(_synthFrame);

  // Temporarily set _synthFrame for fast incremental generation
  auto savedSynthFrame = _synthFrame;
  _synthFrame = bestSynthFrame;

  const int BURN_IN_ITERATIONS = 40;
  for (int iter = 0; iter < BURN_IN_ITERATIONS; ++iter) {
      size_t dIdx = (iter % 2 == 0) ? d1Idx : d2Idx;
      Spheroid saved = cells[dIdx];
      cells[dIdx] = cells[dIdx].getPerturbedCell();
      // ... validate, generateSynthFrameFast, accept if cost decreases
  }
  _synthFrame = savedSynthFrame;  // restore for fair old-vs-new comparison
  return {bestCost - oldCost, callback};
  ```

- **Line 313** (`generateSynthFrame()`): Initial full synth frame with raw
  daughter placement.
- **Lines 318-319**: Save original `_synthFrame` and temporarily replace it
  with the split frame so `generateSynthFrameFast()` can use it as the base
  for incremental z-slice redraws.
- **Lines 324-349**: Burn-in loop. Each iteration:
  1. Picks a daughter (alternating d1/d2 by even/odd iteration)
  2. Creates a perturbed copy via `getPerturbedCell()`
  3. Validates overlap with `checkIfCellsValid()`
  4. Generates trial synth frame via `generateSynthFrameFast()` (fast:
     only redraws z-slices affected by the perturbation)
  5. Accepts if trial cost < best cost; otherwise reverts
- **Lines 354-357**: Debug output `[Split Burn-in]` showing parent name,
  accepted/total burn-in iterations, old cost, new cost, and cost difference.
- **Lines 359-371**: Callback unchanged — if the optimizer accepts the split,
  `_synthFrame` is set to the burn-in's best frame; if rejected, daughters
  are popped and parent is restored.

#### `C++/examples/config.yaml`

- **Line 56** (changed): Split probability increased from 0.01 to 0.05:
  ```yaml
  # Before:
  perturbation: 0.99
  split: 0.01

  # After:
  perturbation: 0.95
  split: 0.05
  ```
  With 2100 total iterations (350 * 6 cells), this gives ~105 split attempts
  instead of ~21. Each attempt now has a meaningful chance of acceptance
  thanks to the burn-in.

---

## 2026-02-17 — Add Split Cost Penalty to Prevent Spurious Splits

The burn-in fix caused the opposite problem: **too many splits**. Frame 2
had 11 cells (should be 6) and frame 3 had 18 (should be 8). Debug.log
showed cascading splits: `e907...5` → `e907...50` → `e907...500` →
`e907...5000`, each accepted with noise-level cost improvements of -0.007
to -0.96 (on a total cost of ~10300).

Root cause: the burn-in allows daughters to perturb into positions that
reduce cost by a tiny amount, even for cells that shouldn't be splitting.
A diff of -0.007 is indistinguishable from noise but still passes
`costDiff < 0`.

### Fixes Applied

1. **Split cost penalty**: A split must reduce cost by more than
   `split_cost` to be accepted. This filters out all noise-level
   improvements.
2. **Reduced burn-in**: From 40 to 20 iterations to prevent over-fitting.
3. **Reduced split probability**: From 5% to 3%.

### Files Changed

#### `C++/includes/ConfigTypes.hpp`

- **Line 50** (added): `float split_cost;` field on `ProbabilityConfig`:
  ```cpp
  float split_cost;
  ```
- **Line 51** (changed): Default constructor initializes `split_cost` to 0:
  ```cpp
  ProbabilityConfig() : perturbation(0.0f), split(0.0f), split_cost(0.0f) {}
  ```
- **Lines 60-62** (added): YAML parsing for `split_cost` in `explodeConfig()`:
  ```cpp
  if (node["split_cost"]) {
      split_cost = node["split_cost"].as<float>();
  }
  ```
- **Line 68** (added): Print `split_cost` in `printConfig()`.

#### `C++/src/Lineage.cpp`

- **Lines 228-251** (rewritten): Split and perturbation now have separate
  acceptance logic in `Lineage::optimize()`:
  ```cpp
  // Before: same threshold for both
  costDiff = result.first;
  accept(costDiff < 0);

  // After: split uses penalty, perturbation unchanged
  if (chosenOption == "perturbation") {
      result = frame.perturb();
      costDiff = result.first;
      accept(costDiff < 0);
  } else if (chosenOption == "split") {
      result = frame.split();
      costDiff = result.first;
      accept(costDiff < -config.prob.split_cost);  // Must overcome penalty
  }
  ```

#### `C++/src/Frame.cpp`

- **Line 324** (changed): Reduced `BURN_IN_ITERATIONS` from 40 to 20:
  ```cpp
  // Before:
  const int BURN_IN_ITERATIONS = 40;
  // After:
  const int BURN_IN_ITERATIONS = 20;
  ```

#### `C++/examples/config.yaml`

- **Lines 56-58** (changed): Adjusted split parameters:
  ```yaml
  # Before:
  perturbation: 0.95
  split: 0.05

  # After:
  perturbation: 0.97
  split: 0.03
  split_cost: 2
  ```
  A `split_cost` of 2 means the two daughters must reduce the total L2
  cost by at least 2.0 compared to the single parent. All observed
  spurious splits had |diff| < 1.0, so this blocks them while leaving
  room for genuine splits (which should show larger improvements).

---

## 2026-02-17 — Two-Phase Optimization: Perturbation First, Then Split Detection

Random split attempts during optimization caused cascading splits (one cell
splitting repeatedly: `e907...5` → `e907...50` → `e907...500`). Even with
a split cost penalty, random attempts mixed splits with perturbation in an
unpredictable order, leading to 11 cells in frame 2 (should be 6) and 18
in frame 3 (should be 8).

Replaced with a deterministic two-phase approach:
1. **Phase 1**: Run ALL perturbation iterations first (no splits). This
   settles every cell into its best position/size/angle.
2. **Phase 2**: After optimization is complete, try splitting each cell
   exactly once. Accept only if cost improves by > `split_cost`. After
   each accepted split, run extra perturbation iterations so daughters
   can settle.

This prevents cascading splits because:
- Each original cell is tried only once (daughters are NOT tried)
- Splits happen after cells are fully optimized, so the cost comparison
  is fair (settled parent vs. burned-in daughters)
- `split_cost` threshold filters noise-level improvements

### Files Changed

#### `C++/includes/Frame.hpp`

- **Line 38** (added): New `trySplitCell` method declaration:
  ```cpp
  CostCallbackPair trySplitCell(size_t cellIndex);
  ```
  Splits a specific cell by index (vs `split()` which picks randomly).

#### `C++/src/Frame.cpp`

- **Lines 269-276** (refactored): `Frame::split()` now delegates to
  `trySplitCell()`:
  ```cpp
  CostCallbackPair Frame::split() {
      // ... random index selection ...
      return trySplitCell(index);
  }
  ```

- **Lines 278-372** (added): New `Frame::trySplitCell(size_t index)` method.
  Contains the split logic previously in `split()`: PCA-based daughter
  placement, overlap check, 20-iteration burn-in, cost comparison, and
  accept/reject callback. Takes a specific cell index rather than random.

#### `C++/src/Lineage.cpp`

- **Lines 163-254** (rewritten): `Lineage::optimize()` completely
  restructured into two phases:

  **Phase 1** (lines 179-208): Perturbation-only optimization loop.
  Removed the random split/perturbation selection (`discrete_distribution`,
  `chosenOption`, etc.). All iterations now call `frame.perturb()`.
  Residual monitoring preserved.

  **Phase 2** (lines 210-253): Post-optimization split detection.
  - Lines 216-219: Collect cell names before the loop (prevents trying
    daughter cells if a split is accepted).
  - Lines 224-233: Find each cell's current index by name. `SIZE_MAX`
    skip handles cells that were already removed (shouldn't happen in
    this flow, but safe).
  - Line 235: Call `frame.trySplitCell(idx)` — deterministic, one attempt
    per cell.
  - Line 239: Accept only if `costDiff < -config.prob.split_cost`.
  - Lines 244-249: After accepted split, run `2 * iterations_per_cell`
    (700) extra perturbation iterations so daughters can settle into
    optimal positions.

---

## 2026-02-17 — Tune Split Parameters: Remove Penalty, Increase Burn-in

With the two-phase approach, cascading splits are already prevented (each
cell tried once, daughters never re-tried). The `split_cost` penalty was
blocking ALL splits because the 20-iteration burn-in wasn't enough for
daughters to improve cost by > 2.0 (earlier logs showed best diffs ~-0.96).

### Files Changed

#### `C++/src/Frame.cpp`

- **Line 319** (changed): Increased `BURN_IN_ITERATIONS` from 20 to 100:
  ```cpp
  const int BURN_IN_ITERATIONS = 100;
  ```
  Each daughter now gets ~50 perturbation attempts (alternating), enough
  to settle into a position that genuinely reduces cost if the split is
  real.

#### `C++/examples/config.yaml`

- **Line 58** (changed): Set `split_cost` to 0:
  ```yaml
  split_cost: 0
  ```
  The two-phase design prevents cascading, so no penalty needed. The cost
  comparison (`costDiff < 0`) alone determines whether a split is genuine.

---

## 2026-02-17 — Fix Daughter Overlap Blocking All Splits

Debug.log showed only 1 of 6 cells even reached the burn-in phase — the
other 5 were silently rejected by `checkIfCellsValid()` (overlap check)
before any cost comparison.

Root cause: daughters were placed `_major_radius / 2.0` apart per side
(total separation = `_major_radius`), but the overlap threshold is
`2 * daughter_major * 0.95 = 2 * 0.794 * _major_radius * 0.95 ≈ 1.51 *
_major_radius`. Since `1.0 < 1.51`, **daughters always overlapped** for
cells with minor/major ratio > 0.66 (5 of 6 cells).

### Files Changed

#### `C++/src/Cells/Spheroid.cpp`

- **Line 338** (changed): Increased daughter offset from `_major_radius / 2.0`
  to `_major_radius * 0.8`:
  ```cpp
  // Before: total separation = _major_radius (< overlap threshold 1.51 * _major_radius)
  cv::Point3f offset = split_axis * static_cast<float>(_major_radius / 2.0);

  // After: total separation = 1.6 * _major_radius (> overlap threshold)
  cv::Point3f offset = split_axis * static_cast<float>(_major_radius * 0.8);
  ```
  With offset = 0.8 * _major_radius per daughter, total separation is
  1.6 * _major_radius, which exceeds the overlap threshold of 1.51 *
  _major_radius for all cells regardless of minor/major ratio.

---

## 2026-02-17 — Use Real Image Data for PCA Split Axis Instead of Cell Geometry

Debug.log showed ALL splits had positive cost diff (cost always went UP),
even with 100-iteration burn-in. Root cause: PCA was run on the **cell's
geometric boundary points** (spheroid equation `val <= 1.0`). For oblate
spheroids where `a == b`, this shape is symmetric in x/y, so PCA gives a
random axis — daughters are placed in arbitrary directions, not where the
real cells actually are.

Fix: PCA now runs on **bright pixels from the real image** within an
expanded boundary (1.5x radius). If a cell has genuinely split, the real
image shows two separate blobs of bright pixels, and PCA finds the axis
between them. For non-splitting cells, bright pixels form a single blob —
PCA finds the blob's shape axis, the split won't improve cost, and it gets
rejected.

### Files Changed

#### `C++/src/Cells/Spheroid.cpp`

- **Lines 260-305** (rewritten): Replaced geometric point collection with
  real-image brightness-based collection in `getSplitCells()`:

  **First pass** (lines 267-280): Compute mean brightness of pixels inside
  the spheroid boundary (val <= 1.0). This adaptive threshold distinguishes
  cell tissue from background without hardcoding brightness values.

  **Second pass** (lines 285-305): Collect bright pixels (above mean) within
  an expanded spheroid boundary (val <= 2.25, i.e., 1.5x radius):
  ```cpp
  // Before: collected ALL points inside spheroid boundary (cell geometry)
  // PCA on symmetric shape → random axis → daughters placed wrong
  if (val <= 1.0) {
      points.emplace_back(x, y, z);
  }

  // After: collect bright real-image pixels in expanded boundary
  // PCA on real data → axis between actual cell blobs → correct placement
  float pixel = image[z].at<float>(y, x);
  if (pixel > meanBrightness && val <= 2.25) {
      points.emplace_back(x, y, z);
  }
  ```
  The 1.5x expansion captures daughter blobs that may extend beyond the
  parent's original boundary (daughters move apart during cell division).

---

## 2026-02-17 — Project PCA Points to X-Y Plane to Fix Z-Dominated Split Axes

Debug.log showed PCA split axes were still z-dominated (z components of
-0.82, -0.97, 0.87) because the expanded 1.5x boundary spans many z-slices.
Bright pixels across 50-100 z-slices produce huge z variance that overwhelms
the x-y variance where the actual split happens.

For oblate spheroids (flat cells, a == b), cell division happens in the x-y
plane. PCA should only see x-y variation to find the split direction.

### Files Changed

#### `C++/src/Cells/Spheroid.cpp`

- **Lines 310-311** (changed): PCA points now use a constant z coordinate
  (the cell's center z), eliminating z variance from PCA:
  ```cpp
  // Before: actual z → z variance dominates PCA → z-oriented axes
  points.emplace_back(x, y, static_cast<float>(z));

  // After: constant z → PCA only sees x-y variance → x-y split axes
  points.emplace_back(x, y, static_cast<float>(_position.z));
  ```
  The third eigenvalue will be ~0 and the third eigenvector will be pure z.
  The first eigenvector (split axis) will lie in the x-y plane, which is
  where oblate spheroid splits actually occur.

---

## 2026-02-17 — Aspect-Ratio PCA Normalization + Centroid-Based Daughter Placement

Two changes to `getSplitCells()` to support splits in **any direction**
(x-y or z) and place daughters where the real data actually is.

### Problem 1: X-Y Projection Blocked Z Splits

The previous fix projected all PCA points to the cell's center z
(`_position.z`), which forced PCA to only find x-y split axes. But cell
division can also happen along the z axis. A z split would be invisible
to PCA since all z coordinates were identical.

### Problem 2: Fixed-Offset Daughter Placement

Daughters were placed at `±0.8 * _major_radius` along the split axis
from the parent center. This arbitrary offset doesn't correspond to where
the actual bright pixel clusters are in the real image. If the two blobs
are asymmetrically positioned, the daughters start in the wrong place
and burn-in can't recover.

### Fix 1: Aspect-Ratio Normalization for PCA

Instead of projecting to x-y, scale the z coordinate by `a/c` (major
radius / minor radius) so the oblate spheroid appears spherical to PCA.
This gives each axis equal weight relative to the cell's physical extent.
After PCA, the eigenvector is transformed back to image space by dividing
the z component by `a/c`, then re-normalizing.

For an oblate spheroid with a=30, c=20: `zScale = 30/20 = 1.5`. A
5-slice z separation becomes 7.5 in PCA space, comparable to a 7.5-pixel
x/y separation. This allows PCA to detect splits in any direction.

### Fix 2: Centroid-Based Daughter Placement

After PCA determines the split axis, each bright pixel is projected onto
the axis through the cell center. Pixels with positive projection go to
group 1, negative to group 2. The 3D centroid of each group becomes the
daughter's initial position. This directly places daughters where the
real bright pixel clusters are.

If centroids are too close (below the overlap threshold), or all pixels
fall on one side, falls back to fixed offset placement using `minSep`.

### Files Changed

#### `C++/src/Cells/Spheroid.cpp`

- **Lines 289-316** (changed): Point collection now stores actual z
  coordinates in `rawPoints` instead of constant `_position.z`:
  ```cpp
  // Before: constant z → only x-y splits possible
  std::vector<cv::Point3f> points;
  ...
  points.emplace_back(x, y, static_cast<float>(_position.z));

  // After: actual z → splits in any direction
  std::vector<cv::Point3f> rawPoints;
  ...
  rawPoints.emplace_back(
      static_cast<float>(x),
      static_cast<float>(y),
      static_cast<float>(z));
  ```

- **Lines 318-376** (rewritten): PCA now uses inline `cv::PCA` with
  aspect-ratio normalization instead of calling `performPCA()`:
  - **Line 322**: Compute `zScale = a / c` (oblate: a > c, so zScale > 1).
  - **Lines 327-332**: Build PCA matrix centered on cell position. The z
    component is scaled by `zScale`:
    ```cpp
    data.at<float>(i, 0) = rawPoints[i].x - _position.x;
    data.at<float>(i, 1) = rawPoints[i].y - _position.y;
    data.at<float>(i, 2) = (rawPoints[i].z - _position.z) * zScale;
    ```
  - **Line 347**: Transform eigenvector back to image space by dividing
    the z component by `zScale`:
    ```cpp
    cv::Point3f ev_image(ev_scaled.x, ev_scaled.y, ev_scaled.z / zScale);
    ```
  - **Lines 350-354**: Normalize the image-space eigenvector to get
    `split_axis`.
  - **Line 367**: Debug output now includes `zScale`.

- **Lines 378-441** (new): Centroid-based daughter placement replaces
  fixed-offset placement:
  - **Lines 383-385**: Compute `volumeScale`, `daughterMajorRadius`,
    `daughterMinorRadius` (unchanged logic, moved earlier).
  - **Lines 390-403**: Project each bright pixel onto split axis. Pixels
    with `projection >= 0` accumulate into `centroid1`; others into
    `centroid2`.
  - **Line 408**: Compute `minSep = 2 * daughterMajorRadius * 0.95`
    (overlap threshold).
  - **Lines 410-433**: If both groups have pixels, compute centroids and
    use as daughter positions. If separation < `minSep`, fall back to
    fixed offset of `(minSep + 1) / 2` per side.
  - **Lines 434-441**: If all pixels are on one side, fall back to fixed
    offset.
  - **Lines 443-452**: Daughter cell construction (unchanged — inherits
    parent rotation, uses volume-scaled radii).

---

## 2026-02-17 — Per-Axis Stddev PCA Normalization + Skip Daughter-Daughter Overlap

Debug.log showed ALL splits rejected across ALL frames. Two root causes:

1. **PCA z-dominance**: For 3 of 6 cells, the split axis had z-component
   > 0.92, placing daughters far apart in z where there's no data.
   The aspect-ratio `zScale` (1.25–1.35) was too weak because oblate
   spheroids have inherently more z-variance (bright pixels span many
   z-slices but cluster near center_x/y in each slice).

2. **Centroid placement always fell back to fixed offset**: Every single
   cell had centroid sep < minSep (e.g., 26.2 < 57.2 for `1f2e...`),
   triggering the fixed-offset fallback. This pushed daughters to
   `±minSep/2` along the wrong (z-dominated) axis, making cost worse.

### Fix 1: Per-Axis Stddev Normalization

Replace aspect-ratio `zScale = a/c` with full per-axis standard deviation
normalization: divide each axis by its empirical stddev before PCA, so
all axes have unit variance. PCA then finds the direction of maximum
*relative* variance — i.e., bimodal structure from two blobs (a split),
regardless of which axis it occurs on. After PCA, the eigenvector is
transformed back to image space by multiplying each component by its
stddev and re-normalizing.

### Fix 2: Always Use Centroid Positions

Remove the `minSep` fallback entirely. Daughters are placed at the 3D
centroids of the bright pixel groups (positive/negative projection onto
split axis). Even if centroids are close together, this is where the
data actually is. Daughters may overlap — that's expected since they
split from a single parent occupying that space.

### Fix 3: Skip Daughter-Daughter Overlap Check

The `checkIfCellsValid()` call in `trySplitCell()` was checking ALL cell
pairs, including the two daughters against each other. Since daughters
start close together (data-driven centroid placement), they always
"overlapped" and the split was blocked. Now:
- Initial overlap check only validates daughters against non-daughter cells
- Burn-in overlap check similarly skips the daughter-daughter pair
- The cost function handles daughter-daughter separation naturally (two
  cells drawing on top of each other makes the synth image too bright)

### Files Changed

#### `C++/src/Cells/Spheroid.cpp`

- **Lines 318-401** (rewritten): PCA normalization in `getSplitCells()`:
  - **Lines 329-331**: Build centered data matrix with raw z (no scaling):
    ```cpp
    data.at<float>(i, 0) = rawPoints[i].x - _position.x;
    data.at<float>(i, 1) = rawPoints[i].y - _position.y;
    data.at<float>(i, 2) = rawPoints[i].z - _position.z;
    ```
  - **Lines 334-343**: Compute per-axis standard deviation:
    ```cpp
    float sx = 0, sy = 0, sz = 0;
    for (int i = 0; i < n; ++i) {
        sx += vx * vx; sy += vy * vy; sz += vz * vz;
    }
    sx = std::sqrt(sx / n); // etc.
    ```
  - **Lines 346-348**: Normalize each column to unit variance:
    ```cpp
    if (sx > 1e-6f) for (int i = 0; i < n; ++i) data.at<float>(i, 0) /= sx;
    ```
  - **Lines 365-368**: Transform eigenvector back to image space by
    multiplying by stddev (inverse of the normalization):
    ```cpp
    cv::Point3f ev_image(ev_norm.x * sx, ev_norm.y * sy, ev_norm.z * sz);
    ```
  - **Line 391**: Debug output now prints `stddev=(sx, sy, sz)` instead
    of `zScale`.

- **Lines 430-460** (changed): Centroid placement always used, no
  `minSep` fallback:
  - Removed `minSep` computation and the `if (sep < minSep)` branch
  - **Lines 432-441**: Always use centroid positions when both groups
    have points
  - **Lines 452-456**: One-sided fallback uses small offset
    (`daughterMajorRadius * 0.5`) instead of the large `minSep`-based
    offset

#### `C++/src/Frame.cpp`

- **Lines 299-324** (rewritten): Initial overlap check in
  `trySplitCell()` now only checks daughters against non-daughter cells:
  ```cpp
  // Before: checked ALL pairs including daughter-daughter
  bool areCellsValid = oldCell.checkIfCellsValid(cells);

  // After: skip daughter-daughter pair
  for (size_t i = 0; i < d1Idx; ++i) {
      for (size_t di : {d1Idx, d2Idx}) {
          // check cells[i] vs cells[di] using same overlap criterion
      }
  }
  ```

- **Lines 342-359** (rewritten): Burn-in overlap check also skips
  daughter-daughter pair:
  ```cpp
  // Before: checked ALL pairs
  if (!saved.checkIfCellsValid(cells)) {

  // After: only check perturbed daughter vs non-daughters
  for (size_t i = 0; i < d1Idx; ++i) {
      // check cells[i] vs cells[dIdx]
  }
  ```
  This allows burn-in perturbations to freely adjust daughter positions
  relative to each other, with only the cost function guiding separation.
