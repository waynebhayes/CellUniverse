# CellUniverse C++ — Detailed Technical Documentation

## Overview

CellUniverse tracks 3D cells across time-lapse microscopy frames. For each frame, it fits 3D spheroid models to real image data using Monte Carlo perturbation, then detects cell divisions (splits) using PCA on bright pixels. The pipeline runs frame-by-frame: optimize cell positions → detect splits → copy cells forward → save results.

---

## 1. Startup & Initialization

### 1.1 Entry Point — `main()` in `C++/src/main.cpp`

**Arguments** (parsed at lines 151–176):
| Arg | Meaning | Example |
|-----|---------|---------|
| `argv[1]` | First frame number | `1` |
| `argv[2]` | Last frame number | `10` |
| `argv[3]` | Input path (directory, pattern, or file) | `input/original_data` |
| `argv[4]` | Output directory | `output_run01` |
| `argv[5]` | YAML config file | `config.yaml` |
| `argv[6]` | Initial cell CSV | `initial.csv` |

### 1.2 Config Loading — `BaseConfig::explodeConfig()` in `C++/includes/ConfigTypes.hpp`

The YAML config (e.g., `config.yaml`) is parsed into three sub-configs:

- **SimulationConfig** (lines 10–43): `iterations_per_cell`, `background_color`, `cell_color`, `z_scaling`, `blur_sigma`, `z_slices`
- **ProbabilityConfig** (lines 45–73): `perturbation`, `split`, `split_cost`, `split_elongation_threshold`
- **SpheroidConfig** (lines 130–172): Per-parameter perturbation settings (`x`, `y`, `z`, `majorRadius`, `minorRadius`, `thetaX`, `thetaY`, `thetaZ`) each with `prob`, `mu`, `sigma` plus global constraints (`minMajorRadius`, `maxMajorRadius`, `minMinorRadius`, `maxMinorRadius`)

Each perturbation parameter has a `PerturbParams` struct (lines 81–104) with:
- `prob`: Probability of applying a random offset (vs. using mean)
- `mu`: Mean offset
- `sigma`: Standard deviation
- `getPerturbOffset()`: Returns `N(mu, sigma)` with probability `prob`, else returns `mu`

### 1.3 Image Loading — `getImageFilePaths()` in `main.cpp` (lines 46–138)

Supports three input modes:
1. **Printf pattern** (e.g., `frame%03d.tif`): Generates paths for the frame range
2. **Directory**: Auto-detects `.tif`/`.tiff` files, sorts alphabetically, selects frame range
3. **Single file**: Returns one path

Calls `updateTiffConfigIfNeeded()` (lines 27–43) to auto-detect z-slice count from the first TIFF file using `cv::imreadmulti()`.

### 1.4 Cell Creation — `CellFactory::createCells()` in `C++/src/CellFactory.cpp` (lines 18–61)

Reads the initial CSV (columns: `file, name, x, y, z, majorRadius, minorRadius, theta_x, theta_y, theta_z`). For each row:
1. Parse fields
2. Scale z-coordinate: `z *= z_scaling` (line 44)
3. Construct `Spheroid` with clamped radii
4. Group by filename: `map<filename, vector<Spheroid>>`

### 1.5 Lineage Construction — `Lineage()` in `C++/src/Lineage.cpp` (lines 138–162)

For each image path:
1. Call `loadFrame()` to load and preprocess the TIFF
2. Create a `Frame` object with the initial cells (first frame) or empty cells (later frames)

---

## 2. Frame Loading & Z-Interpolation

### 2.1 `loadFrame()` in `C++/src/Lineage.cpp` (lines 55–135)

Each TIFF frame is a multi-page file where each page is one z-slice (e.g., 33 slices from a confocal microscope).

**Processing steps:**
1. Load all TIFF pages with `cv::imreadmulti()` (line 65)
2. For each slice:
   - Convert BGR → Grayscale (line 80)
   - Apply Gaussian blur with sigma=1.5 (line 50)
   - Convert to CV_32F normalized to [0, 1] (line 43)
3. **Z-interpolation** (lines 94–107): Expand 33 slices to 225 using linear interpolation:
   - Original slices placed at indices: `0, z_scaling, 2×z_scaling, ...` (where z_scaling=7)
   - Between each pair, insert `z_scaling - 1 = 6` interpolated slices
   - Formula: `interpolated[k] = (1 - t) × slice[i] + t × slice[i+1]` where `t = k / z_scaling`
   - Result: 33 × 7 - 6 = **225 total z-slices**
4. Validate: throws error if result ≠ expected slice count (line 128)

---

## 3. How Cells Are Rendered (Synthetic Image)

### 3.1 Full Render — `Frame::generateSynthFrame()` in `C++/src/Frame.cpp` (lines 56–89)

Creates the entire synthetic image stack from scratch:

```
For each z-slice (0 to 224):
    Create blank image filled with background_color (e.g., 0.39)
    For each cell:
        cell.draw(image, config, nullptr, z)
```

### 3.2 `Spheroid::draw()` in `C++/src/Spheroid.cpp` (lines 113–157)

Renders one spheroid into one z-slice. This is a **per-pixel analytical test** — no voxel matrix is stored.

**Steps:**
1. **Early exit**: Skip if z-slice is beyond `maxR = max(majorR, majorR, minorR)` from cell center (line 129)
2. **Bounding box**: Compute 2D box as `center ± maxR` in x and y (lines 131–134)
3. **Per-pixel test** (lines 136–155):
   ```
   For each pixel (x, y) in bounding box:
       dx = x - center.x
       dy = y - center.y
       dz = z - center.z

       // Transform to cell's local coordinate frame
       (lx, ly, lz) = inverseRotate(dx, dy, dz)

       // Ellipsoid equation test
       val = (lx/a)² + (ly/b)² + (lz/c)²

       if val ≤ 1.0:
           pixel = cell_color  (e.g., 0.53)
   ```

The **inverse rotation** (`inverseRotatePoint()`, lines 19–42) applies:
`R_total^T = Rx(-θx) × Ry(-θy) × Rz(-θz)` to transform world coordinates into the spheroid's local frame where it is axis-aligned.

The spheroid shape is **oblate**: `a = b = majorRadius`, `c = minorRadius`. So:
- `a = b`: Equal equatorial radii
- `c`: Polar radius (shorter for oblate/pancake shape)

### 3.3 Fast Re-render — `Frame::generateSynthFrameFast()` in `Frame.cpp` (lines 115–154)

When only one cell changes (during perturbation), re-renders only the affected region:
1. Compute bounding box enclosing both old and new cell positions
2. Copy unchanged z-slices from existing synthetic frame
3. Re-render only z-slices within the bounding box

### 3.4 Cell Outlines — `Spheroid::drawOutline()` in `Spheroid.cpp` (lines 161–196)

Used when saving output images. Same as `draw()` but only marks **surface pixels** where `0.95 ≤ val ≤ 1.05` in green.

---

## 4. Cost Function

### 4.1 `Frame::calculateCost()` in `Frame.cpp` (lines 100–113)

Measures how well the synthetic image matches the real image:

```
totalCost = 0
For each z-slice i:
    totalCost += ||real[i] - synth[i]||_L2
```

Where `||·||_L2` is the L2 norm (sum of squared pixel differences, square-rooted). Lower cost = better fit.

---

## 5. Phase 1: Perturbation Optimization

### 5.1 `Lineage::optimize()` Phase 1 in `Lineage.cpp` (lines 179–209)

Runs `iterations_per_cell × num_cells` iterations of random perturbation.

**Each iteration calls `Frame::perturb()`** in `Frame.cpp` (lines 219–267):

1. **Select random cell**: `index = rand() % cells.size()` (line 227)
2. **Perturb**: `cells[index] = cells[index].getPerturbedCell()` (line 229)
3. **Overlap check**: If perturbed cell overlaps any other cell → revert immediately (lines 232–238)
4. **Render**: `generateSynthFrameFast(oldCell, newCell)` — only re-renders affected region (line 243)
5. **Cost**: `costDiff = newCost - oldCost` (line 244)
6. **Accept/reject** (lines 248–256):
   - **Accept** (costDiff < 0): Keep new cell position, update synthetic frame
   - **Reject** (costDiff ≥ 0): Revert cell to old state

### 5.2 `Spheroid::getPerturbedCell()` in `Spheroid.cpp` (lines 198–210)

Creates a copy with random offsets applied to every parameter:
- **Position**: `x ± N(mu_x, sigma_x)`, same for y, z
- **Radii**: `majorR ± N(mu_major, sigma_major)`, same for minorR
- **Rotation**: `θx ± N(mu_θx, sigma_θx)`, same for θy, θz

Each offset uses `PerturbParams::getPerturbOffset()`: with probability `prob`, sample from `N(mu, sigma)`; otherwise use `mu` (typically 0).

The new `Spheroid` constructor clamps radii to `[min, max]` bounds and enforces `minorR ≤ majorR`.

### 5.3 Overlap Detection — `Spheroid::checkIfCellsOverlap()` in `Spheroid.cpp` (lines 575–631)

For all pairs of cells `(i, j)`:
```
dist = ||center[i] - center[j]||
majorThresh = (majorR[i] + majorR[j]) × 0.95
minorThresh = (minorR[i] + minorR[j]) × 0.95

overlap if: dist < majorThresh AND dist < minorThresh
```

Both conditions must be true. The 0.95 factor allows slight tangency.

---

## 6. Phase 2: Split Detection

### 6.1 `Lineage::optimize()` Phase 2 in `Lineage.cpp` (lines 212–289)

After Phase 1 settles cells, Phase 2 checks if any cell should split into two daughters.

**Key design: independent evaluation.** Each cell is evaluated against the same baseline state. After evaluation, the split is always reverted so the next cell sees the original state.

```
candidates = []

For each cell (by name):
    result = frame.trySplitCell(cellIndex)
    costDiff = result.first

    if costDiff < -split_cost:
        Record daughters as SplitCandidate

    callback(false)  // Always revert to baseline

// Apply all accepted splits at once
For each candidate:
    Remove parent, add both daughters

if any splits applied:
    frame.regenerateSynthFrame()
    Run 2 × iterations_per_cell × num_splits post-split perturbations
```

### 6.2 `Frame::trySplitCell()` in `Frame.cpp` (lines 284–440)

The main split evaluation function. Takes a cell index and returns `{costDiff, callback}`.

#### Step A: Get Split Geometry (lines 284–310)

```cpp
// Collect neighbor cell centers (excluding the cell being split)
neighborCenters = [center of every other cell]

// Call PCA-based split detection
(child1, child2, valid, elongationRatio) = oldCell.getSplitCells(realFrame, z_scaling, neighborCenters)
```

If `valid == false` (too few bright pixels or constraint failure), logs `[Split Skip]` and returns.

#### Step A.5: Elongation Ratio Filter (lines 312–317)

Before the expensive burn-in, check whether the PCA elongation ratio indicates a likely split:

```
if splitElongationThreshold > 0.0 AND elongationRatio < splitElongationThreshold:
    Log "[Split Skip] <name> elongation_ratio=X < threshold=Y"
    Return {0.0, no-op}
```

- `elongationRatio ≈ 1.0`: Spherical bright pixels → no split signal → skip
- `elongationRatio > threshold (default 1.3)`: Elongated → proceed to burn-in
- Setting `split_elongation_threshold: 0` in config disables the filter (old behavior)

#### Step B: Overlap Checks (lines 305–358)

Remove parent, add both daughters to the cell list, then check:

**Daughter vs. existing cells** (lines 316–330):
```
For each existing cell i, for each daughter d:
    dist = ||center[i] - center[d]||
    majorThresh = (majorR[i] + majorR[d]) × 0.95
    minorThresh = (minorR[i] + minorR[d]) × 0.95

    if dist < majorThresh AND dist < minorThresh:
        OVERLAP → log "[Split Overlap] daughter c1/c2 overlaps with <cell>"
```

**Daughter vs. daughter**: *(removed)* — the cost function handles this naturally. If daughters stack or collapse, the synthetic image won't match the real data and costDiff will be positive, rejecting the split.

If daughter-existing overlap → revert and return `{0.0, no-op}`.

#### Step C: Burn-in Optimization (lines 360–416)

300 iterations of alternating perturbation on daughter 1 and daughter 2:

```
For iter = 0 to 499:
    dIdx = (iter even) ? daughter1 : daughter2

    saved = cells[dIdx]
    cells[dIdx] = cells[dIdx].getPerturbedCell()

    // Check perturbed daughter against existing cells
    if overlap with any existing cell → revert

    // Daughter-daughter check: removed (cost function decides)

    // Evaluate cost
    trialFrame = generateSynthFrameFast(saved, cells[dIdx])
    trialCost = calculateCost(trialFrame)

    if trialCost < bestCost:
        Accept perturbation
        accepted++
    else:
        Revert to saved
```

Logs: `[Split Burn-in] <name> burn_in_accepted=X/300 oldCost=Y newCost=Z diff=W`

#### Step D: Return Result (lines 418–439)

```
costDiff = bestCost - oldCost
callback = function(accept):
    if accept: keep daughters, update synth frame
    if reject: remove daughters, restore parent
return {costDiff, callback}
```

---

## 7. PCA Split Detection (getSplitCells)

### 7.1 `Spheroid::getSplitCells()` in `Spheroid.cpp` (lines 249–499)

This is the core algorithm that determines WHERE to place daughter cells. Returns `tuple<Spheroid, Spheroid, bool, float>` (daughters, validity, elongation ratio).

#### Step 1: Define Search Area (lines 253–265)

```cpp
maxR = max(majorRadius, majorRadius, minorRadius)  // = majorRadius for oblate
splitSearchRadius = maxR × 3.0                     // Generous bounding box

// PCA center: use pre-opt position when available (Phase 1 may shift
// the cell toward one blob, making PCA look spherical from that center)
pcaCenter = (preOptMajorR > 0) ? preOptPosition : currentPosition

// 3D bounding box centered at pcaCenter
minX = max(0, floor(pcaCenter.x - splitSearchRadius))
maxX = min(cols-1, ceil(pcaCenter.x + splitSearchRadius))
// Same for Y, Z
```

#### Step 2: Compute Mean Brightness (lines 276–295)

First pass: iterate over all pixels within the cell's own boundary (`val ≤ 1.0` using the ellipsoid equation) and compute the average brightness. This threshold separates "cell tissue" from "background."

```
For each pixel (x,y,z) in bounding box:
    Transform to local frame: (lx,ly,lz) = inverseRotate(dx,dy,dz)
    val = (lx/a)² + (ly/b)² + (lz/c)²
    if val ≤ 1.0:
        brightnessSum += real_image[z][y][x]
        count++

meanBrightness = brightnessSum / count
```

#### Step 3: Collect Bright Pixels (lines 297–344)

Second pass: collect all bright pixels within the bounding box. These are the input to PCA. No ellipsoidal boundary is applied — the bounding box (3×maxR) and neighbor exclusion naturally limit the area.

```
For each pixel (x,y,z) in bounding box:
    if pixel_brightness ≤ meanBrightness: skip (background)

    // Neighbor exclusion: skip if closer to another cell than to pcaCenter
    // (uses pre-opt position so distance is from original midpoint)
    distToSelf = (x-pcaCenter.x)² + (y-pcaCenter.y)² + (z-pcaCenter.z)²
    for each neighbor center:
        distToNeighbor = (x-nc.x)² + (y-nc.y)² + (z-nc.z)²
        if distToNeighbor < distToSelf: skip (belongs to neighbor)

    rawPoints.add(x, y, z)  // Store in image coordinates
```

**Why this matters:**
- The **brightness threshold** filters out dark background pixels, keeping only cell tissue
- The **bounding box** (3×maxR) sets the computational limit using pre-opt effective radii
- The **neighbor exclusion** prevents bright pixels from adjacent cells from contaminating the PCA

#### Step 4: PCA with Per-Axis Normalization (lines 346–426)

PCA (Principal Component Analysis) finds the direction of maximum spread in the point cloud. If the cell has split, the bright pixels form two clusters and PCA finds the axis connecting them.

```
// Center the data: subtract pcaCenter (pre-opt position when available)
For each point:
    data[i] = (point.x - pcaCenter.x, point.y - pcaCenter.y, point.z - pcaCenter.z)

// Compute per-axis standard deviation
sx = stddev of all x-values
sy = stddev of all y-values
sz = stddev of all z-values

// Normalize each axis to unit variance
// This prevents one axis from dominating PCA
// (e.g., z-slices may have different physical spacing)
data[i].x /= sx
data[i].y /= sy
data[i].z /= sz

// Run PCA on normalized data
pca = cv::PCA(data)
eigenvalues: λ1 ≥ λ2 ≥ λ3
eigenvector1 = direction of maximum variance (in normalized space)

// Transform eigenvector back to image space
ev_image = (ev_norm.x × sx, ev_norm.y × sy, ev_norm.z × sz)
split_axis = normalize(ev_image)

elongation_ratio = λ1 / λ2
```

**Interpreting the results:**
- `elongation_ratio ≈ 1.0`: Point cloud is spherical → no clear split direction
- `elongation_ratio > 1.5`: Point cloud is elongated → likely two clusters (split)
- `split_axis`: The direction connecting the two clusters

#### Step 5: Centroid-Based Daughter Placement (lines 428–487)

Split the bright pixels into two groups and compute their centroids:

```
// Volume conservation: each daughter has half the parent's volume
volumeScale = ∛(0.5) ≈ 0.794
effMajorR = max(currentMajorR, preOptMajorR)  // Use pre-Phase-1 size if larger
effMinorR = max(currentMinorR, preOptMinorR)
daughterMajorR = effMajorR × 0.794
daughterMinorR = effMinorR × 0.794

// Partition pixels by projection onto split axis
For each bright pixel point:
    projection = (point - center) · split_axis

    if projection ≥ 0:  → group1 (positive side)
    if projection < 0:  → group2 (negative side)

centroid1 = mean(group1 points)  // 3D average
centroid2 = mean(group2 points)  // 3D average
```

**Fallback**: If all pixels fall on one side, use fixed offset placement:
`centroid = center ± split_axis × (daughterMajorR × 0.5)`

#### Step 6: Create Daughter Spheroids (lines 490–498)

```
daughter0 = Spheroid(
    name = parent_name + "0",
    position = centroid1,
    majorRadius = daughterMajorR,
    minorRadius = daughterMinorR,
    rotation = parent rotation (θx, θy, θz)  // Inherited
)

daughter1 = Spheroid(
    name = parent_name + "1",
    position = centroid2,
    ...same radii and rotation...
)
```

Both daughters must pass `checkConstraints()` (radii within config bounds). If either fails, `valid = false`.

Logs: `[Split Placement] centroid-based: c1=(x,y,z) c2=(x,y,z) sep=D`

---

## 8. Post-Split & Frame Transition

### 8.1 Applying Splits — `Lineage::optimize()` (lines 262–289)

After all cells are evaluated independently:

```
For each accepted candidate:
    Find parent by name in cell list
    Remove parent
    Add daughter0 and daughter1
    Log "[Split Accepted]"

if any splits were applied:
    frame.regenerateSynthFrame()  // Full re-render with new daughters

    // Post-split perturbation: let daughters settle
    postIters = 2 × iterations_per_cell × num_splits
    For i in 0..postIters:
        frame.perturb()  // Same as Phase 1
```

### 8.2 Copy Forward — `Lineage::copyCellsForward()` (lines 376–384)

After optimizing frame N, copy all cells to frame N+1 as the initial guess:
```
frames[N+1].cells = frames[N].cells  // Deep copy
```

The next frame's Phase 1 will then optimize these positions for the new image.

### 8.3 Save Outputs — `Lineage::saveImages()` (lines 292–329) & `saveCells()` (lines 331–374)

**Images**: For each z-slice (0–224):
- `output/real/{frame}/0.png` ... `224.png` — Real image with green cell outlines
- `output/synth/{frame}/0.png` ... `224.png` — Synthetic rendering

**Cells**: Append to `output/cells.csv`:
```
file, name, x, y, z, majorRadius, minorRadius, theta_x, theta_y, theta_z
```

---

## 9. Complete Execution Flow

```
1. STARTUP
   main.cpp:147-207
   ├── Parse 6 command-line arguments
   ├── Load YAML config → BaseConfig
   ├── Auto-detect z-slices from TIFF metadata
   ├── Load image file paths for frame range
   ├── Create cells from initial CSV (CellFactory)
   └── Initialize Lineage with cells + images

2. FOR EACH FRAME (e.g., frames 1-10):

   2a. LOAD IMAGE (first time only)
       Lineage::loadFrame() [Lineage.cpp:55-135]
       ├── Read multi-page TIFF (33 z-slices)
       ├── Convert to grayscale, blur, normalize to [0,1]
       └── Z-interpolate: 33 → 225 slices (linear, z_scaling=7)

   2b. PHASE 1: PERTURBATION OPTIMIZATION
       Lineage::optimize() [Lineage.cpp:179-209]
       │
       │   iterations = num_cells × iterations_per_cell
       │   (e.g., 6 cells × 350 = 2,100 iterations)
       │
       └── For each iteration:
           Frame::perturb() [Frame.cpp:219-267]
           ├── Pick random cell
           ├── Apply random perturbation (position, radii, rotation)
           │   Spheroid::getPerturbedCell() [Spheroid.cpp:198-210]
           ├── Check overlaps
           │   Spheroid::checkIfCellsOverlap() [Spheroid.cpp:575-631]
           ├── Render changed region only
           │   Frame::generateSynthFrameFast() [Frame.cpp:115-154]
           │   └── Spheroid::draw() [Spheroid.cpp:113-157]
           ├── Compute cost (L2 norm)
           │   Frame::calculateCost() [Frame.cpp:100-113]
           └── Accept if cost decreased, else revert

   2c. PHASE 2: SPLIT DETECTION
       Lineage::optimize() [Lineage.cpp:212-289]
       │
       └── For each cell independently:
           Frame::trySplitCell() [Frame.cpp:278-440]
           │
           ├── PCA SPLIT DETECTION
           │   Spheroid::getSplitCells() [Spheroid.cpp:249-499]
           │   ├── Compute mean brightness inside cell boundary
           │   ├── Collect bright pixels in bounding box
           │   │   (3×maxR radius, brightness + neighbor filtering)
           │   ├── PCA with per-axis normalization
           │   │   → split_axis, elongation_ratio
           │   ├── Partition pixels by split axis → two groups
           │   ├── Compute 3D centroids → daughter positions
           │   └── Create daughter spheroids (0.794× parent radii)
           │
           ├── OVERLAP CHECKS
           │   ├── Each daughter vs. all existing cells
           │   │   (threshold: 0.95 × sum of radii)
           │   └── Daughter vs. daughter: removed (cost function decides)
           │
           ├── BURN-IN (300 iterations)
           │   ├── Alternate perturbing daughter 1 and daughter 2
           │   ├── Check overlaps after each perturbation
           │   ├── Accept only cost-reducing moves
           │   └── Track best configuration
           │
           ├── EVALUATE: costDiff = bestCost - preSplitCost
           │   if costDiff < -split_cost → add to candidates
           │
           └── ALWAYS REVERT (independent evaluation)

       Apply all accepted splits at once
       Post-split perturbation (2 × iters × num_splits)

   2d. SAVE & ADVANCE
       ├── Copy cells to next frame
       │   Lineage::copyCellsForward() [Lineage.cpp:376-384]
       ├── Save real+synth images as PNGs
       │   Lineage::saveImages() [Lineage.cpp:292-329]
       └── Append cell states to CSV
           Lineage::saveCells() [Lineage.cpp:331-374]
```

---

## 10. Key Constants & Thresholds

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `BURN_IN_ITERATIONS` | 500 | Frame.cpp:363 | Perturbations per daughter after split placement |
| `z_scaling` | 7 | config.yaml | Z-interpolation factor (33 → 225 slices) |
| Daughter volume scale | ∛0.5 ≈ 0.794 | Spheroid.cpp:436 | Each daughter has ~half the parent volume |
| Existing-cell overlap | 0.95 × sum | Frame.cpp:322 | Max overlap allowed with neighbors |
| Daughter-daughter min sep | *(removed)* | — | Removed: cost function handles this naturally |
| Split search radius | 3×maxR (effective) | Spheroid.cpp:360 | PCA pixel collection bounding box (uses pre-opt radii) |
| Expanded boundary | *(removed)* | — | Removed: bounding box + neighbor exclusion handle this |
| Split elongation threshold | 1.3 (config) | Frame.cpp:312 | PCA elongation ratio below which burn-in is skipped |
| Surface outline band | 0.95–1.05 | Spheroid.cpp:186 | Pixels drawn as cell outline |

---

## 11. File Reference

| File | Role |
|------|------|
| `C++/src/main.cpp` | Entry point, argument parsing, image loading |
| `C++/src/Lineage.cpp` | Frame management, Phase 1 + Phase 2 optimization, I/O |
| `C++/src/Frame.cpp` | Synthetic rendering, cost function, perturbation, split evaluation |
| `C++/src/Spheroid.cpp` | Cell geometry, drawing, PCA split detection, perturbation |
| `C++/src/CellFactory.cpp` | CSV parsing, cell initialization |
| `C++/includes/Lineage.hpp` | Lineage class declaration |
| `C++/includes/Frame.hpp` | Frame class declaration |
| `C++/includes/Spheroid.hpp` | Spheroid class + SpheroidParams declaration |
| `C++/includes/ConfigTypes.hpp` | All config structs, YAML parsing |
| `C++/includes/types.hpp` | Type aliases (Cost, CostCallbackPair, etc.) |
| `C++/config/config.yaml` | Runtime configuration |
| `C++/config/initial.csv` | Initial cell positions and shapes |
