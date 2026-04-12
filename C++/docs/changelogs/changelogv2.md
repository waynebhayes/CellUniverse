# Changelog

## 2026-02-21

### Added `boundingBoxScale` to `SphereConfig`

**File:** `C++/includes/ConfigTypes.hpp`

- **Line 123:** Added `double boundingBoxScale{1.0};` member to `SphereConfig` class

  ```cpp
  // Before: (no member)
  // After:
  double boundingBoxScale{1.0};
  ```
- **Lines 133-135:** Added optional YAML parsing in `explodeConfig()`:

  ```cpp
  // Before: (no parsing)
  // After:
  if (node["boundingBoxScale"]) {
      boundingBoxScale = node["boundingBoxScale"].as<double>();
  }
  ```

  (Note: user changed default from 1.5 to 1.0 after initial edit)

### Added `debug_log.txt` capture to run script

**File:** `C++/examples/run_celluniverse.sh`

- **Lines 476-494 (replaced lines 476-491):** Replaced the old stderr-only temp file approach with `tee`-based logging that captures both stdout and stderr to `$OUT_DIR/debug_log.txt` while still displaying on terminal. Added start/finish timestamps and command echo to the log. Added "Debug log:" path printout at end of run.

### Added output folders to `.gitignore`

**File:** `.gitignore`

- **Line 358-359:**

  ```
  # Before: (no entries)
  # After:
  C++/examples/output_*/
  C++/examples/user_input_configurations.ini
  ```

### Changed minMinorRadius: from 8 to 5 in config.yaml

**File:** `C++/examples/config.yaml`

```yaml
# Before:
  minMinorRadius: 8
# After:
  minMinorRadius: 5
```

### Frame numbering: use `firstFrame` offset for output and logs

**File:** `C++/includes/Lineage.hpp`

- **Line 28:** Added `int firstFrame = 0` parameter to constructor signature

  ```cpp
  // Before:
  Lineage(std::map<std::string, std::vector<Spheroid>> initialCells, PathVec imagePaths, BaseConfig &config, std::string outputPath, int continueFrom = -1);
  // After:
  Lineage(std::map<std::string, std::vector<Spheroid>> initialCells, PathVec imagePaths, BaseConfig &config, std::string outputPath, int firstFrame = 0, int continueFrom = -1);
  ```
- **Line 45:** Added `int firstFrame;` private member

**File:** `C++/src/Lineage.cpp`

- **Line 138-139:** Updated constructor signature and initializer list

  ```cpp
  // Before:
  Lineage::Lineage(std::map<std::string, std::vector<Spheroid>> initialCells, PathVec imagePaths, BaseConfig &config, std::string outputPath, int continueFrom)
      : config(config), outputPath(outputPath)
  // After:
  Lineage::Lineage(std::map<std::string, std::vector<Spheroid>> initialCells, PathVec imagePaths, BaseConfig &config, std::string outputPath, int firstFrame, int continueFrom)
      : config(config), outputPath(outputPath), firstFrame(firstFrame)
  ```
- **Line 183:** Added `int displayFrame = firstFrame + frameIndex;` local variable
- **Lines 183, 194, 197, 221, 242:** Changed all `frameIndex` references in debug output to `displayFrame`

  ```cpp
  // Before:
  std::cout << "Frame " << frameIndex << ", iteration " << i ...
  // After:
  std::cout << "Frame " << displayFrame << ", iteration " << i ...
  ```
- **Line 265:** Changed saving images log

  ```cpp
  // Before:
  std::cout << "Saving images for frame " << frameIndex << "..." << std::endl;
  // After:
  std::cout << "Saving images for frame " << displayFrame << "..." << std::endl;
  ```
- **Line 266-267:** Fixed bug: was indexing z-slices with frame index

  ```cpp
  // Before:
  std::cout << "Real Image Type: " << realImages[frameIndex].type() << std::endl;
  std::cout << "Synth Image Type: " << synthImages[frameIndex].type() << std::endl;
  // After:
  std::cout << "Real Image Type: " << realImages[0].type() << std::endl;
  std::cout << "Synth Image Type: " << synthImages[0].type() << std::endl;
  ```
- **Lines 269, 280:** Changed output folder naming

  ```cpp
  // Before:
  std::string realOutputPath = outputPath + "/real/" + std::to_string(frameIndex);
  std::string synthOutputPath = outputPath + "/synth/" + std::to_string(frameIndex);
  // After:
  std::string realOutputPath = outputPath + "/real/" + std::to_string(displayFrame);
  std::string synthOutputPath = outputPath + "/synth/" + std::to_string(displayFrame);
  ```
- **Line 335:** Changed cells saved log

  ```cpp
  // Before:
  std::cout << "Saved " << frame.cells.size() << " cells for frame " << frameIndex ...
  // After:
  std::cout << "Saved " << frame.cells.size() << " cells for frame " << (firstFrame + frameIndex) ...
  ```

**File:** `C++/src/main.cpp`

- **Line 190:** Added `args.firstFrame` argument to Lineage constructor call:

  ```cpp
  // Before:
  Lineage(cells, imageFilePaths, config, args.output, args.continueFrom)
  // After:
  Lineage(cells, imageFilePaths, config, args.output, args.firstFrame, args.continueFrom)
  ```

### Increased split burn-in iterations from 100 to 300

**File:** `C++/src/Frame.cpp`

- **Line 334:**

  ```cpp
  // Before:
  const int BURN_IN_ITERATIONS = 100;
  // After:
  const int BURN_IN_ITERATIONS = 300;
  ```

## 2026-02-22

### Independent split evaluation (no sequential dependency)

**File:** `C++/includes/Frame.hpp`

- **Line 41:** Added `regenerateSynthFrame()` public method

  ```cpp
  // Before: (no method)
  // After:
  void regenerateSynthFrame() { _synthFrame = generateSynthFrame(); }
  ```

**File:** `C++/src/Lineage.cpp`

- **Lines 211-282 (replaced lines 211-254):** Rewrote Phase 2 split detection to evaluate all cells independently against the same baseline state.

  ```cpp
  // Before: (sequential split evaluation — each split changes the state for the next)
  for (size_t i = 0; i < totalIterations; ++i) {
      auto result = frame.split();
      costDiff = result.first;
      std::function<void(bool)> accept = result.second;
      accept(costDiff < -config.prob.split_cost);
  }

  // After: (independent evaluation — evaluate all, revert each, apply all together)
  struct SplitCandidate {
      std::string parentName;
      Spheroid child1;
      Spheroid child2;
      double costDiff;
  };
  std::vector<SplitCandidate> candidates;

  for (const auto &name : cellNames) {
      // Find cell, evaluate split, record if accepted
      auto result = frame.trySplitCell(idx, preOptMajorR, preOptMinorR);
      costDiff = result.first;
      if (costDiff < -config.prob.split_cost) {
          candidates.push_back({name, child1, child2, costDiff});
      }
      callback(false); // Always revert — next cell sees same baseline
  }

  // Apply all accepted splits together
  for (const auto &candidate : candidates) {
      frame.cells.erase(parent);
      frame.cells.push_back(candidate.child1);
      frame.cells.push_back(candidate.child2);
  }
  if (!candidates.empty()) {
      frame.regenerateSynthFrame();
      // Post-split perturbation scales with number of splits
  }
  ```

### Added daughter-daughter overlap check to prevent false splits

**File:** `C++/src/Frame.cpp`

- **Lines 338-351 (inserted after existing-cell overlap check):** Daughters must be separated by at least `0.2 * (majorRadius1 + majorRadius2)`.

  ```cpp
  // Before: (no daughter-daughter check)
  // After:
  if (!overlapDetected) {
      cv::Point3f dd = cells[d1Idx].get_center() - cells[d2Idx].get_center();
      float ddDist = std::sqrt(dd.x * dd.x + dd.y * dd.y + dd.z * dd.z);
      auto p1 = cells[d1Idx].getCellParams();
      auto p2 = cells[d2Idx].getCellParams();
      float ddThresh = (p1.majorRadius + p2.majorRadius) * 0.2f;
      if (ddDist < ddThresh) {
          overlapDetected = true;
      }
  }
  ```

  (Initially set to 0.5, reduced to 0.2 because 0.5 was too strict.)
- **Lines 389-399 (inserted in burn-in loop):** Daughter-sibling overlap check during burn-in

  ```cpp
  // Before: (no sibling check during burn-in)
  // After:
  if (burnInValid) {
      size_t siblingIdx = (dIdx == d1Idx) ? d2Idx : d1Idx;
      cv::Point3f dd = cells[dIdx].get_center() - cells[siblingIdx].get_center();
      float ddDist = std::sqrt(dd.x * dd.x + dd.y * dd.y + dd.z * dd.z);
      auto pd = cells[dIdx].getCellParams();
      auto ps = cells[siblingIdx].getCellParams();
      if (ddDist < (pd.majorRadius + ps.majorRadius) * 0.2f) {
          burnInValid = false;
      }
  }
  ```

### Expanded PCA search area from 1.0x to 2.0x radius for split detection

**File:** `C++/src/Cells/Spheroid.cpp`

- **Lines 249-260 (replaced):** Replaced `calculateCorners()` bounding box with `2.0 * maxR` search radius

  ```cpp
  // Before:
  auto [min_corner, max_corner] = calculateCorners();
  int minX = std::max(0, (int)min_corner[0]);
  int maxX = std::min(image[0].cols - 1, (int)max_corner[0]);
  // ... (bounding box was only ±maxR)

  // After:
  float maxR = std::max({(float)a, (float)b, (float)c});
  float splitSearchRadius = maxR * 2.0f;
  int minX = std::max(0, static_cast<int>(std::floor(_position.x - splitSearchRadius)));
  int maxX = std::min(static_cast<int>(image[0].cols) - 1, static_cast<int>(std::ceil(_position.x + splitSearchRadius)));
  // ... (same for Y and Z)
  ```
- **Line 308:** Changed expanded boundary threshold

  ```cpp
  // Before:
  if (val <= 2.25) {  // 1.5²
  // After:
  if (val <= 4.0) {   // 2.0²
  ```

### Added debug prints for split overlap rejections and invalid splits

**File:** `C++/src/Frame.cpp`

- **Lines 300-302:** Added `[Split Skip]` debug print

  ```cpp
  // Before: (no debug print)
  // After:
  std::cout << "[Split Skip] " << oldCell.getCellParams().name
            << " getSplitCells returned invalid" << std::endl;
  ```
- **Lines 326-331:** Added `[Split Overlap]` debug print for daughter-existing overlap

  ```cpp
  // Before: (silent rejection)
  // After:
  std::cout << "[Split Overlap] " << oldCell.getCellParams().name
            << " daughter " << dLabel << " overlaps with " << pi.name
            << " dist=" << dist
            << " majorThresh=" << majorThresh
            << " minorThresh=" << minorThresh << std::endl;
  ```
- **Lines 347-349:** Added `[Split Overlap]` debug print for daughters too close

  ```cpp
  // Before: (silent rejection)
  // After:
  std::cout << "[Split Overlap] " << oldCell.getCellParams().name
            << " daughters too close: dist=" << ddDist
            << " thresh=" << ddThresh << std::endl;
  ```

### Added neighbor-cell filtering to PCA pixel collection (prevent contamination)

**File:** `C++/includes/Spheroid.hpp`

- **Line 111-113:** Added `neighborCenters` parameter to `getSplitCells`

  ```cpp
  // Before:
  std::tuple<Spheroid, Spheroid, bool> getSplitCells(const std::vector<cv::Mat> &image, float z_scaling) const;
  // After:
  std::tuple<Spheroid, Spheroid, bool> getSplitCells(const std::vector<cv::Mat> &image, float z_scaling,
      const std::vector<cv::Point3f> &neighborCenters = {},
      float preOptMajorR = 0.0f, float preOptMinorR = 0.0f) const;
  ```

**File:** `C++/src/Cells/Spheroid.cpp`

- **Lines 334-347 (inserted in second pass loop):** Skip pixels closer to neighbors

  ```cpp
  // Before: (no neighbor filtering)
  // After:
  float distSqToSelf = (float)(dx * dx + dy * dy + dz * dz);
  bool closerToNeighbor = false;
  for (const auto &nc : neighborCenters) {
      float ndx = (float)x - nc.x;
      float ndy = (float)y - nc.y;
      float ndz = (float)z - nc.z;
      float distSqToNeighbor = ndx * ndx + ndy * ndy + ndz * ndz;
      if (distSqToNeighbor < distSqToSelf) {
          closerToNeighbor = true;
          break;
      }
  }
  if (closerToNeighbor) continue;
  ```

**File:** `C++/src/Frame.cpp`

- **Lines 287-292 (inserted before `getSplitCells` call):** Collect neighbor centers

  ```cpp
  // Before: (no neighbor collection)
  // After:
  std::vector<cv::Point3f> neighborCenters;
  for (size_t i = 0; i < cells.size(); ++i) {
      if (i != index) {
          neighborCenters.push_back(cells[i].get_center());
      }
  }
  ```

### Added minimum PCA search radius of 50px

**File:** `C++/src/Cells/Spheroid.cpp`

- **Line 257:**

  ```cpp
  // Before:
  float splitSearchRadius = maxR * 2.0f;
  // After:
  float splitSearchRadius = std::max(maxR * 2.0f, 50.0f);
  ```

### Reverted spherical boundary back to ellipsoidal with minimum effective radius

**File:** `C++/src/Cells/Spheroid.cpp`

- **Lines 315-319 (replaced):** Used `maxR` for all axes

  ```cpp
  // Before: (per-axis ellipsoidal)
  double val = (lx * lx) / (a * a) + (ly * ly) / (b * b) + (lz * lz) / (c * c);
  // After:
  double val = (lx * lx + ly * ly + lz * lz) / (maxR * maxR);
  ```
- **Lines 325-328:** Reverted neighbor distance check to simple Euclidean (removed z_scaling).

## 2026-02-23

### Reverted: minimum 50px PCA search radius (was unnecessary)

**File:** `C++/src/Cells/Spheroid.cpp`

- **Line 258:**

  ```cpp
  // Before:
  float splitSearchRadius = std::max(maxR * 2.0f, 50.0f);
  // After:
  float splitSearchRadius = maxR * 2.0f;
  ```

  Pre-optimization boundary preservation handles the collapsed-cell case.

### Reverted: maxR-for-all-axes spherical boundary (was unnecessary)

**File:** `C++/src/Cells/Spheroid.cpp`

- **Line 332:**

  ```cpp
  // Before:
  double val = (lx * lx + ly * ly + lz * lz) / ((double)maxR * maxR);
  // After:
  double val = (lx * lx) / ((double)effA * effA) + (ly * ly) / ((double)effB * effB) + (lz * lz) / ((double)effC * effC);
  ```

  Now using pre-optimization effective radii instead.

### Added pre-optimization boundary preservation for split detection

Saves each cell's radii before Phase 1 optimization. When Phase 1 collapses an elongated/pancaked cell to a sphere, the PCA search boundary uses `max(current, preOpt)` radii so it never shrinks below the pre-optimization size.

**File:** `C++/includes/Spheroid.hpp`

- **Lines 111-113:** Added `preOptMajorR`, `preOptMinorR` parameters to `getSplitCells`

  ```cpp
  // Before:
  std::tuple<Spheroid, Spheroid, bool> getSplitCells(const std::vector<cv::Mat> &image, float z_scaling,
      const std::vector<cv::Point3f> &neighborCenters = {}) const;
  // After:
  std::tuple<Spheroid, Spheroid, bool> getSplitCells(const std::vector<cv::Mat> &image, float z_scaling,
      const std::vector<cv::Point3f> &neighborCenters = {},
      float preOptMajorR = 0.0f, float preOptMinorR = 0.0f) const;
  ```

**File:** `C++/src/Cells/Spheroid.cpp`

- **Lines 249-258:** Updated definition and compute effective radii

  ```cpp
  // Before:
  std::tuple<Spheroid, Spheroid, bool> Spheroid::getSplitCells(const std::vector<cv::Mat> &image, float z_scaling,
      const std::vector<cv::Point3f> &neighborCenters) const {
      float maxR = std::max({(float)a, (float)b, (float)c});
      float splitSearchRadius = maxR * 2.0f;

  // After:
  std::tuple<Spheroid, Spheroid, bool> Spheroid::getSplitCells(const std::vector<cv::Mat> &image, float z_scaling,
      const std::vector<cv::Point3f> &neighborCenters,
      float preOptMajorR, float preOptMinorR) const {
      float effA = (preOptMajorR > 0.0f) ? std::max((float)a, preOptMajorR) : (float)a;
      float effB = (preOptMajorR > 0.0f) ? std::max((float)b, preOptMajorR) : (float)b;
      float effC = (preOptMinorR > 0.0f) ? std::max((float)c, preOptMinorR) : (float)c;
      float maxR = std::max({effA, effB, effC});
      float splitSearchRadius = maxR * 2.0f;
  ```
- **Lines 267-273:** Added `[Split PreOpt]` debug print when effective radii differ from current

  ```cpp
  // Before: (no debug print)
  // After:
  if (preOptMajorR > 0.0f && (effA > (float)a || effC > (float)c)) {
      std::cout << "[Split PreOpt] " << _name
                << " current=(" << a << "," << b << "," << c << ")"
                << " preOpt=(" << preOptMajorR << "," << preOptMajorR << "," << preOptMinorR << ")"
                << " effective=(" << effA << "," << effB << "," << effC << ")"
                << " searchRadius=" << splitSearchRadius << std::endl;
  }
  ```
- **Line 332:** Boundary check uses effective radii

  ```cpp
  // Before:
  double val = (lx * lx) / (a * a) + (ly * ly) / (b * b) + (lz * lz) / (c * c);
  // After:
  double val = (lx * lx) / ((double)effA * effA) + (ly * ly) / ((double)effB * effB) + (lz * lz) / ((double)effC * effC);
  ```

**File:** `C++/includes/Frame.hpp`

- **Line 38:** Added `preOptMajorR`, `preOptMinorR` to `trySplitCell`

  ```cpp
  // Before:
  CostCallbackPair trySplitCell(size_t cellIndex);
  // After:
  CostCallbackPair trySplitCell(size_t cellIndex, float preOptMajorR = 0.0f, float preOptMinorR = 0.0f);
  ```

**File:** `C++/src/Frame.cpp`

- **Line 278:** Updated `trySplitCell` definition

  ```cpp
  // Before:
  CostCallbackPair Frame::trySplitCell(size_t index)
  // After:
  CostCallbackPair Frame::trySplitCell(size_t index, float preOptMajorR, float preOptMinorR)
  ```
- **Line 297:** Pass through to `getSplitCells`

  ```cpp
  // Before:
  std::tie(child1, child2, valid) = oldCell.getSplitCells(_realFrame, simulationConfig.z_scaling, neighborCenters);
  // After:
  std::tie(child1, child2, valid) = oldCell.getSplitCells(_realFrame, simulationConfig.z_scaling, neighborCenters, preOptMajorR, preOptMinorR);
  ```

**File:** `C++/src/Lineage.cpp`

- **Lines 179-191:** Save pre-optimization radii before Phase 1

  ```cpp
  // Before: (no pre-optimization saving)
  // After:
  struct PreOptShape {
      float majorR;
      float minorR;
  };
  std::map<std::string, PreOptShape> preOptShapes;
  for (const auto &cell : frame.cells) {
      auto params = cell.getCellParams();
      preOptShapes[params.name] = {(float)params.majorRadius, (float)params.minorRadius};
  }
  ```
- **Lines 259-267:** Look up pre-optimization radii in Phase 2

  ```cpp
  // Before:
  auto result = frame.trySplitCell(idx);
  // After:
  float preOptMajorR = 0.0f, preOptMinorR = 0.0f;
  auto it = preOptShapes.find(name);
  if (it != preOptShapes.end()) {
      preOptMajorR = it->second.majorR;
      preOptMinorR = it->second.minorR;
  }
  auto result = frame.trySplitCell(idx, preOptMajorR, preOptMinorR);
  ```

### Use pre-optimization radii for daughter cell sizes

**File:** `C++/src/Cells/Spheroid.cpp`

- **Lines 445-449:** Daughter radii use `max(current, preOpt)`

  ```cpp
  // Before:
  double volumeScale = std::cbrt(0.5);
  double daughterMajorRadius = _major_radius * volumeScale;
  double daughterMinorRadius = _minor_radius * volumeScale;

  // After:
  double effMajorR = (preOptMajorR > 0.0f) ? std::max(_major_radius, (double)preOptMajorR) : _major_radius;
  double effMinorR = (preOptMinorR > 0.0f) ? std::max(_minor_radius, (double)preOptMinorR) : _minor_radius;
  double volumeScale = std::cbrt(0.5);
  double daughterMajorRadius = effMajorR * volumeScale;
  double daughterMinorRadius = effMinorR * volumeScale;
  ```
- **Line 491:** Added `daughterMajorR` and `daughterMinorR` to `[Split Placement]` debug print

  ```cpp
  // Before:
  << " sep=" << sep << std::endl;
  // After:
  << " sep=" << sep
  << " daughterMajorR=" << daughterMajorRadius
  << " daughterMinorR=" << daughterMinorRadius << std::endl;
  ```

### Reverted: Added per-cell intensity parameter

Dim cells were being shrunk by Phase 1 to minimize L2 cost against the fixed-brightness synthetic rendering. Adding a per-cell intensity multiplier lets dim cells stay large while matching the real image brightness.

**File:** `C++/includes/Spheroid.hpp`

- **Line 34:** Added `float intensity` field to `SpheroidParams`

  ```cpp
  // Before: (no intensity field)
  // After:
  float intensity;
  ```
- **Lines 36-40:** Updated all constructors

  ```cpp
  // Before:
  SpheroidParams(const std::string &name, float x, float y, float z, float majorRadius, float minorRadius, float theta_x, float theta_y, float theta_z)
      : CellParams(name), x(x), y(y), z(z), majorRadius(majorRadius), minorRadius(minorRadius), theta_x(theta_x), theta_y(theta_y), theta_z(theta_z) {}
  // After:
  SpheroidParams(const std::string &name, float x, float y, float z, float majorRadius, float minorRadius, float theta_x, float theta_y, float theta_z, float intensity = 1.0f)
      : CellParams(name), x(x), y(y), z(z), majorRadius(majorRadius), minorRadius(minorRadius), theta_x(theta_x), theta_y(theta_y), theta_z(theta_z), intensity(intensity) {}
  ```
- **Line 83:** Added `double _intensity` member to `Spheroid`
- **Line 96:** Added `_intensity(1.0)` to default constructor
- **Line 99:** Added `intensity` to `printCellInfo()`

**File:** `C++/src/Cells/Spheroid.cpp`

- **Line 51:** Constructor init

  ```cpp
  // Before:
  _theta_z(init_props.theta_z),
  dormant(false)
  // After:
  _theta_z(init_props.theta_z),
  _intensity(init_props.intensity),
  dormant(false)
  ```
- **Lines 58-59:** Clamping

  ```cpp
  // Before: (no intensity clamping)
  // After:
  _intensity = std::fmax(_intensity, cellConfig.minIntensity);
  _intensity = std::fmin(_intensity, cellConfig.maxIntensity);
  ```
- **Line 155:** `draw()` intensity scaling

  ```cpp
  // Before:
  image.at<float>(j, i) = simulationConfig.cell_color;
  // After:
  float pixelValue = simulationConfig.background_color
      + (float)_intensity * (simulationConfig.cell_color - simulationConfig.background_color);
  image.at<float>(j, i) = pixelValue;
  ```
- **Line 214:** `getPerturbedCell()` perturbs intensity

  ```cpp
  // Before:
  _theta_z + cellConfig.thetaZ.getPerturbOffset());
  // After:
  _theta_z + cellConfig.thetaZ.getPerturbOffset(),
  _intensity + cellConfig.intensity.getPerturbOffset());
  ```
- **Line 253:** `getParameterizedCell()` passes `_intensity`

  ```cpp
  // Before:
  _theta_z + thetaZOffset);
  // After:
  _theta_z + thetaZOffset,
  _intensity);
  ```
- **Lines 512-515:** `getSplitCells()` daughters inherit `_intensity`

  ```cpp
  // Before:
  SpheroidParams(_name + "0", ..., _theta_x, _theta_y, _theta_z));
  // After:
  SpheroidParams(_name + "0", ..., _theta_x, _theta_y, _theta_z, _intensity));
  ```
- **Line 571:** `getCellParams()` includes `_intensity`

  ```cpp
  // Before:
  return SpheroidParams(_name, ..., _theta_x, _theta_y, _theta_z);
  // After:
  return SpheroidParams(_name, ..., _theta_x, _theta_y, _theta_z, _intensity);
  ```

**File:** `C++/includes/ConfigTypes.hpp`

- **Line 149-156:**

  ```cpp
  // Before: (no intensity config)
  // After:
  PerturbParams intensity{};
  // ...
  double minIntensity{0.3};
  double maxIntensity{1.5};
  ```
- **Lines 169-182:** YAML parsing

  ```cpp
  // Before: (no intensity parsing)
  // After:
  if (node["intensity"]) {
      intensity.explodeParams(node["intensity"]);
  }
  // ...
  if (node["minIntensity"]) {
      minIntensity = node["minIntensity"].as<double>();
  }
  if (node["maxIntensity"]) {
      maxIntensity = node["maxIntensity"].as<double>();
  }
  ```

**File:** `C++/examples/config.yaml`

- **Lines 40-49:**

  ```yaml
  # Before: (no intensity entries)
  # After:
    intensity:
      prob: 0.3
      mu: 0
      sigma: 0.05
    # ...
    minIntensity: 0.3
    maxIntensity: 1.5
  ```

**File:** `C++/src/Lineage.cpp`

- **Line 372:** CSV header

  ```cpp
  // Before:
  file << "file,name,x,y,z,majorRadius,minorRadius,theta_x,theta_y,theta_z" << std::endl;
  // After:
  file << "file,name,x,y,z,majorRadius,minorRadius,theta_x,theta_y,theta_z,intensity" << std::endl;
  ```
- **Line 391:** CSV output

  ```cpp
  // Before:
  << params.theta_z
  << std::endl;
  // After:
  << params.theta_z << ","
  << params.intensity
  << std::endl;
  ```

### Reverted: per-cell intensity parameter

The intensity feature caused cascading problems: daughters at default intensity 1.0 vs a dim parent created massive fake cost improvements (diffs of -117), causing every cell to split every frame. Even with daughter intensity inheritance, pre-optimization daughter sizing inflated daughters further (daughterMajorR=47.6), producing diffs of -10 in frame 1. Reverted all intensity changes across all files. All before/after are the reverse of the "Added per-cell intensity parameter" section above.

### Reverted: Added analytically computed per-cell brightness

Dim cells get shrunk by Phase 1 because the synthetic image renders all cells at fixed `cell_color`. When a real cell is dimmer, L2 cost rewards shrinking. Fix: compute mean real-image brightness within each cell's volume before optimization. Use that brightness during rendering so optimizer finds correct size. Output synth images still render at uniform `cell_color`. This is NOT a perturbable parameter — it is derived from the real image.

**File:** `C++/includes/Spheroid.hpp`

- **Line 83:** Added `double _brightness;` private member

  ```cpp
  // Before:
  double _theta_z;  // rotation angle about z-axis (radians)
  bool dormant;
  // After:
  double _theta_z;  // rotation angle about z-axis (radians)
  double _brightness; // per-cell brightness from real image (-1.0 = unset, use cell_color)
  bool dormant;
  ```
- **Line 96:** Added `_brightness(-1.0)` to default constructor

  ```cpp
  // Before:
  Spheroid() : _major_radius(0), _minor_radius(0), _rotation(0), _theta_x(0), _theta_y(0), _theta_z(0), dormant(false) {}
  // After:
  Spheroid() : _major_radius(0), _minor_radius(0), _rotation(0), _theta_x(0), _theta_y(0), _theta_z(0), _brightness(-1.0), dormant(false) {}
  ```
- **Lines 102-103:** Added public methods

  ```cpp
  // Before: (no brightness methods)
  // After:
  void computeBrightness(const std::vector<cv::Mat> &realFrame, const std::vector<double> &z_slices);
  void resetBrightness() { _brightness = -1.0; }
  ```

**File:** `C++/src/Cells/Spheroid.cpp`

- **Line 51:** Added `_brightness(-1.0)` to parameterized constructor initializer list

  ```cpp
  // Before:
  _theta_z(init_props.theta_z),
  dormant(false)
  // After:
  _theta_z(init_props.theta_z),
  _brightness(-1.0),
  dormant(false)
  ```
- **Lines 111-148:** Added `computeBrightness()` method (iterates over cell volume, computes mean real brightness using inverse rotation + ellipsoid test, same geometry as `draw()`)

  ```cpp
  // Before: (no method)
  // After:
  void Spheroid::computeBrightness(const std::vector<cv::Mat> &realFrame,
                                    const std::vector<double> &z_slices) {
      // ... iterates z-slices, bounding box, inverse rotation, ellipsoid test
      // accumulates realFrame[zi].at<float>(j, i) for pixels inside cell
      _brightness = (count > 0) ? (sum / count) : -1.0;
  }
  ```
- **Lines 192-193:** Modified `draw()` to use `_brightness`

  ```cpp
  // Before:
  image.at<float>(j, i) = simulationConfig.cell_color;
  // After:
  float renderColor = (_brightness > 0) ? (float)_brightness : simulationConfig.cell_color;
  image.at<float>(j, i) = renderColor;
  ```
- **Lines 250-252:** `getPerturbedCell()` copies `_brightness` to perturbed cell

  ```cpp
  // Before:
  return Spheroid(spheroidParams);
  // After:
  Spheroid result(spheroidParams);
  result._brightness = _brightness;
  return result;
  ```
- **Lines 289-291:** `getParameterizedCell()` copies `_brightness` to result

  ```cpp
  // Before:
  return Spheroid(spheroidParams);
  // After:
  Spheroid result(spheroidParams);
  result._brightness = _brightness;
  return result;
  ```
- **Lines 551-555:** `getSplitCells()` daughters inherit parent `_brightness`

  ```cpp
  // Before:
  Spheroid cell1(SpheroidParams(...));
  Spheroid cell2(SpheroidParams(...));
  // After:
  Spheroid cell1(SpheroidParams(...));
  cell1._brightness = _brightness;
  Spheroid cell2(SpheroidParams(...));
  cell2._brightness = _brightness;
  ```

**File:** `C++/includes/Frame.hpp`

- **Line 42:** Added `computeAllCellBrightnesses()` public method

  ```cpp
  // Before: (no method)
  // After:
  void computeAllCellBrightnesses();
  ```

**File:** `C++/src/Frame.cpp`

- **Lines 605-609:** Implemented `computeAllCellBrightnesses()`

  ```cpp
  // Before: (no method)
  // After:
  void Frame::computeAllCellBrightnesses() {
      for (auto &cell : cells) {
          cell.computeBrightness(_realFrame, z_slices);
      }
  }
  ```

**File:** `C++/src/Lineage.cpp`

- **Lines 193-197:** Compute brightness before Phase 1

  ```cpp
  // Before: (Phase 1 starts immediately after preOptShapes saving)
  // After:
  frame.computeAllCellBrightnesses();
  frame.regenerateSynthFrame();
  ```
- **Line 311:** Compute brightness after Phase 2 splits (before post-split perturbation)

  ```cpp
  // Before:
  frame.regenerateSynthFrame();
  // After:
  frame.computeAllCellBrightnesses();
  frame.regenerateSynthFrame();
  ```
- **Lines 328-334:** Reset brightness before saving output

  ```cpp
  // Before: (no brightness reset)
  // After:
  Frame &saveFrame = frames[frameIndex];
  for (auto &cell : saveFrame.cells) {
      cell.resetBrightness();
  }
  saveFrame.regenerateSynthFrame();
  ```

### Reverted: analytically computed per-cell brightness

The per-cell brightness feature weakened the L2 cost gradient at cell boundaries. Rendering at actual brightness (e.g. 0.46 vs cell_color 0.53) reduced the per-pixel penalty for growing into background from (0.53-0.39)²=0.0196 to (0.46-0.39)²=0.0049 — a 75% drop. This caused Phase 1 to grow ALL cells to maxRadius (60/30) because the optimizer faced almost no resistance. Oversized parents then produced oversized daughters, causing 3 false splits in frame 1. All brightness changes reverted across all files.

### Added real image normalization (per-cell) to eliminate brightness bias

**Problem**: Dim cells get shrunk by Phase 1 because synth renders at `cell_color` (0.53) but real cells may be dimmer (e.g. 0.46). L2 cost rewards shrinking. Previous approaches (per-cell `_intensity` and per-cell `_brightness` rendering) both failed because rendering at actual brightness weakened the L2 gradient at cell boundaries (75% drop).

**Fix**: Normalize the real image, not the synth. Before optimization, scale pixels inside each cell's volume in `_realFrame` so their mean matches `cell_color`. `_realFrameCopy` serves as the pristine backup. This preserves full L2 contrast at cell boundaries while eliminating brightness bias.

**File:** `C++/includes/Spheroid.hpp`

- **Lines 140-143:** Added `normalizeRegion()` declaration

  ```cpp
  // Before: (no method)
  // After:
  void normalizeRegion(const std::vector<cv::Mat> &source,
                       std::vector<cv::Mat> &target,
                       const std::vector<double> &z_slices,
                       float targetBrightness) const;
  ```

**File:** `C++/src/Cells/Spheroid.cpp`

- **Lines 98-169:** Implemented `normalizeRegion()` — two-pass approach using same inverse-rotation ellipsoid test as `draw()`. Pass 1 computes mean brightness from source (pristine `_realFrameCopy`). Pass 2 scales pixels in target (`_realFrame`) by `targetBrightness / meanBrightness`, clamped to 1.0.

  ```cpp
  // Before: (no method)
  // After:
  void Spheroid::normalizeRegion(const std::vector<cv::Mat> &source,
                                  std::vector<cv::Mat> &target,
                                  const std::vector<double> &z_slices,
                                  float targetBrightness) const {
      // ... two-pass: compute mean from source, scale pixels in target
      float scale = targetBrightness / meanBrightness;
      target[zi].at<float>(j, i) = std::min(pixel * scale, 1.0f);
  }
  ```

**File:** `C++/includes/Frame.hpp`

- **Lines 42-43:** Added `normalizeRealFrame()` and `restoreRealFrame()` declarations

  ```cpp
  // Before: (no methods)
  // After:
  void normalizeRealFrame();
  void restoreRealFrame();
  ```

**File:** `C++/src/Frame.cpp`

- **Lines 605-618:** Implemented `normalizeRealFrame()` and `restoreRealFrame()`

  ```cpp
  // Before: (no methods)
  // After:
  void Frame::normalizeRealFrame() {
      restoreRealFrame();
      for (const auto &cell : cells) {
          cell.normalizeRegion(_realFrameCopy, _realFrame, z_slices,
                               simulationConfig.cell_color);
      }
  }

  void Frame::restoreRealFrame() {
      for (size_t i = 0; i < _realFrameCopy.size(); ++i) {
          _realFrameCopy[i].copyTo(_realFrame[i]);
      }
  }
  ```

**File:** `C++/src/Lineage.cpp`

- **Lines 193-197:** Call `normalizeRealFrame()` before Phase 1 (after saving preOptShapes)

  ```cpp
  // Before: (Phase 1 starts immediately)
  // After:
  frame.normalizeRealFrame();
  frame.regenerateSynthFrame();
  ```
- **Line 311:** Call `normalizeRealFrame()` after Phase 2 splits (before post-split perturbation)

  ```cpp
  // Before:
  frame.regenerateSynthFrame();
  // After:
  frame.normalizeRealFrame();   // re-normalize with new cell positions
  frame.regenerateSynthFrame();
  ```
- **Lines 328-330:** Call `restoreRealFrame()` before saving output

  ```cpp
  // Before: (no restore)
  // After:
  frames[frameIndex].restoreRealFrame();
  frames[frameIndex].regenerateSynthFrame();  // synth at uniform cell_color for output
  ```

### Fixed: `_realFrameCopy` shallow copy bug (cv::Mat shares data)

`cv::Mat` copy constructor is shallow (reference counting). Both `_realFrameCopy` and `_realFrame` shared the same pixel buffers, so `normalizeRegion()` writing to `_realFrame` also corrupted `_realFrameCopy`. `restoreRealFrame()` then restored corrupted data, causing output images to show normalized brightness instead of original.

**File:** `C++/src/Frame.cpp`

- **Lines 25-31:** Replaced shallow member-initializer copy with explicit `.clone()` in constructor body

  ```cpp
  // Before:
      _realFrameCopy(realFrame), // Make a copy of the input image stack
      _realFrame(realFrame)
  {

  // After:
      _realFrame(realFrame)
  {
      _realFrameCopy.reserve(realFrame.size());
      for (const auto &mat : realFrame) {
          _realFrameCopy.push_back(mat.clone());
      }
  ```

### Increased daughter-daughter overlap threshold from 0.2 to 0.5

Daughters with parent-sized radii were stacking in z to mimic the parent volume. With 0.2, two majorR≈38 daughters only 17px apart passed (17 > 0.2×76=15.2). At 0.5, they're rejected (17 < 0.5×76=38). Legitimate splits (sep≈43, radii≈22) still pass easily (43 > 0.5×44=22).

**File:** `C++/src/Frame.cpp`

- **Line 350:** Initial daughter-daughter overlap check

  ```cpp
  // Before:
  float ddThresh = (p1.majorRadius + p2.majorRadius) * 0.2f;
  // After:
  float ddThresh = (p1.majorRadius + p2.majorRadius) * 0.5f;
  ```
- **Line 402:** Burn-in sibling overlap check

  ```cpp
  // Before:
  if (ddDist < (pd.majorRadius + ps.majorRadius) * 0.2f) {
  // After:
  if (ddDist < (pd.majorRadius + ps.majorRadius) * 0.5f) {
  ```

### Added brightness threshold to normalizeRegion (exclude background pixels)

**File:** `C++/includes/Spheroid.hpp`

- **Lines 140-144:** Added `backgroundBrightness` parameter

  ```cpp
  // Before:
  void normalizeRegion(..., float targetBrightness) const;
  // After:
  void normalizeRegion(..., float targetBrightness, float backgroundBrightness) const;
  ```

**File:** `C++/src/Cells/Spheroid.cpp`

- **Lines 98-182:** Updated implementation to skip pixels below `(background + target) / 2` threshold in both passes. Only actual cell tissue gets scaled, preventing bright halos from anchoring cells.

**File:** `C++/src/Frame.cpp`

- **Lines 612-613:** Pass `background_color` to `normalizeRegion`

  ```cpp
  cell.normalizeRegion(_realFrameCopy, _realFrame, z_slices,
                       simulationConfig.cell_color, simulationConfig.background_color);
  ```

### Reverted: real image normalization (per-cell)

Normalization weakened the optimizer (733 accepted perturbations vs 1545 without), caused false splits where daughters stacked in z to mimic the parent, and froze cells across frames. Even with the brightness threshold fix, the fundamental issue remained: normalizing pixels inside the model ellipsoid couples the cost function to the current model position, creating stale artifacts as cells move. Three approaches to the dim cell problem have now been tried and reverted: perturbable `_intensity`, analytically computed `_brightness` rendering, and real image normalization. All reverted from `Spheroid.hpp`, `Spheroid.cpp`, `Frame.hpp`, `Frame.cpp`, and `Lineage.cpp`. The deep-clone fix for `_realFrameCopy` and the 0.5 daughter-daughter overlap threshold are kept as independent improvements.

## 2026-02-26

### Fixed: `firstFrame` self-initialization bug (merge artifact)

The merge of `yuancen_pu_22_02_2026` into `merge_jh_yp_02_24_2026` removed the `int firstFrame` parameter from the Lineage constructor but left `firstFrame(firstFrame)` in the member initializer list. This caused the member to be initialized with its own uninitialized value (undefined behavior), producing garbage frame numbers in all output paths and logs.

**File:** `C++/includes/Lineage.hpp`

- **Line 28-32:** Restored `int firstFrame = 0` parameter

  ```cpp
  // Before (broken — missing firstFrame parameter):
  Lineage(std::map<std::string, std::vector<Spheroid>> initialCells,
          PathVec imagePaths, BaseConfig &config, std::string outputPath,
          int continueFrom = -1);
  // After:
  Lineage(std::map<std::string, std::vector<Spheroid>> initialCells,
          PathVec imagePaths, BaseConfig &config, std::string outputPath,
          int firstFrame = 0, int continueFrom = -1);
  ```

**File:** `C++/src/Lineage.cpp`

- **Lines 142-147:** Restored `int firstFrame` parameter in constructor definition

  ```cpp
  // Before (broken):
  Lineage::Lineage(..., int continueFrom)
      : config(config), outputPath(outputPath), firstFrame(firstFrame)  // self-init!
  // After:
  Lineage::Lineage(..., int firstFrame, int continueFrom)
      : config(config), outputPath(outputPath), firstFrame(firstFrame)  // now reads parameter
  ```

**File:** `C++/src/main.cpp`

- **Line 217:** Restored `args.firstFrame` argument to Lineage constructor call

  ```cpp
  // Before (broken — firstFrame not passed):
  Lineage lineage = Lineage(cells, imageFilePaths, config, args.output, args.continueFrom);
  // After:
  Lineage lineage = Lineage(cells, imageFilePaths, config, args.output, args.firstFrame, args.continueFrom);
  ```

### Fixed: Pre-optimization radii inflating daughter sizes, blocking all splits

The pre-optimization boundary preservation (added 2026-02-23) used `max(current, preOpt)` radii for BOTH the PCA search boundary AND daughter cell sizing. Using inflated radii for daughter sizing made daughters too large, causing the daughter-daughter overlap threshold (`0.5 × sum majorR`) to reject all valid splits.

**Example (cell `12345679...` in frame 2):**
- Phase 1 collapsed majorR from 29.3 → 21.8
- With pre-opt sizing: daughterMajorR = 29.3 × 0.794 = 23.2, threshold = 23.2, separation = 22.7 → REJECTED
- With current sizing: daughterMajorR = 21.8 × 0.794 = 17.3, threshold = 17.3, separation = 22.7 → PASSES

Fix: use pre-opt radii only for the PCA search boundary (correct — prevents shrinkage), use current radii for daughter sizing.

**File:** `C++/src/Cells/Spheroid.cpp`

- **Lines 554-558:** Changed daughter sizing to use current radii

  ```cpp
  // Before (inflated daughter sizes):
  double effMajorR = (preOptMajorR > 0.0f) ? std::max(_major_radius, (double)preOptMajorR) : _major_radius;
  double effMinorR = (preOptMinorR > 0.0f) ? std::max(_minor_radius, (double)preOptMinorR) : _minor_radius;
  double volumeScale = std::cbrt(0.5);
  double daughterMajorRadius = effMajorR * volumeScale;
  double daughterMinorRadius = effMinorR * volumeScale;

  // After (current radii for daughters):
  double volumeScale = std::cbrt(0.5);
  double daughterMajorRadius = _major_radius * volumeScale;
  double daughterMinorRadius = _minor_radius * volumeScale;
  ```

## 2026-02-27

### Removed daughter-daughter overlap hard gate (let cost function decide)

The daughter-daughter overlap threshold (initially 0.2, raised to 0.5, lowered to 0.3) was blocking legitimate splits even at 0.3. The centroid-based daughter placement often produces daughters closer than the threshold, especially when Phase 1 collapses cells to spheres. The cost function (L2 real vs synth) already handles this: if daughters are placed badly (stacking, collapsing), the synthetic image won't match the real data, costDiff will be positive, and the split is naturally rejected.

**File:** `C++/src/Frame.cpp`

- **Lines 344-357 (removed):** Removed initial daughter-daughter overlap check before burn-in

  ```cpp
  // Before:
  if (!overlapDetected) {
      cv::Point3f dd = cells[d1Idx].get_center() - cells[d2Idx].get_center();
      float ddDist = std::sqrt(dd.x * dd.x + dd.y * dd.y + dd.z * dd.z);
      auto p1 = cells[d1Idx].getCellParams();
      auto p2 = cells[d2Idx].getCellParams();
      float ddThresh = (p1.majorRadius + p2.majorRadius) * 0.3f;
      if (ddDist < ddThresh) {
          overlapDetected = true;
          // ... log ...
      }
  }

  // After: (removed entirely — cost function decides)
  // Daughter-daughter overlap check removed — let the cost function decide.
  // If daughters are placed badly, burn-in won't improve the fit and
  // costDiff will be positive, naturally rejecting the split.
  ```

- **Lines 395-404 (removed):** Removed daughter-daughter check inside burn-in loop

  ```cpp
  // Before:
  if (burnInValid) {
      size_t siblingIdx = (dIdx == d1Idx) ? d2Idx : d1Idx;
      cv::Point3f dd = cells[dIdx].get_center() - cells[siblingIdx].get_center();
      float ddDist = std::sqrt(dd.x * dd.x + dd.y * dd.y + dd.z * dd.z);
      auto pd = cells[dIdx].getCellParams();
      auto ps = cells[siblingIdx].getCellParams();
      if (ddDist < (pd.majorRadius + ps.majorRadius) * 0.3f) {
          burnInValid = false;
      }
  }

  // After: (removed entirely — optimizer explores freely, cost rejects bad configs)
  ```

### Relaxed existing-cell overlap threshold in burn-in from 0.95 to 0.8

Newly born daughters may legitimately start near existing neighbors. The strict 0.95 threshold during burn-in prevented daughters from optimizing their positions near neighbors. Relaxed to 0.8 during burn-in only (Phase 1 perturbation keeps 0.95).

**File:** `C++/src/Frame.cpp`

- **Lines 378-379:** Changed burn-in existing-cell overlap threshold

  ```cpp
  // Before:
  if (dist < (pi.majorRadius + pd.majorRadius) * 0.95f &&
      dist < (pi.minorRadius + pd.minorRadius) * 0.95f) {

  // After:
  if (dist < (pi.majorRadius + pd.majorRadius) * 0.8f &&
      dist < (pi.minorRadius + pd.minorRadius) * 0.8f) {
  ```

### Increased burn-in iterations from 300 to 500

With hard geometric constraints removed, the optimizer needs more iterations to find good daughter positions. Increased from 300 to 500 to compensate.

**File:** `C++/src/Frame.cpp`

- **Line 363:**

  ```cpp
  // Before:
  const int BURN_IN_ITERATIONS = 300;
  // After:
  const int BURN_IN_ITERATIONS = 500;
  ```

### Removed ellipsoidal pixel inclusion limit from PCA split detection

The second pass in `getSplitCells()` used `val <= 4.0` (an ellipsoidal boundary at ~2× radius) to filter which bright pixels are collected for PCA. This had a critical inconsistency: the bounding box (line 360) used pre-opt effective radii, but the ellipsoidal check used current radii (`invA2, invB2, invC2`). If Phase 1 collapsed a cell from radius 28→15, the bounding box extended to 2×28=56 but the ellipsoidal gate only included pixels within 2×15=30 — defeating the pre-opt boundary preservation.

Removed the ellipsoidal gate entirely. The bounding box and neighbor exclusion naturally limit the search area. Brightness thresholding already filters out background pixels.

**File:** `C++/src/Cells/Spheroid.cpp`

- **Lines 421-430:** Removed `expandedThresh` constant and `val <= expandedThresh` condition

  ```cpp
  // Before:
  // Expanded ellipsoidal boundary: val <= 4.0 means ~2.0x the radius
  // Uses effective radii (max of current and pre-optimization) so that
  // Phase 1 collapsing a cell doesn't shrink the search area.
  const double expandedThresh = 4.0;

  scanSpheroidVolume(
      image, minX, maxX, minY, maxY, minZ, maxZ, _position,
      R_T, invA2, invB2, invC2,
      [&](double dx, double dy, double dz, int x, int y, int z, float pixel, double val) {
          if (pixel > meanBrightness && val <= expandedThresh) {

  // After:
  // No ellipsoidal boundary — the bounding box (3×maxR) and neighbor
  // exclusion naturally limit the search area. Removing the ellipsoidal
  // gate fixes the inconsistency where the bounding box used pre-opt
  // radii but the ellipsoid check used current (possibly collapsed) radii.

  scanSpheroidVolume(
      image, minX, maxX, minY, maxY, minZ, maxZ, _position,
      R_T, invA2, invB2, invC2,
      [&](double dx, double dy, double dz, int x, int y, int z, float pixel, double /*val*/) {
          if (pixel > meanBrightness) {
  ```

### Increased split search radius from 2× to 3×

With the ellipsoidal gate removed, the bounding box is the only spatial limit on PCA pixel collection. Increased from 2×maxR to 3×maxR so daughter blobs further from the parent center are captured. The neighbor exclusion check prevents contamination from other cells.

**File:** `C++/src/Cells/Spheroid.cpp`

- **Line 360:**

  ```cpp
  // Before:
  float splitSearchRadius = maxR * 2.0f;
  // After:
  float splitSearchRadius = maxR * 3.0f;
  ```

### Restored pre-opt radii for daughter sizing

Phase 1 collapses the parent cell (e.g. 28→21 radius) when fitting one sphere to a two-cell real image. Daughters sized from the collapsed parent (`21.3 × 0.794 = 16.88`) are too small to cover the real cell mass, resulting in near-zero cost improvement (e.g. diff=-0.033 for e9077 in frame 3). Using pre-opt radii gives `28.3 × 0.794 = 22.5` — much closer to the true daughter size.

This was previously blocked by the daughter-daughter overlap hard gate (inflated daughters couldn't pass the threshold). Now that the gate is removed, pre-opt sizing works safely — bad splits are rejected by the cost function.

**File:** `C++/src/Cells/Spheroid.cpp`

- **Lines 554-560:** Changed daughter sizing from current radii to max(current, pre-opt)

  ```cpp
  // Before:
  // Use CURRENT radii for daughter sizing (not inflated pre-opt radii).
  // Pre-opt radii are used only for the PCA search boundary above.
  // Using pre-opt radii here inflated daughters, making the daughter-daughter
  // overlap threshold (0.5 × sum majorR) impossible to pass.
  double volumeScale = std::cbrt(0.5);
  double daughterMajorRadius = _major_radius * volumeScale;
  double daughterMinorRadius = _minor_radius * volumeScale;

  // After:
  // Use pre-opt radii for daughter sizing when available. Phase 1 may
  // collapse the parent (e.g. 28→21) when fitting one sphere to a two-cell
  // image, making current-radius daughters too small for meaningful cost
  // improvement. Pre-opt radii reflect the cell size before collapse.
  // (Previously this was blocked by daughter-daughter overlap gates, now removed.)
  double effMajorR = (preOptMajorR > 0.0f) ? std::max(_major_radius, (double)preOptMajorR) : _major_radius;
  double effMinorR = (preOptMinorR > 0.0f) ? std::max(_minor_radius, (double)preOptMinorR) : _minor_radius;
  double volumeScale = std::cbrt(0.5);
  double daughterMajorRadius = effMajorR * volumeScale;
  double daughterMinorRadius = effMinorR * volumeScale;
  ```

### Added then reverted: perturbable per-cell `_brightness` parameter

Dim cells shrink to minimum radii because the fixed `cell_color` (0.53) doesn't match their real brightness (~0.45). Added per-cell brightness as a slow, tightly-bounded perturbable parameter (bounds [0.43, 0.63], sigma 0.02, prob 0.15). **Reverted** — same root cause as previous brightness attempts: when brightness drifts lower (e.g. 0.47), the background penalty drops from |0.53-0.39|²=0.0196 to |0.47-0.39|²=0.0064 (67% reduction), causing excessive false splits. This is the 4th failed approach to the dim cell problem. All code below was added then reverted.

**File:** `C++/includes/ConfigTypes.hpp`

- **Line 150:** Added `PerturbParams brightness{}` to SpheroidConfig

  ```cpp
  // Before:
  PerturbParams thetaZ{};
  double minMajorRadius{};
  // After:
  PerturbParams thetaZ{};
  PerturbParams brightness{};
  double minMajorRadius{};
  ```
- **Lines 155-156:** Added `minBrightness` and `maxBrightness` bounds

  ```cpp
  // Before:
  double maxMinorRadius{};
  ~SpheroidConfig() = default;
  // After:
  double maxMinorRadius{};
  double minBrightness{};
  double maxBrightness{};
  ~SpheroidConfig() = default;
  ```
- **Line 167:** Added brightness parsing in `explodeConfig()`

  ```cpp
  // Before:
  thetaZ.explodeParams(node["thetaZ"]);
  // After:
  thetaZ.explodeParams(node["thetaZ"]);
  brightness.explodeParams(node["brightness"]);
  ```
- **Lines 174-175:** Added brightness bounds parsing

  ```cpp
  // Before:
  maxMinorRadius = node["maxMinorRadius"].as<double>();
  // After:
  maxMinorRadius = node["maxMinorRadius"].as<double>();
  minBrightness = node["minBrightness"].as<double>();
  maxBrightness = node["maxBrightness"].as<double>();
  ```

**File:** `C++/includes/Spheroid.hpp`

- **Line 36:** Added `float brightness` to SpheroidParams

  ```cpp
  // Before:
  float theta_z; // rotation about z-axis (radians)
  // After:
  float theta_z; // rotation about z-axis (radians)
  float brightness;
  ```
- **Lines 37-41:** Updated all SpheroidParams constructors to include brightness (default 0.53f)

  ```cpp
  // Before:
  SpheroidParams() : CellParams(""), x(0), ..., theta_z(0) {}
  SpheroidParams(..., float theta_z)
      : CellParams(name), ..., theta_z(theta_z) {}
  // After:
  SpheroidParams() : CellParams(""), ..., theta_z(0), brightness(0.53f) {}
  SpheroidParams(..., float theta_z, float brightness = 0.53f)
      : CellParams(name), ..., theta_z(theta_z), brightness(brightness) {}
  ```
- **Line 86:** Added `double _brightness` private member

  ```cpp
  // Before:
  double _theta_z;
  bool dormant;
  // After:
  double _theta_z;
  double _brightness;
  bool dormant;
  ```
- **Line 104:** Added `_brightness(0.53)` to default constructor

  ```cpp
  // Before:
  Spheroid() : ..., _theta_z(0), dormant(false) {}
  // After:
  Spheroid() : ..., _theta_z(0), _brightness(0.53), dormant(false) {}
  ```
- **Line 106:** Added brightness to `printCellInfo()`

  ```cpp
  // Before:
  << " theta_z: " << _theta_z << " isDormant: " << dormant
  // After:
  << " theta_z: " << _theta_z << " brightness: " << _brightness << " isDormant: " << dormant
  ```

**File:** `C++/src/Cells/Spheroid.cpp`

- **Line 160:** Added `_brightness(init_props.brightness)` to constructor initializer list

  ```cpp
  // Before:
  _theta_z(init_props.theta_z),
  dormant(false)
  // After:
  _theta_z(init_props.theta_z),
  _brightness(init_props.brightness),
  dormant(false)
  ```
- **Lines 168-169:** Added brightness clamping in constructor body

  ```cpp
  // Before: (no brightness clamping)
  // After:
  _brightness = std::fmax(_brightness, cellConfig.minBrightness);
  _brightness = std::fmin(_brightness, cellConfig.maxBrightness);
  ```
- **Line 254:** Changed `draw()` to use `_brightness` instead of `simulationConfig.cell_color`

  ```cpp
  // Before:
  image.at<float>(y, x) = simulationConfig.cell_color;
  // After:
  image.at<float>(y, x) = static_cast<float>(_brightness);
  ```
- **Line 310:** Added brightness perturbation to `getPerturbedCell()`

  ```cpp
  // Before:
  _theta_z + cellConfig.thetaZ.getPerturbOffset());
  // After:
  _theta_z + cellConfig.thetaZ.getPerturbOffset(),
  _brightness + cellConfig.brightness.getPerturbOffset());
  ```
- **Lines 314-346:** Added brightness offset to `getParameterizedCell()`

  ```cpp
  // Before: (no brightness offset)
  // After:
  float brightnessOffset = params["brightness"];
  // ... in empty-params fallback:
  brightnessOffset = Spheroid::cellConfig.brightness.getPerturbOffset();
  // ... clamping:
  float newBrightness = fmin(fmax(Spheroid::cellConfig.minBrightness, _brightness + brightnessOffset), Spheroid::cellConfig.maxBrightness);
  // ... passed to SpheroidParams constructor
  ```
- **Line 672:** Added brightness bounds to `checkConstraints()`

  ```cpp
  // Before:
  return ... && (_minor_radius <= cellConfig.maxMinorRadius);
  // After:
  return ... && (_minor_radius <= cellConfig.maxMinorRadius)
      && (cellConfig.minBrightness <= _brightness) && (_brightness <= cellConfig.maxBrightness);
  ```
- **Line 676:** Added brightness to `getCellParams()` return

  ```cpp
  // Before:
  return SpheroidParams(_name, ..., _theta_z);
  // After:
  return SpheroidParams(_name, ..., _theta_z, _brightness);
  ```
- **Lines 622-627:** Daughters inherit parent brightness in `getSplitCells()`

  ```cpp
  // Before:
  ..., _theta_x, _theta_y, _theta_z));
  // After:
  ..., _theta_x, _theta_y, _theta_z, _brightness));
  ```

**File:** `C++/src/Lineage.cpp`

- **Line 381:** Updated CSV header

  ```cpp
  // Before:
  file << "file,name,x,y,z,majorRadius,minorRadius,theta_x,theta_y,theta_z" << std::endl;
  // After:
  file << "file,name,x,y,z,majorRadius,minorRadius,theta_x,theta_y,theta_z,brightness" << std::endl;
  ```
- **Lines 399-400:** Added brightness to CSV output

  ```cpp
  // Before:
  << params.theta_z
  << std::endl;
  // After:
  << params.theta_z << ","
  << params.brightness
  << std::endl;
  ```

**File:** `C++/examples/config.yaml`

- **Lines 40-44:** Added brightness perturbation section

  ```yaml
  # Before: (no brightness section)
  # After:
    brightness:
      prob: 0.15
      mu: 0
      sigma: 0.02
  ```
- **Lines 48-49:** Added brightness bounds

  ```yaml
  # Before: (no brightness bounds)
  # After:
    minBrightness: 0.43
    maxBrightness: 0.63
  ```

## 2026-03-02

### Added iterative centroid refinement for PCA center

If Phase 1 of the previous frame dragged the cell toward one blob, pcaCenter (pre-opt position) lands on that blob. The neighbor exclusion then clips the far blob's pixels (closer to a neighbor than to the biased pcaCenter), reducing the PCA elongation ratio. For `1f2ed...` in frame 11, this dropped elongation from what should be ~10+ to only 4.6.

After the initial bright pixel collection, compute the centroid of collected pixels. If it differs from pcaCenter by >5 pixels, re-collect using the centroid as the new center. This shifts the Voronoi boundary between the cell and its neighbors to capture both blobs more symmetrically.

**File:** `C++/src/Cells/Spheroid.cpp`
- **Lines 472-522 (new Step 2b):** Added centroid computation after first bright pixel collection, drift check (>5px threshold), and re-collection scan with updated pcaCenter. Logs `[PCA Recenter]` when triggered.

---

### Changed PCA normalization from per-axis to isotropic

Per-axis normalization (dividing x by effA, y by effB, z by effC) suppressed x-y splits in pancaked cells. The small z-radius (minorR) amplified z-noise, inflating all eigenvalues while keeping the ratio low. For `1f2ed...` in frame 11 (majorR=32, minorR=12), elongation ratio was only 3.15 despite clear visual elongation.

Changed to isotropic normalization: all axes divided by `majorR` (effA). This preserves the true shape of the bright pixel distribution regardless of cell aspect ratio.

**File:** `C++/src/Cells/Spheroid.cpp`
- **Lines 496-506:** Replaced per-axis `invSx/invSy/invSz` with single `invS = 1/normR` applied to all axes
- **Lines 521-524:** Transform-back uses `normR` for all axes instead of per-axis radii
- **Line 545:** Log prints single `normR` instead of per-axis `normRadii`

---

### Fixed stale `_synthFrame` after `copyCellsForward`

The Frame constructor generates `_synthFrame` with whatever cells it receives. For frames after the initial one, the Lineage constructor passes empty cells, so `_synthFrame` = background-only. `copyCellsForward()` copies cells but does NOT regenerate `_synthFrame`. This means Phase 1 starts with a stale background-only baseline, causing the first perturbation of any cell to be accepted regardless of direction (any cell drawing beats background). For dim cells already biased toward shrinking, this amplifies the collapse.

**File:** `C++/src/Lineage.cpp`
- **Line 182 (after):** Added `frame.regenerateSynthFrame()` at the start of `optimize()`, right after getting the frame reference.

```cpp
// Before:
    Frame &frame = frames[frameIndex];
    size_t totalIterations = frame.length() * config.simulation.iterations_per_cell;

// After:
    Frame &frame = frames[frameIndex];
    frame.regenerateSynthFrame();
    size_t totalIterations = frame.length() * config.simulation.iterations_per_cell;
```

---

### Added pre-opt position for PCA center in split detection

Phase 1 shifts the cell center toward one blob, making PCA from the shifted center see a spherical distribution (low elongation ratio) even when two blobs exist. Now `getSplitCells` accepts pre-opt position (`preOptX/Y/Z`) and uses it as `pcaCenter` — the center for bounding box, neighbor exclusion, PCA centering, pixel projection, and fallback placement. This ensures PCA sees both blobs from the original midpoint.

**File:** `C++/src/Lineage.cpp`

- **Lines 192-195:** Added `x, y, z` to `PreOptShape` struct:

  ```cpp
  // Before:
  struct PreOptShape { float majorR; float minorR; };
  // After:
  struct PreOptShape { float majorR; float minorR; float x, y, z; };
  ```

- **Lines 199-200:** Save position alongside radii:

  ```cpp
  // Before:
  preOptShapes[params.name] = {(float)params.majorRadius, (float)params.minorRadius};
  // After:
  preOptShapes[params.name] = {(float)params.majorRadius, (float)params.minorRadius,
                               params.x, params.y, params.z};
  ```

- **Lines 270-276:** Look up and pass pre-opt position:

  ```cpp
  // Before:
  float preOptMajorR = 0.0f, preOptMinorR = 0.0f;
  // After:
  float preOptMajorR = 0.0f, preOptMinorR = 0.0f;
  float preOptX = 0.0f, preOptY = 0.0f, preOptZ = 0.0f;
  // ... it->second.x/y/z ...
  frame.trySplitCell(idx, preOptMajorR, preOptMinorR, preOptX, preOptY, preOptZ, ...)
  ```

**File:** `C++/includes/Frame.hpp`

- **Lines 38-40:** Added pre-opt position parameters:

  ```cpp
  // Before:
  CostCallbackPair trySplitCell(size_t cellIndex, float preOptMajorR = 0.0f, float preOptMinorR = 0.0f, float splitElongationThreshold = 1.3f);
  // After:
  CostCallbackPair trySplitCell(size_t cellIndex, float preOptMajorR = 0.0f, float preOptMinorR = 0.0f,
                                float preOptX = 0.0f, float preOptY = 0.0f, float preOptZ = 0.0f,
                                float splitElongationThreshold = 1.3f);
  ```

**File:** `C++/src/Frame.cpp`

- **Lines 284-286:** Updated signature to accept pre-opt position and pass to `getSplitCells`.

**File:** `C++/includes/Spheroid.hpp`

- **Lines 118-121:** Added `preOptX/Y/Z` parameters:

  ```cpp
  // Before:
  ... float preOptMajorR = 0.0f, float preOptMinorR = 0.0f) const;
  // After:
  ... float preOptMajorR = 0.0f, float preOptMinorR = 0.0f,
      float preOptX = 0.0f, float preOptY = 0.0f, float preOptZ = 0.0f) const;
  ```

**File:** `C++/src/Cells/Spheroid.cpp`

- **Lines 349-351:** Updated signature.

- **Lines 369-372:** Create `pcaCenter` from pre-opt position:

  ```cpp
  cv::Point3f pcaCenter = _position;
  if (preOptMajorR > 0.0f) {
      pcaCenter = cv::Point3f(preOptX, preOptY, preOptZ);
  }
  ```

- **Lines 374-384:** Updated `[Split PreOpt]` log to show `pcaCenter` and `currentPos`.

- **Lines 386-393:** Bounding box now centered at `pcaCenter` instead of `_position`.

- **Lines 442-448:** Neighbor exclusion distance computed from `pcaCenter`:

  ```cpp
  // Before:
  float distSqToSelf = static_cast<float>(dx * dx + dy * dy + dz * dz);
  // After:
  float selfDx = static_cast<float>(x) - pcaCenter.x;
  float selfDy = static_cast<float>(y) - pcaCenter.y;
  float selfDz = static_cast<float>(z) - pcaCenter.z;
  float distSqToSelf = selfDx * selfDx + selfDy * selfDy + selfDz * selfDz;
  ```

- **Lines 481-483:** PCA data matrix centered at `pcaCenter`:

  ```cpp
  // Before:
  data.at<float>(i, 0) = rawPoints[i].x - _position.x;
  // After:
  data.at<float>(i, 0) = rawPoints[i].x - pcaCenter.x;
  ```

- **Lines 585-587:** Pixel projection for daughter placement uses `pcaCenter`:

  ```cpp
  // Before:
  const double dx = pt.x - _position.x;
  // After:
  const double dx = pt.x - pcaCenter.x;
  ```

- **Lines 626-627:** Fallback placement uses `pcaCenter`:

  ```cpp
  // Before:
  new_position1 = _position + split_axis * offset;
  // After:
  new_position1 = pcaCenter + split_axis * offset;
  ```

### Changed PCA normalization from per-axis stddev to cell-radius

Per-axis stddev normalization suppressed the Z-direction split signal: if two blobs are separated in Z, Z-stddev is large, and dividing by it compresses the separation. Now normalizes by the cell's effective radii (`effA`, `effB`, `effC`) instead. A non-splitting cell looks spherical in normalized space; a splitting cell shows elongation in the split direction.

**File:** `C++/src/Cells/Spheroid.cpp`

- **Lines 472-480:** Updated comment explaining the change from stddev to cell-radius normalization.

- **Lines 493-498:** Replaced stddev computation with cell-radius normalization:

  ```cpp
  // Before:
  float sx = 0, sy = 0, sz = 0;
  // ... compute stddev from data ...
  invSx = 1.0 / sx;  invSy = 1.0 / sy;  invSz = 1.0 / sz;

  // After:
  float invSx = (effA > 1e-6f) ? (1.0f / effA) : 1.0f;
  float invSy = (effB > 1e-6f) ? (1.0f / effB) : 1.0f;
  float invSz = (effC > 1e-6f) ? (1.0f / effC) : 1.0f;
  ```

- **Lines 540-542:** Eigenvector transform back to image space uses radii:

  ```cpp
  // Before:
  ev_norm.x * sx, ev_norm.y * sy, ev_norm.z * sz
  // After:
  ev_norm.x * effA, ev_norm.y * effB, ev_norm.z * effC
  ```

- **Line 558:** Debug log changed `stddev=(...)` to `normRadii=(...)`.

### Added PCA elongation ratio filter to skip non-splitting cells

Phase 2 split detection now checks the PCA elongation ratio before running the expensive 500-iteration burn-in. Cells with nearly spherical bright pixel distributions (elongation ratio < threshold) are skipped, saving significant runtime.

**File:** `C++/includes/ConfigTypes.hpp`

- **Line 48:** Added `float split_elongation_threshold;` member to `ProbabilityConfig`
- **Line 50:** Updated constructor initializer to include `split_elongation_threshold(1.3f)`
- **Lines 63-65:** Added YAML parsing in `explodeConfig()`:

  ```cpp
  // Before: (no parsing)
  // After:
  if (node["split_elongation_threshold"]) {
      split_elongation_threshold = node["split_elongation_threshold"].as<float>();
  }
  ```

- **Line 71:** Added print in `printConfig()`:

  ```cpp
  // Before: (no print)
  // After:
  std::cout << "split_elongation_threshold: " << split_elongation_threshold << std::endl;
  ```

**File:** `C++/includes/Spheroid.hpp`

- **Line 118:** Changed `getSplitCells` return type:

  ```cpp
  // Before:
  std::tuple<Spheroid, Spheroid, bool> getSplitCells(...) const;
  // After:
  std::tuple<Spheroid, Spheroid, bool, float> getSplitCells(...) const;
  ```

**File:** `C++/src/Cells/Spheroid.cpp`

- **Line 349:** Changed function signature to match new return type:

  ```cpp
  // Before:
  std::tuple<Spheroid, Spheroid, bool> Spheroid::getSplitCells(...)
  // After:
  std::tuple<Spheroid, Spheroid, bool, float> Spheroid::getSplitCells(...)
  ```

- **Line 462:** Declared outer `elongationRatio` variable before PCA branch:

  ```cpp
  // Before: (no variable)
  // After:
  float elongationRatio = 1.0f;
  ```

- **Line 529:** Changed from local declaration to assignment to outer variable:

  ```cpp
  // Before:
  float elongationRatio = (lambda2 > 1e-6f) ? (lambda1 / lambda2) : 1.0f;
  // After:
  elongationRatio = (lambda2 > 1e-6f) ? (lambda1 / lambda2) : 1.0f;
  ```

- **Line 629:** Added `elongationRatio` to return tuple:

  ```cpp
  // Before:
  return std::make_tuple(Spheroid(cell1), Spheroid(cell2), constraints);
  // After:
  return std::make_tuple(Spheroid(cell1), Spheroid(cell2), constraints, elongationRatio);
  ```

**File:** `C++/includes/Frame.hpp`

- **Line 38:** Added threshold parameter to `trySplitCell()`:

  ```cpp
  // Before:
  CostCallbackPair trySplitCell(size_t cellIndex, float preOptMajorR = 0.0f, float preOptMinorR = 0.0f);
  // After:
  CostCallbackPair trySplitCell(size_t cellIndex, float preOptMajorR = 0.0f, float preOptMinorR = 0.0f, float splitElongationThreshold = 1.3f);
  ```

**File:** `C++/src/Frame.cpp`

- **Line 284:** Updated signature:

  ```cpp
  // Before:
  CostCallbackPair Frame::trySplitCell(size_t index, float preOptMajorR, float preOptMinorR)
  // After:
  CostCallbackPair Frame::trySplitCell(size_t index, float preOptMajorR, float preOptMinorR, float splitElongationThreshold)
  ```

- **Lines 303-304:** Unpack 4th tuple element:

  ```cpp
  // Before:
  bool valid;
  std::tie(child1, child2, valid) = oldCell.getSplitCells(...)
  // After:
  bool valid;
  float elongationRatio;
  std::tie(child1, child2, valid, elongationRatio) = oldCell.getSplitCells(...)
  ```

- **Lines 312-317:** Added elongation ratio filter (after validity check, before burn-in):

  ```cpp
  // Before: (no filter)
  // After:
  if (splitElongationThreshold > 0.0f && elongationRatio < splitElongationThreshold) {
      std::cout << "[Split Skip] " << oldCell.getCellParams().name
                << " elongation_ratio=" << elongationRatio
                << " < threshold=" << splitElongationThreshold << std::endl;
      return {0.0, [](bool accept) {}};
  }
  ```

**File:** `C++/src/Lineage.cpp`

- **Line 276:** Pass threshold from config:

  ```cpp
  // Before:
  auto result = frame.trySplitCell(idx, preOptMajorR, preOptMinorR);
  // After:
  auto result = frame.trySplitCell(idx, preOptMajorR, preOptMinorR, config.prob.split_elongation_threshold);
  ```

**File:** `C++/examples/config.yaml`

- **Line 59:** Added parameter under `prob:` section:

  ```yaml
  # Before: (no entry)
  # After:
  split_elongation_threshold: 1.3 # skip split burn-in if PCA elongation ratio < this
  ```

## 2026-03-27

### Reorganized `examples/` into `config/`, `scripts/`, `data/`, `outputs/`

Directory restructuring (no code changes, file moves only).

**Moved to `C++/config/` (6 files):**
- `examples/config.yaml` -> `config/config.yaml`
- `examples/initial.csv` -> `config/initial.csv`
- `examples/initial_auto.csv` -> `config/initial_auto.csv`
- `examples/initial_embryo.csv` -> `config/initial_embryo.csv`
- `examples/user_input_configurations.ini` -> `config/user_input_configurations.ini`
- `examples/runauto.args` -> `config/runauto.args`

**Moved to `C++/scripts/` (6 files):**
- `examples/run_celluniverse.sh` -> `scripts/run_celluniverse.sh`
- `examples/run_original.sh` -> `scripts/run_original.sh`
- `examples/run_embryo.sh` -> `scripts/run_embryo.sh`
- `examples/runauto.sh` -> `scripts/runauto.sh`
- `examples/remove_old_outputs.sh` -> `scripts/remove_old_outputs.sh`
- `examples/convert_png_to_tiff.py` -> `scripts/convert_png_to_tiff.py`

**Moved to `C++/data/`:**
- `examples/input/` -> `data/input/`

**Moved to `C++/outputs/`:**
- `examples/brightness_hist_*.svg`, `runtime_log` -> `outputs/`
- Any `examples/output_*` directories -> `outputs/`

**Removed:** `C++/examples/` (directory deleted)

**Updated paths in `C++/config/user_input_configurations.ini`:**
All preset sections updated:
- `input_path`: prepended `../data/` (e.g., `input/original_data` -> `../data/input/original_data`)
- `output_base_dir`: changed from `.` to `../outputs`
- `build_dir`, `cell_config_file`, `initial_csv_file`, `cli_args_file`: unchanged (same relative depth)

**Updated `C++/docs/details.md` section 11 (File Reference):**
- `C++/examples/config.yaml` -> `C++/config/config.yaml`
- `C++/examples/initial.csv` -> `C++/config/initial.csv`

