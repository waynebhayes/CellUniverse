# Changelog: Project Directory Restructuring (2026-03-27)

## Summary

Reorganized the C++ project directory structure to separate concerns: deprecated dead code moved out of active source tree, config/scripts/data separated from examples/, Spheroid.cpp moved to top-level src/.

---

## 1. Files moved to `C++/deprecated/`

Nine files moved from active source/include directories to `C++/deprecated/`:

| File | Original location | Content |
|------|-------------------|---------|
| `Cell.hpp` | `C++/includes/Cell.hpp` | Abstract base class (unused since Spheroid became standalone) |
| `Sphere.hpp` | `C++/includes/Sphere.hpp` | Sphere subclass of Cell (unused) |
| `Sphere.cpp` | `C++/src/Sphere.cpp` | Sphere implementation (unused) |
| `Bacilli.hpp` | `C++/includes/Bacilli.hpp` | Bacilli subclass of Cell (unused) |
| `Bacilli.cpp` | `C++/src/Bacilli.cpp` | Bacilli implementation (unused) |
| `mathhelper.hpp` | `C++/includes/mathhelper.hpp` | Math utilities (unused) |
| `args.cpp` | `C++/src/args.cpp` | Argument parsing (replaced by direct argv parsing in main.cpp) |
| `pseudo-code.txt` | `C++/src/pseudo-code.txt` | Design notes |
| `README.md` | `C++/deprecated/README.md` | Old readme preserved |

Also already present in deprecated: `frame000.tif` (test image).

## 2. `#include "Cell.hpp"` removed from 4 files

Removed the dead include from files that no longer reference the Cell base class:

- `C++/includes/Spheroid.hpp`
- `C++/includes/Frame.hpp`
- `C++/src/CellFactory.cpp`
- `C++/src/Frame.cpp`

## 3. `#include "Sphere.hpp"` removed from `C++/includes/Frame.hpp`

Dead include for the unused Sphere class.

## 4. Frame method signatures fixed

Removed `const Cell& oldCell` parameter from methods in `C++/includes/Frame.hpp` and `C++/src/Frame.cpp` that referenced the deleted Cell base class:

- `generateSynthFrameFast()` -- parameter changed from `const Cell& oldCell` to `const Spheroid& oldCell`
- Other methods updated to use `Spheroid` directly instead of `Cell` references

## 5. `gradientDescent()` removed

Removed unused `gradientDescent()` method from:
- `C++/includes/Frame.hpp` (declaration)
- `C++/src/Frame.cpp` (implementation)

This method was dead code -- all optimization uses Monte Carlo perturbation.

## 6. `Spheroid.cpp` moved from `src/Cells/` to `src/`

- **From:** `C++/src/Cells/Spheroid.cpp`
- **To:** `C++/src/Spheroid.cpp`
- Include path updated in the file: `#include "Spheroid.hpp"` (path unchanged since includes/ is a flat directory)
- `C++/src/Cells/` directory removed (was the only file in it)

## 7. `CMakeLists.txt` updated

Source file paths updated to reflect:
- `src/Spheroid.cpp` (was `src/Cells/Spheroid.cpp`)
- Removed references to `src/Sphere.cpp`, `src/args.cpp`, `src/Bacilli.cpp`

## 8. `C++/examples/` reorganized into `config/`, `scripts/`, `data/`

The monolithic `examples/` directory was split into purpose-specific directories:

| New location | Files moved | Original location |
|-------------|-------------|-------------------|
| `C++/config/` | `config.yaml`, `initial.csv`, `initial_auto.csv`, `initial_embryo.csv`, `user_input_configurations.ini`, `runauto.args` | `C++/examples/` |
| `C++/scripts/` | `run_celluniverse.sh`, `run_original.sh`, `run_embryo.sh`, `runauto.sh`, `convert_png_to_tiff.py`, `remove_old_outputs.sh` | `C++/examples/` |
| `C++/data/input/` | `original_data/` (input images) | `C++/examples/` |

## 9. INI file paths updated

`C++/config/user_input_configurations.ini` -- all paths updated to reflect the new directory structure:

- Config file paths: `examples/config.yaml` -> `config/config.yaml`
- Initial CSV paths: `examples/initial.csv` -> `config/initial.csv`
- Input data paths: `examples/original_data` -> `data/input/original_data`
- Output paths: updated to `outputs/` or appropriate locations

## 10. Documentation updated

- **`C++/docs/details.md`**: `src/Cells/Spheroid.cpp` -> `src/Spheroid.cpp` (2 occurrences)
- **`C++/README.md`**: Complete rewrite with new directory structure, updated build/run commands
- **`.claude/CLAUDE.md`**: `C++/examples/details.md` -> `C++/docs/details.md` (reference fix)
- **`.claude/rules/codebase.md`**: Removed "(moved from src/Cells/)" parenthetical from Spheroid.cpp entry
- **`.claude/rules/changelog.md`**: `C++/examples/details.md` -> `C++/docs/details.md`
- **MEMORY.md**: `C++/examples/config.yaml` -> `C++/config/config.yaml`

---

## 11. Dead code removed from ConfigTypes, CellFactory, main, and types (Task 10)

### 11a. Removed unused `z_values` from SimulationConfig

**File:** `C++/includes/ConfigTypes.hpp`

**Line 19 (before):**
```cpp
    int z_slices;
    std::vector<int> z_values;
    float sigmoid_k = 75.0f;
```

**Line 18 (after):**
```cpp
    int z_slices;
    float sigmoid_k = 75.0f;
```

**Effect:** `z_values` was populated in main.cpp but never read by any code. Removing the field and its population code.

### 11b. Removed `z_values` population from `updateTiffConfigIfNeeded`

**File:** `C++/src/main.cpp`

**Lines 38-43 (before):**
```cpp
    config.simulation.z_slices = slices;
    config.simulation.z_values.clear();
    for (int j = 0; j < slices; ++j)
    {
        config.simulation.z_values.push_back(j - slices / 2);
    }
```

**Line 37 (after):**
```cpp
    config.simulation.z_slices = slices;
```

### 11c. Fixed dead sphere branch in CellFactory constructor

**File:** `C++/src/CellFactory.cpp`

**Lines 6-14 (before):**
```cpp
    if (cellType == "sphere") {
        // Sphere::cellConfig = *config.cell; // this is ideal, but entire code base must be changed to run it this way
        Spheroid::cellConfig = *config.cell;
    } if (cellType == "spheroid") {
        Spheroid::cellConfig = *config.cell;
    }
    else {
        throw std::invalid_argument("Invalid cell type: " + config.cellType);
    }
```

**Lines 6-10 (after):**
```cpp
    if (cellType == "spheroid") {
        Spheroid::cellConfig = *config.cell;
    } else {
        throw std::invalid_argument("Invalid cell type: " + config.cellType);
    }
```

**Effect:** The old code had a bug: `if (sphere) {...} if (spheroid) {...} else {throw}`. Because the second `if` was not `else if`, passing `cellType == "sphere"` would execute the sphere branch AND fall through to the else, throwing an exception. The sphere branch was dead code anyway (Sphere class is deprecated). Replaced with a clean if/else.

### 11d. Removed dead LineageViewer usage from main.cpp

**File:** `C++/src/main.cpp`

Removed:
1. `#include "LineageViewer.hpp"` (line 14)
2. `LineageViewer viewer;` declaration (line 220) -- object was created but never used
3. The entire CellViz building block inside the frame loop (lines 221-234): vector construction and commented-out `viewer.update()` call
4. The dead infinite wait loop at the end of main (lines 250-260): `cv::waitKey` loop waiting for a window that was never created

**Effect:** Removed ~25 lines of dead code. The program now exits cleanly after processing instead of hanging in a `waitKey` loop. Note: `LineageViewer.hpp` and `LineageViewer.cpp` files remain in the codebase for future use.

### 11e. Removed commented-out Sphere include from types.hpp

**File:** `C++/includes/types.hpp`

**Line 15 (before):**
```cpp
#include <map>
//#include "Sphere.hpp"
```

**Line 14 (after):**
```cpp
#include <map>
```

**Effect:** Dead commented-out include for the deprecated Sphere class.

---

## 12. Dead code removed from Frame (Task 9)

Removed four dead methods from Frame that were superseded by newer implementations.

### Problem/Motivation

Frame.hpp/Frame.cpp contained several methods that are no longer called anywhere in the codebase. `perturb()` was replaced by `perturbCell()` (which uses cached cost and per-cell overlap). `split()` was a trivial wrapper around `trySplitCell()` with a random index but is never called. `padRealFrame()` was never called (only a commented-out reference existed). `costOfPerturb()` and `getSynthPerturbedCells()` were gradient-descent-era helpers that are no longer used.

### Files changed

**File:** `C++/includes/Frame.hpp`

**Line 26 (before):**
```cpp
    void padRealFrame();
```
**Removed.**

**Line 34 (before):**
```cpp
    CostCallbackPair perturb();
    CostCallbackPair perturbCell(size_t index, float overlapWeight = 1000.0f);
```
**After:**
```cpp
    CostCallbackPair perturbCell(size_t index, float overlapWeight = 1000.0f);
```

**Line 36 (before):**
```cpp
    CostCallbackPair split();
    double computeOverlapPenalty(float weight) const;
```
**After:**
```cpp
    double computeOverlapPenalty(float weight) const;
```

**Lines 60-61 (before):**
```cpp
    Cost costOfPerturb(const std::string &perturbParam, float perturbVal, size_t index);
    ParamImageMap getSynthPerturbedCells(size_t index, const ParamValMap &params, float perturbLength);
```
**Removed.**

---

**File:** `C++/src/Frame.cpp`

**Lines 41-42 (before):**
```cpp
    // TODO: Fix padding
    //    padRealImage();
```
**Removed** (dead comment and commented-out call in constructor).

**Lines 47-61 (before):**
```cpp
void Frame::padRealFrame()
{
    int padding = simulationConfig.padding;
    cv::Scalar paddingColor(simulationConfig.background_color);
    cv::Mat paddedFrame;
    cv::copyMakeBorder(_realFrame, paddedFrame, padding, padding, padding, padding, cv::BORDER_CONSTANT, paddingColor);
    _realFrame = paddedFrame;
}
```
**Removed.**

**Lines 226-274 (before):**
```cpp
CostCallbackPair Frame::perturb()
{
    // ~48 lines: picks random cell index, stores old cell, calls getPerturbedCell,
    // checks checkIfCellsValid, generates fast synth frame, computes cost diff,
    // returns callback. Recalculates _synthFrame cost every call (not cached).
}
```
**Removed.** Superseded by `perturbCell()` which uses cached `_currentCost` and per-cell overlap penalty.

**Lines 473-480 (before):**
```cpp
CostCallbackPair Frame::split()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, cells.size() - 1);
    size_t index = distrib(gen);
    return trySplitCell(index);
}
```
**Removed.** Never called -- Phase 2 split evaluation calls `trySplitCell()` directly.

**Lines 638-686 (before):**
```cpp
Cost Frame::costOfPerturb(const std::string &perturbParam, float perturbVal, size_t index)
{
    // ~19 lines: creates perturbParams map, calls getParameterizedCell,
    // does full generateSynthFrame + calculateCost, restores original cell.
}

ParamImageMap Frame::getSynthPerturbedCells(size_t index, const ParamValMap &params, float perturbLength)
{
    // ~22 lines: iterates over param map, for each param calls getParameterizedCell,
    // generates full synth frame, stores in map, restores original cell.
}
```
**Removed.** These were gradient-descent-era helpers. All optimization now uses Monte Carlo perturbation via `perturbCell()`.

### Effect

- Frame.hpp reduced from 63 lines to 58 lines (5 declarations removed)
- Frame.cpp reduced from 692 lines to 565 lines (127 lines of dead code removed)
- All preserved methods (`perturbCell`, `randomSplitCell`, `trySplitCell`, `computeOverlapPenalty`, `computeOverlapForCell`, `computeElongationRatios`, `computeElongationForCell`, `generateSynthFrame`, `generateSynthFrameFast`, `calculateCost`, `generateOutputFrame`, `generateOutputSynthFrame`, `regenerateSynthFrame`, `getSynthFrame`, `getImageName`, `length`) remain unchanged

---

## 13. Dead code removed from Spheroid (Task 8)

Removed dead member variables, dead methods, dead static members, dormant checks, unused `cellMap` parameter, and dead SpheroidParams constructors/methods from the Spheroid class.

### Problem/Motivation

The Spheroid class accumulated dead code from older designs: `_x_vec/_y_vec/_z_vec` vectors (never populated in the active constructor), `_rotation` (unused since Euler angles replaced it), `dormant` (always `false`), `paramClass` (static but never referenced), `major_magnitude()`/`minor_magnitude()` (crash on empty `_x_vec`), `get_matrix_size()`/`getShapeAt()` (never called), `checkIfCellsValid()`/`checkIfCellsOverlap()` (only called from already-removed `perturb()`), the `cellMap` parameter in `draw()` (always `nullptr`), and SpheroidParams' vector constructor and `parseParams()` (never used since cells are created with scalar params).

### Files changed

**File:** `C++/includes/Spheroid.hpp`

**SpheroidParams: removed vector members, vector constructor, and parseParams (lines 29-66 before)**

Lines 29-31 (before):
```cpp
    std::vector<float> x_vec;
    std::vector<float> y_vec;
    std::vector<float> z_vec;
```
Removed.

Lines 43-65 (before):
```cpp
    SpheroidParams(const std::string &name, float x, float y, float z, std::vector<float> _x_vec, std::vector<float> _y_vec, std::vector<float> _z_vec)
        : CellParams(name), x(x), y(y), z(z), x_vec(_x_vec), y_vec(_y_vec), z_vec(_z_vec)
        { ... }

    void parseParams(float x_, float y_, float z_, std::vector<float> _x_vec, std::vector<float> _y_vec, std::vector<float> _z_vec)
    { ... }
```
Removed.

**Spheroid private members (lines 75-87 before):**

```cpp
        std::vector<double> _x_vec;
        std::vector<double> _y_vec;
        std::vector<double> _z_vec;
        double _rotation;
        bool dormant;
```
All five removed.

**Static paramClass (line 99 before):**

```cpp
        static SpheroidParams paramClass;
```
Removed. `static SpheroidConfig cellConfig;` retained (it IS used).

**Default constructor (line 104 before):**

```cpp
        Spheroid() : _major_radius(0), _minor_radius(0), _rotation(0), _theta_x(0), _theta_y(0), _theta_z(0), _brightness(0.5f), dormant(false) {}
```
After:
```cpp
        Spheroid() : _major_radius(0), _minor_radius(0), _theta_x(0), _theta_y(0), _theta_z(0), _brightness(0.5f) {}
```

**printCellInfo (line 107 before):**

Removed `<< " isDormant: " << dormant` from the output.

**draw() signature (line 112 before):**

```cpp
        void draw(cv::Mat &image, SimulationConfig simulationConfig, cv::Mat *cellMap = nullptr, float z = 0) const;
```
After:
```cpp
        void draw(cv::Mat &image, const SimulationConfig &simulationConfig, float z = 0) const;
```
Also changed `SimulationConfig` to `const SimulationConfig &` to avoid copy.

**Dead method declarations removed (lines 110, 133-136, 140, 142, 144, 150 before):**

```cpp
        std::vector<double> getShapeAt(double z) const;
        bool checkIfCellsValid(const std::vector<Spheroid> &spheroids) { ... }
        static bool checkIfCellsOverlap(const std::vector<Spheroid> &spheroids);
        float major_magnitude() const;
        float minor_magnitude() const;
        int get_matrix_size();
```
All removed.

---

**File:** `C++/src/Spheroid.cpp`

**Static paramClass definition (line 90 before):**

```cpp
SpheroidParams Spheroid::paramClass = SpheroidParams();
```
Removed.

**_get_magnitude free function (lines 95-97 before):**

```cpp
static double _get_magnitude(std::vector<double> vec){
    return std::sqrt((vec[0]*vec[0])+(vec[1]*vec[1])+(vec[2]*vec[2]));
}
```
Removed.

**Main constructor initializer list (lines 156-160 before):**

```cpp
          _rotation(0),
          ...
          _brightness(init_props.brightness), dormant(false)
```
After:
```cpp
          ...
          _brightness(init_props.brightness)
```

**major_magnitude and minor_magnitude (lines 189-195 before):**

```cpp
float Spheroid::major_magnitude() const {
    return _get_magnitude(_x_vec);
}

float Spheroid::minor_magnitude() const {
    return _get_magnitude(_z_vec);
}
```
Removed.

**get_matrix_size and getShapeAt (lines 209-219 before):**

```cpp
int Spheroid::get_matrix_size(){ ... }
std::vector<double> Spheroid::getShapeAt(double z) const { ... }
```
Removed.

**draw() implementation (lines 224-230 before):**

```cpp
void Spheroid::draw(cv::Mat &image, SimulationConfig simulationConfig, cv::Mat *cellMap, float z) const{
    (void)cellMap;
    if (dormant) { return; }
```
After:
```cpp
void Spheroid::draw(cv::Mat &image, const SimulationConfig &simulationConfig, float z) const{
```

**drawOutline dormant check (line 262 before):**

```cpp
    if (dormant) return;
```
Removed.

**checkIfCellsOverlap (lines 766-822 before):**

```cpp
bool Spheroid::checkIfCellsOverlap(const std::vector<Spheroid> &spheroids) {
    // ~56 lines building position/distance/radii matrices
}
```
Removed.

---

**File:** `C++/src/Frame.cpp`

**draw() call sites (lines 81, 154 before):**

```cpp
cell.draw(synthImage, simulationConfig, nullptr, z);
```
After:
```cpp
cell.draw(synthImage, simulationConfig, z);
```
Both occurrences in `generateSynthFrame()` and `generateSynthFrameFast()`.

**Stale comment (line 316 before):**

```cpp
        float ri = pi.majorRadius;  // use majorRadius directly (major_magnitude() crashes if _x_vec empty)
```
After:
```cpp
        float ri = pi.majorRadius;
```

---

**File:** `C++/tests/spheroid_test.cc`

**draw() call (line 45 before):**

```cpp
    spheroid.draw(image, simulationConfig, nullptr, 0.0f);
```
After:
```cpp
    spheroid.draw(image, simulationConfig, 0.0f);
```

**DetectsOverlapAndNonOverlap test (lines 51-60 before):**

```cpp
TEST(SpheroidTest, DetectsOverlapAndNonOverlap) {
    // calls Spheroid::checkIfCellsOverlap
}
```
Removed entirely (tested a now-removed method).

---

**File:** `C++/tests/spheroid_rotation_test.cc`

**All draw() calls updated from:**
```cpp
    spheroid.draw(image, cfg, nullptr, 0.0f);
```
To:
```cpp
    spheroid.draw(image, cfg, 0.0f);
```
8 call sites updated across 5 test functions.

### Effect

- `Spheroid.hpp` reduced from 153 lines to 103 lines (50 lines removed)
- `Spheroid.cpp` reduced by ~90 lines (dead methods + member init + overlap check)
- `Frame.cpp` draw calls simplified (removed always-nullptr parameter)
- Test files updated to match new signatures; one dead test removed
- No behavioral change -- all removed code was either never called or always-constant (`dormant` was always `false`, `cellMap` was always `nullptr`)

## 2026-03-28

### Status: ACTIVE

### Step 5: Removed dead `_realFrameCopy` member

**Problem/Motivation:** `_realFrameCopy` was deep-cloned in the Frame constructor but never read anywhere in the codebase. It was a leftover from the reverted real-image normalization approach (see changelogv2.md). Each Frame allocated 225 deep-cloned cv::Mat slices for nothing.

**Files changed:**
- `C++/includes/Frame.hpp`
- `C++/src/Frame.cpp`

**File:** `C++/includes/Frame.hpp`

**Line 54 (before):**
```cpp
    std::vector<cv::Mat> _realFrame;
    std::vector<cv::Mat> _realFrameCopy; // copy of realFrame
    std::vector<cv::Mat> _synthFrame;
```

**Line 53 (after):**
```cpp
    std::vector<cv::Mat> _realFrame;
    std::vector<cv::Mat> _synthFrame;
```

**File:** `C++/src/Frame.cpp`

**Lines 27-33 (before):**
```cpp
    // Deep-clone _realFrameCopy so it has its own pixel buffers.
    // cv::Mat copy constructor is shallow (shared data pointer), so
    // modifications to _realFrame would corrupt _realFrameCopy without this.
    _realFrameCopy.reserve(realFrame.size());
    for (const auto &mat : realFrame) {
        _realFrameCopy.push_back(mat.clone());
    }

    // Calculate z_slices
```

**Lines 27 (after):**
```cpp
    // Calculate z_slices
```

**Effect:** Eliminates 225 unnecessary deep-cloned cv::Mat allocations per frame construction. Reduces memory usage and constructor time.

---

### Step 6: Removed dead `CellConfig` and `SphereConfig` classes from ConfigTypes.hpp

**Problem/Motivation:** `CellConfig` was an abstract base class only inherited by `SphereConfig` (deprecated, Sphere model is unused) and `SpheroidConfig`. `SphereConfig` is never instantiated outside deprecated code. Removing both simplifies the class hierarchy. `SpheroidConfig` no longer inherits from `CellConfig` and its `explodeConfig` method drops the `override` specifier.

**Files changed:**
- `C++/includes/ConfigTypes.hpp`

**File:** `C++/includes/ConfigTypes.hpp`

**Lines 128-162 (before):**
```cpp
class CellConfig {
    // A pure abstract base class for SphereConfig and BacilliConfig
public:
    // pure virtual function for exploding the configuration
    virtual void explodeConfig(const YAML::Node& node) = 0;
    virtual ~CellConfig() = default;
//    virtual CellConfig& operator=(const CellConfig& other) = 0;
};

class SphereConfig: public CellConfig {
public:
    PerturbParams x{};
    PerturbParams y{};
    PerturbParams z{};
    PerturbParams radius{};
    double minRadius{};
    double maxRadius{};
    double boundingBoxScale{1.0};
    ~SphereConfig() = default;

    void explodeConfig(const YAML::Node& node) override
    {
        x.explodeParams(node["x"]);
        y.explodeParams(node["y"]);
        z.explodeParams(node["z"]);
        radius.explodeParams(node["radius"]);
        minRadius = node["minRadius"].as<double>();
        maxRadius = node["maxRadius"].as<double>();
        if (node["boundingBoxScale"]) {
            boundingBoxScale = node["boundingBoxScale"].as<double>();
        }
    }
};

class SpheroidConfig: public CellConfig {
    ...
    void explodeConfig(const YAML::Node& node) override
```

**Lines 122 (after):**
```cpp
class SpheroidConfig {
    ...
    void explodeConfig(const YAML::Node& node)
```

**Effect:** Removes 35 lines of dead code. `SpheroidConfig` is now a standalone class, not part of a polymorphic hierarchy that was never used polymorphically.

Also updated stale comment on PerturbParams class from "Used with a CellConfig" to "Used with a cell config" (line 101).

---

### Step 7: Removed dead type aliases from types.hpp

**Problem/Motivation:** `ParamImageMap` and `ParamValMap` were type aliases for `std::unordered_map` types used by the old `getSynthPerturbedCells` method which was removed during restructuring. No source file references either alias.

**Files changed:**
- `C++/includes/types.hpp`

**File:** `C++/includes/types.hpp`

**Lines 30-31 (before):**
```cpp
typedef std::unordered_map<std::string, ImageStack> ParamImageMap;
typedef std::unordered_map<std::string, float> ParamValMap;
```

**Lines 30-31 (after):**
(removed)

**Effect:** Removes two unused type aliases, reducing include-chain coupling.

---

### Step 8: Replaced `std::endl` with `'\n'` globally

**Problem/Motivation:** `std::endl` outputs a newline AND flushes the stream buffer. In a console application with diagnostic logging in tight loops (perturbation iterations, split evaluation), the flush is unnecessary overhead. `'\n'` outputs the newline without flushing.

**Files changed (all `std::endl` occurrences replaced with `'\n'`):**
- `C++/src/Frame.cpp` (4 occurrences)
- `C++/src/CellUniverse.cpp` (27 occurrences)
- `C++/src/main.cpp` (13 occurrences)
- `C++/src/CellFactory.cpp` (4 occurrences)
- `C++/src/Spheroid.cpp` (9 occurrences)
- `C++/includes/Spheroid.hpp` (1 occurrence)
- `C++/includes/ConfigTypes.hpp` (12 occurrences)

**Example (CellUniverse.cpp line 12):**

Before:
```cpp
std::cout << "The matrix is empty." << std::endl;
```

After:
```cpp
std::cout << "The matrix is empty." << '\n';
```

**Effect:** Eliminates unnecessary buffer flushes across all logging output. Particularly beneficial during the optimizer loop where thousands of diagnostic lines may be printed per frame.

---

### Step 1: Removed dead variables and commented code

**Problem/Motivation:** Multiple files contained unused variables, commented-out code blocks, and debug cout lines that added noise without serving any purpose.

**Files changed:**
- `C++/src/Frame.cpp`
- `C++/src/CellUniverse.cpp`
- `C++/src/Spheroid.cpp`

**File:** `C++/src/Frame.cpp`

**generateSynthFrame() lines 46-66 (before):**
```cpp
    unsigned int x = 0;
    // std::cout << "Num of Cells to draw : " << cells.size() << '\n';
    for (double z : z_slices)
    {
        Image synthImage = cv::Mat(shape, CV_32F, cv::Scalar(simulationConfig.background_color)); // Assuming background color is in cv::Scalar format
        for (const auto &cell : cells)
        {
            // cell.printCellInfo();
            // cell.print();
            cell.draw(synthImage, simulationConfig, z);
        }
        // if(!frame.empty() && x > 0)
        // {
        //     unsigned num_interpolated_slices = 7;
        //     std::vector<cv::Mat> interSlices{...};
        //     ...
        // }
        x += 1;
        frame.push_back(synthImage);
    }
```

**generateSynthFrame() (after):**
```cpp
    for (double z : z_slices)
    {
        Image synthImage = cv::Mat(shape, CV_32F, cv::Scalar(simulationConfig.background_color));
        for (const auto &cell : cells)
        {
            cell.draw(synthImage, simulationConfig, z);
        }
        frame.push_back(synthImage);
    }
```

**Also removed from Frame.cpp:**
- Commented-out `cells.empty()` check block (lines 39-41)
- Commented-out duplicate `length()` definition (lines 191-194)

**File:** `C++/src/CellUniverse.cpp`

**processImage() (before):**
```cpp
    SimulationConfig simConfig = config.simulation;
    //    std::cout << "The blur sigma is: " <<  simConfig.blur_sigma << std::endl;
    cv::GaussianBlur(processedImage, processedImage, cv::Size(0, 0), config.simulation.blur_sigma);
```

**processImage() (after):**
```cpp
    cv::GaussianBlur(processedImage, processedImage, cv::Size(0, 0), config.simulation.blur_sigma);
```

**Also removed from CellUniverse.cpp:**
- Commented assert `// assert(numTiffSlices == 33);`
- Commented cout `// std::cout << "Number of synthetic slices: ...`
- Commented cout `//        std::cout << "Filename: ...`

**File:** `C++/src/Spheroid.cpp`

**Constructor (before):**
```cpp
    // // DEBUG: Print to verify values -- remove after confirming correctness
    // std::cout << "[Spheroid INIT] " << _name
    //         << " a=" << a << " b=" << b << " c=" << c
    //         << " theta=(" << _theta_x << ", " << _theta_y << ", " << _theta_z << ")"
    //         << std::endl;
```
(removed)

**draw() (before):**
```cpp
    // TEMPORARY DEBUG -- remove after testing
    // std::cout << "[DRAW] " << _name
    //         << " a=" << a << " b=" << b << " c=" << c
    //         << " theta_y=" << _theta_y
    //         << " z=" << z << " pos_z=" << _position.z << std::endl;
```
(removed)

**Effect:** Removes dead variables, commented-out code blocks, and debug prints across three source files.

---

### Step 2: Removed dead config fields

**Problem/Motivation:** Three config fields were declared, parsed, and printed but never referenced by any active source code:
- `cell_color` (SimulationConfig) -- replaced by per-cell `_brightness`; only used in deprecated Sphere.cpp
- `padding` (SimulationConfig) -- never referenced anywhere
- `perturbation` (ProbabilityConfig) -- the split/perturb decision is now driven by PCA elongation, not this field

**Files changed:**
- `C++/includes/ConfigTypes.hpp`
- `C++/config/config.yaml`
- `C++/tests/frame_test.cc`
- `C++/tests/spheroid_rotation_test.cc`

**File:** `C++/includes/ConfigTypes.hpp` (SimulationConfig)

**Fields (before):**
```cpp
    int iterations_per_cell;
    float background_color;
    float cell_color;
    int padding;
    float z_scaling;
```

**Fields (after):**
```cpp
    int iterations_per_cell;
    float background_color;
    float z_scaling;
```

Constructor, `explodeConfig()`, and `printConfig()` all updated to remove `cell_color` and `padding`.

**File:** `C++/includes/ConfigTypes.hpp` (ProbabilityConfig)

**Fields (before):**
```cpp
    float perturbation;
    float split;
```

**Fields (after):**
```cpp
    float split;
```

Constructor, `explodeConfig()`, and `printConfig()` all updated to remove `perturbation`.

**File:** `C++/tests/frame_test.cc`

Removed `cfg.cell_color = 1.0f;` and `cfg.padding = 0;` from `MakeFrameConfig()`.

**File:** `C++/tests/spheroid_rotation_test.cc`

Replaced `cfg.cell_color` references with a `constexpr float kTestBrightness = 1.0f` constant. Updated Spheroid construction to pass explicit `brightness=1.0f` so the test compares against the actual rendered brightness, not a removed config field.

**Effect:** Removes three dead config fields, their YAML parsing, their print statements, and fixes test references.

---

### Step 3: Removed dead methods

**Problem/Motivation:** Two Spheroid methods were declared and implemented but never called from any active source code:
- `getParameterizedCell()` -- superseded by `getPerturbedCell()` which uses the config-driven PerturbParams system
- `performPCA()` -- standalone PCA was replaced by inline PCA inside `getSplitCells()` with isotropic normalization

**Files changed:**
- `C++/includes/Spheroid.hpp`
- `C++/src/Spheroid.cpp`
- `C++/tests/spheroid_params_test.cc`

**File:** `C++/includes/Spheroid.hpp`

**Lines 81, 88 (before):**
```cpp
        Spheroid getParameterizedCell(std::unordered_map<std::string, float> params) const;
        ...
        std::vector<std::pair<float, cv::Vec3f>> performPCA(const std::vector<cv::Point3f> &points) const;
```
(both removed)

Also removed `#include <unordered_map>` which was only needed for `getParameterizedCell`.

**File:** `C++/src/Spheroid.cpp`

Removed `getParameterizedCell()` implementation (~36 lines, was at lines 278-313).
Removed `performPCA()` implementation (~37 lines, was at lines 655-691).

**File:** `C++/tests/spheroid_params_test.cc`

Removed three tests that exercised `getParameterizedCell()`:
- `GetParameterizedCellAppliesOffsets`
- `GetParameterizedCellClampsRadiiToBounds`
- `ParameterizationStillEnforcesMinorNotGreaterThanMajor`

Also removed `#include <unordered_map>`.

**Effect:** Removes ~73 lines of dead method code and 3 tests for removed functionality.

---

### Step 4: Extracted magic numbers to config

**Problem/Motivation:** Two magic numbers were hardcoded in the optimizer:
- `500` (burn-in iterations in `trySplitCell`) -- controls how many iterations daughters are optimized before evaluating the split
- `0.5f` (max P(split) cap in `optimize()`) -- caps the adaptive split probability

Both should be configurable without recompilation.

**Files changed:**
- `C++/includes/ConfigTypes.hpp`
- `C++/src/CellUniverse.cpp`
- `C++/config/config.yaml`

**File:** `C++/includes/ConfigTypes.hpp` (ProbabilityConfig)

**Added fields:**
```cpp
    int split_burn_in_iterations = 500;
    float max_split_probability = 0.5f;
```

**Added YAML parsing:**
```cpp
        if (node["split_burn_in_iterations"]) {
            split_burn_in_iterations = node["split_burn_in_iterations"].as<int>();
        }
        if (node["max_split_probability"]) {
            max_split_probability = node["max_split_probability"].as<float>();
        }
```

**File:** `C++/src/CellUniverse.cpp`

**Lines 233, 268 (before):**
```cpp
        ? std::min(0.5f, baseSplitProb + std::max(0.0f, 1.0f - 1.0f / elong))
        ...
        ? std::min(0.5f, baseSplitProb + std::max(0.0f, 1.0f - 1.0f / elongation))
```

**Lines 233, 268 (after):**
```cpp
        ? std::min(config.prob.max_split_probability, baseSplitProb + std::max(0.0f, 1.0f - 1.0f / elong))
        ...
        ? std::min(config.prob.max_split_probability, baseSplitProb + std::max(0.0f, 1.0f - 1.0f / elongation))
```

**File:** `C++/config/config.yaml`

Added to prob section:
```yaml
  split_burn_in_iterations: 500
  max_split_probability: 0.5
```

**Note:** The `BURN_IN_ITERATIONS` constant in `Frame.cpp::trySplitCell()` remains a local constant for now. Threading `config.prob.split_burn_in_iterations` requires passing it through the call chain, which is deferred to a later step.

**Effect:** Makes two key optimizer tuning parameters configurable via YAML without recompilation.

---

# Items 9-12: Code Quality (Direct Accessors, static_cast, <random>, setenv removal)

## Item 9: Direct accessors on Spheroid — ACTIVE

**Problem:** `getCellParams()` returns `SpheroidParams` by value (copies std::string). Called ~25K times/frame in overlap loops.

**Fix:** Added inline accessors `getName()`, `getX()`, `getY()`, `getZ()`, `getMajorRadius()`, `getMinorRadius()`, `getBrightness()` to Spheroid.hpp. Updated `computeOverlapPenalty()`, `computeOverlapForCell()`, and log lines in Frame.cpp to use direct accessors.

## Item 10: C-style casts → static_cast — ACTIVE

**Fix:** All 18 C-style casts in Spheroid.cpp replaced with `static_cast<float>()`, `static_cast<int>()`, `static_cast<double>()`.

## Item 11: rand() → <random> — ACTIVE

**Fix:** Both `rand()/RAND_MAX` sites in getSplitCells() replaced with `thread_local std::mt19937` + `std::uniform_real_distribution`.

## Item 12: setenv/getenv → parameter — ACTIVE

**Fix:** Added `const std::string& firstFrameFile` parameter to `CellFactory::createCells()`. Removed `setenv()` from main.cpp and `getenv()` from CellFactory.cpp.

---

# Items 13-16, 18: Code Quality (size_t, pre-opt, structured bindings, YAML ref, using)

## Item 13: copyCellsForward int → size_t — ACTIVE

**Fix:** Changed parameter from `int to` to `size_t to` in both CellUniverse.hpp and CellUniverse.cpp.

## Item 14: Pre-opt params populated and passed — ACTIVE

**Fix:** Added `PreOptShape` struct and `preOptShapes` map populated before the optimization loop. `trySplitCell` now receives actual pre-optimization positions and radii instead of zeros.

## Item 15: Structured bindings — ACTIVE

**Fix:** Replaced `std::tie(child1, child2, valid, elongationRatio)` with `auto [child1, child2, valid, elongationRatio]` in trySplitCell.

## Item 16: YAML::Node by reference — ACTIVE

**Fix:** `SimulationConfig::explodeConfig(const YAML::Node node)` → `const YAML::Node& node`. Avoids copying YAML tree.

## Item 18: typedef → using — ACTIVE

**Fix:** All 8 `typedef` declarations in types.hpp replaced with `using` aliases.
