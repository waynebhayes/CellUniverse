# Changelog: Brightness-Size Relationship Data Collection

## 2026-04-02

### F1: Per-cell brightness measurement from blurred-only images — **ACTIVE**

**Problem/Motivation:** Need to find the relationship `brightness = alpha * f(size)` where `alpha` is per-cell and `f(size) = R^k` for some universal constant k. This requires measuring actual cell brightness from the real images (blurred only, no sigmoid) at the optimizer's fitted cell positions across all frames.

**Approach:** 
1. Save blurred-only (pre-sigmoid) raw TIFF slices during image loading
2. After each frame's optimization, measure mean brightness inside each cell's 3D ellipsoid boundary from the blurred slices
3. Output `brightness_data.csv` with brightness + size metrics per cell per frame
4. Python script analyzes the data, searches for universal k, and plots results

**Files changed:** `Spheroid.hpp`, `Spheroid.cpp`, `CellUniverse.hpp`, `CellUniverse.cpp`, `main.cpp`, new `scripts/plot_brightness_analysis.py`

---

**File:** `C++/includes/Spheroid.hpp`

**Lines 108-113 (after):**
```cpp
        // Measure mean brightness from blurred-only (pre-sigmoid) raw slices.
        // Returns (meanBrightness, pixelCount, totalBrightness).
        // blurredSlices: raw TIFF slices after blur only (no sigmoid, no z-interpolation).
        // z_scaling: interpolation factor (e.g., 7) to map raw slice index to interpolated z.
        std::tuple<float, int, double> measureMeanBrightness(
            const std::vector<cv::Mat> &blurredSlices, int z_scaling) const;
```

**Effect:** Declares new public method on Spheroid for measuring brightness from raw blurred images.

---

**File:** `C++/src/Spheroid.cpp`

**Lines 267-301 (after):**
```cpp
std::tuple<float, int, double> Spheroid::measureMeanBrightness(
    const std::vector<cv::Mat> &blurredSlices, int z_scaling) const
{
    double totalBrightness = 0.0;
    int pixelCount = 0;

    std::array<double, 9> R_T;
    generateInverseRotationMatrix(R_T);

    const double invA2 = 1.0 / (a * a);
    const double invB2 = 1.0 / (b * b);
    const double invC2 = 1.0 / (c * c);

    for (size_t rawZ = 0; rawZ < blurredSlices.size(); ++rawZ) {
        float interpZ = static_cast<float>(rawZ * z_scaling);

        int minX, maxX, minY, maxY;
        if (!computeSliceBounds(blurredSlices[rawZ], interpZ, minX, maxX, minY, maxY)) continue;

        const cv::Mat &slice = blurredSlices[rawZ];
        scanSpheroidSlice(
            minX, maxX, minY, maxY,
            _position, static_cast<double>(interpZ),
            R_T, invA2, invB2, invC2,
            [&](int x, int y, double val) {
                if (val <= 1.0) {
                    totalBrightness += slice.at<float>(y, x);
                    pixelCount++;
                }
            });
    }

    float meanBrightness = pixelCount > 0 ? static_cast<float>(totalBrightness / pixelCount) : 0.0f;
    return {meanBrightness, pixelCount, totalBrightness};
}
```

**Effect:** Measures mean pixel brightness from blurred-only (pre-sigmoid) images within the cell's 3D ellipsoid boundary. Iterates over raw TIFF slices, maps each to interpolated z-space (`rawZ * z_scaling`), and uses the existing `scanSpheroidSlice` template to test each pixel against the rotated ellipsoid equation. Only counts pixels with `val <= 1.0` (inside boundary).

---

**File:** `C++/includes/CellUniverse.hpp`

**Line 24 (before):**
```cpp
std::vector<cv::Mat> loadFrame(const std::string &imageFile, BaseConfig &config);
```

**Line 24 (after):**
```cpp
std::vector<cv::Mat> loadFrame(const std::string &imageFile, BaseConfig &config,
                                std::vector<cv::Mat> *blurredOut = nullptr);
```

**Line 39 (after) — new method:**
```cpp
    void saveBrightnessData(int frameIndex);
```

**Line 54 (after) — new member:**
```cpp
   std::vector<std::vector<cv::Mat>> _blurredFrames;  // pre-sigmoid blurred raw slices per frame
```

**Effect:** `loadFrame()` gains optional output parameter for blurred-only slices. CellUniverse stores them and exposes `saveBrightnessData()`.

---

**File:** `C++/src/CellUniverse.cpp`

**Line 64-65 (before):**
```cpp
std::vector<cv::Mat> loadFrame(const std::string &imageFile, BaseConfig &config)
{
```

**Line 64-65 (after):**
```cpp
std::vector<cv::Mat> loadFrame(const std::string &imageFile, BaseConfig &config,
                                std::vector<cv::Mat> *blurredOut)
{
```

**Lines 125-131 (after) — new block before sigmoid:**
```cpp
        // Save blurred-only slices before sigmoid (for brightness analysis)
        if (blurredOut) {
            blurredOut->clear();
            blurredOut->reserve(processedZSlices.size());
            for (const auto &slice : processedZSlices) {
                blurredOut->push_back(slice.clone());
            }
        }
```

**Lines 200-204 (after) — constructor stores blurred frames:**
```cpp
        std::vector<cv::Mat> blurredSlices;
        std::vector<Image> real_frame;
        real_frame = loadFrame(imagePaths[i], config, &blurredSlices);
        _blurredFrames.push_back(std::move(blurredSlices));
```

**Lines 539-578 (after) — new method:**
```cpp
void CellUniverse::saveBrightnessData(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size()) return;
    if (static_cast<size_t>(frameIndex) >= _blurredFrames.size()) return;

    std::string path = outputPath + "/brightness_data.csv";
    bool first = (frameIndex == 0);
    std::ofstream file(path, first ? std::ios::trunc : std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << path << " for writing" << '\n';
        return;
    }

    if (first) {
        file << "frame,name,mean_brightness,total_brightness,pixel_count,"
             << "majorR,minorR,volume,cross_area" << '\n';
    }

    int displayFrame = firstFrame + frameIndex;
    Frame &frame = frames[frameIndex];
    const auto &blurredSlices = _blurredFrames[frameIndex];
    int z_scaling = config.simulation.z_scaling;

    for (const auto &cell : frame.cells) {
        auto [meanB, pixCount, totalB] = cell.measureMeanBrightness(blurredSlices, z_scaling);
        auto p = cell.getCellParams();
        double volume = (4.0 / 3.0) * M_PI * p.majorRadius * p.majorRadius * p.minorRadius;
        double crossArea = M_PI * p.majorRadius * p.majorRadius;

        file << displayFrame << "," << p.name << ","
             << meanB << "," << totalB << ","
             << pixCount << "," << p.majorRadius << ","
             << p.minorRadius << "," << volume << ","
             << crossArea << '\n';
    }

    std::cout << "[Brightness] Saved " << frame.cells.size() << " cells for frame "
              << displayFrame << " to " << path << std::endl;
}
```

**Effect:** After each frame's optimization, measures per-cell mean brightness from blurred-only images and writes to `output/brightness_data.csv`. CSV columns: frame, name, mean_brightness, total_brightness, pixel_count, majorR, minorR, volume, cross_area.

---

**File:** `C++/src/main.cpp`

**Line 227 (after):**
```cpp
        lineage.saveBrightnessData(frame);
```

**Effect:** Calls brightness data collection after each frame in the main loop.

---

**File:** `C++/scripts/plot_brightness_analysis.py` — **NEW FILE**

Python script for analyzing the brightness-size relationship:
- Reads `brightness_data.csv` from C++ output
- Plots: brightness vs volume, brightness vs majorR, brightness vs pixel count
- Searches for universal k by minimizing CV of `brightness / majorR^k` per cell
- Plots CV-vs-k curve, per-cell alpha values, ratio stability over time
- Standalone mode: can measure directly from TIFFs + cells.csv with blur only (no sigmoid)

**Usage:**
```bash
# After C++ run (reads output/brightness_data.csv):
python scripts/plot_brightness_analysis.py output/brightness_data.csv

# Standalone (reads TIFFs + cells.csv directly):
python scripts/plot_brightness_analysis.py --standalone --cells_csv output/cells.csv --input_dir data/input/original_data --blur_sigma 3.0
```

---

### Output Format

`brightness_data.csv` columns:
| Column | Description |
|--------|-------------|
| frame | Frame number (display frame, e.g., 1-19) |
| name | Cell name (UUID or daughter name with 0/1 suffix) |
| mean_brightness | Mean pixel value from blurred-only image inside cell boundary |
| total_brightness | Sum of all pixel values inside cell boundary |
| pixel_count | Number of pixels inside 3D ellipsoid (proportional to rendered volume) |
| majorR | Cell major radius at this frame |
| minorR | Cell minor radius at this frame |
| volume | Analytical volume: (4/3)πR²r |
| cross_area | Central cross-section: πR² |

### Memory Notes

- Stores 33 raw blurred slices per frame (~1MB each = ~33MB per frame)
- For 19 frames: ~627MB additional memory
- Acceptable for research workstation; can be optimized later if needed

---

### F2: Remove sigmoid preprocessing — **ACTIVE**

**Problem/Motivation:** The professor's direction: remove all preprocessing except initial Gaussian blur. The sigmoid was compressing the brightness range to near-binary (cells→1.0, bg→0.0), which destroyed the natural brightness-size relationship we need to measure. The optimizer should work on blurred images directly so we can observe and model how brightness varies with cell size.

**Files changed:** `CellUniverse.cpp`, `config.yaml`

---

**File:** `C++/src/CellUniverse.cpp`

**Lines 31-39 (before) — REMOVED:**
```cpp
// Apply sigmoid contrast: output = 1 / (1 + exp(-k * (input - center)))
// Makes cells near-white and background near-black, giving the L2 cost
// function clear gradient signal for cell boundaries.
static void applySigmoid(cv::Mat &image, float k, float center)
{
    image.forEach<float>([k, center](float &pixel, const int *) {
        pixel = 1.0f / (1.0f + std::exp(-k * (pixel - center)));
    });
}
```

**Effect:** Removed the `applySigmoid()` static function entirely. No longer needed.

---

**Lines 89-106 (before — sigmoid calibration):**
```cpp
        // --- Calibrate sigmoid center from background zone, then apply sigmoid ---
        float sigmoidCenter = config.simulation.sigmoid_center;
        if (!processedZSlices.empty()) {
            // ... calibration zone measurement ...
            float bgMean = static_cast<float>(cv::mean(processedZSlices[calZ](roi))[0]);
            sigmoidCenter = bgMean + config.simulation.sigmoid_center_offset;
            std::cout << "[Calibration] bgMean=" << bgMean
                      << " sigmoidCenter=" << sigmoidCenter
                      << " sigmoid_k=" << config.simulation.sigmoid_k << std::endl;
        }
```

**Lines 89-106 (after — background_color calibration):**
```cpp
        // --- Calibrate background_color from cell-free zone ---
        // Measure mean brightness in a known-empty region to set background_color.
        // No sigmoid — the optimizer works on blurred images directly.
        if (!processedZSlices.empty()) {
            // ... same calibration zone measurement ...
            float bgMean = static_cast<float>(cv::mean(processedZSlices[calZ](roi))[0]);
            config.simulation.background_color = bgMean;
            std::cout << "[Calibration] background_color=" << bgMean << std::endl;
        }
```

**Effect:** Repurposed calibration zone to directly set `background_color` to the measured mean, instead of computing a sigmoid center. The optimizer's synthetic background now matches the real image background.

---

**Lines 133-138 (before) — REMOVED:**
```cpp
        // Apply sigmoid to all slices — amplifies cell/background contrast
        if (config.simulation.sigmoid_k > 0) {
            for (auto &slice : processedZSlices) {
                applySigmoid(slice, config.simulation.sigmoid_k, sigmoidCenter);
            }
        }
```

**Effect:** Removed sigmoid application. Image pipeline is now: TIFF → BGR→Gray → normalize [0,1] → Gaussian blur → z-interpolation. No contrast manipulation.

---

**File:** `C++/config/config.yaml`

**Before:**
```yaml
simulation:
  background_color: 0.0  # post-sigmoid background is near 0
  cell_color: 0.95       # near 1.0 to match sharp sigmoid
  blur_sigma: 3.0
  sigmoid_k: 75.0
  sigmoid_center: 0.445
  sigmoid_center_offset: 0.047
```

**After:**
```yaml
simulation:
  background_color: 0.0  # auto-calibrated from cell-free zone at runtime
  cell_color: 0.5        # placeholder — will be determined by brightness-size analysis
  blur_sigma: 3.0
  # sigmoid removed — optimizer works on blurred images directly
  # sigmoid_k: 0
  # sigmoid_center: 0.445
  # sigmoid_center_offset: 0.047
```

**Effect:** `cell_color` set to 0.5 as placeholder. `background_color` is auto-calibrated at runtime. Sigmoid config fields commented out (struct defaults remain in ConfigTypes.hpp but are unused).

---

### Impact Summary

| Before (sigmoid) | After (blur only) |
|---|---|
| Cells → ~1.0, Background → ~0.0 | Cells → ~0.4-0.7, Background → ~0.2-0.4 |
| `cell_color = 0.95` | `cell_color = 0.5` (placeholder) |
| `background_color = 0.0` | `background_color = auto-calibrated` |
| L2 cost sees binary-like contrast | L2 cost sees natural brightness gradients |
| Brightness-size relationship destroyed | Brightness-size relationship preserved |

### ConfigTypes.hpp — Dead Fields

The following fields remain in `SimulationConfig` with defaults but are not read from YAML or used in code:
- `sigmoid_k` (default 75.0)
- `sigmoid_center` (default 0.445)
- `sigmoid_center_offset` (default 0.047)

These can be cleaned up in a future commit.

---

### F3: Review-driven bug fixes — **ACTIVE**

Three review agents (C++ reviewer, architecture reviewer, Python reviewer) identified bugs and improvements.

**Bug fixes applied:**

---

**File:** `C++/src/CellUniverse.cpp`

**Fix 1: Removed shadowed `cellVol` variable (line ~288)**
Inner `double cellVol = ...` shadowed outer declaration at line 281. Both computed identical values. Removed the inner one; the outer `cellVol` is now used throughout the scope.

**Fix 2: Removed stale post-split residual validation (lines ~307-322)**

The post-split residual validation was computing daughter residuals against `_synthFrame` which still showed the *parent* (pre-split). `tryResidualSplit` restores `_synthFrame = savedSynthFrame` before returning, so `computeResidualForCell()` for the daughters was measuring against a frame that didn't include them — making the check unreliable. Removed entirely; cost threshold (`split_cost`) is the primary gate.

**Fix 3: Removed dead `capParams` code (lines ~408-410)**
`std::unordered_map<std::string, float> capParams` was populated but never used. The actual volume cap uses direct `SpheroidParams` construction. Removed.

---

**File:** `C++/scripts/plot_brightness_analysis.py`

**Fix 4: 16-bit TIFF normalization**
Was dividing by 255.0 unconditionally. Microscopy TIFFs can be 16-bit (max 65535). Now detects dtype and normalizes correctly.

**Fix 5: Vectorized inner loop**
Replaced per-pixel Python loop with numpy broadcasting (meshgrid + matrix multiply). ~100x speedup for large cells.

**Fix 6: Mutable default argument**
`find_best_k(k_range=np.arange(...))` — default arg evaluated once at import. Changed to `None` sentinel.

**Fix 7: `plt.close()` after each figure**
Prevents memory leak from accumulated matplotlib figures.

**Fix 8: `short()` edge case**
Was appending `..` even for short names. Now only truncates names > 10 chars.

**Fix 9: Dual-parameter k search**
Architecture reviewer noted: physically, mean brightness depends on optical path length (proportional to `minorR`), not just `majorR`. Added Plot 5 searching `brightness / (majorR^k1 * minorR^k2)` for both exponents. Summary table now compares single vs dual parameter fits.

---

### F4: Switch split detection from residual-based to PCA elongation-based — **ACTIVE**

**Problem/Motivation:** Professor's direction: use PCA elongation to drive split probability. Always have a base 0.03 chance; increase based on elongation. No blocking of split attempts.

**Files changed:** `CellUniverse.cpp`, `CellUniverse.hpp`

**CellUniverse.cpp optimize() — Split logic replaced (lines ~259-330):**

Before (residual-based):
- P(split) driven by `computeResidualForCell()` normalized by volume
- Used `tryResidualSplit()` placing daughter at residual centroid
- Split blacklist prevented retries after failure
- `recentDaughters` blocked next-frame splits

After (PCA elongation-based):
- `P(split) = min(0.5, 0.03 + max(0, 1 - 1/elongation))`
- Uses `trySplitCell()` with PCA-based daughter placement via `getSplitCells()`
- **No blocking** — any cell can attempt splits every iteration
- Elongation cached lazily (one PCA scan per cell per frame via `elongationCache`)
- Cost threshold (`split_cost`) is the sole acceptance gate
- Pre-opt shapes passed for stable daughter sizing

**CellUniverse.hpp — Removed `recentDaughters` member + `<set>` include**

---

### F5: Background subtraction in loadFrame — **ACTIVE**

**Problem/Motivation:** Improve contrast for L2 cost without sigmoid. Measure background mean from calibration zone and subtract from all slices.

**CellUniverse.cpp loadFrame() — Pipeline changed (lines ~89-126):**

Before: Calibration zone set `background_color` to match real bg mean. No image modification.

After:
1. Save blurred slices to `blurredOut` BEFORE subtraction (preserves raw brightness for analysis)
2. Measure `bgMean` from calibration zone
3. Subtract `bgMean` from all slices, clamp to >= 0
4. Set `background_color = 0.0`

Pipeline: TIFF → Gray → [0,1] → blur → save blurred → subtract bg → z-interpolate

**config.yaml:** `cell_color: 0.2`, `background_color: 0.0`

---

### F6: Performance optimizations — delta cost + z-range render — **ACTIVE**

**Problem/Motivation:** Each frame was very slow. Profiling found `calculateCost()` scanning all 225 z-slices even when only ~30-60 were modified by `generateSynthFrameFast()`.

**Frame.cpp perturbCell() (lines ~194-209) — Delta cost:**

Before:
```cpp
auto newSynthFrame = generateSynthFrameFast(oldCell, cells[index]);
double newImageCost = calculateCost(newSynthFrame); // ALL 225 slices
```

After:
```cpp
auto newSynthFrame = generateSynthFrameFast(oldCell, cells[index]);
// Compute z-bounds from old/new cell bounding box
int zMin = ..., zMax = ...;
double costDelta = 0.0;
for (int i = zMin; i <= zMax; ++i) {
    costDelta += cv::norm(real[i], newSynth[i], L2) - cv::norm(real[i], oldSynth[i], L2);
}
double newImageCost = _currentCost + costDelta;
```

Effect: L2 computed over ~30-60 affected slices instead of 225. **4-7x speedup** on cost calculation per perturbation.

**Frame.cpp trySplitCell() (lines ~505-580) — Z-range render + delta cost in burn-in:**

Before: `generateSynthFrame()` rendered all 225 slices at burn-in start. 500 burn-in iterations each called `calculateCost()` over all 225 slices.

After: Z-range render only re-renders slices in parent's bounding box (~60 slices). Burn-in uses same delta cost as perturbCell. **4-7x speedup** on 500-iteration burn-in.

**Spheroid.cpp draw() (lines ~202-213) — Row pointer access:**

Before: `image.at<float>(y, x)` — bounds-checked per pixel
After: `image.ptr<float>(y)[x]` — direct row pointer, cached per row. ~5-10% speedup on rendering.
