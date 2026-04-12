# Residual-Based Split Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PCA-elongation-driven P(split) with residual-based P(split) — split probability driven by unexplained signal in the current frame, not cell shape from the previous frame.

**Architecture:** Each iteration in the unified loop: pick a random cell, compute the positive residual (real - synth) in the cell's neighborhood. High residual = unexplained bright signal nearby = higher P(split). When splitting, place one daughter at the cell center and one at the residual centroid. After burn-in, validate both daughters individually reduce residual. Remove all hard gates (elongation threshold, daughter size ratio, daughter separation, volume floor). Keep only: cost threshold + residual validation.

**Tech Stack:** C++17, OpenCV (cv::Mat, cv::Rect), yaml-cpp

---

## File Structure

| File | Responsibility | Changes |
|------|---------------|---------|
| `includes/Frame.hpp` | Frame class declaration | Add `computeResidualForCell()` method |
| `src/Frame.cpp` | Frame implementation | Implement `computeResidualForCell()`, add `tryResidualSplit()` |
| `src/CellUniverse.cpp` | Optimize loop | Replace PCA-elongation P(split) with residual-based P(split), remove hard gates, call `tryResidualSplit()` |
| `includes/CellUniverse.hpp` | CellUniverse class | Remove `previousElongations` member (no longer needed) |
| `includes/ConfigTypes.hpp` | Config structs | Add `residual_split_threshold` to ProbabilityConfig |
| `config/config.yaml` | Runtime config | Add `residual_split_threshold`, lower `split_cost` |

---

## Task 1: Add `computeResidualForCell()` to Frame

**Files:**
- Modify: `C++/includes/Frame.hpp`
- Modify: `C++/src/Frame.cpp`

- [ ] **Step 1: Add declaration to Frame.hpp**

After the `computeElongationForCell` declaration, add:

```cpp
// Compute total positive residual (unexplained bright signal) in a cell's neighborhood.
// Returns: {totalResidual, centroidX, centroidY, centroidZ, numPixels}
struct ResidualInfo {
    double totalResidual = 0.0;
    float centroidX = 0, centroidY = 0, centroidZ = 0;
    int numPixels = 0;
};
ResidualInfo computeResidualForCell(size_t cellIdx, float searchRadius = 3.0f) const;
```

Put the `ResidualInfo` struct above the `class Frame` definition so it's accessible.

- [ ] **Step 2: Implement in Frame.cpp**

Add after `computeElongationForCell()`:

```cpp
ResidualInfo Frame::computeResidualForCell(size_t cellIdx, float searchRadius) const
{
    ResidualInfo info;
    if (cellIdx >= cells.size() || _realFrame.empty()) return info;

    auto params = cells[cellIdx].getCellParams();
    float maxR = params.majorRadius * searchRadius;

    // Bounding box in pixel space
    int minX = std::max(0, static_cast<int>(params.x - maxR));
    int maxX = std::min(_realFrame[0].cols - 1, static_cast<int>(params.x + maxR));
    int minY = std::max(0, static_cast<int>(params.y - maxR));
    int maxY = std::min(_realFrame[0].rows - 1, static_cast<int>(params.y + maxR));
    int minZ = std::max(0, static_cast<int>(params.z - maxR));
    int maxZ = std::min(static_cast<int>(_realFrame.size()) - 1, static_cast<int>(params.z + maxR));

    double sumX = 0, sumY = 0, sumZ = 0;

    for (int z = minZ; z <= maxZ; ++z) {
        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                float realVal = _realFrame[z].at<float>(y, x);
                float synthVal = _synthFrame[z].at<float>(y, x);
                float residual = realVal - synthVal;
                if (residual > 0.1f) {  // noise floor
                    info.totalResidual += residual;
                    sumX += x * residual;
                    sumY += y * residual;
                    sumZ += z * residual;
                    info.numPixels++;
                }
            }
        }
    }

    if (info.totalResidual > 0) {
        info.centroidX = static_cast<float>(sumX / info.totalResidual);
        info.centroidY = static_cast<float>(sumY / info.totalResidual);
        info.centroidZ = static_cast<float>(sumZ / info.totalResidual);
    }

    return info;
}
```

---

## Task 2: Add `tryResidualSplit()` to Frame

**Files:**
- Modify: `C++/includes/Frame.hpp`
- Modify: `C++/src/Frame.cpp`

- [ ] **Step 1: Add declaration to Frame.hpp**

```cpp
CostCallbackPair tryResidualSplit(size_t cellIndex, float residCentroidX, float residCentroidY,
                                   float residCentroidZ, float overlapWeight = 1000.0f);
```

- [ ] **Step 2: Implement in Frame.cpp**

This method:
1. Removes parent, creates two daughters (one at cell center, one at residual centroid)
2. Daughters get `cbrt(0.5) * parentR` radii (half volume each)
3. Runs 500-iter burn-in with overlap penalty
4. Returns costDiff + callback

```cpp
CostCallbackPair Frame::tryResidualSplit(size_t index, float residCentroidX, float residCentroidY,
                                          float residCentroidZ, float overlapWeight)
{
    if (index >= cells.size()) {
        return {0.0, [](bool) {}};
    }

    Spheroid oldCell = cells[index];
    auto parentParams = oldCell.getCellParams();

    // Daughters at half parent volume
    float volumeScale = std::cbrt(0.5f);
    float dMajorR = parentParams.majorRadius * volumeScale;
    float dMinorR = parentParams.minorRadius * volumeScale;

    // Daughter 1 at parent center, Daughter 2 at residual centroid
    SpheroidParams d1Params(parentParams.name + "0",
        parentParams.x, parentParams.y, parentParams.z,
        dMajorR, dMinorR, parentParams.theta_x, parentParams.theta_y, parentParams.theta_z,
        parentParams.brightness);
    SpheroidParams d2Params(parentParams.name + "1",
        residCentroidX, residCentroidY, residCentroidZ,
        dMajorR, dMinorR, parentParams.theta_x, parentParams.theta_y, parentParams.theta_z,
        parentParams.brightness);

    Spheroid child1(d1Params);
    Spheroid child2(d2Params);

    // Remove parent, add daughters
    cells.erase(cells.begin() + index);
    cells.push_back(child1);
    cells.push_back(child2);

    size_t d1Idx = cells.size() - 2;
    size_t d2Idx = cells.size() - 1;

    // Measure old state cost
    cells.pop_back(); cells.pop_back();
    cells.insert(cells.begin() + index, oldCell);
    double oldOverlap = computeOverlapPenalty(overlapWeight);
    double oldTotalCost = _currentCost + oldOverlap;
    cells.erase(cells.begin() + index);
    cells.push_back(child1); cells.push_back(child2);
    d1Idx = cells.size() - 2;
    d2Idx = cells.size() - 1;

    // Burn-in: optimize daughters with overlap penalty
    auto bestSynthFrame = generateSynthFrame();
    double bestImageCost = calculateCost(bestSynthFrame);
    auto savedSynthFrame = _synthFrame;
    _synthFrame = bestSynthFrame;

    const int BURN_IN_ITERATIONS = 500;
    int accepted = 0;
    for (int iter = 0; iter < BURN_IN_ITERATIONS; ++iter) {
        size_t dIdx = (iter % 2 == 0) ? d1Idx : d2Idx;
        Spheroid saved = cells[dIdx];
        double oldCellOverlap = computeOverlapForCell(dIdx, overlapWeight);
        cells[dIdx] = cells[dIdx].getPerturbedCell();
        double newCellOverlap = computeOverlapForCell(dIdx, overlapWeight);
        auto trialFrame = generateSynthFrameFast(saved, cells[dIdx]);
        double trialImageCost = calculateCost(trialFrame);
        double improvement = (trialImageCost + newCellOverlap) - (bestImageCost + oldCellOverlap);
        if (improvement < 0) {
            bestSynthFrame = trialFrame;
            bestImageCost = trialImageCost;
            _synthFrame = trialFrame;
            accepted++;
        } else {
            cells[dIdx] = saved;
        }
    }

    _synthFrame = savedSynthFrame;

    double bestOverlap = computeOverlapPenalty(overlapWeight);
    double bestTotalCost = bestImageCost + bestOverlap;
    double costDiff = bestTotalCost - oldTotalCost;

    std::cout << "[Residual Split Burn-in] " << oldCell.getName()
              << " burn_in_accepted=" << accepted << "/" << BURN_IN_ITERATIONS
              << " oldCost=" << oldTotalCost << " newCost=" << bestTotalCost
              << " diff=" << costDiff << std::endl;

    CallBackFunc callback = [this, bestSynthFrame, bestImageCost, oldCell, index](bool accept)
    {
        if (accept) {
            this->_synthFrame = bestSynthFrame;
            this->_currentCost = bestImageCost;
        } else {
            this->cells.pop_back();
            this->cells.pop_back();
            this->cells.insert(this->cells.begin() + index, oldCell);
        }
    };

    return {costDiff, callback};
}
```

---

## Task 3: Add config field for residual threshold

**Files:**
- Modify: `C++/includes/ConfigTypes.hpp`
- Modify: `C++/config/config.yaml`

- [ ] **Step 1: Add field to ProbabilityConfig**

In `ProbabilityConfig`, add:

```cpp
float residual_split_threshold = 50.0f;  // min total residual to consider splitting
```

Add YAML parsing:

```cpp
if (node["residual_split_threshold"]) residual_split_threshold = node["residual_split_threshold"].as<float>();
```

- [ ] **Step 2: Update config.yaml**

```yaml
prob:
  split: 0.03
  split_cost: 30              # lowered from 300 — residual validation replaces hard gates
  split_elongation_threshold: 1.5  # kept for trySplitCell internal PCA, not for P(split)
  overlap_penalty_weight: 1000.0
  split_burn_in_iterations: 500
  max_split_probability: 0.5
  residual_split_threshold: 50.0   # min positive residual sum to attempt split
```

---

## Task 4: Rewrite optimize() split logic — residual-based P(split)

**Files:**
- Modify: `C++/src/CellUniverse.cpp`
- Modify: `C++/includes/CellUniverse.hpp`

- [ ] **Step 1: Remove `previousElongations` from CellUniverse.hpp**

Remove the member:
```cpp
std::map<std::string, float> previousElongations;  // no longer needed
```

- [ ] **Step 2: Rewrite the split/perturb decision in optimize()**

Replace the entire P(split) computation block (prevElongation lookup, elongation threshold gate) and the post-split validation block (daughter size ratio, separation check, volume floor) with residual-based logic:

The new main loop body:

```cpp
for (size_t i = 0; i < totalIterations; ++i) {
    if (frame.cells.empty()) break;

    std::uniform_int_distribution<size_t> cellDist(0, frame.cells.size() - 1);
    size_t cellIdx = cellDist(gen);
    auto params = frame.cells[cellIdx].getCellParams();

    // Compute positive residual in cell's neighborhood
    // High residual = unexplained bright signal nearby = try split
    float pSplit = 0.0f;
    ResidualInfo residInfo;
    if (allowSplits && splitBlacklist.find(params.name) == splitBlacklist.end()) {
        residInfo = frame.computeResidualForCell(cellIdx);
        if (residInfo.totalResidual > config.prob.residual_split_threshold) {
            // Scale P(split) by how much residual exceeds threshold
            float residualRatio = static_cast<float>(residInfo.totalResidual / config.prob.residual_split_threshold);
            pSplit = std::min(config.prob.max_split_probability,
                              baseSplitProb + std::min(0.47f, (residualRatio - 1.0f) * 0.1f));
        }
    }

    if (pSplit > 0.0f && uniform01(gen) < pSplit) {
        // --- Try residual-based split ---
        splitAttempted++;
        auto result = frame.tryResidualSplit(cellIdx,
                                              residInfo.centroidX, residInfo.centroidY, residInfo.centroidZ,
                                              overlapWeight);
        double costDiff = result.first;
        auto callback = result.second;

        if (costDiff < -config.prob.split_cost) {
            // Post-split residual validation: both daughters must individually
            // explain some positive residual. If one daughter covers only background
            // (no residual to explain), reject — it's mimicking the parent.
            bool splitValid = true;
            size_t d1Idx = frame.cells.size() - 2;
            size_t d2Idx = frame.cells.size() - 1;
            auto res1 = frame.computeResidualForCell(d1Idx, 1.5f);
            auto res2 = frame.computeResidualForCell(d2Idx, 1.5f);

            // Both daughters should have LOW residual (they're explaining their region)
            // A daughter mimicking the parent leaves the other region with HIGH residual
            // Check: neither daughter should have more than 70% of the original residual
            if (res1.totalResidual > residInfo.totalResidual * 0.7 ||
                res2.totalResidual > residInfo.totalResidual * 0.7) {
                splitValid = false;
                std::cout << "[Split Reject Residual] " << params.name
                          << " d1Residual=" << res1.totalResidual
                          << " d2Residual=" << res2.totalResidual
                          << " originalResidual=" << residInfo.totalResidual << std::endl;
            }

            if (splitValid) {
                callback(true);
                splitAccepted++;
                splitBlacklist.insert(params.name + "0");
                splitBlacklist.insert(params.name + "1");
                std::cout << "[Split Accepted] " << params.name << " frame=" << displayFrame
                          << " iter=" << i << " diff=" << costDiff
                          << " residual=" << residInfo.totalResidual
                          << " P(split)=" << pSplit << std::endl;
            } else {
                callback(false);
                splitBlacklist.insert(params.name);
            }
        } else {
            callback(false);
            splitBlacklist.insert(params.name);
        }
    } else {
        // --- Try perturbation ---
        auto result = frame.perturbCell(cellIdx, overlapWeight);
        double costDiff = result.first;
        auto callback = result.second;

        if (costDiff < 0) {
            callback(true);
            perturbAccepted++;
            residSum += costDiff;
            residCount++;
        } else {
            callback(false);
        }
    }

    // Progress logging every 500 iterations
    if (i % 500 == 0 && i > 0) {
        double avgResid = residCount > 0 ? residSum / residCount : 0.0;
        std::cout << "Frame " << displayFrame << " iter=" << i
                  << " perturb_accepted=" << perturbAccepted
                  << " split_attempts=" << splitAttempted
                  << " split_accepted=" << splitAccepted
                  << " avg_resid=" << avgResid
                  << " cells=" << frame.cells.size() << std::endl;
        residSum = 0;
        residCount = 0;
    }
}
```

- [ ] **Step 3: Remove end-of-frame PCA elongation computation**

Remove the block that computes `previousElongations` at the end of `optimize()` — no longer needed since P(split) is driven by residual, not previous-frame elongation.

- [ ] **Step 4: Remove all old hard gates**

Remove from optimize():
- `daughterVolumeFloor` map and all its logic
- Daughter size ratio checks (d1Ratio, d2Ratio, sizeRatio)
- Daughter separation check (separationRatio)
- The `prevElongation` lookup block
- Pre-computed P(split) logging at frame start (replace with residual-based logging)

Keep:
- `splitBlacklist` (still useful — one attempt per cell per frame)
- `preOptShapes` (still useful for trySplitCell if needed as fallback)
- Volume cap at end of frame

---

## Task 5: Clean up and log

**Files:**
- Modify: `C++/docs/changelogv3.md`

- [ ] **Step 1: Add changelog entry**

Document the residual-based split detection as E22, covering:
- Removed PCA-elongation-driven P(split)
- Removed all daughter validation hard gates
- Added `computeResidualForCell()` and `tryResidualSplit()` to Frame
- P(split) now driven by current-frame residual
- Post-split validation: both daughters must reduce residual
- Added `residual_split_threshold` config field

---

## Summary

| What changed | Why |
|-------------|-----|
| P(split) from residual, not PCA elongation | Residual directly measures "is there unexplained signal?" — no historical dependency, no shape heuristic |
| `tryResidualSplit()` places daughter at residual centroid | Puts the daughter WHERE the unexplained signal is, not along PCA axis |
| Post-split residual validation | Both daughters must explain real signal — catches daughters-mimicking-parent |
| Removed hard gates (size ratio, separation, volume floor, elongation threshold) | These treated symptoms, not root cause. Residual validation replaces them all |
| Kept: split_cost threshold, splitBlacklist, volume cap | Still needed: cost threshold prevents noise-level splits, blacklist prevents repeated attempts, volume cap prevents inflation |
