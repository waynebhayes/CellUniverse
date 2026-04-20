#include "../includes/CellUniverse.hpp"
#include "../includes/ImageHandler.hpp"
#include <set>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>

namespace utils
{
    template <typename T>
    void printMat(const cv::Mat &mat)
    {
        // Check if the matrix is empty
        if (mat.empty())
        {
            std::cout << "The matrix is empty." << '\n';
            return;
        }

        // Iterate over matrix rows
        for (int i = 0; i < mat.rows; ++i)
        {
            for (int j = 0; j < mat.cols; ++j)
            {
                // Print each element. Use mat.at<T>(i,j) to access the element at [i,j] and cast it to the type T.
                // The type T should match the type of the elements in the matrix.
                std::cout << mat.at<T>(i, j) << " ";
            }
            std::cout << '\n'; // Newline for each row
        }
    }
}

static float computeStackMean(const std::vector<cv::Mat> &stack)
{
    double sum = 0.0;
    double count = 0.0;

    for (const auto &slice : stack) {
        if (slice.empty()) {
            continue;
        }
        sum += cv::sum(slice)[0];
        count += static_cast<double>(slice.total());
    }

    return (count > 0.0) ? static_cast<float>(sum / count) : 0.0f;
}

static float computeMeanOfTopFraction(std::vector<float> values, float topFraction)
{
    if (values.empty()) {
        return 0.0f;
    }

    const float clampedFraction = std::clamp(topFraction, 0.0f, 1.0f);
    if (clampedFraction <= 0.0f) {
        return 0.0f;
    }

    const size_t selectedCount = std::max<size_t>(
        1, static_cast<size_t>(std::ceil(clampedFraction * static_cast<float>(values.size()))));
    const size_t thresholdIndex = values.size() - selectedCount;

    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(thresholdIndex),
                     values.end());

    double sum = 0.0;
    for (size_t i = thresholdIndex; i < values.size(); ++i) {
        sum += values[i];
    }

    return static_cast<float>(sum / selectedCount);
}

static void scaleStackBrightness(std::vector<cv::Mat> &stack, float scale)
{
    if (std::abs(scale - 1.0f) <= 1e-6f) {
        return;
    }

    for (auto &slice : stack) {
        if (slice.empty()) {
            continue;
        }
        slice *= scale;
        cv::min(slice, 1.0f, slice);
        cv::max(slice, 0.0f, slice);
    }
}

// === Helpers ported from yp_fix_mask_04172026 (commits 25c5923, b575246) ===
// Used by the new constructor's global-percentile normalization step.

static float computeStackPercentile(const std::vector<cv::Mat> &stack, float percentile)
{
    std::vector<float> values;
    size_t totalCount = 0;
    for (const auto &slice : stack) {
        if (!slice.empty()) totalCount += static_cast<size_t>(slice.total());
    }
    values.reserve(totalCount);
    for (const auto &slice : stack) {
        if (slice.empty()) continue;
        for (int y = 0; y < slice.rows; ++y) {
            const float *row = slice.ptr<float>(y);
            values.insert(values.end(), row, row + slice.cols);
        }
    }
    if (values.empty()) return 0.0f;
    const float clamped = std::clamp(percentile, 0.0f, 1.0f);
    const size_t index = static_cast<size_t>(
        std::floor(clamped * static_cast<float>(values.size() - 1)));
    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(index),
                     values.end());
    return values[index];
}

static float computeStackMax(const std::vector<cv::Mat> &stack)
{
    float maxValue = 0.0f;
    bool foundValue = false;
    for (const auto &slice : stack) {
        if (slice.empty()) continue;
        double sliceMin = 0.0, sliceMax = 0.0;
        cv::minMaxLoc(slice, &sliceMin, &sliceMax);
        if (!foundValue || static_cast<float>(sliceMax) > maxValue) {
            maxValue = static_cast<float>(sliceMax);
            foundValue = true;
        }
    }
    return foundValue ? maxValue : 0.0f;
}

static void normalizeStackToSharedScale(std::vector<cv::Mat> &stack,
                                        float lowReference,
                                        float highReference,
                                        float hardMax)
{
    if (hardMax > 0.0f) highReference = std::min(highReference, hardMax);
    const float denominator = highReference - lowReference;
    if (denominator <= 1e-6f) {
        for (auto &slice : stack) {
            if (!slice.empty()) slice.setTo(0.0f);
        }
        return;
    }
    for (auto &slice : stack) {
        if (slice.empty()) continue;
        if (hardMax > 0.0f) cv::min(slice, hardMax, slice);
        slice -= lowReference;
        slice *= (1.0f / denominator);
        cv::min(slice, 1.0f, slice);
        cv::max(slice, 0.0f, slice);
    }
}

static void exportPreprocessedStack(const std::vector<cv::Mat> &stack,
                                    const fs::path &baseOutputDir,
                                    const fs::path &framePath)
{
    const fs::path frameOutputDir = baseOutputDir / "preprocessed" / framePath.stem();
    fs::create_directories(frameOutputDir);

    for (size_t i = 0; i < stack.size(); ++i) {
        if (stack[i].empty()) {
            continue;
        }

        cv::Mat outputImage;
        if (stack[i].depth() != CV_8U) {
            stack[i].convertTo(outputImage, CV_8U, 255.0);
        } else {
            outputImage = stack[i].clone();
        }

        const fs::path outputFile = frameOutputDir / (std::to_string(i) + ".png");
        cv::imwrite(outputFile.string(), outputImage);
    }
}

// === Signal-guided perturbation helpers (yp ffc1917) ===
//
// Detect bright clusters in the real frame and use them as teleport
// targets for cells that get stuck on local optima. Per-frame init only;
// the actual cell teleport happens in Frame::perturbCell when
// useSignalGuidance=true.

struct BrightBox {
    int ix = 0;
    int iy = 0;
    int iz = 0;
    cv::Point3f center{0.0f, 0.0f, 0.0f};
    float brightness = 0.0f;
    int voxels = 0;
};

static int chooseNearestDivisorSize(int axisLength, float targetSize)
{
    if (axisLength <= 1) return 1;
    const float clampedTarget = std::clamp(targetSize, 1.0f, static_cast<float>(axisLength));
    int bestDivisor = 1;
    float bestDistance = std::abs(static_cast<float>(bestDivisor) - clampedTarget);
    for (int d = 1; d <= axisLength; ++d) {
        if (axisLength % d != 0) continue;
        const float distance = std::abs(static_cast<float>(d) - clampedTarget);
        if (distance < bestDistance ||
            (std::abs(distance - bestDistance) <= 1e-6f && d > bestDivisor)) {
            bestDivisor = d;
            bestDistance = distance;
        }
    }
    return bestDivisor;
}

static std::vector<Frame::SignalCenter> localizeSignalCentersForFrame(
    const Frame &frame,
    const BaseConfig &config,
    int displayFrame)
{
    std::vector<Frame::SignalCenter> centers;
    const auto &realFrame = frame.getRealFrame();
    if (realFrame.empty() || !config.cell) return centers;

    const int sizeX = realFrame[0].cols;
    const int sizeY = realFrame[0].rows;
    const int sizeZ = static_cast<int>(realFrame.size());
    if (sizeX <= 0 || sizeY <= 0 || sizeZ <= 0) return centers;

    const float boxScale = std::max(0.1f, config.simulation.signal_guided_box_size_scale);
    const float maxDiamX = 2.0f * static_cast<float>(config.cell->maxARadius);
    const float maxDiamY = 2.0f * static_cast<float>(
        config.cell->maxBRadius > 0.0 ? config.cell->maxBRadius : config.cell->maxARadius);
    const float maxDiamZ = 2.0f * static_cast<float>(config.cell->maxCRadius);
    const float targetBoxX = std::max(1.0f, (maxDiamX / 3.0f) * boxScale);
    const float targetBoxY = std::max(1.0f, (maxDiamY / 3.0f) * boxScale);
    const float targetBoxZ = std::max(1.0f, (maxDiamZ / 3.0f) * boxScale);

    const int boxSizeX = chooseNearestDivisorSize(sizeX, targetBoxX);
    const int boxSizeY = chooseNearestDivisorSize(sizeY, targetBoxY);
    const int boxSizeZ = chooseNearestDivisorSize(sizeZ, targetBoxZ);
    const int gridX = std::max(1, sizeX / boxSizeX);
    const int gridY = std::max(1, sizeY / boxSizeY);
    const int gridZ = std::max(1, sizeZ / boxSizeZ);

    const float backgroundValue = frame.getBackgroundValue();
    const float minDelta = std::max(0.0f, config.simulation.signal_guided_min_box_brightness_delta);

    std::vector<BrightBox> boxes;
    boxes.reserve(static_cast<size_t>(gridX) * gridY * gridZ);
    for (int iz = 0; iz < gridZ; ++iz) {
        const int z0 = iz * boxSizeZ;
        const int z1 = z0 + boxSizeZ;
        for (int iy = 0; iy < gridY; ++iy) {
            const int y0 = iy * boxSizeY;
            const int y1 = y0 + boxSizeY;
            for (int ix = 0; ix < gridX; ++ix) {
                const int x0 = ix * boxSizeX;
                const int x1 = x0 + boxSizeX;
                double sum = 0.0;
                int voxels = 0;
                for (int z = z0; z < z1; ++z) {
                    for (int y = y0; y < y1; ++y) {
                        const float *row = realFrame[z].ptr<float>(y);
                        for (int x = x0; x < x1; ++x) {
                            sum += row[x];
                            ++voxels;
                        }
                    }
                }
                if (voxels <= 0) continue;
                const float meanBrightness = static_cast<float>(sum / static_cast<double>(voxels));
                if (meanBrightness <= backgroundValue + minDelta) continue;
                boxes.push_back({
                    ix, iy, iz,
                    cv::Point3f(
                        static_cast<float>(x0 + x1 - 1) * 0.5f,
                        static_cast<float>(y0 + y1 - 1) * 0.5f,
                        static_cast<float>(z0 + z1 - 1) * 0.5f),
                    meanBrightness, voxels
                });
            }
        }
    }

    std::sort(boxes.begin(), boxes.end(),
              [](const BrightBox &a, const BrightBox &b) { return a.brightness > b.brightness; });

    std::map<std::tuple<int, int, int>, size_t> boxIndex;
    for (size_t i = 0; i < boxes.size(); ++i) boxIndex[{boxes[i].ix, boxes[i].iy, boxes[i].iz}] = i;

    std::vector<char> visited(boxes.size(), 0);
    float maxCenterBrightness = backgroundValue;
    for (size_t seed = 0; seed < boxes.size(); ++seed) {
        if (visited[seed]) continue;
        std::vector<size_t> stack{seed};
        visited[seed] = 1;
        double weightSum = 0.0, xSum = 0.0, ySum = 0.0, zSum = 0.0, brightnessSum = 0.0;
        int clusterBoxes = 0;
        while (!stack.empty()) {
            const size_t current = stack.back();
            stack.pop_back();
            const BrightBox &box = boxes[current];
            const float weight = std::max(1e-6f, box.brightness - backgroundValue);
            weightSum += weight;
            xSum += weight * static_cast<double>(box.center.x);
            ySum += weight * static_cast<double>(box.center.y);
            zSum += weight * static_cast<double>(box.center.z);
            brightnessSum += static_cast<double>(box.brightness);
            ++clusterBoxes;
            for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        auto it = boxIndex.find({box.ix + dx, box.iy + dy, box.iz + dz});
                        if (it == boxIndex.end()) continue;
                        const size_t neighbor = it->second;
                        if (visited[neighbor]) continue;
                        visited[neighbor] = 1;
                        stack.push_back(neighbor);
                    }
        }
        if (clusterBoxes <= 0 || weightSum <= 0.0) continue;
        Frame::SignalCenter center;
        center.position = cv::Point3f(
            static_cast<float>(xSum / weightSum),
            static_cast<float>(ySum / weightSum),
            static_cast<float>(zSum / weightSum));
        center.brightness = static_cast<float>(brightnessSum / static_cast<double>(clusterBoxes));
        center.boxes = clusterBoxes;
        centers.push_back(center);
        maxCenterBrightness = std::max(maxCenterBrightness, center.brightness);
    }

    for (auto &center : centers) {
        const float normalized = (maxCenterBrightness > backgroundValue + 1e-6f)
            ? std::clamp((center.brightness - backgroundValue) /
                         (maxCenterBrightness - backgroundValue), 0.0f, 1.0f)
            : 0.0f;
        center.sigmaScale = std::max(
            config.simulation.signal_guided_min_sigma_scale,
            1.0f - normalized * (1.0f - config.simulation.signal_guided_min_sigma_scale));
    }

    std::sort(centers.begin(), centers.end(),
              [](const Frame::SignalCenter &a, const Frame::SignalCenter &b) {
                  return a.brightness > b.brightness;
              });

    std::cout << "[Signal Centers] frame " << displayFrame
              << " enabled=" << config.simulation.signal_guided_position_enabled
              << " boxSize=(" << boxSizeX << "," << boxSizeY << "," << boxSizeZ << ")"
              << " grid=(" << gridX << "," << gridY << "," << gridZ << ")"
              << " background=" << backgroundValue
              << " keptBoxes=" << boxes.size()
              << " clusters=" << centers.size() << '\n';
    for (size_t i = 0; i < centers.size(); ++i) {
        const auto &c = centers[i];
        std::cout << "  [Signal Center] idx=" << i
                  << " pos=(" << c.position.x << "," << c.position.y << "," << c.position.z << ")"
                  << " brightness=" << c.brightness
                  << " sigmaScale=" << c.sigmaScale
                  << " boxes=" << c.boxes << '\n';
    }
    return centers;
}

static float estimateAdaptiveBackgroundFromFrame(const Frame &frame,
                                                 const SimulationConfig &simulationConfig)
{
    const auto &realFrame = frame.getRealFrame();
    if (realFrame.empty()) {
        return 0.0f;  // post-sigmoid background invariant
    }

    const float expandFactor = std::max(1.0f, simulationConfig.adaptive_background_expand_factor);
    std::vector<cv::Mat> exclusionMask;
    exclusionMask.reserve(realFrame.size());
    for (const auto &slice : realFrame) {
        exclusionMask.emplace_back(cv::Mat::zeros(slice.size(), CV_32F));
    }

    for (const auto &cell : frame.cells) {
        auto params = cell.getCellParams();
        params.aRadius *= expandFactor;
        params.bRadius     *= expandFactor;
        params.cRadius *= expandFactor;
        Ellipsoid expandedCell(params);
        for (size_t z = 0; z < exclusionMask.size(); ++z) {
            expandedCell.draw(exclusionMask[z], simulationConfig, static_cast<float>(z));
        }
    }

    std::vector<float> backgroundCandidates;
    for (size_t z = 0; z < realFrame.size(); ++z) {
        const cv::Mat &slice = realFrame[z];
        const cv::Mat &mask = exclusionMask[z];
        for (int y = 0; y < slice.rows; ++y) {
            const float *sliceRow = slice.ptr<float>(y);
            const float *maskRow = mask.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x) {
                if (maskRow[x] <= 0.0f) {
                    backgroundCandidates.push_back(sliceRow[x]);
                }
            }
        }
    }

    if (backgroundCandidates.empty()) {
        return 0.0f;  // post-sigmoid background invariant
    }

    return computeMeanOfTopFraction(backgroundCandidates, simulationConfig.adaptive_background_top_fraction);
}

// Preprocessing moved to ImageHandler. Single preprocessed stack via
// ImageHandler::loadFrame.
CellUniverse::CellUniverse(std::map<std::string, std::vector<Ellipsoid>> initialCells,
                           PathVec imagePaths,
                           BaseConfig &config,
                           std::string outputPath,
                           int firstFrame,
                           int continueFrom)
: config(config), outputPath(outputPath), firstFrame(firstFrame)
{
    // Constructor ported from yp_fix_mask_04172026 (commit 25c5923):
    // 4-pass parallel pipeline with global percentile-based intensity
    // normalization. Replaces our prior single-pass per-frame loadFrame
    // + brightness-mean alignment which produced inverted images when
    // combined with yp's preprocessing CODE (yp's iterative loop assumes
    // all frames have already been mapped to a shared low/high scale).
    std::vector<std::vector<cv::Mat>> loadedFrames(imagePaths.size());
    std::vector<std::ostringstream> rawLoadLogs(imagePaths.size());
    std::vector<std::ostringstream> preprocessLogs(imagePaths.size());

    // Pass 1: parallel raw-frame load (no preprocessing yet). Logs are
    // buffered per-frame and flushed in order after the parallel region.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(imagePaths.size()); ++i)
    {
        std::vector<cv::Mat> real_frame =
            ImageHandler::loadRawFrame(imagePaths[static_cast<size_t>(i)].string(),
                                       config,
                                       &rawLoadLogs[static_cast<size_t>(i)]);
        loadedFrames[static_cast<size_t>(i)] = std::move(real_frame);
    }
    for (auto &buf : rawLoadLogs) {
        const std::string s = buf.str();
        if (!s.empty()) std::cout << s;
    }

    // Compute global low/high reference percentiles across ALL loaded slices.
    // This shared scale is what makes yp's preprocessing iterative loop
    // converge correctly (without it, every frame uses its own arbitrary
    // scale and the iterative contrast scoring wanders off-target).
    std::vector<cv::Mat> allLoadedSlices;
    size_t totalSlices = 0;
    for (const auto &frame : loadedFrames) totalSlices += frame.size();
    allLoadedSlices.reserve(totalSlices);
    for (const auto &frame : loadedFrames) {
        for (const auto &slice : frame) allLoadedSlices.push_back(slice);
    }

    float globalLowReference = computeStackPercentile(
        allLoadedSlices, config.simulation.global_intensity_scale_low_percentile);
    float globalHighReference = computeStackPercentile(
        allLoadedSlices, config.simulation.global_intensity_scale_high_percentile);
    if (config.simulation.global_intensity_hard_max > 0.0f &&
        globalHighReference > config.simulation.global_intensity_hard_max) {
        globalHighReference = config.simulation.global_intensity_hard_max;
    }

    if (globalHighReference <= globalLowReference + 1e-6f) {
        const float fallback = computeStackMax(allLoadedSlices);
        if (fallback > globalLowReference + 1e-6f) {
            globalHighReference = fallback;
            if (config.simulation.global_intensity_hard_max > 0.0f) {
                globalHighReference = std::min(globalHighReference,
                                               config.simulation.global_intensity_hard_max);
            }
            std::cout << "[Global Scale Warning] percentile range collapsed;"
                      << " using stack max fallback low_ref=" << globalLowReference
                      << " high_ref=" << globalHighReference << '\n';
        } else {
            globalLowReference = 0.0f;
            globalHighReference = 1.0f;
            if (config.simulation.global_intensity_hard_max > 0.0f) {
                globalHighReference = std::min(globalHighReference,
                                               config.simulation.global_intensity_hard_max);
            }
            std::cout << "[Global Scale Warning] percentile range collapsed; stack max unusable;"
                      << " falling back to low_ref=" << globalLowReference
                      << " high_ref=" << globalHighReference << '\n';
        }
    }
    std::cout << "[Global Scale] low_percentile="
              << config.simulation.global_intensity_scale_low_percentile
              << " high_percentile="
              << config.simulation.global_intensity_scale_high_percentile
              << " low_ref=" << globalLowReference
              << " high_ref=" << globalHighReference
              << " hard_max=" << config.simulation.global_intensity_hard_max << '\n';

    // Pass 2: serially normalize each loaded frame to the shared scale.
    for (size_t i = 0; i < imagePaths.size(); ++i)
    {
        std::vector<cv::Mat> &real_frame = loadedFrames[i];
        normalizeStackToSharedScale(real_frame,
                                    globalLowReference,
                                    globalHighReference,
                                    config.simulation.global_intensity_hard_max);
        std::cout << "[Global Scale] frame=" << imagePaths[i].filename().string()
                  << " mean=" << computeStackMean(real_frame)
                  << " low_ref=" << globalLowReference
                  << " high_ref=" << globalHighReference
                  << " hard_max=" << config.simulation.global_intensity_hard_max << '\n';
    }

    // Pass 3: parallel preprocessing per frame using the shared-scale stacks.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(imagePaths.size()); ++i)
    {
        loadedFrames[static_cast<size_t>(i)] =
            ImageHandler::preprocessLoadedFrame(
                loadedFrames[static_cast<size_t>(i)],
                imagePaths[static_cast<size_t>(i)].string(),
                config,
                &preprocessLogs[static_cast<size_t>(i)]);
    }
    for (auto &buf : preprocessLogs) {
        const std::string s = buf.str();
        if (!s.empty()) std::cout << s;
    }

    // Pass 4: export + construct Frame objects.
    for (size_t i = 0; i < imagePaths.size(); ++i)
    {
        std::vector<cv::Mat> &real_frame = loadedFrames[i];

        if (config.simulation.export_preprocessed_images) {
            exportPreprocessedStack(real_frame, fs::path(outputPath), imagePaths[i]);
        }
        if (config.simulation.quit_after_preprocessing) {
            continue;
        }

        config.simulation.z_slices = real_frame.size();
        Ellipsoid::cellConfig.maxZ = static_cast<float>(real_frame.size()) - 1.0f;

        fs::path path(imagePaths[i]);
        std::string file_name = path.filename();

        if ((continueFrom == -1 || i < continueFrom) && initialCells.find(file_name) != initialCells.end())
        {
            const std::vector<Ellipsoid> &cells = initialCells.at(file_name);
            frames.emplace_back(real_frame, config.simulation, cells, outputPath, file_name);
        }
        else
        {
            frames.emplace_back(real_frame, config.simulation, std::vector<Ellipsoid>(), outputPath, file_name);
        }

        if (config.cell) {
            frames.back().setBackgroundColor(config.cell->backgroundColor);
            frames.back().regenerateSynthFrame();
        }
    }
}
void CellUniverse::optimize(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("Invalid frame index");
    }

    Frame &frame = frames[frameIndex];

    if (frameIndex > 0) {
        const Frame &previousFrame = frames[frameIndex - 1];
        const float previousBackground = estimateAdaptiveBackgroundFromFrame(previousFrame, config.simulation);
        const float previousMeanBrightness = computeStackMean(previousFrame.getRealFrame());
        const float currentMeanBrightness = computeStackMean(frame.getRealFrame());
        const float brightnessScale =
            (previousMeanBrightness > 1e-6f) ? (currentMeanBrightness / previousMeanBrightness) : 1.0f;
        const float updatedBackground = previousBackground * brightnessScale;

        frame.setBackgroundColor(updatedBackground);
        std::cout << "[Adaptive Background] frame " << (firstFrame + frameIndex)
                  << " base=" << previousBackground
                  << " ratio=" << brightnessScale
                  << " background=" << updatedBackground << '\n';
    }

    // Propagate the universal bbox cost flag to the frame BEFORE
    // regenerateSynthFrame so the regenerate can skip the full-image L2
    // cache (refreshFullCostCache) when bbox cost is active — saves ~32M
    // pixel ops per frame for frames 2+.
    //
    // Frame 1 exception: there are no previous-frame snapshots, so bbox
    // mode cannot be snap-anchored (Option A fallback would make every
    // cell use a follow-the-cell bbox and lose the position anchor that's
    // the whole point of bbox cost). Use full-image L2 for frame 1 instead —
    // it's slower but gives the correct global anchoring while cells settle
    // into their initial fit. Frames 2+ use bbox with snap-anchoring.
    const bool bboxActiveThisFrame = config.prob.use_bbox_cost && (frameIndex > 0);
    frame.setUseBboxCost(bboxActiveThisFrame,
                         config.prob.bbox_margin_scale);

    frame.regenerateSynthFrame();

    // Signal-guided perturbation init (yp ffc1917 — full per-frame design).
    // When enabled and enough signal centers are detected (>= previous frame
    // cell count), the entire frame uses signal-guided perturbation with
    // signal_guided_iterations_per_cell. When disabled or not enough centers,
    // fall back to random perturbation with random_iterations_per_cell.
    bool useSignalGuidanceThisFrame = false;
    if (config.simulation.signal_guided_position_enabled) {
        std::vector<Frame::SignalCenter> centers =
            localizeSignalCentersForFrame(frame, config, firstFrame + frameIndex);
        frame.setSignalCenters(centers);
        useSignalGuidanceThisFrame = true;
        if (frameIndex > 0) {
            const size_t previousCellCount = frames[frameIndex - 1].cells.size();
            if (centers.size() < previousCellCount) {
                useSignalGuidanceThisFrame = false;
                std::cout << "[Signal Guidance Fallback] frame " << (firstFrame + frameIndex)
                          << " centers=" << centers.size()
                          << " previousCells=" << previousCellCount
                          << " mode=random\n";
            }
        }
    } else {
        frame.setSignalCenters({});
    }

    const int guidedPerCellIters =
        (config.simulation.signal_guided_iterations_per_cell >= 0)
            ? config.simulation.signal_guided_iterations_per_cell
            : config.simulation.iterations_per_cell;
    const int randomPerCellIters =
        (config.simulation.random_iterations_per_cell >= 0)
            ? config.simulation.random_iterations_per_cell
            : config.simulation.iterations_per_cell;
    const size_t activePerCellIters = static_cast<size_t>(std::max(
        0, useSignalGuidanceThisFrame ? guidedPerCellIters : randomPerCellIters));
    size_t totalIterations = frame.length() * activePerCellIters;
    int displayFrame = firstFrame + frameIndex;

    std::cout << "[Optimize] frame " << displayFrame
              << " (" << frame.cells.size() << " cells, " << totalIterations << " iterations)"
              << " perturbMode=" << (useSignalGuidanceThisFrame ? "signal_guided" : "random")
              << " guidedItersPerCell=" << guidedPerCellIters
              << " randomItersPerCell=" << randomPerCellIters
              << " useBboxCost=" << (bboxActiveThisFrame ? 1 : 0)
              << " bboxMarginScale=" << config.prob.bbox_margin_scale
              << (config.prob.use_bbox_cost && frameIndex == 0
                  ? " (frame-1 forced full-image)" : "")
              << std::endl;

    // Compute mean cell brightness for brightness-proportional overlap
    // scaling in perturbCell. Dim cells get lower overlap weight so the
    // penalty can't overwhelm their lower image cost signal.
    {
        float bSum = 0.0f;
        for (const auto &c : frame.cells) bSum += c.getBrightness();
        frame.setMeanCellBrightness(
            frame.cells.empty() ? 0.0f : bSum / static_cast<float>(frame.cells.size()));
    }

    if (frame.cells.size() <= 24)
    {
        std::cout << "[FrameState Before] frame " << displayFrame << std::endl;
        for (const auto &cell : frame.cells)
        {
            auto p = cell.getCellParams();
            std::cout << "  " << p.name
                      << " pos=(" << p.x << "," << p.y << "," << p.z << ")"
                      << " R=(" << p.aRadius << "," << p.cRadius << ")"
                      << " theta=(" << p.theta_x << "," << p.theta_y << "," << p.theta_z << ")"
                      << " brightness=" << p.brightness
                      << std::endl;
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

    const float overlapWeight = config.prob.overlap_penalty_weight;
    const float baseSplitProb = config.prob.P_split_base;

    // No splits on the first frame — cells can't divide before any time has passed
    bool allowSplits = (frameIndex > 0);

    // Cells that already failed a burn-in this frame — skip all further split attempts.
    std::set<std::string> splitBlacklist;

    // P(split) is a LINEAR ramp on snapshot elongation from P_split_base
    // at elong=1.0 to P_split_max at elong=2.0. Clamped to [P_split_base,
    // P_split_max]. No classification threshold — every cell gets a split
    // probability proportional to how elongated its snapshot was.
    //
    //   elong = 1.0 → P(split) = P_split_base (e.g., 0.03)
    //   elong = 2.0 → P(split) = P_split_max  (e.g., 0.5)
    //   elong > 2.0 → P(split) = P_split_max  (clamped)
    //
    // Elongated cells (likely to be dividing) get tried often; round cells
    // still get a small base chance because snapshot elongation isn't a
    // perfect signal (first-gen splits often have near-round snaps).
    constexpr float kElongRefLow  = 1.0f;
    constexpr float kElongRefHigh = 2.0f;
    const float pMax = config.prob.P_split_max;
    std::map<std::string, float> splitProbabilities;

    std::cout << "[P(split)] frame " << displayFrame
              << (allowSplits ? " (linear elong ramp)" : " (splits disabled)")
              << std::endl;
    for (const auto &cell : frame.cells) {
        auto p = cell.getCellParams();
        float snapElong = 1.0f;
        auto snapIt = previousSnapshots.find(p.name);
        if (snapIt != previousSnapshots.end() && snapIt->second.valid) {
            snapElong = snapIt->second.shapeElongation;
        }
        float ps = 0.0f;
        if (allowSplits) {
            const float t = std::clamp(
                (snapElong - kElongRefLow) / (kElongRefHigh - kElongRefLow),
                0.0f, 1.0f);
            ps = baseSplitProb + t * (pMax - baseSplitProb);
        }
        splitProbabilities[p.name] = ps;
        std::cout << "  " << p.name << " shapeElong=" << snapElong
                  << " P(split)=" << ps << std::endl;
    }

    double residSum = 0;
    double absResidSum = 0;
    double residCount = 0;
    int splitAccepted = 0;
    int splitAttempted = 0;
    int perturbAccepted = 0;

    // ===============================================================
    // Unified pipeline (2026-04-14 — classification removed):
    //   1. Per-cell position calibration.
    //   2. PCA shape fit with snap-mask.
    //   3. Pre-pass for every cell: image-grounded D1/D2 centroids of
    //      bright pixels. Pre-pass direction + midpoint become candidates
    //      for the split attempt, alongside parent-rotation axes and the
    //      true snap center. Cost + bio + bridge gates decide acceptance.
    //   4. Unified perturb + split loop over all cells. Split attempts
    //      are gated only by P(split) which scales linearly with elong.
    //      Cells that aren't really splitting get filtered by:
    //        - bridge gate (valleyRatio ≥ 0.95 → no valley, reject)
    //        - bio gates (volume fraction, daughter size ratio, buried)
    //        - cost gate (must improve by ≥ split_cost)
    // ===============================================================

    // ---- Per-cell position calibration pass (frame 1 only) ----
    //
    // Calibration uses brightness centroid + perturbation refinement to
    // place each cell at its true image position. It is meaningful only
    // when no authoritative position exists from the previous frame — i.e.
    // frame 1, which has only initial CSV positions.
    //
    // From frame 2 onward, every cell has a snap from end-of-previous-frame
    // optimization. Snap is the correct parent center. Running calibration
    // on a snap-valid cell whose bright pixels span two emerging daughter
    // blobs (any cell about to divide, or any elongated cell with
    // asymmetric brightness) drifts the centroid toward the brighter blob,
    // mis-anchoring the parent for the upcoming split attempt. Empirical
    // analysis (see changelog 2026-04-14 evening "Snap-anchored parent")
    // showed snap elongation cannot reliably gate this — e9077 split at
    // snap elong 1.28 was missed by the threshold approach. Pure skip is
    // the only safe rule.
    //
    // For surviving frame-2+ cells, position adjustments for genuine
    // motion happen in the unified perturb+split loop, where each move
    // is gated by `costDiff < 0`.
    const int calibrationIters = std::max(0, config.prob.split_calibration_iterations_per_cell);
    const bool runCalibration = (frameIndex == 0) && (calibrationIters > 0) && !frame.cells.empty();
    if (runCalibration) {
        const float posScale = std::max(0.0f, config.prob.split_burn_in_pos_sigma_scale);

        PerturbParams savedCalX = Ellipsoid::cellConfig.x;
        PerturbParams savedCalY = Ellipsoid::cellConfig.y;
        PerturbParams savedCalZ = Ellipsoid::cellConfig.z;

        Ellipsoid::cellConfig.x.sigma = savedCalX.sigma * posScale;
        Ellipsoid::cellConfig.y.sigma = savedCalY.sigma * posScale;
        Ellipsoid::cellConfig.z.sigma = savedCalZ.sigma * posScale;

        std::cout << "[Calibration] frame " << displayFrame
                  << " cells=" << frame.cells.size()
                  << " itersPerCell=" << calibrationIters
                  << " posScale=" << posScale
                  << " (frame-1-only path)"
                  << std::endl;

        // No previous snapshots exist on frame 1; fall back to live cell
        // positions for Voronoi claim points.
        auto buildCalibrationClaimSet = [&](const std::string &selfName) -> Frame::ClaimSet {
            Frame::ClaimSet others;
            for (const auto &other : frame.cells) {
                const std::string otherName = other.getName();
                if (otherName == selfName) continue;
                others[otherName].push_back(cv::Point3f(
                    other.getX(), other.getY(), other.getZ()));
            }
            return others;
        };

        for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
            const std::string calName = frame.cells[ci].getName();
            const cv::Point3f preCalPos(
                frame.cells[ci].getX(),
                frame.cells[ci].getY(),
                frame.cells[ci].getZ());

            // Step 1: analytic centroid calibration. Frame-1 cells have no
            // snap; build a synthetic snapshot from initial CSV state so
            // calibrateCellPositionViaCentroid has the position+radii
            // anchor it expects.
            PreviousFrameSnapshot bootSnap;
            bootSnap.valid = true;
            bootSnap.position = preCalPos;
            bootSnap.aRadius = frame.cells[ci].getARadius();
            bootSnap.bRadius = frame.cells[ci].getBRadius();
            bootSnap.cRadius = frame.cells[ci].getCRadius();
            bootSnap.thetaX = frame.cells[ci].getThetaX();
            bootSnap.thetaY = frame.cells[ci].getThetaY();
            bootSnap.thetaZ = frame.cells[ci].getThetaZ();

            Frame::ClaimSet calOthers = buildCalibrationClaimSet(calName);
            frame.calibrateCellPositionViaCentroid(ci, bootSnap, calOthers);

            const cv::Point3f postCentroidPos(
                frame.cells[ci].getX(),
                frame.cells[ci].getY(),
                frame.cells[ci].getZ());

            // Step 2: perturbation refinement
            int calAccepts = 0;
            for (int it = 0; it < calibrationIters; ++it) {
                auto calResult = frame.perturbCell(
                    ci, overlapWeight, useSignalGuidanceThisFrame);
                const bool calAccept = calResult.first < 0.0;
                if (calAccept) ++calAccepts;
                calResult.second(calAccept);
            }
            const cv::Point3f postCalPos(
                frame.cells[ci].getX(),
                frame.cells[ci].getY(),
                frame.cells[ci].getZ());
            const float centroidShift = static_cast<float>(cv::norm(postCentroidPos - preCalPos));
            const float perturbDrift = static_cast<float>(cv::norm(postCalPos - postCentroidPos));
            const float totalDrift = static_cast<float>(cv::norm(postCalPos - preCalPos));
            std::cout << "  [Calibration] cell=" << calName
                      << " iters=" << calibrationIters
                      << " accepts=" << calAccepts
                      << " pre=(" << preCalPos.x << "," << preCalPos.y << "," << preCalPos.z << ")"
                      << " postCentroid=(" << postCentroidPos.x << "," << postCentroidPos.y << "," << postCentroidPos.z << ")"
                      << " post=(" << postCalPos.x << "," << postCalPos.y << "," << postCalPos.z << ")"
                      << " centroidShift=" << centroidShift
                      << " perturbDrift=" << perturbDrift
                      << " totalDrift=" << totalDrift
                      << std::endl;
        }

        Ellipsoid::cellConfig.x = savedCalX;
        Ellipsoid::cellConfig.y = savedCalY;
        Ellipsoid::cellConfig.z = savedCalZ;
    }

    // ---- Per-cell iterative PCA shape fit ----
    // Runs AFTER position calibration so Voronoi neighbor exclusion uses
    // refined positions. PCA on the Voronoi-filtered bright pixels inside
    // (maskScale * current ellipsoid) drives rotation, all 3 radii, and
    // centroid. Iterates until shape converges (or maxIters reached).
    const int pcaMaxIters = config.cell ? config.cell->pcaShapeMaxIters : 0;
    if (pcaMaxIters > 0 && !frame.cells.empty()) {
        const float pcaScale    = config.cell->pcaShapeRadiusScale;
        const int   pcaMin      = config.cell->pcaShapeMinPixels;
        const float maskScale   = config.cell->pcaShapeMaskScale;
        const float convR       = config.cell->pcaShapeConvergeRadius;
        const float convAng     = config.cell->pcaShapeConvergeAngleDeg;
        const bool  updatePos   = config.cell->pcaShapeUpdatePosition;
        const float posShiftCap = config.cell->pcaShapeMaxPosShiftFraction;

        std::cout << "[PCA Shape] frame " << displayFrame
                  << " cells=" << frame.cells.size()
                  << " maxIters=" << pcaMaxIters
                  << " scale=" << pcaScale
                  << " minPixels=" << pcaMin
                  << " maskScale=" << maskScale
                  << " updatePos=" << updatePos
                  << std::endl;

        // Snapshot cell positions BEFORE the parallel region to avoid a
        // data race: each OMP thread writes to frame.cells[ci] (radii,
        // rotation, and optionally position), while buildShapeClaimSet
        // reads all cells' positions. Without the snapshot, thread A
        // could read a partially-written Ellipsoid from thread B.
        // The snapshot is read-only during the parallel region.
        struct CellPosSnapshot { std::string name; float x, y, z; };
        std::vector<CellPosSnapshot> cellPosSnap;
        cellPosSnap.reserve(frame.cells.size());
        for (const auto &c : frame.cells) {
            cellPosSnap.push_back({c.getName(), c.getX(), c.getY(), c.getZ()});
        }

        auto buildShapeClaimSet = [&](const std::string &selfName) -> Frame::ClaimSet {
            Frame::ClaimSet others;
            for (const auto &snap : cellPosSnap) {
                if (snap.name == selfName) continue;
                others[snap.name].push_back(cv::Point3f(snap.x, snap.y, snap.z));
            }
            return others;
        };

        // Snap-mask shape fit: pass snapshot radii as the FIXED mask.
        // The mask stays constant across iterations, so the fit cannot
        // collapse onto one emerging daughter (no mask-feedback loop).
        // Radii are free to reach true image extent (shrink or grow).
        // Newly-born cells with no snapshot fall back to their current
        // live radii inside calibrateCellShapeViaPca.
        //
        // Parallel: cells are independent (_realFrame is read-only, each
        // thread writes a unique cells[ci], claim sets read from the
        // pre-snapshot). Per-cell logs are accumulated into per-cell
        // ostringstreams via the optional logSink parameter, then emitted
        // in cell-index order during the serial merge below — log
        // ordering is deterministic regardless of thread count.
        const int nCells = static_cast<int>(frame.cells.size());
        std::vector<std::ostringstream> shapeLogs(nCells);
        #pragma omp parallel for schedule(dynamic)
        for (int ci = 0; ci < nCells; ++ci) {
            const std::string sname = frame.cells[ci].getName();
            Frame::ClaimSet others = buildShapeClaimSet(sname);

            // Mask radii: prefer the FROZEN per-cell reference (set at cell
            // birth, never updated) over the snap radii (which reflect the
            // previous frame's fit and propagate bloat). The reference is
            // captured post-fit at frame 1 for initial cells and post-refit
            // at split-accept for daughters. Falling back to snap covers
            // cells whose reference wasn't registered (shouldn't happen in
            // Mask radii: use BIRTH values (captured once, never updated).
            // The mask defines the pixel-gathering window — it should
            // always be bigger than the cell so PCA sees the full picture.
            // Birth-based mask is decoupled from the bounded ref, so it
            // can't participate in feedback loops (upward bloat or
            // downward thinning). With maskScale=3.0, mask ≈ 3× birth
            // — generous enough that the PCA fit is determined entirely
            // by the pixel distribution, not the mask boundary.
            // Bounded ref (cellShapeReference) is used ONLY for the
            // fit-side growth cap downstream.
            float maskA = 0.0f, maskB = 0.0f, maskC = 0.0f;
            auto birthIt = cellShapeBirth.find(sname);
            if (birthIt != cellShapeBirth.end()) {
                maskA = birthIt->second[0];
                maskB = birthIt->second[1] > 1e-3f ? birthIt->second[1] : maskA;
                maskC = birthIt->second[2];
            } else {
                // First appearance (frame 1 or just-born daughter):
                // no birth ref yet. Fall back to snap, then live radii.
                auto snapIt = previousSnapshots.find(sname);
                if (snapIt != previousSnapshots.end() && snapIt->second.valid) {
                    maskA = snapIt->second.aRadius;
                    maskB = snapIt->second.bRadius > 1e-3f
                            ? snapIt->second.bRadius : maskA;
                    maskC = snapIt->second.cRadius;
                }
            }

            frame.calibrateCellShapeViaPca(ci, others,
                                           pcaMaxIters, pcaScale, pcaMin,
                                           maskScale, convR, convAng,
                                           updatePos, posShiftCap,
                                           maskA, maskB, maskC,
                                           &shapeLogs[ci]);
        }
        // Serial merge: emit per-cell log blocks in cell-index order.
        for (int ci = 0; ci < nCells; ++ci) {
            const std::string buf = shapeLogs[ci].str();
            if (!buf.empty()) std::cout << buf;
        }
        // ---- Fit-side growth cap (anti-bloat within mask) ----
        //
        // Bounded-growth on the REFERENCE (below) caps ref at ±5%/frame,
        // but the fit can still jump within the mask (mask = ref × 1.6
        // allows fit to grow up to 60% above ref in a single frame).
        // Observed in run 091510: e9077..a50 fit jumped aRadius 41.5 →
        // 65.2 at f20 (+57%) and hit the ceiling, cascading chaos at
        // f21/f22.
        //
        // Fit-side growth cap: two-sided bound at ref × (1 ± fitGrowthCap)
        // per frame.
        //   Up:   prevents single-frame bloat (PCA seeing halo, outputting
        //         R=60 when ref=40).
        //   Down: prevents single-frame collapse (PCA outputting a
        //         spuriously small radius due to Voronoi cut or neighbor
        //         overlap, causing e.g. 12345..0 c to crash 22→14 in one
        //         frame of run 082209, then compounding through splits
        //         to hit the 10-px floor). 10% downward still allows
        //         biological pre-split pinching over several frames.
        constexpr float fitGrowthCap = 0.10f;
        const float fitUpFactor = 1.0f + fitGrowthCap;
        const float fitDownFactor = 1.0f - fitGrowthCap;
        int fitsClamped = 0;
        for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
            const std::string &name = frame.cells[ci].getName();
            auto it = cellShapeReference.find(name);
            if (it == cellShapeReference.end()) continue;
            const auto &ref = it->second;
            const float fA = frame.cells[ci].getARadius();
            const float fB = frame.cells[ci].getBRadius();
            const float fC = frame.cells[ci].getCRadius();
            const float capUpA   = ref[0] * fitUpFactor;
            const float capUpB   = ref[1] * fitUpFactor;
            const float capUpC   = ref[2] * fitUpFactor;
            const float capDownA = ref[0] * fitDownFactor;
            const float capDownB = ref[1] * fitDownFactor;
            const float capDownC = ref[2] * fitDownFactor;
            const float newA = std::clamp(fA, capDownA, capUpA);
            const float newB = std::clamp(fB, capDownB, capUpB);
            const float newC = std::clamp(fC, capDownC, capUpC);
            if (newA != fA || newB != fB || newC != fC) {
                frame.cells[ci].setRadii(newA, newB, newC);
                ++fitsClamped;
            }
        }
        if (fitsClamped > 0) {
            std::cout << "[Fit Growth Cap] frame " << displayFrame
                      << " clamped=" << fitsClamped
                      << " maxUp=" << fitGrowthCap << std::endl;
            // Re-render with clamped radii so downstream cost eval sees
            // the capped synth, not the pre-clamp one.
            frame.regenerateSynthFrame();
        } else {
            frame.regenerateSynthFrame();
        }

        // ---- Reference-side growth cap (anti-compounding mask basis) ----
        //
        // Each frame the reference tracks toward the observed fit but is
        // capped at ±refGrowthCap per frame:
        //   ref_new[i] = clamp(fit[i], ref_old[i] * (1-cap), ref_old[i] * (1+cap))
        // This allows legitimate multi-frame growth (daughter cells growing
        // toward adult size at ~5%/frame passes through unchanged) while
        // denying instantaneous bloat (20%/frame compounding hits the cap,
        // mask can't follow, fit can't run away).
        //
        // New cells (frame 1 seeds, newly-born daughters): reference is
        // captured directly from the current fit — they don't have a
        // "previous" reference to bound against yet.
        constexpr float refGrowthCap = 0.05f;   // 5%/frame max Δ, up or down
        const float refLo = 1.0f - refGrowthCap;
        const float refHi = 1.0f + refGrowthCap;
        int refsCaptured = 0;
        int refsUpdated = 0;
        int birthsCaptured = 0;
        for (const auto &cell : frame.cells) {
            const std::string &name = cell.getName();
            const float fA = cell.getARadius();
            const float fB = cell.getBRadius();
            const float fC = cell.getCRadius();

            // Birth capture (once, never updated). Used for mask basis.
            if (cellShapeBirth.find(name) == cellShapeBirth.end()) {
                cellShapeBirth[name] = {fA, fB, fC};
                ++birthsCaptured;
            }

            auto it = cellShapeReference.find(name);
            if (it == cellShapeReference.end()) {
                cellShapeReference[name] = {fA, fB, fC};
                ++refsCaptured;
            } else {
                auto &ref = it->second;
                ref[0] = std::clamp(fA, ref[0] * refLo, ref[0] * refHi);
                ref[1] = std::clamp(fB, ref[1] * refLo, ref[1] * refHi);
                ref[2] = std::clamp(fC, ref[2] * refLo, ref[2] * refHi);
                ++refsUpdated;
            }
        }
        if (refsCaptured > 0 || refsUpdated > 0 || birthsCaptured > 0) {
            std::cout << "[Shape Reference] frame " << displayFrame
                      << " births=" << birthsCaptured
                      << " refCaptured=" << refsCaptured
                      << " refUpdated=" << refsUpdated
                      << " totalRef=" << cellShapeReference.size()
                      << " totalBirth=" << cellShapeBirth.size()
                      << " growthCap=" << refGrowthCap << std::endl;
        }
    }

    // ---- Install snap-anchored per-cell bboxes (Option A) ----
    // Each cell with a valid PreviousFrameSnapshot gets a bbox fixed at
    // the snap position with half-extent = bbox_margin_scale × snapMaxR.
    // perturbCell then uses this fixed bbox for every iteration in this
    // frame — so drifting away from snap incurs an undershoot cost at the
    // abandoned snap voxels. Cells without a snap (frame 1, or newborn
    // daughters after a split accepts mid-frame) have no entry and fall
    // back to the legacy live pre/post-union bbox in perturbCell. Always
    // cleared first so stale names don't leak across frames.
    frame.clearSnapBboxes();
    frame.clearSnapPositions();
    frame.setPositionPriorWeight(config.prob.position_prior_weight);
    frame.setPositionPriorThreshold(config.prob.position_prior_threshold);
    if (bboxActiveThisFrame) {
        int installed = 0;
        for (const auto &cell : frame.cells) {
            const std::string &name = cell.getName();
            auto snapIt = previousSnapshots.find(name);
            if (snapIt == previousSnapshots.end() || !snapIt->second.valid) continue;
            const auto &snap = snapIt->second;
            const float snapMaxR = std::max({snap.aRadius, snap.bRadius, snap.cRadius});
            if (snapMaxR <= 1e-3f) continue;
            BoundingBox3D snapBbox = frame.computeBboxAtPoint(
                snap.position, snapMaxR, config.prob.bbox_margin_scale);
            if (!snapBbox.isValid()) continue;
            frame.setSnapBbox(name, snapBbox);
            frame.setSnapPosition(name, snap.position);
            ++installed;
        }
        std::cout << "[Snap Bbox] frame " << displayFrame
                  << " installed=" << installed
                  << " total=" << frame.cells.size()
                  << " priorWeight=" << config.prob.position_prior_weight
                  << std::endl;
    }

    // Seed expected-daughter positions for EVERY cell with a valid
    // snapshot. The current-frame PCA (pre-pass below) needs a starting
    // point for Voronoi claims — using the snapshot's split axis is the
    // natural seed. Classification no longer gates this: non-classified
    // cells also need image-grounded midpoints when they attempt splits,
    // because snap axis is unreliable for near-round cells.
    std::map<std::string, std::pair<cv::Point3f, cv::Point3f>> expectedDaughters;
    for (const auto &cell : frame.cells) {
        const std::string &name = cell.getName();
        auto snapIt = previousSnapshots.find(name);
        if (snapIt == previousSnapshots.end() || !snapIt->second.valid) continue;
        const auto &snap = snapIt->second;
        const float half = 0.5f * snap.splitAxisLength;
        const cv::Point3f seed1(
            snap.position.x - half * snap.splitAxisDir.x,
            snap.position.y - half * snap.splitAxisDir.y,
            snap.position.z - half * snap.splitAxisDir.z);
        const cv::Point3f seed2(
            snap.position.x + half * snap.splitAxisDir.x,
            snap.position.y + half * snap.splitAxisDir.y,
            snap.position.z + half * snap.splitAxisDir.z);
        expectedDaughters[name] = {seed1, seed2};
    }

    // ---- Pre-pass: image-ground the seeds for ALL cells ----
    // Runs PCA on current-frame bright pixels around each cell to find the
    // midpoint + direction connecting the two bright blobs (if any). This
    // image-grounded direction is the primary signal when snap axis is
    // unreliable (near-round snap = arbitrary axC direction). For single-
    // blob cells, PCA returns the blob's principal direction — harmless,
    // cost gate still rejects because there's no valley.
    const int prePassRounds = std::max(1, config.prob.expected_daughter_pre_pass_iterations);
    if (!frame.cells.empty()) {
        std::cout << "[Pre-Pass] frame " << displayFrame
                  << " rounds=" << prePassRounds
                  << " allCells=" << frame.cells.size()
                  << " mode=image_grounded_pca" << std::endl;
    }
    // Per-cell pre-pass result, accumulated in parallel then merged serially.
    // Parallelization is safe because:
    //  - imageGroundExpectedDaughters is const on Frame (read-only)
    //  - claim set built from `roundSnapshot` (frozen at start of round) and
    //    `frame.cells` (read-only during the parallel region)
    //  - results[ci] is uniquely written per thread
    //  - logs written into thread-local ostringstream; emitted in order
    //    after the parallel region for deterministic log order
    struct PrePassResult {
        bool present = false;
        bool ok = false;
        cv::Point3f groundedD1{0, 0, 0};
        cv::Point3f groundedD2{0, 0, 0};
        cv::Point3f oldD1{0, 0, 0};
        cv::Point3f oldD2{0, 0, 0};
        int keptPixels = 0;
        std::string claimsLog;
        std::string resultLog;
    };

    for (int round = 0; round < prePassRounds; ++round) {
        // Snapshot expectedDaughters at start of round so all parallel
        // readers see consistent neighbor seed positions.
        const auto roundSnapshot = expectedDaughters;
        std::vector<PrePassResult> results(frame.cells.size());

        const int nCells = static_cast<int>(frame.cells.size());
        #pragma omp parallel for schedule(dynamic)
        for (int ci = 0; ci < nCells; ++ci) {
            const std::string name = frame.cells[ci].getName();
            auto roundIt = roundSnapshot.find(name);
            if (roundIt == roundSnapshot.end()) continue;

            auto snapIt = previousSnapshots.find(name);
            if (snapIt == previousSnapshots.end()) continue;

            results[ci].present = true;
            results[ci].oldD1 = roundIt->second.first;
            results[ci].oldD2 = roundIt->second.second;

            const auto &snap = snapIt->second;

            // Build claim-set from the round snapshot (frozen) and live cells.
            // Every other cell contributes BOTH its D1/D2 seed when present
            // (dumbbell-aware Voronoi for dividing neighbors).
            Frame::ClaimSet others;
            for (const auto &other : frame.cells) {
                const std::string otherName = other.getName();
                if (otherName == name) continue;
                auto itD = roundSnapshot.find(otherName);
                if (itD != roundSnapshot.end()) {
                    others[otherName].push_back(itD->second.first);
                    others[otherName].push_back(itD->second.second);
                } else {
                    others[otherName].push_back(cv::Point3f(
                        other.getX(), other.getY(), other.getZ()));
                }
            }

            {
                std::ostringstream claimLog;
                for (const auto &kv : others) {
                    claimLog << " " << kv.first.substr(0, 8)
                             << "[" << kv.second.size() << "]";
                }
                std::ostringstream line;
                line << "  [Pre-Pass Claims] " << name.substr(0, 8)
                     << " nNeighbors=" << others.size()
                     << " claims:" << claimLog.str() << "\n";
                results[ci].claimsLog = line.str();
            }

            cv::Point3f groundedD1, groundedD2;
            int keptPixels = 0;
            const bool ok = frame.imageGroundExpectedDaughters(
                static_cast<size_t>(ci), snap, others, groundedD1, groundedD2, &keptPixels);

            results[ci].ok = ok;
            results[ci].groundedD1 = groundedD1;
            results[ci].groundedD2 = groundedD2;
            results[ci].keptPixels = keptPixels;

            std::ostringstream rlog;
            if (ok) {
                const cv::Point3f delta1 = groundedD1 - results[ci].oldD1;
                const cv::Point3f delta2 = groundedD2 - results[ci].oldD2;
                const float shift1 = static_cast<float>(cv::norm(delta1));
                const float shift2 = static_cast<float>(cv::norm(delta2));
                rlog << "  [Pre-Pass] round=" << round
                     << " cell=" << name
                     << " shapeElong=" << snap.shapeElongation
                     << " kept=" << keptPixels
                     << " snapPos=(" << snap.position.x << "," << snap.position.y << "," << snap.position.z << ")"
                     << " seedD1=(" << results[ci].oldD1.x << "," << results[ci].oldD1.y << "," << results[ci].oldD1.z << ")"
                     << " expD1=(" << groundedD1.x << "," << groundedD1.y << "," << groundedD1.z << ")"
                     << " shift1=" << shift1
                     << " seedD2=(" << results[ci].oldD2.x << "," << results[ci].oldD2.y << "," << results[ci].oldD2.z << ")"
                     << " expD2=(" << groundedD2.x << "," << groundedD2.y << "," << groundedD2.z << ")"
                     << " shift2=" << shift2
                     << "\n";
            } else {
                rlog << "  [Pre-Pass] round=" << round
                     << " cell=" << name
                     << " shapeElong=" << snap.shapeElongation
                     << " kept=" << keptPixels
                     << " result=pca_failed_keep_snapshot_seeds"
                     << " seedD1=(" << results[ci].oldD1.x << "," << results[ci].oldD1.y << "," << results[ci].oldD1.z << ")"
                     << " seedD2=(" << results[ci].oldD2.x << "," << results[ci].oldD2.y << "," << results[ci].oldD2.z << ")"
                     << "\n";
            }
            results[ci].resultLog = rlog.str();
        }

        // Serial merge: apply results to expectedDaughters and emit logs
        // in cell-index order for deterministic output.
        for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
            if (!results[ci].present) continue;
            if (results[ci].ok) {
                const std::string &name = frame.cells[ci].getName();
                expectedDaughters[name] = {results[ci].groundedD1, results[ci].groundedD2};
            }
            std::cout << results[ci].claimsLog << results[ci].resultLog;
        }
    }

    // ---- Helper: build claim-set for other cells during a split attempt ----
    auto buildOtherClaimSet = [&](const std::string &selfName,
                                  const std::set<std::string> &alreadySplitInB,
                                  const std::set<std::string> &splitRejectedInB,
                                  bool phaseB) -> Frame::ClaimSet {
        Frame::ClaimSet others;
        for (const auto &other : frame.cells) {
            const std::string otherName = other.getName();
            if (otherName == selfName) continue;

            // If this neighbor already split-accepted this frame, its
            // daughters are the relevant Voronoi claims.
            if (alreadySplitInB.count(otherName) > 0) {
                const std::string d1n = otherName + "0";
                const std::string d2n = otherName + "1";
                for (const auto &c : frame.cells) {
                    if (c.getName() == d1n || c.getName() == d2n) {
                        others[c.getName()].push_back(cv::Point3f(
                            c.getX(), c.getY(), c.getZ()));
                    }
                }
                continue;
            }

            // If this neighbor's split was rejected this frame OR it has no
            // D1/D2 seed, use its (calibrated) live position.
            if (splitRejectedInB.count(otherName) > 0) {
                others[otherName].push_back(cv::Point3f(
                    other.getX(), other.getY(), other.getZ()));
                continue;
            }

            // Default: use pre-pass D1/D2 if available (covers dumbbell
            // neighbors), else live position.
            auto itD = expectedDaughters.find(otherName);
            if (itD != expectedDaughters.end()) {
                others[otherName].push_back(itD->second.first);
                others[otherName].push_back(itD->second.second);
            } else {
                others[otherName].push_back(cv::Point3f(
                    other.getX(), other.getY(), other.getZ()));
            }
        }
        return others;
    };

    // ---- Single-phase iteration helper ----
    auto runPhase = [&](std::set<std::string> &phaseNames,
                        bool phaseB,
                        std::set<std::string> &splitAcceptedInPhase,
                        std::set<std::string> &splitRejectedInPhase) {
        if (phaseNames.empty()) return;

        const size_t perCellIters = activePerCellIters;
        const size_t totalPhaseIters = perCellIters * phaseNames.size();

        // Maintain eligibility incrementally. Only changes when a split
        // accepts (parent replaced by two daughters) — no cell is ever
        // removed without replacement, and daughters inherit eligibility.
        // Splits that fail blacklist the cell for future SPLIT attempts but
        // leave it eligible for perturbation, so no index maintenance is
        // needed on rejection.
        std::vector<size_t> eligible;
        eligible.reserve(frame.cells.size());
        const auto rebuildEligible = [&]() {
            eligible.clear();
            for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
                const std::string &cname = frame.cells[ci].getName();
                if (phaseNames.count(cname) == 0) continue;
                eligible.push_back(ci);
            }
        };
        rebuildEligible();

        for (size_t i = 0; i < totalPhaseIters; ++i) {
            if (eligible.empty()) break;

            std::uniform_int_distribution<size_t> cellDist(0, eligible.size() - 1);
            const size_t cellIdx = eligible[cellDist(gen)];
            // VALUE COPY — must not be a reference. trySplitCellPhased and
            // perturbCell mutate frame.cells (erase/push_back/full reassign
            // via savedCells), invalidating any reference to the original
            // ellipsoid's _name string. A dangling cellName silently caused
            // splitBlacklist to be inserted with garbage content, defeating
            // the "max one split attempt per cell per frame" invariant —
            // the same cell would re-attempt because its real name wasn't
            // found in the blacklist (the blacklist had a corrupted entry).
            // Observed as the "8cbdf86d gets 2 split attempts" symptom.
            const std::string cellName = frame.cells[cellIdx].getName();

            float pSplit = 0.0f;
            if (allowSplits) {
                auto pIt = splitProbabilities.find(cellName);
                if (pIt != splitProbabilities.end()) pSplit = pIt->second;
            }
            const bool canSplit = pSplit > 0.0f
                               && splitBlacklist.count(cellName) == 0;

            if (canSplit && uniform01(gen) < pSplit) {
                // --- Split attempt ---
                splitAttempted++;
                auto snapIt = previousSnapshots.find(cellName);
                if (snapIt == previousSnapshots.end() || !snapIt->second.valid) {
                    splitBlacklist.insert(cellName);
                    continue;
                }

                // useSnapDir=true adds snapshot-axis candidates alongside
                // image-PCA candidates inside trySplitCellPhased — cost picks
                // the winner. Always true now so every split attempt tries
                // both direction sources.
                const bool useSnapDir = true;

                Frame::ClaimSet others = buildOtherClaimSet(
                    cellName,
                    splitAcceptedInPhase,
                    splitRejectedInPhase,
                    phaseB);

                // Pass pre-pass direction + length via splitSnapshot so
                // trySplitCellPhased can add it as an extra primary axis
                // (labeled "imgPca"). We KEEP splitSnapshot.position at
                // the snap center so the split attempt tries BOTH midpoints:
                //   - snap_* candidates: midpoint = true snap center
                //   - data_* candidates: midpoint = pixel-projection centroid
                //     (which for the imgPca axis equals the pre-pass midpoint)
                // Snap radii preserved for daughter sizing.
                PreviousFrameSnapshot splitSnapshot = snapIt->second;
                auto itD = expectedDaughters.find(cellName);
                if (itD != expectedDaughters.end()) {
                    const cv::Point3f &gd1 = itD->second.first;
                    const cv::Point3f &gd2 = itD->second.second;
                    const cv::Point3f delta = gd2 - gd1;
                    const float len = static_cast<float>(cv::norm(delta));
                    if (len > 1e-3f) {
                        splitSnapshot.splitAxisDir = cv::Point3f(
                            delta.x / len, delta.y / len, delta.z / len);
                        splitSnapshot.splitAxisLength = len;
                    }
                }

                auto result = frame.trySplitCellPhased(
                    cellIdx, splitSnapshot, others, useSnapDir, config.prob);

                double costDiff = result.first;
                auto callback = result.second;

                const bool accept = (costDiff < 0.0);
                callback(accept);
                if (accept) {
                    splitAccepted++;
                    splitAcceptedInPhase.insert(cellName);
                    previousSnapshots.erase(cellName);
                    // Add daughter names to phaseNames so they become eligible
                    // for perturbation during the rest of this frame. Without
                    // this, rebuildEligible skips daughters (their names weren't
                    // in phaseNames at frame start) and they get zero perturbation
                    // iterations — only the split burn-in + refine positioning.
                    phaseNames.insert(cellName + "0");
                    phaseNames.insert(cellName + "1");
                    // Rebuild eligible: the parent cell at cellIdx was replaced
                    // by two daughters appended to the cells vector.
                    rebuildEligible();
                    // Recompute mean brightness — cell count changed.
                    {
                        float bSum = 0.0f;
                        for (const auto &c : frame.cells) bSum += c.getBrightness();
                        frame.setMeanCellBrightness(bSum / static_cast<float>(frame.cells.size()));
                    }
                } else {
                    splitBlacklist.insert(cellName);
                    splitRejectedInPhase.insert(cellName);

                    // Compensation perturb. The revert left cells[cellIdx]
                    // as the parent (in place), so no find_if needed.
                    auto compResult = frame.perturbCell(
                        cellIdx, overlapWeight, useSignalGuidanceThisFrame);
                    const bool compAccept = compResult.first < 0.0;
                    compResult.second(compAccept);
                    if (compAccept) {
                        perturbAccepted++;
                        residSum += compResult.first;
                        absResidSum += std::abs(compResult.first);
                        residCount++;
                    }
                }
            } else {
                // --- Perturbation ---
                auto result = frame.perturbCell(
                    cellIdx, overlapWeight, useSignalGuidanceThisFrame);
                double costDiff = result.first;
                auto callback = result.second;
                if (costDiff < 0) {
                    callback(true);
                    perturbAccepted++;
                    residSum += costDiff;
                    absResidSum += std::abs(costDiff);
                    residCount++;
                } else {
                    callback(false);
                }
            }

            if ((i + 1) % 500 == 0) {
                std::cout << "Frame " << displayFrame
                          << " iter=" << i
                          << " perturb_accepted=" << perturbAccepted
                          << " split_attempts=" << splitAttempted
                          << " split_accepted=" << splitAccepted
                          << " cells=" << frame.cells.size() << std::endl;
            }
        }
    };

    // Unified loop over ALL cells. Classification removed; all cells share
    // the same perturb+split iteration budget, and the split attempt gets
    // both snap-axis and image-PCA candidates regardless of snap elongation.
    std::set<std::string> allNames;
    for (const auto &cell : frame.cells) {
        allNames.insert(cell.getName());
    }
    std::set<std::string> splitAcceptedInLoop;
    std::set<std::string> splitRejectedInLoop;
    runPhase(allNames, /* phaseB */ true, splitAcceptedInLoop, splitRejectedInLoop);

    // End of frame: build PreviousFrameSnapshot for each cell, combining the
    // in-frame running max (from the periodic sampling above) with the
    // end-of-frame PCA measurement. The snapshot is the authoritative source
    // for next-frame split decisions (PR-B trigger, PR-C placement, PR-D peaks).
    // PR-E: previousElongations removed — snapshots are the sole data store.
    //
    // Don't clear previousSnapshots — entries for cells that disappeared this
    // frame (e.g., parents replaced by daughters) become stale but are
    // overwritten below for surviving cells. Daughters get fresh snapshots.
    // We do erase entries that don't exist anymore to prevent unbounded growth.
    {
        std::set<std::string> liveCells;
        for (const auto &cell : frame.cells) {
            liveCells.insert(cell.getName());
        }
        for (auto it = previousSnapshots.begin(); it != previousSnapshots.end();) {
            if (liveCells.find(it->first) == liveCells.end()) {
                it = previousSnapshots.erase(it);
            } else {
                ++it;
            }
        }
    }
    std::cout << "[Snapshot] frame " << displayFrame << std::endl;
    for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
        auto p = frame.cells[ci].getCellParams();

        // Triaxial fitted-shape elongation is the classification signal.
        // max(a,b,c)/min(a,b,c) from the fit, plus the world-space direction
        // and length of the longest axis. No image-PCA anymore.
        const float fitShapeElong = frame.cells[ci].shapeElongation();
        cv::Point3f fitSplitAxisDir;
        float fitSplitAxisLength = 0.0f;
        frame.cells[ci].worldSplitAxis(fitSplitAxisDir, fitSplitAxisLength);

        PreviousFrameSnapshot snap;
        snap.valid = true;
        snap.shapeElongation = fitShapeElong;
        snap.splitAxisDir = fitSplitAxisDir;
        snap.splitAxisLength = fitSplitAxisLength;

        snap.position = cv::Point3f(p.x, p.y, p.z);
        snap.aRadius = p.aRadius;
        snap.bRadius     = p.bRadius;
        snap.cRadius = p.cRadius;
        snap.thetaX = p.theta_x;
        snap.thetaY = p.theta_y;
        snap.thetaZ = p.theta_z;
        snap.brightness = p.brightness;

        previousSnapshots[p.name] = snap;

        std::cout << "  " << p.name
                  << " shapeElong=" << snap.shapeElongation
                  << " splitAxisLen=" << snap.splitAxisLength
                  << " pos=(" << snap.position.x
                  << "," << snap.position.y
                  << "," << snap.position.z << ")"
                  << std::endl;
    }

    const float brightnessBlend = std::clamp(config.cell ? config.cell->brightnessUpdateBlend : 0.0f, 0.0f, 1.0f);
    if (brightnessBlend > 0.0f && config.cell) {
        const auto &realFrame = frame.getRealFrame();
        const float brightnessAmplification = std::max(0.0f, config.cell->brightnessMeanAmplification);
        const float brightnessMeasurementTopPercentile =
            std::clamp(config.cell->brightnessMeasurementTopPercentile, 0.0f, 1.0f);
        for (auto &cell : frame.cells) {
            const float observedBrightness =
                cell.measureMeanBrightness(realFrame, brightnessMeasurementTopPercentile);
            const float amplifiedObservedBrightness = observedBrightness * brightnessAmplification;
            const float updatedBrightness =
                cell.getBrightness() * (1.0f - brightnessBlend) + amplifiedObservedBrightness * brightnessBlend;
            cell.setBrightness(updatedBrightness);
        }
        frame.regenerateSynthFrame();
    }

    std::cout << "[Optimize Done] frame " << displayFrame
              << " perturb_accepted=" << perturbAccepted
              << " split_attempts=" << splitAttempted
              << " split_accepted=" << splitAccepted
              << " final_cells=" << frame.cells.size() << std::endl;
}

void CellUniverse::saveImages(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("Invalid frame index");
    }

    std::vector<Image> realImages = frames[frameIndex].generateOutputFrame();
    std::vector<Image> synthImages = frames[frameIndex].generateOutputSynthFrame();
    int displayFrame = firstFrame + frameIndex;
    std::cout << "Saving images for frame " << displayFrame << "..." << '\n';
    std::cout << "Real Image Type: " << realImages[0].type() << '\n';
    std::cout << "Synth Image Type: " << synthImages[0].type() << '\n';

    std::string realOutputPath = outputPath + "/real/" + std::to_string(displayFrame);
    if (!std::filesystem::exists(realOutputPath))
    {
        std::filesystem::create_directories(realOutputPath);
    }
    std::string synthOutputPath = outputPath + "/synth/" + std::to_string(displayFrame);
    if (!std::filesystem::exists(synthOutputPath))
    {
        std::filesystem::create_directories(synthOutputPath);
    }

    // Parallelize PNG encoding/write — independent across slices and across
    // real/synth. cv::imwrite is CPU-bound (zlib compression). With 225
    // slices × 2 streams = 450 independent writes per frame this typically
    // saves several seconds per frame on a multi-core machine.
    const int nSlices = static_cast<int>(std::max(realImages.size(), synthImages.size()));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nSlices; ++i)
    {
        if (static_cast<size_t>(i) < realImages.size()) {
            cv::imwrite(realOutputPath + "/" + std::to_string(i) + ".png", realImages[i]);
        }
        if (static_cast<size_t>(i) < synthImages.size()) {
            cv::imwrite(synthOutputPath + "/" + std::to_string(i) + ".png", synthImages[i]);
        }
    }

    std::cout << "Done" << '\n';
}

void CellUniverse::saveCells(int frameIndex)
{
    std::string cellsPath = outputPath + "/cells.csv";
    bool fileExists = std::filesystem::exists(cellsPath);

    // Append mode: each frame adds its rows as it finishes optimizing
    std::ofstream file(cellsPath, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << cellsPath << " for writing" << '\n';
        return;
    }

    // Write header only for the first frame
    if (!fileExists || frameIndex == 0) {
        // Truncate if frame 0 (fresh run)
        if (frameIndex == 0) {
            file.close();
            file.open(cellsPath, std::ios::trunc);
        }
        file << "file,name,x,y,z,aRadius,bRadius,cRadius,theta_x,theta_y,theta_z" << '\n';
    }

    Frame &frame = frames[frameIndex];
    std::string imageName = frame.getImageName();

    for (const auto &cell : frame.cells) {
        EllipsoidParams params = cell.getCellParams();
        cell.printCellInfo();
        file << imageName << ","
             << params.name << ","
             << params.x << ","
             << params.y << ","
             << params.z << ","
             << params.aRadius << ","
             << params.bRadius << ","
             << params.cRadius << ","
             << params.theta_x << ","
             << params.theta_y << ","
             << params.theta_z
             << '\n';
    }

    std::cout << "Saved " << frame.cells.size() << " cells for frame " << (firstFrame + frameIndex)
              << " to " << cellsPath << std::endl;
}

void CellUniverse::copyCellsForward(size_t to)
{
    if (to >= frames.size())
    {
        return;
    }
    // assumes cells have deepcopy copy constructors
    frames[to].cells = frames[to - 1].cells;
    if (config.cell) {
        for (auto &cell : frames[to].cells) {
            cell.blendAdaptivePerturbProbabilitiesWithConfig(
                config.cell->brightnessProbabilityTrust,
                config.cell->aRadiusProbabilityTrust,
                config.cell->cRadiusProbabilityTrust,
                config.cell->bRadiusProbabilityTrust);
        }
    }
}

unsigned int CellUniverse::length()
{
    return frames.size();
}

const std::vector<Ellipsoid> &CellUniverse::getCells(int frameIndex) const
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("CellUniverse::getCells - invalid frameIndex");
    }
    return frames[frameIndex].cells;
}

std::vector<std::string> CellUniverse::getCellNames(int frameIndex) const
{
    const auto &cells = getCells(frameIndex);
    std::vector<std::string> names;
    names.reserve(cells.size());
    for (const auto &c : cells)
    {
        names.push_back(c.getCellParams().name);
    }
    return names;
}
