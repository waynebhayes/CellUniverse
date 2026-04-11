#include "../includes/CellUniverse.hpp"
#include "../includes/ImageHandler.hpp"
#include <set>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>

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
        params.majorRadius *= expandFactor;
        params.minorRadius *= expandFactor;
        Spheroid expandedCell(params);
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

CellUniverse::CellUniverse(std::map<std::string, std::vector<Spheroid>> initialCells,
                           PathVec imagePaths,
                           BaseConfig &config,
                           std::string outputPath,
                           int firstFrame,
                           int continueFrom)
: config(config), outputPath(outputPath), firstFrame(firstFrame)
{
    std::vector<std::vector<Image>> loadedFrames;
    std::vector<float> frameMeanBrightness;
    loadedFrames.reserve(imagePaths.size());
    frameMeanBrightness.reserve(imagePaths.size());

    for (size_t i = 0; i < imagePaths.size(); ++i)
    {
        std::vector<Image> real_frame;
        real_frame = ImageHandler::loadFrame(imagePaths[i].string(), config);
        frameMeanBrightness.push_back(computeStackMean(real_frame));
        loadedFrames.push_back(std::move(real_frame));
    }

    const float globalMeanBrightness = frameMeanBrightness.empty()
        ? 0.0f
        : std::accumulate(frameMeanBrightness.begin(), frameMeanBrightness.end(), 0.0f)
            / static_cast<float>(frameMeanBrightness.size());

    for (size_t i = 0; i < imagePaths.size(); ++i)
    {
        std::vector<Image> &real_frame = loadedFrames[i];
        const float currentMeanBrightness = frameMeanBrightness[i];
        const float brightnessScale =
            (currentMeanBrightness > 1e-6f) ? (globalMeanBrightness / currentMeanBrightness) : 1.0f;
        scaleStackBrightness(real_frame, brightnessScale);

        std::cout << "[Brightness Align] frame=" << imagePaths[i].filename().string()
                  << " frame_mean=" << currentMeanBrightness
                  << " global_mean=" << globalMeanBrightness
                  << " scale=" << brightnessScale << '\n';

        if (config.simulation.export_preprocessed_images) {
            exportPreprocessedStack(real_frame, fs::path(outputPath), imagePaths[i]);
        }

        if (config.simulation.quit_after_preprocessing) {
            continue;
        }

        // loadFrame interpolate frames, update to config is needed
        config.simulation.z_slices = real_frame.size();
        // Propagate the interpolated-z upper bound into Spheroid::cellConfig so
        // the Spheroid constructor's z clamp uses the actual stack height.
        // Without this, cells could drift off the top/bottom of the z-stack.
        Spheroid::cellConfig.maxZ = static_cast<float>(real_frame.size()) - 1.0f;

        fs::path path(imagePaths[i]);
        std::string file_name = path.filename();

        if ((continueFrom == -1 || i < continueFrom) && initialCells.find(file_name) != initialCells.end())
        {
            const std::vector<Spheroid> &cells = initialCells.at(file_name);
            frames.emplace_back(real_frame, config.simulation, cells, outputPath, file_name);
        }
        else
        {
            frames.emplace_back(real_frame, config.simulation, std::vector<Spheroid>(), outputPath, file_name);
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
    const bool brightnessPerturbFirstFrameOnly =
        config.cell && config.cell->firstFrameBrightnessPerturbationOnly;
    const PerturbParams originalBrightnessPerturb = Spheroid::cellConfig.brightness;
    if (brightnessPerturbFirstFrameOnly && frameIndex > 0)
    {
        Spheroid::cellConfig.brightness = PerturbParams{};
    }

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

    frame.regenerateSynthFrame();

    size_t totalIterations = frame.length() * config.simulation.iterations_per_cell;
    int displayFrame = firstFrame + frameIndex;

    std::cout << "[Optimize] frame " << displayFrame
              << " (" << frame.cells.size() << " cells, " << totalIterations << " iterations)" << std::endl;

    if (frame.cells.size() <= 24)
    {
        std::cout << "[FrameState Before] frame " << displayFrame << std::endl;
        for (const auto &cell : frame.cells)
        {
            auto p = cell.getCellParams();
            std::cout << "  " << p.name
                      << " pos=(" << p.x << "," << p.y << "," << p.z << ")"
                      << " R=(" << p.majorRadius << "," << p.minorRadius << ")"
                      << " theta=(" << p.theta_x << "," << p.theta_y << "," << p.theta_z << ")"
                      << " brightness=" << p.brightness
                      << std::endl;
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

    const float overlapWeight = config.prob.overlap_penalty_weight;
    const float sizeReductionWeight = config.prob.size_reduction_penalty_weight;
    const float baseSplitProb = config.prob.split; // base probability (e.g., 0.03)

    // No splits on the first frame — cells can't divide before any time has passed
    bool allowSplits = (frameIndex > 0);

    // Save pre-optimization shapes for each cell before Phase 1 perturbation.
    // Phase 1 can collapse a cell toward one blob of a splitting pair;
    // pre-opt position feeds PCA center and pre-opt radii set daughter sizing.
    struct PreOptShape { float majorR, minorR, x, y, z; };
    std::map<std::string, PreOptShape> preOptShapes;
    for (const auto &cell : frame.cells) {
        auto p = cell.getCellParams();
        preOptShapes[p.name] = {p.majorRadius, p.minorRadius, p.x, p.y, p.z};
    }

    // Cells that already failed a burn-in this frame — skip all further split attempts.
    std::set<std::string> splitBlacklist;

    // P(split) is driven by PREVIOUS frame's PCA elongation (stored in previousElongations).
    // If a cell looked elongated last frame, it's likely dividing now → higher P(split).
    // Frame 1 has no previous data → all cells use base rate.
    // The current frame's PCA (via trySplitCell→getSplitCells) is used for the split AXIS.
    std::map<std::string, float> rawSplitProbabilities;
    std::map<std::string, float> splitProbabilities;
    float maxRawSplitProbability = 0.0f;

    for (const auto &cell : frame.cells) {
        auto p = cell.getCellParams();
        float prevElong = 1.0f;
        auto it = previousElongations.find(p.name);
        if (it != previousElongations.end()) {
            prevElong = it->second;
        }
        const float rawProbability = allowSplits
            ? (baseSplitProb + std::max(0.0f, 1.0f - 1.0f / prevElong))
            : 0.0f;
        rawSplitProbabilities[p.name] = rawProbability;
        maxRawSplitProbability = std::max(maxRawSplitProbability, rawProbability);
    }

    const float probabilityScale =
        (allowSplits && maxRawSplitProbability > 1e-6f)
            ? (config.prob.max_split_probability / maxRawSplitProbability)
            : 0.0f;

    std::cout << "[P(split)] frame " << displayFrame
              << (allowSplits ? " (from previous frame PCA)" : " (splits disabled)") << std::endl;
    for (const auto &cell : frame.cells) {
        auto p = cell.getCellParams();
        float prevElong = 1.0f;
        auto it = previousElongations.find(p.name);
        if (it != previousElongations.end()) {
            prevElong = it->second;
        }
        const float ps = allowSplits ? (rawSplitProbabilities[p.name] * probabilityScale) : 0.0f;
        splitProbabilities[p.name] = ps;
        std::cout << "  " << p.name << " prevElong=" << prevElong << " P(split)=" << ps << std::endl;
    }

    double residSum = 0;
    double absResidSum = 0;
    double residCount = 0;
    int splitAccepted = 0;
    int splitAttempted = 0;
    int perturbAccepted = 0;

    for (size_t i = 0; i < totalIterations; ++i) {
        if (frame.cells.empty()) break;

        // Pick random cell
        std::uniform_int_distribution<size_t> cellDist(0, frame.cells.size() - 1);
        size_t cellIdx = cellDist(gen);
        auto params = frame.cells[cellIdx].getCellParams();

        // P(split) driven by PREVIOUS frame's PCA elongation for this cell.
        // Current frame's PCA is used for split axis (inside trySplitCell), not probability.
        float prevElongation = 1.0f;
        {
            auto it = previousElongations.find(params.name);
            if (it != previousElongations.end()) {
                prevElongation = it->second;
            }
        }
        float pSplit = allowSplits ? splitProbabilities[params.name] : 0.0f;

        if (pSplit > 0.0f && uniform01(gen) < pSplit
            && splitBlacklist.find(params.name) == splitBlacklist.end()) {
            // --- Try split (PCA + 500-iter burn-in via trySplitCell) ---
            splitAttempted++;
            float preOptMajorR = 0.0f, preOptMinorR = 0.0f;
            float preOptX = 0.0f, preOptY = 0.0f, preOptZ = 0.0f;
            auto pit = preOptShapes.find(params.name);
            if (pit != preOptShapes.end()) {
                preOptMajorR = pit->second.majorR;
                preOptMinorR = pit->second.minorR;
                preOptX = pit->second.x;
                preOptY = pit->second.y;
                preOptZ = pit->second.z;
            }
            std::cout << "[Split Attempt] " << params.name
                      << " frame=" << displayFrame
                      << " iter=" << i
                      << " prevElong=" << prevElongation
                      << " P(split)=" << pSplit
                      << " preOptPos=(" << preOptX << "," << preOptY << "," << preOptZ << ")"
                      << " preOptR=(" << preOptMajorR << "," << preOptMinorR << ")"
                      << std::endl;
            auto result = frame.trySplitCell(cellIdx, preOptMajorR, preOptMinorR,
                                             preOptX, preOptY, preOptZ,
                                             config.prob.split_elongation_threshold,
                                             overlapWeight,
                                             config.prob.split_fake_overlap_volume_fraction_threshold,
                                             config.prob.split_fake_volume_ratio_threshold,
                                             config.prob.split_search_radius_multiplier,
                                             config.prob.split_minor_axis_alignment_tolerance_degrees,
                                             config.prob.split_minor_axis_alignment_flatness_ratio_threshold,
                                             config.prob.split_minor_axis_alignment_min_radius_disable_threshold,
                                             config.prob.split_fake_bridge_brightness_similarity_threshold,
                                             config.prob.split_pre_burn_in_min_separation_over_major,
                                             config.prob.split_pre_burn_in_z_axis_max_abs,
                                             config.prob.split_pre_burn_in_z_axis_max_separation_over_major,
                                             config.prob.split_pre_burn_in_z_axis_min_drift_over_major,
                                             config.prob.split_post_burn_in_large_recenter_min_drift_over_major,
                                             config.prob.split_post_burn_in_large_recenter_max_cost_diff,
                                             config.prob.split_burn_in_iterations,
                                             config.prob.split_min_inside_count);
            double costDiff = result.first;
            auto callback = result.second;

            if (costDiff < -config.prob.split_cost) {
                callback(true);
                splitAccepted++;
                // Remove parent's previous elongation — daughters are new cells
                previousElongations.erase(params.name);
                std::cout << "[Split Accepted] " << params.name << " frame=" << displayFrame
                          << " iter=" << i << " diff=" << costDiff
                          << " threshold=" << -config.prob.split_cost
                          << " prevElong=" << prevElongation
                          << " P(split)=" << pSplit << std::endl;
            } else {
                callback(false);
                // Blacklist this cell from further split attempts this frame.
                // trySplitCell recomputes PCA internally (ignores cache),
                // so cache reset alone doesn't prevent repeated burn-ins.
                splitBlacklist.insert(params.name);
                std::cout << "[Split RejectCost] " << params.name
                          << " frame=" << displayFrame
                          << " iter=" << i
                          << " diff=" << costDiff
                          << " threshold=" << -config.prob.split_cost
                          << " prevElong=" << prevElongation
                          << " P(split)=" << pSplit
                          << std::endl;
            }
        } else {
            // --- Try perturbation ---
            auto result = frame.perturbCell(cellIdx, overlapWeight, sizeReductionWeight);
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

        // Progress logging
        if (i % 500 == 0 && i > 0) {
            double avgResid = residCount > 0 ? residSum / residCount : 0.0;
            double avgAbsResid = residCount > 0 ? absResidSum / residCount : 0.0;
            std::cout << "Frame " << displayFrame << " iter=" << i
                      << " perturb_accepted=" << perturbAccepted
                      << " split_attempts=" << splitAttempted
                      << " split_accepted=" << splitAccepted
                    //   << " avg_resid=" << avgResid
                      << " avg_abs_resid=" << avgAbsResid
                      << " cells=" << frame.cells.size() << std::endl;
            residSum = 0;
            absResidSum = 0;
            residCount = 0;
        }
    }

    // End of frame: compute PCA elongation for each cell on THIS frame's image.
    // Store in previousElongations for the NEXT frame's P(split) computation.
    previousElongations.clear();
    std::cout << "[Elongation for next frame]" << std::endl;
    for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
        auto p = frame.cells[ci].getCellParams();
        float elong = frame.computeElongationForCell(ci);
        previousElongations[p.name] = elong;
        std::cout << "  " << p.name << " elongation=" << elong << std::endl;
    }

    const float brightnessBlend = std::clamp(config.cell ? config.cell->brightnessUpdateBlend : 0.0f, 0.0f, 1.0f);
    if (brightnessBlend > 0.0f && config.cell) {
        const auto &realFrame = frame.getRealFrame();
        const float brightnessAmplification = std::max(0.0f, config.cell->brightnessMeanAmplification);
        for (auto &cell : frame.cells) {
            const float observedBrightness = cell.measureMeanBrightness(realFrame);
            const float amplifiedObservedBrightness = observedBrightness * brightnessAmplification;
            const float updatedBrightness =
                cell.getBrightness() * (1.0f - brightnessBlend) + amplifiedObservedBrightness * brightnessBlend;
            cell.setBrightness(updatedBrightness);
        }
        frame.regenerateSynthFrame();
    }

    if (frameIndex > 0 && config.cell) {
        const float lossThreshold = std::clamp(config.cell->volumeRecoveryLossFractionThreshold, 0.0f, 0.99f);
        const float maxScaleIncreaseFraction =
            std::max(0.0f, config.cell->volumeRecoveryMaxScaleIncreaseFraction);
        if (lossThreshold > 0.0f) {
            const Frame &previousFrame = frames[frameIndex - 1];
            std::map<std::string, Spheroid> previousCellsByName;
            for (const auto &previousCell : previousFrame.cells) {
                previousCellsByName.emplace(previousCell.getName(), previousCell);
            }

            const auto &realFrame = frame.getRealFrame();
            const auto computeVolume = [](const Spheroid &cell) {
                return static_cast<double>(cell.getMajorRadius()) *
                       static_cast<double>(cell.getBRadius()) *
                       static_cast<double>(cell.getMinorRadius());
            };

            bool recoveredAnyVolume = false;
            for (auto &cell : frame.cells) {
                const auto previousIt = previousCellsByName.find(cell.getName());
                if (previousIt == previousCellsByName.end()) {
                    continue;
                }

                const double previousVolume = computeVolume(previousIt->second);
                const double currentVolume = computeVolume(cell);
                if (previousVolume <= 1e-6) {
                    continue;
                }

                const double minimumAllowedVolume = previousVolume * (1.0 - lossThreshold);
                if (currentVolume >= minimumAllowedVolume) {
                    continue;
                }

                const float baseMajorRadius = cell.getMajorRadius();
                const float baseMinorRadius = cell.getMinorRadius();
                const float baseBrightness = cell.getBrightness();
                const float baseMeanBrightness = cell.measureMeanBrightness(realFrame);
                Spheroid bestCell = cell;
                float bestMeanBrightness = baseMeanBrightness;

                for (float scale = 1.02f; scale <= 1.0f + maxScaleIncreaseFraction + 1e-6f; scale += 0.02f) {
                    const SpheroidParams candidateParams(
                        cell.getName(),
                        cell.getX(),
                        cell.getY(),
                        cell.getZ(),
                        baseMajorRadius * scale,
                        baseMinorRadius * scale,
                        cell.getCellParams().theta_x,
                        cell.getCellParams().theta_y,
                        cell.getCellParams().theta_z,
                        baseBrightness);
                    Spheroid candidate(candidateParams);
                    const float candidateMeanBrightness = candidate.measureMeanBrightness(realFrame);
                    if (candidateMeanBrightness + 1e-6f < bestMeanBrightness) {
                        break;
                    }
                    if (candidateMeanBrightness > bestMeanBrightness + 1e-6f) {
                        bestMeanBrightness = candidateMeanBrightness;
                        bestCell = candidate;
                    }
                }

                if (bestCell.getMajorRadius() > cell.getMajorRadius() + 1e-6f ||
                    bestCell.getMinorRadius() > cell.getMinorRadius() + 1e-6f) {
                    std::cout << "[Volume Recovery] frame " << displayFrame
                              << " cell=" << cell.getName()
                              << " prev_volume=" << previousVolume
                              << " current_volume=" << currentVolume
                              << " threshold_volume=" << minimumAllowedVolume
                              << " mean_brightness=" << baseMeanBrightness
                              << " recovered_mean_brightness=" << bestMeanBrightness
                              << " majorRadius=" << cell.getMajorRadius() << "->" << bestCell.getMajorRadius()
                              << " minorRadius=" << cell.getMinorRadius() << "->" << bestCell.getMinorRadius()
                              << '\n';
                    cell = bestCell;
                    recoveredAnyVolume = true;
                }
            }

            if (recoveredAnyVolume) {
                frame.regenerateSynthFrame();
        }
    }

    Spheroid::cellConfig.brightness = originalBrightnessPerturb;
}

    if (config.cell && config.cell->flatCellRotationRefineEnabled) {
        const float flatnessThreshold =
            std::clamp(config.cell->flatCellRotationRefineFlatnessThreshold, 0.0f, 1.0f);
        const float configuredAngleStep = std::max(0.0f, config.cell->flatCellRotationRefineAngleStep);
        const float maxOffsetRadians = std::max(
            0.0f,
            config.cell->flatCellRotationRefineMaxOffsetDegrees * static_cast<float>(M_PI / 180.0));
        const int refinePasses = std::max(0, config.cell->flatCellRotationRefinePasses);
        if (configuredAngleStep > 0.0f && refinePasses > 0 && maxOffsetRadians > 0.0f) {
            const auto &realFrame = frame.getRealFrame();
            bool refinedAnyRotation = false;
            for (auto &cell : frame.cells) {
                if (cell.getMajorRadius() <= 1e-6f) {
                    continue;
                }
                const float flatnessRatio = cell.getMinorRadius() / cell.getMajorRadius();
                if (flatnessRatio > flatnessThreshold) {
                    continue;
                }

                const auto originalParams = cell.getCellParams();
                const auto [originalMeanBrightness, originalStdBrightness] =
                    cell.measureBrightnessStats(realFrame);
                Spheroid bestCell = cell;
                float bestMeanBrightness = originalMeanBrightness;
                float bestStdBrightness = originalStdBrightness;
                for (int pass = 0; pass < refinePasses; ++pass) {
                    const auto passCenter = bestCell.getCellParams();
                    Spheroid passBestCell = bestCell;
                    float passBestMeanBrightness = bestMeanBrightness;
                    float passBestStdBrightness = bestStdBrightness;

                    for (float deltaX = -maxOffsetRadians; deltaX <= maxOffsetRadians + 1e-6f; deltaX += configuredAngleStep) {
                        for (float deltaY = -maxOffsetRadians; deltaY <= maxOffsetRadians + 1e-6f; deltaY += configuredAngleStep) {
                            for (float deltaZ = -maxOffsetRadians; deltaZ <= maxOffsetRadians + 1e-6f; deltaZ += configuredAngleStep) {
                                if (std::abs(deltaX) < 1e-6f &&
                                    std::abs(deltaY) < 1e-6f &&
                                    std::abs(deltaZ) < 1e-6f) {
                                    continue;
                                }

                                const Spheroid candidate(SpheroidParams(
                                    passCenter.name,
                                    passCenter.x,
                                    passCenter.y,
                                    passCenter.z,
                                    passCenter.majorRadius,
                                    passCenter.minorRadius,
                                    passCenter.theta_x + deltaX,
                                    passCenter.theta_y + deltaY,
                                    passCenter.theta_z + deltaZ,
                                    passCenter.brightness,
                                    passCenter.equatorialAspectRatio));
                                const auto [candidateMeanBrightness, candidateStdBrightness] =
                                    candidate.measureBrightnessStats(realFrame);
                                const bool betterMean = candidateMeanBrightness > passBestMeanBrightness + 1e-6f;
                                const bool lowerStd = candidateStdBrightness < passBestStdBrightness - 1e-6f;
                                const bool nonWorseMean = candidateMeanBrightness >= passBestMeanBrightness - 1e-6f;
                                const bool nonWorseStd = candidateStdBrightness <= passBestStdBrightness + 1e-6f;
                                if ((betterMean && nonWorseStd) || (lowerStd && nonWorseMean)) {
                                    passBestCell = candidate;
                                    passBestMeanBrightness = candidateMeanBrightness;
                                    passBestStdBrightness = candidateStdBrightness;
                                }
                            }
                        }
                    }

                    if (passBestCell.getCellParams().theta_x == bestCell.getCellParams().theta_x &&
                        passBestCell.getCellParams().theta_y == bestCell.getCellParams().theta_y &&
                        passBestCell.getCellParams().theta_z == bestCell.getCellParams().theta_z) {
                        break;
                    }

                    bestCell = passBestCell;
                    bestMeanBrightness = passBestMeanBrightness;
                    bestStdBrightness = passBestStdBrightness;
                    refinedAnyRotation = true;
                }

                if (bestCell.getCellParams().theta_x != cell.getCellParams().theta_x ||
                    bestCell.getCellParams().theta_y != cell.getCellParams().theta_y ||
                    bestCell.getCellParams().theta_z != cell.getCellParams().theta_z) {
                    const auto paramsBefore = cell.getCellParams();
                    const auto paramsAfter = bestCell.getCellParams();
                    std::cout << "[Flat Rotation Refine] frame " << displayFrame
                              << " cell=" << cell.getName()
                              << " flatness_ratio=" << flatnessRatio
                              << " mean=" << originalMeanBrightness << "->" << bestMeanBrightness
                              << " std=" << originalStdBrightness << "->" << bestStdBrightness
                              << " theta=(" << paramsBefore.theta_x << "," << paramsBefore.theta_y << "," << paramsBefore.theta_z
                              << ")->(" << paramsAfter.theta_x << "," << paramsAfter.theta_y << "," << paramsAfter.theta_z << ")"
                              << '\n';
                    cell = bestCell;
                }
            }

            if (refinedAnyRotation) {
                frame.regenerateSynthFrame();
            }
        }
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
    for (size_t i = 0; i < realImages.size(); ++i)
    {
        // Save real images
        cv::imwrite(realOutputPath + "/" + std::to_string(i) + ".png", realImages[i]);
    }

    std::string synthOutputPath = outputPath + "/synth/" + std::to_string(displayFrame);
    if (!std::filesystem::exists(synthOutputPath))
    {
        std::filesystem::create_directories(synthOutputPath);
    }
    for (size_t i = 0; i < synthImages.size(); ++i)
    {
        // Save synthetic images
        cv::imwrite(synthOutputPath + "/" + std::to_string(i) + ".png", synthImages[i]);
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
        file << "file,name,x,y,z,majorRadius,minorRadius,equatorialAspectRatio,theta_x,theta_y,theta_z" << '\n';
    }

    Frame &frame = frames[frameIndex];
    std::string imageName = frame.getImageName();

    for (const auto &cell : frame.cells) {
        SpheroidParams params = cell.getCellParams();
        cell.printCellInfo();
        file << imageName << ","
             << params.name << ","
             << params.x << ","
             << params.y << ","
             << params.z << ","
             << params.majorRadius << ","
             << params.minorRadius << ","
             << params.equatorialAspectRatio << ","
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
}

unsigned int CellUniverse::length()
{
    return frames.size();
}

const std::vector<Spheroid> &CellUniverse::getCells(int frameIndex) const
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
