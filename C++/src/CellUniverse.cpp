#include "../includes/CellUniverse.hpp"
#include "../includes/ImageHandler.hpp"
#include <set>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <chrono>

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

static double secondsSince(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
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
        params.bRadius     *= expandFactor;
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

// Preprocessing moved to ImageHandler. Single preprocessed stack via
// ImageHandler::loadFrame.
CellUniverse::CellUniverse(std::map<std::string, std::vector<Spheroid>> initialCells,
                           PathVec imagePaths,
                           BaseConfig &config,
                           std::string outputPath,
                           int firstFrame,
                           int continueFrom)
: config(config),
  imagePaths(std::move(imagePaths)),
  outputPath(std::move(outputPath)),
  firstFrame(firstFrame),
  continueFrom(continueFrom),
  initialCells(std::move(initialCells))
{
    frames.resize(this->imagePaths.size());
    if (this->config.simulation.quit_after_preprocessing)
    {
        for (size_t i = 0; i < this->imagePaths.size(); ++i)
        {
            std::vector<cv::Mat> realFrame = ImageHandler::loadFrame(this->imagePaths[i].string(), this->config);
            if (this->config.simulation.export_preprocessed_images)
            {
                exportPreprocessedStack(realFrame, fs::path(this->outputPath), this->imagePaths[i]);
            }
        }
    }
    else if (!this->imagePaths.empty())
    {
        ensureFrameLoaded(0);
    }
}

PreviousFrameSnapshot CellUniverse::buildSnapshotFromCell(const Spheroid &cell)
{
    auto p = cell.getCellParams();

    PreviousFrameSnapshot snap;
    snap.valid = true;
    snap.shapeElongation = cell.shapeElongation();
    cell.worldLongAxis(snap.longAxisDir, snap.longAxisLength);
    snap.position = cv::Point3f(p.x, p.y, p.z);
    snap.majorRadius = p.majorRadius;
    snap.bRadius = p.bRadius;
    snap.minorRadius = p.minorRadius;
    snap.thetaX = p.theta_x;
    snap.thetaY = p.theta_y;
    snap.thetaZ = p.theta_z;
    snap.brightness = p.brightness;
    return snap;
}

void CellUniverse::seedPreviousSnapshotsFromCells(const std::vector<Spheroid> &cells,
                                                  int displayFrame,
                                                  const std::string &reason)
{
    previousSnapshots.clear();

    std::cout << "[Resume Snapshot Seed] frame " << displayFrame
              << " reason=" << reason
              << " cells=" << cells.size()
              << std::endl;
    for (const auto &cell : cells) {
        const auto p = cell.getCellParams();
        const PreviousFrameSnapshot snap = buildSnapshotFromCell(cell);
        previousSnapshots[p.name] = snap;

        std::cout << "  " << p.name
                  << " shapeElong=" << snap.shapeElongation
                  << " longAxisLen=" << snap.longAxisLength
                  << " pos=(" << snap.position.x
                  << "," << snap.position.y
                  << "," << snap.position.z << ")"
                  << std::endl;
    }
}

std::vector<Spheroid> CellUniverse::seedCellsForFrame(size_t frameIndex) const
{
    if (frameIndex >= imagePaths.size())
    {
        return {};
    }

    auto carriedIt = carriedCells.find(frameIndex);
    if (carriedIt != carriedCells.end())
    {
        return carriedIt->second;
    }

    const std::string fileName = imagePaths[frameIndex].filename().string();
    if ((continueFrom == -1 || static_cast<int>(frameIndex) < continueFrom) && initialCells.find(fileName) != initialCells.end())
    {
        return initialCells.at(fileName);
    }

    return {};
}

void CellUniverse::ensureFrameLoaded(size_t frameIndex)
{
    if (frameIndex >= imagePaths.size())
    {
        throw std::invalid_argument("CellUniverse::ensureFrameLoaded - invalid frame index");
    }

    if (frames[frameIndex])
    {
        return;
    }

    std::vector<cv::Mat> realFrame = ImageHandler::loadFrame(imagePaths[frameIndex].string(), config);
    const float currentMeanBrightness = computeStackMean(realFrame);
    const double runningReferenceMean =
        (loadedFrameMeanCount > 0)
            ? (accumulatedFrameMeanBrightness / static_cast<double>(loadedFrameMeanCount))
            : static_cast<double>(currentMeanBrightness);
    const float brightnessScale =
        (currentMeanBrightness > 1e-6f && runningReferenceMean > 1e-6)
            ? static_cast<float>(runningReferenceMean / static_cast<double>(currentMeanBrightness))
            : 1.0f;
    scaleStackBrightness(realFrame, brightnessScale);

    if (config.simulation.export_preprocessed_images)
    {
        exportPreprocessedStack(realFrame, fs::path(outputPath), imagePaths[frameIndex]);
    }

    std::cout << "[Brightness Align] frame=" << imagePaths[frameIndex].filename().string()
              << " frame_mean=" << currentMeanBrightness
              << " running_mean=" << runningReferenceMean
              << " scale=" << brightnessScale << '\n';

    config.simulation.z_slices = static_cast<int>(realFrame.size());
    Spheroid::cellConfig.maxZ = static_cast<float>(realFrame.size()) - 1.0f;

    const std::string fileName = imagePaths[frameIndex].filename().string();
    std::vector<Spheroid> cells = seedCellsForFrame(frameIndex);
    carriedCells.erase(frameIndex);

    frames[frameIndex] = std::make_unique<Frame>(realFrame, config.simulation, cells, outputPath, fileName);
    if (config.cell)
    {
        frames[frameIndex]->setBackgroundColor(config.cell->backgroundColor);
        frames[frameIndex]->regenerateSynthFrame();
    }

    accumulatedFrameMeanBrightness += static_cast<double>(currentMeanBrightness);
    ++loadedFrameMeanCount;
}
void CellUniverse::optimize(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("Invalid frame index");
    }

    const auto frameTimerStart = std::chrono::steady_clock::now();
    auto stageTimerStart = std::chrono::steady_clock::now();
    double loadSeconds = 0.0;
    double adaptiveBackgroundSeconds = 0.0;
    double initialRenderSeconds = 0.0;
    double probabilitySeconds = 0.0;
    double prePassSeconds = 0.0;
    double calibrationSeconds = 0.0;
    double phaseASeconds = 0.0;
    double phaseBSeconds = 0.0;
    double snapshotSeconds = 0.0;
    double brightnessUpdateSeconds = 0.0;

    ensureFrameLoaded(static_cast<size_t>(frameIndex));
    loadSeconds = secondsSince(stageTimerStart);
    Frame &frame = *frames[frameIndex];
    const size_t startingCellCount = frame.cells.size();

    stageTimerStart = std::chrono::steady_clock::now();
    if (frameIndex > 0) {
        ensureFrameLoaded(static_cast<size_t>(frameIndex - 1));
        const Frame &previousFrame = *frames[frameIndex - 1];
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
    adaptiveBackgroundSeconds = secondsSince(stageTimerStart);

    stageTimerStart = std::chrono::steady_clock::now();
    frame.regenerateSynthFrame();
    initialRenderSeconds = secondsSince(stageTimerStart);

    size_t totalIterations = frame.length() * config.simulation.iterations_per_cell;
    int displayFrame = firstFrame + frameIndex;

    std::cout << "[Optimize] frame " << displayFrame
              << " (" << frame.cells.size() << " cells, " << totalIterations << " iterations)" << std::endl;
    std::cout << "[Optimize Metric] frame " << displayFrame
              << " primary=seconds_per_frame"
              << " normalized=seconds_per_cell_iteration"
              << " cells_start=" << startingCellCount
              << " iterations_per_cell=" << config.simulation.iterations_per_cell
              << " parallel_threads=" << config.simulation.parallel_threads
              << std::endl;

    const bool isResumeStartFrame = (frameIndex == 0 && firstFrame > 1);
    if (isResumeStartFrame && previousSnapshots.empty() && !frame.cells.empty()) {
        seedPreviousSnapshotsFromCells(frame.cells, displayFrame, "resume_first_frame");
    }

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

    stageTimerStart = std::chrono::steady_clock::now();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

    const float overlapWeight = config.prob.overlap_penalty_weight;
    const float sizeReductionWeight = config.prob.size_reduction_penalty_weight;
    const float baseSplitProb = config.prob.P_split_base;

    // No splits on a true cold-start frame 0/1. Resume runs start with a
    // CSV that already encodes the previous frame's fitted state, so seed
    // previousSnapshots and allow snapshot-driven splits immediately.
    const bool allowSplits =
        (frameIndex > 0) || (isResumeStartFrame && !previousSnapshots.empty());

    // Cells that already failed a burn-in this frame — skip all further split attempts.
    std::set<std::string> splitBlacklist;

    // PR-B (2026-04-09): P(split) now reads from previousSnapshots (the
    // snapshot-driven architecture) instead of previousElongations. The
    // snapshot's shapeElongation is the max of (running max during
    // the previous frame, end-of-previous-frame) — capturing transient
    // elongation peaks that the optimizer flattens out before end-of-frame.
    //
    // Hard cutoff: cells whose snapshot elongation is below
    // MIN_TRIGGER_ELONGATION get P(split) = 0, blocking split attempts
    // Per-cell P(split) using the plan's formula:
    //   rawP_i     = P_split_base + max(0, 1 - 1 / snapshot.shapeElongation_i)
    //   P(split)_i = rawP_i * (P_split_max / max_i rawP_i)
    // All cells get at least `P_split_base` so missed-split recovery works.
    std::map<std::string, float> rawSplitProbabilities;
    std::map<std::string, float> splitProbabilities;
    float maxRawSplitProbability = 0.0f;

    for (const auto &cell : frame.cells) {
        auto p = cell.getCellParams();
        float prevElong = 1.0f;
        auto snapIt = previousSnapshots.find(p.name);
        if (snapIt != previousSnapshots.end() && snapIt->second.valid) {
            prevElong = snapIt->second.shapeElongation;
        }
        const float rawProbability = allowSplits
            ? (baseSplitProb + std::max(0.0f, 1.0f - 1.0f / prevElong))
            : 0.0f;
        rawSplitProbabilities[p.name] = rawProbability;
        maxRawSplitProbability = std::max(maxRawSplitProbability, rawProbability);
    }

    const float probabilityScale =
        (allowSplits && maxRawSplitProbability > 1e-6f)
            ? (config.prob.P_split_max / maxRawSplitProbability)
            : 0.0f;

    std::cout << "[P(split)] frame " << displayFrame
              << (allowSplits ? " (from snapshot)" : " (splits disabled)") << std::endl;
    for (const auto &cell : frame.cells) {
        auto p = cell.getCellParams();
        float prevElong = 1.0f;
        auto snapIt = previousSnapshots.find(p.name);
        if (snapIt != previousSnapshots.end() && snapIt->second.valid) {
            prevElong = snapIt->second.shapeElongation;
        }
        const float ps = allowSplits ? (rawSplitProbabilities[p.name] * probabilityScale) : 0.0f;
        splitProbabilities[p.name] = ps;
        std::cout << "  " << p.name << " shapeElong=" << prevElong << " P(split)=" << ps << std::endl;
    }

    double residSum = 0;
    double absResidSum = 0;
    double residCount = 0;
    int splitAccepted = 0;
    int splitAttempted = 0;
    int perturbAccepted = 0;

    // ===============================================================
    // Triaxial phased pipeline (2026-04-11 redesign):
    //   1. Classify cells into pre_classified and non_classified using
    //      snapshot.shapeElongation.
    //   2. Compute seed D1_exp/D2_exp for every pre_classified cell from
    //      the snapshot long axis.
    //   3. Run a pre-pass: for each pre_classified cell, gather bright
    //      pixels in the snapshot-centered box, PCA split, overwrite the
    //      seed with image-grounded centroids.
    //   4. Phase A — iterate non_classified cells. Split attempts use
    //      snapshot-only claim sets for other cells.
    //   5. Phase B — iterate pre_classified cells. Non_classified cells
    //      are settled, so their live positions drive the claim set.
    //      Other pre_classified cells contribute their pre-pass D1_exp/
    //      D2_exp (or live positions if they've already been processed).
    // ===============================================================

    const float T_classify = config.prob.shape_elongation_classify_threshold;

    // Classify by snapshot.shapeElongation. Cells with no snapshot (frame 1,
    // or just-split daughters) default to non_classified.
    std::set<std::string> preClassifiedNames;
    std::set<std::string> nonClassifiedNames;
    for (const auto &cell : frame.cells) {
        const std::string name = cell.getName();
        auto snapIt = previousSnapshots.find(name);
        const bool isPre = allowSplits
                        && snapIt != previousSnapshots.end()
                        && snapIt->second.valid
                        && snapIt->second.shapeElongation > T_classify;
        if (isPre) preClassifiedNames.insert(name);
        else       nonClassifiedNames.insert(name);
    }

    std::cout << "[Classify] frame " << displayFrame
              << " pre=" << preClassifiedNames.size()
              << " non=" << nonClassifiedNames.size()
              << " T=" << T_classify << std::endl;

    // Seed expected-daughter positions for pre_classified cells.
    std::map<std::string, std::pair<cv::Point3f, cv::Point3f>> expectedDaughters;
    for (const auto &name : preClassifiedNames) {
        const auto &snap = previousSnapshots[name];
        const float half = 0.5f * snap.longAxisLength;
        const cv::Point3f seed1(
            snap.position.x - half * snap.longAxisDir.x,
            snap.position.y - half * snap.longAxisDir.y,
            snap.position.z - half * snap.longAxisDir.z);
        const cv::Point3f seed2(
            snap.position.x + half * snap.longAxisDir.x,
            snap.position.y + half * snap.longAxisDir.y,
            snap.position.z + half * snap.longAxisDir.z);
        expectedDaughters[name] = {seed1, seed2};
    }

    probabilitySeconds = secondsSince(stageTimerStart);

    // ---- Pre-pass: image-ground the seeds ----
    // For each pre_classified cell, gather bright pixels in the snapshot
    // bounding box using the current seed claim sets, run PCA, overwrite
    // D1_exp/D2_exp with the image-grounded centroids. Optionally iterate —
    // round k+1 uses the D1_exp/D2_exp from round k to build the claim sets.
    stageTimerStart = std::chrono::steady_clock::now();
    const int prePassRounds = std::max(1, config.prob.expected_daughter_pre_pass_iterations);
    if (!preClassifiedNames.empty()) {
        std::cout << "[Pre-Pass] frame " << displayFrame
                  << " rounds=" << prePassRounds
                  << " preClassified=" << preClassifiedNames.size()
                  << " mode=image_grounded_pca" << std::endl;
    }
    for (int round = 0; round < prePassRounds; ++round) {
        for (const auto &name : preClassifiedNames) {
            auto it = std::find_if(frame.cells.begin(), frame.cells.end(),
                [&](const Spheroid &c) { return c.getName() == name; });
            if (it == frame.cells.end()) continue;
            size_t ci = static_cast<size_t>(std::distance(frame.cells.begin(), it));

            const auto &snap = previousSnapshots[name];
            const auto &seedsBefore = expectedDaughters[name];

            // Build claim-set for all OTHER cells at pre-pass time: each
            // non_classified contributes its snapshot.center (or live
            // position if no snapshot), each pre_classified contributes its
            // current D1_exp/D2_exp (seed on round 0, image-grounded on
            // later rounds). This matches the plan's neighbor-exclusion
            // table for the pre-pass rows.
            Frame::ClaimSet others;
            for (const auto &other : frame.cells) {
                const std::string otherName = other.getName();
                if (otherName == name) continue;
                if (preClassifiedNames.count(otherName)) {
                    auto itD = expectedDaughters.find(otherName);
                    if (itD != expectedDaughters.end()) {
                        others[otherName].push_back(itD->second.first);
                        others[otherName].push_back(itD->second.second);
                    }
                } else {
                    auto otherSnap = previousSnapshots.find(otherName);
                    if (otherSnap != previousSnapshots.end() && otherSnap->second.valid) {
                        others[otherName].push_back(otherSnap->second.position);
                    } else {
                        others[otherName].push_back(cv::Point3f(
                            other.getX(), other.getY(), other.getZ()));
                    }
                }
            }

            // Run the image-grounded PCA helper on Frame.
            cv::Point3f groundedD1, groundedD2;
            int keptPixels = 0;
            const bool ok = frame.imageGroundExpectedDaughters(
                ci, snap, others, groundedD1, groundedD2, &keptPixels);

            if (ok) {
                const cv::Point3f oldD1 = seedsBefore.first;
                const cv::Point3f oldD2 = seedsBefore.second;
                expectedDaughters[name] = {groundedD1, groundedD2};

                const cv::Point3f delta1 = groundedD1 - oldD1;
                const cv::Point3f delta2 = groundedD2 - oldD2;
                const float shift1 = static_cast<float>(cv::norm(delta1));
                const float shift2 = static_cast<float>(cv::norm(delta2));

                std::cout << "  [Pre-Pass] round=" << round
                          << " cell=" << name
                          << " shapeElong=" << snap.shapeElongation
                          << " kept=" << keptPixels
                          << " snapPos=(" << snap.position.x << "," << snap.position.y << "," << snap.position.z << ")"
                          << " seedD1=(" << oldD1.x << "," << oldD1.y << "," << oldD1.z << ")"
                          << " expD1=(" << groundedD1.x << "," << groundedD1.y << "," << groundedD1.z << ")"
                          << " shift1=" << shift1
                          << " seedD2=(" << oldD2.x << "," << oldD2.y << "," << oldD2.z << ")"
                          << " expD2=(" << groundedD2.x << "," << groundedD2.y << "," << groundedD2.z << ")"
                          << " shift2=" << shift2
                          << std::endl;
            } else {
                // PCA failed or too few pixels — keep the snapshot extrapolation.
                std::cout << "  [Pre-Pass] round=" << round
                          << " cell=" << name
                          << " shapeElong=" << snap.shapeElongation
                          << " kept=" << keptPixels
                          << " result=pca_failed_keep_snapshot_seeds"
                          << " seedD1=(" << seedsBefore.first.x << "," << seedsBefore.first.y << "," << seedsBefore.first.z << ")"
                          << " seedD2=(" << seedsBefore.second.x << "," << seedsBefore.second.y << "," << seedsBefore.second.z << ")"
                          << std::endl;
            }
        }
    }
    prePassSeconds = secondsSince(stageTimerStart);

    // ---- Per-cell position calibration pass ----
    // Refine each cell's position with tight position sigmas and frozen
    // radii BEFORE Phase A/B runs. Purpose: prevent the "Phase A parks
    // parent on one incipient daughter" pathology where radii shrink and
    // position drifts onto the stronger bright region, leaving the split
    // attempt with a half-collapsed baseline parent that only weakly
    // improves cost when replaced by daughters.
    //
    // Tight position sigmas (reuse split_burn_in_pos_sigma_scale) let the
    // cell refine its center by a few voxels per iteration but not escape
    // the snapshot footprint. Radius sigmas forced to 0 — cell cannot
    // shrink or grow during calibration. After calibration, Phase A/B runs
    // at normal sigmas and may still drift, but by then the calibrated
    // state is used as the starting point.
    stageTimerStart = std::chrono::steady_clock::now();
    const int calibrationIters = std::max(0, config.prob.split_calibration_iterations_per_cell);
    if (calibrationIters > 0 && !frame.cells.empty()) {
        const float posScale = std::max(0.0f, config.prob.split_burn_in_pos_sigma_scale);

        PerturbParams savedCalX = Spheroid::cellConfig.x;
        PerturbParams savedCalY = Spheroid::cellConfig.y;
        PerturbParams savedCalZ = Spheroid::cellConfig.z;
        PerturbParams savedCalMajor = Spheroid::cellConfig.majorRadius;
        PerturbParams savedCalB = Spheroid::cellConfig.bRadius;
        PerturbParams savedCalMinor = Spheroid::cellConfig.minorRadius;

        Spheroid::cellConfig.x.sigma = savedCalX.sigma * posScale;
        Spheroid::cellConfig.y.sigma = savedCalY.sigma * posScale;
        Spheroid::cellConfig.z.sigma = savedCalZ.sigma * posScale;
        Spheroid::cellConfig.majorRadius.sigma = 0.0f;
        Spheroid::cellConfig.bRadius.sigma     = 0.0f;
        Spheroid::cellConfig.minorRadius.sigma = 0.0f;

        std::cout << "[Calibration] frame " << displayFrame
                  << " cells=" << frame.cells.size()
                  << " itersPerCell=" << calibrationIters
                  << " posScale=" << posScale
                  << " radiusSigma=0 (frozen)"
                  << std::endl;

        // Helper to build the frame-start claim set for a given cell.
        // Used both for the centroid calibration step and the existing
        // pre-pass loop. For every OTHER cell: pre-classified contribute
        // their image-grounded expected-daughter pair (from pre-pass),
        // non-classified contribute their raw snapshot center.
        auto buildCalibrationClaimSet = [&](const std::string &selfName) -> Frame::ClaimSet {
            Frame::ClaimSet others;
            for (const auto &other : frame.cells) {
                const std::string otherName = other.getName();
                if (otherName == selfName) continue;
                if (preClassifiedNames.count(otherName)) {
                    auto itD = expectedDaughters.find(otherName);
                    if (itD != expectedDaughters.end()) {
                        others[otherName].push_back(itD->second.first);
                        others[otherName].push_back(itD->second.second);
                    }
                } else {
                    auto otherSnap = previousSnapshots.find(otherName);
                    if (otherSnap != previousSnapshots.end() && otherSnap->second.valid) {
                        others[otherName].push_back(otherSnap->second.position);
                    } else {
                        others[otherName].push_back(cv::Point3f(
                            other.getX(), other.getY(), other.getZ()));
                    }
                }
            }
            return others;
        };

        for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
            const std::string calName = frame.cells[ci].getName();
            const cv::Point3f preCalPos(
                frame.cells[ci].getX(),
                frame.cells[ci].getY(),
                frame.cells[ci].getZ());

            // Step 1: analytic centroid calibration — try the bright-pixel
            // weighted mean as a candidate position, keep it if it gives
            // lower L2 cost than the current (snapshot-inherited) position.
            // This is a one-shot move, not an iterative refinement.
            auto calSnapIt = previousSnapshots.find(calName);
            if (calSnapIt != previousSnapshots.end() && calSnapIt->second.valid) {
                Frame::ClaimSet calOthers = buildCalibrationClaimSet(calName);
                frame.calibrateCellPositionViaCentroid(ci, calSnapIt->second, calOthers);
            }

            const cv::Point3f postCentroidPos(
                frame.cells[ci].getX(),
                frame.cells[ci].getY(),
                frame.cells[ci].getZ());

            // Step 2: perturbation refinement around whatever starting
            // position the centroid step chose (either the original snapshot
            // position or the centroid, whichever was better).
            int calAccepts = 0;
            for (int it = 0; it < calibrationIters; ++it) {
                auto calResult = frame.perturbCell(ci, overlapWeight, sizeReductionWeight);
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

        Spheroid::cellConfig.x = savedCalX;
        Spheroid::cellConfig.y = savedCalY;
        Spheroid::cellConfig.z = savedCalZ;
        Spheroid::cellConfig.majorRadius = savedCalMajor;
        Spheroid::cellConfig.bRadius     = savedCalB;
        Spheroid::cellConfig.minorRadius = savedCalMinor;
    }
    calibrationSeconds = secondsSince(stageTimerStart);

    // ---- Helper: build claim-set for other cells during a split attempt ----
    auto buildOtherClaimSet = [&](const std::string &selfName,
                                  const std::set<std::string> &alreadySplitInB,
                                  const std::set<std::string> &splitRejectedInB,
                                  bool phaseB) -> Frame::ClaimSet {
        Frame::ClaimSet others;
        for (const auto &other : frame.cells) {
            const std::string otherName = other.getName();
            if (otherName == selfName) continue;

            const bool isOtherPre = preClassifiedNames.count(otherName) > 0;

            if (phaseB) {
                if (isOtherPre) {
                    // If already split-accepted in B → use the daughters'
                    // live positions. If rejected → use parent live position.
                    // Otherwise → use expected daughters seed.
                    if (alreadySplitInB.count(otherName) > 0) {
                        // Daughters are new cells with names like otherName+"0"/"1".
                        // They will appear in frame.cells with current live pos.
                        // Find them by name prefix.
                        const std::string d1n = otherName + "0";
                        const std::string d2n = otherName + "1";
                        for (const auto &c : frame.cells) {
                            if (c.getName() == d1n || c.getName() == d2n) {
                                others[c.getName()].push_back(cv::Point3f(c.getX(), c.getY(), c.getZ()));
                            }
                        }
                    } else if (splitRejectedInB.count(otherName) > 0) {
                        others[otherName].push_back(cv::Point3f(other.getX(), other.getY(), other.getZ()));
                    } else {
                        auto itD = expectedDaughters.find(otherName);
                        if (itD != expectedDaughters.end()) {
                            others[otherName].push_back(itD->second.first);
                            others[otherName].push_back(itD->second.second);
                        }
                    }
                } else {
                    // non_classified: Phase A settled them — use live pos.
                    others[otherName].push_back(cv::Point3f(other.getX(), other.getY(), other.getZ()));
                }
            } else {
                // Phase A: snapshot-only.
                if (isOtherPre) {
                    auto itD = expectedDaughters.find(otherName);
                    if (itD != expectedDaughters.end()) {
                        others[otherName].push_back(itD->second.first);
                        others[otherName].push_back(itD->second.second);
                    }
                } else {
                    auto otherSnap = previousSnapshots.find(otherName);
                    if (otherSnap != previousSnapshots.end() && otherSnap->second.valid) {
                        others[otherName].push_back(otherSnap->second.position);
                    } else {
                        others[otherName].push_back(cv::Point3f(other.getX(), other.getY(), other.getZ()));
                    }
                }
            }
        }
        return others;
    };

    // ---- Single-phase iteration helper ----
    auto runPhase = [&](const std::set<std::string> &phaseNames,
                        bool phaseB,
                        std::set<std::string> &splitAcceptedInPhase,
                        std::set<std::string> &splitRejectedInPhase,
                        size_t &iterationsExecuted) {
        if (phaseNames.empty()) return;

        const size_t perCellIters = static_cast<size_t>(config.simulation.iterations_per_cell);
        const size_t totalPhaseIters = perCellIters * phaseNames.size();
        const auto phaseTimerStart = std::chrono::steady_clock::now();

        for (size_t i = 0; i < totalPhaseIters; ++i) {
            if (frame.cells.empty()) break;

            // Find all live cells whose name is in phaseNames and not blacklisted.
            std::vector<size_t> eligible;
            eligible.reserve(phaseNames.size());
            for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
                const std::string cname = frame.cells[ci].getName();
                if (phaseNames.count(cname) == 0) continue;
                if (splitBlacklist.count(cname) > 0) {
                    // Still eligible for perturbation — the blacklist gates
                    // split attempts, not perturbation.
                }
                eligible.push_back(ci);
            }
            if (eligible.empty()) break;
            ++iterationsExecuted;

            std::uniform_int_distribution<size_t> cellDist(0, eligible.size() - 1);
            const size_t cellIdx = eligible[cellDist(gen)];
            const std::string cellName = frame.cells[cellIdx].getName();

            const float pSplit = allowSplits ? splitProbabilities[cellName] : 0.0f;
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

                const bool isPre = preClassifiedNames.count(cellName) > 0;
                const bool useSnapDir = isPre;

                Frame::ClaimSet others = buildOtherClaimSet(
                    cellName,
                    splitAcceptedInPhase,
                    splitRejectedInPhase,
                    phaseB);

                // Pass a snapshot copy whose long-axis direction / length
                // / position reflect the image-grounded {D1_exp, D2_exp}
                // from the pre-pass (when available). Without this override,
                // `trySplitCellPhased` would rebuild its self-claim from the
                // raw snapshot.longAxisDir/longAxisLength and miss the
                // grounded positions the pre-pass computed. Non-pre-classified
                // cells have no entry in `expectedDaughters`, so they pass
                // the original snapshot through unchanged.
                PreviousFrameSnapshot splitSnapshot = snapIt->second;
                if (isPre) {
                    auto itD = expectedDaughters.find(cellName);
                    if (itD != expectedDaughters.end()) {
                        const cv::Point3f &gd1 = itD->second.first;
                        const cv::Point3f &gd2 = itD->second.second;
                        const cv::Point3f mid(
                            0.5f * (gd1.x + gd2.x),
                            0.5f * (gd1.y + gd2.y),
                            0.5f * (gd1.z + gd2.z));
                        const cv::Point3f delta = gd2 - gd1;
                        const float len = static_cast<float>(cv::norm(delta));
                        if (len > 1e-3f) {
                            splitSnapshot.position = mid;
                            splitSnapshot.longAxisDir = cv::Point3f(
                                delta.x / len, delta.y / len, delta.z / len);
                            splitSnapshot.longAxisLength = len;
                        }
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
                } else {
                    splitBlacklist.insert(cellName);
                    splitRejectedInPhase.insert(cellName);

                    // Plan compensation: "revert; blacklist i; perturb i once".
                    // Run a single perturbation on the (now-reverted) parent
                    // so a rejected split doesn't waste the whole iteration
                    // budget for this cell. Re-lookup the cell index by name
                    // because the cells vector is untouched by revert but
                    // cellIdx is only cheap-safe immediately after the revert.
                    auto parentIt = std::find_if(frame.cells.begin(), frame.cells.end(),
                        [&](const Spheroid &c) { return c.getName() == cellName; });
                    if (parentIt != frame.cells.end()) {
                        const size_t parentIdx = static_cast<size_t>(
                            std::distance(frame.cells.begin(), parentIt));
                        auto compResult = frame.perturbCell(
                            parentIdx, overlapWeight, sizeReductionWeight);
                        const bool compAccept = compResult.first < 0.0;
                        compResult.second(compAccept);
                        if (compAccept) {
                            perturbAccepted++;
                            residSum += compResult.first;
                            absResidSum += std::abs(compResult.first);
                            residCount++;
                        }
                    }
                }
            } else {
                // --- Perturbation ---
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

            if ((i + 1) % 500 == 0) {
                const double phaseElapsed = secondsSince(phaseTimerStart);
                const double iterPerSecond =
                    (phaseElapsed > 1e-9)
                        ? (static_cast<double>(i + 1) / phaseElapsed)
                        : 0.0;
                std::cout << "Frame " << displayFrame
                          << (phaseB ? " PhaseB " : " PhaseA ")
                          << "iter=" << i
                          << " perturb_accepted=" << perturbAccepted
                          << " split_attempts=" << splitAttempted
                          << " split_accepted=" << splitAccepted
                          << " cells=" << frame.cells.size()
                          << " elapsed_sec=" << phaseElapsed
                          << " iter_per_sec=" << iterPerSecond
                          << " sec_per_iter="
                          << ((iterPerSecond > 1e-9) ? (1.0 / iterPerSecond) : 0.0)
                          << std::endl;
            }
        }
    };

    std::set<std::string> phaseASplitAccepted;
    std::set<std::string> phaseASplitRejected;
    size_t phaseAIterationsExecuted = 0;
    stageTimerStart = std::chrono::steady_clock::now();
    runPhase(nonClassifiedNames, /* phaseB */ false, phaseASplitAccepted, phaseASplitRejected,
             phaseAIterationsExecuted);
    phaseASeconds = secondsSince(stageTimerStart);

    std::set<std::string> phaseBSplitAccepted;
    std::set<std::string> phaseBSplitRejected;
    size_t phaseBIterationsExecuted = 0;
    stageTimerStart = std::chrono::steady_clock::now();
    runPhase(preClassifiedNames, /* phaseB */ true, phaseBSplitAccepted, phaseBSplitRejected,
             phaseBIterationsExecuted);
    phaseBSeconds = secondsSince(stageTimerStart);

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
    stageTimerStart = std::chrono::steady_clock::now();
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
        PreviousFrameSnapshot snap = buildSnapshotFromCell(frame.cells[ci]);
        previousSnapshots[p.name] = snap;

        std::cout << "  " << p.name
                  << " shapeElong=" << snap.shapeElongation
                  << " longAxisLen=" << snap.longAxisLength
                  << " pos=(" << snap.position.x
                  << "," << snap.position.y
                  << "," << snap.position.z << ")"
                  << std::endl;
    }
    snapshotSeconds = secondsSince(stageTimerStart);

    stageTimerStart = std::chrono::steady_clock::now();
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
    brightnessUpdateSeconds = secondsSince(stageTimerStart);

    const double optimizeSeconds = secondsSince(frameTimerStart);
    const size_t scheduledCellIterations =
        (static_cast<size_t>(std::max(0, calibrationIters)) * startingCellCount) +
        phaseAIterationsExecuted +
        phaseBIterationsExecuted;
    const double secondsPerCellIteration =
        scheduledCellIterations > 0
            ? optimizeSeconds / static_cast<double>(scheduledCellIterations)
            : 0.0;
    const double iterationsPerSecond =
        optimizeSeconds > 1e-9
            ? static_cast<double>(scheduledCellIterations) / optimizeSeconds
            : 0.0;

    std::cout << "[Optimize Done] frame " << displayFrame
              << " perturb_accepted=" << perturbAccepted
              << " split_attempts=" << splitAttempted
              << " split_accepted=" << splitAccepted
              << " final_cells=" << frame.cells.size() << std::endl;
    std::cout << "[Frame Timing] frame " << displayFrame
              << " total_sec=" << optimizeSeconds
              << " load_sec=" << loadSeconds
              << " adaptive_background_sec=" << adaptiveBackgroundSeconds
              << " initial_render_sec=" << initialRenderSeconds
              << " probability_classify_sec=" << probabilitySeconds
              << " prepass_sec=" << prePassSeconds
              << " calibration_sec=" << calibrationSeconds
              << " phaseA_sec=" << phaseASeconds
              << " phaseB_sec=" << phaseBSeconds
              << " snapshot_sec=" << snapshotSeconds
              << " brightness_update_sec=" << brightnessUpdateSeconds
              << " scheduled_cell_iterations=" << scheduledCellIterations
              << " phaseA_iterations=" << phaseAIterationsExecuted
              << " phaseB_iterations=" << phaseBIterationsExecuted
              << " split_attempts=" << splitAttempted
              << " sec_per_cell_iteration=" << secondsPerCellIteration
              << " iter_per_sec=" << iterationsPerSecond
              << " cells_start=" << startingCellCount
              << " cells_final=" << frame.cells.size()
              << std::endl;
}

void CellUniverse::saveImages(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("Invalid frame index");
    }

    ensureFrameLoaded(static_cast<size_t>(frameIndex));
    std::vector<Image> realImages = frames[frameIndex]->generateOutputFrame();
    std::vector<Image> synthImages = frames[frameIndex]->generateOutputSynthFrame();
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
        file << "file,name,x,y,z,majorRadius,bRadius,minorRadius,theta_x,theta_y,theta_z" << '\n';
    }

    ensureFrameLoaded(static_cast<size_t>(frameIndex));
    Frame &frame = *frames[frameIndex];
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
             << params.bRadius << ","
             << params.minorRadius << ","
             << params.theta_x << ","
             << params.theta_y << ","
             << params.theta_z
             << '\n';
    }

    std::cout << "Saved " << frame.cells.size() << " cells for frame " << (firstFrame + frameIndex)
              << " to " << cellsPath << std::endl;

    if (frameIndex > 0)
    {
        frames[frameIndex - 1].reset();
    }
}

void CellUniverse::copyCellsForward(size_t to)
{
    if (to >= frames.size())
    {
        return;
    }

    ensureFrameLoaded(to - 1);
    carriedCells[to] = frames[to - 1]->cells;
    if (config.cell) {
        for (auto &cell : carriedCells[to]) {
            cell.blendAdaptivePerturbProbabilitiesWithConfig(
                config.cell->brightnessProbabilityTrust,
                config.cell->majorRadiusProbabilityTrust,
                config.cell->minorRadiusProbabilityTrust,
                config.cell->abRatioProbabilityTrust);
        }
    }
}

unsigned int CellUniverse::length()
{
    return static_cast<unsigned int>(imagePaths.size());
}

const std::vector<Spheroid> &CellUniverse::getCells(int frameIndex) const
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("CellUniverse::getCells - invalid frameIndex");
    }
    if (!frames[frameIndex])
    {
        throw std::runtime_error("CellUniverse::getCells - requested frame is not loaded");
    }
    return frames[frameIndex]->cells;
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
