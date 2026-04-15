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
    std::vector<std::vector<cv::Mat>> loadedFrames;
    std::vector<float> frameMeanBrightness;
    loadedFrames.reserve(imagePaths.size());
    frameMeanBrightness.reserve(imagePaths.size());

    // Pass 1: load and preprocess every frame, record per-frame mean brightness.
    for (size_t i = 0; i < imagePaths.size(); ++i)
    {
        std::vector<cv::Mat> real_frame =
            ImageHandler::loadFrame(imagePaths[i].string(), config);
        frameMeanBrightness.push_back(computeStackMean(real_frame));
        loadedFrames.push_back(std::move(real_frame));
    }

    // Global mean across frames — rescale each frame so brightness is consistent.
    const float globalMeanBrightness = frameMeanBrightness.empty()
        ? 0.0f
        : std::accumulate(frameMeanBrightness.begin(), frameMeanBrightness.end(), 0.0f)
            / static_cast<float>(frameMeanBrightness.size());

    // Pass 2: scale each frame to the global mean, export, construct Frame.
    for (size_t i = 0; i < imagePaths.size(); ++i)
    {
        std::vector<cv::Mat> &real_frame = loadedFrames[i];
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

        // loadFrame interpolates frames; update config to the new slice count.
        config.simulation.z_slices = real_frame.size();
        // Propagate the interpolated-z upper bound into Ellipsoid::cellConfig so
        // the Ellipsoid constructor's z clamp uses the actual stack height.
        // Without this, cells could drift off the top/bottom of the z-stack.
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

    // ---- Per-cell position calibration pass (runs BEFORE pre-pass) ----
    // Refine each cell's position FIRST so Voronoi claim points in the
    // pre-pass use calibrated positions, not raw snapshot positions.
    // This makes neighbor exclusion more accurate.
    const int calibrationIters = std::max(0, config.prob.split_calibration_iterations_per_cell);
    if (calibrationIters > 0 && !frame.cells.empty()) {
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
                  << std::endl;

        // Claim set uses snapshot positions only (pre-pass hasn't run yet).
        auto buildCalibrationClaimSet = [&](const std::string &selfName) -> Frame::ClaimSet {
            Frame::ClaimSet others;
            for (const auto &other : frame.cells) {
                const std::string otherName = other.getName();
                if (otherName == selfName) continue;
                auto otherSnap = previousSnapshots.find(otherName);
                if (otherSnap != previousSnapshots.end() && otherSnap->second.valid) {
                    others[otherName].push_back(otherSnap->second.position);
                } else {
                    others[otherName].push_back(cv::Point3f(
                        other.getX(), other.getY(), other.getZ()));
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

            // Step 1: analytic centroid calibration
            auto calSnapIt = previousSnapshots.find(calName);
            if (calSnapIt != previousSnapshots.end() && calSnapIt->second.valid) {
                Frame::ClaimSet calOthers = buildCalibrationClaimSet(calName);
                frame.calibrateCellPositionViaCentroid(ci, calSnapIt->second, calOthers);
            }

            const cv::Point3f postCentroidPos(
                frame.cells[ci].getX(),
                frame.cells[ci].getY(),
                frame.cells[ci].getZ());

            // Step 2: perturbation refinement
            int calAccepts = 0;
            for (int it = 0; it < calibrationIters; ++it) {
                auto calResult = frame.perturbCell(ci, overlapWeight);
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

        auto buildShapeClaimSet = [&](const std::string &selfName) -> Frame::ClaimSet {
            Frame::ClaimSet others;
            for (const auto &other : frame.cells) {
                const std::string otherName = other.getName();
                if (otherName == selfName) continue;
                others[otherName].push_back(cv::Point3f(
                    other.getX(), other.getY(), other.getZ()));
            }
            return others;
        };

        // Snap-mask shape fit: pass snapshot radii as the FIXED mask.
        // The mask stays constant across iterations, so the fit cannot
        // collapse onto one emerging daughter (no mask-feedback loop).
        // Radii are free to reach true image extent (shrink or grow).
        // Newly-born cells with no snapshot fall back to their current
        // live radii inside calibrateCellShapeViaPca.
        for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
            const std::string sname = frame.cells[ci].getName();
            Frame::ClaimSet others = buildShapeClaimSet(sname);

            float maskA = 0.0f, maskB = 0.0f, maskC = 0.0f;
            auto snapIt = previousSnapshots.find(sname);
            if (snapIt != previousSnapshots.end() && snapIt->second.valid) {
                maskA = snapIt->second.aRadius;
                maskB = snapIt->second.bRadius > 1e-3f
                        ? snapIt->second.bRadius : maskA;
                maskC = snapIt->second.cRadius;
            }

            frame.calibrateCellShapeViaPca(ci, others,
                                           pcaMaxIters, pcaScale, pcaMin,
                                           maskScale, convR, convAng,
                                           updatePos, posShiftCap,
                                           maskA, maskB, maskC);
        }
        frame.regenerateSynthFrame();
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
    for (int round = 0; round < prePassRounds; ++round) {
        for (const auto &cellRef : frame.cells) {
            const std::string name = cellRef.getName();
            if (expectedDaughters.find(name) == expectedDaughters.end()) continue;
            auto it = std::find_if(frame.cells.begin(), frame.cells.end(),
                [&](const Ellipsoid &c) { return c.getName() == name; });
            if (it == frame.cells.end()) continue;
            size_t ci = static_cast<size_t>(std::distance(frame.cells.begin(), it));

            const auto &snap = previousSnapshots[name];
            const auto &seedsBefore = expectedDaughters[name];

            // Build claim-set for all OTHER cells at pre-pass time.
            // Every cell contributes BOTH its D1/D2 seed (from the seed
            // map initialized upstream — even non-splitting cells get a
            // snap-based seed) so Voronoi correctly partitions pixels
            // around dumbbell neighbors. Cells without a seed entry fall
            // back to live position. Calibration has already refined
            // positions, so live is accurate.
            Frame::ClaimSet others;
            for (const auto &other : frame.cells) {
                const std::string otherName = other.getName();
                if (otherName == name) continue;
                auto itD = expectedDaughters.find(otherName);
                if (itD != expectedDaughters.end()) {
                    others[otherName].push_back(itD->second.first);
                    others[otherName].push_back(itD->second.second);
                } else {
                    others[otherName].push_back(cv::Point3f(
                        other.getX(), other.getY(), other.getZ()));
                }
            }

            // Log claim set size for this cell.
            {
                std::ostringstream claimLog;
                for (const auto &kv : others) {
                    claimLog << " " << kv.first.substr(0,8)
                             << "[" << kv.second.size() << "]";
                }
                std::cout << "  [Pre-Pass Claims] " << name.substr(0,8)
                          << " nNeighbors=" << others.size()
                          << " claims:" << claimLog.str() << std::endl;
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
    auto runPhase = [&](const std::set<std::string> &phaseNames,
                        bool phaseB,
                        std::set<std::string> &splitAcceptedInPhase,
                        std::set<std::string> &splitRejectedInPhase) {
        if (phaseNames.empty()) return;

        const size_t perCellIters = static_cast<size_t>(config.simulation.iterations_per_cell);
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
            const std::string &cellName = frame.cells[cellIdx].getName();

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
                // Snapshot cellName before callback() invalidates the
                // const-ref (accept replaces cells[cellIdx] with daughters).
                const std::string cellNameCopy(cellName);
                callback(accept);
                if (accept) {
                    splitAccepted++;
                    splitAcceptedInPhase.insert(cellNameCopy);
                    previousSnapshots.erase(cellNameCopy);
                    // Rebuild eligible: the parent cell at cellIdx was replaced
                    // by two daughters appended to the cells vector.
                    rebuildEligible();
                } else {
                    splitBlacklist.insert(cellNameCopy);
                    splitRejectedInPhase.insert(cellNameCopy);

                    // Compensation perturb. The revert left cells[cellIdx]
                    // as the parent (in place), so no find_if needed.
                    auto compResult = frame.perturbCell(cellIdx, overlapWeight);
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
                auto result = frame.perturbCell(cellIdx, overlapWeight);
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
