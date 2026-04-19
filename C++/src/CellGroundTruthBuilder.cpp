#include "../includes/CellGroundTruthBuilder.hpp"
#include "../includes/ImageHandler.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace
{
std::vector<float> collectSampledValues(const std::vector<cv::Mat> &volume)
{
    std::vector<float> values;
    values.reserve(200000);

    for (const auto &slice : volume)
    {
        for (int y = 0; y < slice.rows; ++y)
        {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x)
            {
                values.push_back(row[x]);
            }
        }
    }

    return values;
}

float percentileFromValues(std::vector<float> values, float percentileFraction)
{
    if (values.empty())
    {
        return 0.0f;
    }

    percentileFraction = std::clamp(percentileFraction, 0.0f, 1.0f);
    const size_t index = static_cast<size_t>(
        std::floor(percentileFraction * static_cast<float>(values.size() - 1)));
    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(index),
                     values.end());
    return values[index];
}

cv::Scalar markerColorForIndex(size_t index)
{
    static const std::array<cv::Scalar, 8> palette = {
        cv::Scalar(0, 0, 255),
        cv::Scalar(0, 255, 255),
        cv::Scalar(255, 255, 0),
        cv::Scalar(255, 0, 255),
        cv::Scalar(0, 255, 0),
        cv::Scalar(255, 128, 0),
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 128, 255)
    };
    return palette[index % palette.size()];
}

void drawCrosshair(cv::Mat &image, const cv::Point &center, const cv::Scalar &color, int radius)
{
    cv::circle(image, center, radius, color, 2, cv::LINE_AA);
    cv::line(image,
             cv::Point(center.x - radius - 2, center.y),
             cv::Point(center.x + radius + 2, center.y),
             color,
             2,
             cv::LINE_AA);
    cv::line(image,
             cv::Point(center.x, center.y - radius - 2),
             cv::Point(center.x, center.y + radius + 2),
             color,
             2,
             cv::LINE_AA);
}
} // namespace

CellGroundTruthBuilder::CellGroundTruthBuilder(BaseConfig config, const fs::path &outputDir)
    : config(std::move(config)),
      outputDir(outputDir),
      tracker(this->config, outputDir.string())
{
    tracker.setDebugVerbose(false);
}

float CellGroundTruthBuilder::clampf(float value, float lo, float hi)
{
    return std::max(lo, std::min(value, hi));
}

std::string CellGroundTruthBuilder::extractFrameFolderName(const fs::path &imageFile)
{
    const std::string stem = imageFile.stem().string();
    std::string digits;
    for (char ch : stem)
    {
        if (std::isdigit(static_cast<unsigned char>(ch)))
        {
            digits.push_back(ch);
        }
    }
    return digits.empty() ? stem : std::to_string(std::stoi(digits));
}

std::string CellGroundTruthBuilder::makeCellName(const std::string &frameStem, int index)
{
    std::ostringstream name;
    name << frameStem << "_cell_" << std::setfill('0') << std::setw(3) << index;
    return name.str();
}

float CellGroundTruthBuilder::estimateBackgroundValue(const std::vector<cv::Mat> &volume) const
{
    return percentileFromValues(collectSampledValues(volume),
                                clampf(config.simulation.post_process_black_percentile, 0.0f, 1.0f));
}

int CellGroundTruthBuilder::computeMinComponentVoxels() const
{
    if (!config.cell)
    {
        return 80;
    }

    const double minMajor = config.cell->minMajorRadius;
    const double minMinor = config.cell->minMinorRadius;
    const double estimatedMinVolume = (4.0 / 3.0) * M_PI * minMajor * minMajor * minMinor;
    return std::max(80, static_cast<int>(std::lround(estimatedMinVolume * 0.03)));
}

bool CellGroundTruthBuilder::componentContainsBrightSeed(
    const EmbryoBrightTracker::Comp3DStat &component,
    const std::vector<EmbryoBrightTracker::Comp3DStat> &highComponents) const
{
    for (const auto &high : highComponents)
    {
        const cv::Point3f ctr = high.center();
        if (ctr.x >= component.x0 && ctr.x <= component.x1 &&
            ctr.y >= component.y0 && ctr.y <= component.y1 &&
            ctr.z >= component.z0 && ctr.z <= component.z1)
        {
            return true;
        }
    }
    return false;
}

CellGroundTruthBuilder::DetectedCell CellGroundTruthBuilder::makeDetectedCellFromComponent(
    const EmbryoBrightTracker::Comp3DStat &component) const
{
    const float dx = static_cast<float>(component.x1 - component.x0 + 1);
    const float dy = static_cast<float>(component.y1 - component.y0 + 1);
    const float dz = static_cast<float>(component.z1 - component.z0 + 1);

    DetectedCell cell;
    cell.centerScaled = component.center();
    cell.majorRadius = std::max(0.5f * std::max(dx, dy),
                                config.cell ? static_cast<float>(config.cell->minMajorRadius) : 10.0f);
    cell.bRadius = std::max(0.5f * std::min(dx, dy), 1.0f);
    cell.minorRadius = std::max(0.5f * dz,
                                config.cell ? static_cast<float>(config.cell->minMinorRadius) : 5.0f);
    cell.bRadius = std::min(cell.bRadius, cell.majorRadius);
    cell.minorRadius = std::min(cell.minorRadius, cell.majorRadius);
    cell.voxelCount = component.vox;
    cell.meanIntensity = component.meanI();
    cell.component = component;
    return cell;
}

std::optional<CellGroundTruthBuilder::DetectedCell> CellGroundTruthBuilder::detectLocalSeededCell(
    const std::vector<cv::Mat> &volume,
    const EmbryoBrightTracker::Comp3DStat &seedComponent,
    float thresholdLow) const
{
    if (volume.empty())
    {
        return std::nullopt;
    }

    const cv::Point3f seedCenter = seedComponent.center();
    const int Z = static_cast<int>(volume.size());
    const int Y = volume[0].rows;
    const int X = volume[0].cols;

    const float minMajor = config.cell ? static_cast<float>(config.cell->minMajorRadius) : 10.0f;
    const float windowDiameter = std::clamp(
        std::max(seedComponent.diamXY() * 2.4f, minMajor * 3.0f),
        minMajor * 3.0f,
        minMajor * 5.0f);

    EmbryoBrightTracker::CellState seedState;
    seedState.id = "seed";
    seedState.center = seedCenter;
    seedState.voxelCount = std::max(seedComponent.vox, 1);
    seedState.bbox = tracker.makeBBoxForAnalysis(seedCenter, windowDiameter, Z, Y, X);
    seedState.alive = true;

    bool ok = false;
    std::vector<EmbryoBrightTracker::Comp3DStat> localComponents;
    EmbryoBrightTracker::CellState tracked = tracker.trackSingleCellByCCInBBoxForAnalysis(
        0, volume, seedState, thresholdLow, ok, &localComponents);

    if (!ok || localComponents.empty())
    {
        return std::nullopt;
    }

    int bestIndex = 0;
    float bestDist2 = std::numeric_limits<float>::max();
    for (int i = 0; i < static_cast<int>(localComponents.size()); ++i)
    {
        const cv::Point3f center = localComponents[i].center();
        const float dx = center.x - tracked.center.x;
        const float dy = center.y - tracked.center.y;
        const float dz = center.z - tracked.center.z;
        const float dist2 = dx * dx + dy * dy + dz * dz;
        if (dist2 < bestDist2)
        {
            bestDist2 = dist2;
            bestIndex = i;
        }
    }

    DetectedCell cell = makeDetectedCellFromComponent(localComponents[bestIndex]);
    cell.centerScaled = tracked.center;
    cell.voxelCount = tracked.voxelCount;
    cell.meanIntensity = tracked.meanIntensity;
    return cell;
}

std::vector<CellGroundTruthBuilder::DetectedCell> CellGroundTruthBuilder::detectCellsAtPercentile(
    const std::vector<cv::Mat> &volume,
    const std::string &frameStem,
    float percentileHigh) const
{
    if (volume.empty())
    {
        return {};
    }

    const int Z = static_cast<int>(volume.size());
    const int Y = volume[0].rows;
    const int X = volume[0].cols;

    float thresholdHigh = tracker.percentileThresholdForAnalysis(volume, percentileHigh);
    thresholdHigh = clampf(thresholdHigh, 0.05f, 0.95f);
    const float thresholdLow = clampf(thresholdHigh * 0.50f, 0.02f, 0.95f);

    std::cout << "[GroundTruth Detect] percentileHigh=" << percentileHigh
              << " thresholdHigh=" << thresholdHigh
              << " thresholdLow=" << thresholdLow
              << std::endl;

    std::vector<EmbryoBrightTracker::Comp3DStat> lowComponents =
        tracker.extractConnectedComponents3DForAnalysis(volume, thresholdLow, 0, Z - 1, 0, Y - 1, 0, X - 1, true);

    std::vector<EmbryoBrightTracker::Comp3DStat> highComponents =
        tracker.extractConnectedComponents3DForAnalysis(volume, thresholdHigh, 0, Z - 1, 0, Y - 1, 0, X - 1, true);

    const int minComponentVoxels = computeMinComponentVoxels();
    std::vector<DetectedCell> cells;
    cells.reserve(lowComponents.size());

    for (const auto &component : lowComponents)
    {
        if (component.vox < minComponentVoxels)
        {
            continue;
        }

        if (!componentContainsBrightSeed(component, highComponents))
        {
            continue;
        }

        std::vector<EmbryoBrightTracker::Comp3DStat> containedSeeds;
        for (const auto &high : highComponents)
        {
            const cv::Point3f ctr = high.center();
            if (ctr.x >= component.x0 && ctr.x <= component.x1 &&
                ctr.y >= component.y0 && ctr.y <= component.y1 &&
                ctr.z >= component.z0 && ctr.z <= component.z1)
            {
                containedSeeds.push_back(high);
            }
        }

        DetectedCell coarseCell = makeDetectedCellFromComponent(component);
        const float maxAllowedMajor = config.cell ? static_cast<float>(config.cell->maxMajorRadius) * 1.5f : 60.0f;
        const bool needsSeededSplit = containedSeeds.size() > 1 || coarseCell.majorRadius > maxAllowedMajor;

        if (needsSeededSplit)
        {
            for (const auto &seed : containedSeeds)
            {
                std::optional<DetectedCell> refined = detectLocalSeededCell(volume, seed, thresholdLow);
                if (refined)
                {
                    cells.push_back(*refined);
                }
            }
            continue;
        }

        cells.push_back(coarseCell);
    }

    std::sort(cells.begin(), cells.end(), [](const DetectedCell &lhs, const DetectedCell &rhs) {
        if (std::abs(lhs.centerScaled.y - rhs.centerScaled.y) > 1e-3f)
        {
            return lhs.centerScaled.y < rhs.centerScaled.y;
        }
        return lhs.centerScaled.x < rhs.centerScaled.x;
    });

    for (size_t i = 0; i < cells.size(); ++i)
    {
        cells[i].name = makeCellName(frameStem, static_cast<int>(i + 1));
        cells[i].zForCsv = cells[i].centerScaled.z / std::max(1.0f, config.simulation.z_scaling);
    }

    std::vector<DetectedCell> deduped;
    const float mergeRadius = config.cell ? std::max(4.0f, static_cast<float>(config.cell->minMajorRadius) * 0.75f) : 7.5f;
    const float mergeRadiusSq = mergeRadius * mergeRadius;
    for (const auto &cell : cells)
    {
        bool merged = false;
        for (auto &existing : deduped)
        {
            const float dx = existing.centerScaled.x - cell.centerScaled.x;
            const float dy = existing.centerScaled.y - cell.centerScaled.y;
            const float dz = existing.centerScaled.z - cell.centerScaled.z;
            if (dx * dx + dy * dy + dz * dz <= mergeRadiusSq)
            {
                if (cell.voxelCount > existing.voxelCount)
                {
                    existing = cell;
                }
                merged = true;
                break;
            }
        }

        if (!merged)
        {
            deduped.push_back(cell);
        }
    }

    for (size_t i = 0; i < deduped.size(); ++i)
    {
        deduped[i].name = makeCellName(frameStem, static_cast<int>(i + 1));
        deduped[i].zForCsv = deduped[i].centerScaled.z / std::max(1.0f, config.simulation.z_scaling);
    }

    return deduped;
}

std::vector<CellGroundTruthBuilder::DetectedCell> CellGroundTruthBuilder::detectCellsInVolume(
    const std::vector<cv::Mat> &volume,
    const std::string &frameStem)
{
    const std::array<float, 4> percentiles = {99.3f, 99.0f, 98.7f, 98.5f};

    std::vector<DetectedCell> bestCells;
    float bestScore = -std::numeric_limits<float>::max();
    const float minMinor = config.cell ? static_cast<float>(config.cell->minMinorRadius) : 5.0f;
    const float maxMajor = config.cell ? static_cast<float>(config.cell->maxMajorRadius) : 40.0f;
    for (float percentile : percentiles)
    {
        std::vector<DetectedCell> cells = detectCellsAtPercentile(volume, frameStem, percentile);
        int totalVoxels = 0;
        int clampedMinorCount = 0;
        int verySmallCount = 0;
        int veryLargeCount = 0;
        for (const auto &cell : cells)
        {
            totalVoxels += cell.voxelCount;
            if (cell.minorRadius <= minMinor + 0.25f)
            {
                clampedMinorCount++;
            }
            if (cell.majorRadius <= (config.cell ? static_cast<float>(config.cell->minMajorRadius) + 1.0f : 11.0f))
            {
                verySmallCount++;
            }
            if (cell.majorRadius >= maxMajor * 1.25f)
            {
                veryLargeCount++;
            }
        }

        const float score = static_cast<float>(cells.size())
                            - 1.25f * static_cast<float>(clampedMinorCount)
                            - 0.20f * static_cast<float>(verySmallCount)
                            - 2.50f * static_cast<float>(veryLargeCount);

        std::cout << "[GroundTruth Candidate] percentile=" << percentile
                  << " cells=" << cells.size()
                  << " total_voxels=" << totalVoxels
                  << " minor_floor=" << clampedMinorCount
                  << " very_small=" << verySmallCount
                  << " very_large=" << veryLargeCount
                  << " score=" << score
                  << std::endl;

        if (score > bestScore)
        {
            bestCells = std::move(cells);
            bestScore = score;
        }
    }

    if (bestCells.empty())
    {
        throw std::runtime_error("CellGroundTruthBuilder found no valid connected components after preprocessing.");
    }

    return bestCells;
}

std::vector<Spheroid> CellGroundTruthBuilder::makeSpheroids(const std::vector<DetectedCell> &cells) const
{
    std::vector<Spheroid> spheroids;
    spheroids.reserve(cells.size());

    for (const auto &cell : cells)
    {
        SpheroidParams params(cell.name,
                              cell.centerScaled.x,
                              cell.centerScaled.y,
                              cell.centerScaled.z,
                              cell.majorRadius,
                              cell.minorRadius,
                              0.0f,
                              0.0f,
                              0.0f,
                              config.cell ? config.cell->initialBrightness : 0.2f);
        params.bRadius = cell.bRadius;
        spheroids.emplace_back(params);
    }

    return spheroids;
}

void CellGroundTruthBuilder::saveInitialCsv(const fs::path &csvOutputPath,
                                            const std::string &frameFileName,
                                            const std::vector<DetectedCell> &cells) const
{
    fs::create_directories(csvOutputPath.parent_path());
    std::ofstream out(csvOutputPath);
    if (!out.is_open())
    {
        throw std::runtime_error("Failed to open output CSV: " + csvOutputPath.string());
    }

    out << "file,name,x,y,z,majorRadius,minorRadius\n";
    out << std::setprecision(15);
    for (const auto &cell : cells)
    {
        out << frameFileName << ","
            << cell.name << ","
            << cell.centerScaled.x << ","
            << cell.centerScaled.y << ","
            << cell.zForCsv << ","
            << cell.majorRadius << ","
            << cell.minorRadius << "\n";
    }
}

void CellGroundTruthBuilder::saveFrameOutputs(const fs::path &imageFile,
                                              const std::vector<cv::Mat> &realFrame,
                                              const std::vector<DetectedCell> &cells) const
{
    std::vector<Spheroid> spheroids = makeSpheroids(cells);

    Frame frame(realFrame,
                config.simulation,
                spheroids,
                outputDir.string(),
                imageFile.filename().string());
    frame.setBackgroundColor(estimateBackgroundValue(realFrame));
    frame.regenerateSynthFrame();

    std::vector<cv::Mat> realImages = frame.generateOutputFrame();
    std::vector<cv::Mat> synthImages = frame.generateOutputSynthFrame();

    std::vector<cv::Mat> markedSynthImages;
    markedSynthImages.reserve(synthImages.size());
    for (const auto &slice : synthImages)
    {
        cv::Mat colorSlice;
        cv::cvtColor(slice, colorSlice, cv::COLOR_GRAY2BGR);
        markedSynthImages.push_back(colorSlice);
    }

    for (size_t i = 0; i < cells.size(); ++i)
    {
        const auto &cell = cells[i];
        const cv::Scalar color = markerColorForIndex(i);
        const int centerZ = std::clamp(static_cast<int>(std::lround(cell.centerScaled.z)),
                                       0,
                                       static_cast<int>(markedSynthImages.size()) - 1);
        const cv::Point center(static_cast<int>(std::lround(cell.centerScaled.x)),
                               static_cast<int>(std::lround(cell.centerScaled.y)));

        for (int dz = -1; dz <= 1; ++dz)
        {
            const int z = centerZ + dz;
            if (z < 0 || z >= static_cast<int>(markedSynthImages.size()))
            {
                continue;
            }

            drawCrosshair(markedSynthImages[z], center, color, dz == 0 ? 7 : 4);
            if (dz == 0)
            {
                cv::putText(markedSynthImages[z],
                            std::to_string(i + 1),
                            cv::Point(center.x + 8, center.y - 8),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.45,
                            color,
                            1,
                            cv::LINE_AA);
            }
        }
    }

    const std::string frameDirName = extractFrameFolderName(imageFile);
    const fs::path realOutputDir = outputDir / "real" / frameDirName;
    const fs::path synthOutputDir = outputDir / "synth" / frameDirName;
    fs::create_directories(realOutputDir);
    fs::create_directories(synthOutputDir);

    for (size_t i = 0; i < realImages.size(); ++i)
    {
        cv::imwrite((realOutputDir / (std::to_string(i) + ".png")).string(), realImages[i]);
    }

    for (size_t i = 0; i < markedSynthImages.size(); ++i)
    {
        cv::imwrite((synthOutputDir / (std::to_string(i) + ".png")).string(), markedSynthImages[i]);
    }
}

std::vector<CellGroundTruthBuilder::DetectedCell> CellGroundTruthBuilder::buildInitialCsvForFrame(
    const fs::path &imageFile,
    const fs::path &csvOutputPath)
{
    if (!fs::exists(imageFile))
    {
        throw std::runtime_error("Ground-truth frame file not found: " + imageFile.string());
    }

    PathVec discoveredInput = ImageHandler::getImageFilePaths(imageFile.string(), 0, 0, config);
    if (discoveredInput.empty())
    {
        throw std::runtime_error("Ground-truth input discovery returned no files.");
    }

    std::vector<cv::Mat> realFrame = ImageHandler::loadFrame(discoveredInput.front().string(), config);
    config.simulation.z_slices = static_cast<int>(realFrame.size());

    if (config.cell)
    {
        Spheroid::cellConfig = *config.cell;
        Spheroid::cellConfig.maxZ = static_cast<float>(realFrame.size()) - 1.0f;
    }

    const std::string frameStem = imageFile.stem().string();
    std::vector<DetectedCell> cells = detectCellsInVolume(realFrame, frameStem);

    std::cout << "[GroundTruth Result] frame=" << imageFile.filename().string()
              << " detected_cells=" << cells.size()
              << std::endl;
    for (const auto &cell : cells)
    {
        std::cout << "  [Cell] name=" << cell.name
                  << " centerScaled=(" << cell.centerScaled.x << "," << cell.centerScaled.y << "," << cell.centerScaled.z << ")"
                  << " zCsv=" << cell.zForCsv
                  << " major=" << cell.majorRadius
                  << " minor=" << cell.minorRadius
                  << " vox=" << cell.voxelCount
                  << " meanI=" << cell.meanIntensity
                  << std::endl;
    }

    saveInitialCsv(csvOutputPath, imageFile.filename().string(), cells);
    saveFrameOutputs(imageFile, realFrame, cells);
    return cells;
}
