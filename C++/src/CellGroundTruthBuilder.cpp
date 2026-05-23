#include "../includes/CellGroundTruthBuilder.hpp"
#include "../includes/ImageHandler.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <chrono>
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

float stackPercentile(const std::vector<cv::Mat> &volume,
                      float percentileFraction,
                      bool excludeZeros)
{
    std::vector<float> values;
    size_t totalCount = 0;
    for (const auto &slice : volume)
    {
        if (!slice.empty())
        {
            totalCount += static_cast<size_t>(slice.total());
        }
    }
    values.reserve(totalCount);

    for (const auto &slice : volume)
    {
        if (slice.empty()) continue;
        for (int y = 0; y < slice.rows; ++y)
        {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x)
            {
                const float value = row[x];
                if (!std::isfinite(value)) continue;
                if (excludeZeros && value == 0.0f) continue;
                values.push_back(value);
            }
        }
    }

    return percentileFromValues(std::move(values), percentileFraction);
}

float stackMaxValue(const std::vector<cv::Mat> &volume)
{
    float maxValue = 0.0f;
    bool foundValue = false;
    for (const auto &slice : volume)
    {
        if (slice.empty()) continue;
        double sliceMin = 0.0;
        double sliceMax = 0.0;
        cv::minMaxLoc(slice, &sliceMin, &sliceMax);
        if (!foundValue || static_cast<float>(sliceMax) > maxValue)
        {
            maxValue = static_cast<float>(sliceMax);
            foundValue = true;
        }
    }
    return foundValue ? maxValue : 0.0f;
}

float stackMeanValue(const std::vector<cv::Mat> &volume)
{
    double sum = 0.0;
    size_t count = 0;
    for (const auto &slice : volume)
    {
        if (slice.empty()) continue;
        for (int y = 0; y < slice.rows; ++y)
        {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x)
            {
                const float value = row[x];
                if (!std::isfinite(value)) continue;
                sum += value;
                ++count;
            }
        }
    }
    if (count == 0) return 0.0f;
    return static_cast<float>(sum / static_cast<double>(count));
}

std::vector<cv::Mat> loadGroundTruthRawStack(const fs::path &imageFile)
{
    std::vector<cv::Mat> rawSlices;
    std::string extension = imageFile.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    if (extension == ".tif" || extension == ".tiff")
    {
        std::vector<cv::Mat> tiffImage;
        cv::imreadmulti(imageFile.string(), tiffImage, cv::IMREAD_ANYDEPTH | cv::IMREAD_COLOR);
        if (tiffImage.empty())
        {
            throw std::runtime_error("Ground-truth TIFF has 0 slices: " + imageFile.string());
        }

        rawSlices.resize(tiffImage.size());
        for (size_t i = 0; i < tiffImage.size(); ++i)
        {
            cv::Mat slice = tiffImage[i];
            if (slice.channels() == 3)
            {
                cv::cvtColor(slice, slice, cv::COLOR_BGR2GRAY);
            }
            slice.convertTo(rawSlices[i], CV_32F);
        }
    }
    else
    {
        cv::Mat image = cv::imread(imageFile.string(), cv::IMREAD_ANYDEPTH | cv::IMREAD_COLOR);
        if (image.empty())
        {
            throw std::runtime_error("Could not read ground-truth image: " + imageFile.string());
        }
        if (image.channels() == 3)
        {
            cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        }
        cv::Mat raw;
        image.convertTo(raw, CV_32F);
        rawSlices.push_back(raw);
    }

    return rawSlices;
}

std::vector<cv::Mat> normalizeStackForPreview(const std::vector<cv::Mat> &volume)
{
    std::vector<cv::Mat> preview;
    preview.reserve(volume.size());
    if (volume.empty())
    {
        return preview;
    }

    float low = stackPercentile(volume, 0.01f, false);
    float high = stackPercentile(volume, 0.995f, false);
    if (high <= low + 1e-6f)
    {
        low = 0.0f;
        high = std::max(1.0f, stackMaxValue(volume));
    }

    const float denom = high - low;
    for (const auto &slice : volume)
    {
        cv::Mat normalized;
        slice.convertTo(normalized, CV_32F, 1.0 / denom, -low / denom);
        cv::max(normalized, 0.0f, normalized);
        cv::min(normalized, 1.0f, normalized);
        preview.push_back(normalized);
    }
    return preview;
}

cv::Mat makeTiffReadySlice(const cv::Mat &slice)
{
    if (slice.empty())
    {
        return {};
    }

    cv::Mat gray;
    if (slice.channels() == 1)
    {
        gray = slice;
    }
    else if (slice.channels() == 3)
    {
        cv::cvtColor(slice, gray, cv::COLOR_BGR2GRAY);
    }
    else if (slice.channels() == 4)
    {
        cv::cvtColor(slice, gray, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        throw std::runtime_error("Unsupported channel count for ground-truth TIFF export: " +
                                 std::to_string(slice.channels()));
    }

    cv::Mat output;
    if (gray.depth() == CV_8U)
    {
        output = gray.clone();
    }
    else
    {
        cv::Mat clipped = gray.clone();
        cv::patchNaNs(clipped, 0.0);
        cv::min(clipped, 1.0f, clipped);
        cv::max(clipped, 0.0f, clipped);
        clipped.convertTo(output, CV_8U, 255.0);
    }
    return output;
}

void writeGroundTruthTiffStack(const fs::path &path, const std::vector<cv::Mat> &stack)
{
    std::vector<cv::Mat> output;
    output.reserve(stack.size());

    cv::Size expectedSize;
    for (const auto &slice : stack)
    {
        cv::Mat converted = makeTiffReadySlice(slice);
        if (converted.empty())
        {
            continue;
        }
        if (expectedSize.empty())
        {
            expectedSize = converted.size();
        }
        else if (converted.size() != expectedSize)
        {
            throw std::runtime_error("Ground-truth TIFF export requires same-size slices: " +
                                     path.string());
        }
        output.push_back(std::move(converted));
    }

    if (output.empty())
    {
        throw std::runtime_error("Ground-truth TIFF export received an empty stack: " +
                                 path.string());
    }

    const std::vector<int> params = {
        cv::IMWRITE_TIFF_COMPRESSION, 1
    };
    if (!cv::imwritemulti(path.string(), output, params))
    {
        throw std::runtime_error("Failed to write ground-truth TIFF stack: " + path.string());
    }
}

float squaredDistance(const cv::Point3f &lhs, const cv::Point3f &rhs)
{
    const float dx = lhs.x - rhs.x;
    const float dy = lhs.y - rhs.y;
    const float dz = lhs.z - rhs.z;
    return dx * dx + dy * dy + dz * dz;
}

float squaredDistanceXY(const cv::Point3f &lhs, const cv::Point3f &rhs)
{
    const float dx = lhs.x - rhs.x;
    const float dy = lhs.y - rhs.y;
    return dx * dx + dy * dy;
}

float absoluteDistanceZ(const cv::Point3f &lhs, const cv::Point3f &rhs)
{
    return std::abs(lhs.z - rhs.z);
}
} // namespace

CellGroundTruthBuilder::CellGroundTruthBuilder(BaseConfig config, const fs::path &outputDir)
    : config(std::move(config)),
      outputDir(outputDir),
      tracker(this->config, outputDir.string()),
      activeProfile{}
{
    tracker.setDebugVerbose(false);
    activeProfile.effectiveZScaling = std::max(1.0f, this->config.simulation.z_scaling);
}

float CellGroundTruthBuilder::clampf(float value, float lo, float hi)
{
    return std::max(lo, std::min(value, hi));
}

std::string CellGroundTruthBuilder::toLowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
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

CellGroundTruthBuilder::DatasetProfile CellGroundTruthBuilder::inferDatasetProfile(const fs::path &imageFile) const
{
    DatasetProfile profile;
    profile.effectiveZScaling = std::max(1.0f, config.simulation.z_scaling);

    const float minMajor = config.cell ? static_cast<float>(config.cell->minARadius) : 10.0f;
    const int minComponentVoxels = computeMinComponentVoxels();
    profile.minHighSeedVoxels = std::max(8, minComponentVoxels / 8);
    profile.seedMergeDistance = std::max(6.0f, minMajor * 0.95f);
    profile.seedSplitSeparation = std::max(8.0f, minMajor * 1.10f);
    profile.dedupDistance = std::max(5.0f, minMajor * 0.90f);

    const std::string imagePath = toLowerCopy(imageFile.string());
    if (imagePath.find("simulated_nuclei_hl60 cells_stained_with_hoechst") != std::string::npos ||
        imagePath.find("fluo-n3dh-sim+") != std::string::npos)
    {
        profile.label = "hl60_sim";
        profile.effectiveZScaling = 0.200f / 0.125f;
        profile.minHighSeedVoxels = std::max(profile.minHighSeedVoxels, 14);
        profile.seedMergeDistance = std::max(profile.seedMergeDistance, minMajor * 1.10f);
        profile.seedSplitSeparation = std::max(profile.seedSplitSeparation, minMajor * 1.35f);
        profile.dedupDistance = std::max(profile.dedupDistance, minMajor * 1.05f);
        return profile;
    }

    if (imagePath.find("c.elegans_developing embryo_fluo-n3dh-ce_training") != std::string::npos ||
        imagePath.find("fluo-n3dh-ce") != std::string::npos)
    {
        profile.label = "celegans_embryo";
        profile.minHighSeedVoxels = std::max(profile.minHighSeedVoxels, 14);
        profile.seedMergeDistance = std::max(profile.seedMergeDistance, minMajor * 1.20f);
        profile.seedSplitSeparation = std::max(profile.seedSplitSeparation, minMajor * 1.50f);
        profile.dedupDistance = std::max(profile.dedupDistance, minMajor * 1.20f);
        return profile;
    }

    if (imagePath.find("original_data") != std::string::npos)
    {
        profile.label = "original_data";
    }

    return profile;
}

float CellGroundTruthBuilder::effectiveZScaling() const
{
    return std::max(1.0f, activeProfile.effectiveZScaling);
}

float CellGroundTruthBuilder::estimateBackgroundValue(const std::vector<cv::Mat> &volume) const
{
    return percentileFromValues(collectSampledValues(volume),
                                clampf(config.simulation.post_alignment_black_percentile, 0.0f, 1.0f));
}

int CellGroundTruthBuilder::computeMinComponentVoxels() const
{
    if (!config.cell)
    {
        return 80;
    }

    const double minMajor = config.cell->minARadius;
    const double minMinor = config.cell->minCRadius;
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
    DetectedCell cell;
    cell.centerScaled = component.center();
    cell.centerScaled.z *= effectiveZScaling();
    estimateAdaptiveRadii(component, cell.majorRadius, cell.bRadius, cell.minorRadius);
    cell.voxelCount = component.vox;
    cell.meanIntensity = component.meanI();
    cell.component = component;
    return cell;
}

void CellGroundTruthBuilder::estimateAdaptiveRadii(const EmbryoBrightTracker::Comp3DStat &component,
                                                   float &majorRadius,
                                                   float &bRadius,
                                                   float &minorRadius) const
{
    const float dx = static_cast<float>(component.x1 - component.x0 + 1);
    const float dy = static_cast<float>(component.y1 - component.y0 + 1);
    const float dz = static_cast<float>(component.z1 - component.z0 + 1) * effectiveZScaling();
    const float xyMaxRadius = 0.5f * std::max(dx, dy);
    const float xyMinRadius = 0.5f * std::min(dx, dy);
    const float zRadius = 0.5f * dz;
    const float componentVolume = std::max(1.0f, static_cast<float>(component.vox));
    const float equivalentRadius = std::cbrt((3.0f * componentVolume) / (4.0f * static_cast<float>(M_PI)));
    const float bboxVolume = std::max(1.0f, dx * dy * dz);
    const float fillRatio = clampf(componentVolume / bboxVolume, 0.0f, 1.0f);

    const auto blendTowardAxis = [&](float axisRadius, float supportFactor) {
        const float safeAxis = std::max(axisRadius, 1e-3f);
        const float safeEquiv = std::max(equivalentRadius, 1e-3f);
        const float agreement = std::min(safeAxis, safeEquiv) / std::max(safeAxis, safeEquiv);
        const float blend = clampf(fillRatio * agreement * supportFactor, 0.0f, 1.0f);
        return equivalentRadius + blend * (axisRadius - equivalentRadius);
    };

    const float minMajor = config.cell ? static_cast<float>(config.cell->minARadius) : 10.0f;
    const float maxMajor = config.cell ? static_cast<float>(config.cell->maxARadius) : 50.0f;
    const float minMinor = config.cell ? static_cast<float>(config.cell->minCRadius) : 5.0f;
    const float maxMinor = config.cell ? static_cast<float>(config.cell->maxCRadius) : 45.0f;
    const float minB = (config.cell && config.cell->maxBRadius > 0.0f)
        ? static_cast<float>(config.cell->minBRadius)
        : minMajor;
    const float maxB = (config.cell && config.cell->maxBRadius > 0.0f)
        ? static_cast<float>(config.cell->maxBRadius)
        : maxMajor;

    const float xySupport = clampf(xyMinRadius / std::max(equivalentRadius, 1e-3f), 0.35f, 1.0f);
    const float zSupport = clampf(zRadius / std::max(equivalentRadius, 1e-3f), 0.10f, 1.0f);

    majorRadius = clampf(std::max(equivalentRadius, blendTowardAxis(xyMaxRadius, 1.0f)), minMajor, maxMajor);
    bRadius = clampf(blendTowardAxis(xyMinRadius, xySupport), minB, majorRadius);
    minorRadius = clampf(blendTowardAxis(zRadius, zSupport * zSupport), minMinor, std::min(maxMinor, majorRadius));
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

    const float minMajor = config.cell ? static_cast<float>(config.cell->minARadius) : 10.0f;
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
    cell.centerScaled = cv::Point3f(
        tracked.center.x,
        tracked.center.y,
        tracked.center.z * effectiveZScaling());
    cell.voxelCount = tracked.voxelCount;
    cell.meanIntensity = tracked.meanIntensity;
    return cell;
}

std::vector<EmbryoBrightTracker::Comp3DStat> CellGroundTruthBuilder::collapseNearbySeeds(
    const std::vector<EmbryoBrightTracker::Comp3DStat> &seeds,
    float mergeDistance) const
{
    if (seeds.empty())
    {
        return {};
    }

    std::vector<EmbryoBrightTracker::Comp3DStat> sortedSeeds = seeds;
    std::sort(sortedSeeds.begin(), sortedSeeds.end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.vox != rhs.vox)
        {
            return lhs.vox > rhs.vox;
        }
        return lhs.meanI() > rhs.meanI();
    });

    const float mergeDistanceSq = mergeDistance * mergeDistance;
    std::vector<EmbryoBrightTracker::Comp3DStat> merged;
    for (const auto &seed : sortedSeeds)
    {
        bool duplicate = false;
        for (const auto &existing : merged)
        {
            const float distXY = std::sqrt(squaredDistanceXY(seed.center(), existing.center()));
            const float distZ = absoluteDistanceZ(seed.center(), existing.center());
            const float sizeRatio = static_cast<float>(std::min(seed.vox, existing.vox)) /
                                    static_cast<float>(std::max(seed.vox, existing.vox));
            const float intensityRatio = std::min(seed.meanI(), existing.meanI()) /
                                         std::max(std::max(seed.meanI(), existing.meanI()), 1e-6f);

            const bool nearlySamePeak =
                distXY * distXY <= (0.40f * mergeDistance) * (0.40f * mergeDistance) &&
                distZ <= mergeDistance;
            const bool likelySatellite =
                distXY * distXY <= mergeDistanceSq &&
                distZ <= mergeDistance * 2.0f &&
                sizeRatio < 0.45f &&
                intensityRatio < 0.95f;

            if (nearlySamePeak || likelySatellite)
            {
                duplicate = true;
                break;
            }
        }

        if (!duplicate)
        {
            merged.push_back(seed);
        }
    }

    return merged;
}

bool CellGroundTruthBuilder::shouldSplitCoarseComponent(
    const DetectedCell &coarseCell,
    const std::vector<EmbryoBrightTracker::Comp3DStat> &containedSeeds) const
{
    const float maxAllowedMajor = config.cell ? static_cast<float>(config.cell->maxARadius) * 1.15f : 46.0f;
    if (coarseCell.majorRadius > maxAllowedMajor && !containedSeeds.empty())
    {
        return true;
    }

    if (containedSeeds.size() <= 1)
    {
        return false;
    }

    float maxSeedSeparationXY = 0.0f;
    for (size_t i = 0; i < containedSeeds.size(); ++i)
    {
        for (size_t j = i + 1; j < containedSeeds.size(); ++j)
        {
            maxSeedSeparationXY = std::max(
                maxSeedSeparationXY,
                std::sqrt(squaredDistanceXY(containedSeeds[i].center(), containedSeeds[j].center())));
        }
    }

    const float dominantSeedVox = static_cast<float>(containedSeeds.front().vox);
    const float secondSeedVox = static_cast<float>(containedSeeds[1].vox);
    const float balance = secondSeedVox / std::max(dominantSeedVox, 1.0f);

    const float softSplitThreshold = std::max(4.0f, coarseCell.majorRadius * 0.45f);
    if (balance >= 0.28f && maxSeedSeparationXY >= softSplitThreshold)
    {
        return true;
    }

    const float hardSplitThreshold = std::max(activeProfile.seedSplitSeparation, coarseCell.majorRadius * 0.75f);
    return maxSeedSeparationXY >= hardSplitThreshold;
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
    const float maxValue = stackMaxValue(volume);
    const bool unitScaleInput = maxValue <= 1.5f;
    if (unitScaleInput)
    {
        thresholdHigh = clampf(thresholdHigh, 0.05f, 0.95f);
    }
    else
    {
        thresholdHigh = std::clamp(thresholdHigh, 0.0f, maxValue);
    }
    const float thresholdLow = unitScaleInput
        ? clampf(thresholdHigh * 0.50f, 0.02f, 0.95f)
        : std::clamp(thresholdHigh * 0.50f, 0.0f, maxValue);

    std::cout << "[GroundTruth Detect] percentileHigh=" << percentileHigh
              << " input_scale=" << (unitScaleInput ? "unit" : "raw")
              << " maxValue=" << maxValue
              << " thresholdHigh=" << thresholdHigh
              << " thresholdLow=" << thresholdLow
              << std::endl;

    std::vector<EmbryoBrightTracker::Comp3DStat> lowComponents =
        tracker.extractConnectedComponents3DForAnalysis(volume, thresholdLow, 0, Z - 1, 0, Y - 1, 0, X - 1, true);

    std::vector<EmbryoBrightTracker::Comp3DStat> highComponents =
        tracker.extractConnectedComponents3DForAnalysis(volume, thresholdHigh, 0, Z - 1, 0, Y - 1, 0, X - 1, true);

    std::vector<EmbryoBrightTracker::Comp3DStat> strongHighComponents;
    strongHighComponents.reserve(highComponents.size());
    for (const auto &component : highComponents)
    {
        if (component.vox >= activeProfile.minHighSeedVoxels)
        {
            strongHighComponents.push_back(component);
        }
    }

    const int minComponentVoxels = computeMinComponentVoxels();
    std::vector<DetectedCell> cells;
    cells.reserve(lowComponents.size());

    for (const auto &component : lowComponents)
    {
        if (component.vox < minComponentVoxels)
        {
            continue;
        }

        if (!componentContainsBrightSeed(component, strongHighComponents))
        {
            continue;
        }

        std::vector<EmbryoBrightTracker::Comp3DStat> containedSeeds;
        for (const auto &high : strongHighComponents)
        {
            const cv::Point3f ctr = high.center();
            if (ctr.x >= component.x0 && ctr.x <= component.x1 &&
                ctr.y >= component.y0 && ctr.y <= component.y1 &&
                ctr.z >= component.z0 && ctr.z <= component.z1)
            {
                containedSeeds.push_back(high);
            }
        }
        containedSeeds = collapseNearbySeeds(containedSeeds, activeProfile.seedMergeDistance);

        DetectedCell coarseCell = makeDetectedCellFromComponent(component);
        const bool needsSeededSplit = shouldSplitCoarseComponent(coarseCell, containedSeeds);

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
        cells[i].zForCsv = cells[i].centerScaled.z / effectiveZScaling();
    }

    std::vector<DetectedCell> deduped;
    for (const auto &cell : cells)
    {
        bool merged = false;
        for (auto &existing : deduped)
        {
            const float mergeRadius = std::max(activeProfile.dedupDistance,
                                               0.60f * std::min(existing.majorRadius, cell.majorRadius));
            if (squaredDistanceXY(existing.centerScaled, cell.centerScaled) <= mergeRadius * mergeRadius &&
                absoluteDistanceZ(existing.centerScaled, cell.centerScaled) <= std::max(10.0f, mergeRadius * 2.0f))
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

    deduped = pruneLikelySatelliteCells(deduped);

    for (size_t i = 0; i < deduped.size(); ++i)
    {
        deduped[i].name = makeCellName(frameStem, static_cast<int>(i + 1));
        deduped[i].zForCsv = deduped[i].centerScaled.z / effectiveZScaling();
    }

    return deduped;
}

std::vector<CellGroundTruthBuilder::DetectedCell> CellGroundTruthBuilder::pruneLikelySatelliteCells(
    const std::vector<DetectedCell> &cells) const
{
    if (cells.empty())
    {
        return {};
    }

    std::vector<int> voxelCounts;
    voxelCounts.reserve(cells.size());
    for (const auto &cell : cells)
    {
        voxelCounts.push_back(cell.voxelCount);
    }
    std::sort(voxelCounts.begin(), voxelCounts.end());
    const float medianVoxelCount = static_cast<float>(voxelCounts[voxelCounts.size() / 2]);
    const float minMinor = config.cell ? static_cast<float>(config.cell->minCRadius) : 5.0f;
    const float minMajor = config.cell ? static_cast<float>(config.cell->minARadius) : 10.0f;
    std::vector<float> majorRadii;
    std::vector<float> meanIntensities;
    majorRadii.reserve(cells.size());
    meanIntensities.reserve(cells.size());
    for (const auto &cell : cells)
    {
        majorRadii.push_back(cell.majorRadius);
        meanIntensities.push_back(cell.meanIntensity);
    }
    std::sort(majorRadii.begin(), majorRadii.end());
    std::sort(meanIntensities.begin(), meanIntensities.end());
    const float medianMajor = majorRadii[majorRadii.size() / 2];
    const float medianMeanIntensity = meanIntensities[meanIntensities.size() / 2];
    const float satelliteVoxelCutoff = std::max(3500.0f, medianVoxelCount * 0.50f);
    const float embryoThinFragmentCutoff = std::max(10000.0f, medianVoxelCount * 0.35f);
    const float smallSatelliteVoxelCutoff = std::max(2200.0f, medianVoxelCount * 0.35f);
    const float smallSatelliteMajorCutoff = std::max(minMajor + 1.0f, medianMajor * 0.82f);
    const float microFragmentVoxelCutoff = std::max(1400.0f, medianVoxelCount * 0.16f);
    const float microFragmentMajorCutoff = std::max(minMajor, medianMajor * 0.62f);
    const float microFragmentIntensityCutoff = std::max(0.08f, medianMeanIntensity * 0.70f);

    std::vector<DetectedCell> filtered;
    filtered.reserve(cells.size());
    for (size_t i = 0; i < cells.size(); ++i)
    {
        const auto &cell = cells[i];
        bool isSatellite = false;
        if (activeProfile.label == "celegans_embryo" &&
            cell.minorRadius <= minMinor + 0.25f &&
            static_cast<float>(cell.voxelCount) < embryoThinFragmentCutoff)
        {
            isSatellite = true;
        }

        if (!isSatellite &&
            cell.minorRadius <= minMinor + 0.25f &&
            static_cast<float>(cell.voxelCount) < satelliteVoxelCutoff)
        {
            for (size_t j = 0; j < cells.size(); ++j)
            {
                if (i == j)
                {
                    continue;
                }

                const auto &other = cells[j];
                if (other.voxelCount < std::max(static_cast<int>(cell.voxelCount * 1.5f),
                                                static_cast<int>(medianVoxelCount * 0.85f)))
                {
                    continue;
                }

                const float nearThreshold = std::max(activeProfile.dedupDistance * 1.50f,
                                                     0.90f * std::max(cell.majorRadius, other.majorRadius));
                if (squaredDistanceXY(cell.centerScaled, other.centerScaled) <= nearThreshold * nearThreshold &&
                    absoluteDistanceZ(cell.centerScaled, other.centerScaled) <= std::max(16.0f, nearThreshold * 2.5f))
                {
                    isSatellite = true;
                    break;
                }
            }
        }

        if (!isSatellite &&
            static_cast<float>(cell.voxelCount) < smallSatelliteVoxelCutoff &&
            cell.majorRadius < smallSatelliteMajorCutoff)
        {
            for (size_t j = 0; j < cells.size(); ++j)
            {
                if (i == j)
                {
                    continue;
                }

                const auto &other = cells[j];
                if (other.voxelCount < std::max(static_cast<int>(cell.voxelCount * 1.8f),
                                                static_cast<int>(medianVoxelCount * 0.90f)))
                {
                    continue;
                }

                const float nearThreshold = std::max(activeProfile.dedupDistance * 1.75f,
                                                     0.95f * std::max(cell.majorRadius, other.majorRadius));
                if (squaredDistanceXY(cell.centerScaled, other.centerScaled) <= nearThreshold * nearThreshold &&
                    absoluteDistanceZ(cell.centerScaled, other.centerScaled) <= std::max(18.0f, nearThreshold * 3.0f))
                {
                    isSatellite = true;
                    break;
                }
            }
        }

        if (!isSatellite &&
            static_cast<float>(cell.voxelCount) < microFragmentVoxelCutoff &&
            cell.majorRadius < microFragmentMajorCutoff &&
            cell.meanIntensity < microFragmentIntensityCutoff)
        {
            for (size_t j = 0; j < cells.size(); ++j)
            {
                if (i == j)
                {
                    continue;
                }

                const auto &other = cells[j];
                const float nearThreshold = std::max(activeProfile.dedupDistance * 2.25f,
                                                     1.10f * std::max(cell.majorRadius, other.majorRadius));
                if (squaredDistanceXY(cell.centerScaled, other.centerScaled) <= nearThreshold * nearThreshold &&
                    absoluteDistanceZ(cell.centerScaled, other.centerScaled) <= std::max(20.0f, nearThreshold * 4.0f))
                {
                    isSatellite = true;
                    break;
                }
            }
        }

        if (!isSatellite)
        {
            filtered.push_back(cell);
        }
    }

    return filtered;
}

float CellGroundTruthBuilder::scoreCandidateCells(const std::vector<DetectedCell> &cells,
                                                  int &totalVoxels,
                                                  int &clampedMinorCount,
                                                  int &verySmallCount,
                                                  int &veryLargeCount,
                                                  int &nearDuplicatePairs,
                                                  int &flattenedCount,
                                                  int &tinyFragmentCount) const
{
    totalVoxels = 0;
    clampedMinorCount = 0;
    verySmallCount = 0;
    veryLargeCount = 0;
    nearDuplicatePairs = 0;
    flattenedCount = 0;
    tinyFragmentCount = 0;

    if (cells.empty())
    {
        return -std::numeric_limits<float>::max();
    }

    const float minMinor = config.cell ? static_cast<float>(config.cell->minCRadius) : 5.0f;
    const float minMajor = config.cell ? static_cast<float>(config.cell->minARadius) : 10.0f;
    const float maxMajor = config.cell ? static_cast<float>(config.cell->maxARadius) : 40.0f;

    std::vector<float> majorRadii;
    std::vector<int> voxelCounts;
    majorRadii.reserve(cells.size());
    voxelCounts.reserve(cells.size());
    for (const auto &cell : cells)
    {
        totalVoxels += cell.voxelCount;
        majorRadii.push_back(cell.majorRadius);
        voxelCounts.push_back(cell.voxelCount);
        if (cell.minorRadius <= minMinor + 0.25f)
        {
            clampedMinorCount++;
        }
        if (cell.majorRadius <= minMajor + 1.0f)
        {
            verySmallCount++;
        }
        if (cell.majorRadius >= maxMajor * 1.25f)
        {
            veryLargeCount++;
        }
    }

    std::sort(voxelCounts.begin(), voxelCounts.end());
    for (size_t i = 0; i < cells.size(); ++i)
    {
        for (size_t j = i + 1; j < cells.size(); ++j)
        {
            const float nearThreshold = std::max(4.0f, 0.55f * std::min(cells[i].majorRadius, cells[j].majorRadius));
            if (squaredDistanceXY(cells[i].centerScaled, cells[j].centerScaled) <= nearThreshold * nearThreshold &&
                absoluteDistanceZ(cells[i].centerScaled, cells[j].centerScaled) <= std::max(10.0f, nearThreshold * 2.0f))
            {
                nearDuplicatePairs++;
            }
        }
    }

    std::sort(majorRadii.begin(), majorRadii.end());
    const float medianMajor = majorRadii[majorRadii.size() / 2];
    const float medianVoxelCount = static_cast<float>(voxelCounts[voxelCounts.size() / 2]);
    const float meanVoxelCount = static_cast<float>(totalVoxels) / static_cast<float>(cells.size());

    for (const auto &cell : cells)
    {
        const float ratio = cell.minorRadius / std::max(cell.majorRadius, 1e-3f);
        if (ratio < 0.72f)
        {
            flattenedCount++;
        }
        if (static_cast<float>(cell.voxelCount) < std::max(1200.0f, medianVoxelCount * 0.25f) &&
            cell.majorRadius < medianMajor * 0.85f)
        {
            tinyFragmentCount++;
        }
    }

    return 0.028f * meanVoxelCount
           + 0.20f * medianMajor
           - 2.50f * static_cast<float>(clampedMinorCount)
           - 1.25f * static_cast<float>(verySmallCount)
           - 3.00f * static_cast<float>(veryLargeCount)
           - 1.50f * static_cast<float>(nearDuplicatePairs)
           - 1.75f * static_cast<float>(flattenedCount)
           - 1.25f * static_cast<float>(tinyFragmentCount);
}

std::vector<CellGroundTruthBuilder::DetectedCell> CellGroundTruthBuilder::detectCellsInVolume(
    const std::vector<cv::Mat> &volume,
    const std::string &frameStem)
{
    const std::array<float, 4> percentiles = {99.3f, 99.0f, 98.7f, 98.5f};

    std::vector<DetectedCell> bestCells;
    float bestScore = -std::numeric_limits<float>::max();
    for (float percentile : percentiles)
    {
        std::vector<DetectedCell> cells = detectCellsAtPercentile(volume, frameStem, percentile);
        int totalVoxels = 0;
        int clampedMinorCount = 0;
        int verySmallCount = 0;
        int veryLargeCount = 0;
        int nearDuplicatePairs = 0;
        int flattenedCount = 0;
        int tinyFragmentCount = 0;
        const float score = scoreCandidateCells(
            cells,
            totalVoxels,
            clampedMinorCount,
            verySmallCount,
            veryLargeCount,
            nearDuplicatePairs,
            flattenedCount,
            tinyFragmentCount);

        std::cout << "[GroundTruth Candidate] percentile=" << percentile
                  << " cells=" << cells.size()
                  << " total_voxels=" << totalVoxels
                  << " minor_floor=" << clampedMinorCount
                  << " very_small=" << verySmallCount
                  << " very_large=" << veryLargeCount
                  << " near_pairs=" << nearDuplicatePairs
                  << " flattened=" << flattenedCount
                  << " tiny_fragments=" << tinyFragmentCount
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

std::vector<Ellipsoid> CellGroundTruthBuilder::makeEllipsoids(const std::vector<DetectedCell> &cells) const
{
    std::vector<Ellipsoid> ellipsoids;
    ellipsoids.reserve(cells.size());

    for (const auto &cell : cells)
    {
        EllipsoidParams params(cell.name,
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
        ellipsoids.emplace_back(params);
    }

    return ellipsoids;
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
    std::vector<DetectedCell> displayCells = cells;
    const float zScale = effectiveZScaling();
    for (auto &cell : displayCells)
    {
        cell.centerScaled.z = cell.zForCsv;
        if (zScale > 1e-6f)
        {
            cell.minorRadius = std::max(1.0f, cell.minorRadius / zScale);
        }
    }

    std::vector<cv::Mat> displayFrame = normalizeStackForPreview(realFrame);
    std::vector<Ellipsoid> ellipsoids = makeEllipsoids(displayCells);

    Frame frame(displayFrame,
                config.simulation,
                ellipsoids,
                outputDir.string(),
                imageFile.filename().string());
    frame.setBackgroundColor(estimateBackgroundValue(displayFrame));
    frame.regenerateSynthFrame();

    std::vector<cv::Mat> realImages = frame.generateOutputFrame();
    std::vector<cv::Mat> synthImages = frame.generateOutputSynthFrame();

    const std::string frameDirName = extractFrameFolderName(imageFile);
    fs::create_directories(outputDir);
    const fs::path realTiffPath = outputDir / (frameDirName + "_real.tif");
    const fs::path synthTiffPath = outputDir / (frameDirName + "_synth.tif");
    writeGroundTruthTiffStack(realTiffPath, realImages);
    writeGroundTruthTiffStack(synthTiffPath, synthImages);
    std::cout << "[GroundTruth Output] real_tif=" << realTiffPath
              << " synth_tif=" << synthTiffPath
              << std::endl;

    const fs::path realOutputDir = outputDir / "real" / frameDirName;
    const fs::path synthOutputDir = outputDir / "synth" / frameDirName;
    fs::create_directories(realOutputDir);
    fs::create_directories(synthOutputDir);

    for (size_t i = 0; i < realImages.size(); ++i)
    {
        cv::imwrite((realOutputDir / (std::to_string(i) + ".png")).string(), realImages[i]);
    }

    for (size_t i = 0; i < synthImages.size(); ++i)
    {
        cv::imwrite((synthOutputDir / (std::to_string(i) + ".png")).string(), synthImages[i]);
    }
}

std::vector<CellGroundTruthBuilder::DetectedCell> CellGroundTruthBuilder::buildInitialCsvForFrame(
    const fs::path &imageFile,
    const fs::path &csvOutputPath)
{
    using Clock = std::chrono::steady_clock;
    const auto totalStart = Clock::now();
    auto lastMark = totalStart;
    auto logElapsed = [&](const std::string &stage) {
        const auto now = Clock::now();
        const double stageSeconds =
            std::chrono::duration<double>(now - lastMark).count();
        const double totalSeconds =
            std::chrono::duration<double>(now - totalStart).count();
        std::cout << "[GroundTruth Timing] frame=" << imageFile.filename().string()
                  << " stage=" << stage
                  << " stage_sec=" << std::fixed << std::setprecision(6) << stageSeconds
                  << " total_sec=" << totalSeconds
                  << std::defaultfloat
                  << std::endl;
        lastMark = now;
    };

    if (!fs::exists(imageFile))
    {
        throw std::runtime_error("Ground-truth frame file not found: " + imageFile.string());
    }

    activeProfile = inferDatasetProfile(imageFile);
    config.simulation.z_scaling = effectiveZScaling();
    std::cout << "[GroundTruth Dataset] profile=" << activeProfile.label
              << " effective_z_scaling=" << effectiveZScaling();
    if (activeProfile.label == "hl60_sim")
    {
        std::cout << " voxel_size_xyz=(0.125,0.125,0.200)";
    }
    std::cout << std::endl;

    PathVec discoveredInput = ImageHandler::getImageFilePaths(imageFile.string(), 0, 0, config);
    if (discoveredInput.empty())
    {
        throw std::runtime_error("Ground-truth input discovery returned no files.");
    }
    logElapsed("input_discovery");

    std::vector<cv::Mat> realFrame = loadGroundTruthRawStack(discoveredInput.front());
    std::cout << "[GroundTruth RawInput] frame=" << imageFile.filename().string()
              << " mode=no_preprocess"
              << " slices=" << realFrame.size()
              << " min=" << stackPercentile(realFrame, 0.0f, false)
              << " max=" << stackMaxValue(realFrame)
              << " mean=" << stackMeanValue(realFrame)
              << std::endl;
    config.simulation.z_slices = static_cast<int>(realFrame.size());
    logElapsed("load_raw_frame_no_preprocess");

    if (config.cell)
    {
        Ellipsoid::cellConfig = *config.cell;
        Ellipsoid::cellConfig.maxZ = static_cast<float>(realFrame.size()) - 1.0f;
    }

    const std::string frameStem = imageFile.stem().string();
    std::vector<DetectedCell> cells = detectCellsInVolume(realFrame, frameStem);
    logElapsed("detect_cells");

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
    logElapsed("save_csv_and_preview_images");
    std::cout << "[GroundTruth Timing Summary] frame=" << imageFile.filename().string()
              << " detected_cells=" << cells.size()
              << " total_sec=" << std::fixed << std::setprecision(6)
              << std::chrono::duration<double>(Clock::now() - totalStart).count()
              << std::defaultfloat
              << std::endl;
    return cells;
}
