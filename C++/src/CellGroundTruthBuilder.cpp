#include "../includes/CellGroundTruthBuilder.hpp"
#include "../includes/ImageHandler.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
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

std::vector<float> groundTruthPercentiles()
{
    const char *overrideValue = std::getenv("CELLUNIVERSE_GT_PERCENTILES");
    if (overrideValue == nullptr || std::string(overrideValue).empty())
    {
        return {99.3f, 99.0f, 98.7f, 98.5f, 98.3f, 98.0f, 97.7f};
    }

    std::vector<float> percentiles;
    std::stringstream stream(overrideValue);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        try
        {
            const float percentile = std::stof(token);
            if (percentile > 0.0f && percentile < 100.0f)
            {
                percentiles.push_back(percentile);
            }
        }
        catch (const std::exception &)
        {
        }
    }
    if (percentiles.empty())
    {
        return {99.3f, 99.0f, 98.7f, 98.5f, 98.3f, 98.0f, 97.7f};
    }
    return percentiles;
}

bool shouldUseTraMaskWhenAvailable()
{
    const char *value = std::getenv("CELLUNIVERSE_GT_USE_TRA_MASK");
    return value != nullptr && std::string(value) == "1";
}

bool shouldSkipGroundTruthTiffOutput()
{
    const char *value = std::getenv("CELLUNIVERSE_GT_SKIP_TIFF");
    return value != nullptr && std::string(value) == "1";
}

bool shouldDisableGroundTruthFragmentMerge()
{
    const char *value = std::getenv("CELLUNIVERSE_GT_DISABLE_FRAGMENT_MERGE");
    return value != nullptr && std::string(value) == "1";
}

float groundTruthGeometricCenterBlend(const std::string &profileLabel)
{
    const char *value = std::getenv("CELLUNIVERSE_GT_GEOMETRIC_CENTER_BLEND");
    if (value != nullptr && std::string(value).size() > 0)
    {
        try
        {
            return std::clamp(std::stof(value), 0.0f, 1.0f);
        }
        catch (const std::exception &)
        {
        }
    }

    return profileLabel == "celegans_embryo" ? 0.35f : 0.15f;
}

int frameNumberFromPath(const fs::path &imageFile)
{
    std::string digits;
    for (char ch : imageFile.stem().string())
    {
        if (std::isdigit(static_cast<unsigned char>(ch)))
        {
            digits.push_back(ch);
        }
    }
    if (digits.empty())
    {
        return -1;
    }
    return std::stoi(digits);
}

std::string formatFrameNumber3(int frameNumber)
{
    std::ostringstream stream;
    stream << std::setfill('0') << std::setw(3) << frameNumber;
    return stream.str();
}

std::optional<fs::path> findPairedTraMask(const fs::path &imageFile)
{
    if (!shouldUseTraMaskWhenAvailable())
    {
        return std::nullopt;
    }

    const int frameNumber = frameNumberFromPath(imageFile);
    if (frameNumber < 0)
    {
        return std::nullopt;
    }

    const std::string frameText = formatFrameNumber3(frameNumber);
    const fs::path maskName = "man_track" + frameText + ".tif";
    if (imageFile.filename() == maskName)
    {
        return imageFile;
    }

    std::vector<fs::path> candidates;
    const fs::path sequenceDir = imageFile.parent_path();
    const fs::path datasetRoot = sequenceDir.parent_path();
    candidates.push_back(datasetRoot / "GroundTruth_Embryo" / "TRA" / maskName);
    candidates.push_back(datasetRoot / (sequenceDir.filename().string() + "_GT") / "TRA" / maskName);
    candidates.push_back(datasetRoot / "TRA" / maskName);

    for (const auto &candidate : candidates)
    {
        if (fs::exists(candidate))
        {
            return candidate;
        }
    }
    return std::nullopt;
}

int labelValueAt(const cv::Mat &slice, int y, int x)
{
    if (slice.channels() != 1)
    {
        throw std::runtime_error("TRA mask must be a single-channel label image.");
    }

    switch (slice.depth())
    {
        case CV_8U:
            return static_cast<int>(slice.at<unsigned char>(y, x));
        case CV_16U:
            return static_cast<int>(slice.at<unsigned short>(y, x));
        case CV_16S:
            return static_cast<int>(slice.at<short>(y, x));
        case CV_32S:
            return slice.at<int>(y, x);
        case CV_32F:
            return static_cast<int>(std::lround(slice.at<float>(y, x)));
        case CV_64F:
            return static_cast<int>(std::lround(slice.at<double>(y, x)));
        default:
            throw std::runtime_error("Unsupported TRA mask pixel depth.");
    }
}

std::vector<cv::Mat> loadTraMaskStack(const fs::path &maskFile)
{
    std::vector<cv::Mat> slices;
    cv::imreadmulti(maskFile.string(), slices, cv::IMREAD_UNCHANGED);
    if (slices.empty())
    {
        throw std::runtime_error("TRA mask has 0 slices: " + maskFile.string());
    }
    return slices;
}

std::vector<cv::Mat> interpolateStackForPreview(const std::vector<cv::Mat> &volume, int zScale)
{
    if (volume.empty())
    {
        return {};
    }
    if (volume.size() == 1 || zScale <= 1)
    {
        return volume;
    }

    const int outputSlices = zScale * (static_cast<int>(volume.size()) - 1) + 1;
    std::vector<cv::Mat> interpolated(static_cast<size_t>(outputSlices));
    for (int outputZ = 0; outputZ < outputSlices; ++outputZ)
    {
        const int sourceZ = outputZ / zScale;
        const int offset = outputZ % zScale;
        if (offset == 0)
        {
            interpolated[static_cast<size_t>(outputZ)] =
                volume[static_cast<size_t>(sourceZ)].clone();
            continue;
        }

        const float alpha = static_cast<float>(offset) / static_cast<float>(zScale);
        cv::Mat blended;
        cv::addWeighted(volume[static_cast<size_t>(sourceZ)],
                        1.0f - alpha,
                        volume[static_cast<size_t>(sourceZ + 1)],
                        alpha,
                        0.0,
                        blended);
        interpolated[static_cast<size_t>(outputZ)] = blended;
    }
    return interpolated;
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

int findRoot(std::vector<int> &parent, int index)
{
    while (parent[index] != index)
    {
        parent[index] = parent[parent[index]];
        index = parent[index];
    }
    return index;
}

void unionRoots(std::vector<int> &parent, int a, int b)
{
    a = findRoot(parent, a);
    b = findRoot(parent, b);
    if (a != b)
    {
        parent[b] = a;
    }
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
    const cv::Point3f weightedCenter = component.center();
    const cv::Point3f geometricCenter = component.voxelCenter();
    const float geometricBlend = groundTruthGeometricCenterBlend(activeProfile.label);
    cell.centerScaled = weightedCenter * (1.0f - geometricBlend) + geometricCenter * geometricBlend;
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

std::vector<CellGroundTruthBuilder::DetectedCell> CellGroundTruthBuilder::detectCellsFromTraMask(
    const fs::path &traMaskFile,
    const std::vector<cv::Mat> &rawVolume,
    const std::string &frameStem) const
{
    const std::vector<cv::Mat> labelSlices = loadTraMaskStack(traMaskFile);
    if (rawVolume.empty())
    {
        throw std::runtime_error("TRA mask mode requires the matching raw frame volume.");
    }

    const bool rawMatchesMask =
        rawVolume.size() == labelSlices.size() &&
        !rawVolume.front().empty() &&
        rawVolume.front().size() == labelSlices.front().size();
    if (!rawMatchesMask)
    {
        std::cout << "[GroundTruth TRA] warning=raw_mask_size_mismatch"
                  << " raw_slices=" << rawVolume.size()
                  << " mask_slices=" << labelSlices.size()
                  << std::endl;
    }

    std::map<int, EmbryoBrightTracker::Comp3DStat> labelStats;
    for (int z = 0; z < static_cast<int>(labelSlices.size()); ++z)
    {
        const cv::Mat &maskSlice = labelSlices[static_cast<size_t>(z)];
        for (int y = 0; y < maskSlice.rows; ++y)
        {
            for (int x = 0; x < maskSlice.cols; ++x)
            {
                const int label = labelValueAt(maskSlice, y, x);
                if (label <= 0)
                {
                    continue;
                }

                auto [it, inserted] = labelStats.try_emplace(label);
                auto &stat = it->second;
                if (inserted)
                {
                    stat.x0 = stat.x1 = x;
                    stat.y0 = stat.y1 = y;
                    stat.z0 = stat.z1 = z;
                }

                stat.vox++;
                stat.sumW += 1.0;
                stat.sx += static_cast<double>(x);
                stat.sy += static_cast<double>(y);
                stat.sz += static_cast<double>(z);
                stat.ux += static_cast<double>(x);
                stat.uy += static_cast<double>(y);
                stat.uz += static_cast<double>(z);
                stat.x0 = std::min(stat.x0, x);
                stat.x1 = std::max(stat.x1, x);
                stat.y0 = std::min(stat.y0, y);
                stat.y1 = std::max(stat.y1, y);
                stat.z0 = std::min(stat.z0, z);
                stat.z1 = std::max(stat.z1, z);

                if (rawMatchesMask)
                {
                    stat.sumI += rawVolume[static_cast<size_t>(z)].ptr<float>(y)[x];
                }
            }
        }
    }

    const float minMajor = config.cell ? static_cast<float>(config.cell->minARadius) : 6.0f;
    const float maxMajor = config.cell ? static_cast<float>(config.cell->maxARadius) : 35.0f;
    const float minMinor = config.cell ? static_cast<float>(config.cell->minCRadius) : 4.0f;
    const float maxMinor = config.cell ? static_cast<float>(config.cell->maxCRadius) : 25.0f;
    const float minB = (config.cell && config.cell->maxBRadius > 0.0f)
        ? static_cast<float>(config.cell->minBRadius)
        : minMajor;
    const float maxB = (config.cell && config.cell->maxBRadius > 0.0f)
        ? static_cast<float>(config.cell->maxBRadius)
        : maxMajor;
    const float zScale = effectiveZScaling();

    auto estimateRadiiFromRawNeighborhood = [&](const cv::Point3f &centerRaw,
                                                float &major,
                                                float &bRadius,
                                                float &minor) {
        major = std::max(minMajor, 12.0f);
        bRadius = std::max(minB, 10.0f);
        minor = std::max(minMinor, 8.0f);
        if (!rawMatchesMask)
        {
            return;
        }

        const float searchXY = std::clamp(std::max(minMajor * 4.8f, 22.0f), 18.0f, 34.0f);
        const int rx = static_cast<int>(std::ceil(searchXY));
        const int ry = rx;
        const int rz = std::max(2, static_cast<int>(std::ceil((searchXY / zScale) * 1.35f)));
        const int cz = static_cast<int>(std::lround(centerRaw.z));
        const int cy = static_cast<int>(std::lround(centerRaw.y));
        const int cx = static_cast<int>(std::lround(centerRaw.x));

        std::vector<float> neighborhoodValues;
        for (int z = std::max(0, cz - rz); z <= std::min(static_cast<int>(rawVolume.size()) - 1, cz + rz); ++z)
        {
            const cv::Mat &slice = rawVolume[static_cast<size_t>(z)];
            for (int y = std::max(0, cy - ry); y <= std::min(slice.rows - 1, cy + ry); ++y)
            {
                const float dy = static_cast<float>(y) - centerRaw.y;
                for (int x = std::max(0, cx - rx); x <= std::min(slice.cols - 1, cx + rx); ++x)
                {
                    const float dx = static_cast<float>(x) - centerRaw.x;
                    const float dzScaled = (static_cast<float>(z) - centerRaw.z) * zScale;
                    const float normalized = (dx * dx + dy * dy) / (searchXY * searchXY) +
                                             (dzScaled * dzScaled) / (searchXY * searchXY);
                    if (normalized > 1.0f)
                    {
                        continue;
                    }
                    const float value = slice.ptr<float>(y)[x];
                    if (std::isfinite(value) && value > 0.0f)
                    {
                        neighborhoodValues.push_back(value);
                    }
                }
            }
        }

        if (neighborhoodValues.size() < 16)
        {
            return;
        }

        const float threshold = percentileFromValues(neighborhoodValues, 0.72f);
        double sumW = 0.0;
        double sumDx2 = 0.0;
        double sumDy2 = 0.0;
        double sumDz2 = 0.0;
        for (int z = std::max(0, cz - rz); z <= std::min(static_cast<int>(rawVolume.size()) - 1, cz + rz); ++z)
        {
            const cv::Mat &slice = rawVolume[static_cast<size_t>(z)];
            for (int y = std::max(0, cy - ry); y <= std::min(slice.rows - 1, cy + ry); ++y)
            {
                const float dy = static_cast<float>(y) - centerRaw.y;
                for (int x = std::max(0, cx - rx); x <= std::min(slice.cols - 1, cx + rx); ++x)
                {
                    const float dx = static_cast<float>(x) - centerRaw.x;
                    const float dzScaled = (static_cast<float>(z) - centerRaw.z) * zScale;
                    const float normalized = (dx * dx + dy * dy) / (searchXY * searchXY) +
                                             (dzScaled * dzScaled) / (searchXY * searchXY);
                    if (normalized > 1.0f)
                    {
                        continue;
                    }

                    const float value = slice.ptr<float>(y)[x];
                    const float signal = value - threshold;
                    if (signal <= 0.0f)
                    {
                        continue;
                    }
                    const double weight = static_cast<double>(signal) * static_cast<double>(signal);
                    sumW += weight;
                    sumDx2 += weight * static_cast<double>(dx * dx);
                    sumDy2 += weight * static_cast<double>(dy * dy);
                    sumDz2 += weight * static_cast<double>(dzScaled * dzScaled);
                }
            }
        }

        if (sumW <= 1e-6)
        {
            return;
        }

        const float radiusScale = 2.55f;
        const float rxMoment = radiusScale * std::sqrt(static_cast<float>(sumDx2 / sumW));
        const float ryMoment = radiusScale * std::sqrt(static_cast<float>(sumDy2 / sumW));
        const float rzMoment = radiusScale * std::sqrt(static_cast<float>(sumDz2 / sumW));
        major = clampf(std::max(rxMoment, ryMoment), minMajor, maxMajor);
        bRadius = clampf(std::min(rxMoment, ryMoment), minB, std::min(maxB, major));
        minor = clampf(rzMoment, minMinor, std::min(maxMinor, major));
    };

    std::vector<DetectedCell> cells;
    cells.reserve(labelStats.size());
    for (const auto &[label, stat] : labelStats)
    {
        DetectedCell cell = makeDetectedCellFromComponent(stat);
        const cv::Point3f centerRaw = stat.center();
        cell.centerScaled = cv::Point3f(centerRaw.x, centerRaw.y, centerRaw.z * zScale);
        cell.zForCsv = centerRaw.z;
        estimateRadiiFromRawNeighborhood(centerRaw, cell.majorRadius, cell.bRadius, cell.minorRadius);
        std::ostringstream name;
        name << frameStem << "_label_" << label;
        cell.name = name.str();
        cells.push_back(cell);
    }

    std::cout << "[GroundTruth TRA] mask=" << traMaskFile
              << " labels=" << cells.size()
              << " raw_radius_estimate=" << (rawMatchesMask ? 1 : 0)
              << std::endl;

    return cells;
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
    int skippedSmallComponents = 0;
    int skippedNoSeedComponents = 0;
    int splitComponents = 0;
    int refinedSplitCells = 0;
    const bool sparseSeedField =
        activeProfile.label == "celegans_embryo" && strongHighComponents.size() < 80;

    for (const auto &component : lowComponents)
    {
        if (component.vox < minComponentVoxels)
        {
            skippedSmallComponents++;
            continue;
        }

        if (!componentContainsBrightSeed(component, strongHighComponents))
        {
            skippedNoSeedComponents++;
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
        const bool needsSeededSplit =
            !sparseSeedField && shouldSplitCoarseComponent(coarseCell, containedSeeds);

        if (needsSeededSplit)
        {
            splitComponents++;
            for (const auto &seed : containedSeeds)
            {
                std::optional<DetectedCell> refined = detectLocalSeededCell(volume, seed, thresholdLow);
                if (refined)
                {
                    cells.push_back(*refined);
                    refinedSplitCells++;
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

    const size_t beforeDedupCount = cells.size();
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

    const size_t beforePruneCount = deduped.size();
    deduped = pruneLikelySatelliteCells(deduped);
    const size_t beforeFragmentMergeCount = deduped.size();
    deduped = mergeLikelySameCellFragments(deduped, volume, thresholdLow, thresholdHigh);
    std::cout << "[GroundTruth CCStats] percentileHigh=" << percentileHigh
              << " low_components=" << lowComponents.size()
              << " high_components=" << highComponents.size()
              << " strong_high=" << strongHighComponents.size()
              << " min_component_voxels=" << minComponentVoxels
              << " skipped_small=" << skippedSmallComponents
              << " skipped_no_seed=" << skippedNoSeedComponents
              << " sparse_seed_field=" << (sparseSeedField ? 1 : 0)
              << " split_components=" << splitComponents
              << " refined_split_cells=" << refinedSplitCells
              << " before_dedup=" << beforeDedupCount
              << " after_dedup=" << beforePruneCount
              << " after_prune=" << beforeFragmentMergeCount
              << " after_fragment_merge=" << deduped.size()
              << std::endl;

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

std::vector<CellGroundTruthBuilder::DetectedCell> CellGroundTruthBuilder::mergeLikelySameCellFragments(
    const std::vector<DetectedCell> &cells,
    const std::vector<cv::Mat> &volume,
    float thresholdLow,
    float thresholdHigh) const
{
    if (cells.size() < 2 || volume.empty())
    {
        return cells;
    }

    std::vector<float> majorRadii;
    majorRadii.reserve(cells.size());
    for (const auto &cell : cells)
    {
        majorRadii.push_back(cell.majorRadius);
    }
    std::sort(majorRadii.begin(), majorRadii.end());
    const float medianMajor = majorRadii[majorRadii.size() / 2];
    const float zScale = effectiveZScaling();
    if (shouldDisableGroundTruthFragmentMerge())
    {
        std::cout << "[GroundTruth FragmentMerge]"
                  << " before=" << cells.size()
                  << " after=" << cells.size()
                  << " merged_pairs=0"
                  << " median_major=" << medianMajor
                  << " mode=disabled_by_env"
                  << std::endl;
        return cells;
    }

    if (activeProfile.label == "celegans_embryo" && cells.size() >= 245)
    {
        std::cout << "[GroundTruth FragmentMerge]"
                  << " before=" << cells.size()
                  << " after=" << cells.size()
                  << " merged_pairs=0"
                  << " median_major=" << medianMajor
                  << " mode=dense_skip"
                  << std::endl;
        return cells;
    }

    const bool sparseMode = activeProfile.label == "celegans_embryo" && cells.size() < 90;
    const bool moderateMode = activeProfile.label == "celegans_embryo" && cells.size() < 170;

    auto bridgeLooksContinuous = [&](const DetectedCell &lhs, const DetectedCell &rhs) {
        const cv::Point3f a(lhs.centerScaled.x, lhs.centerScaled.y, lhs.zForCsv);
        const cv::Point3f b(rhs.centerScaled.x, rhs.centerScaled.y, rhs.zForCsv);
        const float dx = b.x - a.x;
        const float dy = b.y - a.y;
        const float dzScaled = (b.z - a.z) * zScale;
        const float distance = std::sqrt(dx * dx + dy * dy + dzScaled * dzScaled);
        const int steps = std::clamp(static_cast<int>(std::ceil(distance / 1.5f)), 8, 48);

        std::vector<float> samples;
        samples.reserve(static_cast<size_t>(steps + 1));
        for (int s = 0; s <= steps; ++s)
        {
            const float t = static_cast<float>(s) / static_cast<float>(steps);
            const int x = static_cast<int>(std::lround(a.x + t * dx));
            const int y = static_cast<int>(std::lround(a.y + t * dy));
            const int z = static_cast<int>(std::lround(a.z + t * (b.z - a.z)));
            if (z < 0 || z >= static_cast<int>(volume.size()))
            {
                continue;
            }
            const cv::Mat &slice = volume[static_cast<size_t>(z)];
            if (y < 0 || y >= slice.rows || x < 0 || x >= slice.cols)
            {
                continue;
            }
            const float value = slice.ptr<float>(y)[x];
            if (std::isfinite(value))
            {
                samples.push_back(value);
            }
        }

        if (samples.size() < 5)
        {
            return false;
        }
        std::sort(samples.begin(), samples.end());
        const float q20 = samples[static_cast<size_t>(0.20f * static_cast<float>(samples.size() - 1))];
        const float q50 = samples[samples.size() / 2];
        const float localSignal = std::max(1.0f, std::min(lhs.meanIntensity, rhs.meanIntensity) - thresholdLow);
        const float bridgeSignal = q20 - thresholdLow;
        const bool highContinuousBridge = q20 >= thresholdHigh * 0.58f;
        const bool lowContinuousBridge = bridgeSignal >= localSignal * 0.30f && q50 >= thresholdLow * 1.12f;
        const bool sparseContinuousBridge =
            sparseMode && q20 >= thresholdLow * 0.52f && q50 >= thresholdLow * 0.82f;
        const bool moderateContinuousBridge =
            moderateMode && q20 >= thresholdLow * 0.70f && q50 >= thresholdLow * 0.95f;
        return highContinuousBridge || lowContinuousBridge ||
               sparseContinuousBridge || moderateContinuousBridge;
    };

    std::vector<int> parent(cells.size());
    for (size_t i = 0; i < parent.size(); ++i)
    {
        parent[i] = static_cast<int>(i);
    }

    int mergedPairs = 0;
    for (size_t i = 0; i < cells.size(); ++i)
    {
        for (size_t j = i + 1; j < cells.size(); ++j)
        {
            const float dzScaled = (cells[i].zForCsv - cells[j].zForCsv) * zScale;
            const float dx = cells[i].centerScaled.x - cells[j].centerScaled.x;
            const float dy = cells[i].centerScaled.y - cells[j].centerScaled.y;
            const float distance = std::sqrt(dx * dx + dy * dy + dzScaled * dzScaled);
            float mergeDistance = std::max({
                13.0f,
                medianMajor * 0.92f,
                0.62f * std::max(cells[i].majorRadius, cells[j].majorRadius)
            });
            if (sparseMode)
            {
                mergeDistance = std::max(mergeDistance, std::max(30.0f, medianMajor * 3.1f));
            }
            else if (moderateMode)
            {
                mergeDistance = std::max(mergeDistance, std::max(20.0f, medianMajor * 1.8f));
            }
            if (distance > mergeDistance)
            {
                continue;
            }

            if (bridgeLooksContinuous(cells[i], cells[j]))
            {
                unionRoots(parent, static_cast<int>(i), static_cast<int>(j));
                mergedPairs++;
            }
        }
    }

    std::map<int, std::vector<int>> groups;
    for (size_t i = 0; i < cells.size(); ++i)
    {
        groups[findRoot(parent, static_cast<int>(i))].push_back(static_cast<int>(i));
    }

    std::vector<DetectedCell> merged;
    merged.reserve(groups.size());
    for (const auto &[root, members] : groups)
    {
        if (members.size() == 1)
        {
            merged.push_back(cells[static_cast<size_t>(members.front())]);
            continue;
        }

        DetectedCell cell;
        double totalWeight = 0.0;
        double totalVoxelWeight = 0.0;
        int x0 = std::numeric_limits<int>::max();
        int y0 = std::numeric_limits<int>::max();
        int z0 = std::numeric_limits<int>::max();
        int x1 = std::numeric_limits<int>::min();
        int y1 = std::numeric_limits<int>::min();
        int z1 = std::numeric_limits<int>::min();
        for (int index : members)
        {
            const DetectedCell &part = cells[static_cast<size_t>(index)];
            const double weight = std::max(1.0, static_cast<double>(part.voxelCount) *
                                                  std::max(1.0f, part.meanIntensity));
            totalWeight += weight;
            totalVoxelWeight += static_cast<double>(part.voxelCount);
            cell.centerScaled.x += static_cast<float>(weight * part.centerScaled.x);
            cell.centerScaled.y += static_cast<float>(weight * part.centerScaled.y);
            cell.zForCsv += static_cast<float>(weight * part.zForCsv);
            cell.majorRadius = std::max(cell.majorRadius, part.majorRadius);
            cell.bRadius = std::max(cell.bRadius, part.bRadius);
            cell.minorRadius = std::max(cell.minorRadius, part.minorRadius);
            cell.meanIntensity += static_cast<float>(weight * part.meanIntensity);
            x0 = std::min(x0, part.component.x0);
            x1 = std::max(x1, part.component.x1);
            y0 = std::min(y0, part.component.y0);
            y1 = std::max(y1, part.component.y1);
            z0 = std::min(z0, part.component.z0);
            z1 = std::max(z1, part.component.z1);
        }

        cell.centerScaled.x = static_cast<float>(cell.centerScaled.x / totalWeight);
        cell.centerScaled.y = static_cast<float>(cell.centerScaled.y / totalWeight);
        cell.zForCsv = static_cast<float>(cell.zForCsv / totalWeight);
        cell.centerScaled.z = cell.zForCsv * zScale;
        cell.voxelCount = static_cast<int>(std::lround(totalVoxelWeight));
        cell.meanIntensity = static_cast<float>(cell.meanIntensity / totalWeight);
        cell.component.x0 = x0;
        cell.component.x1 = x1;
        cell.component.y0 = y0;
        cell.component.y1 = y1;
        cell.component.z0 = z0;
        cell.component.z1 = z1;
        cell.component.vox = cell.voxelCount;
        cell.component.sumW = totalVoxelWeight;
        cell.component.sx = static_cast<double>(cell.centerScaled.x) * totalVoxelWeight;
        cell.component.sy = static_cast<double>(cell.centerScaled.y) * totalVoxelWeight;
        cell.component.sz = static_cast<double>(cell.zForCsv) * totalVoxelWeight;
        cell.component.sumI = static_cast<double>(cell.meanIntensity) * totalVoxelWeight;
        merged.push_back(cell);
    }

    std::sort(merged.begin(), merged.end(), [](const DetectedCell &lhs, const DetectedCell &rhs) {
        if (std::abs(lhs.centerScaled.y - rhs.centerScaled.y) > 1e-3f)
        {
            return lhs.centerScaled.y < rhs.centerScaled.y;
        }
        return lhs.centerScaled.x < rhs.centerScaled.x;
    });

    std::cout << "[GroundTruth FragmentMerge]"
              << " before=" << cells.size()
              << " after=" << merged.size()
              << " merged_pairs=" << mergedPairs
              << " median_major=" << medianMajor
              << std::endl;

    return merged;
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

    return 0.010f * meanVoxelCount
           + 0.16f * static_cast<float>(cells.size())
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
    const std::vector<float> percentiles = groundTruthPercentiles();

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
    if (shouldSkipGroundTruthTiffOutput())
    {
        std::cout << "[GroundTruth Output] skip_tiff=1 frame=" << imageFile.filename().string()
                  << " reason=CELLUNIVERSE_GT_SKIP_TIFF"
                  << std::endl;
        return;
    }

    std::vector<DetectedCell> displayCells = cells;
    const float zScale = effectiveZScaling();
    for (auto &cell : displayCells)
    {
        cell.centerScaled.z = cell.zForCsv * zScale;
    }

    std::vector<cv::Mat> displayFrame = interpolateStackForPreview(
        normalizeStackForPreview(realFrame),
        static_cast<int>(std::lround(zScale)));
    std::vector<Ellipsoid> ellipsoids = makeEllipsoids(displayCells);
    SimulationConfig displayConfig = config.simulation;
    displayConfig.z_slices = static_cast<int>(displayFrame.size());

    Frame frame(displayFrame,
                displayConfig,
                ellipsoids,
                outputDir.string(),
                imageFile.filename().string());
    frame.setBackgroundColor(0.0f);
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
    std::vector<DetectedCell> cells;
    if (const std::optional<fs::path> traMask = findPairedTraMask(imageFile))
    {
        cells = detectCellsFromTraMask(*traMask, realFrame, frameStem);
    }
    else
    {
        cells = detectCellsInVolume(realFrame, frameStem);
    }
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
