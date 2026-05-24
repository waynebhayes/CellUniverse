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
#include <queue>
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

std::vector<float> groundTruthPercentiles(const GroundTruthConfig &config)
{
    const std::vector<float> envPercentiles = groundTruthPercentiles();
    if (!config.percentiles.empty())
    {
        std::vector<float> cleaned;
        cleaned.reserve(config.percentiles.size());
        for (float percentile : config.percentiles)
        {
            if (percentile > 0.0f && percentile < 100.0f)
            {
                cleaned.push_back(percentile);
            }
        }
        if (!cleaned.empty())
        {
            return cleaned;
        }
    }
    return envPercentiles;
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

float groundTruthGeometricCenterBlend(const BaseConfig &config,
                                      const std::string &profileLabel)
{
    if (config.groundTruth.geometricCenterBlend >= 0.0f)
    {
        return std::clamp(config.groundTruth.geometricCenterBlend, 0.0f, 1.0f);
    }
    return groundTruthGeometricCenterBlend(profileLabel);
}

float continuousEllipsoidVolume(float aRadius, float bRadius, float cRadius)
{
    return static_cast<float>((4.0 / 3.0) * M_PI *
                              std::max(aRadius, 1e-3f) *
                              std::max(bRadius, 1e-3f) *
                              std::max(cRadius, 1e-3f));
}

float meanOfTopFraction(std::vector<float> values, float fraction)
{
    if (values.empty())
    {
        return 0.0f;
    }
    const float clamped = std::clamp(fraction, 0.0f, 1.0f);
    if (clamped <= 0.0f)
    {
        return 0.0f;
    }
    const size_t keep = std::max<size_t>(
        1,
        static_cast<size_t>(std::ceil(clamped * static_cast<float>(values.size()))));
    const size_t index = values.size() - keep;
    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(index),
                     values.end());
    double sum = 0.0;
    for (size_t i = index; i < values.size(); ++i)
    {
        sum += values[i];
    }
    return static_cast<float>(sum / static_cast<double>(keep));
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
    }
    else if (imagePath.find("original_data") != std::string::npos)
    {
        profile.label = "original_data";
    }

    if (config.groundTruth.minHighSeedVoxels > 0)
    {
        profile.minHighSeedVoxels = config.groundTruth.minHighSeedVoxels;
    }
    if (config.groundTruth.seedMergeDistance > 0.0f)
    {
        profile.seedMergeDistance = config.groundTruth.seedMergeDistance;
    }
    if (config.groundTruth.seedSplitSeparation > 0.0f)
    {
        profile.seedSplitSeparation = config.groundTruth.seedSplitSeparation;
    }
    if (config.groundTruth.dedupDistance > 0.0f)
    {
        profile.dedupDistance = config.groundTruth.dedupDistance;
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
    if (config.groundTruth.minComponentVoxels > 0)
    {
        return config.groundTruth.minComponentVoxels;
    }

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
    const float geometricBlend = groundTruthGeometricCenterBlend(config, activeProfile.label);
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
    const float volumeScale = config.groundTruth.useScaledVoxelVolumeForRadius
        ? effectiveZScaling()
        : 1.0f;
    const float componentVolume = std::max(1.0f, static_cast<float>(component.vox) * volumeScale);
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

    const float radiusScale = std::max(0.1f, config.groundTruth.radiusInflationScale);
    const float outputMinMajor = config.groundTruth.minOutputMajorRadius > 0.0f
        ? config.groundTruth.minOutputMajorRadius
        : minMajor;
    const float outputMinB = config.groundTruth.minOutputBRadius > 0.0f
        ? config.groundTruth.minOutputBRadius
        : minB;
    const float outputMinMinor = config.groundTruth.minOutputMinorRadius > 0.0f
        ? config.groundTruth.minOutputMinorRadius
        : minMinor;
    const float outputMaxMajor = config.groundTruth.maxOutputMajorRadius > 0.0f
        ? config.groundTruth.maxOutputMajorRadius
        : maxMajor;
    const float outputMaxB = config.groundTruth.maxOutputBRadius > 0.0f
        ? config.groundTruth.maxOutputBRadius
        : maxB;
    const float outputMaxMinor = config.groundTruth.maxOutputMinorRadius > 0.0f
        ? config.groundTruth.maxOutputMinorRadius
        : maxMinor;

    majorRadius = radiusScale * std::max(equivalentRadius, blendTowardAxis(xyMaxRadius, 1.0f));
    bRadius = radiusScale * blendTowardAxis(xyMinRadius, xySupport);
    minorRadius = radiusScale * blendTowardAxis(zRadius, zSupport * zSupport);

    majorRadius = clampf(majorRadius, outputMinMajor, outputMaxMajor);
    bRadius = clampf(bRadius, outputMinB, std::min(outputMaxB, majorRadius));
    minorRadius = clampf(minorRadius, outputMinMinor, std::min(outputMaxMinor, majorRadius));
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

std::vector<EmbryoBrightTracker::Comp3DStat> CellGroundTruthBuilder::extractLocalMaximumSeeds(
    const std::vector<cv::Mat> &volume,
    float thresholdHigh,
    const std::vector<EmbryoBrightTracker::Comp3DStat> &componentSeeds) const
{
    if (!config.groundTruth.localMaxSeedEnabled || volume.empty())
    {
        return componentSeeds;
    }

    if (config.groundTruth.localMaxSeedMinStrongSeedCount > 0 &&
        static_cast<int>(componentSeeds.size()) < config.groundTruth.localMaxSeedMinStrongSeedCount)
    {
        return componentSeeds;
    }

    struct Peak
    {
        float value = 0.0f;
        int x = 0;
        int y = 0;
        int z = 0;
    };

    const int Z = static_cast<int>(volume.size());
    const int Y = volume.front().rows;
    const int X = volume.front().cols;
    std::vector<Peak> peaks;
    peaks.reserve(componentSeeds.size() * 2 + 64);

    for (int z = 1; z + 1 < Z; ++z)
    {
        for (int y = 1; y + 1 < Y; ++y)
        {
            const float *row = volume[static_cast<size_t>(z)].ptr<float>(y);
            for (int x = 1; x + 1 < X; ++x)
            {
                const float value = row[x];
                if (value < thresholdHigh)
                {
                    continue;
                }

                bool isLocalMax = true;
                for (int dz = -1; dz <= 1 && isLocalMax; ++dz)
                {
                    const cv::Mat &slice = volume[static_cast<size_t>(z + dz)];
                    for (int dy = -1; dy <= 1 && isLocalMax; ++dy)
                    {
                        const float *neighborRow = slice.ptr<float>(y + dy);
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            if (dx == 0 && dy == 0 && dz == 0)
                            {
                                continue;
                            }
                            if (neighborRow[x + dx] > value)
                            {
                                isLocalMax = false;
                                break;
                            }
                        }
                    }
                }

                if (isLocalMax)
                {
                    peaks.push_back(Peak{value, x, y, z});
                }
            }
        }
    }

    std::sort(peaks.begin(), peaks.end(), [](const Peak &lhs, const Peak &rhs) {
        return lhs.value > rhs.value;
    });

    const float minDistance = std::max(1.0f, config.groundTruth.localMaxSeedMinDistance);
    const float minDistanceSq = minDistance * minDistance;
    const float zScale = effectiveZScaling();
    const size_t maxSeeds = config.groundTruth.localMaxSeedMaxPerFrameFactor > 0.0f
        ? static_cast<size_t>(std::ceil(config.groundTruth.localMaxSeedMaxPerFrameFactor *
                                        static_cast<float>(std::max<size_t>(componentSeeds.size(), 1))))
        : std::numeric_limits<size_t>::max();

    std::vector<EmbryoBrightTracker::Comp3DStat> seeds;
    seeds.reserve(std::min(peaks.size(), maxSeeds));
    for (const Peak &peak : peaks)
    {
        bool duplicate = false;
        for (const auto &existing : seeds)
        {
            const cv::Point3f center = existing.center();
            const float dx = static_cast<float>(peak.x) - center.x;
            const float dy = static_cast<float>(peak.y) - center.y;
            const float dz = (static_cast<float>(peak.z) - center.z) * zScale;
            if (dx * dx + dy * dy + dz * dz <= minDistanceSq)
            {
                duplicate = true;
                break;
            }
        }
        if (duplicate)
        {
            continue;
        }

        EmbryoBrightTracker::Comp3DStat seed;
        seed.vox = 1;
        seed.sumW = std::max(1.0f, peak.value);
        seed.sx = static_cast<double>(peak.x) * seed.sumW;
        seed.sy = static_cast<double>(peak.y) * seed.sumW;
        seed.sz = static_cast<double>(peak.z) * seed.sumW;
        seed.ux = static_cast<double>(peak.x);
        seed.uy = static_cast<double>(peak.y);
        seed.uz = static_cast<double>(peak.z);
        seed.sumI = peak.value;
        seed.x0 = seed.x1 = peak.x;
        seed.y0 = seed.y1 = peak.y;
        seed.z0 = seed.z1 = peak.z;
        seeds.push_back(seed);

        if (seeds.size() >= maxSeeds)
        {
            break;
        }
    }

    std::cout << "[GroundTruth LocalMaxSeeds]"
              << " component_seeds=" << componentSeeds.size()
              << " local_peaks=" << peaks.size()
              << " kept=" << seeds.size()
              << " min_distance=" << minDistance
              << " threshold_high=" << thresholdHigh
              << std::endl;

    return seeds.empty() ? componentSeeds : seeds;
}

std::vector<cv::Mat> CellGroundTruthBuilder::buildLocalContrastVolume(
    const std::vector<cv::Mat> &volume) const
{
    std::vector<cv::Mat> contrast;
    contrast.reserve(volume.size());
    if (volume.empty())
    {
        return contrast;
    }

    const double signalSigma = std::max(0.0f, config.groundTruth.seededWatershedSignalSigma);
    const double backgroundSigma = std::max(
        signalSigma + 0.5,
        static_cast<double>(std::max(1.0f, config.groundTruth.seededWatershedBackgroundSigma)));

    for (const auto &slice : volume)
    {
        cv::Mat signal;
        if (signalSigma > 0.0)
        {
            cv::GaussianBlur(slice, signal, cv::Size(0, 0), signalSigma, signalSigma, cv::BORDER_REPLICATE);
        }
        else
        {
            signal = slice.clone();
        }

        cv::Mat background;
        cv::GaussianBlur(slice, background, cv::Size(0, 0), backgroundSigma, backgroundSigma, cv::BORDER_REPLICATE);

        cv::Mat local = signal - background;
        cv::max(local, 0.0f, local);
        contrast.push_back(local);
    }

    float normalizer = stackPercentile(contrast, 0.995f, true);
    if (normalizer <= 1e-6f)
    {
        normalizer = std::max(1.0f, stackMaxValue(contrast));
    }

    for (auto &slice : contrast)
    {
        slice.convertTo(slice, CV_32F, 1.0f / normalizer);
        cv::min(slice, 1.0f, slice);
        cv::max(slice, 0.0f, slice);
    }

    std::cout << "[GroundTruth SeededWatershed Contrast]"
              << " signal_sigma=" << signalSigma
              << " background_sigma=" << backgroundSigma
              << " normalizer=" << normalizer
              << std::endl;
    return contrast;
}

std::vector<CellGroundTruthBuilder::DetectedCell> CellGroundTruthBuilder::detectCellsBySeededWatershed(
    const std::vector<cv::Mat> &volume,
    const std::string &frameStem) const
{
    if (!config.groundTruth.seededWatershedEnabled || volume.empty())
    {
        return {};
    }

    const std::vector<cv::Mat> contrast = buildLocalContrastVolume(volume);
    if (contrast.empty())
    {
        return {};
    }

    const int Z = static_cast<int>(contrast.size());
    const int Y = contrast[0].rows;
    const int X = contrast[0].cols;
    const float zScale = effectiveZScaling();

    const float seedThreshold = std::max(
        config.groundTruth.seededWatershedSeedThresholdFloor,
        stackPercentile(contrast,
                        clampf(config.groundTruth.seededWatershedSeedPercentile / 100.0f, 0.0f, 1.0f),
                        true));
    float maskThreshold = std::max(
        config.groundTruth.seededWatershedMaskThresholdFloor,
        stackPercentile(contrast,
                        clampf(config.groundTruth.seededWatershedMaskPercentile / 100.0f, 0.0f, 1.0f),
                        true));
    maskThreshold = std::min(maskThreshold, seedThreshold * 0.95f);

    struct Peak
    {
        float value = 0.0f;
        int x = 0;
        int y = 0;
        int z = 0;
    };

    std::vector<Peak> peaks;
    peaks.reserve(512);
    for (int z = 1; z + 1 < Z; ++z)
    {
        for (int y = 1; y + 1 < Y; ++y)
        {
            const float *row = contrast[static_cast<size_t>(z)].ptr<float>(y);
            for (int x = 1; x + 1 < X; ++x)
            {
                const float value = row[x];
                if (value < seedThreshold)
                {
                    continue;
                }

                bool isLocalMaximum = true;
                for (int dz = -1; dz <= 1 && isLocalMaximum; ++dz)
                {
                    const cv::Mat &neighborSlice = contrast[static_cast<size_t>(z + dz)];
                    for (int dy = -1; dy <= 1 && isLocalMaximum; ++dy)
                    {
                        const float *neighborRow = neighborSlice.ptr<float>(y + dy);
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            if (dx == 0 && dy == 0 && dz == 0)
                            {
                                continue;
                            }
                            if (neighborRow[x + dx] > value)
                            {
                                isLocalMaximum = false;
                                break;
                            }
                        }
                    }
                }

                if (isLocalMaximum)
                {
                    peaks.push_back(Peak{value, x, y, z});
                }
            }
        }
    }

    std::sort(peaks.begin(), peaks.end(), [](const Peak &lhs, const Peak &rhs) {
        return lhs.value > rhs.value;
    });

    const float minSeedDistance = std::max(1.0f, config.groundTruth.seededWatershedMinSeedDistance);
    const float minSeedDistanceSq = minSeedDistance * minSeedDistance;
    const size_t maxSeeds = config.groundTruth.seededWatershedMaxSeeds > 0
        ? static_cast<size_t>(config.groundTruth.seededWatershedMaxSeeds)
        : std::numeric_limits<size_t>::max();
    std::vector<Peak> seeds;
    seeds.reserve(std::min(peaks.size(), maxSeeds));

    for (const Peak &peak : peaks)
    {
        bool tooClose = false;
        for (const Peak &seed : seeds)
        {
            const float dx = static_cast<float>(peak.x - seed.x);
            const float dy = static_cast<float>(peak.y - seed.y);
            const float dz = static_cast<float>(peak.z - seed.z) * zScale;
            if (dx * dx + dy * dy + dz * dz <= minSeedDistanceSq)
            {
                tooClose = true;
                break;
            }
        }
        if (!tooClose)
        {
            seeds.push_back(peak);
            if (seeds.size() >= maxSeeds)
            {
                break;
            }
        }
    }

    if (seeds.empty())
    {
        std::cout << "[GroundTruth SeededWatershed] no_seeds=1"
                  << " peaks=" << peaks.size()
                  << " seed_threshold=" << seedThreshold
                  << " mask_threshold=" << maskThreshold
                  << std::endl;
        return {};
    }

    const auto lin = [Y, X](int z, int y, int x) -> size_t {
        return (static_cast<size_t>(z) * static_cast<size_t>(Y) + static_cast<size_t>(y)) *
               static_cast<size_t>(X) + static_cast<size_t>(x);
    };

    struct GrowNode
    {
        float priority = 0.0f;
        int seedIndex = 0;
        int x = 0;
        int y = 0;
        int z = 0;
    };
    struct GrowNodeCompare
    {
        bool operator()(const GrowNode &lhs, const GrowNode &rhs) const
        {
            return lhs.priority < rhs.priority;
        }
    };

    std::priority_queue<GrowNode, std::vector<GrowNode>, GrowNodeCompare> frontier;
    std::vector<int> labels(static_cast<size_t>(Z) * static_cast<size_t>(Y) * static_cast<size_t>(X), 0);
    std::vector<EmbryoBrightTracker::Comp3DStat> stats(seeds.size());
    for (auto &stat : stats)
    {
        stat.x0 = X;
        stat.y0 = Y;
        stat.z0 = Z;
        stat.x1 = 0;
        stat.y1 = 0;
        stat.z1 = 0;
    }

    for (int i = 0; i < static_cast<int>(seeds.size()); ++i)
    {
        const Peak &seed = seeds[static_cast<size_t>(i)];
        frontier.push(GrowNode{seed.value, i, seed.x, seed.y, seed.z});
    }

    const float maxGrowDistance = std::max(1.0f, config.groundTruth.seededWatershedMaxGrowDistance);
    const float maxGrowDistanceSq = maxGrowDistance * maxGrowDistance;
    const std::array<cv::Point3i, 26> neighbors = {
        cv::Point3i{-1, -1, -1}, cv::Point3i{0, -1, -1}, cv::Point3i{1, -1, -1},
        cv::Point3i{-1, 0, -1}, cv::Point3i{0, 0, -1}, cv::Point3i{1, 0, -1},
        cv::Point3i{-1, 1, -1}, cv::Point3i{0, 1, -1}, cv::Point3i{1, 1, -1},
        cv::Point3i{-1, -1, 0}, cv::Point3i{0, -1, 0}, cv::Point3i{1, -1, 0},
        cv::Point3i{-1, 0, 0}, cv::Point3i{1, 0, 0},
        cv::Point3i{-1, 1, 0}, cv::Point3i{0, 1, 0}, cv::Point3i{1, 1, 0},
        cv::Point3i{-1, -1, 1}, cv::Point3i{0, -1, 1}, cv::Point3i{1, -1, 1},
        cv::Point3i{-1, 0, 1}, cv::Point3i{0, 0, 1}, cv::Point3i{1, 0, 1},
        cv::Point3i{-1, 1, 1}, cv::Point3i{0, 1, 1}, cv::Point3i{1, 1, 1}
    };

    size_t assignedVoxels = 0;
    while (!frontier.empty())
    {
        GrowNode node = frontier.top();
        frontier.pop();

        if (node.x < 0 || node.x >= X || node.y < 0 || node.y >= Y || node.z < 0 || node.z >= Z)
        {
            continue;
        }

        const size_t index = lin(node.z, node.y, node.x);
        if (labels[index] != 0)
        {
            continue;
        }

        const float value = contrast[static_cast<size_t>(node.z)].ptr<float>(node.y)[node.x];
        if (value < maskThreshold)
        {
            continue;
        }

        const Peak &seed = seeds[static_cast<size_t>(node.seedIndex)];
        const float dxSeed = static_cast<float>(node.x - seed.x);
        const float dySeed = static_cast<float>(node.y - seed.y);
        const float dzSeed = static_cast<float>(node.z - seed.z) * zScale;
        const float distSeedSq = dxSeed * dxSeed + dySeed * dySeed + dzSeed * dzSeed;
        if (distSeedSq > maxGrowDistanceSq)
        {
            continue;
        }

        labels[index] = node.seedIndex + 1;
        ++assignedVoxels;

        EmbryoBrightTracker::Comp3DStat &stat = stats[static_cast<size_t>(node.seedIndex)];
        const float rawValue = volume[static_cast<size_t>(node.z)].ptr<float>(node.y)[node.x];
        const double weight = static_cast<double>(std::max(1e-4f, value - maskThreshold));
        stat.vox++;
        stat.sumW += weight;
        stat.sx += weight * static_cast<double>(node.x);
        stat.sy += weight * static_cast<double>(node.y);
        stat.sz += weight * static_cast<double>(node.z);
        stat.ux += node.x;
        stat.uy += node.y;
        stat.uz += node.z;
        stat.sumI += rawValue;
        stat.x0 = std::min(stat.x0, node.x);
        stat.x1 = std::max(stat.x1, node.x);
        stat.y0 = std::min(stat.y0, node.y);
        stat.y1 = std::max(stat.y1, node.y);
        stat.z0 = std::min(stat.z0, node.z);
        stat.z1 = std::max(stat.z1, node.z);

        for (const auto &delta : neighbors)
        {
            const int nx = node.x + delta.x;
            const int ny = node.y + delta.y;
            const int nz = node.z + delta.z;
            if (nx < 0 || nx >= X || ny < 0 || ny >= Y || nz < 0 || nz >= Z)
            {
                continue;
            }
            const size_t neighborIndex = lin(nz, ny, nx);
            if (labels[neighborIndex] != 0)
            {
                continue;
            }
            const float neighborValue = contrast[static_cast<size_t>(nz)].ptr<float>(ny)[nx];
            if (neighborValue < maskThreshold)
            {
                continue;
            }

            const float ndx = static_cast<float>(nx - seed.x);
            const float ndy = static_cast<float>(ny - seed.y);
            const float ndz = static_cast<float>(nz - seed.z) * zScale;
            const float ndistSq = ndx * ndx + ndy * ndy + ndz * ndz;
            if (ndistSq > maxGrowDistanceSq)
            {
                continue;
            }

            const float distancePenalty = 0.01f * std::sqrt(ndistSq) / maxGrowDistance;
            frontier.push(GrowNode{neighborValue - distancePenalty, node.seedIndex, nx, ny, nz});
        }
    }

    const int minVoxels = config.groundTruth.seededWatershedMinVoxels > 0
        ? config.groundTruth.seededWatershedMinVoxels
        : computeMinComponentVoxels();
    std::vector<DetectedCell> cells;
    cells.reserve(stats.size());
    int skippedSmall = 0;
    for (const auto &stat : stats)
    {
        if (stat.vox < minVoxels || stat.sumW <= 1e-9)
        {
            skippedSmall++;
            continue;
        }
        cells.push_back(makeDetectedCellFromComponent(stat));
    }

    cells = applyBiologicalPriors(cells, volume, "seeded_watershed");

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

    std::cout << "[GroundTruth SeededWatershed]"
              << " peaks=" << peaks.size()
              << " seeds=" << seeds.size()
              << " seed_threshold=" << seedThreshold
              << " mask_threshold=" << maskThreshold
              << " min_seed_distance=" << minSeedDistance
              << " max_grow_distance=" << maxGrowDistance
              << " assigned_voxels=" << assignedVoxels
              << " min_voxels=" << minVoxels
              << " skipped_small=" << skippedSmall
              << " cells=" << cells.size()
              << std::endl;

    return cells;
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
    const float maxAllowedMajor = config.groundTruth.seededSplitMaxMajorRadius > 0.0f
        ? config.groundTruth.seededSplitMaxMajorRadius
        : (config.cell ? static_cast<float>(config.cell->maxARadius) * 1.15f : 46.0f);
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
    const float minSeedBalance = config.groundTruth.seededSplitMinSeedBalance;
    if (balance >= minSeedBalance && maxSeedSeparationXY >= softSplitThreshold)
    {
        return true;
    }

    const float configuredHardSplit = config.groundTruth.seededSplitMinSeedSeparation > 0.0f
        ? config.groundTruth.seededSplitMinSeedSeparation
        : activeProfile.seedSplitSeparation;
    const float hardSplitThreshold = std::max(configuredHardSplit, coarseCell.majorRadius * 0.75f);
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
    strongHighComponents = extractLocalMaximumSeeds(volume, thresholdHigh, strongHighComponents);

    const int minComponentVoxels = computeMinComponentVoxels();
    std::vector<DetectedCell> cells;
    cells.reserve(lowComponents.size());
    int skippedSmallComponents = 0;
    int skippedNoSeedComponents = 0;
    int splitComponents = 0;
    int refinedSplitCells = 0;
    const bool sparseSeedField =
        activeProfile.label == "celegans_embryo" && strongHighComponents.size() < 80;
    float effectiveSeedMergeDistance = activeProfile.seedMergeDistance;
    if (config.groundTruth.adaptiveSeedMergeEnabled &&
        activeProfile.label == "celegans_embryo")
    {
        const int strongSeedCount = static_cast<int>(strongHighComponents.size());
        if (config.groundTruth.adaptiveSeedMergeDenseHighSeedCount > 0 &&
            strongSeedCount >= config.groundTruth.adaptiveSeedMergeDenseHighSeedCount)
        {
            if (config.groundTruth.adaptiveSeedMergeDenseDistance > 0.0f)
            {
                effectiveSeedMergeDistance = config.groundTruth.adaptiveSeedMergeDenseDistance;
            }
        }
        else if (config.groundTruth.adaptiveSeedMergeModerateHighSeedCount > 0 &&
                 strongSeedCount >= config.groundTruth.adaptiveSeedMergeModerateHighSeedCount)
        {
            if (config.groundTruth.adaptiveSeedMergeModerateDistance > 0.0f)
            {
                effectiveSeedMergeDistance = config.groundTruth.adaptiveSeedMergeModerateDistance;
            }
        }
    }

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
        containedSeeds = collapseNearbySeeds(containedSeeds, effectiveSeedMergeDistance);

        DetectedCell coarseCell = makeDetectedCellFromComponent(component);
        const bool needsSeededSplit =
            (!sparseSeedField || config.groundTruth.allowSeededSplitInSparseField) &&
            shouldSplitCoarseComponent(coarseCell, containedSeeds);

        if (needsSeededSplit)
        {
            splitComponents++;
            for (const auto &seed : containedSeeds)
            {
                if (config.groundTruth.useHighSeedCenterForSplitCells)
                {
                    DetectedCell seedCell = makeDetectedCellFromComponent(seed);
                    cells.push_back(seedCell);
                    refinedSplitCells++;
                }
                else
                {
                    std::optional<DetectedCell> refined = detectLocalSeededCell(volume, seed, thresholdLow);
                    if (refined)
                    {
                        cells.push_back(*refined);
                        refinedSplitCells++;
                    }
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

    float effectiveDedupDistance = activeProfile.dedupDistance;
    float effectiveDedupRadiusScale = config.groundTruth.dedupRadiusScale;
    if (config.groundTruth.adaptiveDedupEnabled &&
        activeProfile.label == "celegans_embryo")
    {
        const int strongSeedCount = static_cast<int>(strongHighComponents.size());
        if (config.groundTruth.adaptiveDedupDenseHighSeedCount > 0 &&
            strongSeedCount >= config.groundTruth.adaptiveDedupDenseHighSeedCount)
        {
            if (config.groundTruth.adaptiveDedupDenseDistance > 0.0f)
            {
                effectiveDedupDistance = config.groundTruth.adaptiveDedupDenseDistance;
            }
            if (config.groundTruth.adaptiveDedupDenseRadiusScale > 0.0f)
            {
                effectiveDedupRadiusScale = config.groundTruth.adaptiveDedupDenseRadiusScale;
            }
        }
        else if (config.groundTruth.adaptiveDedupModerateHighSeedCount > 0 &&
                 strongSeedCount >= config.groundTruth.adaptiveDedupModerateHighSeedCount)
        {
            if (config.groundTruth.adaptiveDedupModerateDistance > 0.0f)
            {
                effectiveDedupDistance = config.groundTruth.adaptiveDedupModerateDistance;
            }
            if (config.groundTruth.adaptiveDedupModerateRadiusScale > 0.0f)
            {
                effectiveDedupRadiusScale = config.groundTruth.adaptiveDedupModerateRadiusScale;
            }
        }
    }

    const size_t beforeDedupCount = cells.size();
    std::vector<DetectedCell> deduped;
    for (const auto &cell : cells)
    {
        bool merged = false;
        for (auto &existing : deduped)
        {
            const float mergeRadius = std::max(effectiveDedupDistance,
                                               effectiveDedupRadiusScale * std::min(existing.majorRadius, cell.majorRadius));
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
    const size_t beforeBiologicalPriorsCount = deduped.size();
    deduped = applyBiologicalPriors(deduped, volume, "post_fragment_merge");
    std::cout << "[GroundTruth CCStats] percentileHigh=" << percentileHigh
              << " low_components=" << lowComponents.size()
              << " high_components=" << highComponents.size()
              << " strong_high=" << strongHighComponents.size()
              << " min_component_voxels=" << minComponentVoxels
              << " seed_merge_distance=" << effectiveSeedMergeDistance
              << " dedup_distance=" << effectiveDedupDistance
              << " dedup_radius_scale=" << effectiveDedupRadiusScale
              << " skipped_small=" << skippedSmallComponents
              << " skipped_no_seed=" << skippedNoSeedComponents
              << " sparse_seed_field=" << (sparseSeedField ? 1 : 0)
              << " split_components=" << splitComponents
              << " refined_split_cells=" << refinedSplitCells
              << " before_dedup=" << beforeDedupCount
              << " after_dedup=" << beforePruneCount
              << " after_prune=" << beforeFragmentMergeCount
              << " after_fragment_merge=" << beforeBiologicalPriorsCount
              << " before_bio_priors=" << beforeBiologicalPriorsCount
              << " after_bio_priors=" << deduped.size()
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

void CellGroundTruthBuilder::annotateSignalStats(const std::vector<cv::Mat> &volume,
                                                 DetectedCell &cell) const
{
    cell.shellVoxelCount = 0;
    cell.top10Intensity = cell.meanIntensity;
    cell.shellMeanIntensity = 0.0f;
    cell.meanMinusShell = 0.0f;
    cell.top10MinusShell = 0.0f;
    if (!config.groundTruth.signalStatsEnabled || volume.empty())
    {
        return;
    }

    const float zScale = effectiveZScaling();
    const float innerScale = std::max(1.0f, config.groundTruth.shellInnerScale);
    const float outerScale = std::max(innerScale + 0.01f, config.groundTruth.shellOuterScale);
    const float a = std::max(cell.majorRadius, 1e-3f);
    const float b = std::max(cell.bRadius, 1e-3f);
    const float c = std::max(cell.minorRadius, 1e-3f);
    const float maxR = std::max({a, b, c}) * outerScale;

    const int Z = static_cast<int>(volume.size());
    const int Y = volume[0].rows;
    const int X = volume[0].cols;
    const int minX = std::max(0, static_cast<int>(std::floor(cell.centerScaled.x - maxR)));
    const int maxX = std::min(X - 1, static_cast<int>(std::ceil(cell.centerScaled.x + maxR)));
    const int minY = std::max(0, static_cast<int>(std::floor(cell.centerScaled.y - maxR)));
    const int maxY = std::min(Y - 1, static_cast<int>(std::ceil(cell.centerScaled.y + maxR)));
    const int minZ = std::max(0, static_cast<int>(std::floor((cell.centerScaled.z - maxR) / zScale)));
    const int maxZ = std::min(Z - 1, static_cast<int>(std::ceil((cell.centerScaled.z + maxR) / zScale)));
    if (minX > maxX || minY > maxY || minZ > maxZ)
    {
        return;
    }

    std::vector<float> cellValues;
    std::vector<float> shellValues;
    const float invA2 = 1.0f / (a * a);
    const float invB2 = 1.0f / (b * b);
    const float invC2 = 1.0f / (c * c);
    const float inner2 = innerScale * innerScale;
    const float outer2 = outerScale * outerScale;
    for (int z = minZ; z <= maxZ; ++z)
    {
        const float dz = static_cast<float>(z) * zScale - cell.centerScaled.z;
        const cv::Mat &slice = volume[static_cast<size_t>(z)];
        for (int y = minY; y <= maxY; ++y)
        {
            const float dy = static_cast<float>(y) - cell.centerScaled.y;
            const float *row = slice.ptr<float>(y);
            for (int x = minX; x <= maxX; ++x)
            {
                const float dx = static_cast<float>(x) - cell.centerScaled.x;
                const float value = (dx * dx) * invA2 +
                                    (dy * dy) * invB2 +
                                    (dz * dz) * invC2;
                const float pixel = row[x];
                if (!std::isfinite(pixel))
                {
                    continue;
                }
                if (value <= 1.0f)
                {
                    cellValues.push_back(pixel);
                }
                else if (value > inner2 && value <= outer2)
                {
                    shellValues.push_back(pixel);
                }
            }
        }
    }

    if (!cellValues.empty())
    {
        cell.top10Intensity = meanOfTopFraction(cellValues, 0.10f);
        double sum = 0.0;
        for (float value : cellValues)
        {
            sum += value;
        }
        cell.meanIntensity = static_cast<float>(sum / static_cast<double>(cellValues.size()));
    }
    if (!shellValues.empty())
    {
        cell.shellVoxelCount = static_cast<int>(shellValues.size());
        double sum = 0.0;
        for (float value : shellValues)
        {
            sum += value;
        }
        cell.shellMeanIntensity = static_cast<float>(sum / static_cast<double>(shellValues.size()));
    }
    cell.meanMinusShell = cell.meanIntensity - cell.shellMeanIntensity;
    cell.top10MinusShell = cell.top10Intensity - cell.shellMeanIntensity;
}

std::vector<CellGroundTruthBuilder::DetectedCell> CellGroundTruthBuilder::applyBiologicalPriors(
    const std::vector<DetectedCell> &cells,
    const std::vector<cv::Mat> &volume,
    const std::string &stage) const
{
    if (cells.empty())
    {
        return {};
    }

    std::vector<DetectedCell> filtered;
    filtered.reserve(cells.size());
    int rejectedTiny = 0;
    int rejectedContrast = 0;
    for (DetectedCell cell : cells)
    {
        annotateSignalStats(volume, cell);
        const float componentDx = static_cast<float>(cell.component.x1 - cell.component.x0 + 1);
        const float componentDy = static_cast<float>(cell.component.y1 - cell.component.y0 + 1);
        const float componentDz = static_cast<float>(cell.component.z1 - cell.component.z0 + 1) * effectiveZScaling();
        const float componentSpanRadius = 0.5f * std::max({componentDx, componentDy, componentDz});

        bool reject = false;
        if (config.groundTruth.rejectTinyBrightArtifacts)
        {
            const bool biologicallyTinyByVoxel =
                config.groundTruth.minBiologicalVoxelCount > 0 &&
                cell.voxelCount < config.groundTruth.minBiologicalVoxelCount;
            const bool biologicallyTinyBySpan =
                config.groundTruth.minBiologicalMajorRadius > 0.0f &&
                componentSpanRadius < config.groundTruth.minBiologicalMajorRadius * 0.70f;
            const bool allowSparseArtifactRule =
                config.groundTruth.smallArtifactSparseMaxCells <= 0 ||
                static_cast<int>(cells.size()) <= config.groundTruth.smallArtifactSparseMaxCells;
            const bool tinyArtifactByConfig =
                allowSparseArtifactRule &&
                config.groundTruth.smallArtifactMaxVoxels > 0 &&
                cell.voxelCount <= config.groundTruth.smallArtifactMaxVoxels &&
                (config.groundTruth.smallArtifactMaxTop10Intensity < 0.0f ||
                 cell.top10Intensity <= config.groundTruth.smallArtifactMaxTop10Intensity) &&
                (config.groundTruth.smallArtifactMaxMeanIntensity < 0.0f ||
                 cell.meanIntensity <= config.groundTruth.smallArtifactMaxMeanIntensity);
            if ((biologicallyTinyByVoxel && biologicallyTinyBySpan) || tinyArtifactByConfig)
            {
                reject = true;
                rejectedTiny++;
            }
        }

        if (!reject && config.groundTruth.signalStatsEnabled)
        {
            if (config.groundTruth.minTop10MinusShell >= 0.0f &&
                cell.top10MinusShell < config.groundTruth.minTop10MinusShell)
            {
                reject = true;
                rejectedContrast++;
            }
            if (!reject &&
                config.groundTruth.minMeanMinusShell > -999999.0f &&
                cell.meanMinusShell < config.groundTruth.minMeanMinusShell)
            {
                reject = true;
                rejectedContrast++;
            }
        }

        if (!reject)
        {
            filtered.push_back(cell);
        }
    }

    int mergedPairs = 0;
    float effectiveMinCenterDistance = config.groundTruth.minBiologicalCenterDistance;
    if (config.groundTruth.adaptiveNearDuplicateMergeEnabled &&
        config.groundTruth.adaptiveNearDuplicateDistance > 0.0f)
    {
        const int cellCount = static_cast<int>(filtered.size());
        const bool aboveMin = config.groundTruth.adaptiveNearDuplicateMinCells <= 0 ||
                              cellCount >= config.groundTruth.adaptiveNearDuplicateMinCells;
        const bool belowMax = config.groundTruth.adaptiveNearDuplicateMaxCells <= 0 ||
                              cellCount <= config.groundTruth.adaptiveNearDuplicateMaxCells;
        bool pairRatioOk = true;
        if (config.groundTruth.adaptiveNearDuplicateMinPairRatio > 0.0f)
        {
            int nearPairs = 0;
            for (size_t i = 0; i < filtered.size(); ++i)
            {
                for (size_t j = i + 1; j < filtered.size(); ++j)
                {
                    const float nearThreshold = std::max(
                        4.0f,
                        0.55f * std::min(filtered[i].majorRadius, filtered[j].majorRadius));
                    if (squaredDistanceXY(filtered[i].centerScaled, filtered[j].centerScaled) <=
                            nearThreshold * nearThreshold &&
                        absoluteDistanceZ(filtered[i].centerScaled, filtered[j].centerScaled) <=
                            std::max(10.0f, nearThreshold * 2.0f))
                    {
                        nearPairs++;
                    }
                }
            }
            const float pairRatio = static_cast<float>(nearPairs) /
                                    static_cast<float>(std::max<int>(1, cellCount));
            pairRatioOk = pairRatio >= config.groundTruth.adaptiveNearDuplicateMinPairRatio;
        }
        if (aboveMin && belowMax && pairRatioOk)
        {
            effectiveMinCenterDistance = std::max(
                effectiveMinCenterDistance,
                config.groundTruth.adaptiveNearDuplicateDistance);
        }
    }
    if (config.groundTruth.biologicalNearDuplicateMergeEnabled &&
        effectiveMinCenterDistance > 0.0f &&
        filtered.size() >= 2)
    {
        std::vector<DetectedCell> merged;
        merged.reserve(filtered.size());
        for (DetectedCell cell : filtered)
        {
            bool absorbed = false;
            for (auto &existing : merged)
            {
                const float dz = cell.centerScaled.z - existing.centerScaled.z;
                const float dx = cell.centerScaled.x - existing.centerScaled.x;
                const float dy = cell.centerScaled.y - existing.centerScaled.y;
                const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (distance > effectiveMinCenterDistance)
                {
                    continue;
                }

                const float leftWeight = std::max(1.0f, static_cast<float>(existing.voxelCount));
                const float rightWeight = std::max(1.0f, static_cast<float>(cell.voxelCount));
                const float totalWeight = leftWeight + rightWeight;
                existing.centerScaled.x = (existing.centerScaled.x * leftWeight + cell.centerScaled.x * rightWeight) / totalWeight;
                existing.centerScaled.y = (existing.centerScaled.y * leftWeight + cell.centerScaled.y * rightWeight) / totalWeight;
                existing.centerScaled.z = (existing.centerScaled.z * leftWeight + cell.centerScaled.z * rightWeight) / totalWeight;
                existing.zForCsv = existing.centerScaled.z / effectiveZScaling();
                existing.majorRadius = std::max(existing.majorRadius, cell.majorRadius);
                existing.bRadius = std::max(existing.bRadius, cell.bRadius);
                existing.minorRadius = std::max(existing.minorRadius, cell.minorRadius);
                existing.voxelCount += cell.voxelCount;
                existing.meanIntensity = (existing.meanIntensity * leftWeight + cell.meanIntensity * rightWeight) / totalWeight;
                existing.component.x0 = std::min(existing.component.x0, cell.component.x0);
                existing.component.x1 = std::max(existing.component.x1, cell.component.x1);
                existing.component.y0 = std::min(existing.component.y0, cell.component.y0);
                existing.component.y1 = std::max(existing.component.y1, cell.component.y1);
                existing.component.z0 = std::min(existing.component.z0, cell.component.z0);
                existing.component.z1 = std::max(existing.component.z1, cell.component.z1);
                annotateSignalStats(volume, existing);
                absorbed = true;
                mergedPairs++;
                break;
            }
            if (!absorbed)
            {
                merged.push_back(cell);
            }
        }
        filtered = std::move(merged);
    }

    std::cout << "[GroundTruth BioPriors]"
              << " stage=" << stage
              << " before=" << cells.size()
              << " after=" << filtered.size()
              << " rejected_tiny=" << rejectedTiny
              << " rejected_contrast=" << rejectedContrast
              << " merged_near_duplicates=" << mergedPairs
              << " min_component_voxels=" << config.groundTruth.minComponentVoxels
              << " min_top10_minus_shell=" << config.groundTruth.minTop10MinusShell
              << " min_center_distance=" << effectiveMinCenterDistance
              << std::endl;
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
    const bool disabledByCellCount =
        config.groundTruth.fragmentMergeMaxInputCells > 0 &&
        static_cast<int>(cells.size()) > config.groundTruth.fragmentMergeMaxInputCells;
    if (shouldDisableGroundTruthFragmentMerge() ||
        !config.groundTruth.fragmentMergeEnabled ||
        disabledByCellCount)
    {
        std::cout << "[GroundTruth FragmentMerge]"
                  << " before=" << cells.size()
                  << " after=" << cells.size()
                  << " merged_pairs=0"
                  << " median_major=" << medianMajor
                  << " mode="
                  << (shouldDisableGroundTruthFragmentMerge()
                          ? "disabled_by_env"
                          : (!config.groundTruth.fragmentMergeEnabled ? "disabled_by_config" : "disabled_by_cell_count"))
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
    std::vector<float> top10MinusShellValues;
    std::vector<float> meanMinusShellValues;
    std::vector<int> voxelCounts;
    majorRadii.reserve(cells.size());
    top10MinusShellValues.reserve(cells.size());
    meanMinusShellValues.reserve(cells.size());
    voxelCounts.reserve(cells.size());
    for (const auto &cell : cells)
    {
        totalVoxels += cell.voxelCount;
        majorRadii.push_back(cell.majorRadius);
        top10MinusShellValues.push_back(cell.top10MinusShell);
        meanMinusShellValues.push_back(cell.meanMinusShell);
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
    std::sort(top10MinusShellValues.begin(), top10MinusShellValues.end());
    std::sort(meanMinusShellValues.begin(), meanMinusShellValues.end());
    const float medianTop10MinusShell = top10MinusShellValues[top10MinusShellValues.size() / 2];
    const float medianMeanMinusShell = meanMinusShellValues[meanMinusShellValues.size() / 2];
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
    const float meanVoxelPenalty =
        config.groundTruth.maxCandidateMeanVoxelCount > 0.0f && meanVoxelCount > config.groundTruth.maxCandidateMeanVoxelCount
            ? (meanVoxelCount - config.groundTruth.maxCandidateMeanVoxelCount) /
                  std::max(1.0f, config.groundTruth.maxCandidateMeanVoxelCount)
            : 0.0f;
    const float cellCountWeight = config.groundTruth.candidateCellCountWeight > 0.0f
                                      ? config.groundTruth.candidateCellCountWeight
                                      : 0.16f;
    const float meanVoxelPenaltyWeight = config.groundTruth.candidateMeanVoxelPenaltyWeight > 0.0f
                                             ? config.groundTruth.candidateMeanVoxelPenaltyWeight
                                             : 1.0f;

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

    return cellCountWeight * static_cast<float>(cells.size())
           + 0.20f * medianMajor
           + 0.020f * medianTop10MinusShell
           + 0.010f * medianMeanMinusShell
           - 2.50f * static_cast<float>(clampedMinorCount)
           - 1.25f * static_cast<float>(verySmallCount)
           - 3.00f * static_cast<float>(veryLargeCount)
           - config.groundTruth.candidateNearPairPenaltyWeight * static_cast<float>(nearDuplicatePairs)
           - 1.75f * static_cast<float>(flattenedCount)
           - 1.25f * static_cast<float>(tinyFragmentCount)
           - meanVoxelPenaltyWeight * meanVoxelPenalty;
}

std::vector<CellGroundTruthBuilder::DetectedCell> CellGroundTruthBuilder::detectCellsInVolume(
    const std::vector<cv::Mat> &volume,
    const std::string &frameStem)
{
    const std::vector<float> percentiles = groundTruthPercentiles(config.groundTruth);

    std::vector<DetectedCell> bestCells;
    float bestScore = -std::numeric_limits<float>::max();
    std::vector<DetectedCell> watershedRescueCandidates;
    if (config.groundTruth.seededWatershedEnabled)
    {
        std::vector<DetectedCell> watershedCells = detectCellsBySeededWatershed(volume, frameStem);
        if (!watershedCells.empty())
        {
            int totalVoxels = 0;
            int clampedMinorCount = 0;
            int verySmallCount = 0;
            int veryLargeCount = 0;
            int nearDuplicatePairs = 0;
            int flattenedCount = 0;
            int tinyFragmentCount = 0;
            const float watershedScore = scoreCandidateCells(
                watershedCells,
                totalVoxels,
                clampedMinorCount,
                verySmallCount,
                veryLargeCount,
                nearDuplicatePairs,
                flattenedCount,
                tinyFragmentCount);
            std::cout << "[GroundTruth Candidate] method=seeded_watershed"
                      << " cells=" << watershedCells.size()
                      << " total_voxels=" << totalVoxels
                      << " minor_floor=" << clampedMinorCount
                      << " very_small=" << verySmallCount
                      << " very_large=" << veryLargeCount
                      << " near_pairs=" << nearDuplicatePairs
                      << " flattened=" << flattenedCount
                      << " tiny_fragments=" << tinyFragmentCount
                      << " score=" << watershedScore
                      << std::endl;

            if (config.groundTruth.seededWatershedPreferOverPercentile)
            {
                return watershedCells;
            }

            watershedRescueCandidates = std::move(watershedCells);
        }
    }

    int firstCandidateCellCount = -1;
    for (float percentile : percentiles)
    {
        std::vector<DetectedCell> cells = detectCellsAtPercentile(volume, frameStem, percentile);
        if (firstCandidateCellCount < 0)
        {
            firstCandidateCellCount = static_cast<int>(cells.size());
        }
        int totalVoxels = 0;
        int clampedMinorCount = 0;
        int verySmallCount = 0;
        int veryLargeCount = 0;
        int nearDuplicatePairs = 0;
        int flattenedCount = 0;
        int tinyFragmentCount = 0;
        float score = scoreCandidateCells(
            cells,
            totalVoxels,
            clampedMinorCount,
            verySmallCount,
            veryLargeCount,
            nearDuplicatePairs,
            flattenedCount,
            tinyFragmentCount);
        // Prefer the highest threshold when candidates are otherwise close. This
        // keeps the raw-only detector from expanding into dim halo signal just
        // because the lower threshold produces slightly larger connected
        // components.
        score += config.groundTruth.candidatePercentilePreferenceWeight * (percentile - 99.0f);
        const bool protectEarlySparseFrame =
            config.groundTruth.candidateOvergrowthAlwaysBelowFirstCount > 0 &&
            firstCandidateCellCount > 0 &&
            firstCandidateCellCount <= config.groundTruth.candidateOvergrowthAlwaysBelowFirstCount;
        const bool protectNormalFrame =
            config.groundTruth.candidateOvergrowthMinFirstCount <= 0 ||
            firstCandidateCellCount >= config.groundTruth.candidateOvergrowthMinFirstCount;
        if (config.groundTruth.maxCandidateCountGrowthFromFirst > 1.0f &&
            config.groundTruth.candidateOvergrowthPenaltyWeight > 0.0f &&
            firstCandidateCellCount > 0 &&
            (protectEarlySparseFrame || protectNormalFrame))
        {
            const float growth = static_cast<float>(cells.size()) /
                                 static_cast<float>(std::max(1, firstCandidateCellCount));
            if (growth > config.groundTruth.maxCandidateCountGrowthFromFirst)
            {
                const float excess = growth - config.groundTruth.maxCandidateCountGrowthFromFirst;
                score -= config.groundTruth.candidateOvergrowthPenaltyWeight *
                         excess * excess *
                         static_cast<float>(firstCandidateCellCount);
            }
        }

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

    if (!watershedRescueCandidates.empty() &&
        config.groundTruth.seededWatershedPreferBaseCountMin > 0 &&
        config.groundTruth.seededWatershedPreferBaseCountMax >= config.groundTruth.seededWatershedPreferBaseCountMin)
    {
        const int baseCount = static_cast<int>(bestCells.size());
        const bool baseCountInRange =
            baseCount >= config.groundTruth.seededWatershedPreferBaseCountMin &&
            baseCount <= config.groundTruth.seededWatershedPreferBaseCountMax;
        if (baseCountInRange)
        {
            std::cout << "[GroundTruth SeededWatershed DensitySwitch]"
                      << " base_cells=" << baseCount
                      << " watershed_cells=" << watershedRescueCandidates.size()
                      << " min_base_count=" << config.groundTruth.seededWatershedPreferBaseCountMin
                      << " max_base_count=" << config.groundTruth.seededWatershedPreferBaseCountMax
                      << " selected=seeded_watershed"
                      << std::endl;
            return watershedRescueCandidates;
        }
        std::cout << "[GroundTruth SeededWatershed DensitySwitch]"
                  << " base_cells=" << baseCount
                  << " watershed_cells=" << watershedRescueCandidates.size()
                  << " min_base_count=" << config.groundTruth.seededWatershedPreferBaseCountMin
                  << " max_base_count=" << config.groundTruth.seededWatershedPreferBaseCountMax
                  << " selected=percentile_cc"
                  << std::endl;
    }

    if (config.groundTruth.seededWatershedRescueEnabled && !watershedRescueCandidates.empty())
    {
        std::sort(watershedRescueCandidates.begin(),
                  watershedRescueCandidates.end(),
                  [](const DetectedCell &lhs, const DetectedCell &rhs) {
                      if (std::abs(lhs.top10MinusShell - rhs.top10MinusShell) > 1e-3f)
                      {
                          return lhs.top10MinusShell > rhs.top10MinusShell;
                      }
                      return lhs.voxelCount > rhs.voxelCount;
                  });

        const float rescueDistance = std::max(1.0f, config.groundTruth.seededWatershedRescueMinDistance);
        const float rescueDistanceSq = rescueDistance * rescueDistance;
        const int fractionCap = config.groundTruth.seededWatershedRescueMaxAddedFraction > 0.0f
            ? static_cast<int>(std::ceil(config.groundTruth.seededWatershedRescueMaxAddedFraction *
                                         static_cast<float>(bestCells.size())))
            : std::numeric_limits<int>::max();
        const int absoluteCap = config.groundTruth.seededWatershedRescueMaxAdded > 0
            ? config.groundTruth.seededWatershedRescueMaxAdded
            : std::numeric_limits<int>::max();
        const int maxAdded = std::max(0, std::min(fractionCap, absoluteCap));

        int considered = 0;
        int rejectedNearExisting = 0;
        int rejectedWeakSignal = 0;
        int added = 0;
        for (const DetectedCell &candidate : watershedRescueCandidates)
        {
            if (added >= maxAdded)
            {
                break;
            }
            ++considered;

            if (config.groundTruth.minTop10MinusShell > 0.0f &&
                candidate.top10MinusShell < config.groundTruth.minTop10MinusShell)
            {
                ++rejectedWeakSignal;
                continue;
            }

            bool nearExisting = false;
            for (const DetectedCell &existing : bestCells)
            {
                if (squaredDistance(candidate.centerScaled, existing.centerScaled) <= rescueDistanceSq)
                {
                    nearExisting = true;
                    break;
                }
            }
            if (nearExisting)
            {
                ++rejectedNearExisting;
                continue;
            }

            bestCells.push_back(candidate);
            ++added;
        }

        std::sort(bestCells.begin(), bestCells.end(), [](const DetectedCell &lhs, const DetectedCell &rhs) {
            if (std::abs(lhs.centerScaled.y - rhs.centerScaled.y) > 1e-3f)
            {
                return lhs.centerScaled.y < rhs.centerScaled.y;
            }
            return lhs.centerScaled.x < rhs.centerScaled.x;
        });
        for (size_t i = 0; i < bestCells.size(); ++i)
        {
            bestCells[i].name = makeCellName(frameStem, static_cast<int>(i + 1));
            bestCells[i].zForCsv = bestCells[i].centerScaled.z / effectiveZScaling();
        }

        std::cout << "[GroundTruth SeededWatershed Rescue]"
                  << " candidates=" << watershedRescueCandidates.size()
                  << " considered=" << considered
                  << " added=" << added
                  << " max_added=" << maxAdded
                  << " rescue_distance=" << rescueDistance
                  << " rejected_near_existing=" << rejectedNearExisting
                  << " rejected_weak_signal=" << rejectedWeakSignal
                  << " final_cells=" << bestCells.size()
                  << std::endl;
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

    const EllipsoidConfig previousCellConfig = Ellipsoid::cellConfig;
    if (config.cell)
    {
        Ellipsoid::cellConfig = *config.cell;
        Ellipsoid::cellConfig.maxZ = static_cast<float>(displayFrame.size()) - 1.0f;
    }
    std::vector<Ellipsoid> ellipsoids = makeEllipsoids(displayCells);
    Ellipsoid::cellConfig = previousCellConfig;

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
                  << " top10I=" << cell.top10Intensity
                  << " shellMeanI=" << cell.shellMeanIntensity
                  << " top10MinusShell=" << cell.top10MinusShell
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
