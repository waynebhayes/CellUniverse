#include "../includes/CellLumen.hpp"
#include "../includes/CsvHandler.hpp"
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

std::vector<cv::Mat> loadCellLumenRawStack(const fs::path &imageFile)
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
            throw std::runtime_error("CellLumen TIFF has 0 slices: " + imageFile.string());
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
            throw std::runtime_error("Could not read CellLumen image: " + imageFile.string());
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

std::vector<float> cellLumenPercentiles()
{
    const char *overrideValue = std::getenv("CELLUNIVERSE_CELL_LUMEN_PERCENTILES");
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

std::vector<float> cellLumenPercentiles(const CellLumenConfig &config)
{
    const std::vector<float> envPercentiles = cellLumenPercentiles();
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

bool shouldSkipCellLumenTiffOutput()
{
    const char *value = std::getenv("CELLUNIVERSE_CELL_LUMEN_SKIP_TIFF");
    return value != nullptr && std::string(value) == "1";
}

bool shouldDisableCellLumenFragmentMerge()
{
    const char *value = std::getenv("CELLUNIVERSE_CELL_LUMEN_DISABLE_FRAGMENT_MERGE");
    return value != nullptr && std::string(value) == "1";
}

float cellLumenGeometricCenterBlend(const std::string &profileLabel)
{
    const char *value = std::getenv("CELLUNIVERSE_CELL_LUMEN_GEOMETRIC_CENTER_BLEND");
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

float cellLumenGeometricCenterBlend(const BaseConfig &config,
                                      const std::string &profileLabel)
{
    if (config.cellLumen.geometricCenterBlend >= 0.0f)
    {
        return std::clamp(config.cellLumen.geometricCenterBlend, 0.0f, 1.0f);
    }
    return cellLumenGeometricCenterBlend(profileLabel);
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
        throw std::runtime_error("Unsupported channel count for CellLumen TIFF export: " +
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

void writeCellLumenTiffStack(const fs::path &path, const std::vector<cv::Mat> &stack)
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
            throw std::runtime_error("CellLumen TIFF export requires same-size slices: " +
                                     path.string());
        }
        output.push_back(std::move(converted));
    }

    if (output.empty())
    {
        throw std::runtime_error("CellLumen TIFF export received an empty stack: " +
                                 path.string());
    }

    const std::vector<int> params = {
        cv::IMWRITE_TIFF_COMPRESSION, 1
    };
    if (!cv::imwritemulti(path.string(), output, params))
    {
        throw std::runtime_error("Failed to write CellLumen TIFF stack: " + path.string());
    }
}

float squaredDistance(const cv::Point3f &lhs, const cv::Point3f &rhs)
{
    const float dx = lhs.x - rhs.x;
    const float dy = lhs.y - rhs.y;
    const float dz = lhs.z - rhs.z;
    return dx * dx + dy * dy + dz * dz;
}

float sampleLineMinimum(const std::vector<cv::Mat> &volume,
                        const cv::Point3f &a,
                        const cv::Point3f &b)
{
    if (volume.empty())
    {
        return 0.0f;
    }

    const float dx = b.x - a.x;
    const float dy = b.y - a.y;
    const float dz = b.z - a.z;
    const int steps = std::max(4, static_cast<int>(std::ceil(
                                      std::sqrt(dx * dx + dy * dy + dz * dz))));
    float minValue = std::numeric_limits<float>::max();
    for (int i = 0; i <= steps; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(steps);
        const int x = static_cast<int>(std::lround(a.x + t * dx));
        const int y = static_cast<int>(std::lround(a.y + t * dy));
        const int z = static_cast<int>(std::lround(a.z + t * dz));
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
            minValue = std::min(minValue, value);
        }
    }

    return minValue == std::numeric_limits<float>::max() ? 0.0f : minValue;
}

struct LineIntensityProfile
{
    int count = 0;
    float minimum = 0.0f;
    float q20 = 0.0f;
    float median = 0.0f;
};

std::optional<LineIntensityProfile> sampleLineProfile(const std::vector<cv::Mat> &volume,
                                                      const cv::Point3f &a,
                                                      const cv::Point3f &b,
                                                      float zScale)
{
    if (volume.empty())
    {
        return std::nullopt;
    }

    const float dx = b.x - a.x;
    const float dy = b.y - a.y;
    const float dzScaled = (b.z - a.z) * zScale;
    const float distance = std::sqrt(dx * dx + dy * dy + dzScaled * dzScaled);
    const int steps = std::clamp(static_cast<int>(std::ceil(distance / 1.5f)), 8, 64);
    std::vector<float> samples;
    samples.reserve(static_cast<size_t>(steps + 1));
    for (int i = 0; i <= steps; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(steps);
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
        return std::nullopt;
    }

    std::sort(samples.begin(), samples.end());
    LineIntensityProfile profile;
    profile.count = static_cast<int>(samples.size());
    profile.minimum = samples.front();
    profile.q20 = samples[static_cast<size_t>(0.20f * static_cast<float>(samples.size() - 1))];
    profile.median = samples[samples.size() / 2];
    return profile;
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

CellLumen::CellLumen(BaseConfig config, const fs::path &outputDir)
    : config(std::move(config)),
      outputDir(outputDir),
      tracker(this->config, outputDir.string()),
      activeProfile{}
{
    tracker.setDebugVerbose(false);
    activeProfile.effectiveZScaling = std::max(1.0f, this->config.simulation.z_scaling);
}

float CellLumen::clampf(float value, float lo, float hi)
{
    return std::max(lo, std::min(value, hi));
}

std::string CellLumen::toLowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string CellLumen::extractFrameFolderName(const fs::path &imageFile)
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

std::string CellLumen::makeCellName(const std::string &frameStem, int index)
{
    std::ostringstream name;
    name << frameStem << "_cell_" << std::setfill('0') << std::setw(3) << index;
    return name.str();
}

CellLumen::DatasetProfile CellLumen::inferDatasetProfile(const fs::path &imageFile) const
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

    if (config.cellLumen.minHighSeedVoxels > 0)
    {
        profile.minHighSeedVoxels = config.cellLumen.minHighSeedVoxels;
    }
    if (config.cellLumen.seedMergeDistance > 0.0f)
    {
        profile.seedMergeDistance = config.cellLumen.seedMergeDistance;
    }
    if (config.cellLumen.seedSplitSeparation > 0.0f)
    {
        profile.seedSplitSeparation = config.cellLumen.seedSplitSeparation;
    }
    if (config.cellLumen.dedupDistance > 0.0f)
    {
        profile.dedupDistance = config.cellLumen.dedupDistance;
    }

    return profile;
}

float CellLumen::effectiveZScaling() const
{
    return std::max(1.0f, activeProfile.effectiveZScaling);
}

float CellLumen::estimateBackgroundValue(const std::vector<cv::Mat> &volume) const
{
    return percentileFromValues(collectSampledValues(volume),
                                clampf(config.simulation.post_alignment_black_percentile, 0.0f, 1.0f));
}

int CellLumen::computeMinComponentVoxels() const
{
    if (config.cellLumen.minComponentVoxels > 0)
    {
        return config.cellLumen.minComponentVoxels;
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

bool CellLumen::componentContainsBrightSeed(
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

CellLumen::DetectedCell CellLumen::makeDetectedCellFromComponent(
    const EmbryoBrightTracker::Comp3DStat &component) const
{
    DetectedCell cell;
    const cv::Point3f weightedCenter = component.center();
    const cv::Point3f geometricCenter = component.voxelCenter();
    const float geometricBlend = cellLumenGeometricCenterBlend(config, activeProfile.label);
    cell.centerScaled = weightedCenter * (1.0f - geometricBlend) + geometricCenter * geometricBlend;
    cell.centerScaled.z *= effectiveZScaling();
    cell.centerScaled.z += config.cellLumen.zCenterOffsetScaled;
    estimateAdaptiveRadii(component, cell.majorRadius, cell.bRadius, cell.minorRadius);
    cell.voxelCount = component.vox;
    cell.meanIntensity = component.meanI();
    cell.component = component;
    return cell;
}

void CellLumen::estimateAdaptiveRadii(const EmbryoBrightTracker::Comp3DStat &component,
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
    const float volumeScale = config.cellLumen.useScaledVoxelVolumeForRadius
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

    const float radiusScale = std::max(0.1f, config.cellLumen.radiusInflationScale);
    const float outputMinMajor = config.cellLumen.minOutputMajorRadius > 0.0f
        ? config.cellLumen.minOutputMajorRadius
        : minMajor;
    const float outputMinB = config.cellLumen.minOutputBRadius > 0.0f
        ? config.cellLumen.minOutputBRadius
        : minB;
    const float outputMinMinor = config.cellLumen.minOutputMinorRadius > 0.0f
        ? config.cellLumen.minOutputMinorRadius
        : minMinor;
    const float outputMaxMajor = config.cellLumen.maxOutputMajorRadius > 0.0f
        ? config.cellLumen.maxOutputMajorRadius
        : maxMajor;
    const float outputMaxB = config.cellLumen.maxOutputBRadius > 0.0f
        ? config.cellLumen.maxOutputBRadius
        : maxB;
    const float outputMaxMinor = config.cellLumen.maxOutputMinorRadius > 0.0f
        ? config.cellLumen.maxOutputMinorRadius
        : maxMinor;

    majorRadius = radiusScale * std::max(equivalentRadius, blendTowardAxis(xyMaxRadius, 1.0f));
    bRadius = radiusScale * blendTowardAxis(xyMinRadius, xySupport);
    minorRadius = radiusScale * blendTowardAxis(zRadius, zSupport * zSupport);

    majorRadius = clampf(majorRadius, outputMinMajor, outputMaxMajor);
    bRadius = clampf(bRadius, outputMinB, std::min(outputMaxB, majorRadius));
    minorRadius = clampf(minorRadius, outputMinMinor, std::min(outputMaxMinor, majorRadius));
}

std::optional<CellLumen::DetectedCell> CellLumen::detectLocalSeededCell(
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
        tracked.center.z * effectiveZScaling() + config.cellLumen.zCenterOffsetScaled);
    cell.voxelCount = tracked.voxelCount;
    cell.meanIntensity = tracked.meanIntensity;
    return cell;
}

std::vector<EmbryoBrightTracker::Comp3DStat> CellLumen::extractLocalMaximumSeeds(
    const std::vector<cv::Mat> &volume,
    float thresholdHigh,
    const std::vector<EmbryoBrightTracker::Comp3DStat> &componentSeeds) const
{
    if (!config.cellLumen.localMaxSeedEnabled || volume.empty())
    {
        return componentSeeds;
    }

    if (config.cellLumen.localMaxSeedMinStrongSeedCount > 0 &&
        static_cast<int>(componentSeeds.size()) < config.cellLumen.localMaxSeedMinStrongSeedCount)
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

    const float minDistance = std::max(1.0f, config.cellLumen.localMaxSeedMinDistance);
    const float minDistanceSq = minDistance * minDistance;
    const float zScale = effectiveZScaling();
    const size_t maxSeeds = config.cellLumen.localMaxSeedMaxPerFrameFactor > 0.0f
        ? static_cast<size_t>(std::ceil(config.cellLumen.localMaxSeedMaxPerFrameFactor *
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

    std::cout << "[CellLumen LocalMaxSeeds]"
              << " component_seeds=" << componentSeeds.size()
              << " local_peaks=" << peaks.size()
              << " kept=" << seeds.size()
              << " min_distance=" << minDistance
              << " threshold_high=" << thresholdHigh
              << std::endl;

    return seeds.empty() ? componentSeeds : seeds;
}

std::vector<cv::Mat> CellLumen::buildLocalContrastVolume(
    const std::vector<cv::Mat> &volume) const
{
    std::vector<cv::Mat> contrast;
    contrast.reserve(volume.size());
    if (volume.empty())
    {
        return contrast;
    }

    const double signalSigma = std::max(0.0f, config.cellLumen.seededWatershedSignalSigma);
    const double backgroundSigma = std::max(
        signalSigma + 0.5,
        static_cast<double>(std::max(1.0f, config.cellLumen.seededWatershedBackgroundSigma)));

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

    std::cout << "[CellLumen SeededWatershed Contrast]"
              << " signal_sigma=" << signalSigma
              << " background_sigma=" << backgroundSigma
              << " normalizer=" << normalizer
              << std::endl;
    return contrast;
}

std::vector<CellLumen::DetectedCell> CellLumen::detectCellsBySeededWatershed(
    const std::vector<cv::Mat> &volume,
    const std::string &frameStem) const
{
    if (!config.cellLumen.seededWatershedEnabled || volume.empty())
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
        config.cellLumen.seededWatershedSeedThresholdFloor,
        stackPercentile(contrast,
                        clampf(config.cellLumen.seededWatershedSeedPercentile / 100.0f, 0.0f, 1.0f),
                        true));
    float maskThreshold = std::max(
        config.cellLumen.seededWatershedMaskThresholdFloor,
        stackPercentile(contrast,
                        clampf(config.cellLumen.seededWatershedMaskPercentile / 100.0f, 0.0f, 1.0f),
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

    const float minSeedDistance = std::max(1.0f, config.cellLumen.seededWatershedMinSeedDistance);
    const float minSeedDistanceSq = minSeedDistance * minSeedDistance;
    const float closeSeedMinDistance = std::max(1.0f, config.cellLumen.seededWatershedCloseSeedMinDistance);
    const float closeSeedMinDistanceSq = closeSeedMinDistance * closeSeedMinDistance;
    const float minValleyDrop = std::max(0.0f, config.cellLumen.seededWatershedMinValleyDrop);
    const size_t maxSeeds = config.cellLumen.seededWatershedMaxSeeds > 0
        ? static_cast<size_t>(config.cellLumen.seededWatershedMaxSeeds)
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
            const float distSq = dx * dx + dy * dy + dz * dz;
            if (distSq <= closeSeedMinDistanceSq)
            {
                tooClose = true;
                break;
            }
            if (distSq <= minSeedDistanceSq)
            {
                const cv::Point3f peakPoint(static_cast<float>(peak.x),
                                            static_cast<float>(peak.y),
                                            static_cast<float>(peak.z));
                const cv::Point3f seedPoint(static_cast<float>(seed.x),
                                            static_cast<float>(seed.y),
                                            static_cast<float>(seed.z));
                const float valley = sampleLineMinimum(contrast, peakPoint, seedPoint);
                const float valleyDrop = std::min(peak.value, seed.value) - valley;
                if (valleyDrop < minValleyDrop)
                {
                    tooClose = true;
                    break;
                }
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
        std::cout << "[CellLumen SeededWatershed] no_seeds=1"
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

    const size_t voxelTotal = static_cast<size_t>(Z) * static_cast<size_t>(Y) * static_cast<size_t>(X);
    std::vector<float> distanceToBoundary;
    size_t foregroundVoxelCount = 0;
    size_t boundaryVoxelCount = 0;
    if (config.cellLumen.seededWatershedUseDistanceTransform)
    {
        std::vector<unsigned char> foreground(voxelTotal, 0);
        for (int z = 0; z < Z; ++z)
        {
            const cv::Mat &slice = contrast[static_cast<size_t>(z)];
            for (int y = 0; y < Y; ++y)
            {
                const float *row = slice.ptr<float>(y);
                for (int x = 0; x < X; ++x)
                {
                    if (row[x] >= maskThreshold)
                    {
                        foreground[lin(z, y, x)] = 1;
                        foregroundVoxelCount++;
                    }
                }
            }
        }

        struct DistanceNode
        {
            float distance = 0.0f;
            int x = 0;
            int y = 0;
            int z = 0;
        };
        struct DistanceNodeCompare
        {
            bool operator()(const DistanceNode &lhs, const DistanceNode &rhs) const
            {
                return lhs.distance > rhs.distance;
            }
        };
        const std::array<cv::Point3i, 6> distanceNeighbors = {
            cv::Point3i{-1, 0, 0}, cv::Point3i{1, 0, 0},
            cv::Point3i{0, -1, 0}, cv::Point3i{0, 1, 0},
            cv::Point3i{0, 0, -1}, cv::Point3i{0, 0, 1}
        };

        distanceToBoundary.assign(voxelTotal, std::numeric_limits<float>::max());
        std::priority_queue<DistanceNode,
                            std::vector<DistanceNode>,
                            DistanceNodeCompare> distanceQueue;
        for (int z = 0; z < Z; ++z)
        {
            for (int y = 0; y < Y; ++y)
            {
                for (int x = 0; x < X; ++x)
                {
                    const size_t index = lin(z, y, x);
                    if (foreground[index] == 0)
                    {
                        continue;
                    }

                    bool touchesBackground = false;
                    for (const auto &delta : distanceNeighbors)
                    {
                        const int nx = x + delta.x;
                        const int ny = y + delta.y;
                        const int nz = z + delta.z;
                        if (nx < 0 || nx >= X || ny < 0 || ny >= Y || nz < 0 || nz >= Z ||
                            foreground[lin(nz, ny, nx)] == 0)
                        {
                            touchesBackground = true;
                            break;
                        }
                    }

                    if (touchesBackground)
                    {
                        distanceToBoundary[index] = 0.0f;
                        distanceQueue.push(DistanceNode{0.0f, x, y, z});
                        boundaryVoxelCount++;
                    }
                }
            }
        }

        while (!distanceQueue.empty())
        {
            DistanceNode node = distanceQueue.top();
            distanceQueue.pop();
            const size_t index = lin(node.z, node.y, node.x);
            if (node.distance > distanceToBoundary[index] + 1e-4f)
            {
                continue;
            }

            for (const auto &delta : distanceNeighbors)
            {
                const int nx = node.x + delta.x;
                const int ny = node.y + delta.y;
                const int nz = node.z + delta.z;
                if (nx < 0 || nx >= X || ny < 0 || ny >= Y || nz < 0 || nz >= Z)
                {
                    continue;
                }

                const size_t neighborIndex = lin(nz, ny, nx);
                if (foreground[neighborIndex] == 0)
                {
                    continue;
                }

                const float step = delta.z == 0 ? 1.0f : zScale;
                const float nextDistance = node.distance + step;
                if (nextDistance < distanceToBoundary[neighborIndex])
                {
                    distanceToBoundary[neighborIndex] = nextDistance;
                    distanceQueue.push(DistanceNode{nextDistance, nx, ny, nz});
                }
            }
        }

        std::cout << "[CellLumen SeededWatershed DistanceMap]"
                  << " foreground_voxels=" << foregroundVoxelCount
                  << " boundary_voxels=" << boundaryVoxelCount
                  << " z_scale=" << zScale
                  << std::endl;
    }

    const float configuredMaxGrowDistance =
        std::max(1.0f, config.cellLumen.seededWatershedMaxGrowDistance);
    const auto voxelPriority = [&](int z, int y, int x, int seedIndex) -> float {
        const float value = contrast[static_cast<size_t>(z)].ptr<float>(y)[x];
        const Peak &seed = seeds[static_cast<size_t>(seedIndex)];
        const float dx = static_cast<float>(x - seed.x);
        const float dy = static_cast<float>(y - seed.y);
        const float dz = static_cast<float>(z - seed.z) * zScale;
        const float distancePenalty =
            config.cellLumen.seededWatershedSeedDistancePenaltyWeight *
            std::sqrt(dx * dx + dy * dy + dz * dz) /
            configuredMaxGrowDistance;
        if (!distanceToBoundary.empty())
        {
            const float boundaryDistance = distanceToBoundary[lin(z, y, x)];
            const float safeBoundaryDistance =
                std::isfinite(boundaryDistance) ? boundaryDistance : 0.0f;
            return config.cellLumen.seededWatershedDistancePriorityWeight * safeBoundaryDistance +
                   config.cellLumen.seededWatershedContrastPriorityWeight * value -
                   distancePenalty;
        }
        return value - distancePenalty;
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
    std::vector<int> labels(voxelTotal, 0);
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
        frontier.push(GrowNode{voxelPriority(seed.z, seed.y, seed.x, i), i, seed.x, seed.y, seed.z});
    }

    const float maxGrowDistance = configuredMaxGrowDistance;
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

            frontier.push(GrowNode{voxelPriority(nz, ny, nx, node.seedIndex), node.seedIndex, nx, ny, nz});
        }
    }

    const int configuredMinVoxels = config.cellLumen.seededWatershedMinVoxels > 0
        ? config.cellLumen.seededWatershedMinVoxels
        : computeMinComponentVoxels();
    int defaultQualifiedCells = 0;
    for (const auto &stat : stats)
    {
        if (stat.vox >= configuredMinVoxels && stat.sumW > 1e-9)
        {
            defaultQualifiedCells++;
        }
    }
    int minVoxels = configuredMinVoxels;
    if (config.cellLumen.seededWatershedAdaptiveMinVoxelsEnabled &&
        config.cellLumen.seededWatershedAdaptiveMinVoxelsCellCount > 0 &&
        config.cellLumen.seededWatershedAdaptiveDenseMinVoxels > 0 &&
        configuredMinVoxels > config.cellLumen.seededWatershedAdaptiveDenseMinVoxels &&
        defaultQualifiedCells >= config.cellLumen.seededWatershedAdaptiveMinVoxelsCellCount)
    {
        minVoxels = config.cellLumen.seededWatershedAdaptiveDenseMinVoxels;
    }
    std::vector<DetectedCell> cells;
    cells.reserve(stats.size());
    int skippedSmall = 0;
    const float centerSeedBlend = clampf(config.cellLumen.seededWatershedCenterSeedBlend, 0.0f, 1.0f);
    for (size_t i = 0; i < stats.size(); ++i)
    {
        const auto &stat = stats[i];
        if (stat.vox < minVoxels || stat.sumW <= 1e-9)
        {
            skippedSmall++;
            continue;
        }
        DetectedCell cell = makeDetectedCellFromComponent(stat);
        if (centerSeedBlend > 0.0f)
        {
            const Peak &seed = seeds[i];
            const cv::Point3f seedCenter(static_cast<float>(seed.x),
                                         static_cast<float>(seed.y),
                                         static_cast<float>(seed.z) * zScale +
                                             config.cellLumen.zCenterOffsetScaled);
            cell.centerScaled.x = (1.0f - centerSeedBlend) * cell.centerScaled.x +
                                  centerSeedBlend * seedCenter.x;
            cell.centerScaled.y = (1.0f - centerSeedBlend) * cell.centerScaled.y +
                                  centerSeedBlend * seedCenter.y;
            cell.centerScaled.z = (1.0f - centerSeedBlend) * cell.centerScaled.z +
                                  centerSeedBlend * seedCenter.z;
            cell.zForCsv = cell.centerScaled.z / zScale;
        }
        cells.push_back(cell);
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

    std::cout << "[CellLumen SeededWatershed]"
              << " peaks=" << peaks.size()
              << " seeds=" << seeds.size()
              << " seed_threshold=" << seedThreshold
              << " mask_threshold=" << maskThreshold
              << " min_seed_distance=" << minSeedDistance
              << " close_seed_min_distance=" << closeSeedMinDistance
              << " min_valley_drop=" << minValleyDrop
              << " max_grow_distance=" << maxGrowDistance
              << " distance_transform=" << (distanceToBoundary.empty() ? 0 : 1)
              << " center_seed_blend=" << centerSeedBlend
              << " assigned_voxels=" << assignedVoxels
              << " min_voxels=" << minVoxels
              << " configured_min_voxels=" << configuredMinVoxels
              << " default_qualified_cells=" << defaultQualifiedCells
              << " skipped_small=" << skippedSmall
              << " cells=" << cells.size()
              << std::endl;

    return cells;
}

std::vector<EmbryoBrightTracker::Comp3DStat> CellLumen::collapseNearbySeeds(
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
            const float zDistanceScale = config.cellLumen.seedMergeUseScaledZ ? effectiveZScaling() : 1.0f;
            const float distZ = absoluteDistanceZ(seed.center(), existing.center()) * zDistanceScale;
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

bool CellLumen::shouldSplitCoarseComponent(
    const DetectedCell &coarseCell,
    const std::vector<EmbryoBrightTracker::Comp3DStat> &containedSeeds) const
{
    const float maxAllowedMajor = config.cellLumen.seededSplitMaxMajorRadius > 0.0f
        ? config.cellLumen.seededSplitMaxMajorRadius
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
    float maxSeedSeparation3D = 0.0f;
    float maxSeedSeparationZ = 0.0f;
    const float zScale = effectiveZScaling();
    for (size_t i = 0; i < containedSeeds.size(); ++i)
    {
        for (size_t j = i + 1; j < containedSeeds.size(); ++j)
        {
            const cv::Point3f lhs = containedSeeds[i].center();
            const cv::Point3f rhs = containedSeeds[j].center();
            const float dx = lhs.x - rhs.x;
            const float dy = lhs.y - rhs.y;
            const float dzScaled = (lhs.z - rhs.z) * zScale;
            const float separationXY = std::sqrt(dx * dx + dy * dy);
            const float separation3D = std::sqrt(dx * dx + dy * dy + dzScaled * dzScaled);
            maxSeedSeparationXY = std::max(maxSeedSeparationXY, separationXY);
            maxSeedSeparation3D = std::max(maxSeedSeparation3D, separation3D);
            maxSeedSeparationZ = std::max(maxSeedSeparationZ, std::abs(dzScaled));
        }
    }

    const float dominantSeedVox = static_cast<float>(containedSeeds.front().vox);
    const float secondSeedVox = static_cast<float>(containedSeeds[1].vox);
    const float balance = secondSeedVox / std::max(dominantSeedVox, 1.0f);

    const float softSplitThreshold = std::max(4.0f, coarseCell.majorRadius * 0.45f);
    const float minSeedBalance = config.cellLumen.seededSplitMinSeedBalance;
    if (balance >= minSeedBalance && maxSeedSeparationXY >= softSplitThreshold)
    {
        return true;
    }

    const float configuredHardSplit = config.cellLumen.seededSplitMinSeedSeparation > 0.0f
        ? config.cellLumen.seededSplitMinSeedSeparation
        : activeProfile.seedSplitSeparation;
    const float hardSplitThreshold = std::max(configuredHardSplit, coarseCell.majorRadius * 0.75f);
    if (config.cellLumen.seededSplitUseScaled3DSeparation && balance >= minSeedBalance)
    {
        const float configured3D = config.cellLumen.seededSplitMinScaled3DSeparation > 0.0f
            ? config.cellLumen.seededSplitMinScaled3DSeparation
            : configuredHardSplit;
        const float configuredZ = config.cellLumen.seededSplitMinScaledZSeparation > 0.0f
            ? config.cellLumen.seededSplitMinScaledZSeparation
            : configured3D;
        const float hard3DThreshold = std::max(configured3D, coarseCell.majorRadius * 0.65f);
        if (maxSeedSeparation3D >= hard3DThreshold &&
            maxSeedSeparationZ >= configuredZ)
        {
            return true;
        }
    }
    return maxSeedSeparationXY >= hardSplitThreshold;
}

std::vector<CellLumen::DetectedCell> CellLumen::detectCellsAtPercentile(
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

    std::cout << "[CellLumen Detect] percentileHigh=" << percentileHigh
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
    if (config.cellLumen.adaptiveSeedMergeEnabled &&
        activeProfile.label == "celegans_embryo")
    {
        const int strongSeedCount = static_cast<int>(strongHighComponents.size());
        if (config.cellLumen.adaptiveSeedMergeDenseHighSeedCount > 0 &&
            strongSeedCount >= config.cellLumen.adaptiveSeedMergeDenseHighSeedCount)
        {
            if (config.cellLumen.adaptiveSeedMergeDenseDistance > 0.0f)
            {
                effectiveSeedMergeDistance = config.cellLumen.adaptiveSeedMergeDenseDistance;
            }
        }
        else if (config.cellLumen.adaptiveSeedMergeModerateHighSeedCount > 0 &&
                 strongSeedCount >= config.cellLumen.adaptiveSeedMergeModerateHighSeedCount)
        {
            if (config.cellLumen.adaptiveSeedMergeModerateDistance > 0.0f)
            {
                effectiveSeedMergeDistance = config.cellLumen.adaptiveSeedMergeModerateDistance;
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
            (!sparseSeedField || config.cellLumen.allowSeededSplitInSparseField) &&
            shouldSplitCoarseComponent(coarseCell, containedSeeds);

        if (needsSeededSplit)
        {
            splitComponents++;
            for (const auto &seed : containedSeeds)
            {
                if (config.cellLumen.useHighSeedCenterForSplitCells)
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
    float effectiveDedupRadiusScale = config.cellLumen.dedupRadiusScale;
    if (config.cellLumen.adaptiveDedupEnabled &&
        activeProfile.label == "celegans_embryo")
    {
        const int strongSeedCount = static_cast<int>(strongHighComponents.size());
        if (config.cellLumen.adaptiveDedupDenseHighSeedCount > 0 &&
            strongSeedCount >= config.cellLumen.adaptiveDedupDenseHighSeedCount)
        {
            if (config.cellLumen.adaptiveDedupDenseDistance > 0.0f)
            {
                effectiveDedupDistance = config.cellLumen.adaptiveDedupDenseDistance;
            }
            if (config.cellLumen.adaptiveDedupDenseRadiusScale > 0.0f)
            {
                effectiveDedupRadiusScale = config.cellLumen.adaptiveDedupDenseRadiusScale;
            }
        }
        else if (config.cellLumen.adaptiveDedupModerateHighSeedCount > 0 &&
                 strongSeedCount >= config.cellLumen.adaptiveDedupModerateHighSeedCount)
        {
            if (config.cellLumen.adaptiveDedupModerateDistance > 0.0f)
            {
                effectiveDedupDistance = config.cellLumen.adaptiveDedupModerateDistance;
            }
            if (config.cellLumen.adaptiveDedupModerateRadiusScale > 0.0f)
            {
                effectiveDedupRadiusScale = config.cellLumen.adaptiveDedupModerateRadiusScale;
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
    std::cout << "[CellLumen CCStats] percentileHigh=" << percentileHigh
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

std::vector<CellLumen::DetectedCell> CellLumen::pruneLikelySatelliteCells(
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

void CellLumen::annotateSignalStats(const std::vector<cv::Mat> &volume,
                                                 DetectedCell &cell) const
{
    cell.shellVoxelCount = 0;
    cell.top10Intensity = cell.meanIntensity;
    cell.shellMeanIntensity = 0.0f;
    cell.meanMinusShell = 0.0f;
    cell.top10MinusShell = 0.0f;
    if (!config.cellLumen.signalStatsEnabled || volume.empty())
    {
        return;
    }

    const float zScale = effectiveZScaling();
    const float innerScale = std::max(1.0f, config.cellLumen.shellInnerScale);
    const float outerScale = std::max(innerScale + 0.01f, config.cellLumen.shellOuterScale);
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

std::vector<CellLumen::DetectedCell> CellLumen::applyBiologicalPriors(
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
    int rejectedZRange = 0;
    for (DetectedCell cell : cells)
    {
        annotateSignalStats(volume, cell);
        const float componentDx = static_cast<float>(cell.component.x1 - cell.component.x0 + 1);
        const float componentDy = static_cast<float>(cell.component.y1 - cell.component.y0 + 1);
        const float componentDz = static_cast<float>(cell.component.z1 - cell.component.z0 + 1) * effectiveZScaling();
        const float componentSpanRadius = 0.5f * std::max({componentDx, componentDy, componentDz});

        bool reject = false;
        if (config.cellLumen.minOutputCenterZScaled >= 0.0f &&
            cell.centerScaled.z < config.cellLumen.minOutputCenterZScaled)
        {
            reject = true;
            rejectedZRange++;
        }
        if (!reject &&
            config.cellLumen.maxOutputCenterZScaled >= 0.0f &&
            cell.centerScaled.z > config.cellLumen.maxOutputCenterZScaled)
        {
            reject = true;
            rejectedZRange++;
        }

        if (!reject && config.cellLumen.rejectTinyBrightArtifacts)
        {
            const bool biologicallyTinyByVoxel =
                config.cellLumen.minBiologicalVoxelCount > 0 &&
                cell.voxelCount < config.cellLumen.minBiologicalVoxelCount;
            const bool biologicallyTinyBySpan =
                config.cellLumen.minBiologicalMajorRadius > 0.0f &&
                componentSpanRadius < config.cellLumen.minBiologicalMajorRadius * 0.70f;
            const bool allowSparseArtifactRule =
                config.cellLumen.smallArtifactSparseMaxCells <= 0 ||
                static_cast<int>(cells.size()) <= config.cellLumen.smallArtifactSparseMaxCells;
            const bool tinyArtifactByConfig =
                allowSparseArtifactRule &&
                config.cellLumen.smallArtifactMaxVoxels > 0 &&
                cell.voxelCount <= config.cellLumen.smallArtifactMaxVoxels &&
                (config.cellLumen.smallArtifactMaxTop10Intensity < 0.0f ||
                 cell.top10Intensity <= config.cellLumen.smallArtifactMaxTop10Intensity) &&
                (config.cellLumen.smallArtifactMaxMeanIntensity < 0.0f ||
                 cell.meanIntensity <= config.cellLumen.smallArtifactMaxMeanIntensity);
            if ((biologicallyTinyByVoxel && biologicallyTinyBySpan) || tinyArtifactByConfig)
            {
                reject = true;
                rejectedTiny++;
            }
        }

        if (!reject && config.cellLumen.signalStatsEnabled)
        {
            if (config.cellLumen.minTop10MinusShell >= 0.0f &&
                cell.top10MinusShell < config.cellLumen.minTop10MinusShell)
            {
                reject = true;
                rejectedContrast++;
            }
            if (!reject &&
                config.cellLumen.minMeanMinusShell > -999999.0f &&
                cell.meanMinusShell < config.cellLumen.minMeanMinusShell)
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
    float effectiveMinCenterDistance = config.cellLumen.minBiologicalCenterDistance;
    if (config.cellLumen.adaptiveNearDuplicateMergeEnabled &&
        config.cellLumen.adaptiveNearDuplicateDistance > 0.0f)
    {
        const int cellCount = static_cast<int>(filtered.size());
        const bool aboveMin = config.cellLumen.adaptiveNearDuplicateMinCells <= 0 ||
                              cellCount >= config.cellLumen.adaptiveNearDuplicateMinCells;
        const bool belowMax = config.cellLumen.adaptiveNearDuplicateMaxCells <= 0 ||
                              cellCount <= config.cellLumen.adaptiveNearDuplicateMaxCells;
        bool pairRatioOk = true;
        if (config.cellLumen.adaptiveNearDuplicateMinPairRatio > 0.0f)
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
            pairRatioOk = pairRatio >= config.cellLumen.adaptiveNearDuplicateMinPairRatio;
        }
        if (aboveMin && belowMax && pairRatioOk)
        {
            effectiveMinCenterDistance = std::max(
                effectiveMinCenterDistance,
                config.cellLumen.adaptiveNearDuplicateDistance);
        }
    }
    if (config.cellLumen.biologicalNearDuplicateMergeEnabled &&
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

    std::cout << "[CellLumen BioPriors]"
              << " stage=" << stage
              << " before=" << cells.size()
              << " after=" << filtered.size()
              << " rejected_tiny=" << rejectedTiny
              << " rejected_contrast=" << rejectedContrast
              << " rejected_z_range=" << rejectedZRange
              << " merged_near_duplicates=" << mergedPairs
              << " min_component_voxels=" << config.cellLumen.minComponentVoxels
              << " min_top10_minus_shell=" << config.cellLumen.minTop10MinusShell
              << " min_center_distance=" << effectiveMinCenterDistance
              << std::endl;
    return filtered;
}

std::vector<CellLumen::DetectedCell> CellLumen::filterWeakSatelliteCells(
    const std::vector<cv::Mat> &volume,
    const std::vector<DetectedCell> &cells) const
{
    if (!config.cellLumen.finalWeakSatelliteFilterEnabled ||
        cells.size() < 2 ||
        volume.empty())
    {
        return cells;
    }

    std::vector<DetectedCell> annotated = cells;
    for (DetectedCell &cell : annotated)
    {
        annotateSignalStats(volume, cell);
    }

    const float neighborDistance =
        std::max(0.0f, config.cellLumen.finalWeakSatelliteNeighborDistance);
    const float neighborDistanceSq = neighborDistance * neighborDistance;
    const int maxVoxels = config.cellLumen.finalWeakSatelliteMaxVoxels;
    const float maxSignal = config.cellLumen.finalWeakSatelliteMaxTop10MinusShell;
    const float maxMinor = config.cellLumen.finalWeakSatelliteMaxMinorRadius;
    const int neighborMinVoxels = config.cellLumen.finalWeakSatelliteNeighborMinVoxels;
    const float neighborVoxelRatio =
        std::max(1.0f, config.cellLumen.finalWeakSatelliteNeighborVoxelRatio);

    std::vector<DetectedCell> filtered;
    filtered.reserve(annotated.size());
    int rejected = 0;
    for (size_t i = 0; i < annotated.size(); ++i)
    {
        const DetectedCell &cell = annotated[i];
        const bool weakSmallCandidate =
            (maxVoxels <= 0 || cell.voxelCount <= maxVoxels) &&
            (maxSignal < 0.0f || cell.top10MinusShell <= maxSignal) &&
            (maxMinor < 0.0f || cell.minorRadius <= maxMinor);

        bool hasDominantNeighbor = false;
        if (weakSmallCandidate && neighborDistance > 0.0f)
        {
            for (size_t j = 0; j < annotated.size(); ++j)
            {
                if (i == j)
                {
                    continue;
                }

                const DetectedCell &neighbor = annotated[j];
                if (neighbor.voxelCount < neighborMinVoxels ||
                    static_cast<float>(neighbor.voxelCount) <
                        neighborVoxelRatio * static_cast<float>(std::max(1, cell.voxelCount)))
                {
                    continue;
                }

                if (squaredDistance(cell.centerScaled, neighbor.centerScaled) <= neighborDistanceSq)
                {
                    hasDominantNeighbor = true;
                    break;
                }
            }
        }

        if (weakSmallCandidate && hasDominantNeighbor)
        {
            rejected++;
            continue;
        }
        filtered.push_back(cell);
    }

    std::cout << "[CellLumen WeakSatelliteFilter]"
              << " enabled=1"
              << " before=" << cells.size()
              << " after=" << filtered.size()
              << " rejected=" << rejected
              << " neighbor_distance=" << neighborDistance
              << " max_voxels=" << maxVoxels
              << " max_top10_minus_shell=" << maxSignal
              << " max_minor_radius=" << maxMinor
              << " neighbor_min_voxels=" << neighborMinVoxels
              << " neighbor_voxel_ratio=" << neighborVoxelRatio
              << std::endl;
    return filtered;
}

std::vector<CellLumen::DetectedCell> CellLumen::mergeLikelySameCellFragments(
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
        config.cellLumen.fragmentMergeMaxInputCells > 0 &&
        static_cast<int>(cells.size()) > config.cellLumen.fragmentMergeMaxInputCells;
    if (shouldDisableCellLumenFragmentMerge() ||
        !config.cellLumen.fragmentMergeEnabled ||
        disabledByCellCount)
    {
        std::cout << "[CellLumen FragmentMerge]"
                  << " before=" << cells.size()
                  << " after=" << cells.size()
                  << " merged_pairs=0"
                  << " median_major=" << medianMajor
                  << " mode="
                  << (shouldDisableCellLumenFragmentMerge()
                          ? "disabled_by_env"
                          : (!config.cellLumen.fragmentMergeEnabled ? "disabled_by_config" : "disabled_by_cell_count"))
                  << std::endl;
        return cells;
    }

    if (activeProfile.label == "celegans_embryo" && cells.size() >= 245)
    {
        std::cout << "[CellLumen FragmentMerge]"
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
                mergeDistance = std::max(
                    mergeDistance,
                    std::max(config.cellLumen.fragmentMergeSparseMinDistance,
                             medianMajor * config.cellLumen.fragmentMergeSparseMajorScale));
            }
            else if (moderateMode)
            {
                mergeDistance = std::max(
                    mergeDistance,
                    std::max(config.cellLumen.fragmentMergeModerateMinDistance,
                             medianMajor * config.cellLumen.fragmentMergeModerateMajorScale));
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

    std::cout << "[CellLumen FragmentMerge]"
              << " before=" << cells.size()
              << " after=" << merged.size()
              << " merged_pairs=" << mergedPairs
              << " median_major=" << medianMajor
              << std::endl;

    return merged;
}

void CellLumen::refineCentersByZColumn(const std::vector<cv::Mat> &volume,
                                                    std::vector<DetectedCell> &cells) const
{
    if (!config.cellLumen.finalZColumnRefineEnabled ||
        volume.empty() ||
        cells.empty())
    {
        return;
    }

    const int cellCount = static_cast<int>(cells.size());
    if ((config.cellLumen.finalZColumnRefineMinCells > 0 &&
         cellCount < config.cellLumen.finalZColumnRefineMinCells) ||
        (config.cellLumen.finalZColumnRefineMaxCells > 0 &&
         cellCount > config.cellLumen.finalZColumnRefineMaxCells))
    {
        std::cout << "[CellLumen FinalZColumnRefine]"
                  << " enabled=1"
                  << " skipped=count_gate"
                  << " cells=" << cellCount
                  << " min_cells=" << config.cellLumen.finalZColumnRefineMinCells
                  << " max_cells=" << config.cellLumen.finalZColumnRefineMaxCells
                  << std::endl;
        return;
    }

    const float zScale = effectiveZScaling();
    const int Z = static_cast<int>(volume.size());
    const int Y = volume[0].rows;
    const int X = volume[0].cols;
    const float radiusXY = std::max(1.0f, config.cellLumen.finalZColumnRefineRadiusXY);
    const float radiusXYSq = radiusXY * radiusXY;
    const float halfWindowScaled =
        std::max(zScale, config.cellLumen.finalZColumnRefineHalfWindowScaled);
    const float quantile = clampf(config.cellLumen.finalZColumnRefineQuantile, 0.0f, 0.99f);
    const float minScoreFraction =
        clampf(config.cellLumen.finalZColumnRefineMinScoreFraction, 0.0f, 1.0f);
    const float maxMoveScaled =
        std::max(0.0f, config.cellLumen.finalZColumnRefineMaxMoveScaled);
    const float blend = clampf(config.cellLumen.finalZColumnRefineBlend, 0.0f, 1.0f);
    const bool positiveOnly = config.cellLumen.finalZColumnRefinePositiveOnly;
    int refinedCenters = 0;

    for (DetectedCell &cell : cells)
    {
        const int minX = std::max(0, static_cast<int>(std::floor(cell.centerScaled.x - radiusXY)));
        const int maxX = std::min(X - 1, static_cast<int>(std::ceil(cell.centerScaled.x + radiusXY)));
        const int minY = std::max(0, static_cast<int>(std::floor(cell.centerScaled.y - radiusXY)));
        const int maxY = std::min(Y - 1, static_cast<int>(std::ceil(cell.centerScaled.y + radiusXY)));
        const int centerZRaw = static_cast<int>(std::lround(cell.centerScaled.z / zScale));
        const int zWindow = std::max(1, static_cast<int>(std::ceil(halfWindowScaled / zScale)));
        const int minZ = std::max(0, centerZRaw - zWindow);
        const int maxZ = std::min(Z - 1, centerZRaw + zWindow);

        struct SliceScore
        {
            float zScaled = 0.0f;
            float score = 0.0f;
        };
        std::vector<SliceScore> sliceScores;
        sliceScores.reserve(static_cast<size_t>(std::max(0, maxZ - minZ + 1)));
        float bestScore = 0.0f;

        for (int z = minZ; z <= maxZ; ++z)
        {
            std::vector<float> values;
            values.reserve(static_cast<size_t>(
                std::max(1, maxY - minY + 1) *
                std::max(1, maxX - minX + 1)));
            const cv::Mat &slice = volume[static_cast<size_t>(z)];
            for (int y = minY; y <= maxY; ++y)
            {
                const float dy = static_cast<float>(y) - cell.centerScaled.y;
                const float *row = slice.ptr<float>(y);
                for (int x = minX; x <= maxX; ++x)
                {
                    const float dx = static_cast<float>(x) - cell.centerScaled.x;
                    const float value = row[x];
                    if (dx * dx + dy * dy <= radiusXYSq &&
                        std::isfinite(value))
                    {
                        values.push_back(value);
                    }
                }
            }

            if (values.size() < 12)
            {
                continue;
            }

            const float median = percentileFromValues(values, 0.50f);
            const float highCutoff = percentileFromValues(values, quantile);
            double topSum = 0.0;
            int topCount = 0;
            for (float value : values)
            {
                if (value >= highCutoff)
                {
                    topSum += static_cast<double>(value);
                    topCount++;
                }
            }
            if (topCount == 0)
            {
                continue;
            }

            const float topMean = static_cast<float>(topSum / static_cast<double>(topCount));
            const float score = std::max(0.0f, topMean - median);
            sliceScores.push_back({static_cast<float>(z) * zScale, score});
            bestScore = std::max(bestScore, score);
        }

        if (sliceScores.empty() || bestScore <= 0.0f)
        {
            continue;
        }

        const float scoreFloor = bestScore * minScoreFraction;
        double weightedZ = 0.0;
        double totalWeight = 0.0;
        for (const SliceScore &sliceScore : sliceScores)
        {
            if (sliceScore.score < scoreFloor)
            {
                continue;
            }
            const double weight = static_cast<double>(sliceScore.score * sliceScore.score);
            weightedZ += weight * static_cast<double>(sliceScore.zScaled);
            totalWeight += weight;
        }

        if (totalWeight <= 0.0)
        {
            continue;
        }

        const float proposedZ = static_cast<float>(weightedZ / totalWeight);
        const float deltaZ = proposedZ - cell.centerScaled.z;
        if (positiveOnly && deltaZ < 0.0f)
        {
            continue;
        }
        if (std::abs(deltaZ) > maxMoveScaled)
        {
            continue;
        }

        const float refinedZ = cell.centerScaled.z + blend * deltaZ;
        if (!std::isfinite(refinedZ))
        {
            continue;
        }

        cell.centerScaled.z = refinedZ;
        cell.zForCsv = cell.centerScaled.z / zScale;
        annotateSignalStats(volume, cell);
        refinedCenters++;
    }

    std::cout << "[CellLumen FinalZColumnRefine]"
              << " enabled=1"
              << " radius_xy=" << radiusXY
              << " half_window_scaled=" << halfWindowScaled
              << " quantile=" << quantile
              << " min_score_fraction=" << minScoreFraction
              << " max_move_scaled=" << maxMoveScaled
              << " blend=" << blend
              << " positive_only=" << positiveOnly
              << " refined_centers=" << refinedCenters
              << " final_cells=" << cells.size()
              << std::endl;
}

std::vector<CellLumen::DetectedCell> CellLumen::splitCellsByLocalZPeaks(
    const std::vector<cv::Mat> &volume,
    const std::vector<DetectedCell> &cells,
    const std::string &frameStem) const
{
    if (!config.cellLumen.finalZPeakSplitEnabled ||
        volume.empty() ||
        cells.empty())
    {
        return cells;
    }

    const int cellCount = static_cast<int>(cells.size());
    if ((config.cellLumen.finalZPeakSplitMinCells > 0 &&
         cellCount < config.cellLumen.finalZPeakSplitMinCells) ||
        (config.cellLumen.finalZPeakSplitMaxCells > 0 &&
         cellCount > config.cellLumen.finalZPeakSplitMaxCells))
    {
        std::cout << "[CellLumen FinalZPeakSplit]"
                  << " enabled=1"
                  << " skipped=count_gate"
                  << " cells=" << cellCount
                  << " min_cells=" << config.cellLumen.finalZPeakSplitMinCells
                  << " max_cells=" << config.cellLumen.finalZPeakSplitMaxCells
                  << std::endl;
        return cells;
    }

    struct SlicePeak
    {
        int zRaw = 0;
        float zScaled = 0.0f;
        float score = 0.0f;
        float x = 0.0f;
        float y = 0.0f;
        int topCount = 0;
    };

    const float zScale = effectiveZScaling();
    const int Z = static_cast<int>(volume.size());
    const int Y = volume[0].rows;
    const int X = volume[0].cols;
    const int maxAdded = std::max(0, config.cellLumen.finalZPeakSplitMaxAdded);
    const bool addOnly = config.cellLumen.finalZPeakSplitAddOnlyEnabled;
    const float minMajor = std::max(0.0f, config.cellLumen.finalZPeakSplitMinMajorRadius);
    const float radiusXY = std::max(3.0f, config.cellLumen.finalZPeakSplitRadiusXY);
    const float radiusXYSq = radiusXY * radiusXY;
    const float quantile = clampf(config.cellLumen.finalZPeakSplitQuantile, 0.50f, 0.98f);
    const float minPeakFraction =
        clampf(config.cellLumen.finalZPeakSplitMinPeakScoreFraction, 0.05f, 0.95f);
    const float minSeparation = std::max(zScale, config.cellLumen.finalZPeakSplitMinSeparationScaled);
    const float maxSeparation = config.cellLumen.finalZPeakSplitMaxSeparationScaled > 0.0f
        ? config.cellLumen.finalZPeakSplitMaxSeparationScaled
        : std::numeric_limits<float>::max();
    const float maxShiftXY = std::max(0.0f, config.cellLumen.finalZPeakSplitMaxCenterShiftXY);
    const float radiusScale = clampf(config.cellLumen.finalZPeakSplitRadiusScale, 0.30f, 1.0f);
    const bool prioritizeCandidates =
        addOnly && config.cellLumen.finalZPeakSplitPrioritizeCandidates;
    const float priorityMinPeakShift =
        std::max(0.0f, config.cellLumen.finalZPeakSplitPriorityMinPeakShiftScaled);
    const float priorityScoreWeight =
        std::max(0.0f, config.cellLumen.finalZPeakSplitPriorityScoreWeight);
    const int zWindow = std::max(2, static_cast<int>(std::ceil(maxSeparation / zScale)) + 1);
    const bool local3DFallbackEnabled =
        config.cellLumen.finalZPeakSplitLocal3DFallbackEnabled;
    const bool preferLocal3D =
        config.cellLumen.finalZPeakSplitPreferLocal3D;
    const float local3DCentroidRadiusXY =
        std::max(1.0f, config.cellLumen.finalZPeakSplitLocal3DCentroidRadiusXY);
    const float local3DCentroidRadiusXYSq =
        local3DCentroidRadiusXY * local3DCentroidRadiusXY;
    const float local3DCentroidHalfWindowScaled =
        std::max(zScale, config.cellLumen.finalZPeakSplitLocal3DCentroidHalfWindowScaled);
    const int local3DCentroidHalfWindowRaw =
        std::max(1, static_cast<int>(std::ceil(local3DCentroidHalfWindowScaled / zScale)));

    std::vector<DetectedCell> splitCells;
    splitCells.reserve(cells.size() + static_cast<size_t>(maxAdded));
    struct SplitOption
    {
        DetectedCell lhs;
        DetectedCell rhs;
        float priority = 0.0f;
        bool usedLocal3D = false;
    };
    std::vector<SplitOption> prioritizedOptions;
    if (prioritizeCandidates)
    {
        prioritizedOptions.reserve(cells.size());
    }
    int considered = 0;
    int splitParents = 0;
    int local3DSplitParents = 0;
    int added = 0;
    int rejectedParentSignal = 0;
    int rejectedNoPeaks = 0;
    int rejectedSeparation = 0;
    int rejectedShift = 0;
    int rejectedPriorityShift = 0;

    for (const DetectedCell &cell : cells)
    {
        if ((!prioritizeCandidates && added >= maxAdded) || cell.majorRadius < minMajor)
        {
            splitCells.push_back(cell);
            continue;
        }

        considered++;
        if (config.cellLumen.finalZPeakSplitMaxParentTop10MinusShell >= 0.0f &&
            cell.top10MinusShell > config.cellLumen.finalZPeakSplitMaxParentTop10MinusShell)
        {
            rejectedParentSignal++;
            splitCells.push_back(cell);
            continue;
        }

        const int centerZRaw = static_cast<int>(std::lround(cell.centerScaled.z / zScale));
        const int minZ = std::max(0, centerZRaw - zWindow);
        const int maxZ = std::min(Z - 1, centerZRaw + zWindow);
        const int minX = std::max(0, static_cast<int>(std::floor(cell.centerScaled.x - radiusXY)));
        const int maxX = std::min(X - 1, static_cast<int>(std::ceil(cell.centerScaled.x + radiusXY)));
        const int minY = std::max(0, static_cast<int>(std::floor(cell.centerScaled.y - radiusXY)));
        const int maxY = std::min(Y - 1, static_cast<int>(std::ceil(cell.centerScaled.y + radiusXY)));

        struct BrightVoxel
        {
            int x = 0;
            int y = 0;
            int zRaw = 0;
            float score = 0.0f;
        };

        std::vector<SlicePeak> peaks;
        peaks.reserve(static_cast<size_t>(std::max(0, maxZ - minZ + 1)));
        for (int z = minZ; z <= maxZ; ++z)
        {
            const cv::Mat &slice = volume[static_cast<size_t>(z)];
            std::vector<float> values;
            values.reserve(static_cast<size_t>(
                std::max(1, maxY - minY + 1) *
                std::max(1, maxX - minX + 1)));
            for (int y = minY; y <= maxY; ++y)
            {
                const float dy = static_cast<float>(y) - cell.centerScaled.y;
                const float *row = slice.ptr<float>(y);
                for (int x = minX; x <= maxX; ++x)
                {
                    const float dx = static_cast<float>(x) - cell.centerScaled.x;
                    const float value = row[x];
                    if (dx * dx + dy * dy <= radiusXYSq && std::isfinite(value))
                    {
                        values.push_back(value);
                    }
                }
            }
            if (values.size() < 16)
            {
                continue;
            }

            const float median = percentileFromValues(values, 0.50f);
            const float highCutoff = percentileFromValues(values, quantile);
            double topSum = 0.0;
            double weightSum = 0.0;
            double weightedX = 0.0;
            double weightedY = 0.0;
            int topCount = 0;
            for (int y = minY; y <= maxY; ++y)
            {
                const float dy = static_cast<float>(y) - cell.centerScaled.y;
                const float *row = slice.ptr<float>(y);
                for (int x = minX; x <= maxX; ++x)
                {
                    const float dx = static_cast<float>(x) - cell.centerScaled.x;
                    const float value = row[x];
                    if (dx * dx + dy * dy > radiusXYSq ||
                        !std::isfinite(value) ||
                        value < highCutoff)
                    {
                        continue;
                    }
                    const double weight = std::max(0.001f, value - median);
                    topSum += static_cast<double>(value);
                    weightSum += weight;
                    weightedX += weight * static_cast<double>(x);
                    weightedY += weight * static_cast<double>(y);
                    topCount++;
                }
            }
            if (topCount < 6 || weightSum <= 0.0)
            {
                continue;
            }

            const float topMean = static_cast<float>(topSum / static_cast<double>(topCount));
            const float score = std::max(0.0f, topMean - median);
            if (score <= 0.0f)
            {
                continue;
            }
            peaks.push_back({
                z,
                static_cast<float>(z) * zScale,
                score,
                static_cast<float>(weightedX / weightSum),
                static_cast<float>(weightedY / weightSum),
                topCount
            });
        }

        auto findLocal3DPeakPair = [&]() -> std::optional<std::pair<SlicePeak, SlicePeak>> {
            std::vector<float> values;
            values.reserve(static_cast<size_t>(
                std::max(1, maxZ - minZ + 1) *
                std::max(1, maxY - minY + 1) *
                std::max(1, maxX - minX + 1)));
            for (int z = minZ; z <= maxZ; ++z)
            {
                const cv::Mat &slice = volume[static_cast<size_t>(z)];
                for (int y = minY; y <= maxY; ++y)
                {
                    const float dy = static_cast<float>(y) - cell.centerScaled.y;
                    const float *row = slice.ptr<float>(y);
                    for (int x = minX; x <= maxX; ++x)
                    {
                        const float dx = static_cast<float>(x) - cell.centerScaled.x;
                        const float value = row[x];
                        if (dx * dx + dy * dy <= radiusXYSq && std::isfinite(value))
                        {
                            values.push_back(value);
                        }
                    }
                }
            }
            if (values.size() < 32)
            {
                return std::nullopt;
            }

            const float median = percentileFromValues(values, 0.50f);
            const float highCutoff = percentileFromValues(values, quantile);
            std::vector<BrightVoxel> brightVoxels;
            brightVoxels.reserve(values.size() / 10 + 1);
            for (int z = minZ; z <= maxZ; ++z)
            {
                const cv::Mat &slice = volume[static_cast<size_t>(z)];
                for (int y = minY; y <= maxY; ++y)
                {
                    const float dy = static_cast<float>(y) - cell.centerScaled.y;
                    const float *row = slice.ptr<float>(y);
                    for (int x = minX; x <= maxX; ++x)
                    {
                        const float dx = static_cast<float>(x) - cell.centerScaled.x;
                        const float value = row[x];
                        if (dx * dx + dy * dy > radiusXYSq ||
                            !std::isfinite(value) ||
                            value < highCutoff)
                        {
                            continue;
                        }
                        const float score = std::max(0.0f, value - median);
                        if (score > 0.0f)
                        {
                            brightVoxels.push_back({x, y, z, score});
                        }
                    }
                }
            }
            if (brightVoxels.size() < 12)
            {
                return std::nullopt;
            }

            std::sort(brightVoxels.begin(), brightVoxels.end(), [](const BrightVoxel &lhs,
                                                                    const BrightVoxel &rhs) {
                return lhs.score > rhs.score;
            });
            const BrightVoxel firstVoxel = brightVoxels.front();
            std::optional<BrightVoxel> secondVoxel;
            for (size_t i = 1; i < brightVoxels.size(); ++i)
            {
                const float dx = static_cast<float>(brightVoxels[i].x - firstVoxel.x);
                const float dy = static_cast<float>(brightVoxels[i].y - firstVoxel.y);
                const float dz = static_cast<float>(brightVoxels[i].zRaw - firstVoxel.zRaw) * zScale;
                const float separation = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (separation < minSeparation || separation > maxSeparation)
                {
                    continue;
                }
                if (brightVoxels[i].score < firstVoxel.score * minPeakFraction)
                {
                    continue;
                }
                secondVoxel = brightVoxels[i];
                break;
            }
            if (!secondVoxel)
            {
                return std::nullopt;
            }

            auto centroidAround = [&](const BrightVoxel &peak) -> std::optional<SlicePeak> {
                double topSum = 0.0;
                double weightSum = 0.0;
                double weightedX = 0.0;
                double weightedY = 0.0;
                double weightedZ = 0.0;
                int topCount = 0;
                const int peakMinZ = std::max(minZ, peak.zRaw - local3DCentroidHalfWindowRaw);
                const int peakMaxZ = std::min(maxZ, peak.zRaw + local3DCentroidHalfWindowRaw);
                const int peakMinX = std::max(minX, static_cast<int>(std::floor(peak.x - local3DCentroidRadiusXY)));
                const int peakMaxX = std::min(maxX, static_cast<int>(std::ceil(peak.x + local3DCentroidRadiusXY)));
                const int peakMinY = std::max(minY, static_cast<int>(std::floor(peak.y - local3DCentroidRadiusXY)));
                const int peakMaxY = std::min(maxY, static_cast<int>(std::ceil(peak.y + local3DCentroidRadiusXY)));
                for (int z = peakMinZ; z <= peakMaxZ; ++z)
                {
                    const cv::Mat &slice = volume[static_cast<size_t>(z)];
                    for (int y = peakMinY; y <= peakMaxY; ++y)
                    {
                        const float dy = static_cast<float>(y - peak.y);
                        const float *row = slice.ptr<float>(y);
                        for (int x = peakMinX; x <= peakMaxX; ++x)
                        {
                            const float dx = static_cast<float>(x - peak.x);
                            const float value = row[x];
                            if (dx * dx + dy * dy > local3DCentroidRadiusXYSq ||
                                !std::isfinite(value) ||
                                value < highCutoff)
                            {
                                continue;
                            }
                            const double weight = std::max(0.001f, value - median);
                            topSum += static_cast<double>(value);
                            weightSum += weight;
                            weightedX += weight * static_cast<double>(x);
                            weightedY += weight * static_cast<double>(y);
                            weightedZ += weight * static_cast<double>(z) * static_cast<double>(zScale);
                            topCount++;
                        }
                    }
                }
                if (topCount < 6 || weightSum <= 0.0)
                {
                    return std::nullopt;
                }
                return SlicePeak{
                    peak.zRaw,
                    static_cast<float>(weightedZ / weightSum),
                    std::max(0.0f, static_cast<float>(topSum / static_cast<double>(topCount)) - median),
                    static_cast<float>(weightedX / weightSum),
                    static_cast<float>(weightedY / weightSum),
                    topCount
                };
            };

            auto firstPeak3D = centroidAround(firstVoxel);
            auto secondPeak3D = centroidAround(*secondVoxel);
            if (!firstPeak3D || !secondPeak3D)
            {
                return std::nullopt;
            }
            return std::make_pair(*firstPeak3D, *secondPeak3D);
        };

        SlicePeak firstPeak;
        std::optional<SlicePeak> secondPeak;
        bool usedLocal3D = false;
        bool hadEnoughSlicePeaks = peaks.size() >= 2;
        if (local3DFallbackEnabled && preferLocal3D)
        {
            auto local3DPair = findLocal3DPeakPair();
            if (local3DPair)
            {
                firstPeak = local3DPair->first;
                secondPeak = local3DPair->second;
                usedLocal3D = true;
            }
        }

        if (!secondPeak && hadEnoughSlicePeaks)
        {
            std::sort(peaks.begin(), peaks.end(), [](const SlicePeak &lhs, const SlicePeak &rhs) {
                return lhs.score > rhs.score;
            });
            firstPeak = peaks.front();
            for (size_t i = 1; i < peaks.size(); ++i)
            {
                const float separation = std::abs(peaks[i].zScaled - firstPeak.zScaled);
                if (separation < minSeparation || separation > maxSeparation)
                {
                    continue;
                }
                if (peaks[i].score < firstPeak.score * minPeakFraction)
                {
                    continue;
                }
                secondPeak = peaks[i];
                break;
            }
        }

        if (!secondPeak && local3DFallbackEnabled)
        {
            auto local3DPair = findLocal3DPeakPair();
            if (local3DPair)
            {
                firstPeak = local3DPair->first;
                secondPeak = local3DPair->second;
                usedLocal3D = true;
            }
        }

        if (!secondPeak)
        {
            if (hadEnoughSlicePeaks)
            {
                rejectedSeparation++;
            }
            else
            {
                rejectedNoPeaks++;
            }
            splitCells.push_back(cell);
            continue;
        }

        const auto shiftXY = [&](const SlicePeak &peak) {
            const float dx = peak.x - cell.centerScaled.x;
            const float dy = peak.y - cell.centerScaled.y;
            return std::sqrt(dx * dx + dy * dy);
        };
        if (shiftXY(firstPeak) > maxShiftXY || shiftXY(*secondPeak) > maxShiftXY)
        {
            rejectedShift++;
            splitCells.push_back(cell);
            continue;
        }

        DetectedCell lhs = cell;
        DetectedCell rhs = cell;
        lhs.centerScaled = cv::Point3f(firstPeak.x, firstPeak.y, firstPeak.zScaled);
        rhs.centerScaled = cv::Point3f(secondPeak->x, secondPeak->y, secondPeak->zScaled);
        lhs.zForCsv = lhs.centerScaled.z / zScale;
        rhs.zForCsv = rhs.centerScaled.z / zScale;
        lhs.majorRadius *= radiusScale;
        lhs.bRadius *= radiusScale;
        lhs.minorRadius *= radiusScale;
        rhs.majorRadius *= radiusScale;
        rhs.bRadius *= radiusScale;
        rhs.minorRadius *= radiusScale;
        lhs.voxelCount = std::max(1, cell.voxelCount / 2);
        rhs.voxelCount = std::max(1, cell.voxelCount - lhs.voxelCount);
        annotateSignalStats(volume, lhs);
        annotateSignalStats(volume, rhs);

        bool parentSplitRecorded = false;
        if (addOnly)
        {
            splitCells.push_back(cell);
            if (prioritizeCandidates)
            {
                const auto peakDistance = [&](const DetectedCell &candidate) {
                    const float dx = candidate.centerScaled.x - cell.centerScaled.x;
                    const float dy = candidate.centerScaled.y - cell.centerScaled.y;
                    const float dz = candidate.centerScaled.z - cell.centerScaled.z;
                    return std::sqrt(dx * dx + dy * dy + dz * dz);
                };
                const float peakShift =
                    std::max(peakDistance(lhs), peakDistance(rhs));
                if (peakShift < priorityMinPeakShift)
                {
                    rejectedPriorityShift++;
                }
                else
                {
                    const float secondaryScore =
                        std::min(firstPeak.score, secondPeak->score);
                    const float priority =
                        peakShift + priorityScoreWeight * secondaryScore +
                        (usedLocal3D ? 3.0f : 0.0f);
                    prioritizedOptions.push_back({lhs, rhs, priority, usedLocal3D});
                }
            }
            else if (added < maxAdded)
            {
                splitCells.push_back(lhs);
                added++;
                parentSplitRecorded = true;
            }
            if (!prioritizeCandidates && added < maxAdded)
            {
                splitCells.push_back(rhs);
                added++;
                parentSplitRecorded = true;
            }
        }
        else
        {
            splitCells.push_back(lhs);
            splitCells.push_back(rhs);
            added++;
            parentSplitRecorded = true;
        }

        if (parentSplitRecorded)
        {
            splitParents++;
            if (usedLocal3D)
            {
                local3DSplitParents++;
            }
        }
    }

    if (prioritizeCandidates && !prioritizedOptions.empty())
    {
        std::sort(prioritizedOptions.begin(), prioritizedOptions.end(),
                  [](const SplitOption &lhs, const SplitOption &rhs) {
            return lhs.priority > rhs.priority;
        });
        for (const SplitOption &option : prioritizedOptions)
        {
            bool selectedParent = false;
            if (added < maxAdded)
            {
                splitCells.push_back(option.lhs);
                added++;
                selectedParent = true;
            }
            if (added < maxAdded)
            {
                splitCells.push_back(option.rhs);
                added++;
                selectedParent = true;
            }
            if (selectedParent)
            {
                splitParents++;
                if (option.usedLocal3D)
                {
                    local3DSplitParents++;
                }
            }
            if (added >= maxAdded)
            {
                break;
            }
        }
    }

    if (splitParents > 0)
    {
        std::sort(splitCells.begin(), splitCells.end(), [](const DetectedCell &lhs, const DetectedCell &rhs) {
            if (std::abs(lhs.centerScaled.y - rhs.centerScaled.y) > 1e-3f)
            {
                return lhs.centerScaled.y < rhs.centerScaled.y;
            }
            return lhs.centerScaled.x < rhs.centerScaled.x;
        });
        for (size_t i = 0; i < splitCells.size(); ++i)
        {
            splitCells[i].name = makeCellName(frameStem, static_cast<int>(i + 1));
            splitCells[i].zForCsv = splitCells[i].centerScaled.z / zScale;
        }
    }

    std::cout << "[CellLumen FinalZPeakSplit]"
              << " enabled=1"
              << " before=" << cells.size()
              << " after=" << splitCells.size()
              << " considered=" << considered
              << " split_parents=" << splitParents
              << " local3d_split_parents=" << local3DSplitParents
              << " added=" << added
              << " max_added=" << maxAdded
              << " add_only=" << addOnly
              << " prioritize=" << prioritizeCandidates
              << " prioritized_candidates=" << prioritizedOptions.size()
              << " rejected_parent_signal=" << rejectedParentSignal
              << " rejected_no_peaks=" << rejectedNoPeaks
              << " rejected_separation=" << rejectedSeparation
              << " rejected_shift=" << rejectedShift
              << " rejected_priority_shift=" << rejectedPriorityShift
              << " max_parent_top10_minus_shell="
              << config.cellLumen.finalZPeakSplitMaxParentTop10MinusShell
              << " min_major=" << minMajor
              << " radius_xy=" << radiusXY
              << " min_separation_scaled=" << minSeparation
              << " max_separation_scaled=" << maxSeparation
              << std::endl;

    return splitCells;
}

std::vector<CellLumen::DetectedCell> CellLumen::addZProfileRescueCandidates(
    const std::vector<cv::Mat> &volume,
    const std::vector<DetectedCell> &cells,
    const std::string &frameStem) const
{
    if (!config.cellLumen.finalZProfileRescueAddEnabled ||
        volume.empty() ||
        cells.empty())
    {
        return cells;
    }

    const int cellCount = static_cast<int>(cells.size());
    if ((config.cellLumen.finalZProfileRescueMinCells > 0 &&
         cellCount < config.cellLumen.finalZProfileRescueMinCells) ||
        (config.cellLumen.finalZProfileRescueMaxCells > 0 &&
         cellCount > config.cellLumen.finalZProfileRescueMaxCells))
    {
        std::cout << "[CellLumen ZProfileRescue]"
                  << " enabled=1 skipped=count_gate"
                  << " cells=" << cellCount
                  << " min_cells=" << config.cellLumen.finalZProfileRescueMinCells
                  << " max_cells=" << config.cellLumen.finalZProfileRescueMaxCells
                  << std::endl;
        return cells;
    }

    struct SliceScore
    {
        float zScaled = 0.0f;
        float score = 0.0f;
    };
    struct RescueOption
    {
        DetectedCell cell;
        float priority = 0.0f;
        cv::Point3f parentCenter{};
        float shift = 0.0f;
        float score = 0.0f;
    };

    const float zScale = effectiveZScaling();
    const int Z = static_cast<int>(volume.size());
    const int Y = volume[0].rows;
    const int X = volume[0].cols;
    const float radiusXY = std::max(1.0f, config.cellLumen.finalZProfileRescueRadiusXY);
    const float radiusXYSq = radiusXY * radiusXY;
    const float halfWindowScaled =
        std::max(zScale, config.cellLumen.finalZProfileRescueHalfWindowScaled);
    const int zWindow = std::max(1, static_cast<int>(std::ceil(halfWindowScaled / zScale)));
    const float quantile = clampf(config.cellLumen.finalZProfileRescueQuantile, 0.0f, 0.99f);
    const float minScoreFraction =
        clampf(config.cellLumen.finalZProfileRescueMinScoreFraction, 0.05f, 0.99f);
    const float minShift = std::max(0.0f, config.cellLumen.finalZProfileRescueMinShiftScaled);
    const float maxShift = config.cellLumen.finalZProfileRescueMaxShiftScaled > 0.0f
        ? config.cellLumen.finalZProfileRescueMaxShiftScaled
        : std::numeric_limits<float>::max();
    const float minExistingDistance =
        std::max(0.0f, config.cellLumen.finalZProfileRescueMinExistingDistance);
    const float minExistingDistanceSq = minExistingDistance * minExistingDistance;
    const float radiusScale =
        clampf(config.cellLumen.finalZProfileRescueRadiusScale, 0.30f, 1.0f);
    int maxAdded = std::max(0, config.cellLumen.finalZProfileRescueMaxAdded);
    const bool boostWindowMatches =
        config.cellLumen.finalZProfileRescueBoostMaxAdded > maxAdded &&
        (config.cellLumen.finalZProfileRescueBoostMinCells <= 0 ||
         cellCount >= config.cellLumen.finalZProfileRescueBoostMinCells) &&
        (config.cellLumen.finalZProfileRescueBoostMaxCells <= 0 ||
         cellCount <= config.cellLumen.finalZProfileRescueBoostMaxCells);
    if (boostWindowMatches)
    {
        // This narrow recall boost is for sparse frames where collapse leaves a
        // small, clean-looking candidate set but one true z-separated center is
        // still present in lower-ranked image evidence. A global budget raise
        // creates too many internal fragments, so the boost is count-gated.
        maxAdded = config.cellLumen.finalZProfileRescueBoostMaxAdded;
    }

    std::vector<RescueOption> options;
    options.reserve(cells.size());
    int considered = 0;
    int rejectedNoProfile = 0;
    int rejectedNoUpperPeak = 0;
    int rejectedExistingDistance = 0;

    for (const DetectedCell &cell : cells)
    {
        considered++;
        const int minX = std::max(0, static_cast<int>(std::floor(cell.centerScaled.x - radiusXY)));
        const int maxX = std::min(X - 1, static_cast<int>(std::ceil(cell.centerScaled.x + radiusXY)));
        const int minY = std::max(0, static_cast<int>(std::floor(cell.centerScaled.y - radiusXY)));
        const int maxY = std::min(Y - 1, static_cast<int>(std::ceil(cell.centerScaled.y + radiusXY)));
        const int centerZRaw = static_cast<int>(std::lround(cell.centerScaled.z / zScale));
        const int minZ = std::max(0, centerZRaw - zWindow);
        const int maxZ = std::min(Z - 1, centerZRaw + zWindow);

        std::vector<SliceScore> scores;
        scores.reserve(static_cast<size_t>(std::max(0, maxZ - minZ + 1)));
        float bestScore = 0.0f;
        for (int z = minZ; z <= maxZ; ++z)
        {
            std::vector<float> values;
            values.reserve(static_cast<size_t>(
                std::max(1, maxY - minY + 1) *
                std::max(1, maxX - minX + 1)));
            const cv::Mat &slice = volume[static_cast<size_t>(z)];
            for (int y = minY; y <= maxY; ++y)
            {
                const float dy = static_cast<float>(y) - cell.centerScaled.y;
                const float *row = slice.ptr<float>(y);
                for (int x = minX; x <= maxX; ++x)
                {
                    const float dx = static_cast<float>(x) - cell.centerScaled.x;
                    const float value = row[x];
                    if (dx * dx + dy * dy <= radiusXYSq && std::isfinite(value))
                    {
                        values.push_back(value);
                    }
                }
            }
            if (values.size() < 12)
            {
                continue;
            }

            const float median = percentileFromValues(values, 0.50f);
            const float cutoff = percentileFromValues(values, quantile);
            double topSum = 0.0;
            int topCount = 0;
            for (float value : values)
            {
                if (value >= cutoff)
                {
                    topSum += static_cast<double>(value);
                    topCount++;
                }
            }
            if (topCount == 0)
            {
                continue;
            }
            const float topMean = static_cast<float>(topSum / static_cast<double>(topCount));
            const float score = std::max(0.0f, topMean - median);
            scores.push_back({static_cast<float>(z) * zScale, score});
            bestScore = std::max(bestScore, score);
        }

        if (scores.empty() || bestScore <= 0.0f)
        {
            rejectedNoProfile++;
            continue;
        }

        const float scoreFloor = bestScore * minScoreFraction;
        std::optional<SliceScore> bestUpper;
        for (const SliceScore &score : scores)
        {
            const float shift = score.zScaled - cell.centerScaled.z;
            if (shift < minShift || shift > maxShift || score.score < scoreFloor)
            {
                continue;
            }
            if (!bestUpper ||
                score.score > bestUpper->score ||
                (std::abs(score.score - bestUpper->score) < 1e-3f &&
                 score.zScaled > bestUpper->zScaled))
            {
                bestUpper = score;
            }
        }
        if (!bestUpper)
        {
            rejectedNoUpperPeak++;
            continue;
        }

        DetectedCell rescue = cell;
        rescue.centerScaled.z = bestUpper->zScaled;
        rescue.zForCsv = rescue.centerScaled.z / zScale;
        rescue.majorRadius *= radiusScale;
        rescue.bRadius *= radiusScale;
        rescue.minorRadius *= radiusScale;
        annotateSignalStats(volume, rescue);

        bool tooClose = false;
        for (const DetectedCell &existing : cells)
        {
            if (squaredDistance(rescue.centerScaled, existing.centerScaled) < minExistingDistanceSq)
            {
                tooClose = true;
                break;
            }
        }
        if (tooClose)
        {
            rejectedExistingDistance++;
            continue;
        }

        const float shift = rescue.centerScaled.z - cell.centerScaled.z;
        // The original rescue priority rewarded larger z shifts, which could
        // spend the tiny add-back budget on far jumps before a nearer true
        // z-separated peak. Keep the old behavior by default; tuning YAMLs can
        // lower or invert this weight when the local z density evidence should
        // dominate over shift length.
        const float priority = bestUpper->score +
                               config.cellLumen.finalZProfileRescueShiftPriorityWeight * shift;
        options.push_back({rescue, priority, cell.centerScaled, shift, bestUpper->score});
    }

    std::sort(options.begin(), options.end(), [](const RescueOption &lhs,
                                                 const RescueOption &rhs) {
        return lhs.priority > rhs.priority;
    });

    // Keep this diagnostic attached to the rescue stage rather than a separate
    // ad hoc script. The whole purpose of this default-off rescue is to recover
    // z-separated doublets without GT, so the runtime log must show which image
    // evidence won the small candidate budget.
    const int debugCount = std::min<int>(static_cast<int>(options.size()), 12);
    for (int i = 0; i < debugCount; ++i)
    {
        const RescueOption &option = options[static_cast<size_t>(i)];
        std::cout << "[CellLumen ZProfileRescue Option]"
                  << " rank=" << (i + 1)
                  << " parent=(" << option.parentCenter.x
                  << "," << option.parentCenter.y
                  << "," << option.parentCenter.z << ")"
                  << " rescue=(" << option.cell.centerScaled.x
                  << "," << option.cell.centerScaled.y
                  << "," << option.cell.centerScaled.z << ")"
                  << " shift=" << option.shift
                  << " score=" << option.score
                  << " priority=" << option.priority
                  << std::endl;
    }

    std::vector<DetectedCell> output = cells;
    int added = 0;
    for (const RescueOption &option : options)
    {
        if (added >= maxAdded)
        {
            break;
        }
        output.push_back(option.cell);
        added++;
    }

    if (added > 0)
    {
        std::sort(output.begin(), output.end(), [](const DetectedCell &lhs,
                                                   const DetectedCell &rhs) {
            if (std::abs(lhs.centerScaled.y - rhs.centerScaled.y) > 1e-3f)
            {
                return lhs.centerScaled.y < rhs.centerScaled.y;
            }
            return lhs.centerScaled.x < rhs.centerScaled.x;
        });
        for (size_t i = 0; i < output.size(); ++i)
        {
            output[i].name = makeCellName(frameStem, static_cast<int>(i + 1));
            output[i].zForCsv = output[i].centerScaled.z / zScale;
        }
    }

    std::cout << "[CellLumen ZProfileRescue]"
              << " enabled=1"
              << " before=" << cells.size()
              << " after=" << output.size()
              << " considered=" << considered
              << " options=" << options.size()
              << " added=" << added
              << " max_added=" << maxAdded
              << " boost_window=" << boostWindowMatches
              << " min_shift=" << minShift
              << " max_shift=" << maxShift
              << " min_score_fraction=" << minScoreFraction
              << " rejected_no_profile=" << rejectedNoProfile
              << " rejected_no_upper_peak=" << rejectedNoUpperPeak
              << " rejected_existing_distance=" << rejectedExistingDistance
              << std::endl;

    return output;
}

std::vector<CellLumen::DetectedCell> CellLumen::rescueBrightPairMidpoints(
    const std::vector<cv::Mat> &volume,
    const std::vector<DetectedCell> &cells,
    const std::string &frameStem) const
{
    if (!config.cellLumen.finalBrightPairMidpointRescueEnabled ||
        volume.empty() ||
        cells.size() < 2)
    {
        return cells;
    }

    const int cellCount = static_cast<int>(cells.size());
    if ((config.cellLumen.finalBrightPairMidpointRescueMinCells > 0 &&
         cellCount < config.cellLumen.finalBrightPairMidpointRescueMinCells) ||
        (config.cellLumen.finalBrightPairMidpointRescueMaxCells > 0 &&
         cellCount > config.cellLumen.finalBrightPairMidpointRescueMaxCells))
    {
        std::cout << "[CellLumen BrightPairMidpointRescue]"
                  << " enabled=1"
                  << " skipped=count_gate"
                  << " cells=" << cellCount
                  << " min_cells=" << config.cellLumen.finalBrightPairMidpointRescueMinCells
                  << " max_cells=" << config.cellLumen.finalBrightPairMidpointRescueMaxCells
                  << std::endl;
        return cells;
    }

    struct MidpointCandidate
    {
        DetectedCell cell;
        float pairDistance = 0.0f;
        float signal = 0.0f;
        float nearestExistingDistance = 0.0f;
        size_t lhs = 0;
        size_t rhs = 0;
    };

    const float minDistance =
        std::max(0.0f, config.cellLumen.finalBrightPairMidpointRescueMinDistance);
    const float maxDistance =
        std::max(minDistance, config.cellLumen.finalBrightPairMidpointRescueMaxDistance);
    const float minSignal =
        std::max(0.0f, config.cellLumen.finalBrightPairMidpointRescueMinTop10MinusShell);
    const float radiusScale =
        clampf(config.cellLumen.finalBrightPairMidpointRescueRadiusScale, 0.25f, 1.25f);
    const float minExistingDistance =
        config.cellLumen.finalBrightPairMidpointRescueMinExistingDistance;
    const float minExistingDistanceSq = minExistingDistance * minExistingDistance;
    const bool requireExistingDistance = minExistingDistance > 0.0f;
    const bool sortByExistingDistance =
        config.cellLumen.finalBrightPairMidpointRescueSortByExistingDistance;

    std::vector<MidpointCandidate> candidates;
    int rejectedNearExisting = 0;
    for (size_t i = 0; i < cells.size(); ++i)
    {
        for (size_t j = i + 1; j < cells.size(); ++j)
        {
            const float distance = std::sqrt(squaredDistance(cells[i].centerScaled,
                                                             cells[j].centerScaled));
            if (distance < minDistance || distance > maxDistance)
            {
                continue;
            }

            DetectedCell midpoint = cells[i].top10MinusShell >= cells[j].top10MinusShell
                ? cells[i]
                : cells[j];
            midpoint.centerScaled.x = 0.5f * (cells[i].centerScaled.x + cells[j].centerScaled.x);
            midpoint.centerScaled.y = 0.5f * (cells[i].centerScaled.y + cells[j].centerScaled.y);
            midpoint.centerScaled.z = 0.5f * (cells[i].centerScaled.z + cells[j].centerScaled.z);
            midpoint.zForCsv = midpoint.centerScaled.z / effectiveZScaling();
            midpoint.majorRadius =
                0.5f * (cells[i].majorRadius + cells[j].majorRadius) * radiusScale;
            midpoint.bRadius =
                0.5f * (cells[i].bRadius + cells[j].bRadius) * radiusScale;
            midpoint.minorRadius =
                0.5f * (cells[i].minorRadius + cells[j].minorRadius) * radiusScale;
            midpoint.voxelCount =
                std::max(1, (cells[i].voxelCount + cells[j].voxelCount) / 2);
            annotateSignalStats(volume, midpoint);
            if (midpoint.top10MinusShell < minSignal)
            {
                continue;
            }
            float nearestExistingDistanceSq = std::numeric_limits<float>::max();
            for (const DetectedCell &existing : cells)
            {
                nearestExistingDistanceSq =
                    std::min(nearestExistingDistanceSq,
                             squaredDistance(midpoint.centerScaled,
                                             existing.centerScaled));
            }
            if (requireExistingDistance &&
                nearestExistingDistanceSq < minExistingDistanceSq)
            {
                rejectedNearExisting++;
                continue;
            }
            candidates.push_back({
                midpoint,
                distance,
                midpoint.top10MinusShell,
                std::sqrt(nearestExistingDistanceSq),
                i,
                j
            });
        }
    }

    std::sort(candidates.begin(), candidates.end(), [&](const MidpointCandidate &lhs,
                                                        const MidpointCandidate &rhs) {
        if (sortByExistingDistance &&
            std::abs(lhs.nearestExistingDistance - rhs.nearestExistingDistance) > 1e-3f)
        {
            return lhs.nearestExistingDistance > rhs.nearestExistingDistance;
        }
        if (std::abs(lhs.signal - rhs.signal) > 1e-3f)
        {
            return lhs.signal > rhs.signal;
        }
        return lhs.pairDistance < rhs.pairDistance;
    });

    std::vector<DetectedCell> rescued = cells;
    const int maxAdded = std::max(0, config.cellLumen.finalBrightPairMidpointRescueMaxAdded);
    const float minAddedDistance =
        std::max(0.0f, config.cellLumen.finalBrightPairMidpointRescueMinAddedDistance);
    const float minAddedDistanceSq = minAddedDistance * minAddedDistance;
    int added = 0;
    int rejectedNearAdded = 0;
    for (const MidpointCandidate &candidate : candidates)
    {
        if (added >= maxAdded)
        {
            break;
        }
        bool nearAdded = false;
        for (size_t k = cells.size(); k < rescued.size(); ++k)
        {
            if (squaredDistance(candidate.cell.centerScaled,
                                rescued[k].centerScaled) <= minAddedDistanceSq)
            {
                nearAdded = true;
                break;
            }
        }
        if (nearAdded)
        {
            rejectedNearAdded++;
            continue;
        }
        rescued.push_back(candidate.cell);
        added++;
    }

    if (added > 0)
    {
        std::sort(rescued.begin(), rescued.end(), [](const DetectedCell &lhs,
                                                     const DetectedCell &rhs) {
            if (std::abs(lhs.centerScaled.y - rhs.centerScaled.y) > 1e-3f)
            {
                return lhs.centerScaled.y < rhs.centerScaled.y;
            }
            return lhs.centerScaled.x < rhs.centerScaled.x;
        });
        for (size_t i = 0; i < rescued.size(); ++i)
        {
            rescued[i].name = makeCellName(frameStem, static_cast<int>(i + 1));
            rescued[i].zForCsv = rescued[i].centerScaled.z / effectiveZScaling();
        }
    }

    std::cout << "[CellLumen BrightPairMidpointRescue]"
              << " enabled=1"
              << " before=" << cells.size()
              << " after=" << rescued.size()
              << " candidates=" << candidates.size()
              << " added=" << added
              << " max_added=" << maxAdded
              << " min_distance=" << minDistance
              << " max_distance=" << maxDistance
              << " min_signal=" << minSignal
              << " min_existing_distance=" << minExistingDistance
              << " sort_by_existing_distance=" << sortByExistingDistance
              << " rejected_near_existing=" << rejectedNearExisting
              << " rejected_near_added=" << rejectedNearAdded
              << std::endl;

    return rescued;
}

std::vector<CellLumen::DetectedCell> CellLumen::addClusterCentroidRecallCandidates(
    const std::vector<cv::Mat> &volume,
    const std::vector<DetectedCell> &cells,
    const std::string &frameStem) const
{
    if (!config.cellLumen.finalClusterCentroidRecallRescueEnabled ||
        volume.empty() ||
        cells.size() < 2)
    {
        return cells;
    }

    const int cellCount = static_cast<int>(cells.size());
    if ((config.cellLumen.finalClusterCentroidRecallRescueMinCells > 0 &&
         cellCount < config.cellLumen.finalClusterCentroidRecallRescueMinCells) ||
        (config.cellLumen.finalClusterCentroidRecallRescueMaxCells > 0 &&
         cellCount > config.cellLumen.finalClusterCentroidRecallRescueMaxCells))
    {
        std::cout << "[CellLumen ClusterCentroidRecallRescue]"
                  << " enabled=1"
                  << " skipped=count_gate"
                  << " cells=" << cellCount
                  << " min_cells=" << config.cellLumen.finalClusterCentroidRecallRescueMinCells
                  << " max_cells=" << config.cellLumen.finalClusterCentroidRecallRescueMaxCells
                  << std::endl;
        return cells;
    }

    std::vector<DetectedCell> annotated = cells;
    for (DetectedCell &cell : annotated)
    {
        annotateSignalStats(volume, cell);
    }

    const float clusterDistance =
        std::max(1.0f, config.cellLumen.finalClusterCentroidRecallRescueClusterDistance);
    const float clusterDistanceSq = clusterDistance * clusterDistance;
    const int minClusterSize =
        std::max(2, config.cellLumen.finalClusterCentroidRecallRescueMinClusterSize);
    const int maxAdded =
        std::max(0, config.cellLumen.finalClusterCentroidRecallRescueMaxAdded);
    const float minAddedDistance =
        std::max(0.0f, config.cellLumen.finalClusterCentroidRecallRescueMinAddedDistance);
    const float minAddedDistanceSq = minAddedDistance * minAddedDistance;
    const float minSignal =
        config.cellLumen.finalClusterCentroidRecallRescueMinTop10MinusShell;
    const float radiusScale =
        clampf(config.cellLumen.finalClusterCentroidRecallRescueRadiusScale, 0.25f, 1.25f);

    std::vector<int> parent(annotated.size());
    for (size_t i = 0; i < parent.size(); ++i)
    {
        parent[i] = static_cast<int>(i);
    }
    for (size_t i = 0; i < annotated.size(); ++i)
    {
        for (size_t j = i + 1; j < annotated.size(); ++j)
        {
            if (squaredDistance(annotated[i].centerScaled,
                                annotated[j].centerScaled) <= clusterDistanceSq)
            {
                unionRoots(parent, static_cast<int>(i), static_cast<int>(j));
            }
        }
    }

    std::map<int, std::vector<int>> groups;
    for (size_t i = 0; i < annotated.size(); ++i)
    {
        groups[findRoot(parent, static_cast<int>(i))].push_back(static_cast<int>(i));
    }

    struct RecallCandidate
    {
        DetectedCell cell;
        int members = 0;
        float score = 0.0f;
    };

    std::vector<RecallCandidate> candidates;
    int rejectedSmallCluster = 0;
    int rejectedWeakSignal = 0;
    for (const auto &[root, members] : groups)
    {
        (void)root;
        if (static_cast<int>(members.size()) < minClusterSize)
        {
            rejectedSmallCluster++;
            continue;
        }

        const DetectedCell *templateCell = &annotated[static_cast<size_t>(members.front())];
        for (int index : members)
        {
            const DetectedCell &cell = annotated[static_cast<size_t>(index)];
            if (cell.top10MinusShell > templateCell->top10MinusShell ||
                (std::abs(cell.top10MinusShell - templateCell->top10MinusShell) < 1e-3f &&
                 cell.voxelCount > templateCell->voxelCount))
            {
                templateCell = &cell;
            }
        }

        DetectedCell centroid = *templateCell;
        double totalWeight = 0.0;
        double weightedX = 0.0;
        double weightedY = 0.0;
        double weightedZ = 0.0;
        double weightedMean = 0.0;
        double totalVoxels = 0.0;
        int x0 = std::numeric_limits<int>::max();
        int y0 = std::numeric_limits<int>::max();
        int z0 = std::numeric_limits<int>::max();
        int x1 = std::numeric_limits<int>::min();
        int y1 = std::numeric_limits<int>::min();
        int z1 = std::numeric_limits<int>::min();
        float maxMajor = 0.0f;
        float maxB = 0.0f;
        float maxMinor = 0.0f;
        float maxSignal = -std::numeric_limits<float>::max();
        for (int index : members)
        {
            const DetectedCell &cell = annotated[static_cast<size_t>(index)];
            const double signalWeight =
                static_cast<double>(std::max(1.0f, cell.top10MinusShell));
            const double voxelWeight =
                static_cast<double>(std::max(1, cell.voxelCount));
            const double weight = signalWeight * std::sqrt(voxelWeight);
            totalWeight += weight;
            weightedX += weight * static_cast<double>(cell.centerScaled.x);
            weightedY += weight * static_cast<double>(cell.centerScaled.y);
            weightedZ += weight * static_cast<double>(cell.centerScaled.z);
            weightedMean += weight * static_cast<double>(cell.meanIntensity);
            totalVoxels += voxelWeight;
            x0 = std::min(x0, cell.component.x0);
            y0 = std::min(y0, cell.component.y0);
            z0 = std::min(z0, cell.component.z0);
            x1 = std::max(x1, cell.component.x1);
            y1 = std::max(y1, cell.component.y1);
            z1 = std::max(z1, cell.component.z1);
            maxMajor = std::max(maxMajor, cell.majorRadius);
            maxB = std::max(maxB, cell.bRadius);
            maxMinor = std::max(maxMinor, cell.minorRadius);
            maxSignal = std::max(maxSignal, cell.top10MinusShell);
        }
        if (totalWeight <= 0.0)
        {
            rejectedWeakSignal++;
            continue;
        }

        centroid.centerScaled.x = static_cast<float>(weightedX / totalWeight);
        centroid.centerScaled.y = static_cast<float>(weightedY / totalWeight);
        centroid.centerScaled.z = static_cast<float>(weightedZ / totalWeight);
        centroid.zForCsv = centroid.centerScaled.z / effectiveZScaling();
        centroid.majorRadius = std::max(1.0f, maxMajor * radiusScale);
        centroid.bRadius = std::max(1.0f, maxB * radiusScale);
        centroid.minorRadius = std::max(1.0f, maxMinor * radiusScale);
        centroid.voxelCount = std::max(1, static_cast<int>(std::lround(totalVoxels /
                                                                       static_cast<double>(members.size()))));
        centroid.meanIntensity = static_cast<float>(weightedMean / totalWeight);
        centroid.component.x0 = x0;
        centroid.component.y0 = y0;
        centroid.component.z0 = z0;
        centroid.component.x1 = x1;
        centroid.component.y1 = y1;
        centroid.component.z1 = z1;
        centroid.component.vox = centroid.voxelCount;
        centroid.component.sumW = static_cast<double>(centroid.voxelCount);
        centroid.component.sx = static_cast<double>(centroid.centerScaled.x) * centroid.component.sumW;
        centroid.component.sy = static_cast<double>(centroid.centerScaled.y) * centroid.component.sumW;
        centroid.component.sz = static_cast<double>(centroid.zForCsv) * centroid.component.sumW;
        centroid.component.sumI = static_cast<double>(centroid.meanIntensity) * centroid.component.sumW;
        annotateSignalStats(volume, centroid);

        if (minSignal >= 0.0f && centroid.top10MinusShell < minSignal)
        {
            rejectedWeakSignal++;
            continue;
        }

        const float score = centroid.top10MinusShell +
                            25.0f * static_cast<float>(members.size()) +
                            0.002f * maxSignal;
        candidates.push_back({centroid, static_cast<int>(members.size()), score});
    }

    std::sort(candidates.begin(),
              candidates.end(),
              [](const RecallCandidate &lhs, const RecallCandidate &rhs) {
                  if (std::abs(lhs.score - rhs.score) > 1e-3f)
                  {
                      return lhs.score > rhs.score;
                  }
                  return lhs.members > rhs.members;
              });

    std::vector<DetectedCell> rescued = annotated;
    int added = 0;
    int rejectedNearExisting = 0;
    for (const RecallCandidate &candidate : candidates)
    {
        if (added >= maxAdded)
        {
            break;
        }
        bool nearExisting = false;
        if (minAddedDistance > 0.0f)
        {
            for (const DetectedCell &existing : rescued)
            {
                if (squaredDistance(candidate.cell.centerScaled,
                                    existing.centerScaled) <= minAddedDistanceSq)
                {
                    nearExisting = true;
                    break;
                }
            }
        }
        if (nearExisting)
        {
            rejectedNearExisting++;
            continue;
        }
        rescued.push_back(candidate.cell);
        added++;
    }

    if (added > 0)
    {
        std::sort(rescued.begin(), rescued.end(), [](const DetectedCell &lhs,
                                                     const DetectedCell &rhs) {
            if (std::abs(lhs.centerScaled.y - rhs.centerScaled.y) > 1e-3f)
            {
                return lhs.centerScaled.y < rhs.centerScaled.y;
            }
            return lhs.centerScaled.x < rhs.centerScaled.x;
        });
        for (size_t i = 0; i < rescued.size(); ++i)
        {
            rescued[i].name = makeCellName(frameStem, static_cast<int>(i + 1));
            rescued[i].zForCsv = rescued[i].centerScaled.z / effectiveZScaling();
        }
    }

    std::cout << "[CellLumen ClusterCentroidRecallRescue]"
              << " enabled=1"
              << " before=" << cells.size()
              << " after=" << rescued.size()
              << " groups=" << groups.size()
              << " candidates=" << candidates.size()
              << " added=" << added
              << " max_added=" << maxAdded
              << " cluster_distance=" << clusterDistance
              << " min_cluster_size=" << minClusterSize
              << " min_added_distance=" << minAddedDistance
              << " min_signal=" << minSignal
              << " rejected_small_cluster=" << rejectedSmallCluster
              << " rejected_weak_signal=" << rejectedWeakSignal
              << " rejected_near_existing=" << rejectedNearExisting
              << std::endl;

    return rescued;
}

float CellLumen::computeInitialPriorClusterLinkDistance(float fallbackDistance) const
{
    if (!config.cellLumen.initialPriorClusterCollapseEnabled ||
        config.cellLumen.initialPriorCsvPath.empty())
    {
        return fallbackDistance;
    }

    // The initial CSV is a valid runtime prior for Cell Universe. Here we use
    // it only to estimate the embryo's early cell spacing scale, then collapse
    // nearby CellLumen fragments that are much closer than that scale. This
    // avoids hard-coded frame numbers and does not use any current-frame GT.
    try
    {
        const float radiusScale = config.cell ? config.cell->initialRadiusScale : 1.0f;
        const std::vector<InitialCellRecord> records = CsvHandler::loadInitialCells(
            config.cellLumen.initialPriorCsvPath,
            "",
            effectiveZScaling(),
            radiusScale,
            config.simulation.initial_z_space);

        std::vector<cv::Point3f> centers;
        centers.reserve(records.size());
        for (const InitialCellRecord &record : records)
        {
            if (record.isTrash)
            {
                continue;
            }
            centers.emplace_back(record.x, record.y, record.z);
        }

        if (centers.size() < 2)
        {
            std::cout << "[CellLumen InitialPriorClusterScale]"
                      << " enabled=1 skipped=too_few_initial_cells"
                      << " path=" << config.cellLumen.initialPriorCsvPath
                      << " cells=" << centers.size()
                      << " fallback_link_distance=" << fallbackDistance
                      << std::endl;
            return fallbackDistance;
        }

        std::vector<float> nearestDistances;
        nearestDistances.reserve(centers.size());
        for (size_t i = 0; i < centers.size(); ++i)
        {
            float nearest = std::numeric_limits<float>::max();
            for (size_t j = 0; j < centers.size(); ++j)
            {
                if (i == j)
                {
                    continue;
                }
                nearest = std::min(nearest, std::sqrt(squaredDistance(centers[i], centers[j])));
            }
            if (std::isfinite(nearest))
            {
                nearestDistances.push_back(nearest);
            }
        }

        if (nearestDistances.empty())
        {
            return fallbackDistance;
        }

        std::sort(nearestDistances.begin(), nearestDistances.end());
        const float medianNearest =
            nearestDistances[nearestDistances.size() / 2];
        const float scaledLink =
            medianNearest * config.cellLumen.initialPriorClusterCollapseLinkScale;
        const float minLink =
            std::max(1.0f, config.cellLumen.initialPriorClusterCollapseMinLinkDistance);
        const float maxLink =
            std::max(minLink, config.cellLumen.initialPriorClusterCollapseMaxLinkDistance);
        const float linkDistance = clampf(scaledLink, minLink, maxLink);

        std::cout << "[CellLumen InitialPriorClusterScale]"
                  << " enabled=1"
                  << " initial_cells=" << centers.size()
                  << " median_initial_nn=" << medianNearest
                  << " link_scale=" << config.cellLumen.initialPriorClusterCollapseLinkScale
                  << " link_distance=" << linkDistance
                  << " min_link=" << minLink
                  << " max_link=" << maxLink
                  << " path=" << config.cellLumen.initialPriorCsvPath
                  << std::endl;

        return linkDistance;
    }
    catch (const std::exception &e)
    {
        std::cout << "[CellLumen InitialPriorClusterScale]"
                  << " enabled=1 skipped=parse_error"
                  << " path=" << config.cellLumen.initialPriorCsvPath
                  << " error=\"" << e.what() << "\""
                  << " fallback_link_distance=" << fallbackDistance
                  << std::endl;
        return fallbackDistance;
    }
}

std::optional<CellLumen::ClusterDensityShape> CellLumen::measureClusterDensityShape(
    const std::vector<cv::Mat> &volume,
    const std::vector<DetectedCell> &cells,
    const std::vector<int> &members) const
{
    if (volume.empty() || members.size() < 2)
    {
        return std::nullopt;
    }

    const int Z = static_cast<int>(volume.size());
    const int Y = volume.front().rows;
    const int X = volume.front().cols;
    if (Z <= 0 || Y <= 0 || X <= 0)
    {
        return std::nullopt;
    }

    const int pad = std::max(
        0,
        static_cast<int>(std::lround(
            std::max(0.0f, config.cellLumen.initialPriorClusterCollapseDensityPadding))));
    int x0 = X - 1;
    int y0 = Y - 1;
    int z0 = Z - 1;
    int x1 = 0;
    int y1 = 0;
    int z1 = 0;
    int firstIndex = members.front();
    int secondIndex = members.back();
    float memberSpan = 0.0f;
    for (size_t a = 0; a < members.size(); ++a)
    {
        const DetectedCell &lhs = cells[static_cast<size_t>(members[a])];
        x0 = std::min(x0, lhs.component.x0);
        y0 = std::min(y0, lhs.component.y0);
        z0 = std::min(z0, lhs.component.z0);
        x1 = std::max(x1, lhs.component.x1);
        y1 = std::max(y1, lhs.component.y1);
        z1 = std::max(z1, lhs.component.z1);
        for (size_t b = a + 1; b < members.size(); ++b)
        {
            const DetectedCell &rhs = cells[static_cast<size_t>(members[b])];
            const float distance = std::sqrt(squaredDistance(lhs.centerScaled, rhs.centerScaled));
            if (distance > memberSpan)
            {
                memberSpan = distance;
                firstIndex = members[a];
                secondIndex = members[b];
            }
        }
    }

    x0 = std::clamp(x0 - pad, 0, X - 1);
    y0 = std::clamp(y0 - pad, 0, Y - 1);
    z0 = std::clamp(z0 - std::max(1, pad / 4), 0, Z - 1);
    x1 = std::clamp(x1 + pad, 0, X - 1);
    y1 = std::clamp(y1 + pad, 0, Y - 1);
    z1 = std::clamp(z1 + std::max(1, pad / 4), 0, Z - 1);
    if (x1 < x0 || y1 < y0 || z1 < z0)
    {
        return std::nullopt;
    }

    std::vector<float> localValues;
    localValues.reserve(static_cast<size_t>(x1 - x0 + 1) *
                        static_cast<size_t>(y1 - y0 + 1) *
                        static_cast<size_t>(z1 - z0 + 1));
    for (int z = z0; z <= z1; ++z)
    {
        const cv::Mat &slice = volume[static_cast<size_t>(z)];
        for (int y = y0; y <= y1; ++y)
        {
            const float *row = slice.ptr<float>(y);
            for (int x = x0; x <= x1; ++x)
            {
                const float value = row[x];
                if (std::isfinite(value))
                {
                    localValues.push_back(value);
                }
            }
        }
    }
    if (localValues.empty())
    {
        return std::nullopt;
    }

    const float quantile = clampf(
        config.cellLumen.initialPriorClusterCollapseDensityQuantile,
        0.05f,
        0.95f);
    const float localThreshold = percentileFromValues(localValues, quantile);
    const float threshold = config.cellLumen.initialPriorClusterCollapseDensityUseFrameThreshold
        ? stackPercentile(
              volume,
              clampf(config.cellLumen.initialPriorClusterCollapseDensityFrameQuantile,
                     0.50f,
                     0.995f),
              true)
        : localThreshold;
    const float zScale = effectiveZScaling();

    double totalWeight = 0.0;
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    double sx2 = 0.0;
    double sy2 = 0.0;
    double sz2 = 0.0;
    int supportVoxels = 0;

    const cv::Point3f &a = cells[static_cast<size_t>(firstIndex)].centerScaled;
    const cv::Point3f &b = cells[static_cast<size_t>(secondIndex)].centerScaled;
    cv::Point3f axis(b.x - a.x, b.y - a.y, b.z - a.z);
    const float axisLength = std::max(1e-3f, std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z));
    axis.x /= axisLength;
    axis.y /= axisLength;
    axis.z /= axisLength;
    const auto project = [&](float x, float y, float zScaled) {
        return (x - a.x) * axis.x + (y - a.y) * axis.y + (zScaled - a.z) * axis.z;
    };
    const float endpointA = project(a.x, a.y, a.z);
    const float endpointB = project(b.x, b.y, b.z);
    const float projMin = std::min(endpointA, endpointB);
    const float projMax = std::max(endpointA, endpointB);
    const float projRange = std::max(1.0f, projMax - projMin);
    std::array<double, 17> projectionBins{};
    std::array<double, 17> supportBins{};

    for (int z = z0; z <= z1; ++z)
    {
        const cv::Mat &slice = volume[static_cast<size_t>(z)];
        const float zScaled = static_cast<float>(z) * zScale;
        for (int y = y0; y <= y1; ++y)
        {
            const float *row = slice.ptr<float>(y);
            for (int x = x0; x <= x1; ++x)
            {
                const float value = row[x];
                if (!std::isfinite(value) || value < threshold)
                {
                    continue;
                }
                const double weight = std::max(1e-3f, value - threshold + 1.0f);
                totalWeight += weight;
                sx += weight * static_cast<double>(x);
                sy += weight * static_cast<double>(y);
                sz += weight * static_cast<double>(zScaled);
                sx2 += weight * static_cast<double>(x) * static_cast<double>(x);
                sy2 += weight * static_cast<double>(y) * static_cast<double>(y);
                sz2 += weight * static_cast<double>(zScaled) * static_cast<double>(zScaled);
                supportVoxels++;

                const float t = std::clamp((project(static_cast<float>(x),
                                                    static_cast<float>(y),
                                                    zScaled) - projMin) / projRange,
                                           0.0f,
                                           0.999f);
                const int bin = static_cast<int>(t * static_cast<float>(projectionBins.size()));
                projectionBins[static_cast<size_t>(bin)] += weight;
                supportBins[static_cast<size_t>(bin)] += 1.0;
            }
        }
    }

    if (supportVoxels < std::max(1, config.cellLumen.initialPriorClusterCollapseDensityMinVoxels) ||
        totalWeight <= 0.0)
    {
        return std::nullopt;
    }

    ClusterDensityShape stats;
    stats.centroidScaled = cv::Point3f(
        static_cast<float>(sx / totalWeight),
        static_cast<float>(sy / totalWeight),
        static_cast<float>(sz / totalWeight));
    stats.supportVoxels = supportVoxels;
    stats.threshold = threshold;
    stats.memberSpan = memberSpan;
    const auto weightedStd = [totalWeight](double sum, double sumSquares) {
        const double mean = sum / totalWeight;
        const double variance = std::max(0.0, sumSquares / totalWeight - mean * mean);
        return static_cast<float>(std::sqrt(variance));
    };
    const float radiusSigmaScale = std::max(
        0.25f,
        config.cellLumen.initialPriorClusterCollapseDensityRadiusSigmaScale);
    // This is the PCA-like part of the density shape model. A local-peak
    // candidate can be only a tiny fragment, but the above-threshold support
    // cloud gives a better estimate of the full visible cell body.
    stats.radiusX = radiusSigmaScale * weightedStd(sx, sx2);
    stats.radiusY = radiusSigmaScale * weightedStd(sy, sy2);
    stats.radiusZ = radiusSigmaScale * weightedStd(sz, sz2);

    const auto peakInRange = [&](int begin, int end) {
        double peak = 0.0;
        int peakIndex = begin;
        for (int i = begin; i <= end; ++i)
        {
            if (projectionBins[static_cast<size_t>(i)] > peak)
            {
                peak = projectionBins[static_cast<size_t>(i)];
                peakIndex = i;
            }
        }
        return std::make_pair(peak, peakIndex);
    };
    const auto [leftPeak, leftIndex] = peakInRange(0, 7);
    const auto [rightPeak, rightIndex] = peakInRange(9, 16);
    const double minPeak = std::min(leftPeak, rightPeak);
    const double leftSupportPeak = supportBins[static_cast<size_t>(leftIndex)];
    const double rightSupportPeak = supportBins[static_cast<size_t>(rightIndex)];
    const double minSupportPeak = std::min(leftSupportPeak, rightSupportPeak);
    double valley = minPeak;
    double valleySupport = minSupportPeak;
    if (leftIndex + 1 < rightIndex)
    {
        for (int i = leftIndex + 1; i < rightIndex; ++i)
        {
            valley = std::min(valley, projectionBins[static_cast<size_t>(i)]);
            valleySupport = std::min(valleySupport, supportBins[static_cast<size_t>(i)]);
        }
    }
    stats.valleyRatio = static_cast<float>(valley / std::max(1e-6, minPeak));
    stats.valleySupportRatio =
        static_cast<float>(valleySupport / std::max(1e-6, minSupportPeak));
    stats.valleyDrop = static_cast<float>(minPeak - valley);
    stats.twoLobeGuard =
        memberSpan >= std::max(0.0f, config.cellLumen.initialPriorClusterCollapseDensityTwoLobeMinDistance) &&
        stats.valleyRatio <= clampf(
            config.cellLumen.initialPriorClusterCollapseDensityTwoLobeMaxValleyRatio,
            0.05f,
            0.99f) &&
        stats.valleySupportRatio <= clampf(
            config.cellLumen.initialPriorClusterCollapseDensityTwoLobeMaxSupportRatio,
            0.01f,
            0.99f) &&
        stats.valleyDrop >= std::max(
            0.0f,
            config.cellLumen.initialPriorClusterCollapseDensityTwoLobeMinDrop);

    return stats;
}

std::vector<CellLumen::DetectedCell> CellLumen::collapseClusteredCandidates(
    const std::vector<cv::Mat> &volume,
    const std::vector<DetectedCell> &cells,
    const std::string &frameStem) const
{
    // Keep the old fixed-distance collapse available. The initial-prior mode is
    // default-off and is meant for early sparse frames where one large nucleus
    // often contains several local brightness maxima.
    const bool initialPriorMode =
        config.cellLumen.initialPriorClusterCollapseEnabled;
    if ((!config.cellLumen.finalClusterCentroidCollapseEnabled && !initialPriorMode) ||
        cells.size() < 2)
    {
        return cells;
    }

    const int cellCount = static_cast<int>(cells.size());
    const int minCells = initialPriorMode
        ? config.cellLumen.initialPriorClusterCollapseMinCells
        : config.cellLumen.finalClusterCentroidCollapseMinCells;
    const int maxCells = initialPriorMode
        ? config.cellLumen.initialPriorClusterCollapseMaxCells
        : config.cellLumen.finalClusterCentroidCollapseMaxCells;
    if ((minCells > 0 && cellCount < minCells) ||
        (maxCells > 0 && cellCount > maxCells))
    {
        std::cout << "[CellLumen ClusterCentroidCollapse]"
                  << " enabled=1"
                  << " skipped=count_gate"
                  << " mode=" << (initialPriorMode ? "initial_prior" : "fixed")
                  << " cells=" << cellCount
                  << " min_cells=" << minCells
                  << " max_cells=" << maxCells
                  << std::endl;
        return cells;
    }

    std::vector<DetectedCell> annotated = cells;
    if (!volume.empty())
    {
        for (DetectedCell &cell : annotated)
        {
            annotateSignalStats(volume, cell);
        }
    }

    const float configuredLinkDistance =
        std::max(1.0f, config.cellLumen.finalClusterCentroidCollapseLinkDistance);
    const float linkDistance = initialPriorMode
        ? computeInitialPriorClusterLinkDistance(configuredLinkDistance)
        : configuredLinkDistance;
    const float linkDistanceSq = linkDistance * linkDistance;
    const int minClusterSize = std::max(
        2,
        initialPriorMode
            ? config.cellLumen.initialPriorClusterCollapseMinClusterSize
            : config.cellLumen.finalClusterCentroidCollapseMinClusterSize);
    const float radiusScale = clampf(
        initialPriorMode
            ? config.cellLumen.initialPriorClusterCollapseRadiusScale
            : config.cellLumen.finalClusterCentroidCollapseRadiusScale,
        0.25f,
        1.50f);
    const bool useSignalWeights = initialPriorMode
        ? config.cellLumen.initialPriorClusterCollapseUseSignalWeights
        : config.cellLumen.finalClusterCentroidCollapseUseSignalWeights;
    const float maxGroupDiameter = initialPriorMode
        ? config.cellLumen.initialPriorClusterCollapseMaxGroupDiameter
        : -1.0f;
    const bool valleyGuardEnabled =
        initialPriorMode &&
        config.cellLumen.initialPriorClusterCollapseValleyGuardEnabled &&
        !volume.empty();
    const float valleyMaxQ20Ratio = clampf(
        config.cellLumen.initialPriorClusterCollapseValleyMaxQ20Ratio,
        0.05f,
        0.99f);
    const float valleyMinDrop =
        std::max(0.0f, config.cellLumen.initialPriorClusterCollapseValleyMinDrop);
    const bool densityShapeEnabled =
        initialPriorMode &&
        config.cellLumen.initialPriorClusterCollapseDensityShapeEnabled &&
        !volume.empty();
    const bool densityCentroidEnabled =
        densityShapeEnabled &&
        config.cellLumen.initialPriorClusterCollapseDensityCentroidEnabled;
    const bool densityTwoLobeGuardEnabled =
        densityShapeEnabled &&
        config.cellLumen.initialPriorClusterCollapseDensityTwoLobeGuardEnabled;
    const bool densityMomentRadiiEnabled =
        densityShapeEnabled &&
        config.cellLumen.initialPriorClusterCollapseDensityMomentRadiiEnabled;
    const bool ambiguousAddbackEnabled =
        initialPriorMode &&
        config.cellLumen.initialPriorClusterCollapseAmbiguousAddbackEnabled &&
        (config.cellLumen.initialPriorClusterCollapseAmbiguousAddbackMinCells <= 0 ||
         cellCount >= config.cellLumen.initialPriorClusterCollapseAmbiguousAddbackMinCells) &&
        (config.cellLumen.initialPriorClusterCollapseAmbiguousAddbackMaxCells <= 0 ||
         cellCount <= config.cellLumen.initialPriorClusterCollapseAmbiguousAddbackMaxCells);
    const float zScale = effectiveZScaling();
    auto hasBlockingValley = [&](const DetectedCell &lhs, const DetectedCell &rhs) {
        const cv::Point3f a(lhs.centerScaled.x, lhs.centerScaled.y, lhs.zForCsv);
        const cv::Point3f b(rhs.centerScaled.x, rhs.centerScaled.y, rhs.zForCsv);
        const auto profile = sampleLineProfile(volume, a, b, zScale);
        if (!profile)
        {
            return false;
        }
        const float edgeLevel = std::max(
            1.0f,
            std::min(lhs.meanIntensity, rhs.meanIntensity));
        const float q20Ratio = profile->q20 / edgeLevel;
        const float valleyDrop = edgeLevel - profile->q20;
        return q20Ratio <= valleyMaxQ20Ratio && valleyDrop >= valleyMinDrop;
    };
    int valleyGuardedPairs = 0;

    std::vector<int> parent(annotated.size());
    for (size_t i = 0; i < parent.size(); ++i)
    {
        parent[i] = static_cast<int>(i);
    }
    for (size_t i = 0; i < annotated.size(); ++i)
    {
        for (size_t j = i + 1; j < annotated.size(); ++j)
        {
            if (squaredDistance(annotated[i].centerScaled,
                                annotated[j].centerScaled) <= linkDistanceSq)
            {
                if (valleyGuardEnabled &&
                    hasBlockingValley(annotated[i], annotated[j]))
                {
                    valleyGuardedPairs++;
                    continue;
                }
                unionRoots(parent, static_cast<int>(i), static_cast<int>(j));
            }
        }
    }

    std::map<int, std::vector<int>> groups;
    for (size_t i = 0; i < annotated.size(); ++i)
    {
        groups[findRoot(parent, static_cast<int>(i))].push_back(static_cast<int>(i));
    }

    if (initialPriorMode &&
        config.cellLumen.initialPriorClusterCollapseSkipAboveGroupCount > 0 &&
        static_cast<int>(groups.size()) >
            config.cellLumen.initialPriorClusterCollapseSkipAboveGroupCount)
    {
        // Raw candidate count confuses early over-segmentation with real
        // mid-density frames. Group count is measured after the same local-link
        // clustering collapse would use, so it better reflects whether the
        // frame already contains many separated bodies. In that case, keeping
        // high-recall candidates is safer than compressing real neighbors.
        std::cout << "[CellLumen ClusterCentroidCollapse]"
                  << " enabled=1"
                  << " skipped=group_count_gate"
                  << " mode=initial_prior"
                  << " cells=" << cellCount
                  << " groups=" << groups.size()
                  << " max_groups="
                  << config.cellLumen.initialPriorClusterCollapseSkipAboveGroupCount
                  << " link_distance=" << linkDistance
                  << " valley_guarded_pairs=" << valleyGuardedPairs
                  << std::endl;
        return annotated;
    }

    const bool diameterSkipWindow =
        initialPriorMode &&
        maxGroupDiameter > 0.0f &&
        config.cellLumen.initialPriorClusterCollapseSkipDiameterGuardMinCells > 0 &&
        cellCount >= config.cellLumen.initialPriorClusterCollapseSkipDiameterGuardMinCells &&
        (config.cellLumen.initialPriorClusterCollapseSkipDiameterGuardMaxCells <= 0 ||
         cellCount <= config.cellLumen.initialPriorClusterCollapseSkipDiameterGuardMaxCells);
    if (diameterSkipWindow)
    {
        bool hasOversizedGroup = false;
        float largestGroupDiameter = 0.0f;
        for (const auto &[root, members] : groups)
        {
            (void)root;
            if (static_cast<int>(members.size()) < minClusterSize)
            {
                continue;
            }
            float groupDiameter = 0.0f;
            for (size_t a = 0; a < members.size(); ++a)
            {
                const DetectedCell &lhs = annotated[static_cast<size_t>(members[a])];
                for (size_t b = a + 1; b < members.size(); ++b)
                {
                    const DetectedCell &rhs = annotated[static_cast<size_t>(members[b])];
                    groupDiameter = std::max(
                        groupDiameter,
                        std::sqrt(squaredDistance(lhs.centerScaled, rhs.centerScaled)));
                }
            }
            largestGroupDiameter = std::max(largestGroupDiameter, groupDiameter);
            if (groupDiameter > maxGroupDiameter)
            {
                hasOversizedGroup = true;
            }
        }
        if (hasOversizedGroup)
        {
            // At moderate density, an oversized linked group usually means the
            // single-link chain crossed a real cell boundary. Keeping the
            // high-recall candidates is safer than collapsing several nearby
            // cells into a shifted centroid.
            std::cout << "[CellLumen ClusterCentroidCollapse]"
                      << " enabled=1"
                      << " skipped=diameter_group_gate"
                      << " mode=initial_prior"
                      << " cells=" << cellCount
                      << " groups=" << groups.size()
                      << " largest_group_diameter=" << largestGroupDiameter
                      << " max_group_diameter=" << maxGroupDiameter
                      << " min_cells="
                      << config.cellLumen.initialPriorClusterCollapseSkipDiameterGuardMinCells
                      << " max_cells="
                      << config.cellLumen.initialPriorClusterCollapseSkipDiameterGuardMaxCells
                      << " link_distance=" << linkDistance
                      << " valley_guarded_pairs=" << valleyGuardedPairs
                      << std::endl;
            return annotated;
        }
    }

    std::vector<DetectedCell> collapsed;
    collapsed.reserve(groups.size());
    int mergedGroups = 0;
    int mergedCells = 0;
    int singletonGroups = 0;
    int tooSmallGroups = 0;
    int diameterGuardedGroups = 0;
    int diameterGuardedCells = 0;
    int densityGuardedGroups = 0;
    int densityGuardedCells = 0;
    int densityCentroidGroups = 0;
    int densityMomentRadiiGroups = 0;
    int densityMeasuredGroups = 0;
    int ambiguousAddbackGroups = 0;
    int ambiguousAddbackCells = 0;
    int ambiguousAddbackRejectedNear = 0;

    for (const auto &[root, members] : groups)
    {
        (void)root;
        if (static_cast<int>(members.size()) < minClusterSize)
        {
            if (members.size() == 1)
            {
                singletonGroups++;
            }
            else
            {
                tooSmallGroups++;
            }
            for (int index : members)
            {
                collapsed.push_back(annotated[static_cast<size_t>(index)]);
            }
            continue;
        }

        float groupDiameter = 0.0f;
        for (size_t a = 0; a < members.size(); ++a)
        {
            const DetectedCell &lhs = annotated[static_cast<size_t>(members[a])];
            for (size_t b = a + 1; b < members.size(); ++b)
            {
                const DetectedCell &rhs = annotated[static_cast<size_t>(members[b])];
                groupDiameter = std::max(
                    groupDiameter,
                    std::sqrt(squaredDistance(lhs.centerScaled, rhs.centerScaled)));
            }
        }

        if (maxGroupDiameter > 0.0f)
        {
            if (groupDiameter > maxGroupDiameter)
            {
                // Single-link clustering can connect two real cells through a
                // chain of internal fragments. When the group diameter is too
                // large, keeping the original candidates is safer for Cell
                // Universe than collapsing them into one midpoint-like miss.
                diameterGuardedGroups++;
                diameterGuardedCells += static_cast<int>(members.size());
                for (int index : members)
                {
                    collapsed.push_back(annotated[static_cast<size_t>(index)]);
                }
                continue;
            }
        }

        std::optional<ClusterDensityShape> densityShape;
        if (densityShapeEnabled)
        {
            densityShape = measureClusterDensityShape(volume, annotated, members);
            if (densityShape)
            {
                densityMeasuredGroups++;
            }
            if (densityTwoLobeGuardEnabled &&
                densityShape &&
                densityShape->twoLobeGuard)
            {
                // A human reviewer separates close daughters when there is a
                // real low-intensity saddle between two bright lobes. This
                // default-off guard gives CellLumen the same group-level check:
                // keep the original candidates instead of collapsing a true
                // two-lobe density body into one midpoint.
                densityGuardedGroups++;
                densityGuardedCells += static_cast<int>(members.size());
                for (int index : members)
                {
                    collapsed.push_back(annotated[static_cast<size_t>(index)]);
                }
                continue;
            }
        }

        const DetectedCell *templateCell = &annotated[static_cast<size_t>(members.front())];
        for (int index : members)
        {
            const DetectedCell &cell = annotated[static_cast<size_t>(index)];
            if (cell.top10MinusShell > templateCell->top10MinusShell ||
                (std::abs(cell.top10MinusShell - templateCell->top10MinusShell) < 1e-3f &&
                 cell.voxelCount > templateCell->voxelCount))
            {
                templateCell = &cell;
            }
        }

        DetectedCell centroid = *templateCell;
        double totalWeight = 0.0;
        double weightedX = 0.0;
        double weightedY = 0.0;
        double weightedZ = 0.0;
        double weightedMean = 0.0;
        double weightedMajor = 0.0;
        double weightedB = 0.0;
        double weightedMinor = 0.0;
        double weightedVoxels = 0.0;
        int x0 = std::numeric_limits<int>::max();
        int y0 = std::numeric_limits<int>::max();
        int z0 = std::numeric_limits<int>::max();
        int x1 = std::numeric_limits<int>::min();
        int y1 = std::numeric_limits<int>::min();
        int z1 = std::numeric_limits<int>::min();
        for (int index : members)
        {
            const DetectedCell &cell = annotated[static_cast<size_t>(index)];
            const double signalWeight =
                static_cast<double>(std::max(1.0f, cell.top10MinusShell));
            const double weight = useSignalWeights ? signalWeight : 1.0;
            totalWeight += weight;
            weightedX += weight * static_cast<double>(cell.centerScaled.x);
            weightedY += weight * static_cast<double>(cell.centerScaled.y);
            weightedZ += weight * static_cast<double>(cell.centerScaled.z);
            weightedMean += weight * static_cast<double>(cell.meanIntensity);
            weightedMajor += weight * static_cast<double>(cell.majorRadius);
            weightedB += weight * static_cast<double>(cell.bRadius);
            weightedMinor += weight * static_cast<double>(cell.minorRadius);
            weightedVoxels += weight * static_cast<double>(std::max(1, cell.voxelCount));
            x0 = std::min(x0, cell.component.x0);
            y0 = std::min(y0, cell.component.y0);
            z0 = std::min(z0, cell.component.z0);
            x1 = std::max(x1, cell.component.x1);
            y1 = std::max(y1, cell.component.y1);
            z1 = std::max(z1, cell.component.z1);
        }
        if (totalWeight <= 0.0)
        {
            for (int index : members)
            {
                collapsed.push_back(annotated[static_cast<size_t>(index)]);
            }
            continue;
        }

        centroid.centerScaled.x = static_cast<float>(weightedX / totalWeight);
        centroid.centerScaled.y = static_cast<float>(weightedY / totalWeight);
        centroid.centerScaled.z = static_cast<float>(weightedZ / totalWeight);
        if (densityCentroidEnabled && densityShape)
        {
            // Distance-only candidate averaging follows the internal seed
            // locations. For large early cells, the brightness-density centroid
            // is closer to how the PCA shape fitter sees one continuous cell.
            centroid.centerScaled = densityShape->centroidScaled;
            densityCentroidGroups++;
        }
        centroid.zForCsv = centroid.centerScaled.z / effectiveZScaling();
        centroid.majorRadius = std::max(1.0f, static_cast<float>(weightedMajor / totalWeight) * radiusScale);
        centroid.bRadius = std::max(1.0f, static_cast<float>(weightedB / totalWeight) * radiusScale);
        centroid.minorRadius = std::max(1.0f, static_cast<float>(weightedMinor / totalWeight) * radiusScale);
        if (densityMomentRadiiEnabled && densityShape &&
            densityShape->radiusX > 0.0f &&
            densityShape->radiusY > 0.0f &&
            densityShape->radiusZ > 0.0f)
        {
            // A group made of several seed fragments should inherit the shape
            // of the full bright-density support, not the average size of the
            // fragments. This stays image-only and is controlled by YAML.
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
            const float outputMinMajor = config.cellLumen.minOutputMajorRadius > 0.0f
                ? config.cellLumen.minOutputMajorRadius
                : minMajor;
            const float outputMinB = config.cellLumen.minOutputBRadius > 0.0f
                ? config.cellLumen.minOutputBRadius
                : minB;
            const float outputMinMinor = config.cellLumen.minOutputMinorRadius > 0.0f
                ? config.cellLumen.minOutputMinorRadius
                : minMinor;
            const float outputMaxMajor = config.cellLumen.maxOutputMajorRadius > 0.0f
                ? config.cellLumen.maxOutputMajorRadius
                : maxMajor;
            const float outputMaxB = config.cellLumen.maxOutputBRadius > 0.0f
                ? config.cellLumen.maxOutputBRadius
                : maxB;
            const float outputMaxMinor = config.cellLumen.maxOutputMinorRadius > 0.0f
                ? config.cellLumen.maxOutputMinorRadius
                : maxMinor;
            const float xyMajor = std::max(densityShape->radiusX, densityShape->radiusY);
            const float xyMinor = std::min(densityShape->radiusX, densityShape->radiusY);
            centroid.majorRadius = clampf(radiusScale * xyMajor, outputMinMajor, outputMaxMajor);
            centroid.bRadius = clampf(radiusScale * xyMinor, outputMinB, std::min(outputMaxB, centroid.majorRadius));
            centroid.minorRadius = clampf(
                radiusScale * densityShape->radiusZ,
                outputMinMinor,
                std::min(outputMaxMinor, centroid.majorRadius));
            densityMomentRadiiGroups++;
        }
        centroid.voxelCount = std::max(1, static_cast<int>(std::lround(weightedVoxels / totalWeight)));
        centroid.meanIntensity = static_cast<float>(weightedMean / totalWeight);
        centroid.component.x0 = x0;
        centroid.component.y0 = y0;
        centroid.component.z0 = z0;
        centroid.component.x1 = x1;
        centroid.component.y1 = y1;
        centroid.component.z1 = z1;
        centroid.component.vox = centroid.voxelCount;
        centroid.component.sumW = static_cast<double>(centroid.voxelCount);
        centroid.component.sx = static_cast<double>(centroid.centerScaled.x) * centroid.component.sumW;
        centroid.component.sy = static_cast<double>(centroid.centerScaled.y) * centroid.component.sumW;
        centroid.component.sz = static_cast<double>(centroid.zForCsv) * centroid.component.sumW;
        centroid.component.sumI = static_cast<double>(centroid.meanIntensity) * centroid.component.sumW;
        if (!volume.empty())
        {
            annotateSignalStats(volume, centroid);
        }
        collapsed.push_back(centroid);
        mergedGroups++;
        mergedCells += static_cast<int>(members.size());

        if (ambiguousAddbackEnabled &&
            ambiguousAddbackCells < std::max(0, config.cellLumen.initialPriorClusterCollapseAmbiguousAddbackMaxAdded) &&
            groupDiameter >= std::max(0.0f, config.cellLumen.initialPriorClusterCollapseAmbiguousAddbackMinGroupDiameter))
        {
            struct AddbackCandidate
            {
                float distanceFromCentroid = 0.0f;
                int index = 0;
            };
            std::vector<AddbackCandidate> addbackCandidates;
            addbackCandidates.reserve(members.size());
            for (int index : members)
            {
                const DetectedCell &member = annotated[static_cast<size_t>(index)];
                if (config.cellLumen.initialPriorClusterCollapseAmbiguousAddbackMinTop10MinusShell >= 0.0f &&
                    member.top10MinusShell <
                        config.cellLumen.initialPriorClusterCollapseAmbiguousAddbackMinTop10MinusShell)
                {
                    continue;
                }
                const float distanceFromCentroid =
                    std::sqrt(squaredDistance(member.centerScaled, centroid.centerScaled));
                if (distanceFromCentroid <
                    std::max(0.0f, config.cellLumen.initialPriorClusterCollapseAmbiguousAddbackMinCentroidDistance))
                {
                    continue;
                }
                addbackCandidates.push_back(AddbackCandidate{distanceFromCentroid, index});
            }
            std::sort(addbackCandidates.begin(),
                      addbackCandidates.end(),
                      [&](const AddbackCandidate &lhs, const AddbackCandidate &rhs) {
                          const DetectedCell &lhsCell = annotated[static_cast<size_t>(lhs.index)];
                          const DetectedCell &rhsCell = annotated[static_cast<size_t>(rhs.index)];
                          if (std::abs(lhs.distanceFromCentroid - rhs.distanceFromCentroid) > 1e-3f)
                          {
                              return lhs.distanceFromCentroid > rhs.distanceFromCentroid;
                          }
                          if (std::abs(lhsCell.top10MinusShell - rhsCell.top10MinusShell) > 1e-3f)
                          {
                              return lhsCell.top10MinusShell > rhsCell.top10MinusShell;
                          }
                          return lhsCell.voxelCount > rhsCell.voxelCount;
                      });

            int addedForGroup = 0;
            const int maxPerGroup = std::max(1, config.cellLumen.initialPriorClusterCollapseAmbiguousAddbackMaxPerGroup);
            const int maxTotal = std::max(0, config.cellLumen.initialPriorClusterCollapseAmbiguousAddbackMaxAdded);
            const float minAddDistance = std::max(
                0.0f,
                config.cellLumen.initialPriorClusterCollapseAmbiguousAddbackMinCentroidDistance);
            const float minAddDistanceSq = minAddDistance * minAddDistance;
            for (const AddbackCandidate &candidate : addbackCandidates)
            {
                if (addedForGroup >= maxPerGroup || ambiguousAddbackCells >= maxTotal)
                {
                    break;
                }
                const DetectedCell &member = annotated[static_cast<size_t>(candidate.index)];
                bool nearExisting = false;
                for (size_t i = 0; i + 1 < collapsed.size(); ++i)
                {
                    if (squaredDistance(member.centerScaled, collapsed[i].centerScaled) <= minAddDistanceSq)
                    {
                        nearExisting = true;
                        break;
                    }
                }
                if (nearExisting)
                {
                    ambiguousAddbackRejectedNear++;
                    continue;
                }
                collapsed.push_back(member);
                addedForGroup++;
                ambiguousAddbackCells++;
            }
            if (addedForGroup > 0)
            {
                ambiguousAddbackGroups++;
            }
        }
    }

    if (collapsed.size() != cells.size())
    {
        std::sort(collapsed.begin(), collapsed.end(), [](const DetectedCell &lhs,
                                                         const DetectedCell &rhs) {
            if (std::abs(lhs.centerScaled.y - rhs.centerScaled.y) > 1e-3f)
            {
                return lhs.centerScaled.y < rhs.centerScaled.y;
            }
            return lhs.centerScaled.x < rhs.centerScaled.x;
        });
        for (size_t i = 0; i < collapsed.size(); ++i)
        {
            collapsed[i].name = makeCellName(frameStem, static_cast<int>(i + 1));
            collapsed[i].zForCsv = collapsed[i].centerScaled.z / effectiveZScaling();
        }
    }

    std::cout << "[CellLumen ClusterCentroidCollapse]"
              << " enabled=1"
              << " mode=" << (initialPriorMode ? "initial_prior" : "fixed")
              << " before=" << cells.size()
              << " after=" << collapsed.size()
              << " groups=" << groups.size()
              << " merged_groups=" << mergedGroups
              << " merged_cells=" << mergedCells
              << " singleton_groups=" << singletonGroups
              << " too_small_groups=" << tooSmallGroups
              << " diameter_guarded_groups=" << diameterGuardedGroups
              << " diameter_guarded_cells=" << diameterGuardedCells
              << " density_shape=" << densityShapeEnabled
              << " density_measured_groups=" << densityMeasuredGroups
              << " density_guarded_groups=" << densityGuardedGroups
              << " density_guarded_cells=" << densityGuardedCells
              << " density_centroid_groups=" << densityCentroidGroups
              << " density_moment_radii_groups=" << densityMomentRadiiGroups
              << " ambiguous_addback=" << ambiguousAddbackEnabled
              << " ambiguous_addback_groups=" << ambiguousAddbackGroups
              << " ambiguous_addback_cells=" << ambiguousAddbackCells
              << " ambiguous_addback_rejected_near=" << ambiguousAddbackRejectedNear
              << " valley_guard=" << valleyGuardEnabled
              << " valley_guarded_pairs=" << valleyGuardedPairs
              << " link_distance=" << linkDistance
              << " max_group_diameter=" << maxGroupDiameter
              << " min_cluster_size=" << minClusterSize
              << " radius_scale=" << radiusScale
              << " signal_weights=" << useSignalWeights
              << std::endl;

    return collapsed;
}

std::vector<CellLumen::DetectedCell> CellLumen::filterDominatedDuplicateCandidates(
    const std::vector<cv::Mat> &volume,
    const std::vector<DetectedCell> &cells,
    const std::string &frameStem) const
{
    (void)frameStem;
    if (!config.cellLumen.finalDominatedDuplicateFilterEnabled ||
        cells.size() < 2 ||
        volume.empty())
    {
        return cells;
    }

    const int cellCount = static_cast<int>(cells.size());
    if ((config.cellLumen.finalDominatedDuplicateFilterMinCells > 0 &&
         cellCount < config.cellLumen.finalDominatedDuplicateFilterMinCells) ||
        (config.cellLumen.finalDominatedDuplicateFilterMaxCells > 0 &&
         cellCount > config.cellLumen.finalDominatedDuplicateFilterMaxCells))
    {
        std::cout << "[CellLumen DominatedDuplicateFilter]"
                  << " enabled=1"
                  << " skipped=count_gate"
                  << " cells=" << cellCount
                  << " min_cells=" << config.cellLumen.finalDominatedDuplicateFilterMinCells
                  << " max_cells=" << config.cellLumen.finalDominatedDuplicateFilterMaxCells
                  << std::endl;
        return cells;
    }

    std::vector<DetectedCell> annotated = cells;
    for (DetectedCell &cell : annotated)
    {
        annotateSignalStats(volume, cell);
    }

    const float distance =
        std::max(0.0f, config.cellLumen.finalDominatedDuplicateFilterDistance);
    const float distanceSq = distance * distance;
    const float voxelRatio =
        std::max(1.0f, config.cellLumen.finalDominatedDuplicateFilterMinVoxelRatio);
    const float signalRatio =
        std::max(1.0f, config.cellLumen.finalDominatedDuplicateFilterMinSignalRatio);
    const float radiusRatio =
        std::max(1.0f, config.cellLumen.finalDominatedDuplicateFilterMinRadiusRatio);
    const float maxLoserSignal =
        config.cellLumen.finalDominatedDuplicateFilterMaxLoserTop10MinusShell;
    const bool requireTwoSignals =
        config.cellLumen.finalDominatedDuplicateFilterRequireTwoSignals;

    std::vector<bool> reject(annotated.size(), false);
    int rejected = 0;
    int rejectedVoxelSignal = 0;
    int rejectedVoxelRadius = 0;
    int rejectedSignalRadius = 0;
    for (size_t i = 0; i < annotated.size(); ++i)
    {
        const DetectedCell &cell = annotated[i];
        const bool loserSignalAllowed =
            maxLoserSignal < 0.0f || cell.top10MinusShell <= maxLoserSignal;
        if (!loserSignalAllowed)
        {
            continue;
        }

        for (size_t j = 0; j < annotated.size(); ++j)
        {
            if (i == j)
            {
                continue;
            }
            const DetectedCell &neighbor = annotated[j];
            if (squaredDistance(cell.centerScaled, neighbor.centerScaled) > distanceSq)
            {
                continue;
            }

            const bool voxelDominated =
                static_cast<float>(neighbor.voxelCount) >=
                voxelRatio * static_cast<float>(std::max(1, cell.voxelCount));
            const bool signalDominated =
                neighbor.top10MinusShell >=
                    signalRatio * std::max(1.0f, cell.top10MinusShell) &&
                neighbor.top10MinusShell > cell.top10MinusShell + 5.0f;
            const bool radiusDominated =
                neighbor.majorRadius >= radiusRatio * std::max(1.0f, cell.majorRadius);
            const int evidenceCount =
                static_cast<int>(voxelDominated) +
                static_cast<int>(signalDominated) +
                static_cast<int>(radiusDominated);
            if (evidenceCount < (requireTwoSignals ? 2 : 1))
            {
                continue;
            }

            reject[i] = true;
            rejected++;
            if (voxelDominated && signalDominated)
            {
                rejectedVoxelSignal++;
            }
            else if (voxelDominated && radiusDominated)
            {
                rejectedVoxelRadius++;
            }
            else if (signalDominated && radiusDominated)
            {
                rejectedSignalRadius++;
            }
            break;
        }
    }

    if (rejected == 0)
    {
        std::cout << "[CellLumen DominatedDuplicateFilter]"
                  << " enabled=1"
                  << " before=" << cells.size()
                  << " after=" << cells.size()
                  << " rejected=0"
                  << " distance=" << distance
                  << std::endl;
        return annotated;
    }

    std::vector<DetectedCell> filtered;
    filtered.reserve(annotated.size() - static_cast<size_t>(rejected));
    for (size_t i = 0; i < annotated.size(); ++i)
    {
        if (!reject[i])
        {
            filtered.push_back(annotated[i]);
        }
    }
    std::sort(filtered.begin(), filtered.end(), [](const DetectedCell &lhs,
                                                   const DetectedCell &rhs) {
        if (std::abs(lhs.centerScaled.y - rhs.centerScaled.y) > 1e-3f)
        {
            return lhs.centerScaled.y < rhs.centerScaled.y;
        }
        return lhs.centerScaled.x < rhs.centerScaled.x;
    });
    for (size_t i = 0; i < filtered.size(); ++i)
    {
        filtered[i].name = makeCellName(frameStem, static_cast<int>(i + 1));
        filtered[i].zForCsv = filtered[i].centerScaled.z / effectiveZScaling();
    }

    std::cout << "[CellLumen DominatedDuplicateFilter]"
              << " enabled=1"
              << " before=" << cells.size()
              << " after=" << filtered.size()
              << " rejected=" << rejected
              << " distance=" << distance
              << " voxel_ratio=" << voxelRatio
              << " signal_ratio=" << signalRatio
              << " radius_ratio=" << radiusRatio
              << " max_loser_signal=" << maxLoserSignal
              << " rejected_voxel_signal=" << rejectedVoxelSignal
              << " rejected_voxel_radius=" << rejectedVoxelRadius
              << " rejected_signal_radius=" << rejectedSignalRadius
              << std::endl;

    return filtered;
}

std::vector<CellLumen::DetectedCell> CellLumen::filterLowDensityArtifacts(
    const std::vector<DetectedCell> &cells,
    const std::string &frameStem) const
{
    if (!config.cellLumen.finalLowDensityArtifactFilterEnabled ||
        cells.size() < 2)
    {
        return cells;
    }

    const int cellCount = static_cast<int>(cells.size());
    if ((config.cellLumen.finalLowDensityArtifactFilterMinCells > 0 &&
         cellCount < config.cellLumen.finalLowDensityArtifactFilterMinCells) ||
        (config.cellLumen.finalLowDensityArtifactFilterMaxCells > 0 &&
         cellCount > config.cellLumen.finalLowDensityArtifactFilterMaxCells))
    {
        std::cout << "[CellLumen LowDensityArtifactFilter]"
                  << " enabled=1 skipped=count_gate"
                  << " cells=" << cellCount
                  << " min_cells=" << config.cellLumen.finalLowDensityArtifactFilterMinCells
                  << " max_cells=" << config.cellLumen.finalLowDensityArtifactFilterMaxCells
                  << std::endl;
        return cells;
    }

    std::vector<float> meanValues;
    std::vector<float> voxelValues;
    std::vector<float> majorValues;
    std::vector<float> minorValues;
    meanValues.reserve(cells.size());
    voxelValues.reserve(cells.size());
    majorValues.reserve(cells.size());
    minorValues.reserve(cells.size());
    for (const DetectedCell &cell : cells)
    {
        if (std::isfinite(cell.meanIntensity) && cell.meanIntensity > 0.0f)
        {
            meanValues.push_back(cell.meanIntensity);
        }
        if (cell.voxelCount > 0)
        {
            voxelValues.push_back(static_cast<float>(cell.voxelCount));
        }
        if (std::isfinite(cell.majorRadius) && cell.majorRadius > 0.0f)
        {
            majorValues.push_back(cell.majorRadius);
        }
        if (std::isfinite(cell.minorRadius) && cell.minorRadius > 0.0f)
        {
            minorValues.push_back(cell.minorRadius);
        }
    }

    const auto medianOrZero = [](std::vector<float> values) -> float {
        if (values.empty())
        {
            return 0.0f;
        }
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };

    const float medianMean = medianOrZero(meanValues);
    const float medianVoxels = medianOrZero(voxelValues);
    const float medianMajor = medianOrZero(majorValues);
    const float medianMinor = medianOrZero(minorValues);
    if (medianMean <= 0.0f || medianVoxels <= 0.0f)
    {
        std::cout << "[CellLumen LowDensityArtifactFilter]"
                  << " enabled=1 skipped=missing_reference_stats"
                  << " cells=" << cellCount
                  << " median_mean=" << medianMean
                  << " median_voxels=" << medianVoxels
                  << std::endl;
        return cells;
    }

    const float maxMeanRatio =
        std::max(0.0f, config.cellLumen.finalLowDensityArtifactMaxMeanRatio);
    const float maxVoxelRatio =
        std::max(0.0f, config.cellLumen.finalLowDensityArtifactMaxVoxelRatio);
    const float minNearest =
        std::max(0.0f, config.cellLumen.finalLowDensityArtifactMinNearestDistance);
    const float maxMajorRatio =
        config.cellLumen.finalLowDensityArtifactMaxMajorRadiusRatio;
    const float maxMinorRatio =
        config.cellLumen.finalLowDensityArtifactMaxMinorRadiusRatio;

    std::vector<DetectedCell> filtered;
    filtered.reserve(cells.size());
    int rejected = 0;
    for (size_t i = 0; i < cells.size(); ++i)
    {
        const DetectedCell &cell = cells[i];
        float nearestDistance = std::numeric_limits<float>::max();
        for (size_t j = 0; j < cells.size(); ++j)
        {
            if (i == j)
            {
                continue;
            }
            nearestDistance = std::min(
                nearestDistance,
                std::sqrt(squaredDistance(cell.centerScaled, cells[j].centerScaled)));
        }

        const float meanRatio = cell.meanIntensity / medianMean;
        const float voxelRatio =
            static_cast<float>(std::max(1, cell.voxelCount)) / medianVoxels;
        const bool lowMean = meanRatio <= maxMeanRatio;
        const bool lowVoxelSupport = voxelRatio <= maxVoxelRatio;
        const bool isolated = nearestDistance >= minNearest;
        const bool smallMajorEnough =
            maxMajorRatio <= 0.0f ||
            (medianMajor > 0.0f && cell.majorRadius / medianMajor <= maxMajorRatio);
        const bool smallMinorEnough =
            maxMinorRatio <= 0.0f ||
            (medianMinor > 0.0f && cell.minorRadius / medianMinor <= maxMinorRatio);

        if (lowMean && lowVoxelSupport && isolated &&
            smallMajorEnough && smallMinorEnough)
        {
            rejected++;
            continue;
        }
        filtered.push_back(cell);
    }

    if (rejected > 0)
    {
        std::sort(filtered.begin(), filtered.end(), [](const DetectedCell &lhs,
                                                       const DetectedCell &rhs) {
            if (std::abs(lhs.centerScaled.y - rhs.centerScaled.y) > 1e-3f)
            {
                return lhs.centerScaled.y < rhs.centerScaled.y;
            }
            return lhs.centerScaled.x < rhs.centerScaled.x;
        });
        for (size_t i = 0; i < filtered.size(); ++i)
        {
            filtered[i].name = makeCellName(frameStem, static_cast<int>(i + 1));
            filtered[i].zForCsv = filtered[i].centerScaled.z / effectiveZScaling();
        }
    }

    // This filter targets persistent floating dust spots. Those artifacts can
    // look radius-like after PCA inflation, so the decision uses relative mean
    // intensity and component support plus isolation from the biological field.
    std::cout << "[CellLumen LowDensityArtifactFilter]"
              << " enabled=1"
              << " before=" << cells.size()
              << " after=" << filtered.size()
              << " rejected=" << rejected
              << " median_mean=" << medianMean
              << " median_voxels=" << medianVoxels
              << " max_mean_ratio=" << maxMeanRatio
              << " max_voxel_ratio=" << maxVoxelRatio
              << " min_nearest_distance=" << minNearest
              << " max_major_ratio=" << maxMajorRatio
              << " max_minor_ratio=" << maxMinorRatio
              << std::endl;

    return filtered;
}

float CellLumen::scoreCandidateCells(const std::vector<DetectedCell> &cells,
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
        config.cellLumen.maxCandidateMeanVoxelCount > 0.0f && meanVoxelCount > config.cellLumen.maxCandidateMeanVoxelCount
            ? (meanVoxelCount - config.cellLumen.maxCandidateMeanVoxelCount) /
                  std::max(1.0f, config.cellLumen.maxCandidateMeanVoxelCount)
            : 0.0f;
    const float cellCountWeight = config.cellLumen.candidateCellCountWeight > 0.0f
                                      ? config.cellLumen.candidateCellCountWeight
                                      : 0.16f;
    const float meanVoxelPenaltyWeight = config.cellLumen.candidateMeanVoxelPenaltyWeight > 0.0f
                                             ? config.cellLumen.candidateMeanVoxelPenaltyWeight
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
           - config.cellLumen.candidateNearPairPenaltyWeight * static_cast<float>(nearDuplicatePairs)
           - 1.75f * static_cast<float>(flattenedCount)
           - 1.25f * static_cast<float>(tinyFragmentCount)
           - meanVoxelPenaltyWeight * meanVoxelPenalty;
}

std::vector<CellLumen::DetectedCell> CellLumen::detectCellsInVolume(
    const std::vector<cv::Mat> &volume,
    const std::string &frameStem)
{
    const std::vector<float> percentiles = cellLumenPercentiles(config.cellLumen);

    std::vector<DetectedCell> bestCells;
    float bestScore = -std::numeric_limits<float>::max();
    std::vector<DetectedCell> watershedRescueCandidates;
    if (config.cellLumen.seededWatershedEnabled)
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
            std::cout << "[CellLumen Candidate] method=seeded_watershed"
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

            if (config.cellLumen.seededWatershedPreferOverPercentile)
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
        score += config.cellLumen.candidatePercentilePreferenceWeight * (percentile - 99.0f);
        const bool protectEarlySparseFrame =
            config.cellLumen.candidateOvergrowthAlwaysBelowFirstCount > 0 &&
            firstCandidateCellCount > 0 &&
            firstCandidateCellCount <= config.cellLumen.candidateOvergrowthAlwaysBelowFirstCount;
        const bool protectNormalFrame =
            config.cellLumen.candidateOvergrowthMinFirstCount <= 0 ||
            firstCandidateCellCount >= config.cellLumen.candidateOvergrowthMinFirstCount;
        if (config.cellLumen.maxCandidateCountGrowthFromFirst > 1.0f &&
            config.cellLumen.candidateOvergrowthPenaltyWeight > 0.0f &&
            firstCandidateCellCount > 0 &&
            (protectEarlySparseFrame || protectNormalFrame))
        {
            const float growth = static_cast<float>(cells.size()) /
                                 static_cast<float>(std::max(1, firstCandidateCellCount));
            if (growth > config.cellLumen.maxCandidateCountGrowthFromFirst)
            {
                const float excess = growth - config.cellLumen.maxCandidateCountGrowthFromFirst;
                score -= config.cellLumen.candidateOvergrowthPenaltyWeight *
                         excess * excess *
                         static_cast<float>(firstCandidateCellCount);
            }
        }

        std::cout << "[CellLumen Candidate] percentile=" << percentile
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
        throw std::runtime_error("CellLumen found no valid connected components after preprocessing.");
    }

    if (!watershedRescueCandidates.empty() &&
        config.cellLumen.seededWatershedPreferBaseCountMin > 0 &&
        config.cellLumen.seededWatershedPreferBaseCountMax >= config.cellLumen.seededWatershedPreferBaseCountMin)
    {
        const int baseCount = static_cast<int>(bestCells.size());
        const bool baseCountInRange =
            baseCount >= config.cellLumen.seededWatershedPreferBaseCountMin &&
            baseCount <= config.cellLumen.seededWatershedPreferBaseCountMax;
        if (baseCountInRange)
        {
            std::cout << "[CellLumen SeededWatershed DensitySwitch]"
                      << " base_cells=" << baseCount
                      << " watershed_cells=" << watershedRescueCandidates.size()
                      << " min_base_count=" << config.cellLumen.seededWatershedPreferBaseCountMin
                      << " max_base_count=" << config.cellLumen.seededWatershedPreferBaseCountMax
                      << " selected=seeded_watershed"
                      << std::endl;
            bestCells = watershedRescueCandidates;
        }
        else
        {
            std::cout << "[CellLumen SeededWatershed DensitySwitch]"
                      << " base_cells=" << baseCount
                      << " watershed_cells=" << watershedRescueCandidates.size()
                      << " min_base_count=" << config.cellLumen.seededWatershedPreferBaseCountMin
                      << " max_base_count=" << config.cellLumen.seededWatershedPreferBaseCountMax
                      << " selected=percentile_cc"
                      << std::endl;
        }
    }

    if (config.cellLumen.seededWatershedRescueEnabled && !watershedRescueCandidates.empty())
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

        const float rescueDistance = std::max(1.0f, config.cellLumen.seededWatershedRescueMinDistance);
        const float rescueDistanceSq = rescueDistance * rescueDistance;
        const int fractionCap = config.cellLumen.seededWatershedRescueMaxAddedFraction > 0.0f
            ? static_cast<int>(std::ceil(config.cellLumen.seededWatershedRescueMaxAddedFraction *
                                         static_cast<float>(bestCells.size())))
            : std::numeric_limits<int>::max();
        const int absoluteCap = config.cellLumen.seededWatershedRescueMaxAdded > 0
            ? config.cellLumen.seededWatershedRescueMaxAdded
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

            if (config.cellLumen.minTop10MinusShell > 0.0f &&
                candidate.top10MinusShell < config.cellLumen.minTop10MinusShell)
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

        std::cout << "[CellLumen SeededWatershed Rescue]"
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

    const int lowRescueBaseCellCount = static_cast<int>(bestCells.size());
    const bool lowRescueBaseCountAllowed =
        (config.cellLumen.seededWatershedLowRescueMinBaseCells <= 0 ||
         lowRescueBaseCellCount >= config.cellLumen.seededWatershedLowRescueMinBaseCells) &&
        (config.cellLumen.seededWatershedLowRescueMaxBaseCells <= 0 ||
         lowRescueBaseCellCount <= config.cellLumen.seededWatershedLowRescueMaxBaseCells);
    if (config.cellLumen.seededWatershedLowRescueEnabled && !lowRescueBaseCountAllowed)
    {
        std::cout << "[CellLumen SeededWatershed LowRescue]"
                  << " skipped=base_count_gate"
                  << " base_cells=" << lowRescueBaseCellCount
                  << " min_base_cells=" << config.cellLumen.seededWatershedLowRescueMinBaseCells
                  << " max_base_cells=" << config.cellLumen.seededWatershedLowRescueMaxBaseCells
                  << std::endl;
    }
    if (config.cellLumen.seededWatershedLowRescueEnabled && lowRescueBaseCountAllowed)
    {
        const CellLumenConfig savedCellLumenConfig = config.cellLumen;
        config.cellLumen.seededWatershedSeedPercentile =
            config.cellLumen.seededWatershedLowRescueSeedPercentile;
        config.cellLumen.seededWatershedMinVoxels =
            config.cellLumen.seededWatershedLowRescueMinVoxels;
        if (savedCellLumenConfig.seededWatershedLowRescueMinSeedDistance > 0.0f)
        {
            config.cellLumen.seededWatershedMinSeedDistance =
                savedCellLumenConfig.seededWatershedLowRescueMinSeedDistance;
        }
        if (savedCellLumenConfig.seededWatershedLowRescueCloseSeedMinDistance > 0.0f)
        {
            config.cellLumen.seededWatershedCloseSeedMinDistance =
                savedCellLumenConfig.seededWatershedLowRescueCloseSeedMinDistance;
        }
        if (savedCellLumenConfig.seededWatershedLowRescueMinValleyDrop >= 0.0f)
        {
            config.cellLumen.seededWatershedMinValleyDrop =
                savedCellLumenConfig.seededWatershedLowRescueMinValleyDrop;
        }

        std::vector<DetectedCell> lowRescueCandidates =
            detectCellsBySeededWatershed(volume, frameStem);
        config.cellLumen = savedCellLumenConfig;
        const size_t rawLowRescueCandidateCount = lowRescueCandidates.size();

        int refinedCenters = 0;
        if (config.cellLumen.seededWatershedLowRescueRefineEnabled &&
            !lowRescueCandidates.empty())
        {
            const float refineDistance =
                std::max(1.0f, config.cellLumen.seededWatershedLowRescueRefineDistance);
            const float refineDistanceSq = refineDistance * refineDistance;
            for (DetectedCell &cell : bestCells)
            {
                double totalWeight = 0.0;
                double sx = 0.0;
                double sy = 0.0;
                double sz = 0.0;
                int contributors = 0;
                for (const DetectedCell &candidate : lowRescueCandidates)
                {
                    if (squaredDistance(cell.centerScaled, candidate.centerScaled) > refineDistanceSq)
                    {
                        continue;
                    }
                    const double weight = static_cast<double>(
                        std::max(1, candidate.voxelCount));
                    totalWeight += weight;
                    sx += weight * static_cast<double>(candidate.centerScaled.x);
                    sy += weight * static_cast<double>(candidate.centerScaled.y);
                    sz += weight * static_cast<double>(candidate.centerScaled.z);
                    contributors++;
                }

                if (contributors > 0 && totalWeight > 0.0)
                {
                    cell.centerScaled.x = static_cast<float>(sx / totalWeight);
                    cell.centerScaled.y = static_cast<float>(sy / totalWeight);
                    cell.centerScaled.z = static_cast<float>(sz / totalWeight);
                    cell.zForCsv = cell.centerScaled.z / effectiveZScaling();
                    refinedCenters++;
                }
            }
        }

        int clusteredLowRescueCandidates = 0;
        if (config.cellLumen.seededWatershedLowRescueClusterDistance > 0.0f &&
            !lowRescueCandidates.empty())
        {
            const float clusterDistance = config.cellLumen.seededWatershedLowRescueClusterDistance;
            const float clusterDistanceSq = clusterDistance * clusterDistance;
            const int minCandidateVoxels =
                std::max(1, config.cellLumen.seededWatershedLowRescueMinCandidateVoxels);
            const int softMinCandidateVoxels = std::max(1, minCandidateVoxels / 2);
            const float minSignal = config.cellLumen.seededWatershedLowRescueMinTop10MinusShell;
            const float softMinSignal = minSignal - 8.0f;

            std::vector<DetectedCell> eligible;
            eligible.reserve(lowRescueCandidates.size());
            for (const DetectedCell &candidate : lowRescueCandidates)
            {
                if (candidate.voxelCount < softMinCandidateVoxels)
                {
                    continue;
                }
                if (candidate.top10MinusShell < softMinSignal)
                {
                    continue;
                }
                eligible.push_back(candidate);
            }

            std::vector<char> visited(eligible.size(), 0);
            std::vector<DetectedCell> clustered;
            clustered.reserve(eligible.size());
            for (size_t start = 0; start < eligible.size(); ++start)
            {
                if (visited[start])
                {
                    continue;
                }

                std::vector<size_t> members;
                std::queue<size_t> queue;
                visited[start] = 1;
                queue.push(start);
                while (!queue.empty())
                {
                    const size_t index = queue.front();
                    queue.pop();
                    members.push_back(index);

                    for (size_t other = 0; other < eligible.size(); ++other)
                    {
                        if (visited[other])
                        {
                            continue;
                        }
                        if (squaredDistance(eligible[index].centerScaled,
                                            eligible[other].centerScaled) <= clusterDistanceSq)
                        {
                            visited[other] = 1;
                            queue.push(other);
                        }
                    }
                }

                DetectedCell merged = eligible[members.front()];
                double totalWeight = 0.0;
                double sx = 0.0;
                double sy = 0.0;
                double sz = 0.0;
                double meanSum = 0.0;
                double shellSum = 0.0;
                int shellVoxels = 0;
                int totalVoxels = 0;
                merged.component.x0 = std::numeric_limits<int>::max();
                merged.component.y0 = std::numeric_limits<int>::max();
                merged.component.z0 = std::numeric_limits<int>::max();
                merged.component.x1 = std::numeric_limits<int>::min();
                merged.component.y1 = std::numeric_limits<int>::min();
                merged.component.z1 = std::numeric_limits<int>::min();
                merged.majorRadius = 0.0f;
                merged.bRadius = 0.0f;
                merged.minorRadius = 0.0f;
                merged.top10Intensity = 0.0f;
                merged.top10MinusShell = std::numeric_limits<float>::lowest();

                for (size_t member : members)
                {
                    const DetectedCell &part = eligible[member];
                    const int weight = std::max(1, part.voxelCount);
                    totalWeight += static_cast<double>(weight);
                    sx += static_cast<double>(weight) * part.centerScaled.x;
                    sy += static_cast<double>(weight) * part.centerScaled.y;
                    sz += static_cast<double>(weight) * part.centerScaled.z;
                    meanSum += static_cast<double>(weight) * part.meanIntensity;
                    shellSum += static_cast<double>(std::max(0, part.shellVoxelCount)) *
                                part.shellMeanIntensity;
                    shellVoxels += std::max(0, part.shellVoxelCount);
                    totalVoxels += part.voxelCount;
                    merged.component.x0 = std::min(merged.component.x0, part.component.x0);
                    merged.component.y0 = std::min(merged.component.y0, part.component.y0);
                    merged.component.z0 = std::min(merged.component.z0, part.component.z0);
                    merged.component.x1 = std::max(merged.component.x1, part.component.x1);
                    merged.component.y1 = std::max(merged.component.y1, part.component.y1);
                    merged.component.z1 = std::max(merged.component.z1, part.component.z1);
                    merged.majorRadius = std::max(merged.majorRadius, part.majorRadius);
                    merged.bRadius = std::max(merged.bRadius, part.bRadius);
                    merged.minorRadius = std::max(merged.minorRadius, part.minorRadius);
                    merged.top10Intensity = std::max(merged.top10Intensity, part.top10Intensity);
                    merged.top10MinusShell = std::max(merged.top10MinusShell, part.top10MinusShell);
                }

                if (totalWeight <= 0.0)
                {
                    continue;
                }
                merged.centerScaled.x = static_cast<float>(sx / totalWeight);
                merged.centerScaled.y = static_cast<float>(sy / totalWeight);
                merged.centerScaled.z = static_cast<float>(sz / totalWeight);
                merged.zForCsv = merged.centerScaled.z / effectiveZScaling();
                merged.voxelCount = totalVoxels;
                merged.meanIntensity = static_cast<float>(meanSum / totalWeight);
                merged.shellVoxelCount = shellVoxels;
                if (shellVoxels > 0)
                {
                    merged.shellMeanIntensity =
                        static_cast<float>(shellSum / static_cast<double>(shellVoxels));
                }
                merged.meanMinusShell = merged.meanIntensity - merged.shellMeanIntensity;
                if (merged.top10MinusShell == std::numeric_limits<float>::lowest())
                {
                    merged.top10MinusShell = merged.top10Intensity - merged.shellMeanIntensity;
                }
                merged.component.vox = totalVoxels;
                merged.component.sumI = meanSum;
                merged.component.sumW = totalWeight;
                merged.component.sx = merged.centerScaled.x * totalWeight;
                merged.component.sy = merged.centerScaled.y * totalWeight;
                merged.component.sz = (merged.centerScaled.z / effectiveZScaling()) * totalWeight;

                clustered.push_back(merged);
            }

            clusteredLowRescueCandidates = static_cast<int>(clustered.size());
            lowRescueCandidates = std::move(clustered);
        }

        const auto minDistanceToBest = [&](const DetectedCell &candidate) {
            float bestDistanceSq = std::numeric_limits<float>::max();
            for (const DetectedCell &existing : bestCells)
            {
                bestDistanceSq = std::min(bestDistanceSq,
                                          squaredDistance(candidate.centerScaled,
                                                          existing.centerScaled));
            }
            return bestDistanceSq;
        };

        std::sort(lowRescueCandidates.begin(),
                  lowRescueCandidates.end(),
                  [&](const DetectedCell &lhs, const DetectedCell &rhs) {
                      if (config.cellLumen.seededWatershedLowRescueSortByDistance)
                      {
                          const float lhsDistanceSq = minDistanceToBest(lhs);
                          const float rhsDistanceSq = minDistanceToBest(rhs);
                          if (std::abs(lhsDistanceSq - rhsDistanceSq) > 1e-3f)
                          {
                              return lhsDistanceSq > rhsDistanceSq;
                          }
                      }
                      if (lhs.top10MinusShell != rhs.top10MinusShell)
                      {
                          return lhs.top10MinusShell > rhs.top10MinusShell;
                      }
                      return lhs.voxelCount > rhs.voxelCount;
                  });

        const float minDistance = std::max(1.0f, config.cellLumen.seededWatershedLowRescueMinDistance);
        const float minDistanceSq = minDistance * minDistance;
        const int maxAdded = std::max(0, config.cellLumen.seededWatershedLowRescueMaxAdded);
        int considered = 0;
        int added = 0;
        int rejectedNearExisting = 0;
        int rejectedSmall = 0;
        int rejectedWeak = 0;
        for (const DetectedCell &candidate : lowRescueCandidates)
        {
            if (added >= maxAdded)
            {
                break;
            }
            considered++;
            if (config.cellLumen.seededWatershedLowRescueMinCandidateVoxels > 0 &&
                candidate.voxelCount < config.cellLumen.seededWatershedLowRescueMinCandidateVoxels)
            {
                rejectedSmall++;
                continue;
            }
            if (candidate.top10MinusShell <
                config.cellLumen.seededWatershedLowRescueMinTop10MinusShell)
            {
                rejectedWeak++;
                continue;
            }

            bool nearExisting = false;
            for (const DetectedCell &existing : bestCells)
            {
                if (squaredDistance(candidate.centerScaled, existing.centerScaled) <= minDistanceSq)
                {
                    nearExisting = true;
                    break;
                }
            }
            if (nearExisting)
            {
                rejectedNearExisting++;
                continue;
            }

            bestCells.push_back(candidate);
            added++;
            std::cout << "[CellLumen SeededWatershed LowRescue Added]"
                      << " index=" << added
                      << " center=(" << candidate.centerScaled.x
                      << "," << candidate.centerScaled.y
                      << "," << candidate.centerScaled.z << ")"
                      << " voxels=" << candidate.voxelCount
                      << " top10_minus_shell=" << candidate.top10MinusShell
                      << std::endl;
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

        std::cout << "[CellLumen SeededWatershed LowRescue]"
                  << " candidates=" << rawLowRescueCandidateCount
                  << " clustered_candidates=" << clusteredLowRescueCandidates
                  << " considered=" << considered
                  << " added=" << added
                  << " max_added=" << maxAdded
                  << " min_distance=" << minDistance
                  << " cluster_distance=" << config.cellLumen.seededWatershedLowRescueClusterDistance
                  << " min_candidate_voxels=" << config.cellLumen.seededWatershedLowRescueMinCandidateVoxels
                  << " min_top10_minus_shell=" << config.cellLumen.seededWatershedLowRescueMinTop10MinusShell
                  << " refined_centers=" << refinedCenters
                  << " refine_distance=" << config.cellLumen.seededWatershedLowRescueRefineDistance
                  << " rejected_near_existing=" << rejectedNearExisting
                  << " rejected_small=" << rejectedSmall
                  << " rejected_weak=" << rejectedWeak
                  << " final_cells=" << bestCells.size()
                  << std::endl;
    }

    if (config.cellLumen.finalDuplicateMergeDistance > 0.0f &&
        bestCells.size() >= 2)
    {
        const float mergeDistance = config.cellLumen.finalDuplicateMergeDistance;
        std::vector<DetectedCell> merged;
        merged.reserve(bestCells.size());
        int mergedPairs = 0;
        for (DetectedCell cell : bestCells)
        {
            bool absorbed = false;
            for (DetectedCell &existing : merged)
            {
                if (squaredDistance(cell.centerScaled, existing.centerScaled) >
                    mergeDistance * mergeDistance)
                {
                    continue;
                }

                const float leftWeight = std::max(1.0f, static_cast<float>(existing.voxelCount));
                const float rightWeight = std::max(1.0f, static_cast<float>(cell.voxelCount));
                const float totalWeight = leftWeight + rightWeight;
                existing.centerScaled.x =
                    (existing.centerScaled.x * leftWeight + cell.centerScaled.x * rightWeight) / totalWeight;
                existing.centerScaled.y =
                    (existing.centerScaled.y * leftWeight + cell.centerScaled.y * rightWeight) / totalWeight;
                existing.centerScaled.z =
                    (existing.centerScaled.z * leftWeight + cell.centerScaled.z * rightWeight) / totalWeight;
                existing.zForCsv = existing.centerScaled.z / effectiveZScaling();
                existing.majorRadius = std::max(existing.majorRadius, cell.majorRadius);
                existing.bRadius = std::max(existing.bRadius, cell.bRadius);
                existing.minorRadius = std::max(existing.minorRadius, cell.minorRadius);
                existing.voxelCount += cell.voxelCount;
                existing.meanIntensity =
                    (existing.meanIntensity * leftWeight + cell.meanIntensity * rightWeight) / totalWeight;
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

        if (mergedPairs > 0)
        {
            bestCells = std::move(merged);
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
        }

        std::cout << "[CellLumen FinalDuplicateMerge]"
                  << " distance=" << mergeDistance
                  << " merged_pairs=" << mergedPairs
                  << " final_cells=" << bestCells.size()
                  << std::endl;
    }

    if (config.cellLumen.finalLocalRefineEnabled &&
        !bestCells.empty() &&
        !volume.empty())
    {
        const float radius = std::max(1.0f, config.cellLumen.finalLocalRefineRadius);
        const float radiusSq = radius * radius;
        const float quantile = clampf(config.cellLumen.finalLocalRefineQuantile, 0.0f, 0.99f);
        const float centerBlend = clampf(config.cellLumen.finalLocalRefineBlend, 0.0f, 1.0f);
        const float zBlend = config.cellLumen.finalLocalRefineZBlend >= 0.0f
            ? clampf(config.cellLumen.finalLocalRefineZBlend, 0.0f, 1.0f)
            : centerBlend;
        const float zScale = effectiveZScaling();
        const int Z = static_cast<int>(volume.size());
        const int Y = volume[0].rows;
        const int X = volume[0].cols;
        int refinedCenters = 0;

        for (DetectedCell &cell : bestCells)
        {
            const int minX = std::max(0, static_cast<int>(std::floor(cell.centerScaled.x - radius)));
            const int maxX = std::min(X - 1, static_cast<int>(std::ceil(cell.centerScaled.x + radius)));
            const int minY = std::max(0, static_cast<int>(std::floor(cell.centerScaled.y - radius)));
            const int maxY = std::min(Y - 1, static_cast<int>(std::ceil(cell.centerScaled.y + radius)));
            const int zRadius = std::max(1, static_cast<int>(std::ceil(radius / zScale)) + 2);
            const int centerZRaw = static_cast<int>(std::lround(cell.centerScaled.z / zScale));
            const int minZ = std::max(0, centerZRaw - zRadius);
            const int maxZ = std::min(Z - 1, centerZRaw + zRadius);

            std::vector<float> localValues;
            localValues.reserve(static_cast<size_t>(
                std::max(1, maxZ - minZ + 1) *
                std::max(1, maxY - minY + 1) *
                std::max(1, maxX - minX + 1)));
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
                        if (dx * dx + dy * dy + dz * dz <= radiusSq &&
                            std::isfinite(row[x]))
                        {
                            localValues.push_back(row[x]);
                        }
                    }
                }
            }

            if (localValues.size() < 20)
            {
                continue;
            }

            const float threshold = percentileFromValues(localValues, quantile);
            double totalWeight = 0.0;
            double sx = 0.0;
            double sy = 0.0;
            double sz = 0.0;
            int usedVoxels = 0;
            for (int z = minZ; z <= maxZ; ++z)
            {
                const float zScaled = static_cast<float>(z) * zScale;
                const float dz = zScaled - cell.centerScaled.z;
                const cv::Mat &slice = volume[static_cast<size_t>(z)];
                for (int y = minY; y <= maxY; ++y)
                {
                    const float dy = static_cast<float>(y) - cell.centerScaled.y;
                    const float *row = slice.ptr<float>(y);
                    for (int x = minX; x <= maxX; ++x)
                    {
                        const float dx = static_cast<float>(x) - cell.centerScaled.x;
                        const float value = row[x];
                        if (dx * dx + dy * dy + dz * dz > radiusSq ||
                            !std::isfinite(value) ||
                            value < threshold)
                        {
                            continue;
                        }

                        const double weight = static_cast<double>(
                            std::max(0.0f, value - threshold) + 1e-3f);
                        totalWeight += weight;
                        sx += weight * static_cast<double>(x);
                        sy += weight * static_cast<double>(y);
                        sz += weight * static_cast<double>(zScaled);
                        usedVoxels++;
                    }
                }
            }

            if (usedVoxels < 20 || totalWeight <= 0.0)
            {
                continue;
            }

            const cv::Point3f refinedCenter(static_cast<float>(sx / totalWeight),
                                             static_cast<float>(sy / totalWeight),
                                             static_cast<float>(sz / totalWeight));
            cell.centerScaled.x += centerBlend * (refinedCenter.x - cell.centerScaled.x);
            cell.centerScaled.y += centerBlend * (refinedCenter.y - cell.centerScaled.y);
            cell.centerScaled.z += zBlend * (refinedCenter.z - cell.centerScaled.z);
            cell.zForCsv = cell.centerScaled.z / effectiveZScaling();
            annotateSignalStats(volume, cell);
            refinedCenters++;
        }

        int postRefineMergedPairs = 0;
        if (config.cellLumen.finalPostRefineDuplicateMergeDistance > 0.0f &&
            bestCells.size() >= 2)
        {
            const float mergeDistance = config.cellLumen.finalPostRefineDuplicateMergeDistance;
            const float mergeDistanceSq = mergeDistance * mergeDistance;
            std::vector<DetectedCell> merged;
            merged.reserve(bestCells.size());
            for (DetectedCell cell : bestCells)
            {
                bool absorbed = false;
                for (DetectedCell &existing : merged)
                {
                    if (squaredDistance(cell.centerScaled, existing.centerScaled) > mergeDistanceSq)
                    {
                        continue;
                    }

                    const float leftWeight = std::max(1.0f, static_cast<float>(existing.voxelCount));
                    const float rightWeight = std::max(1.0f, static_cast<float>(cell.voxelCount));
                    const float totalWeight = leftWeight + rightWeight;
                    existing.centerScaled.x =
                        (existing.centerScaled.x * leftWeight + cell.centerScaled.x * rightWeight) / totalWeight;
                    existing.centerScaled.y =
                        (existing.centerScaled.y * leftWeight + cell.centerScaled.y * rightWeight) / totalWeight;
                    existing.centerScaled.z =
                        (existing.centerScaled.z * leftWeight + cell.centerScaled.z * rightWeight) / totalWeight;
                    existing.zForCsv = existing.centerScaled.z / effectiveZScaling();
                    existing.majorRadius = std::max(existing.majorRadius, cell.majorRadius);
                    existing.bRadius = std::max(existing.bRadius, cell.bRadius);
                    existing.minorRadius = std::max(existing.minorRadius, cell.minorRadius);
                    existing.voxelCount += cell.voxelCount;
                    existing.meanIntensity =
                        (existing.meanIntensity * leftWeight + cell.meanIntensity * rightWeight) / totalWeight;
                    annotateSignalStats(volume, existing);
                    absorbed = true;
                    postRefineMergedPairs++;
                    break;
                }

                if (!absorbed)
                {
                    merged.push_back(cell);
                }
            }
            bestCells = std::move(merged);
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

        std::cout << "[CellLumen FinalLocalRefine]"
                  << " enabled=1"
                  << " radius=" << radius
                  << " quantile=" << quantile
                  << " blend=" << centerBlend
                  << " z_blend=" << zBlend
                  << " refined_centers=" << refinedCenters
                  << " post_refine_merged_pairs=" << postRefineMergedPairs
                  << " final_cells=" << bestCells.size()
                  << std::endl;
    }

    if (config.cellLumen.finalZColumnRefineBeforeCollapseEnabled)
    {
        refineCentersByZColumn(volume, bestCells);
    }

    const size_t beforeZPeakSplit = bestCells.size();
    if (config.cellLumen.finalZPeakSplitBeforeCollapseEnabled)
    {
        bestCells = splitCellsByLocalZPeaks(volume, bestCells, frameStem);
    }
    if (bestCells.size() != beforeZPeakSplit)
    {
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
    }

    bestCells = filterLowDensityArtifacts(bestCells, frameStem);

    if (config.cellLumen.finalSparseIsolatedFloorFilterEnabled &&
        bestCells.size() >= 2)
    {
        const int cellCount = static_cast<int>(bestCells.size());
        const bool sparseCountMode =
            config.cellLumen.finalSparseIsolatedFloorFilterMaxCells <= 0 ||
            cellCount <= config.cellLumen.finalSparseIsolatedFloorFilterMaxCells;
        const bool weakSignalMode =
            config.cellLumen.finalSparseIsolatedFloorWeakSignalMaxCells > 0 &&
            (config.cellLumen.finalSparseIsolatedFloorWeakSignalMinCells <= 0 ||
             cellCount >= config.cellLumen.finalSparseIsolatedFloorWeakSignalMinCells) &&
            cellCount <= config.cellLumen.finalSparseIsolatedFloorWeakSignalMaxCells &&
            config.cellLumen.finalSparseIsolatedFloorMaxTop10MinusShell >= 0.0f;
        const float maxMajor = config.cellLumen.finalSparseIsolatedFloorMaxMajorRadius;
        const float maxMinor = config.cellLumen.finalSparseIsolatedFloorMaxMinorRadius;
        const float minNearest = std::max(0.0f, config.cellLumen.finalSparseIsolatedFloorMinNearestDistance);
        std::vector<DetectedCell> filtered;
        filtered.reserve(bestCells.size());
        int rejected = 0;
        int rejectedBySparseCount = 0;
        int rejectedByWeakSignal = 0;
        for (size_t i = 0; i < bestCells.size(); ++i)
        {
            const DetectedCell &cell = bestCells[i];
            const bool floorSized =
                cell.majorRadius <= maxMajor &&
                cell.minorRadius <= maxMinor;
            float nearestDistance = std::numeric_limits<float>::max();
            for (size_t j = 0; j < bestCells.size(); ++j)
            {
                if (i == j)
                {
                    continue;
                }
                nearestDistance = std::min(
                    nearestDistance,
                    std::sqrt(squaredDistance(cell.centerScaled, bestCells[j].centerScaled)));
            }

            const bool weakFloorSignal =
                weakSignalMode &&
                cell.top10MinusShell <= config.cellLumen.finalSparseIsolatedFloorMaxTop10MinusShell;
            const bool rejectAsSparseFloor =
                floorSized &&
                nearestDistance >= minNearest &&
                (sparseCountMode || weakFloorSignal);
            if (rejectAsSparseFloor)
            {
                rejected++;
                if (sparseCountMode)
                {
                    rejectedBySparseCount++;
                }
                else
                {
                    rejectedByWeakSignal++;
                }
                continue;
            }
            filtered.push_back(cell);
        }
        if (rejected > 0)
        {
            bestCells = std::move(filtered);
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
        }
        std::cout << "[CellLumen SparseIsolatedFloorFilter]"
                  << " enabled=1"
                  << " before=" << (bestCells.size() + static_cast<size_t>(rejected))
                  << " after=" << bestCells.size()
                  << " rejected=" << rejected
                  << " max_cells=" << config.cellLumen.finalSparseIsolatedFloorFilterMaxCells
                  << " weak_signal_min_cells=" << config.cellLumen.finalSparseIsolatedFloorWeakSignalMinCells
                  << " weak_signal_max_cells=" << config.cellLumen.finalSparseIsolatedFloorWeakSignalMaxCells
                  << " max_top10_minus_shell=" << config.cellLumen.finalSparseIsolatedFloorMaxTop10MinusShell
                  << " rejected_sparse_count=" << rejectedBySparseCount
                  << " rejected_weak_signal=" << rejectedByWeakSignal
                  << " max_major=" << maxMajor
                  << " max_minor=" << maxMinor
                  << " min_nearest_distance=" << minNearest
                  << std::endl;
    }
    const size_t beforeWeakSatelliteFilter = bestCells.size();
    bestCells = filterWeakSatelliteCells(volume, bestCells);
    if (bestCells.size() != beforeWeakSatelliteFilter)
    {
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
    }

    const size_t beforeBrightPairMidpointRescue = bestCells.size();
    bestCells = rescueBrightPairMidpoints(volume, bestCells, frameStem);
    if (bestCells.size() != beforeBrightPairMidpointRescue)
    {
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
    }

    const size_t beforeClusterCentroidRecallRescue = bestCells.size();
    bestCells = addClusterCentroidRecallCandidates(volume, bestCells, frameStem);
    if (bestCells.size() != beforeClusterCentroidRecallRescue)
    {
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
    }

    const size_t beforeClusterCentroidCollapse = bestCells.size();
    bestCells = collapseClusteredCandidates(volume, bestCells, frameStem);
    if (bestCells.size() != beforeClusterCentroidCollapse)
    {
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
    }

    if (config.cellLumen.finalPostCollapseZColumnRefineEnabled)
    {
        // Collapse can average several early bright fragments back toward a
        // biased z slice. This optional second pass corrects the final collapsed
        // centers without adding candidates or changing lineage evidence.
        refineCentersByZColumn(volume, bestCells);
    }

    if (config.cellLumen.finalPostCollapseZPeakSplitEnabled)
    {
        // A few sparse early frames contain two real centers that are close in
        // x/y but separated along z. Running a small add-only z split after
        // collapse keeps the low-extra collapse path while giving the reviewer
        // and Cell Universe one extra local candidate for the second center.
        const size_t beforePostCollapseZPeakSplit = bestCells.size();
        bestCells = splitCellsByLocalZPeaks(volume, bestCells, frameStem);
        if (bestCells.size() != beforePostCollapseZPeakSplit)
        {
            std::sort(bestCells.begin(), bestCells.end(), [](const DetectedCell &lhs,
                                                             const DetectedCell &rhs) {
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
        }
    }

    const size_t beforeZProfileRescue = bestCells.size();
    bestCells = addZProfileRescueCandidates(volume, bestCells, frameStem);
    if (bestCells.size() != beforeZProfileRescue)
    {
        std::sort(bestCells.begin(), bestCells.end(), [](const DetectedCell &lhs,
                                                         const DetectedCell &rhs) {
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
    }

    // Run the same artifact filter again after cluster recall/collapse because
    // those final stages can leave a low-density floating candidate in the
    // output even if the pre-collapse candidate list was already cleaned.
    bestCells = filterLowDensityArtifacts(bestCells, frameStem);

    const size_t beforeDominatedDuplicateFilter = bestCells.size();
    bestCells = filterDominatedDuplicateCandidates(volume, bestCells, frameStem);
    if (bestCells.size() != beforeDominatedDuplicateFilter)
    {
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
    }

    return bestCells;
}

std::vector<Ellipsoid> CellLumen::makeEllipsoids(const std::vector<DetectedCell> &cells) const
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

void CellLumen::saveInitialCsv(const fs::path &csvOutputPath,
                                            const std::string &frameFileName,
                                            const std::vector<DetectedCell> &cells) const
{
    fs::create_directories(csvOutputPath.parent_path());
    std::ofstream out(csvOutputPath);
    if (!out.is_open())
    {
        throw std::runtime_error("Failed to open output CSV: " + csvOutputPath.string());
    }

    out << "file,name,x,y,z,majorRadius,bRadius,minorRadius\n";
    out << std::setprecision(15);
    for (const auto &cell : cells)
    {
        out << frameFileName << ","
            << cell.name << ","
            << cell.centerScaled.x << ","
            << cell.centerScaled.y << ","
            << cell.zForCsv << ","
            << cell.majorRadius << ","
            << cell.bRadius << ","
            << cell.minorRadius << "\n";
    }
}

void CellLumen::saveFrameOutputs(const fs::path &imageFile,
                                              const std::vector<cv::Mat> &realFrame,
                                              const std::vector<DetectedCell> &cells) const
{
    if (shouldSkipCellLumenTiffOutput())
    {
        std::cout << "[CellLumen Output] skip_tiff=1 frame=" << imageFile.filename().string()
                  << " reason=CELLUNIVERSE_CELL_LUMEN_SKIP_TIFF"
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
    writeCellLumenTiffStack(realTiffPath, realImages);
    writeCellLumenTiffStack(synthTiffPath, synthImages);
    std::cout << "[CellLumen Output] real_tif=" << realTiffPath
              << " synth_tif=" << synthTiffPath
              << std::endl;
}

std::vector<CellLumen::DetectedCell> CellLumen::buildInitialCsvForFrame(
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
        std::cout << "[CellLumen Timing] frame=" << imageFile.filename().string()
                  << " stage=" << stage
                  << " stage_sec=" << std::fixed << std::setprecision(6) << stageSeconds
                  << " total_sec=" << totalSeconds
                  << std::defaultfloat
                  << std::endl;
        lastMark = now;
    };

    if (!fs::exists(imageFile))
    {
        throw std::runtime_error("CellLumen frame file not found: " + imageFile.string());
    }

    activeProfile = inferDatasetProfile(imageFile);
    config.simulation.z_scaling = effectiveZScaling();
    std::cout << "[CellLumen Dataset] profile=" << activeProfile.label
              << " effective_z_scaling=" << effectiveZScaling();
    if (activeProfile.label == "hl60_sim")
    {
        std::cout << " voxel_size_xyz=(0.125,0.125,0.200)";
    }
    std::cout << std::endl;

    PathVec discoveredInput = ImageHandler::getImageFilePaths(imageFile.string(), 0, 0, config);
    if (discoveredInput.empty())
    {
        throw std::runtime_error("CellLumen input discovery returned no files.");
    }
    logElapsed("input_discovery");

    std::vector<cv::Mat> realFrame = loadCellLumenRawStack(discoveredInput.front());
    std::cout << "[CellLumen RawInput] frame=" << imageFile.filename().string()
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

    std::cout << "[CellLumen Result] frame=" << imageFile.filename().string()
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
    std::cout << "[CellLumen Timing Summary] frame=" << imageFile.filename().string()
              << " detected_cells=" << cells.size()
              << " total_sec=" << std::fixed << std::setprecision(6)
              << std::chrono::duration<double>(Clock::now() - totalStart).count()
              << std::defaultfloat
              << std::endl;
    return cells;
}

std::vector<CellLumen::DetectedCell> CellLumen::detectCellsForFrame(
    const fs::path &imageFile,
    bool printCellDetails)
{
    using Clock = std::chrono::steady_clock;
    const auto totalStart = Clock::now();

    if (!fs::exists(imageFile))
    {
        throw std::runtime_error("CellLumen frame file not found: " + imageFile.string());
    }

    activeProfile = inferDatasetProfile(imageFile);
    config.simulation.z_scaling = effectiveZScaling();
    std::cout << "[CellLumen Fusion Dataset] profile=" << activeProfile.label
              << " effective_z_scaling=" << effectiveZScaling()
              << " frame=" << imageFile.filename().string()
              << std::endl;

    PathVec discoveredInput = ImageHandler::getImageFilePaths(imageFile.string(), 0, 0, config);
    if (discoveredInput.empty())
    {
        throw std::runtime_error("CellLumen input discovery returned no files.");
    }

    std::vector<cv::Mat> realFrame = loadCellLumenRawStack(discoveredInput.front());
    config.simulation.z_slices = static_cast<int>(realFrame.size());

    if (config.cell)
    {
        Ellipsoid::cellConfig = *config.cell;
        Ellipsoid::cellConfig.maxZ = static_cast<float>(realFrame.size()) - 1.0f;
    }

    const std::string frameStem = imageFile.stem().string();
    std::vector<DetectedCell> cells = detectCellsInVolume(realFrame, frameStem);

    std::cout << "[CellLumen Fusion Result] frame=" << imageFile.filename().string()
              << " detected_cells=" << cells.size()
              << " total_sec=" << std::fixed << std::setprecision(6)
              << std::chrono::duration<double>(Clock::now() - totalStart).count()
              << std::defaultfloat
              << std::endl;

    if (printCellDetails)
    {
        for (const auto &cell : cells)
        {
            std::cout << "  [CellLumen Fusion Cell] name=" << cell.name
                      << " centerScaled=(" << cell.centerScaled.x << ","
                      << cell.centerScaled.y << "," << cell.centerScaled.z << ")"
                      << " major=" << cell.majorRadius
                      << " b=" << cell.bRadius
                      << " minor=" << cell.minorRadius
                      << " vox=" << cell.voxelCount
                      << " top10MinusShell=" << cell.top10MinusShell
                      << std::endl;
        }
    }

    return cells;
}
