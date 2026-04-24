#include "../includes/ImageHandler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

namespace
{
int makeOddAtLeast(int value, int minimum = 3)
{
    int adjusted = std::max(value, minimum);
    if (adjusted % 2 == 0)
    {
        ++adjusted;
    }
    return adjusted;
}

bool shouldIgnoreImagePath(const fs::path &file)
{
    const std::string name = file.filename().string();
    return name.empty() || name[0] == '.' || name.rfind("._", 0) == 0;
}

void updateTiffConfigIfNeeded(const fs::path &file, BaseConfig &config)
{
    if (shouldIgnoreImagePath(file) || !(file.extension() == ".tif" || file.extension() == ".tiff"))
    {
        return;
    }

    std::vector<cv::Mat> images;
    cv::imreadmulti(file.string(), images, cv::IMREAD_UNCHANGED);
    config.simulation.z_slices = static_cast<int>(images.size());
}

struct StackStats
{
    double minValue = 0.0;
    double maxValue = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
    std::size_t count = 0;
};

StackStats computeStackStats(const ImageStack &stack)
{
    StackStats stats;
    double sum = 0.0;
    double sumSq = 0.0;
    double minValue = std::numeric_limits<double>::infinity();
    double maxValue = -std::numeric_limits<double>::infinity();
    std::size_t count = 0;

    #pragma omp parallel for schedule(static) reduction(+:sum,sumSq,count) reduction(min:minValue) reduction(max:maxValue)
    for (int i = 0; i < static_cast<int>(stack.size()); ++i)
    {
        const auto &slice = stack[static_cast<std::size_t>(i)];
        CV_Assert(slice.type() == CV_32F);

        double sliceMin = 0.0;
        double sliceMax = 0.0;
        cv::minMaxLoc(slice, &sliceMin, &sliceMax);
        minValue = std::min(minValue, sliceMin);
        maxValue = std::max(maxValue, sliceMax);

        for (int y = 0; y < slice.rows; ++y)
        {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x)
            {
                const double value = row[x];
                sum += value;
                sumSq += value * value;
                ++count;
            }
        }
    }

    stats.count = count;
    if (stats.count == 0)
    {
        return stats;
    }

    stats.minValue = minValue;
    stats.maxValue = maxValue;
    stats.mean = sum / static_cast<double>(stats.count);
    const double variance =
        std::max(0.0, sumSq / static_cast<double>(stats.count) - stats.mean * stats.mean);
    stats.stddev = std::sqrt(variance);
    return stats;
}

void printStackStats(std::ostream &log,
                     const std::string &stage,
                     const std::string &imageFile,
                     const ImageStack &stack)
{
    const StackStats stats = computeStackStats(stack);
    log << "[Preprocess] file=" << fs::path(imageFile).filename().string()
        << " stage=" << stage
        << " slices=" << stack.size()
        << " voxels=" << stats.count
        << " min=" << stats.minValue
        << " max=" << stats.maxValue
        << " mean=" << stats.mean
        << " stddev=" << stats.stddev
        << std::endl;
}

ImageStack cloneStack(const ImageStack &sequence)
{
    ImageStack cloned(sequence.size());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(sequence.size()); ++i)
    {
        cloned[static_cast<std::size_t>(i)] = sequence[static_cast<std::size_t>(i)].clone();
    }
    return cloned;
}

ImageStack applyAdaptiveCubePooling(const ImageStack &stack,
                                    const BaseConfig &config,
                                    std::ostream &log)
{
    if (stack.empty() || !config.cell || !config.simulation.adaptive_cube_pooling_enabled)
    {
        return cloneStack(stack);
    }

    const int depth = static_cast<int>(stack.size());
    const int rows = stack[0].rows;
    const int cols = stack[0].cols;
    if (depth <= 0 || rows <= 0 || cols <= 0)
    {
        return cloneStack(stack);
    }

    const float minRadius = std::max(
        1.0f,
        static_cast<float>(std::min({config.cell->minARadius,
                                     config.cell->minBRadius,
                                     config.cell->minCRadius})));
    const float cubeSizeTarget =
        std::max(1.0f, config.simulation.adaptive_cube_pooling_cube_size_scale * minRadius);
    const int cubeSize = std::max(1, static_cast<int>(std::lround(cubeSizeTarget)));
    const float zeroThreshold = std::max(0.0f, config.simulation.adaptive_cube_pooling_zero_threshold);
    const float majorityThreshold =
        std::clamp(config.simulation.adaptive_cube_pooling_majority_threshold, 0.0f, 1.0f);

    const int gridZ = (depth + cubeSize - 1) / cubeSize;
    const int gridY = (rows + cubeSize - 1) / cubeSize;
    const int gridX = (cols + cubeSize - 1) / cubeSize;

    struct CubeStats
    {
        float mean = 0.0f;
        float maxValue = 0.0f;
        float zeroFraction = 0.0f;
    };

    std::vector<CubeStats> cubeStats(static_cast<size_t>(gridZ) * gridY * gridX);
    auto cubeIndex = [gridX, gridY](int gz, int gy, int gx) -> size_t {
        return static_cast<size_t>((gz * gridY + gy) * gridX + gx);
    };

    // Parallelize cube-stats computation. Each cube writes only to its
    // own cubeStats[idx] -> no races.
    #pragma omp parallel for collapse(3) schedule(static)
    for (int gz = 0; gz < gridZ; ++gz)
    {
        for (int gy = 0; gy < gridY; ++gy)
        {
            for (int gx = 0; gx < gridX; ++gx)
            {
                const int z0 = gz * cubeSize;
                const int z1 = std::min(depth, z0 + cubeSize);
                const int y0 = gy * cubeSize;
                const int y1 = std::min(rows, y0 + cubeSize);
                const int x0 = gx * cubeSize;
                const int x1 = std::min(cols, x0 + cubeSize);

                double sum = 0.0;
                float maxValue = 0.0f;
                int zeroCount = 0;
                int voxelCount = 0;
                for (int z = z0; z < z1; ++z)
                {
                    const cv::Mat &slice = stack[static_cast<size_t>(z)];
                    for (int y = y0; y < y1; ++y)
                    {
                        const float *row = slice.ptr<float>(y);
                        for (int x = x0; x < x1; ++x)
                        {
                            const float value = row[x];
                            sum += value;
                            maxValue = std::max(maxValue, value);
                            zeroCount += (value <= zeroThreshold) ? 1 : 0;
                            ++voxelCount;
                        }
                    }
                }

                CubeStats &stats = cubeStats[cubeIndex(gz, gy, gx)];
                if (voxelCount > 0)
                {
                    stats.mean = static_cast<float>(sum / static_cast<double>(voxelCount));
                    stats.maxValue = maxValue;
                    stats.zeroFraction = static_cast<float>(zeroCount) /
                                         static_cast<float>(voxelCount);
                }
            }
        }
    }

    ImageStack pooled(depth);
    for (int z = 0; z < depth; ++z)
    {
        pooled[static_cast<size_t>(z)] = cv::Mat::zeros(rows, cols, CV_32F);
    }
    std::vector<float> pooledCubeValues(static_cast<size_t>(gridZ) * gridY * gridX, 0.0f);

    int meanPooledCubes = 0;
    int maxPooledCubes = 0;
    // Parallelize cube-reweighting + voxel fill. Each iteration reads from
    // cubeStats (read-only) and writes to its own cube voxel range in pooled
    // (disjoint per-tuple) and pooledCubeValues[cubeIndex(gz,gy,gx)] (disjoint).
    // Counters updated via reduction.
    #pragma omp parallel for collapse(3) schedule(static) reduction(+:meanPooledCubes, maxPooledCubes)
    for (int gz = 0; gz < gridZ; ++gz)
    {
        for (int gy = 0; gy < gridY; ++gy)
        {
            for (int gx = 0; gx < gridX; ++gx)
            {
                const int z0 = gz * cubeSize;
                const int z1 = std::min(depth, z0 + cubeSize);
                const int y0 = gy * cubeSize;
                const int y1 = std::min(rows, y0 + cubeSize);
                const int x0 = gx * cubeSize;
                const int x1 = std::min(cols, x0 + cubeSize);
                const CubeStats &stats = cubeStats[cubeIndex(gz, gy, gx)];

                float neighborMean = 0.0f;
                int neighborCount = 0;
                for (int nz = std::max(0, gz - 1); nz <= std::min(gridZ - 1, gz + 1); ++nz)
                {
                    for (int ny = std::max(0, gy - 1); ny <= std::min(gridY - 1, gy + 1); ++ny)
                    {
                        for (int nx = std::max(0, gx - 1); nx <= std::min(gridX - 1, gx + 1); ++nx)
                        {
                            if (nz == gz && ny == gy && nx == gx)
                            {
                                continue;
                            }
                            neighborMean += cubeStats[cubeIndex(nz, ny, nx)].mean;
                            ++neighborCount;
                        }
                    }
                }
                if (neighborCount > 0)
                {
                    neighborMean /= static_cast<float>(neighborCount);
                }

                const bool useMeanPooling =
                    stats.zeroFraction >= majorityThreshold &&
                    neighborMean <= zeroThreshold;
                const float pooledValue = useMeanPooling ? stats.mean : stats.maxValue;
                pooledCubeValues[cubeIndex(gz, gy, gx)] = pooledValue;
                if (useMeanPooling)
                {
                    ++meanPooledCubes;
                }
                else
                {
                    ++maxPooledCubes;
                }

                for (int z = z0; z < z1; ++z)
                {
                    cv::Mat &slice = pooled[static_cast<size_t>(z)];
                    for (int y = y0; y < y1; ++y)
                    {
                        float *row = slice.ptr<float>(y);
                        for (int x = x0; x < x1; ++x)
                        {
                            row[x] = pooledValue;
                        }
                    }
                }
            }
        }
    }

    int removedIsolatedBrightCubes = 0;
    int isolatedBrightCandidateCubes = 0;
    if (config.simulation.adaptive_cube_pooling_remove_isolated_bright_cubes)
    {
        const float isolatedBrightThreshold =
            std::max(zeroThreshold,
                     config.simulation.adaptive_cube_pooling_isolated_bright_cube_threshold);
        std::vector<char> clearCube(static_cast<size_t>(gridZ) * gridY * gridX, 0);
        // Parallelize neighbor-check: each cube only writes to its own
        // clearCube[idx] and reads from pooledCubeValues (read-only here).
        // Candidate counter via reduction.
        #pragma omp parallel for collapse(3) schedule(static) reduction(+:isolatedBrightCandidateCubes)
        for (int gz = 0; gz < gridZ; ++gz)
        {
            for (int gy = 0; gy < gridY; ++gy)
            {
                for (int gx = 0; gx < gridX; ++gx)
                {
                    const size_t idx = cubeIndex(gz, gy, gx);
                    const float centerValue = pooledCubeValues[idx];
                    if (centerValue <= isolatedBrightThreshold)
                    {
                        continue;
                    }
                    ++isolatedBrightCandidateCubes;

                    bool hasBrightNeighbor = false;
                    for (int nz = std::max(0, gz - 1); nz <= std::min(gridZ - 1, gz + 1) && !hasBrightNeighbor; ++nz)
                    {
                        for (int ny = std::max(0, gy - 1); ny <= std::min(gridY - 1, gy + 1) && !hasBrightNeighbor; ++ny)
                        {
                            for (int nx = std::max(0, gx - 1); nx <= std::min(gridX - 1, gx + 1); ++nx)
                            {
                                if (nz == gz && ny == gy && nx == gx)
                                {
                                    continue;
                                }
                                if (pooledCubeValues[cubeIndex(nz, ny, nx)] > zeroThreshold)
                                {
                                    hasBrightNeighbor = true;
                                    break;
                                }
                            }
                        }
                    }

                    if (!hasBrightNeighbor)
                    {
                        clearCube[idx] = 1;
                    }
                }
            }
        }

        // Parallel voxel-zero pass. Each cube writes to its own disjoint
        // voxel range in pooled. Counter via reduction.
        #pragma omp parallel for collapse(3) schedule(static) reduction(+:removedIsolatedBrightCubes)
        for (int gz = 0; gz < gridZ; ++gz)
        {
            for (int gy = 0; gy < gridY; ++gy)
            {
                for (int gx = 0; gx < gridX; ++gx)
                {
                    const int z0 = gz * cubeSize;
                    const int z1 = std::min(depth, z0 + cubeSize);
                    const int y0 = gy * cubeSize;
                    const int y1 = std::min(rows, y0 + cubeSize);
                    if (!clearCube[cubeIndex(gz, gy, gx)])
                    {
                        continue;
                    }

                    ++removedIsolatedBrightCubes;
                    const int x0 = gx * cubeSize;
                    const int x1 = std::min(cols, x0 + cubeSize);
                    for (int z = z0; z < z1; ++z)
                    {
                        cv::Mat &slice = pooled[static_cast<size_t>(z)];
                        for (int y = y0; y < y1; ++y)
                        {
                            float *row = slice.ptr<float>(y);
                            for (int x = x0; x < x1; ++x)
                            {
                                row[x] = 0.0f;
                            }
                        }
                    }
                }
            }
        }
    }

    log << "[AdaptiveCubePooling]"
        << " cubeSize=" << cubeSize
        << " grid=" << gridX << "x" << gridY << "x" << gridZ
        << " zeroThreshold=" << zeroThreshold
        << " majorityThreshold=" << majorityThreshold
        << " meanCubes=" << meanPooledCubes
        << " maxCubes=" << maxPooledCubes
        << " isolatedBrightThreshold="
        << (config.simulation.adaptive_cube_pooling_remove_isolated_bright_cubes
                ? std::max(zeroThreshold,
                           config.simulation.adaptive_cube_pooling_isolated_bright_cube_threshold)
                : 0.0f)
        << " isolatedBrightCandidates=" << isolatedBrightCandidateCubes
        << " removedIsolatedBrightCubes=" << removedIsolatedBrightCubes
        << std::endl;

    return pooled;
}

struct BrightBox
{
    int ix = 0;
    int iy = 0;
    int iz = 0;
    cv::Point3f center{0.0f, 0.0f, 0.0f};
    float brightness = 0.0f;
    int voxels = 0;
};

int chooseNearestDivisorSize(int axisLength, float targetSize)
{
    if (axisLength <= 1)
    {
        return 1;
    }

    const float clampedTarget = std::clamp(targetSize, 1.0f, static_cast<float>(axisLength));
    int bestDivisor = 1;
    float bestDistance = std::abs(static_cast<float>(bestDivisor) - clampedTarget);
    for (int d = 1; d <= axisLength; ++d)
    {
        if (axisLength % d != 0)
        {
            continue;
        }
        const float distance = std::abs(static_cast<float>(d) - clampedTarget);
        if (distance < bestDistance ||
            (std::abs(distance - bestDistance) <= 1e-6f && d > bestDivisor))
        {
            bestDivisor = d;
            bestDistance = distance;
        }
    }
    return bestDivisor;
}

std::vector<Frame::SignalCenter> localizeSignalCentersInStack(const ImageStack &stack,
                                                              const BaseConfig &config)
{
    std::vector<Frame::SignalCenter> centers;
    if (stack.empty() || !config.cell)
    {
        return centers;
    }

    const int sizeX = stack[0].cols;
    const int sizeY = stack[0].rows;
    const int sizeZ = static_cast<int>(stack.size());
    if (sizeX <= 0 || sizeY <= 0 || sizeZ <= 0)
    {
        return centers;
    }

    const float boxScale = std::max(0.1f, config.simulation.signal_guided_box_size_scale);
    const float zShrink = std::max(1.0f, config.simulation.z_scaling);
    const float maxDiamX = 2.0f * static_cast<float>(config.cell->maxARadius);
    const float maxDiamY = 2.0f * static_cast<float>(
        config.cell->maxBRadius > 0.0 ? config.cell->maxBRadius : config.cell->maxARadius);
    const float maxDiamZ = 2.0f * static_cast<float>(config.cell->maxCRadius);
    const float targetBoxX = std::max(1.0f, (maxDiamX / 3.0f) * boxScale);
    const float targetBoxY = std::max(1.0f, (maxDiamY / 3.0f) * boxScale);
    const float targetBoxZ = std::max(1.0f, (maxDiamZ / 3.0f) * (boxScale / zShrink));

    const int boxSizeX = chooseNearestDivisorSize(sizeX, targetBoxX);
    const int boxSizeY = chooseNearestDivisorSize(sizeY, targetBoxY);
    const int boxSizeZ = chooseNearestDivisorSize(sizeZ, targetBoxZ);
    const int gridX = std::max(1, sizeX / boxSizeX);
    const int gridY = std::max(1, sizeY / boxSizeY);
    const int gridZ = std::max(1, sizeZ / boxSizeZ);

    const float backgroundValue = 0.0f;
    const float minDelta = std::max(0.0f, config.simulation.signal_guided_min_box_brightness_delta);

    std::vector<BrightBox> boxes;
    boxes.reserve(static_cast<size_t>(gridX) * gridY * gridZ);
    for (int iz = 0; iz < gridZ; ++iz)
    {
        const int z0 = iz * boxSizeZ;
        const int z1 = z0 + boxSizeZ;
        for (int iy = 0; iy < gridY; ++iy)
        {
            const int y0 = iy * boxSizeY;
            const int y1 = y0 + boxSizeY;
            for (int ix = 0; ix < gridX; ++ix)
            {
                const int x0 = ix * boxSizeX;
                const int x1 = x0 + boxSizeX;

                double sum = 0.0;
                int voxels = 0;
                for (int z = z0; z < z1; ++z)
                {
                    for (int y = y0; y < y1; ++y)
                    {
                        const float *row = stack[static_cast<size_t>(z)].ptr<float>(y);
                        for (int x = x0; x < x1; ++x)
                        {
                            sum += row[x];
                            ++voxels;
                        }
                    }
                }
                if (voxels <= 0)
                {
                    continue;
                }

                const float meanBrightness = static_cast<float>(sum / static_cast<double>(voxels));
                if (meanBrightness <= backgroundValue + minDelta)
                {
                    continue;
                }

                boxes.push_back({
                    ix, iy, iz,
                    cv::Point3f(
                        static_cast<float>(x0 + x1 - 1) * 0.5f,
                        static_cast<float>(y0 + y1 - 1) * 0.5f,
                        static_cast<float>(z0 + z1 - 1) * 0.5f),
                    meanBrightness,
                    voxels
                });
            }
        }
    }

    std::sort(boxes.begin(), boxes.end(),
              [](const BrightBox &a, const BrightBox &b)
              {
                  return a.brightness > b.brightness;
              });

    std::map<std::tuple<int, int, int>, size_t> boxIndex;
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        boxIndex[{boxes[i].ix, boxes[i].iy, boxes[i].iz}] = i;
    }

    std::vector<char> visited(boxes.size(), 0);
    float maxCenterBrightness = backgroundValue;
    for (size_t seed = 0; seed < boxes.size(); ++seed)
    {
        if (visited[seed])
        {
            continue;
        }

        std::vector<size_t> stackIndices{seed};
        visited[seed] = 1;
        double weightSum = 0.0;
        double xSum = 0.0;
        double ySum = 0.0;
        double zSum = 0.0;
        double brightnessSum = 0.0;
        int clusterBoxes = 0;

        while (!stackIndices.empty())
        {
            const size_t current = stackIndices.back();
            stackIndices.pop_back();
            const BrightBox &box = boxes[current];
            const float weight = std::max(1e-6f, box.brightness - backgroundValue);
            weightSum += weight;
            xSum += weight * static_cast<double>(box.center.x);
            ySum += weight * static_cast<double>(box.center.y);
            zSum += weight * static_cast<double>(box.center.z);
            brightnessSum += static_cast<double>(box.brightness);
            ++clusterBoxes;

            for (int dz = -1; dz <= 1; ++dz)
            {
                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        if (dx == 0 && dy == 0 && dz == 0)
                        {
                            continue;
                        }
                        auto it = boxIndex.find({box.ix + dx, box.iy + dy, box.iz + dz});
                        if (it == boxIndex.end())
                        {
                            continue;
                        }
                        const size_t neighbor = it->second;
                        if (visited[neighbor])
                        {
                            continue;
                        }
                        visited[neighbor] = 1;
                        stackIndices.push_back(neighbor);
                    }
                }
            }
        }

        if (clusterBoxes <= 0 || weightSum <= 0.0)
        {
            continue;
        }

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

    for (auto &center : centers)
    {
        const float normalized = (maxCenterBrightness > backgroundValue + 1e-6f)
            ? std::clamp((center.brightness - backgroundValue) /
                         (maxCenterBrightness - backgroundValue), 0.0f, 1.0f)
            : 0.0f;
        center.sigmaScale = std::max(
            config.simulation.signal_guided_min_sigma_scale,
            1.0f - normalized * (1.0f - config.simulation.signal_guided_min_sigma_scale));
    }

    std::sort(centers.begin(), centers.end(),
              [](const Frame::SignalCenter &a, const Frame::SignalCenter &b)
              {
                  return a.brightness > b.brightness;
              });

    return centers;
}

void clipStack(ImageStack &sequence)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(sequence.size()); ++i)
    {
        auto &slice = sequence[static_cast<std::size_t>(i)];
        cv::min(slice, 1.0f, slice);
        cv::max(slice, 0.0f, slice);
    }
}
} // namespace

Image ImageHandler::processImage(const Image &image, const BaseConfig &config)
{
    Image processedImage;

    if (image.channels() == 3)
    {
        cv::cvtColor(image, processedImage, cv::COLOR_RGB2GRAY);
    }
    else
    {
        processedImage = image.clone();
    }

    // Preserve the source intensity scale here. Frames are normalized later
    // using one shared percentile-based scale across the selected run.
    processedImage.convertTo(processedImage, CV_32F);

    if (config.simulation.blur_sigma > 0.0f)
    {
        cv::GaussianBlur(processedImage,
                         processedImage,
                         cv::Size(0, 0),
                         config.simulation.blur_sigma,
                         config.simulation.blur_sigma);
    }

    return processedImage;
}

float ImageHandler::computePercentileFromValues(std::vector<float> values, float percentileFraction)
{
    if (values.empty())
    {
        return 0.0f;
    }

    const float clampedPercentile = std::clamp(percentileFraction, 0.0f, 1.0f);
    const std::size_t index = static_cast<std::size_t>(
        std::floor(clampedPercentile * static_cast<float>(values.size() - 1)));

    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(index),
                     values.end());
    return values[index];
}

float computeMeanOfTopFraction(std::vector<float> values, float topFraction)
{
    if (values.empty())
    {
        return 0.0f;
    }

    const float clampedFraction = std::clamp(topFraction, 0.0f, 1.0f);
    if (clampedFraction <= 0.0f)
    {
        return 0.0f;
    }

    const std::size_t selectedCount = std::max<std::size_t>(
        1,
        static_cast<std::size_t>(
            std::ceil(clampedFraction * static_cast<float>(values.size()))));
    const std::size_t thresholdIndex = values.size() - selectedCount;

    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(thresholdIndex),
                     values.end());

    double sum = 0.0;
    for (std::size_t i = thresholdIndex; i < values.size(); ++i)
    {
        sum += values[i];
    }

    return static_cast<float>(sum / static_cast<double>(selectedCount));
}

float ImageHandler::computePercentileFromSlice(const cv::Mat &slice, float percentileFraction)
{
    CV_Assert(slice.type() == CV_32F);

    std::vector<float> values;
    values.reserve(static_cast<std::size_t>(slice.total()));
    for (int y = 0; y < slice.rows; ++y)
    {
        const float *row = slice.ptr<float>(y);
        values.insert(values.end(), row, row + slice.cols);
    }
    return computePercentileFromValues(std::move(values), percentileFraction);
}

cv::Mat ImageHandler::boxMean(const cv::Mat &image, int windowSize)
{
    if (windowSize < 1 || windowSize % 2 == 0)
    {
        throw std::invalid_argument("windowSize must be a positive odd integer");
    }

    cv::Mat output;
    cv::blur(image, output, cv::Size(windowSize, windowSize), cv::Point(-1, -1), cv::BORDER_REPLICATE);
    return output;
}

float ImageHandler::evaluateSequenceContrastScore(const ImageStack &sequence, const BaseConfig &config)
{
    const float aRadius =
        (config.cell ? static_cast<float>(config.cell->maxARadius) : 40.0f);
    const float cRadius =
        (config.cell ? static_cast<float>(config.cell->maxCRadius) : 35.0f);
    const float maxRadius = std::max(1.0f, std::max(aRadius, cRadius));
    return evaluateSequenceContrastScoreForRadius(sequence, config, maxRadius);
}

float ImageHandler::evaluateBestWindowContrastScore(const ImageStack &sequence, const BaseConfig &config)
{
    const float defaultMinRadius = 5.0f;
    const float defaultMaxRadius = 65.0f;
    const float minRadius = config.cell
        ? std::max(1.0f, static_cast<float>(std::min({
              config.cell->minARadius,
              config.cell->minBRadius,
              config.cell->minCRadius})))
        : defaultMinRadius;
    const float maxRadius = config.cell
        ? std::max(minRadius, static_cast<float>(std::max({
              config.cell->maxARadius,
              config.cell->maxBRadius > 0.0 ? config.cell->maxBRadius : config.cell->maxARadius,
              config.cell->maxCRadius})))
        : defaultMaxRadius;
    const float radiusStep = std::max(1.0f, config.simulation.contrast_window_radius_step);

    std::vector<float> scoringRadii;
    for (float radius = minRadius; radius < maxRadius - 1e-6f; radius += radiusStep) {
        scoringRadii.push_back(radius);
    }
    if (scoringRadii.empty() || std::abs(scoringRadii.back() - maxRadius) > 1e-6f) {
        scoringRadii.push_back(maxRadius);
    }

    std::vector<float> scores(scoringRadii.size(), 0.0f);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(scoringRadii.size()); ++i) {
        scores[static_cast<std::size_t>(i)] = evaluateSequenceContrastScoreForRadius(
            sequence, config, scoringRadii[static_cast<std::size_t>(i)]);
    }

    return *std::max_element(scores.begin(), scores.end());
}

float ImageHandler::evaluateSequenceContrastScoreForRadius(const ImageStack &sequence,
                                                           const BaseConfig &config,
                                                           float radiusAtScale)
{
    const int innerWindow = makeOddAtLeast(
        static_cast<int>(std::lround(radiusAtScale * 2.0f + 1.0f)));
    const int outerWindow = makeOddAtLeast(
        static_cast<int>(std::lround(radiusAtScale * 4.0f + 1.0f)),
        innerWindow + 2);

    std::vector<float> sliceScores(sequence.size(), 0.0f);

    #pragma omp parallel for schedule(dynamic)
    for (int sliceIndex = 0; sliceIndex < static_cast<int>(sequence.size()); ++sliceIndex)
    {
        const auto &slice = sequence[static_cast<std::size_t>(sliceIndex)];
        CV_Assert(slice.type() == CV_32F);

        const cv::Mat innerMean = boxMean(slice, innerWindow);
        const cv::Mat outerMean = boxMean(slice, outerWindow);

        std::vector<float> contrastValues;
        std::vector<float> brightInteriorValues;
        contrastValues.reserve(static_cast<std::size_t>(slice.total()));
        brightInteriorValues.reserve(static_cast<std::size_t>(slice.total() / 8));

        for (int y = 0; y < slice.rows; ++y)
        {
            const float *sliceRow = slice.ptr<float>(y);
            const float *innerRow = innerMean.ptr<float>(y);
            const float *outerRow = outerMean.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x)
            {
                const float localDifference = innerRow[x] - outerRow[x];
                if (localDifference < config.simulation.contrast_structure_threshold)
                {
                    continue;
                }

                const float stableBackground =
                    std::max(outerRow[x], config.simulation.contrast_background_floor);
                const float localContrast =
                    localDifference / (stableBackground + config.simulation.contrast_eps);
                contrastValues.push_back(localContrast);

                // Reward bright filled interiors in regions that already
                // exhibit positive local contrast.
                brightInteriorValues.push_back(sliceRow[x]);
            }
        }

        if (contrastValues.empty())
        {
            sliceScores[static_cast<std::size_t>(sliceIndex)] = 0.0f;
            continue;
        }

        const float contrastScore = computePercentileFromValues(
            std::move(contrastValues),
            config.simulation.iterative_score_percentile);
        const float brightInteriorScore = computeMeanOfTopFraction(
            std::move(brightInteriorValues),
            config.simulation.contrast_bright_fraction);

        sliceScores[static_cast<std::size_t>(sliceIndex)] =
            contrastScore +
            config.simulation.contrast_brightness_weight * brightInteriorScore;
    }

    if (sliceScores.empty())
    {
        return 0.0f;
    }

    return computePercentileFromValues(std::move(sliceScores), 0.5f);
}

ImageStack ImageHandler::processPreparedSequence(const ImageStack &sequence,
                                                const BaseConfig &config,
                                                std::ostream &log,
                                                const std::string &imageFile)
{
    const int maxIterations = std::max(1, config.simulation.iterative_max_count);
    const float gateStart = config.simulation.iterative_reward_gate;
    const float gateMin = std::min(gateStart, config.simulation.iterative_reward_gate_min);
    const float gateStep = config.simulation.iterative_reward_gate_decrement;

    std::vector<float> rewardGates;
    if (gateStep <= 1e-6f || gateStart <= gateMin + 1e-6f) {
        rewardGates.push_back(gateStart);
    } else {
        for (float gate = gateStart; gate > gateMin + 1e-6f; gate -= gateStep) {
            rewardGates.push_back(gate);
        }
        if (rewardGates.empty() || std::abs(rewardGates.back() - gateMin) > 1e-6f) {
            rewardGates.push_back(gateMin);
        }
    }

    const double learningRate = std::clamp(
        static_cast<double>(config.simulation.iterative_reward_gate_learning_rate),
        0.0,
        1.0);
    const int noImprovementPatience = std::max(1, config.simulation.iterative_no_improvement_patience);
    const float explosionThreshold = std::max(
        config.simulation.iterative_score_explosion_threshold,
        config.simulation.iterative_score_max);
    const std::size_t imageHash = std::hash<std::string>{}(fs::path(imageFile).filename().string());

    const float defaultMinRadius = 5.0f;
    const float defaultMaxRadius = 65.0f;
    const float minRadius = config.cell
        ? std::max(1.0f, static_cast<float>(std::min({
              config.cell->minARadius,
              config.cell->minBRadius,
              config.cell->minCRadius})))
        : defaultMinRadius;
    const float maxRadius = config.cell
        ? std::max(minRadius, static_cast<float>(std::max({
              config.cell->maxARadius,
              config.cell->maxBRadius > 0.0 ? config.cell->maxBRadius : config.cell->maxARadius,
              config.cell->maxCRadius})))
        : defaultMaxRadius;
    const float radiusStep = std::max(1.0f, config.simulation.contrast_window_radius_step);

    std::vector<float> scoringRadii;
    for (float radius = minRadius; radius < maxRadius - 1e-6f; radius += radiusStep) {
        scoringRadii.push_back(radius);
    }
    if (scoringRadii.empty() || std::abs(scoringRadii.back() - maxRadius) > 1e-6f) {
        scoringRadii.push_back(maxRadius);
    }

    struct PreprocessTrialResult
    {
        ImageStack sequence;
        float score = -std::numeric_limits<float>::infinity();
        float radius = 0.0f;
        std::string logText;
    };

    auto runRadiusTrial = [&](float scoringRadius, int radiusIndex) {
        std::ostringstream trialLog;
        ImageStack current = cloneStack(sequence);
        ImageStack bestSequence = cloneStack(current);
        float currentScore = evaluateSequenceContrastScoreForRadius(current, config, scoringRadius);
        float bestScore = currentScore;
        float currentPenalty = config.simulation.iterative_penalty;

        std::vector<double> gateProbabilities(
            rewardGates.size(), 1.0 / static_cast<double>(rewardGates.size()));
        const double minProbability = std::min(
            std::max(0.0, static_cast<double>(config.simulation.iterative_reward_gate_min_probability)),
            1.0 / static_cast<double>(rewardGates.size()));
        auto normalizeGateProbabilities = [&]() {
            double sum = 0.0;
            for (double &probability : gateProbabilities) {
                probability = std::max(probability, minProbability);
                sum += probability;
            }
            if (sum <= 0.0) {
                const double uniform = 1.0 / static_cast<double>(gateProbabilities.size());
                std::fill(gateProbabilities.begin(), gateProbabilities.end(), uniform);
                return;
            }
            for (double &probability : gateProbabilities) {
                probability /= sum;
            }
        };

        auto updateGateProbability = [&](std::size_t gateIndex, bool improved) {
            const double multiplier = improved ? (1.0 + learningRate) : (1.0 - learningRate);
            gateProbabilities[gateIndex] *= multiplier;
            normalizeGateProbabilities();
        };

        const unsigned int rngSeed =
            static_cast<unsigned int>(config.simulation.iterative_reward_gate_seed) ^
            static_cast<unsigned int>(imageHash & 0xffffffffU) ^
            (static_cast<unsigned int>(radiusIndex) * 0x9e3779b9U);
        std::mt19937 rng(rngSeed);

        trialLog << "[IterPreprocess] radius=" << scoringRadius
                 << " initial_score=" << currentScore
                 << " gate_count=" << rewardGates.size()
                 << " learning_rate=" << learningRate
                 << " min_probability=" << minProbability
                 << std::endl;

        int noImprovementCount = 0;
        for (int count = 0; count < maxIterations; ++count)
        {
            std::discrete_distribution<int> gateDistribution(
                gateProbabilities.begin(), gateProbabilities.end());
            const int selectedGateIndex = gateDistribution(rng);
            const float selectedGate = rewardGates[static_cast<std::size_t>(selectedGateIndex)];

            ImageStack candidate = cloneStack(current);

            #pragma omp parallel for schedule(static)
            for (int sliceIndex = 0; sliceIndex < static_cast<int>(candidate.size()); ++sliceIndex)
            {
                auto &slice = candidate[static_cast<std::size_t>(sliceIndex)];
                for (int y = 0; y < slice.rows; ++y)
                {
                    float *row = slice.ptr<float>(y);
                    for (int x = 0; x < slice.cols; ++x)
                    {
                        if (row[x] < config.simulation.iterative_penalty_range)
                        {
                            const float normalizedDistanceToKeep =
                                (config.simulation.iterative_penalty_range > 1e-6f)
                                    ? ((config.simulation.iterative_penalty_range - row[x]) /
                                       config.simulation.iterative_penalty_range)
                                    : 1.0f;
                            const float penaltyStrength =
                                std::clamp(normalizedDistanceToKeep, 0.0f, 1.0f);
                            row[x] -= currentPenalty * penaltyStrength;
                            if (row[x] < 0.0f)
                            {
                                row[x] = 0.0f;
                            }
                        }
                    }
                }
            }

            #pragma omp parallel for schedule(static)
            for (int sliceIndex = 0; sliceIndex < static_cast<int>(candidate.size()); ++sliceIndex)
            {
                auto &slice = candidate[static_cast<std::size_t>(sliceIndex)];
                for (int y = 0; y < slice.rows; ++y)
                {
                    float *row = slice.ptr<float>(y);
                    for (int x = 0; x < slice.cols; ++x)
                    {
                        if (row[x] > selectedGate)
                        {
                            row[x] += config.simulation.iterative_reward;
                            if (row[x] > 1.0f)
                            {
                                row[x] = 1.0f;
                            }
                        }
                    }
                }
            }

            const float score = evaluateSequenceContrastScoreForRadius(candidate, config, scoringRadius);
            if (!std::isfinite(score) || score > explosionThreshold) {
                trialLog << "[IterPreprocess] radius=" << scoringRadius
                         << " stop=score_explosion"
                         << " round=" << count
                         << " score=" << score
                         << " threshold=" << explosionThreshold
                         << std::endl;
                break;
            }

            const bool improvedCurrent =
                score > currentScore + config.simulation.iterative_improvement_tolerance;
            updateGateProbability(static_cast<std::size_t>(selectedGateIndex), improvedCurrent);

            if (improvedCurrent) {
                currentPenalty = config.simulation.iterative_penalty;
                currentScore = score;
                current = std::move(candidate);
            } else {
                currentPenalty = std::max(
                    config.simulation.iterative_min_penalty,
                    currentPenalty * config.simulation.iterative_collapse_backoff);
            }

            const bool improvedBest =
                currentScore > bestScore + config.simulation.iterative_improvement_tolerance;
            if (improvedBest) {
                bestScore = currentScore;
                bestSequence = cloneStack(current);
                noImprovementCount = 0;
            } else {
                ++noImprovementCount;
            }

            if (count % 50 == 0)
            {
                trialLog << "[IterPreprocess] radius=" << scoringRadius
                         << " round=" << count
                         << " score=" << score
                         << " current_score=" << currentScore
                         << " best_score=" << bestScore
                         << " gate=" << selectedGate
                         << " gate_probability="
                         << gateProbabilities[static_cast<std::size_t>(selectedGateIndex)]
                         << std::endl;
            }

            if (bestScore >= config.simulation.iterative_score_max)
            {
                break;
            }

            if (noImprovementCount >= noImprovementPatience) {
                trialLog << "[IterPreprocess] radius=" << scoringRadius
                         << " stop=no_improvement"
                         << " round=" << count
                         << " patience=" << noImprovementPatience
                         << " best_score=" << bestScore
                         << std::endl;
                break;
            }
        }

        trialLog << "[IterPreprocess] radius=" << scoringRadius
                 << " best_score=" << bestScore << std::endl;
        return PreprocessTrialResult{
            std::move(bestSequence),
            bestScore,
            scoringRadius,
            trialLog.str()
        };
    };

    log << "[IterPreprocess] radius_trials=" << scoringRadii.size()
        << " radius_min=" << minRadius
        << " radius_max=" << maxRadius
        << " radius_step=" << radiusStep
        << std::endl;

    std::vector<PreprocessTrialResult> trialResults(scoringRadii.size());
    #pragma omp parallel for schedule(dynamic)
    for (int radiusIndex = 0; radiusIndex < static_cast<int>(scoringRadii.size()); ++radiusIndex)
    {
        trialResults[static_cast<std::size_t>(radiusIndex)] =
            runRadiusTrial(scoringRadii[static_cast<std::size_t>(radiusIndex)], radiusIndex);
    }

    std::size_t bestTrialIndex = 0;
    for (std::size_t i = 1; i < trialResults.size(); ++i) {
        if (trialResults[i].score > trialResults[bestTrialIndex].score) {
            bestTrialIndex = i;
        }
    }

    for (const auto &result : trialResults) {
        log << result.logText;
    }
    ImageStack bestSequence = std::move(trialResults[bestTrialIndex].sequence);
    const float bestScore = trialResults[bestTrialIndex].score;
    log << "[IterPreprocess] selected_radius=" << trialResults[bestTrialIndex].radius
        << " best_score=" << bestScore << std::endl;

    #pragma omp parallel for schedule(static)
    for (int sliceIndex = 0; sliceIndex < static_cast<int>(bestSequence.size()); ++sliceIndex)
    {
        auto &slice = bestSequence[static_cast<std::size_t>(sliceIndex)];
        if (config.simulation.post_process_blur_sigma > 0.0f)
        {
            cv::GaussianBlur(slice,
                             slice,
                             cv::Size(0, 0),
                             config.simulation.post_process_blur_sigma,
                             config.simulation.post_process_blur_sigma);
        }

        for (int y = 0; y < slice.rows; ++y)
        {
            float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x)
            {
                if (row[x] < config.simulation.post_process_black_percentile)
                {
                    row[x] = 0.0f;
                }
                else if (row[x] < config.simulation.post_process_white_percentile)
                {
                    row[x] *= config.simulation.post_process_amplification;
                }
            }
        }

    }

    if (config.simulation.post_process_final_blur_sigma > 0.0f)
    {
        const float directWeight = std::clamp(
            config.simulation.post_process_final_direct_weight,
            0.0f,
            1.0f);
        const float blurredWeight = 1.0f - directWeight;
        const float directAmplification =
            std::max(0.0f, config.simulation.post_process_final_direct_amplification);
        const float blurredAmplification =
            std::max(0.0f, config.simulation.post_process_final_blurred_amplification);

        #pragma omp parallel for schedule(static)
        for (int sliceIndex = 0; sliceIndex < static_cast<int>(bestSequence.size()); ++sliceIndex)
        {
            auto &slice = bestSequence[static_cast<std::size_t>(sliceIndex)];
            cv::Mat directSlice = slice.clone();
            cv::Mat blurredSlice;
            cv::GaussianBlur(slice,
                             blurredSlice,
                             cv::Size(0, 0),
                             config.simulation.post_process_final_blur_sigma,
                             config.simulation.post_process_final_blur_sigma);

            if (directAmplification != 1.0f)
            {
                directSlice *= directAmplification;
            }
            if (blurredAmplification != 1.0f)
            {
                blurredSlice *= blurredAmplification;
            }

            cv::addWeighted(directSlice,
                            directWeight,
                            blurredSlice,
                            blurredWeight,
                            0.0,
                            slice);
        }
    }

    clipStack(bestSequence);
    return bestSequence;
}

std::vector<cv::Mat> ImageHandler::loadRawFrame(const std::string &imageFile,
                                                const BaseConfig &config,
                                                std::ostream *logSink)
{
    std::ostream &log = logSink ? *logSink : std::cout;
    std::vector<cv::Mat> normalizedSlices;

    const std::string extension = imageFile.substr(imageFile.find_last_of('.') + 1);
    if (extension == "tiff" || extension == "tif")
    {
        std::vector<cv::Mat> tiffImage;
        cv::imreadmulti(imageFile, tiffImage, cv::IMREAD_ANYDEPTH | cv::IMREAD_COLOR);

        const auto numTiffSlices = tiffImage.size();
        if (numTiffSlices == 0)
        {
            throw std::runtime_error("TIFF has 0 slices: " + imageFile);
        }

        const cv::Mat &firstSlice = tiffImage.front();
        if (firstSlice.empty())
        {
            std::cout << "Error: Could not read the TIFF image" << '\n';
            return normalizedSlices;
        }

        log << "[LoadFrame] file=" << fs::path(imageFile).filename().string()
            << " rawSlices=" << numTiffSlices
            << " rawType=" << firstSlice.type()
            << " rawChannels=" << firstSlice.channels()
            << " rawRows=" << firstSlice.rows
            << " rawCols=" << firstSlice.cols
            << std::endl;

        normalizedSlices.resize(numTiffSlices);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(numTiffSlices); ++i)
        {
            cv::Mat slice = tiffImage[static_cast<std::size_t>(i)];
            if (slice.channels() == 3)
            {
                cv::cvtColor(slice, slice, cv::COLOR_BGR2GRAY);
            }
            normalizedSlices[static_cast<std::size_t>(i)] = processImage(slice, config);
        }
    }
    else
    {
        cv::Mat image = cv::imread(imageFile);
        if (image.empty())
        {
            std::cout << "Error: Could not read the image" << '\n';
            return normalizedSlices;
        }

        if (image.channels() == 3)
        {
            cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        }

        normalizedSlices.push_back(processImage(image, config));
    }

    printStackStats(log, "normalized_input", imageFile, normalizedSlices);
    return normalizedSlices;
}

std::vector<cv::Mat> ImageHandler::preprocessLoadedFrame(const std::vector<cv::Mat> &normalizedSlices,
                                                         const std::string &imageFile,
                                                         const BaseConfig &config,
                                                         std::ostream *logSink)
{
    std::ostream &log = logSink ? *logSink : std::cout;
    std::vector<cv::Mat> processedZSlices = cloneStack(normalizedSlices);
    std::vector<cv::Mat> interpolatedZSlices;

    processedZSlices = processPreparedSequence(processedZSlices, config, log, imageFile);

    const float localScore = evaluateBestWindowContrastScore(processedZSlices, config);
    log << "[PreprocessScores] file=" << fs::path(imageFile).filename().string()
        << " local=" << localScore
        << std::endl;

    printStackStats(log, "processed_sequence", imageFile, processedZSlices);

    if (processedZSlices.empty())
    {
        return interpolatedZSlices;
    }

    if (processedZSlices.size() == 1)
    {
        interpolatedZSlices = processedZSlices;
    }
    else
    {
        const int expandFactor = config.simulation.z_scaling;
        const unsigned numSynthSlices =
            static_cast<unsigned>(expandFactor) * (processedZSlices.size() - 1U) + 1U;

        interpolatedZSlices.resize(numSynthSlices);
        #pragma omp parallel for schedule(static)
        for (int synthSliceIndex = 0; synthSliceIndex < static_cast<int>(numSynthSlices); ++synthSliceIndex)
        {
            const unsigned synthSlice = static_cast<unsigned>(synthSliceIndex);
            const int sourceSlice = static_cast<int>(synthSlice / expandFactor);
            if (synthSlice % expandFactor == 0)
            {
                interpolatedZSlices[static_cast<std::size_t>(synthSlice)] =
                    processedZSlices[static_cast<std::size_t>(sourceSlice)];
            }
            else
            {
                const double t = static_cast<double>(synthSlice % expandFactor) /
                                 static_cast<double>(expandFactor);
                interpolatedZSlices[static_cast<std::size_t>(synthSlice)] =
                    (1.0 - t) * processedZSlices[static_cast<std::size_t>(sourceSlice)] +
                    t * processedZSlices[static_cast<std::size_t>(sourceSlice + 1)];
            }
        }

        if (interpolatedZSlices.size() != numSynthSlices)
        {
            throw std::runtime_error(
                "interpolatedZSlices must have exactly " + std::to_string(numSynthSlices) +
                " slices, but has " + std::to_string(interpolatedZSlices.size()) + " slices");
        }
    }

    printStackStats(log, "post_interpolation", imageFile, interpolatedZSlices);
    interpolatedZSlices = applyAdaptiveCubePooling(interpolatedZSlices, config, log);
    printStackStats(log, "post_cube_pooling", imageFile, interpolatedZSlices);
    log << std::to_string(interpolatedZSlices.size()) << "slices built successfully" << std::endl;
    return interpolatedZSlices;
}

std::vector<cv::Mat> ImageHandler::loadFrame(const std::string &imageFile,
                                             BaseConfig &config,
                                             std::ostream *logSink)
{
    const std::vector<cv::Mat> normalizedSlices = loadRawFrame(imageFile, config, logSink);
    return preprocessLoadedFrame(normalizedSlices, imageFile, config, logSink);
}

PathVec ImageHandler::getImageFilePaths(const std::string &input, int firstFrame, int lastFrame, BaseConfig &config)
{
    PathVec imagePaths;

    if (input.find('%') != std::string::npos)
    {
        for (int i = firstFrame; lastFrame == -1 || i <= lastFrame; ++i)
        {
            char buffer[1024];
            std::snprintf(buffer, sizeof(buffer), input.c_str(), i);
            fs::path file(buffer);

            if (fs::exists(file) && fs::is_regular_file(file))
            {
                imagePaths.push_back(file);
                continue;
            }

            std::cerr << "Input file not found \"" << file << "\"" << '\n';
            throw std::runtime_error("Input file not found");
        }
    }
    else if (fs::is_directory(input))
    {
        PathVec allFiles;
        for (const auto &entry : fs::directory_iterator(input))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }

            const fs::path &path = entry.path();
            if (shouldIgnoreImagePath(path))
            {
                continue;
            }

            if (path.extension() == ".tif" || path.extension() == ".tiff")
            {
                allFiles.push_back(path);
            }
        }

        if (allFiles.empty())
        {
            throw std::runtime_error("No .tif/.tiff files found in directory: " + input);
        }

        std::sort(allFiles.begin(), allFiles.end());

        if (firstFrame < 0)
        {
            throw std::runtime_error("firstFrame must be >= 0 for directory input");
        }

        const int start = firstFrame;
        const int end = (lastFrame < 0) ? static_cast<int>(allFiles.size()) - 1
                                        : std::min(lastFrame, static_cast<int>(allFiles.size()) - 1);

        if (start >= static_cast<int>(allFiles.size()))
        {
            throw std::runtime_error("firstFrame is out of range for directory input");
        }
        if (start > end)
        {
            throw std::runtime_error("Invalid frame range for directory input");
        }

        for (int i = start; i <= end; ++i)
        {
            imagePaths.push_back(allFiles[i]);
        }
    }
    else if (fs::exists(input) && fs::is_regular_file(input))
    {
        imagePaths.push_back(input);
    }
    else
    {
        throw std::runtime_error("Input is neither a pattern, directory, nor file: " + input);
    }

    if (!imagePaths.empty())
    {
        updateTiffConfigIfNeeded(imagePaths.front(), config);
    }

    for (const auto &path : imagePaths)
    {
        std::cout << path << '\n';
    }

    return imagePaths;
}
