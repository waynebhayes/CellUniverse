#include "../includes/CellUniverse.hpp"
#include "../includes/ImageHandler.hpp"
#include <set>
#include <cmath>
#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <numeric>
#include <utility>
#include <sstream>
#include <cstdint>
#include <queue>
#include <deque>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

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

static cv::Mat makeNapariFriendlyTiffSlice(const cv::Mat &slice)
{
    if (slice.empty()) {
        return {};
    }

    cv::Mat gray;
    if (slice.channels() == 1) {
        gray = slice;
    } else if (slice.channels() == 3) {
        cv::cvtColor(slice, gray, cv::COLOR_BGR2GRAY);
    } else if (slice.channels() == 4) {
        cv::cvtColor(slice, gray, cv::COLOR_BGRA2GRAY);
    } else {
        throw std::runtime_error("Unsupported channel count for TIFF export: " +
                                 std::to_string(slice.channels()));
    }

    cv::Mat output;
    if (gray.depth() == CV_8U) {
        output = gray.clone();
    } else {
        cv::Mat clipped = gray.clone();
        cv::patchNaNs(clipped, 0.0);
        cv::min(clipped, 1.0f, clipped);
        cv::max(clipped, 0.0f, clipped);
        clipped.convertTo(output, CV_8U, 255.0);
    }
    return output;
}

static std::vector<cv::Mat> makeNapariFriendlyTiffStack(const std::vector<cv::Mat> &stack)
{
    std::vector<cv::Mat> output;
    output.reserve(stack.size());

    cv::Size expectedSize;
    for (const auto &slice : stack) {
        cv::Mat converted = makeNapariFriendlyTiffSlice(slice);
        if (converted.empty()) {
            continue;
        }
        if (expectedSize.empty()) {
            expectedSize = converted.size();
        } else if (converted.size() != expectedSize) {
            throw std::runtime_error("TIFF export requires all slices to have the same size");
        }
        output.push_back(std::move(converted));
    }

    if (output.empty()) {
        throw std::runtime_error("TIFF export received an empty image stack");
    }
    return output;
}

static void writeNapariFriendlyTiffStack(const std::string &path,
                                         const std::vector<cv::Mat> &stack)
{
    std::vector<cv::Mat> output = makeNapariFriendlyTiffStack(stack);
    const std::vector<int> params = {
        cv::IMWRITE_TIFF_COMPRESSION, 1 // COMPRESSION_NONE: easiest for TIFF readers.
    };

    if (!cv::imwritemulti(path, output, params)) {
        throw std::runtime_error("Failed to write TIFF stack: " + path);
    }
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
// Used by the per-frame percentile normalization step.

static float computeStackPercentile(const std::vector<cv::Mat> &stack,
                                    float percentile,
                                    bool excludeZeros = false)
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
            for (int x = 0; x < slice.cols; ++x) {
                const float value = row[x];
                if (!std::isfinite(value)) {
                    continue;
                }
                if (excludeZeros && value == 0.0f) {
                    continue;
                }
                values.push_back(value);
            }
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

static void normalizeStackToFrameScale(std::vector<cv::Mat> &stack,
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

static std::pair<float, float> normalizeStackToFrameIntensity(std::vector<cv::Mat> &stack,
                                                              const SimulationConfig &config)
{
    float lowReference = computeStackPercentile(
        stack,
        config.frame_intensity_scale_low_percentile,
        config.frame_intensity_percentile_exclude_zeros);
    float highReference = computeStackPercentile(
        stack,
        config.frame_intensity_scale_high_percentile,
        config.frame_intensity_percentile_exclude_zeros);
    if (config.frame_intensity_hard_max > 0.0f &&
        highReference > config.frame_intensity_hard_max) {
        highReference = config.frame_intensity_hard_max;
    }

    if (highReference <= lowReference + 1e-6f) {
        const float fallback = computeStackMax(stack);
        if (fallback > lowReference + 1e-6f) {
            highReference = fallback;
            if (config.frame_intensity_hard_max > 0.0f) {
                highReference = std::min(highReference, config.frame_intensity_hard_max);
            }
        } else {
            lowReference = 0.0f;
            highReference = 1.0f;
            if (config.frame_intensity_hard_max > 0.0f) {
                highReference = std::min(highReference, config.frame_intensity_hard_max);
            }
        }
    }

    normalizeStackToFrameScale(stack,
                               lowReference,
                               highReference,
                               config.frame_intensity_hard_max);
    return {lowReference, highReference};
}

static int clampEdgeLimit(int requestedLimit, int axisLength)
{
    if (axisLength <= 0) {
        return 0;
    }
    return std::clamp(requestedLimit, 0, axisLength);
}

static float computeEdgeBrightnessMean(const std::vector<cv::Mat> &stack,
                                       const SimulationConfig &config)
{
    double sum = 0.0;
    double count = 0.0;

    for (const auto &slice : stack) {
        if (slice.empty()) continue;
        CV_Assert(slice.type() == CV_32F);

        const int margin = std::max(1, config.edge_brightness_alignment_xy_margin);
        const int leftOffset = std::max(0, config.edge_brightness_alignment_left_offset);
        const int rightOffset = std::max(0, config.edge_brightness_alignment_right_offset);
        const int topOffset = std::max(0, config.edge_brightness_alignment_top_offset);
        const int bottomOffset = std::max(0, config.edge_brightness_alignment_bottom_offset);
        const int xInnerStart = clampEdgeLimit(leftOffset, slice.cols);
        const int xInnerEnd = clampEdgeLimit(leftOffset + margin, slice.cols);
        const int xOuterStart = clampEdgeLimit(slice.cols - rightOffset - margin, slice.cols);
        const int xOuterEnd = clampEdgeLimit(slice.cols - rightOffset, slice.cols);
        const int yInnerStart = clampEdgeLimit(topOffset, slice.rows);
        const int yInnerEnd = clampEdgeLimit(topOffset + margin, slice.rows);
        const int yOuterStart = clampEdgeLimit(slice.rows - bottomOffset - margin, slice.rows);
        const int yOuterEnd = clampEdgeLimit(slice.rows - bottomOffset, slice.rows);

        for (int y = 0; y < slice.rows; ++y) {
            const float *row = slice.ptr<float>(y);
            const bool inYBand =
                (y >= yInnerStart && y < yInnerEnd) ||
                (y >= yOuterStart && y < yOuterEnd);
            for (int x = 0; x < slice.cols; ++x) {
                const bool inXBand =
                    (x >= xInnerStart && x < xInnerEnd) ||
                    (x >= xOuterStart && x < xOuterEnd);
                if (!inYBand && !inXBand) {
                    continue;
                }
                const float value = row[x];
                if (!std::isfinite(value)) {
                    continue;
                }
                sum += value;
                count += 1.0;
            }
        }
    }

    return count > 0.0 ? static_cast<float>(sum / count) : 0.0f;
}

static void alignStackToEdgeBrightness(std::vector<cv::Mat> &stack,
                                       const SimulationConfig &config,
                                       float targetEdgeMean,
                                       const fs::path &framePath,
                                       std::ostream &log)
{
    if (!config.edge_brightness_alignment_enabled || stack.empty()) {
        return;
    }

    const float edgeMean = computeEdgeBrightnessMean(
        stack, config);
    const float maxShift = std::max(0.0f, config.edge_brightness_alignment_max_shift);
    const float shift = std::clamp(edgeMean - targetEdgeMean, -maxShift, maxShift);
    if (std::abs(shift) <= 1e-6f) {
        log << "[EdgeBrightnessAlignment] frame=" << framePath.filename().string()
            << " edge_mean=" << edgeMean
            << " target=" << targetEdgeMean
            << " shift=0"
            << " xy_margin=" << std::max(1, config.edge_brightness_alignment_xy_margin)
            << " offsets=("
            << std::max(0, config.edge_brightness_alignment_left_offset) << ","
            << std::max(0, config.edge_brightness_alignment_right_offset) << ","
            << std::max(0, config.edge_brightness_alignment_top_offset) << ","
            << std::max(0, config.edge_brightness_alignment_bottom_offset) << ")"
            << std::endl;
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int sliceIndex = 0; sliceIndex < static_cast<int>(stack.size()); ++sliceIndex) {
        auto &slice = stack[static_cast<size_t>(sliceIndex)];
        if (slice.empty()) continue;
        slice -= shift;
        cv::min(slice, 1.0f, slice);
        cv::max(slice, 0.0f, slice);
    }

    const float alignedEdgeMean = computeEdgeBrightnessMean(
        stack, config);
    log << "[EdgeBrightnessAlignment] frame=" << framePath.filename().string()
        << " edge_mean=" << edgeMean
        << " target=" << targetEdgeMean
        << " shift=" << shift
        << " aligned_edge_mean=" << alignedEdgeMean
        << " xy_margin=" << std::max(1, config.edge_brightness_alignment_xy_margin)
        << " offsets=("
        << std::max(0, config.edge_brightness_alignment_left_offset) << ","
        << std::max(0, config.edge_brightness_alignment_right_offset) << ","
        << std::max(0, config.edge_brightness_alignment_top_offset) << ","
        << std::max(0, config.edge_brightness_alignment_bottom_offset) << ")"
        << " max_shift=" << maxShift
        << std::endl;
}

static void blackThresholdStackAfterAlignment(std::vector<cv::Mat> &stack,
                                              const SimulationConfig &config,
                                              const fs::path &framePath,
                                              std::ostream &log)
{
    const float threshold = std::max(0.0f, config.post_alignment_black_threshold);
    if (threshold <= 0.0f || stack.empty()) {
        return;
    }

    std::size_t changedCount = 0;
    std::size_t totalCount = 0;
    #pragma omp parallel for schedule(static) reduction(+:changedCount,totalCount)
    for (int sliceIndex = 0; sliceIndex < static_cast<int>(stack.size()); ++sliceIndex) {
        auto &slice = stack[static_cast<size_t>(sliceIndex)];
        if (slice.empty()) continue;
        CV_Assert(slice.type() == CV_32F);
        for (int y = 0; y < slice.rows; ++y) {
            float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x) {
                ++totalCount;
                if (row[x] < threshold) {
                    if (row[x] != 0.0f) {
                        ++changedCount;
                    }
                    row[x] = 0.0f;
                }
            }
        }
    }

    const double changedFraction = totalCount > 0
        ? static_cast<double>(changedCount) / static_cast<double>(totalCount)
        : 0.0;
    log << "[PostAlignmentBlackThreshold] frame=" << framePath.filename().string()
        << " threshold=" << threshold
        << " changed_fraction=" << changedFraction
        << std::endl;
}

static std::vector<cv::Mat> cloneMatStack(const std::vector<cv::Mat> &stack)
{
    std::vector<cv::Mat> cloned(stack.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(stack.size()); ++i) {
        cloned[static_cast<std::size_t>(i)] = stack[static_cast<std::size_t>(i)].clone();
    }
    return cloned;
}

static float percentileValue(std::vector<float> values, float percentile)
{
    if (values.empty()) {
        return 0.0f;
    }
    percentile = std::clamp(percentile, 0.0f, 100.0f);
    const float rank = (percentile / 100.0f) * static_cast<float>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(rank));
    const size_t hi = static_cast<size_t>(std::ceil(rank));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(lo), values.end());
    const float loVal = values[lo];
    if (hi == lo) {
        return loVal;
    }
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(hi), values.end());
    const float hiVal = values[hi];
    return loVal + (hiVal - loVal) * (rank - static_cast<float>(lo));
}

static std::vector<cv::Mat> gaussianBlurStack3D(const std::vector<cv::Mat> &stack,
                                                float sigma)
{
    if (stack.empty() || sigma <= 0.0f) {
        return cloneMatStack(stack);
    }

    std::vector<cv::Mat> xyBlurred(stack.size());
    #pragma omp parallel for schedule(static)
    for (int z = 0; z < static_cast<int>(stack.size()); ++z) {
        cv::GaussianBlur(stack[static_cast<size_t>(z)],
                         xyBlurred[static_cast<size_t>(z)],
                         cv::Size(0, 0),
                         sigma,
                         sigma,
                         cv::BORDER_REPLICATE);
    }

    const int radius = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    std::vector<float> kernel(static_cast<size_t>(2 * radius + 1));
    float kernelSum = 0.0f;
    for (int dz = -radius; dz <= radius; ++dz) {
        const float w = std::exp(-0.5f * static_cast<float>(dz * dz) / (sigma * sigma));
        kernel[static_cast<size_t>(dz + radius)] = w;
        kernelSum += w;
    }
    for (float &w : kernel) {
        w /= std::max(kernelSum, 1e-12f);
    }

    std::vector<cv::Mat> blurred(stack.size());
    #pragma omp parallel for schedule(static)
    for (int z = 0; z < static_cast<int>(xyBlurred.size()); ++z) {
        blurred[static_cast<size_t>(z)] = cv::Mat::zeros(xyBlurred[static_cast<size_t>(z)].size(), CV_32F);
        for (int dz = -radius; dz <= radius; ++dz) {
            const int zz = std::clamp(z + dz, 0, static_cast<int>(xyBlurred.size()) - 1);
            blurred[static_cast<size_t>(z)] +=
                kernel[static_cast<size_t>(dz + radius)] * xyBlurred[static_cast<size_t>(zz)];
        }
    }
    return blurred;
}

static std::vector<cv::Mat> buildSignalMapStack(const std::vector<cv::Mat> &realFrame,
                                                const SimulationConfig &simulation,
                                                const fs::path &framePath,
                                                std::ostream &log)
{
    std::vector<cv::Mat> signalMap = cloneMatStack(realFrame);
    if (!simulation.signal_map_enabled || signalMap.empty()) {
        return {};
    }

    std::vector<float> nonzeroValues;
    for (const auto &slice : realFrame) {
        for (int y = 0; y < slice.rows; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x) {
                if (row[x] > 0.0f) {
                    nonzeroValues.push_back(row[x]);
                }
            }
        }
    }

    const float threshold = nonzeroValues.empty()
        ? 0.0f
        : percentileValue(std::move(nonzeroValues), simulation.signal_map_bright_center_percentile);
    double targetSum = 0.0;
    std::size_t targetCount = 0;
    for (const auto &slice : realFrame) {
        for (int y = 0; y < slice.rows; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x) {
                if (row[x] >= threshold) {
                    targetSum += row[x];
                    ++targetCount;
                }
            }
        }
    }

    const float eps = std::max(simulation.signal_map_epsilon, 1e-12f);
    const double targetMean = targetCount > 0
        ? targetSum / static_cast<double>(targetCount)
        : 0.0;
    const int iterations = std::max(0, simulation.signal_map_max_iterations);
    for (int iter = 0; iter < iterations; ++iter) {
        signalMap = gaussianBlurStack3D(signalMap, simulation.signal_map_blur_sigma);
        double blurredSum = 0.0;
        std::size_t blurredCount = 0;
        for (size_t z = 0; z < signalMap.size(); ++z) {
            const auto &slice = signalMap[z];
            const auto &referenceSlice = realFrame[z];
            for (int y = 0; y < slice.rows; ++y) {
                const float *row = slice.ptr<float>(y);
                const float *referenceRow = referenceSlice.ptr<float>(y);
                for (int x = 0; x < slice.cols; ++x) {
                    if (referenceRow[x] >= threshold) {
                        blurredSum += row[x];
                        ++blurredCount;
                    }
                }
            }
        }
        const double blurredMean = blurredCount > 0
            ? blurredSum / static_cast<double>(blurredCount)
            : 0.0;
        const float recoveryFactor = targetMean > eps
            ? static_cast<float>(targetMean / std::max(blurredMean, static_cast<double>(eps)))
            : 1.0f;
        #pragma omp parallel for schedule(static)
        for (int z = 0; z < static_cast<int>(signalMap.size()); ++z) {
            signalMap[static_cast<size_t>(z)] *= recoveryFactor;
        }
    }

    log << "[Signal Map] frame=" << framePath.filename().string()
        << " enabled=1"
        << " sigma=" << simulation.signal_map_blur_sigma
        << " iterations=" << iterations
        << " bright_center_percentile=" << simulation.signal_map_bright_center_percentile
        << " target_mean=" << targetMean
        << " threshold=" << threshold
        << std::endl;
    return signalMap;
}

struct BlackPercentileResult
{
    bool applied = false;
    float percentile = 0.0f;
    float cutoff = 0.0f;
    std::size_t nonzeroSampleCount = 0;
    double changedFraction = 0.0;
};

static BlackPercentileResult blackPercentileStackAfterAlignment(std::vector<cv::Mat> &stack,
                                                                const SimulationConfig &config,
                                                                const fs::path &framePath,
                                                                std::ostream &log,
                                                                float percentileOverride = -1.0f)
{
    BlackPercentileResult result;
    const float requestedPercentile = percentileOverride >= 0.0f
        ? percentileOverride
        : config.post_alignment_black_percentile;
    const float percentile = std::clamp(requestedPercentile, 0.0f, 1.0f);
    result.percentile = percentile;
    if (percentile <= 0.0f || stack.empty()) {
        return result;
    }

    std::vector<float> values;
    std::size_t totalCount = 0;
    for (const auto &slice : stack) {
        if (slice.empty()) continue;
        CV_Assert(slice.type() == CV_32F);
        totalCount += slice.total();
    }
    values.reserve(totalCount);

    for (const auto &slice : stack) {
        if (slice.empty()) continue;
        for (int y = 0; y < slice.rows; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x) {
                const float value = row[x];
                if (std::isfinite(value) && value > 0.0f) {
                    values.push_back(value);
                }
            }
        }
    }

    if (values.empty()) {
        return result;
    }

    const std::size_t cutoffIndex = static_cast<std::size_t>(
        std::floor(percentile * static_cast<float>(values.size() - 1)));
    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(cutoffIndex),
                     values.end());
    const float cutoff = values[cutoffIndex];
    result.cutoff = cutoff;
    result.nonzeroSampleCount = values.size();

    std::size_t changedCount = 0;
    std::size_t finiteCount = 0;
    #pragma omp parallel for schedule(static) reduction(+:changedCount,finiteCount)
    for (int sliceIndex = 0; sliceIndex < static_cast<int>(stack.size()); ++sliceIndex) {
        auto &slice = stack[static_cast<size_t>(sliceIndex)];
        if (slice.empty()) continue;
        CV_Assert(slice.type() == CV_32F);
        for (int y = 0; y < slice.rows; ++y) {
            float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x) {
                if (!std::isfinite(row[x])) {
                    continue;
                }
                ++finiteCount;
                if (row[x] <= cutoff) {
                    if (row[x] != 0.0f) {
                        ++changedCount;
                    }
                    row[x] = 0.0f;
                }
            }
        }
    }

    const double changedFraction = finiteCount > 0
        ? static_cast<double>(changedCount) / static_cast<double>(finiteCount)
        : 0.0;
    result.applied = true;
    result.changedFraction = changedFraction;
    log << "[PostAlignmentBlackPercentile] frame=" << framePath.filename().string()
        << " percentile=" << percentile
        << " cutoff=" << cutoff
        << " nonzero_sample_count=" << values.size()
        << " changed_fraction=" << changedFraction
        << std::endl;
    return result;
}

static int countSeparatedChunksInSizeRange(const std::vector<cv::Mat> &stack,
                                           const SimulationConfig &config,
                                           int stopAfterCount = -1)
{
    if (stack.empty()) {
        return 0;
    }

    const int depth = static_cast<int>(stack.size());
    const int rows = stack.front().rows;
    const int cols = stack.front().cols;
    if (depth <= 0 || rows <= 0 || cols <= 0) {
        return 0;
    }
    for (const auto &slice : stack) {
        if (slice.empty() || slice.rows != rows || slice.cols != cols) {
            return 0;
        }
        CV_Assert(slice.type() == CV_32F);
    }

    const int minSize = std::max(1, config.post_alignment_chunk_min_size);
    const int maxSize = std::max(minSize, config.post_alignment_chunk_max_size);
    const std::size_t planeSize = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    const std::size_t voxelCount = static_cast<std::size_t>(depth) * planeSize;
    std::vector<std::uint8_t> foreground(voxelCount, 0U);
    std::vector<std::uint8_t> visited(voxelCount, 0U);
    std::vector<std::size_t> pending;
    int matchingChunkCount = 0;

    auto indexOf = [&](int z, int y, int x) {
        return static_cast<std::size_t>(z) * planeSize +
               static_cast<std::size_t>(y) * static_cast<std::size_t>(cols) +
               static_cast<std::size_t>(x);
    };

    int detectorThreads = std::max(1, config.post_alignment_chunk_detector_threads);
#ifdef _OPENMP
    detectorThreads = std::min(detectorThreads, std::max(1, omp_get_max_threads()));
    if (omp_in_parallel()) {
        detectorThreads = 1;
    }
#endif

    #pragma omp parallel for schedule(static) num_threads(detectorThreads)
    for (int z = 0; z < depth; ++z) {
        const cv::Mat &slice = stack[static_cast<std::size_t>(z)];
        for (int y = 0; y < rows; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < cols; ++x) {
                const float value = row[x];
                foreground[indexOf(z, y, x)] =
                    (std::isfinite(value) && value > 0.0f) ? 1U : 0U;
            }
        }
    }

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                const std::size_t startIndex = indexOf(z, y, x);
                if (visited[startIndex] || !foreground[startIndex]) {
                    visited[startIndex] = 1U;
                    continue;
                }

                int minZ = z;
                int maxZ = z;
                int minY = y;
                int maxY = y;
                int minX = x;
                int maxX = x;
                visited[startIndex] = 1U;
                pending.clear();
                pending.push_back(startIndex);

                while (!pending.empty()) {
                    const std::size_t currentIndex = pending.back();
                    pending.pop_back();
                    const int currentZ = static_cast<int>(currentIndex / planeSize);
                    const std::size_t inPlane = currentIndex % planeSize;
                    const int currentY = static_cast<int>(inPlane / static_cast<std::size_t>(cols));
                    const int currentX = static_cast<int>(inPlane % static_cast<std::size_t>(cols));

                    minZ = std::min(minZ, currentZ);
                    maxZ = std::max(maxZ, currentZ);
                    minY = std::min(minY, currentY);
                    maxY = std::max(maxY, currentY);
                    minX = std::min(minX, currentX);
                    maxX = std::max(maxX, currentX);

                    for (int dz = -1; dz <= 1; ++dz) {
                        const int nz = currentZ + dz;
                        if (nz < 0 || nz >= depth) continue;
                        for (int dy = -1; dy <= 1; ++dy) {
                            const int ny = currentY + dy;
                            if (ny < 0 || ny >= rows) continue;
                            for (int dx = -1; dx <= 1; ++dx) {
                                if (dz == 0 && dy == 0 && dx == 0) continue;
                                const int nx = currentX + dx;
                                if (nx < 0 || nx >= cols) continue;

                                const std::size_t neighborIndex = indexOf(nz, ny, nx);
                                if (visited[neighborIndex]) continue;
                                visited[neighborIndex] = 1U;
                                if (foreground[neighborIndex]) {
                                    pending.push_back(neighborIndex);
                                }
                            }
                        }
                    }
                }

                const int zSize = maxZ - minZ + 1;
                const int ySize = maxY - minY + 1;
                const int xSize = maxX - minX + 1;
                if (zSize >= minSize && zSize <= maxSize &&
                    ySize >= minSize && ySize <= maxSize &&
                    xSize >= minSize && xSize <= maxSize) {
                    ++matchingChunkCount;
                    if (stopAfterCount > 0 && matchingChunkCount >= stopAfterCount) {
                        return matchingChunkCount;
                    }
                }
            }
        }
    }

    return matchingChunkCount;
}

static void removeTinyIsolatedParticles(std::vector<cv::Mat> &stack,
                                        const BaseConfig &config,
                                        const fs::path &framePath,
                                        std::ostream &log)
{
    if (!config.simulation.post_alignment_tiny_particle_removal_enabled || stack.empty()) {
        return;
    }

    const int depth = static_cast<int>(stack.size());
    const int rows = stack.front().rows;
    const int cols = stack.front().cols;
    if (depth <= 0 || rows <= 0 || cols <= 0) {
        return;
    }
    for (const auto &slice : stack) {
        if (slice.empty() || slice.rows != rows || slice.cols != cols) {
            return;
        }
        CV_Assert(slice.type() == CV_32F);
    }

    const double minARadius = config.cell ? config.cell->minARadius : 5.0;
    const double minBRadius = config.cell
        ? (config.cell->minBRadius > 0.0 ? config.cell->minBRadius : config.cell->minARadius)
        : 5.0;
    const double minCRadius = config.cell ? config.cell->minCRadius : 5.0;
    const int minXSize = std::max(1, static_cast<int>(std::ceil(2.0 * minARadius)));
    const int minYSize = std::max(1, static_cast<int>(std::ceil(2.0 * minBRadius)));
    const int minZSize = std::max(1, static_cast<int>(std::ceil(2.0 * minCRadius)));

    const std::size_t planeSize = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    const std::size_t voxelCount = static_cast<std::size_t>(depth) * planeSize;
    std::vector<std::uint8_t> foreground(voxelCount, 0U);
    std::vector<std::uint8_t> visited(voxelCount, 0U);
    std::vector<std::size_t> pending;
    std::vector<std::size_t> componentVoxels;

    auto indexOf = [&](int z, int y, int x) {
        return static_cast<std::size_t>(z) * planeSize +
               static_cast<std::size_t>(y) * static_cast<std::size_t>(cols) +
               static_cast<std::size_t>(x);
    };

    int detectorThreads = std::max(1, config.simulation.post_alignment_chunk_detector_threads);
#ifdef _OPENMP
    detectorThreads = std::min(detectorThreads, std::max(1, omp_get_max_threads()));
    if (omp_in_parallel()) {
        detectorThreads = 1;
    }
#endif

    #pragma omp parallel for schedule(static) num_threads(detectorThreads)
    for (int z = 0; z < depth; ++z) {
        const cv::Mat &slice = stack[static_cast<std::size_t>(z)];
        for (int y = 0; y < rows; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < cols; ++x) {
                const float value = row[x];
                foreground[indexOf(z, y, x)] =
                    (std::isfinite(value) && value > 0.0f) ? 1U : 0U;
            }
        }
    }

    int removedComponentCount = 0;
    std::size_t removedVoxelCount = 0;
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                const std::size_t startIndex = indexOf(z, y, x);
                if (visited[startIndex] || !foreground[startIndex]) {
                    visited[startIndex] = 1U;
                    continue;
                }

                int minZ = z;
                int maxZ = z;
                int minY = y;
                int maxY = y;
                int minX = x;
                int maxX = x;
                visited[startIndex] = 1U;
                pending.clear();
                componentVoxels.clear();
                pending.push_back(startIndex);

                while (!pending.empty()) {
                    const std::size_t currentIndex = pending.back();
                    pending.pop_back();
                    componentVoxels.push_back(currentIndex);
                    const int currentZ = static_cast<int>(currentIndex / planeSize);
                    const std::size_t inPlane = currentIndex % planeSize;
                    const int currentY = static_cast<int>(inPlane / static_cast<std::size_t>(cols));
                    const int currentX = static_cast<int>(inPlane % static_cast<std::size_t>(cols));

                    minZ = std::min(minZ, currentZ);
                    maxZ = std::max(maxZ, currentZ);
                    minY = std::min(minY, currentY);
                    maxY = std::max(maxY, currentY);
                    minX = std::min(minX, currentX);
                    maxX = std::max(maxX, currentX);

                    for (int dz = -1; dz <= 1; ++dz) {
                        const int nz = currentZ + dz;
                        if (nz < 0 || nz >= depth) continue;
                        for (int dy = -1; dy <= 1; ++dy) {
                            const int ny = currentY + dy;
                            if (ny < 0 || ny >= rows) continue;
                            for (int dx = -1; dx <= 1; ++dx) {
                                if (dz == 0 && dy == 0 && dx == 0) continue;
                                const int nx = currentX + dx;
                                if (nx < 0 || nx >= cols) continue;

                                const std::size_t neighborIndex = indexOf(nz, ny, nx);
                                if (visited[neighborIndex]) continue;
                                visited[neighborIndex] = 1U;
                                if (foreground[neighborIndex]) {
                                    pending.push_back(neighborIndex);
                                }
                            }
                        }
                    }
                }

                const int zSize = maxZ - minZ + 1;
                const int ySize = maxY - minY + 1;
                const int xSize = maxX - minX + 1;
                if (zSize < minZSize && ySize < minYSize && xSize < minXSize) {
                    ++removedComponentCount;
                    removedVoxelCount += componentVoxels.size();
                    for (const std::size_t voxelIndex : componentVoxels) {
                        const int voxelZ = static_cast<int>(voxelIndex / planeSize);
                        const std::size_t inPlane = voxelIndex % planeSize;
                        const int voxelY = static_cast<int>(inPlane / static_cast<std::size_t>(cols));
                        const int voxelX = static_cast<int>(inPlane % static_cast<std::size_t>(cols));
                        stack[static_cast<std::size_t>(voxelZ)].ptr<float>(voxelY)[voxelX] = 0.0f;
                    }
                }
            }
        }
    }

    log << "[PostAlignmentTinyParticleRemoval] frame=" << framePath.filename().string()
        << " removed_components=" << removedComponentCount
        << " removed_voxels=" << removedVoxelCount
        << " min_bbox=(" << minZSize << "," << minYSize << "," << minXSize << ")"
        << " detector_threads=" << detectorThreads
        << std::endl;
}

static void adaptBlackPercentileToChunkCount(std::vector<cv::Mat> &stack,
                                             const std::vector<cv::Mat> &unblackoffedStack,
                                             const SimulationConfig &config,
                                             const fs::path &framePath,
                                             std::ostream &log)
{
    if (!config.post_alignment_chunk_blackoff_enabled || stack.empty()) {
        return;
    }

    const int targetCount = std::max(0, config.post_alignment_chunk_target_count);
    const float percentileStep = std::max(0.0f, config.post_alignment_chunk_percentile_step);
    const float maxPercentile = std::clamp(config.post_alignment_chunk_max_percentile, 0.0f, 1.0f);
    const int nonImprovementPatience = std::max(0, config.post_alignment_chunk_non_improvement_patience);
    const int disableBelowCount = std::max(0, config.post_alignment_chunk_disable_below_count);
    float currentPercentile = std::clamp(config.post_alignment_black_percentile, 0.0f, maxPercentile);
    int chunkCount = countSeparatedChunksInSizeRange(stack, config);
    int bestChunkCount = chunkCount;
    int nonImprovementCount = 0;
    int iteration = 0;

    log << "[PostAlignmentChunkBlackoff] frame=" << framePath.filename().string()
        << " iteration=" << iteration
        << " percentile=" << currentPercentile
        << " chunk_count=" << chunkCount
        << " target_count=" << targetCount
        << " min_size=" << std::max(1, config.post_alignment_chunk_min_size)
        << " max_size=" << std::max(std::max(1, config.post_alignment_chunk_min_size),
                                    config.post_alignment_chunk_max_size)
        << " non_improvement_patience=" << nonImprovementPatience
        << " disable_below_count=" << disableBelowCount
        << " detector_threads=" << std::max(1, config.post_alignment_chunk_detector_threads)
        << std::endl;

    if (chunkCount < disableBelowCount) {
        log << "[PostAlignmentChunkBlackoff] frame=" << framePath.filename().string()
            << " disabled_reason=chunk_count_below_configured_threshold"
            << " chunk_count=" << chunkCount
            << " disable_below_count=" << disableBelowCount
            << std::endl;
        return;
    }

    while (chunkCount > targetCount &&
           percentileStep > 0.0f &&
           currentPercentile + 1e-6f < maxPercentile &&
           nonImprovementCount < nonImprovementPatience) {
        currentPercentile = std::min(maxPercentile, currentPercentile + percentileStep);
        ++iteration;
        stack = cloneMatStack(unblackoffedStack);
        const BlackPercentileResult result =
            blackPercentileStackAfterAlignment(stack, config, framePath, log, currentPercentile);
        if (!result.applied) {
            break;
        }
        chunkCount = countSeparatedChunksInSizeRange(stack, config);
        if (chunkCount < bestChunkCount) {
            bestChunkCount = chunkCount;
            nonImprovementCount = 0;
        } else {
            ++nonImprovementCount;
        }
        log << "[PostAlignmentChunkBlackoff] frame=" << framePath.filename().string()
            << " iteration=" << iteration
            << " percentile=" << currentPercentile
            << " chunk_count=" << chunkCount
            << " target_count=" << targetCount
            << " best_chunk_count=" << bestChunkCount
            << " non_improvement_count=" << nonImprovementCount
            << " non_improvement_patience=" << nonImprovementPatience
            << std::endl;
    }
}

static void applyFinalPreprocessingBlur(std::vector<cv::Mat> &stack,
                                        const SimulationConfig &config,
                                        const fs::path &framePath,
                                        std::ostream &log)
{
    const float sigma = std::max(0.0f, config.post_alignment_final_blur_sigma);
    if (sigma <= 0.0f || stack.empty()) {
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int sliceIndex = 0; sliceIndex < static_cast<int>(stack.size()); ++sliceIndex) {
        auto &slice = stack[static_cast<size_t>(sliceIndex)];
        if (slice.empty()) {
            continue;
        }

        cv::GaussianBlur(slice, slice, cv::Size(0, 0), sigma, sigma);
        cv::patchNaNs(slice, 0.0);
        cv::min(slice, 1.0f, slice);
        cv::max(slice, 0.0f, slice);
    }

    log << "[PostAlignmentFinalBlur] frame=" << framePath.filename().string()
        << " sigma=" << sigma
        << " slices=" << stack.size()
        << std::endl;
}

static void exportStackToSubdir(const std::vector<cv::Mat> &stack,
                                const fs::path &baseOutputDir,
                                const fs::path &subdir,
                                const fs::path &framePath,
                                bool exportPng,
                                bool exportTiff)
{
    const fs::path outputDir = baseOutputDir / subdir;

    if (exportPng) {
        const fs::path frameOutputDir = outputDir / framePath.stem();
        fs::create_directories(frameOutputDir);

        for (size_t i = 0; i < stack.size(); ++i) {
            if (stack[i].empty()) {
                continue;
            }

            cv::Mat outputImage;
            if (stack[i].depth() != CV_8U) {
                cv::Mat clipped = stack[i].clone();
                cv::patchNaNs(clipped, 0.0);
                cv::min(clipped, 1.0f, clipped);
                cv::max(clipped, 0.0f, clipped);
                clipped.convertTo(outputImage, CV_8U, 255.0);
            } else {
                outputImage = stack[i].clone();
            }

            const fs::path outputFile = frameOutputDir / (std::to_string(i) + ".png");
            cv::imwrite(outputFile.string(), outputImage);
        }
    }

    if (exportTiff) {
        fs::create_directories(outputDir);
        const fs::path outputFile = outputDir / (framePath.stem().string() + ".tif");
        writeNapariFriendlyTiffStack(outputFile.string(), stack);
    }
}

static void exportPreprocessedStack(const std::vector<cv::Mat> &stack,
                                    const fs::path &baseOutputDir,
                                    const fs::path &framePath,
                                    bool exportPng,
                                    bool exportTiff)
{
    exportStackToSubdir(stack, baseOutputDir, "preprocessed", framePath,
                        exportPng, exportTiff);
}

static std::vector<cv::Mat> makeEmptyDebugStackLike(const std::vector<cv::Mat> &realFrame)
{
    std::vector<cv::Mat> stack;
    stack.reserve(realFrame.size());
    for (const auto &slice : realFrame) {
        stack.emplace_back(cv::Mat::zeros(slice.size(), CV_32F));
    }
    return stack;
}

static void accumulateDebugCellPlacement(std::vector<cv::Mat> &stack,
                                         const Ellipsoid &cell,
                                         const SimulationConfig &simulationConfig,
                                         float brightness)
{
    if (stack.empty()) {
        return;
    }

    Ellipsoid debugCell = cell;
    debugCell.setBrightness(std::max(0.0f, brightness));
    const float maxR = std::max({debugCell.getARadius(),
                                 debugCell.getBRadius(),
                                 debugCell.getCRadius()});
    const int zMin = std::max(0, static_cast<int>(std::floor(debugCell.getZ() - maxR)));
    const int zMax = std::min(static_cast<int>(stack.size()) - 1,
                              static_cast<int>(std::ceil(debugCell.getZ() + maxR)));
    for (int z = zMin; z <= zMax; ++z) {
        cv::Mat temp = cv::Mat::zeros(stack[static_cast<size_t>(z)].size(), CV_32F);
        debugCell.draw(temp, simulationConfig, static_cast<float>(z));
        stack[static_cast<size_t>(z)] += temp;
    }
}

static std::vector<cv::Mat> makeCellCenterLabelDebugStack(
    const std::vector<cv::Mat> &realFrame,
    const std::vector<Ellipsoid> &cells,
    int cubeRadius)
{
    std::vector<cv::Mat> stack = makeEmptyDebugStackLike(realFrame);
    if (stack.empty()) {
        return stack;
    }

    const int radius = std::max(1, cubeRadius);
    const int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    const double fontScale = 0.45;
    const int thickness = 1;
    const float value = 1.0f;

    for (const auto &cell : cells) {
        const int cx = static_cast<int>(std::round(cell.getX()));
        const int cy = static_cast<int>(std::round(cell.getY()));
        const int cz = static_cast<int>(std::round(cell.getZ()));
        if (cz < 0 || cz >= static_cast<int>(stack.size())) {
            continue;
        }

        const int zMin = std::max(0, cz - radius);
        const int zMax = std::min(static_cast<int>(stack.size()) - 1, cz + radius);
        for (int z = zMin; z <= zMax; ++z) {
            cv::Mat &slice = stack[static_cast<size_t>(z)];
            if (slice.empty()) {
                continue;
            }
            const int xMin = std::max(0, cx - radius);
            const int xMax = std::min(slice.cols - 1, cx + radius);
            const int yMin = std::max(0, cy - radius);
            const int yMax = std::min(slice.rows - 1, cy + radius);
            for (int y = yMin; y <= yMax; ++y) {
                float *row = slice.ptr<float>(y);
                for (int x = xMin; x <= xMax; ++x) {
                    row[x] = value;
                }
            }
        }

        cv::Mat &centerSlice = stack[static_cast<size_t>(cz)];
        if (centerSlice.empty()) {
            continue;
        }
        int baseline = 0;
        const cv::Size textSize = cv::getTextSize(
            cell.getName(), fontFace, fontScale, thickness, &baseline);
        int textX = cx + radius + 4;
        if (textX + textSize.width >= centerSlice.cols) {
            textX = cx - radius - 4 - textSize.width;
        }
        textX = std::clamp(textX, 0, std::max(0, centerSlice.cols - textSize.width - 1));
        int textY = std::clamp(cy - radius - 4, textSize.height + 1,
                               std::max(textSize.height + 1, centerSlice.rows - 2));
        cv::putText(centerSlice, cell.getName(), cv::Point(textX, textY),
                    fontFace, fontScale, cv::Scalar(value), thickness, cv::LINE_AA);
    }

    return stack;
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

static constexpr std::array<std::array<int, 3>, 6> kSignalFaceNeighbors{{
    {{ 1,  0,  0}}, {{-1,  0,  0}},
    {{ 0,  1,  0}}, {{ 0, -1,  0}},
    {{ 0,  0,  1}}, {{ 0,  0, -1}}
}};

static std::vector<size_t> keepBrightestBoxFraction(const std::vector<BrightBox> &boxes,
                                                    std::vector<size_t> members,
                                                    float fraction)
{
    if (members.empty()) {
        return members;
    }

    const float clampedFraction = std::clamp(fraction, 0.0f, 1.0f);
    if (clampedFraction >= 1.0f - 1e-6f) {
        return members;
    }
    if (clampedFraction <= 0.0f) {
        return {};
    }

    std::sort(members.begin(), members.end(),
              [&](size_t lhs, size_t rhs) {
                  return boxes[lhs].brightness > boxes[rhs].brightness;
              });
    const size_t keepCount = std::max<size_t>(
        1, static_cast<size_t>(std::ceil(clampedFraction * static_cast<float>(members.size()))));
    members.resize(std::min(keepCount, members.size()));
    return members;
}

static std::vector<std::vector<size_t>> connectedSignalBoxComponents(
    const std::vector<BrightBox> &boxes,
    const std::vector<size_t> &members)
{
    std::map<std::tuple<int, int, int>, size_t> memberIndex;
    for (size_t local = 0; local < members.size(); ++local) {
        const BrightBox &box = boxes[members[local]];
        memberIndex[{box.ix, box.iy, box.iz}] = local;
    }

    std::vector<std::vector<size_t>> components;
    std::vector<char> visited(members.size(), 0);
    for (size_t seedLocal = 0; seedLocal < members.size(); ++seedLocal) {
        if (visited[seedLocal]) continue;

        std::vector<size_t> component;
        std::queue<size_t> queue;
        queue.push(seedLocal);
        visited[seedLocal] = 1;
        while (!queue.empty()) {
            const size_t local = queue.front();
            queue.pop();
            const size_t boxIdx = members[local];
            component.push_back(boxIdx);
            const BrightBox &box = boxes[boxIdx];

            for (const auto &offset : kSignalFaceNeighbors) {
                auto it = memberIndex.find({
                    box.ix + offset[0],
                    box.iy + offset[1],
                    box.iz + offset[2]});
                if (it == memberIndex.end()) continue;
                const size_t neighborLocal = it->second;
                if (visited[neighborLocal]) continue;
                visited[neighborLocal] = 1;
                queue.push(neighborLocal);
            }
        }
        components.push_back(std::move(component));
    }
    return components;
}

static Frame::SignalCenter makeSignalCenterFromBoxes(const std::vector<BrightBox> &boxes,
                                                     const std::vector<size_t> &members,
                                                     float backgroundValue)
{
    double weightSum = 0.0;
    double xSum = 0.0;
    double ySum = 0.0;
    double zSum = 0.0;
    double brightnessSum = 0.0;
    for (size_t boxIdx : members) {
        const BrightBox &box = boxes[boxIdx];
        const float weight = std::max(1e-6f, box.brightness - backgroundValue);
        weightSum += weight;
        xSum += weight * static_cast<double>(box.center.x);
        ySum += weight * static_cast<double>(box.center.y);
        zSum += weight * static_cast<double>(box.center.z);
        brightnessSum += static_cast<double>(box.brightness);
    }

    Frame::SignalCenter center;
    if (!members.empty() && weightSum > 0.0) {
        center.position = cv::Point3f(
            static_cast<float>(xSum / weightSum),
            static_cast<float>(ySum / weightSum),
            static_cast<float>(zSum / weightSum));
        center.brightness = static_cast<float>(brightnessSum / static_cast<double>(members.size()));
        center.boxes = static_cast<int>(members.size());
    }
    return center;
}

static void splitSignalClusterRecursive(const std::vector<BrightBox> &boxes,
                                        const std::vector<size_t> &members,
                                        float backgroundValue,
                                        float recursiveBrightFraction,
                                        int depth,
                                        int maxDepth,
                                        std::vector<Frame::SignalCenter> &outCenters)
{
    if (members.empty()) {
        return;
    }

    if (depth >= maxDepth || members.size() < 2) {
        outCenters.push_back(makeSignalCenterFromBoxes(boxes, members, backgroundValue));
        return;
    }

    std::vector<size_t> splitMembers =
        keepBrightestBoxFraction(boxes, members, recursiveBrightFraction);
    if (splitMembers.size() < 2) {
        outCenters.push_back(makeSignalCenterFromBoxes(boxes, members, backgroundValue));
        return;
    }

    std::vector<std::vector<size_t>> components =
        connectedSignalBoxComponents(boxes, splitMembers);
    if (components.size() < 2) {
        outCenters.push_back(makeSignalCenterFromBoxes(boxes, members, backgroundValue));
        return;
    }

    std::sort(components.begin(), components.end(),
              [&](const std::vector<size_t> &lhs, const std::vector<size_t> &rhs) {
                  const Frame::SignalCenter a = makeSignalCenterFromBoxes(boxes, lhs, backgroundValue);
                  const Frame::SignalCenter b = makeSignalCenterFromBoxes(boxes, rhs, backgroundValue);
                  return a.brightness > b.brightness;
              });

    for (const auto &component : components) {
        splitSignalClusterRecursive(boxes, component, backgroundValue,
                                    recursiveBrightFraction, depth + 1,
                                    maxDepth, outCenters);
    }
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

    const float targetBoxSize =
        static_cast<float>(std::max(1, config.simulation.signal_guided_box_side_length));

    const int boxSizeX = chooseNearestDivisorSize(sizeX, targetBoxSize);
    const int boxSizeY = chooseNearestDivisorSize(sizeY, targetBoxSize);
    const int boxSizeZ = chooseNearestDivisorSize(sizeZ, targetBoxSize);
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

    if (boxes.empty()) {
        std::cout << "[Signal Centers] frame " << displayFrame
                  << " enabled=" << config.simulation.signal_guided_position_enabled
                  << " boxSize=(" << boxSizeX << "," << boxSizeY << "," << boxSizeZ << ")"
                  << " grid=(" << gridX << "," << gridY << "," << gridZ << ")"
                  << " background=" << backgroundValue
                  << " minDelta=" << minDelta
                  << " keptBoxes=0 clusters=0"
                  << " reason=no_boxes_above_background_delta"
                  << '\n';
        return centers;
    }

    std::vector<size_t> initialMembers(boxes.size());
    std::iota(initialMembers.begin(), initialMembers.end(), 0);
    const float initialBrightFraction =
        std::clamp(config.simulation.signal_guided_initial_bright_fraction, 0.0f, 1.0f);
    const float recursiveBrightFraction =
        std::clamp(config.simulation.signal_guided_recursive_bright_fraction, 0.0f, 1.0f);
    const int maxRecursiveDepth =
        std::max(0, config.simulation.signal_guided_max_recursive_depth);
    initialMembers = keepBrightestBoxFraction(boxes, std::move(initialMembers),
                                              initialBrightFraction);

    std::vector<std::vector<size_t>> initialComponents =
        connectedSignalBoxComponents(boxes, initialMembers);
    float maxCenterBrightness = backgroundValue;
    for (const auto &component : initialComponents) {
        splitSignalClusterRecursive(boxes, component, backgroundValue,
                                    recursiveBrightFraction, 0,
                                    maxRecursiveDepth, centers);
    }

    for (const auto &center : centers) {
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
              << " minDelta=" << minDelta
              << " keptBoxes=" << boxes.size()
              << " initialBrightFraction=" << initialBrightFraction
              << " initialBoxes=" << initialMembers.size()
              << " recursiveBrightFraction=" << recursiveBrightFraction
              << " maxRecursiveDepth=" << maxRecursiveDepth
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

static std::vector<cv::Mat> buildSignalProbabilityStack(
    const std::vector<cv::Mat> &realFrame,
    const std::vector<Frame::SignalCenter> &centers,
    const BaseConfig &config)
{
    std::vector<cv::Mat> probabilityDebug;
    probabilityDebug.reserve(realFrame.size());
    for (const auto &slice : realFrame) {
        probabilityDebug.emplace_back(cv::Mat::zeros(slice.size(), CV_32F));
    }

    if (realFrame.empty() || centers.empty()) {
        return probabilityDebug;
    }

    std::vector<cv::Mat> probability;
    probability.reserve(realFrame.size());
    for (const auto &slice : realFrame) {
        probability.emplace_back(cv::Mat::zeros(slice.size(), CV_32F));
    }

    const float minSigmaScale =
        std::max(0.0f, config.simulation.signal_guided_min_sigma_scale);
    const float sigmaRangeMultiplier =
        std::max(1e-3f, config.simulation.signal_guided_sigma_range_multiplier);

    float maxProb = 0.0f;
    #pragma omp parallel for schedule(static) reduction(max:maxProb)
    for (int z = 0; z < static_cast<int>(probability.size()); ++z) {
        cv::Mat &probSlice = probability[static_cast<size_t>(z)];
        for (int y = 0; y < probSlice.rows; ++y) {
            float *probRow = probSlice.ptr<float>(y);
            for (int x = 0; x < probSlice.cols; ++x) {
                float bestProb = 0.0f;
                for (const auto &center : centers) {
                    const float sigmaScale =
                        std::max(minSigmaScale, center.sigmaScale);
                    const float sx = std::max(
                        1e-3f, Ellipsoid::cellConfig.x.sigma * sigmaScale * sigmaRangeMultiplier);
                    const float sy = std::max(
                        1e-3f, Ellipsoid::cellConfig.y.sigma * sigmaScale * sigmaRangeMultiplier);
                    const float sz = std::max(
                        1e-3f, Ellipsoid::cellConfig.z.sigma * sigmaScale * sigmaRangeMultiplier);
                    const float dx = (static_cast<float>(x) - center.position.x) / sx;
                    const float dy = (static_cast<float>(y) - center.position.y) / sy;
                    const float dz = (static_cast<float>(z) - center.position.z) / sz;
                    const float p = std::exp(-0.5f * (dx * dx + dy * dy + dz * dz));
                    bestProb = std::max(bestProb, p);
                }
                probRow[x] = bestProb;
                maxProb = std::max(maxProb, bestProb);
            }
        }
    }

    if (maxProb > 1e-6f) {
        #pragma omp parallel for schedule(static)
        for (int z = 0; z < static_cast<int>(probabilityDebug.size()); ++z) {
            cv::Mat &debugSlice = probabilityDebug[static_cast<size_t>(z)];
            const cv::Mat &probSlice = probability[static_cast<size_t>(z)];
            for (int y = 0; y < debugSlice.rows; ++y) {
                float *debugRow = debugSlice.ptr<float>(y);
                const float *probRow = probSlice.ptr<float>(y);
                for (int x = 0; x < debugSlice.cols; ++x) {
                    debugRow[x] = std::clamp(probRow[x] / maxProb, 0.0f, 1.0f);
                }
            }
        }
    }

    return probabilityDebug;
}

static void exportSignalDebugStacks(const std::vector<cv::Mat> &realFrame,
                                    const std::vector<Frame::SignalCenter> &centers,
                                    const std::vector<cv::Mat> &probabilityDebug,
                                    const BaseConfig &config,
                                    const fs::path &baseOutputDir,
                                    const fs::path &framePath)
{
    if (!config.simulation.export_signal_debug_images || realFrame.empty()) {
        return;
    }

    std::vector<cv::Mat> centerCubes;
    centerCubes.reserve(realFrame.size());
    for (const auto &slice : realFrame) {
        centerCubes.emplace_back(cv::Mat::zeros(slice.size(), CV_32F));
    }

    constexpr int kCubeHalfSize = 2;
    const int zCount = static_cast<int>(realFrame.size());
    for (const auto &center : centers) {
        const int cx = static_cast<int>(std::round(center.position.x));
        const int cy = static_cast<int>(std::round(center.position.y));
        const int cz = static_cast<int>(std::round(center.position.z));
        const int z0 = std::max(0, cz - kCubeHalfSize);
        const int z1 = std::min(zCount - 1, cz + kCubeHalfSize);
        for (int z = z0; z <= z1; ++z) {
            cv::Mat &slice = centerCubes[static_cast<size_t>(z)];
            const int x0 = std::max(0, cx - kCubeHalfSize);
            const int x1 = std::min(slice.cols - 1, cx + kCubeHalfSize);
            const int y0 = std::max(0, cy - kCubeHalfSize);
            const int y1 = std::min(slice.rows - 1, cy + kCubeHalfSize);
            for (int y = y0; y <= y1; ++y) {
                float *row = slice.ptr<float>(y);
                for (int x = x0; x <= x1; ++x) {
                    row[x] = 1.0f;
                }
            }
        }
    }

    double probabilityMin = 0.0;
    double probabilityMax = 0.0;
    int nonzeroVoxels = 0;
    bool statsInitialized = false;
    for (const auto &slice : probabilityDebug) {
        if (slice.empty()) continue;
        double sliceMin = 0.0;
        double sliceMax = 0.0;
        cv::minMaxLoc(slice, &sliceMin, &sliceMax);
        if (!statsInitialized) {
            probabilityMin = sliceMin;
            probabilityMax = sliceMax;
            statsInitialized = true;
        } else {
            probabilityMin = std::min(probabilityMin, sliceMin);
            probabilityMax = std::max(probabilityMax, sliceMax);
        }
        nonzeroVoxels += cv::countNonZero(slice > 0.0f);
    }

    std::cout << "[Signal Debug Export] frame=" << framePath.filename().string()
              << " centers=" << centers.size()
              << " probability_min=" << probabilityMin
              << " probability_max=" << probabilityMax
              << " probability_nonzero_voxels=" << nonzeroVoxels
              << " sigma=(" << Ellipsoid::cellConfig.x.sigma
              << "," << Ellipsoid::cellConfig.y.sigma
              << "," << Ellipsoid::cellConfig.z.sigma << ")"
              << " sigmaRangeMultiplier="
              << config.simulation.signal_guided_sigma_range_multiplier
              << " output_dir=" << (baseOutputDir / "signal_debug").string()
              << '\n';

    exportStackToSubdir(centerCubes, baseOutputDir, "signal_debug/centers",
                        framePath,
                        config.simulation.export_frame_png,
                        config.simulation.export_frame_tiff);
    exportStackToSubdir(probabilityDebug, baseOutputDir, "signal_debug/perturb_probability",
                        framePath,
                        config.simulation.export_frame_png,
                        config.simulation.export_frame_tiff);
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

static std::size_t countBackgroundVoxelsAfterCells(const Frame &frame,
                                                   const SimulationConfig &simulationConfig)
{
    const auto &realFrame = frame.getRealFrame();
    if (realFrame.empty()) {
        return 0;
    }

    std::size_t backgroundVoxels = 0;
    for (size_t z = 0; z < realFrame.size(); ++z) {
        cv::Mat mask = cv::Mat::zeros(realFrame[z].size(), CV_32F);
        for (const auto &cell : frame.cells) {
            EllipsoidParams params = cell.getCellParams();
            params.brightness = 1.0f;
            Ellipsoid maskCell(params);
            maskCell.draw(mask, simulationConfig, static_cast<float>(z));
        }

        for (int y = 0; y < mask.rows; ++y) {
            const float *row = mask.ptr<float>(y);
            for (int x = 0; x < mask.cols; ++x) {
                if (row[x] <= 0.0f) {
                    ++backgroundVoxels;
                }
            }
        }
    }

    return backgroundVoxels;
}

static float blackVoxelFractionInsideCell(const Frame &frame,
                                          const Ellipsoid &cell,
                                          const SimulationConfig &simulationConfig)
{
    const auto &realFrame = frame.getRealFrame();
    if (realFrame.empty()) {
        return 0.0f;
    }

    EllipsoidParams params = cell.getCellParams();
    params.brightness = 1.0f;
    Ellipsoid maskCell(params);

    std::size_t inside = 0;
    std::size_t black = 0;
    const float blackThreshold = std::max(0.0f, simulationConfig.post_alignment_black_threshold);
    for (size_t z = 0; z < realFrame.size(); ++z) {
        cv::Mat mask = cv::Mat::zeros(realFrame[z].size(), CV_32F);
        maskCell.draw(mask, simulationConfig, static_cast<float>(z));

        for (int y = 0; y < mask.rows; ++y) {
            const float *maskRow = mask.ptr<float>(y);
            const float *realRow = realFrame[z].ptr<float>(y);
            for (int x = 0; x < mask.cols; ++x) {
                if (maskRow[x] <= 0.0f) {
                    continue;
                }
                ++inside;
                if (realRow[x] <= blackThreshold) {
                    ++black;
                }
            }
        }
    }

    if (inside == 0) {
        return 0.0f;
    }
    return static_cast<float>(black) / static_cast<float>(inside);
}

// Preprocessing moved to ImageHandler. Single preprocessed stack via
// ImageHandler::loadFrame.
CellUniverse::CellUniverse(std::map<std::string, std::vector<Ellipsoid>> initialCells,
                           PathVec imagePaths,
                           BaseConfig &config,
                           std::string outputPath,
                           int firstFrame,
                           int continueFrom)
: config(config), outputPath(outputPath), firstFrame(firstFrame),
  imagePaths(imagePaths), initialCells(initialCells), continueFrom(continueFrom)
{
    std::cout << "[Frame Intensity Scale] enabled="
              << config.simulation.frame_intensity_normalization_enabled
              << " low_percentile="
              << config.simulation.frame_intensity_scale_low_percentile
              << " high_percentile="
              << config.simulation.frame_intensity_scale_high_percentile
              << " exclude_zeros="
              << config.simulation.frame_intensity_percentile_exclude_zeros
              << " hard_max=" << config.simulation.frame_intensity_hard_max
              << '\n';

    if (config.simulation.edge_brightness_alignment_enabled) {
        std::cout << "[EdgeBrightnessAlignment] enabled=1"
                  << " target=pending_all_frames_min_background"
                  << " xy_margin=" << std::max(1, config.simulation.edge_brightness_alignment_xy_margin)
                  << " offsets=("
                  << std::max(0, config.simulation.edge_brightness_alignment_left_offset) << ","
                  << std::max(0, config.simulation.edge_brightness_alignment_right_offset) << ","
                  << std::max(0, config.simulation.edge_brightness_alignment_top_offset) << ","
                  << std::max(0, config.simulation.edge_brightness_alignment_bottom_offset) << ")"
                  << " max_shift=" << std::max(0.0f, config.simulation.edge_brightness_alignment_max_shift)
                  << '\n';
    } else {
        std::cout << "[EdgeBrightnessAlignment] enabled=0\n";
    }

    // Determine post-interpolation z_slices depth: run ONE frame through the
    // full preprocess pipeline so we see the true interpolated size (raw
    // TIFF depth * z_scaling roughly). Preprocess-only export does not need
    // cell z-bounds, so skip this expensive probe there to avoid processing
    // frame 1 twice.
    if (!config.simulation.quit_after_preprocessing) {
        std::ostringstream probeRaw, probePre;
        std::vector<cv::Mat> probe = ImageHandler::loadRawFrame(
            imagePaths.front().string(), config, &probeRaw);
        if (config.simulation.frame_intensity_normalization_enabled) {
            normalizeStackToFrameIntensity(probe, config.simulation);
        }
        probe = ImageHandler::preprocessLoadedFrame(
            probe, imagePaths.front().string(), config, &probePre);
        config.simulation.z_slices = static_cast<int>(probe.size());
        Ellipsoid::cellConfig.maxZ = static_cast<float>(probe.size()) - 1.0f;
        std::cout << "[M2 Probe] post-preprocess z_slices=" << probe.size()
                  << " maxZ=" << Ellipsoid::cellConfig.maxZ << '\n';
    } else {
        std::cout << "[M2 Probe] skipped for preprocess-only mode\n";
    }

    // Allocate lazy-load Frame placeholders. Each holds only cells + config;
    // image stacks populated by `prepareFrame(i)` just before optimize.
    frames.reserve(imagePaths.size());
    for (size_t i = 0; i < imagePaths.size(); ++i)
    {
        fs::path path(imagePaths[i]);
        std::string file_name = path.filename();
        const bool hasInitial =
            (continueFrom == -1 || static_cast<int>(i) < continueFrom) &&
            initialCells.find(file_name) != initialCells.end();
        const std::vector<Ellipsoid> &cells =
            hasInitial ? initialCells.at(file_name) : std::vector<Ellipsoid>{};
        frames.emplace_back(config.simulation, cells, outputPath, file_name);
        if (config.cell) {
            frames.back().setBackgroundColor(config.cell->backgroundColor);
        }
    }
    perFrameAdaptiveBackground.assign(imagePaths.size(), 0.0f);
    perFrameMeanBrightness.assign(imagePaths.size(), 0.0f);
}

void CellUniverse::prepareFrame(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("prepareFrame: invalid frame index");
    }
    if (frames[frameIndex].hasImageStacks()) {
        return;  // already loaded
    }

    // Load + normalize + preprocess just this frame.
    std::ostringstream rawLog;
    std::ostringstream preprocessLog;
    std::vector<cv::Mat> real_frame =
        ImageHandler::loadRawFrame(imagePaths[static_cast<size_t>(frameIndex)].string(),
                                   config, &rawLog);
    std::cout << rawLog.str();

    if (config.simulation.frame_intensity_normalization_enabled) {
        const auto [lowReference, highReference] =
            normalizeStackToFrameIntensity(real_frame, config.simulation);
        std::cout << "[Frame Intensity Scale] frame="
                  << imagePaths[static_cast<size_t>(frameIndex)].filename().string()
                  << " mean=" << computeStackMean(real_frame)
                  << " low_ref=" << lowReference
                  << " high_ref=" << highReference
                  << " hard_max=" << config.simulation.frame_intensity_hard_max << '\n';
    } else {
        std::cout << "[Frame Intensity Scale] frame="
                  << imagePaths[static_cast<size_t>(frameIndex)].filename().string()
                  << " enabled=0"
                  << " mean=" << computeStackMean(real_frame)
                  << '\n';
    }

    real_frame = ImageHandler::preprocessLoadedFrame(
        real_frame,
        imagePaths[static_cast<size_t>(frameIndex)].string(),
        config, &preprocessLog);
    std::cout << preprocessLog.str();
    if (config.simulation.edge_brightness_alignment_enabled &&
        !edgeBrightnessAlignmentTargetInitialized) {
        edgeBrightnessAlignmentTarget =
            computeEdgeBrightnessMean(real_frame, config.simulation);
        edgeBrightnessAlignmentTargetInitialized = true;
        std::cout << "[EdgeBrightnessAlignment] target_initialized_from="
                  << imagePaths[static_cast<size_t>(frameIndex)].filename().string()
                  << " target=" << edgeBrightnessAlignmentTarget << '\n';
    }
    alignStackToEdgeBrightness(real_frame,
                               config.simulation,
                               edgeBrightnessAlignmentTarget,
                               imagePaths[static_cast<size_t>(frameIndex)],
                               std::cout);
    blackThresholdStackAfterAlignment(real_frame,
                                      config.simulation,
                                      imagePaths[static_cast<size_t>(frameIndex)],
                                      std::cout);
    const std::vector<cv::Mat> unblackoffedFrame = cloneMatStack(real_frame);
    blackPercentileStackAfterAlignment(real_frame,
                                       config.simulation,
                                       imagePaths[static_cast<size_t>(frameIndex)],
                                       std::cout);
    adaptBlackPercentileToChunkCount(real_frame,
                                     unblackoffedFrame,
                                     config.simulation,
                                     imagePaths[static_cast<size_t>(frameIndex)],
                                     std::cout);
    removeTinyIsolatedParticles(real_frame,
                                config,
                                imagePaths[static_cast<size_t>(frameIndex)],
                                std::cout);
    applyFinalPreprocessingBlur(real_frame,
                                config.simulation,
                                imagePaths[static_cast<size_t>(frameIndex)],
                                std::cout);
    std::vector<cv::Mat> signalMap = buildSignalMapStack(
        real_frame,
        config.simulation,
        imagePaths[static_cast<size_t>(frameIndex)],
        std::cout);

    if (config.simulation.export_preprocessed_images) {
        exportPreprocessedStack(real_frame, fs::path(outputPath),
                                imagePaths[static_cast<size_t>(frameIndex)],
                                config.simulation.export_frame_png,
                                config.simulation.export_frame_tiff);
    }

    // loadImageStacks already generates _synthFrame + refreshes the full-image
    // cost cache. The previous `if (config.cell) regenerateSynthFrame()` was
    // redundant work (2x render + 2x cost cache per frame).
    frames[frameIndex].loadImageStacks(real_frame);
    frames[frameIndex].setSignalMap(std::move(signalMap));
    prepareSignalCentersForFrame(frameIndex, real_frame, true);
}

void CellUniverse::prepareSignalCentersForFrame(int frameIndex,
                                                const std::vector<cv::Mat> &realFrame,
                                                bool keepLoaded)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("prepareSignalCentersForFrame: invalid frame index");
    }

    Frame &frame = frames[static_cast<size_t>(frameIndex)];
    if (!frame.hasImageStacks()) {
        frame.loadImageStacks(realFrame);
    }

    if (config.simulation.signal_guided_position_enabled ||
        config.simulation.export_signal_debug_images) {
        std::vector<Frame::SignalCenter> centers = localizeSignalCentersForFrame(
            frame, config, firstFrame + frameIndex);
        std::vector<cv::Mat> probability =
            buildSignalProbabilityStack(realFrame, centers, config);
        exportSignalDebugStacks(realFrame, centers, probability, config, fs::path(outputPath),
                                imagePaths[static_cast<size_t>(frameIndex)]);
        if (config.simulation.signal_guided_position_enabled) {
            frame.setSignalCenters(std::move(centers));
            frame.setSignalProbability(std::move(probability));
        } else {
            frame.setSignalCenters({});
            frame.setSignalProbability({});
        }
    } else {
        frame.setSignalCenters({});
        frame.setSignalProbability({});
    }

    if (!keepLoaded) {
        frame.releaseImageStacks();
    }
}

void CellUniverse::preprocessAllFramesAlignedToMinimumBackground(bool loadIntoFrames)
{
    struct PreprocessedFrame
    {
        std::vector<cv::Mat> stack;
        float sampledBackground = 0.0f;
    };

    std::vector<PreprocessedFrame> preprocessedFrames(imagePaths.size());
    float minimumSampledBackground = std::numeric_limits<float>::infinity();
    const int frameCount = static_cast<int>(imagePaths.size());
    const bool alignmentEnabled = config.simulation.edge_brightness_alignment_enabled;

    for (int frameIndex = 0; frameIndex < frameCount; ++frameIndex)
    {
        std::ostringstream rawLog;
        std::ostringstream preprocessLog;
        std::vector<cv::Mat> realFrame =
            ImageHandler::loadRawFrame(imagePaths[static_cast<size_t>(frameIndex)].string(),
                                       config,
                                       &rawLog);
        std::cout << rawLog.str();

        if (config.simulation.frame_intensity_normalization_enabled) {
            const auto [lowReference, highReference] =
                normalizeStackToFrameIntensity(realFrame, config.simulation);
            std::cout << "[Frame Intensity Scale] frame="
                      << imagePaths[static_cast<size_t>(frameIndex)].filename().string()
                      << " mean=" << computeStackMean(realFrame)
                      << " low_ref=" << lowReference
                      << " high_ref=" << highReference
                      << " hard_max=" << config.simulation.frame_intensity_hard_max << '\n';
        } else {
            std::cout << "[Frame Intensity Scale] frame="
                      << imagePaths[static_cast<size_t>(frameIndex)].filename().string()
                      << " enabled=0"
                      << " mean=" << computeStackMean(realFrame)
                      << '\n';
        }

        realFrame = ImageHandler::preprocessLoadedFrame(
            realFrame,
            imagePaths[static_cast<size_t>(frameIndex)].string(),
            config,
            &preprocessLog);
        std::cout << preprocessLog.str();

        if (!alignmentEnabled) {
            blackThresholdStackAfterAlignment(realFrame,
                                              config.simulation,
                                              imagePaths[static_cast<size_t>(frameIndex)],
                                              std::cout);
            const std::vector<cv::Mat> unblackoffedFrame = cloneMatStack(realFrame);
            blackPercentileStackAfterAlignment(realFrame,
                                               config.simulation,
                                               imagePaths[static_cast<size_t>(frameIndex)],
                                               std::cout);
            adaptBlackPercentileToChunkCount(realFrame,
                                             unblackoffedFrame,
                                             config.simulation,
                                             imagePaths[static_cast<size_t>(frameIndex)],
                                             std::cout);
            removeTinyIsolatedParticles(realFrame,
                                        config,
                                        imagePaths[static_cast<size_t>(frameIndex)],
                                        std::cout);
            applyFinalPreprocessingBlur(realFrame,
                                        config.simulation,
                                        imagePaths[static_cast<size_t>(frameIndex)],
                                        std::cout);
            std::vector<cv::Mat> signalMap = buildSignalMapStack(
                realFrame,
                config.simulation,
                imagePaths[static_cast<size_t>(frameIndex)],
                std::cout);

            if (config.simulation.export_preprocessed_images) {
                exportPreprocessedStack(realFrame, fs::path(outputPath),
                                        imagePaths[static_cast<size_t>(frameIndex)],
                                        config.simulation.export_frame_png,
                                        config.simulation.export_frame_tiff);
            }

            if (loadIntoFrames) {
                frames[static_cast<size_t>(frameIndex)].loadImageStacks(realFrame);
                frames[static_cast<size_t>(frameIndex)].setSignalMap(std::move(signalMap));
                prepareSignalCentersForFrame(frameIndex, realFrame, true);
            } else if (config.simulation.export_signal_debug_images) {
                prepareSignalCentersForFrame(frameIndex, realFrame, false);
            }
            continue;
        }

        const float sampledBackground =
            computeEdgeBrightnessMean(realFrame, config.simulation);
        minimumSampledBackground =
            std::min(minimumSampledBackground, sampledBackground);
        preprocessedFrames[static_cast<size_t>(frameIndex)] = {
            std::move(realFrame),
            sampledBackground
        };

        std::cout << "[EdgeBrightnessAlignment] frame="
                  << imagePaths[static_cast<size_t>(frameIndex)].filename().string()
                  << " sampled_background=" << sampledBackground
                  << " pending_minimum_background=1"
                  << std::endl;
    }

    if (!alignmentEnabled) {
        return;
    }

    if (!std::isfinite(minimumSampledBackground))
    {
        minimumSampledBackground = 0.0f;
    }

    edgeBrightnessAlignmentTarget = minimumSampledBackground;
    edgeBrightnessAlignmentTargetInitialized = true;
    std::cout << "[EdgeBrightnessAlignment] batch_target=min_sampled_background"
              << " target=" << edgeBrightnessAlignmentTarget
              << " frame_count=" << preprocessedFrames.size()
              << std::endl;

    for (int frameIndex = 0; frameIndex < static_cast<int>(preprocessedFrames.size()); ++frameIndex)
    {
        auto &item = preprocessedFrames[static_cast<size_t>(frameIndex)];
        alignStackToEdgeBrightness(item.stack,
                                   config.simulation,
                                   edgeBrightnessAlignmentTarget,
                                   imagePaths[static_cast<size_t>(frameIndex)],
                                   std::cout);
        blackThresholdStackAfterAlignment(item.stack,
                                          config.simulation,
                                          imagePaths[static_cast<size_t>(frameIndex)],
                                          std::cout);
        const std::vector<cv::Mat> unblackoffedFrame = cloneMatStack(item.stack);
        blackPercentileStackAfterAlignment(item.stack,
                                           config.simulation,
                                           imagePaths[static_cast<size_t>(frameIndex)],
                                           std::cout);
        adaptBlackPercentileToChunkCount(item.stack,
                                         unblackoffedFrame,
                                         config.simulation,
                                         imagePaths[static_cast<size_t>(frameIndex)],
                                         std::cout);
        removeTinyIsolatedParticles(item.stack,
                                    config,
                                    imagePaths[static_cast<size_t>(frameIndex)],
                                    std::cout);
        applyFinalPreprocessingBlur(item.stack,
                                    config.simulation,
                                    imagePaths[static_cast<size_t>(frameIndex)],
                                    std::cout);
        std::vector<cv::Mat> signalMap = buildSignalMapStack(
            item.stack,
            config.simulation,
            imagePaths[static_cast<size_t>(frameIndex)],
            std::cout);

        if (config.simulation.export_preprocessed_images) {
            exportPreprocessedStack(item.stack, fs::path(outputPath),
                                    imagePaths[static_cast<size_t>(frameIndex)],
                                    config.simulation.export_frame_png,
                                    config.simulation.export_frame_tiff);
        }

        if (loadIntoFrames) {
            frames[static_cast<size_t>(frameIndex)].loadImageStacks(item.stack);
            frames[static_cast<size_t>(frameIndex)].setSignalMap(std::move(signalMap));
            prepareSignalCentersForFrame(frameIndex, item.stack, true);
        } else if (config.simulation.export_signal_debug_images) {
            prepareSignalCentersForFrame(frameIndex, item.stack, false);
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
    const int absoluteFrame = firstFrame + frameIndex;

    const bool hasPreviousFrameSummary =
        frameIndex > 0 ||
        (frameIndex == 0 && resumePreviousFrameSummaryValid);
    if (hasPreviousFrameSummary) {
        // Prev-frame summaries cached at end of optimize(frameIndex-1) so we
        // don't need frames[frameIndex-1]'s image stacks. Falls back to 0 if
        // not cached (shouldn't happen if prepareFrame + optimize sequence is
        // respected).
        const float previousBackground = (frameIndex > 0)
            ? perFrameAdaptiveBackground[frameIndex - 1]
            : resumePreviousAdaptiveBackground;
        const float previousMeanBrightness = (frameIndex > 0)
            ? perFrameMeanBrightness[frameIndex - 1]
            : resumePreviousMeanBrightness;
        const float currentMeanBrightness = computeStackMean(frame.getRealFrame());
        const float brightnessScale =
            (previousMeanBrightness > 1e-6f) ? (currentMeanBrightness / previousMeanBrightness) : 1.0f;
        const float updatedBackground = previousBackground * brightnessScale;

        frame.setBackgroundColor(updatedBackground);
        std::cout << "[Adaptive Background] frame " << absoluteFrame
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
    const bool bboxActiveThisFrame =
        config.prob.use_bbox_cost &&
        absoluteFrame > 0 &&
        (frameIndex > 0 || !previousSnapshots.empty());
    frame.setUseBboxCost(bboxActiveThisFrame,
                         config.prob.bbox_margin_scale);

    frame.regenerateSynthFrame();

    // Signal-guided perturbation activation (yp ffc1917 — full per-frame design).
    // When enabled and enough signal centers are detected (>= previous frame
    // cell count), the prepared frame uses signal-guided perturbation with
    // signal_guided_iterations_per_cell. When disabled or not enough centers,
    // fall back to random perturbation with random_iterations_per_cell.
    bool useSignalGuidanceThisFrame = false;
    if (config.simulation.signal_guided_position_enabled) {
        const auto &centers = frame.getSignalCenters();
        useSignalGuidanceThisFrame = true;
        const bool hasPreviousCellCount =
            frameIndex > 0 ||
            (frameIndex == 0 && resumePreviousFrameSummaryValid);
        if (hasPreviousCellCount) {
            const size_t previousCellCount = (frameIndex > 0)
                ? frames[frameIndex - 1].cells.size()
                : resumePreviousCellCount;
            if (centers.size() < previousCellCount) {
                useSignalGuidanceThisFrame = false;
                std::cout << "[Signal Guidance Fallback] frame " << absoluteFrame
                          << " centers=" << centers.size()
                          << " previousCells=" << previousCellCount
                          << " mode=random\n";
            }
        }
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
    const float randomPerturbRadiusRatio =
        config.cell ? config.cell->randomPerturbRadiusRatio : 0.5f;

    std::cout << "[Optimize] frame " << displayFrame
              << " (" << frame.cells.size() << " cells, " << totalIterations << " iterations)"
              << " perturbMode=" << (useSignalGuidanceThisFrame ? "signal_guided" : "random")
              << " guidedItersPerCell=" << guidedPerCellIters
              << " randomItersPerCell=" << randomPerCellIters
              << " randomPerturbRadiusRatio=" << randomPerturbRadiusRatio
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
        if (allowSplits && !cell.isTrash()) {
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
    //   2. Pre-pass for every cell: image-grounded D1/D2 centroids of
    //      bright pixels. Pre-pass direction + midpoint become candidates
    //      for the split attempt, alongside parent-rotation axes and the
    //      true snap center. Cost + bio + bridge gates decide acceptance.
    //   3. Unified perturb + split loop over all cells. Split attempts
    //      are gated only by P(split) which scales linearly with elong.
    //      Cells that aren't really splitting get filtered by:
    //        - bridge gate (valleyRatio ≥ 0.95 → no valley, reject)
    //        - bio gates (volume fraction, daughter size ratio, buried)
    //        - cost gate (must improve by ≥ split_cost)
    //   4. PCA shape fit with snap-mask at the final post-perturb positions.
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

    auto runPcaShapeFit = [&]() {
    // ---- Per-cell iterative PCA shape fit ----
    // Runs after the current-frame perturb/split loop so the final frame
    // shape is fitted at the post-perturb positions before snapshot/export.
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
        const bool trashPcaShapeFitEnabled =
            config.cell && config.cell->trashPcaShapeFitEnabled;
        const float trashMaxOriginalRadiusFactor = std::max(
            1.0f,
            config.cell ? config.cell->trashPcaShapeMaxOriginalRadiusFactor : 2.0f);
        if (trashPcaShapeFitEnabled) {
            for (const auto &cell : frame.cells) {
                if (!cell.isTrash()) continue;
                const std::string &name = cell.getName();
                if (cellShapeBirth.find(name) == cellShapeBirth.end()) {
                    cellShapeBirth[name] = {
                        cell.getARadius(), cell.getBRadius(), cell.getCRadius()};
                }
            }
        }
        const int nCells = static_cast<int>(frame.cells.size());
        std::vector<std::ostringstream> shapeLogs(nCells);
        #pragma omp parallel for schedule(dynamic)
        for (int ci = 0; ci < nCells; ++ci) {
            const std::string sname = frame.cells[ci].getName();
            const bool isTrashCell = frame.cells[ci].isTrash();
            if (isTrashCell && !trashPcaShapeFitEnabled) {
                shapeLogs[ci] << "  [PCA Shape] cell=" << sname
                              << " skipped=trash_fixed_size" << std::endl;
                continue;
            }
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
            if (isTrashCell) {
                auto originalIt = cellShapeBirth.find(sname);
                if (originalIt != cellShapeBirth.end()) {
                    const auto &original = originalIt->second;
                    const float preA = frame.cells[ci].getARadius();
                    const float preB = frame.cells[ci].getBRadius();
                    const float preC = frame.cells[ci].getCRadius();
                    const float capA = original[0] * trashMaxOriginalRadiusFactor;
                    const float capB = original[1] * trashMaxOriginalRadiusFactor;
                    const float capC = original[2] * trashMaxOriginalRadiusFactor;
                    const float newA = std::min(preA, capA);
                    const float newB = std::min(preB, capB);
                    const float newC = std::min(preC, capC);
                    if (newA != preA || newB != preB || newC != preC) {
                        frame.cells[ci].setRadii(newA, newB, newC);
                        shapeLogs[ci] << "  [Trash PCA Radius Cap] cell=" << sname
                                      << " factor=" << trashMaxOriginalRadiusFactor
                                      << " original=(" << original[0] << "," << original[1] << "," << original[2] << ")"
                                      << " pre=(" << preA << "," << preB << "," << preC << ")"
                                      << " post=(" << newA << "," << newB << "," << newC << ")"
                                      << std::endl;
                    }
                }
            }
        }
        // Serial merge: emit per-cell log blocks in cell-index order.
        for (int ci = 0; ci < nCells; ++ci) {
            const std::string buf = shapeLogs[ci].str();
            if (!buf.empty()) std::cout << buf;
        }

        // PCA-bridge discovery moved out of this end-of-frame lambda — it
        // now runs BEFORE the main split loop (see "PCA Bridge Propose"
        // block earlier in optimize()), so its (left, right) daughter
        // centroids can be injected as a "bridge_primary" candidate into
        // trySplitCellPhased and validated by the main path's full gate
        // stack instead of an independent accepting path.

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
        const bool fitGrowthCapEnabled =
            config.cell && config.cell->pcaShapeFitGrowthCapEnabled;
        const float fitGrowthCap = std::clamp(
            config.cell ? config.cell->pcaShapeFitGrowthCap : 0.10f,
            0.0f, 1.0f);
        const float fitUpFactor = 1.0f + fitGrowthCap;
        const float fitDownFactor = 1.0f - fitGrowthCap;
        int fitsClamped = 0;
        if (fitGrowthCapEnabled) {
            for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
                const std::string &name = frame.cells[ci].getName();
                if (frame.cells[ci].isTrash()) continue;
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

        // ---- Birth-relative growth cap (anti-bloat over long horizons) ----
        //
        // Caps each radius at `birthR × birth_growth_cap_factor`. Unlike the
        // per-frame fit growth cap above (which compounds over time since
        // cellShapeReference updates every frame), this is a cumulative
        // ceiling tied to the frozen birth radii. A cell that has existed
        // for many frames and slowly bloated (observed: a511 grew a-axis
        // from 28.6 at birth to 58.6 after 25 frames) gets caught here.
        //
        // When a bloated cell wants to grow but hits the cap, the only way
        // to reduce its image cost is to split — which is exactly the
        // biologically correct behavior.
        const float birthCapFactor = config.prob.birth_growth_cap_factor;
        const float birthCapElongThresh = config.prob.birth_growth_cap_elong_threshold;
        if (birthCapFactor > 0.0f) {
            int birthCapped = 0;
            for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
                const std::string &name = frame.cells[ci].getName();
                if (frame.cells[ci].isTrash()) continue;
                auto birthIt = cellShapeBirth.find(name);
                if (birthIt == cellShapeBirth.end()) continue;  // not yet captured
                const auto &birth = birthIt->second;
                const float capA = birth[0] * birthCapFactor;
                const float capB = birth[1] * birthCapFactor;
                const float capC = birth[2] * birthCapFactor;
                const float fA = frame.cells[ci].getARadius();
                const float fB = frame.cells[ci].getBRadius();
                const float fC = frame.cells[ci].getCRadius();
                const bool radiusOver = (fA > capA || fB > capB || fC > capC);
                if (!radiusOver) continue;
                // Gate on elongation too — only classic bloat (big + elongated)
                // gets clamped. Healthy growing cells that exceed birth×factor
                // without elongation pass through unchanged.
                const float elong = frame.cells[ci].shapeElongation();
                if (elong < birthCapElongThresh) continue;
                frame.cells[ci].setRadii(std::min(fA, capA),
                                         std::min(fB, capB),
                                         std::min(fC, capC));
                ++birthCapped;
            }
            if (birthCapped > 0) {
                std::cout << "[Birth Growth Cap] frame " << displayFrame
                          << " clamped=" << birthCapped
                          << " factor=" << birthCapFactor
                          << " elongThresh=" << birthCapElongThresh << std::endl;
                frame.regenerateSynthFrame();
            }
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
    };

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
    frame.setMaxPerturbDriftXY(config.prob.max_perturb_drift_xy);
    frame.setMaxPerturbDriftZ(config.prob.max_perturb_drift_z);
    // Voronoi map is needed for EITHER the (legacy) cost-replacement filter
    // or the additive bleed penalty — enable map build when either is on.
    const bool voronoiMapNeeded =
        config.simulation.voronoi_cost_enabled ||
        (config.simulation.voronoi_bleed_penalty_enabled &&
         config.simulation.voronoi_bleed_penalty_weight > 0.0f);
    frame.enableVoronoiCost(voronoiMapNeeded);
    frame.setVoronoiBleedWeight(
        config.simulation.voronoi_bleed_penalty_enabled
            ? config.simulation.voronoi_bleed_penalty_weight
            : 0.0f);
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

    // Build the static-Voronoi cost territory map for this frame. Anchors
    // are snap positions (for cells with a valid snap) or live centers
    // (frame 1, newborn daughters). Pixels in each cell's bbox that are
    // closer to another anchor contribute zero to this cell's image cost
    // during perturbation — prevents bloat-induced territory annexation.
    // Rebuilt after each split accept below.
    if (voronoiMapNeeded) {
        const auto vorT0 = std::chrono::steady_clock::now();
        frame.rebuildVoronoiMap();
        const auto vorT1 = std::chrono::steady_clock::now();
        const double vorMs = std::chrono::duration<double, std::milli>(
            vorT1 - vorT0).count();
        std::cout << "[Voronoi Map] frame " << displayFrame
                  << " anchors=" << frame.cells.size()
                  << " bleed_w=" << frame.getVoronoiBleedWeight()
                  << " build_ms=" << vorMs << std::endl;
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

    std::vector<cv::Mat> movementPerturbDebugPlacements;
    std::vector<cv::Mat> splitPerturbDebugPlacements;
    int movementPerturbDebugPlacementCount = 0;
    int splitPerturbDebugPlacementCount = 0;
    const bool exportPerturbDebug =
        config.simulation.export_perturb_debug_images &&
        !frame.getRealFrame().empty() &&
        (config.simulation.export_frame_png || config.simulation.export_frame_tiff);
    const bool exportPerturbCenterDebug =
        config.simulation.export_perturb_cell_center_debug_images &&
        !frame.getRealFrame().empty() &&
        (config.simulation.export_frame_png || config.simulation.export_frame_tiff);
    if (exportPerturbDebug) {
        movementPerturbDebugPlacements = makeEmptyDebugStackLike(frame.getRealFrame());
        splitPerturbDebugPlacements = makeEmptyDebugStackLike(frame.getRealFrame());
    }

    // ---- PCA-bridge proposal pre-pass (split-gate-overlap-analysis.md) ----
    //
    // The PCA bridge no longer accepts splits on its own. For each elongated
    // cell with a valid dark-bridge gap, we discover a (left, right)
    // daughter-centroid proposal here and store it by cell name. The
    // proposal is then injected as a "bridge_primary" candidate inside the
    // main split loop's trySplitCellPhased call below — the standard
    // burn-in + bio + daughter-overlap + bridge + asymmetric L2 + adaptive
    // cost gates decide acceptance.
    //
    // Cells use the previous frame's PCA-fitted shape at this point (the
    // current frame's PCA fit happens in runPcaShapeFit at end of frame),
    // but the bright-pixel input is the current frame's image — same input
    // the original end-of-frame bridge had access to via the next frame's
    // start. Daughter centroids go through 50-iter burn-in inside
    // trySplitCellPhased, so any seed-position drift is corrected.
    std::unordered_map<std::string, BridgeSplitProposal> bridgeProposals;
    if (config.prob.pca_bridge_split_enabled) {
        float maxBridgeElong = 0.0f;
        int bridgeEligible = 0;
        for (const auto &cell : frame.cells) {
            if (cell.isTrash()) continue;
            const float elong = cell.shapeElongation();
            maxBridgeElong = std::max(maxBridgeElong, elong);
            if (elong >= config.prob.pca_bridge_elongation_ratio) {
                ++bridgeEligible;
            }
        }
        for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
            const std::string parentName = frame.cells[ci].getName();
            BridgeSplitProposal proposal;
            if (frame.discoverPcaBridgeProposal(ci, config.prob, proposal)) {
                bridgeProposals[parentName] = proposal;
            }
        }
        std::cout << "[PCA Bridge Propose] frame " << (firstFrame + frameIndex)
                  << " scanned=" << frame.cells.size()
                  << " eligible=" << bridgeEligible
                  << " proposalsFound=" << bridgeProposals.size()
                  << " maxElong=" << maxBridgeElong
                  << " threshold=" << config.prob.pca_bridge_elongation_ratio
                  << std::endl;
    }

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

        struct PerturbOscillationState {
            size_t attempts = 0;
            bool hasLastCost = false;
            double lastCost = 0.0;
            std::deque<int> trendSigns;
            float multiplier = 1.0f;
        };

        std::unordered_map<std::string, PerturbOscillationState> perturbOscillation;
        const float oscillationWarmupFraction = std::clamp(
            config.simulation.perturb_oscillation_warmup_fraction, 0.0f, 1.0f);
        const size_t oscillationWarmupAttempts = static_cast<size_t>(std::ceil(
            static_cast<double>(perCellIters) * oscillationWarmupFraction));
        const size_t oscillationWindow = static_cast<size_t>(std::max(
            2, config.simulation.perturb_oscillation_window));
        const size_t oscillationMinSamples = static_cast<size_t>(std::max(
            2, config.simulation.perturb_oscillation_min_samples));
        const float oscillationSignChangeRatio = std::clamp(
            config.simulation.perturb_oscillation_sign_change_ratio, 0.0f, 1.0f);
        const float oscillationBoostRatio = std::max(
            1.0f, config.simulation.perturb_oscillation_boost_ratio);
        const float oscillationMaxMultiplier = std::max(
            1.0f, config.simulation.perturb_oscillation_max_multiplier);
        const float oscillationSmallStepProbability = std::clamp(
            config.simulation.perturb_oscillation_small_step_probability, 0.0f, 1.0f);
        const float oscillationSmallStepMultiplier = std::clamp(
            config.simulation.perturb_oscillation_small_step_multiplier, 0.0f, 1.0f);
        const bool oscillationBlackPixelBoostEnabled =
            config.simulation.perturb_oscillation_black_pixel_boost_enabled;
        const float oscillationBlackPixelBoostWeight = std::max(
            0.0f, config.simulation.perturb_oscillation_black_pixel_boost_weight);

        auto oscillationEnabledFor = [&](const Ellipsoid &cell) {
            return cell.isTrash()
                ? config.simulation.perturb_oscillation_boost_trash_enabled
                : config.simulation.perturb_oscillation_boost_cells_enabled;
        };

        auto perturbRatioFor = [&](const std::string &name) {
            auto it = perturbOscillation.find(name);
            const float mult = (it != perturbOscillation.end()) ? it->second.multiplier : 1.0f;
            if (mult > 1.0f && oscillationSmallStepProbability > 0.0f) {
                std::bernoulli_distribution smallStepDist(oscillationSmallStepProbability);
                if (smallStepDist(gen)) {
                    return randomPerturbRadiusRatio * oscillationSmallStepMultiplier;
                }
            }
            return randomPerturbRadiusRatio * mult;
        };

        auto recordPerturbLoss = [&](const std::string &name,
                                     const Ellipsoid &cell,
                                     double costDiff,
                                     bool accepted) {
            if (!oscillationEnabledFor(cell) || perCellIters == 0 ||
                oscillationWarmupAttempts == 0) {
                return;
            }

            auto &state = perturbOscillation[name];
            ++state.attempts;
            if (accepted && config.simulation.perturb_oscillation_reset_on_accept) {
                if (state.multiplier > 1.0f) {
                    const float oldMultiplier = state.multiplier;
                    state.multiplier = std::max(1.0f, state.multiplier / oscillationBoostRatio);
                    std::cout << "[Perturb Oscillation Boost Step Down] frame " << displayFrame
                              << " cell=" << name
                              << " attempts=" << state.attempts
                              << " multiplier=" << oldMultiplier << "->" << state.multiplier
                              << std::endl;
                } else {
                    state.multiplier = 1.0f;
                }
                state.trendSigns.clear();
                state.hasLastCost = false;
                return;
            }

            if (state.attempts > oscillationWarmupAttempts) {
                state.hasLastCost = true;
                state.lastCost = costDiff;
                return;
            }

            if (state.hasLastCost) {
                constexpr double kLossEpsilon = 1e-9;
                const double delta = costDiff - state.lastCost;
                int sign = 0;
                if (delta > kLossEpsilon) sign = 1;
                if (delta < -kLossEpsilon) sign = -1;
                if (sign != 0) {
                    state.trendSigns.push_back(sign);
                    while (state.trendSigns.size() > oscillationWindow) {
                        state.trendSigns.pop_front();
                    }
                }
            }
            state.hasLastCost = true;
            state.lastCost = costDiff;

            if (state.trendSigns.size() < oscillationMinSamples ||
                state.multiplier >= oscillationMaxMultiplier) {
                return;
            }

            int signChanges = 0;
            for (size_t si = 1; si < state.trendSigns.size(); ++si) {
                if (state.trendSigns[si] != state.trendSigns[si - 1]) {
                    ++signChanges;
                }
            }
            const float ratio = static_cast<float>(signChanges) /
                                static_cast<float>(std::max<size_t>(1, state.trendSigns.size() - 1));
            if (ratio < oscillationSignChangeRatio) {
                return;
            }

            float blackFraction = 0.0f;
            float blackBoostFactor = 1.0f;
            if (oscillationBlackPixelBoostEnabled && oscillationBlackPixelBoostWeight > 0.0f) {
                blackFraction = blackVoxelFractionInsideCell(frame, cell, config.simulation);
                blackBoostFactor += blackFraction * oscillationBlackPixelBoostWeight;
            }

            const float oldMultiplier = state.multiplier;
            const float effectiveBoostRatio = oscillationBoostRatio * blackBoostFactor;
            state.multiplier = std::min(oscillationMaxMultiplier,
                                        state.multiplier * effectiveBoostRatio);
            state.trendSigns.clear();
            std::cout << "[Perturb Oscillation Boost] frame " << displayFrame
                      << " cell=" << name
                      << " attempts=" << state.attempts
                      << " sign_change_ratio=" << ratio
                      << " black_fraction=" << blackFraction
                      << " black_boost_factor=" << blackBoostFactor
                      << " effective_boost_ratio=" << effectiveBoostRatio
                      << " multiplier=" << oldMultiplier << "->" << state.multiplier
                      << " base_radius_ratio=" << randomPerturbRadiusRatio
                      << std::endl;
        };

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

                // If the post-PCA pass produced a bridge proposal for this
                // cell, hand it to trySplitCellPhased so it competes as
                // "bridge_primary" alongside data_/snap_ candidates.
                const BridgeSplitProposal *bridgeForCell = nullptr;
                if (!bridgeProposals.empty()) {
                    auto bIt = bridgeProposals.find(cellName);
                    if (bIt != bridgeProposals.end()) {
                        bridgeForCell = &bIt->second;
                    }
                }
                auto result = frame.trySplitCellPhased(
                    cellIdx, splitSnapshot, others, useSnapDir, config.prob,
                    exportPerturbDebug ? &splitPerturbDebugPlacements : nullptr,
                    exportPerturbDebug ? &splitPerturbDebugPlacementCount : nullptr,
                    config.simulation.perturb_debug_cell_brightness,
                    bridgeForCell);

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
                    // Rebuild Voronoi territory — cell count and positions
                    // changed (parent replaced, two daughters appended).
                    // Daughters don't have a snap entry yet, so their
                    // live positions become their Voronoi anchors going
                    // forward. Without a rebuild, subsequent perturbations
                    // of the daughters would be scored against the parent's
                    // stale claim-set.
                    if (voronoiMapNeeded) {
                        frame.rebuildVoronoiMap();
                    }
                } else {
                    splitBlacklist.insert(cellName);
                    splitRejectedInPhase.insert(cellName);

                    // Compensation perturb. The revert left cells[cellIdx]
                    // as the parent (in place), so no find_if needed.
                    auto compResult = frame.perturbCell(
                        cellIdx, overlapWeight, useSignalGuidanceThisFrame,
                        randomPerturbRadiusRatio,
                        /*pcaRefitWellFilledMove=*/false,
                        /*useSignalMapGuidance=*/false);
                    const bool compAccept = compResult.first < 0.0;
                    if (exportPerturbDebug) {
                        accumulateDebugCellPlacement(
                            splitPerturbDebugPlacements,
                            frame.cells[cellIdx],
                            config.simulation,
                            config.simulation.perturb_debug_cell_brightness);
                        ++splitPerturbDebugPlacementCount;
                    }
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
                const float effectivePerturbRadiusRatio = perturbRatioFor(cellName);
                auto result = frame.perturbCell(
                    cellIdx, overlapWeight, useSignalGuidanceThisFrame,
                    effectivePerturbRadiusRatio,
                    /*pcaRefitWellFilledMove=*/true);
                double costDiff = result.first;
                auto callback = result.second;
                const bool perturbAccept = costDiff < 0;
                if (exportPerturbDebug) {
                    accumulateDebugCellPlacement(
                        movementPerturbDebugPlacements,
                        frame.cells[cellIdx],
                        config.simulation,
                        config.simulation.perturb_debug_cell_brightness);
                    ++movementPerturbDebugPlacementCount;
                }
                recordPerturbLoss(cellName, frame.cells[cellIdx], costDiff, perturbAccept);
                if (perturbAccept) {
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

    if (exportPerturbDebug) {
        const fs::path framePath = (frameIndex >= 0 && static_cast<size_t>(frameIndex) < imagePaths.size())
            ? imagePaths[static_cast<size_t>(frameIndex)]
            : fs::path(frame.getImageName());
        std::cout << "[Perturb Debug Export] frame " << displayFrame
                  << " movement_placements=" << movementPerturbDebugPlacementCount
                  << " split_placements=" << splitPerturbDebugPlacementCount
                  << " brightness=" << config.simulation.perturb_debug_cell_brightness
                  << " output_dir=" << (fs::path(outputPath) / "perturb_debug").string()
                  << std::endl;
        exportStackToSubdir(movementPerturbDebugPlacements,
                            fs::path(outputPath),
                            "perturb_debug/movement_placements",
                            framePath,
                            config.simulation.export_frame_png,
                            config.simulation.export_frame_tiff);
        exportStackToSubdir(splitPerturbDebugPlacements,
                            fs::path(outputPath),
                            "perturb_debug/split_placements",
                            framePath,
                            config.simulation.export_frame_png,
                            config.simulation.export_frame_tiff);
    }

    if (exportPerturbCenterDebug) {
        const fs::path framePath = (frameIndex >= 0 && static_cast<size_t>(frameIndex) < imagePaths.size())
            ? imagePaths[static_cast<size_t>(frameIndex)]
            : fs::path(frame.getImageName());
        std::vector<cv::Mat> cellCenterLabels = makeCellCenterLabelDebugStack(
            frame.getRealFrame(), frame.cells,
            config.simulation.perturb_debug_center_cube_radius);
        std::cout << "[Perturb Debug Center Export] frame " << displayFrame
                  << " cells=" << frame.cells.size()
                  << " cube_radius=" << config.simulation.perturb_debug_center_cube_radius
                  << " output_dir=" << (fs::path(outputPath) / "perturb_debug/cell_centers").string()
                  << std::endl;
        exportStackToSubdir(cellCenterLabels,
                            fs::path(outputPath),
                            "perturb_debug/cell_centers",
                            framePath,
                            config.simulation.export_frame_png,
                            config.simulation.export_frame_tiff);
    }

    runPcaShapeFit();

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

    bool trashRemovalUpdatedBackground = false;
    if (config.cell && config.cell->trashRemovalEnabled) {
        const float threshold = config.cell->trashRemovalBrightnessThreshold;
        float removedBrightnessSum = 0.0f;
        std::vector<std::string> removedNames;

        auto &cells = frame.cells;
        for (auto it = cells.begin(); it != cells.end();) {
            if (it->isTrash() && it->getBrightness() < threshold) {
                removedBrightnessSum += it->getBrightness();
                removedNames.push_back(it->getName());
                it = cells.erase(it);
            } else {
                ++it;
            }
        }

        if (!removedNames.empty()) {
            for (const std::string &name : removedNames) {
                previousSnapshots.erase(name);
                cellShapeReference.erase(name);
                cellShapeBirth.erase(name);
            }

            std::size_t backgroundVoxels =
                countBackgroundVoxelsAfterCells(frame, config.simulation);
            if (backgroundVoxels == 0 && !frame.getRealFrame().empty()) {
                const auto &realFrame = frame.getRealFrame();
                backgroundVoxels = static_cast<std::size_t>(realFrame.front().rows) *
                                   static_cast<std::size_t>(realFrame.front().cols) *
                                   realFrame.size();
            }
            backgroundVoxels = std::max<std::size_t>(backgroundVoxels, 1);

            const float backgroundDelta =
                removedBrightnessSum / static_cast<float>(backgroundVoxels);
            const float updatedBackground = std::clamp(
                frame.getBackgroundValue() + backgroundDelta, 0.0f, 1.0f);
            frame.setBackgroundColor(updatedBackground);
            frame.regenerateSynthFrame();
            trashRemovalUpdatedBackground = true;

            std::cout << "[Trash Removal] frame " << displayFrame
                      << " removed=" << removedNames.size()
                      << " brightnessSum=" << removedBrightnessSum
                      << " backgroundVoxels=" << backgroundVoxels
                      << " backgroundDelta=" << backgroundDelta
                      << " background=" << updatedBackground
                      << std::endl;
        }
    }

    std::cout << "[Optimize Done] frame " << displayFrame
              << " perturb_accepted=" << perturbAccepted
              << " split_attempts=" << splitAttempted
              << " split_accepted=" << splitAccepted
              << " final_cells=" << frame.cells.size() << std::endl;

    // M1/M2 cache per-frame summaries so optimize(frameIndex+1) doesn't need
    // frames[frameIndex]'s image stacks.
    perFrameAdaptiveBackground[frameIndex] = trashRemovalUpdatedBackground
        ? frame.getBackgroundValue()
        : estimateAdaptiveBackgroundFromFrame(frame, config.simulation);
    perFrameMeanBrightness[frameIndex] = computeStackMean(frame.getRealFrame());
}

void CellUniverse::releaseFrameImages(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        return;
    }
    frames[frameIndex].releaseImageStacks();
}

void CellUniverse::saveImages(int frameIndex, const std::string &stage)
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
    const std::string exportRoot = stage.empty() ? outputPath : (outputPath + "/" + stage);

    if (config.simulation.export_frame_png)
    {
        std::string realOutputPath = exportRoot + "/png/real/" + std::to_string(displayFrame);
        if (!std::filesystem::exists(realOutputPath))
        {
            std::filesystem::create_directories(realOutputPath);
        }
        std::string synthOutputPath = exportRoot + "/png/synth/" + std::to_string(displayFrame);
        if (!std::filesystem::exists(synthOutputPath))
        {
            std::filesystem::create_directories(synthOutputPath);
        }

        // Parallelize PNG encoding/write: independent across slices and stacks.
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
    }

    if (config.simulation.export_frame_tiff)
    {
        const std::string realTiffOutputPath = exportRoot + "/tiff/real";
        const std::string synthTiffOutputPath = exportRoot + "/tiff/synth";
        std::filesystem::create_directories(realTiffOutputPath);
        std::filesystem::create_directories(synthTiffOutputPath);

        writeNapariFriendlyTiffStack(realTiffOutputPath + "/" + std::to_string(displayFrame) + ".tif",
                                     realImages);
        writeNapariFriendlyTiffStack(synthTiffOutputPath + "/" + std::to_string(displayFrame) + ".tif",
                                     synthImages);
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
        file << "file,name,x,y,z,aRadius,bRadius,cRadius,theta_x,theta_y,theta_z,isTrash" << '\n';
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
             << params.theta_z << ","
             << (params.isTrash ? 1 : 0)
             << '\n';
    }

    std::cout << "Saved " << frame.cells.size() << " cells for frame " << (firstFrame + frameIndex)
              << " to " << cellsPath << std::endl;
}

// ---------------------------------------------------------------------------
// Checkpoint save / load (Approach 2).
// ---------------------------------------------------------------------------
//
// Format: simple line-oriented text. One directive per line, space-separated
// fields. Each checkpoint file holds the full state needed to resume an
// optimization run at frameIndex+1.

void CellUniverse::saveCheckpoint(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size()) return;
    const std::string dir = outputPath + "/checkpoints";
    fs::create_directories(dir);
    char buf[64];
    const int absoluteFrame = firstFrame + frameIndex;
    std::snprintf(buf, sizeof(buf), "frame_%03d.txt", absoluteFrame);
    const std::string path = dir + "/" + buf;

    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "[Checkpoint] failed to open " << path << '\n';
        return;
    }

    // Scalar header
    out << "frame " << absoluteFrame << '\n';
    out << "z_slices " << config.simulation.z_slices << '\n';
    out << "maxZ " << Ellipsoid::cellConfig.maxZ << '\n';

    // Per-frame summaries (indexed by frameIndex, since we cache AFTER
    // optimize(frameIndex) completes).
    const float adaptiveBg = (static_cast<size_t>(frameIndex) < perFrameAdaptiveBackground.size())
        ? perFrameAdaptiveBackground[frameIndex] : 0.0f;
    const float meanBright = (static_cast<size_t>(frameIndex) < perFrameMeanBrightness.size())
        ? perFrameMeanBrightness[frameIndex] : 0.0f;
    out << "perFrameAdaptiveBackground " << adaptiveBg << '\n';
    out << "perFrameMeanBrightness " << meanBright << '\n';

    // Next frame's background value (what optimize(frameIndex+1) will use
    // as initial bg for the frame being optimized).
    if (static_cast<size_t>(frameIndex + 1) < frames.size()) {
        out << "nextFrameBackgroundValue " << frames[frameIndex + 1].getBackgroundValue() << '\n';
    }

    // Cells of frameIndex + 1 (already copied forward, ready for optimize).
    // If frame+1 doesn't exist (last frame), save frame N's cells instead.
    const size_t cellsFrameIdx = (static_cast<size_t>(frameIndex + 1) < frames.size())
        ? static_cast<size_t>(frameIndex + 1) : static_cast<size_t>(frameIndex);
    const auto &cells = frames[cellsFrameIdx].cells;
    for (const auto &cell : cells) {
        auto p = cell.getCellParams();
        out << "cell " << p.name
            << " " << p.x << " " << p.y << " " << p.z
            << " " << p.aRadius << " " << p.bRadius << " " << p.cRadius
            << " " << p.theta_x << " " << p.theta_y << " " << p.theta_z
            << " " << p.brightness
            << " " << (p.isTrash ? 1 : 0) << '\n';
    }

    // Previous-frame snapshots (per-cell PCA fit state for split detection).
    for (const auto &kv : previousSnapshots) {
        const auto &s = kv.second;
        out << "snap " << kv.first
            << " " << (s.valid ? 1 : 0)
            << " " << s.shapeElongation
            << " " << s.splitAxisDir.x << " " << s.splitAxisDir.y << " " << s.splitAxisDir.z
            << " " << s.splitAxisLength
            << " " << s.position.x << " " << s.position.y << " " << s.position.z
            << " " << s.aRadius << " " << s.bRadius << " " << s.cRadius
            << " " << s.thetaX << " " << s.thetaY << " " << s.thetaZ
            << " " << s.brightness << '\n';
    }

    // Shape reference (bounded growth ref for fit-cap).
    for (const auto &kv : cellShapeReference) {
        out << "ref " << kv.first
            << " " << kv.second[0] << " " << kv.second[1] << " " << kv.second[2] << '\n';
    }

    // Shape birth (frozen birth radii for mask + birth cap).
    for (const auto &kv : cellShapeBirth) {
        out << "birth " << kv.first
            << " " << kv.second[0] << " " << kv.second[1] << " " << kv.second[2] << '\n';
    }

    out.close();
    std::cout << "[Checkpoint] saved frame " << absoluteFrame << " → " << path << std::endl;
}

bool CellUniverse::loadCheckpoint(int frameIndex, const std::string &checkpointPath)
{
    return loadCheckpoint(frameIndex, frameIndex + 1, checkpointPath);
}

bool CellUniverse::loadCheckpoint(int checkpointFrameIndex,
                                  int targetFrameIndex,
                                  const std::string &checkpointPath)
{
    std::ifstream in(checkpointPath);
    if (!in.is_open()) {
        std::cerr << "[Checkpoint] cannot open " << checkpointPath << '\n';
        return false;
    }

    // Clear maps that will be repopulated.
    previousSnapshots.clear();
    cellShapeReference.clear();
    cellShapeBirth.clear();
    resumePreviousFrameSummaryValid = false;
    resumePreviousAdaptiveBackground = 0.0f;
    resumePreviousMeanBrightness = 0.0f;
    resumePreviousCellCount = 0;

    if (targetFrameIndex < 0 || static_cast<size_t>(targetFrameIndex) >= frames.size()) {
        std::cerr << "[Checkpoint] target frame out of range: " << targetFrameIndex << '\n';
        return false;
    }
    const size_t cellsFrameIdx = static_cast<size_t>(targetFrameIndex);
    frames[cellsFrameIdx].cells.clear();

    const int targetAbsoluteFrame = firstFrame + targetFrameIndex;
    const bool checkpointMatchesLocal =
        checkpointFrameIndex == (targetFrameIndex - 1);
    const int expectedCheckpointFrame = targetAbsoluteFrame - 1;
    if (checkpointFrameIndex != expectedCheckpointFrame && !checkpointMatchesLocal) {
        std::cerr << "[Checkpoint] caller checkpoint frame " << checkpointFrameIndex
                  << " does not match expected absolute " << expectedCheckpointFrame
                  << " for target local frame " << targetFrameIndex << '\n';
    }

    std::string line;
    int loadedCells = 0, loadedSnaps = 0, loadedRefs = 0, loadedBirths = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "frame") {
            int f; ss >> f;
            // Accept both the new absolute frame tag and legacy local tags
            // from older first_frame=0 runs.
            if (f != checkpointFrameIndex && f != targetFrameIndex - 1) {
                std::cerr << "[Checkpoint] frame mismatch: file=" << f
                          << " expected=" << checkpointFrameIndex << '\n';
            }
        } else if (tag == "z_slices") {
            int v; ss >> v;
            config.simulation.z_slices = v;
        } else if (tag == "maxZ") {
            float v; ss >> v;
            Ellipsoid::cellConfig.maxZ = v;
        } else if (tag == "perFrameAdaptiveBackground") {
            float v; ss >> v;
            resumePreviousAdaptiveBackground = v;
            resumePreviousFrameSummaryValid = true;
            if (targetFrameIndex > 0) {
                perFrameAdaptiveBackground[targetFrameIndex - 1] = v;
            }
        } else if (tag == "perFrameMeanBrightness") {
            float v; ss >> v;
            resumePreviousMeanBrightness = v;
            resumePreviousFrameSummaryValid = true;
            if (targetFrameIndex > 0) {
                perFrameMeanBrightness[targetFrameIndex - 1] = v;
            }
        } else if (tag == "nextFrameBackgroundValue") {
            float v; ss >> v;
            frames[cellsFrameIdx].setBackgroundColor(v);
        } else if (tag == "cell") {
            EllipsoidParams p;
            ss >> p.name >> p.x >> p.y >> p.z
               >> p.aRadius >> p.bRadius >> p.cRadius
               >> p.theta_x >> p.theta_y >> p.theta_z
               >> p.brightness;
            int isTrashInt = 0;
            if (ss >> isTrashInt) {
                p.isTrash = (isTrashInt != 0);
            }
            frames[cellsFrameIdx].cells.emplace_back(p);
            ++loadedCells;
        } else if (tag == "snap") {
            PreviousFrameSnapshot s;
            std::string name;
            int validInt = 0;
            ss >> name >> validInt >> s.shapeElongation
               >> s.splitAxisDir.x >> s.splitAxisDir.y >> s.splitAxisDir.z
               >> s.splitAxisLength
               >> s.position.x >> s.position.y >> s.position.z
               >> s.aRadius >> s.bRadius >> s.cRadius
               >> s.thetaX >> s.thetaY >> s.thetaZ
               >> s.brightness;
            s.valid = (validInt != 0);
            previousSnapshots[name] = s;
            ++loadedSnaps;
        } else if (tag == "ref") {
            std::string name; std::array<float, 3> r;
            ss >> name >> r[0] >> r[1] >> r[2];
            cellShapeReference[name] = r;
            ++loadedRefs;
        } else if (tag == "birth") {
            std::string name; std::array<float, 3> r;
            ss >> name >> r[0] >> r[1] >> r[2];
            cellShapeBirth[name] = r;
            ++loadedBirths;
        }
    }
    in.close();
    resumePreviousCellCount = frames[cellsFrameIdx].cells.size();

    std::cout << "[Checkpoint] loaded frame " << checkpointFrameIndex
              << " into local frame " << targetFrameIndex
              << " (absolute " << targetAbsoluteFrame << ")"
              << " cells=" << loadedCells
              << " snaps=" << loadedSnaps
              << " refs=" << loadedRefs
              << " births=" << loadedBirths << std::endl;
    return true;
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
