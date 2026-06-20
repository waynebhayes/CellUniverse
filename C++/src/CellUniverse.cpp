#include "../includes/CellUniverse.hpp"
#include "../includes/ImageHandler.hpp"
#include "../includes/CellLumen.hpp"
#include <set>
#include <cmath>
#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <numeric>
#include <utility>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <queue>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <system_error>
#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_max_threads()
{
    return 1;
}

static int omp_in_parallel()
{
    return 0;
}
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

static float squaredDistance(const cv::Point3f &a, const cv::Point3f &b)
{
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

struct CandidateGraphRow {
    int frame = 0;
    std::string kind;
    std::string source;
    std::string parent;
    std::string candidateA;
    std::string candidateB;
    int selected = 0;
    double score = 0.0;
    double rawScore = 0.0;
    double imageGain = 0.0;
    double overlapCost = 0.0;
    double bridgeMetric = 0.0;
    float sep = 0.0f;
    float minSep = 0.0f;
    float maxSep = 0.0f;
    float midpointDist = 0.0f;
    float parentShape = 0.0f;
    float parentPersistencePenalty = 0.0f;
    float neighborClaimPenalty = 0.0f;
    float parentDistNear = 0.0f;
    float parentDistFar = 0.0f;
    float parentDistBalance = 0.0f;
    cv::Point3f d1;
    cv::Point3f d2;
    int voxA = 0;
    int voxB = 0;
    float signalA = 0.0f;
    float signalB = 0.0f;
    std::string note;
};

static std::string csvEscape(const std::string &value)
{
    const bool needsQuotes =
        value.find_first_of(",\"\n\r") != std::string::npos;
    if (!needsQuotes) {
        return value;
    }

    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char ch : value) {
        if (ch == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}

static std::ofstream openCandidateGraphLog(const std::string &outputPath,
                                           int absoluteFrame)
{
    const fs::path graphDir = fs::path(outputPath) / "candidate_graph";
    std::error_code ec;
    fs::create_directories(graphDir, ec);
    if (ec) {
        return {};
    }

    const fs::path csvPath =
        graphDir / ("frame_" + std::to_string(absoluteFrame) + "_candidates.csv");
    bool writeHeader = true;
    if (fs::exists(csvPath, ec) && !ec) {
        const auto size = fs::file_size(csvPath, ec);
        writeHeader = ec || size == 0;
    }

    std::ofstream out(csvPath, std::ios::app);
    if (out && writeHeader) {
        out << "frame,kind,source,parent,candidate_a,candidate_b,selected,"
            << "score,raw_score,image_gain,overlap_cost,bridge_metric,"
            << "sep,min_sep,max_sep,midpoint_dist,parent_shape,"
            << "parent_persistence_penalty,neighbor_claim_penalty,parent_dist_near,"
            << "parent_dist_far,parent_dist_balance,x1,y1,z1,x2,y2,z2,vox_a,vox_b,"
            << "signal_a,signal_b,note\n";
    }
    return out;
}

static void writeCandidateGraphRow(std::ostream &out,
                                   const CandidateGraphRow &row)
{
    if (!out) {
        return;
    }
    out << row.frame << ','
        << csvEscape(row.kind) << ','
        << csvEscape(row.source) << ','
        << csvEscape(row.parent) << ','
        << csvEscape(row.candidateA) << ','
        << csvEscape(row.candidateB) << ','
        << row.selected << ','
        << row.score << ','
        << row.rawScore << ','
        << row.imageGain << ','
        << row.overlapCost << ','
        << row.bridgeMetric << ','
        << row.sep << ','
        << row.minSep << ','
        << row.maxSep << ','
        << row.midpointDist << ','
        << row.parentShape << ','
        << row.parentPersistencePenalty << ','
        << row.neighborClaimPenalty << ','
        << row.parentDistNear << ','
        << row.parentDistFar << ','
        << row.parentDistBalance << ','
        << row.d1.x << ','
        << row.d1.y << ','
        << row.d1.z << ','
        << row.d2.x << ','
        << row.d2.y << ','
        << row.d2.z << ','
        << row.voxA << ','
        << row.voxB << ','
        << row.signalA << ','
        << row.signalB << ','
        << csvEscape(row.note) << '\n';
}

struct BrightnessNeighborhoodStats {
    int innerVoxels = 0;
    int shellVoxels = 0;
    float innerMean = 0.0f;
    float shellMean = 0.0f;
    float innerTop10Mean = 0.0f;
    float innerMax = 0.0f;
};

static float readStackVoxelAsFloat(const cv::Mat &slice, int y, int x)
{
    switch (slice.type()) {
    case CV_32F:
        return slice.at<float>(y, x);
    case CV_64F:
        return static_cast<float>(slice.at<double>(y, x));
    case CV_8U:
        return static_cast<float>(slice.at<unsigned char>(y, x)) / 255.0f;
    case CV_16U:
        return static_cast<float>(slice.at<unsigned short>(y, x)) / 65535.0f;
    default:
        return static_cast<float>(slice.at<float>(y, x));
    }
}

static float medianOrZero(std::vector<float> values)
{
    if (values.empty()) {
        return 0.0f;
    }
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    float median = values[mid];
    if (values.size() % 2 == 0) {
        const auto lowerIt = std::max_element(values.begin(), values.begin() + mid);
        median = 0.5f * (median + *lowerIt);
    }
    return median;
}

static BrightnessNeighborhoodStats sampleBrightnessNeighborhood(
    const std::vector<cv::Mat> &stack,
    const cv::Point3f &center,
    float innerRadius,
    float shellRadius)
{
    BrightnessNeighborhoodStats stats;
    if (stack.empty() || stack[0].empty()) {
        return stats;
    }

    innerRadius = std::max(1.0f, innerRadius);
    shellRadius = std::max(innerRadius + 1.0f, shellRadius);
    const float innerSq = innerRadius * innerRadius;
    const float shellSq = shellRadius * shellRadius;
    const int maxZ = static_cast<int>(stack.size()) - 1;
    const int maxY = stack[0].rows - 1;
    const int maxX = stack[0].cols - 1;
    const int minZ = std::max(0, static_cast<int>(std::floor(center.z - shellRadius)));
    const int maxZBound = std::min(maxZ, static_cast<int>(std::ceil(center.z + shellRadius)));
    const int minY = std::max(0, static_cast<int>(std::floor(center.y - shellRadius)));
    const int maxYBound = std::min(maxY, static_cast<int>(std::ceil(center.y + shellRadius)));
    const int minX = std::max(0, static_cast<int>(std::floor(center.x - shellRadius)));
    const int maxXBound = std::min(maxX, static_cast<int>(std::ceil(center.x + shellRadius)));

    double innerSum = 0.0;
    double shellSum = 0.0;
    std::vector<float> innerValues;
    for (int z = minZ; z <= maxZBound; ++z) {
        const cv::Mat &slice = stack[static_cast<size_t>(z)];
        for (int y = minY; y <= maxYBound; ++y) {
            for (int x = minX; x <= maxXBound; ++x) {
                const float dx = static_cast<float>(x) - center.x;
                const float dy = static_cast<float>(y) - center.y;
                const float dz = static_cast<float>(z) - center.z;
                const float distSq = dx * dx + dy * dy + dz * dz;
                if (distSq > shellSq) {
                    continue;
                }
                const float value = readStackVoxelAsFloat(slice, y, x);
                if (distSq <= innerSq) {
                    innerSum += value;
                    innerValues.push_back(value);
                    stats.innerMax = std::max(stats.innerMax, value);
                    ++stats.innerVoxels;
                } else {
                    shellSum += value;
                    ++stats.shellVoxels;
                }
            }
        }
    }

    if (stats.innerVoxels > 0) {
        stats.innerMean = static_cast<float>(innerSum / stats.innerVoxels);
        const size_t topCount = std::max<size_t>(
            1, static_cast<size_t>(std::ceil(innerValues.size() * 0.10)));
        std::nth_element(innerValues.begin(),
                         innerValues.begin() + static_cast<std::ptrdiff_t>(topCount - 1),
                         innerValues.end(),
                         std::greater<float>());
        double topSum = 0.0;
        for (size_t i = 0; i < topCount; ++i) {
            topSum += innerValues[i];
        }
        stats.innerTop10Mean = static_cast<float>(topSum / topCount);
    }
    if (stats.shellVoxels > 0) {
        stats.shellMean = static_cast<float>(shellSum / stats.shellVoxels);
    }
    return stats;
}

class ScopedStageTimer {
public:
    ScopedStageTimer(int frame, std::string stage)
        : frame(frame),
          stage(std::move(stage)),
          start(std::chrono::steady_clock::now())
    {
    }

    ~ScopedStageTimer()
    {
        finish();
    }

    double finish()
    {
        if (!active) {
            return elapsedSeconds;
        }
        const auto end = std::chrono::steady_clock::now();
        elapsedSeconds = std::chrono::duration<double>(end - start).count();
        active = false;
        std::cout << "[Stage Timing] frame " << frame
                  << " stage=" << stage
                  << " seconds=" << elapsedSeconds
                  << std::endl;
        return elapsedSeconds;
    }

private:
    int frame = 0;
    std::string stage;
    std::chrono::steady_clock::time_point start;
    bool active = true;
    double elapsedSeconds = 0.0;
};

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

static bool usesIterativePostPreprocessing(const SimulationConfig &config)
{
    return config.preprocess_mode == "iterative";
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

    // Determine post-interpolation z_slices depth. getImageFilePaths() records
    // the raw TIFF slice count in config.simulation.z_slices, but optimizer
    // cells live in the interpolated z space. Keep Ellipsoid::cellConfig.maxZ
    // in that same coordinate system, otherwise newly built split daughters get
    // clamped back to the raw stack depth.
    if (!config.simulation.quit_after_preprocessing &&
        config.simulation.z_slices > 0) {
        const int rawSlices = config.simulation.z_slices;
        const int zScale = std::max(1, static_cast<int>(config.simulation.z_scaling));
        const int runtimeSlices = (rawSlices > 1)
            ? zScale * (rawSlices - 1) + 1
            : rawSlices;
        config.simulation.z_slices = runtimeSlices;
        Ellipsoid::cellConfig.maxZ =
            static_cast<float>(config.simulation.z_slices) - 1.0f;
        std::cout << "[M2 Probe] computed_from_raw_z_slices="
                  << rawSlices
                  << " z_scaling=" << zScale
                  << " runtime_z_slices=" << config.simulation.z_slices
                  << " maxZ=" << Ellipsoid::cellConfig.maxZ << '\n';
    } else if (!config.simulation.quit_after_preprocessing) {
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

    // By default, loaded initial cells keep the legacy mature marker so resume
    // runs are unchanged. Sparse embryo runs can opt in to firstFrame aging so
    // the split-prior parent-age guard prevents immediate CellLumen over-splits
    // from bright sub-peaks inside a large early cell.
    const int oldInitialFirstSeenFrame =
        config.cellLumen.fusionSplitPriorTreatInitialCellsAsNew
            ? firstFrame
            : firstFrame - 100000;
    for (const auto &entry : initialCells) {
        for (const auto &cell : entry.second) {
            cellFirstSeenFrame.try_emplace(cell.getName(), oldInitialFirstSeenFrame);
        }
    }
}

void CellUniverse::applyCellLumenRescue(int frameIndex)
{
    const CellLumenConfig &lumenConfig = config.cellLumen;
    if (!lumenConfig.enabled || !lumenConfig.fusionEnabled) {
        return;
    }
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size()) {
        throw std::invalid_argument("applyCellLumenRescue: invalid frame index");
    }

    const int absoluteFrame = firstFrame + frameIndex;
    if (absoluteFrame < lumenConfig.fusionStartFrame) {
        return;
    }
    const int everyN = std::max(1, lumenConfig.fusionEveryNFrames);
    if (((absoluteFrame - lumenConfig.fusionStartFrame) % everyN) != 0) {
        return;
    }
    if (lumenConfig.fusionMaxAddedPerFrame == 0 &&
        !lumenConfig.fusionCenterPriorEnabled &&
        !lumenConfig.fusionRepairCloseCellsEnabled &&
        !lumenConfig.fusionSplitPriorEnabled) {
        return;
    }

    Frame &frame = frames[static_cast<size_t>(frameIndex)];
    const fs::path framePath = imagePaths[static_cast<size_t>(frameIndex)];
    auto &splitPriorsForFrame = cellLumenSplitPriors[frameIndex];
    splitPriorsForFrame.clear();
    auto &centerReanchorCandidateIdsForFrame =
        cellLumenCenterReanchorCandidateIds[frameIndex];
    centerReanchorCandidateIdsForFrame.clear();
    auto &badSplitPriorParentsForFrame = cellLumenSplitPriorRejectedBadParents[frameIndex];
    badSplitPriorParentsForFrame.clear();
    auto &collapsedCenterParentsForFrame = cellLumenCollapsedCenterParents[frameIndex];
    collapsedCenterParentsForFrame.clear();
    auto &centerCandidatesForFrame = cellLumenCenterCandidates[frameIndex];
    centerCandidatesForFrame.clear();

    EllipsoidConfig savedCellConfig = Ellipsoid::cellConfig;
    std::vector<CellLumen::DetectedCell> candidates;
    try {
        CellLumen lumen(config, fs::path(outputPath) / "cell_lumen_fusion");
        candidates = lumen.detectCellsForFrame(framePath, false);
    } catch (const std::exception &ex) {
        Ellipsoid::cellConfig = savedCellConfig;
        std::cout << "[CellLumen Fusion] frame=" << absoluteFrame
                  << " status=skipped error=\"" << ex.what() << "\""
                  << std::endl;
        return;
    }
    Ellipsoid::cellConfig = savedCellConfig;

    std::sort(candidates.begin(), candidates.end(),
              [](const CellLumen::DetectedCell &a, const CellLumen::DetectedCell &b) {
                  if (a.top10MinusShell != b.top10MinusShell) {
                      return a.top10MinusShell > b.top10MinusShell;
                  }
                  return a.voxelCount > b.voxelCount;
              });

    std::ofstream candidateGraphLog = openCandidateGraphLog(outputPath, absoluteFrame);
    if (candidateGraphLog) {
        for (const auto &cell : frame.cells) {
            CandidateGraphRow row;
            row.frame = absoluteFrame;
            row.kind = "continuation";
            row.source = "current_cell_state";
            row.parent = cell.getName();
            row.selected = 1;
            row.parentShape = cell.shapeElongation();
            row.d1 = cv::Point3f(cell.getX(), cell.getY(), cell.getZ());
            row.note = "state_before_cell_lumen_rescue";
            writeCandidateGraphRow(candidateGraphLog, row);
        }
        for (size_t candIdx = 0; candIdx < candidates.size(); ++candIdx) {
            const auto &candidate = candidates[candIdx];
            CandidateGraphRow row;
            row.frame = absoluteFrame;
            row.kind = "lumen_center";
            row.source = "cell_lumen_high_recall";
            row.candidateA = std::to_string(candIdx);
            row.d1 = candidate.centerScaled;
            row.voxA = candidate.voxelCount;
            row.signalA = candidate.top10MinusShell;
            row.note = "cell_lumen_high_recall_center";
            writeCandidateGraphRow(candidateGraphLog, row);
        }
    }

    const float minDistance = std::max(0.0f, lumenConfig.fusionMinDistanceToExisting);
    const float minDistanceSq = minDistance * minDistance;
    const int maxAdded = lumenConfig.fusionMaxAddedPerFrame < 0
                             ? std::numeric_limits<int>::max()
                             : lumenConfig.fusionMaxAddedPerFrame;
    const float radiusScale = std::max(0.05f, lumenConfig.fusionRadiusScale);
    const float brightness = (lumenConfig.fusionBrightness >= 0.0f)
                                 ? lumenConfig.fusionBrightness
                                 : (config.cell ? config.cell->initialBrightness : 0.5f);

    const float centerCandidateMaxDistance = std::max(
        0.0f,
        lumenConfig.fusionCenterPriorMaxDistance > 0.0f
            ? lumenConfig.fusionCenterPriorMaxDistance
            : lumenConfig.fusionMinDistanceToExisting);
    const float centerCandidateMaxDistanceSq =
        centerCandidateMaxDistance * centerCandidateMaxDistance;
    struct CenterPriorClusterItem {
        size_t candidateIndex = 0;
        float distance = 0.0f;
    };
    const int centerClusterMaxCells =
        lumenConfig.fusionCenterPriorClusterCollapseMaxCells;
    const float centerClusterRadius = std::max(
        centerCandidateMaxDistance,
        std::max(0.0f, lumenConfig.fusionCenterPriorClusterCollapseRadius));
    const float centerClusterRadiusSq = centerClusterRadius * centerClusterRadius;
    const bool centerClusterCollapseEnabled =
        lumenConfig.fusionCenterPriorClusterCollapseEnabled &&
        centerClusterMaxCells > 0 &&
        static_cast<int>(frame.cells.size()) <= centerClusterMaxCells &&
        centerClusterRadius > 0.0f;
    const bool farSingleCenterPriorEnabled =
        lumenConfig.fusionCenterPriorFarSingleEnabled &&
        lumenConfig.fusionCenterPriorFarSingleMaxCells > 0 &&
        static_cast<int>(frame.cells.size()) <=
            lumenConfig.fusionCenterPriorFarSingleMaxCells;
    const float farSingleMinDistance = std::max(
        centerCandidateMaxDistance,
        lumenConfig.fusionCenterPriorFarSingleMinDistance);
    const float farSingleMaxDistance = std::max(
        farSingleMinDistance,
        lumenConfig.fusionCenterPriorFarSingleMaxDistance);
    const float farSingleMaxDistanceSq =
        farSingleMaxDistance * farSingleMaxDistance;
    const int farSingleMinVoxels =
        std::max(0, lumenConfig.fusionCenterPriorFarSingleMinVoxels);
    const float farSingleMinSignal =
        std::max(0.0f, lumenConfig.fusionCenterPriorFarSingleMinSignal);
    const float farSinglePositionBlend = std::clamp(
        lumenConfig.fusionCenterPriorFarSinglePositionBlend, 0.0f, 1.0f);
    const bool youngFarSingleCenterPriorEnabled =
        lumenConfig.fusionCenterPriorYoungFarSingleEnabled &&
        lumenConfig.fusionCenterPriorYoungFarSingleMaxCells > 0 &&
        static_cast<int>(frame.cells.size()) <=
            lumenConfig.fusionCenterPriorYoungFarSingleMaxCells;
    const int youngFarSingleMaxAgeFrames =
        std::max(0, lumenConfig.fusionCenterPriorYoungFarSingleMaxAgeFrames);
    const float youngFarSingleMinDistance = std::max(
        centerCandidateMaxDistance,
        lumenConfig.fusionCenterPriorYoungFarSingleMinDistance);
    const float youngFarSingleMaxDistance = std::max(
        youngFarSingleMinDistance,
        lumenConfig.fusionCenterPriorYoungFarSingleMaxDistance);
    const float youngFarSingleMaxDistanceSq =
        youngFarSingleMaxDistance * youngFarSingleMaxDistance;
    const int youngFarSingleMinVoxels =
        std::max(0, lumenConfig.fusionCenterPriorYoungFarSingleMinVoxels);
    const float youngFarSingleMinSignal =
        std::max(0.0f, lumenConfig.fusionCenterPriorYoungFarSingleMinSignal);
    const float youngFarSinglePositionBlend = std::clamp(
        lumenConfig.fusionCenterPriorYoungFarSinglePositionBlend, 0.0f, 1.0f);
    const bool youngFarSinglePreferPositiveZShift =
        lumenConfig.fusionCenterPriorYoungFarSinglePreferPositiveZShift;
    const float youngFarSingleMinPositiveZShift =
        std::max(0.0f, lumenConfig.fusionCenterPriorYoungFarSingleMinPositiveZShift);
    const float youngFarSingleMinExtraZShift =
        std::max(0.0f, lumenConfig.fusionCenterPriorYoungFarSingleMinExtraZShift);
    std::unordered_map<size_t, std::vector<CenterPriorClusterItem>>
        centerPriorClusterBuckets;
    std::unordered_map<size_t, std::unordered_set<int>>
        collapsedCenterClusterMembersByParent;
    for (size_t candIdx = 0; candIdx < candidates.size(); ++candIdx) {
        const auto &candidate = candidates[candIdx];
        const bool passesBaseFusionGate =
            candidate.voxelCount >= lumenConfig.fusionMinVoxels &&
            candidate.top10MinusShell >= lumenConfig.fusionMinTop10MinusShell;
        // Newborn daughters can produce smaller Cell Lumen blobs during the
        // first continuation frame. Let only the gated young far-single path
        // inspect those blobs; ordinary center priors and cluster collapse keep
        // using the older conservative fusionMin* gate.
        const bool passesYoungFarSingleSignalGate =
            youngFarSingleCenterPriorEnabled &&
            candidate.voxelCount >= youngFarSingleMinVoxels &&
            candidate.top10MinusShell >= youngFarSingleMinSignal;
        if (!passesBaseFusionGate && !passesYoungFarSingleSignalGate) {
            continue;
        }

        size_t nearestIndex = std::numeric_limits<size_t>::max();
        float nearestDistanceSq = std::numeric_limits<float>::max();
        for (size_t existingIdx = 0; existingIdx < frame.cells.size(); ++existingIdx) {
            const cv::Point3f existingCenter(frame.cells[existingIdx].getX(),
                                             frame.cells[existingIdx].getY(),
                                             frame.cells[existingIdx].getZ());
            const float distSq = squaredDistance(candidate.centerScaled, existingCenter);
            if (distSq < nearestDistanceSq) {
                nearestDistanceSq = distSq;
                nearestIndex = existingIdx;
            }
        }
        if (nearestIndex == std::numeric_limits<size_t>::max() ||
            nearestDistanceSq > centerCandidateMaxDistanceSq) {
            if (passesBaseFusionGate &&
                centerClusterCollapseEnabled &&
                nearestIndex != std::numeric_limits<size_t>::max() &&
                nearestDistanceSq <= centerClusterRadiusSq) {
                centerPriorClusterBuckets[nearestIndex].push_back(
                    {candIdx, std::sqrt(nearestDistanceSq)});
            }
            if ((farSingleCenterPriorEnabled ||
                 youngFarSingleCenterPriorEnabled) &&
                nearestIndex != std::numeric_limits<size_t>::max() &&
                (nearestDistanceSq <= farSingleMaxDistanceSq ||
                 nearestDistanceSq <= youngFarSingleMaxDistanceSq)) {
                const float distance = std::sqrt(nearestDistanceSq);
                const std::string parentName =
                    frame.cells[nearestIndex].getName();
                const bool strongFarSingle =
                    farSingleCenterPriorEnabled &&
                    distance >= farSingleMinDistance &&
                    distance <= farSingleMaxDistance &&
                    candidate.voxelCount >= farSingleMinVoxels &&
                    candidate.top10MinusShell >= farSingleMinSignal;
                int parentAgeFrames = std::numeric_limits<int>::max();
                if (const auto firstSeenIt = cellFirstSeenFrame.find(parentName);
                    firstSeenIt != cellFirstSeenFrame.end()) {
                    parentAgeFrames = std::max(0, absoluteFrame - firstSeenIt->second);
                }
                const bool strongYoungFarSingle =
                    !strongFarSingle &&
                    youngFarSingleCenterPriorEnabled &&
                    parentAgeFrames <= youngFarSingleMaxAgeFrames &&
                    distance >= youngFarSingleMinDistance &&
                    distance <= youngFarSingleMaxDistance &&
                    candidate.voxelCount >= youngFarSingleMinVoxels &&
                    candidate.top10MinusShell >= youngFarSingleMinSignal;
                if (strongFarSingle || strongYoungFarSingle) {
                    const float selectedPositionBlend =
                        strongYoungFarSingle
                            ? youngFarSinglePositionBlend
                            : farSinglePositionBlend;
                    const float candidateZShift =
                        candidate.centerScaled.z -
                        frame.cells[nearestIndex].getZ();
                    auto existing = centerCandidatesForFrame.find(parentName);
                    const bool replaceByYoungPositiveZShift =
                        strongYoungFarSingle &&
                        youngFarSinglePreferPositiveZShift &&
                        existing != centerCandidatesForFrame.end() &&
                        existing->second.youngFarSingle &&
                        candidateZShift >= youngFarSingleMinPositiveZShift &&
                        candidateZShift >
                            existing->second.parentZShift +
                                youngFarSingleMinExtraZShift;
                    const bool replaceExisting =
                        existing == centerCandidatesForFrame.end() ||
                        replaceByYoungPositiveZShift ||
                        (existing->second.distance >
                         centerCandidateMaxDistance + 1e-3f &&
                         distance < existing->second.distance - 1e-3f) ||
                        (existing->second.distance >
                         centerCandidateMaxDistance + 1e-3f &&
                         std::abs(distance - existing->second.distance) <=
                             1e-3f &&
                         candidate.top10MinusShell > existing->second.signal);
                    if (replaceExisting) {
                        CellLumenCenterCandidate stored;
                        stored.position = candidate.centerScaled;
                        stored.distance = distance;
                        stored.voxelCount = candidate.voxelCount;
                        stored.signal = candidate.top10MinusShell;
                        stored.candidateId = static_cast<int>(candIdx);
                        stored.positionBlendOverride =
                            selectedPositionBlend;
                        stored.youngFarSingle = strongYoungFarSingle;
                        stored.parentZShift = candidateZShift;
                        centerCandidatesForFrame[parentName] = stored;
                        if (candidateGraphLog) {
                            CandidateGraphRow row;
                            row.frame = absoluteFrame;
                            row.kind = "center_prior";
                            row.source =
                                strongYoungFarSingle
                                    ? "cell_lumen_young_far_single_center_prior"
                                    : "cell_lumen_far_single_center_prior";
                            row.parent = parentName;
                            row.candidateA = std::to_string(candIdx);
                            row.selected = 1;
                            row.score = distance;
                            row.d1 = candidate.centerScaled;
                            row.voxA = candidate.voxelCount;
                            row.signalA = candidate.top10MinusShell;
                            std::ostringstream note;
                            note << "position_blend="
                                 << selectedPositionBlend
                                 << ";min_distance="
                                 << (strongYoungFarSingle
                                         ? youngFarSingleMinDistance
                                         : farSingleMinDistance)
                                 << ";max_distance="
                                 << (strongYoungFarSingle
                                         ? youngFarSingleMaxDistance
                                         : farSingleMaxDistance)
                                 << ";parent_age_frames="
                                 << parentAgeFrames
                                 << ";parent_z_shift="
                                 << candidateZShift
                                 << ";positive_z_replace="
                                 << (replaceByYoungPositiveZShift ? 1 : 0)
                                 << ";reason="
                                 << (strongYoungFarSingle
                                         ? "newborn_fast_motion_single_lumen_center"
                                         : "sparse_fast_motion_single_lumen_center");
                            row.note = note.str();
                            writeCandidateGraphRow(candidateGraphLog, row);
                        }
                    }
                }
            }
            continue;
        }

        if (!passesBaseFusionGate) {
            continue;
        }

        if (centerClusterCollapseEnabled &&
            nearestDistanceSq <= centerClusterRadiusSq) {
            centerPriorClusterBuckets[nearestIndex].push_back(
                {candIdx, std::sqrt(nearestDistanceSq)});
        }

        const std::string parentName = frame.cells[nearestIndex].getName();
        const float distance = std::sqrt(nearestDistanceSq);
        auto existing = centerCandidatesForFrame.find(parentName);
        const bool replaceExisting =
            existing == centerCandidatesForFrame.end() ||
            distance < existing->second.distance - 1e-3f ||
            (std::abs(distance - existing->second.distance) <= 1e-3f &&
             candidate.top10MinusShell > existing->second.signal);
        if (!replaceExisting) {
            continue;
        }

        CellLumenCenterCandidate stored;
        stored.position = candidate.centerScaled;
        stored.distance = distance;
        stored.voxelCount = candidate.voxelCount;
        stored.signal = candidate.top10MinusShell;
        stored.candidateId = static_cast<int>(candIdx);
        centerCandidatesForFrame[parentName] = stored;
    }
    if (centerClusterCollapseEnabled) {
        const int minClusterCandidates = std::max(
            2, lumenConfig.fusionCenterPriorClusterCollapseMinCandidates);
        const float maxClusterDiameter =
            lumenConfig.fusionCenterPriorClusterCollapseMaxDiameter;
        const float distanceWeightScale = std::max(
            1.0f,
            lumenConfig.fusionCenterPriorClusterCollapseDistanceWeightScale);
        const float clusterPositionBlend = std::clamp(
            lumenConfig.fusionCenterPriorClusterCollapsePositionBlend,
            0.0f, 1.0f);
        int collapsedCenterPriors = 0;
        int skippedWideClusters = 0;

        for (const auto &bucketEntry : centerPriorClusterBuckets) {
            const size_t parentIdx = bucketEntry.first;
            const auto &bucket = bucketEntry.second;
            if (parentIdx >= frame.cells.size() ||
                static_cast<int>(bucket.size()) < minClusterCandidates) {
                continue;
            }

            float maxPairDistance = 0.0f;
            for (size_t i = 0; i < bucket.size(); ++i) {
                for (size_t j = i + 1; j < bucket.size(); ++j) {
                    const auto &a = candidates[bucket[i].candidateIndex];
                    const auto &b = candidates[bucket[j].candidateIndex];
                    maxPairDistance = std::max(
                        maxPairDistance,
                        std::sqrt(squaredDistance(a.centerScaled, b.centerScaled)));
                }
            }
            if (maxClusterDiameter > 0.0f &&
                maxPairDistance > maxClusterDiameter) {
                ++skippedWideClusters;
                continue;
            }

            double weightedX = 0.0;
            double weightedY = 0.0;
            double weightedZ = 0.0;
            double totalWeight = 0.0;
            int totalVoxels = 0;
            float maxSignal = 0.0f;
            int representativeCandidateId = -1;
            double representativeWeight = -1.0;
            for (const auto &item : bucket) {
                const auto &candidate = candidates[item.candidateIndex];
                const float signalAboveGate = std::max(
                    1.0f,
                    candidate.top10MinusShell -
                        lumenConfig.fusionMinTop10MinusShell + 1.0f);
                const float voxelWeight = std::sqrt(
                    std::max(1, candidate.voxelCount) / 1000.0f);
                const float distanceWeight =
                    1.0f / (1.0f + item.distance / distanceWeightScale);
                const double weight =
                    static_cast<double>(signalAboveGate) *
                    static_cast<double>(voxelWeight) *
                    static_cast<double>(distanceWeight);
                weightedX += candidate.centerScaled.x * weight;
                weightedY += candidate.centerScaled.y * weight;
                weightedZ += candidate.centerScaled.z * weight;
                totalWeight += weight;
                totalVoxels += candidate.voxelCount;
                maxSignal = std::max(maxSignal, candidate.top10MinusShell);
                if (weight > representativeWeight) {
                    representativeWeight = weight;
                    representativeCandidateId = static_cast<int>(item.candidateIndex);
                }
            }
            if (totalWeight <= 0.0) {
                continue;
            }

            const cv::Point3f collapsedCenter(
                static_cast<float>(weightedX / totalWeight),
                static_cast<float>(weightedY / totalWeight),
                static_cast<float>(weightedZ / totalWeight));
            const cv::Point3f parentCenter(frame.cells[parentIdx].getX(),
                                           frame.cells[parentIdx].getY(),
                                           frame.cells[parentIdx].getZ());
            const float collapsedDistance =
                std::sqrt(squaredDistance(collapsedCenter, parentCenter));
            if (collapsedDistance > centerClusterRadius + 1e-3f) {
                continue;
            }

            const std::string parentName = frame.cells[parentIdx].getName();
            CellLumenCenterCandidate stored;
            stored.position = collapsedCenter;
            stored.distance = collapsedDistance;
            stored.voxelCount = totalVoxels;
            stored.signal = maxSignal;
            stored.candidateId = representativeCandidateId;
            stored.clusterCandidateCount = static_cast<int>(bucket.size());
            stored.positionBlendOverride = clusterPositionBlend;
            stored.clusterCollapsed = true;
            centerCandidatesForFrame[parentName] = stored;
            auto &collapsedMembers =
                collapsedCenterClusterMembersByParent[parentIdx];
            for (const auto &item : bucket) {
                collapsedMembers.insert(static_cast<int>(item.candidateIndex));
            }
            collapsedCenterParentsForFrame.insert(parentName);
            ++collapsedCenterPriors;

            if (candidateGraphLog) {
                CandidateGraphRow row;
                row.frame = absoluteFrame;
                row.kind = "center_prior";
                row.source = "cell_lumen_cluster_collapse";
                row.parent = parentName;
                row.candidateA = std::to_string(representativeCandidateId);
                row.selected = 1;
                row.score = collapsedDistance;
                row.d1 = collapsedCenter;
                row.voxA = totalVoxels;
                row.signalA = maxSignal;
                std::ostringstream note;
                note << "cluster_count=" << bucket.size()
                     << ";position_blend=" << clusterPositionBlend
                     << ";radius=" << centerClusterRadius
                     << ";max_pair_distance=" << maxPairDistance
                     << ";reason=early_large_cell_internal_lumen_peaks";
                row.note = note.str();
                writeCandidateGraphRow(candidateGraphLog, row);
            }
        }

        if (collapsedCenterPriors > 0 || skippedWideClusters > 0) {
            std::cout << "[CellLumen Fusion CenterPrior ClusterCollapse] frame="
                      << absoluteFrame
                      << " collapsed=" << collapsedCenterPriors
                      << " skipped_wide=" << skippedWideClusters
                      << " buckets=" << centerPriorClusterBuckets.size()
                      << " radius=" << centerClusterRadius
                      << " min_candidates=" << minClusterCandidates
                      << " position_blend=" << clusterPositionBlend
                      << std::endl;
        }
    }
    std::unordered_map<int, std::string> centerCandidateOwnerById;
    std::unordered_map<int, float> centerCandidateOwnerDistanceById;
    for (const auto &entry : centerCandidatesForFrame) {
        const auto &candidate = entry.second;
        if (candidate.candidateId < 0) {
            continue;
        }
        centerCandidateOwnerById[candidate.candidateId] = entry.first;
        centerCandidateOwnerDistanceById[candidate.candidateId] =
            candidate.distance;
    }

    struct FusionCandidateRef {
        int candidateId = -1;
        cv::Point3f center;
        int voxelCount = 0;
        float signal = 0.0f;
        float majorRadius = 0.0f;
        float bRadius = 0.0f;
        float minorRadius = 0.0f;
        bool assignedByTemporalRepairCatch = false;
    };

    const int splitPriorMinVoxels =
        lumenConfig.fusionSplitPriorMinVoxels >= 0
            ? lumenConfig.fusionSplitPriorMinVoxels
            : lumenConfig.fusionMinVoxels;
    const float splitPriorMinTop10MinusShell =
        lumenConfig.fusionSplitPriorMinTop10MinusShell >= 0.0f
            ? lumenConfig.fusionSplitPriorMinTop10MinusShell
            : lumenConfig.fusionMinTop10MinusShell;
    auto passesFusionSplitPriorSignalGate = [&](const CellLumen::DetectedCell &candidate) {
        return candidate.voxelCount >= splitPriorMinVoxels &&
               candidate.top10MinusShell >= splitPriorMinTop10MinusShell;
    };

    auto &currentFrameLookaheadCandidates =
        cellLumenLookaheadCandidates[frameIndex];
    currentFrameLookaheadCandidates.clear();
    for (size_t candIdx = 0; candIdx < candidates.size(); ++candIdx) {
        const auto &candidate = candidates[candIdx];
        if (!passesFusionSplitPriorSignalGate(candidate)) {
            continue;
        }
        currentFrameLookaheadCandidates.push_back({
            candidate.centerScaled,
            candidate.voxelCount,
            candidate.top10MinusShell,
            static_cast<int>(candIdx)
        });
    }

    int splitPriorCandidateAssignments = 0;
    int splitPriorCandidatesPassedGate = 0;
    int splitPriorRejectedParentDistance = 0;
    int splitPriorRejectedSeparation = 0;
    int splitPriorRejectedMidpoint = 0;
    int splitPriorRejectedScore = 0;
    int splitPriorRejectedNeighborClaim = 0;
    int splitPriorRejectedContinuationOwner = 0;
    int splitPriorRejectedSignal = 0;
    int splitPriorTemporalCatchAssignments = 0;
    int splitPriorRejectedTemporalCatchPairs = 0;
    int splitPriorEarlyLargeCatchAssignments = 0;
    int splitPriorRejectedCollapsedCenterCluster = 0;
    int splitPriorSoftAllowedCollapsedCenterCluster = 0;
    if (lumenConfig.fusionSplitPriorEnabled && frame.cells.size() > 0) {
        int splitPriorLiveCells = 0;
        for (const auto &cell : frame.cells) {
            if (!cell.isTrash()) {
                ++splitPriorLiveCells;
            }
        }
        std::unordered_map<size_t, std::vector<FusionCandidateRef>> candidatesByParent;
        struct RankedSplitPrior {
            size_t parentIdx = 0;
            BridgeSplitProposal proposal;
            double score = 0.0;
            double rawScore = 0.0;
            float sep = 0.0f;
            float minSep = 0.0f;
            float maxSep = 0.0f;
            float midpointDist = 0.0f;
            int candidateA = -1;
            int candidateB = -1;
            int voxA = 0;
            int voxB = 0;
            float signalA = 0.0f;
            float signalB = 0.0f;
            float nearParentDist = 0.0f;
            float farParentDist = 0.0f;
            float parentDistanceBalance = 1.0f;
            float parentShapeElongation = 1.0f;
            float parentLongMidRatio = 1.0f;
            float parentMidShortRatio = 1.0f;
            float parentPersistencePenalty = 0.0f;
            float neighborClaimPenalty = 0.0f;
            float rankingSoftPenalty = 0.0f;
            float windowSupportScore = 0.0f;
            int windowBothDaughtersSupported = 0;
            int windowMissingDaughterCount = 0;
            int windowParentPersists = 0;
            float continuationClaimSoftPenalty = 0.0f;
            float balancedWindowBonus = 0.0f;
            std::vector<size_t> continuationClaimBlockers;
            std::string continuationClaimBlockerNames;
            bool conflictReplacementEligible = false;
            bool elongatedParentRescued = false;
            bool parentAnchored = false;
            bool temporalRepairEligible = false;
            bool temporalRepairCatchExpanded = false;
        };
        std::vector<RankedSplitPrior> rankedPriors;
        struct WindowSupportResult {
            float score = 0.0f;
            int bothDaughtersSupported = 0;
            int missingDaughterCount = 0;
            int parentPersists = 0;
        };
        std::map<int, const std::vector<CellLumenLookaheadCandidate> *> windowCandidatesByOffset;
        const bool windowEnabled =
            lumenConfig.fusionSplitPriorWindowEnabled &&
            lumenConfig.fusionSplitPriorWindowSize > 1;
        if (windowEnabled) {
            const int windowSize = std::clamp(
                lumenConfig.fusionSplitPriorWindowSize,
                2,
                5);
            for (int offset = 1; offset < windowSize; ++offset) {
                const int lookaheadFrameIndex = frameIndex + offset;
                if (lookaheadFrameIndex < 0 ||
                    static_cast<size_t>(lookaheadFrameIndex) >= frames.size()) {
                    break;
                }
                const auto &futureCandidates =
                    getCellLumenLookaheadCandidates(lookaheadFrameIndex);
                windowCandidatesByOffset[offset] = &futureCandidates;
            }
            std::cout << "[CellLumen Window] frame=" << absoluteFrame
                      << " enabled=1"
                      << " window_size=" << windowSize
                      << " loaded_offsets=" << windowCandidatesByOffset.size()
                      << " match_distance=" << lumenConfig.fusionSplitPriorWindowMatchDistance
                      << " match_distance_per_frame="
                      << lumenConfig.fusionSplitPriorWindowMatchDistancePerFrame
                      << std::endl;
        }

        auto nearestWindowCandidateDistance =
            [](const std::vector<CellLumenLookaheadCandidate> &futureCandidates,
               const cv::Point3f &point) -> float {
                float best = std::numeric_limits<float>::infinity();
                for (const auto &candidate : futureCandidates) {
                    best = std::min(
                        best,
                        static_cast<float>(cv::norm(candidate.position - point)));
                }
                return best;
            };

        auto computeWindowSupport = [&](const cv::Point3f &parentCenter,
                                        const cv::Point3f &d1,
                                        const cv::Point3f &d2) {
            WindowSupportResult result;
            if (!windowEnabled || windowCandidatesByOffset.empty()) {
                return result;
            }
            for (const auto &entry : windowCandidatesByOffset) {
                const int offset = entry.first;
                const auto *futureCandidates = entry.second;
                if (futureCandidates == nullptr || futureCandidates->empty()) {
                    result.missingDaughterCount += 2;
                    continue;
                }
                const float matchDistance =
                    std::max(0.0f, lumenConfig.fusionSplitPriorWindowMatchDistance) +
                    std::max(0.0f, lumenConfig.fusionSplitPriorWindowMatchDistancePerFrame) *
                        static_cast<float>(std::max(0, offset - 1));
                const float d1Dist = nearestWindowCandidateDistance(*futureCandidates, d1);
                const float d2Dist = nearestWindowCandidateDistance(*futureCandidates, d2);
                const float parentDist =
                    nearestWindowCandidateDistance(*futureCandidates, parentCenter);
                const bool d1Supported = d1Dist <= matchDistance;
                const bool d2Supported = d2Dist <= matchDistance;
                const bool parentPersists = parentDist <= matchDistance;
                const float offsetWeight = 1.0f / static_cast<float>(offset);
                if (d1Supported && d2Supported) {
                    ++result.bothDaughtersSupported;
                    result.score -=
                        std::max(0.0f, lumenConfig.fusionSplitPriorWindowDaughterSupportBonus) *
                        offsetWeight;
                } else {
                    const int missingCount =
                        (d1Supported ? 0 : 1) + (d2Supported ? 0 : 1);
                    result.missingDaughterCount += missingCount;
                    result.score +=
                        std::max(0.0f, lumenConfig.fusionSplitPriorWindowMissingDaughterPenalty) *
                        static_cast<float>(missingCount) *
                        0.5f *
                        offsetWeight;
                }
                if (parentPersists && !(d1Supported && d2Supported)) {
                    ++result.parentPersists;
                    result.score +=
                        std::max(0.0f, lumenConfig.fusionSplitPriorWindowParentPersistencePenalty) *
                        offsetWeight;
                }
            }
            return result;
        };

        for (size_t candIdx = 0; candIdx < candidates.size(); ++candIdx) {
            const auto &candidate = candidates[candIdx];
            if (!passesFusionSplitPriorSignalGate(candidate)) {
                ++splitPriorRejectedSignal;
                continue;
            }
            ++splitPriorCandidatesPassedGate;
            bool assignedToAnyParent = false;
            for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
                const auto &cell = frame.cells[ci];
                const float dx = cell.getX() - candidate.centerScaled.x;
                const float dy = cell.getY() - candidate.centerScaled.y;
                const float dz = cell.getZ() - candidate.centerScaled.z;
                const float distSq = dx * dx + dy * dy + dz * dz;

                const Ellipsoid &parent = frame.cells[ci];
                const float parentMaxR = std::max({parent.getARadius(),
                                                   parent.getBRadius(),
                                                   parent.getCRadius()});
                const float parentMinR = std::max(
                    1e-5f,
                    std::min({parent.getARadius(),
                              parent.getBRadius(),
                              parent.getCRadius()}));
                const float parentShapeElongation = parentMaxR / parentMinR;
                const float absoluteCatch = std::max(0.0f, lumenConfig.fusionSplitPriorMaxParentDistance);
                const float scaledCatch = std::max(0.0f, parentMaxR *
                                                         lumenConfig.fusionSplitPriorParentRadiusScale);
                const float catchRadius = (absoluteCatch > 0.0f && scaledCatch > 0.0f)
                                              ? std::min(absoluteCatch, scaledCatch)
                                              : std::max(absoluteCatch, scaledCatch);
                const bool insidePrimaryCatch =
                    catchRadius > 0.0f && distSq <= catchRadius * catchRadius;
                const bool temporalRepairCatchEnabled =
                    windowEnabled &&
                    lumenConfig.fusionSplitPriorTemporalRepairEnabled &&
                    (lumenConfig.fusionSplitPriorTemporalRepairMaxParentDistance > 0.0f ||
                     lumenConfig.fusionSplitPriorTemporalRepairParentRadiusScale > 0.0f);
                const float temporalAbsoluteCatch = std::max(
                    0.0f,
                    lumenConfig.fusionSplitPriorTemporalRepairMaxParentDistance);
                const float temporalScaledCatch = std::max(
                    0.0f,
                    parentMaxR *
                        lumenConfig.fusionSplitPriorTemporalRepairParentRadiusScale);
                const float temporalCatchRadius =
                    temporalRepairCatchEnabled
                        ? std::max(temporalAbsoluteCatch, temporalScaledCatch)
                        : 0.0f;
                const bool insideTemporalRepairCatch =
                    temporalCatchRadius > catchRadius &&
                    distSq <= temporalCatchRadius * temporalCatchRadius;
                const bool earlyLargeAssignFrame =
                    lumenConfig
                        .fusionSplitPriorEarlyLargeSeparationAssignBeyondCatchEnabled &&
                    lumenConfig
                        .fusionSplitPriorEarlyLargeSeparationRescueEnabled &&
                    (!lumenConfig
                          .fusionSplitPriorEarlyLargeSeparationRescueFirstFrameOnly ||
                     frameIndex == 0) &&
                    (lumenConfig
                             .fusionSplitPriorEarlyLargeSeparationRescueMaxFrame < 0 ||
                     absoluteFrame <=
                         lumenConfig
                             .fusionSplitPriorEarlyLargeSeparationRescueMaxFrame);
                const float parentDistance = std::sqrt(distSq);
                const bool insideEarlyLargeSeparationAssignCatch =
                    earlyLargeAssignFrame &&
                    !insidePrimaryCatch &&
                    parentShapeElongation >=
                        std::max(
                            1.0f,
                            lumenConfig
                                .fusionSplitPriorEarlyLargeSeparationMinParentShape) &&
                    parentDistance <=
                        std::max(
                            catchRadius,
                            lumenConfig
                                .fusionSplitPriorEarlyLargeSeparationAssignMaxParentDistance) &&
                    candidate.voxelCount >=
                        std::max(
                            0,
                            lumenConfig
                                .fusionSplitPriorEarlyLargeSeparationAssignMinVoxels) &&
                    candidate.top10MinusShell >=
                        std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorEarlyLargeSeparationAssignMinSignal);
                if (!insidePrimaryCatch && !insideTemporalRepairCatch &&
                    !insideEarlyLargeSeparationAssignCatch) {
                    continue;
                }
                if (lumenConfig.fusionSplitPriorContinuationClaimGuardEnabled) {
                    const int candidateId = static_cast<int>(candIdx);
                    const auto ownerIt = centerCandidateOwnerById.find(candidateId);
                    if (ownerIt != centerCandidateOwnerById.end() &&
                        ownerIt->second != cell.getName()) {
                        const auto ownerDistanceIt =
                            centerCandidateOwnerDistanceById.find(candidateId);
                        const float ownerDistance =
                            ownerDistanceIt != centerCandidateOwnerDistanceById.end()
                                ? ownerDistanceIt->second
                                : std::numeric_limits<float>::infinity();
                        const float tieMargin = std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorContinuationClaimTieMargin);
                        if (std::isfinite(ownerDistance) &&
                            ownerDistance + tieMargin < parentDistance) {
                            const bool elongatedParentCanCarrySoftConflict =
                                parentShapeElongation >=
                                    std::max(
                                        0.0f,
                                        lumenConfig
                                            .fusionSplitPriorElongatedParentMinShape) &&
                                parentDistance <= catchRadius;
                            // A strong early daughter can sit just outside the
                            // ordinary continuation catch radius. Keep it in
                            // the split graph so the later pair scorer can use
                            // separation, window support, and cost gates to
                            // decide instead of losing it at assignment time.
                            if (!elongatedParentCanCarrySoftConflict &&
                                !insideEarlyLargeSeparationAssignCatch) {
                                ++splitPriorRejectedContinuationOwner;
                                continue;
                            }
                        }
                    }
                }

                candidatesByParent[ci].push_back({
                    static_cast<int>(candIdx),
                    candidate.centerScaled,
                    candidate.voxelCount,
                    candidate.top10MinusShell,
                    candidate.majorRadius,
                    candidate.bRadius,
                    candidate.minorRadius,
                    !insidePrimaryCatch && insideTemporalRepairCatch &&
                        !insideEarlyLargeSeparationAssignCatch});
                if (insideEarlyLargeSeparationAssignCatch) {
                    ++splitPriorEarlyLargeCatchAssignments;
                } else if (!insidePrimaryCatch && insideTemporalRepairCatch) {
                    ++splitPriorTemporalCatchAssignments;
                }
                ++splitPriorCandidateAssignments;
                assignedToAnyParent = true;
            }
            if (!assignedToAnyParent) {
                ++splitPriorRejectedParentDistance;
            }
        }

        int splitPriorParentsWithTwoCandidates = 0;
        for (const auto &entry : candidatesByParent) {
            const size_t parentIdx = entry.first;
            if (parentIdx >= frame.cells.size()) {
                continue;
            }
            std::vector<FusionCandidateRef> list = entry.second;
            const Ellipsoid &parent = frame.cells[parentIdx];
            const cv::Point3f parentCenter(parent.getX(), parent.getY(), parent.getZ());
            const float parentMaxR = std::max({parent.getARadius(),
                                               parent.getBRadius(),
                                               parent.getCRadius()});
            const float parentMinR = std::max(
                1e-5f,
                std::min({parent.getARadius(), parent.getBRadius(), parent.getCRadius()}));
            const float parentMidR = std::max(
                1e-5f,
                parent.getARadius() + parent.getBRadius() + parent.getCRadius() -
                    parentMaxR - parentMinR);
            const float parentShapeElongation = parentMaxR / parentMinR;
            const float parentLongMidRatio = parentMaxR / parentMidR;
            const float parentMidShortRatio = parentMidR / parentMinR;
            const float absoluteCatch = std::max(0.0f, lumenConfig.fusionSplitPriorMaxParentDistance);
            const float scaledCatch = std::max(0.0f, parentMaxR *
                                                         lumenConfig.fusionSplitPriorParentRadiusScale);
            const float catchRadius = (absoluteCatch > 0.0f && scaledCatch > 0.0f)
                                          ? std::min(absoluteCatch, scaledCatch)
                                          : std::max(absoluteCatch, scaledCatch);
            const int parentAnchorCandidateId =
                2000000000 - static_cast<int>(std::min<size_t>(parentIdx, 1000000));
            const float parentAnchorMinShape = std::max(
                1.0f,
                lumenConfig.fusionSplitPriorParentAnchorMinShape);

            if (windowEnabled &&
                list.size() < 2 &&
                catchRadius > 0.0f &&
                parentShapeElongation >=
                    std::max(2.0f, lumenConfig.fusionSplitPriorElongatedParentMinShape)) {
                struct LookaheadRef {
                    FusionCandidateRef ref;
                    float distance = 0.0f;
                };
                std::vector<LookaheadRef> lookaheadRefs;
                for (const auto &windowEntry : windowCandidatesByOffset) {
                    const int offset = windowEntry.first;
                    const auto *futureCandidates = windowEntry.second;
                    if (futureCandidates == nullptr || futureCandidates->empty()) {
                        continue;
                    }
                    const float futureCatchRadius =
                        catchRadius +
                        std::max(0.0f,
                                 lumenConfig.fusionSplitPriorWindowMatchDistancePerFrame) *
                            static_cast<float>(std::max(0, offset - 1));
                    for (const auto &future : *futureCandidates) {
                        const float distance =
                            static_cast<float>(cv::norm(future.position - parentCenter));
                        if (distance > futureCatchRadius) {
                            continue;
                        }
                        bool duplicatesCurrentCandidate = false;
                        for (const auto &current : list) {
                            if (cv::norm(current.center - future.position) < 2.0f) {
                                duplicatesCurrentCandidate = true;
                                break;
                            }
                        }
                        if (duplicatesCurrentCandidate) {
                            continue;
                        }
                        FusionCandidateRef ref;
                        ref.candidateId =
                            1000000 + offset * 100000 + std::max(0, future.candidateId);
                        ref.center = future.position;
                        ref.voxelCount = future.voxelCount;
                        ref.signal = future.signal;
                        lookaheadRefs.push_back({ref, distance});
                    }
                }
                std::sort(lookaheadRefs.begin(), lookaheadRefs.end(),
                          [](const LookaheadRef &a, const LookaheadRef &b) {
                              if (std::abs(a.distance - b.distance) > 1e-4f) {
                                  return a.distance < b.distance;
                              }
                              if (std::abs(a.ref.signal - b.ref.signal) > 1e-4f) {
                                  return a.ref.signal > b.ref.signal;
                              }
                              return a.ref.candidateId < b.ref.candidateId;
                          });
                const size_t maxLookaheadRefs = std::min<size_t>(3, lookaheadRefs.size());
                for (size_t idx = 0; idx < maxLookaheadRefs; ++idx) {
                    list.push_back(lookaheadRefs[idx].ref);
                }
                if (maxLookaheadRefs > 0) {
                    std::cout << "[CellLumen Fusion Lookahead Inject] frame="
                              << absoluteFrame
                              << " parent=" << parent.getName()
                              << " added=" << maxLookaheadRefs
                              << " parentShapeElong=" << parentShapeElongation
                              << " catchRadius=" << catchRadius
                              << std::endl;
                }
            }

            if (windowEnabled &&
                catchRadius > 0.0f &&
                !list.empty() &&
                parentShapeElongation >= parentAnchorMinShape) {
                bool hasParentCenterCandidate = false;
                for (const auto &current : list) {
                    if (cv::norm(current.center - parentCenter) < 2.0f) {
                        hasParentCenterCandidate = true;
                        break;
                    }
                }
                if (!hasParentCenterCandidate) {
                    FusionCandidateRef parentAnchor;
                    parentAnchor.candidateId = parentAnchorCandidateId;
                    parentAnchor.center = parentCenter;
                    parentAnchor.voxelCount = 0;
                    parentAnchor.signal = 0.0f;
                    list.push_back(parentAnchor);
                    std::cout << "[CellLumen Fusion Parent Anchor] frame="
                              << absoluteFrame
                              << " parent=" << parent.getName()
                              << " parentShapeElong=" << parentShapeElongation
                              << " candidates=" << (list.size() - 1)
                              << std::endl;
                }
            }

            if (list.size() < 2) {
                continue;
            }
            ++splitPriorParentsWithTwoCandidates;
            const float minSep = std::max(
                lumenConfig.fusionSplitPriorMinSeparation,
                parentMaxR * lumenConfig.fusionSplitPriorMinSeparationRadiusScale);
            const float maxSepByScale = parentMaxR * lumenConfig.fusionSplitPriorMaxSeparationRadiusScale;
            const float maxSep = lumenConfig.fusionSplitPriorMaxSeparation > 0.0f
                                     ? std::min(lumenConfig.fusionSplitPriorMaxSeparation, maxSepByScale)
                                     : maxSepByScale;
            const bool temporalRepairEnabled =
                lumenConfig.fusionSplitPriorTemporalRepairEnabled && windowEnabled;
            const float temporalMaxSepByScale =
                parentMaxR *
                std::max(
                    0.0f,
                    lumenConfig
                        .fusionSplitPriorTemporalRepairMaxCandidateSeparationRadiusScale);
            const float temporalMaxSepByAbsolute =
                std::max(
                    0.0f,
                    lumenConfig
                        .fusionSplitPriorTemporalRepairMaxCandidateSeparation);
            const float temporalCandidateMaxSep =
                temporalRepairEnabled
                    ? ((temporalMaxSepByAbsolute > 0.0f && temporalMaxSepByScale > 0.0f)
                           ? std::max(temporalMaxSepByAbsolute, temporalMaxSepByScale)
                           : std::max(temporalMaxSepByAbsolute, temporalMaxSepByScale))
                    : maxSep;
            const float targetSep = std::max(minSep, parentMaxR * 1.55f);
            const bool rankingSoftGateEnabled =
                lumenConfig.fusionSplitPriorRankingSoftGateEnabled;
            struct NeighborClaim {
                size_t index = std::numeric_limits<size_t>::max();
                float distance = std::numeric_limits<float>::infinity();
                float maxRadius = 0.0f;
            };
            auto nearestOtherCellClaim = [&](const cv::Point3f &point) {
                NeighborClaim best;
                for (size_t otherIdx = 0; otherIdx < frame.cells.size(); ++otherIdx) {
                    if (otherIdx == parentIdx) {
                        continue;
                    }
                    const auto &other = frame.cells[otherIdx];
                    const cv::Point3f otherCenter(other.getX(), other.getY(), other.getZ());
                    const float distance = static_cast<float>(cv::norm(point - otherCenter));
                    if (distance < best.distance) {
                        best.index = otherIdx;
                        best.distance = distance;
                        best.maxRadius = std::max({other.getARadius(),
                                                   other.getBRadius(),
                                                   other.getCRadius()});
                    }
                }
                return best;
            };
            auto appendUniqueBlockerName = [&](std::vector<size_t> &blockers,
                                               std::string &names,
                                               size_t blockerIdx) {
                if (blockerIdx >= frame.cells.size()) {
                    return;
                }
                if (std::find(blockers.begin(), blockers.end(), blockerIdx) != blockers.end()) {
                    return;
                }
                blockers.push_back(blockerIdx);
                if (!names.empty()) {
                    names += '|';
                }
                names += frame.cells[blockerIdx].getName();
            };
            auto areImmediateLineageSiblings = [](const std::string &a,
                                                  const std::string &b) {
                if (a.size() != b.size() || a.size() < 2 || a == b) {
                    return false;
                }
                const char aLast = a.back();
                const char bLast = b.back();
                if ((aLast != '0' && aLast != '1') ||
                    (bLast != '0' && bLast != '1') ||
                    aLast == bLast) {
                    return false;
                }
                return a.compare(0, a.size() - 1, b, 0, b.size() - 1) == 0;
            };

            for (size_t i = 0; i < list.size(); ++i) {
                for (size_t j = i + 1; j < list.size(); ++j) {
                    double rankingSoftPenalty = 0.0;
                    const bool parentAnchoredPair =
                        list[i].candidateId == parentAnchorCandidateId ||
                        list[j].candidateId == parentAnchorCandidateId;
                    const float sep =
                        static_cast<float>(cv::norm(list[i].center - list[j].center));
                    // Center-prior cluster collapse means these current-frame
                    // Cell Lumen peaks are being treated as one parent center.
                    // Reusing the exact same peaks as a split pair in the same
                    // parent creates a contradictory graph and caused f078 to
                    // produce an extra daughter after the continuation evidence
                    // had already collapsed the pair. A wide, future-supported
                    // pair inside an elongated parent is allowed as a local
                    // exception so true f085-like splits are not missed.
                    const auto collapsedIt =
                        collapsedCenterClusterMembersByParent.find(parentIdx);
                    const bool collapsedCenterClusterPairRaw =
                        lumenConfig
                            .fusionSplitPriorRejectCollapsedCenterClusterPairs &&
                        !parentAnchoredPair &&
                        collapsedIt !=
                            collapsedCenterClusterMembersByParent.end() &&
                        collapsedIt->second.count(list[i].candidateId) > 0 &&
                        collapsedIt->second.count(list[j].candidateId) > 0;
                    WindowSupportResult collapsedCenterPairWindowSupport;
                    if (collapsedCenterClusterPairRaw &&
                        (lumenConfig
                             .fusionSplitPriorCollapsedCenterPairRescueEnabled ||
                         lumenConfig
                             .fusionSplitPriorCollapsedCenterPairSoftRescueEnabled)) {
                        collapsedCenterPairWindowSupport =
                            computeWindowSupport(parentCenter,
                                                 list[i].center,
                                                 list[j].center);
                    }
                    const int collapsedPairMinWindowBoth =
                        std::max(
                            0,
                            lumenConfig
                                .fusionSplitPriorCollapsedCenterPairRescueMinWindowBoth);
                    const bool collapsedCenterClusterPairRescued =
                        collapsedCenterClusterPairRaw &&
                        lumenConfig
                            .fusionSplitPriorCollapsedCenterPairRescueEnabled &&
                        sep >=
                            lumenConfig
                                .fusionSplitPriorCollapsedCenterPairRescueMinSeparation &&
                        parentShapeElongation >=
                            lumenConfig
                            .fusionSplitPriorCollapsedCenterPairRescueMinParentShape &&
                        collapsedCenterPairWindowSupport.bothDaughtersSupported >=
                            collapsedPairMinWindowBoth;
                    const int softRescueMaxCells = std::max(
                        0,
                        lumenConfig
                            .fusionSplitPriorCollapsedCenterPairSoftRescueMaxCells);
                    const float softRescueMinSeparation =
                        lumenConfig
                            .fusionSplitPriorCollapsedCenterPairRescueMinSeparation *
                        std::clamp(
                            lumenConfig
                                .fusionSplitPriorCollapsedCenterPairSoftRescueMinSeparationFraction,
                            0.0f,
                            1.0f);
                    const int softRescueMinVoxels = std::max(
                        0,
                        lumenConfig
                            .fusionSplitPriorCollapsedCenterPairSoftRescueMinVoxels);
                    const float softRescueMinSignal = std::max(
                        0.0f,
                        lumenConfig
                            .fusionSplitPriorCollapsedCenterPairSoftRescueMinSignal);
                    const bool softRescueSignalOk =
                        list[i].voxelCount >= softRescueMinVoxels &&
                        list[j].voxelCount >= softRescueMinVoxels &&
                        list[i].signal >= softRescueMinSignal &&
                        list[j].signal >= softRescueMinSignal;
                    const bool softRescueSupportOk =
                        collapsedCenterPairWindowSupport.bothDaughtersSupported >=
                            collapsedPairMinWindowBoth ||
                        sep >= lumenConfig
                                   .fusionSplitPriorCollapsedCenterPairRescueMinSeparation;
                    const bool collapsedCenterClusterPairSoftRescued =
                        collapsedCenterClusterPairRaw &&
                        !collapsedCenterClusterPairRescued &&
                        lumenConfig
                            .fusionSplitPriorCollapsedCenterPairSoftRescueEnabled &&
                        splitPriorLiveCells <= softRescueMaxCells &&
                        sep >= softRescueMinSeparation &&
                        softRescueSignalOk &&
                        softRescueSupportOk;
                    double collapsedCenterSoftPenalty = 0.0;
                    if (collapsedCenterClusterPairSoftRescued) {
                        const float sepShortfall = std::max(
                            0.0f,
                            lumenConfig
                                    .fusionSplitPriorCollapsedCenterPairRescueMinSeparation -
                                sep);
                        const float shapeShortfall = std::max(
                            0.0f,
                            lumenConfig
                                    .fusionSplitPriorCollapsedCenterPairRescueMinParentShape -
                                parentShapeElongation);
                        collapsedCenterSoftPenalty =
                            static_cast<double>(sepShortfall) *
                                static_cast<double>(std::max(
                                    0.0f,
                                    lumenConfig
                                        .fusionSplitPriorCollapsedCenterPairSoftRescueSeparationPenaltyWeight)) +
                            static_cast<double>(shapeShortfall) *
                                static_cast<double>(std::max(
                                    0.0f,
                                    lumenConfig
                                        .fusionSplitPriorCollapsedCenterPairSoftRescueShapePenaltyWeight));
                        rankingSoftPenalty += collapsedCenterSoftPenalty;
                    }
                    const bool collapsedCenterClusterPair =
                        collapsedCenterClusterPairRaw &&
                        !collapsedCenterClusterPairRescued &&
                        !collapsedCenterClusterPairSoftRescued;
                    if (collapsedCenterClusterPairRescued) {
                        std::cout
                            << "[CellLumen Fusion SplitPrior Allow CollapsedCenterClusterPair] frame="
                            << absoluteFrame
                            << " parent=" << parent.getName()
                            << " candidateIds=(" << list[i].candidateId
                            << "," << list[j].candidateId << ")"
                            << " sep=" << sep
                            << " minSep="
                            << lumenConfig
                                   .fusionSplitPriorCollapsedCenterPairRescueMinSeparation
                            << " parentShapeElong=" << parentShapeElongation
                            << " minParentShape="
                            << lumenConfig
                                   .fusionSplitPriorCollapsedCenterPairRescueMinParentShape
                            << " windowBoth="
                            << collapsedCenterPairWindowSupport
                                   .bothDaughtersSupported
                            << " minWindowBoth=" << collapsedPairMinWindowBoth
                            << " reason=wide_future_supported_pair_in_elongated_parent"
                            << std::endl;
                    }
                    if (collapsedCenterClusterPairSoftRescued) {
                        ++splitPriorSoftAllowedCollapsedCenterCluster;
                        std::cout
                            << "[CellLumen Fusion SplitPrior SoftAllow CollapsedCenterClusterPair] frame="
                            << absoluteFrame
                            << " parent=" << parent.getName()
                            << " candidateIds=(" << list[i].candidateId
                            << "," << list[j].candidateId << ")"
                            << " sep=" << sep
                            << " softMinSep=" << softRescueMinSeparation
                            << " hardMinSep="
                            << lumenConfig
                                   .fusionSplitPriorCollapsedCenterPairRescueMinSeparation
                            << " parentShapeElong=" << parentShapeElongation
                            << " hardMinParentShape="
                            << lumenConfig
                                   .fusionSplitPriorCollapsedCenterPairRescueMinParentShape
                            << " windowBoth="
                            << collapsedCenterPairWindowSupport
                                   .bothDaughtersSupported
                            << " liveCells=" << splitPriorLiveCells
                            << " maxCells=" << softRescueMaxCells
                            << " vox=(" << list[i].voxelCount
                            << "," << list[j].voxelCount << ")"
                            << " signal=(" << list[i].signal
                            << "," << list[j].signal << ")"
                            << " softPenalty=" << collapsedCenterSoftPenalty
                            << " reason=sparse_strong_collapsed_pair_kept_as_soft_candidate"
                            << std::endl;
                    }
                    if (collapsedCenterClusterPair) {
                        ++splitPriorRejectedCollapsedCenterCluster;
                        std::cout
                            << "[CellLumen Fusion SplitPrior Reject CollapsedCenterClusterPair] frame="
                            << absoluteFrame
                            << " parent=" << parent.getName()
                            << " candidateIds=(" << list[i].candidateId
                            << "," << list[j].candidateId << ")"
                            << " sep=" << sep
                            << " rescueMinSep="
                            << lumenConfig
                                   .fusionSplitPriorCollapsedCenterPairRescueMinSeparation
                            << " parentShapeElong=" << parentShapeElongation
                            << " rescueMinParentShape="
                            << lumenConfig
                                   .fusionSplitPriorCollapsedCenterPairRescueMinParentShape
                            << " windowBoth="
                            << collapsedCenterPairWindowSupport
                                   .bothDaughtersSupported
                            << " rescueMinWindowBoth=" << collapsedPairMinWindowBoth
                            << " reason=center_prior_already_collapsed_internal_peaks"
                            << std::endl;
                        continue;
                    }
                    if (sep < minSep || sep > maxSep) {
                        const float softMinSep = minSep * 0.70f;
                        const float softMaxSep = temporalRepairEnabled
                                                     ? std::max(maxSep * 1.30f,
                                                                temporalCandidateMaxSep)
                                                     : maxSep * 1.30f;
                        const bool hardSeparationReject =
                            (!rankingSoftGateEnabled && !temporalRepairEnabled) ||
                            sep < softMinSep ||
                            sep > softMaxSep;
                        if (hardSeparationReject) {
                            ++splitPriorRejectedSeparation;
                            continue;
                        }
                        const float excess =
                            sep < minSep ? (minSep - sep) : (sep - maxSep);
                        rankingSoftPenalty +=
                            static_cast<double>(excess) *
                            static_cast<double>(
                                std::max(0.0f,
                                         lumenConfig
                                             .fusionSplitPriorRankingSoftSeparationPenalty));
                    }
                    if (lumenConfig.fusionSplitPriorRoundParentMinSeparationRadiusScale > 0.0f &&
                        parentShapeElongation <= lumenConfig.fusionSplitPriorRoundParentMaxShape) {
                        const float roundParentMinSep =
                            parentMaxR *
                            lumenConfig.fusionSplitPriorRoundParentMinSeparationRadiusScale;
                        if (sep < roundParentMinSep) {
                            const bool hardRoundSeparationReject =
                                !rankingSoftGateEnabled ||
                                sep < roundParentMinSep * 0.80f;
                            if (hardRoundSeparationReject) {
                                ++splitPriorRejectedSeparation;
                                continue;
                            }
                            rankingSoftPenalty +=
                                static_cast<double>(roundParentMinSep - sep) *
                                static_cast<double>(
                                    std::max(0.0f,
                                             lumenConfig
                                                 .fusionSplitPriorRankingSoftSeparationPenalty));
                        }
                    }
                    const cv::Point3f midpoint = 0.5f * (list[i].center + list[j].center);
                    const float midpointDist = static_cast<float>(cv::norm(midpoint - parentCenter));
                    const float maxMidpointDist = std::max(
                        lumenConfig.fusionSplitPriorMaxMidpointDistance,
                        parentMaxR * lumenConfig.fusionSplitPriorMaxMidpointRadiusScale);
                    if (maxMidpointDist > 0.0f && midpointDist > maxMidpointDist) {
                        const bool hardMidpointReject =
                            !rankingSoftGateEnabled ||
                            midpointDist > maxMidpointDist * 1.60f;
                        if (hardMidpointReject) {
                            ++splitPriorRejectedMidpoint;
                            continue;
                        }
                        rankingSoftPenalty +=
                            static_cast<double>(midpointDist - maxMidpointDist) *
                            static_cast<double>(
                                std::max(0.0f,
                                         lumenConfig
                                             .fusionSplitPriorRankingSoftMidpointPenalty));
                    }
                    const float parentDistA = static_cast<float>(cv::norm(list[i].center - parentCenter));
                    const float parentDistB = static_cast<float>(cv::norm(list[j].center - parentCenter));
                    const float nearParentDist = std::min(parentDistA, parentDistB);
                    const float farParentDist = std::max(parentDistA, parentDistB);
                    const float parentDistanceBalance =
                        parentAnchoredPair
                            ? 1.0f
                            : (farParentDist > 1e-5f ? nearParentDist / farParentDist : 1.0f);
                    const float minDaughterParentDist = std::max(
                        lumenConfig.fusionSplitPriorMinDaughterParentDistance,
                        parentMaxR * lumenConfig.fusionSplitPriorMinDaughterParentDistanceRadiusScale);
                    double parentPersistencePenalty = 0.0;
                    if (nearParentDist < minDaughterParentDist) {
                        parentPersistencePenalty +=
                            static_cast<double>(minDaughterParentDist - nearParentDist) *
                            static_cast<double>(lumenConfig.fusionSplitPriorParentPersistencePenalty);
                    }
                    if (parentDistanceBalance < lumenConfig.fusionSplitPriorMinParentDistanceBalance) {
                        parentPersistencePenalty +=
                            static_cast<double>(lumenConfig.fusionSplitPriorMinParentDistanceBalance -
                                                parentDistanceBalance) *
                            static_cast<double>(lumenConfig.fusionSplitPriorParentPersistencePenalty);
                    }
                    if (parentAnchoredPair) {
                        parentPersistencePenalty = 0.0;
                    }
                    double neighborClaimPenalty = 0.0;
                    const NeighborClaim nearestOtherA = nearestOtherCellClaim(list[i].center);
                    const NeighborClaim nearestOtherB = nearestOtherCellClaim(list[j].center);
                    const float neighborMargin = std::max(0.0f, lumenConfig.fusionSplitPriorNeighborClaimMargin);
                    if (std::isfinite(nearestOtherA.distance) &&
                        nearestOtherA.distance + neighborMargin < parentDistA) {
                        neighborClaimPenalty +=
                            static_cast<double>(parentDistA - nearestOtherA.distance - neighborMargin) *
                            static_cast<double>(lumenConfig.fusionSplitPriorNeighborClaimPenalty);
                    }
                    if (std::isfinite(nearestOtherB.distance) &&
                        nearestOtherB.distance + neighborMargin < parentDistB) {
                        neighborClaimPenalty +=
                            static_cast<double>(parentDistB - nearestOtherB.distance - neighborMargin) *
                            static_cast<double>(lumenConfig.fusionSplitPriorNeighborClaimPenalty);
                    }
                    std::vector<size_t> continuationClaimBlockers;
                    std::string continuationClaimBlockerNames;
                    bool siblingContinuationBlocked = false;
                    float closestContinuationClaimDistance =
                        std::numeric_limits<float>::infinity();
                    if (lumenConfig.fusionSplitPriorContinuationClaimGuardEnabled) {
                        const float claimRadiusScale = std::max(
                            0.0f,
                            lumenConfig.fusionSplitPriorContinuationClaimRadiusScale);
                        const float tieMargin = std::max(
                            0.0f,
                            lumenConfig.fusionSplitPriorContinuationClaimTieMargin);
                        auto maybeAddClaimBlocker =
                            [&](const NeighborClaim &claim, float parentDist) {
                                if (claim.index >= frame.cells.size() ||
                                    !std::isfinite(claim.distance) ||
                                    claim.maxRadius <= 0.0f) {
                                    return;
                                }
                                const bool insideNeighborContinuation =
                                    claim.distance <= claim.maxRadius * claimRadiusScale;
                                const bool neighborIsCompetitive =
                                    claim.distance <= parentDist + tieMargin;
                                if (insideNeighborContinuation && neighborIsCompetitive) {
                                    appendUniqueBlockerName(continuationClaimBlockers,
                                                            continuationClaimBlockerNames,
                                                            claim.index);
                                    siblingContinuationBlocked =
                                        siblingContinuationBlocked ||
                                        areImmediateLineageSiblings(
                                            parent.getName(),
                                            frame.cells[claim.index].getName());
                                    closestContinuationClaimDistance =
                                        std::min(closestContinuationClaimDistance,
                                                 claim.distance);
	                                }
	                            };
                        maybeAddClaimBlocker(nearestOtherA, parentDistA);
                        maybeAddClaimBlocker(nearestOtherB, parentDistB);
                    }
                    const bool conflictReplacementEligible =
                        lumenConfig.fusionSplitPriorConflictReplacementEnabled &&
                        nearParentDist <= parentMaxR *
                                              lumenConfig.fusionSplitPriorConflictCloseParentRadiusScale &&
                        sep >= parentMaxR *
                                   lumenConfig.fusionSplitPriorConflictMinNewSeparationRadiusScale;
                    if (conflictReplacementEligible &&
                        lumenConfig.fusionSplitPriorSuppressConflictPenaltiesForAsymmetricClose) {
                        parentPersistencePenalty = 0.0;
                        neighborClaimPenalty = 0.0;
                    }
                    if (lumenConfig.fusionSplitPriorMaxNeighborClaimPenalty >= 0.0f &&
                        neighborClaimPenalty >
                            static_cast<double>(lumenConfig.fusionSplitPriorMaxNeighborClaimPenalty)) {
                        const double excess =
                            neighborClaimPenalty -
                            static_cast<double>(
                                lumenConfig.fusionSplitPriorMaxNeighborClaimPenalty);
                        const bool hardNeighborReject =
                            (!rankingSoftGateEnabled && !temporalRepairEnabled) ||
                            excess > 24.0;
                        if (hardNeighborReject) {
                            ++splitPriorRejectedNeighborClaim;
                            continue;
                        }
                        rankingSoftPenalty +=
                            excess *
                            static_cast<double>(
                                std::max(0.0f,
                                         lumenConfig
                                             .fusionSplitPriorRankingSoftNeighborPenalty));
                    }
                    const float sepPenalty = std::abs(sep - targetSep);
                    const float signalBonus =
                        lumenConfig.fusionSplitPriorSignalBonusWeight *
                        (list[i].signal + list[j].signal);
                    const WindowSupportResult windowSupport =
                        computeWindowSupport(parentCenter,
                                             list[i].center,
                                             list[j].center);
                    double continuationClaimSoftPenalty = 0.0;
	                    if (lumenConfig.fusionSplitPriorContinuationClaimGuardEnabled &&
	                        !continuationClaimBlockers.empty()) {
	                        const float closeParentRadius =
	                            parentMaxR *
	                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorContinuationClaimCloseParentRadiusScale);
                        if (closeParentRadius > 0.0f && nearParentDist < closeParentRadius) {
                            continuationClaimSoftPenalty +=
                                static_cast<double>(closeParentRadius - nearParentDist) *
	                                static_cast<double>(
	                                    std::max(
	                                        0.0f,
	                                        lumenConfig
	                                            .fusionSplitPriorContinuationClaimCloseParentPenalty));
	                        }
	                        const float tightContinuationRadius =
	                            parentMaxR *
	                            std::max(
	                                0.85f,
	                                lumenConfig
	                                    .fusionSplitPriorContinuationClaimCloseParentRadiusScale);
                        if (std::isfinite(closestContinuationClaimDistance) &&
                            tightContinuationRadius > 0.0f &&
                            closestContinuationClaimDistance < tightContinuationRadius) {
	                            continuationClaimSoftPenalty +=
	                                static_cast<double>(
	                                    tightContinuationRadius -
	                                    closestContinuationClaimDistance) *
	                                static_cast<double>(
	                                    std::max(
	                                        0.0f,
	                                        lumenConfig
                                            .fusionSplitPriorContinuationClaimCloseParentPenalty));
                        }
                        if (siblingContinuationBlocked) {
                            continuationClaimSoftPenalty += 100.0;
                        }
                    }
                    const bool cleanStrongWindowSupported =
                        windowSupport.bothDaughtersSupported >= 2 &&
                        windowSupport.missingDaughterCount == 0 &&
                        windowSupport.parentPersists == 0;
                    const bool tolerableContinuationClaim =
                        continuationClaimBlockers.empty() ||
                        (continuationClaimBlockers.size() == 1 &&
                         parentDistanceBalance >=
                             std::max(
                                 0.0f,
                                 lumenConfig
                                     .fusionSplitPriorMinParentDistanceBalance) &&
                         nearParentDist >=
                             std::max(
                                 6.0f,
                                 lumenConfig
                                     .fusionSplitPriorMinDaughterParentDistance));
                    double balancedWindowBonus = 0.0;
                    // Future support is not strong evidence if a daughter is
                    // also claimed by a neighboring cell's continuation.
                    const bool balancedWindowSupported =
                        cleanStrongWindowSupported &&
                        continuationClaimBlockers.empty() &&
                        midpointDist <=
                            std::max(parentMaxR,
                                     minSep * 1.25f) &&
                        parentDistanceBalance >=
                            lumenConfig
                                .fusionSplitPriorWindowBalancedMinParentDistanceBalance &&
                        nearParentDist >=
                            parentMaxR *
                                std::max(
                                    0.0f,
                                    lumenConfig
                                        .fusionSplitPriorWindowBalancedMinNearParentRadiusScale);
                    if (balancedWindowSupported) {
                        balancedWindowBonus =
                            static_cast<double>(
                                std::max(
                                    0.0f,
                                    lumenConfig
                                        .fusionSplitPriorWindowBalancedDaughterBonus));
                    }
                    double parentAnchorBonus = 0.0;
                    const bool parentAnchorWindowSupported =
                        parentAnchoredPair &&
                        cleanStrongWindowSupported &&
                        continuationClaimBlockers.empty() &&
                        rankingSoftPenalty <= 1e-5 &&
                        neighborClaimPenalty <= 1e-5 &&
                        farParentDist >= std::max(minDaughterParentDist, minSep * 1.75f) &&
                        parentShapeElongation >= parentAnchorMinShape;
                    if (parentAnchorWindowSupported) {
                        parentAnchorBonus = std::min(
                            8.0,
                            static_cast<double>(std::max(
                                0.0f,
                                lumenConfig.fusionSplitPriorWindowBalancedDaughterBonus)) *
                                0.30);
                    }
                    const int temporalMinWindowBoth =
                        std::max(
                            1,
                            lumenConfig
                                .fusionSplitPriorTemporalRepairMinWindowBoth);
                    const int temporalMaxWindowMissing =
                        std::max(
                            0,
                            lumenConfig
                                .fusionSplitPriorTemporalRepairMaxWindowMissing);
                    const int temporalMaxWindowParentPersists =
                        std::max(
                            0,
                            lumenConfig
                                .fusionSplitPriorTemporalRepairMaxWindowParentPersists);
                    const float temporalMinSeparation =
                        std::max(
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorTemporalRepairMinSeparation),
                            parentMaxR *
                                std::max(
                                    0.0f,
                                    lumenConfig
                                        .fusionSplitPriorTemporalRepairMinSeparationRadiusScale));
                    const float temporalMinParentBalance =
                        std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorTemporalRepairMinParentDistanceBalance);
                    const float temporalStrongMinParentBalance =
                        std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorTemporalRepairStrongMinParentDistanceBalance);
                    const double temporalMaxNeighborClaim =
                        static_cast<double>(
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorTemporalRepairMaxNeighborClaimPenalty));
                    const double temporalStrongMaxNeighborClaim =
                        static_cast<double>(
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorTemporalRepairStrongMaxNeighborClaimPenalty));
                    const bool currentFrameTwoRealPair =
                        list[i].candidateId >= 0 && list[i].candidateId < 1000000 &&
                        list[j].candidateId >= 0 && list[j].candidateId < 1000000;
                    const bool strongTemporalDaughterEvidence =
                        currentFrameTwoRealPair &&
                        list[i].voxelCount >=
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorTemporalRepairStrongMinVoxels) &&
                        list[j].voxelCount >=
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorTemporalRepairStrongMinVoxels) &&
                        list[i].signal >=
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorTemporalRepairStrongMinSignal) &&
                        list[j].signal >=
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorTemporalRepairStrongMinSignal);
                    const bool temporalBalanceOk =
                        parentDistanceBalance >= temporalMinParentBalance ||
                        (strongTemporalDaughterEvidence &&
                         parentDistanceBalance >= temporalStrongMinParentBalance);
                    const bool temporalNeighborClaimOk =
                        neighborClaimPenalty <= temporalMaxNeighborClaim ||
                        (strongTemporalDaughterEvidence &&
                         neighborClaimPenalty <= temporalStrongMaxNeighborClaim);
                    const bool temporalAsymmetricPair =
                        parentDistanceBalance <
                        std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorTemporalRepairStrongAsymmetryBalance);
                    const bool temporalAsymmetrySignalOk =
                        !temporalAsymmetricPair || strongTemporalDaughterEvidence;
                    const bool temporalRepairEligible =
                        temporalRepairEnabled &&
                        !parentAnchoredPair &&
                        windowSupport.bothDaughtersSupported >= temporalMinWindowBoth &&
                        windowSupport.missingDaughterCount <= temporalMaxWindowMissing &&
                        windowSupport.parentPersists <= temporalMaxWindowParentPersists &&
                        sep >= temporalMinSeparation &&
                        temporalBalanceOk &&
                        temporalAsymmetrySignalOk &&
                        nearParentDist >=
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorTemporalRepairMinNearParentDistance) &&
                        temporalNeighborClaimOk &&
                        continuationClaimSoftPenalty <=
                            static_cast<double>(
                                std::max(
                                    0.0f,
                                    lumenConfig
                                        .fusionSplitPriorTemporalRepairMaxContinuationClaimSoftPenalty)) &&
                        (!lumenConfig
                              .fusionSplitPriorTemporalRepairRejectClaimBlockers ||
                         continuationClaimBlockers.empty());
                    const bool temporalRepairCatchPair =
                        list[i].assignedByTemporalRepairCatch ||
                        list[j].assignedByTemporalRepairCatch;
                    const FusionCandidateRef *parentAnchorRealRef = nullptr;
                    if (parentAnchoredPair) {
                        parentAnchorRealRef =
                            list[i].candidateId == parentAnchorCandidateId
                                ? &list[j]
                                : &list[i];
                    }
                    // A sparse early daughter may be far enough from the old
                    // parent center that it enters only through the temporal
                    // catch radius. Before this guard, pairing that current
                    // Cell Lumen center with the parent anchor was discarded
                    // because temporal repair required two real candidates.
                    // Keep this narrow and YAML-gated: it only preserves a
                    // current-frame lumen center when future support agrees and
                    // no neighboring continuation owns the candidate.
                    const bool temporalCatchParentAnchorAllowed =
                        parentAnchoredPair &&
                        temporalRepairCatchPair &&
                        parentAnchorRealRef != nullptr &&
                        parentAnchorRealRef->candidateId >= 0 &&
                        parentAnchorRealRef->candidateId < 1000000 &&
                        lumenConfig.fusionSplitPriorPartialParentAnchorRescueEnabled &&
                        parentShapeElongation >=
                            std::max(
                                1.0f,
                                lumenConfig
                                    .fusionSplitPriorPartialParentAnchorMinShape) &&
                        parentAnchorRealRef->voxelCount >=
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorPartialParentAnchorMinRealVoxels) &&
                        parentAnchorRealRef->signal >=
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorPartialParentAnchorMinRealSignal) &&
                        windowSupport.bothDaughtersSupported >=
                            std::max(
                                1,
                                lumenConfig
                                    .fusionSplitPriorPartialParentAnchorMinWindowBoth) &&
                        windowSupport.missingDaughterCount <=
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorPartialParentAnchorMaxWindowMissing) &&
                        windowSupport.parentPersists <=
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorPartialParentAnchorMaxWindowParentPersists) &&
                        continuationClaimBlockers.empty() &&
                        farParentDist >=
                            std::max(
                                minDaughterParentDist,
                                std::max(
                                    0.0f,
                                    lumenConfig
                                        .fusionSplitPriorTemporalRepairMinSeparation));
                    if (temporalCatchParentAnchorAllowed) {
                        std::cout
                            << "[CellLumen Fusion TemporalCatch ParentAnchor Rescue] frame="
                            << absoluteFrame
                            << " parent=" << parent.getName()
                            << " candidateId="
                            << parentAnchorRealRef->candidateId
                            << " candidateDistance=" << farParentDist
                            << " parentShapeElong=" << parentShapeElongation
                            << " vox=" << parentAnchorRealRef->voxelCount
                            << " signal=" << parentAnchorRealRef->signal
                            << " windowBoth="
                            << windowSupport.bothDaughtersSupported
                            << " windowMissing="
                            << windowSupport.missingDaughterCount
                            << " windowParentPersists="
                            << windowSupport.parentPersists
                            << std::endl;
                    }
                    if (temporalRepairCatchPair && !temporalRepairEligible &&
                        !temporalCatchParentAnchorAllowed) {
                        ++splitPriorRejectedTemporalCatchPairs;
                        continue;
                    }
                    double rawScore =
                        static_cast<double>(lumenConfig.fusionSplitPriorMidpointWeight) *
                            static_cast<double>(midpointDist) +
                        static_cast<double>(lumenConfig.fusionSplitPriorSeparationPenaltyWeight) *
                            static_cast<double>(sepPenalty) -
                                         static_cast<double>(signalBonus) +
                                         parentPersistencePenalty +
                                         neighborClaimPenalty +
                                         rankingSoftPenalty +
                                         continuationClaimSoftPenalty +
                                         static_cast<double>(windowSupport.score) -
                                         balancedWindowBonus -
                                         parentAnchorBonus;
                    const bool elongatedRescueNeighborClaimOk =
                        lumenConfig.fusionSplitPriorElongatedParentMaxNeighborClaimPenalty < 0.0f ||
                        neighborClaimPenalty <=
                            static_cast<double>(
                                lumenConfig
                                    .fusionSplitPriorElongatedParentMaxNeighborClaimPenalty);
                    const bool elongatedParentRescued =
                        lumenConfig.fusionSplitPriorElongatedParentRescueEnabled &&
                        parentShapeElongation >= lumenConfig.fusionSplitPriorElongatedParentMinShape &&
                        (lumenConfig.fusionSplitPriorElongatedParentMaxScore <= 0.0f ||
                         rawScore <= static_cast<double>(lumenConfig.fusionSplitPriorElongatedParentMaxScore)) &&
                        elongatedRescueNeighborClaimOk;
                    if (lumenConfig.fusionSplitPriorMaxScore > 0.0f &&
                        rawScore > static_cast<double>(lumenConfig.fusionSplitPriorMaxScore) &&
                        !elongatedParentRescued) {
                        const double excess =
                            rawScore -
                            static_cast<double>(lumenConfig.fusionSplitPriorMaxScore);
                        const bool temporalRepairScoreSoftPass =
                            temporalRepairEligible &&
                            rawScore <=
                                static_cast<double>(
                                    std::max(
                                        0.0f,
                                        lumenConfig
                                            .fusionSplitPriorTemporalRepairMaxScore));
                        const bool hardScoreReject =
                            !temporalRepairScoreSoftPass &&
                            ((!rankingSoftGateEnabled && !temporalRepairEnabled) ||
                             rawScore >
                                 static_cast<double>(
                                     lumenConfig.fusionSplitPriorMaxScore) *
                                     2.0 +
                                 20.0);
                        if (hardScoreReject) {
                            ++splitPriorRejectedScore;
                            continue;
                        }
                        rankingSoftPenalty +=
                            excess *
                            static_cast<double>(
                                std::max(0.0f,
                                         lumenConfig
                                             .fusionSplitPriorRankingSoftScorePenalty));
                        rawScore +=
                            excess *
                            static_cast<double>(
                                std::max(0.0f,
                                         lumenConfig
                                             .fusionSplitPriorRankingSoftScorePenalty));
                    }
                    const double score =
                        rawScore -
                        (elongatedParentRescued
                             ? static_cast<double>(lumenConfig.fusionSplitPriorElongatedParentScoreBonus)
                             : 0.0);

                    BridgeSplitProposal proposal;
                    proposal.d1Pos = list[i].center;
                    proposal.d2Pos = list[j].center;
                    proposal.elongation = static_cast<float>(score);
                    proposal.parentShapeElongation = parentShapeElongation;
                    proposal.elongatedParentRescued = elongatedParentRescued;
                    proposal.candidateIdA = list[i].candidateId;
                    proposal.candidateIdB = list[j].candidateId;
                    proposal.windowBothDaughtersSupported =
                        windowSupport.bothDaughtersSupported;
                    proposal.windowMissingDaughterCount =
                        windowSupport.missingDaughterCount;
                    proposal.windowParentPersists =
                        windowSupport.parentPersists;
                    proposal.windowSupportScore = windowSupport.score;
                    proposal.balancedWindowBonus =
                        static_cast<float>(balancedWindowBonus);
                    proposal.parentDistanceBalance = parentDistanceBalance;
                    proposal.parentPersistencePenalty =
                        static_cast<float>(parentPersistencePenalty);
                    proposal.neighborClaimPenalty =
                        static_cast<float>(neighborClaimPenalty);
                    proposal.continuationClaimSoftPenalty =
                        static_cast<float>(continuationClaimSoftPenalty);
                    proposal.continuationClaimBlockerNames =
                        continuationClaimBlockerNames;
                    proposal.parentAnchored = parentAnchoredPair;
                    if (cleanStrongWindowSupported &&
                        tolerableContinuationClaim &&
                        lumenConfig
                                .fusionSplitPriorWindowHighConfidenceMaxOverlapCostFraction >=
                            0.0f) {
                        proposal.maxOverlapCostFractionOverride =
                            lumenConfig
                                .fusionSplitPriorWindowHighConfidenceMaxOverlapCostFraction;
                    }
                    proposal.leftPixelCount = list[i].voxelCount;
                    proposal.rightPixelCount = list[j].voxelCount;
                    rankedPriors.push_back({
                        parentIdx,
                        proposal,
                        score,
                        rawScore,
                        sep,
                        minSep,
                        maxSep,
                        midpointDist,
                        list[i].candidateId,
                        list[j].candidateId,
                        list[i].voxelCount,
                        list[j].voxelCount,
                        list[i].signal,
                        list[j].signal,
                        nearParentDist,
                        farParentDist,
                        parentDistanceBalance,
                        parentShapeElongation,
                        parentLongMidRatio,
                        parentMidShortRatio,
                        static_cast<float>(parentPersistencePenalty),
                        static_cast<float>(neighborClaimPenalty),
                        static_cast<float>(rankingSoftPenalty),
                        windowSupport.score,
                        windowSupport.bothDaughtersSupported,
                        windowSupport.missingDaughterCount,
                        windowSupport.parentPersists,
                        static_cast<float>(continuationClaimSoftPenalty),
                        static_cast<float>(balancedWindowBonus),
                        continuationClaimBlockers,
                        continuationClaimBlockerNames,
                        conflictReplacementEligible,
                        elongatedParentRescued,
                        parentAnchoredPair,
                        temporalRepairEligible,
                        temporalRepairCatchPair});
                }
            }
        }
        auto splitPriorAxisStats = [](const RankedSplitPrior &ranked) {
            const cv::Point3f delta =
                ranked.proposal.d2Pos - ranked.proposal.d1Pos;
            const float axisLen = static_cast<float>(cv::norm(delta));
            const float lateral =
                std::sqrt(delta.x * delta.x + delta.y * delta.y);
            const float zDominance =
                axisLen > 1e-3f ? std::abs(delta.z) / axisLen : 0.0f;
            return std::pair<float, float>(lateral, zDominance);
        };
        const float zDominantTieBreakMargin =
            std::max(0.0f,
                     lumenConfig.fusionSplitPriorZDominantTieBreakScoreMargin);
        const float zDominantTieBreakMinZ =
            std::clamp(
                lumenConfig.fusionSplitPriorZDominantTieBreakMinZDominance,
                0.0f,
                1.0f);
        const float zDominantTieBreakMaxLateralScale =
            std::max(
                0.0f,
                lumenConfig
                    .fusionSplitPriorZDominantTieBreakMaxLateralRadiusScale);
        auto isZDominantTieBreakCandidate =
            [&](const RankedSplitPrior &ranked) {
                if (zDominantTieBreakMargin <= 0.0f ||
                    ranked.parentIdx >= frame.cells.size()) {
                    return false;
                }
                const bool twoCurrentFrameLumenCenters =
                    ranked.candidateA >= 0 && ranked.candidateA < 1000000 &&
                    ranked.candidateB >= 0 && ranked.candidateB < 1000000;
                if (!twoCurrentFrameLumenCenters) {
                    return false;
                }
                const auto stats = splitPriorAxisStats(ranked);
                const Ellipsoid &parent = frame.cells[ranked.parentIdx];
                const float parentMaxR =
                    std::max({parent.getARadius(),
                              parent.getBRadius(),
                              parent.getCRadius()});
                const float lateralLimit =
                    zDominantTieBreakMaxLateralScale > 0.0f
                        ? parentMaxR * zDominantTieBreakMaxLateralScale
                        : std::numeric_limits<float>::infinity();
                return stats.second >= zDominantTieBreakMinZ &&
                       stats.first <= lateralLimit;
            };
        auto isLowShapeZDominantPair =
            [&](const RankedSplitPrior &ranked) {
                return lumenConfig.fusionSplitPriorRejectLowShapeZDominantPairs &&
                       !ranked.parentAnchored &&
                       isZDominantTieBreakCandidate(ranked) &&
                       ranked.parentShapeElongation <=
                           std::max(
                               1.0f,
                               lumenConfig
                                   .fusionSplitPriorLowShapeZDominantMaxParentShape);
            };
        auto isCleanWindowSeedColumnZStackPair =
            [&](const RankedSplitPrior &ranked) {
                if (!lumenConfig
                         .fusionSplitPriorRejectCleanWindowSeedColumnZStack ||
                    ranked.parentAnchored ||
                    ranked.candidateA < 0 ||
                    ranked.candidateA >= 1000000 ||
                    ranked.candidateB < 0 ||
                    ranked.candidateB >= 1000000 ||
                    ranked.windowBothDaughtersSupported < 2 ||
                    ranked.windowMissingDaughterCount != 0 ||
                    ranked.windowParentPersists != 0) {
                    return false;
                }
                if (lumenConfig
                        .fusionSplitPriorCleanWindowSeedColumnRequireBalancedBonus &&
                    ranked.balancedWindowBonus <= 0.0f) {
                    return false;
                }
                const auto stats = splitPriorAxisStats(ranked);
                const float maxLateral =
                    std::max(
                        0.0f,
                        lumenConfig
                            .fusionSplitPriorCleanWindowSeedColumnMaxLateralSeparation);
                const float minZDominance =
                    std::clamp(
                        lumenConfig
                            .fusionSplitPriorCleanWindowSeedColumnMinZDominance,
                        0.0f,
                        1.0f);
                const float minParentShape =
                    std::max(
                        1.0f,
                        lumenConfig
                            .fusionSplitPriorCleanWindowSeedColumnMinParentShape);
                const float maxParentShape =
                    std::max(
                        minParentShape,
                        lumenConfig
                            .fusionSplitPriorCleanWindowSeedColumnMaxParentShape);
                return stats.first <= maxLateral &&
                       stats.second >= minZDominance &&
                       ranked.parentShapeElongation >= minParentShape &&
                       ranked.parentShapeElongation <= maxParentShape;
            };
        auto preferRankedSplitPrior =
            [&](const RankedSplitPrior &a, const RankedSplitPrior &b) {
                if (a.parentIdx == b.parentIdx &&
                    zDominantTieBreakMargin > 0.0f &&
                    std::abs(a.score - b.score) <=
                        static_cast<double>(zDominantTieBreakMargin)) {
                    const bool aZDominant = isZDominantTieBreakCandidate(a);
                    const bool bZDominant = isZDominantTieBreakCandidate(b);
                    if (aZDominant != bZDominant) {
                        return !aZDominant;
                    }
                    if (aZDominant && bZDominant) {
                        const auto aStats = splitPriorAxisStats(a);
                        const auto bStats = splitPriorAxisStats(b);
                        if (std::abs(aStats.second - bStats.second) > 1e-4f) {
                            return aStats.second < bStats.second;
                        }
                        if (std::abs(aStats.first - bStats.first) > 1e-3f) {
                            return aStats.first > bStats.first;
                        }
                    }
                }
                if (std::abs(a.score - b.score) > 1e-9) {
                    return a.score < b.score;
                }
                if (a.parentIdx != b.parentIdx) {
                    return a.parentIdx < b.parentIdx;
                }
                if (a.candidateA != b.candidateA) {
                    return a.candidateA < b.candidateA;
                }
                return a.candidateB < b.candidateB;
            };
        std::sort(rankedPriors.begin(), rankedPriors.end(),
                  preferRankedSplitPrior);
        if (lumenConfig.fusionSplitPriorPrepassFallbackRejectBadLumenParent) {
            std::set<std::string> parentsSeenWithBestPair;
            for (const auto &ranked : rankedPriors) {
                if (ranked.parentIdx >= frame.cells.size()) {
                    continue;
                }
                const std::string parentName = frame.cells[ranked.parentIdx].getName();
                if (!parentsSeenWithBestPair.insert(parentName).second) {
                    continue;
                }
                const bool scoreTooHigh =
                    lumenConfig.fusionSplitPriorPrepassFallbackBadLumenMaxScore >= 0.0f &&
                    ranked.score >
                        static_cast<double>(
                            lumenConfig.fusionSplitPriorPrepassFallbackBadLumenMaxScore);
                const bool neighborClaimTooHigh =
                    lumenConfig.fusionSplitPriorPrepassFallbackBadLumenMaxNeighborClaimPenalty >= 0.0f &&
                    ranked.neighborClaimPenalty >
                        lumenConfig.fusionSplitPriorPrepassFallbackBadLumenMaxNeighborClaimPenalty;
                if (scoreTooHigh || neighborClaimTooHigh) {
                    badSplitPriorParentsForFrame.insert(parentName);
                }
            }
            if (!badSplitPriorParentsForFrame.empty()) {
                std::cout << "[CellLumen Fusion SplitPrior BadParent Summary] frame="
                          << absoluteFrame
                          << " parents=" << badSplitPriorParentsForFrame.size()
                          << " max_score="
                          << lumenConfig.fusionSplitPriorPrepassFallbackBadLumenMaxScore
                          << " max_neighbor_claim_penalty="
                          << lumenConfig
                                 .fusionSplitPriorPrepassFallbackBadLumenMaxNeighborClaimPenalty
                          << std::endl;
            }
        }
        const int debugTopN = std::max(0, lumenConfig.fusionSplitPriorDebugTopN);
        for (int debugIdx = 0;
             debugIdx < static_cast<int>(rankedPriors.size()) && debugIdx < debugTopN;
             ++debugIdx) {
            const auto &ranked = rankedPriors[static_cast<size_t>(debugIdx)];
            if (ranked.parentIdx >= frame.cells.size()) {
                continue;
            }
            const Ellipsoid &parent = frame.cells[ranked.parentIdx];
            std::cout << "[CellLumen Fusion SplitPrior Ranked] frame=" << absoluteFrame
                      << " rank=" << debugIdx
                      << " parent=" << parent.getName()
                      << " score=" << ranked.score
                      << " sep=" << ranked.sep
                      << " pairLateralSep="
                      << splitPriorAxisStats(ranked).first
                      << " pairZDominance="
                      << splitPriorAxisStats(ranked).second
                      << " midpointDist=" << ranked.midpointDist
                      << " parentDistNearFar=(" << ranked.nearParentDist
                      << "," << ranked.farParentDist << ")"
                      << " parentDistBalance=" << ranked.parentDistanceBalance
                      << " parentShapeElong=" << ranked.parentShapeElongation
                      << " parentPersistencePenalty=" << ranked.parentPersistencePenalty
                      << " neighborClaimPenalty=" << ranked.neighborClaimPenalty
                      << " rankingSoftPenalty=" << ranked.rankingSoftPenalty
                      << " windowScore=" << ranked.windowSupportScore
                      << " windowBoth=" << ranked.windowBothDaughtersSupported
                      << " windowMissing=" << ranked.windowMissingDaughterCount
                      << " windowParentPersists=" << ranked.windowParentPersists
                      << " continuationClaimSoftPenalty="
                      << ranked.continuationClaimSoftPenalty
                      << " balancedWindowBonus=" << ranked.balancedWindowBonus
                      << " continuationClaimBlockers="
                      << (ranked.continuationClaimBlockerNames.empty()
                              ? "none"
                              : ranked.continuationClaimBlockerNames)
                      << " elongatedParentRescued=" << ranked.elongatedParentRescued
                      << " parentAnchored=" << ranked.parentAnchored
                      << " temporalRepairEligible=" << ranked.temporalRepairEligible
                      << " candidateIds=(" << ranked.candidateA << "," << ranked.candidateB << ")"
                      << std::endl;
        }
        const int maxPriors =
            lumenConfig.fusionSplitPriorMaxPriorsPerFrame < 0
                ? std::numeric_limits<int>::max()
                : lumenConfig.fusionSplitPriorMaxPriorsPerFrame;
        auto logStoredSplitPrior = [&](const RankedSplitPrior &ranked,
                                       const std::string &parentName) {
            std::cout << "[CellLumen Fusion SplitPrior] frame=" << absoluteFrame
                      << " parent=" << parentName
                      << " d1=(" << ranked.proposal.d1Pos.x << "," << ranked.proposal.d1Pos.y
                      << "," << ranked.proposal.d1Pos.z << ")"
                      << " d2=(" << ranked.proposal.d2Pos.x << "," << ranked.proposal.d2Pos.y
                      << "," << ranked.proposal.d2Pos.z << ")"
                      << " sep=" << ranked.sep
                      << " pairLateralSep="
                      << splitPriorAxisStats(ranked).first
                      << " pairZDominance="
                      << splitPriorAxisStats(ranked).second
                      << " minSep=" << ranked.minSep
                      << " maxSep=" << ranked.maxSep
                      << " midpointDist=" << ranked.midpointDist
                      << " parentDistNearFar=(" << ranked.nearParentDist
                      << "," << ranked.farParentDist << ")"
                      << " parentDistBalance=" << ranked.parentDistanceBalance
                      << " parentShapeElong=" << ranked.parentShapeElongation
                      << " parentPersistencePenalty=" << ranked.parentPersistencePenalty
                      << " neighborClaimPenalty=" << ranked.neighborClaimPenalty
                      << " rankingSoftPenalty=" << ranked.rankingSoftPenalty
                      << " windowScore=" << ranked.windowSupportScore
                      << " windowBoth=" << ranked.windowBothDaughtersSupported
                      << " windowMissing=" << ranked.windowMissingDaughterCount
                      << " windowParentPersists=" << ranked.windowParentPersists
                      << " continuationClaimSoftPenalty="
                      << ranked.continuationClaimSoftPenalty
                      << " balancedWindowBonus=" << ranked.balancedWindowBonus
                      << " continuationClaimBlockers="
                      << (ranked.continuationClaimBlockerNames.empty()
                              ? "none"
                              : ranked.continuationClaimBlockerNames)
                      << " conflictReplacementEligible=" << ranked.conflictReplacementEligible
                      << " elongatedParentRescued=" << ranked.elongatedParentRescued
                      << " parentAnchored=" << ranked.parentAnchored
                      << " temporalRepairEligible=" << ranked.temporalRepairEligible
                      << " score=" << ranked.score
                      << " candidateIds=(" << ranked.candidateA << "," << ranked.candidateB << ")"
                      << " vox=(" << ranked.voxA << "," << ranked.voxB << ")"
                      << " signal=(" << ranked.signalA << "," << ranked.signalB << ")"
                      << std::endl;
        };
        struct ParentPriorGroup {
            std::string parentName;
            std::vector<size_t> rankedIndexes;
            double bestScore = 0.0;
        };

        std::map<std::string, ParentPriorGroup> groupedByParent;
        for (size_t rankedIdx = 0; rankedIdx < rankedPriors.size(); ++rankedIdx) {
            const auto &ranked = rankedPriors[rankedIdx];
            if (ranked.parentIdx >= frame.cells.size()) {
                continue;
            }
            if (ranked.candidateA < 0 || ranked.candidateB < 0 ||
                ranked.candidateA == ranked.candidateB) {
                continue;
            }
            const std::string parentName = frame.cells[ranked.parentIdx].getName();
            auto &group = groupedByParent[parentName];
            group.parentName = parentName;
            group.rankedIndexes.push_back(rankedIdx);
        }

        std::vector<ParentPriorGroup> parentGroups;
        parentGroups.reserve(groupedByParent.size());
        for (auto &entry : groupedByParent) {
            auto &group = entry.second;
            std::sort(group.rankedIndexes.begin(), group.rankedIndexes.end(),
                      [&](size_t a, size_t b) {
                          return preferRankedSplitPrior(rankedPriors[a],
                                                        rankedPriors[b]);
                      });
            group.bestScore = group.rankedIndexes.empty()
                                  ? std::numeric_limits<double>::infinity()
                                  : rankedPriors[group.rankedIndexes.front()].score;
            parentGroups.push_back(group);
        }
        std::sort(parentGroups.begin(), parentGroups.end(),
                  [](const ParentPriorGroup &a, const ParentPriorGroup &b) {
                      if (std::abs(a.bestScore - b.bestScore) > 1e-9) {
                          return a.bestScore < b.bestScore;
                      }
                      return a.parentName < b.parentName;
                  });

        std::set<size_t> selectedRankedPriorIndexes;
        size_t searchNodes = 0;
        bool searchTruncated = false;
        const double maxSelectableCost =
            static_cast<double>(lumenConfig.fusionSplitPriorGlobalSelectMaxCost);
        auto hasCleanWindowSupport = [](const RankedSplitPrior &ranked) {
            return ranked.windowBothDaughtersSupported > 0 &&
                   ranked.windowMissingDaughterCount == 0 &&
                   ranked.windowParentPersists == 0;
        };
        auto parentAgeFramesForRanked = [&](const RankedSplitPrior &ranked) {
            if (ranked.parentIdx >= frame.cells.size()) {
                return std::numeric_limits<int>::max() / 4;
            }
            const std::string &parentName = frame.cells[ranked.parentIdx].getName();
            const auto firstSeenIt = cellFirstSeenFrame.find(parentName);
            if (firstSeenIt == cellFirstSeenFrame.end()) {
                return std::numeric_limits<int>::max() / 4;
            }
            return std::max(0, absoluteFrame - firstSeenIt->second);
        };
        auto temporalRepairParentAgeOk = [&](const RankedSplitPrior &ranked) {
            const int minAge = std::max(
                0,
                lumenConfig.fusionSplitPriorTemporalRepairMinParentAgeFrames);
            if (minAge <= 0 || !ranked.temporalRepairEligible) {
                return true;
            }
            return parentAgeFramesForRanked(ranked) >= minAge;
        };
        auto isFutureContinuationConflictRescue =
            [&](const RankedSplitPrior &ranked) {
                if (ranked.parentAnchored || !hasCleanWindowSupport(ranked)) {
                    return false;
                }
                return ranked.continuationClaimBlockers.size() == 1 &&
                       ranked.neighborClaimPenalty <= 1e-5f &&
                       ranked.parentPersistencePenalty <= 1e-5f &&
                       ranked.continuationClaimSoftPenalty >= 15.0f &&
                       ranked.rankingSoftPenalty >= 4.0f &&
                       ranked.parentDistanceBalance >=
                           std::max(
                               0.45f,
                               lumenConfig
                                   .fusionSplitPriorMinParentDistanceBalance) &&
                       ranked.nearParentDist >=
                           std::max(
                               8.0f,
                               lumenConfig
                                   .fusionSplitPriorMinDaughterParentDistance) &&
                       ranked.farParentDist >=
                           std::max(ranked.minSep * 1.45f, 16.0f) &&
                       ranked.midpointDist <=
                           std::max(ranked.minSep * 0.55f, 7.0f) &&
                       ranked.sep >= ranked.minSep * 2.0f &&
                       ranked.score <= maxSelectableCost;
            };
        if (maxPriors > 0 && !parentGroups.empty()) {
            std::set<int> activeCandidateIds;
            std::set<size_t> activeParentIndexes;
            std::vector<size_t> activeSelection;
            std::vector<size_t> bestSelection;
            int bestCount = 0;
            double bestScore = 0.0;
            const int finiteMaxPriors =
                maxPriors == std::numeric_limits<int>::max()
                    ? static_cast<int>(parentGroups.size())
                    : std::min<int>(maxPriors, static_cast<int>(parentGroups.size()));
            std::vector<std::vector<size_t>> viableRankedIndexesByGroup(parentGroups.size());
            auto hasTolerableContinuationClaim = [&](const RankedSplitPrior &ranked) {
                if (ranked.continuationClaimBlockers.empty()) {
                    return true;
                }
                return ranked.continuationClaimBlockers.size() == 1 &&
                       ranked.parentDistanceBalance >=
                           std::max(0.0f,
                                    lumenConfig
                                        .fusionSplitPriorMinParentDistanceBalance) &&
                       ranked.nearParentDist >=
                           std::max(6.0f,
                                    lumenConfig
                                        .fusionSplitPriorMinDaughterParentDistance);
            };
            auto isWindowBackedConflictResolution = [&](const RankedSplitPrior &ranked) {
                if (!hasCleanWindowSupport(ranked)) {
                    return false;
                }
                const double hardScoreCeiling =
                    static_cast<double>(std::max(
                        lumenConfig.fusionSplitPriorMaxScore * 2.0f + 20.0f,
                        0.0f));
                return ranked.neighborClaimPenalty >
                           std::max(0.0f, lumenConfig.fusionSplitPriorMaxNeighborClaimPenalty) &&
                       ranked.parentDistanceBalance >=
                           std::max(0.0f, lumenConfig.fusionSplitPriorMinParentDistanceBalance) &&
                       ranked.nearParentDist >= 10.0f &&
                       ranked.score <= hardScoreCeiling;
            };
            auto isWeakAsymmetricElongatedRescue = [&](const RankedSplitPrior &ranked) {
                const bool cleanAsymmetricWindowPair =
                    !ranked.parentAnchored &&
                    hasCleanWindowSupport(ranked) &&
                    ranked.continuationClaimBlockers.empty() &&
                    ranked.neighborClaimPenalty <= 1e-5f &&
                    ranked.parentShapeElongation >=
                        std::max(
                            1.50f,
                            lumenConfig.fusionSplitPriorElongatedParentMinShape) &&
                    ranked.parentDistanceBalance >= 0.40f &&
                    ranked.nearParentDist >=
                        std::max(
                            6.0f,
                            lumenConfig.fusionSplitPriorMinDaughterParentDistance) &&
                    ranked.score <= maxSelectableCost;
                if (cleanAsymmetricWindowPair) {
                    return false;
                }
                return !ranked.parentAnchored &&
                       ranked.parentDistanceBalance <
                           std::max(
                               0.60f,
                               lumenConfig
                                   .fusionSplitPriorWindowBalancedMinParentDistanceBalance) &&
                       ranked.rawScore > 0.0 &&
                       ranked.score <= 0.0;
            };
            auto isStrongParentAnchoredOneSided = [&](const RankedSplitPrior &ranked) {
                if (!ranked.parentAnchored || !hasCleanWindowSupport(ranked)) {
                    return false;
                }
                return ranked.continuationClaimBlockers.empty() &&
                       ranked.rankingSoftPenalty <= 1e-5f &&
                       ranked.neighborClaimPenalty <= 1e-5f &&
                       ranked.parentShapeElongation >=
                           std::max(
                               1.0f,
                               lumenConfig.fusionSplitPriorParentAnchorMinShape) &&
                       ranked.farParentDist >=
                           std::max(ranked.minSep * 1.75f,
                                    std::max(6.0f,
                                             lumenConfig
                                                 .fusionSplitPriorMinDaughterParentDistance)) &&
                       ranked.score <= maxSelectableCost;
            };
            auto isRealLumenCandidateId = [](int candidateId) {
                return candidateId >= 0 && candidateId < 1000000000;
            };
            auto isCurrentFrameLumenCandidateId = [](int candidateId) {
                return candidateId >= 0 && candidateId < 1000000;
            };
            auto isFutureOnlyParentAnchorPositiveRaw =
                [&](const RankedSplitPrior &ranked) {
                    if (!lumenConfig
                             .fusionSplitPriorRejectFutureOnlyParentAnchorPositiveRawEnabled ||
                        !ranked.parentAnchored) {
                        return false;
                    }
                    if (static_cast<int>(frame.cells.size()) <
                        std::max(
                            0,
                            lumenConfig
                                .fusionSplitPriorRejectFutureOnlyParentAnchorPositiveRawMinLiveCells)) {
                        return false;
                    }
                    const bool realA = isRealLumenCandidateId(ranked.candidateA);
                    const bool realB = isRealLumenCandidateId(ranked.candidateB);
                    if (realA == realB) {
                        return false;
                    }
                    const bool currentRealA =
                        realA && isCurrentFrameLumenCandidateId(ranked.candidateA);
                    const bool currentRealB =
                        realB && isCurrentFrameLumenCandidateId(ranked.candidateB);
                    if (currentRealA || currentRealB) {
                        return false;
                    }
                    // f060 showed a future-only lookahead center plus the old
                    // parent anchor can invent an extra daughter after the
                    // current-frame Cell Lumen field is already rich enough.
                    // Keep true current-frame one-real rescues available, but
                    // make future-only rescues prove a negative raw cost first.
                    return ranked.rawScore >
                               static_cast<double>(
                                   lumenConfig
                                       .fusionSplitPriorRejectFutureOnlyParentAnchorPositiveRawMaxRawScore) &&
                           ranked.windowBothDaughtersSupported >= 2 &&
                           ranked.windowMissingDaughterCount == 0 &&
                           ranked.windowParentPersists == 0 &&
                           ranked.neighborClaimPenalty <= 1e-5f &&
                           ranked.continuationClaimSoftPenalty <= 1e-5f &&
                           ranked.parentPersistencePenalty <= 1e-5f;
                };
            auto isWeakParentAnchoredOneReal =
                [&](const RankedSplitPrior &ranked) {
                    if (!lumenConfig
                             .fusionSplitPriorRejectWeakParentAnchoredOneRealEnabled ||
                        !ranked.parentAnchored) {
                        return false;
                    }
                    const bool realA =
                        isCurrentFrameLumenCandidateId(ranked.candidateA);
                    const bool realB =
                        isCurrentFrameLumenCandidateId(ranked.candidateB);
                    if (realA == realB) {
                        return false;
                    }
                    const int realVoxels = realA ? ranked.voxA : ranked.voxB;
                    const float realSignal =
                        realA ? ranked.signalA : ranked.signalB;
                    const auto stats = splitPriorAxisStats(ranked);
                    const bool widePartialParentAnchorBypass =
                        lumenConfig
                            .fusionSplitPriorWeakParentAnchoredPartialBypassEnabled &&
                        ranked.parentShapeElongation >=
                            std::max(
                                1.0f,
                                lumenConfig
                                    .fusionSplitPriorWeakParentAnchoredPartialBypassMinParentShape) &&
                        ranked.score <=
                            static_cast<double>(
                                std::max(
                                    0.0f,
                                    lumenConfig
                                        .fusionSplitPriorWeakParentAnchoredPartialBypassMaxScore)) &&
                        realVoxels >=
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorWeakParentAnchoredPartialBypassMinRealVoxels) &&
                        realSignal >=
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorWeakParentAnchoredPartialBypassMinRealSignal) &&
                        ranked.windowBothDaughtersSupported >=
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorWeakParentAnchoredPartialBypassMinWindowBoth) &&
                        ranked.windowMissingDaughterCount <=
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorWeakParentAnchoredPartialBypassMaxWindowMissing) &&
                        ranked.windowParentPersists <=
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorWeakParentAnchoredPartialBypassMaxWindowParentPersists) &&
                        ranked.continuationClaimBlockers.empty() &&
                        ranked.neighborClaimPenalty <= 1e-5f &&
                        ranked.continuationClaimSoftPenalty <= 1e-5f &&
                        ranked.parentPersistencePenalty <= 1e-5f &&
                        ranked.sep >=
                            ranked.minSep *
                                std::max(
                                    0.0f,
                                    lumenConfig
                                        .fusionSplitPriorWeakParentAnchoredPartialBypassMinSeparationRadiusScale) &&
                        stats.first >=
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorWeakParentAnchoredPartialBypassMinLateralSeparation) &&
                        stats.second <=
                            std::clamp(
                                lumenConfig
                                    .fusionSplitPriorWeakParentAnchoredPartialBypassMaxZDominance,
                                0.0f, 1.0f) &&
                        ranked.midpointDist <=
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorWeakParentAnchoredPartialBypassMaxMidpointDistance);
                    // f018 Cell type 1_10 is a real partial parent-anchor
                    // split: one current Cell Lumen center is dimmer than the
                    // weak-anchor guard wants, but it is laterally separated
                    // from the parent anchor and has partial future support.
                    // Do not classify that geometry as an internal texture peak.
                    if (widePartialParentAnchorBypass) {
                        return false;
                    }
                    // Sparse large cells often keep one stable internal bright
                    // peak while the old parent center remains valid. That
                    // looks like one real CellLumen center plus the synthetic
                    // parent anchor. Reject it unless the real center is strong
                    // enough and not just a tight same-column texture peak.
                    return realVoxels <
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorWeakParentAnchoredOneRealMinVoxels) ||
                           realSignal <
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorWeakParentAnchoredOneRealMinSignal) ||
                           ranked.sep <
                               ranked.minSep *
                                   std::max(
                                       0.0f,
                                       lumenConfig
                                           .fusionSplitPriorWeakParentAnchoredOneRealMinSeparationRadiusScale) ||
                           stats.second >
                               std::clamp(
                                   lumenConfig
                                       .fusionSplitPriorWeakParentAnchoredOneRealMaxZDominance,
                                   0.0f, 1.0f) ||
                           stats.first <
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorWeakParentAnchoredOneRealMinLateralSeparation);
                };
            auto isTightCleanWindowInternalPair =
                [&](const RankedSplitPrior &ranked) {
                    if (!lumenConfig
                             .fusionSplitPriorRejectTightCleanWindowInternalPairEnabled ||
                        ranked.parentAnchored ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateA) ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateB) ||
                        !hasCleanWindowSupport(ranked)) {
                        return false;
                    }
                    // A clean future window can repeat the same internal cell
                    // texture peaks. If the two centers are only about one
                    // daughter radius apart and the midpoint is not close to
                    // the parent center, treat it as an internal split, not a
                    // true lineage event. Strong balanced-window evidence is
                    // allowed to pass because real emerging daughters often
                    // have symmetric support.
                    return ranked.balancedWindowBonus <=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorTightCleanWindowInternalMaxBalancedBonus) &&
                           ranked.neighborClaimPenalty <=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorTightCleanWindowInternalMaxNeighborClaimPenalty) &&
                           ranked.sep <
                               ranked.minSep *
                                   std::max(
                                       0.0f,
                                       lumenConfig
                                           .fusionSplitPriorTightCleanWindowInternalMinSeparationRadiusScale) &&
                           ranked.midpointDist >=
                               ranked.minSep *
                                   std::max(
                                       0.0f,
                                           lumenConfig
                                               .fusionSplitPriorTightCleanWindowInternalMinMidpointRadiusScale);
                };
            auto isWeakBalancedCleanWindowPair =
                [&](const RankedSplitPrior &ranked) {
                    if (!lumenConfig
                             .fusionSplitPriorRejectWeakBalancedCleanWindowPairEnabled ||
                        ranked.parentAnchored ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateA) ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateB) ||
                        !hasCleanWindowSupport(ranked) ||
                        ranked.balancedWindowBonus <
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorWeakBalancedCleanWindowMinBonus)) {
                        return false;
                    }
                    // f058 exposed a new early sparse-frame failure mode: stable
                    // internal bright texture peaks can appear in the future
                    // window and receive the balanced-window bonus, but their
                    // voxel support or signal is still much weaker than true
                    // daughter centers. Keep this default-off so dense-stage
                    // high-recall behavior is untouched unless YAML opts in.
                    const int minVoxels =
                        std::max(
                            0,
                            lumenConfig
                                .fusionSplitPriorWeakBalancedCleanWindowMinVoxels);
                    const float minSignal =
                        std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorWeakBalancedCleanWindowMinSignal);
                    const bool cleanTwoRealWeakBalancedBypass =
                        lumenConfig
                            .fusionSplitPriorCleanTwoRealWeakBalancedBypassEnabled &&
                        ranked.windowBothDaughtersSupported >= 2 &&
                        ranked.windowMissingDaughterCount == 0 &&
                        ranked.windowParentPersists == 0 &&
                        ranked.continuationClaimBlockers.empty() &&
                        ranked.neighborClaimPenalty <= 1e-5f &&
                        ranked.parentPersistencePenalty <= 1e-5f &&
                        ranked.continuationClaimSoftPenalty <= 1e-5f &&
                        ranked.parentShapeElongation <=
                            std::max(
                                1.0f,
                                lumenConfig
                                    .fusionSplitPriorCleanTwoRealWeakBalancedBypassMaxParentShape) &&
                        ranked.sep >=
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorCleanTwoRealWeakBalancedBypassMinSeparation) &&
                        ranked.midpointDist <=
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorCleanTwoRealWeakBalancedBypassMaxMidpointDistance) &&
                        ranked.parentDistanceBalance >=
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorCleanTwoRealWeakBalancedBypassMinParentDistanceBalance) &&
                        ranked.signalA >=
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorCleanTwoRealWeakBalancedBypassMinSignal) &&
                        ranked.signalB >=
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorCleanTwoRealWeakBalancedBypassMinSignal) &&
                        ranked.score <=
                            static_cast<double>(
                                std::max(
                                    0.0f,
                                    lumenConfig
                                        .fusionSplitPriorCleanTwoRealWeakBalancedBypassMaxScore)) &&
                        ranked.rawScore <=
                            static_cast<double>(
                                lumenConfig
                                    .fusionSplitPriorCleanTwoRealWeakBalancedBypassMaxRawScore);
                    const float negativeGeometryWeakSignal =
                        std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorCleanTwoRealWeakBalancedNegativeGeometryMinWeakSignal);
                    const float negativeGeometryStrongSignal =
                        std::max(
                            negativeGeometryWeakSignal,
                            lumenConfig
                                .fusionSplitPriorCleanTwoRealWeakBalancedNegativeGeometryMinStrongSignal);
                    // f011 exposed the opposite edge case from f058: a true
                    // two-real split can have one dim daughter, but the pair is
                    // wide, centered, balanced, future-supported, and already
                    // negative-cost. Let that evidence reach the global
                    // selector without weakening the default texture guard.
                    const bool cleanTwoRealWeakBalancedNegativeGeometryBypass =
                        lumenConfig
                            .fusionSplitPriorCleanTwoRealWeakBalancedNegativeGeometryBypassEnabled &&
                        ranked.windowBothDaughtersSupported >= 2 &&
                        ranked.windowMissingDaughterCount == 0 &&
                        ranked.windowParentPersists == 0 &&
                        ranked.continuationClaimBlockers.empty() &&
                        ranked.neighborClaimPenalty <= 1e-5f &&
                        ranked.parentPersistencePenalty <= 1e-5f &&
                        ranked.continuationClaimSoftPenalty <= 1e-5f &&
                        ranked.parentShapeElongation <=
                            std::max(
                                1.0f,
                                lumenConfig
                                    .fusionSplitPriorCleanTwoRealWeakBalancedNegativeGeometryMaxParentShape) &&
                        ranked.sep >=
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorCleanTwoRealWeakBalancedNegativeGeometryMinSeparation) &&
                        ranked.midpointDist <=
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorCleanTwoRealWeakBalancedNegativeGeometryMaxMidpointDistance) &&
                        ranked.parentDistanceBalance >=
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorCleanTwoRealWeakBalancedNegativeGeometryMinParentDistanceBalance) &&
                        ranked.voxA >=
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorCleanTwoRealWeakBalancedNegativeGeometryMinVoxels) &&
                        ranked.voxB >=
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorCleanTwoRealWeakBalancedNegativeGeometryMinVoxels) &&
                        std::min(ranked.signalA, ranked.signalB) >=
                            negativeGeometryWeakSignal &&
                        std::max(ranked.signalA, ranked.signalB) >=
                            negativeGeometryStrongSignal &&
                        ranked.score <=
                            static_cast<double>(
                                lumenConfig
                                    .fusionSplitPriorCleanTwoRealWeakBalancedNegativeGeometryMaxScore);
                    // f060 has one low-voxel daughter but otherwise clean
                    // two-real evidence. Let that narrow pattern reach the
                    // global selector instead of being treated like internal
                    // texture from f058-style false balanced pairs.
                    if (cleanTwoRealWeakBalancedBypass ||
                        cleanTwoRealWeakBalancedNegativeGeometryBypass) {
                        return false;
                    }
                    return ranked.voxA < minVoxels ||
                           ranked.voxB < minVoxels ||
                           ranked.signalA < minSignal ||
                           ranked.signalB < minSignal;
                };
            auto isWeakAsymmetricCleanWindowPair =
                [&](const RankedSplitPrior &ranked) {
                    if (!lumenConfig
                             .fusionSplitPriorRejectWeakAsymmetricCleanWindowPairEnabled ||
                        ranked.parentAnchored ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateA) ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateB) ||
                        !hasCleanWindowSupport(ranked)) {
                        return false;
                    }
                    // f058 Cell type 1_410 showed a future-supported pair can
                    // still be two weak internal texture peaks. The pair had
                    // no balanced daughter bonus, an asymmetric parent-distance
                    // layout, and one weak Cell Lumen side. Keep these centers
                    // available for continuation, but do not commit a lineage
                    // split unless the evidence is stronger.
                    if (ranked.windowBothDaughtersSupported <
                            std::max(
                                1,
                                lumenConfig
                                    .fusionSplitPriorWeakAsymmetricCleanWindowMinWindowBoth) ||
                        ranked.balancedWindowBonus >
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorWeakAsymmetricCleanWindowMaxBalancedBonus) ||
                        ranked.parentDistanceBalance >
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorWeakAsymmetricCleanWindowMaxParentDistanceBalance) ||
                        ranked.rawScore <=
                            static_cast<double>(
                                lumenConfig
                                    .fusionSplitPriorWeakAsymmetricCleanWindowMinRawScore)) {
                        return false;
                    }
                    const int minVoxels =
                        std::max(
                            0,
                            lumenConfig
                                .fusionSplitPriorWeakAsymmetricCleanWindowMinVoxels);
                    const float minSignal =
                        std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorWeakAsymmetricCleanWindowMinSignal);
	                    return ranked.voxA < minVoxels ||
	                           ranked.voxB < minVoxels ||
	                           ranked.signalA < minSignal ||
	                           ranked.signalB < minSignal;
	                };
            auto isTriaxialNoBalancedCleanWindowPair =
                [&](const RankedSplitPrior &ranked) {
                    if (!lumenConfig
                             .fusionSplitPriorRejectTriaxialNoBalancedCleanWindowPairEnabled ||
                        ranked.parentAnchored ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateA) ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateB) ||
                        !hasCleanWindowSupport(ranked)) {
                        return false;
                    }
                    // Some early large cells are flattened bright bodies rather
                    // than one clear bridge between two daughters. Their internal
                    // texture peaks can persist in future Cell Lumen frames, so a
                    // clean future window is not enough if the parent shape is
                    // triaxial and the pair did not earn the balanced daughter
                    // bonus. Keep this YAML-gated so verified dense behavior is
                    // untouched unless a profile explicitly enables it.
                    return ranked.windowBothDaughtersSupported >=
                               std::max(
                                   1,
                                   lumenConfig
                                       .fusionSplitPriorTriaxialNoBalancedCleanWindowMinWindowBoth) &&
                           ranked.balancedWindowBonus <=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorTriaxialNoBalancedCleanWindowMaxBalancedBonus) &&
                           ranked.parentShapeElongation >=
                               std::max(
                                   1.0f,
                                   lumenConfig
                                       .fusionSplitPriorTriaxialNoBalancedCleanWindowMinParentShape) &&
                           ranked.parentLongMidRatio <=
                               std::max(
                                   1.0f,
                                   lumenConfig
                                       .fusionSplitPriorTriaxialNoBalancedCleanWindowMaxLongMidRatio) &&
                           ranked.parentMidShortRatio >=
                               std::max(
                                   1.0f,
                                   lumenConfig
                                       .fusionSplitPriorTriaxialNoBalancedCleanWindowMinMidShortRatio) &&
                           ranked.parentDistanceBalance >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorTriaxialNoBalancedCleanWindowMinParentDistanceBalance) &&
                           ranked.rawScore <=
                               static_cast<double>(
                                   lumenConfig
                                       .fusionSplitPriorTriaxialNoBalancedCleanWindowMaxRawScore);
                };
            auto isParentAnchoredSeedColumnDuplicate =
                [&](const RankedSplitPrior &ranked) {
                    if (!lumenConfig
                             .fusionSplitPriorRejectParentAnchoredSeedColumnDuplicate ||
                        !ranked.parentAnchored) {
                        return false;
                    }
                    const bool realA =
                        isCurrentFrameLumenCandidateId(ranked.candidateA);
                    const bool realB =
                        isCurrentFrameLumenCandidateId(ranked.candidateB);
                    if (realA == realB) {
                        return false;
                    }
                    const float minParentShape =
                        std::max(
                            1.0f,
                            lumenConfig
                                .fusionSplitPriorParentAnchoredSeedColumnMinParentShape);
                    const float maxParentShape =
                        std::max(
                            minParentShape,
                            lumenConfig
                                .fusionSplitPriorParentAnchoredSeedColumnMaxParentShape);
                    if (ranked.parentShapeElongation < minParentShape ||
                        ranked.parentShapeElongation > maxParentShape) {
                        return false;
                    }
                    const int realCandidateId =
                        realA ? ranked.candidateA : ranked.candidateB;
                    const cv::Point3f realPos =
                        realA ? ranked.proposal.d1Pos
                              : ranked.proposal.d2Pos;
                    const cv::Point3f anchorPos =
                        realA ? ranked.proposal.d2Pos
                              : ranked.proposal.d1Pos;
                    const float selectedAnchorDistance =
                        static_cast<float>(cv::norm(realPos - anchorPos));
                    const float maxSiblingLateral =
                        std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorParentAnchoredSeedColumnMaxSiblingLateralSeparation);
                    const float minSiblingZ =
                        std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorParentAnchoredSeedColumnMinSiblingZSeparation);
                    const float closerMargin =
                        std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorParentAnchoredSeedColumnSiblingCloserMargin);
                    const float maxSiblingAnchorDistance =
                        std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorParentAnchoredSeedColumnMaxSiblingAnchorDistance);
                    for (const auto &sibling : currentFrameLookaheadCandidates) {
                        if (sibling.candidateId == realCandidateId) {
                            continue;
                        }
                        const cv::Point3f delta =
                            sibling.position - realPos;
                        const float lateral =
                            std::sqrt(delta.x * delta.x + delta.y * delta.y);
                        const float zSep = std::abs(delta.z);
                        if (lateral > maxSiblingLateral || zSep < minSiblingZ) {
                            continue;
                        }
                        const float siblingAnchorDistance =
                            static_cast<float>(
                                cv::norm(sibling.position - anchorPos));
                        // This guard exists to stop one-real parent-anchored
                        // splits from turning a same-XY z column into a fake
                        // lineage split. Keep it narrow: a partial-window
                        // rescue can still be valid when the selected real
                        // seed is the better anchor-side evidence.
                        const bool siblingIsCloserToAnchor =
                            siblingAnchorDistance + closerMargin <=
                            selectedAnchorDistance;
                        const bool cleanWindowSameColumn =
                            ranked.windowBothDaughtersSupported >= 2 &&
                            ranked.windowMissingDaughterCount == 0;
                        const bool cleanColumnHasSafeAnchorAlternative =
                            cleanWindowSameColumn &&
                            siblingAnchorDistance <= maxSiblingAnchorDistance;
                        const bool wideSameColumnZDuplicate =
                            zSep >= minSiblingZ;
                        const bool siblingIsSaferContinuation =
                            siblingIsCloserToAnchor ||
                            cleanColumnHasSafeAnchorAlternative ||
                            wideSameColumnZDuplicate;
                        if (siblingIsSaferContinuation) {
                            return true;
                        }
                    }
                    return false;
                };
            auto isCompactParentAnchoredWindowRescue = [&](const RankedSplitPrior &ranked) {
                if (!lumenConfig
                         .fusionSplitPriorCompactParentAnchorWindowRescueEnabled ||
                    !ranked.parentAnchored || !hasCleanWindowSupport(ranked)) {
                    return false;
                }
                const bool realA = isRealLumenCandidateId(ranked.candidateA);
                const bool realB = isRealLumenCandidateId(ranked.candidateB);
                if (realA == realB ||
                    !ranked.continuationClaimBlockers.empty()) {
                    return false;
                }
                const int realVoxels = realA ? ranked.voxA : ranked.voxB;
                const float realSignal = realA ? ranked.signalA : ranked.signalB;
                return ranked.parentShapeElongation >=
                           std::max(
                               1.0f,
                               lumenConfig
                                   .fusionSplitPriorCompactParentAnchorMinShape) &&
                       ranked.score <=
                           static_cast<double>(
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCompactParentAnchorMaxScore)) &&
                       realVoxels >=
                           std::max(
                               0,
                               lumenConfig
                                   .fusionSplitPriorCompactParentAnchorMinRealVoxels) &&
                       realSignal >=
                           std::max(
                               0.0f,
                               lumenConfig
                                   .fusionSplitPriorCompactParentAnchorMinRealSignal) &&
                       ranked.farParentDist >=
                           ranked.minSep *
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCompactParentAnchorMinFarDistanceRadiusScale) &&
                       ranked.neighborClaimPenalty <= 1e-5f &&
                       ranked.continuationClaimSoftPenalty <= 1e-5f &&
                       ranked.parentPersistencePenalty <= 1e-5f;
            };
            auto isParentAnchoredSingleBlockerWindowRescue =
                [&](const RankedSplitPrior &ranked) {
                    if (!lumenConfig
                             .fusionSplitPriorParentAnchorSingleBlockerRescueEnabled ||
                        !ranked.parentAnchored ||
                        !hasCleanWindowSupport(ranked)) {
                        return false;
                    }
                    const bool realA = isRealLumenCandidateId(ranked.candidateA);
                    const bool realB = isRealLumenCandidateId(ranked.candidateB);
                    if (realA == realB ||
                        ranked.continuationClaimBlockers.size() != 1) {
                        return false;
                    }
                    const int realVoxels = realA ? ranked.voxA : ranked.voxB;
                    const float realSignal = realA ? ranked.signalA : ranked.signalB;
                    return ranked.parentShapeElongation >=
                               std::max(
                                   1.0f,
                                   lumenConfig
                                       .fusionSplitPriorParentAnchorSingleBlockerMinShape) &&
                           ranked.score <=
                               static_cast<double>(
                                   std::max(
                                       0.0f,
                                       lumenConfig
                                           .fusionSplitPriorParentAnchorSingleBlockerMaxScore)) &&
                           realVoxels >=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorParentAnchorSingleBlockerMinRealVoxels) &&
                           realSignal >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorParentAnchorSingleBlockerMinRealSignal) &&
                           ranked.farParentDist >=
                               ranked.minSep *
                                   std::max(
                                       0.0f,
                                       lumenConfig
                                           .fusionSplitPriorParentAnchorSingleBlockerMinFarDistanceRadiusScale) &&
                           ranked.continuationClaimSoftPenalty <=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorParentAnchorSingleBlockerMaxContinuationPenalty) &&
                           ranked.neighborClaimPenalty <= 1e-5f &&
                           ranked.parentPersistencePenalty <= 1e-5f;
                };
            auto isPartialParentAnchoredWindowRescue =
                [&](const RankedSplitPrior &ranked) {
                    if (!lumenConfig
                             .fusionSplitPriorPartialParentAnchorRescueEnabled ||
                        !ranked.parentAnchored) {
                        return false;
                    }
                    const bool realA = isRealLumenCandidateId(ranked.candidateA);
                    const bool realB = isRealLumenCandidateId(ranked.candidateB);
                    if (realA == realB) {
                        return false;
                    }
                    const bool conflictWindowEvidence =
                        ranked.windowMissingDaughterCount > 0 ||
                        ranked.windowParentPersists > 0;
                    const bool partialWindow =
                        conflictWindowEvidence ||
                        ranked.windowBothDaughtersSupported <
                            std::max(
                                2,
                                lumenConfig
                                    .fusionSplitPriorPartialParentAnchorMinWindowBoth);
                    // Temporal-catch parent-anchor pairs are already a narrow
                    // sparse-frame rescue: one current Cell Lumen center sits
                    // outside the normal catch radius, but the future window
                    // supports both the old parent center and the new daughter.
                    // Treat this clean-window case as selectable too; otherwise
                    // the candidate is ranked but skipped because its score is
                    // slightly positive.
                    const bool cleanTemporalCatchWindow =
                        ranked.temporalRepairCatchExpanded &&
                        ranked.windowBothDaughtersSupported >= 2 &&
                        ranked.windowMissingDaughterCount == 0 &&
                        ranked.windowParentPersists == 0;
                    if (!partialWindow && !cleanTemporalCatchWindow) {
                        return false;
                    }
                    if (lumenConfig
                            .fusionSplitPriorParentAnchorWeakGainPartialRequireConflictEvidence &&
                        !conflictWindowEvidence &&
                        !cleanTemporalCatchWindow) {
                        return false;
                    }
                    const int realVoxels = realA ? ranked.voxA : ranked.voxB;
                    const float realSignal = realA ? ranked.signalA : ranked.signalB;
                    return ranked.parentShapeElongation >=
                               std::max(
                                   1.0f,
                                   lumenConfig
                                       .fusionSplitPriorPartialParentAnchorMinShape) &&
                           ranked.score <=
                               static_cast<double>(
                                   lumenConfig
                                       .fusionSplitPriorPartialParentAnchorMaxScore) &&
                           realVoxels >=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorPartialParentAnchorMinRealVoxels) &&
                           realSignal >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorPartialParentAnchorMinRealSignal) &&
                           ranked.windowBothDaughtersSupported >=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorPartialParentAnchorMinWindowBoth) &&
                           ranked.windowMissingDaughterCount <=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorPartialParentAnchorMaxWindowMissing) &&
                           ranked.windowParentPersists <=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorPartialParentAnchorMaxWindowParentPersists) &&
                           ranked.neighborClaimPenalty <= 1e-5f &&
                           ranked.continuationClaimSoftPenalty <= 1e-5f &&
                           ranked.parentPersistencePenalty <= 1e-5f;
                };
            auto isTemporalCatchParentAnchoredPartialWindowRescue =
                [&](const RankedSplitPrior &ranked) {
                    if (!lumenConfig
                             .fusionSplitPriorTemporalCatchParentAnchorPartialWindowRescueEnabled ||
                        !ranked.parentAnchored ||
                        !ranked.temporalRepairCatchExpanded) {
                        return false;
                    }
                    const bool realA =
                        isCurrentFrameLumenCandidateId(ranked.candidateA);
                    const bool realB =
                        isCurrentFrameLumenCandidateId(ranked.candidateB);
                    if (realA == realB ||
                        !ranked.continuationClaimBlockers.empty()) {
                        return false;
                    }
                    const int realVoxels = realA ? ranked.voxA : ranked.voxB;
                    const float realSignal = realA ? ranked.signalA : ranked.signalB;
                    // f059 had a real far daughter that was too positive for
                    // the normal skip-zero selector because the clean future
                    // evidence was only one frame deep. Keep this separate
                    // from the generic partial-anchor rescue so we do not
                    // reopen weak same-cell texture peaks from f058.
                    const float minFarDistance =
                        std::max(
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorTemporalCatchParentAnchorPartialWindowMinFarDistance),
                            ranked.minSep *
                                std::max(
                                    0.0f,
                                    lumenConfig
                                        .fusionSplitPriorTemporalCatchParentAnchorPartialWindowMinFarDistanceRadiusScale));
                    return ranked.parentShapeElongation >=
                               std::max(
                                   1.0f,
                                   lumenConfig
                                       .fusionSplitPriorTemporalCatchParentAnchorPartialWindowMinShape) &&
                           ranked.score <=
                               static_cast<double>(
                                   std::max(
                                       0.0f,
                                       lumenConfig
                                           .fusionSplitPriorTemporalCatchParentAnchorPartialWindowMaxScore)) &&
                           realVoxels >=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorTemporalCatchParentAnchorPartialWindowMinRealVoxels) &&
                           realSignal >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorTemporalCatchParentAnchorPartialWindowMinRealSignal) &&
                           ranked.windowBothDaughtersSupported >=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorTemporalCatchParentAnchorPartialWindowMinWindowBoth) &&
                           ranked.windowMissingDaughterCount <=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorTemporalCatchParentAnchorPartialWindowMaxWindowMissing) &&
                           ranked.windowParentPersists <=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorTemporalCatchParentAnchorPartialWindowMaxWindowParentPersists) &&
                           ranked.farParentDist >= minFarDistance &&
                           ranked.rankingSoftPenalty <=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorTemporalCatchParentAnchorPartialWindowMaxRankingSoftPenalty) &&
                           ranked.neighborClaimPenalty <= 1e-5f &&
                           ranked.continuationClaimSoftPenalty <= 1e-5f &&
                           ranked.parentPersistencePenalty <= 1e-5f;
                };
		            auto hasStrongTemporalDaughterEvidence =
		                [&](const RankedSplitPrior &ranked) {
		                    const bool twoRealCurrent =
		                        isCurrentFrameLumenCandidateId(ranked.candidateA) &&
		                        isCurrentFrameLumenCandidateId(ranked.candidateB);
		                    if (!twoRealCurrent) {
		                        return false;
		                    }
		                    const bool genericStrong =
		                        ranked.voxA >=
		                            std::max(
		                                0,
		                                lumenConfig
		                                    .fusionSplitPriorTemporalRepairStrongMinVoxels) &&
		                        ranked.voxB >=
		                            std::max(
		                                0,
		                                lumenConfig
		                                    .fusionSplitPriorTemporalRepairStrongMinVoxels) &&
		                        ranked.signalA >=
		                            std::max(
		                                0.0f,
		                                lumenConfig
		                                    .fusionSplitPriorTemporalRepairStrongMinSignal) &&
		                        ranked.signalB >=
		                            std::max(
		                                0.0f,
		                                lumenConfig
		                                    .fusionSplitPriorTemporalRepairStrongMinSignal);
		                    if (genericStrong) {
		                        return true;
		                    }
		                    // f045: a real wide split had two-frame future support
		                    // and strong voxel evidence, but one daughter signal was
		                    // just below the generic threshold. Only relax that case
		                    // for candidates that entered through temporal catch, so
		                    // ordinary close-to-parent repair stays conservative.
		                    return lumenConfig
		                               .fusionSplitPriorTemporalRepairCatchStrongEvidenceEnabled &&
		                           ranked.temporalRepairCatchExpanded &&
		                           ranked.parentShapeElongation >=
		                               std::max(
		                                   1.0f,
		                                   lumenConfig
		                                       .fusionSplitPriorTemporalRepairCatchStrongMinParentShape) &&
		                           ranked.rankingSoftPenalty <=
		                               std::max(
		                                   0.0f,
		                                   lumenConfig
		                                       .fusionSplitPriorTemporalRepairCatchStrongMaxRankingSoftPenalty) &&
		                           ranked.voxA >=
		                               std::max(
		                                   0,
		                                   lumenConfig
		                                       .fusionSplitPriorTemporalRepairCatchStrongMinVoxels) &&
		                           ranked.voxB >=
		                               std::max(
		                                   0,
		                                   lumenConfig
		                                       .fusionSplitPriorTemporalRepairCatchStrongMinVoxels) &&
		                           ranked.signalA >=
		                               std::max(
		                                   0.0f,
		                                   lumenConfig
		                                       .fusionSplitPriorTemporalRepairCatchStrongMinSignal) &&
		                           ranked.signalB >=
		                               std::max(
		                                   0.0f,
		                                   lumenConfig
		                                       .fusionSplitPriorTemporalRepairCatchStrongMinSignal);
		                };
		            auto isCleanTwoRealWindowPair = [&](const RankedSplitPrior &ranked) {
		                return !ranked.parentAnchored &&
		                       isRealLumenCandidateId(ranked.candidateA) &&
		                       isRealLumenCandidateId(ranked.candidateB) &&
	                       ranked.windowBothDaughtersSupported >= 2 &&
	                       ranked.windowMissingDaughterCount == 0 &&
	                       ranked.windowParentPersists == 0 &&
	                       ranked.continuationClaimSoftPenalty <= 1e-5f &&
	                       ranked.parentDistanceBalance >=
	                           std::max(
	                               0.0f,
	                               lumenConfig
	                                   .fusionSplitPriorCleanTwoRealWindowPairMinParentDistanceBalance) &&
	                       ranked.nearParentDist >=
	                           std::max(10.0f, ranked.minSep * 1.35f) &&
	                       ranked.score <=
	                           std::min(
	                               static_cast<double>(
	                                   std::max(
	                                       0.0f,
	                                       lumenConfig
	                                           .fusionSplitPriorCleanTwoRealWindowPairMaxScore)),
	                               maxSelectableCost);
	            };
            auto isCleanTwoRealSingleBlockerRescue =
                [&](const RankedSplitPrior &ranked) {
                    if (!lumenConfig
                             .fusionSplitPriorCleanTwoRealSingleBlockerRescueEnabled ||
                        ranked.parentAnchored ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateA) ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateB) ||
                        !hasCleanWindowSupport(ranked) ||
                        ranked.continuationClaimBlockers.size() != 1 ||
                        ranked.neighborClaimPenalty > 1e-5f ||
                        ranked.parentPersistencePenalty > 1e-5f) {
                        return false;
                    }
                    return ranked.voxA >=
                               std::max(0,
                                        lumenConfig
                                            .fusionSplitPriorCleanTwoRealSingleBlockerMinVoxels) &&
                           ranked.voxB >=
                               std::max(0,
                                        lumenConfig
                                            .fusionSplitPriorCleanTwoRealSingleBlockerMinVoxels) &&
                           ranked.signalA >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealSingleBlockerMinSignal) &&
                           ranked.signalB >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealSingleBlockerMinSignal) &&
                           ranked.parentDistanceBalance >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealSingleBlockerMinParentDistanceBalance) &&
                           ranked.nearParentDist >=
                               std::max(
                                   8.0f,
                                   lumenConfig
                                       .fusionSplitPriorMinDaughterParentDistance) &&
                           ranked.farParentDist >=
                               std::max(ranked.minSep * 1.45f, 16.0f) &&
                           ranked.midpointDist <=
                               std::max(ranked.minSep * 0.70f, 8.0f) &&
                           ranked.sep >= ranked.minSep * 2.0f &&
                           ranked.score <=
                               std::min(
                                   static_cast<double>(
                                       std::max(
                                           0.0f,
                                           lumenConfig
                                               .fusionSplitPriorCleanTwoRealSingleBlockerMaxScore)),
                                   maxSelectableCost);
                };
            auto isCleanTwoRealAsymmetricRescue =
                [&](const RankedSplitPrior &ranked) {
                    if (!lumenConfig
                             .fusionSplitPriorCleanTwoRealAsymmetricRescueEnabled ||
                        ranked.parentAnchored ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateA) ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateB) ||
                        !hasCleanWindowSupport(ranked) ||
                        !ranked.continuationClaimBlockers.empty() ||
                        ranked.parentPersistencePenalty > 1e-5f ||
                        ranked.continuationClaimSoftPenalty > 1e-5f) {
                        return false;
                    }
                    return ranked.voxA >=
                               std::max(0,
                                        lumenConfig
                                            .fusionSplitPriorCleanTwoRealAsymmetricMinVoxels) &&
                           ranked.voxB >=
                               std::max(0,
                                        lumenConfig
                                            .fusionSplitPriorCleanTwoRealAsymmetricMinVoxels) &&
                           ranked.signalA >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealAsymmetricMinSignal) &&
                           ranked.signalB >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealAsymmetricMinSignal) &&
                           ranked.neighborClaimPenalty <=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealAsymmetricMaxNeighborClaimPenalty) &&
                           ranked.parentDistanceBalance >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealAsymmetricMinParentDistanceBalance) &&
                           ranked.nearParentDist >=
                               std::max(
                                   6.0f,
                                   lumenConfig
                                       .fusionSplitPriorMinDaughterParentDistance) &&
                           ranked.farParentDist >=
                               std::max(ranked.minSep * 1.40f, 14.0f) &&
                           ranked.midpointDist <=
                               std::max(ranked.minSep, 10.0f) &&
                           ranked.sep >= ranked.minSep * 2.0f &&
                           ranked.score <=
                               static_cast<double>(
                                   std::max(
                                       0.0f,
                                       lumenConfig
                                           .fusionSplitPriorCleanTwoRealAsymmetricMaxScore));
                };
            auto isCleanTwoRealHighNeighborClaimRescue =
                [&](const RankedSplitPrior &ranked) {
                    if (!lumenConfig
                             .fusionSplitPriorCleanTwoRealHighNeighborClaimRescueEnabled ||
                        ranked.parentAnchored ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateA) ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateB) ||
                        !hasCleanWindowSupport(ranked) ||
                        !ranked.continuationClaimBlockers.empty() ||
                        ranked.parentPersistencePenalty > 1e-5f) {
                        return false;
                    }
                    return ranked.voxA >=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealHighNeighborClaimMinVoxels) &&
                           ranked.voxB >=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealHighNeighborClaimMinVoxels) &&
                           ranked.signalA >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealHighNeighborClaimMinSignal) &&
                           ranked.signalB >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealHighNeighborClaimMinSignal) &&
                           ranked.neighborClaimPenalty >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealHighNeighborClaimMinNeighborClaimPenalty) &&
                           ranked.continuationClaimSoftPenalty <=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealHighNeighborClaimMaxContinuationClaimPenalty) &&
                           ranked.parentDistanceBalance >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealHighNeighborClaimMinParentDistanceBalance) &&
                           ranked.nearParentDist >=
                               std::max(10.0f, ranked.minSep * 1.35f) &&
                           ranked.sep >= ranked.minSep * 2.0f &&
                           ranked.score <=
                               static_cast<double>(
                                   std::max(
                                       0.0f,
                                       lumenConfig
                                           .fusionSplitPriorCleanTwoRealHighNeighborClaimMaxScore));
                };
            auto isCleanTwoRealCompactRescue =
                [&](const RankedSplitPrior &ranked) {
                    if (!lumenConfig
                             .fusionSplitPriorCleanTwoRealCompactRescueEnabled ||
                        ranked.parentAnchored ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateA) ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateB) ||
                        !hasCleanWindowSupport(ranked) ||
                        !ranked.continuationClaimBlockers.empty() ||
                        ranked.neighborClaimPenalty > 1e-5f ||
                        ranked.continuationClaimSoftPenalty > 1e-5f) {
                        return false;
                    }
                    return ranked.voxA >=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealCompactMinVoxels) &&
                           ranked.voxB >=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealCompactMinVoxels) &&
                           ranked.signalA >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealCompactMinSignal) &&
                           ranked.signalB >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealCompactMinSignal) &&
                           ranked.parentPersistencePenalty <=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealCompactMaxParentPersistencePenalty) &&
                           ranked.parentDistanceBalance >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealCompactMinParentDistanceBalance) &&
                           ranked.nearParentDist >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealCompactMinNearParentDistance) &&
                           ranked.sep >=
                               ranked.minSep *
                                   std::max(
                                       0.0f,
                                       lumenConfig
                                           .fusionSplitPriorCleanTwoRealCompactMinSeparationRadiusScale) &&
                           ranked.midpointDist <=
                               std::max(
                                   ranked.minSep,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealCompactMaxMidpointDistance) &&
                           ranked.score <=
                               static_cast<double>(
                                   std::max(
                                       0.0f,
                                       lumenConfig
                                           .fusionSplitPriorCleanTwoRealCompactMaxScore));
                };
            auto sharesParentAnchorAlternative =
                [&](const RankedSplitPrior &ranked) {
                    const double maxAnchorScoreDelta =
                        static_cast<double>(
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorCleanTwoRealParentAnchorReplacementMaxAnchorScoreDelta));
                    for (const RankedSplitPrior &other : rankedPriors) {
                        if (&other == &ranked ||
                            other.parentIdx != ranked.parentIdx ||
                            !other.parentAnchored) {
                            continue;
                        }
                        const bool otherRealA =
                            isCurrentFrameLumenCandidateId(other.candidateA);
                        const bool otherRealB =
                            isCurrentFrameLumenCandidateId(other.candidateB);
                        if (otherRealA == otherRealB ||
                            other.windowBothDaughtersSupported <
                                std::max(
                                    0,
                                    lumenConfig
                                        .fusionSplitPriorCleanTwoRealParentAnchorReplacementMinWindowBoth) ||
                            other.windowMissingDaughterCount >
                                std::max(
                                    0,
                                    lumenConfig
                                        .fusionSplitPriorCleanTwoRealParentAnchorReplacementMaxWindowMissing) ||
                            other.windowParentPersists >
                                std::max(
                                    0,
                                    lumenConfig
                                        .fusionSplitPriorCleanTwoRealParentAnchorReplacementMaxWindowParentPersists)) {
                            continue;
                        }
                        const int otherRealId =
                            otherRealA ? other.candidateA : other.candidateB;
                        const bool sharesReal =
                            ranked.candidateA == otherRealId ||
                            ranked.candidateB == otherRealId;
                        if (sharesReal &&
                            ranked.score - other.score <= maxAnchorScoreDelta) {
                            return true;
                        }
                    }
                    return false;
                };
            auto isCleanTwoRealParentAnchorReplacementRescue =
                [&](const RankedSplitPrior &ranked) {
                    if (!lumenConfig
                             .fusionSplitPriorCleanTwoRealParentAnchorReplacementRescueEnabled ||
                        ranked.parentAnchored ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateA) ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateB) ||
                        ranked.windowBothDaughtersSupported <
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorCleanTwoRealParentAnchorReplacementMinWindowBoth) ||
                        ranked.windowMissingDaughterCount >
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorCleanTwoRealParentAnchorReplacementMaxWindowMissing) ||
                        ranked.windowParentPersists >
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorCleanTwoRealParentAnchorReplacementMaxWindowParentPersists) ||
                        !ranked.continuationClaimBlockers.empty() ||
                        ranked.neighborClaimPenalty > 1e-5f ||
                        ranked.parentPersistencePenalty >
                            std::max(1e-5f, ranked.minSep * 0.05f) ||
                        ranked.continuationClaimSoftPenalty > 1e-5f ||
                        !sharesParentAnchorAlternative(ranked)) {
                        return false;
                    }
                    // Prefer measured two-center evidence only for the narrow
                    // f086 pattern: the parent-anchor option shares one real
                    // center, but two current Cell Lumen centers persist across
                    // the future window. The voxel/signal and midpoint checks
                    // keep this from becoming a general "accept compact pair"
                    // shortcut in denser frames.
                    return ranked.voxA >=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealParentAnchorReplacementMinVoxels) &&
                           ranked.voxB >=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealParentAnchorReplacementMinVoxels) &&
                           ranked.signalA >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealParentAnchorReplacementMinSignal) &&
                           ranked.signalB >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealParentAnchorReplacementMinSignal) &&
                           ranked.parentDistanceBalance >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealParentAnchorReplacementMinParentDistanceBalance) &&
                           ranked.nearParentDist >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealParentAnchorReplacementMinNearParentDistance) &&
                           ranked.sep >=
                               ranked.minSep *
                                   std::max(
                                       0.0f,
                                       lumenConfig
                                           .fusionSplitPriorCleanTwoRealParentAnchorReplacementMinSeparationRadiusScale) &&
                           ranked.midpointDist <=
                               std::max(
                                   ranked.minSep,
                                   lumenConfig
                                       .fusionSplitPriorCleanTwoRealParentAnchorReplacementMaxMidpointDistance) &&
                           ranked.score <=
                               static_cast<double>(
                                   std::max(
                                       0.0f,
                                       lumenConfig
                                           .fusionSplitPriorCleanTwoRealParentAnchorReplacementMaxScore));
                };
            auto isParentAnchorYoungStrongLocalRescue =
                [&](const RankedSplitPrior &ranked) {
                    if (!lumenConfig
                             .fusionSplitPriorParentAnchorYoungStrongLocalRescueEnabled ||
                        !ranked.parentAnchored ||
                        !hasTolerableContinuationClaim(ranked) ||
                        ranked.windowBothDaughtersSupported <
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorParentAnchorYoungStrongLocalMinWindowBoth) ||
                        ranked.windowMissingDaughterCount >
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorParentAnchorYoungStrongLocalMaxWindowMissing) ||
                        ranked.windowParentPersists >
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorParentAnchorYoungStrongLocalMaxWindowParentPersists)) {
                        return false;
                    }
                    const bool realA =
                        isCurrentFrameLumenCandidateId(ranked.candidateA);
                    const bool realB =
                        isCurrentFrameLumenCandidateId(ranked.candidateB);
                    if (realA == realB) {
                        return false;
                    }
                    const int realVoxels = realA ? ranked.voxA : ranked.voxB;
                    const float realSignal =
                        realA ? ranked.signalA : ranked.signalB;
                    const auto stats = splitPriorAxisStats(ranked);
                    const bool parentWithinMaxShape =
                        lumenConfig
                                    .fusionSplitPriorParentAnchorYoungStrongLocalMaxParentShape <=
                                0.0f ||
                        ranked.parentShapeElongation <=
                            lumenConfig
                                .fusionSplitPriorParentAnchorYoungStrongLocalMaxParentShape;
                    return ranked.parentShapeElongation >=
                               std::max(
                                   1.0f,
                                   lumenConfig
                                       .fusionSplitPriorParentAnchorYoungStrongLocalMinParentShape) &&
                           parentWithinMaxShape &&
                           ranked.score <=
                               static_cast<double>(
                                   std::max(
                                       0.0f,
                                       lumenConfig
                                           .fusionSplitPriorParentAnchorYoungStrongLocalMaxScore)) &&
                           realVoxels >=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorParentAnchorYoungStrongLocalMinRealVoxels) &&
                           realSignal >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorParentAnchorYoungStrongLocalMinRealSignal) &&
                           ranked.farParentDist >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorParentAnchorYoungStrongLocalMinFarDistance) &&
                           stats.first >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorParentAnchorYoungStrongLocalMinLateralSeparation) &&
                           stats.second <=
                               std::clamp(
                                   lumenConfig
                                       .fusionSplitPriorParentAnchorYoungStrongLocalMaxZDominance,
                                   0.0f, 1.0f) &&
                           ranked.neighborClaimPenalty <= 1e-5f &&
                           ranked.continuationClaimSoftPenalty <= 1e-5f &&
                           ranked.parentPersistencePenalty <= 1e-5f;
                };
            auto isShortUnbalancedCleanWindowDuplicate =
                [&](const RankedSplitPrior &ranked) {
                    if (!lumenConfig
                             .fusionSplitPriorRejectShortUnbalancedCleanWindowDuplicate ||
                        ranked.parentAnchored ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateA) ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateB)) {
                        return false;
                    }
                    const auto axisStats = splitPriorAxisStats(ranked);
                    return ranked.windowBothDaughtersSupported >=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorShortUnbalancedCleanWindowMinWindowBoth) &&
                           ranked.windowMissingDaughterCount <=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorShortUnbalancedCleanWindowMaxWindowMissing) &&
                           ranked.windowParentPersists <=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorShortUnbalancedCleanWindowMaxWindowParentPersists) &&
                           ranked.balancedWindowBonus <=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorShortUnbalancedCleanWindowMaxBalancedBonus) &&
                           ranked.parentShapeElongation >=
                               std::max(
                                   1.0f,
                                   lumenConfig
                                       .fusionSplitPriorShortUnbalancedCleanWindowMinParentShape) &&
                           ranked.sep <=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorShortUnbalancedCleanWindowMaxSeparation) &&
                           ranked.parentDistanceBalance <=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorShortUnbalancedCleanWindowMaxParentDistanceBalance) &&
                           axisStats.second >=
                               std::clamp(
                                   lumenConfig
                                       .fusionSplitPriorShortUnbalancedCleanWindowMinZDominance,
                                   0.0f, 1.0f) &&
                           axisStats.second <=
                               std::clamp(
                                   lumenConfig
                                       .fusionSplitPriorShortUnbalancedCleanWindowMaxZDominance,
                                   0.0f, 1.0f) &&
                           ranked.rawScore <=
                               static_cast<double>(
                                   lumenConfig
                                       .fusionSplitPriorShortUnbalancedCleanWindowMaxRawScore) &&
                           ranked.neighborClaimPenalty <= 1e-5f &&
                           ranked.continuationClaimSoftPenalty <= 1e-5f &&
                           ranked.parentPersistencePenalty <= 1e-5f;
                };
            auto isParentAnchorCleanWindowLowZDuplicate =
                [&](const RankedSplitPrior &ranked) {
                    if (!lumenConfig
                             .fusionSplitPriorRejectParentAnchorCleanWindowLowZDuplicate ||
                        !ranked.parentAnchored) {
                        return false;
                    }
                    const bool realA =
                        isCurrentFrameLumenCandidateId(ranked.candidateA);
                    const bool realB =
                        isCurrentFrameLumenCandidateId(ranked.candidateB);
                    if (realA == realB) {
                        return false;
                    }
                    const auto axisStats = splitPriorAxisStats(ranked);
                    return ranked.windowBothDaughtersSupported >=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorParentAnchorCleanWindowLowZMinWindowBoth) &&
                           ranked.windowMissingDaughterCount <=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorParentAnchorCleanWindowLowZMaxWindowMissing) &&
                           ranked.windowParentPersists <=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorParentAnchorCleanWindowLowZMaxWindowParentPersists) &&
                           ranked.parentShapeElongation >=
                               std::max(
                                   1.0f,
                                   lumenConfig
                                       .fusionSplitPriorParentAnchorCleanWindowLowZMinParentShape) &&
                           ranked.sep <=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorParentAnchorCleanWindowLowZMaxSeparation) &&
                           axisStats.second <=
                               std::clamp(
                                   lumenConfig
                                       .fusionSplitPriorParentAnchorCleanWindowLowZMaxZDominance,
                                   0.0f, 1.0f) &&
                           ranked.score <=
                               static_cast<double>(
                                   lumenConfig
                                       .fusionSplitPriorParentAnchorCleanWindowLowZMaxScore) &&
                           ranked.neighborClaimPenalty <= 1e-5f &&
                           ranked.continuationClaimSoftPenalty <= 1e-5f &&
                           ranked.parentPersistencePenalty <= 1e-5f;
                };
            const bool earlyLargeSeparationRescueFrame =
                lumenConfig.fusionSplitPriorEarlyLargeSeparationRescueEnabled &&
                (!lumenConfig
                      .fusionSplitPriorEarlyLargeSeparationRescueFirstFrameOnly ||
                 frameIndex == 0) &&
                (lumenConfig.fusionSplitPriorEarlyLargeSeparationRescueMaxFrame < 0 ||
                 absoluteFrame <=
                     lumenConfig.fusionSplitPriorEarlyLargeSeparationRescueMaxFrame);
            auto isEarlyLargeSeparationWindowRescue =
                [&](const RankedSplitPrior &ranked) {
                    if (!earlyLargeSeparationRescueFrame || ranked.parentAnchored ||
                        !isRealLumenCandidateId(ranked.candidateA) ||
                        !isRealLumenCandidateId(ranked.candidateB)) {
                        return false;
                    }
                    if (!ranked.continuationClaimBlockers.empty()) {
                        return false;
                    }
                    const int minWindowBoth =
                        std::max(0, lumenConfig
                                      .fusionSplitPriorEarlyLargeSeparationMinWindowBoth);
                    const int maxWindowMissing =
                        std::max(0, lumenConfig
                                      .fusionSplitPriorEarlyLargeSeparationMaxWindowMissing);
                    const int maxParentPersists =
                        std::max(0, lumenConfig
                                      .fusionSplitPriorEarlyLargeSeparationMaxWindowParentPersists);
                    const auto stats = splitPriorAxisStats(ranked);
                    const int strictBalanceAfterFrame =
                        lumenConfig
                            .fusionSplitPriorEarlyLargeSeparationStrictBalanceAfterFrame;
                    if (strictBalanceAfterFrame >= 0 &&
                        absoluteFrame >= strictBalanceAfterFrame &&
                        ranked.parentDistanceBalance >
                            std::clamp(
                                lumenConfig
                                    .fusionSplitPriorEarlyLargeSeparationMaxParentDistanceBalanceAfterFrame,
                                0.0f, 1.0f)) {
                        return false;
                    }
                    return ranked.windowBothDaughtersSupported >= minWindowBoth &&
                           ranked.windowMissingDaughterCount <= maxWindowMissing &&
                           ranked.windowParentPersists <= maxParentPersists &&
                           ranked.parentShapeElongation >=
                               std::max(
                                   1.0f,
                                   lumenConfig
                                       .fusionSplitPriorEarlyLargeSeparationMinParentShape) &&
                           ranked.sep >=
                               std::max(0.0f,
                                        lumenConfig
                                            .fusionSplitPriorEarlyLargeSeparationMinSeparation) &&
                           stats.first >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorEarlyLargeSeparationMinLateralSeparation) &&
                           ranked.parentDistanceBalance >=
                               std::max(0.0f,
                                        lumenConfig
                                            .fusionSplitPriorEarlyLargeSeparationMinParentDistanceBalance) &&
                           ranked.nearParentDist >=
                               std::max(
                                   6.0f,
                                   lumenConfig
                                       .fusionSplitPriorMinDaughterParentDistance) &&
                           ranked.score <=
                               static_cast<double>(
                                   std::max(
                                       0.0f,
                                       lumenConfig
                                           .fusionSplitPriorEarlyLargeSeparationMaxScore));
                };
            auto isPartialWindowWideLateralRescue =
                [&](const RankedSplitPrior &ranked) {
                    if (!lumenConfig
                             .fusionSplitPriorPartialWindowWideLateralRescueEnabled ||
                        ranked.parentAnchored ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateA) ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateB) ||
                        !ranked.continuationClaimBlockers.empty()) {
                        return false;
                    }
                    const auto stats = splitPriorAxisStats(ranked);
                    // f052: CellLumen already had both real daughter centers,
                    // but the future window was only partial because the old
                    // parent still persisted once. Let this through only for
                    // wide lateral, strong-signal, elongated-parent splits so
                    // z-column duplicates and weak noisy pairs remain blocked.
                    return ranked.windowBothDaughtersSupported >=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorPartialWindowWideLateralMinWindowBoth) &&
                           ranked.windowMissingDaughterCount <=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorPartialWindowWideLateralMaxWindowMissing) &&
                           ranked.windowParentPersists <=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorPartialWindowWideLateralMaxParentPersists) &&
                           ranked.parentShapeElongation >=
                               std::max(
                                   1.0f,
                                   lumenConfig
                                       .fusionSplitPriorPartialWindowWideLateralMinParentShape) &&
                           ranked.sep >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorPartialWindowWideLateralMinSeparation) &&
                           stats.first >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorPartialWindowWideLateralMinLateralSeparation) &&
                           stats.second <=
                               std::clamp(
                                   lumenConfig
                                       .fusionSplitPriorPartialWindowWideLateralMaxZDominance,
                                   0.0f, 1.0f) &&
                           ranked.midpointDist <=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorPartialWindowWideLateralMaxMidpointDistance) &&
                           ranked.parentDistanceBalance >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorPartialWindowWideLateralMinParentDistanceBalance) &&
                           ranked.voxA >=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorPartialWindowWideLateralMinVoxels) &&
                           ranked.voxB >=
                               std::max(
                                   0,
                                   lumenConfig
                                       .fusionSplitPriorPartialWindowWideLateralMinVoxels) &&
                           ranked.signalA >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorPartialWindowWideLateralMinSignal) &&
                           ranked.signalB >=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorPartialWindowWideLateralMinSignal) &&
                           ranked.score <=
                               static_cast<double>(
                                   std::max(
                                       0.0f,
                                       lumenConfig
                                           .fusionSplitPriorPartialWindowWideLateralMaxScore)) &&
                           ranked.rankingSoftPenalty <=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorPartialWindowWideLateralMaxRankingSoftPenalty) &&
                           ranked.neighborClaimPenalty <=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorPartialWindowWideLateralMaxNeighborClaimPenalty) &&
                           ranked.continuationClaimSoftPenalty <=
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorPartialWindowWideLateralMaxContinuationClaimPenalty);
                };
            auto isPartialWindowZReplacementRescue =
                [&](const RankedSplitPrior &ranked) {
                    if (!lumenConfig
                             .fusionSplitPriorPartialWindowZReplacementRescueEnabled ||
                        ranked.parentAnchored ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateA) ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateB) ||
                        !ranked.continuationClaimBlockers.empty()) {
                        return false;
                    }
                    const auto stats = splitPriorAxisStats(ranked);
                    if (ranked.windowBothDaughtersSupported <
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorPartialWindowWideLateralMinWindowBoth) ||
                        ranked.windowMissingDaughterCount >
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorPartialWindowWideLateralMaxWindowMissing) ||
                        ranked.windowParentPersists >
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorPartialWindowWideLateralMaxParentPersists) ||
                        ranked.parentShapeElongation <
                            std::max(
                                1.0f,
                                lumenConfig
                                    .fusionSplitPriorPartialWindowZReplacementMinParentShape) ||
                        ranked.sep <
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorPartialWindowZReplacementMinSeparation) ||
                        stats.first <
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorPartialWindowZReplacementMinLateralSeparation) ||
                        stats.second >
                            std::clamp(
                                lumenConfig
                                    .fusionSplitPriorPartialWindowZReplacementMaxZDominance,
                                0.0f, 1.0f) ||
                        ranked.midpointDist >
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorPartialWindowZReplacementMaxMidpointDistance) ||
                        ranked.parentDistanceBalance <
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorPartialWindowZReplacementMinParentDistanceBalance) ||
                        ranked.voxA <
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorPartialWindowZReplacementMinVoxels) ||
                        ranked.voxB <
                            std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorPartialWindowZReplacementMinVoxels) ||
                        ranked.signalA <
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorPartialWindowZReplacementMinSignal) ||
                        ranked.signalB <
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorPartialWindowZReplacementMinSignal) ||
                        ranked.score >
                            static_cast<double>(
                                std::max(
                                    0.0f,
                                    lumenConfig
                                        .fusionSplitPriorPartialWindowZReplacementMaxScore)) ||
                        ranked.rankingSoftPenalty >
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorPartialWindowWideLateralMaxRankingSoftPenalty) ||
                        ranked.neighborClaimPenalty >
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorPartialWindowWideLateralMaxNeighborClaimPenalty) ||
                        ranked.continuationClaimSoftPenalty >
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorPartialWindowWideLateralMaxContinuationClaimPenalty)) {
                        return false;
                    }
                    auto sharesExactlyOneCandidate =
                        [](const RankedSplitPrior &a,
                           const RankedSplitPrior &b) {
                            const bool shareA =
                                a.candidateA == b.candidateA ||
                                a.candidateA == b.candidateB;
                            const bool shareB =
                                a.candidateB == b.candidateA ||
                                a.candidateB == b.candidateB;
                            return shareA != shareB;
                        };
                    auto nonSharedCandidateId =
                        [](const RankedSplitPrior &candidate,
                           const RankedSplitPrior &other) {
                            const bool aShared =
                                candidate.candidateA == other.candidateA ||
                                candidate.candidateA == other.candidateB;
                            return aShared ? candidate.candidateB
                                           : candidate.candidateA;
                        };
                    auto candidatePosition =
                        [](const RankedSplitPrior &candidate, int id) {
                            return candidate.candidateA == id
                                       ? candidate.proposal.d1Pos
                                       : candidate.proposal.d2Pos;
                        };
                    auto candidateVoxels =
                        [](const RankedSplitPrior &candidate, int id) {
                            return candidate.candidateA == id
                                       ? candidate.voxA
                                       : candidate.voxB;
                        };
                    const float minZSeparation =
                        std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorPartialWindowZReplacementMinZSeparation);
                    const float minVoxelRatio =
                        std::max(
                            1.0f,
                            lumenConfig
                                .fusionSplitPriorPartialWindowZReplacementMinVoxelRatio);
                    for (const RankedSplitPrior &other : rankedPriors) {
                        if (&other == &ranked ||
                            other.parentIdx != ranked.parentIdx ||
                            other.parentAnchored ||
                            !isCurrentFrameLumenCandidateId(other.candidateA) ||
                            !isCurrentFrameLumenCandidateId(other.candidateB) ||
                            !hasCleanWindowSupport(other) ||
                            other.balancedWindowBonus <= 0.0f ||
                            !sharesExactlyOneCandidate(ranked, other) ||
                            !isWeakBalancedCleanWindowPair(other)) {
                            continue;
                        }
                        const int replacementId =
                            nonSharedCandidateId(ranked, other);
                        const int blockedId =
                            nonSharedCandidateId(other, ranked);
                        const cv::Point3f replacementPos =
                            candidatePosition(ranked, replacementId);
                        const cv::Point3f blockedPos =
                            candidatePosition(other, blockedId);
                        const int replacementVoxels =
                            candidateVoxels(ranked, replacementId);
                        const int blockedVoxels =
                            std::max(1, candidateVoxels(other, blockedId));
                        const float zSeparation =
                            std::abs(replacementPos.z - blockedPos.z);
                        if (zSeparation >= minZSeparation &&
                            static_cast<float>(replacementVoxels) >=
                                static_cast<float>(blockedVoxels) *
                                    minVoxelRatio) {
                            return true;
                        }
                    }
                    return false;
                };
            auto isTemporalRepairWindowRescue =
                [&](const RankedSplitPrior &ranked) {
                    if (!ranked.temporalRepairEligible ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateA) ||
                        !isCurrentFrameLumenCandidateId(ranked.candidateB)) {
                        return false;
                    }
                    const bool strongTemporalCatchEvidence =
                        ranked.temporalRepairCatchExpanded &&
                        hasStrongTemporalDaughterEvidence(ranked);
	                    if (!hasCleanWindowSupport(ranked) ||
	                        !hasTolerableContinuationClaim(ranked) ||
	                        ranked.neighborClaimPenalty >
	                            std::max(8.0f,
	                                     lumenConfig
	                                         .fusionSplitPriorMaxNeighborClaimPenalty) ||
	                        (ranked.rankingSoftPenalty > 12.0f &&
	                         !strongTemporalCatchEvidence) ||
	                        ranked.parentDistanceBalance <
	                            (hasStrongTemporalDaughterEvidence(ranked)
	                                 ? std::max(
	                                       0.0f,
	                                       lumenConfig
	                                           .fusionSplitPriorTemporalRepairStrongMinParentDistanceBalance)
	                                 : std::max(
	                                       0.25f,
	                                       lumenConfig
	                                           .fusionSplitPriorMinParentDistanceBalance *
	                                           0.65f)) ||
	                        ranked.nearParentDist <
	                            std::max(6.0f,
	                                     lumenConfig
	                                         .fusionSplitPriorMinDaughterParentDistance)) {
                        return false;
                    }
                    return ranked.score <=
                           static_cast<double>(
                               std::max(
                                   0.0f,
                                   lumenConfig
                                       .fusionSplitPriorTemporalRepairMaxScore));
                };
            auto isSelectableRankedPrior = [&](const RankedSplitPrior &ranked) {
                const int minParentAge = std::max(
                    0,
                    lumenConfig.fusionSplitPriorMinParentAgeFrames);
                const bool youngStrongLocalRescue =
                    isParentAnchorYoungStrongLocalRescue(ranked);
                if (isLowShapeZDominantPair(ranked) ||
                    isCleanWindowSeedColumnZStackPair(ranked) ||
                    isParentAnchoredSeedColumnDuplicate(ranked) ||
                    isFutureOnlyParentAnchorPositiveRaw(ranked) ||
                    isWeakParentAnchoredOneReal(ranked) ||
                    isTightCleanWindowInternalPair(ranked) ||
                    isWeakBalancedCleanWindowPair(ranked) ||
                    isWeakAsymmetricCleanWindowPair(ranked) ||
                    isTriaxialNoBalancedCleanWindowPair(ranked) ||
                    isShortUnbalancedCleanWindowDuplicate(ranked) ||
                    isParentAnchorCleanWindowLowZDuplicate(ranked)) {
                    return false;
                }
                if (minParentAge > 0 &&
                    parentAgeFramesForRanked(ranked) < minParentAge &&
                    !youngStrongLocalRescue) {
                    return false;
                }
                if (ranked.temporalRepairEligible &&
                    !temporalRepairParentAgeOk(ranked)) {
                    return false;
                }
                if (ranked.parentAnchored) {
                    return youngStrongLocalRescue ||
                           isStrongParentAnchoredOneSided(ranked) ||
                           isCompactParentAnchoredWindowRescue(ranked) ||
                           isParentAnchoredSingleBlockerWindowRescue(ranked) ||
                           isTemporalCatchParentAnchoredPartialWindowRescue(ranked) ||
                           isPartialParentAnchoredWindowRescue(ranked);
                }
                if (isCleanTwoRealSingleBlockerRescue(ranked)) {
                    return true;
                }
                if (lumenConfig
                        .fusionSplitPriorCleanTwoRealRescueBeforeDefaultNegativeGate &&
                    (isCleanTwoRealAsymmetricRescue(ranked) ||
                     isCleanTwoRealHighNeighborClaimRescue(ranked))) {
                    return true;
                }
                if (isWeakAsymmetricElongatedRescue(ranked)) {
                    return false;
                }
                if (isPartialWindowWideLateralRescue(ranked)) {
                    return true;
                }
                if (isPartialWindowZReplacementRescue(ranked)) {
                    return true;
                }
                if (isCleanTwoRealParentAnchorReplacementRescue(ranked)) {
                    return true;
                }
                if (ranked.score <= 0.0) {
                    if (!ranked.continuationClaimBlockers.empty() &&
                        !isFutureContinuationConflictRescue(ranked)) {
                        return false;
                    }
                    return hasCleanWindowSupport(ranked) &&
                           ranked.parentDistanceBalance >=
                               std::max(0.0f,
                                        lumenConfig.fusionSplitPriorMinParentDistanceBalance) &&
                           ranked.nearParentDist >=
                               std::max(6.0f,
                                        lumenConfig.fusionSplitPriorMinDaughterParentDistance);
                }
	                if (isFutureContinuationConflictRescue(ranked)) {
	                    return true;
	                }
	                if (isCleanTwoRealWindowPair(ranked)) {
	                    return true;
	                }
                if (isCleanTwoRealSingleBlockerRescue(ranked)) {
                    return true;
                }
                if (isCleanTwoRealAsymmetricRescue(ranked)) {
                    return true;
                }
                if (isCleanTwoRealCompactRescue(ranked)) {
                    return true;
                }
                if (isCleanTwoRealParentAnchorReplacementRescue(ranked)) {
                    return true;
                }
                if (isCleanTwoRealHighNeighborClaimRescue(ranked)) {
                    return true;
                }
                if (isTemporalRepairWindowRescue(ranked)) {
                    return true;
                }
                if (isEarlyLargeSeparationWindowRescue(ranked)) {
                    return true;
                }
	                if (hasCleanWindowSupport(ranked) &&
	                    hasTolerableContinuationClaim(ranked) &&
	                    ranked.score <= maxSelectableCost) {
                    return true;
                }
                return isWindowBackedConflictResolution(ranked);
            };
            std::set<int> protectedTwoRealCandidateIds;
            if (lumenConfig
                    .fusionSplitPriorParentAnchorOneRealSharedCandidatePenalty >
                0.0f) {
                for (const RankedSplitPrior &ranked : rankedPriors) {
                    const bool bothReal =
                        isCurrentFrameLumenCandidateId(ranked.candidateA) &&
                        isCurrentFrameLumenCandidateId(ranked.candidateB);
                    const bool cleanWindowTwoReal =
                        bothReal &&
                        !ranked.parentAnchored &&
                        hasCleanWindowSupport(ranked) &&
                        ranked.continuationClaimBlockers.empty() &&
                        ranked.parentPersistencePenalty <= 1e-5f &&
                        ranked.continuationClaimSoftPenalty <= 1e-5f &&
                        ranked.score <= maxSelectableCost &&
                        ranked.parentDistanceBalance >=
                            std::max(0.0f,
                                     lumenConfig
                                         .fusionSplitPriorMinParentDistanceBalance) &&
                        ranked.nearParentDist >=
                            std::max(6.0f,
                                     lumenConfig
                                         .fusionSplitPriorMinDaughterParentDistance);
                    const double windowPairSelectionBonus =
                        std::min(18.0, std::max(0.0, maxSelectableCost * 0.36));
                    const bool windowPairWouldImproveSelection =
                        isCleanTwoRealWindowPair(ranked) &&
                        ranked.score - windowPairSelectionBonus <= 0.0;
                    if ((cleanWindowTwoReal && ranked.score <= 0.0) ||
                        isCleanTwoRealSingleBlockerRescue(ranked) ||
                        isCleanTwoRealAsymmetricRescue(ranked) ||
                        isCleanTwoRealCompactRescue(ranked) ||
                        isCleanTwoRealParentAnchorReplacementRescue(ranked) ||
                        isCleanTwoRealHighNeighborClaimRescue(ranked) ||
                        isPartialWindowWideLateralRescue(ranked) ||
                        isPartialWindowZReplacementRescue(ranked) ||
                        windowPairWouldImproveSelection) {
                        protectedTwoRealCandidateIds.insert(ranked.candidateA);
                        protectedTwoRealCandidateIds.insert(ranked.candidateB);
                    }
                }
            }
            std::set<size_t> zDominantTieBreakDemotedIndexes;
            auto selectionCost = [&](const RankedSplitPrior &ranked) {
                const size_t rankedIndex =
                    static_cast<size_t>(&ranked - rankedPriors.data());
                const double zTieBreakSelectionPenalty =
                    zDominantTieBreakDemotedIndexes.count(rankedIndex) > 0
                        ? std::max(0.25,
                                   static_cast<double>(
                                       zDominantTieBreakMargin))
                        : 0.0;
                auto withZTieBreakPenalty = [&](double baseCost) {
                    return baseCost + zTieBreakSelectionPenalty;
                };
                if (ranked.parentAnchored) {
                    const bool realA = isRealLumenCandidateId(ranked.candidateA);
                    const bool realB = isRealLumenCandidateId(ranked.candidateB);
                    const int realCandidateId =
                        realA != realB ? (realA ? ranked.candidateA
                                                : ranked.candidateB)
                                       : -1;
                    const double sharedTwoRealPenalty =
                        realCandidateId >= 0 &&
                                protectedTwoRealCandidateIds.count(
                                    realCandidateId) > 0
                            ? static_cast<double>(
                                  std::max(
                                      0.0f,
                                      lumenConfig
                                          .fusionSplitPriorParentAnchorOneRealSharedCandidatePenalty))
                            : 0.0;
                    if (isPartialParentAnchoredWindowRescue(ranked) &&
                        ranked.temporalRepairCatchExpanded) {
                        // Temporal-catch parent-anchor pairs are sparse-frame
                        // repairs. They can also satisfy the generic strong
                        // one-sided test, so apply the rescue bonus before the
                        // generic branch or the global selector will skip them
                        // as mildly positive-cost splits.
                        return withZTieBreakPenalty(ranked.score - 55.0 +
                                                   sharedTwoRealPenalty);
                    }
                    if (isParentAnchorYoungStrongLocalRescue(ranked)) {
                        return withZTieBreakPenalty(
                            ranked.score -
                                std::min(
                                    120.0,
                                    static_cast<double>(
                                        std::max(
                                            0.0f,
                                            lumenConfig
                                                .fusionSplitPriorParentAnchorYoungStrongLocalSelectionBonus))) +
                            sharedTwoRealPenalty);
                    }
                    if (isStrongParentAnchoredOneSided(ranked)) {
                        return withZTieBreakPenalty(ranked.score +
                                                   sharedTwoRealPenalty);
                    }
                    if (isCompactParentAnchoredWindowRescue(ranked)) {
                        return withZTieBreakPenalty(
                            ranked.score -
                            std::min(
                                40.0,
                                std::max(0.0, maxSelectableCost * 0.80)) +
                            sharedTwoRealPenalty);
                    }
                    if (isParentAnchoredSingleBlockerWindowRescue(ranked)) {
                        return withZTieBreakPenalty(
                            ranked.score -
                            std::min(
                                120.0,
                                static_cast<double>(
                                    std::max(
                                        0.0f,
                                        lumenConfig
                                            .fusionSplitPriorParentAnchorSingleBlockerSelectionBonus))) +
                            sharedTwoRealPenalty);
                    }
                    if (isTemporalCatchParentAnchoredPartialWindowRescue(ranked)) {
                        return withZTieBreakPenalty(
                            ranked.score -
                            std::min(
                                120.0,
                                static_cast<double>(
                                    std::max(
                                        0.0f,
                                        lumenConfig
                                            .fusionSplitPriorTemporalCatchParentAnchorPartialWindowSelectionBonus))) +
                            sharedTwoRealPenalty);
                    }
                    if (isPartialParentAnchoredWindowRescue(ranked)) {
                        return withZTieBreakPenalty(ranked.score - 55.0 +
                                                   sharedTwoRealPenalty);
                    }
                    return withZTieBreakPenalty(sharedTwoRealPenalty);
                }
	                if (isFutureContinuationConflictRescue(ranked)) {
	                    return withZTieBreakPenalty(
	                        ranked.score -
	                        std::min(
	                            60.0,
	                            static_cast<double>(
	                                ranked.continuationClaimSoftPenalty) +
	                                35.0));
	                }
                if (isCleanTwoRealSingleBlockerRescue(ranked)) {
                    return withZTieBreakPenalty(
                        ranked.score -
                        std::min(
                            80.0,
                            static_cast<double>(
                                std::max(
                                    0.0f,
                                        lumenConfig
                                        .fusionSplitPriorCleanTwoRealSingleBlockerSelectionBonus))));
                }
                if (isCleanTwoRealAsymmetricRescue(ranked)) {
                    return withZTieBreakPenalty(
                        ranked.score -
                        std::min(
                            120.0,
                            static_cast<double>(
                                std::max(
                                    0.0f,
                                    lumenConfig
                                        .fusionSplitPriorCleanTwoRealAsymmetricSelectionBonus))));
                }
                if (isCleanTwoRealCompactRescue(ranked)) {
                    return withZTieBreakPenalty(
                        ranked.score -
                        std::min(
                            120.0,
                            static_cast<double>(
                                std::max(
                                    0.0f,
                                lumenConfig
                                    .fusionSplitPriorCleanTwoRealCompactSelectionBonus))));
                }
                if (isCleanTwoRealParentAnchorReplacementRescue(ranked)) {
                    return withZTieBreakPenalty(
                        ranked.score -
                        std::min(
                            120.0,
                            static_cast<double>(
                                std::max(
                                    0.0f,
                                    lumenConfig
                                        .fusionSplitPriorCleanTwoRealParentAnchorReplacementSelectionBonus))));
                }
                if (isCleanTwoRealHighNeighborClaimRescue(ranked)) {
                    return withZTieBreakPenalty(
                        ranked.score -
                        std::min(
                            140.0,
                            static_cast<double>(
                                std::max(
                                    0.0f,
                                    lumenConfig
                                        .fusionSplitPriorCleanTwoRealHighNeighborClaimSelectionBonus))));
                }
                if (isTemporalRepairWindowRescue(ranked)) {
                    return withZTieBreakPenalty(
                        ranked.score -
                        std::min(
                            120.0,
                            static_cast<double>(
                                std::max(
                                    0.0f,
                                    lumenConfig
                                        .fusionSplitPriorTemporalRepairSelectionBonus))));
                }
                if (isPartialWindowWideLateralRescue(ranked)) {
                    return withZTieBreakPenalty(
                        ranked.score -
                        std::min(
                            120.0,
                            static_cast<double>(
                                std::max(
                                    0.0f,
                                    lumenConfig
                                        .fusionSplitPriorPartialWindowWideLateralSelectionBonus))));
                }
                if (isPartialWindowZReplacementRescue(ranked)) {
                    return withZTieBreakPenalty(
                        ranked.score -
                        std::min(
                            120.0,
                            static_cast<double>(
                                std::max(
                                    0.0f,
                                    lumenConfig
                                        .fusionSplitPriorPartialWindowZReplacementSelectionBonus))));
                }
	                if (isCleanTwoRealWindowPair(ranked)) {
	                    return withZTieBreakPenalty(
	                        ranked.score -
	                        std::min(18.0, std::max(0.0, maxSelectableCost * 0.36)));
	                }
                if (isEarlyLargeSeparationWindowRescue(ranked)) {
                    const double rescueMaxScore =
                        static_cast<double>(
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorEarlyLargeSeparationMaxScore));
                    return withZTieBreakPenalty(
                        ranked.score - std::min(95.0, rescueMaxScore + 5.0));
                }
                if (isWindowBackedConflictResolution(ranked)) {
                    return withZTieBreakPenalty(
                        ranked.score -
                        static_cast<double>(
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorWindowBackedConflictSelectionBonus)));
                }
                if (ranked.score <= 0.0) {
                    return withZTieBreakPenalty(ranked.score);
                }
	                const bool elongatedCleanWindowRescue =
	                    hasCleanWindowSupport(ranked) &&
	                    hasTolerableContinuationClaim(ranked) &&
                    ranked.parentShapeElongation >=
                        std::max(
                            0.0f,
                            lumenConfig.fusionSplitPriorElongatedParentMinShape) &&
                    ranked.parentDistanceBalance >=
                        std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorWindowBalancedMinParentDistanceBalance) &&
                    ranked.score <= maxSelectableCost;
                if (elongatedCleanWindowRescue) {
                    return withZTieBreakPenalty(
                        ranked.score -
                        std::min(30.0, std::max(0.0, maxSelectableCost * 0.60)));
                }
                if (hasCleanWindowSupport(ranked) &&
                    hasTolerableContinuationClaim(ranked)) {
                    return withZTieBreakPenalty(ranked.score);
                }
                return withZTieBreakPenalty(0.0);
            };
	            for (size_t groupIdx = 0; groupIdx < parentGroups.size(); ++groupIdx) {
	                for (const size_t rankedIdx : parentGroups[groupIdx].rankedIndexes) {
	                    const RankedSplitPrior &ranked = rankedPriors[rankedIdx];
	                    const bool selectable = isSelectableRankedPrior(ranked);
	                    if (selectable) {
	                        viableRankedIndexesByGroup[groupIdx].push_back(rankedIdx);
	                    }
                }
            }
            for (const auto &group : parentGroups) {
                for (const size_t rankedIdx : group.rankedIndexes) {
                    const RankedSplitPrior &ranked = rankedPriors[rankedIdx];
                    if (!isSelectableRankedPrior(ranked) ||
                        !isZDominantTieBreakCandidate(ranked)) {
                        continue;
                    }
                    for (const size_t altIdx : group.rankedIndexes) {
                        if (altIdx == rankedIdx) {
                            continue;
                        }
                        const RankedSplitPrior &alt = rankedPriors[altIdx];
                        if (!isSelectableRankedPrior(alt) ||
                            isZDominantTieBreakCandidate(alt)) {
                            continue;
                        }
                        if (std::abs(alt.score - ranked.score) <=
                                static_cast<double>(zDominantTieBreakMargin) &&
                            preferRankedSplitPrior(alt, ranked)) {
                            zDominantTieBreakDemotedIndexes.insert(rankedIdx);
                            break;
                        }
                    }
                }
            }
            std::vector<double> suffixBestPossible(parentGroups.size() + 1, 0.0);
            for (int groupIdx = static_cast<int>(parentGroups.size()) - 1; groupIdx >= 0; --groupIdx) {
                double bestGroupScore = 0.0;
                for (const size_t rankedIdx : viableRankedIndexesByGroup[static_cast<size_t>(groupIdx)]) {
                    bestGroupScore = std::min(
                        bestGroupScore,
                        selectionCost(rankedPriors[rankedIdx]));
                }
                suffixBestPossible[static_cast<size_t>(groupIdx)] =
                    suffixBestPossible[static_cast<size_t>(groupIdx + 1)] + bestGroupScore;
            }
            const size_t maxSearchNodes = 1000000;

            auto updateBest = [&](int count, double score) {
                if (score < bestScore - 1e-9 ||
                    (std::abs(score - bestScore) <= 1e-9 && count > bestCount)) {
                    bestCount = count;
                    bestScore = score;
                    bestSelection = activeSelection;
                }
            };

            std::function<void(size_t, int, double)> search =
                [&](size_t groupIdx, int count, double score) {
                    if (searchTruncated) {
                        return;
                    }
                    ++searchNodes;
                    if (searchNodes > maxSearchNodes) {
                        searchTruncated = true;
                        return;
                    }
                    if (score + suffixBestPossible[groupIdx] > bestScore + 1e-9) {
                        return;
                    }
                    if (count >= finiteMaxPriors || groupIdx >= parentGroups.size()) {
                        updateBest(count, score);
                        return;
                    }

                    for (const size_t rankedIdx : viableRankedIndexesByGroup[groupIdx]) {
                        const RankedSplitPrior &ranked = rankedPriors[rankedIdx];
                        if (activeCandidateIds.count(ranked.candidateA) > 0 ||
                            activeCandidateIds.count(ranked.candidateB) > 0) {
                            continue;
                        }
                        activeCandidateIds.insert(ranked.candidateA);
                        activeCandidateIds.insert(ranked.candidateB);
                        activeParentIndexes.insert(ranked.parentIdx);
                        activeSelection.push_back(rankedIdx);
                        search(groupIdx + 1, count + 1, score + selectionCost(ranked));
                        activeSelection.pop_back();
                        activeParentIndexes.erase(ranked.parentIdx);
                        activeCandidateIds.erase(ranked.candidateA);
                        activeCandidateIds.erase(ranked.candidateB);
                    }

                    search(groupIdx + 1, count, score);
                };

            search(0, 0, 0.0);
            selectedRankedPriorIndexes.insert(bestSelection.begin(), bestSelection.end());
            for (const size_t rankedIdx : selectedRankedPriorIndexes) {
                if (rankedIdx >= rankedPriors.size()) {
                    continue;
                }
                const RankedSplitPrior &ranked = rankedPriors[rankedIdx];
                const bool twoRealAnchorReplacement =
                    isCleanTwoRealParentAnchorReplacementRescue(ranked);
                const bool youngParentAnchor =
                    isParentAnchorYoungStrongLocalRescue(ranked);
                if (!twoRealAnchorReplacement && !youngParentAnchor) {
                    continue;
                }
                std::cout << "[CellLumen Fusion SplitPrior SelectionTag] frame="
                          << absoluteFrame
                          << " parent="
                          << frame.cells[ranked.parentIdx].getName()
                          << " twoRealParentAnchorReplacement="
                          << twoRealAnchorReplacement
                          << " youngParentAnchorStrongLocal="
                          << youngParentAnchor
                          << " parentAge="
                          << parentAgeFramesForRanked(ranked)
                          << " score=" << ranked.score
                          << " sep=" << ranked.sep
                          << " pairLateralSep="
                          << splitPriorAxisStats(ranked).first
                          << " pairZDominance="
                          << splitPriorAxisStats(ranked).second
                          << " windowBoth="
                          << ranked.windowBothDaughtersSupported
                          << " windowMissing="
                          << ranked.windowMissingDaughterCount
                          << " windowParentPersists="
                          << ranked.windowParentPersists
                          << " parentDistBalance="
                          << ranked.parentDistanceBalance
                          << " vox=(" << ranked.voxA << ","
                          << ranked.voxB << ")"
                          << " signal=(" << ranked.signalA << ","
                          << ranked.signalB << ")"
                          << " candidateIds=(" << ranked.candidateA
                          << "," << ranked.candidateB << ")"
                          << std::endl;
            }
        }

	        std::map<size_t, size_t> promotedRankedPriorIndexes;
	        std::map<size_t, int> promotedCandidateIdBySource;
	        std::map<size_t, float> promotedZShiftBySource;
	        std::set<size_t> promotedRankedPriorTargets;
        if (lumenConfig.fusionSplitPriorZStackDaughterPromotionEnabled &&
            !selectedRankedPriorIndexes.empty()) {
            std::set<int> selectedCandidateIds;
            for (const size_t rankedIdx : selectedRankedPriorIndexes) {
                if (rankedIdx >= rankedPriors.size()) {
                    continue;
                }
                selectedCandidateIds.insert(rankedPriors[rankedIdx].candidateA);
                selectedCandidateIds.insert(rankedPriors[rankedIdx].candidateB);
            }
            auto candidatePositionInRanked =
                [](const RankedSplitPrior &ranked, int candidateId,
                   cv::Point3f &positionOut) {
                    if (ranked.candidateA == candidateId) {
                        positionOut = ranked.proposal.d1Pos;
                        return true;
                    }
                    if (ranked.candidateB == candidateId) {
                        positionOut = ranked.proposal.d2Pos;
                        return true;
                    }
                    return false;
                };
            auto candidateVoxelsInRanked =
                [](const RankedSplitPrior &ranked, int candidateId) {
                    if (ranked.candidateA == candidateId) {
                        return ranked.voxA;
                    }
                    if (ranked.candidateB == candidateId) {
                        return ranked.voxB;
                    }
                    return 0;
                };
            auto candidateSignalInRanked =
                [](const RankedSplitPrior &ranked, int candidateId) {
                    if (ranked.candidateA == candidateId) {
                        return ranked.signalA;
                    }
                    if (ranked.candidateB == candidateId) {
                        return ranked.signalB;
                    }
                    return 0.0f;
                };
            auto isCurrentFramePromotionCandidateId = [](int candidateId) {
                return candidateId >= 0 && candidateId < 1000000;
            };
            const float maxPromotionLateral = std::max(
                0.0f,
                lumenConfig.fusionSplitPriorZStackDaughterPromotionMaxLateral);
            const float minPromotionZShift = std::max(
                0.0f,
                lumenConfig.fusionSplitPriorZStackDaughterPromotionMinZShift);
            const float maxPromotionScoreDelta = std::max(
                0.0f,
                lumenConfig
                    .fusionSplitPriorZStackDaughterPromotionMaxScoreDelta);
            const int minPromotionVoxels = std::max(
                0,
                lumenConfig.fusionSplitPriorZStackDaughterPromotionMinVoxels);
            const float minPromotionSignal = std::max(
                0.0f,
                lumenConfig.fusionSplitPriorZStackDaughterPromotionMinSignal);
            const int minPromotionWindowBoth = std::max(
                0,
                lumenConfig
                    .fusionSplitPriorZStackDaughterPromotionMinWindowBoth);
            const int maxPromotionWindowMissing = std::max(
                0,
                lumenConfig
                    .fusionSplitPriorZStackDaughterPromotionMaxWindowMissing);
            const int maxPromotionParentPersists = std::max(
                0,
                lumenConfig
                    .fusionSplitPriorZStackDaughterPromotionMaxWindowParentPersists);

            for (const size_t selectedIdx : selectedRankedPriorIndexes) {
                if (selectedIdx >= rankedPriors.size()) {
                    continue;
                }
                const RankedSplitPrior &selected = rankedPriors[selectedIdx];
                if (selected.parentAnchored ||
                    !isCurrentFramePromotionCandidateId(selected.candidateA) ||
                    !isCurrentFramePromotionCandidateId(selected.candidateB)) {
                    continue;
                }
                size_t bestAltIdx = std::numeric_limits<size_t>::max();
                int bestSelectedOtherId = -1;
                int bestAltOtherId = -1;
                float bestZShift = 0.0f;
                float bestLateral = 0.0f;
                double bestAltScore = std::numeric_limits<double>::infinity();
                for (size_t altIdx = 0; altIdx < rankedPriors.size(); ++altIdx) {
                    if (altIdx == selectedIdx) {
                        continue;
                    }
                    const RankedSplitPrior &alt = rankedPriors[altIdx];
                    if (alt.parentIdx != selected.parentIdx ||
                        alt.parentAnchored ||
                        !isCurrentFramePromotionCandidateId(alt.candidateA) ||
                        !isCurrentFramePromotionCandidateId(alt.candidateB) ||
                        alt.windowBothDaughtersSupported < minPromotionWindowBoth ||
                        alt.windowMissingDaughterCount > maxPromotionWindowMissing ||
                        alt.windowParentPersists > maxPromotionParentPersists ||
                        alt.neighborClaimPenalty > 1e-5f ||
                        alt.continuationClaimSoftPenalty > 1e-5f ||
                        alt.score >
                            selected.score +
                                static_cast<double>(maxPromotionScoreDelta)) {
                        continue;
                    }

                    int selectedOtherId = -1;
                    int altOtherId = -1;
                    if (selected.candidateA == alt.candidateA) {
                        selectedOtherId = selected.candidateB;
                        altOtherId = alt.candidateB;
                    } else if (selected.candidateA == alt.candidateB) {
                        selectedOtherId = selected.candidateB;
                        altOtherId = alt.candidateA;
                    } else if (selected.candidateB == alt.candidateA) {
                        selectedOtherId = selected.candidateA;
                        altOtherId = alt.candidateB;
                    } else if (selected.candidateB == alt.candidateB) {
                        selectedOtherId = selected.candidateA;
                        altOtherId = alt.candidateA;
                    } else {
                        continue;
                    }
                    if (!isCurrentFramePromotionCandidateId(selectedOtherId) ||
                        !isCurrentFramePromotionCandidateId(altOtherId) ||
                        selectedCandidateIds.count(altOtherId) > 0) {
                        continue;
                    }
                    if (candidateVoxelsInRanked(alt, altOtherId) <
                            minPromotionVoxels ||
                        candidateSignalInRanked(alt, altOtherId) <
                            minPromotionSignal) {
                        continue;
                    }
                    cv::Point3f selectedOtherPos;
                    cv::Point3f altOtherPos;
                    if (!candidatePositionInRanked(selected, selectedOtherId,
                                                   selectedOtherPos) ||
                        !candidatePositionInRanked(alt, altOtherId,
                                                   altOtherPos)) {
                        continue;
                    }
                    const cv::Point3f delta = altOtherPos - selectedOtherPos;
                    const float lateral =
                        std::sqrt(delta.x * delta.x + delta.y * delta.y);
                    const float zShift = delta.z;
                    if (lateral > maxPromotionLateral ||
                        std::abs(zShift) < minPromotionZShift ||
                        (lumenConfig
                             .fusionSplitPriorZStackDaughterPromotionPositiveOnly &&
                         zShift <= 0.0f)) {
                        continue;
                    }
                    if (bestAltIdx == std::numeric_limits<size_t>::max() ||
                        alt.score < bestAltScore - 1e-9 ||
                        (std::abs(alt.score - bestAltScore) <= 1e-9 &&
                         std::abs(zShift) > std::abs(bestZShift))) {
                        bestAltIdx = altIdx;
                        bestSelectedOtherId = selectedOtherId;
                        bestAltOtherId = altOtherId;
                        bestZShift = zShift;
                        bestLateral = lateral;
                        bestAltScore = alt.score;
                    }
                }
	                if (bestAltIdx != std::numeric_limits<size_t>::max()) {
	                    promotedRankedPriorIndexes[selectedIdx] = bestAltIdx;
	                    promotedCandidateIdBySource[selectedIdx] = bestAltOtherId;
	                    promotedZShiftBySource[selectedIdx] = bestZShift;
	                    promotedRankedPriorTargets.insert(bestAltIdx);
                    selectedCandidateIds.erase(bestSelectedOtherId);
                    selectedCandidateIds.insert(bestAltOtherId);
                    const std::string parentName =
                        selected.parentIdx < frame.cells.size()
                            ? frame.cells[selected.parentIdx].getName()
                            : std::string("<invalid>");
                    std::cout
                        << "[CellLumen Fusion SplitPrior ZStackDaughterPromotion] frame="
                        << absoluteFrame
                        << " parent=" << parentName
                        << " fromCandidate=" << bestSelectedOtherId
                        << " toCandidate=" << bestAltOtherId
                        << " lateral=" << bestLateral
                        << " zShift=" << bestZShift
                        << " selectedScore=" << selected.score
                        << " promotedScore=" << bestAltScore
                        << std::endl;
                }
            }
        }

        auto promotedCandidateIdForZReplacement =
            [&](const RankedSplitPrior &ranked) -> int {
                auto isCurrentFrameCandidateId = [](int candidateId) {
                    return candidateId >= 0 && candidateId < 1000000;
                };
                if (!lumenConfig
                         .fusionSplitPriorPartialWindowZReplacementRescueEnabled ||
                    ranked.parentAnchored ||
                    !isCurrentFrameCandidateId(ranked.candidateA) ||
                    !isCurrentFrameCandidateId(ranked.candidateB) ||
                    !ranked.continuationClaimBlockers.empty() ||
                    ranked.windowBothDaughtersSupported <
                        std::max(
                            0,
                            lumenConfig
                                .fusionSplitPriorPartialWindowWideLateralMinWindowBoth) ||
                    ranked.windowMissingDaughterCount >
                        std::max(
                            0,
                            lumenConfig
                                .fusionSplitPriorPartialWindowWideLateralMaxWindowMissing) ||
                    ranked.windowParentPersists >
                        std::max(
                            0,
                            lumenConfig
                                .fusionSplitPriorPartialWindowWideLateralMaxParentPersists) ||
                    ranked.parentShapeElongation <
                        std::max(
                            1.0f,
                            lumenConfig
                                .fusionSplitPriorPartialWindowZReplacementMinParentShape) ||
                    ranked.sep <
                        std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorPartialWindowZReplacementMinSeparation) ||
                    ranked.midpointDist >
                        std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorPartialWindowZReplacementMaxMidpointDistance) ||
                    ranked.parentDistanceBalance <
                        std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorPartialWindowZReplacementMinParentDistanceBalance) ||
                    ranked.voxA <
                        std::max(
                            0,
                            lumenConfig
                                .fusionSplitPriorPartialWindowZReplacementMinVoxels) ||
                    ranked.voxB <
                        std::max(
                            0,
                            lumenConfig
                                .fusionSplitPriorPartialWindowZReplacementMinVoxels) ||
                    ranked.signalA <
                        std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorPartialWindowZReplacementMinSignal) ||
                    ranked.signalB <
                        std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorPartialWindowZReplacementMinSignal) ||
                    ranked.score >
                        static_cast<double>(
                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorPartialWindowZReplacementMaxScore))) {
                    return -1;
                }
                const cv::Point3f rankedDelta =
                    ranked.proposal.d1Pos - ranked.proposal.d2Pos;
                const float rankedLateral =
                    std::sqrt(rankedDelta.x * rankedDelta.x +
                              rankedDelta.y * rankedDelta.y);
                const float rankedSep =
                    std::max(1e-4f, ranked.sep);
                const float rankedZDominance =
                    std::abs(rankedDelta.z) / rankedSep;
                if (rankedLateral <
                        std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorPartialWindowZReplacementMinLateralSeparation) ||
                    rankedZDominance >
                        std::clamp(
                            lumenConfig
                                .fusionSplitPriorPartialWindowZReplacementMaxZDominance,
                            0.0f, 1.0f)) {
                    return -1;
                }
                auto sharesExactlyOneCandidate =
                    [](const RankedSplitPrior &a,
                       const RankedSplitPrior &b) {
                        const bool shareA =
                            a.candidateA == b.candidateA ||
                            a.candidateA == b.candidateB;
                        const bool shareB =
                            a.candidateB == b.candidateA ||
                            a.candidateB == b.candidateB;
                        return shareA != shareB;
                    };
                auto nonSharedCandidateId =
                    [](const RankedSplitPrior &candidate,
                       const RankedSplitPrior &other) {
                        const bool aShared =
                            candidate.candidateA == other.candidateA ||
                            candidate.candidateA == other.candidateB;
                        return aShared ? candidate.candidateB
                                       : candidate.candidateA;
                    };
                auto candidatePosition =
                    [](const RankedSplitPrior &candidate, int id) {
                        return candidate.candidateA == id
                                   ? candidate.proposal.d1Pos
                                   : candidate.proposal.d2Pos;
                    };
                auto candidateVoxels =
                    [](const RankedSplitPrior &candidate, int id) {
                        return candidate.candidateA == id
                                   ? candidate.voxA
                                   : candidate.voxB;
                    };
                const float minZSeparation =
                    std::max(
                        0.0f,
                        lumenConfig
                            .fusionSplitPriorPartialWindowZReplacementMinZSeparation);
                const float minVoxelRatio =
                    std::max(
                        1.0f,
                        lumenConfig
                            .fusionSplitPriorPartialWindowZReplacementMinVoxelRatio);
                const int weakMinVoxels =
                    std::max(
                        0,
                        lumenConfig
                            .fusionSplitPriorWeakBalancedCleanWindowMinVoxels);
                const float weakMinSignal =
                    std::max(
                        0.0f,
                        lumenConfig
                            .fusionSplitPriorWeakBalancedCleanWindowMinSignal);
                const float minBalancedBonus =
                    std::max(
                        0.0f,
                        lumenConfig
                            .fusionSplitPriorWeakBalancedCleanWindowMinBonus);
                for (const RankedSplitPrior &other : rankedPriors) {
                    if (&other == &ranked ||
                        other.parentIdx != ranked.parentIdx ||
                        other.parentAnchored ||
                        !isCurrentFrameCandidateId(other.candidateA) ||
                        !isCurrentFrameCandidateId(other.candidateB) ||
                        other.windowBothDaughtersSupported < 2 ||
                        other.windowMissingDaughterCount != 0 ||
                        other.windowParentPersists != 0 ||
                        !other.continuationClaimBlockers.empty() ||
                        other.balancedWindowBonus < minBalancedBonus ||
                        !sharesExactlyOneCandidate(ranked, other)) {
                        continue;
                    }
                    const bool weakCleanOther =
                        other.voxA < weakMinVoxels ||
                        other.voxB < weakMinVoxels ||
                        other.signalA < weakMinSignal ||
                        other.signalB < weakMinSignal;
                    if (!weakCleanOther) {
                        continue;
                    }
                    const int replacementId =
                        nonSharedCandidateId(ranked, other);
                    const int blockedId =
                        nonSharedCandidateId(other, ranked);
                    const cv::Point3f replacementPos =
                        candidatePosition(ranked, replacementId);
                    const cv::Point3f blockedPos =
                        candidatePosition(other, blockedId);
                    const int replacementVoxels =
                        candidateVoxels(ranked, replacementId);
                    const int blockedVoxels =
                        std::max(1, candidateVoxels(other, blockedId));
                    if (std::abs(replacementPos.z - blockedPos.z) >=
                            minZSeparation &&
                        static_cast<float>(replacementVoxels) >=
                            static_cast<float>(blockedVoxels) *
                                minVoxelRatio) {
                        return replacementId;
                    }
                }
                return -1;
            };

        for (const size_t rankedIdx : selectedRankedPriorIndexes) {
            const size_t effectiveRankedIdx =
                promotedRankedPriorIndexes.count(rankedIdx) > 0
                    ? promotedRankedPriorIndexes[rankedIdx]
                    : rankedIdx;
            const RankedSplitPrior &ranked = rankedPriors[effectiveRankedIdx];
            if (ranked.parentIdx >= frame.cells.size()) {
                continue;
            }
	            const std::string parentName = frame.cells[ranked.parentIdx].getName();
	            BridgeSplitProposal proposal = ranked.proposal;
	            if (promotedRankedPriorIndexes.count(rankedIdx) > 0) {
	                proposal.zStackDaughterPromotion = true;
	                auto promotedIdIt = promotedCandidateIdBySource.find(rankedIdx);
	                if (promotedIdIt != promotedCandidateIdBySource.end()) {
	                    proposal.zStackPromotedCandidateId = promotedIdIt->second;
	                }
	                auto promotedShiftIt = promotedZShiftBySource.find(rankedIdx);
	                if (promotedShiftIt != promotedZShiftBySource.end()) {
	                    proposal.zStackPromotionZShift = promotedShiftIt->second;
	                }
	            }
                const int zReplacementPromotedCandidateId =
                    promotedCandidateIdForZReplacement(ranked);
                if (zReplacementPromotedCandidateId >= 0) {
                    proposal.zStackDaughterPromotion = true;
                    proposal.zStackPromotedCandidateId =
                        zReplacementPromotedCandidateId;
                    std::cout
                        << "[CellLumen Fusion SplitPrior ZReplacementSeedLock] frame="
                        << absoluteFrame
                        << " parent=" << parentName
                        << " promotedCandidate="
                        << zReplacementPromotedCandidateId
                        << " score=" << ranked.score
                        << std::endl;
                }
	            proposal.futureContinuationConflictRescued =
	                isFutureContinuationConflictRescue(ranked);
            proposal.continuationClaimBlockerNames =
                ranked.continuationClaimBlockerNames;
            splitPriorsForFrame[parentName] = proposal;
            logStoredSplitPrior(ranked, parentName);
        }

        if (lumenConfig.fusionTemporalCenterRepairParentAnchorReanchorEnabled) {
            const int reanchorMinWindowSupport = std::max(
                1,
                lumenConfig.fusionTemporalCenterRepairMinWindowSupport);
            const float reanchorMinShape = std::max(
                1.0f,
                lumenConfig
                    .fusionTemporalCenterRepairParentAnchorReanchorMinShape);
            for (const RankedSplitPrior &ranked : rankedPriors) {
                if (ranked.parentIdx >= frame.cells.size()) {
                    continue;
                }
                const bool realA = ranked.candidateA >= 0 &&
                                   ranked.candidateA < 1000000000;
                const bool realB = ranked.candidateB >= 0 &&
                                   ranked.candidateB < 1000000000;
                const bool cleanOneRealParentAnchor =
                    ranked.parentAnchored &&
                    ranked.parentShapeElongation >= reanchorMinShape &&
                    (realA != realB) &&
                    ranked.windowBothDaughtersSupported >=
                        reanchorMinWindowSupport &&
                    ranked.windowMissingDaughterCount == 0 &&
                    ranked.windowParentPersists == 0 &&
                    ranked.neighborClaimPenalty <= 1e-5f &&
                    ranked.continuationClaimSoftPenalty <= 1e-5f &&
                    ranked.continuationClaimBlockers.empty();
                if (!cleanOneRealParentAnchor) {
                    continue;
                }
                const std::string parentName =
                    frame.cells[ranked.parentIdx].getName();
                if (realA) {
                    centerReanchorCandidateIdsForFrame[parentName].insert(
                        ranked.candidateA);
                }
                if (realB) {
                    centerReanchorCandidateIdsForFrame[parentName].insert(
                        ranked.candidateB);
                }
            }
        }

        double selectedScore = 0.0;
        int selectedTemporalRepair = 0;
        for (const size_t rankedIdx : selectedRankedPriorIndexes) {
            selectedScore += rankedPriors[rankedIdx].score;
            if (rankedPriors[rankedIdx].temporalRepairEligible) {
                ++selectedTemporalRepair;
            }
        }
        std::cout << "[CellLumen Fusion SplitPrior GlobalSelect] frame=" << absoluteFrame
                  << " ranked=" << rankedPriors.size()
                  << " parent_groups=" << parentGroups.size()
                  << " selected=" << selectedRankedPriorIndexes.size()
                  << " selected_temporal_repair=" << selectedTemporalRepair
                  << " max_priors=" << maxPriors
                  << " objective=min_cost_skip_zero"
                  << " max_selectable_cost=" << maxSelectableCost
                  << " selected_score=" << selectedScore
                  << " search_nodes=" << searchNodes
                  << " search_truncated=" << (searchTruncated ? 1 : 0)
                  << std::endl;

        if (candidateGraphLog) {
            for (size_t rankedIdx = 0; rankedIdx < rankedPriors.size(); ++rankedIdx) {
                const RankedSplitPrior &ranked = rankedPriors[rankedIdx];
                if (ranked.parentIdx >= frame.cells.size()) {
                    continue;
                }
                CandidateGraphRow row;
                row.frame = absoluteFrame;
                row.kind = "split_pair";
                row.source = "cell_lumen_split_prior";
                row.parent = frame.cells[ranked.parentIdx].getName();
                row.candidateA = std::to_string(ranked.candidateA);
                row.candidateB = std::to_string(ranked.candidateB);
                bool rowSelected =
                    selectedRankedPriorIndexes.count(rankedIdx) > 0;
                if (promotedRankedPriorIndexes.count(rankedIdx) > 0) {
                    rowSelected = false;
                }
                if (promotedRankedPriorTargets.count(rankedIdx) > 0) {
                    rowSelected = true;
                }
                row.selected = rowSelected ? 1 : 0;
                row.score = ranked.score;
                row.rawScore = ranked.rawScore;
                row.bridgeMetric = ranked.elongatedParentRescued ? 1.0 : 0.0;
                row.sep = ranked.sep;
                row.minSep = ranked.minSep;
                row.maxSep = ranked.maxSep;
                row.midpointDist = ranked.midpointDist;
                row.parentShape = ranked.parentShapeElongation;
                row.parentPersistencePenalty = ranked.parentPersistencePenalty;
                row.neighborClaimPenalty = ranked.neighborClaimPenalty;
                row.parentDistNear = ranked.nearParentDist;
                row.parentDistFar = ranked.farParentDist;
                row.parentDistBalance = ranked.parentDistanceBalance;
                row.d1 = ranked.proposal.d1Pos;
                row.d2 = ranked.proposal.d2Pos;
                row.voxA = ranked.voxA;
                row.voxB = ranked.voxB;
                row.signalA = ranked.signalA;
                row.signalB = ranked.signalB;
                std::ostringstream note;
                if (ranked.conflictReplacementEligible) {
                    note << "conflict_replacement_eligible;";
                }
                if (ranked.parentAnchored) {
                    note << "parent_anchored;";
                }
                if (ranked.temporalRepairEligible) {
                    note << "temporal_repair_eligible;";
                    note << "parent_age_frames=" << parentAgeFramesForRanked(ranked)
                         << ";min_parent_age="
                         << std::max(
                                0,
                                lumenConfig
                                    .fusionSplitPriorTemporalRepairMinParentAgeFrames)
                         << ";parent_age_ok="
                         << (temporalRepairParentAgeOk(ranked) ? 1 : 0)
                         << ";";
                }
                if (ranked.temporalRepairCatchExpanded) {
                    note << "temporal_repair_catch_expanded;";
                }
                note << "ranking_soft_penalty=" << ranked.rankingSoftPenalty
                     << ";window_score=" << ranked.windowSupportScore
                     << ";window_both=" << ranked.windowBothDaughtersSupported
                     << ";window_missing=" << ranked.windowMissingDaughterCount
                     << ";window_parent_persists=" << ranked.windowParentPersists
                     << ";continuation_claim_soft_penalty="
                     << ranked.continuationClaimSoftPenalty
                     << ";balanced_window_bonus=" << ranked.balancedWindowBonus
                     << ";continuation_claim_blockers="
                     << (ranked.continuationClaimBlockerNames.empty()
                             ? "none"
                             : ranked.continuationClaimBlockerNames);
                row.note = note.str();
                writeCandidateGraphRow(candidateGraphLog, row);
            }
        }
        std::cout << "[CellLumen Fusion SplitPrior Summary] frame=" << absoluteFrame
                  << " candidates=" << candidates.size()
                  << " passed_gate=" << splitPriorCandidatesPassedGate
                  << " assigned_candidates=" << splitPriorCandidateAssignments
                  << " parents_with_2plus=" << splitPriorParentsWithTwoCandidates
                  << " ranked_pairs=" << rankedPriors.size()
                  << " priors=" << splitPriorsForFrame.size()
                  << " max_priors=" << maxPriors
                  << " rejected_parent_distance=" << splitPriorRejectedParentDistance
                  << " rejected_separation_pairs=" << splitPriorRejectedSeparation
                  << " rejected_midpoint_pairs=" << splitPriorRejectedMidpoint
                  << " rejected_score_pairs=" << splitPriorRejectedScore
                  << " rejected_neighbor_claim_pairs=" << splitPriorRejectedNeighborClaim
                  << " rejected_continuation_owner_assignments="
                  << splitPriorRejectedContinuationOwner
                  << " rejected_signal_candidates=" << splitPriorRejectedSignal
                  << " temporal_catch_assignments="
                  << splitPriorTemporalCatchAssignments
                  << " early_large_catch_assignments="
                  << splitPriorEarlyLargeCatchAssignments
                  << " rejected_temporal_catch_pairs="
                  << splitPriorRejectedTemporalCatchPairs
                  << " rejected_collapsed_center_cluster_pairs="
                  << splitPriorRejectedCollapsedCenterCluster
                  << " soft_allowed_collapsed_center_cluster_pairs="
                  << splitPriorSoftAllowedCollapsedCenterCluster
                  << " window_enabled=" << (windowEnabled ? 1 : 0)
                  << " window_offsets=" << windowCandidatesByOffset.size()
                  << " ranking_soft_gate="
                  << (lumenConfig.fusionSplitPriorRankingSoftGateEnabled ? 1 : 0)
                  << " split_min_voxels=" << splitPriorMinVoxels
                  << " split_min_top10_minus_shell=" << splitPriorMinTop10MinusShell
                  << std::endl;
    }

    int rejectedSize = 0;
    int rejectedSignal = 0;
    int rejectedClose = 0;
    int repairedClose = 0;
    int centerPriorGuided = 0;
    int added = 0;
    std::vector<bool> centerPriorUsed(frame.cells.size(), false);

    for (const auto &candidate : candidates) {
        if (added >= maxAdded) {
            break;
        }
        if (candidate.voxelCount < lumenConfig.fusionMinVoxels) {
            ++rejectedSize;
            continue;
        }
        if (candidate.top10MinusShell < lumenConfig.fusionMinTop10MinusShell) {
            ++rejectedSignal;
            continue;
        }

        const float candidateA = std::max(1.0f, candidate.majorRadius * radiusScale);
        const float candidateB = std::max(1.0f, candidate.bRadius * radiusScale);
        const float candidateC = std::max(1.0f, candidate.minorRadius * radiusScale);

        bool tooClose = false;
        size_t nearestIndex = std::numeric_limits<size_t>::max();
        float nearestDistanceSq = std::numeric_limits<float>::max();
        for (size_t existingIdx = 0; existingIdx < frame.cells.size(); ++existingIdx) {
            const cv::Point3f existingCenter(frame.cells[existingIdx].getX(),
                                             frame.cells[existingIdx].getY(),
                                             frame.cells[existingIdx].getZ());
            const float d2 = squaredDistance(candidate.centerScaled, existingCenter);
            if (d2 < nearestDistanceSq) {
                nearestDistanceSq = d2;
                nearestIndex = existingIdx;
            }
            if (d2 < minDistanceSq) {
                tooClose = true;
            }
        }
        if (tooClose) {
            const float centerPriorMaxDistance =
                std::max(0.0f, lumenConfig.fusionCenterPriorMaxDistance);
            const float centerPriorMaxDistanceSq =
                centerPriorMaxDistance * centerPriorMaxDistance;
            if (lumenConfig.fusionCenterPriorEnabled &&
                nearestIndex != std::numeric_limits<size_t>::max() &&
                nearestIndex < centerPriorUsed.size() &&
                !centerPriorUsed[nearestIndex] &&
                nearestDistanceSq <= centerPriorMaxDistanceSq) {
                Ellipsoid &target = frame.cells[nearestIndex];
                const float posBlend = std::clamp(lumenConfig.fusionCenterPriorPositionBlend,
                                                  0.0f, 1.0f);
                const float radiusBlend = std::clamp(lumenConfig.fusionCenterPriorRadiusBlend,
                                                     0.0f, 1.0f);
                const float oldX = target.getX();
                const float oldY = target.getY();
                const float oldZ = target.getZ();
                const float oldA = target.getARadius();
                const float oldB = target.getBRadius();
                const float oldC = target.getCRadius();

                target.setPosition(oldX * (1.0f - posBlend) +
                                       candidate.centerScaled.x * posBlend,
                                   oldY * (1.0f - posBlend) +
                                       candidate.centerScaled.y * posBlend,
                                   oldZ * (1.0f - posBlend) +
                                       candidate.centerScaled.z * posBlend);
                target.setRadii(std::max(1.0f, oldA * (1.0f - radiusBlend) +
                                                    candidateA * radiusBlend),
                                std::max(1.0f, oldB * (1.0f - radiusBlend) +
                                                    candidateB * radiusBlend),
                                std::max(1.0f, oldC * (1.0f - radiusBlend) +
                                                    candidateC * radiusBlend));

                const std::string &targetName = target.getName();
                const std::array<float, 3> guidedRadii{
                    target.getARadius(), target.getBRadius(), target.getCRadius()};
                cellShapeBirth[targetName] = guidedRadii;
                cellShapeReference[targetName] = guidedRadii;
                centerPriorUsed[nearestIndex] = true;
                ++centerPriorGuided;
                std::cout << "[CellLumen Fusion CenterPrior] frame=" << absoluteFrame
                          << " target=" << targetName
                          << " candidate_center=(" << candidate.centerScaled.x << ","
                          << candidate.centerScaled.y << "," << candidate.centerScaled.z << ")"
                          << " old_center=(" << oldX << "," << oldY << "," << oldZ << ")"
                          << " new_center=(" << target.getX() << ","
                          << target.getY() << "," << target.getZ() << ")"
                          << " old_radii=(" << oldA << "," << oldB << "," << oldC << ")"
                          << " new_radii=(" << target.getARadius() << ","
                          << target.getBRadius() << "," << target.getCRadius() << ")"
                          << " distance=" << std::sqrt(nearestDistanceSq)
                          << " vox=" << candidate.voxelCount
                          << " top10MinusShell=" << candidate.top10MinusShell
                          << std::endl;
            }

            const float repairMaxDistance =
                std::max(0.0f, lumenConfig.fusionRepairMaxDistance);
            const float repairMaxDistanceSq = repairMaxDistance * repairMaxDistance;
            if (lumenConfig.fusionRepairCloseCellsEnabled &&
                nearestIndex != std::numeric_limits<size_t>::max() &&
                nearestDistanceSq <= repairMaxDistanceSq) {
                Ellipsoid &target = frame.cells[nearestIndex];
                const float minCandidateRadius = std::min({candidateA, candidateB, candidateC});
                const float repairRatio = std::clamp(lumenConfig.fusionRepairRadiusRatio,
                                                     0.0f, 1.5f);
                const bool radiusLooksCollapsed =
                    target.getMinorRadius() < minCandidateRadius * repairRatio ||
                    target.getARadius() < candidateA * repairRatio ||
                    target.getBRadius() < candidateB * repairRatio ||
                    target.getCRadius() < candidateC * repairRatio;
                if (radiusLooksCollapsed) {
                    const float posBlend = std::clamp(lumenConfig.fusionRepairPositionBlend,
                                                      0.0f, 1.0f);
                    const float radiusBlend = std::clamp(lumenConfig.fusionRepairRadiusBlend,
                                                         0.0f, 1.0f);
                    const float oldX = target.getX();
                    const float oldY = target.getY();
                    const float oldZ = target.getZ();
                    target.setPosition(oldX * (1.0f - posBlend) +
                                           candidate.centerScaled.x * posBlend,
                                       oldY * (1.0f - posBlend) +
                                           candidate.centerScaled.y * posBlend,
                                       oldZ * (1.0f - posBlend) +
                                           candidate.centerScaled.z * posBlend);
                    const float newA = std::max(target.getARadius(),
                                                target.getARadius() * (1.0f - radiusBlend) +
                                                    candidateA * radiusBlend);
                    const float newB = std::max(target.getBRadius(),
                                                target.getBRadius() * (1.0f - radiusBlend) +
                                                    candidateB * radiusBlend);
                    const float newC = std::max(target.getCRadius(),
                                                target.getCRadius() * (1.0f - radiusBlend) +
                                                    candidateC * radiusBlend);
                    target.setRadii(newA, newB, newC);

                    const std::string &targetName = target.getName();
                    const std::array<float, 3> repairedRadii{
                        target.getARadius(), target.getBRadius(), target.getCRadius()};
                    cellShapeBirth[targetName] = repairedRadii;
                    cellShapeReference[targetName] = repairedRadii;
                    ++repairedClose;
                    std::cout << "[CellLumen Fusion Repair] frame=" << absoluteFrame
                              << " target=" << targetName
                              << " candidate_center=(" << candidate.centerScaled.x << ","
                              << candidate.centerScaled.y << "," << candidate.centerScaled.z << ")"
                              << " new_center=(" << target.getX() << ","
                              << target.getY() << "," << target.getZ() << ")"
                              << " new_radii=(" << target.getARadius() << ","
                              << target.getBRadius() << "," << target.getCRadius() << ")"
                              << " distance=" << std::sqrt(nearestDistanceSq)
                              << " vox=" << candidate.voxelCount
                              << " top10MinusShell=" << candidate.top10MinusShell
                              << std::endl;
                }
            }
            ++rejectedClose;
            continue;
        }

        const std::string name = lumenConfig.fusionNamePrefix + "_" +
                                 std::to_string(absoluteFrame) + "_" +
                                 std::to_string(added + 1);
        EllipsoidParams params(name,
                               candidate.centerScaled.x,
                               candidate.centerScaled.y,
                               candidate.centerScaled.z,
                               candidateA,
                               candidateC,
                               0.0f,
                               0.0f,
                               0.0f,
                               brightness);
        params.bRadius = candidateB;

        Ellipsoid rescuedCell(params);
        frame.cells.push_back(rescuedCell);
        centerPriorUsed.push_back(true);
        cellShapeBirth[name] = {rescuedCell.getARadius(),
                                rescuedCell.getBRadius(),
                                rescuedCell.getCRadius()};
        cellShapeReference[name] = cellShapeBirth[name];
        ++added;

        std::cout << "[CellLumen Fusion Add] frame=" << absoluteFrame
                  << " name=" << name
                  << " center=(" << candidate.centerScaled.x << ","
                  << candidate.centerScaled.y << "," << candidate.centerScaled.z << ")"
                  << " radii=(" << rescuedCell.getARadius() << ","
                  << rescuedCell.getBRadius() << "," << rescuedCell.getCRadius() << ")"
                  << " vox=" << candidate.voxelCount
                  << " top10MinusShell=" << candidate.top10MinusShell
                  << std::endl;
    }

    std::cout << "[CellLumen Fusion Summary] frame=" << absoluteFrame
              << " candidates=" << candidates.size()
              << " added=" << added
              << " rejected_size=" << rejectedSize
              << " rejected_signal=" << rejectedSignal
              << " rejected_close=" << rejectedClose
              << " center_prior_guided=" << centerPriorGuided
              << " repaired_close=" << repairedClose
              << " existing_after=" << frame.cells.size()
              << std::endl;
}

void CellUniverse::prepareFrame(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("prepareFrame: invalid frame index");
    }
    applyRuntimeDensityProfileForFrame(frameIndex, "prepare");
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
    if (usesIterativePostPreprocessing(config.simulation)) {
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
    } else {
        std::cout << "[PostPreprocess] frame="
                  << imagePaths[static_cast<size_t>(frameIndex)].filename().string()
                  << " skipped_for_mode=" << config.simulation.preprocess_mode
                  << '\n';
    }
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

    {
        ScopedStageTimer timer(firstFrame + frameIndex,
                               "cell_lumen_fusion_prepare");
        applyCellLumenRescue(frameIndex);
    }

    // loadImageStacks already generates _synthFrame + refreshes the full-image
    // cost cache. The previous `if (config.cell) regenerateSynthFrame()` was
    // redundant work (2x render + 2x cost cache per frame).
    frames[frameIndex].loadImageStacks(real_frame);
    frames[frameIndex].setSignalMap(std::move(signalMap));
    prepareSignalCentersForFrame(frameIndex, real_frame, true);
}

float CellUniverse::computeFrameMedianNearestNeighbor(int frameIndex) const
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size()) {
        throw std::invalid_argument(
            "computeFrameMedianNearestNeighbor: invalid frame index");
    }

    const auto &cells = frames[static_cast<size_t>(frameIndex)].cells;
    if (cells.size() < 2) {
        return 1000000000.0f;
    }

    std::vector<float> nearestDistances;
    nearestDistances.reserve(cells.size());
    for (size_t i = 0; i < cells.size(); ++i) {
        const EllipsoidParams params = cells[i].getCellParams();
        const cv::Point3f pos(params.x, params.y, params.z);
        float nearest = std::numeric_limits<float>::max();
        for (size_t j = 0; j < cells.size(); ++j) {
            if (i == j) {
                continue;
            }
            const EllipsoidParams otherParams = cells[j].getCellParams();
            const cv::Point3f otherPos(
                otherParams.x, otherParams.y, otherParams.z);
            nearest = std::min(nearest,
                               static_cast<float>(cv::norm(pos - otherPos)));
        }
        if (nearest < std::numeric_limits<float>::max()) {
            nearestDistances.push_back(nearest);
        }
    }
    return medianOrZero(nearestDistances);
}

void CellUniverse::applyRuntimeDensityProfileForFrame(
    int frameIndex,
    const std::string &phase)
{
    if (!config.runtimeDensityProfileSelectionEnabled ||
        config.runtimeDensityProfiles.empty()) {
        return;
    }
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size()) {
        throw std::invalid_argument(
            "applyRuntimeDensityProfileForFrame: invalid frame index");
    }

    const float medianNearest =
        computeFrameMedianNearestNeighbor(frameIndex);
    const int absoluteFrame = firstFrame + frameIndex;
    int liveCellCount = 0;
    for (const auto &cell : frames[static_cast<size_t>(frameIndex)].cells) {
        if (!cell.isTrash()) {
            ++liveCellCount;
        }
    }

    const BaseConfig::RuntimeDensityProfile *selected = nullptr;
    for (const auto &profile : config.runtimeDensityProfiles) {
        const bool medianOk =
            medianNearest >= profile.minMedianNearestNeighborPx &&
            medianNearest < profile.maxMedianNearestNeighborPx;
        const bool frameOk =
            absoluteFrame >= profile.minFrame &&
            absoluteFrame <= profile.maxFrame;
        const bool cellCountOk =
            liveCellCount >= profile.minLiveCells &&
            liveCellCount <= profile.maxLiveCells;
        if (medianOk && frameOk && cellCountOk) {
            selected = &profile;
            break;
        }
    }
    if (selected == nullptr && !config.runtimeDensityDefaultProfile.empty()) {
        for (const auto &profile : config.runtimeDensityProfiles) {
            if (profile.name == config.runtimeDensityDefaultProfile) {
                selected = &profile;
                break;
            }
        }
    }
    if (selected == nullptr) {
        selected = &config.runtimeDensityProfiles.front();
    }

    const std::string selectedName = selected->name;
    const bool changed =
        (selectedName != config.runtimeDensityActiveProfile);
    if (changed) {
        BaseConfig selectedConfig;
        selectedConfig.explodeConfig(selected->expandedConfig);

        const auto runtimeProfiles = config.runtimeDensityProfiles;
        const bool runtimeEnabled =
            config.runtimeDensityProfileSelectionEnabled;
        const std::string runtimeMetric =
            config.runtimeDensityProfileMetric;
        const std::string runtimeDefault =
            config.runtimeDensityDefaultProfile;

        const int runtimeZSlices = config.simulation.z_slices;
        const float runtimeZScaling = config.simulation.z_scaling;
        const int parallelThreads = config.simulation.parallel_threads;
        const int parallelMinSlices = config.simulation.parallel_min_slices;
        const int resumeFrom = config.simulation.resume_from;
        const std::string resumeSourceDir =
            config.simulation.resume_source_dir;

        selectedConfig.runtimeDensityProfiles = runtimeProfiles;
        selectedConfig.runtimeDensityProfileSelectionEnabled =
            runtimeEnabled;
        selectedConfig.runtimeDensityProfileMetric = runtimeMetric;
        selectedConfig.runtimeDensityDefaultProfile = runtimeDefault;
        selectedConfig.runtimeDensityActiveProfile = selectedName;

        selectedConfig.simulation.z_slices = runtimeZSlices;
        selectedConfig.simulation.z_scaling = runtimeZScaling;
        selectedConfig.simulation.parallel_threads = parallelThreads;
        selectedConfig.simulation.parallel_min_slices = parallelMinSlices;
        selectedConfig.simulation.resume_from = resumeFrom;
        selectedConfig.simulation.resume_source_dir = resumeSourceDir;

        config = selectedConfig;
        if (config.cell) {
            config.cell->maxZ =
                static_cast<float>(config.simulation.z_slices) - 1.0f;
            Ellipsoid::cellConfig = *config.cell;
        }
    }
    frames[static_cast<size_t>(frameIndex)].setSimulationConfig(
        config.simulation);

    std::cout << "[Runtime Density Profile] frame="
              << absoluteFrame
              << " phase=" << phase
              << " metric=" << config.runtimeDensityProfileMetric
              << " medianNearestNeighbor=" << medianNearest
              << " liveCells=" << liveCellCount
              << " selectedProfile=" << selectedName
              << " changed=" << (changed ? 1 : 0)
              << std::endl;
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
    const bool heavyPostPreprocessing = usesIterativePostPreprocessing(config.simulation);
    const bool alignmentEnabled =
        heavyPostPreprocessing && config.simulation.edge_brightness_alignment_enabled;

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
            if (heavyPostPreprocessing) {
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
            } else {
                std::cout << "[PostPreprocess] frame="
                          << imagePaths[static_cast<size_t>(frameIndex)].filename().string()
                          << " skipped_for_mode=" << config.simulation.preprocess_mode
                          << '\n';
            }
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

const std::vector<CellUniverse::CellLumenLookaheadCandidate> &
CellUniverse::getCellLumenLookaheadCandidates(int frameIndex)
{
    static const std::vector<CellLumenLookaheadCandidate> empty;
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= imagePaths.size()) {
        return empty;
    }

    const auto cached = cellLumenLookaheadCandidates.find(frameIndex);
    if (cached != cellLumenLookaheadCandidates.end()) {
        return cached->second;
    }

    auto &stored = cellLumenLookaheadCandidates[frameIndex];
    const CellLumenConfig &lumenConfig = config.cellLumen;
    const int minVoxels =
        lumenConfig.fusionSplitPriorMinVoxels >= 0
            ? lumenConfig.fusionSplitPriorMinVoxels
            : lumenConfig.fusionMinVoxels;
    const float minSignal =
        lumenConfig.fusionSplitPriorMinTop10MinusShell >= 0.0f
            ? lumenConfig.fusionSplitPriorMinTop10MinusShell
            : lumenConfig.fusionMinTop10MinusShell;

    EllipsoidConfig savedCellConfig = Ellipsoid::cellConfig;
    try {
        CellLumen lumen(config, fs::path(outputPath) / "cell_lumen_fusion_window");
        const auto detected =
            lumen.detectCellsForFrame(imagePaths[static_cast<size_t>(frameIndex)],
                                      false);
        int candidateId = 0;
        for (const auto &candidate : detected) {
            if (candidate.voxelCount < minVoxels ||
                candidate.top10MinusShell < minSignal) {
                ++candidateId;
                continue;
            }
            stored.push_back({
                candidate.centerScaled,
                candidate.voxelCount,
                candidate.top10MinusShell,
                candidateId
            });
            ++candidateId;
        }
        std::sort(stored.begin(), stored.end(),
                  [](const CellLumenLookaheadCandidate &a,
                     const CellLumenLookaheadCandidate &b) {
                      if (a.signal != b.signal) {
                          return a.signal > b.signal;
                      }
                      return a.voxelCount > b.voxelCount;
                  });
        std::cout << "[CellLumen Window Candidates] frame=" << (firstFrame + frameIndex)
                  << " candidates=" << stored.size()
                  << " min_voxels=" << minVoxels
                  << " min_signal=" << minSignal
                  << std::endl;
    } catch (const std::exception &ex) {
        std::cout << "[CellLumen Window Candidates] frame=" << (firstFrame + frameIndex)
                  << " status=skipped error=\"" << ex.what() << "\""
                  << std::endl;
    }
    Ellipsoid::cellConfig = savedCellConfig;
    return stored;
}

void CellUniverse::writeDensityBrightnessMetrics(int frameIndex,
                                                 const std::string &phase)
{
    const CellLumenConfig &lumenConfig = config.cellLumen;
    if (!lumenConfig.fusionDensityMetricsEnabled) {
        return;
    }
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size()) {
        return;
    }

    Frame &frame = frames[static_cast<size_t>(frameIndex)];
    const int absoluteFrame = firstFrame + frameIndex;
    const std::vector<cv::Mat> &stack = frame.getRealFrame();
    const auto lookaheadIt = cellLumenLookaheadCandidates.find(frameIndex);
    const std::vector<CellLumenLookaheadCandidate> emptyCandidates;
    const auto &lumenCandidates =
        lookaheadIt != cellLumenLookaheadCandidates.end()
            ? lookaheadIt->second
            : emptyCandidates;

    std::error_code ec;
    fs::create_directories(outputPath, ec);
    const fs::path csvPath =
        fs::path(outputPath) / "density_brightness_metrics.csv";
    bool writeHeader = true;
    if (fs::exists(csvPath, ec) && !ec) {
        const auto size = fs::file_size(csvPath, ec);
        writeHeader = ec || size == 0;
    }
    std::ofstream out(csvPath, std::ios::app);
    if (!out) {
        std::cerr << "[Density Metrics] frame=" << absoluteFrame
                  << " status=skipped reason=cannot_open_csv path="
                  << csvPath << std::endl;
        return;
    }
    if (writeHeader) {
        out << "frame,phase,cell_name,x,y,z,a_radius,b_radius,c_radius,"
            << "shape,cell_brightness,density_radius,local_neighbor_count,"
            << "nearest_neighbor_distance,knn_distance,density_per_1000_voxels,"
            << "lumen_candidate_count,lumen_within_density_radius,"
            << "nearest_lumen_distance,nearest_lumen_signal,"
            << "brightness_radius,brightness_shell_radius,inner_voxels,"
            << "shell_voxels,inner_mean,shell_mean,inner_top10_mean,"
            << "inner_max,mean_minus_shell,top10_minus_shell,"
            << "density_adaptive_enabled,brightness_adaptive_enabled\n";
    }

    const int k = std::max(1, lumenConfig.fusionDensityMetricsK);
    const float densityScale = std::max(
        0.0f, lumenConfig.fusionDensityMetricsRadiusScale);
    const float brightnessScale = std::max(
        0.0f, lumenConfig.fusionBrightnessMetricsRadiusScale);
    const float shellScale = std::max(
        1.01f, lumenConfig.fusionBrightnessMetricsShellScale);
    std::vector<float> nearestDistances;
    std::vector<float> knnDistances;
    std::vector<float> densityValues;
    std::vector<float> top10MinusShellValues;

    for (size_t i = 0; i < frame.cells.size(); ++i) {
        const Ellipsoid &cell = frame.cells[i];
        const EllipsoidParams params = cell.getCellParams();
        const cv::Point3f pos(params.x, params.y, params.z);
        const float maxRadius = std::max(
            1.0f,
            std::max(params.aRadius, std::max(params.bRadius, params.cRadius)));
        const float densityRadius = std::max(1.0f, densityScale * maxRadius);

        std::vector<float> distances;
        distances.reserve(frame.cells.size() > 0 ? frame.cells.size() - 1 : 0);
        int localNeighborCount = 0;
        for (size_t j = 0; j < frame.cells.size(); ++j) {
            if (i == j) {
                continue;
            }
            const EllipsoidParams otherParams = frame.cells[j].getCellParams();
            const cv::Point3f otherPos(otherParams.x, otherParams.y, otherParams.z);
            const float dist = cv::norm(pos - otherPos);
            distances.push_back(dist);
            if (dist <= densityRadius) {
                ++localNeighborCount;
            }
        }
        std::sort(distances.begin(), distances.end());
        const float nearestDistance =
            distances.empty() ? 0.0f : distances.front();
        const float knnDistance =
            distances.empty()
                ? 0.0f
                : distances[std::min<size_t>(static_cast<size_t>(k - 1),
                                             distances.size() - 1)];
        constexpr float pi = 3.14159265358979323846f;
        const float densityVolume =
            (4.0f / 3.0f) * pi *
            densityRadius * densityRadius * densityRadius;
        const float densityPer1000 =
            densityVolume > 1e-6f
                ? (static_cast<float>(localNeighborCount) / densityVolume) *
                      1000.0f
                : 0.0f;

        int lumenWithinDensityRadius = 0;
        float nearestLumenDistance = 0.0f;
        float nearestLumenSignal = 0.0f;
        bool hasLumenDistance = false;
        for (const auto &candidate : lumenCandidates) {
            const float dist = cv::norm(pos - candidate.position);
            if (dist <= densityRadius) {
                ++lumenWithinDensityRadius;
            }
            if (!hasLumenDistance || dist < nearestLumenDistance) {
                hasLumenDistance = true;
                nearestLumenDistance = dist;
                nearestLumenSignal = candidate.signal;
            }
        }

        const float brightnessRadius =
            std::max(1.0f, brightnessScale * maxRadius);
        const float shellRadius =
            std::max(brightnessRadius + 1.0f, brightnessRadius * shellScale);
        const BrightnessNeighborhoodStats brightnessStats =
            sampleBrightnessNeighborhood(stack, pos, brightnessRadius, shellRadius);
        const float meanMinusShell =
            brightnessStats.innerMean - brightnessStats.shellMean;
        const float top10MinusShell =
            brightnessStats.innerTop10Mean - brightnessStats.shellMean;

        nearestDistances.push_back(nearestDistance);
        knnDistances.push_back(knnDistance);
        densityValues.push_back(densityPer1000);
        top10MinusShellValues.push_back(top10MinusShell);

        out << absoluteFrame << ','
            << csvEscape(phase) << ','
            << csvEscape(params.name) << ','
            << params.x << ','
            << params.y << ','
            << params.z << ','
            << params.aRadius << ','
            << params.bRadius << ','
            << params.cRadius << ','
            << cell.shapeElongation() << ','
            << params.brightness << ','
            << densityRadius << ','
            << localNeighborCount << ','
            << nearestDistance << ','
            << knnDistance << ','
            << densityPer1000 << ','
            << lumenCandidates.size() << ','
            << lumenWithinDensityRadius << ','
            << nearestLumenDistance << ','
            << nearestLumenSignal << ','
            << brightnessRadius << ','
            << shellRadius << ','
            << brightnessStats.innerVoxels << ','
            << brightnessStats.shellVoxels << ','
            << brightnessStats.innerMean << ','
            << brightnessStats.shellMean << ','
            << brightnessStats.innerTop10Mean << ','
            << brightnessStats.innerMax << ','
            << meanMinusShell << ','
            << top10MinusShell << ','
            << (lumenConfig.fusionDensityAdaptiveGateEnabled ? 1 : 0) << ','
            << (lumenConfig.fusionBrightnessAdaptiveGateEnabled ? 1 : 0)
            << '\n';
    }

    std::cout << "[Density Metrics] frame=" << absoluteFrame
              << " phase=" << phase
              << " cells=" << frame.cells.size()
              << " lumenCandidates=" << lumenCandidates.size()
              << " medianNearestNeighbor=" << medianOrZero(nearestDistances)
              << " medianKnnDistance=" << medianOrZero(knnDistances)
              << " medianDensityPer1000Voxels=" << medianOrZero(densityValues)
              << " medianTop10MinusShell="
              << medianOrZero(top10MinusShellValues)
              << " densityAdaptive="
              << (lumenConfig.fusionDensityAdaptiveGateEnabled ? 1 : 0)
              << " brightnessAdaptive="
              << (lumenConfig.fusionBrightnessAdaptiveGateEnabled ? 1 : 0)
              << std::endl;
}

void CellUniverse::optimize(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("Invalid frame index");
    }

    Frame &frame = frames[frameIndex];
    const int absoluteFrame = firstFrame + frameIndex;
    if (!frame.hasImageStacks()) {
        applyRuntimeDensityProfileForFrame(frameIndex, "optimize");
    }

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
    const auto lumenSplitPriorForFrameIt = cellLumenSplitPriors.find(frameIndex);
    const bool fusionSplitPriorPreparedForFrame =
        config.cellLumen.fusionSplitPriorEnabled &&
        lumenSplitPriorForFrameIt != cellLumenSplitPriors.end();
    const bool fusionSplitPriorForFrame =
        fusionSplitPriorPreparedForFrame &&
        !lumenSplitPriorForFrameIt->second.empty();
    int effectiveGuidedPerCellIters = guidedPerCellIters;
    int effectiveRandomPerCellIters = randomPerCellIters;
    const bool reducedFusionPerturb =
        fusionSplitPriorPreparedForFrame &&
        config.cellLumen.fusionReducePostSplitPerturbEnabled;
    if (reducedFusionPerturb) {
        const int fusionPerturbIters =
            std::max(0, config.cellLumen.fusionPostSplitPerturbItersPerCell);
        effectiveGuidedPerCellIters = std::min(effectiveGuidedPerCellIters, fusionPerturbIters);
        effectiveRandomPerCellIters = std::min(effectiveRandomPerCellIters, fusionPerturbIters);
    }
    const size_t activePerCellIters = static_cast<size_t>(std::max(
        0, useSignalGuidanceThisFrame ? effectiveGuidedPerCellIters : effectiveRandomPerCellIters));
    size_t totalIterations = frame.length() * activePerCellIters;
    int displayFrame = firstFrame + frameIndex;
    ScopedStageTimer optimizeTimer(displayFrame, "optimize_total");
    const float randomPerturbRadiusRatio =
        config.cell ? config.cell->randomPerturbRadiusRatio : 0.5f;
    std::ofstream candidateGraphLog = openCandidateGraphLog(outputPath, displayFrame);

    std::cout << "[Optimize] frame " << displayFrame
              << " (" << frame.cells.size() << " cells, " << totalIterations << " iterations)"
              << " perturbMode=" << (useSignalGuidanceThisFrame ? "signal_guided" : "random")
              << " guidedItersPerCell=" << guidedPerCellIters
              << " randomItersPerCell=" << randomPerCellIters
              << " effectiveGuidedItersPerCell=" << effectiveGuidedPerCellIters
              << " effectiveRandomItersPerCell=" << effectiveRandomPerCellIters
              << " fusionPriorPrepared=" << (fusionSplitPriorPreparedForFrame ? 1 : 0)
              << " fusionPriorCount="
              << (fusionSplitPriorPreparedForFrame
                      ? static_cast<int>(lumenSplitPriorForFrameIt->second.size())
                      : 0)
              << " fusionPerturbReduced=" << (reducedFusionPerturb ? 1 : 0)
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
    writeDensityBrightnessMetrics(frameIndex, "pre_optimize");

    auto logPerturbCandidate = [&](const std::string &source,
                                   const std::string &cellName,
                                   const Ellipsoid &cell,
                                   double costDiff,
                                   bool accepted,
                                   float radiusRatio,
                                   const std::string &note) {
        if (!candidateGraphLog) {
            return;
        }
        CandidateGraphRow row;
        row.frame = displayFrame;
        row.kind = "perturb";
        row.source = source;
        row.parent = cellName;
        row.selected = accepted ? 1 : 0;
        row.score = costDiff;
        row.imageGain = -costDiff;
        row.sep = radiusRatio;
        row.parentShape = cell.shapeElongation();
        row.d1 = cv::Point3f(cell.getX(), cell.getY(), cell.getZ());
        row.note = note;
        writeCandidateGraphRow(candidateGraphLog, row);
    };

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

    std::mt19937 &gen = cellUniverseRandomGenerator();
    std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

    const float overlapWeight = config.prob.overlap_penalty_weight;
    const float baseSplitProb = config.prob.P_split_base;

    // Resume CSVs are usually the validated state from the previous frame
    // copied onto the first requested image. Treat those cells as snapshots so
    // the first resumed frame can split. A fresh run from frame 0 still starts
    // with no split attempts because no previous biological time step exists.
    if (frameIndex == 0 && firstFrame > 0 && previousSnapshots.empty()) {
        std::cout << "[Snapshot Bootstrap] frame " << displayFrame
                  << " source=initial_cells cells=" << frame.cells.size()
                  << std::endl;
        for (const auto &cell : frame.cells) {
            auto p = cell.getCellParams();
            cv::Point3f fitSplitAxisDir;
            float fitSplitAxisLength = 0.0f;
            cell.worldSplitAxis(fitSplitAxisDir, fitSplitAxisLength);

            PreviousFrameSnapshot snap;
            snap.valid = true;
            snap.shapeElongation = cell.shapeElongation();
            snap.splitAxisDir = fitSplitAxisDir;
            snap.splitAxisLength = fitSplitAxisLength;
            snap.position = cv::Point3f(p.x, p.y, p.z);
            snap.aRadius = p.aRadius;
            snap.bRadius = p.bRadius;
            snap.cRadius = p.cRadius;
            snap.thetaX = p.theta_x;
            snap.thetaY = p.theta_y;
            snap.thetaZ = p.theta_z;
            snap.brightness = p.brightness;
            previousSnapshots[p.name] = snap;
        }
    }

    const bool allowSplits =
        frameIndex > 0 ||
        (frameIndex == 0 && firstFrame > 0 && !previousSnapshots.empty());

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
    const bool runCalibration =
        (frameIndex == 0) &&
        previousSnapshots.empty() &&
        (calibrationIters > 0) &&
        !frame.cells.empty();
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
                logPerturbCandidate("calibration_refine",
                                    calName,
                                    frame.cells[ci],
                                    calResult.first,
                                    calAccept,
                                    randomPerturbRadiusRatio,
                                    "score_is_total_cost_diff");
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

    std::set<std::string> pcaPositionLockedCells;
    std::unordered_map<std::string, std::string> pcaPositionLockReasons;
	    std::set<std::string> pcaPositionGuardRelaxedCells;
	    std::set<std::string> pcaPositionGuardWideRelaxedCells;
	    std::set<std::string> pcaPositionGuardRevertedCells;
	    std::set<std::string> sameFramePerturbLockedCells;
	    std::unordered_map<std::string, std::string> pcaPositionGuardRelaxReasons;
    auto lockPcaPosition = [&](const std::string &cellName,
                               const std::string &reason) {
        pcaPositionLockedCells.insert(cellName);
        auto reasonIt = pcaPositionLockReasons.find(cellName);
        if (reasonIt == pcaPositionLockReasons.end()) {
            pcaPositionLockReasons[cellName] = reason;
        } else if (reasonIt->second.find(reason) == std::string::npos) {
            reasonIt->second += "|";
            reasonIt->second += reason;
        }
    };
    auto relaxPcaPositionGuard = [&](const std::string &cellName,
                                     const std::string &reason) {
        if (cellName.empty()) {
            return;
        }
        pcaPositionGuardRelaxedCells.insert(cellName);
        auto reasonIt = pcaPositionGuardRelaxReasons.find(cellName);
        if (reasonIt == pcaPositionGuardRelaxReasons.end()) {
            pcaPositionGuardRelaxReasons[cellName] = reason;
        } else if (reasonIt->second.find(reason) == std::string::npos) {
            reasonIt->second += "|";
            reasonIt->second += reason;
        }
    };
    auto relaxPcaPositionGuardWide = [&](const std::string &cellName,
                                         const std::string &reason) {
        relaxPcaPositionGuard(cellName, reason);
        pcaPositionGuardWideRelaxedCells.insert(cellName);
    };

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
        std::vector<std::string> pcaGuardRevertedNames(nCells);
        #pragma omp parallel for schedule(dynamic)
        for (int ci = 0; ci < nCells; ++ci) {
            const std::string sname = frame.cells[ci].getName();
            const bool isTrashCell = frame.cells[ci].isTrash();
            const cv::Point3f prePcaPos(frame.cells[ci].getX(),
                                        frame.cells[ci].getY(),
                                        frame.cells[ci].getZ());
            const float prePcaMaxR = std::max({frame.cells[ci].getARadius(),
                                               frame.cells[ci].getBRadius(),
                                               frame.cells[ci].getCRadius()});
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

            const bool pcaPositionLocked =
                pcaPositionLockedCells.count(sname) > 0;
            const bool updatePosForCell = updatePos && !pcaPositionLocked;
            if (pcaPositionLocked && updatePos) {
                const auto reasonIt = pcaPositionLockReasons.find(sname);
                shapeLogs[ci] << "  [PCA Shape Position Lock] cell=" << sname
                              << " reason="
                              << (reasonIt != pcaPositionLockReasons.end()
                                      ? reasonIt->second
                                      : std::string("unknown"))
                              << std::endl;
            }

            frame.calibrateCellShapeViaPca(ci, others,
                                           pcaMaxIters, pcaScale, pcaMin,
                                           maskScale, convR, convAng,
                                           updatePosForCell, posShiftCap,
                                           maskA, maskB, maskC,
                                           &shapeLogs[ci]);
            // 2026-06-03 center-offset diagnostic guard. This block is only
            // active when pcaShapeCenterDriftGuardEnabled is explicitly set in
            // YAML. It was added to test whether visual center drift comes from
            // the PCA shape pass; it is not part of the default verified path.
            if (updatePosForCell && !isTrashCell &&
                config.cell && config.cell->pcaShapeCenterDriftGuardEnabled) {
                const cv::Point3f postPcaPos(frame.cells[ci].getX(),
                                             frame.cells[ci].getY(),
                                             frame.cells[ci].getZ());
                const cv::Point3f drift = postPcaPos - prePcaPos;
                const float driftMove =
                    static_cast<float>(cv::norm(drift));
                const float driftXY =
                    std::sqrt(drift.x * drift.x + drift.y * drift.y);
                const float maxMove =
                    std::max(0.0f,
                             config.cell->pcaShapeCenterDriftGuardMaxMove);
                const float maxAbsZ =
                    std::max(0.0f,
                             config.cell->pcaShapeCenterDriftGuardMaxAbsZShift);
                const float maxNegativeZ =
                    std::max(0.0f,
                             config.cell->pcaShapeCenterDriftGuardMaxNegativeZShift);
                const float maxAbsXY =
                    std::max(0.0f,
                             config.cell->pcaShapeCenterDriftGuardMaxAbsXYShift);
                const bool moveTooLarge =
                    maxMove > 0.0f && driftMove > maxMove;
                const bool zTooLarge =
                    maxAbsZ > 0.0f && std::abs(drift.z) > maxAbsZ;
                const bool negativeZTooLarge =
                    maxNegativeZ > 0.0f && drift.z < -maxNegativeZ;
                const bool xyTooLarge =
                    maxAbsXY > 0.0f && driftXY > maxAbsXY;
                if (moveTooLarge || zTooLarge || negativeZTooLarge || xyTooLarge) {
                    frame.cells[ci].setPosition(prePcaPos.x,
                                                prePcaPos.y,
                                                prePcaPos.z);
                    shapeLogs[ci]
                        << "  [PCA Shape Center Drift Guard] cell=" << sname
                        << " move=" << driftMove
                        << " xy=" << driftXY
                        << " dz=" << drift.z
                        << " maxMove=" << maxMove
                        << " maxAbsZ=" << maxAbsZ
                        << " maxNegativeZ=" << maxNegativeZ
                        << " maxAbsXY=" << maxAbsXY
                        << " action=revert_position_keep_shape"
                        << std::endl;
                }
            }
            const bool hasPreviousSnapshot =
                previousSnapshots.find(sname) != previousSnapshots.end();
            if (updatePosForCell && !isTrashCell && hasPreviousSnapshot) {
                const cv::Point3f postPcaPos(frame.cells[ci].getX(),
                                             frame.cells[ci].getY(),
                                             frame.cells[ci].getZ());
                const float cumulativeMove =
                    static_cast<float>(cv::norm(postPcaPos - prePcaPos));
                const bool pcaPositionGuardRelaxed =
                    pcaPositionGuardRelaxedCells.count(sname) > 0;
                const bool pcaPositionGuardWideRelaxed =
                    pcaPositionGuardWideRelaxedCells.count(sname) > 0;
                const float normalCumulativeMoveLimit =
                    std::max(10.0f, std::min(14.0f, prePcaMaxR * 0.55f));
                const float wideCumulativeMoveLimit = std::max(
                    20.0f,
                    std::min(
                        std::max(
                            20.0f,
                            config.cellLumen.fusionTemporalCenterRepairMaxDistance),
                        std::max(20.0f, prePcaMaxR * 2.0f)));
                const float cumulativeMoveLimit =
                    pcaPositionGuardWideRelaxed
                        ? wideCumulativeMoveLimit
                        : pcaPositionGuardRelaxed
                        ? std::max(10.0f,
                                   std::min(20.0f, prePcaMaxR * 0.85f))
                        : normalCumulativeMoveLimit;
                if (cumulativeMove > cumulativeMoveLimit) {
                    frame.cells[ci].setPosition(prePcaPos.x, prePcaPos.y, prePcaPos.z);
                    pcaGuardRevertedNames[ci] = sname;
                    shapeLogs[ci] << "  [PCA Shape Position Guard] cell=" << sname
                                  << " cumulativeMove=" << cumulativeMove
                                  << " limit=" << cumulativeMoveLimit
                                  << " action=revert_position_keep_shape"
                                  << std::endl;
                } else if (pcaPositionGuardRelaxed) {
                    const auto reasonIt =
                        pcaPositionGuardRelaxReasons.find(sname);
                    shapeLogs[ci] << "  [PCA Shape Position Guard] cell=" << sname
                                  << " cumulativeMove=" << cumulativeMove
                                  << " normalLimit="
                                  << normalCumulativeMoveLimit
                                  << " relaxedLimit=" << cumulativeMoveLimit
                                  << " action=keep_relaxed_position"
                                  << " reason="
                                  << (reasonIt !=
                                              pcaPositionGuardRelaxReasons.end()
                                          ? reasonIt->second
                                          : std::string("unknown"))
                                  << std::endl;
                }
            }
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
            if (!pcaGuardRevertedNames[ci].empty()) {
                pcaPositionGuardRevertedCells.insert(pcaGuardRevertedNames[ci]);
            }
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
    std::map<std::string, std::pair<cv::Point3f, cv::Point3f>> snapshotSeedDaughters;
    std::map<std::string, float> expectedDaughterMaxShift;
    std::map<std::string, int> expectedDaughterKeptPixels;
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
        snapshotSeedDaughters[name] = {seed1, seed2};
        expectedDaughterMaxShift[name] = 0.0f;
    }

    // ---- Pre-pass: image-ground the seeds for ALL cells ----
    // Runs PCA on current-frame bright pixels around each cell to find the
    // midpoint + direction connecting the two bright blobs (if any). This
    // image-grounded direction is the primary signal when snap axis is
    // unreliable (near-round snap = arbitrary axC direction). For single-
    // blob cells, PCA returns the blob's principal direction — harmless,
    // cost gate still rejects because there's no valley.
    const int prePassRounds = std::max(1, config.prob.expected_daughter_pre_pass_iterations);
    ScopedStageTimer prePassTimer(displayFrame, "pre_pass_image_grounded_pca");
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
                size_t claimPointCount = 0;
                for (const auto &kv : others) {
                    claimPointCount += kv.second.size();
                }
                constexpr bool verbosePrePassClaimNames = false;
                std::ostringstream claimLog;
                if (verbosePrePassClaimNames) {
                    for (const auto &kv : others) {
                        claimLog << " " << kv.first.substr(0, 8)
                                 << "[" << kv.second.size() << "]";
                    }
                }
                std::ostringstream line;
                line << "  [Pre-Pass Claims] " << name.substr(0, 8)
                     << " nNeighbors=" << others.size()
                     << " claim_points=" << claimPointCount;
                if (verbosePrePassClaimNames) {
                    line << " claims:" << claimLog.str();
                }
                line << "\n";
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
                const float shift1 = static_cast<float>(
                    cv::norm(results[ci].groundedD1 - results[ci].oldD1));
                const float shift2 = static_cast<float>(
                    cv::norm(results[ci].groundedD2 - results[ci].oldD2));
                expectedDaughters[name] = {results[ci].groundedD1, results[ci].groundedD2};
                expectedDaughterKeptPixels[name] = results[ci].keptPixels;
                expectedDaughterMaxShift[name] = std::max(shift1, shift2);
            }
            std::cout << results[ci].claimsLog << results[ci].resultLog;
        }
    }
    prePassTimer.finish();

    std::set<std::string> forcedSplitNames;
    if (allowSplits && config.prob.expected_daughter_force_split_enabled) {
        const float configuredMinFraction =
            config.prob.expected_daughter_force_min_separation_parent_fraction;
        const float minSepFraction = configuredMinFraction > 0.0f
            ? configuredMinFraction
            : config.prob.bio_min_daughter_separation_parent_fraction;
        const int minKeptPixels =
            std::max(0, config.prob.expected_daughter_force_min_kept_pixels);

        int candidatesChecked = 0;
        for (const auto &cell : frame.cells) {
            const std::string &name = cell.getName();
            if (cell.isTrash()) continue;

            auto snapIt = previousSnapshots.find(name);
            auto dIt = expectedDaughters.find(name);
            auto keptIt = expectedDaughterKeptPixels.find(name);
            if (snapIt == previousSnapshots.end() || !snapIt->second.valid ||
                dIt == expectedDaughters.end() || keptIt == expectedDaughterKeptPixels.end()) {
                continue;
            }
            ++candidatesChecked;
            if (keptIt->second < minKeptPixels) continue;

            const auto &snap = snapIt->second;
            const cv::Point3f delta = dIt->second.second - dIt->second.first;
            const float separation = static_cast<float>(cv::norm(delta));
            const float parentMaxRadius = std::max({snap.aRadius, snap.bRadius, snap.cRadius});
            if (parentMaxRadius <= 1e-3f) continue;

            const float minSeparation = parentMaxRadius * minSepFraction;
            if (separation < minSeparation) continue;

            forcedSplitNames.insert(name);
            const float ratio = separation / parentMaxRadius;
            const float pSplit = splitProbabilities.count(name) > 0
                ? splitProbabilities[name]
                : 0.0f;
            std::cout << "[Pre-Pass Force Split] frame " << displayFrame
                      << " cell=" << name
                      << " separation=" << separation
                      << " parentMaxR=" << parentMaxRadius
                      << " ratio=" << ratio
                      << " minRatio=" << minSepFraction
                      << " kept=" << keptIt->second
                      << " P(split)=" << pSplit
                      << std::endl;
        }

        std::cout << "[Pre-Pass Force Split] frame " << displayFrame
                  << " checked=" << candidatesChecked
                  << " forced=" << forcedSplitNames.size()
                  << " minRatio=" << minSepFraction
                  << " minKept=" << minKeptPixels
                  << std::endl;
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

    std::unordered_map<std::string, BridgeSplitProposal> combinedLumenSplitPriors;
    const std::unordered_map<std::string, BridgeSplitProposal> *lumenSplitPriorsForFrame = nullptr;
    auto lumenPriorIt = cellLumenSplitPriors.find(frameIndex);
    const bool lumenSplitPriorPrepared =
        config.cellLumen.fusionSplitPriorEnabled &&
        lumenPriorIt != cellLumenSplitPriors.end();
    if (lumenSplitPriorPrepared && !lumenPriorIt->second.empty()) {
        combinedLumenSplitPriors = lumenPriorIt->second;
    }

    if (config.cellLumen.fusionSplitPriorEnabled &&
        config.cellLumen.fusionSplitPriorPrepassFallbackEnabled &&
        config.cellLumen.fusionSplitPriorPrepassFallbackMaxPriors > 0) {
        ScopedStageTimer fallbackTimer(displayFrame,
                                       "cell_lumen_prepass_fallback");
        struct PrepassFallbackCandidate {
            std::string parentName;
            BridgeSplitProposal proposal;
            float score = 0.0f;
            float separationRatio = 0.0f;
            float midpointDistance = 0.0f;
            float parentShape = 1.0f;
            float d1ParentMargin = 0.0f;
            float d2ParentMargin = 0.0f;
            float sourceMaxShift = 0.0f;
            int keptPixels = 0;
            int parentAgeFrames = std::numeric_limits<int>::max() / 4;
            bool overridesExistingPrior = false;
            int existingWindowBoth = -1;
            int existingWindowMissing = -1;
            int existingWindowParentPersists = -1;
            std::string source = "prepass_pca";
        };

        std::vector<PrepassFallbackCandidate> fallbackCandidates;
        int fallbackConsidered = 0;
        int fallbackRejectedExistingPrior = 0;
        int fallbackRejectedMissingData = 0;
        int fallbackRejectedKept = 0;
        int fallbackRejectedShape = 0;
        int fallbackRejectedSeparation = 0;
        int fallbackRejectedClaim = 0;
        int fallbackRejectedScore = 0;
        int fallbackRejectedBadLumenParent = 0;
        int fallbackRejectedCollapsedCenterParent = 0;
        int fallbackRejectedParentAge = 0;
        int fallbackOverrodeExistingPrior = 0;
        const int fallbackMinParentAgeFrames = std::max(
            0,
            config.cellLumen.fusionSplitPriorPrepassFallbackMinParentAgeFrames);

        const auto badParentIt = cellLumenSplitPriorRejectedBadParents.find(frameIndex);
        const std::set<std::string> *badLumenParentsForFrame =
            badParentIt != cellLumenSplitPriorRejectedBadParents.end()
                ? &badParentIt->second
                : nullptr;
        const auto collapsedParentIt = cellLumenCollapsedCenterParents.find(frameIndex);
        const std::set<std::string> *collapsedCenterParentsForFrame =
            collapsedParentIt != cellLumenCollapsedCenterParents.end()
                ? &collapsedParentIt->second
                : nullptr;

        const auto nearestOtherDistance = [&](const std::string &selfName,
                                              const cv::Point3f &point) -> float {
            float best = std::numeric_limits<float>::max();
            for (const auto &other : frame.cells) {
                if (other.isTrash() || other.getName() == selfName) continue;
                const cv::Point3f otherPos(other.getX(), other.getY(), other.getZ());
                best = std::min(best, static_cast<float>(cv::norm(point - otherPos)));
            }
            return best;
        };
        const auto weakExistingPriorAllowsPrepassFallback =
            [&](const BridgeSplitProposal &proposal) {
                if (!config.cellLumen
                         .fusionSplitPriorPrepassFallbackOverrideWeakExistingPriorEnabled) {
                    return false;
                }
                // f086 showed that an existing Cell Lumen prior can be weak in
                // the future window while the pre-pass PCA shape already found a
                // cleaner two-daughter split. Only those weak-window priors are
                // eligible for replacement; clean future-backed priors stay in
                // control.
                return proposal.windowBothDaughtersSupported <
                           std::max(
                               0,
                               config.cellLumen
                                   .fusionSplitPriorPrepassFallbackOverrideMinExistingWindowBoth) ||
                       proposal.windowMissingDaughterCount >
                           std::max(
                               0,
                               config.cellLumen
                                   .fusionSplitPriorPrepassFallbackOverrideMaxExistingWindowMissing) ||
                       proposal.windowParentPersists >
                           std::max(
                               0,
                               config.cellLumen
                                   .fusionSplitPriorPrepassFallbackOverrideMaxExistingWindowParentPersists);
            };

        for (size_t cellIdx = 0; cellIdx < frame.cells.size(); ++cellIdx) {
            const auto &cell = frame.cells[cellIdx];
            if (cell.isTrash()) continue;
            const std::string parentName = cell.getName();
            ++fallbackConsidered;
            int parentAgeFrames = std::numeric_limits<int>::max() / 4;
            if (const auto firstSeenIt = cellFirstSeenFrame.find(parentName);
                firstSeenIt != cellFirstSeenFrame.end()) {
                parentAgeFrames = std::max(0, displayFrame - firstSeenIt->second);
            }
            if (fallbackMinParentAgeFrames > 0 &&
                parentAgeFrames < fallbackMinParentAgeFrames) {
                ++fallbackRejectedParentAge;
                continue;
            }

            const auto existingPriorIt =
                combinedLumenSplitPriors.find(parentName);
            const bool hasExistingPrior =
                existingPriorIt != combinedLumenSplitPriors.end();
            const bool existingPriorCanBeOverridden =
                hasExistingPrior &&
                weakExistingPriorAllowsPrepassFallback(existingPriorIt->second);
            if (hasExistingPrior && !existingPriorCanBeOverridden) {
                ++fallbackRejectedExistingPrior;
                continue;
            }
            if (config.cellLumen.fusionSplitPriorPrepassFallbackRejectBadLumenParent &&
                badLumenParentsForFrame != nullptr &&
                badLumenParentsForFrame->count(parentName) > 0) {
                ++fallbackRejectedBadLumenParent;
                continue;
            }
            if (config.cellLumen
                    .fusionSplitPriorPrepassFallbackRejectCollapsedCenterParent &&
                collapsedCenterParentsForFrame != nullptr &&
                collapsedCenterParentsForFrame->count(parentName) > 0) {
                ++fallbackRejectedCollapsedCenterParent;
                std::cout
                    << "[CellLumen Prepass Fallback Reject CollapsedCenterParent] frame "
                    << (firstFrame + frameIndex)
                    << " parent=" << parentName
                    << " reason=center_prior_already_collapsed_internal_peaks"
                    << std::endl;
                continue;
            }

            auto snapIt = previousSnapshots.find(parentName);
            auto daughterIt = expectedDaughters.find(parentName);
            auto seedIt = snapshotSeedDaughters.find(parentName);
            auto shiftIt = expectedDaughterMaxShift.find(parentName);
            auto keptIt = expectedDaughterKeptPixels.find(parentName);
            if (snapIt == previousSnapshots.end() || !snapIt->second.valid ||
                daughterIt == expectedDaughters.end() || seedIt == snapshotSeedDaughters.end() ||
                keptIt == expectedDaughterKeptPixels.end()) {
                ++fallbackRejectedMissingData;
                continue;
            }

            const int keptPixels = keptIt->second;
            if (keptPixels < config.cellLumen.fusionSplitPriorPrepassFallbackMinKeptPixels) {
                ++fallbackRejectedKept;
                continue;
            }

            const auto &snap = snapIt->second;
            const float parentShape = std::max(cell.shapeElongation(), snap.shapeElongation);
            if (parentShape < config.cellLumen.fusionSplitPriorPrepassFallbackMinShape) {
                ++fallbackRejectedShape;
                continue;
            }

            const float parentMaxRadius = std::max({snap.aRadius, snap.bRadius, snap.cRadius});
            const float sourceMaxShift = shiftIt != expectedDaughterMaxShift.end()
                ? shiftIt->second
                : 0.0f;
            cv::Point3f d1 = daughterIt->second.first;
            cv::Point3f d2 = daughterIt->second.second;
            std::string source = "prepass_pca";
            if (config.cellLumen.fusionSplitPriorPrepassFallbackUseSnapshotSeedOnLargeDrift &&
                sourceMaxShift > config.cellLumen.fusionSplitPriorPrepassFallbackSeedMaxShift &&
                parentShape >= config.cellLumen.fusionSplitPriorPrepassFallbackSeedMinShape &&
                parentMaxRadius > 1e-3f) {
                const cv::Point3f seedDelta = seedIt->second.second - seedIt->second.first;
                const float seedSeparation = static_cast<float>(cv::norm(seedDelta));
                const float seedRatio = seedSeparation / parentMaxRadius;
                if (seedRatio >= config.cellLumen.fusionSplitPriorPrepassFallbackSeedMinSeparationRadiusScale &&
                    seedRatio <= config.cellLumen.fusionSplitPriorPrepassFallbackSeedMaxSeparationRadiusScale) {
                    d1 = seedIt->second.first;
                    d2 = seedIt->second.second;
                    source = "snapshot_seed_large_shift";
                }
            }
            const cv::Point3f delta = d2 - d1;
            const float separation = static_cast<float>(cv::norm(delta));
            if (parentMaxRadius <= 1e-3f || separation <= 1e-3f) {
                ++fallbackRejectedSeparation;
                continue;
            }

            const float separationRatio = separation / parentMaxRadius;
            const float minSeparationRatio = source == "snapshot_seed_large_shift"
                ? config.cellLumen.fusionSplitPriorPrepassFallbackSeedMinSeparationRadiusScale
                : config.cellLumen.fusionSplitPriorPrepassFallbackMinSeparationRadiusScale;
            const float maxSeparationRatio = source == "snapshot_seed_large_shift"
                ? config.cellLumen.fusionSplitPriorPrepassFallbackSeedMaxSeparationRadiusScale
                : config.cellLumen.fusionSplitPriorPrepassFallbackMaxSeparationRadiusScale;
            float effectiveMaxSeparationRatio = maxSeparationRatio;
            if (existingPriorCanBeOverridden &&
                config.cellLumen
                        .fusionSplitPriorPrepassFallbackOverrideMaxSeparationRadiusScale >
                    0.0f) {
                // Only weak-window existing priors may use this looser ratio.
                // This protects normal fallback from becoming a broad split
                // source while allowing f086-like PCA evidence to replace a
                // bad current Cell Lumen pair.
                effectiveMaxSeparationRatio = std::max(
                    effectiveMaxSeparationRatio,
                    config.cellLumen
                        .fusionSplitPriorPrepassFallbackOverrideMaxSeparationRadiusScale);
            }
            if (separationRatio < minSeparationRatio ||
                separationRatio > effectiveMaxSeparationRatio) {
                ++fallbackRejectedSeparation;
                continue;
            }

            const cv::Point3f parentPos(cell.getX(), cell.getY(), cell.getZ());
            const cv::Point3f midpoint = 0.5f * (d1 + d2);
            const float d1ParentDist = static_cast<float>(cv::norm(d1 - parentPos));
            const float d2ParentDist = static_cast<float>(cv::norm(d2 - parentPos));
            const float d1NearestOther = nearestOtherDistance(parentName, d1);
            const float d2NearestOther = nearestOtherDistance(parentName, d2);
            const float d1ParentMargin = d1NearestOther - d1ParentDist;
            const float d2ParentMargin = d2NearestOther - d2ParentDist;
            if (d1ParentMargin < config.cellLumen.fusionSplitPriorPrepassFallbackParentClaimMargin ||
                d2ParentMargin < config.cellLumen.fusionSplitPriorPrepassFallbackParentClaimMargin) {
                ++fallbackRejectedClaim;
                continue;
            }

            const float midpointDistance = static_cast<float>(cv::norm(midpoint - parentPos));
            const float score =
                midpointDistance +
                std::abs(separationRatio - 1.0f) * 25.0f -
                std::max(0.0f, parentShape - config.cellLumen.fusionSplitPriorPrepassFallbackMinShape) * 5.0f;
            const float maxScoreForSource = source == "snapshot_seed_large_shift"
                ? config.cellLumen.fusionSplitPriorPrepassFallbackSeedMaxScore
                : config.cellLumen.fusionSplitPriorPrepassFallbackMaxScore;
            if (score > maxScoreForSource) {
                ++fallbackRejectedScore;
                continue;
            }
            bool overridesExistingPrior = false;
            int existingWindowBoth = -1;
            int existingWindowMissing = -1;
            int existingWindowParentPersists = -1;
            if (hasExistingPrior) {
                existingWindowBoth =
                    existingPriorIt->second.windowBothDaughtersSupported;
                existingWindowMissing =
                    existingPriorIt->second.windowMissingDaughterCount;
                existingWindowParentPersists =
                    existingPriorIt->second.windowParentPersists;
                overridesExistingPrior =
                    existingPriorCanBeOverridden &&
                    parentShape >=
                        std::max(
                            1.0f,
                            config.cellLumen
                                .fusionSplitPriorPrepassFallbackOverrideMinParentShape) &&
                    score <=
                        std::max(
                            0.0f,
                            config.cellLumen
                                .fusionSplitPriorPrepassFallbackOverrideMaxScore);
                if (!overridesExistingPrior) {
                    ++fallbackRejectedExistingPrior;
                    continue;
                }
            }

            BridgeSplitProposal proposal;
            proposal.d1Pos = d1;
            proposal.d2Pos = d2;
            proposal.elongation = score;
            proposal.parentShapeElongation = parentShape;
            proposal.elongatedParentRescued =
                parentShape >= config.cellLumen.fusionSplitPriorPositiveGateElongatedParentMinShape;
            // -2 marks a normal pre-pass fallback proposal.  -3 is a stricter
            // marker for the snapshot-seed fallback path, where the PCA
            // daughter centroid drifted far away and the original split seeds
            // are more trustworthy than the contaminated PCA centroids.
            proposal.gapStartBin = (source == "snapshot_seed_large_shift") ? -3 : -2;
            proposal.gapEndBin = proposal.gapStartBin;
            proposal.leftPixelCount = keptPixels / 2;
            proposal.rightPixelCount = keptPixels - proposal.leftPixelCount;

            fallbackCandidates.push_back(PrepassFallbackCandidate{
                parentName,
                proposal,
                score,
                separationRatio,
                midpointDistance,
                parentShape,
                d1ParentMargin,
                d2ParentMargin,
                sourceMaxShift,
                keptPixels,
                parentAgeFrames,
                overridesExistingPrior,
                existingWindowBoth,
                existingWindowMissing,
                existingWindowParentPersists,
                source
            });
        }

        std::sort(fallbackCandidates.begin(), fallbackCandidates.end(),
                  [](const PrepassFallbackCandidate &a, const PrepassFallbackCandidate &b) {
                      if (std::abs(a.score - b.score) > 1e-6f) return a.score < b.score;
                      if (std::abs(a.parentShape - b.parentShape) > 1e-6f) {
                          return a.parentShape > b.parentShape;
                      }
                      const bool aSeed = a.source == "snapshot_seed_large_shift";
                      const bool bSeed = b.source == "snapshot_seed_large_shift";
                      if (aSeed != bSeed) return !aSeed;
                      return a.parentName < b.parentName;
                  });

        const int maxFallbackPriors =
            std::max(0, config.cellLumen.fusionSplitPriorPrepassFallbackMaxPriors);
        int fallbackAdded = 0;
        std::set<std::string> selectedFallbackParents;
        for (const auto &candidate : fallbackCandidates) {
            if (fallbackAdded >= maxFallbackPriors) break;
            const bool overridingExistingPrior =
                combinedLumenSplitPriors.count(candidate.parentName) > 0 &&
                candidate.overridesExistingPrior;
            if (combinedLumenSplitPriors.count(candidate.parentName) > 0 &&
                !candidate.overridesExistingPrior) {
                continue;
            }
            combinedLumenSplitPriors[candidate.parentName] = candidate.proposal;
            ++fallbackAdded;
            if (overridingExistingPrior) {
                ++fallbackOverrodeExistingPrior;
            }
            selectedFallbackParents.insert(candidate.parentName);
            std::cout << "[CellLumen Prepass Fallback Prior] frame " << (firstFrame + frameIndex)
                      << " parent=" << candidate.parentName
                      << " score=" << candidate.score
                      << " sepRatio=" << candidate.separationRatio
                      << " midpointDist=" << candidate.midpointDistance
                      << " parentShape=" << candidate.parentShape
                      << " parentAgeFrames=" << candidate.parentAgeFrames
                      << " minParentAge=" << fallbackMinParentAgeFrames
                      << " overrideWeakExistingPrior="
                      << (candidate.overridesExistingPrior ? 1 : 0)
                      << " existingWindowBoth="
                      << candidate.existingWindowBoth
                      << " existingWindowMissing="
                      << candidate.existingWindowMissing
                      << " existingWindowParentPersists="
                      << candidate.existingWindowParentPersists
                      << " kept=" << candidate.keptPixels
                      << " d1ParentMargin=" << candidate.d1ParentMargin
                      << " d2ParentMargin=" << candidate.d2ParentMargin
                      << " source=" << candidate.source
                      << " sourceMaxShift=" << candidate.sourceMaxShift
                      << " d1=(" << candidate.proposal.d1Pos.x << "," << candidate.proposal.d1Pos.y << "," << candidate.proposal.d1Pos.z << ")"
                      << " d2=(" << candidate.proposal.d2Pos.x << "," << candidate.proposal.d2Pos.y << "," << candidate.proposal.d2Pos.z << ")"
                      << std::endl;
        }

        if (candidateGraphLog) {
            for (const auto &candidate : fallbackCandidates) {
                CandidateGraphRow row;
                row.frame = displayFrame;
                row.kind = "split_pair";
                row.source = candidate.source;
                row.parent = candidate.parentName;
                row.selected = selectedFallbackParents.count(candidate.parentName) > 0 ? 1 : 0;
                row.score = candidate.score;
                row.sep = candidate.separationRatio;
                row.midpointDist = candidate.midpointDistance;
                row.parentShape = candidate.parentShape;
                row.parentDistNear = candidate.d1ParentMargin;
                row.parentDistFar = candidate.d2ParentMargin;
                row.bridgeMetric = candidate.sourceMaxShift;
                row.d1 = candidate.proposal.d1Pos;
                row.d2 = candidate.proposal.d2Pos;
                row.voxA = candidate.proposal.leftPixelCount;
                row.voxB = candidate.proposal.rightPixelCount;
                row.note = "prepass_fallback_sep_column_is_radius_ratio"
                           ";parent_age_frames=" +
                           std::to_string(candidate.parentAgeFrames) +
                           ";min_parent_age=" +
                           std::to_string(fallbackMinParentAgeFrames) +
                           ";override_weak_existing_prior=" +
                           (candidate.overridesExistingPrior ? "1" : "0") +
                           ";existing_window_both=" +
                           std::to_string(candidate.existingWindowBoth) +
                           ";existing_window_missing=" +
                           std::to_string(candidate.existingWindowMissing) +
                           ";existing_window_parent_persists=" +
                           std::to_string(candidate.existingWindowParentPersists);
                writeCandidateGraphRow(candidateGraphLog, row);
            }
        }

        std::cout << "[CellLumen Prepass Fallback Summary] frame " << (firstFrame + frameIndex)
                  << " considered=" << fallbackConsidered
                  << " ranked=" << fallbackCandidates.size()
                  << " added=" << fallbackAdded
                  << " rejected_existing_prior=" << fallbackRejectedExistingPrior
                  << " rejected_missing_data=" << fallbackRejectedMissingData
                  << " rejected_kept=" << fallbackRejectedKept
                  << " rejected_shape=" << fallbackRejectedShape
                  << " rejected_separation=" << fallbackRejectedSeparation
                  << " rejected_claim=" << fallbackRejectedClaim
                  << " rejected_score=" << fallbackRejectedScore
                  << " rejected_bad_lumen_parent=" << fallbackRejectedBadLumenParent
                  << " rejected_collapsed_center_parent="
                  << fallbackRejectedCollapsedCenterParent
                  << " rejected_parent_age=" << fallbackRejectedParentAge
                  << " overrode_existing_prior="
                  << fallbackOverrodeExistingPrior
                  << std::endl;
    }

    if (!combinedLumenSplitPriors.empty()) {
        lumenSplitPriorsForFrame = &combinedLumenSplitPriors;
        std::cout << "[CellLumen SplitPrior Ready] frame " << (firstFrame + frameIndex)
                  << " priors=" << lumenSplitPriorsForFrame->size()
                  << " guided_only=" << config.cellLumen.fusionSplitPriorGuidedOnly
                  << " skip_random=" << config.cellLumen.fusionSplitPriorSkipRandomSplits
                  << std::endl;
    }
    const bool lumenSplitPriorActive =
        config.cellLumen.fusionSplitPriorEnabled &&
        lumenSplitPriorsForFrame != nullptr &&
        !lumenSplitPriorsForFrame->empty();
    const bool lumenSplitRandomDisabled =
        lumenSplitPriorPrepared &&
        config.cellLumen.fusionSplitPriorSkipRandomSplits;
    if (lumenSplitRandomDisabled) {
        std::cout << "[CellLumen SplitPrior Mode] frame " << (firstFrame + frameIndex)
                  << " random_splits_disabled=1"
                  << " priors=" << (lumenSplitPriorsForFrame ? lumenSplitPriorsForFrame->size() : 0)
                  << " reason=deterministic_cell_lumen_split_config"
                  << std::endl;
        if (!lumenSplitPriorActive) {
            std::cout << "[CellLumen SplitPrior Mode] frame " << (firstFrame + frameIndex)
                      << " no_cell_lumen_split_priors=1"
                      << " action=skip_classic_random_split"
                      << " reason=cell_lumen_high_recall_found_no_ranked_parent_pair"
                      << std::endl;
        }
    }

    const auto lumenCenterCandidatesIt = cellLumenCenterCandidates.find(frameIndex);
    const std::unordered_map<std::string, CellLumenCenterCandidate> *lumenCenterCandidatesForFrame =
        lumenCenterCandidatesIt != cellLumenCenterCandidates.end()
            ? &lumenCenterCandidatesIt->second
            : nullptr;
    if (lumenCenterCandidatesForFrame != nullptr && !lumenCenterCandidatesForFrame->empty()) {
        std::cout << "[CellLumen CenterCandidate Ready] frame " << displayFrame
                  << " candidates=" << lumenCenterCandidatesForFrame->size()
                  << std::endl;
    }

    std::set<int> reservedLumenSplitCandidateIds;
    std::unordered_map<int, std::string> reservedLumenSplitCandidateOwners;
    if (lumenSplitPriorsForFrame != nullptr) {
        for (const auto &entry : *lumenSplitPriorsForFrame) {
            const auto reserveCandidate = [&](int candidateId) {
                if (candidateId < 0) {
                    return;
                }
                reservedLumenSplitCandidateIds.insert(candidateId);
                reservedLumenSplitCandidateOwners[candidateId] = entry.first;
            };
            reserveCandidate(entry.second.candidateIdA);
            reserveCandidate(entry.second.candidateIdB);
        }
    }
    if (!reservedLumenSplitCandidateIds.empty()) {
        std::cout << "[CellLumen SplitPrior Reserved Candidates] frame "
                  << displayFrame
                  << " count=" << reservedLumenSplitCandidateIds.size()
                  << std::endl;
    }
    if (lumenSplitPriorsForFrame != nullptr) {
        int relaxedBlockers = 0;
        for (const auto &entry : *lumenSplitPriorsForFrame) {
            const BridgeSplitProposal &proposal = entry.second;
            if (!proposal.futureContinuationConflictRescued ||
                proposal.continuationClaimBlockerNames.empty()) {
                continue;
            }
            std::stringstream blockers(proposal.continuationClaimBlockerNames);
            std::string blockerName;
            while (std::getline(blockers, blockerName, '|')) {
                const bool wasAlreadyRelaxed =
                    pcaPositionGuardRelaxedCells.count(blockerName) > 0;
                relaxPcaPositionGuard(
                    blockerName,
                    "future_window_split_continuation_conflict");
                if (!wasAlreadyRelaxed) {
                    ++relaxedBlockers;
                }
            }
        }
        if (relaxedBlockers > 0) {
            std::cout << "[PCA Shape Position Guard Relax] frame "
                      << displayFrame
                      << " cells=" << relaxedBlockers
                      << " reason=future_window_split_continuation_conflict"
                      << std::endl;
        }
    }
    if (lumenCenterCandidatesForFrame != nullptr &&
        !reservedLumenSplitCandidateIds.empty()) {
        for (const auto &entry : *lumenCenterCandidatesForFrame) {
            const auto &centerCandidate = entry.second;
            if (centerCandidate.candidateId >= 0 &&
                reservedLumenSplitCandidateIds.count(centerCandidate.candidateId) > 0) {
                lockPcaPosition(entry.first, "reserved_lumen_split_candidate");
            }
        }
        if (!pcaPositionLockedCells.empty()) {
            std::cout << "[CellLumen SplitPrior Continuation Locks] frame "
                      << displayFrame
                      << " cells=" << pcaPositionLockedCells.size()
                      << " reason=reserved_lumen_split_candidate"
                      << std::endl;
        }
    }
    if (lumenCenterCandidatesForFrame != nullptr) {
        int elongatedCenterAnchorLocks = 0;
        int temporalElongatedCenterAnchorLocks = 0;
        int moderateAnchorGuardRelaxed = 0;
        auto futureWindowSupportForCenter = [&](const cv::Point3f &point) {
            int support = 0;
            const int windowSize =
                std::clamp(config.cellLumen.fusionSplitPriorWindowSize, 2, 5);
            for (int offset = 1; offset < windowSize; ++offset) {
                const int lookaheadFrameIndex = frameIndex + offset;
                if (lookaheadFrameIndex < 0 ||
                    static_cast<size_t>(lookaheadFrameIndex) >= frames.size()) {
                    break;
                }
                const auto &futureCandidates =
                    getCellLumenLookaheadCandidates(lookaheadFrameIndex);
                if (futureCandidates.empty()) {
                    continue;
                }
                float best = std::numeric_limits<float>::infinity();
                for (const auto &future : futureCandidates) {
                    best = std::min(
                        best,
                        static_cast<float>(cv::norm(future.position - point)));
                }
                const float matchDistance =
                    std::max(0.0f,
                             config.cellLumen.fusionSplitPriorWindowMatchDistance) +
                    std::max(0.0f,
                             config.cellLumen
                                 .fusionSplitPriorWindowMatchDistancePerFrame) *
                        static_cast<float>(std::max(0, offset - 1));
                if (best <= matchDistance) {
                    ++support;
                }
            }
            return support;
        };
        for (const auto &entry : *lumenCenterCandidatesForFrame) {
            auto cellIt = std::find_if(
                frame.cells.begin(),
                frame.cells.end(),
                [&](const Ellipsoid &cell) {
                    return cell.getName() == entry.first;
                });
            if (cellIt == frame.cells.end()) {
                continue;
            }
            const auto &centerCandidate = entry.second;
            const float maxR = std::max({cellIt->getARadius(),
                                         cellIt->getBRadius(),
                                         cellIt->getCRadius()});
            const float normalMoveLimit =
                std::max(10.0f, std::min(14.0f, maxR * 0.55f));
            const float lockDistance = std::min(
                std::max(0.0f, config.cellLumen.fusionCenterPriorMaxDistance),
                std::max(8.0f, maxR * 0.75f));
            const float minSignal = std::max(
                100.0f,
                std::max(0.0f, config.cellLumen.fusionMinTop10MinusShell));
            const bool highConfidenceCenter =
                centerCandidate.distance <= lockDistance &&
                centerCandidate.signal >= minSignal &&
                centerCandidate.voxelCount >=
                    std::max(0, config.cellLumen.fusionMinVoxels);
            const bool elongatedCell =
                cellIt->shapeElongation() >=
                std::max(1.85f,
                         config.cellLumen.fusionSplitPriorElongatedParentMinShape + 0.35f);
            const int temporalMinWindowSupport =
                std::max(1,
                         config.cellLumen
                             .fusionTemporalCenterRepairMinWindowSupport);
            const float temporalLockDistance = std::min(
                std::max(0.0f, config.cellLumen.fusionCenterPriorMaxDistance),
                std::max(16.0f, maxR * 0.90f));
            const float temporalMinSignal = std::max(
                std::max(0.0f, config.cellLumen.fusionMinTop10MinusShell),
                config.cellLumen.fusionTemporalCenterRepairOldSupportedMinSignal > 0.0f
                    ? config.cellLumen.fusionTemporalCenterRepairOldSupportedMinSignal
                    : 0.0f);
            const int temporalMinVoxels =
                std::max(0, config.cellLumen.fusionMinVoxels);
            const int temporalWindowSupport =
                futureWindowSupportForCenter(centerCandidate.position);
            const bool temporallySupportedElongatedCenter =
                cellIt->shapeElongation() >=
                    std::max(2.25f,
                             config.cellLumen
                                     .fusionSplitPriorElongatedParentMinShape +
                                 0.75f) &&
                centerCandidate.distance <= temporalLockDistance &&
                centerCandidate.distance >= std::max(8.0f, normalMoveLimit * 0.80f) &&
                centerCandidate.signal >= temporalMinSignal &&
                centerCandidate.voxelCount >= temporalMinVoxels &&
                temporalWindowSupport >= temporalMinWindowSupport;
            const bool moderateAnchorRelaxEnabled =
                config.cellLumen
                    .fusionCenterPriorModerateAnchorPcaGuardRelaxEnabled &&
                config.cellLumen
                        .fusionCenterPriorModerateAnchorPcaGuardRelaxMaxCells >
                    0 &&
                static_cast<int>(frame.cells.size()) <=
                    config.cellLumen
                        .fusionCenterPriorModerateAnchorPcaGuardRelaxMaxCells;
            const bool moderateAnchorPcaGuardRelax =
                moderateAnchorRelaxEnabled &&
                centerCandidate.distance <=
                    std::max(
                        0.0f,
                        config.cellLumen
                            .fusionCenterPriorModerateAnchorPcaGuardRelaxMaxDistance) &&
                centerCandidate.signal >=
                    std::max(
                        0.0f,
                        config.cellLumen
                            .fusionCenterPriorModerateAnchorPcaGuardRelaxMinSignal) &&
                centerCandidate.voxelCount >=
                    std::max(
                        0,
                        config.cellLumen
                            .fusionCenterPriorModerateAnchorPcaGuardRelaxMinVoxels) &&
                cellIt->shapeElongation() >=
                    std::max(
                        1.0f,
                        config.cellLumen
                            .fusionCenterPriorModerateAnchorPcaGuardRelaxMinShape);
            if (moderateAnchorPcaGuardRelax) {
                const bool wasAlreadyRelaxed =
                    pcaPositionGuardRelaxedCells.count(entry.first) > 0;
                relaxPcaPositionGuard(entry.first,
                                      "moderate_lumen_center_anchor");
                if (!wasAlreadyRelaxed) {
                    ++moderateAnchorGuardRelaxed;
                }
            }
            if ((!highConfidenceCenter && !temporallySupportedElongatedCenter) ||
                !elongatedCell) {
                continue;
            }
            const bool wasAlreadyLocked =
                pcaPositionLockedCells.count(entry.first) > 0;
            lockPcaPosition(
                entry.first,
                temporallySupportedElongatedCenter
                    ? "elongated_temporal_lumen_center_anchor"
                    : "elongated_cell_lumen_center_anchor");
            if (!wasAlreadyLocked) {
                ++elongatedCenterAnchorLocks;
                if (temporallySupportedElongatedCenter) {
                    ++temporalElongatedCenterAnchorLocks;
                }
            }
        }
        if (elongatedCenterAnchorLocks > 0) {
            std::cout << "[CellLumen Center Anchor PCA Locks] frame "
                      << displayFrame
                      << " cells=" << elongatedCenterAnchorLocks
                      << " temporal_supported="
                      << temporalElongatedCenterAnchorLocks
                      << " reason=elongated_cell_lumen_center_anchor"
                      << std::endl;
        }
        if (moderateAnchorGuardRelaxed > 0) {
            std::cout << "[PCA Shape Position Guard Relax] frame "
                      << displayFrame
                      << " cells=" << moderateAnchorGuardRelaxed
                      << " reason=moderate_lumen_center_anchor"
                      << std::endl;
        }
    }
    if (!pcaPositionLockedCells.empty()) {
        std::cout << "[PCA Shape Position Locks] frame "
                  << displayFrame
                  << " cells=" << pcaPositionLockedCells.size()
                  << std::endl;
    }

    struct AcceptedLumenSplitRecord {
        std::string parentName;
        BridgeSplitProposal proposal;
    };
    std::vector<AcceptedLumenSplitRecord> acceptedLumenSplits;

    // ---- Single-phase iteration helper ----
    auto runPhase = [&](std::set<std::string> &phaseNames,
                        bool phaseB,
                        std::set<std::string> &splitAcceptedInPhase,
                        std::set<std::string> &splitRejectedInPhase) {
        if (phaseNames.empty()) return;
        ScopedStageTimer phaseTimer(displayFrame, "phase_split_and_perturb");

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

    std::set<std::string> lumenClassicFallbackNames;

    auto attemptSplitAtIndex = [&](size_t cellIdx,
                                   const std::string &cellName,
                                   const std::string &scheduleReason,
                                       bool useLumenPrior) {
            ScopedStageTimer splitTimer(displayFrame,
                                        "split_attempt_" + scheduleReason);
            ++splitAttempted;
            std::cout << "[Split Schedule] frame " << displayFrame
                      << " cell=" << cellName
                      << " reason=" << scheduleReason
                      << std::endl;

            auto snapIt = previousSnapshots.find(cellName);
            if (snapIt == previousSnapshots.end() || !snapIt->second.valid) {
                splitBlacklist.insert(cellName);
                std::cout << "[Split Schedule Reject] frame " << displayFrame
                          << " cell=" << cellName
                          << " reason=no_valid_snapshot"
                          << std::endl;
                return;
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
            const BridgeSplitProposal *lumenForCell = nullptr;
            if (useLumenPrior && lumenSplitPriorsForFrame != nullptr) {
                auto lIt = lumenSplitPriorsForFrame->find(cellName);
                if (lIt != lumenSplitPriorsForFrame->end()) {
                    lumenForCell = &lIt->second;
                }
            }
            if (lumenForCell != nullptr) {
                const cv::Point3f delta = lumenForCell->d2Pos - lumenForCell->d1Pos;
                const float len = static_cast<float>(cv::norm(delta));
                if (len > 1e-3f) {
                    splitSnapshot.splitAxisDir = cv::Point3f(
                        delta.x / len, delta.y / len, delta.z / len);
                    splitSnapshot.splitAxisLength = len;
                    splitSnapshot.position = 0.5f * (lumenForCell->d1Pos + lumenForCell->d2Pos);
                }
            } else if (auto itD = expectedDaughters.find(cellName); itD != expectedDaughters.end()) {
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
                bridgeForCell,
                lumenForCell,
                lumenForCell != nullptr && config.cellLumen.fusionSplitPriorGuidedOnly,
                config.cellLumen.fusionSplitPriorBurnInIterations,
                config.cellLumen.fusionSplitPriorRefineIterations,
                config.cellLumen.fusionSplitPriorUseDedicatedCostGate,
                config.cellLumen.fusionSplitPriorUseImageCostGate,
                config.cellLumen.fusionSplitPriorCost,
                config.cellLumen.fusionSplitPriorCostFraction,
                config.cellLumen.fusionSplitPriorMaxPositiveCostFraction,
                config.cellLumen.fusionSplitPriorPositiveGateMinImageGain,
                config.cellLumen.fusionSplitPriorPositiveGateMinImageGainPenaltyRatio,
                config.cellLumen.fusionSplitPriorPositiveGateElongatedParentMinShape,
                config.cellLumen.fusionSplitPriorPositiveGateElongatedMaxRawWorsening,
                config.cellLumen.fusionSplitPriorPositiveGateElongatedMaxSoftPenaltyFraction,
                config.cellLumen.fusionSplitPriorPositiveGateElongatedMaxScore,
                config.cellLumen.fusionSplitPriorMaxOverlapCostFraction,
                config.cellLumen.fusionSplitPriorHighConfidenceMaxScore,
                config.cellLumen.fusionSplitPriorHighConfidenceMaxOverlapCostFraction,
                config.cellLumen.fusionSplitPriorHighConfidenceAxisAlignmentDegrees,
                config.cellLumen.fusionSplitPriorDaughterVolumeScale,
                config.cellLumen.fusionSplitPriorPrefilterMaxValleyRatio,
                config.cellLumen.fusionSplitPriorBridgeMaxValleyRatio,
                config.cellLumen.fusionSplitPriorMinBridgeGapWidth,
                config.cellLumen.fusionSplitPriorMinEdgeBrightness,
                config.cellLumen.fusionSplitPriorMaxDaughterOverlapFraction,
                config.cellLumen.fusionSplitPriorSoftGateEnabled,
                config.cellLumen.fusionSplitPriorSoftAxisPenaltyFraction,
                config.cellLumen.fusionSplitPriorSoftDaughterOverlapPenaltyFraction,
                config.cellLumen.fusionSplitPriorSoftValleyPenaltyFraction,
                config.cellLumen.fusionSplitPriorSoftBridgeGapPenaltyFraction,
                config.cellLumen.fusionSplitPriorSoftOverlapCostPenaltyWeight,
                config.cellLumen.fusionSplitPriorBridgeEvidenceWaivesOverlapSoftPenalty,
                config.cellLumen.fusionSplitPriorHardMaxDaughterOverlapFraction,
                config.cellLumen.fusionSplitPriorHardMaxValleyRatio,
                config.cellLumen.fusionSplitPriorHardMaxOverlapCostFraction,
                config.cellLumen.fusionSplitPriorMinPostRefitLateralSeparation,
                config.cellLumen.fusionSplitPriorMinPostRefitLateralSeparationRadiusScale,
                config.cellLumen.fusionSplitPriorMaxZDominanceForLowLateralSeparation,
                config.cellLumen.fusionSplitPriorDynamicOverlapEnabled,
                config.cellLumen.fusionSplitPriorLocalDensityRadiusScale,
                config.cellLumen.fusionSplitPriorLocalDensityOverlapBonus,
                config.cellLumen.fusionSplitPriorMaxDynamicDaughterOverlapFraction,
                config.cellLumen.fusionSplitPriorSnapshotSeedMaxRefitDrift,
                config.cellLumen
                    .fusionSplitPriorSnapshotSeedEarlyRefitWaiverEnabled,
                config.cellLumen
                    .fusionSplitPriorSnapshotSeedEarlyRefitMaxDrift,
                config.cellLumen
                    .fusionSplitPriorSnapshotSeedEarlyRefitMinParentShape,
                config.cellLumen
                    .fusionSplitPriorSnapshotSeedEarlyRefitMinFinalAxisScale,
                config.cellLumen
                    .fusionSplitPriorSnapshotSeedEarlyRefitMinTotalGainFraction,
                config.cellLumen.fusionSplitPriorSkipExistingCellBuriedCheck,
                config.cellLumen.fusionSplitPriorSkipNeighborBridgeCheck,
                config.cellLumen
                    .fusionSplitPriorAllowWindowBackedDuplicateHandoff,
                config.cellLumen
                    .fusionSplitPriorParentAnchorCleanFutureDriftRescueMinImageGain,
                config.cellLumen
                    .fusionSplitPriorParentAnchorWeakGainDuplicateRescueEnabled,
                config.cellLumen
                    .fusionSplitPriorParentAnchorWeakGainCleanMinShape,
                config.cellLumen
                    .fusionSplitPriorParentAnchorWeakGainPartialMinShape,
                config.cellLumen
                    .fusionSplitPriorParentAnchorWeakGainCleanMaxScore,
                config.cellLumen
                    .fusionSplitPriorParentAnchorWeakGainPartialMaxScore,
                config.cellLumen
                    .fusionSplitPriorParentAnchorWeakGainCleanMaxOverlapCostFraction,
                config.cellLumen
                    .fusionSplitPriorParentAnchorWeakGainPartialMaxOverlapCostFraction,
                config.cellLumen
                    .fusionSplitPriorParentAnchorWeakGainCleanMinImageGain,
                config.cellLumen
                    .fusionSplitPriorParentAnchorWeakGainPartialMinImageGain,
                config.cellLumen
                    .fusionSplitPriorParentAnchorWeakGainMaxBridgeValleyRatio,
                config.cellLumen
                    .fusionSplitPriorParentAnchorWeakGainPartialRequireConflictEvidence,
                config.cellLumen
                    .fusionSplitPriorParentAnchorCompactPositiveOneRealRescueEnabled,
                config.cellLumen
                    .fusionSplitPriorParentAnchorCompactPositiveOneRealMaxParentShape,
                config.cellLumen
                    .fusionSplitPriorParentAnchorCompactPositiveOneRealMinPriorScore,
                config.cellLumen
                    .fusionSplitPriorParentAnchorCompactPositiveOneRealMaxPriorScore,
                config.cellLumen
                    .fusionSplitPriorParentAnchorCompactPositiveOneRealMinImageGain,
                config.cellLumen
                    .fusionSplitPriorParentAnchorCompactPositiveOneRealMaxOverlapCostFraction,
                config.cellLumen
                    .fusionSplitPriorParentAnchorCompactPositiveOneRealMaxBridgeValleyRatio,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealPostRefitGuardEnabled,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealMaxRefitDrift,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealRefitDriftRescueEnabled,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealPartialRefitDriftRescueEnabled,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinImageGain,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealRefitDriftRescueMaxBridgeValleyRatio,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinBridgeGapWidth,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinParentShape,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealRefitDriftRescueMaxScore,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealCleanHighOverlapMinCost,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealCleanHighOverlapMinImageGain,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealCleanHighOverlapMaxBridgeValleyRatio,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateEnabled,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinOverlapCost,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMaxBridgeGapWidth,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinBridgeValleyRatio,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMaxParentShape,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinImageGain,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealSeedLockOnRefitCollapse,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealSeedLockMaxScore,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealSeedLockMinSeedSeparation,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealSeedLockMaxFinalSeedAxisRatio,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealMinImageGainGuardEnabled,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealCleanMinImageGain,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealPartialMinImageGain,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealPositiveWindowRescueEnabled,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMinShape,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxShape,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxScore,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxImageWorsening,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxTotalWorsening,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxOverlapCostFraction,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxSoftPenaltyFraction,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxBridgeValleyRatio,
                config.cellLumen
                    .fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMinRealVoxels,
                config.cellLumen
                    .fusionSplitPriorCleanTwoRealDuplicateBypassEnabled,
                config.cellLumen
                    .fusionSplitPriorCleanTwoRealDuplicateBypassMinImageGain,
                config.cellLumen
                    .fusionSplitPriorCleanTwoRealDuplicateBypassMaxBridgeValleyRatio,
                config.cellLumen
                    .fusionSplitPriorCleanTwoRealDuplicateBypassMinBridgeGapWidth,
                config.cellLumen
                    .fusionSplitPriorCleanTwoRealDuplicateBypassMinParentDistanceBalance,
                config.cellLumen
                    .fusionSplitPriorCleanTwoRealDuplicateBypassMaxScore,
                config.cellLumen
                    .fusionSplitPriorCleanTwoRealCompactDuplicateBypassEnabled,
                config.cellLumen
                    .fusionSplitPriorCleanTwoRealCompactDuplicateBypassMinImageGain,
                config.cellLumen
                    .fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxBridgeValleyRatio,
                config.cellLumen
                    .fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxOverlapCostFraction,
                config.cellLumen
                    .fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxSoftPenaltyFraction,
                config.cellLumen
                    .fusionSplitPriorCleanTwoRealCompactDuplicateBypassMinParentDistanceBalance,
                config.cellLumen
                    .fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxScore,
                config.cellLumen
                    .fusionSplitPriorRejectNonWindowLowShapeOverlapDuplicate,
                config.cellLumen
                    .fusionSplitPriorNonWindowLowShapeMaxParentShape,
                config.cellLumen
                    .fusionSplitPriorNonWindowLowShapeMinOverlapCost,
                config.cellLumen
                    .fusionSplitPriorNonWindowLowShapeMaxBridgeGapWidth,
                config.cellLumen
                    .fusionSplitPriorRejectPrepassFallbackWeakImageOverlapDuplicate,
                config.cellLumen
                    .fusionSplitPriorPrepassFallbackWeakImageOverlapMaxImageGain,
                config.cellLumen
                    .fusionSplitPriorPrepassFallbackWeakImageOverlapMinOverlapCost,
                config.cellLumen
                    .fusionSplitPriorPrepassFallbackWeakImageOverlapMaxBridgeGapWidth,
                config.cellLumen
                    .fusionSplitPriorPrepassFallbackWeakImageOverlapMinBridgeValleyRatio,
                config.cellLumen
                    .fusionSplitPriorRejectPrepassFallbackOverlapNoValleyEnabled,
                config.cellLumen
                    .fusionSplitPriorPrepassFallbackOverlapNoValleyMaxTotalDiff,
                config.cellLumen
                    .fusionSplitPriorPrepassFallbackOverlapNoValleyMinOverlapCost,
                config.cellLumen
                    .fusionSplitPriorPrepassFallbackOverlapNoValleyMinOverlapToImageGainRatio,
                config.cellLumen
                    .fusionSplitPriorPrepassFallbackOverlapNoValleyMaxBridgeGapWidth,
                config.cellLumen
                    .fusionSplitPriorPrepassFallbackOverlapNoValleyMinBridgeValleyRatio,
                config.cellLumen
                    .fusionSplitPriorPrepassFallbackOverlapNoValleyMinParentShape,
                config.cellLumen
                    .fusionSplitPriorPrepassFallbackOverlapNoValleyMaxPriorScore,
                config.cellLumen
                    .fusionSplitPriorPrepassFallbackOverlapNoValleyMaxFinalAxisLen,
                config.cellLumen
                    .fusionSplitPriorRejectWeakWindowNoValleyOverlapDuplicate,
                config.cellLumen
                    .fusionSplitPriorWeakWindowNoValleyMaxImageGain,
                config.cellLumen
                    .fusionSplitPriorWeakWindowNoValleyMinOverlapCost,
                config.cellLumen
                    .fusionSplitPriorWeakWindowNoValleyMaxBridgeGapWidth,
                config.cellLumen
                    .fusionSplitPriorWeakWindowNoValleyMinBridgeValleyRatio,
                config.cellLumen
                    .fusionSplitPriorRejectWindowOneSidedNoValleyUnbalancedPairEnabled,
                config.cellLumen
                    .fusionSplitPriorWindowOneSidedNoValleyMinWindowBoth,
                config.cellLumen
                    .fusionSplitPriorWindowOneSidedNoValleyMaxBalancedBonus,
                config.cellLumen
                    .fusionSplitPriorWindowOneSidedNoValleyMinParentShape,
                config.cellLumen
                    .fusionSplitPriorWindowOneSidedNoValleyMaxBridgeGapWidth,
                config.cellLumen
                    .fusionSplitPriorWindowOneSidedNoValleyMinWorstValleyRatio,
                config.cellLumen
                    .fusionSplitPriorWindowOneSidedNoValleyMinImageGain,
                config.cellLumen
                    .fusionSplitPriorRejectWindowNoValleyOverlapDominatedDuplicate,
                config.cellLumen
                    .fusionSplitPriorWindowNoValleyOverlapDominatedMinWindowBoth,
                config.cellLumen
                    .fusionSplitPriorWindowNoValleyOverlapDominatedMinOverlapCost,
                config.cellLumen
                    .fusionSplitPriorWindowNoValleyOverlapDominatedMinOverlapToImageGainRatio,
                config.cellLumen
                    .fusionSplitPriorWindowNoValleyOverlapDominatedMaxBridgeGapWidth,
                config.cellLumen
                    .fusionSplitPriorWindowNoValleyOverlapDominatedMinBridgeValleyRatio,
                config.cellLumen
                    .fusionSplitPriorRejectSeedZColumnNoValleyOverlapDuplicate,
                config.cellLumen
                    .fusionSplitPriorSeedZColumnNoValleyMaxSeedLateralSeparation,
                config.cellLumen
                    .fusionSplitPriorSeedZColumnNoValleyMinSeedZDominance,
                config.cellLumen
                    .fusionSplitPriorSeedZColumnNoValleyMinWindowBoth,
                config.cellLumen
                    .fusionSplitPriorSeedZColumnNoValleyMinOverlapCost,
                config.cellLumen
                    .fusionSplitPriorSeedZColumnNoValleyMinOverlapToImageGainRatio,
                config.cellLumen
                    .fusionSplitPriorSeedZColumnNoValleyMaxBridgeGapWidth,
                config.cellLumen
                    .fusionSplitPriorSeedZColumnNoValleyMinBridgeValleyRatio,
                config.cellLumen
                    .fusionSplitPriorRejectWeakFutureSeedZColumnNoValleyDuplicate,
                config.cellLumen
                    .fusionSplitPriorWeakFutureSeedZColumnNoValleyMaxSeedLateralSeparation,
                config.cellLumen
                    .fusionSplitPriorWeakFutureSeedZColumnNoValleyMinSeedZDominance,
                config.cellLumen
                    .fusionSplitPriorWeakFutureSeedZColumnNoValleyMaxWindowBoth,
                config.cellLumen
                    .fusionSplitPriorWeakFutureSeedZColumnNoValleyMaxWindowMissing,
                config.cellLumen
                    .fusionSplitPriorWeakFutureSeedZColumnNoValleyMaxParentPersists,
                config.cellLumen
                    .fusionSplitPriorWeakFutureSeedZColumnNoValleyMaxBridgeGapWidth,
                config.cellLumen
                    .fusionSplitPriorWeakFutureSeedZColumnNoValleyMinWorstBridgeValleyRatio,
                config.cellLumen
                    .fusionSplitPriorWeakFutureSeedZColumnNoValleyMinBridgeValleyFromBright,
                config.cellLumen
                    .fusionSplitPriorReanchorNonParentDuplicateToOtherDaughter,
                config.cellLumen
                    .fusionSplitPriorNonParentDuplicateReanchorMinImageGain,
                config.cellLumen
                    .fusionSplitPriorNonParentDuplicateReanchorMaxBridgeValleyRatio,
                config.cellLumen
                    .fusionSplitPriorNonParentDuplicateReanchorMinBridgeGapWidth,
                config.cellLumen
                    .fusionSplitPriorNonParentDuplicateReanchorMinMove,
                config.cellLumen
                    .fusionSplitPriorNonParentDuplicateReanchorMinParentDistanceBalance);

            double costDiff = result.first;
            auto callback = result.second;

            const bool accept = (costDiff < 0.0);
            callback(accept);
            if (accept) {
                ++splitAccepted;
                splitAcceptedInPhase.insert(cellName);
                previousSnapshots.erase(cellName);
                cellFirstSeenFrame.erase(cellName);
                cellFirstSeenFrame[cellName + "0"] = displayFrame;
                cellFirstSeenFrame[cellName + "1"] = displayFrame;
	                if (lumenForCell != nullptr) {
	                    acceptedLumenSplits.push_back({cellName, *lumenForCell});
	                    // f035 showed that the split-stage z-stack seed lock can
	                    // place the promoted daughter correctly, then the final
	                    // per-frame PCA pass can slide that newborn back toward
	                    // the lower shell. Keep this narrow: only the daughter
	                    // explicitly marked by z-stack promotion is position
	                    // locked for final PCA, while PCA can still refit shape.
	                    if (config.prob
	                            .split_daughter_refit_zstack_seed_lock_enabled &&
	                        lumenForCell->zStackDaughterPromotion) {
	                        if (lumenForCell->candidateIdA ==
	                            lumenForCell->zStackPromotedCandidateId) {
	                            sameFramePerturbLockedCells.insert(
	                                cellName + "0");
	                            lockPcaPosition(
	                                cellName + "0",
	                                "zstack_promoted_daughter_seed");
	                        }
	                        if (lumenForCell->candidateIdB ==
	                            lumenForCell->zStackPromotedCandidateId) {
	                            sameFramePerturbLockedCells.insert(
	                                cellName + "1");
	                            lockPcaPosition(
	                                cellName + "1",
	                                "zstack_promoted_daughter_seed");
	                        }
	                    }
	                    if (config.cellLumen
	                            .fusionSplitPriorLockAcceptedDaughtersForFinalPca) {
	                        lockPcaPosition(cellName + "0",
	                                        "accepted_cell_lumen_split");
                        lockPcaPosition(cellName + "1",
                                        "accepted_cell_lumen_split");
                    }
                }
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
                    frame.setMeanCellBrightness(
                        bSum / static_cast<float>(frame.cells.size()));
                }
                // Rebuild Voronoi territory — cell count and positions
                // changed (parent replaced, two daughters appended).
                // Daughters don't have a snap entry yet, so their live
                // positions become their Voronoi anchors going forward.
                if (voronoiMapNeeded) {
                    frame.rebuildVoronoiMap();
                }
                return;
            }

            const bool wasCellLumenPriorAttempt = (scheduleReason == "cell_lumen_prior");
            const bool allowClassicFallback =
                wasCellLumenPriorAttempt &&
                config.cellLumen.fusionSplitPriorFallbackToClassicOnReject;
            if (!allowClassicFallback) {
                splitBlacklist.insert(cellName);
                splitRejectedInPhase.insert(cellName);
            } else {
                lumenClassicFallbackNames.insert(cellName);
                std::cout << "[CellLumen SplitPrior Fallback] frame " << displayFrame
                          << " cell=" << cellName
                          << " reason=prior_rejected_schedule_classic_once"
                          << std::endl;
            }

            bool compensationHandled = false;
            if (config.cellLumen
                    .fusionSplitRejectCompensateWithSameParentCenterEnabled &&
                wasCellLumenPriorAttempt &&
                lumenForCell != nullptr &&
                lumenCenterCandidatesForFrame != nullptr) {
                const auto centerIt =
                    lumenCenterCandidatesForFrame->find(cellName);
                if (centerIt != lumenCenterCandidatesForFrame->end()) {
                    const auto &centerCandidate = centerIt->second;
                    const bool sameParentSplitCenter =
                        centerCandidate.candidateId >= 0 &&
                        (centerCandidate.candidateId ==
                             lumenForCell->candidateIdA ||
                         centerCandidate.candidateId ==
                             lumenForCell->candidateIdB);
                    const cv::Point3f oldPos(frame.cells[cellIdx].getX(),
                                             frame.cells[cellIdx].getY(),
                                             frame.cells[cellIdx].getZ());
                    const float moveDistance = static_cast<float>(
                        cv::norm(centerCandidate.position - oldPos));
                    const float minMove = std::max(
                        0.0f,
                        config.cellLumen
                            .fusionSplitRejectCompensateWithSameParentCenterMinDistance);
                    const float maxMove = std::max(
                        minMove,
                        config.cellLumen
                            .fusionSplitRejectCompensateWithSameParentCenterMaxDistance);
                    const int minVoxels = std::max(
                        0,
                        config.cellLumen
                            .fusionSplitRejectCompensateWithSameParentCenterMinVoxels);
                    const float minSignal = std::max(
                        0.0f,
                        config.cellLumen
                            .fusionSplitRejectCompensateWithSameParentCenterMinSignal);
                    const bool strongCenter =
                        centerCandidate.voxelCount >= minVoxels &&
                        centerCandidate.signal >= minSignal &&
                        moveDistance >= minMove &&
                        moveDistance <= maxMove;
                    if (sameParentSplitCenter && strongCenter) {
                        // A rejected Cell Lumen split can still identify the
                        // correct continuation center for the same parent. Try
                        // that reserved center before falling back to random
                        // compensation so the good evidence is not discarded
                        // only because the division itself was unsafe.
                        const float blend = std::clamp(
                            config.cellLumen
                                .fusionSplitRejectCompensateWithSameParentCenterBlend,
                            0.0f, 1.0f);
                        cv::Point3f target =
                            oldPos * (1.0f - blend) +
                            centerCandidate.position * blend;
                        float appliedZBlend = blend;
                        if (centerCandidate.clusterCollapsed &&
                            config.cellLumen
                                .fusionCenterPriorClusterCollapseUseSeparateZBlend) {
                            // Cluster-collapsed early centers often fix a real
                            // z-slice bias. Let YAML opt in to preserving that
                            // z evidence without forcing a stronger XY jump.
                            const float zBlend = std::clamp(
                                config.cellLumen
                                    .fusionCenterPriorClusterCollapseZBlend,
                                0.0f, 1.0f);
                            target.z =
                                oldPos.z * (1.0f - zBlend) +
                                centerCandidate.position.z * zBlend;
                            appliedZBlend = zBlend;
                        }
                        auto centerCompResult = frame.perturbCell(
                            cellIdx, overlapWeight, useSignalGuidanceThisFrame,
                            randomPerturbRadiusRatio,
                            /*pcaRefitWellFilledMove=*/true,
                            /*useSignalMapGuidance=*/false,
                            &target);
                        const bool centerCompAccept =
                            centerCompResult.first < 0.0;
                        std::ostringstream note;
                        note << "score_is_total_cost_diff"
                             << ";lumen_candidate_id="
                             << centerCandidate.candidateId
                             << ";lumen_distance=" << moveDistance
                             << ";lumen_signal=" << centerCandidate.signal
                             << ";lumen_voxels="
                             << centerCandidate.voxelCount
                             << ";blend=" << blend
                             << ";z_blend=" << appliedZBlend
                             << ";same_parent_split_center=1";
                        logPerturbCandidate(
                            "split_reject_lumen_center_compensation",
                            cellName,
                            frame.cells[cellIdx],
                            centerCompResult.first,
                            centerCompAccept,
                            randomPerturbRadiusRatio,
                            note.str());
                        if (exportPerturbDebug) {
                            accumulateDebugCellPlacement(
                                splitPerturbDebugPlacements,
                                frame.cells[cellIdx],
                                config.simulation,
                                config.simulation
                                    .perturb_debug_cell_brightness);
                            ++splitPerturbDebugPlacementCount;
                        }
                        centerCompResult.second(centerCompAccept);
                        if (centerCompAccept) {
                            lockPcaPosition(
                                cellName,
                                "split_reject_lumen_center_compensation");
                            ++perturbAccepted;
                            residSum += centerCompResult.first;
                            absResidSum += std::abs(centerCompResult.first);
                            ++residCount;
                            compensationHandled = true;
                        }
                    }
                }
            }

            // Compensation perturb. The revert left cells[cellIdx] as the
            // parent, so no find_if is needed as long as cellIdx still points
            // at the same parent after the rejected split callback.
            if (!compensationHandled) {
                auto compResult = frame.perturbCell(
                    cellIdx, overlapWeight, useSignalGuidanceThisFrame,
                    randomPerturbRadiusRatio,
                    /*pcaRefitWellFilledMove=*/false,
                    /*useSignalMapGuidance=*/false);
                const bool compAccept = compResult.first < 0.0;
                logPerturbCandidate("split_reject_compensation",
                                    cellName,
                                    frame.cells[cellIdx],
                                    compResult.first,
                                    compAccept,
                                    randomPerturbRadiusRatio,
                                    "score_is_total_cost_diff");
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
                    ++perturbAccepted;
                    residSum += compResult.first;
                    absResidSum += std::abs(compResult.first);
                    ++residCount;
                }
            }
        };

        // Image-grounded pre-pass can identify an obvious two-blob split
        // even when snapshot elongation is close to round. Try those
        // candidates once up front so correctness does not depend on random
        // scheduling. Acceptance still uses the same gates as every normal
        // random split attempt.
        if (allowSplits && !forcedSplitNames.empty() &&
            !lumenSplitRandomDisabled) {
            for (const auto &forcedName : forcedSplitNames) {
                if (phaseNames.count(forcedName) == 0 ||
                    splitBlacklist.count(forcedName) > 0) {
                    continue;
                }
                auto it = std::find_if(
                    frame.cells.begin(), frame.cells.end(),
                    [&](const Ellipsoid &cell) {
                        return cell.getName() == forcedName;
                    });
                if (it == frame.cells.end()) continue;
                const size_t cellIdx = static_cast<size_t>(
                    std::distance(frame.cells.begin(), it));
                attemptSplitAtIndex(cellIdx, forcedName, "prepass_force",
                                    /*useLumenPrior=*/false);
            }
            rebuildEligible();
        }

        if (allowSplits &&
            lumenSplitPriorActive &&
            config.cellLumen.fusionSplitPriorForceSchedule) {
            for (const auto &entry : *lumenSplitPriorsForFrame) {
                const std::string &forcedName = entry.first;
                if (phaseNames.count(forcedName) == 0 ||
                    splitBlacklist.count(forcedName) > 0) {
                    continue;
                }
                auto it = std::find_if(
                    frame.cells.begin(), frame.cells.end(),
                    [&](const Ellipsoid &cell) {
                        return cell.getName() == forcedName;
                    });
                if (it == frame.cells.end()) continue;
                const size_t cellIdx = static_cast<size_t>(
                    std::distance(frame.cells.begin(), it));
                attemptSplitAtIndex(cellIdx, forcedName, "cell_lumen_prior",
                                    /*useLumenPrior=*/true);
            }
            rebuildEligible();
        }

        if (allowSplits && !lumenClassicFallbackNames.empty()) {
            const std::vector<std::string> fallbackNames(
                lumenClassicFallbackNames.begin(), lumenClassicFallbackNames.end());
            lumenClassicFallbackNames.clear();
            for (const auto &fallbackName : fallbackNames) {
                if (phaseNames.count(fallbackName) == 0 ||
                    splitBlacklist.count(fallbackName) > 0) {
                    continue;
                }
                auto it = std::find_if(
                    frame.cells.begin(), frame.cells.end(),
                    [&](const Ellipsoid &cell) {
                        return cell.getName() == fallbackName;
                    });
                if (it == frame.cells.end()) continue;
                const size_t cellIdx = static_cast<size_t>(
                    std::distance(frame.cells.begin(), it));
                attemptSplitAtIndex(cellIdx, fallbackName,
                                    "cell_lumen_classic_fallback",
                                    /*useLumenPrior=*/false);
            }
            rebuildEligible();
        }

        ScopedStageTimer perturbLoopTimer(displayFrame, "perturb_iteration_loop");
        for (size_t i = 0; i < totalPhaseIters; ++i) {
            if (eligible.empty()) break;

            size_t cellIdx = 0;
            if (config.cellLumen.fusionPerturbVisitEachCellOnceEnabled &&
                i < eligible.size()) {
                // Sparse early frames run only a few effective perturb tries.
                // Visiting each cell once prevents Cell Lumen center evidence
                // from being skipped just because random sampling picked the
                // same neighbor repeatedly.
                cellIdx = eligible[i];
            } else {
                std::uniform_int_distribution<size_t> cellDist(0, eligible.size() - 1);
                cellIdx = eligible[cellDist(gen)];
            }
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
	            if (sameFramePerturbLockedCells.count(cellName) > 0) {
	                logPerturbCandidate(
	                    "same_frame_zstack_promoted_daughter_locked",
	                    cellName,
	                    frame.cells[cellIdx],
	                    0.0,
	                    false,
	                    perturbRatioFor(cellName),
	                    "zstack_promoted_daughter_seed_preserved_until_next_frame");
	                continue;
	            }

	            float pSplit = 0.0f;
            const bool randomSplitsAllowed =
                allowSplits &&
                !lumenSplitRandomDisabled;
            if (randomSplitsAllowed) {
                auto pIt = splitProbabilities.find(cellName);
                if (pIt != splitProbabilities.end()) pSplit = pIt->second;
            }
            const bool canSplit = pSplit > 0.0f
                               && splitBlacklist.count(cellName) == 0;

            if (canSplit && uniform01(gen) < pSplit) {
                attemptSplitAtIndex(cellIdx, cellName, "random",
                                    /*useLumenPrior=*/false);
            } else {
                // --- Perturbation ---
                const float effectivePerturbRadiusRatio = perturbRatioFor(cellName);
                bool perturbHandled = false;
                if (lumenCenterCandidatesForFrame != nullptr) {
                    auto centerIt = lumenCenterCandidatesForFrame->find(cellName);
                    if (centerIt != lumenCenterCandidatesForFrame->end()) {
                        const auto &centerCandidate = centerIt->second;
                        const bool reservedBySplit =
                            centerCandidate.candidateId >= 0 &&
                            reservedLumenSplitCandidateIds.count(
                                centerCandidate.candidateId) > 0;
                        if (reservedBySplit) {
                            auto ownerIt =
                                reservedLumenSplitCandidateOwners.find(
                                    centerCandidate.candidateId);
                            std::ostringstream note;
                            note << "score_is_total_cost_diff"
                                 << ";lumen_candidate_id="
                                 << centerCandidate.candidateId
                                 << ";lumen_distance="
                                 << centerCandidate.distance
                                 << ";lumen_signal="
                                 << centerCandidate.signal
                                 << ";lumen_voxels="
                                 << centerCandidate.voxelCount
                                 << ";reserved_by_split_parent="
                                 << (ownerIt !=
                                             reservedLumenSplitCandidateOwners.end()
                                         ? ownerIt->second
                                         : std::string("unknown"));
                            logPerturbCandidate(
                                "cell_lumen_center_candidate_reserved_by_split",
                                cellName,
                                frame.cells[cellIdx],
                                0.0,
                                false,
                                effectivePerturbRadiusRatio,
                                note.str());
                            lockPcaPosition(cellName,
                                            "reserved_lumen_split_candidate");
                            perturbHandled = true;
                        } else {
                        const cv::Point3f oldPos(frame.cells[cellIdx].getX(),
                                                 frame.cells[cellIdx].getY(),
                                                 frame.cells[cellIdx].getZ());
	                        const bool clusterForceReanchor =
	                            centerCandidate.clusterCollapsed &&
	                            config.cellLumen
	                                .fusionCenterPriorClusterCollapseForceReanchorEnabled &&
                            absoluteFrame >=
                                std::max(
                                    0,
                                    config.cellLumen
                                        .fusionCenterPriorClusterCollapseForceReanchorMinFrame) &&
                            centerCandidate.distance >=
                                std::max(
                                    0.0f,
                                    config.cellLumen
                                        .fusionCenterPriorClusterCollapseForceReanchorMinDistance) &&
                            centerCandidate.distance <=
                                std::max(
                                    config.cellLumen
                                        .fusionCenterPriorClusterCollapseForceReanchorMinDistance,
                                    config.cellLumen
                                        .fusionCenterPriorClusterCollapseForceReanchorMaxDistance) &&
                            centerCandidate.voxelCount >=
                                std::max(
                                    0,
                                    config.cellLumen
                                        .fusionCenterPriorClusterCollapseForceReanchorMinVoxels) &&
	                            centerCandidate.signal >=
	                                std::max(
	                                    0.0f,
	                                    config.cellLumen
	                                        .fusionCenterPriorClusterCollapseForceReanchorMinSignal);
	                        const bool youngFarSingleForceReanchor =
	                            centerCandidate.youngFarSingle &&
	                            config.cellLumen
	                                .fusionCenterPriorYoungFarSingleForceReanchorEnabled &&
	                            centerCandidate.parentZShift >=
	                                std::max(
	                                    0.0f,
	                                    config.cellLumen
	                                        .fusionCenterPriorYoungFarSingleForceReanchorMinPositiveZShift);
	                        if (clusterForceReanchor ||
	                            youngFarSingleForceReanchor) {
	                            // A cluster-collapsed Cell Lumen center is not a
	                            // random perturb target; it is already a local
	                            // density estimate from multiple nearby lumen
	                            // peaks. In sparse early frames, f034 showed the
	                            // cost perturb path can reject this good center
	                            // and let PCA pull z back to the old ellipsoid.
	                            // f036 showed the same failure mode for a newborn
	                            // daughter with strong upward z Cell Lumen evidence:
	                            // local cost search slid the center back to the lower
	                            // shell. These switch-gated reanchors keep the
	                            // candidate only when its support is inside a narrow,
	                            // explicit YAML range.
	                            frame.cells[cellIdx].setPosition(
	                                centerCandidate.position.x,
	                                centerCandidate.position.y,
	                                centerCandidate.position.z);
	                            const std::string reanchorReason =
	                                youngFarSingleForceReanchor
	                                    ? "cell_lumen_young_far_single_force_reanchor"
	                                    : "cell_lumen_cluster_center_force_reanchor";
	                            lockPcaPosition(
	                                cellName,
	                                reanchorReason);
	                            std::ostringstream note;
	                            note << (youngFarSingleForceReanchor
	                                         ? "direct_young_far_single_reanchor"
	                                         : "direct_cluster_center_reanchor")
	                                 << ";lumen_candidate_id="
	                                 << centerCandidate.candidateId
	                                 << ";lumen_distance="
                                 << centerCandidate.distance
                                 << ";lumen_signal="
	                                 << centerCandidate.signal
	                                 << ";lumen_voxels="
	                                 << centerCandidate.voxelCount
	                                 << ";cluster_count="
	                                 << centerCandidate.clusterCandidateCount
	                                 << ";young_far_single="
	                                 << (centerCandidate.youngFarSingle ? 1 : 0)
	                                 << ";parent_z_shift="
	                                 << centerCandidate.parentZShift;
	                            logPerturbCandidate(
	                                reanchorReason,
	                                cellName,
	                                frame.cells[cellIdx],
                                -1.0,
                                true,
                                effectivePerturbRadiusRatio,
                                note.str());
                            frame.regenerateSynthFrame();
                            if (voronoiMapNeeded) {
                                frame.rebuildVoronoiMap();
                            }
                            perturbAccepted++;
                            residSum += -1.0;
                            absResidSum += 1.0;
                            residCount++;
                            perturbHandled = true;
                        } else {
                        const float configuredBlend =
                            centerCandidate.positionBlendOverride >= 0.0f
                                ? centerCandidate.positionBlendOverride
                                : config.cellLumen.fusionCenterPriorPositionBlend;
                        const float blend =
                            std::clamp(configuredBlend, 0.0f, 1.0f);
                        cv::Point3f target =
                            oldPos * (1.0f - blend) + centerCandidate.position * blend;
                        float appliedZBlend = blend;
                        if (centerCandidate.clusterCollapsed &&
                            config.cellLumen
                                .fusionCenterPriorClusterCollapseUseSeparateZBlend) {
                            // f034 showed that blending all axes equally can
                            // erase the useful z position from a collapsed
                            // Cell Lumen center prior. This keeps the default
                            // behavior unchanged unless the YAML explicitly
                            // enables a separate z blend.
                            const float zBlend = std::clamp(
                                config.cellLumen
                                    .fusionCenterPriorClusterCollapseZBlend,
                                0.0f, 1.0f);
                            target.z =
                                oldPos.z * (1.0f - zBlend) +
                                centerCandidate.position.z * zBlend;
                            appliedZBlend = zBlend;
                        }
                        auto centerResult = frame.perturbCell(
                            cellIdx, overlapWeight, useSignalGuidanceThisFrame,
                            effectivePerturbRadiusRatio,
                            /*pcaRefitWellFilledMove=*/true,
                            /*useSignalMapGuidance=*/false,
                            &target);
                        double centerCostDiff = centerResult.first;
                        auto centerCallback = centerResult.second;
                        const bool centerAccept = centerCostDiff < 0;
                        std::ostringstream note;
                        note << "score_is_total_cost_diff"
                             << ";lumen_candidate_id=" << centerCandidate.candidateId
                             << ";lumen_distance=" << centerCandidate.distance
                             << ";lumen_signal=" << centerCandidate.signal
                             << ";lumen_voxels=" << centerCandidate.voxelCount
                             << ";blend=" << blend
                             << ";z_blend=" << appliedZBlend
                             << ";cluster_collapsed="
                             << (centerCandidate.clusterCollapsed ? 1 : 0)
                             << ";cluster_count="
                             << centerCandidate.clusterCandidateCount;
                        logPerturbCandidate("cell_lumen_center_candidate",
                                            cellName,
                                            frame.cells[cellIdx],
                                            centerCostDiff,
                                            centerAccept,
                                            effectivePerturbRadiusRatio,
                                            note.str());
                        if (exportPerturbDebug) {
                            accumulateDebugCellPlacement(
                                movementPerturbDebugPlacements,
                                frame.cells[cellIdx],
                                config.simulation,
                                config.simulation.perturb_debug_cell_brightness);
                            ++movementPerturbDebugPlacementCount;
                        }
                        recordPerturbLoss(cellName, frame.cells[cellIdx],
                                          centerCostDiff, centerAccept);
                        if (centerAccept) {
                            centerCallback(true);
                            perturbAccepted++;
                            residSum += centerCostDiff;
                            absResidSum += std::abs(centerCostDiff);
                            residCount++;
                            perturbHandled = true;
                        } else {
                            centerCallback(false);
                        }
                        }
                        }
                    }
                }

                if (!perturbHandled) {
                    auto result = frame.perturbCell(
                        cellIdx, overlapWeight, useSignalGuidanceThisFrame,
                        effectivePerturbRadiusRatio,
                        /*pcaRefitWellFilledMove=*/true);
                    double costDiff = result.first;
                    auto callback = result.second;
                    bool perturbAccept = costDiff < 0;
                    std::ostringstream perturbNote;
                    perturbNote << "score_is_total_cost_diff";
                    if (perturbAccept && !useSignalGuidanceThisFrame) {
                        auto snapIt = previousSnapshots.find(cellName);
                        if (snapIt != previousSnapshots.end() && snapIt->second.valid) {
                            const PreviousFrameSnapshot &snap = snapIt->second;
                            const cv::Point3f candidatePos(frame.cells[cellIdx].getX(),
                                                           frame.cells[cellIdx].getY(),
                                                           frame.cells[cellIdx].getZ());
                            const float snapMaxR =
                                std::max({snap.aRadius, snap.bRadius, snap.cRadius});
                            const float driftFromSnapshot =
                                static_cast<float>(cv::norm(candidatePos - snap.position));
                            const float driftLimit =
                                std::max(10.0f, std::min(14.0f, snapMaxR * 0.55f));
                            bool hasNearbyLumenEvidence = false;
                            if (lumenCenterCandidatesForFrame != nullptr) {
                                auto centerIt =
                                    lumenCenterCandidatesForFrame->find(cellName);
                                if (centerIt != lumenCenterCandidatesForFrame->end()) {
                                    const auto &centerCandidate = centerIt->second;
                                    const float lumenDist =
                                        static_cast<float>(cv::norm(
                                            candidatePos - centerCandidate.position));
                                    hasNearbyLumenEvidence =
                                        lumenDist <= std::max(8.0f, snapMaxR * 0.45f) &&
                                        centerCandidate.signal >= 100.0f;
                                }
                            }
                            if (driftFromSnapshot > driftLimit &&
                                !hasNearbyLumenEvidence) {
                                perturbAccept = false;
                                perturbNote << ";large_random_drift_rejected=1"
                                            << ";drift_from_snapshot="
                                            << driftFromSnapshot
                                            << ";drift_limit=" << driftLimit;
                            }
                        }
                    }
                    logPerturbCandidate(useSignalGuidanceThisFrame
                                            ? "signal_guided_local_search"
                                            : "random_local_search",
                                        cellName,
                                        frame.cells[cellIdx],
                                        costDiff,
                                        perturbAccept,
                                        effectivePerturbRadiusRatio,
                                        perturbNote.str());
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
        perturbLoopTimer.finish();
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

    if (!acceptedLumenSplits.empty()) {
        auto isRealCandidateId = [](int candidateId) {
            return candidateId >= 0 && candidateId < 1000000000;
        };
        auto findCellIndexByName = [&](const std::string &name) -> int {
            for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
                if (frame.cells[ci].getName() == name) {
                    return static_cast<int>(ci);
                }
            }
            return -1;
        };
        auto maxRadiusOf = [](const Ellipsoid &cell) {
            return std::max({cell.getARadius(), cell.getBRadius(), cell.getCRadius()});
        };
        auto centerOf = [](const Ellipsoid &cell) {
            return cv::Point3f(cell.getX(), cell.getY(), cell.getZ());
        };

        int continuationMergeCount = 0;
        int duplicateHandoffPruneCount = 0;
        std::set<std::string> mergedContinuations;
        for (const AcceptedLumenSplitRecord &record : acceptedLumenSplits) {
            const BridgeSplitProposal &proposal = record.proposal;
            const bool realA = isRealCandidateId(proposal.candidateIdA);
            const bool realB = isRealCandidateId(proposal.candidateIdB);
            const bool parentAnchorOneReal =
                proposal.parentAnchored && (realA != realB);
            const bool cleanWindow =
                proposal.windowBothDaughtersSupported >= 2 &&
                proposal.windowMissingDaughterCount == 0 &&
                proposal.windowParentPersists == 0;
            const float configuredParentAnchorMergeShapeMin =
                config.cellLumen
                    .fusionSplitPriorParentAnchorContinuationMergeMinShape;
            const float parentAnchorMergeShapeMin =
                configuredParentAnchorMergeShapeMin >= 0.0f
                    ? std::max(1.0f, configuredParentAnchorMergeShapeMin)
                    : std::max(
                          1.50f,
                          config.cellLumen.fusionSplitPriorElongatedParentMinShape);
            const bool parentAnchorMergeMode =
                parentAnchorOneReal &&
                cleanWindow &&
                proposal.parentShapeElongation >= parentAnchorMergeShapeMin &&
                proposal.neighborClaimPenalty <= 1e-5f &&
                proposal.parentPersistencePenalty <= 1e-5f &&
                (!config.cellLumen
                      .fusionSplitPriorParentAnchorContinuationMergeRejectClaimConflict ||
                 (proposal.continuationClaimSoftPenalty <= 1e-5f &&
                  proposal.continuationClaimBlockerNames.empty()));
            const bool weakNeighborHandoffMode =
                !proposal.parentAnchored &&
                cleanWindow &&
                proposal.elongatedParentRescued &&
                proposal.parentDistanceBalance < 0.68f &&
                proposal.neighborClaimPenalty <= 1e-5f &&
                proposal.continuationClaimSoftPenalty <= 1e-5f &&
                proposal.parentPersistencePenalty <= 1e-5f;
            if (!parentAnchorMergeMode && !weakNeighborHandoffMode) {
                continue;
            }

            std::vector<std::string> daughterNames;
            if (parentAnchorMergeMode) {
                daughterNames.push_back(realA ? record.parentName + "1"
                                              : record.parentName + "0");
            } else {
                daughterNames.push_back(record.parentName + "0");
                daughterNames.push_back(record.parentName + "1");
            }

            bool mergedThisSplit = false;
            for (const std::string &daughterName : daughterNames) {
                const int daughterIdx = findCellIndexByName(daughterName);
                if (daughterIdx < 0) {
                    continue;
                }
                const Ellipsoid &daughter = frame.cells[static_cast<size_t>(daughterIdx)];
                const cv::Point3f dPos = centerOf(daughter);
                const float dMaxR = maxRadiusOf(daughter);

                int bestOtherIdx = -1;
                float bestMergeScore = std::numeric_limits<float>::infinity();
                float bestNorm = std::numeric_limits<float>::infinity();
                float bestDist = std::numeric_limits<float>::infinity();
                for (size_t oi = 0; oi < frame.cells.size(); ++oi) {
                    if (static_cast<int>(oi) == daughterIdx) continue;
                    const Ellipsoid &other = frame.cells[oi];
                    const std::string &otherName = other.getName();
                    if (other.isTrash()) continue;
                    if (otherName == record.parentName) continue;
                    if (otherName == record.parentName + "0" ||
                        otherName == record.parentName + "1") {
                        continue;
                    }
                    if (allNames.count(otherName) == 0) continue;
                    if (splitAcceptedInLoop.count(otherName) > 0) continue;
                    if (mergedContinuations.count(otherName) > 0) continue;

                    const float oMaxR = maxRadiusOf(other);
                    const float dist = static_cast<float>(cv::norm(dPos - centerOf(other)));
                    const float norm = dist / std::max(1.0f, dMaxR + oMaxR);
                    const float normLimit =
                        parentAnchorMergeMode
                            ? std::max(
                                  0.0f,
                                  config.cellLumen
                                      .fusionSplitPriorParentAnchorContinuationMergeNormLimit)
                            : 0.68f;
                    const float distLimit =
                        parentAnchorMergeMode
                            ? std::max(
                                  0.0f,
                                  config.cellLumen
                                      .fusionSplitPriorParentAnchorContinuationMergeDistanceLimit)
                            : 32.0f;
                    if (norm > normLimit || dist > distLimit) {
                        continue;
                    }
                    size_t commonPrefix = 0;
                    const size_t commonMax =
                        std::min(record.parentName.size(), otherName.size());
                    while (commonPrefix < commonMax &&
                           record.parentName[commonPrefix] ==
                               otherName[commonPrefix]) {
                        ++commonPrefix;
                    }
                    const bool lineageRelated =
                        commonPrefix + 2 >= record.parentName.size();
                    const float lineageBonus =
                        lineageRelated
                            ? std::max(
                                  0.0f,
                                  config.cellLumen
                                      .fusionSplitPriorParentAnchorContinuationMergeLineageBonus)
                            : 0.0f;
                    const float mergeScore = norm - lineageBonus;
                    if (mergeScore < bestMergeScore) {
                        bestMergeScore = mergeScore;
                        bestNorm = norm;
                        bestDist = dist;
                        bestOtherIdx = static_cast<int>(oi);
                    }
                }

                if (bestOtherIdx < 0) {
                    continue;
                }

                const std::string continuationName =
                    frame.cells[static_cast<size_t>(bestOtherIdx)].getName();
                EllipsoidParams mergedParams = daughter.getCellParams();
                mergedParams.name = continuationName;
                frame.cells[static_cast<size_t>(bestOtherIdx)] = Ellipsoid(mergedParams);
                frame.cells.erase(frame.cells.begin() + daughterIdx);
                previousSnapshots.erase(daughterName);
                cellShapeReference.erase(daughterName);
                cellShapeBirth.erase(daughterName);
                cellFirstSeenFrame.erase(daughterName);
                mergedContinuations.insert(continuationName);
                ++continuationMergeCount;
                mergedThisSplit = true;
                std::cout << "[Split Continuation Merge] frame " << displayFrame
                          << " parent=" << record.parentName
                          << " daughter=" << daughterName
                          << " continuation=" << continuationName
                          << " dist=" << bestDist
                          << " normalized=" << bestNorm
                          << " mode="
                          << (parentAnchorMergeMode
                                  ? "parent_anchor_one_real"
                                  : "weak_neighbor_handoff")
                          << " parent_shape_min=" << parentAnchorMergeShapeMin
                          << std::endl;
                break;
            }

            (void)mergedThisSplit;
        }

        if (config.cellLumen.fusionSplitPriorAllowWindowBackedDuplicateHandoff) {
            const float minContinuationPenalty = std::max(
                0.0f,
                config.cellLumen
                    .fusionSplitPriorDuplicateHandoffMinContinuationPenalty);
            const float normLimit = std::max(
                0.0f,
                config.cellLumen.fusionSplitPriorDuplicateHandoffNormLimit);
            const float distLimit = std::max(
                0.0f,
                config.cellLumen.fusionSplitPriorDuplicateHandoffDistanceLimit);
            std::set<std::string> duplicateHandoffRemoveNames;

            for (const AcceptedLumenSplitRecord &record : acceptedLumenSplits) {
                const BridgeSplitProposal &proposal = record.proposal;
                const bool cleanWindow =
                    proposal.windowBothDaughtersSupported >= 2 &&
                    proposal.windowMissingDaughterCount == 0 &&
                    proposal.windowParentPersists == 0;
                if (proposal.parentAnchored ||
                    !cleanWindow ||
                    proposal.continuationClaimBlockerNames.empty() ||
                    proposal.continuationClaimSoftPenalty < minContinuationPenalty) {
                    continue;
                }

                const std::array<std::string, 2> daughterNames = {
                    record.parentName + "0",
                    record.parentName + "1"};
                std::vector<int> daughterIndexes;
                for (const std::string &daughterName : daughterNames) {
                    const int daughterIdx = findCellIndexByName(daughterName);
                    if (daughterIdx >= 0) {
                        daughterIndexes.push_back(daughterIdx);
                    }
                }
                if (daughterIndexes.empty()) {
                    continue;
                }

                std::stringstream blockers(proposal.continuationClaimBlockerNames);
                std::string blockerName;
                while (std::getline(blockers, blockerName, '|')) {
                    if (blockerName.empty() ||
                        blockerName == record.parentName ||
                        blockerName == daughterNames[0] ||
                        blockerName == daughterNames[1] ||
                        splitAcceptedInLoop.count(blockerName) > 0 ||
                        mergedContinuations.count(blockerName) > 0) {
                        continue;
                    }
                    const int blockerIdx = findCellIndexByName(blockerName);
                    if (blockerIdx < 0) {
                        continue;
                    }

                    const Ellipsoid &blocker =
                        frame.cells[static_cast<size_t>(blockerIdx)];
                    const cv::Point3f blockerPos = centerOf(blocker);
                    const float blockerMaxR = maxRadiusOf(blocker);
                    float bestDist = std::numeric_limits<float>::infinity();
                    float bestNorm = std::numeric_limits<float>::infinity();
                    std::string bestDaughterName;
                    for (const int daughterIdx : daughterIndexes) {
                        const Ellipsoid &daughter =
                            frame.cells[static_cast<size_t>(daughterIdx)];
                        const float dist = static_cast<float>(
                            cv::norm(centerOf(daughter) - blockerPos));
                        const float norm = dist / std::max(
                            1.0f,
                            maxRadiusOf(daughter) + blockerMaxR);
                        if (norm < bestNorm) {
                            bestNorm = norm;
                            bestDist = dist;
                            bestDaughterName = daughter.getName();
                        }
                    }

                    if (bestNorm <= normLimit && bestDist <= distLimit) {
                        duplicateHandoffRemoveNames.insert(blockerName);
                        std::cout << "[Split Duplicate Handoff Candidate] frame "
                                  << displayFrame
                                  << " parent=" << record.parentName
                                  << " daughter=" << bestDaughterName
                                  << " blocker=" << blockerName
                                  << " dist=" << bestDist
                                  << " normalized=" << bestNorm
                                  << " continuationPenalty="
                                  << proposal.continuationClaimSoftPenalty
                                  << std::endl;
                    }
	                }
	            }

	            const auto &currentLumenCentersForHandoff =
	                getCellLumenLookaheadCandidates(frameIndex);
	            auto isCurrentFrameLumenCandidateIdForHandoff = [](int candidateId) {
	                return candidateId >= 0 && candidateId < 1000000;
	            };
	            for (const AcceptedLumenSplitRecord &record : acceptedLumenSplits) {
	                const BridgeSplitProposal &proposal = record.proposal;
	                const bool realA =
	                    isCurrentFrameLumenCandidateIdForHandoff(proposal.candidateIdA);
	                const bool realB =
	                    isCurrentFrameLumenCandidateIdForHandoff(proposal.candidateIdB);
	                const bool cleanWindow =
	                    proposal.windowBothDaughtersSupported >= 2 &&
	                    proposal.windowMissingDaughterCount == 0 &&
	                    proposal.windowParentPersists == 0;
	                if (proposal.parentAnchored || !cleanWindow || !realA || !realB ||
	                    currentLumenCentersForHandoff.empty()) {
	                    continue;
	                }

	                const std::array<std::string, 2> daughterNames = {
	                    record.parentName + "0",
	                    record.parentName + "1"};
	                std::vector<int> daughterIndexes;
	                for (const std::string &daughterName : daughterNames) {
	                    const int daughterIdx = findCellIndexByName(daughterName);
	                    if (daughterIdx >= 0) {
	                        daughterIndexes.push_back(daughterIdx);
	                    }
	                }
	                if (daughterIndexes.empty()) {
	                    continue;
	                }

	                const std::set<int> splitCandidateIds = {
	                    proposal.candidateIdA,
	                    proposal.candidateIdB};
	                for (size_t oi = 0; oi < frame.cells.size(); ++oi) {
	                    const Ellipsoid &other = frame.cells[oi];
	                    const std::string &otherName = other.getName();
	                    if (other.isTrash() ||
	                        otherName == record.parentName ||
	                        otherName == daughterNames[0] ||
	                        otherName == daughterNames[1] ||
	                        splitAcceptedInLoop.count(otherName) > 0 ||
	                        mergedContinuations.count(otherName) > 0 ||
	                        duplicateHandoffRemoveNames.count(otherName) > 0) {
	                        continue;
	                    }

	                    const cv::Point3f otherPos = centerOf(other);
	                    const float otherMaxR = maxRadiusOf(other);
	                    float bestDaughterDist =
	                        std::numeric_limits<float>::infinity();
	                    float bestDaughterNorm =
	                        std::numeric_limits<float>::infinity();
	                    std::string bestDaughterName;
	                    for (const int daughterIdx : daughterIndexes) {
	                        const Ellipsoid &daughter =
	                            frame.cells[static_cast<size_t>(daughterIdx)];
	                        const float dist = static_cast<float>(
	                            cv::norm(centerOf(daughter) - otherPos));
	                        const float norm = dist / std::max(
	                            1.0f,
	                            maxRadiusOf(daughter) + otherMaxR);
	                        if (norm < bestDaughterNorm) {
	                            bestDaughterNorm = norm;
	                            bestDaughterDist = dist;
	                            bestDaughterName = daughter.getName();
	                        }
	                    }
	                    if (bestDaughterNorm > normLimit ||
	                        bestDaughterDist > distLimit) {
	                        continue;
	                    }

	                    float nearestAcceptedCenter =
	                        std::numeric_limits<float>::infinity();
	                    float nearestIndependentCenter =
	                        std::numeric_limits<float>::infinity();
	                    for (const auto &center : currentLumenCentersForHandoff) {
	                        const float centerDist = static_cast<float>(
	                            cv::norm(center.position - otherPos));
	                        if (splitCandidateIds.count(center.candidateId) > 0) {
	                            nearestAcceptedCenter =
	                                std::min(nearestAcceptedCenter, centerDist);
	                        } else {
	                            nearestIndependentCenter =
	                                std::min(nearestIndependentCenter, centerDist);
	                        }
	                    }
	                    if (!std::isfinite(nearestAcceptedCenter) ||
	                        nearestAcceptedCenter > distLimit) {
	                        continue;
	                    }
	                    if (std::isfinite(nearestIndependentCenter) &&
	                        nearestIndependentCenter + 4.0f <
	                            nearestAcceptedCenter) {
	                        continue;
	                    }

	                    duplicateHandoffRemoveNames.insert(otherName);
	                    std::cout << "[Split Duplicate Handoff Nearby Candidate] frame "
	                              << displayFrame
	                              << " parent=" << record.parentName
	                              << " daughter=" << bestDaughterName
	                              << " stale=" << otherName
	                              << " daughterDist=" << bestDaughterDist
	                              << " normalized=" << bestDaughterNorm
	                              << " acceptedCenterDist="
	                              << nearestAcceptedCenter
	                              << " independentCenterDist="
	                              << nearestIndependentCenter
	                              << std::endl;
	                }
	            }

	            for (const std::string &removeName : duplicateHandoffRemoveNames) {
	                for (auto it = frame.cells.begin(); it != frame.cells.end();
	                     ++it) {
                    if (it->getName() != removeName) {
                        continue;
                    }
                    frame.cells.erase(it);
                    previousSnapshots.erase(removeName);
                    cellShapeReference.erase(removeName);
                    cellShapeBirth.erase(removeName);
                    cellFirstSeenFrame.erase(removeName);
                    ++duplicateHandoffPruneCount;
                    std::cout << "[Split Duplicate Handoff Prune] frame "
                              << displayFrame
                              << " removed=" << removeName
                              << std::endl;
                    break;
                }
            }
        }

        if (continuationMergeCount > 0 || duplicateHandoffPruneCount > 0) {
            frame.regenerateSynthFrame();
            if (voronoiMapNeeded) {
                frame.rebuildVoronoiMap();
            }
            std::cout << "[Split Continuation Merge Summary] frame "
                      << displayFrame
                      << " merged=" << continuationMergeCount
                      << " duplicate_handoff_pruned="
                      << duplicateHandoffPruneCount
                      << " cells=" << frame.cells.size()
                      << std::endl;
        }
    }

    auto applyGlobalLumenCenterAssignment = [&]() {
        if (!config.cellLumen.fusionCenterPriorEnabled) {
            return;
        }

        const auto &centers = getCellLumenLookaheadCandidates(frameIndex);
        if (centers.empty() || frame.cells.empty()) {
            return;
        }

        struct CenterEdge {
            size_t cellIdx = 0;
            size_t centerIdx = 0;
            float distance = 0.0f;
            float score = 0.0f;
            float benefit = 0.0f;
        };

        const float configuredCenterMax =
            std::max(0.0f, config.cellLumen.fusionCenterPriorMaxDistance);
        const float windowCenterMax =
            std::max(0.0f, config.cellLumen.fusionSplitPriorWindowMatchDistance) +
            std::max(0.0f, config.cellLumen.fusionSplitPriorWindowMatchDistancePerFrame);
        const float assignMaxDistance =
            std::max(configuredCenterMax, windowCenterMax);
        const float minSignal = std::max(
            0.0f,
            config.cellLumen.fusionSplitPriorMinTop10MinusShell >= 0.0f
                ? config.cellLumen.fusionSplitPriorMinTop10MinusShell
                : config.cellLumen.fusionMinTop10MinusShell);
        const int minVoxels =
            config.cellLumen.fusionSplitPriorMinVoxels >= 0
                ? config.cellLumen.fusionSplitPriorMinVoxels
                : config.cellLumen.fusionMinVoxels;

        std::vector<CenterEdge> edges;
        edges.reserve(frame.cells.size() * std::min<size_t>(centers.size(), 8));
        for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
            const Ellipsoid &cell = frame.cells[ci];
            if (cell.isTrash()) {
                continue;
            }
            const cv::Point3f cellPos(cell.getX(), cell.getY(), cell.getZ());
            for (size_t centerIdx = 0; centerIdx < centers.size(); ++centerIdx) {
                const auto &center = centers[centerIdx];
                if (center.voxelCount < minVoxels || center.signal < minSignal) {
                    continue;
                }
                const float distance =
                    static_cast<float>(cv::norm(cellPos - center.position));
                if (distance > assignMaxDistance) {
                    continue;
                }
                const float signalBonus =
                    std::min(8.0f, std::max(0.0f, center.signal - minSignal) * 0.04f);
                const float voxelBonus =
                    std::min(4.0f, std::log1p(static_cast<float>(center.voxelCount)) * 0.35f);
                const float nearBonus =
                    distance <= configuredCenterMax ? 4.0f : 0.0f;
                CenterEdge edge;
                edge.cellIdx = ci;
                edge.centerIdx = centerIdx;
                edge.distance = distance;
                edge.score = distance - signalBonus - voxelBonus - nearBonus;
                edge.benefit = assignMaxDistance - distance +
                               signalBonus + voxelBonus + nearBonus;
                if (edge.benefit > 0.0f) {
                    edges.push_back(edge);
                }
            }
        }

        if (edges.empty()) {
            return;
        }

        std::sort(edges.begin(), edges.end(),
                  [](const CenterEdge &a, const CenterEdge &b) {
                      if (a.benefit != b.benefit) {
                          return a.benefit > b.benefit;
                      }
                      if (a.score != b.score) {
                          return a.score < b.score;
                      }
                      return a.distance < b.distance;
                  });

        std::vector<int> assignedCenterForCell(frame.cells.size(), -1);
        std::vector<char> usedCenter(centers.size(), 0);
        int selectedEdges = 0;
        for (const CenterEdge &edge : edges) {
            if (assignedCenterForCell[edge.cellIdx] >= 0 ||
                usedCenter[edge.centerIdx]) {
                continue;
            }
            assignedCenterForCell[edge.cellIdx] =
                static_cast<int>(edge.centerIdx);
            usedCenter[edge.centerIdx] = 1;
            ++selectedEdges;
        }

        int moved = 0;
        for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
            const int assignedCenterIdx = assignedCenterForCell[ci];
            if (assignedCenterIdx < 0) {
                continue;
            }
            Ellipsoid &cell = frame.cells[ci];
            const auto &center = centers[static_cast<size_t>(assignedCenterIdx)];
            const cv::Point3f oldPos(cell.getX(), cell.getY(), cell.getZ());
            const float moveDistance =
                static_cast<float>(cv::norm(oldPos - center.position));
            if (moveDistance < 1.0f) {
                continue;
            }

            cell.setPosition(center.position.x, center.position.y, center.position.z);
            lockPcaPosition(cell.getName(), "global_lumen_center_assignment");
            ++moved;

            if (candidateGraphLog) {
                CandidateGraphRow row;
                row.frame = displayFrame;
                row.kind = "continuation";
                row.source = "global_lumen_center_assignment";
                row.parent = cell.getName();
                row.candidateA = std::to_string(center.candidateId);
                row.selected = 1;
                row.score = moveDistance;
                row.rawScore = selectedEdges;
                row.imageGain = center.signal;
                row.sep = assignMaxDistance;
                row.parentShape = cell.shapeElongation();
                row.d1 = oldPos;
                row.d2 = center.position;
                row.voxA = center.voxelCount;
                row.signalA = center.signal;
                std::ostringstream note;
                note << "one_to_one_lumen_center_assignment"
                     << ";assign_max_distance=" << assignMaxDistance
                     << ";min_signal=" << minSignal
                     << ";min_voxels=" << minVoxels;
                row.note = note.str();
                writeCandidateGraphRow(candidateGraphLog, row);
            }

            std::cout << "[CellLumen Global Center Assignment] frame "
                      << displayFrame
                      << " cell=" << cell.getName()
                      << " candidate=" << center.candidateId
                      << " move=" << moveDistance
                      << " old=(" << oldPos.x << "," << oldPos.y << "," << oldPos.z << ")"
                      << " new=(" << center.position.x << "," << center.position.y
                      << "," << center.position.z << ")"
                      << " signal=" << center.signal
                      << " voxels=" << center.voxelCount
                      << std::endl;
        }

        if (moved > 0) {
            frame.regenerateSynthFrame();
            if (voronoiMapNeeded) {
                frame.rebuildVoronoiMap();
            }
            std::cout << "[CellLumen Global Center Assignment Summary] frame "
                      << displayFrame
                      << " edges=" << edges.size()
                      << " selected=" << selectedEdges
                      << " moved=" << moved
                      << " centers=" << centers.size()
                      << " assign_max_distance=" << assignMaxDistance
                      << std::endl;
        }
    };

    auto applyTemporalLumenCenterRepair = [&]() {
        const CellLumenConfig &lumenConfig = config.cellLumen;
        if (!lumenConfig.fusionTemporalCenterRepairEnabled ||
            !lumenConfig.fusionSplitPriorWindowEnabled ||
            lumenConfig.fusionSplitPriorWindowSize <= 1) {
            return;
        }

        const auto &centers = getCellLumenLookaheadCandidates(frameIndex);
        if (centers.empty() || frame.cells.empty()) {
            return;
        }

        const int windowSize =
            std::clamp(lumenConfig.fusionSplitPriorWindowSize, 2, 5);
        std::map<int, const std::vector<CellLumenLookaheadCandidate> *> futureByOffset;
        for (int offset = 1; offset < windowSize; ++offset) {
            const int lookaheadFrameIndex = frameIndex + offset;
            if (lookaheadFrameIndex < 0 ||
                static_cast<size_t>(lookaheadFrameIndex) >= frames.size()) {
                break;
            }
            const auto &futureCandidates =
                getCellLumenLookaheadCandidates(lookaheadFrameIndex);
            futureByOffset[offset] = &futureCandidates;
        }
        if (futureByOffset.empty()) {
            return;
        }

        struct PointWindowSupport {
            int supported = 0;
            int checked = 0;
            float bestDistanceSum = 0.0f;
        };
        auto pointWindowSupport = [&](const cv::Point3f &point) {
            PointWindowSupport support;
            for (const auto &entry : futureByOffset) {
                const int offset = entry.first;
                const auto *futureCandidates = entry.second;
                if (futureCandidates == nullptr || futureCandidates->empty()) {
                    continue;
                }
                ++support.checked;
                float best = std::numeric_limits<float>::infinity();
                for (const auto &future : *futureCandidates) {
                    best = std::min(
                        best,
                        static_cast<float>(cv::norm(future.position - point)));
                }
                const float matchDistance =
                    std::max(0.0f, lumenConfig.fusionSplitPriorWindowMatchDistance) +
                    std::max(0.0f,
                             lumenConfig.fusionSplitPriorWindowMatchDistancePerFrame) *
                        static_cast<float>(std::max(0, offset - 1));
                if (best <= matchDistance) {
                    ++support.supported;
                    support.bestDistanceSum += best;
                }
            }
            return support;
        };

        auto isRealLumenCandidateId = [](int candidateId) {
            return candidateId >= 0 && candidateId < 1000000000;
        };
        std::set<int> acceptedSplitCandidateIds;
        for (const AcceptedLumenSplitRecord &record : acceptedLumenSplits) {
            if (isRealLumenCandidateId(record.proposal.candidateIdA)) {
                acceptedSplitCandidateIds.insert(record.proposal.candidateIdA);
            }
            if (isRealLumenCandidateId(record.proposal.candidateIdB)) {
                acceptedSplitCandidateIds.insert(record.proposal.candidateIdB);
            }
        }
        std::set<int> riskySplitRepairBlockedCandidateIds;
        std::unordered_map<std::string, std::unordered_set<int>>
            splitCandidateIdsByParentForRepair;
        const auto splitPriorItForRepair = cellLumenSplitPriors.find(frameIndex);
        if (splitPriorItForRepair != cellLumenSplitPriors.end()) {
            for (const auto &entry : splitPriorItForRepair->second) {
                const std::string &parentName = entry.first;
                const BridgeSplitProposal &proposal = entry.second;
                if (isRealLumenCandidateId(proposal.candidateIdA)) {
                    splitCandidateIdsByParentForRepair[parentName].insert(
                        proposal.candidateIdA);
                }
                if (isRealLumenCandidateId(proposal.candidateIdB)) {
                    splitCandidateIdsByParentForRepair[parentName].insert(
                        proposal.candidateIdB);
                }
                const float priorScoreNoWindowBonus =
                    proposal.elongation +
                    std::max(0.0f, proposal.balancedWindowBonus);
                const bool riskyNeighborClaimSplit =
                    proposal.neighborClaimPenalty >= 8.0f &&
                    proposal.parentDistanceBalance < 0.75f &&
                    priorScoreNoWindowBonus >= 20.0f;
                if (!riskyNeighborClaimSplit) {
                    continue;
                }
                if (isRealLumenCandidateId(proposal.candidateIdA)) {
                    riskySplitRepairBlockedCandidateIds.insert(proposal.candidateIdA);
                }
                if (isRealLumenCandidateId(proposal.candidateIdB)) {
                    riskySplitRepairBlockedCandidateIds.insert(proposal.candidateIdB);
                }
            }
        }

        const int minVoxels =
            lumenConfig.fusionSplitPriorMinVoxels >= 0
                ? lumenConfig.fusionSplitPriorMinVoxels
                : lumenConfig.fusionMinVoxels;
        const float minSignal =
            lumenConfig.fusionSplitPriorMinTop10MinusShell >= 0.0f
                ? lumenConfig.fusionSplitPriorMinTop10MinusShell
                : lumenConfig.fusionMinTop10MinusShell;
        const float minMove =
            std::max(0.0f, lumenConfig.fusionTemporalCenterRepairMinDistance);
        const float maxMove =
            std::max(minMove, lumenConfig.fusionTemporalCenterRepairMaxDistance);
        const int minWindowSupport =
            std::max(1, lumenConfig.fusionTemporalCenterRepairMinWindowSupport);
        const int maxOldWindowSupport =
            std::max(0, lumenConfig.fusionTemporalCenterRepairMaxOldWindowSupport);
        const float minWindowDistanceGain = std::max(
            0.0f,
            lumenConfig.fusionTemporalCenterRepairMinWindowDistanceGain);
        const float claimMargin =
            std::max(0.0f, lumenConfig.fusionTemporalCenterRepairClaimMargin);
        const int minCellAge =
            std::max(0, lumenConfig.fusionTemporalCenterRepairMinCellAgeFrames);
        const int oldSupportedMinVoxels =
            lumenConfig.fusionTemporalCenterRepairOldSupportedMinVoxels;
        const float oldSupportedMinSignal =
            lumenConfig.fusionTemporalCenterRepairOldSupportedMinSignal;
        const bool parentAnchorReanchorEnabled =
            lumenConfig.fusionTemporalCenterRepairParentAnchorReanchorEnabled;
        const float parentAnchorReanchorMinShape = std::max(
            1.0f,
            lumenConfig.fusionTemporalCenterRepairParentAnchorReanchorMinShape);
        const float parentAnchorReanchorMinMove = std::max(
            0.0f,
            lumenConfig.fusionTemporalCenterRepairParentAnchorReanchorMinDistance);
        const bool sameParentSplitReanchorEnabled =
            lumenConfig.fusionTemporalCenterRepairSameParentSplitReanchorEnabled;
        const float sameParentSplitReanchorMinMove = std::max(
            0.0f,
            lumenConfig.fusionTemporalCenterRepairSameParentSplitMinDistance);
        const int sameParentSplitReanchorMinVoxels = std::max(
            0,
            lumenConfig.fusionTemporalCenterRepairSameParentSplitMinVoxels);
        const float sameParentSplitReanchorMinSignal = std::max(
            0.0f,
            lumenConfig.fusionTemporalCenterRepairSameParentSplitMinSignal);
        const int sameParentSplitReanchorMinWindowSupport = std::max(
            1,
            lumenConfig.fusionTemporalCenterRepairSameParentSplitMinWindowSupport);
        const bool sameParentRequireSupportGainWhenOldSupported =
            lumenConfig
                .fusionTemporalCenterRepairSameParentRequireSupportGainWhenOldSupported;
        std::unordered_map<std::string, std::unordered_set<int>>
            parentAnchorReanchorCandidateIdsByParent;
        const auto centerReanchorIt =
            cellLumenCenterReanchorCandidateIds.find(frameIndex);
        if (parentAnchorReanchorEnabled &&
            centerReanchorIt != cellLumenCenterReanchorCandidateIds.end()) {
            for (const auto &entry : centerReanchorIt->second) {
                for (int candidateId : entry.second) {
                    parentAnchorReanchorCandidateIdsByParent[entry.first].insert(
                        candidateId);
                }
            }
        }
        if (parentAnchorReanchorEnabled &&
            splitPriorItForRepair != cellLumenSplitPriors.end()) {
            for (const auto &entry : splitPriorItForRepair->second) {
                const std::string &parentName = entry.first;
                const BridgeSplitProposal &proposal = entry.second;
                const bool realA = isRealLumenCandidateId(proposal.candidateIdA);
                const bool realB = isRealLumenCandidateId(proposal.candidateIdB);
                const bool oneRealParentAnchor =
                    proposal.parentAnchored &&
                    proposal.parentShapeElongation >=
                        parentAnchorReanchorMinShape &&
                    (realA != realB) &&
                    proposal.windowBothDaughtersSupported >= minWindowSupport &&
                    proposal.windowMissingDaughterCount == 0 &&
                    proposal.windowParentPersists == 0;
                if (!oneRealParentAnchor) {
                    continue;
                }
                if (realA) {
                    parentAnchorReanchorCandidateIdsByParent[parentName].insert(
                        proposal.candidateIdA);
                }
                if (realB) {
                    parentAnchorReanchorCandidateIdsByParent[parentName].insert(
                        proposal.candidateIdB);
                }
            }
        }

        struct RepairEdge {
            size_t cellIdx = 0;
            size_t centerIdx = 0;
            float moveDistance = 0.0f;
            float nearestOtherDistance = std::numeric_limits<float>::infinity();
            PointWindowSupport newSupport;
            PointWindowSupport oldSupport;
            float score = 0.0f;
            bool parentAnchorReanchor = false;
        };

        std::vector<RepairEdge> edges;
        int considered = 0;
        int rejectedAge = 0;
        int rejectedSignal = 0;
        int rejectedReserved = 0;
        int rejectedDistance = 0;
        int rejectedOldSupport = 0;
        int rejectedCurrentFrameSupport = 0;
        int rejectedWindow = 0;
        int rejectedClaim = 0;
        int rejectedRiskySplitCandidate = 0;
        int allowedRejectedSplitCenterRepair = 0;
        int allowedClaimByOtherOwnSupport = 0;
        const float currentFrameSupportDistance = std::max(
            0.0f,
            std::min(minMove, lumenConfig.fusionSplitPriorWindowMatchDistance));

        for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
            const Ellipsoid &cell = frame.cells[ci];
            if (cell.isTrash()) {
                continue;
            }
            const std::string &cellName = cell.getName();
            int cellAge = std::numeric_limits<int>::max() / 4;
            if (const auto firstSeenIt = cellFirstSeenFrame.find(cellName);
                firstSeenIt != cellFirstSeenFrame.end()) {
                cellAge = std::max(0, displayFrame - firstSeenIt->second);
            }
            if (cellAge < minCellAge) {
                ++rejectedAge;
                continue;
            }

            const cv::Point3f oldPos(cell.getX(), cell.getY(), cell.getZ());
            const PointWindowSupport oldSupport = pointWindowSupport(oldPos);
            float oldNearestCurrentFrameCenterDistance =
                std::numeric_limits<float>::infinity();
            int oldNearestCurrentFrameCandidateId = -1;
            for (const auto &center : centers) {
                if (center.voxelCount < minVoxels || center.signal < minSignal) {
                    continue;
                }
                if (center.candidateId >= 0 &&
                    acceptedSplitCandidateIds.count(center.candidateId) > 0) {
                    continue;
                }
                const float oldCenterDistance =
                    static_cast<float>(cv::norm(oldPos - center.position));
                if (oldCenterDistance < oldNearestCurrentFrameCenterDistance) {
                    oldNearestCurrentFrameCenterDistance = oldCenterDistance;
                    oldNearestCurrentFrameCandidateId = center.candidateId;
                }
            }
            const bool oldHasCurrentFrameCenterSupport =
                std::isfinite(oldNearestCurrentFrameCenterDistance) &&
                oldNearestCurrentFrameCenterDistance <= currentFrameSupportDistance;

            for (size_t centerIdx = 0; centerIdx < centers.size(); ++centerIdx) {
                const auto &center = centers[centerIdx];
                ++considered;
                if (center.voxelCount < minVoxels || center.signal < minSignal) {
                    ++rejectedSignal;
                    continue;
                }
                if (center.candidateId >= 0 &&
                    acceptedSplitCandidateIds.count(center.candidateId) > 0) {
                    ++rejectedReserved;
                    continue;
                }
                const bool riskySplitCandidate =
                    center.candidateId >= 0 &&
                    riskySplitRepairBlockedCandidateIds.count(center.candidateId) > 0;
                bool parentAnchorReanchorCandidate = false;
                if (center.candidateId >= 0) {
                    const auto reanchorIt =
                        parentAnchorReanchorCandidateIdsByParent.find(cellName);
                    parentAnchorReanchorCandidate =
                        reanchorIt !=
                            parentAnchorReanchorCandidateIdsByParent.end() &&
                        reanchorIt->second.count(center.candidateId) > 0;
                }
                bool sameParentSplitCandidate = false;
                if (!parentAnchorReanchorCandidate && center.candidateId >= 0) {
                    const auto splitCandidateIt =
                        splitCandidateIdsByParentForRepair.find(cellName);
                    if (splitCandidateIt !=
                            splitCandidateIdsByParentForRepair.end() &&
                        splitCandidateIt->second.count(center.candidateId) > 0) {
                        sameParentSplitCandidate = true;
                    }
                }
                const bool strongNearestCurrentSupportForReanchor =
                    sameParentSplitReanchorEnabled &&
                    !parentAnchorReanchorCandidate &&
                    oldHasCurrentFrameCenterSupport &&
                    center.candidateId == oldNearestCurrentFrameCandidateId &&
                    center.voxelCount >= sameParentSplitReanchorMinVoxels &&
                    center.signal >= sameParentSplitReanchorMinSignal;
                // A one-real-candidate split proposal can actually be the
                // current daughter/parent's continuation center. Rejected split
                // proposals are not stored in cellLumenSplitPriors, so the
                // nearest current-frame Cell Lumen support is used as the
                // conservative fallback signal for this same-track reanchor.
                const bool sameParentSplitReanchorCandidate =
                    strongNearestCurrentSupportForReanchor &&
                    (!sameParentSplitCandidate ||
                     !oldHasCurrentFrameCenterSupport ||
                     center.candidateId == oldNearestCurrentFrameCandidateId);
                if (!parentAnchorReanchorCandidate &&
                    !sameParentSplitReanchorCandidate &&
                    oldHasCurrentFrameCenterSupport &&
                    center.candidateId != oldNearestCurrentFrameCandidateId) {
                    ++rejectedCurrentFrameSupport;
                    continue;
                }
                const float moveDistance =
                    static_cast<float>(cv::norm(oldPos - center.position));
                const float effectiveMinMove =
                    parentAnchorReanchorCandidate
                        ? std::min(minMove, parentAnchorReanchorMinMove)
                        : sameParentSplitReanchorCandidate
                            ? std::min(minMove, sameParentSplitReanchorMinMove)
                        : minMove;
                if (moveDistance < effectiveMinMove || moveDistance > maxMove) {
                    ++rejectedDistance;
                    continue;
                }

                float nearestOtherDistance = std::numeric_limits<float>::infinity();
                int nearestOtherIdx = -1;
                for (size_t oi = 0; oi < frame.cells.size(); ++oi) {
                    if (oi == ci || frame.cells[oi].isTrash()) {
                        continue;
                    }
                    const cv::Point3f otherPos(frame.cells[oi].getX(),
                                               frame.cells[oi].getY(),
                                               frame.cells[oi].getZ());
                    const float otherDistance =
                        static_cast<float>(cv::norm(otherPos - center.position));
                    if (otherDistance < nearestOtherDistance) {
                        nearestOtherDistance = otherDistance;
                        nearestOtherIdx = static_cast<int>(oi);
                    }
                }
                if (std::isfinite(nearestOtherDistance) &&
                    nearestOtherDistance + claimMargin < moveDistance) {
                    bool nearestOtherHasOwnSupportedCenter = false;
                    if (nearestOtherIdx >= 0 &&
                        static_cast<size_t>(nearestOtherIdx) < frame.cells.size()) {
                        const Ellipsoid &nearestOther =
                            frame.cells[static_cast<size_t>(nearestOtherIdx)];
                        const cv::Point3f nearestOtherPos(nearestOther.getX(),
                                                          nearestOther.getY(),
                                                          nearestOther.getZ());
                        for (size_t otherCenterIdx = 0;
                             otherCenterIdx < centers.size();
                             ++otherCenterIdx) {
                            if (otherCenterIdx == centerIdx) {
                                continue;
                            }
                            const auto &otherCenter = centers[otherCenterIdx];
                            if (otherCenter.voxelCount < minVoxels ||
                                otherCenter.signal < minSignal) {
                                continue;
                            }
                            if (otherCenter.candidateId >= 0 &&
                                acceptedSplitCandidateIds.count(
                                    otherCenter.candidateId) > 0) {
                                continue;
                            }
                            const float otherOwnDistance =
                                static_cast<float>(cv::norm(
                                    nearestOtherPos - otherCenter.position));
                            const PointWindowSupport otherOwnSupport =
                                pointWindowSupport(otherCenter.position);
                            if (otherOwnSupport.supported >= minWindowSupport &&
                                otherOwnDistance + claimMargin <
                                    nearestOtherDistance) {
                                nearestOtherHasOwnSupportedCenter = true;
                                break;
                            }
                        }
                    }
                    if (!nearestOtherHasOwnSupportedCenter) {
                        ++rejectedClaim;
                        continue;
                    }
                    ++allowedClaimByOtherOwnSupport;
                }

                const PointWindowSupport newSupport =
                    pointWindowSupport(center.position);
                const int requiredWindowSupport =
                    sameParentSplitReanchorCandidate
                        ? std::min(minWindowSupport,
                                   sameParentSplitReanchorMinWindowSupport)
                        : minWindowSupport;
                if (newSupport.supported < requiredWindowSupport) {
                    ++rejectedWindow;
                    continue;
                }
                const bool strongRejectedSplitCenter =
                    center.voxelCount >= std::max(minVoxels * 2, 1000) &&
                    center.signal >= std::max(minSignal, 60.0f);
                const bool sameParentRejectedSplitCenterRepair =
                    sameParentSplitCandidate &&
                    !parentAnchorReanchorCandidate &&
                    oldSupport.supported <= maxOldWindowSupport &&
                    (strongRejectedSplitCenter ||
                     sameParentSplitReanchorCandidate);
                if (riskySplitCandidate &&
                    !sameParentRejectedSplitCenterRepair) {
                    ++rejectedRiskySplitCandidate;
                    continue;
                }
                if (!parentAnchorReanchorCandidate &&
                    sameParentSplitCandidate &&
                    !sameParentRejectedSplitCenterRepair) {
                    ++rejectedRiskySplitCandidate;
                    continue;
                }
                if (sameParentRejectedSplitCenterRepair) {
                    ++allowedRejectedSplitCenterRepair;
                }
                if (oldSupport.supported > maxOldWindowSupport) {
                    const float oldAverageDistance =
                        oldSupport.supported > 0
                            ? oldSupport.bestDistanceSum /
                                  static_cast<float>(oldSupport.supported)
                            : std::numeric_limits<float>::infinity();
                    const float newAverageDistance =
                        newSupport.supported > 0
                            ? newSupport.bestDistanceSum /
                                  static_cast<float>(newSupport.supported)
                            : std::numeric_limits<float>::infinity();
                    const bool newHasMoreSupport =
                        newSupport.supported > oldSupport.supported;
                    const bool newIsClearlyCloser =
                        newSupport.supported >= oldSupport.supported &&
                        newAverageDistance + minWindowDistanceGain <
                            oldAverageDistance;
                    const bool strongEnoughToOverrideOldSupport =
                        (oldSupportedMinVoxels < 0 ||
                         center.voxelCount >= oldSupportedMinVoxels) &&
                        (oldSupportedMinSignal < 0.0f ||
                         center.signal >= oldSupportedMinSignal);
                    const bool parentAnchorReanchorOverridesOldSupport =
                        parentAnchorReanchorCandidate &&
                        strongEnoughToOverrideOldSupport &&
                        newHasMoreSupport;
                    const bool oldCenterStillFullySupported =
                        oldSupport.supported >= minWindowSupport;
                    // f034 showed that same-parent reanchor can be too eager:
                    // a current-frame Cell Lumen center may be the nearest
                    // local bright blob, but if the old cell center already
                    // has future support and the new center does not increase
                    // that support count, moving the cell can turn a nearly
                    // correct continuation into a threshold miss. Keep this
                    // rule switch-gated so older validated profiles are not
                    // affected unless the YAML opts in.
                    const bool sameParentNoSupportGainFromOldSupported =
                        sameParentRequireSupportGainWhenOldSupported &&
                        sameParentSplitReanchorCandidate &&
                        oldSupport.supported > 0 &&
                        !newHasMoreSupport;
                    const bool futureEvidenceOverridesOldSupport =
                        !sameParentNoSupportGainFromOldSupported &&
                        !oldCenterStillFullySupported &&
                        strongEnoughToOverrideOldSupport &&
                        (newHasMoreSupport || newIsClearlyCloser);
                    if (!futureEvidenceOverridesOldSupport &&
                        !parentAnchorReanchorOverridesOldSupport) {
                        ++rejectedOldSupport;
                        continue;
                    }
                }

                RepairEdge edge;
                edge.cellIdx = ci;
                edge.centerIdx = centerIdx;
                edge.moveDistance = moveDistance;
                edge.nearestOtherDistance = nearestOtherDistance;
                edge.newSupport = newSupport;
                edge.oldSupport = oldSupport;
                edge.parentAnchorReanchor = parentAnchorReanchorCandidate;
                edge.score =
                    moveDistance -
                    static_cast<float>(newSupport.supported) * 12.0f +
                    oldSupport.bestDistanceSum * 0.05f -
                    std::min(6.0f, std::max(0.0f, center.signal - minSignal) * 0.03f);
                edges.push_back(edge);
            }
        }

        if (edges.empty()) {
            std::cout << "[CellLumen Temporal Center Repair Summary] frame "
                      << displayFrame
                      << " moved=0"
                      << " considered=" << considered
                      << " accepted_split_reserved=" << acceptedSplitCandidateIds.size()
                      << " rejected_age=" << rejectedAge
                      << " rejected_signal=" << rejectedSignal
                      << " rejected_reserved=" << rejectedReserved
                      << " rejected_distance=" << rejectedDistance
                      << " rejected_old_support=" << rejectedOldSupport
                      << " rejected_current_frame_support="
                      << rejectedCurrentFrameSupport
                      << " rejected_window=" << rejectedWindow
                      << " rejected_claim=" << rejectedClaim
                      << " rejected_risky_split_candidate="
                      << rejectedRiskySplitCandidate
                      << " allowed_rejected_split_center_repair="
                      << allowedRejectedSplitCenterRepair
                      << " allowed_claim_by_other_own_support="
                      << allowedClaimByOtherOwnSupport
                      << " min_move=" << minMove
                      << " max_move=" << maxMove
                      << " min_window_support=" << minWindowSupport
                      << " max_old_window_support=" << maxOldWindowSupport
                      << " min_window_distance_gain=" << minWindowDistanceGain
                      << " current_frame_support_distance="
                      << currentFrameSupportDistance
                      << " old_supported_min_signal=" << oldSupportedMinSignal
                      << " old_supported_min_voxels=" << oldSupportedMinVoxels
                      << std::endl;
            return;
        }

        std::sort(edges.begin(), edges.end(),
                  [](const RepairEdge &a, const RepairEdge &b) {
                      if (a.newSupport.supported != b.newSupport.supported) {
                          return a.newSupport.supported > b.newSupport.supported;
                      }
                      if (std::abs(a.score - b.score) > 1e-5f) {
                          return a.score < b.score;
                      }
                      return a.moveDistance < b.moveDistance;
                  });

        std::vector<int> assignedCenterForCell(frame.cells.size(), -1);
        std::vector<char> usedCenter(centers.size(), 0);
        int selected = 0;
        for (const RepairEdge &edge : edges) {
            if (assignedCenterForCell[edge.cellIdx] >= 0 ||
                usedCenter[edge.centerIdx]) {
                continue;
            }
            assignedCenterForCell[edge.cellIdx] =
                static_cast<int>(edge.centerIdx);
            usedCenter[edge.centerIdx] = 1;
            ++selected;
        }

        int moved = 0;
        for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
            const int assignedCenterIdx = assignedCenterForCell[ci];
            if (assignedCenterIdx < 0) {
                continue;
            }
            Ellipsoid &cell = frame.cells[ci];
            const auto &center = centers[static_cast<size_t>(assignedCenterIdx)];
            const cv::Point3f oldPos(cell.getX(), cell.getY(), cell.getZ());
            const float moveDistance =
                static_cast<float>(cv::norm(oldPos - center.position));
            cell.setPosition(center.position.x, center.position.y, center.position.z);
            lockPcaPosition(cell.getName(), "temporal_lumen_center_repair");
            ++moved;

            const PointWindowSupport oldSupport = pointWindowSupport(oldPos);
            const PointWindowSupport newSupport = pointWindowSupport(center.position);

            if (candidateGraphLog) {
                CandidateGraphRow row;
                row.frame = displayFrame;
                row.kind = "continuation";
                row.source = "temporal_lumen_center_repair";
                row.parent = cell.getName();
                row.candidateA = std::to_string(center.candidateId);
                row.selected = 1;
                row.score = moveDistance;
                row.imageGain = center.signal;
                row.sep = maxMove;
                row.parentShape = cell.shapeElongation();
                row.d1 = oldPos;
                row.d2 = center.position;
                row.voxA = center.voxelCount;
                row.signalA = center.signal;
                std::ostringstream note;
                note << "future_supported_continuation_center_repair"
                     << ";min_move=" << minMove
                     << ";max_move=" << maxMove
                     << ";new_window_support=" << newSupport.supported
                     << ";new_window_checked=" << newSupport.checked
                     << ";old_window_support=" << oldSupport.supported
                     << ";old_window_checked=" << oldSupport.checked
                     << ";max_old_window_support=" << maxOldWindowSupport
                     << ";min_window_support=" << minWindowSupport
                     << ";min_window_distance_gain=" << minWindowDistanceGain
                     << ";min_signal=" << minSignal
                     << ";min_voxels=" << minVoxels
                     << ";old_supported_min_signal=" << oldSupportedMinSignal
                     << ";old_supported_min_voxels=" << oldSupportedMinVoxels;
                row.note = note.str();
                writeCandidateGraphRow(candidateGraphLog, row);
            }

            std::cout << "[CellLumen Temporal Center Repair] frame "
                      << displayFrame
                      << " cell=" << cell.getName()
                      << " candidate=" << center.candidateId
                      << " move=" << moveDistance
                      << " old=(" << oldPos.x << "," << oldPos.y << "," << oldPos.z << ")"
                      << " new=(" << center.position.x << "," << center.position.y
                      << "," << center.position.z << ")"
                      << " old_window_support=" << oldSupport.supported
                     << " new_window_support=" << newSupport.supported
                     << " signal=" << center.signal
                     << " voxels=" << center.voxelCount
                      << std::endl;
        }

        if (moved > 0) {
            frame.regenerateSynthFrame();
            if (voronoiMapNeeded) {
                frame.rebuildVoronoiMap();
            }
        }

        std::cout << "[CellLumen Temporal Center Repair Summary] frame "
                  << displayFrame
                  << " moved=" << moved
                  << " selected=" << selected
                  << " candidates=" << edges.size()
                  << " considered=" << considered
                  << " accepted_split_reserved=" << acceptedSplitCandidateIds.size()
                  << " rejected_age=" << rejectedAge
                  << " rejected_signal=" << rejectedSignal
                  << " rejected_reserved=" << rejectedReserved
                  << " rejected_distance=" << rejectedDistance
                  << " rejected_old_support=" << rejectedOldSupport
                  << " rejected_current_frame_support="
                  << rejectedCurrentFrameSupport
                  << " rejected_window=" << rejectedWindow
                  << " rejected_claim=" << rejectedClaim
                  << " rejected_risky_split_candidate="
                  << rejectedRiskySplitCandidate
                  << " allowed_rejected_split_center_repair="
                  << allowedRejectedSplitCenterRepair
                  << " allowed_claim_by_other_own_support="
                  << allowedClaimByOtherOwnSupport
                  << " min_move=" << minMove
                  << " max_move=" << maxMove
                  << " min_window_support=" << minWindowSupport
                  << " max_old_window_support=" << maxOldWindowSupport
                  << " min_window_distance_gain=" << minWindowDistanceGain
                  << " current_frame_support_distance="
                  << currentFrameSupportDistance
                  << " old_supported_min_signal=" << oldSupportedMinSignal
                  << " old_supported_min_voxels=" << oldSupportedMinVoxels
                  << std::endl;
    };

    if (config.cellLumen.fusionTemporalCenterRepairEnabled) {
        ScopedStageTimer temporalCenterTimer(displayFrame,
                                             "temporal_lumen_center_repair");
        applyTemporalLumenCenterRepair();
    }

    const bool globalLumenCenterAssignmentEnabled = false;
    if (globalLumenCenterAssignmentEnabled) {
        ScopedStageTimer globalCenterTimer(displayFrame, "global_lumen_center_assignment");
        applyGlobalLumenCenterAssignment();
    }

    auto applyStrongLumenPcaMotionRelax = [&]() {
        const auto &centers = getCellLumenLookaheadCandidates(frameIndex);
        if (centers.empty() || frame.cells.empty()) {
            return;
        }
        const int windowSize =
            std::clamp(config.cellLumen.fusionSplitPriorWindowSize, 2, 5);
        std::map<int, const std::vector<CellLumenLookaheadCandidate> *> futureByOffset;
        for (int offset = 1; offset < windowSize; ++offset) {
            const int lookaheadFrameIndex = frameIndex + offset;
            if (lookaheadFrameIndex < 0 ||
                static_cast<size_t>(lookaheadFrameIndex) >= frames.size()) {
                break;
            }
            const auto &futureCandidates =
                getCellLumenLookaheadCandidates(lookaheadFrameIndex);
            futureByOffset[offset] = &futureCandidates;
        }
        if (futureByOffset.empty()) {
            return;
        }
        struct PointWindowSupport {
            int supported = 0;
            float bestDistanceSum = 0.0f;
        };
        auto pointWindowSupport = [&](const cv::Point3f &point) {
            PointWindowSupport support;
            for (const auto &entry : futureByOffset) {
                const int offset = entry.first;
                const auto *futureCandidates = entry.second;
                if (futureCandidates == nullptr || futureCandidates->empty()) {
                    continue;
                }
                float best = std::numeric_limits<float>::infinity();
                for (const auto &future : *futureCandidates) {
                    best = std::min(
                        best,
                        static_cast<float>(cv::norm(future.position - point)));
                }
                const float matchDistance =
                    std::max(
                        0.0f,
                        config.cellLumen.fusionSplitPriorWindowMatchDistance) +
                    std::max(
                        0.0f,
                        config.cellLumen
                            .fusionSplitPriorWindowMatchDistancePerFrame) *
                        static_cast<float>(std::max(0, offset - 1));
                if (best <= matchDistance) {
                    ++support.supported;
                    support.bestDistanceSum += best;
                }
            }
            return support;
        };

        const int minWindowSupport = std::max(
            1,
            config.cellLumen.fusionTemporalCenterRepairMinWindowSupport);
        const int maxOldWindowSupport = std::max(
            0,
            config.cellLumen.fusionTemporalCenterRepairMaxOldWindowSupport);
        const float minWindowDistanceGain = std::max(
            0.0f,
            config.cellLumen.fusionTemporalCenterRepairMinWindowDistanceGain);
        const int minVoxels = std::max(
            0,
            config.cellLumen.fusionSplitPriorTemporalRepairStrongMinVoxels);
        const float minSignal = std::max(
            0.0f,
            config.cellLumen.fusionSplitPriorTemporalRepairStrongMinSignal);
        const float maxMove = std::max(
            20.0f,
            config.cellLumen.fusionTemporalCenterRepairMaxDistance);
        const float claimMargin = std::max(
            0.0f,
            config.cellLumen.fusionTemporalCenterRepairClaimMargin);
        const bool allowSharedClaimPcaRelax =
            config.cellLumen.fusionTemporalCenterRepairAllowSharedClaimPcaRelax;
        const float sharedClaimMinShape =
            std::max(
                1.0f,
                config.cellLumen
                    .fusionTemporalCenterRepairSharedClaimPcaRelaxMinShape);
        int relaxed = 0;
        for (const auto &cell : frame.cells) {
            if (cell.isTrash()) {
                continue;
            }
            const std::string &cellName = cell.getName();
            const cv::Point3f oldPos(cell.getX(), cell.getY(), cell.getZ());
            const float preMaxR =
                std::max({cell.getARadius(), cell.getBRadius(), cell.getCRadius()});
            const float preMinR =
                std::max(
                    1e-3f,
                    std::min({cell.getARadius(),
                              cell.getBRadius(),
                              cell.getCRadius()}));
            const float cellShapeElongation = preMaxR / preMinR;
            const float normalMoveLimit =
                std::max(10.0f, std::min(14.0f, preMaxR * 0.55f));
            const PointWindowSupport oldSupport = pointWindowSupport(oldPos);

            int bestCenterIdx = -1;
            float bestScore = std::numeric_limits<float>::infinity();
            for (size_t centerIdx = 0; centerIdx < centers.size(); ++centerIdx) {
                const auto &center = centers[centerIdx];
                if (center.voxelCount < minVoxels || center.signal < minSignal) {
                    continue;
                }
                const float moveDistance =
                    static_cast<float>(cv::norm(oldPos - center.position));
                if (moveDistance <= normalMoveLimit || moveDistance > maxMove) {
                    continue;
                }

                const PointWindowSupport newSupport =
                    pointWindowSupport(center.position);
                if (newSupport.supported < minWindowSupport) {
                    continue;
                }

                float nearestOtherDistance =
                    std::numeric_limits<float>::infinity();
                for (const auto &other : frame.cells) {
                    if (other.isTrash() || other.getName() == cellName) {
                        continue;
                    }
                    const cv::Point3f otherPos(other.getX(),
                                               other.getY(),
                                               other.getZ());
                    nearestOtherDistance = std::min(
                        nearestOtherDistance,
                        static_cast<float>(
                            cv::norm(otherPos - center.position)));
                }
                const bool sharedClaimRelaxOk =
                    allowSharedClaimPcaRelax &&
                    cellShapeElongation >= sharedClaimMinShape &&
                    std::isfinite(nearestOtherDistance) &&
                    nearestOtherDistance <= maxMove &&
                    newSupport.supported >= minWindowSupport;
                if (std::isfinite(nearestOtherDistance) &&
                    nearestOtherDistance + claimMargin < moveDistance &&
                    !sharedClaimRelaxOk) {
                    continue;
                }

                if (oldSupport.supported > maxOldWindowSupport) {
                    const float oldAverage =
                        oldSupport.supported > 0
                            ? oldSupport.bestDistanceSum /
                                  static_cast<float>(oldSupport.supported)
                            : std::numeric_limits<float>::infinity();
                    const float newAverage =
                        newSupport.supported > 0
                            ? newSupport.bestDistanceSum /
                                  static_cast<float>(newSupport.supported)
                            : std::numeric_limits<float>::infinity();
                    const bool newHasMoreSupport =
                        newSupport.supported > oldSupport.supported;
                    const bool newIsClearlyCloser =
                        newSupport.supported >= oldSupport.supported &&
                        newAverage + minWindowDistanceGain < oldAverage;
                    if (!newHasMoreSupport && !newIsClearlyCloser) {
                        continue;
                    }
                }

                const float score =
                    moveDistance -
                    static_cast<float>(newSupport.supported) * 12.0f -
                    std::min(8.0f, std::max(0.0f, center.signal - minSignal) * 0.04f);
                if (score < bestScore) {
                    bestScore = score;
                    bestCenterIdx = static_cast<int>(centerIdx);
                }
            }

            if (bestCenterIdx < 0) {
                continue;
            }
            relaxPcaPositionGuardWide(cellName,
                                      "strong_lumen_future_motion");
            ++relaxed;
        }
        if (relaxed > 0) {
            std::cout << "[PCA Shape Position Guard Relax] frame "
                      << displayFrame
                      << " cells=" << relaxed
                      << " reason=strong_lumen_future_motion"
                      << std::endl;
        }
    };

    if (config.cellLumen.fusionTemporalCenterRepairEnabled) {
        applyStrongLumenPcaMotionRelax();
    }

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

    {
        ScopedStageTimer pcaTimer(displayFrame, "final_pca_shape");
        runPcaShapeFit();
    }

    if (config.cellLumen.fusionStaleContinuationPruneEnabled) {
        const auto &centers = getCellLumenLookaheadCandidates(frameIndex);
        if (!centers.empty()) {
            const float maxCurrentDistance = std::max(
                0.0f,
                config.cellLumen.fusionStaleContinuationPruneMaxCurrentDistance);
            const int maxFutureSupport = std::max(
                0,
                config.cellLumen.fusionStaleContinuationPruneMaxFutureSupport);
            const int minCellAge = std::max(
                0,
                config.cellLumen.fusionStaleContinuationPruneMinCellAgeFrames);
            const int windowSize = std::clamp(
                config.cellLumen.fusionSplitPriorWindowSize,
                2,
                5);

            std::set<std::string> protectedNames;
            for (const AcceptedLumenSplitRecord &record : acceptedLumenSplits) {
                protectedNames.insert(record.parentName + "0");
                protectedNames.insert(record.parentName + "1");
            }

            auto bestDistanceToCenters =
                [](const cv::Point3f &point,
                   const std::vector<CellLumenLookaheadCandidate> &candidateSet) {
                    float best = std::numeric_limits<float>::infinity();
                    for (const auto &center : candidateSet) {
                        best = std::min(
                            best,
                            static_cast<float>(cv::norm(center.position - point)));
                    }
                    return best;
                };

            auto futureSupportForPoint = [&](const cv::Point3f &point) {
                int support = 0;
                for (int offset = 1; offset < windowSize; ++offset) {
                    const int lookaheadFrameIndex = frameIndex + offset;
                    if (lookaheadFrameIndex < 0 ||
                        static_cast<size_t>(lookaheadFrameIndex) >= frames.size()) {
                        break;
                    }
                    const auto &futureCandidates =
                        getCellLumenLookaheadCandidates(lookaheadFrameIndex);
                    if (futureCandidates.empty()) {
                        continue;
                    }
                    if (bestDistanceToCenters(point, futureCandidates) <=
                        maxCurrentDistance) {
                        ++support;
                    }
                }
                return support;
            };
            int availableFutureOffsets = 0;
            for (int offset = 1; offset < windowSize; ++offset) {
                const int lookaheadFrameIndex = frameIndex + offset;
                if (lookaheadFrameIndex < 0 ||
                    static_cast<size_t>(lookaheadFrameIndex) >= frames.size()) {
                    break;
                }
                const auto &futureCandidates =
                    getCellLumenLookaheadCandidates(lookaheadFrameIndex);
                if (!futureCandidates.empty()) {
                    ++availableFutureOffsets;
                }
            }

            auto isRealLumenCandidateId = [](int candidateId) {
                return candidateId >= 0 && candidateId < 1000000000;
            };
            const auto splitPriorItForStalePrune =
                cellLumenSplitPriors.find(frameIndex);
            const bool staleReanchorOneRealEnabled =
                config.cellLumen
                    .fusionStaleContinuationPruneReanchorParentAnchorOneRealEnabled &&
                splitPriorItForStalePrune != cellLumenSplitPriors.end();
            const int staleReanchorMinWindowBoth = std::max(
                0,
                config.cellLumen
                    .fusionStaleContinuationPruneReanchorMinWindowBoth);
            const int staleReanchorMaxWindowMissing = std::max(
                0,
                config.cellLumen
                    .fusionStaleContinuationPruneReanchorMaxWindowMissing);
            const int staleReanchorMaxWindowParentPersists = std::max(
                0,
                config.cellLumen
                    .fusionStaleContinuationPruneReanchorMaxWindowParentPersists);
            const int staleReanchorMinVoxels = std::max(
                0,
                config.cellLumen.fusionStaleContinuationPruneReanchorMinVoxels);
            const float staleReanchorMinSignal = std::max(
                0.0f,
                config.cellLumen.fusionStaleContinuationPruneReanchorMinSignal);
            const float staleReanchorMinMove = std::max(
                0.0f,
                config.cellLumen.fusionStaleContinuationPruneReanchorMinMove);
            const bool staleLowQualityEnabled =
                config.cellLumen
                    .fusionStaleContinuationPruneLowQualityEnabled;
            const float staleLowQualityMaxBrightness = std::max(
                0.0f,
                config.cellLumen
                    .fusionStaleContinuationPruneLowQualityMaxBrightness);
            const float staleLowQualityMaxMajorRadius = std::max(
                0.0f,
                config.cellLumen
                    .fusionStaleContinuationPruneLowQualityMaxMajorRadius);
            const float staleLowQualityMaxMinorRadius = std::max(
                0.0f,
                config.cellLumen
                    .fusionStaleContinuationPruneLowQualityMaxMinorRadius);
            const int staleLowQualityMaxCellAge = std::max(
                0,
                config.cellLumen
                    .fusionStaleContinuationPruneLowQualityMaxCellAgeFrames);
            const bool staleLowQualityIgnoresProtection =
                config.cellLumen
                    .fusionStaleContinuationPruneLowQualityIgnoresProtection;
            const bool staleClaimedCenterEnabled =
                config.cellLumen
                    .fusionStaleContinuationPruneClaimedCenterEnabled;
            const float staleClaimedCenterMargin = std::max(
                0.0f,
                config.cellLumen
                    .fusionStaleContinuationPruneClaimedCenterMargin);
            const bool staleClaimedCenterRequireGuardOrNoFuture =
                config.cellLumen
                    .fusionStaleContinuationPruneClaimedCenterRequireGuardOrNoFuture;
            const bool staleGuardedNoFutureIgnoresCurrentCenter =
                config.cellLumen
                    .fusionStaleContinuationPruneGuardedNoFutureIgnoresCurrentCenterEnabled;
            const bool staleCrowdedBridgeEnabled =
                config.cellLumen
                    .fusionStaleContinuationPruneCrowdedBridgeEnabled;
            const float staleCrowdedBridgeMaxNeighborDistance = std::max(
                0.0f,
                config.cellLumen
                    .fusionStaleContinuationPruneCrowdedBridgeMaxNeighborDistance);
            const float staleCrowdedBridgeMinBrightnessAdvantage = std::max(
                0.0f,
                config.cellLumen
                    .fusionStaleContinuationPruneCrowdedBridgeMinBrightnessAdvantage);
            const float staleCrowdedBridgeMinGuardedBrightness = std::max(
                0.0f,
                config.cellLumen
                    .fusionStaleContinuationPruneCrowdedBridgeMinGuardedBrightness);
            const float staleCrowdedBridgeMaxGuardedBrightness = std::max(
                0.0f,
                config.cellLumen
                    .fusionStaleContinuationPruneCrowdedBridgeMaxGuardedBrightness);
            const bool staleProtectedOneRealSyntheticEnabled =
                config.cellLumen
                    .fusionStaleContinuationPruneProtectedOneRealSyntheticEnabled;
            const float staleProtectedOneRealSyntheticMaxBrightness =
                std::max(
                    0.0f,
                    config.cellLumen
                        .fusionStaleContinuationPruneProtectedOneRealSyntheticMaxBrightness);
            const float staleProtectedOneRealSyntheticMaxMajorRadius =
                std::max(
                    0.0f,
                    config.cellLumen
                        .fusionStaleContinuationPruneProtectedOneRealSyntheticMaxMajorRadius);
            const float staleProtectedOneRealSyntheticMaxMinorRadius =
                std::max(
                    0.0f,
                    config.cellLumen
                        .fusionStaleContinuationPruneProtectedOneRealSyntheticMaxMinorRadius);
            const float staleProtectedOneRealSyntheticMaxSiblingDistance =
                std::max(
                    0.0f,
                    config.cellLumen
                        .fusionStaleContinuationPruneProtectedOneRealSyntheticMaxSiblingDistance);
            const float staleProtectedOneRealSyntheticMaxIndependentCenterDistance =
                std::max(
                    0.0f,
                    config.cellLumen
                        .fusionStaleContinuationPruneProtectedOneRealSyntheticMaxIndependentCenterDistance);

            auto findCurrentLumenCenterById =
                [&](int candidateId) -> const CellLumenLookaheadCandidate * {
                for (const auto &center : centers) {
                    if (center.candidateId == candidateId) {
                        return &center;
                    }
                }
                return nullptr;
            };
            auto findNearestCurrentLumenCenter =
                [&](const cv::Point3f &point,
                    const std::set<int> &excludedCandidateIds,
                    float &bestDistance)
                    -> const CellLumenLookaheadCandidate * {
                const CellLumenLookaheadCandidate *bestCenter = nullptr;
                bestDistance = std::numeric_limits<float>::infinity();
                for (const auto &center : centers) {
                    if (excludedCandidateIds.count(center.candidateId) > 0) {
                        continue;
                    }
                    const float distance = static_cast<float>(
                        cv::norm(center.position - point));
                    if (distance < bestDistance) {
                        bestDistance = distance;
                        bestCenter = &center;
                    }
                }
                return bestCenter;
            };
            auto closestOtherCellDistanceToPoint =
                [&](const cv::Point3f &point,
                    const std::string &cellName,
                    std::string &closestName) {
                float bestDistance = std::numeric_limits<float>::infinity();
                closestName.clear();
                for (const auto &other : frame.cells) {
                    if (other.isTrash() || other.getName() == cellName) {
                        continue;
                    }
                    const cv::Point3f otherPos(
                        other.getX(), other.getY(), other.getZ());
                    const float distance = static_cast<float>(
                        cv::norm(otherPos - point));
                    if (distance < bestDistance) {
                        bestDistance = distance;
                        closestName = other.getName();
                    }
                }
                return bestDistance;
            };
            auto findCellIndexByNameForStale =
                [&](const std::string &name) -> int {
                for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
                    if (frame.cells[ci].getName() == name) {
                        return static_cast<int>(ci);
                    }
                }
                return -1;
            };

            struct StaleReanchorRecord {
                cv::Point3f position{0.0f, 0.0f, 0.0f};
                int candidateId = -1;
                int voxelCount = 0;
                float signal = 0.0f;
                float moveDistance = 0.0f;
                int windowBoth = 0;
                int windowMissing = 0;
                int windowParentPersists = 0;
            };
            auto parentAnchorOneRealStaleReanchor =
                [&](const std::string &cellName,
                    const cv::Point3f &pos,
                    StaleReanchorRecord &record) {
                if (!staleReanchorOneRealEnabled) {
                    return false;
                }
                const auto priorIt =
                    splitPriorItForStalePrune->second.find(cellName);
                if (priorIt == splitPriorItForStalePrune->second.end()) {
                    return false;
                }
                const BridgeSplitProposal &proposal = priorIt->second;
                const bool realA =
                    isRealLumenCandidateId(proposal.candidateIdA);
                const bool realB =
                    isRealLumenCandidateId(proposal.candidateIdB);
                if (!proposal.parentAnchored || realA == realB) {
                    return false;
                }
                if (proposal.windowBothDaughtersSupported <
                        staleReanchorMinWindowBoth ||
                    proposal.windowMissingDaughterCount >
                        staleReanchorMaxWindowMissing ||
                    proposal.windowParentPersists >
                        staleReanchorMaxWindowParentPersists) {
                    return false;
                }
                const int realCandidateId =
                    realA ? proposal.candidateIdA : proposal.candidateIdB;
                const auto *center =
                    findCurrentLumenCenterById(realCandidateId);
                if (center == nullptr ||
                    center->voxelCount < staleReanchorMinVoxels ||
                    center->signal < staleReanchorMinSignal) {
                    return false;
                }
                const float moveDistance =
                    static_cast<float>(cv::norm(center->position - pos));
                if (moveDistance < staleReanchorMinMove) {
                    return false;
                }
                float nearestOtherDistance =
                    std::numeric_limits<float>::infinity();
                for (const auto &other : frame.cells) {
                    if (other.isTrash() || other.getName() == cellName) {
                        continue;
                    }
                    const cv::Point3f otherPos(
                        other.getX(), other.getY(), other.getZ());
                    nearestOtherDistance = std::min(
                        nearestOtherDistance,
                        static_cast<float>(
                            cv::norm(otherPos - center->position)));
                }
                if (std::isfinite(nearestOtherDistance) &&
                    nearestOtherDistance + 4.0f < moveDistance) {
                    return false;
                }

                record.position = center->position;
                record.candidateId = center->candidateId;
                record.voxelCount = center->voxelCount;
                record.signal = center->signal;
                record.moveDistance = moveDistance;
                record.windowBoth = proposal.windowBothDaughtersSupported;
                record.windowMissing = proposal.windowMissingDaughterCount;
                record.windowParentPersists = proposal.windowParentPersists;
                return true;
            };
            auto protectedOneRealSyntheticPrune =
                [&](const std::string &cellName,
                    const Ellipsoid &cell,
                    const float cellMajorRadius,
                    const float cellMinorRadius,
                    std::string &parentName,
                    std::string &realDaughterName,
                    float &siblingDistance,
                    float &nearestIndependentCenterDistance) {
                if (!staleProtectedOneRealSyntheticEnabled ||
                    cell.getBrightness() >
                        staleProtectedOneRealSyntheticMaxBrightness ||
                    cellMajorRadius >
                        staleProtectedOneRealSyntheticMaxMajorRadius ||
                    cellMinorRadius >
                        staleProtectedOneRealSyntheticMaxMinorRadius) {
                    return false;
                }
                const cv::Point3f cellPos(cell.getX(), cell.getY(), cell.getZ());
                for (const AcceptedLumenSplitRecord &record :
                     acceptedLumenSplits) {
                    const BridgeSplitProposal &proposal = record.proposal;
                    const bool realA =
                        isRealLumenCandidateId(proposal.candidateIdA);
                    const bool realB =
                        isRealLumenCandidateId(proposal.candidateIdB);
                    if (!proposal.parentAnchored || realA == realB) {
                        continue;
                    }
                    const std::string syntheticName =
                        realA ? record.parentName + "1"
                              : record.parentName + "0";
                    if (cellName != syntheticName) {
                        continue;
                    }
                    realDaughterName =
                        realA ? record.parentName + "0"
                              : record.parentName + "1";
                    const int realDaughterIdx =
                        findCellIndexByNameForStale(realDaughterName);
                    if (realDaughterIdx < 0) {
                        continue;
                    }
                    const Ellipsoid &realDaughter =
                        frame.cells[static_cast<size_t>(realDaughterIdx)];
                    const cv::Point3f realDaughterPos(
                        realDaughter.getX(),
                        realDaughter.getY(),
                        realDaughter.getZ());
                    siblingDistance = static_cast<float>(
                        cv::norm(realDaughterPos - cellPos));
                    if (siblingDistance >
                        staleProtectedOneRealSyntheticMaxSiblingDistance) {
                        continue;
                    }

                    std::set<int> excludedIds;
                    if (realA) {
                        excludedIds.insert(proposal.candidateIdA);
                    }
                    if (realB) {
                        excludedIds.insert(proposal.candidateIdB);
                    }
                    findNearestCurrentLumenCenter(
                        cellPos,
                        excludedIds,
                        nearestIndependentCenterDistance);
                    if (nearestIndependentCenterDistance <=
                        staleProtectedOneRealSyntheticMaxIndependentCenterDistance) {
                        continue;
                    }
                    parentName = record.parentName;
                    return true;
                }
                return false;
            };
            auto isProtectedOneRealSyntheticName =
                [&](const std::string &cellName) {
                for (const AcceptedLumenSplitRecord &record :
                     acceptedLumenSplits) {
                    const BridgeSplitProposal &proposal = record.proposal;
                    const bool realA =
                        isRealLumenCandidateId(proposal.candidateIdA);
                    const bool realB =
                        isRealLumenCandidateId(proposal.candidateIdB);
                    if (!proposal.parentAnchored || realA == realB) {
                        continue;
                    }
                    const std::string syntheticName =
                        realA ? record.parentName + "1"
                              : record.parentName + "0";
                    if (cellName == syntheticName) {
                        return true;
                    }
                }
                return false;
            };
            auto isAcceptedSplitPcaLocked =
                [&](const std::string &cellName) {
                const auto reasonIt = pcaPositionLockReasons.find(cellName);
                return reasonIt != pcaPositionLockReasons.end() &&
                       reasonIt->second.find("accepted_cell_lumen_split") !=
                           std::string::npos;
            };
            auto brighterNeighborCount =
                [&](const std::string &cellName,
                    const cv::Point3f &cellPos,
                    float cellBrightness) {
                int count = 0;
                for (const auto &other : frame.cells) {
                    if (other.isTrash() || other.getName() == cellName) {
                        continue;
                    }
                    if (other.getBrightness() <
                        cellBrightness +
                            staleCrowdedBridgeMinBrightnessAdvantage) {
                        continue;
                    }
                    const cv::Point3f otherPos(
                        other.getX(), other.getY(), other.getZ());
                    if (cv::norm(otherPos - cellPos) <=
                        staleCrowdedBridgeMaxNeighborDistance) {
                        ++count;
                    }
                }
                return count;
            };

            std::set<std::string> staleRemoveNames;
            std::map<std::string, StaleReanchorRecord> staleReanchorRecords;
            for (const auto &cell : frame.cells) {
                if (cell.isTrash()) {
                    continue;
                }
                const std::string &cellName = cell.getName();
                int cellAge = std::numeric_limits<int>::max() / 4;
                if (const auto firstSeenIt = cellFirstSeenFrame.find(cellName);
                    firstSeenIt != cellFirstSeenFrame.end()) {
                    cellAge = std::max(0, displayFrame - firstSeenIt->second);
                }
                const bool isProtected = protectedNames.count(cellName) > 0;

                const float cellMajorRadius = std::max(
                    {cell.getARadius(), cell.getBRadius(), cell.getCRadius()});
                const float cellMinorRadius = std::min(
                    {cell.getARadius(), cell.getBRadius(), cell.getCRadius()});
                const cv::Point3f pos(cell.getX(), cell.getY(), cell.getZ());
                const bool guardReverted =
                    pcaPositionGuardRevertedCells.count(cellName) > 0;
                std::string syntheticParentName;
                std::string syntheticRealDaughterName;
                float syntheticSiblingDistance =
                    std::numeric_limits<float>::infinity();
                float syntheticIndependentCenterDistance =
                    std::numeric_limits<float>::infinity();
                if (protectedOneRealSyntheticPrune(
                        cellName,
                        cell,
                        cellMajorRadius,
                        cellMinorRadius,
                        syntheticParentName,
                        syntheticRealDaughterName,
                        syntheticSiblingDistance,
                        syntheticIndependentCenterDistance)) {
                    staleRemoveNames.insert(cellName);
                    std::cout << "[CellLumen Stale Continuation Protected OneReal Synthetic Prune Candidate] frame "
                              << displayFrame
                              << " cell=" << cellName
                              << " parent=" << syntheticParentName
                              << " realDaughter=" << syntheticRealDaughterName
                              << " brightness=" << cell.getBrightness()
                              << " majorRadius=" << cellMajorRadius
                              << " minorRadius=" << cellMinorRadius
                              << " siblingDistance="
                              << syntheticSiblingDistance
                              << " nearestIndependentCenter="
                              << syntheticIndependentCenterDistance
                              << std::endl;
                    continue;
                }
                const bool crowdedBridgeWeakSynthetic =
                    (isProtectedOneRealSyntheticName(cellName) ||
                     isProtected ||
                     isAcceptedSplitPcaLocked(cellName)) &&
                    cell.getBrightness() <=
                        staleProtectedOneRealSyntheticMaxBrightness &&
                    cellMajorRadius <=
                        staleProtectedOneRealSyntheticMaxMajorRadius &&
                    cellMinorRadius <=
                        staleProtectedOneRealSyntheticMaxMinorRadius;
                const bool crowdedBridgeGuarded =
                    !isProtected &&
                    guardReverted &&
                    cell.getBrightness() >=
                        staleCrowdedBridgeMinGuardedBrightness &&
                    cell.getBrightness() <=
                        staleCrowdedBridgeMaxGuardedBrightness;
                if (staleCrowdedBridgeEnabled &&
                    (crowdedBridgeWeakSynthetic || crowdedBridgeGuarded)) {
                    const int brightNeighbors = brighterNeighborCount(
                        cellName,
                        pos,
                        cell.getBrightness());
                    if (brightNeighbors >= 2) {
                        staleRemoveNames.insert(cellName);
                        std::cout << "[CellLumen Stale Continuation CrowdedBridge Prune Candidate] frame "
                                  << displayFrame
                                  << " cell=" << cellName
                                  << " brightness=" << cell.getBrightness()
                                  << " majorRadius=" << cellMajorRadius
                                  << " minorRadius=" << cellMinorRadius
                                  << " brightNeighbors="
                                  << brightNeighbors
                                  << " maxNeighborDistance="
                                  << staleCrowdedBridgeMaxNeighborDistance
                                  << " brightnessAdvantage="
                                  << staleCrowdedBridgeMinBrightnessAdvantage
                                  << " protectedOneRealSynthetic="
                                  << (crowdedBridgeWeakSynthetic ? 1 : 0)
                                  << " acceptedSplitPcaLocked="
                                  << (isAcceptedSplitPcaLocked(cellName) ? 1
                                                                        : 0)
                                  << " pcaGuardReverted="
                                  << (guardReverted ? 1 : 0)
                                  << std::endl;
                        continue;
                    }
                }
                if (staleLowQualityEnabled &&
                    (!isProtected || staleLowQualityIgnoresProtection) &&
                    cellAge <= staleLowQualityMaxCellAge &&
                    cell.getBrightness() <= staleLowQualityMaxBrightness &&
                    cellMajorRadius <= staleLowQualityMaxMajorRadius &&
                    cellMinorRadius <= staleLowQualityMaxMinorRadius) {
                    staleRemoveNames.insert(cellName);
                    std::cout << "[CellLumen Stale Continuation LowQuality Prune Candidate] frame "
                              << displayFrame
                              << " cell=" << cellName
                              << " brightness=" << cell.getBrightness()
                              << " majorRadius=" << cellMajorRadius
                              << " minorRadius=" << cellMinorRadius
                              << " age=" << cellAge
                              << " protected=" << isProtected
                              << std::endl;
                    continue;
                }

                if (isProtected) {
                    continue;
                }

                if (cellAge < minCellAge) {
                    continue;
                }

                const float nearestAnyCenter = bestDistanceToCenters(pos, centers);
                float ownCenterDistance = std::numeric_limits<float>::infinity();
                if (lumenCenterCandidatesForFrame != nullptr) {
                    const auto ownIt =
                        lumenCenterCandidatesForFrame->find(cellName);
                    if (ownIt != lumenCenterCandidatesForFrame->end()) {
                        ownCenterDistance = static_cast<float>(
                            cv::norm(ownIt->second.position - pos));
                    }
                }
                const int futureSupport = futureSupportForPoint(pos);
                if (staleGuardedNoFutureIgnoresCurrentCenter &&
                    guardReverted &&
                    availableFutureOffsets > 0 &&
                    futureSupport <= maxFutureSupport) {
                    staleRemoveNames.insert(cellName);
                    std::cout << "[CellLumen Stale Continuation GuardedNoFuture Prune Candidate] frame "
                              << displayFrame
                              << " cell=" << cellName
                              << " ownCenterDistance=" << ownCenterDistance
                              << " nearestAnyCenter=" << nearestAnyCenter
                              << " futureSupport=" << futureSupport
                              << " availableFutureOffsets="
                              << availableFutureOffsets
                              << " age=" << cellAge
                              << " pcaGuardReverted=1"
                              << std::endl;
                    continue;
                }
                if (ownCenterDistance <= maxCurrentDistance) {
                    continue;
                }

                StaleReanchorRecord reanchorRecord;
                if (parentAnchorOneRealStaleReanchor(
                        cellName, pos, reanchorRecord)) {
                    staleReanchorRecords[cellName] = reanchorRecord;
                    std::cout << "[CellLumen Stale Continuation Reanchor Candidate] frame "
                              << displayFrame
                              << " cell=" << cellName
                              << " candidateId=" << reanchorRecord.candidateId
                              << " move=" << reanchorRecord.moveDistance
                              << " voxels=" << reanchorRecord.voxelCount
                              << " signal=" << reanchorRecord.signal
                              << " windowBoth=" << reanchorRecord.windowBoth
                              << " windowMissing=" << reanchorRecord.windowMissing
                              << " windowParentPersists="
                              << reanchorRecord.windowParentPersists
                              << " ownCenterDistance=" << ownCenterDistance
                              << " nearestAnyCenter=" << nearestAnyCenter
                              << " futureSupport=" << futureSupport
                              << " age=" << cellAge
                              << std::endl;
                    continue;
                }
                if (staleClaimedCenterEnabled &&
                    ownCenterDistance > maxCurrentDistance &&
                    nearestAnyCenter <= maxCurrentDistance) {
                    float claimedCenterDistance =
                        std::numeric_limits<float>::infinity();
                    const auto *claimedCenter = findNearestCurrentLumenCenter(
                        pos,
                        std::set<int>{},
                        claimedCenterDistance);
                    std::string closestCellName;
                    const float closestOtherDistance =
                        claimedCenter != nullptr
                            ? closestOtherCellDistanceToPoint(
                                  claimedCenter->position,
                                  cellName,
                                  closestCellName)
                            : std::numeric_limits<float>::infinity();
                    const bool claimedByOther =
                        claimedCenter != nullptr &&
                        std::isfinite(closestOtherDistance) &&
                        closestOtherDistance + staleClaimedCenterMargin <
                            claimedCenterDistance;
                    const bool supportAllowsPrune =
                        !staleClaimedCenterRequireGuardOrNoFuture ||
                        guardReverted ||
                        futureSupport <= maxFutureSupport;
                    if (claimedByOther && supportAllowsPrune) {
                        staleRemoveNames.insert(cellName);
                        std::cout << "[CellLumen Stale Continuation ClaimedCenter Prune Candidate] frame "
                                  << displayFrame
                                  << " cell=" << cellName
                                  << " candidateId="
                                  << claimedCenter->candidateId
                                  << " claimedCenterDistance="
                                  << claimedCenterDistance
                                  << " closestOther=" << closestCellName
                                  << " closestOtherDistance="
                                  << closestOtherDistance
                                  << " margin="
                                  << staleClaimedCenterMargin
                                  << " ownCenterDistance="
                                  << ownCenterDistance
                                  << " futureSupport="
                                  << futureSupport
                                  << " pcaGuardReverted="
                                  << (guardReverted ? 1 : 0)
                                  << std::endl;
                        continue;
                    }
                }
                if (nearestAnyCenter <= maxCurrentDistance) {
                    continue;
                }
                if (futureSupport > maxFutureSupport && !guardReverted) {
                    continue;
                }

                staleRemoveNames.insert(cellName);
                std::cout << "[CellLumen Stale Continuation Prune Candidate] frame "
                          << displayFrame
                          << " cell=" << cellName
                          << " ownCenterDistance=" << ownCenterDistance
                          << " nearestAnyCenter=" << nearestAnyCenter
                          << " futureSupport=" << futureSupport
                          << " age=" << cellAge
                          << " pcaGuardReverted=" << (guardReverted ? 1 : 0)
                          << " maxCurrentDistance=" << maxCurrentDistance
                          << std::endl;
            }

            int reanchored = 0;
            for (const auto &entry : staleReanchorRecords) {
                const std::string &cellName = entry.first;
                const StaleReanchorRecord &record = entry.second;
                for (auto &cell : frame.cells) {
                    if (cell.getName() != cellName) {
                        continue;
                    }
                    const cv::Point3f oldPos(
                        cell.getX(), cell.getY(), cell.getZ());
                    cell.setPosition(
                        record.position.x,
                        record.position.y,
                        record.position.z);
                    ++reanchored;
                    std::cout << "[CellLumen Stale Continuation Reanchor] frame "
                              << displayFrame
                              << " cell=" << cellName
                              << " candidateId=" << record.candidateId
                              << " from=(" << oldPos.x << "," << oldPos.y
                              << "," << oldPos.z << ")"
                              << " to=(" << record.position.x << ","
                              << record.position.y << "," << record.position.z
                              << ")"
                              << " move=" << record.moveDistance
                              << " voxels=" << record.voxelCount
                              << " signal=" << record.signal
                              << std::endl;
                    break;
                }
            }

            int removed = 0;
            for (const std::string &removeName : staleRemoveNames) {
                for (auto it = frame.cells.begin(); it != frame.cells.end();
                     ++it) {
                    if (it->getName() != removeName) {
                        continue;
                    }
                    frame.cells.erase(it);
                    previousSnapshots.erase(removeName);
                    cellShapeReference.erase(removeName);
                    cellShapeBirth.erase(removeName);
                    cellFirstSeenFrame.erase(removeName);
                    ++removed;
                    std::cout << "[CellLumen Stale Continuation Prune] frame "
                              << displayFrame
                              << " removed=" << removeName
                              << std::endl;
                    break;
                }
            }

            if (reanchored > 0 || removed > 0) {
                frame.regenerateSynthFrame();
                if (voronoiMapNeeded) {
                    frame.rebuildVoronoiMap();
                }
                std::cout << "[CellLumen Stale Continuation Prune Summary] frame "
                          << displayFrame
                          << " reanchored=" << reanchored
                          << " removed=" << removed
                          << " cells=" << frame.cells.size()
                          << std::endl;
            }
        }
    }

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
                cellFirstSeenFrame.erase(name);
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
    writeDensityBrightnessMetrics(frameIndex, "post_optimize");

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

    // Long OpenLab runs only need the current frame and the short future window
    // for Cell Lumen fusion. Older per-frame candidate caches can be dropped
    // after export/checkpoint so a one-shot 0-249 run does not retain stale
    // graph state for hundreds of frames.
    const int keepWindow = std::max(
        2,
        std::clamp(config.cellLumen.fusionSplitPriorWindowSize, 2, 5) + 1);
    const int keepFromFrame = frameIndex - keepWindow;
    if (keepFromFrame <= 0) {
        return;
    }

    auto pruneFrameCache = [&](auto &cache) {
        int erased = 0;
        for (auto it = cache.begin(); it != cache.end();) {
            if (it->first < keepFromFrame) {
                it = cache.erase(it);
                ++erased;
            } else {
                ++it;
            }
        }
        return erased;
    };

    const int erasedSplitPriors = pruneFrameCache(cellLumenSplitPriors);
    const int erasedReanchors = pruneFrameCache(cellLumenCenterReanchorCandidateIds);
    const int erasedBadParents = pruneFrameCache(cellLumenSplitPriorRejectedBadParents);
    const int erasedCollapsedParents = pruneFrameCache(cellLumenCollapsedCenterParents);
    const int erasedCenterCandidates = pruneFrameCache(cellLumenCenterCandidates);
    const int erasedLookahead = pruneFrameCache(cellLumenLookaheadCandidates);
    const int totalErased = erasedSplitPriors + erasedReanchors + erasedBadParents +
                            erasedCollapsedParents + erasedCenterCandidates +
                            erasedLookahead;
    if (totalErased > 0) {
        std::cout << "[Runtime CacheRelease] frame=" << (firstFrame + frameIndex)
                  << " keep_from_frame=" << (firstFrame + keepFromFrame)
                  << " split_priors=" << erasedSplitPriors
                  << " reanchors=" << erasedReanchors
                  << " bad_parents=" << erasedBadParents
                  << " collapsed_parents=" << erasedCollapsedParents
                  << " center_candidates=" << erasedCenterCandidates
                  << " lookahead=" << erasedLookahead
                  << std::endl;
    }
}

void CellUniverse::saveImages(int frameIndex, const std::string &stage)
{
    if (!config.simulation.export_frame_png &&
        !config.simulation.export_frame_tiff)
    {
        return;
    }

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
        std::filesystem::create_directories(exportRoot);

        writeNapariFriendlyTiffStack(exportRoot + "/" + std::to_string(displayFrame) + "_real.tif",
                                     realImages);
        writeNapariFriendlyTiffStack(exportRoot + "/" + std::to_string(displayFrame) + "_synth.tif",
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

namespace {

bool parseCheckpointPayloadWithNumericTail(const std::string &line,
                                           const std::string &tag,
                                           size_t numericCount,
                                           std::string &name,
                                           std::vector<double> &values)
{
    values.clear();
    name.clear();

    std::string rest = line.substr(tag.size());
    const size_t first = rest.find_first_not_of(" \t");
    if (first == std::string::npos) {
        return false;
    }
    rest = rest.substr(first);

    if (!rest.empty() && rest.front() == '"') {
        std::istringstream quoted(rest);
        if (!(quoted >> std::quoted(name))) {
            return false;
        }
        values.reserve(numericCount);
        for (size_t i = 0; i < numericCount; ++i) {
            double v = 0.0;
            if (!(quoted >> v)) {
                return false;
            }
            values.push_back(v);
        }
        return true;
    }

    std::istringstream raw(rest);
    std::vector<std::string> tokens;
    for (std::string token; raw >> token;) {
        tokens.push_back(token);
    }
    if (tokens.size() <= numericCount) {
        return false;
    }

    const size_t nameTokenCount = tokens.size() - numericCount;
    std::ostringstream nameStream;
    for (size_t i = 0; i < nameTokenCount; ++i) {
        if (i > 0) {
            nameStream << ' ';
        }
        nameStream << tokens[i];
    }
    name = nameStream.str();

    values.reserve(numericCount);
    for (size_t i = nameTokenCount; i < tokens.size(); ++i) {
        try {
            values.push_back(std::stod(tokens[i]));
        } catch (const std::exception &) {
            return false;
        }
    }
    return values.size() == numericCount;
}

} // namespace

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
    {
        std::ostringstream rngState;
        rngState << cellUniverseRandomGenerator();
        out << "rngState " << rngState.str() << '\n';
    }
    out << "edgeBrightnessAlignmentTargetInitialized "
        << (edgeBrightnessAlignmentTargetInitialized ? 1 : 0) << '\n';
    out << "edgeBrightnessAlignmentTarget "
        << edgeBrightnessAlignmentTarget << '\n';

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
        out << "cell " << std::quoted(p.name)
            << " " << p.x << " " << p.y << " " << p.z
            << " " << p.aRadius << " " << p.bRadius << " " << p.cRadius
            << " " << p.theta_x << " " << p.theta_y << " " << p.theta_z
            << " " << p.brightness
            << " " << (p.isTrash ? 1 : 0) << '\n';
    }

    // Previous-frame snapshots (per-cell PCA fit state for split detection).
    for (const auto &kv : previousSnapshots) {
        const auto &s = kv.second;
        out << "snap " << std::quoted(kv.first)
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
        out << "ref " << std::quoted(kv.first)
            << " " << kv.second[0] << " " << kv.second[1] << " " << kv.second[2] << '\n';
    }

    // Shape birth (frozen birth radii for mask + birth cap).
    for (const auto &kv : cellShapeBirth) {
        out << "birth " << std::quoted(kv.first)
            << " " << kv.second[0] << " " << kv.second[1] << " " << kv.second[2] << '\n';
    }

    // First-seen frame metadata used by rescue/fallback anti-cascade gates.
    // Use quoted names so cells with spaces round-trip correctly without
    // changing the legacy checkpoint tags.
    for (const auto &kv : cellFirstSeenFrame) {
        out << "firstSeen " << std::quoted(kv.first)
            << " " << kv.second << '\n';
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
    cellFirstSeenFrame.clear();
    cellLumenSplitPriors.clear();
    cellLumenCenterReanchorCandidateIds.clear();
    cellLumenSplitPriorRejectedBadParents.clear();
    cellLumenCollapsedCenterParents.clear();
    cellLumenCenterCandidates.clear();
    cellLumenLookaheadCandidates.clear();
    resumePreviousFrameSummaryValid = false;
    resumePreviousAdaptiveBackground = 0.0f;
    resumePreviousMeanBrightness = 0.0f;
    resumePreviousCellCount = 0;
    edgeBrightnessAlignmentTargetInitialized = false;
    edgeBrightnessAlignmentTarget = 0.0f;

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
    int loadedFirstSeen = 0;
    bool loadedRngState = false;
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
                return false;
            }
        } else if (tag == "z_slices") {
            int v; ss >> v;
            config.simulation.z_slices = v;
        } else if (tag == "maxZ") {
            float v; ss >> v;
            Ellipsoid::cellConfig.maxZ = v;
        } else if (tag == "rngState") {
            const size_t payloadStart = line.find_first_not_of(" \t", tag.size());
            if (payloadStart == std::string::npos) {
                std::cerr << "[Checkpoint] malformed rngState line\n";
                continue;
            }
            std::istringstream rngIn(line.substr(payloadStart));
            if (rngIn >> cellUniverseRandomGenerator()) {
                loadedRngState = true;
            } else {
                std::cerr << "[Checkpoint] malformed rngState payload\n";
            }
        } else if (tag == "edgeBrightnessAlignmentTargetInitialized") {
            int v; ss >> v;
            edgeBrightnessAlignmentTargetInitialized = (v != 0);
        } else if (tag == "edgeBrightnessAlignmentTarget") {
            float v; ss >> v;
            edgeBrightnessAlignmentTarget = v;
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
            std::vector<double> values;
            if (!parseCheckpointPayloadWithNumericTail(line, tag, 11, p.name, values)) {
                std::cerr << "[Checkpoint] malformed cell line: " << line << '\n';
                continue;
            }
            p.x = static_cast<float>(values[0]);
            p.y = static_cast<float>(values[1]);
            p.z = static_cast<float>(values[2]);
            p.aRadius = static_cast<float>(values[3]);
            p.bRadius = static_cast<float>(values[4]);
            p.cRadius = static_cast<float>(values[5]);
            p.theta_x = static_cast<float>(values[6]);
            p.theta_y = static_cast<float>(values[7]);
            p.theta_z = static_cast<float>(values[8]);
            p.brightness = static_cast<float>(values[9]);
            p.isTrash = (static_cast<int>(values[10]) != 0);
            frames[cellsFrameIdx].cells.emplace_back(p);
            ++loadedCells;
        } else if (tag == "snap") {
            PreviousFrameSnapshot s;
            std::string name;
            std::vector<double> values;
            if (!parseCheckpointPayloadWithNumericTail(line, tag, 16, name, values)) {
                std::cerr << "[Checkpoint] malformed snap line: " << line << '\n';
                continue;
            }
            const int validInt = static_cast<int>(values[0]);
            s.valid = (validInt != 0);
            s.shapeElongation = static_cast<float>(values[1]);
            s.splitAxisDir.x = static_cast<float>(values[2]);
            s.splitAxisDir.y = static_cast<float>(values[3]);
            s.splitAxisDir.z = static_cast<float>(values[4]);
            s.splitAxisLength = static_cast<float>(values[5]);
            s.position.x = static_cast<float>(values[6]);
            s.position.y = static_cast<float>(values[7]);
            s.position.z = static_cast<float>(values[8]);
            s.aRadius = static_cast<float>(values[9]);
            s.bRadius = static_cast<float>(values[10]);
            s.cRadius = static_cast<float>(values[11]);
            s.thetaX = static_cast<float>(values[12]);
            s.thetaY = static_cast<float>(values[13]);
            s.thetaZ = static_cast<float>(values[14]);
            s.brightness = static_cast<float>(values[15]);
            previousSnapshots[name] = s;
            ++loadedSnaps;
        } else if (tag == "ref") {
            std::string name;
            std::vector<double> values;
            if (!parseCheckpointPayloadWithNumericTail(line, tag, 3, name, values)) {
                std::cerr << "[Checkpoint] malformed ref line: " << line << '\n';
                continue;
            }
            std::array<float, 3> r = {
                static_cast<float>(values[0]),
                static_cast<float>(values[1]),
                static_cast<float>(values[2])};
            cellShapeReference[name] = r;
            ++loadedRefs;
        } else if (tag == "birth") {
            std::string name;
            std::vector<double> values;
            if (!parseCheckpointPayloadWithNumericTail(line, tag, 3, name, values)) {
                std::cerr << "[Checkpoint] malformed birth line: " << line << '\n';
                continue;
            }
            std::array<float, 3> r = {
                static_cast<float>(values[0]),
                static_cast<float>(values[1]),
                static_cast<float>(values[2])};
            cellShapeBirth[name] = r;
            ++loadedBirths;
        } else if (tag == "firstSeen") {
            std::string name;
            int seenFrame = targetAbsoluteFrame - 100000;
            if (!(ss >> std::quoted(name) >> seenFrame)) {
                std::vector<double> values;
                if (!parseCheckpointPayloadWithNumericTail(line, tag, 1, name, values)) {
                    std::cerr << "[Checkpoint] malformed firstSeen line: " << line << '\n';
                    continue;
                }
                seenFrame = static_cast<int>(values[0]);
            }
            if (!name.empty()) {
                cellFirstSeenFrame[name] = seenFrame;
                ++loadedFirstSeen;
            }
        }
    }
    in.close();
    resumePreviousCellCount = frames[cellsFrameIdx].cells.size();
    const int oldLoadedFirstSeenFrame = targetAbsoluteFrame - 100000;
    for (const auto &cell : frames[cellsFrameIdx].cells) {
        cellFirstSeenFrame.try_emplace(cell.getName(), oldLoadedFirstSeenFrame);
    }

    std::cout << "[Checkpoint] loaded frame " << checkpointFrameIndex
              << " into local frame " << targetFrameIndex
              << " (absolute " << targetAbsoluteFrame << ")"
              << " cells=" << loadedCells
              << " snaps=" << loadedSnaps
              << " refs=" << loadedRefs
              << " births=" << loadedBirths
              << " firstSeen=" << loadedFirstSeen
              << " rngState=" << (loadedRngState ? 1 : 0)
              << " edgeTargetInitialized="
              << (edgeBrightnessAlignmentTargetInitialized ? 1 : 0)
              << std::endl;
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
