#include "../includes/ImageHandler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
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

void applySigmoid(cv::Mat &image, float k, float center)
{
    image.forEach<float>([k, center](float &pixel, const int *) {
        pixel = 1.0f / (1.0f + std::exp(-k * (pixel - center)));
    });
}

float computePercentileFromStack(const std::vector<cv::Mat> &stack, float percentile)
{
    std::vector<float> values;
    size_t totalValues = 0;
    for (const auto &slice : stack) {
        if (slice.type() != CV_32F || slice.empty()) {
            continue;
        }
        totalValues += static_cast<size_t>(slice.total());
    }

    if (totalValues == 0) {
        return 0.0f;
    }

    values.reserve(totalValues);
    for (const auto &slice : stack) {
        if (slice.type() != CV_32F || slice.empty()) {
            continue;
        }
        for (int y = 0; y < slice.rows; ++y) {
            const float *row = slice.ptr<float>(y);
            values.insert(values.end(), row, row + slice.cols);
        }
    }

    const float clampedPercentile = std::clamp(percentile, 0.0f, 1.0f);
    const size_t index = static_cast<size_t>(
        std::floor(clampedPercentile * static_cast<float>(values.size() - 1)));

    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(index),
                     values.end());
    return values[index];
}

float computePercentileFromStackRoi(const std::vector<cv::Mat> &stack, const cv::Rect &roi, float percentile)
{
    std::vector<float> values;
    size_t totalValues = 0;
    for (const auto &slice : stack) {
        if (slice.type() != CV_32F || slice.empty()) {
            continue;
        }
        totalValues += static_cast<size_t>(roi.area());
    }

    if (totalValues == 0 || roi.width <= 0 || roi.height <= 0) {
        return 0.0f;
    }

    values.reserve(totalValues);
    for (const auto &slice : stack) {
        if (slice.type() != CV_32F || slice.empty()) {
            continue;
        }
        const cv::Mat roiView = slice(roi);
        for (int y = 0; y < roiView.rows; ++y) {
            const float *row = roiView.ptr<float>(y);
            values.insert(values.end(), row, row + roiView.cols);
        }
    }

    const float clampedPercentile = std::clamp(percentile, 0.0f, 1.0f);
    const size_t index = static_cast<size_t>(
        std::floor(clampedPercentile * static_cast<float>(values.size() - 1)));

    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(index),
                     values.end());
    return values[index];
}

void applySafeGaussianBlur(cv::Mat &image, double sigma)
{
    CV_Assert(image.type() == CV_32F);

    cv::Mat validMask;
    cv::compare(image, 0.0f, validMask, cv::CMP_GT);

    if (cv::countNonZero(validMask) == 0) {
        return;
    }

    cv::Mat validMaskFloat;
    validMask.convertTo(validMaskFloat, CV_32F, 1.0 / 255.0);

    cv::Mat weightedImage = image.mul(validMaskFloat);
    cv::Mat blurredWeighted;
    cv::Mat blurredWeights;

    cv::GaussianBlur(weightedImage, blurredWeighted, cv::Size(0, 0), sigma, sigma, cv::BORDER_CONSTANT);
    cv::GaussianBlur(validMaskFloat, blurredWeights, cv::Size(0, 0), sigma, sigma, cv::BORDER_CONSTANT);

    image.setTo(0.0f);
    cv::divide(blurredWeighted, blurredWeights, image, 1.0, CV_32F);
    image.setTo(0.0f, blurredWeights <= 1e-6f);
}

void applyPostSigmoidAdjustments(cv::Mat &slice, float stackDimmestPercentileValue,
                                 float transitionWidth, float transitionGradient)
{
    const float safeTransitionWidth = std::max(0.0f, transitionWidth);
    const float lowerTransitionBound = std::max(0.0f, stackDimmestPercentileValue - safeTransitionWidth);
    const float safeGradient = std::max(1.0f, transitionGradient);

    slice.forEach<float>([stackDimmestPercentileValue, safeTransitionWidth, lowerTransitionBound, safeGradient](float &pixel, const int *) {
        if (pixel > stackDimmestPercentileValue) {
            return;
        }

        float subtractionScale = 1.0f;
        if (safeTransitionWidth > 0.0f && pixel >= lowerTransitionBound) {
            const float normalizedDistance =
                (stackDimmestPercentileValue - pixel) / safeTransitionWidth;
            subtractionScale = std::pow(std::clamp(normalizedDistance, 0.0f, 1.0f), safeGradient);
        }

        pixel -= stackDimmestPercentileValue * subtractionScale;
        pixel = std::max(0.0f, pixel);
    });
}

struct StackStats
{
    double minValue = std::numeric_limits<double>::infinity();
    double maxValue = -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    double sumSq = 0.0;
    std::size_t count = 0;
};

StackStats computeStackStats(const std::vector<cv::Mat> &stack)
{
    StackStats stats;
    for (const auto &slice : stack)
    {
        CV_Assert(slice.type() == CV_32F);
        double sliceMin = 0.0;
        double sliceMax = 0.0;
        cv::minMaxLoc(slice, &sliceMin, &sliceMax);
        stats.minValue = std::min(stats.minValue, sliceMin);
        stats.maxValue = std::max(stats.maxValue, sliceMax);

        for (int y = 0; y < slice.rows; ++y)
        {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x)
            {
                const double v = row[x];
                stats.sum += v;
                stats.sumSq += v * v;
                ++stats.count;
            }
        }
    }

    if (stats.count == 0)
    {
        stats.minValue = 0.0;
        stats.maxValue = 0.0;
    }
    return stats;
}

void printStackStats(const std::string &stage, const std::string &imageFile, const std::vector<cv::Mat> &stack)
{
    const StackStats stats = computeStackStats(stack);
    const double mean = (stats.count > 0) ? stats.sum / static_cast<double>(stats.count) : 0.0;
    const double variance = (stats.count > 0)
        ? std::max(0.0, stats.sumSq / static_cast<double>(stats.count) - mean * mean)
        : 0.0;
    const double stddev = std::sqrt(variance);

    std::cout << "[Preprocess] file=" << fs::path(imageFile).filename().string()
              << " stage=" << stage
              << " slices=" << stack.size()
              << " voxels=" << stats.count
              << " min=" << stats.minValue
              << " max=" << stats.maxValue
              << " mean=" << mean
              << " stddev=" << stddev
              << std::endl;
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

    processedImage.convertTo(processedImage, CV_32F, 1.0 / 255.0);

    if (config.simulation.blur_sigma > 0) {
        applySafeGaussianBlur(processedImage, config.simulation.blur_sigma);
    }

    return processedImage;
}

std::vector<cv::Mat> ImageHandler::loadFrame(const std::string &imageFile, BaseConfig &config)
{
    std::vector<cv::Mat> processedZSlices;
    std::vector<cv::Mat> interpolatedZSlices;

    std::string extension = imageFile.substr(imageFile.find_last_of('.') + 1);
    if (extension == "tiff" || extension == "tif")
    {
        std::vector<cv::Mat> tiffImage;
        cv::imreadmulti(imageFile, tiffImage, cv::IMREAD_ANYDEPTH | cv::IMREAD_COLOR);

        const auto numTiffSlices = tiffImage.size();
        if (numTiffSlices == 0) {
            throw std::runtime_error("TIFF has 0 slices: " + imageFile);
        }

        cv::Mat img = tiffImage[0];

        if (img.empty())
        {
            std::cout << "Error: Could not read the TIFF image" << '\n';
            return processedZSlices;
        }

        std::cout << "[LoadFrame] file=" << fs::path(imageFile).filename().string()
                  << " rawSlices=" << numTiffSlices
                  << " rawType=" << img.type()
                  << " rawChannels=" << img.channels()
                  << " rawRows=" << img.rows
                  << " rawCols=" << img.cols
                  << std::endl;

        for (size_t i = 0; i < numTiffSlices; ++i)
        {
            cv::Mat slice = tiffImage[i].clone();
            cv::cvtColor(slice, slice, cv::COLOR_BGR2GRAY);
            processedZSlices.push_back(processImage(slice, config));
        }

        printStackStats("post_blur_pre_sigmoid", imageFile, processedZSlices);

        float sigmoidCenter = 0.445f;
        if (!processedZSlices.empty()) {
            int calX = config.simulation.calibration_x;
            int calY = config.simulation.calibration_y;
            int calW = std::min(config.simulation.calibration_width,
                               processedZSlices[0].cols - calX);
            int calH = std::min(config.simulation.calibration_height,
                               processedZSlices[0].rows - calY);

            if (calW > 0 && calH > 0) {
                cv::Rect roi(calX, calY, calW, calH);
                float bgPercentile = computePercentileFromStackRoi(
                    processedZSlices,
                    roi,
                    config.simulation.sigmoid_center_percentile);
                sigmoidCenter = bgPercentile;
                std::cout << "[Calibration] stack_bg_percentile=" << bgPercentile
                          << " percentile=" << config.simulation.sigmoid_center_percentile
                          << " sigmoidCenter=" << sigmoidCenter
                          << " sigmoid_k=" << config.simulation.sigmoid_k << std::endl;
            }
        }

        if (config.simulation.sigmoid_k > 0) {
            for (auto &slice : processedZSlices) {
                applySigmoid(slice, config.simulation.sigmoid_k, sigmoidCenter);
            }

            const float stackDimmestPercentileValue = computePercentileFromStack(
                processedZSlices,
                config.simulation.post_sigmoid_dimmest_percentile);

            for (auto &slice : processedZSlices) {
                applyPostSigmoidAdjustments(slice,
                                            stackDimmestPercentileValue,
                                            config.simulation.post_sigmoid_dimmest_transition_width,
                                            config.simulation.post_sigmoid_dimmest_transition_gradient);
            }
        }

        printStackStats("post_sigmoid", imageFile, processedZSlices);

        const int expandFactor = config.simulation.z_scaling;
        const unsigned numSynthSlices = expandFactor * (numTiffSlices - 1) + 1;

        for (unsigned synthSlice = 0; synthSlice < numSynthSlices; ++synthSlice) {
            const int tiffSlice = static_cast<int>(synthSlice / expandFactor);
            if (synthSlice % expandFactor == 0) {
                interpolatedZSlices.push_back(processedZSlices[tiffSlice]);
            } else if (synthSlice % expandFactor == 1) {
                interpolateSlices(processedZSlices[tiffSlice],
                                  processedZSlices[tiffSlice + 1],
                                  interpolatedZSlices,
                                  expandFactor - 1);
            }
        }

        if (interpolatedZSlices.size() != numSynthSlices) {
            std::string errorMessage =
                "interpolatedZSlices must have exactly " + std::to_string(numSynthSlices) +
                " slices, but has " + std::to_string(interpolatedZSlices.size()) + " slices";
            throw std::runtime_error(errorMessage);
        }

        printStackStats("post_interpolation", imageFile, interpolatedZSlices);
    }
    else
    {
        cv::Mat img = cv::imread(imageFile);
        if (img.empty())
        {
            std::cout << "Error: Could not read the image" << '\n';
            return processedZSlices;
        }

        if (img.channels() == 3)
        {
            cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        }

        processedZSlices.push_back(processImage(img, config));
        interpolatedZSlices = processedZSlices;
    }

    std::cout << std::to_string(interpolatedZSlices.size()) << "slices built successfully" << std::endl;
    return interpolatedZSlices;
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
