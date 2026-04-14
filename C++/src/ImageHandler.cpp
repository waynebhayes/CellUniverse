#include "../includes/ImageHandler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
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
    bool firstValue = true;

    for (const auto &slice : stack)
    {
        CV_Assert(slice.type() == CV_32F);

        double sliceMin = 0.0;
        double sliceMax = 0.0;
        cv::minMaxLoc(slice, &sliceMin, &sliceMax);
        if (firstValue)
        {
            stats.minValue = sliceMin;
            stats.maxValue = sliceMax;
            firstValue = false;
        }
        else
        {
            stats.minValue = std::min(stats.minValue, sliceMin);
            stats.maxValue = std::max(stats.maxValue, sliceMax);
        }

        for (int y = 0; y < slice.rows; ++y)
        {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x)
            {
                const double value = row[x];
                sum += value;
                sumSq += value * value;
                ++stats.count;
            }
        }
    }

    if (stats.count == 0)
    {
        return stats;
    }

    stats.mean = sum / static_cast<double>(stats.count);
    const double variance =
        std::max(0.0, sumSq / static_cast<double>(stats.count) - stats.mean * stats.mean);
    stats.stddev = std::sqrt(variance);
    return stats;
}

void printStackStats(const std::string &stage, const std::string &imageFile, const ImageStack &stack)
{
    const StackStats stats = computeStackStats(stack);
    std::cout << "[Preprocess] file=" << fs::path(imageFile).filename().string()
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
    ImageStack cloned;
    cloned.reserve(sequence.size());
    for (const auto &slice : sequence)
    {
        cloned.push_back(slice.clone());
    }
    return cloned;
}

void clipStack(ImageStack &sequence)
{
    for (auto &slice : sequence)
    {
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

    processedImage.convertTo(processedImage, CV_32F, 1.0 / 255.0);

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
    const std::array<float, 2> radii = {
        std::max(1.0f, aRadius),
        std::max(1.0f, cRadius)
    };

    std::vector<float> scaleScores;
    scaleScores.reserve(radii.size());

    for (const float radiusAtScale : radii)
    {
        const int innerWindow = makeOddAtLeast(
            static_cast<int>(std::lround(radiusAtScale * 2.0f + 1.0f)));
        const int outerWindow = makeOddAtLeast(
            static_cast<int>(std::lround(radiusAtScale * 4.0f + 1.0f)),
            innerWindow + 2);

        std::vector<float> sliceScores;
        sliceScores.reserve(sequence.size());

        for (const auto &slice : sequence)
        {
            CV_Assert(slice.type() == CV_32F);

            const cv::Mat innerMean = boxMean(slice, innerWindow);
            const cv::Mat outerMean = boxMean(slice, outerWindow);

            std::vector<float> contrastValues;
            contrastValues.reserve(static_cast<std::size_t>(slice.total()));

            for (int y = 0; y < slice.rows; ++y)
            {
                const float *innerRow = innerMean.ptr<float>(y);
                const float *outerRow = outerMean.ptr<float>(y);
                for (int x = 0; x < slice.cols; ++x)
                {
                    const float localDifference = std::abs(innerRow[x] - outerRow[x]);
                    if (localDifference < config.simulation.contrast_structure_threshold)
                    {
                        continue;
                    }

                    const float localContrast =
                        localDifference / (outerRow[x] + config.simulation.contrast_eps);
                    contrastValues.push_back(localContrast);
                }
            }

            if (contrastValues.empty())
            {
                sliceScores.push_back(0.0f);
                continue;
            }

            sliceScores.push_back(computePercentileFromValues(
                std::move(contrastValues),
                config.simulation.iterative_score_percentile));
        }

        if (sliceScores.empty())
        {
            scaleScores.push_back(0.0f);
            continue;
        }

        scaleScores.push_back(computePercentileFromValues(std::move(sliceScores), 0.5f));
    }

    if (scaleScores.empty())
    {
        return 0.0f;
    }

    float sumScores = 0.0f;
    for (float score : scaleScores)
    {
        sumScores += score;
    }
    return sumScores / static_cast<float>(scaleScores.size());
}

float ImageHandler::evaluateSequencePercentileMichelsonContrast(const ImageStack &sequence, const BaseConfig &config)
{
    std::vector<float> sliceScores;
    sliceScores.reserve(sequence.size());

    for (const auto &slice : sequence)
    {
        const float lowValue = computePercentileFromSlice(
            slice, config.simulation.michelson_low_percentile);
        const float highValue = computePercentileFromSlice(
            slice, config.simulation.michelson_high_percentile);
        sliceScores.push_back(
            (highValue - lowValue) /
            (highValue + lowValue + config.simulation.michelson_eps));
    }

    if (sliceScores.empty())
    {
        return 0.0f;
    }

    return computePercentileFromValues(std::move(sliceScores), 0.5f);
}

float ImageHandler::evaluateSequencePercentileWeberContrast(const ImageStack &sequence, const BaseConfig &config)
{
    std::vector<float> sliceScores;
    sliceScores.reserve(sequence.size());

    for (const auto &slice : sequence)
    {
        const float backgroundValue = computePercentileFromSlice(
            slice, config.simulation.weber_background_percentile);
        const float signalValue = computePercentileFromSlice(
            slice, config.simulation.weber_signal_percentile);
        const float stableBackground =
            std::max(backgroundValue, config.simulation.weber_background_floor);
        sliceScores.push_back(
            (signalValue - stableBackground) /
            (stableBackground + config.simulation.weber_eps));
    }

    if (sliceScores.empty())
    {
        return 0.0f;
    }

    return computePercentileFromValues(std::move(sliceScores), 0.5f);
}

ImageStack ImageHandler::processPreparedSequence(const ImageStack &sequence, const BaseConfig &config)
{
    ImageStack current = cloneStack(sequence);

    ImageStack bestSequence = cloneStack(current);
    float bestScore = -std::numeric_limits<float>::infinity();
    float previousScore = 0.0f;
    bool hasPreviousScore = false;

    int count = 0;
    bool rewardNextRound = true;
    float currentPenalty = config.simulation.iterative_penalty;
    int noImprovementCount = 0;
    bool restoreBestBeforeReward = false;
    float scorePercentile = config.simulation.iterative_score_percentile;
    float rewardGate = config.simulation.iterative_reward_gate;

    while (true)
    {
        for (auto &slice : current)
        {
            for (int y = 0; y < slice.rows; ++y)
            {
                float *row = slice.ptr<float>(y);
                for (int x = 0; x < slice.cols; ++x)
                {
                    if (row[x] < config.simulation.iterative_penalty_range)
                    {
                        row[x] -= currentPenalty;
                        if (row[x] < 0.0f)
                        {
                            row[x] = 0.0f;
                        }
                    }
                }
            }
        }

        if (rewardNextRound)
        {
            if (restoreBestBeforeReward)
            {
                current = cloneStack(bestSequence);
                restoreBestBeforeReward = false;
            }

            for (auto &slice : current)
            {
                for (int y = 0; y < slice.rows; ++y)
                {
                    float *row = slice.ptr<float>(y);
                    for (int x = 0; x < slice.cols; ++x)
                    {
                        if (row[x] > rewardGate)
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

            scorePercentile = std::min(
                scorePercentile + config.simulation.iterative_score_percentile_increment,
                config.simulation.iterative_score_percentile_max);
            rewardGate = std::max(
                config.simulation.iterative_reward_gate_min,
                rewardGate -
                    config.simulation.iterative_reward_gate_decrement);
        }

        rewardNextRound = !rewardNextRound;

        BaseConfig scoringConfig = config;
        scoringConfig.simulation.iterative_score_percentile = scorePercentile;
        const float score = evaluateSequenceContrastScore(current, scoringConfig);

        if (hasPreviousScore &&
            previousScore - score >= config.simulation.iterative_score_drop_stop_threshold)
        {
            rewardNextRound = true;
            restoreBestBeforeReward = true;
            currentPenalty = std::max(
                config.simulation.iterative_min_penalty,
                currentPenalty * config.simulation.iterative_collapse_backoff);
            previousScore = score;
            hasPreviousScore = true;
            ++count;
            continue;
        }

        if (score > bestScore + config.simulation.iterative_improvement_tolerance)
        {
            bestScore = score;
            bestSequence = cloneStack(current);
            noImprovementCount = 0;
        }
        else
        {
            ++noImprovementCount;
        }

        if (count % 50 == 0)
        {
            std::cout << "[IterPreprocess] round=" << count
                      << " score=" << score << std::endl;
        }

        previousScore = score;
        hasPreviousScore = true;
        ++count;

        if (score >= config.simulation.iterative_score_max)
        {
            break;
        }

        if (score == 0.0f)
        {
            currentPenalty = std::max(
                config.simulation.iterative_min_penalty,
                currentPenalty * config.simulation.iterative_collapse_backoff);
            current = cloneStack(bestSequence);
            rewardNextRound = true;
            continue;
        }

        if (noImprovementCount >= config.simulation.iterative_no_improvement_patience)
        {
            if (currentPenalty <= config.simulation.iterative_min_penalty)
            {
                break;
            }

            currentPenalty = std::max(
                config.simulation.iterative_min_penalty,
                currentPenalty * config.simulation.iterative_collapse_backoff);
            current = cloneStack(bestSequence);
            noImprovementCount = 0;
            rewardNextRound = true;
            continue;
        }

        if (count >= config.simulation.iterative_max_count)
        {
            break;
        }
    }

    std::cout << "[IterPreprocess] best_score=" << bestScore << std::endl;

    for (auto &slice : bestSequence)
    {
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

        if (config.simulation.post_process_blur_sigma > 0.0f)
        {
            cv::GaussianBlur(slice,
                             slice,
                             cv::Size(0, 0),
                             config.simulation.post_process_blur_sigma,
                             config.simulation.post_process_blur_sigma);
        }
    }

    clipStack(bestSequence);
    return bestSequence;
}

std::vector<cv::Mat> ImageHandler::loadFrame(const std::string &imageFile, BaseConfig &config)
{
    std::vector<cv::Mat> processedZSlices;
    std::vector<cv::Mat> interpolatedZSlices;

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
            return processedZSlices;
        }

        std::cout << "[LoadFrame] file=" << fs::path(imageFile).filename().string()
                  << " rawSlices=" << numTiffSlices
                  << " rawType=" << firstSlice.type()
                  << " rawChannels=" << firstSlice.channels()
                  << " rawRows=" << firstSlice.rows
                  << " rawCols=" << firstSlice.cols
                  << std::endl;

        processedZSlices.reserve(numTiffSlices);
        for (const auto &rawSlice : tiffImage)
        {
            cv::Mat slice = rawSlice;
            if (slice.channels() == 3)
            {
                cv::cvtColor(slice, slice, cv::COLOR_BGR2GRAY);
            }
            processedZSlices.push_back(processImage(slice, config));
        }
    }
    else
    {
        cv::Mat image = cv::imread(imageFile);
        if (image.empty())
        {
            std::cout << "Error: Could not read the image" << '\n';
            return processedZSlices;
        }

        if (image.channels() == 3)
        {
            cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        }

        processedZSlices.push_back(processImage(image, config));
    }

    printStackStats("normalized_input", imageFile, processedZSlices);

    processedZSlices = processPreparedSequence(processedZSlices, config);

    const float localScore = evaluateSequenceContrastScore(processedZSlices, config);
    const float michelsonScore = evaluateSequencePercentileMichelsonContrast(processedZSlices, config);
    const float weberScore = evaluateSequencePercentileWeberContrast(processedZSlices, config);

    std::cout << "[PreprocessScores] file=" << fs::path(imageFile).filename().string()
              << " local=" << localScore
              << " michelson=" << michelsonScore
              << " weber=" << weberScore
              << std::endl;

    printStackStats("processed_sequence", imageFile, processedZSlices);

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

        for (unsigned synthSlice = 0; synthSlice < numSynthSlices; ++synthSlice)
        {
            const int sourceSlice = static_cast<int>(synthSlice / expandFactor);
            if (synthSlice % expandFactor == 0)
            {
                interpolatedZSlices.push_back(processedZSlices[sourceSlice]);
            }
            else if (synthSlice % expandFactor == 1)
            {
                interpolateSlices(processedZSlices[sourceSlice],
                                  processedZSlices[sourceSlice + 1],
                                  interpolatedZSlices,
                                  expandFactor - 1);
            }
        }

        if (interpolatedZSlices.size() != numSynthSlices)
        {
            throw std::runtime_error(
                "interpolatedZSlices must have exactly " + std::to_string(numSynthSlices) +
                " slices, but has " + std::to_string(interpolatedZSlices.size()) + " slices");
        }
    }

    printStackStats("post_interpolation", imageFile, interpolatedZSlices);
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
