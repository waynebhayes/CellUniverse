#ifndef IMAGEHANDLER_HPP
#define IMAGEHANDLER_HPP

#include "ConfigTypes.hpp"
#include "Frame.hpp"
#include "types.hpp"

#include <opencv2/opencv.hpp>

class ImageHandler
{
public:
    static Image processImage(const Image &image, const BaseConfig &config);
    static std::vector<cv::Mat> loadFrame(const std::string &imageFile, BaseConfig &config);
    static PathVec getImageFilePaths(const std::string &input, int firstFrame, int lastFrame, BaseConfig &config);
    static void applyDatasetRuntimeProfile(const std::string &input, BaseConfig &config);

private:
    static ImageStack processPreparedSequence(const ImageStack &sequence, const BaseConfig &config);
    static float evaluateSequenceContrastScore(const ImageStack &sequence, const BaseConfig &config);
    static float evaluateSequencePercentileMichelsonContrast(const ImageStack &sequence, const BaseConfig &config);
    static float evaluateSequencePercentileWeberContrast(const ImageStack &sequence, const BaseConfig &config);
    static cv::Mat boxMean(const cv::Mat &image, int windowSize);
    static float computePercentileFromSlice(const cv::Mat &slice, float percentileFraction);
    static float computePercentileFromValues(std::vector<float> values, float percentileFraction);
};

#endif
