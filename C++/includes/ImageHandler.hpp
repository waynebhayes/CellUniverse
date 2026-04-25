#ifndef IMAGEHANDLER_HPP
#define IMAGEHANDLER_HPP

#include "ConfigTypes.hpp"
#include "Frame.hpp"
#include "types.hpp"

#include <opencv2/opencv.hpp>
#include <ostream>

class ImageHandler
{
public:
    static Image processImage(const Image &image, const BaseConfig &config);
    static std::vector<cv::Mat> loadRawFrame(const std::string &imageFile,
                                             const BaseConfig &config,
                                             std::ostream *logSink = nullptr);
    static std::vector<cv::Mat> preprocessLoadedFrame(const std::vector<cv::Mat> &normalizedSlices,
                                                      const std::string &imageFile,
                                                      const BaseConfig &config,
                                                      std::ostream *logSink = nullptr);
    static float evaluateSequenceContrastScore(const ImageStack &sequence, const BaseConfig &config);
    static float evaluateBestWindowContrastScore(const ImageStack &sequence, const BaseConfig &config);
    static std::vector<cv::Mat> loadFrame(const std::string &imageFile,
                                          BaseConfig &config,
                                          std::ostream *logSink = nullptr);
    static PathVec getImageFilePaths(const std::string &input, int firstFrame, int lastFrame, BaseConfig &config);

private:
    static ImageStack processPreparedSequence(const ImageStack &sequence,
                                             const BaseConfig &config,
                                             std::ostream &log,
                                             const std::string &imageFile);
    static ImageStack applyPostBlackpointLift(const ImageStack &sequence,
                                             const BaseConfig &config,
                                             std::ostream &log,
                                             const std::string &imageFile);
    static cv::Mat boxMean(const cv::Mat &image, int windowSize);
    static float evaluateSequenceContrastScoreForRadius(const ImageStack &sequence,
                                                        const BaseConfig &config,
                                                        float radiusAtScale);
    static float computePercentileFromSlice(const cv::Mat &slice, float percentileFraction);
    static float computePercentileFromValues(std::vector<float> values, float percentileFraction);
};

#endif
