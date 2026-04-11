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
};

#endif
