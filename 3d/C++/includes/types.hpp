//
// Created by yuant on 1/30/2024.
//

#ifndef CELLUNIVERSE_TYPES_HPP
#define CELLUNIVERSE_TYPES_HPP


#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>

typedef std::string Path;
typedef cv::Mat Image;
typedef std::vector<cv::Mat> ImageStack;
typedef std::pair<double, std::function<void(bool)>> CostCallbackPair;
typedef std::function<void(bool)> CallBackFunc;
typedef std::unordered_map<std::string, ImageStack> ParamImageMap;
typedef std::unordered_map<std::string, float> ParamValMap;
typedef double Cost;

#endif //CELLUNIVERSE_TYPES_HPP
