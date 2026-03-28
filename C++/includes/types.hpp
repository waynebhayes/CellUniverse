//
// Created by yuant on 1/30/2024.
//

#ifndef CELLUNIVERSE_TYPES_HPP
#define CELLUNIVERSE_TYPES_HPP


#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <map>

enum argKeywords {
    ff = 1,
    lf,
    input,
    output,
    config,
    initial
};

using Path = std::string;
using Image = cv::Mat;
using ImageStack = std::vector<cv::Mat>;
using CostCallbackPair = std::pair<double, std::function<void(bool)>>;
using CallBackFunc = std::function<void(bool)>;
// using CellMap = std::map<Path, std::vector<Sphere>>;
using Cost = double;
using Corner = std::vector<float>;
using MinBox = std::pair<Corner, Corner>;
namespace fs = std::filesystem;
using PathVec = std::vector<fs::path>;

#endif //CELLUNIVERSE_TYPES_HPP
