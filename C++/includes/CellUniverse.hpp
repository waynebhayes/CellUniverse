#ifndef CELLUNIVERSE_HPP
#define CELLUNIVERSE_HPP

#include <opencv2/opencv.hpp>
#include "ConfigTypes.hpp"
#include "Frame.hpp"
#include "types.hpp"
#include "Spheroid.hpp"

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

Image processImage(const Image &image, const BaseConfig &config);
std::vector<cv::Mat> loadFrame(const std::string &imageFile, const BaseConfig &config);

class CellUniverse
{
public:
    CellUniverse(std::map<std::string, std::vector<Spheroid>> initialCells,
                 PathVec imagePaths,
                 BaseConfig &config,
                 std::string outputPath,
                 int firstFrame = 0,
                 int continueFrom = -1);

    void optimize(int frameIndex);
    void saveImages(int frameIndex);
    void saveCells(int frameIndex);
    void copyCellsForward(int to);
    unsigned int length();

    // ---- Added for realtime viewer ----
    const std::vector<Spheroid> &getCells(int frameIndex) const;
    std::vector<std::string> getCellNames(int frameIndex) const;

private:
   BaseConfig config;
   std::vector<Frame> frames;
   std::string outputPath;
   int firstFrame;
};

#endif
