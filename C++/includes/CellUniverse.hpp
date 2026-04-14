#ifndef CELLUNIVERSE_HPP
#define CELLUNIVERSE_HPP

#include <opencv2/opencv.hpp>
#include "ConfigTypes.hpp"
#include "Frame.hpp"
#include "types.hpp"
#include "Ellipsoid.hpp"

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

class CellUniverse
{
public:
    CellUniverse(std::map<std::string, std::vector<Ellipsoid>> initialCells,
                 PathVec imagePaths,
                 BaseConfig &config,
                 std::string outputPath,
                 int firstFrame = 0,
                 int continueFrom = -1);

    void optimize(int frameIndex);
    void saveImages(int frameIndex);
    void saveCells(int frameIndex);
    void copyCellsForward(size_t to);
    unsigned int length();

    // ---- Added for realtime viewer ----
    const std::vector<Ellipsoid> &getCells(int frameIndex) const;
    std::vector<std::string> getCellNames(int frameIndex) const;

private:
   BaseConfig config;
   std::vector<Frame> frames;
   std::string outputPath;
   int firstFrame;
   std::map<std::string, PreviousFrameSnapshot> previousSnapshots;
};

#endif
