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
#include <memory>

namespace fs = std::filesystem;

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
    void copyCellsForward(size_t to);
    unsigned int length();

    // ---- Added for realtime viewer ----
    const std::vector<Spheroid> &getCells(int frameIndex) const;
    std::vector<std::string> getCellNames(int frameIndex) const;

private:
   BaseConfig config;
   PathVec imagePaths;
   std::vector<std::unique_ptr<Frame>> frames;
   std::string outputPath;
   int firstFrame;
   int continueFrom;
   std::map<std::string, std::vector<Spheroid>> initialCells;
   std::map<size_t, std::vector<Spheroid>> carriedCells;
   double accumulatedFrameMeanBrightness = 0.0;
   size_t loadedFrameMeanCount = 0;
   std::map<std::string, PreviousFrameSnapshot> previousSnapshots;

   void ensureFrameLoaded(size_t frameIndex);
   std::vector<Spheroid> seedCellsForFrame(size_t frameIndex) const;
   static PreviousFrameSnapshot buildSnapshotFromCell(const Spheroid &cell);
   void seedPreviousSnapshotsFromCells(const std::vector<Spheroid> &cells, int displayFrame,
                                       const std::string &reason);
};

#endif
