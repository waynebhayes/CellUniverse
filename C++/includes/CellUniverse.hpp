#ifndef CELLUNIVERSE_HPP
#define CELLUNIVERSE_HPP

#include <opencv2/opencv.hpp>
#include "ConfigTypes.hpp"
#include "Frame.hpp"
#include "types.hpp"
#include "Ellipsoid.hpp"

#include <array>
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
    // Memory optimization (M1): after this frame has been optimized, saved,
    // and its snapshot captured, release its image stacks. Cells and
    // snapshot metadata are retained. Enables long-horizon runs (60+ frames)
    // without 13+ GB memory peaks.
    void releaseFrameImages(int frameIndex);
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
   // Frozen per-cell shape reference (a, b, c radii). Captured at cell
   // birth (frame 1 for initial cells; post-refit at split-accept for
   // daughters) and NEVER updated. Used as the pixel-collection mask
   // basis in subsequent frames' shape fits, decoupled from snap radii,
   // so a bloated fit in frame N can't compound into an even bigger
   // mask for frame N+1. See 2026-04-15 compounding-bloat analysis.
   std::map<std::string, std::array<float, 3>> cellShapeReference;
   // Birth-time radii. Captured once at first appearance, NEVER updated.
   // Used as the pixel-collection mask basis: mask = birth × maskScale.
   // Decoupled from the bounded ref so the mask can't participate in
   // feedback loops (neither upward bloat nor downward thinning).
   // The bounded ref is used ONLY for the fit-side growth cap.
   std::map<std::string, std::array<float, 3>> cellShapeBirth;
};

#endif
