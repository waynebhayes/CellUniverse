// Lineage.hpp
#ifndef LINEAGE_HPP
#define LINEAGE_HPP

#include <opencv2/opencv.hpp>
#include "Cell.hpp"
#include "ConfigTypes.hpp"
#include "Frame.hpp"
#include "types.hpp"
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include "Sphere.hpp"

namespace fs = std::filesystem;

cv::Mat processImage(const cv::Mat &image, const BaseConfig &config);

std::vector<cv::Mat> loadFrame(const std::string &imageFile, const BaseConfig &config);

class Lineage
{
public:
   Lineage(std::map<std::string, std::vector<Sphere>> initialCells, PathVec imagePaths, BaseConfig &config, std::string outputPath, int continueFrom = -1);

   void optimize(int frameIndex);

   void saveFrame(int frameIndex);

   void saveCells(int frameIndex);

   void copyCellsForward(int to);

   unsigned int length();

private:
   BaseConfig config;
   std::vector<Frame> frames;
   std::string outputPath;
};

#endif
