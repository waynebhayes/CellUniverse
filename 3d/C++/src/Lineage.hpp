// Lineage.hpp
#ifndef LINEAGE_HPP
#define LINEAGE_HPP

#include "Cells/Cell.hpp"
#include "Config/BaseConfig.hpp"
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;



class Lineage
{
public:
    Lineage(std::map<std::string, std::vector<Cell>> initialCells, std::vector<fs::path> imagePaths, BaseConfig config, fs::path outputPath, int continueFrom = -1);

    void optimize(int frameIndex);

    void saveImages(int frameIndex);

    void saveCells(int frameIndex);

    void copyCellsForward(int to);

    unsigned int getLength();

private:
    BaseConfig config;
    std::vector<Cell> frames;
    fs::path outputPath;
};

#endif
