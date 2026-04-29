#ifndef CSVHANDLER_HPP
#define CSVHANDLER_HPP

#include "types.hpp"

#include <string>
#include <vector>

struct InitialCellRecord
{
    std::string filePath;
    std::string cellName;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float aRadius = 0.0f;
    float bRadius = 0.0f;
    float cRadius = 0.0f;
    bool isTrash = false;
};

class CsvHandler
{
public:
    static std::vector<InitialCellRecord> loadInitialCells(const Path &initParamsPath,
                                                           const std::string &firstFrameFile,
                                                           float zScaling,
                                                           float initialRadiusScale);
};

#endif
