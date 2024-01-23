// Frame.hpp
#ifndef FRAME_HPP
#define FRAME_HPP

#include "Cell.hpp"
#include "Config.hpp"
#include <vector>

class Frame
{
public:
    size_t size();
    std::vector<Cell> cells;
    std::vector<cv::Mat> generateOutputImages();
    std::vector<cv::Mat> generateOutputSynthImages();
};

#endif