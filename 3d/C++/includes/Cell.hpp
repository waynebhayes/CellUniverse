// Cell.hpp
#ifndef CELL_HPP
#define CELL_HPP

#include <iostream>
#include <vector>
#include "ConfigTypes.hpp"
#include "yaml-cpp/yaml.h"
#include "opencv2/opencv.hpp"

class Cell {
    //The Cell class stores information about a particular cell.
//    CellParams paramClass;
//    CellConfig cellConfig; //?unsure
public:
//    Cell(const CellParams& initProps) : paramClass(initProps) {}
    Cell() {}
    virtual ~Cell() = default;
    virtual void draw(cv::Mat& image, SimulationConfig simulationConfig, cv::Mat* cellMap = nullptr, float z = 0) const = 0;
    virtual void draw_outline(cv::Mat& image, float color, float z = 0) const = 0;
    virtual Cell* get_perturbed_cell() const = 0;
    virtual Cell* get_parameterized_cell(std::unordered_map<std::string, float> params = {}) const = 0;
    virtual std::tuple<Cell*, Cell*, bool> get_split_cells() const = 0;
    virtual CellParams get_cell_params() const = 0;
    static bool check_if_cells_valid(const std::vector<Cell*>& cells) { return false; }
    virtual std::pair<std::vector<float>, std::vector<float>> calculate_corners() const = 0;
    virtual std::pair<std::vector<float>, std::vector<float>> calculate_minimum_box(Cell& perturbed_cell) const = 0;
};
#endif
