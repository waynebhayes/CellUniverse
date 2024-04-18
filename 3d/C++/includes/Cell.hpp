// Cell.hpp
#ifndef CELL_HPP
#define CELL_HPP

#include <iostream>
#include <vector>
#include "ConfigTypes.hpp"
#include "yaml-cpp/yaml.h"
#include "opencv2/opencv.hpp"
#include "types.hpp"

class Cell {
    //The Cell class stores information about a particular cell.
//    CellParams paramClass;
//    CellConfig cellConfig; //?unsure
public:
//    Cell(const CellParams& initProps) : paramClass(initProps) {}
    Cell() {}
    virtual ~Cell() = default;
    virtual void draw(cv::Mat& image, SimulationConfig simulationConfig, cv::Mat* cellMap = nullptr, float z = 0) const = 0;
    virtual void drawOutline(cv::Mat& image, float color, float z = 0) const = 0;
    virtual Cell* getPerturbedCell() const = 0;
    virtual Cell* getParameterizedCell(std::unordered_map<std::string, float> params = {}) const = 0;
    virtual std::tuple<Cell*, Cell*, bool> getSplitCells() const = 0;
    virtual CellParams getCellParams() const = 0;
    static bool checkIfCellsValid(const std::vector<Cell*>& cells) { return false; }
    virtual MinBox calculateCorners() const = 0;
    virtual MinBox calculateMinimumBox(Cell& perturbed_cell) const = 0;
};
#endif
