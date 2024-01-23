// Cell.cpp
//Yuxuan's version
#include <iostream>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <random>

class CellParams {
    //The CellParams class stores the parameters of a particular cell.
public:
    std::string name;
    //CellParams(const std::string& name) : name(name) {} //Krishna
};

class CellConfig {
    // Abstract base class for cell configurations.
};

class PerturbParams {
    //Used with a CellConfig to add perturb parameters.
public:
    float prob;
    float mu;
    float sigma;

    float get_perturb_offset() const {
        if (rand() / static_cast<float>(RAND_MAX) < prob) {
            return mu + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * sigma;
        }
        else {
            return mu;
        }
    }
};

class Cell {
    //The Cell class stores information about a particular cell.
protected:
    CellParams paramClass;
    CellConfig cellConfig;
public:
    Cell(const CellParams& initProps) : paramClass(initProps) {}
    virtual ~Cell() = default;

    virtual void draw() const = 0;
    virtual void draw_outline() const = 0;
    virtual Cell* get_perturbed_cell() const = 0;
    virtual Cell* get_paramaterized_cell(std::unordered_map<std::string, float> params = {}) const = 0;
    virtual std::tuple<Cell*, Cell*, bool> get_split_cells() const = 0;
    virtual CellParams get_cell_params() const = 0;
    static bool check_if_cells_valid(const std::vector<Cell*>& cells) { return false; }
    virtual std::tuple<std::vector<float>, std::vector<float>> calculate_corners() const = 0;
    virtual std::tuple<std::vector<float>, std::vector<float>> calculate_minimum_box(const Cell& perturbed_cell) const = 0;
};
