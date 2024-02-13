// ConfigTypes.hpp
#ifndef CONFIGTYPES_HPP
#define CONFIGTYPES_HPP
#include "Cell.hpp"

class SimulationConfig {
public:
    int iterationsPerCell;
    float backgroundColor;
    float cellColor;
    int padding = 0;
    float zScaling = 1;
    float blurSigma = 0;
private:
    int zSlices = -1;
    std::vector<int> zValues;

    SimulationConfig(const YAML::Node& node) {
        iterationsPerCell = node["iterations_per_cell"].as<int>();
        backgroundColor = node["background_color"].as<float>();
        cellColor = node["cell_color"].as<float>();
        padding = node["padding"].as<int>();
        zScaling = node["z_scaling"].as<float>();
        blurSigma = node["blur_sigma"].as<float>();

    }
    // No need for C++
    //void checkZValues() const {
    //    if (!zValues.empty()) {
    //        throw std::invalid_argument("zValues should not be set manually");
    //    }
    //}

    //void checkZSlices() const {
    //    if (zSlices != -1) {
    //        throw std::invalid_argument("zSlices should not be set manually");
    //    }
    //}
};

class ProbabilityConfig {
public:
    float perturbation;
    float split;

    static void checkProbability(std::vector<float> values) {//TODO
    	}
    ProbabilityConfig(const YAML::Node& node)
    {
        perturbation = node["perturbation"].as<float>();
        split = node["split"].as<float>();
    }
};

class BaseConfig {
public:
    CellConfig cell;
    char CellType; //Added new attribute for type check: 'b'/'s' stands for bacilli/sphere
    SimulationConfig simulation;
    ProbabilityConfig prob;
    BaseConfig(const YAML::Node& node) :
        simulation(node), prob(node), cell(node) {
        // Parse YAML node and initialize config object
    }
};
#endif
