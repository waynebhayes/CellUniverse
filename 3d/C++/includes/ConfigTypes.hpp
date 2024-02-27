// ConfigTypes.hpp
#ifndef CONFIGTYPES_HPP
#define CONFIGTYPES_HPP
#include <vector>
#include <string>
#include "Sphere.hpp"
#include "yaml-cpp/yaml.h"

class SimulationConfig {
public:
    int iterations_per_cell;
    float background_color;
    float cell_color;
    int padding;
    float z_scaling;
    float blur_sigma;
    int z_slices;
    std::vector<int> z_values;

    // Constructor with default values
    SimulationConfig() : iterations_per_cell(0), background_color(0.0f), cell_color(0.0f),
                         padding(0), z_scaling(1.0), blur_sigma(0.0f), z_slices(-1) {
    }
    void explodeConfig(const YAML::Node node) {
        iterations_per_cell = node["iterations_per_cell"].as<int>();
        background_color = node["background_color"].as<float>();
        cell_color = node["cell_color"].as<float>();
        padding = node["padding"].as<int>();
        z_scaling = node["z_scaling"].as<float>();
        blur_sigma = node["blur_sigma"].as<float>();
    }
};

class ProbabilityConfig {
public:
    float perturbation;
    float split;
    ProbabilityConfig() : perturbation(0.0f), split(0.0f) {
    }

    void explodeConfig(const YAML::Node& node) {
        if (node["perturbation"]) {
            perturbation = node["perturbation"].as<float>();
        }
        if (node["split"]) {
            split = node["split"].as<float>();
        }
    }
};


class BaseConfig {
public:
    std::string cellType;
    // TODO: Change to a template to support more different types of config
    CellConfig* cell;
    SimulationConfig simulation;
    ProbabilityConfig prob;
    BaseConfig() :cell(nullptr) {};
    // load the BaseConfig with a YAML node
    void explodeConfig(const YAML::Node& node) {
        cellType = node["cellType"].as<std::string>();
        // TODO: Now cell is always pointing to a sphere config, make it more dynamic later
        cell = new SphereConfig;
        cell->explodeConfig(node["cell"]);
        simulation.explodeConfig(node["simulation"]);
        prob.explodeConfig(node["prob"]);
    }
};

#endif
