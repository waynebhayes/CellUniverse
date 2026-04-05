// ConfigTypes.hpp
#ifndef CONFIGTYPES_HPP
#define CONFIGTYPES_HPP
#include <vector>
#include <random>
#include <string>
#include <memory>
#include "yaml-cpp/yaml.h"
#include <iostream>

class SimulationConfig {
public:
    int iterations_per_cell;
    float background_color;
    float cell_color;
    float z_scaling;
    float blur_sigma;
    int z_slices;
    float sigmoid_k = 75.0f;
    float sigmoid_center = 0.445f;
    float sigmoid_center_offset = 0.047f;
    int calibration_x = 20;
    int calibration_y = 20;
    int calibration_z = 0;
    int calibration_width = 50;
    int calibration_height = 31;

    // Constructor with default values
    SimulationConfig() : iterations_per_cell(0), background_color(0.0f),
                         cell_color(0.6f), z_scaling(1.0), blur_sigma(0.0f), z_slices(-1) {
    }
    void explodeConfig(const YAML::Node& node) {
        iterations_per_cell = node["iterations_per_cell"].as<int>();
        background_color = node["background_color"].as<float>();
        if (node["cell_color"]) cell_color = node["cell_color"].as<float>();
        z_scaling = node["z_scaling"].as<float>();
        blur_sigma = node["blur_sigma"].as<float>();
        if (node["sigmoid_k"]) sigmoid_k = node["sigmoid_k"].as<float>();
        if (node["sigmoid_center"]) sigmoid_center = node["sigmoid_center"].as<float>();
        if (node["sigmoid_center_offset"]) sigmoid_center_offset = node["sigmoid_center_offset"].as<float>();
        if (node["calibration_x"]) calibration_x = node["calibration_x"].as<int>();
        if (node["calibration_y"]) calibration_y = node["calibration_y"].as<int>();
        if (node["calibration_z"]) calibration_z = node["calibration_z"].as<int>();
        if (node["calibration_width"]) calibration_width = node["calibration_width"].as<int>();
        if (node["calibration_height"]) calibration_height = node["calibration_height"].as<int>();
    }
    void printConfig() const {
        std::cout << "Simulation Config\n";
        std::cout << "iterations_per_cell: " << iterations_per_cell << '\n';
        std::cout << "background_color: " << background_color << '\n';
        std::cout << "z_scaling: " << z_scaling << '\n';
        std::cout << "blur_sigma: " << blur_sigma << '\n';
        std::cout << "z_slices: " << z_slices << std::endl;
    }
};

class ProbabilityConfig {
public:
    float split;
    float split_cost;
    float split_elongation_threshold;
    float overlap_penalty_weight;
    int split_burn_in_iterations = 500;
    float max_split_probability = 0.5f;
    ProbabilityConfig() : split(0.0f), split_cost(0.0f),
                          split_elongation_threshold(1.3f), overlap_penalty_weight(1000.0f) {
    }

    void explodeConfig(const YAML::Node& node) {
        if (node["split"]) {
            split = node["split"].as<float>();
        }
        if (node["split_cost"]) {
            split_cost = node["split_cost"].as<float>();
        }
        if (node["split_elongation_threshold"]) {
            split_elongation_threshold = node["split_elongation_threshold"].as<float>();
        }
        if (node["overlap_penalty_weight"]) {
            overlap_penalty_weight = node["overlap_penalty_weight"].as<float>();
        }
        if (node["split_burn_in_iterations"]) {
            split_burn_in_iterations = node["split_burn_in_iterations"].as<int>();
        }
        if (node["max_split_probability"]) {
            max_split_probability = node["max_split_probability"].as<float>();
        }
    }
    void printConfig() const {
        std::cout << "Probability Config\n";
        std::cout << "split: " << split << '\n';
        std::cout << "split_cost: " << split_cost << '\n';
        std::cout << "split_elongation_threshold: " << split_elongation_threshold << '\n';
        std::cout << "overlap_penalty_weight: " << overlap_penalty_weight << std::endl;
    }
};

class CellParams {
    //The CellParams class stores the parameters of a particular cell.
public:
    std::string name;
    CellParams(const std::string& name_)
        : name(name_){
    }
};

class PerturbParams {
    //Used with a cell config to add perturb parameters.
public:
    float prob  = 0.0f;
    float mu    = 0.0f;
    float sigma = 0.0f;
    void explodeParams(const YAML::Node& node) {
        prob = node["prob"].as<float>();
        mu = node["mu"].as<float>();
        sigma = node["sigma"].as<float>();
    }
    [[nodiscard]] float getPerturbOffset() const {
        thread_local std::mt19937 gen{std::random_device{}()};
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        if (dis(gen) < prob) {
            std::normal_distribution<float> d(mu, sigma);
            return d(gen);
        }
        return mu;
    }
};

class SpheroidConfig {
public:
    PerturbParams x{};
    PerturbParams y{};
    PerturbParams z{};
    PerturbParams majorRadius{};
    PerturbParams minorRadius{};
    PerturbParams thetaX{};
    PerturbParams thetaY{};
    PerturbParams thetaZ{};
    PerturbParams brightness{};
    double minMajorRadius{};
    double maxMajorRadius{};
    double minMinorRadius{};
    double maxMinorRadius{};
    double minBrightness{0.1};
    double maxBrightness{1.0};
    float splitBrightestFraction{0.10f};
    ~SpheroidConfig() = default;

    void explodeConfig(const YAML::Node& node)
    {
        x.explodeParams(node["x"]);
        y.explodeParams(node["y"]);
        z.explodeParams(node["z"]);
        majorRadius.explodeParams(node["majorRadius"]);
        minorRadius.explodeParams(node["minorRadius"]);
        thetaX.explodeParams(node["thetaX"]);
        thetaY.explodeParams(node["thetaY"]);
        thetaZ.explodeParams(node["thetaZ"]);
        if (node["brightness"]) brightness.explodeParams(node["brightness"]);

        minMajorRadius = node["minMajorRadius"].as<double>();
        maxMajorRadius = node["maxMajorRadius"].as<double>();
        minMinorRadius = node["minMinorRadius"].as<double>();
        maxMinorRadius = node["maxMinorRadius"].as<double>();
        if (node["minBrightness"]) minBrightness = node["minBrightness"].as<double>();
        if (node["maxBrightness"]) maxBrightness = node["maxBrightness"].as<double>();
        if (node["splitBrightestFraction"]) splitBrightestFraction = node["splitBrightestFraction"].as<float>();
    }
};

class BaseConfig {
public:
    std::string cellType;
    std::unique_ptr<SpheroidConfig> cell;
    SimulationConfig simulation;
    ProbabilityConfig prob;

    BaseConfig() = default;
    ~BaseConfig() = default;

    // Deep copy (unique_ptr requires explicit copy)
    BaseConfig(const BaseConfig& other)
        : cellType(other.cellType),
          cell(other.cell ? std::make_unique<SpheroidConfig>(*other.cell) : nullptr),
          simulation(other.simulation),
          prob(other.prob) {}

    BaseConfig& operator=(const BaseConfig& other) {
        if (this != &other) {
            cellType = other.cellType;
            cell = other.cell ? std::make_unique<SpheroidConfig>(*other.cell) : nullptr;
            simulation = other.simulation;
            prob = other.prob;
        }
        return *this;
    }

    // Move operations auto-generated correctly by unique_ptr
    BaseConfig(BaseConfig&&) noexcept = default;
    BaseConfig& operator=(BaseConfig&&) noexcept = default;

    void explodeConfig(const YAML::Node& node) {
        cellType = node["cellType"].as<std::string>();
        cell = std::make_unique<SpheroidConfig>();
        cell->explodeConfig(node["cell"]);
        simulation.explodeConfig(node["simulation"]);
        prob.explodeConfig(node["prob"]);
    }

    void printConfig() const {
        simulation.printConfig();
        prob.printConfig();
    }
};

#endif
