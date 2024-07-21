// ConfigTypes.hpp
#ifndef CONFIGTYPES_HPP
#define CONFIGTYPES_HPP
#include <vector>
#include <random>
#include <string>
#include "yaml-cpp/yaml.h"
#include <iostream>

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
    void printConfig() {
        std::cout << "Simulation Config\n";
        std::cout << "iterations_per_cell: " << iterations_per_cell << std::endl;
        std::cout << "background_color: " << background_color << std::endl;
        std::cout << "cell_color: " << cell_color << std::endl;
        std::cout << "padding: " << padding << std::endl;
        std::cout << "z_scaling: " << z_scaling << std::endl;
        std::cout << "blur_sigma: " << blur_sigma << std::endl;
        std::cout << "z_slices: " << z_slices << std::endl;
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
    void printConfig() {
        std::cout << "Probability Config\n";
        std::cout << "perturbation: " << perturbation << std::endl;
        std::cout << "split: " << split << std::endl;
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
    //Used with a CellConfig to add perturb parameters.
public:
    float prob;
    float mu;
    float sigma;
    void explodeParams(const YAML::Node& node) {
        prob = node["prob"].as<float>();
        mu = node["mu"].as<float>();
        sigma = node["sigma"].as<float>();
    }
    [[nodiscard]] float getPerturbOffset() const {
        std::random_device rd; // Obtain a random number from hardware
        std::mt19937 gen(rd()); // Seed the generator
        std::uniform_real_distribution<> dis(0.0, 1.0); // Distribution for probability

        if (dis(gen) < prob) {
            std::normal_distribution<> d(mu, sigma); // Distribution for the Gaussian
            return d(gen);
        } else {
            return mu;
        }
    }
};

class CellConfig {
    // A pure abstract base class for SphereConfig and BacilliConfig
public:
    // pure virtual function for exploding the configuration
    virtual void explodeConfig(const YAML::Node& node) = 0;
    virtual ~CellConfig() = default;
//    virtual CellConfig& operator=(const CellConfig& other) = 0;
};

class SphereConfig: public CellConfig {
public:
    PerturbParams x{};
    PerturbParams y{};
    PerturbParams z{};
    PerturbParams radius{};
    double minRadius{};
    double maxRadius{};
    ~SphereConfig() = default;

    void explodeConfig(const YAML::Node& node) override
    {
        x.explodeParams(node["x"]);
        y.explodeParams(node["y"]);
        z.explodeParams(node["z"]);
        radius.explodeParams(node["radius"]);
        minRadius = node["minRadius"].as<double>();
        maxRadius = node["maxRadius"].as<double>();
    }
};

class BaseConfig {
public:
    std::string cellType;
    SphereConfig* cell;
    SimulationConfig simulation;
    ProbabilityConfig prob;
    BaseConfig() :cell(nullptr) {};
    BaseConfig& operator=(const BaseConfig& other) {
        if (this != &other) {
            if (other.cell != nullptr) {
                cell = new SphereConfig(*other.cell);
            }
            else {
                cell = nullptr;
            }
            simulation = other.simulation;
            prob = other.prob;
        }
        return *this;
    }
    BaseConfig(const BaseConfig& other) {
        if (other.cell != nullptr) {
            cell = new SphereConfig(*other.cell);
        }
        else {
            cell = nullptr;
        }
        simulation = other.simulation;
        prob = other.prob;
    }
    ~BaseConfig() {
        delete cell;
    }
    // load the BaseConfig with a YAML node
    void explodeConfig(const YAML::Node& node) {
        cellType = node["cellType"].as<std::string>();
        // TODO: Now cell is always pointing to a sphere config, make it more dynamic later
        cell = new SphereConfig;
        cell->explodeConfig(node["cell"]);
        simulation.explodeConfig(node["simulation"]);
        prob.explodeConfig(node["prob"]);
    }
    // for debug
    void printConfig() {
        simulation.printConfig();
        prob.printConfig();
    }
};

#endif
