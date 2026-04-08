// ConfigTypes.hpp
#ifndef CONFIGTYPES_HPP
#define CONFIGTYPES_HPP
#include <vector>
#include <random>
#include <string>
#include <memory>
#include <algorithm>
#include <cmath>
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
    float sigmoid_center_percentile = 0.4f;
    float sigmoid_center_offset = 0.047f;
    float post_sigmoid_dimmest_percentile = 0.45f;
    float post_sigmoid_dimmest_transition_width = 0.02f;
    float post_sigmoid_dimmest_transition_gradient = 4.0f;
    float adaptive_background_expand_factor = 1.1f;
    float adaptive_background_top_fraction = 0.4f;
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
        if (node["sigmoid_center_percentile"]) sigmoid_center_percentile = node["sigmoid_center_percentile"].as<float>();
        if (node["sigmoid_center_offset"]) sigmoid_center_offset = node["sigmoid_center_offset"].as<float>();
        if (node["post_sigmoid_dimmest_percentile"]) post_sigmoid_dimmest_percentile = node["post_sigmoid_dimmest_percentile"].as<float>();
        if (node["post_sigmoid_dimmest_transition_width"]) post_sigmoid_dimmest_transition_width = node["post_sigmoid_dimmest_transition_width"].as<float>();
        if (node["post_sigmoid_dimmest_transition_gradient"]) post_sigmoid_dimmest_transition_gradient = node["post_sigmoid_dimmest_transition_gradient"].as<float>();
        if (node["adaptive_background_expand_factor"]) adaptive_background_expand_factor = node["adaptive_background_expand_factor"].as<float>();
        if (node["adaptive_background_top_fraction"]) adaptive_background_top_fraction = node["adaptive_background_top_fraction"].as<float>();
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
        std::cout << "sigmoid_center_percentile: " << sigmoid_center_percentile << '\n';
        std::cout << "post_sigmoid_dimmest_percentile: " << post_sigmoid_dimmest_percentile << '\n';
        std::cout << "post_sigmoid_dimmest_transition_width: " << post_sigmoid_dimmest_transition_width << '\n';
        std::cout << "post_sigmoid_dimmest_transition_gradient: " << post_sigmoid_dimmest_transition_gradient << '\n';
        std::cout << "adaptive_background_expand_factor: " << adaptive_background_expand_factor << '\n';
        std::cout << "adaptive_background_top_fraction: " << adaptive_background_top_fraction << '\n';
        std::cout << "z_slices: " << z_slices << std::endl;
    }
};

class ProbabilityConfig {
public:
    float split;
    float split_cost;
    float split_elongation_threshold;
    float overlap_penalty_weight;
    float size_reduction_penalty_weight;
    float split_fake_overlap_volume_fraction_threshold;
    float split_fake_volume_ratio_threshold;
    float split_minor_axis_alignment_tolerance_degrees;
    float split_minor_axis_alignment_flatness_ratio_threshold;
    float split_minor_axis_alignment_min_radius_disable_threshold;
    float split_search_radius_multiplier;
    float split_fake_bridge_brightness_similarity_threshold;
    int split_burn_in_iterations = 500;
    float max_split_probability = 0.5f;
    float split_pre_burn_in_min_separation_over_major = 0.35f;
    float split_pre_burn_in_z_axis_max_abs = 0.92f;
    float split_pre_burn_in_z_axis_max_separation_over_major = 1.30f;
    float split_pre_burn_in_z_axis_min_drift_over_major = 0.40f;
    float split_post_burn_in_large_recenter_min_drift_over_major = 0.85f;
    float split_post_burn_in_large_recenter_max_cost_diff = -40.0f;
    ProbabilityConfig() : split(0.0f), split_cost(0.0f),
                          split_elongation_threshold(1.3f), overlap_penalty_weight(1000.0f),
                          size_reduction_penalty_weight(0.0f),
                          split_fake_overlap_volume_fraction_threshold(0.30f),
                          split_fake_volume_ratio_threshold(2.0f),
                          split_minor_axis_alignment_tolerance_degrees(20.0f),
                          split_minor_axis_alignment_flatness_ratio_threshold(0.5f),
                          split_minor_axis_alignment_min_radius_disable_threshold(0.0f),
                          split_search_radius_multiplier(3.0f),
                          split_fake_bridge_brightness_similarity_threshold(0.9f) {
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
        if (node["size_reduction_penalty_weight"]) {
            size_reduction_penalty_weight = node["size_reduction_penalty_weight"].as<float>();
        }
        if (node["split_fake_overlap_volume_fraction_threshold"]) {
            split_fake_overlap_volume_fraction_threshold =
                node["split_fake_overlap_volume_fraction_threshold"].as<float>();
        }
        if (node["split_fake_volume_ratio_threshold"]) {
            split_fake_volume_ratio_threshold =
                node["split_fake_volume_ratio_threshold"].as<float>();
        } else if (node["split_fake_radius_ratio_threshold"]) {
            split_fake_volume_ratio_threshold =
                node["split_fake_radius_ratio_threshold"].as<float>();
        }
        if (node["split_minor_axis_alignment_tolerance_degrees"]) {
            split_minor_axis_alignment_tolerance_degrees =
                node["split_minor_axis_alignment_tolerance_degrees"].as<float>();
        }
        if (node["split_minor_axis_alignment_flatness_ratio_threshold"]) {
            split_minor_axis_alignment_flatness_ratio_threshold =
                node["split_minor_axis_alignment_flatness_ratio_threshold"].as<float>();
        }
        if (node["split_minor_axis_alignment_min_radius_disable_threshold"]) {
            split_minor_axis_alignment_min_radius_disable_threshold =
                node["split_minor_axis_alignment_min_radius_disable_threshold"].as<float>();
        }
        if (node["split_search_radius_multiplier"]) {
            split_search_radius_multiplier = node["split_search_radius_multiplier"].as<float>();
        }
        if (node["split_fake_bridge_brightness_similarity_threshold"]) {
            split_fake_bridge_brightness_similarity_threshold =
                node["split_fake_bridge_brightness_similarity_threshold"].as<float>();
        }
        if (node["split_burn_in_iterations"]) {
            split_burn_in_iterations = node["split_burn_in_iterations"].as<int>();
        }
        if (node["max_split_probability"]) {
            max_split_probability = node["max_split_probability"].as<float>();
        }
        if (node["split_pre_burn_in_min_separation_over_major"]) {
            split_pre_burn_in_min_separation_over_major =
                node["split_pre_burn_in_min_separation_over_major"].as<float>();
        }
        if (node["split_pre_burn_in_z_axis_max_abs"]) {
            split_pre_burn_in_z_axis_max_abs =
                node["split_pre_burn_in_z_axis_max_abs"].as<float>();
        }
        if (node["split_pre_burn_in_z_axis_max_separation_over_major"]) {
            split_pre_burn_in_z_axis_max_separation_over_major =
                node["split_pre_burn_in_z_axis_max_separation_over_major"].as<float>();
        }
        if (node["split_pre_burn_in_z_axis_min_drift_over_major"]) {
            split_pre_burn_in_z_axis_min_drift_over_major =
                node["split_pre_burn_in_z_axis_min_drift_over_major"].as<float>();
        }
        if (node["split_post_burn_in_large_recenter_min_drift_over_major"]) {
            split_post_burn_in_large_recenter_min_drift_over_major =
                node["split_post_burn_in_large_recenter_min_drift_over_major"].as<float>();
        }
        if (node["split_post_burn_in_large_recenter_max_cost_diff"]) {
            split_post_burn_in_large_recenter_max_cost_diff =
                node["split_post_burn_in_large_recenter_max_cost_diff"].as<float>();
        }
    }
    void printConfig() const {
        std::cout << "Probability Config\n";
        std::cout << "split: " << split << '\n';
        std::cout << "split_cost: " << split_cost << '\n';
        std::cout << "split_elongation_threshold: " << split_elongation_threshold << '\n';
        std::cout << "overlap_penalty_weight: " << overlap_penalty_weight << '\n';
        std::cout << "size_reduction_penalty_weight: " << size_reduction_penalty_weight << '\n';
        std::cout << "split_fake_overlap_volume_fraction_threshold: "
                  << split_fake_overlap_volume_fraction_threshold << '\n';
        std::cout << "split_fake_volume_ratio_threshold: "
                  << split_fake_volume_ratio_threshold << '\n';
        std::cout << "split_minor_axis_alignment_tolerance_degrees: "
                  << split_minor_axis_alignment_tolerance_degrees << '\n';
        std::cout << "split_minor_axis_alignment_flatness_ratio_threshold: "
                  << split_minor_axis_alignment_flatness_ratio_threshold << '\n';
        std::cout << "split_minor_axis_alignment_min_radius_disable_threshold: "
                  << split_minor_axis_alignment_min_radius_disable_threshold << '\n';
        std::cout << "split_search_radius_multiplier: "
                  << split_search_radius_multiplier << '\n';
        std::cout << "split_fake_bridge_brightness_similarity_threshold: "
                  << split_fake_bridge_brightness_similarity_threshold << '\n';
        std::cout << "split_pre_burn_in_min_separation_over_major: "
                  << split_pre_burn_in_min_separation_over_major << '\n';
        std::cout << "split_pre_burn_in_z_axis_max_abs: "
                  << split_pre_burn_in_z_axis_max_abs << '\n';
        std::cout << "split_pre_burn_in_z_axis_max_separation_over_major: "
                  << split_pre_burn_in_z_axis_max_separation_over_major << '\n';
        std::cout << "split_pre_burn_in_z_axis_min_drift_over_major: "
                  << split_pre_burn_in_z_axis_min_drift_over_major << '\n';
        std::cout << "split_post_burn_in_large_recenter_min_drift_over_major: "
                  << split_post_burn_in_large_recenter_min_drift_over_major << '\n';
        std::cout << "split_post_burn_in_large_recenter_max_cost_diff: "
                  << split_post_burn_in_large_recenter_max_cost_diff << std::endl;
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
    float increase_prob = -1.0f;
    float decrease_prob = -1.0f;
    float mu    = 0.0f;
    float sigma = 0.0f;
    void explodeParams(const YAML::Node& node) {
        if (node["prob"]) prob = node["prob"].as<float>();
        if (node["increase_prob"]) increase_prob = node["increase_prob"].as<float>();
        if (node["decrease_prob"]) decrease_prob = node["decrease_prob"].as<float>();
        mu = node["mu"].as<float>();
        sigma = node["sigma"].as<float>();
    }
    [[nodiscard]] float getPerturbOffset() const {
        thread_local std::mt19937 gen{std::random_device{}()};
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        const bool hasSeparateSignProbabilities = increase_prob >= 0.0f || decrease_prob >= 0.0f;
        if (!hasSeparateSignProbabilities) {
            if (dis(gen) < prob) {
                std::normal_distribution<float> d(mu, sigma);
                return d(gen);
            }
            return mu;
        }

        const float incProb = std::clamp(increase_prob >= 0.0f ? increase_prob : 0.0f, 0.0f, 1.0f);
        const float decProb = std::clamp(decrease_prob >= 0.0f ? decrease_prob : 0.0f, 0.0f, 1.0f);
        const float roll = dis(gen);
        const float magnitude = (sigma > 0.0f)
            ? std::abs(std::normal_distribution<float>(mu, sigma)(gen))
            : std::abs(mu);

        if (roll < incProb) {
            return magnitude;
        }
        if (roll < incProb + decProb) {
            return -magnitude;
        }
        return 0.0f;
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
    float brightnessUpdateBlend{0.2f};
    float brightnessMeanAmplification{1.0f};
    float volumeRecoveryLossFractionThreshold{0.4f};
    float volumeRecoveryMaxScaleIncreaseFraction{0.3f};
    bool flatCellRotationRefineEnabled{true};
    float flatCellRotationRefineFlatnessThreshold{0.8f};
    float flatCellRotationRefineAngleStep{0.15f};
    float flatCellRotationRefineMaxOffsetDegrees{15.0f};
    int flatCellRotationRefinePasses{2};
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
        if (node["brightnessUpdateBlend"]) brightnessUpdateBlend = node["brightnessUpdateBlend"].as<float>();
        if (node["brightnessMeanAmplification"]) brightnessMeanAmplification = node["brightnessMeanAmplification"].as<float>();
        if (node["volumeRecoveryLossFractionThreshold"]) {
            volumeRecoveryLossFractionThreshold = node["volumeRecoveryLossFractionThreshold"].as<float>();
        }
        if (node["volumeRecoveryMaxScaleIncreaseFraction"]) {
            volumeRecoveryMaxScaleIncreaseFraction =
                node["volumeRecoveryMaxScaleIncreaseFraction"].as<float>();
        }
        if (node["flatCellRotationRefineEnabled"]) {
            flatCellRotationRefineEnabled =
                node["flatCellRotationRefineEnabled"].as<bool>();
        }
        if (node["flatCellRotationRefineFlatnessThreshold"]) {
            flatCellRotationRefineFlatnessThreshold =
                node["flatCellRotationRefineFlatnessThreshold"].as<float>();
        }
        if (node["flatCellRotationRefineAngleStep"]) {
            flatCellRotationRefineAngleStep = node["flatCellRotationRefineAngleStep"].as<float>();
        }
        if (node["flatCellRotationRefineMaxOffsetDegrees"]) {
            flatCellRotationRefineMaxOffsetDegrees =
                node["flatCellRotationRefineMaxOffsetDegrees"].as<float>();
        }
        if (node["flatCellRotationRefinePasses"]) {
            flatCellRotationRefinePasses = node["flatCellRotationRefinePasses"].as<int>();
        }
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
