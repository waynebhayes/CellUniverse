// ConfigTypes.hpp
#ifndef CONFIGTYPES_HPP
#define CONFIGTYPES_HPP
#include <vector>
#include <random>
#include <string>
#include <memory>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <filesystem>
#include "yaml-cpp/yaml.h"
#include <iostream>

inline std::mt19937 &cellUniverseRandomGenerator()
{
    static thread_local std::mt19937 gen([] {
        const char *seedEnv = std::getenv("CELLUNIVERSE_SEED");
        if (seedEnv != nullptr && seedEnv[0] != '\0') {
            char *end = nullptr;
            const unsigned long seed = std::strtoul(seedEnv, &end, 10);
            if (end != seedEnv && *end == '\0') {
                return std::mt19937(static_cast<std::mt19937::result_type>(seed));
            }
        }
        return std::mt19937(std::random_device{}());
    }());
    return gen;
}

namespace CellUniverseConfig {

inline YAML::Node mergeYamlNodes(const YAML::Node &base, const YAML::Node &overrides)
{
    if (!base || !base.IsMap() || !overrides || !overrides.IsMap()) {
        return overrides ? YAML::Clone(overrides) : YAML::Clone(base);
    }

    YAML::Node merged = YAML::Clone(base);
    for (const auto &entry : overrides) {
        const std::string key = entry.first.as<std::string>();
        if (key == "base_config") {
            continue;
        }

        if (merged[key] && merged[key].IsMap() && entry.second.IsMap()) {
            merged[key] = mergeYamlNodes(merged[key], entry.second);
        } else {
            merged[key] = YAML::Clone(entry.second);
        }
    }
    return merged;
}

inline YAML::Node loadConfigYamlNode(const std::string &path)
{
    const YAML::Node node = YAML::LoadFile(path);
    YAML::Node loaded;
    if (!node["base_config"]) {
        loaded = node;
    } else {
        std::filesystem::path basePath(node["base_config"].as<std::string>());
        if (basePath.is_relative()) {
            basePath = std::filesystem::path(path).parent_path() / basePath;
        }

        loaded = mergeYamlNodes(loadConfigYamlNode(basePath.string()), node);
    }

    if (!loaded["unified_density_profile_yaml"] || !loaded["profiles"]) {
        return loaded;
    }

    const YAML::Node meta = loaded["unified_density_profile_yaml"];
    const YAML::Node profiles = loaded["profiles"];
    if (!profiles || !profiles.IsMap()) {
        return loaded;
    }

    std::string defaultProfile;
    if (meta["default_profile"]) {
        defaultProfile = meta["default_profile"].as<std::string>();
    } else {
        for (const auto &entry : profiles) {
            defaultProfile = entry.first.as<std::string>();
            break;
        }
    }
    if (defaultProfile.empty() || !profiles[defaultProfile] ||
        !profiles[defaultProfile]["expanded_config"]) {
        throw std::runtime_error(
            "Unified density profile YAML has no usable default profile.");
    }

    YAML::Node runtime = YAML::Clone(profiles[defaultProfile]["expanded_config"]);
    YAML::Node runtimeProfiles;
    runtimeProfiles["enabled"] =
        meta["runtime_density_profile_selection_enabled"]
            ? meta["runtime_density_profile_selection_enabled"].as<bool>()
            : false;
    runtimeProfiles["metric"] =
        meta["density_profile_selection_metric"]
            ? meta["density_profile_selection_metric"].as<std::string>()
            : std::string("median_nearest_neighbor_px");
    runtimeProfiles["default_profile"] = defaultProfile;
    runtimeProfiles["active_profile"] = defaultProfile;

    YAML::Node rulesByProfile;
    const bool hasDensitySelectionRules =
        meta["density_profile_selection"] &&
        meta["density_profile_selection"].IsSequence();
    if (hasDensitySelectionRules) {
        for (const auto &rule : meta["density_profile_selection"]) {
            if (!rule["profile"]) {
                continue;
            }
            const std::string name = rule["profile"].as<std::string>();
            rulesByProfile[name] = YAML::Clone(rule);
        }
    }

    YAML::Node profileList(YAML::NodeType::Sequence);
    for (const auto &entry : profiles) {
        const std::string name = entry.first.as<std::string>();
        if (!entry.second["expanded_config"]) {
            continue;
        }
        if (hasDensitySelectionRules && !rulesByProfile[name]) {
            continue;
        }
        YAML::Node profile;
        profile["name"] = name;
        profile["expanded_config"] = YAML::Clone(entry.second["expanded_config"]);
        if (rulesByProfile[name]) {
            profile["min_median_nearest_neighbor_px"] =
                rulesByProfile[name]["min_median_nearest_neighbor_px"]
                    ? rulesByProfile[name]["min_median_nearest_neighbor_px"].as<float>()
                    : -1000000000.0f;
            profile["max_median_nearest_neighbor_px"] =
                rulesByProfile[name]["max_median_nearest_neighbor_px"]
                    ? rulesByProfile[name]["max_median_nearest_neighbor_px"].as<float>()
                    : 1000000000.0f;
        }
        profileList.push_back(profile);
    }
    runtimeProfiles["profiles"] = profileList;
    runtime["runtime_density_profiles"] = runtimeProfiles;
    return runtime;
}

} // namespace CellUniverseConfig

class SimulationConfig {
public:
    int iterations_per_cell;
    int signal_guided_iterations_per_cell = -1;
    int random_iterations_per_cell = -1;
    float z_scaling;
    float blur_sigma;
    int z_slices;
    // Preprocessing mode:
    // iterative = existing contrast optimization plus post alignment cleanup.
    // light = percentile normalization, optional gamma, and z interpolation.
    // none = percentile normalization and z interpolation only.
    std::string preprocess_mode = "iterative";
    float light_preprocess_gamma = 1.0f;
    float iterative_penalty = 0.1f;
    float iterative_penalty_range = 0.9f;
    float iterative_reward_gate = 1.0f;
    float iterative_reward_gate_step = 0.008f;
    float iterative_reward_gate_min = 0.025f;
    float iterative_reward = 0.1f;
    float iterative_reward_gate_learning_rate = 0.05f;
    float iterative_reward_gate_min_probability = 0.001f;
    int iterative_no_improvement_patience = 50;
    float iterative_score_explosion_threshold = 5.0f;
    float iterative_candidate_max_mean = 0.40f;
    float iterative_candidate_max_saturated_fraction = 0.02f;
    int iterative_reward_gate_seed = 12345;
    float iterative_score_max = 1.5f;
    float iterative_score_target_tolerance = 0.05f;
    int iterative_max_count = 300;
    float iterative_improvement_tolerance = 0.01f;
    float iterative_score_percentile = 0.05f;
    float contrast_structure_threshold = 0.02f;
    float contrast_background_floor = 0.02f;
    float contrast_bright_fraction = 0.02f;
    float contrast_brightness_weight = 0.25f;
    float contrast_window_radius_step = 5.0f;
    float contrast_penalty_min_radius_scale = 2.0f;
    float contrast_reward_weight = 1.0f;
    float contrast_penalty_weight = 1.0f;
    float contrast_eps = 1e-6f;
    bool frame_intensity_normalization_enabled = true;
    bool frame_intensity_percentile_exclude_zeros = false;
    float frame_intensity_scale_low_percentile = 0.01f;
    float frame_intensity_scale_high_percentile = 0.995f;
    float frame_intensity_hard_max = 0.0f;
    bool edge_brightness_alignment_enabled = false;
    int edge_brightness_alignment_xy_margin = 24;
    int edge_brightness_alignment_left_offset = 0;
    int edge_brightness_alignment_right_offset = 0;
    int edge_brightness_alignment_top_offset = 0;
    int edge_brightness_alignment_bottom_offset = 0;
    float edge_brightness_alignment_max_shift = 0.20f;
    float post_alignment_black_threshold = 0.05f;
    float post_alignment_black_percentile = 0.30f;
    bool post_alignment_chunk_blackoff_enabled = true;
    int post_alignment_chunk_target_count = 30;
    int post_alignment_chunk_min_size = 5;
    int post_alignment_chunk_max_size = 10;
    float post_alignment_chunk_percentile_step = 0.001f;
    float post_alignment_chunk_max_percentile = 0.999f;
    int post_alignment_chunk_non_improvement_patience = 10;
    int post_alignment_chunk_disable_below_count = 0;
    int post_alignment_chunk_detector_threads = 4;
    bool post_alignment_tiny_particle_removal_enabled = false;
    int preprocess_radius_batch_size = 5;
    float post_process_blur_sigma = 2.5f;
    float post_process_final_blur_sigma = 0.0f;
    float post_process_final_direct_weight = 1.0f;
    float post_process_final_direct_amplification = 1.0f;
    float post_process_final_blurred_amplification = 1.0f;
    float post_alignment_final_blur_sigma = 0.0f;
    bool export_preprocessed_images = false;
    bool export_signal_debug_images = false;
    bool export_perturb_debug_images = false;
    bool export_perturb_cell_center_debug_images = false;
    bool export_frame_png = true;
    bool export_frame_tiff = false;
    bool quit_after_preprocessing = false;
    bool enable_lineage_tree_window = false;
    bool prepare_analyze_one_frame = false;
    bool release_analyzed_exported_frames = true;
    bool perturb_oscillation_boost_cells_enabled = false;
    bool perturb_oscillation_boost_trash_enabled = false;
    float perturb_oscillation_warmup_fraction = 0.30f;
    int perturb_oscillation_window = 12;
    int perturb_oscillation_min_samples = 8;
    float perturb_oscillation_sign_change_ratio = 0.60f;
    float perturb_oscillation_boost_ratio = 1.50f;
    float perturb_oscillation_max_multiplier = 4.0f;
    bool perturb_oscillation_reset_on_accept = true;
    float perturb_oscillation_small_step_probability = 0.20f;
    float perturb_oscillation_small_step_multiplier = 1.0f;
    bool perturb_oscillation_black_pixel_boost_enabled = false;
    float perturb_oscillation_black_pixel_boost_weight = 1.0f;
    float perturb_debug_cell_brightness = 0.30f;
    int perturb_debug_center_cube_radius = 2;

    // Checkpoint resume (Approach 2): skip frames 0..resume_from-1 and
    // load state from {resume_source_dir}/checkpoints/frame_{resume_from-1:03d}.txt.
    // Set resume_from=0 or -1 to disable. resume_source_dir is the output
    // folder of the run being resumed FROM.
    int resume_from = 0;
    std::string resume_source_dir = "";
    bool cube_pooling_enabled = false;
    bool cube_pooling_cost_comparison_enabled = true;
    int cube_pooling_cube_size = 5;
    std::string cube_pooling_mode = "mean";
    float cube_pooling_top_fraction = 0.10f;
    float cube_pooling_low_fraction = 0.10f;
    float adaptive_background_expand_factor = 1.1f;
    float adaptive_background_top_fraction = 0.4f;
    bool signal_guided_position_enabled = false;
    int signal_guided_box_side_length = 5;
    float signal_guided_min_box_brightness_delta = 0.0f;
    float signal_guided_min_sigma_scale = 0.35f;
    float signal_guided_sigma_range_multiplier = 1.0f;
    float signal_guided_initial_bright_fraction = 0.70f;
    float signal_guided_recursive_bright_fraction = 0.50f;
    int signal_guided_max_recursive_depth = 2;
    bool signal_map_enabled = true;
    float signal_map_blur_sigma = 2.5f;
    int signal_map_max_iterations = 15;
    float signal_map_bright_center_percentile = 0.0f;
    float signal_map_epsilon = 1e-6f;
    bool signal_map_perturb_guidance_enabled = true;
    float signal_map_cell_radius_scale = 1.2f;
    float signal_map_min_gradient_norm = 1e-6f;
    float signal_map_direction_probability_boost = 0.5f;
    float signal_map_opposing_move_damping = 0.0f;
    float signal_map_guide_strength = 0.0f;

    // Asymmetric L2 cost weight (Fix E). Per-voxel squared error is
    // multiplied by this factor when synth > real (cell covers darker
    // image region). When synth < real (cell undershoots bright region),
    // the unweighted squared error is used. 1.0 = symmetric L2 (disabled).
    // Typical: 2.0 - 4.0. Makes "parent covering dark valley between two
    // daughters" always cost more than "two daughters covering bright
    // blobs," so split-vs-no-split cost comparison is deterministic on
    // bright-coverage geometry.
    float asymmetric_cost_weight = 1.0f;
    // Minimum overshoot (synth - real) to trigger asymK amplification.
    // Below this threshold, overshoot is penalized at 1x (symmetric) —
    // treats small boundary rendering artifacts as noise. Above, the
    // full asymK applies. Eliminates double-boundary bias where two
    // daughters' boundary artifacts cost more than one parent's.
    float asymmetric_cost_threshold = 0.03f;

    // Static-Voronoi cost territory (2026-04-21). When true, each pixel in
    // a cell's bbox contributes to that cell's image cost only if the pixel
    // is closer to THIS cell's snap-anchor than to any other cell's snap-
    // anchor. Anchors are captured once per frame from previousSnapshots
    // (snap position, held fixed for the whole frame), so the Voronoi
    // boundary does NOT shift with the perturbed cell — unlike the earlier
    // live-center Voronoi attempt, this preserves the snap-anchor drift
    // penalty because a cell's own claim-region stays put even when the
    // cell moves. Prevents bloat-induced territory annexation (a511-case):
    // growing cell X to cover pixels assigned to neighbor Y gives zero
    // image-cost improvement, so there is no gradient toward bloat.
    // Rebuilt at frame start and after every split accept.
    //
    // DEFAULT OFF (2026-04-21): when enabled, cells deform to match their
    // Voronoi territory geometry — a cell surrounded by neighbors sits in a
    // polygonal wedge, and with its cost restricted to that wedge it
    // elongates to match. High elongation triggers early false-positive
    // splits (observed: e3d03 FP at f4 on 2026-04-21 with factor=static).
    // Keep disabled until the replacement "additive bleed penalty" design
    // lands — that version adds a penalty for ellipsoid volume outside own
    // territory WITHOUT replacing image cost, so shape fitting is preserved
    // and bloat is still suppressed.
    bool voronoi_cost_enabled = false;

    // Additive Voronoi bleed penalty (2026-04-21 — the fundamental bloat
    // fix). For each perturbation, we add:
    //   penalty = voronoi_bleed_penalty_weight × (count of voxels inside
    //             this cell's ellipsoid that are assigned to ANOTHER cell
    //             in the snap-anchored Voronoi map)
    // to the cost. This is ADDITIVE — it does NOT replace image cost, so
    // the normal brightness-fitting gradient is intact. But growing into
    // a neighbor's territory costs extra (proportional to the volume of
    // intrusion), so the Monte Carlo optimizer no longer has a free lunch
    // by annexing neighbor pixels.
    //
    // Why additive not replacement: the earlier replacement-cost approach
    // (voronoi_cost_enabled above) distorted cell shapes to match polygonal
    // territories → premature false splits. Additive only adds cost where
    // intrusion happens; cells still fit their own bright regions normally.
    //
    // Three pathologies directly addressed:
    //   (1) Bloat / neighbor-particle capture — expansion into neighbor
    //       territory incurs roughly quadratic growth in bleed.
    //   (2) False-positive splits — cells that would have bloated to
    //       trigger premature split-attempts now stay at normal size.
    //   (3) Cascading missed splits — FP daughters stop appearing, so
    //       they no longer block neighbor split attempts via the bio-gate
    //       "buried_in" check.
    //
    // Weight tuning: bleed counts are O(10k) for a moderately-bloating
    // cell (radius 40 in xy, 20 in z). With weight 0.5 that's 5k cost —
    // comparable to asymmetric L2 image cost per voxel times a few
    // percent of the cell volume. Tune up if bloat slips through; down
    // if legitimate growth is resisted.
    bool voronoi_bleed_penalty_enabled = true;
    float voronoi_bleed_penalty_weight = 0.5f;

    // Runtime parallelism for independent z-slice work. Keep the YAML default
    // conservative for laptop runs; OpenLab jobs can override it with
    // CELLUNIVERSE_THREADS after Slurm grants the requested CPU allocation.
    int parallel_threads = 1;
    int parallel_min_slices = 8;

    // How initial CSV z coordinates should be interpreted.
    // auto: resume CSVs with theta columns are already in optimizer z space;
    // raw: CSV z is raw TIFF/GT slice index and must be multiplied by z_scaling;
    // scaled: CSV z is already in interpolated optimizer z space.
    std::string initial_z_space = "auto";

    // Constructor with default values
    SimulationConfig() : iterations_per_cell(0),
                         z_scaling(1.0), blur_sigma(0.0f), z_slices(-1) {
    }
    void explodeConfig(const YAML::Node& node) {
        iterations_per_cell = node["iterations_per_cell"].as<int>();
        if (node["signal_guided_iterations_per_cell"]) signal_guided_iterations_per_cell = node["signal_guided_iterations_per_cell"].as<int>();
        if (node["random_iterations_per_cell"]) random_iterations_per_cell = node["random_iterations_per_cell"].as<int>();
        z_scaling = node["z_scaling"].as<float>();
        blur_sigma = node["blur_sigma"].as<float>();
        if (node["preprocess_mode"]) preprocess_mode = node["preprocess_mode"].as<std::string>();
        if (node["light_preprocess_gamma"]) light_preprocess_gamma = node["light_preprocess_gamma"].as<float>();
        if (node["iterative_penalty"]) iterative_penalty = node["iterative_penalty"].as<float>();
        if (node["iterative_penalty_range"]) iterative_penalty_range = node["iterative_penalty_range"].as<float>();
        if (node["iterative_reward_gate"]) iterative_reward_gate = node["iterative_reward_gate"].as<float>();
        if (node["iterative_reward_gate_step"]) iterative_reward_gate_step = node["iterative_reward_gate_step"].as<float>();
        if (node["iterative_reward_gate_min"]) iterative_reward_gate_min = node["iterative_reward_gate_min"].as<float>();
        if (node["iterative_reward"]) iterative_reward = node["iterative_reward"].as<float>();
        if (node["iterative_reward_gate_learning_rate"]) iterative_reward_gate_learning_rate = node["iterative_reward_gate_learning_rate"].as<float>();
        if (node["iterative_reward_gate_min_probability"]) iterative_reward_gate_min_probability = node["iterative_reward_gate_min_probability"].as<float>();
        if (node["iterative_no_improvement_patience"]) iterative_no_improvement_patience = node["iterative_no_improvement_patience"].as<int>();
        if (node["iterative_score_explosion_threshold"]) iterative_score_explosion_threshold = node["iterative_score_explosion_threshold"].as<float>();
        if (node["iterative_candidate_max_mean"]) iterative_candidate_max_mean = node["iterative_candidate_max_mean"].as<float>();
        if (node["iterative_candidate_max_saturated_fraction"]) iterative_candidate_max_saturated_fraction = node["iterative_candidate_max_saturated_fraction"].as<float>();
        if (node["iterative_reward_gate_seed"]) iterative_reward_gate_seed = node["iterative_reward_gate_seed"].as<int>();
        if (node["iterative_score_max"]) iterative_score_max = node["iterative_score_max"].as<float>();
        if (node["iterative_score_target_tolerance"]) iterative_score_target_tolerance = node["iterative_score_target_tolerance"].as<float>();
        if (node["iterative_max_count"]) iterative_max_count = node["iterative_max_count"].as<int>();
        if (node["iterative_improvement_tolerance"]) iterative_improvement_tolerance = node["iterative_improvement_tolerance"].as<float>();
        if (node["iterative_score_percentile"]) iterative_score_percentile = node["iterative_score_percentile"].as<float>();
        if (node["contrast_structure_threshold"]) contrast_structure_threshold = node["contrast_structure_threshold"].as<float>();
        if (node["contrast_background_floor"]) contrast_background_floor = node["contrast_background_floor"].as<float>();
        if (node["contrast_bright_fraction"]) contrast_bright_fraction = node["contrast_bright_fraction"].as<float>();
        if (node["contrast_brightness_weight"]) contrast_brightness_weight = node["contrast_brightness_weight"].as<float>();
        if (node["contrast_window_radius_step"]) contrast_window_radius_step = node["contrast_window_radius_step"].as<float>();
        if (node["contrast_penalty_min_radius_scale"]) contrast_penalty_min_radius_scale = node["contrast_penalty_min_radius_scale"].as<float>();
        if (node["contrast_reward_weight"]) contrast_reward_weight = node["contrast_reward_weight"].as<float>();
        if (node["contrast_penalty_weight"]) contrast_penalty_weight = node["contrast_penalty_weight"].as<float>();
        if (node["contrast_eps"]) contrast_eps = node["contrast_eps"].as<float>();
        if (node["frame_intensity_normalization_enabled"]) frame_intensity_normalization_enabled = node["frame_intensity_normalization_enabled"].as<bool>();
        if (node["frame_intensity_percentile_exclude_zeros"]) frame_intensity_percentile_exclude_zeros = node["frame_intensity_percentile_exclude_zeros"].as<bool>();
        if (node["frame_intensity_scale_low_percentile"]) frame_intensity_scale_low_percentile = node["frame_intensity_scale_low_percentile"].as<float>();
        if (node["frame_intensity_scale_high_percentile"]) frame_intensity_scale_high_percentile = node["frame_intensity_scale_high_percentile"].as<float>();
        if (node["frame_intensity_hard_max"]) frame_intensity_hard_max = node["frame_intensity_hard_max"].as<float>();
        if (node["edge_brightness_alignment_enabled"]) edge_brightness_alignment_enabled = node["edge_brightness_alignment_enabled"].as<bool>();
        if (node["edge_brightness_alignment_xy_margin"]) edge_brightness_alignment_xy_margin = node["edge_brightness_alignment_xy_margin"].as<int>();
        if (node["edge_brightness_alignment_left_offset"]) edge_brightness_alignment_left_offset = node["edge_brightness_alignment_left_offset"].as<int>();
        if (node["edge_brightness_alignment_right_offset"]) edge_brightness_alignment_right_offset = node["edge_brightness_alignment_right_offset"].as<int>();
        if (node["edge_brightness_alignment_top_offset"]) edge_brightness_alignment_top_offset = node["edge_brightness_alignment_top_offset"].as<int>();
        if (node["edge_brightness_alignment_bottom_offset"]) edge_brightness_alignment_bottom_offset = node["edge_brightness_alignment_bottom_offset"].as<int>();
        if (node["edge_brightness_alignment_max_shift"]) edge_brightness_alignment_max_shift = node["edge_brightness_alignment_max_shift"].as<float>();
        if (node["post_alignment_black_threshold"]) post_alignment_black_threshold = node["post_alignment_black_threshold"].as<float>();
        if (node["post_alignment_black_percentile"]) post_alignment_black_percentile = node["post_alignment_black_percentile"].as<float>();
        if (node["post_alignment_chunk_blackoff_enabled"]) post_alignment_chunk_blackoff_enabled = node["post_alignment_chunk_blackoff_enabled"].as<bool>();
        if (node["post_alignment_chunk_target_count"]) post_alignment_chunk_target_count = node["post_alignment_chunk_target_count"].as<int>();
        if (node["post_alignment_chunk_min_size"]) post_alignment_chunk_min_size = node["post_alignment_chunk_min_size"].as<int>();
        if (node["post_alignment_chunk_max_size"]) post_alignment_chunk_max_size = node["post_alignment_chunk_max_size"].as<int>();
        if (node["post_alignment_chunk_percentile_step"]) post_alignment_chunk_percentile_step = node["post_alignment_chunk_percentile_step"].as<float>();
        if (node["post_alignment_chunk_max_percentile"]) post_alignment_chunk_max_percentile = node["post_alignment_chunk_max_percentile"].as<float>();
        if (node["post_alignment_chunk_non_improvement_patience"]) post_alignment_chunk_non_improvement_patience = node["post_alignment_chunk_non_improvement_patience"].as<int>();
        if (node["post_alignment_chunk_disable_below_count"]) post_alignment_chunk_disable_below_count = node["post_alignment_chunk_disable_below_count"].as<int>();
        if (node["post_alignment_chunk_detector_threads"]) post_alignment_chunk_detector_threads = node["post_alignment_chunk_detector_threads"].as<int>();
        if (node["post_alignment_tiny_particle_removal_enabled"]) post_alignment_tiny_particle_removal_enabled = node["post_alignment_tiny_particle_removal_enabled"].as<bool>();
        if (node["preprocess_radius_batch_size"]) preprocess_radius_batch_size = node["preprocess_radius_batch_size"].as<int>();
        if (node["post_process_blur_sigma"]) post_process_blur_sigma = node["post_process_blur_sigma"].as<float>();
        if (node["post_process_final_blur_sigma"]) post_process_final_blur_sigma = node["post_process_final_blur_sigma"].as<float>();
        if (node["post_process_final_direct_weight"]) post_process_final_direct_weight = node["post_process_final_direct_weight"].as<float>();
        if (node["post_process_final_direct_amplification"]) post_process_final_direct_amplification = node["post_process_final_direct_amplification"].as<float>();
        if (node["post_process_final_blurred_amplification"]) post_process_final_blurred_amplification = node["post_process_final_blurred_amplification"].as<float>();
        if (node["post_alignment_final_blur_sigma"]) post_alignment_final_blur_sigma = node["post_alignment_final_blur_sigma"].as<float>();
        if (node["export_preprocessed_images"]) export_preprocessed_images = node["export_preprocessed_images"].as<bool>();
        if (node["export_signal_debug_images"]) export_signal_debug_images = node["export_signal_debug_images"].as<bool>();
        if (node["export_perturb_debug_images"]) export_perturb_debug_images = node["export_perturb_debug_images"].as<bool>();
        if (node["export_perturb_cell_center_debug_images"]) export_perturb_cell_center_debug_images = node["export_perturb_cell_center_debug_images"].as<bool>();
        if (node["export_frame_png"]) export_frame_png = node["export_frame_png"].as<bool>();
        if (node["export_frame_tiff"]) export_frame_tiff = node["export_frame_tiff"].as<bool>();
        if (node["quit_after_preprocessing"]) quit_after_preprocessing = node["quit_after_preprocessing"].as<bool>();
        if (node["enable_lineage_tree_window"]) enable_lineage_tree_window = node["enable_lineage_tree_window"].as<bool>();
        if (node["lineage_tree_window"]) enable_lineage_tree_window = node["lineage_tree_window"].as<bool>();
        if (node["prepare_analyze_one_frame"]) prepare_analyze_one_frame = node["prepare_analyze_one_frame"].as<bool>();
        if (node["release_analyzed_exported_frames"]) release_analyzed_exported_frames = node["release_analyzed_exported_frames"].as<bool>();
        if (node["perturb_oscillation_boost_cells_enabled"]) perturb_oscillation_boost_cells_enabled = node["perturb_oscillation_boost_cells_enabled"].as<bool>();
        if (node["perturb_oscillation_boost_trash_enabled"]) perturb_oscillation_boost_trash_enabled = node["perturb_oscillation_boost_trash_enabled"].as<bool>();
        if (node["perturb_oscillation_warmup_fraction"]) perturb_oscillation_warmup_fraction = node["perturb_oscillation_warmup_fraction"].as<float>();
        if (node["perturb_oscillation_window"]) perturb_oscillation_window = node["perturb_oscillation_window"].as<int>();
        if (node["perturb_oscillation_min_samples"]) perturb_oscillation_min_samples = node["perturb_oscillation_min_samples"].as<int>();
        if (node["perturb_oscillation_sign_change_ratio"]) perturb_oscillation_sign_change_ratio = node["perturb_oscillation_sign_change_ratio"].as<float>();
        if (node["perturb_oscillation_boost_ratio"]) perturb_oscillation_boost_ratio = node["perturb_oscillation_boost_ratio"].as<float>();
        if (node["perturb_oscillation_max_multiplier"]) perturb_oscillation_max_multiplier = node["perturb_oscillation_max_multiplier"].as<float>();
        if (node["perturb_oscillation_reset_on_accept"]) perturb_oscillation_reset_on_accept = node["perturb_oscillation_reset_on_accept"].as<bool>();
        if (node["perturb_oscillation_small_step_probability"]) perturb_oscillation_small_step_probability = node["perturb_oscillation_small_step_probability"].as<float>();
        if (node["perturb_oscillation_small_step_multiplier"]) perturb_oscillation_small_step_multiplier = node["perturb_oscillation_small_step_multiplier"].as<float>();
        if (node["perturb_oscillation_black_pixel_boost_enabled"]) perturb_oscillation_black_pixel_boost_enabled = node["perturb_oscillation_black_pixel_boost_enabled"].as<bool>();
        if (node["perturb_oscillation_black_pixel_boost_weight"]) perturb_oscillation_black_pixel_boost_weight = node["perturb_oscillation_black_pixel_boost_weight"].as<float>();
        if (node["perturb_debug_cell_brightness"]) perturb_debug_cell_brightness = node["perturb_debug_cell_brightness"].as<float>();
        if (node["perturb_debug_center_cube_radius"]) perturb_debug_center_cube_radius = node["perturb_debug_center_cube_radius"].as<int>();
        if (node["resume_from"]) resume_from = node["resume_from"].as<int>();
        if (node["resume_source_dir"]) resume_source_dir = node["resume_source_dir"].as<std::string>();
        if (node["cube_pooling_enabled"]) cube_pooling_enabled = node["cube_pooling_enabled"].as<bool>();
        if (node["cube_pooling_cost_comparison_enabled"]) cube_pooling_cost_comparison_enabled = node["cube_pooling_cost_comparison_enabled"].as<bool>();
        if (node["cube_pooling_cube_size"]) cube_pooling_cube_size = node["cube_pooling_cube_size"].as<int>();
        if (node["cube_pooling_mode"]) cube_pooling_mode = node["cube_pooling_mode"].as<std::string>();
        if (node["cube_pooling_top_fraction"]) cube_pooling_top_fraction = node["cube_pooling_top_fraction"].as<float>();
        if (node["cube_pooling_low_fraction"]) cube_pooling_low_fraction = node["cube_pooling_low_fraction"].as<float>();
        if (node["adaptive_background_expand_factor"]) adaptive_background_expand_factor = node["adaptive_background_expand_factor"].as<float>();
        if (node["adaptive_background_top_fraction"]) adaptive_background_top_fraction = node["adaptive_background_top_fraction"].as<float>();
        if (node["signal_guided_position_enabled"]) signal_guided_position_enabled = node["signal_guided_position_enabled"].as<bool>();
        if (node["signal_guided_box_side_length"]) signal_guided_box_side_length = node["signal_guided_box_side_length"].as<int>();
        if (node["signal_guided_min_box_brightness_delta"]) signal_guided_min_box_brightness_delta = node["signal_guided_min_box_brightness_delta"].as<float>();
        if (node["signal_guided_min_sigma_scale"]) signal_guided_min_sigma_scale = node["signal_guided_min_sigma_scale"].as<float>();
        if (node["signal_guided_sigma_range_multiplier"]) signal_guided_sigma_range_multiplier = node["signal_guided_sigma_range_multiplier"].as<float>();
        if (node["signal_guided_initial_bright_fraction"]) signal_guided_initial_bright_fraction = node["signal_guided_initial_bright_fraction"].as<float>();
        if (node["signal_guided_recursive_bright_fraction"]) signal_guided_recursive_bright_fraction = node["signal_guided_recursive_bright_fraction"].as<float>();
        if (node["signal_guided_max_recursive_depth"]) signal_guided_max_recursive_depth = node["signal_guided_max_recursive_depth"].as<int>();
        if (node["signal_map_enabled"]) signal_map_enabled = node["signal_map_enabled"].as<bool>();
        if (node["signal_map_blur_sigma"]) signal_map_blur_sigma = node["signal_map_blur_sigma"].as<float>();
        if (node["signal_map_max_iterations"]) signal_map_max_iterations = node["signal_map_max_iterations"].as<int>();
        if (node["signal_map_bright_center_percentile"]) signal_map_bright_center_percentile = node["signal_map_bright_center_percentile"].as<float>();
        if (node["signal_map_epsilon"]) signal_map_epsilon = node["signal_map_epsilon"].as<float>();
        if (node["signal_map_perturb_guidance_enabled"]) signal_map_perturb_guidance_enabled = node["signal_map_perturb_guidance_enabled"].as<bool>();
        if (node["signal_map_cell_radius_scale"]) signal_map_cell_radius_scale = node["signal_map_cell_radius_scale"].as<float>();
        if (node["signal_map_min_gradient_norm"]) signal_map_min_gradient_norm = node["signal_map_min_gradient_norm"].as<float>();
        if (node["signal_map_direction_probability_boost"]) signal_map_direction_probability_boost = node["signal_map_direction_probability_boost"].as<float>();
        if (node["signal_map_opposing_move_damping"]) signal_map_opposing_move_damping = node["signal_map_opposing_move_damping"].as<float>();
        if (node["signal_map_guide_strength"]) signal_map_guide_strength = node["signal_map_guide_strength"].as<float>();
        if (node["asymmetric_cost_weight"]) asymmetric_cost_weight = node["asymmetric_cost_weight"].as<float>();
        if (node["asymmetric_cost_threshold"]) asymmetric_cost_threshold = node["asymmetric_cost_threshold"].as<float>();
        if (node["voronoi_cost_enabled"]) voronoi_cost_enabled = node["voronoi_cost_enabled"].as<bool>();
        if (node["voronoi_bleed_penalty_enabled"]) voronoi_bleed_penalty_enabled = node["voronoi_bleed_penalty_enabled"].as<bool>();
        if (node["voronoi_bleed_penalty_weight"]) voronoi_bleed_penalty_weight = node["voronoi_bleed_penalty_weight"].as<float>();
        if (node["parallel_threads"]) parallel_threads = node["parallel_threads"].as<int>();
        if (node["parallel_min_slices"]) parallel_min_slices = node["parallel_min_slices"].as<int>();
        if (node["initial_z_space"]) initial_z_space = node["initial_z_space"].as<std::string>();
        parallel_threads = std::max(1, parallel_threads);
        parallel_min_slices = std::max(1, parallel_min_slices);
        if (initial_z_space != "auto" &&
            initial_z_space != "raw" &&
            initial_z_space != "scaled") {
            throw std::invalid_argument("simulation.initial_z_space must be one of: auto, raw, scaled");
        }
        if (preprocess_mode != "iterative" &&
            preprocess_mode != "light" &&
            preprocess_mode != "none") {
            throw std::invalid_argument("simulation.preprocess_mode must be one of: iterative, light, none");
        }
        light_preprocess_gamma = std::max(0.01f, light_preprocess_gamma);
    }
    void printConfig() const {
        std::cout << "Simulation Config\n";
        std::cout << "iterations_per_cell: " << iterations_per_cell << '\n';
        std::cout << "signal_guided_iterations_per_cell: " << signal_guided_iterations_per_cell << '\n';
        std::cout << "random_iterations_per_cell: " << random_iterations_per_cell << '\n';
        std::cout << "z_scaling: " << z_scaling << '\n';
        std::cout << "blur_sigma: " << blur_sigma << '\n';
        std::cout << "preprocess_mode: " << preprocess_mode << '\n';
        std::cout << "light_preprocess_gamma: " << light_preprocess_gamma << '\n';
        std::cout << "iterative_penalty: " << iterative_penalty << '\n';
        std::cout << "iterative_penalty_range: " << iterative_penalty_range << '\n';
        std::cout << "iterative_reward_gate: " << iterative_reward_gate << '\n';
        std::cout << "iterative_reward_gate_step: " << iterative_reward_gate_step << '\n';
        std::cout << "iterative_reward_gate_min: " << iterative_reward_gate_min << '\n';
        std::cout << "iterative_reward: " << iterative_reward << '\n';
        std::cout << "iterative_reward_gate_learning_rate: " << iterative_reward_gate_learning_rate << '\n';
        std::cout << "iterative_reward_gate_min_probability: " << iterative_reward_gate_min_probability << '\n';
        std::cout << "iterative_no_improvement_patience: " << iterative_no_improvement_patience << '\n';
        std::cout << "iterative_score_explosion_threshold: " << iterative_score_explosion_threshold << '\n';
        std::cout << "iterative_candidate_max_mean: " << iterative_candidate_max_mean << '\n';
        std::cout << "iterative_candidate_max_saturated_fraction: " << iterative_candidate_max_saturated_fraction << '\n';
        std::cout << "iterative_reward_gate_seed: " << iterative_reward_gate_seed << '\n';
        std::cout << "iterative_score_max: " << iterative_score_max << '\n';
        std::cout << "iterative_score_target_tolerance: " << iterative_score_target_tolerance << '\n';
        std::cout << "iterative_max_count: " << iterative_max_count << '\n';
        std::cout << "iterative_improvement_tolerance: " << iterative_improvement_tolerance << '\n';
        std::cout << "iterative_score_percentile: " << iterative_score_percentile << '\n';
        std::cout << "contrast_structure_threshold: " << contrast_structure_threshold << '\n';
        std::cout << "contrast_background_floor: " << contrast_background_floor << '\n';
        std::cout << "contrast_bright_fraction: " << contrast_bright_fraction << '\n';
        std::cout << "contrast_brightness_weight: " << contrast_brightness_weight << '\n';
        std::cout << "contrast_window_radius_step: " << contrast_window_radius_step << '\n';
        std::cout << "contrast_penalty_min_radius_scale: " << contrast_penalty_min_radius_scale << '\n';
        std::cout << "contrast_reward_weight: " << contrast_reward_weight << '\n';
        std::cout << "contrast_penalty_weight: " << contrast_penalty_weight << '\n';
        std::cout << "contrast_eps: " << contrast_eps << '\n';
        std::cout << "frame_intensity_normalization_enabled: " << frame_intensity_normalization_enabled << '\n';
        std::cout << "frame_intensity_percentile_exclude_zeros: " << frame_intensity_percentile_exclude_zeros << '\n';
        std::cout << "frame_intensity_scale_low_percentile: " << frame_intensity_scale_low_percentile << '\n';
        std::cout << "frame_intensity_scale_high_percentile: " << frame_intensity_scale_high_percentile << '\n';
        std::cout << "frame_intensity_hard_max: " << frame_intensity_hard_max << '\n';
        std::cout << "edge_brightness_alignment_enabled: " << edge_brightness_alignment_enabled << '\n';
        std::cout << "edge_brightness_alignment_xy_margin: " << edge_brightness_alignment_xy_margin << '\n';
        std::cout << "edge_brightness_alignment_left_offset: " << edge_brightness_alignment_left_offset << '\n';
        std::cout << "edge_brightness_alignment_right_offset: " << edge_brightness_alignment_right_offset << '\n';
        std::cout << "edge_brightness_alignment_top_offset: " << edge_brightness_alignment_top_offset << '\n';
        std::cout << "edge_brightness_alignment_bottom_offset: " << edge_brightness_alignment_bottom_offset << '\n';
        std::cout << "edge_brightness_alignment_max_shift: " << edge_brightness_alignment_max_shift << '\n';
        std::cout << "post_alignment_black_threshold: " << post_alignment_black_threshold << '\n';
        std::cout << "post_alignment_black_percentile: " << post_alignment_black_percentile << '\n';
        std::cout << "post_alignment_chunk_blackoff_enabled: " << post_alignment_chunk_blackoff_enabled << '\n';
        std::cout << "post_alignment_chunk_target_count: " << post_alignment_chunk_target_count << '\n';
        std::cout << "post_alignment_chunk_min_size: " << post_alignment_chunk_min_size << '\n';
        std::cout << "post_alignment_chunk_max_size: " << post_alignment_chunk_max_size << '\n';
        std::cout << "post_alignment_chunk_percentile_step: " << post_alignment_chunk_percentile_step << '\n';
        std::cout << "post_alignment_chunk_max_percentile: " << post_alignment_chunk_max_percentile << '\n';
        std::cout << "post_alignment_chunk_non_improvement_patience: " << post_alignment_chunk_non_improvement_patience << '\n';
        std::cout << "post_alignment_chunk_disable_below_count: " << post_alignment_chunk_disable_below_count << '\n';
        std::cout << "post_alignment_chunk_detector_threads: " << post_alignment_chunk_detector_threads << '\n';
        std::cout << "post_alignment_tiny_particle_removal_enabled: " << post_alignment_tiny_particle_removal_enabled << '\n';
        std::cout << "preprocess_radius_batch_size: " << preprocess_radius_batch_size << '\n';
        std::cout << "post_process_blur_sigma: " << post_process_blur_sigma << '\n';
        std::cout << "post_process_final_blur_sigma: " << post_process_final_blur_sigma << '\n';
        std::cout << "post_process_final_direct_weight: " << post_process_final_direct_weight << '\n';
        std::cout << "post_process_final_direct_amplification: " << post_process_final_direct_amplification << '\n';
        std::cout << "post_process_final_blurred_amplification: " << post_process_final_blurred_amplification << '\n';
        std::cout << "post_alignment_final_blur_sigma: " << post_alignment_final_blur_sigma << '\n';
        std::cout << "export_preprocessed_images: " << export_preprocessed_images << '\n';
        std::cout << "export_signal_debug_images: " << export_signal_debug_images << '\n';
        std::cout << "export_perturb_debug_images: " << export_perturb_debug_images << '\n';
        std::cout << "export_perturb_cell_center_debug_images: " << export_perturb_cell_center_debug_images << '\n';
        std::cout << "export_frame_png: " << export_frame_png << '\n';
        std::cout << "export_frame_tiff: " << export_frame_tiff << '\n';
        std::cout << "quit_after_preprocessing: " << quit_after_preprocessing << '\n';
        std::cout << "enable_lineage_tree_window: " << enable_lineage_tree_window << '\n';
        std::cout << "prepare_analyze_one_frame: " << prepare_analyze_one_frame << '\n';
        std::cout << "release_analyzed_exported_frames: " << release_analyzed_exported_frames << '\n';
        std::cout << "perturb_oscillation_boost_cells_enabled: " << perturb_oscillation_boost_cells_enabled << '\n';
        std::cout << "perturb_oscillation_boost_trash_enabled: " << perturb_oscillation_boost_trash_enabled << '\n';
        std::cout << "perturb_oscillation_warmup_fraction: " << perturb_oscillation_warmup_fraction << '\n';
        std::cout << "perturb_oscillation_window: " << perturb_oscillation_window << '\n';
        std::cout << "perturb_oscillation_min_samples: " << perturb_oscillation_min_samples << '\n';
        std::cout << "perturb_oscillation_sign_change_ratio: " << perturb_oscillation_sign_change_ratio << '\n';
        std::cout << "perturb_oscillation_boost_ratio: " << perturb_oscillation_boost_ratio << '\n';
        std::cout << "perturb_oscillation_max_multiplier: " << perturb_oscillation_max_multiplier << '\n';
        std::cout << "perturb_oscillation_reset_on_accept: " << perturb_oscillation_reset_on_accept << '\n';
        std::cout << "perturb_oscillation_small_step_probability: " << perturb_oscillation_small_step_probability << '\n';
        std::cout << "perturb_oscillation_small_step_multiplier: " << perturb_oscillation_small_step_multiplier << '\n';
        std::cout << "perturb_oscillation_black_pixel_boost_enabled: " << perturb_oscillation_black_pixel_boost_enabled << '\n';
        std::cout << "perturb_oscillation_black_pixel_boost_weight: " << perturb_oscillation_black_pixel_boost_weight << '\n';
        std::cout << "perturb_debug_cell_brightness: " << perturb_debug_cell_brightness << '\n';
        std::cout << "perturb_debug_center_cube_radius: " << perturb_debug_center_cube_radius << '\n';
        std::cout << "cube_pooling_enabled: " << cube_pooling_enabled << '\n';
        std::cout << "cube_pooling_cost_comparison_enabled: " << cube_pooling_cost_comparison_enabled << '\n';
        std::cout << "cube_pooling_cube_size: " << cube_pooling_cube_size << '\n';
        std::cout << "cube_pooling_mode: " << cube_pooling_mode << '\n';
        std::cout << "cube_pooling_top_fraction: " << cube_pooling_top_fraction << '\n';
        std::cout << "cube_pooling_low_fraction: " << cube_pooling_low_fraction << '\n';
        std::cout << "adaptive_background_expand_factor: " << adaptive_background_expand_factor << '\n';
        std::cout << "adaptive_background_top_fraction: " << adaptive_background_top_fraction << '\n';
        std::cout << "signal_guided_position_enabled: " << signal_guided_position_enabled << '\n';
        std::cout << "signal_guided_box_side_length: " << signal_guided_box_side_length << '\n';
        std::cout << "signal_guided_min_box_brightness_delta: " << signal_guided_min_box_brightness_delta << '\n';
        std::cout << "signal_guided_min_sigma_scale: " << signal_guided_min_sigma_scale << '\n';
        std::cout << "signal_guided_sigma_range_multiplier: " << signal_guided_sigma_range_multiplier << '\n';
        std::cout << "signal_guided_initial_bright_fraction: " << signal_guided_initial_bright_fraction << '\n';
        std::cout << "signal_guided_recursive_bright_fraction: " << signal_guided_recursive_bright_fraction << '\n';
        std::cout << "signal_guided_max_recursive_depth: " << signal_guided_max_recursive_depth << '\n';
        std::cout << "signal_map_enabled: " << signal_map_enabled << '\n';
        std::cout << "signal_map_blur_sigma: " << signal_map_blur_sigma << '\n';
        std::cout << "signal_map_max_iterations: " << signal_map_max_iterations << '\n';
        std::cout << "signal_map_bright_center_percentile: " << signal_map_bright_center_percentile << '\n';
        std::cout << "signal_map_perturb_guidance_enabled: " << signal_map_perturb_guidance_enabled << '\n';
        std::cout << "signal_map_cell_radius_scale: " << signal_map_cell_radius_scale << '\n';
        std::cout << "signal_map_direction_probability_boost: " << signal_map_direction_probability_boost << '\n';
        std::cout << "parallel_threads: " << parallel_threads << '\n';
        std::cout << "parallel_min_slices: " << parallel_min_slices << '\n';
        std::cout << "initial_z_space: " << initial_z_space << '\n';
        std::cout << "z_slices: " << z_slices << std::endl;
    }
};

class ProbabilityConfig {
public:
    // ---- Triaxial pipeline fields (2026-04-11 redesign) ----
    float P_split_base = 0.03f;
    float P_split_max = 0.5f;

    float overlap_penalty_weight = 500.0f;
    float split_cost = 80.0f;
    // Proportional split cost: threshold = max(split_cost, fraction × baselineImageCost).
    // 0.0 = disabled (use fixed split_cost only). Typical: 0.03 = split must
    // improve cost by at least 3% of the baseline to accept.
    float split_cost_fraction = 0.0f;
    // Optional rescue for visually strong split bridges whose bbox image cost
    // is worse than the live single-parent baseline.
    bool split_bridge_cost_rescue_enabled = false;
    float split_bridge_cost_rescue_max_positive_fraction = 0.20f;
    float split_bridge_cost_rescue_max_valley_ratio = 0.40f;
    float split_bridge_cost_rescue_max_gap_density = 0.05f;
    // false keeps the original one-sided bridge rescue behavior, where the
    // bridge only needs to be darker than the brighter daughter edge. Light
    // preprocessing can enable this stricter check so both daughter sides must
    // show a real valley before a positive cost split is rescued.
    bool split_bridge_cost_rescue_require_two_sided_valley = false;
    // Quadratic position prior weight for perturbCell.
    // penalty = weight × ||cell.pos - snap.pos||²
    // Temporal anchor independent of image evidence. Prevents drift
    // when overlap/bright-neighbor halos would otherwise overpower
    // the image-based snap anchor. Typical: 50.0 (20 px drift = 20K
    // penalty; same scale as bbox image cost).
    float position_prior_weight = 0.0f;
    // Free-motion threshold (px). Drift below this pays no penalty.
    // Above, quadratic penalty: w × (drift - threshold)². 20 px covers
    // the p95 of biological motion in best22 (24 px is the p95 of
    // drift per transition).
    float position_prior_threshold = 20.0f;

    // Hard per-frame velocity cap on cell position drift from snap.
    // Any perturbation proposing a position more than this many voxels
    // from the snap position is rejected outright (not penalized, not
    // clamped — just refused). Complements the soft position_prior
    // penalty: while the prior provides a quadratic pull-back signal,
    // the cap acts as a guardrail against runaway Monte Carlo random
    // walks that the prior fails to stop in time.
    // xy limit is typically smaller (cells move slowly in plane) than
    // z limit (z can have larger per-frame drift from splits or
    // interpolation artifacts). Set to -1 to disable.
    // Recommended: xy=15, z=20 for most datasets. Observed bug case:
    // a510 drifted +18 in z (f41→f42) then +23 (f42→f43) before
    // hitting z=224 ceiling. A z cap of 15 would have blocked both.
    // 2026-04-21 (drop the cap): disabled by default. At 15 vx/frame xy the
    // cap was clipping legit biological motion during early frames (e.g.
    // 1f2ed10d and e3d03 moved 20-27 vx/frame in xy between f3-f6 in
    // perfect_45, hitting the cap and lagging ~5-13 vx per frame → ~28 vx
    // cumulative drift visible as off-center ellipsoids by f5-f6). The
    // position_prior_weight quadratic penalty (non-saturating) remains
    // active and handles late-frame drift without clipping fast biology.
    // Set to > 0 to re-enable.
    float max_perturb_drift_xy = 0.0f;
    float max_perturb_drift_z = 0.0f;

    // Birth-relative growth budget (anti-bloat). A cell's radii are capped
    // at `birthRadii × birth_growth_cap_factor` per axis, BUT only when the
    // cell also shows shape elongation >= birth_growth_cap_elong_threshold.
    // Both conditions must be true to trigger clamping — this catches the
    // classic bloat signature (big radius + elongated, meaning a cell that
    // should have split but merged into one oversize ellipsoid) while
    // leaving healthy growing cells alone.
    //
    // Example tuning (current defaults):
    //   a511 at f37: radius 1.9× birth + elong 2.07 → BOTH conditions met → clamped
    //   e3d03 at f28: radius 1.2× birth + elong 1.83 → radius condition not met → NOT clamped
    //
    // Set factor to 0 or negative to disable.
    float birth_growth_cap_factor = 1.8f;
    float birth_growth_cap_elong_threshold = 1.8f;

    int expected_daughter_pre_pass_iterations = 1;
    // Optional deterministic split scheduling from the image-grounded
    // pre-pass. When enabled, cells whose pre-pass daughter centers are
    // already well separated get exactly one split attempt before the random
    // perturb loop starts. Acceptance still goes through the normal bio,
    // bridge, and cost gates, so this only prevents probability scheduling
    // from missing a strong candidate.
    bool expected_daughter_force_split_enabled = false;
    float expected_daughter_force_min_separation_parent_fraction = 0.0f;
    int expected_daughter_force_min_kept_pixels = 0;

    // Number of candidate placements per split attempt. With two midpoint
    // options (PCA and snapshot) each generating 5 variants (primary + 2
    // rotation + 2 translation), the natural total is 10 per primary
    // direction. Raised from 5 on 2026-04-11 when the dual-midpoint logic
    // was added so both midpoints fit inside the cap.
    int split_candidates_per_attempt = 10;
    int split_candidate_burn_in_iterations = 20;
    // After the K-candidate loop picks a winner, run this many extra
    // perturb iterations on JUST the winning pair of daughters (under the
    // same tight burn-in sigmas) before the bio/cost gates fire. Lets the
    // chosen daughters settle to a refined local optimum without paying
    // the K× cost of longer per-candidate burn-ins.
    int split_final_refine_iterations = 30;
    // Per-cell position calibration pass between the pre-pass and Phase A/B.
    // Each cell gets this many perturbCell iterations with tight position
    // sigmas and ALL radius sigmas forced to 0 so the parent refines its
    // center without collapsing radii onto one incipient daughter.
    // Addresses Phase A/B's tendency to park the parent on one daughter
    // before split attempts fire.
    int split_calibration_iterations_per_cell = 50;
    float split_candidate_rotation_delta_degrees = 8.0f;
    float split_candidate_translation_delta_fraction = 0.2f;

    // Geometry gate for accepted split candidates. This rejects cases where
    // the optimizer makes a low-cost "split" by letting daughters walk far
    // away from the proposed division geometry. The limits are evaluated
    // after daughter refine and before bridge/cost gates.
    bool split_geometry_gate_enabled = true;
    float split_max_daughter_seed_drift_fraction = 1.25f;
    float split_max_daughter_axis_expansion = 1.8f;
    bool split_axis_alignment_gate_enabled = true;
    float split_axis_alignment_sphere_angle_degrees = 120.0f;
    float split_axis_alignment_elongation_shrink = 0.75f;
    float split_axis_alignment_min_angle_degrees = 20.0f;
    bool split_daughter_overlap_gate_enabled = true;
    float split_max_daughter_overlap_fraction = 0.20f;
    float split_daughter_overlap_scale = 1.0f;

    float bio_daughter_size_ratio_max = 1.5f;
    float bio_combined_volume_min_fraction = 0.6f;
    float bio_combined_volume_max_fraction = 1.3f;
    // Optional minimum daughter separation, normalized by the snapshot parent
    // max radius. Disabled at 0.0. This blocks false splits where two
    // daughters sit inside the same parent-sized blob, while allowing true
    // divisions whose daughter centers separate across the parent body.
    float bio_min_daughter_separation_parent_fraction = 0.0f;

    // Single-daughter volume gate. Reject when either daughter's volume
    // exceeds `bio_max_single_daughter_volume_fraction * refParentVolume`.
    // For a real division each daughter is ~0.5 * parent volume. One
    // daughter >0.65 * parent means "one daughter is essentially the
    // parent" — catches asymmetric "mimicking" splits like 1f2ed f4
    // (d2 volume = 0.74 * parent) where one daughter inherits the parent
    // and the other is a small bolt-on.
    float bio_max_single_daughter_volume_fraction = 0.65f;

    // Bridge brightness gate — catches the "two equal-size daughters at
    // parent center covering one continuous cell" pattern that passes
    // every other gate.
    //
    // Reuses the Voronoi-filtered bright-pixel set that PCA already
    // computed for the split attempt. Each pixel is projected onto the
    // split axis (d1→d2), normalized so -1 is d1 center and +1 is d2
    // center. Two independent signals are computed:
    //
    //   gap_density  = fraction of in-range pixels with |t| < 0.3
    //                  (in the middle ~30% of the axis between daughters)
    //   valley_ratio = mean(brightness of gap pixels) /
    //                  mean(brightness of edge pixels at 0.6 < |t| < 1.1)
    //
    // For a real division the dividing groove empties the gap (low
    // density) AND dims it (low ratio). For a fake continuous cell
    // both signals are flat — high density AND ratio ~1.0. The gate
    // fires only when BOTH signals indicate "flat profile", so real
    // divisions with a partial groove still pass.
    float bio_bridge_max_gap_density = 0.18f;
    float bio_bridge_max_valley_ratio = 0.85f;
    // Bridge tier-1: hard rejection threshold on the worst per-daughter
    // valley ratio (gap / min edge). When gap brightness equals or exceeds
    // an edge's brightness, there is no valley at all on that daughter's
    // side and the split is clearly phantom. Applied to
    // max(valleyRatio1, valleyRatio2).
    float bio_bridge_no_valley_hard_threshold = 0.95f;
    // Absolute minimum edge brightness (real-image units above background).
    // Rejects splits where one daughter's edge zone is near-background —
    // a phantom daughter sitting in empty space regardless of ratios.
    // Typical cell cores are 0.1-0.3 above background; 0.05 is ~3× bg noise.
    float bio_bridge_min_edge_brightness_absolute = 0.05f;

    // Multiplier applied to the Ellipsoid x/y/z perturbation sigmas during
    // candidate burn-in. The main-loop sigmas (x=5, y=5, z=8) let daughters
    // wander 15-25 voxels across a 20-iter burn-in, enough to leave the
    // parent footprint. 0.4 scales them to (2, 2, 3.2), limiting drift to
    // refinement distances (<10 voxels) while still letting daughters
    // settle to the local image optimum.
    float split_burn_in_pos_sigma_scale = 0.4f;

    // When true, every cost evaluation for perturbation and split uses
    // a per-cell bbox with Voronoi neighbor exclusion instead of
    // full-image asymmetric L2. Concentrates cost signal, eliminates
    // cross-cell contamination, and runs faster. See
    // `C++/docs/plans/2026-04-14-universal-bbox-cost.md`.
    // Bbox margin = bbox_margin_scale * max(a,b,c) of the cell (D1=3.0).
    bool use_bbox_cost = false;
    float bbox_margin_scale = 3.0f;

    // Per-daughter PCA radius refit in the split refine phase (A1).
    // After positional refine settles both daughters, each daughter runs
    // a short PCA shape fit (using its own built-in radii as the mask)
    // so its aRadius/bRadius/cRadius match the actual blob it covers.
    // Real daughters tighten → lower image cost, widening the cost gap
    // vs false splits where a phantom daughter has no compact cloud
    // to fit. Clamped at `split_daughter_refit_min_radius_fraction *
    // built_radius` to prevent the collapsed-sliver regression.
    // Set to 0 to disable.
    int split_daughter_refit_iterations = 3;
    float split_daughter_refit_min_radius_fraction = 0.6f;
    // Upper cap on post-refit radii: clamp at max * built_radii per axis.
    // Prevents newborn-daughter bloat when the sibling Voronoi boundary
    // is immature and the mask absorbs neighbor/halo pixels. Typically
    // 1.1 (10% over built).
    float split_daughter_refit_max_radius_fraction = 1.1f;

    // Built-time per-axis radius scale for newly-spawned daughters.
    // Daughter radii = scale × snapshot parent radii.
    //   ∛(0.5) ≈ 0.7937: volume-preserving split (each daughter has half
    //                    the parent's volume; geometric default).
    //   1.0:             daughters as large as parent (covers more
    //                    image material on first cost eval; risk of
    //                    overlap penalty).
    //   0.85 - 0.95:     enlarged daughters — useful when the parent
    //                    PCA fit is tight and you want daughters to
    //                    cover the full real cell extent immediately
    //                    (relies on bio gates to reject if too big).
    //   0.5 - 0.7:       tighter daughters than volume-preserving;
    //                    relies on per-daughter PCA refit (A1) to grow
    //                    them back if the real cell is bigger.
    // Applied per axis equally — daughters keep the parent's aspect ratio
    // before the per-daughter PCA refit re-shapes them.
    float split_daughter_volume_scale = 0.7937f;

    // Post-PCA rescue for false fits where one ellipsoid stretches across
    // two bright blobs with a dark bridge between them. Runs after the
    // per-frame PCA shape fit, before growth caps/reference updates.
    bool pca_bridge_split_enabled = false;
    float pca_bridge_elongation_ratio = 3.0f;
    // Prolate-shape pre-filter (ported from yp_opt_speed 1ec91e7). Reject
    // bridge candidates where the high elongation comes from one collapsed
    // axis instead of a clean rod shape. R=(43,30,12) has long/short=3.58
    // (passes elong gate) but mid/short=2.5 — wedge, not rod, and the
    // bin-analysis "valley" along the long axis is not biological.
    //   long/mid >= pca_bridge_min_long_mid_ratio  → one clearly long axis
    //   mid/short <= pca_bridge_max_mid_short_ratio → two non-long axes
    //                                                 close in size
    // Set either to 0 to disable that half of the gate.
    float pca_bridge_min_long_mid_ratio = 1.35f;
    float pca_bridge_max_mid_short_ratio = 1.35f;
    float pca_bridge_black_threshold = 0.05f;
    float pca_bridge_min_black_fraction = 0.75f;
    float pca_bridge_gap_center_fraction = 0.35f;
    int pca_bridge_profile_bins = 21;
    int pca_bridge_min_gap_bins = 2;
    int pca_bridge_min_side_voxels = 40;
    float pca_bridge_daughter_radius_scale = 2.236f;
    float pca_bridge_min_radius_fraction = 0.35f;
    float pca_bridge_max_radius_fraction = 0.90f;
    float pca_bridge_min_cost_improvement = 0.0f;
    float pca_bridge_overlap_weight = -1.0f;

    ProbabilityConfig() = default;

    void explodeConfig(const YAML::Node& node) {
        if (node["P_split_base"]) P_split_base = node["P_split_base"].as<float>();
        if (node["P_split_max"]) P_split_max = node["P_split_max"].as<float>();
        if (node["overlap_penalty_weight"]) overlap_penalty_weight = node["overlap_penalty_weight"].as<float>();
        if (node["split_cost"]) split_cost = node["split_cost"].as<float>();
        if (node["split_cost_fraction"]) split_cost_fraction = node["split_cost_fraction"].as<float>();
        if (node["split_bridge_cost_rescue_enabled"]) split_bridge_cost_rescue_enabled = node["split_bridge_cost_rescue_enabled"].as<bool>();
        if (node["split_bridge_cost_rescue_max_positive_fraction"]) split_bridge_cost_rescue_max_positive_fraction = node["split_bridge_cost_rescue_max_positive_fraction"].as<float>();
        if (node["split_bridge_cost_rescue_max_valley_ratio"]) split_bridge_cost_rescue_max_valley_ratio = node["split_bridge_cost_rescue_max_valley_ratio"].as<float>();
        if (node["split_bridge_cost_rescue_max_gap_density"]) split_bridge_cost_rescue_max_gap_density = node["split_bridge_cost_rescue_max_gap_density"].as<float>();
        if (node["split_bridge_cost_rescue_require_two_sided_valley"]) split_bridge_cost_rescue_require_two_sided_valley = node["split_bridge_cost_rescue_require_two_sided_valley"].as<bool>();
        if (node["position_prior_weight"]) position_prior_weight = node["position_prior_weight"].as<float>();
        if (node["position_prior_threshold"]) position_prior_threshold = node["position_prior_threshold"].as<float>();
        if (node["max_perturb_drift_xy"]) max_perturb_drift_xy = node["max_perturb_drift_xy"].as<float>();
        if (node["max_perturb_drift_z"]) max_perturb_drift_z = node["max_perturb_drift_z"].as<float>();
        if (node["birth_growth_cap_factor"]) birth_growth_cap_factor = node["birth_growth_cap_factor"].as<float>();
        if (node["birth_growth_cap_elong_threshold"]) birth_growth_cap_elong_threshold = node["birth_growth_cap_elong_threshold"].as<float>();
        if (node["expected_daughter_pre_pass_iterations"]) expected_daughter_pre_pass_iterations = node["expected_daughter_pre_pass_iterations"].as<int>();
        if (node["expected_daughter_force_split_enabled"]) expected_daughter_force_split_enabled = node["expected_daughter_force_split_enabled"].as<bool>();
        if (node["expected_daughter_force_min_separation_parent_fraction"]) expected_daughter_force_min_separation_parent_fraction = node["expected_daughter_force_min_separation_parent_fraction"].as<float>();
        if (node["expected_daughter_force_min_kept_pixels"]) expected_daughter_force_min_kept_pixels = node["expected_daughter_force_min_kept_pixels"].as<int>();
        if (node["split_candidates_per_attempt"]) split_candidates_per_attempt = node["split_candidates_per_attempt"].as<int>();
        if (node["split_candidate_burn_in_iterations"]) split_candidate_burn_in_iterations = node["split_candidate_burn_in_iterations"].as<int>();
        if (node["split_final_refine_iterations"]) split_final_refine_iterations = node["split_final_refine_iterations"].as<int>();
        if (node["split_calibration_iterations_per_cell"]) split_calibration_iterations_per_cell = node["split_calibration_iterations_per_cell"].as<int>();
        if (node["split_candidate_rotation_delta_degrees"]) split_candidate_rotation_delta_degrees = node["split_candidate_rotation_delta_degrees"].as<float>();
        if (node["split_candidate_translation_delta_fraction"]) split_candidate_translation_delta_fraction = node["split_candidate_translation_delta_fraction"].as<float>();
        if (node["split_geometry_gate_enabled"]) split_geometry_gate_enabled = node["split_geometry_gate_enabled"].as<bool>();
        if (node["split_max_daughter_seed_drift_fraction"]) split_max_daughter_seed_drift_fraction = node["split_max_daughter_seed_drift_fraction"].as<float>();
        if (node["split_max_daughter_axis_expansion"]) split_max_daughter_axis_expansion = node["split_max_daughter_axis_expansion"].as<float>();
        if (node["split_axis_alignment_gate_enabled"]) split_axis_alignment_gate_enabled = node["split_axis_alignment_gate_enabled"].as<bool>();
        if (node["split_axis_alignment_sphere_angle_degrees"]) split_axis_alignment_sphere_angle_degrees = node["split_axis_alignment_sphere_angle_degrees"].as<float>();
        if (node["split_axis_alignment_elongation_shrink"]) split_axis_alignment_elongation_shrink = node["split_axis_alignment_elongation_shrink"].as<float>();
        if (node["split_axis_alignment_min_angle_degrees"]) split_axis_alignment_min_angle_degrees = node["split_axis_alignment_min_angle_degrees"].as<float>();
        if (node["split_daughter_overlap_gate_enabled"]) split_daughter_overlap_gate_enabled = node["split_daughter_overlap_gate_enabled"].as<bool>();
        if (node["split_max_daughter_overlap_fraction"]) split_max_daughter_overlap_fraction = node["split_max_daughter_overlap_fraction"].as<float>();
        if (node["split_daughter_overlap_scale"]) split_daughter_overlap_scale = node["split_daughter_overlap_scale"].as<float>();
        if (node["split_parent_overlap_gate_enabled"]) split_daughter_overlap_gate_enabled = node["split_parent_overlap_gate_enabled"].as<bool>();
        if (node["split_max_parent_overlap_fraction"]) split_max_daughter_overlap_fraction = node["split_max_parent_overlap_fraction"].as<float>();
        if (node["split_parent_overlap_daughter_scale"]) split_daughter_overlap_scale = node["split_parent_overlap_daughter_scale"].as<float>();
        if (node["bio_daughter_size_ratio_max"]) bio_daughter_size_ratio_max = node["bio_daughter_size_ratio_max"].as<float>();
        if (node["bio_combined_volume_min_fraction"]) bio_combined_volume_min_fraction = node["bio_combined_volume_min_fraction"].as<float>();
        if (node["bio_combined_volume_max_fraction"]) bio_combined_volume_max_fraction = node["bio_combined_volume_max_fraction"].as<float>();
        if (node["bio_min_daughter_separation_parent_fraction"]) bio_min_daughter_separation_parent_fraction = node["bio_min_daughter_separation_parent_fraction"].as<float>();
        if (node["bio_max_single_daughter_volume_fraction"]) bio_max_single_daughter_volume_fraction = node["bio_max_single_daughter_volume_fraction"].as<float>();
        if (node["bio_bridge_max_gap_density"]) bio_bridge_max_gap_density = node["bio_bridge_max_gap_density"].as<float>();
        if (node["bio_bridge_max_valley_ratio"]) bio_bridge_max_valley_ratio = node["bio_bridge_max_valley_ratio"].as<float>();
        if (node["bio_bridge_no_valley_hard_threshold"]) bio_bridge_no_valley_hard_threshold = node["bio_bridge_no_valley_hard_threshold"].as<float>();
        if (node["bio_bridge_min_edge_brightness_absolute"]) bio_bridge_min_edge_brightness_absolute = node["bio_bridge_min_edge_brightness_absolute"].as<float>();
        if (node["split_burn_in_pos_sigma_scale"]) split_burn_in_pos_sigma_scale = node["split_burn_in_pos_sigma_scale"].as<float>();
        if (node["use_bbox_cost"]) use_bbox_cost = node["use_bbox_cost"].as<bool>();
        if (node["bbox_margin_scale"]) bbox_margin_scale = node["bbox_margin_scale"].as<float>();
        if (node["split_daughter_refit_iterations"]) split_daughter_refit_iterations = node["split_daughter_refit_iterations"].as<int>();
        if (node["split_daughter_refit_min_radius_fraction"]) split_daughter_refit_min_radius_fraction = node["split_daughter_refit_min_radius_fraction"].as<float>();
        if (node["split_daughter_refit_max_radius_fraction"]) split_daughter_refit_max_radius_fraction = node["split_daughter_refit_max_radius_fraction"].as<float>();
        if (node["split_daughter_volume_scale"]) split_daughter_volume_scale = node["split_daughter_volume_scale"].as<float>();
        if (node["pca_bridge_split_enabled"]) pca_bridge_split_enabled = node["pca_bridge_split_enabled"].as<bool>();
        if (node["pca_bridge_elongation_ratio"]) pca_bridge_elongation_ratio = node["pca_bridge_elongation_ratio"].as<float>();
        if (node["pca_bridge_min_long_mid_ratio"]) pca_bridge_min_long_mid_ratio = node["pca_bridge_min_long_mid_ratio"].as<float>();
        if (node["pca_bridge_max_mid_short_ratio"]) pca_bridge_max_mid_short_ratio = node["pca_bridge_max_mid_short_ratio"].as<float>();
        if (node["pca_bridge_black_threshold"]) pca_bridge_black_threshold = node["pca_bridge_black_threshold"].as<float>();
        if (node["pca_bridge_min_black_fraction"]) pca_bridge_min_black_fraction = node["pca_bridge_min_black_fraction"].as<float>();
        if (node["pca_bridge_gap_center_fraction"]) pca_bridge_gap_center_fraction = node["pca_bridge_gap_center_fraction"].as<float>();
        if (node["pca_bridge_profile_bins"]) pca_bridge_profile_bins = node["pca_bridge_profile_bins"].as<int>();
        if (node["pca_bridge_min_gap_bins"]) pca_bridge_min_gap_bins = node["pca_bridge_min_gap_bins"].as<int>();
        if (node["pca_bridge_min_side_voxels"]) pca_bridge_min_side_voxels = node["pca_bridge_min_side_voxels"].as<int>();
        if (node["pca_bridge_daughter_radius_scale"]) pca_bridge_daughter_radius_scale = node["pca_bridge_daughter_radius_scale"].as<float>();
        if (node["pca_bridge_min_radius_fraction"]) pca_bridge_min_radius_fraction = node["pca_bridge_min_radius_fraction"].as<float>();
        if (node["pca_bridge_max_radius_fraction"]) pca_bridge_max_radius_fraction = node["pca_bridge_max_radius_fraction"].as<float>();
        if (node["pca_bridge_min_cost_improvement"]) pca_bridge_min_cost_improvement = node["pca_bridge_min_cost_improvement"].as<float>();
        if (node["pca_bridge_overlap_weight"]) pca_bridge_overlap_weight = node["pca_bridge_overlap_weight"].as<float>();

        // Legacy YAML aliases — silently map to the new names.
        if (node["split"]) P_split_base = node["split"].as<float>();
        if (node["max_split_probability"]) P_split_max = node["max_split_probability"].as<float>();
    }

    void printConfig() const {
        std::cout << "Probability Config\n";
        std::cout << "P_split_base: " << P_split_base << '\n';
        std::cout << "P_split_max: " << P_split_max << '\n';
        std::cout << "split_cost: " << split_cost << '\n';
        std::cout << "split_cost_fraction: " << split_cost_fraction << '\n';
        std::cout << "split_bridge_cost_rescue_enabled: " << split_bridge_cost_rescue_enabled << '\n';
        std::cout << "split_bridge_cost_rescue_max_positive_fraction: " << split_bridge_cost_rescue_max_positive_fraction << '\n';
        std::cout << "split_bridge_cost_rescue_max_valley_ratio: " << split_bridge_cost_rescue_max_valley_ratio << '\n';
        std::cout << "split_bridge_cost_rescue_max_gap_density: " << split_bridge_cost_rescue_max_gap_density << '\n';
        std::cout << "split_bridge_cost_rescue_require_two_sided_valley: " << split_bridge_cost_rescue_require_two_sided_valley << '\n';
        std::cout << "overlap_penalty_weight: " << overlap_penalty_weight << '\n';
        std::cout << "expected_daughter_force_split_enabled: " << expected_daughter_force_split_enabled << '\n';
        std::cout << "expected_daughter_force_min_separation_parent_fraction: " << expected_daughter_force_min_separation_parent_fraction << '\n';
        std::cout << "expected_daughter_force_min_kept_pixels: " << expected_daughter_force_min_kept_pixels << '\n';
        std::cout << "split_candidates_per_attempt: " << split_candidates_per_attempt << '\n';
        std::cout << "split_candidate_burn_in_iterations: " << split_candidate_burn_in_iterations << '\n';
        std::cout << "split_geometry_gate_enabled: " << split_geometry_gate_enabled << '\n';
        std::cout << "split_max_daughter_seed_drift_fraction: " << split_max_daughter_seed_drift_fraction << '\n';
        std::cout << "split_max_daughter_axis_expansion: " << split_max_daughter_axis_expansion << '\n';
        std::cout << "split_axis_alignment_gate_enabled: " << split_axis_alignment_gate_enabled << '\n';
        std::cout << "split_axis_alignment_sphere_angle_degrees: " << split_axis_alignment_sphere_angle_degrees << '\n';
        std::cout << "split_axis_alignment_elongation_shrink: " << split_axis_alignment_elongation_shrink << '\n';
        std::cout << "split_axis_alignment_min_angle_degrees: " << split_axis_alignment_min_angle_degrees << '\n';
        std::cout << "split_daughter_overlap_gate_enabled: " << split_daughter_overlap_gate_enabled << '\n';
        std::cout << "split_max_daughter_overlap_fraction: " << split_max_daughter_overlap_fraction << '\n';
        std::cout << "split_daughter_overlap_scale: " << split_daughter_overlap_scale << '\n';
        std::cout << "bio_daughter_size_ratio_max: " << bio_daughter_size_ratio_max << std::endl;
        std::cout << "bio_min_daughter_separation_parent_fraction: " << bio_min_daughter_separation_parent_fraction << '\n';
        std::cout << "pca_bridge_split_enabled: " << pca_bridge_split_enabled << '\n';
        std::cout << "pca_bridge_elongation_ratio: " << pca_bridge_elongation_ratio << '\n';
        std::cout << "pca_bridge_min_long_mid_ratio: " << pca_bridge_min_long_mid_ratio << '\n';
        std::cout << "pca_bridge_max_mid_short_ratio: " << pca_bridge_max_mid_short_ratio << '\n';
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
    struct Sample {
        float offset = 0.0f;
        int direction = 0; // -1 decrease, +1 increase, 0 no signed move
    };

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
    [[nodiscard]] Sample samplePerturb() const {
        std::mt19937 &gen = cellUniverseRandomGenerator();
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        const bool hasSeparateSignProbabilities = increase_prob >= 0.0f || decrease_prob >= 0.0f;
        if (!hasSeparateSignProbabilities) {
            if (dis(gen) < prob) {
                std::normal_distribution<float> d(mu, sigma);
                return {d(gen), 0};
            }
            return {mu, 0};
        }

        const float incProb = std::clamp(increase_prob >= 0.0f ? increase_prob : 0.0f, 0.0f, 1.0f);
        const float decProbRaw = std::clamp(decrease_prob >= 0.0f ? decrease_prob : 0.0f, 0.0f, 1.0f);
        const float decProb = std::min(decProbRaw, 1.0f - incProb);
        const float roll = dis(gen);
        const float magnitude = (sigma > 0.0f)
            ? std::abs(std::normal_distribution<float>(mu, sigma)(gen))
            : std::abs(mu);

        if (roll < incProb) {
            return {magnitude, 1};
        }
        if (roll < incProb + decProb) {
            return {-magnitude, -1};
        }
        return {0.0f, 0};
    }
    [[nodiscard]] Sample samplePerturbWithSignedProbabilityBias(float signedBias,
                                                                float boost) const {
        const bool hasSeparateSignProbabilities = increase_prob >= 0.0f || decrease_prob >= 0.0f;
        if (!hasSeparateSignProbabilities || std::abs(signedBias) <= 1e-6f || boost <= 0.0f) {
            return samplePerturb();
        }

        PerturbParams biased = *this;
        float inc = std::clamp(increase_prob >= 0.0f ? increase_prob : 0.0f, 0.0f, 1.0f);
        float dec = std::clamp(decrease_prob >= 0.0f ? decrease_prob : 0.0f, 0.0f, 1.0f);
        if (inc + dec > 1.0f) {
            dec = 1.0f - inc;
        }

        float amount = std::clamp(boost, 0.0f, 1.0f) * std::min(1.0f, std::abs(signedBias));
        if (signedBias > 0.0f) {
            const float fromDec = std::min(dec, amount);
            dec -= fromDec;
            inc += fromDec;
            amount -= fromDec;
            const float fromNone = std::min(1.0f - inc - dec, amount);
            inc += fromNone;
        } else {
            const float fromInc = std::min(inc, amount);
            inc -= fromInc;
            dec += fromInc;
            amount -= fromInc;
            const float fromNone = std::min(1.0f - inc - dec, amount);
            dec += fromNone;
        }

        biased.increase_prob = std::clamp(inc, 0.0f, 1.0f);
        biased.decrease_prob = std::clamp(dec, 0.0f, 1.0f - biased.increase_prob);
        return biased.samplePerturb();
    }
    [[nodiscard]] float getPerturbOffset() const {
        return samplePerturb().offset;
    }
    void adjustSignedProbability(int direction, float delta) {
        if (direction == 0 || std::abs(delta) <= 1e-9f) {
            return;
        }
        const bool hasSeparateSignProbabilities = increase_prob >= 0.0f || decrease_prob >= 0.0f;
        if (!hasSeparateSignProbabilities) {
            return;
        }

        if (direction > 0) {
            const float other = std::clamp(decrease_prob >= 0.0f ? decrease_prob : 0.0f, 0.0f, 1.0f);
            increase_prob = std::clamp((increase_prob >= 0.0f ? increase_prob : 0.0f) + delta, 0.0f, 1.0f - other);
            return;
        }

        const float other = std::clamp(increase_prob >= 0.0f ? increase_prob : 0.0f, 0.0f, 1.0f);
        decrease_prob = std::clamp((decrease_prob >= 0.0f ? decrease_prob : 0.0f) + delta, 0.0f, 1.0f - other);
    }
};

class EllipsoidConfig {
public:
    PerturbParams x{};
    PerturbParams y{};
    PerturbParams z{};
    PerturbParams aRadius{};
    PerturbParams bRadius{};
    PerturbParams cRadius{};
    PerturbParams thetaX{};
    PerturbParams thetaY{};
    PerturbParams thetaZ{};
    PerturbParams brightness{};
    double minARadius{};
    double maxARadius{};
    double minBRadius{};
    double maxBRadius{};
    double minCRadius{};
    double maxCRadius{};
    bool minAnyRadiusEnabled{false};
    double minAnyRadius{0.0};
    float initialBrightness{0.2f};
    float initialRadiusScale{1.0f};
    float backgroundColor{0.0f};
    double minBrightness{0.1};
    double maxBrightness{1.0};
    float brightnessProbabilityStep{0.02f};
    float brightnessProbabilityTrust{1.0f};
    float aRadiusProbabilityStep{0.02f};
    float aRadiusProbabilityTrust{1.0f};
    float bRadiusProbabilityStep{0.02f};
    float bRadiusProbabilityTrust{1.0f};
    float cRadiusProbabilityStep{0.02f};
    float cRadiusProbabilityTrust{1.0f};
    float brightnessUpdateBlend{0.2f};
    float brightnessMeanAmplification{1.0f};
    float brightnessMeasurementTopPercentile{0.3f};
    // Iterative PCA shape fit (per frame). Converges rotation + radii +
    // optional centroid position to the bright-pixel cloud inside a
    // Voronoi-filtered ellipsoid mask. See Frame::calibrateCellShapeViaPca.
    // Disabled when pcaShapeMaxIters <= 0.
    int pcaShapeMaxIters{15};
    // Scale factor: radius = pcaShapeRadiusScale * sqrt(eigenvalue).
    // sqrt(5) ≈ 2.236 matches variance of a uniformly-filled ellipsoid.
    float pcaShapeRadiusScale{2.236f};
    // Minimum pixel count before accepting a PCA iteration. Below this, stop.
    int pcaShapeMinPixels{50};
    // Ellipsoid-mask scale-up: bright pixels inside (scale * current ellipsoid)
    // are candidates. 1.3 allows modest growth between iterations.
    float pcaShapeMaskScale{1.3f};
    // Per-iter convergence thresholds. Iteration stops when the max radius
    // change is below this AND the max axis rotation is below the angle
    // tolerance (degrees).
    float pcaShapeConvergeRadius{0.3f};
    float pcaShapeConvergeAngleDeg{2.0f};
    // When true, PCA centroid drives the cell's position (capped per iter).
    bool  pcaShapeUpdatePosition{true};
    float pcaShapeMaxPosShiftFraction{0.5f}; // cap per-iter shift at fraction * maxR
    // 2026-06-03 center-offset diagnostic switch. This was added while
    // investigating visually shifted embryo cells in frames 85~100. It is
    // default-off because previous validated runs rely on the historical PCA
    // center movement. When enabled, PCA can still fit radii and rotation, but
    // suspicious center movement is reverted to the pre-PCA position.
    bool pcaShapeCenterDriftGuardEnabled{false};
    float pcaShapeCenterDriftGuardMaxMove{0.0f};
    float pcaShapeCenterDriftGuardMaxAbsZShift{0.0f};
    float pcaShapeCenterDriftGuardMaxNegativeZShift{0.0f};
    float pcaShapeCenterDriftGuardMaxAbsXYShift{0.0f};
    // Per-pixel intensity exponent in the PCA centroid + covariance moments.
    // Higher values → stronger emphasis on the bright core, smaller fitted
    // radii. Lower values → halo pixels contribute more, larger fitted radii.
    //   exp=1.0: linear (halo-inclusive; bloat).
    //   exp=1.5: balanced (current default) — cell tracks core+inner halo.
    //   exp=2.0: quadratic — suppresses halo ~25× vs core; tighter fits.
    //   exp=3.0+: very strong core emphasis.
    //   exp=0.5: root weighting — approximates equal per-pixel weight.
    // Fast paths detected automatically for 1.0 and 2.0; other values use
    // std::pow per pixel (small overhead).
    float pcaShapeWeightExponent{1.5f};
    // Per-cell adaptive exponent. When enabled, each cell picks its own
    // exponent based on how "core-dominated" its Voronoi-filtered bright
    // pixel cloud is. Rationale: a uniform `exp=1.3` over-shrinks bright
    // cells (core pixels swamp halo by ~10×, fit snaps tight to core);
    // dim cells have near-uniform weights so `exp=1.3` is already correct.
    // Metric: pCore = fraction of raw pixels with weight > coreThreshold.
    // Map: pCore ≤ fracLow → expDim (1.3), pCore ≥ fracHigh → expBright
    // (1.15), linear ramp between. `expDim` is `pcaShapeWeightExponent`.
    // Floor `expBright` at 1.15 — below ~1.2 halo-dominated regime begins
    // (cliff observed at exp=1.1 in run 021558).
    bool pcaShapeAdaptiveExponent{false};
    float pcaShapeWeightExponentBright{1.15f};
    float pcaShapeCoreBrightnessThreshold{0.6f};
    float pcaShapeCoreFractionLow{0.10f};
    float pcaShapeCoreFractionHigh{0.40f};
    // Adaptive output-radius inflation for peaked cells. The PCA variance
    // formula `radii = sqrt(5) * sqrt(variance)` gives the 97% containment
    // radius for a Gaussian-like (peaked) distribution — but real bright
    // cells have visible halo extending to ~3σ (99%+ containment), so the
    // fit looks too tight. Uniform/dim cells are correctly fit by sqrt(5).
    // When adaptive exponent is enabled, we apply the SAME pCore ramp to
    // a per-cell radius multiplier: uniform cells get 1.0× (unchanged),
    // peaked cells get pcaShapeRadiusInflationBright× (e.g. 1.15 for 15%
    // inflation). Overfitted cells (halo-bleed) typically have LOW pCore
    // (halo dilutes the core fraction), so they don't get additional
    // inflation — bounded-growth reference handles their cap separately.
    float pcaShapeRadiusInflationBright{1.15f};
    // Percentile for the radius computation in calibrateCellShapeViaPca.
    // For each PCA axis, the radius = Nth percentile of |projection of
    // bright pixels onto that axis|. Higher = larger radii (captures more
    // halo). Lower = tighter to core. Default 0.90.
    float pcaShapeRadiusPercentile{0.90f};
    bool pcaShapeFitGrowthCapEnabled{true};
    float pcaShapeFitGrowthCap{0.10f};
    bool trashPcaShapeFitEnabled{true};
    float trashPcaShapeMaxOriginalRadiusFactor{2.0f};
    bool trashRemovalEnabled{false};
    float trashRemovalBrightnessThreshold{0.05f};
    // Reference radius for proportional perturbation sigma scaling.
    // positionScale = max(cell.a, cell.b, cell.c) / perturbSigmaReferenceRadius.
    // Cells larger than refR take bigger steps; smaller cells take smaller
    // steps. Set to 0 to disable (all cells use base sigma unchanged).
    float perturbSigmaReferenceRadius{0.0f};
    // Random perturbation radius multiplier. Only the random perturb path
    // applies this to the radius used for position-sigma scaling; fitted
    // cell radii are not modified. 0.5 means random steps behave as if the
    // cell radius were half its current fitted radius.
    float randomPerturbRadiusRatio{0.5f};
    // Random perturb bright-core guidance. After the usual reduced random
    // position step, the candidate is nudged toward the brightest weighted
    // centroid inside the current ellipsoid. This is separate from global
    // signal-center guidance and works on the local enclosed image content.
    bool randomPerturbBrightCoreGuidanceEnabled{true};
    // Keep pixels >= this fraction of the brightest pixel inside the cell
    // when estimating the local bright core. Higher = tighter to peak.
    float randomPerturbBrightCoreThresholdFraction{0.5f};
    // Pixel weight exponent for bright-core centroid. 1=linear, 2=quadratic,
    // larger values pull harder toward the brightest center.
    float randomPerturbBrightCoreWeightExponent{2.0f};
    // Minimum trust given to a detected bright core.
    float randomPerturbBrightCoreBaseTrust{0.35f};
    // Extra trust contributed by black-vs-bright imbalance inside the cell.
    float randomPerturbBrightCoreBlackTrust{0.65f};
    // Extra trust contributed by absolute brightness of the bright core.
    float randomPerturbBrightCoreBrightnessTrust{0.65f};
    // Multiplier on the capped push toward the bright core. 0 disables the
    // added push but still allows damping movement away from the bright core.
    float randomPerturbBrightCoreGuideStrength{0.75f};
    // Maximum valid z position (interpolated z-space). Used to clamp Ellipsoid
    // center z in the constructor, preventing cells from drifting off the z-stack.
    // Default 224 = (z_slices=225) - 1. Runtime-updated by CellUniverse::loadFrame
    // to the actual interpolated slice count - 1. Not parsed from YAML.
    float maxZ{224.0f};
    ~EllipsoidConfig() = default;

    void explodeConfig(const YAML::Node& node)
    {
        x.explodeParams(node["x"]);
        y.explodeParams(node["y"]);
        z.explodeParams(node["z"]);
        thetaX.explodeParams(node["thetaX"]);
        thetaY.explodeParams(node["thetaY"]);
        thetaZ.explodeParams(node["thetaZ"]);
        // Radius + brightness perturbation blocks are optional: PCA drives
        // shape and brightness is EMA-driven.
        if (node["aRadius"]) aRadius.explodeParams(node["aRadius"]);
        if (node["bRadius"]) bRadius.explodeParams(node["bRadius"]);
        if (node["cRadius"]) cRadius.explodeParams(node["cRadius"]);
        if (node["brightness"]) brightness.explodeParams(node["brightness"]);

        minARadius = node["minARadius"].as<double>();
        maxARadius = node["maxARadius"].as<double>();
        if (node["minBRadius"]) minBRadius = node["minBRadius"].as<double>();
        if (node["maxBRadius"]) maxBRadius = node["maxBRadius"].as<double>();
        minCRadius = node["minCRadius"].as<double>();
        maxCRadius = node["maxCRadius"].as<double>();
        if (node["minAnyRadiusEnabled"]) minAnyRadiusEnabled = node["minAnyRadiusEnabled"].as<bool>();
        if (node["minAnyRadius"]) minAnyRadius = node["minAnyRadius"].as<double>();
        if (node["initialBrightness"]) initialBrightness = node["initialBrightness"].as<float>();
        if (node["initialRadiusScale"]) initialRadiusScale = node["initialRadiusScale"].as<float>();
        if (node["backgroundColor"]) backgroundColor = node["backgroundColor"].as<float>();
        if (node["minBrightness"]) minBrightness = node["minBrightness"].as<double>();
        if (node["maxBrightness"]) maxBrightness = node["maxBrightness"].as<double>();
        if (node["brightnessProbabilityStep"]) brightnessProbabilityStep = node["brightnessProbabilityStep"].as<float>();
        if (node["brightnessProbabilityTrust"]) brightnessProbabilityTrust = node["brightnessProbabilityTrust"].as<float>();
        aRadiusProbabilityStep = brightnessProbabilityStep;
        aRadiusProbabilityTrust = brightnessProbabilityTrust;
        bRadiusProbabilityStep = brightnessProbabilityStep;
        bRadiusProbabilityTrust = brightnessProbabilityTrust;
        cRadiusProbabilityStep = brightnessProbabilityStep;
        cRadiusProbabilityTrust = brightnessProbabilityTrust;
        if (node["aRadiusProbabilityStep"]) aRadiusProbabilityStep = node["aRadiusProbabilityStep"].as<float>();
        if (node["aRadiusProbabilityTrust"]) aRadiusProbabilityTrust = node["aRadiusProbabilityTrust"].as<float>();
        if (node["bRadiusProbabilityStep"]) bRadiusProbabilityStep = node["bRadiusProbabilityStep"].as<float>();
        if (node["bRadiusProbabilityTrust"]) bRadiusProbabilityTrust = node["bRadiusProbabilityTrust"].as<float>();
        if (node["cRadiusProbabilityStep"]) cRadiusProbabilityStep = node["cRadiusProbabilityStep"].as<float>();
        if (node["cRadiusProbabilityTrust"]) cRadiusProbabilityTrust = node["cRadiusProbabilityTrust"].as<float>();
        if (node["brightnessUpdateBlend"]) brightnessUpdateBlend = node["brightnessUpdateBlend"].as<float>();
        if (node["brightnessMeanAmplification"]) brightnessMeanAmplification = node["brightnessMeanAmplification"].as<float>();
        if (node["brightnessMeasurementTopPercentile"]) brightnessMeasurementTopPercentile = node["brightnessMeasurementTopPercentile"].as<float>();
        if (node["pcaShapeMaxIters"]) pcaShapeMaxIters = node["pcaShapeMaxIters"].as<int>();
        if (node["pcaShapeRadiusScale"]) pcaShapeRadiusScale = node["pcaShapeRadiusScale"].as<float>();
        if (node["pcaShapeMinPixels"]) pcaShapeMinPixels = node["pcaShapeMinPixels"].as<int>();
        if (node["pcaShapeMaskScale"]) pcaShapeMaskScale = node["pcaShapeMaskScale"].as<float>();
        if (node["pcaShapeConvergeRadius"]) pcaShapeConvergeRadius = node["pcaShapeConvergeRadius"].as<float>();
        if (node["pcaShapeConvergeAngleDeg"]) pcaShapeConvergeAngleDeg = node["pcaShapeConvergeAngleDeg"].as<float>();
        if (node["pcaShapeUpdatePosition"]) pcaShapeUpdatePosition = node["pcaShapeUpdatePosition"].as<bool>();
        if (node["pcaShapeMaxPosShiftFraction"]) pcaShapeMaxPosShiftFraction = node["pcaShapeMaxPosShiftFraction"].as<float>();
        if (node["pcaShapeCenterDriftGuardEnabled"]) pcaShapeCenterDriftGuardEnabled = node["pcaShapeCenterDriftGuardEnabled"].as<bool>();
        if (node["pcaShapeCenterDriftGuardMaxMove"]) pcaShapeCenterDriftGuardMaxMove = node["pcaShapeCenterDriftGuardMaxMove"].as<float>();
        if (node["pcaShapeCenterDriftGuardMaxAbsZShift"]) pcaShapeCenterDriftGuardMaxAbsZShift = node["pcaShapeCenterDriftGuardMaxAbsZShift"].as<float>();
        if (node["pcaShapeCenterDriftGuardMaxNegativeZShift"]) pcaShapeCenterDriftGuardMaxNegativeZShift = node["pcaShapeCenterDriftGuardMaxNegativeZShift"].as<float>();
        if (node["pcaShapeCenterDriftGuardMaxAbsXYShift"]) pcaShapeCenterDriftGuardMaxAbsXYShift = node["pcaShapeCenterDriftGuardMaxAbsXYShift"].as<float>();
        if (node["pcaShapeWeightExponent"]) pcaShapeWeightExponent = node["pcaShapeWeightExponent"].as<float>();
        if (node["pcaShapeAdaptiveExponent"]) pcaShapeAdaptiveExponent = node["pcaShapeAdaptiveExponent"].as<bool>();
        if (node["pcaShapeWeightExponentBright"]) pcaShapeWeightExponentBright = node["pcaShapeWeightExponentBright"].as<float>();
        if (node["pcaShapeCoreBrightnessThreshold"]) pcaShapeCoreBrightnessThreshold = node["pcaShapeCoreBrightnessThreshold"].as<float>();
        if (node["pcaShapeCoreFractionLow"]) pcaShapeCoreFractionLow = node["pcaShapeCoreFractionLow"].as<float>();
        if (node["pcaShapeCoreFractionHigh"]) pcaShapeCoreFractionHigh = node["pcaShapeCoreFractionHigh"].as<float>();
        if (node["pcaShapeRadiusInflationBright"]) pcaShapeRadiusInflationBright = node["pcaShapeRadiusInflationBright"].as<float>();
        if (node["pcaShapeRadiusPercentile"]) pcaShapeRadiusPercentile = node["pcaShapeRadiusPercentile"].as<float>();
        if (node["pcaShapeFitGrowthCapEnabled"]) pcaShapeFitGrowthCapEnabled = node["pcaShapeFitGrowthCapEnabled"].as<bool>();
        if (node["pcaShapeFitGrowthCap"]) pcaShapeFitGrowthCap = node["pcaShapeFitGrowthCap"].as<float>();
        if (node["trashPcaShapeFitEnabled"]) trashPcaShapeFitEnabled = node["trashPcaShapeFitEnabled"].as<bool>();
        if (node["trashPcaShapeMaxOriginalRadiusFactor"]) trashPcaShapeMaxOriginalRadiusFactor = node["trashPcaShapeMaxOriginalRadiusFactor"].as<float>();
        if (node["trashRemovalEnabled"]) trashRemovalEnabled = node["trashRemovalEnabled"].as<bool>();
        if (node["trashRemovalBrightnessThreshold"]) trashRemovalBrightnessThreshold = node["trashRemovalBrightnessThreshold"].as<float>();
        if (node["perturbSigmaReferenceRadius"]) perturbSigmaReferenceRadius = node["perturbSigmaReferenceRadius"].as<float>();
        if (node["randomPerturbRadiusRatio"]) randomPerturbRadiusRatio = node["randomPerturbRadiusRatio"].as<float>();
        if (node["randomPerturbBrightCoreGuidanceEnabled"]) randomPerturbBrightCoreGuidanceEnabled = node["randomPerturbBrightCoreGuidanceEnabled"].as<bool>();
        if (node["randomPerturbBrightCoreThresholdFraction"]) randomPerturbBrightCoreThresholdFraction = node["randomPerturbBrightCoreThresholdFraction"].as<float>();
        if (node["randomPerturbBrightCoreWeightExponent"]) randomPerturbBrightCoreWeightExponent = node["randomPerturbBrightCoreWeightExponent"].as<float>();
        if (node["randomPerturbBrightCoreBaseTrust"]) randomPerturbBrightCoreBaseTrust = node["randomPerturbBrightCoreBaseTrust"].as<float>();
        if (node["randomPerturbBrightCoreBlackTrust"]) randomPerturbBrightCoreBlackTrust = node["randomPerturbBrightCoreBlackTrust"].as<float>();
        if (node["randomPerturbBrightCoreBrightnessTrust"]) randomPerturbBrightCoreBrightnessTrust = node["randomPerturbBrightCoreBrightnessTrust"].as<float>();
        if (node["randomPerturbBrightCoreGuideStrength"]) randomPerturbBrightCoreGuideStrength = node["randomPerturbBrightCoreGuideStrength"].as<float>();
    }
};

class CellLumenConfig {
public:
    // Dataset-specific controls for CellLumen only. These values
    // are intentionally separated from the main optimizer so raw connected
    // component initialization can use biological priors without changing the
    // Cell Universe tracking loop.
    bool enabled = false;
    std::vector<float> percentiles{};
    float geometricCenterBlend = -1.0f;

    int minComponentVoxels = -1;
    int minHighSeedVoxels = -1;
    float seedMergeDistance = -1.0f;
    bool adaptiveSeedMergeEnabled = false;
    int adaptiveSeedMergeModerateHighSeedCount = -1;
    int adaptiveSeedMergeDenseHighSeedCount = -1;
    float adaptiveSeedMergeModerateDistance = -1.0f;
    float adaptiveSeedMergeDenseDistance = -1.0f;
    bool seedMergeUseScaledZ = false;
    float seedSplitSeparation = -1.0f;
    float dedupDistance = -1.0f;
    float dedupRadiusScale = 0.60f;
    bool adaptiveDedupEnabled = false;
    int adaptiveDedupModerateHighSeedCount = -1;
    int adaptiveDedupDenseHighSeedCount = -1;
    float adaptiveDedupModerateDistance = -1.0f;
    float adaptiveDedupModerateRadiusScale = -1.0f;
    float adaptiveDedupDenseDistance = -1.0f;
    float adaptiveDedupDenseRadiusScale = -1.0f;
    bool fragmentMergeEnabled = true;
    int fragmentMergeMaxInputCells = -1;
    float fragmentMergeSparseMinDistance = 30.0f;
    float fragmentMergeSparseMajorScale = 3.10f;
    float fragmentMergeModerateMinDistance = 20.0f;
    float fragmentMergeModerateMajorScale = 1.80f;
    bool allowSeededSplitInSparseField = false;
    float seededSplitMaxMajorRadius = -1.0f;
    float seededSplitMinSeedBalance = 0.28f;
    float seededSplitMinSeedSeparation = -1.0f;
    bool seededSplitUseScaled3DSeparation = false;
    float seededSplitMinScaled3DSeparation = -1.0f;
    float seededSplitMinScaledZSeparation = -1.0f;

    bool useScaledVoxelVolumeForRadius = true;
    float radiusInflationScale = 1.0f;
    float zCenterOffsetScaled = 0.0f;
    float minOutputMajorRadius = -1.0f;
    float minOutputBRadius = -1.0f;
    float minOutputMinorRadius = -1.0f;
    float maxOutputMajorRadius = -1.0f;
    float maxOutputBRadius = -1.0f;
    float maxOutputMinorRadius = -1.0f;
    float minOutputCenterZScaled = -1.0f;
    float maxOutputCenterZScaled = -1.0f;

    bool signalStatsEnabled = false;
    float shellInnerScale = 1.05f;
    float shellOuterScale = 1.35f;
    float minTop10MinusShell = -1.0f;
    float minMeanMinusShell = -1000000.0f;
    int minBiologicalVoxelCount = -1;
    float minBiologicalMajorRadius = -1.0f;
    int smallArtifactSparseMaxCells = -1;
    int smallArtifactMaxVoxels = -1;
    float smallArtifactMaxTop10Intensity = -1.0f;
    float smallArtifactMaxMeanIntensity = -1.0f;
    bool rejectTinyBrightArtifacts = false;

    bool biologicalNearDuplicateMergeEnabled = false;
    float minBiologicalCenterDistance = -1.0f;
    bool adaptiveNearDuplicateMergeEnabled = false;
    int adaptiveNearDuplicateMinCells = -1;
    int adaptiveNearDuplicateMaxCells = -1;
    float adaptiveNearDuplicateDistance = -1.0f;
    float adaptiveNearDuplicateMinPairRatio = -1.0f;
    bool localMaxSeedEnabled = false;
    int localMaxSeedMinStrongSeedCount = -1;
    float localMaxSeedMinDistance = 8.0f;
    float localMaxSeedMaxPerFrameFactor = 2.5f;
    bool useHighSeedCenterForSplitCells = false;

    // Experimental raw intensity seed segmentation path. This keeps the old
    // percentile connected component method available, but lets CellLumen
    // try a marker controlled region grow that is closer to seeded watershed:
    // local background subtraction finds markers, then each marker claims only
    // nearby permissive mask voxels. It is intended for dim or bridged embryo
    // frames where one global threshold either loses cells or merges them.
    bool seededWatershedEnabled = false;
    bool seededWatershedPreferOverPercentile = false;
    bool seededWatershedRescueEnabled = false;
    int seededWatershedPreferBaseCountMin = -1;
    int seededWatershedPreferBaseCountMax = -1;
    float seededWatershedSignalSigma = 1.0f;
    float seededWatershedBackgroundSigma = 12.0f;
    float seededWatershedSeedPercentile = 99.25f;
    float seededWatershedMaskPercentile = 94.0f;
    float seededWatershedSeedThresholdFloor = 0.08f;
    float seededWatershedMaskThresholdFloor = 0.025f;
    float seededWatershedMinSeedDistance = 18.0f;
    float seededWatershedCloseSeedMinDistance = 10.0f;
    float seededWatershedMinValleyDrop = 0.0f;
    float seededWatershedMaxGrowDistance = 30.0f;
    bool seededWatershedUseDistanceTransform = false;
    float seededWatershedDistancePriorityWeight = 1.0f;
    float seededWatershedContrastPriorityWeight = 1.0f;
    float seededWatershedSeedDistancePenaltyWeight = 0.01f;
    float seededWatershedCenterSeedBlend = 0.0f;
    int seededWatershedMinVoxels = 350;
    bool seededWatershedAdaptiveMinVoxelsEnabled = false;
    int seededWatershedAdaptiveMinVoxelsCellCount = 0;
    int seededWatershedAdaptiveDenseMinVoxels = 350;
    int seededWatershedMaxSeeds = 650;
    float seededWatershedRescueMinDistance = 24.0f;
    float seededWatershedRescueMaxAddedFraction = 0.12f;
    int seededWatershedRescueMaxAdded = 40;
    bool seededWatershedLowRescueEnabled = false;
    float seededWatershedLowRescueSeedPercentile = 70.0f;
    int seededWatershedLowRescueMinVoxels = 1000;
    float seededWatershedLowRescueMinSeedDistance = -1.0f;
    float seededWatershedLowRescueCloseSeedMinDistance = -1.0f;
    float seededWatershedLowRescueMinValleyDrop = -1.0f;
    bool seededWatershedLowRescueRefineEnabled = false;
    float seededWatershedLowRescueRefineDistance = 24.0f;
    float seededWatershedLowRescueMinDistance = 40.0f;
    int seededWatershedLowRescueMinCandidateVoxels = 2000;
    float seededWatershedLowRescueMinTop10MinusShell = 72.0f;
    float seededWatershedLowRescueClusterDistance = -1.0f;
    bool seededWatershedLowRescueSortByDistance = false;
    int seededWatershedLowRescueMaxAdded = 16;
    int seededWatershedLowRescueMinBaseCells = -1;
    int seededWatershedLowRescueMaxBaseCells = -1;
    float finalDuplicateMergeDistance = -1.0f;
    bool finalLocalRefineEnabled = false;
    float finalLocalRefineRadius = 24.0f;
    float finalLocalRefineQuantile = 0.75f;
    float finalLocalRefineBlend = 1.0f;
    bool finalZColumnRefineEnabled = false;
    float finalZColumnRefineRadiusXY = 10.0f;
    float finalZColumnRefineHalfWindowScaled = 28.0f;
    float finalZColumnRefineQuantile = 0.75f;
    float finalZColumnRefineMinScoreFraction = 0.65f;
    float finalZColumnRefineMaxMoveScaled = 24.0f;
    float finalZColumnRefineBlend = 1.0f;
    int finalZColumnRefineMinCells = -1;
    int finalZColumnRefineMaxCells = -1;
    bool finalZPeakSplitEnabled = false;
    int finalZPeakSplitMinCells = -1;
    int finalZPeakSplitMaxCells = -1;
    int finalZPeakSplitMaxAdded = 2;
    bool finalZPeakSplitAddOnlyEnabled = false;
    float finalZPeakSplitMinMajorRadius = 22.0f;
    float finalZPeakSplitRadiusXY = 24.0f;
    float finalZPeakSplitQuantile = 0.82f;
    float finalZPeakSplitMinPeakScoreFraction = 0.55f;
    float finalZPeakSplitMaxParentTop10MinusShell = -1.0f;
    bool finalZPeakSplitLocal3DFallbackEnabled = false;
    bool finalZPeakSplitPreferLocal3D = false;
    float finalZPeakSplitLocal3DCentroidRadiusXY = 10.0f;
    float finalZPeakSplitLocal3DCentroidHalfWindowScaled = 14.0f;
    float finalZPeakSplitMinSeparationScaled = 24.0f;
    float finalZPeakSplitMaxSeparationScaled = 70.0f;
    float finalZPeakSplitMaxCenterShiftXY = 45.0f;
    float finalZPeakSplitRadiusScale = 0.78f;
    bool finalBrightPairMidpointRescueEnabled = false;
    int finalBrightPairMidpointRescueMinCells = -1;
    int finalBrightPairMidpointRescueMaxCells = -1;
    int finalBrightPairMidpointRescueMaxAdded = 2;
    float finalBrightPairMidpointRescueMinDistance = 34.0f;
    float finalBrightPairMidpointRescueMaxDistance = 48.0f;
    float finalBrightPairMidpointRescueMinTop10MinusShell = 55.0f;
    float finalBrightPairMidpointRescueRadiusScale = 0.85f;
    float finalBrightPairMidpointRescueMinAddedDistance = 18.0f;
    float finalBrightPairMidpointRescueMinExistingDistance = -1.0f;
    bool finalBrightPairMidpointRescueSortByExistingDistance = false;
    bool finalClusterCentroidRecallRescueEnabled = false;
    int finalClusterCentroidRecallRescueMinCells = -1;
    int finalClusterCentroidRecallRescueMaxCells = -1;
    int finalClusterCentroidRecallRescueMaxAdded = 12;
    int finalClusterCentroidRecallRescueMinClusterSize = 2;
    float finalClusterCentroidRecallRescueClusterDistance = 32.0f;
    float finalClusterCentroidRecallRescueMinAddedDistance = 6.0f;
    float finalClusterCentroidRecallRescueMinTop10MinusShell = -1.0f;
    float finalClusterCentroidRecallRescueRadiusScale = 0.90f;
    bool finalClusterCentroidCollapseEnabled = false;
    int finalClusterCentroidCollapseMinCells = -1;
    int finalClusterCentroidCollapseMaxCells = -1;
    int finalClusterCentroidCollapseMinClusterSize = 2;
    float finalClusterCentroidCollapseLinkDistance = 34.0f;
    float finalClusterCentroidCollapseRadiusScale = 1.0f;
    bool finalClusterCentroidCollapseUseSignalWeights = false;
    bool finalDominatedDuplicateFilterEnabled = false;
    int finalDominatedDuplicateFilterMinCells = -1;
    int finalDominatedDuplicateFilterMaxCells = -1;
    float finalDominatedDuplicateFilterDistance = 10.0f;
    float finalDominatedDuplicateFilterMinVoxelRatio = 2.0f;
    float finalDominatedDuplicateFilterMinSignalRatio = 1.35f;
    float finalDominatedDuplicateFilterMinRadiusRatio = 1.25f;
    float finalDominatedDuplicateFilterMaxLoserTop10MinusShell = -1.0f;
    bool finalDominatedDuplicateFilterRequireTwoSignals = true;
    float finalPostRefineDuplicateMergeDistance = -1.0f;
    bool finalWeakSatelliteFilterEnabled = false;
    float finalWeakSatelliteNeighborDistance = 30.0f;
    int finalWeakSatelliteMaxVoxels = 1300;
    float finalWeakSatelliteMaxTop10MinusShell = 100.0f;
    float finalWeakSatelliteMaxMinorRadius = 14.5f;
    int finalWeakSatelliteNeighborMinVoxels = 2400;
    float finalWeakSatelliteNeighborVoxelRatio = 1.25f;
    bool finalSparseIsolatedFloorFilterEnabled = false;
    int finalSparseIsolatedFloorFilterMaxCells = 16;
    float finalSparseIsolatedFloorMaxMajorRadius = 18.2f;
    float finalSparseIsolatedFloorMaxMinorRadius = 18.2f;
    float finalSparseIsolatedFloorMinNearestDistance = 45.0f;
    int finalSparseIsolatedFloorWeakSignalMinCells = -1;
    int finalSparseIsolatedFloorWeakSignalMaxCells = -1;
    float finalSparseIsolatedFloorMaxTop10MinusShell = -1.0f;

    // Optional fusion path for the main Cell Universe optimizer. CellLumen is
    // used only as a raw frame center rescue source; candidates still have to
    // pass distance, size, and shell contrast gates before becoming real
    // optimizer cells.
    bool fusionEnabled = false;
    int fusionStartFrame = 0;
    int fusionEveryNFrames = 1;
    float fusionMinDistanceToExisting = 24.0f;
    int fusionMinVoxels = 1000;
    float fusionMinTop10MinusShell = 60.0f;
    int fusionMaxAddedPerFrame = 12;
    float fusionRadiusScale = 1.0f;
    float fusionBrightness = -1.0f;
    std::string fusionNamePrefix = "CellLumen_rescue";
    bool fusionRepairCloseCellsEnabled = false;
    float fusionRepairMaxDistance = 18.0f;
    float fusionRepairRadiusRatio = 0.75f;
    float fusionRepairPositionBlend = 0.20f;
    float fusionRepairRadiusBlend = 0.75f;
    // Conservative CellLumen center prior for Cell Universe. This does not add a
    // new cell and does not trust CellLumen as final ground truth. It only nudges
    // an already existing Cell Universe cell toward a nearby high-recall raw
    // intensity center before the normal optimizer starts.
    bool fusionCenterPriorEnabled = false;
    float fusionCenterPriorMaxDistance = 24.0f;
    float fusionCenterPriorPositionBlend = 0.35f;
    float fusionCenterPriorRadiusBlend = 0.25f;
    // CellLumen-guided split prior for Cell Universe. When two raw-intensity
    // CellLumen centers fall near one existing parent, Cell Universe can use
    // those two centers as the only daughter candidate pair. This preserves
    // the normal split validation gates but avoids broad random directions.
    bool fusionSplitPriorEnabled = false;
    bool fusionSplitPriorForceSchedule = true;
    bool fusionSplitPriorSkipRandomSplits = true;
    bool fusionSplitPriorGuidedOnly = true;
    float fusionSplitPriorMaxParentDistance = 42.0f;
    float fusionSplitPriorParentRadiusScale = 2.8f;
    float fusionSplitPriorMinSeparation = 8.0f;
    float fusionSplitPriorMinSeparationRadiusScale = 0.55f;
    // Extra CellLumen-prior-only guard for round parents. If the parent has
    // not become elongated, two very close CellLumen peaks are more likely to
    // be one cell's internal bright structure than two biological daughters.
    // A negative scale disables this guard.
    float fusionSplitPriorRoundParentMaxShape = 1.25f;
    float fusionSplitPriorRoundParentMinSeparationRadiusScale = -1.0f;
    float fusionSplitPriorMaxSeparation = 58.0f;
    float fusionSplitPriorMaxSeparationRadiusScale = 3.4f;
    float fusionSplitPriorMaxMidpointDistance = 18.0f;
    float fusionSplitPriorMaxMidpointRadiusScale = 0.80f;
    float fusionSplitPriorMaxScore = 18.0f;
    // Split priors need higher recall than rescue-cell insertion. If these are
    // negative, the older fusionMin* rescue thresholds are reused.
    int fusionSplitPriorMinVoxels = -1;
    float fusionSplitPriorMinTop10MinusShell = -1.0f;
    // Score rescue for elongated parents. A parent that PCA already fits as a
    // long ellipsoid is biologically more likely to be entering mitosis, so a
    // CellLumen pair near that parent should not be discarded by the same
    // global score cutoff used for round mature cells.
    bool fusionSplitPriorElongatedParentRescueEnabled = false;
    float fusionSplitPriorElongatedParentMinShape = 1.55f;
    float fusionSplitPriorElongatedParentMaxScore = 42.0f;
    float fusionSplitPriorElongatedParentScoreBonus = 6.0f;
    // Elongated parents can be rescued past the normal score cutoff, but not
    // when the proposed daughter centers are clearly claimed by neighboring
    // continuing cells. A negative value disables this extra guard.
    float fusionSplitPriorElongatedParentMaxNeighborClaimPenalty = -1.0f;
    // Scoring weights for assigning two CellLumen centers to one parent. Dense
    // embryo frames can put a real daughter closer to a neighboring parent, so
    // the score must balance midpoint distance with biologically plausible
    // daughter separation instead of only choosing the nearest midpoint.
    float fusionSplitPriorMidpointWeight = 1.0f;
    float fusionSplitPriorSeparationPenaltyWeight = 0.20f;
    float fusionSplitPriorSignalBonusWeight = 0.001f;
    bool fusionSplitPriorConflictReplacementEnabled = false;
    float fusionSplitPriorConflictCloseParentRadiusScale = 0.45f;
    float fusionSplitPriorConflictMinNewSeparationRadiusScale = 1.40f;
    float fusionSplitPriorConflictMinSeparationDrop = 15.0f;
    float fusionSplitPriorConflictMinOldScoreForReplacement = 10.0f;
    bool fusionSplitPriorSuppressConflictPenaltiesForAsymmetricClose = false;
    int fusionSplitPriorMaxPriorsPerFrame = 18;
    bool fusionSplitPriorWindowEnabled = false;
    int fusionSplitPriorWindowSize = 1;
    float fusionSplitPriorWindowMatchDistance = 28.0f;
    float fusionSplitPriorWindowMatchDistancePerFrame = 4.0f;
    float fusionSplitPriorWindowDaughterSupportBonus = 4.0f;
    float fusionSplitPriorWindowMissingDaughterPenalty = 5.0f;
    float fusionSplitPriorWindowParentPersistencePenalty = 6.0f;
    float fusionSplitPriorWindowBalancedDaughterBonus = 0.0f;
    float fusionSplitPriorWindowBalancedMinParentDistanceBalance = 0.65f;
    float fusionSplitPriorCleanTwoRealWindowPairMinParentDistanceBalance = 0.75f;
    float fusionSplitPriorCleanTwoRealWindowPairMaxScore = 8.0f;
    float fusionSplitPriorWindowBackedConflictSelectionBonus = 0.0f;
    float fusionSplitPriorWindowBalancedMinNearParentRadiusScale = 0.60f;
    float fusionSplitPriorWindowHighConfidenceMaxOverlapCostFraction = -1.0f;
    bool fusionSplitPriorRankingSoftGateEnabled = false;
    float fusionSplitPriorRankingSoftSeparationPenalty = 12.0f;
    float fusionSplitPriorRankingSoftMidpointPenalty = 10.0f;
    float fusionSplitPriorRankingSoftScorePenalty = 1.0f;
    float fusionSplitPriorRankingSoftNeighborPenalty = 1.0f;
    float fusionSplitPriorGlobalSelectMaxCost = 0.0f;
    // Default-off rescue for very early embryo frames where true daughters can
    // move far while CellLumen still sees a residual parent-like bright spot.
    bool fusionSplitPriorEarlyLargeSeparationRescueEnabled = false;
    bool fusionSplitPriorEarlyLargeSeparationRescueFirstFrameOnly = true;
    int fusionSplitPriorEarlyLargeSeparationRescueMaxFrame = -1;
    float fusionSplitPriorEarlyLargeSeparationMinSeparation = 44.0f;
    float fusionSplitPriorEarlyLargeSeparationMinParentDistanceBalance = 0.25f;
    float fusionSplitPriorEarlyLargeSeparationMaxScore = 90.0f;
    int fusionSplitPriorEarlyLargeSeparationMinWindowBoth = 1;
    int fusionSplitPriorEarlyLargeSeparationMaxWindowMissing = 1;
    int fusionSplitPriorEarlyLargeSeparationMaxWindowParentPersists = 1;
    // Default-off, frame-agnostic temporal repair path. This promotes a split
    // only when future CellLumen centers keep supporting two daughters while
    // the old parent center does not persist. It replaces frame-number based
    // early rescue tuning for long-window validation runs.
    bool fusionSplitPriorTemporalRepairEnabled = false;
    int fusionSplitPriorTemporalRepairMinWindowBoth = 2;
    int fusionSplitPriorTemporalRepairMaxWindowMissing = 0;
    int fusionSplitPriorTemporalRepairMaxWindowParentPersists = 0;
    float fusionSplitPriorTemporalRepairMinSeparation = 0.0f;
    float fusionSplitPriorTemporalRepairMinSeparationRadiusScale = 1.45f;
    float fusionSplitPriorTemporalRepairMaxParentDistance = 0.0f;
    float fusionSplitPriorTemporalRepairParentRadiusScale = 0.0f;
    float fusionSplitPriorTemporalRepairMaxCandidateSeparation = -1.0f;
    float fusionSplitPriorTemporalRepairMaxCandidateSeparationRadiusScale = 4.8f;
    float fusionSplitPriorTemporalRepairMinParentDistanceBalance = 0.30f;
    float fusionSplitPriorTemporalRepairMinNearParentDistance = 6.0f;
    float fusionSplitPriorTemporalRepairMaxScore = 90.0f;
    float fusionSplitPriorTemporalRepairSelectionBonus = 95.0f;
    float fusionSplitPriorTemporalRepairMaxNeighborClaimPenalty = 1e-5f;
    float fusionSplitPriorTemporalRepairMaxContinuationClaimSoftPenalty = 1e-5f;
    bool fusionSplitPriorTemporalRepairRejectClaimBlockers = true;
    int fusionSplitPriorTemporalRepairStrongMinVoxels = 1500;
    float fusionSplitPriorTemporalRepairStrongMinSignal = 120.0f;
    float fusionSplitPriorTemporalRepairStrongMaxNeighborClaimPenalty = 25.0f;
    float fusionSplitPriorTemporalRepairStrongMinParentDistanceBalance = 0.25f;
    float fusionSplitPriorTemporalRepairStrongAsymmetryBalance = 0.55f;
    int fusionSplitPriorTemporalRepairMinParentAgeFrames = 0;
    int fusionSplitPriorMinParentAgeFrames = 0;
    float fusionSplitPriorParentAnchorMinShape = 1.45f;
    // Default-off continuation center repair. Unlike the older global lumen
    // assignment, this only moves an existing cell to a current CellLumen
    // center when a future window keeps supporting the new center and stops
    // supporting the old position.
    bool fusionTemporalCenterRepairEnabled = false;
    float fusionTemporalCenterRepairMinDistance = 24.0f;
    float fusionTemporalCenterRepairMaxDistance = 48.0f;
    int fusionTemporalCenterRepairMinWindowSupport = 2;
    int fusionTemporalCenterRepairMaxOldWindowSupport = 0;
    float fusionTemporalCenterRepairMinWindowDistanceGain = 6.0f;
    float fusionTemporalCenterRepairClaimMargin = 4.0f;
    int fusionTemporalCenterRepairMinCellAgeFrames = 2;
    int fusionTemporalCenterRepairOldSupportedMinVoxels = -1;
    float fusionTemporalCenterRepairOldSupportedMinSignal = -1.0f;
    bool fusionTemporalCenterRepairParentAnchorReanchorEnabled = false;
    float fusionTemporalCenterRepairParentAnchorReanchorMinShape = 2.0f;
    float fusionTemporalCenterRepairParentAnchorReanchorMinDistance = 12.0f;
    bool fusionSplitPriorContinuationClaimGuardEnabled = false;
    float fusionSplitPriorContinuationClaimRadiusScale = 1.15f;
    float fusionSplitPriorContinuationClaimTieMargin = 4.0f;
    float fusionSplitPriorContinuationClaimCloseParentRadiusScale = 0.60f;
    float fusionSplitPriorContinuationClaimCloseParentPenalty = 0.0f;
    bool fusionSplitPriorAllowWindowBackedDuplicateHandoff = false;
    float fusionSplitPriorParentAnchorCleanFutureDriftRescueMinImageGain = 0.0f;
    float fusionSplitPriorDuplicateHandoffMinContinuationPenalty = 15.0f;
    float fusionSplitPriorDuplicateHandoffNormLimit = 0.75f;
    float fusionSplitPriorDuplicateHandoffDistanceLimit = 32.0f;
    bool fusionStaleContinuationPruneEnabled = false;
    float fusionStaleContinuationPruneMaxCurrentDistance = 24.0f;
    int fusionStaleContinuationPruneMaxFutureSupport = 0;
    int fusionStaleContinuationPruneMinCellAgeFrames = 1;
    bool fusionStaleContinuationPruneReanchorParentAnchorOneRealEnabled = false;
    int fusionStaleContinuationPruneReanchorMinWindowBoth = 1;
    int fusionStaleContinuationPruneReanchorMaxWindowMissing = 1;
    int fusionStaleContinuationPruneReanchorMaxWindowParentPersists = 0;
    int fusionStaleContinuationPruneReanchorMinVoxels = 1000;
    float fusionStaleContinuationPruneReanchorMinSignal = 100.0f;
    float fusionStaleContinuationPruneReanchorMinMove = 8.0f;
    bool fusionStaleContinuationPruneLowQualityEnabled = false;
    float fusionStaleContinuationPruneLowQualityMaxBrightness = 0.12f;
    float fusionStaleContinuationPruneLowQualityMaxMajorRadius = 8.0f;
    float fusionStaleContinuationPruneLowQualityMaxMinorRadius = 5.5f;
    int fusionStaleContinuationPruneLowQualityMaxCellAgeFrames = 2;
    bool fusionStaleContinuationPruneLowQualityIgnoresProtection = false;
    bool fusionStaleContinuationPruneClaimedCenterEnabled = false;
    float fusionStaleContinuationPruneClaimedCenterMargin = 4.0f;
    bool fusionStaleContinuationPruneClaimedCenterRequireGuardOrNoFuture = true;
    bool fusionStaleContinuationPruneGuardedNoFutureIgnoresCurrentCenterEnabled = false;
    bool fusionStaleContinuationPruneCrowdedBridgeEnabled = false;
    float fusionStaleContinuationPruneCrowdedBridgeMaxNeighborDistance = 30.0f;
    float fusionStaleContinuationPruneCrowdedBridgeMinBrightnessAdvantage = 0.03f;
    float fusionStaleContinuationPruneCrowdedBridgeMinGuardedBrightness = 0.0f;
    float fusionStaleContinuationPruneCrowdedBridgeMaxGuardedBrightness = 0.95f;
    bool fusionStaleContinuationPruneProtectedOneRealSyntheticEnabled = false;
    float fusionStaleContinuationPruneProtectedOneRealSyntheticMaxBrightness = 0.55f;
    float fusionStaleContinuationPruneProtectedOneRealSyntheticMaxMajorRadius = 12.5f;
    float fusionStaleContinuationPruneProtectedOneRealSyntheticMaxMinorRadius = 5.5f;
    float fusionStaleContinuationPruneProtectedOneRealSyntheticMaxSiblingDistance = 20.0f;
    float fusionStaleContinuationPruneProtectedOneRealSyntheticMaxIndependentCenterDistance = 18.0f;
    float fusionSplitPriorParentAnchorContinuationMergeMinShape = -1.0f;
    float fusionSplitPriorParentAnchorContinuationMergeNormLimit = 0.72f;
    float fusionSplitPriorParentAnchorContinuationMergeDistanceLimit = 30.0f;
    float fusionSplitPriorParentAnchorContinuationMergeLineageBonus = 0.0f;
    bool fusionSplitPriorParentAnchorContinuationMergeRejectClaimConflict = false;
    bool fusionTemporalCenterRepairAllowSharedClaimPcaRelax = false;
    float fusionTemporalCenterRepairSharedClaimPcaRelaxMinShape = 1.45f;
    bool fusionSplitPriorPartialParentAnchorRescueEnabled = false;
    float fusionSplitPriorPartialParentAnchorMinShape = 1.75f;
    float fusionSplitPriorPartialParentAnchorMaxScore = 50.0f;
    int fusionSplitPriorPartialParentAnchorMinRealVoxels = 1500;
    float fusionSplitPriorPartialParentAnchorMinRealSignal = 120.0f;
    int fusionSplitPriorPartialParentAnchorMinWindowBoth = 1;
    int fusionSplitPriorPartialParentAnchorMaxWindowMissing = 1;
    int fusionSplitPriorPartialParentAnchorMaxWindowParentPersists = 1;
    bool fusionSplitPriorParentAnchorWeakGainDuplicateRescueEnabled = false;
    float fusionSplitPriorParentAnchorWeakGainCleanMinShape = 1.60f;
    float fusionSplitPriorParentAnchorWeakGainPartialMinShape = 1.70f;
    float fusionSplitPriorParentAnchorWeakGainCleanMaxScore = 8.0f;
    float fusionSplitPriorParentAnchorWeakGainPartialMaxScore = 50.0f;
    float fusionSplitPriorParentAnchorWeakGainCleanMaxOverlapCostFraction = 0.25f;
    float fusionSplitPriorParentAnchorWeakGainPartialMaxOverlapCostFraction = 1.05f;
    float fusionSplitPriorParentAnchorWeakGainCleanMinImageGain = 0.0f;
    float fusionSplitPriorParentAnchorWeakGainPartialMinImageGain = 0.0f;
    float fusionSplitPriorParentAnchorWeakGainMaxBridgeValleyRatio = 0.95f;
    bool fusionSplitPriorParentAnchorWeakGainPartialRequireConflictEvidence = true;
    bool fusionSplitPriorParentAnchorCompactPositiveOneRealRescueEnabled = false;
    float fusionSplitPriorParentAnchorCompactPositiveOneRealMaxParentShape = 1.35f;
    float fusionSplitPriorParentAnchorCompactPositiveOneRealMinPriorScore = 0.0f;
    float fusionSplitPriorParentAnchorCompactPositiveOneRealMaxPriorScore = 4.0f;
    float fusionSplitPriorParentAnchorCompactPositiveOneRealMinImageGain = 4200.0f;
    float fusionSplitPriorParentAnchorCompactPositiveOneRealMaxOverlapCostFraction = 0.08f;
    float fusionSplitPriorParentAnchorCompactPositiveOneRealMaxBridgeValleyRatio = 0.95f;
    bool fusionSplitPriorParentAnchorOneRealPostRefitGuardEnabled = false;
    float fusionSplitPriorParentAnchorOneRealMaxRefitDrift = -1.0f;
    bool fusionSplitPriorParentAnchorOneRealRefitDriftRescueEnabled = false;
    float fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinImageGain = 20000.0f;
    float fusionSplitPriorParentAnchorOneRealRefitDriftRescueMaxBridgeValleyRatio = 0.35f;
    float fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinBridgeGapWidth = 8.0f;
    float fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinParentShape = 1.80f;
    float fusionSplitPriorParentAnchorOneRealRefitDriftRescueMaxScore = 80.0f;
    float fusionSplitPriorParentAnchorOneRealCleanHighOverlapMinCost = -1.0f;
    float fusionSplitPriorParentAnchorOneRealCleanHighOverlapMinImageGain = -1.0f;
    float fusionSplitPriorParentAnchorOneRealCleanHighOverlapMaxBridgeValleyRatio = -1.0f;
    bool fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateEnabled = false;
    float fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinOverlapCost = 10000.0f;
    float fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMaxBridgeGapWidth = 0.0f;
    float fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinBridgeValleyRatio = 0.55f;
    float fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMaxParentShape = 1.75f;
    float fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinImageGain = 0.0f;
    bool fusionSplitPriorParentAnchorOneRealSeedLockOnRefitCollapse = false;
    float fusionSplitPriorParentAnchorOneRealSeedLockMaxScore = 8.0f;
    float fusionSplitPriorParentAnchorOneRealSeedLockMinSeedSeparation = 14.0f;
    float fusionSplitPriorParentAnchorOneRealSeedLockMaxFinalSeedAxisRatio = 0.75f;
    bool fusionSplitPriorParentAnchorOneRealMinImageGainGuardEnabled = false;
    float fusionSplitPriorParentAnchorOneRealCleanMinImageGain = 0.0f;
    float fusionSplitPriorParentAnchorOneRealPartialMinImageGain = 0.0f;
    bool fusionSplitPriorParentAnchorOneRealPositiveWindowRescueEnabled = false;
    float fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMinShape = 1.45f;
    float fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxShape = 999.0f;
    float fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxScore = 4.0f;
    float fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxImageWorsening = 3500.0f;
    float fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxTotalWorsening = 10000.0f;
    float fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxOverlapCostFraction = 0.05f;
    float fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxSoftPenaltyFraction = 0.01f;
    float fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxBridgeValleyRatio = 0.65f;
    int fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMinRealVoxels = 1000;
    bool fusionSplitPriorCompactParentAnchorWindowRescueEnabled = false;
    float fusionSplitPriorCompactParentAnchorMinShape = 1.60f;
    float fusionSplitPriorCompactParentAnchorMaxScore = 0.0f;
    int fusionSplitPriorCompactParentAnchorMinRealVoxels = 1500;
    float fusionSplitPriorCompactParentAnchorMinRealSignal = 120.0f;
    float fusionSplitPriorCompactParentAnchorMinFarDistanceRadiusScale = 0.95f;
    bool fusionSplitPriorParentAnchorSingleBlockerRescueEnabled = false;
    float fusionSplitPriorParentAnchorSingleBlockerMinShape = 1.80f;
    float fusionSplitPriorParentAnchorSingleBlockerMaxScore = 80.0f;
    int fusionSplitPriorParentAnchorSingleBlockerMinRealVoxels = 1500;
    float fusionSplitPriorParentAnchorSingleBlockerMinRealSignal = 120.0f;
    float fusionSplitPriorParentAnchorSingleBlockerMinFarDistanceRadiusScale = 1.65f;
    float fusionSplitPriorParentAnchorSingleBlockerMaxContinuationPenalty = 90.0f;
    float fusionSplitPriorParentAnchorSingleBlockerSelectionBonus = 85.0f;
    bool fusionSplitPriorLockAcceptedDaughtersForFinalPca = false;
    bool fusionSplitPriorCleanTwoRealSingleBlockerRescueEnabled = false;
    float fusionSplitPriorCleanTwoRealSingleBlockerMinParentDistanceBalance = 0.80f;
    float fusionSplitPriorCleanTwoRealSingleBlockerMaxScore = 8.0f;
    int fusionSplitPriorCleanTwoRealSingleBlockerMinVoxels = 1800;
    float fusionSplitPriorCleanTwoRealSingleBlockerMinSignal = 120.0f;
    float fusionSplitPriorCleanTwoRealSingleBlockerSelectionBonus = 35.0f;
    bool fusionSplitPriorCleanTwoRealAsymmetricRescueEnabled = false;
    float fusionSplitPriorCleanTwoRealAsymmetricMinParentDistanceBalance = 0.45f;
    float fusionSplitPriorCleanTwoRealAsymmetricMaxScore = 18.0f;
    int fusionSplitPriorCleanTwoRealAsymmetricMinVoxels = 1500;
    float fusionSplitPriorCleanTwoRealAsymmetricMinSignal = 120.0f;
    float fusionSplitPriorCleanTwoRealAsymmetricMaxNeighborClaimPenalty = 12.0f;
    float fusionSplitPriorCleanTwoRealAsymmetricSelectionBonus = 24.0f;
    bool fusionSplitPriorCleanTwoRealCompactRescueEnabled = false;
    float fusionSplitPriorCleanTwoRealCompactMinParentDistanceBalance = 0.45f;
    float fusionSplitPriorCleanTwoRealCompactMaxScore = 18.0f;
    int fusionSplitPriorCleanTwoRealCompactMinVoxels = 1400;
    float fusionSplitPriorCleanTwoRealCompactMinSignal = 120.0f;
    float fusionSplitPriorCleanTwoRealCompactMaxParentPersistencePenalty = 20.0f;
    float fusionSplitPriorCleanTwoRealCompactMinNearParentDistance = 6.0f;
    float fusionSplitPriorCleanTwoRealCompactMinSeparationRadiusScale = 1.0f;
    float fusionSplitPriorCleanTwoRealCompactMaxMidpointDistance = 10.0f;
    float fusionSplitPriorCleanTwoRealCompactSelectionBonus = 30.0f;
    bool fusionSplitPriorCleanTwoRealRescueBeforeDefaultNegativeGate = false;
    float fusionSplitPriorParentAnchorOneRealSharedCandidatePenalty = 0.0f;
    bool fusionSplitPriorCleanTwoRealHighNeighborClaimRescueEnabled = false;
    float fusionSplitPriorCleanTwoRealHighNeighborClaimMinParentDistanceBalance = 0.65f;
    float fusionSplitPriorCleanTwoRealHighNeighborClaimMaxScore = 60.0f;
    int fusionSplitPriorCleanTwoRealHighNeighborClaimMinVoxels = 1500;
    float fusionSplitPriorCleanTwoRealHighNeighborClaimMinSignal = 120.0f;
    float fusionSplitPriorCleanTwoRealHighNeighborClaimMinNeighborClaimPenalty = 12.0f;
    float fusionSplitPriorCleanTwoRealHighNeighborClaimMaxContinuationClaimPenalty = 1e-5f;
    float fusionSplitPriorCleanTwoRealHighNeighborClaimSelectionBonus = 90.0f;
    bool fusionSplitPriorCleanTwoRealDuplicateBypassEnabled = false;
    float fusionSplitPriorCleanTwoRealDuplicateBypassMinImageGain = 0.0f;
    float fusionSplitPriorCleanTwoRealDuplicateBypassMaxBridgeValleyRatio = 0.70f;
    float fusionSplitPriorCleanTwoRealDuplicateBypassMinBridgeGapWidth = 0.0f;
    float fusionSplitPriorCleanTwoRealDuplicateBypassMinParentDistanceBalance = 0.80f;
    float fusionSplitPriorCleanTwoRealDuplicateBypassMaxScore = 8.0f;
    bool fusionSplitPriorCleanTwoRealCompactDuplicateBypassEnabled = false;
    float fusionSplitPriorCleanTwoRealCompactDuplicateBypassMinImageGain = 3500.0f;
    float fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxBridgeValleyRatio = 0.90f;
    float fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxOverlapCostFraction = 0.16f;
    float fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxSoftPenaltyFraction = 0.012f;
    float fusionSplitPriorCleanTwoRealCompactDuplicateBypassMinParentDistanceBalance = 0.45f;
    float fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxScore = 18.0f;
    bool fusionSplitPriorRejectLowShapeZDominantPairs = false;
    float fusionSplitPriorLowShapeZDominantMaxParentShape = 1.45f;
    bool fusionSplitPriorRejectNonWindowLowShapeOverlapDuplicate = false;
    float fusionSplitPriorNonWindowLowShapeMaxParentShape = 1.45f;
    float fusionSplitPriorNonWindowLowShapeMinOverlapCost = 8000.0f;
    float fusionSplitPriorNonWindowLowShapeMaxBridgeGapWidth = 0.0f;
    bool fusionSplitPriorReanchorNonParentDuplicateToOtherDaughter = false;
    float fusionSplitPriorNonParentDuplicateReanchorMinImageGain = 5000.0f;
    float fusionSplitPriorNonParentDuplicateReanchorMaxBridgeValleyRatio = 0.50f;
    float fusionSplitPriorNonParentDuplicateReanchorMinBridgeGapWidth = 8.0f;
    float fusionSplitPriorNonParentDuplicateReanchorMinMove = 12.0f;
    float fusionSplitPriorNonParentDuplicateReanchorMinParentDistanceBalance = 0.60f;
    // A false CellLumen split often looks like "the old parent center is still
    // present, plus one nearby neighbor." These terms keep the prior score from
    // treating that pattern as a real division while still allowing asymmetric
    // true daughter pairs in dense embryo frames.
    float fusionSplitPriorMinDaughterParentDistance = 6.0f;
    float fusionSplitPriorMinDaughterParentDistanceRadiusScale = 0.35f;
    float fusionSplitPriorMinParentDistanceBalance = 0.35f;
    float fusionSplitPriorParentPersistencePenalty = 8.0f;
    float fusionSplitPriorNeighborClaimPenalty = 3.0f;
    float fusionSplitPriorNeighborClaimMargin = 2.0f;
    // Optional hard cap after the neighbor-claim score is computed. Use this
    // when a proposed daughter pair is close enough to a continuing neighbor
    // that the pair should not enter ranked split-prior selection at all.
    float fusionSplitPriorMaxNeighborClaimPenalty = -1.0f;
    int fusionSplitPriorDebugTopN = 25;
    bool fusionSplitPriorFallbackToClassicOnReject = true;
    bool fusionSplitPriorUseDedicatedCostGate = true;
    bool fusionSplitPriorUseImageCostGate = false;
    float fusionSplitPriorCost = 0.0f;
    float fusionSplitPriorCostFraction = 0.0f;
    float fusionSplitPriorMaxPositiveCostFraction = 0.012f;
    // A positive Lumen gate allowance is only trusted when the raw image
    // improvement is strong enough to justify the geometry/overlap penalty.
    // This keeps true dense-tissue splits from being rejected by overlap
    // alone, while blocking weak duplicate splits whose image gain is tiny.
    float fusionSplitPriorPositiveGateMinImageGain = 0.0f;
    float fusionSplitPriorPositiveGateMinImageGainPenaltyRatio = 0.0f;
    // Some true CellLumen splits slightly worsen raw image cost after PCA refit
    // because daughters are close and the raw embryo background is not black.
    // Only allow that small worsening when the previous parent is elongated
    // enough to look division-ready and the soft geometry penalty is small.
    float fusionSplitPriorPositiveGateElongatedParentMinShape = -1.0f;
    float fusionSplitPriorPositiveGateElongatedMaxRawWorsening = -1.0f;
    float fusionSplitPriorPositiveGateElongatedMaxSoftPenaltyFraction = -1.0f;
    float fusionSplitPriorPositiveGateElongatedMaxScore = -1.0f;
    // When the CellLumen path evaluates raw image cost only, a false split can
    // still look good in the image term while producing a huge overlap penalty.
    // Keep this as a separate raw-fusion gate so heavy-preprocess split_cost
    // behavior is not changed.
    float fusionSplitPriorMaxOverlapCostFraction = -1.0f;
    float fusionSplitPriorHighConfidenceMaxScore = 10.0f;
    float fusionSplitPriorHighConfidenceMaxOverlapCostFraction = -1.0f;
    // CellLumen prior centers are image-derived, not random split directions.
    // A low-score prior may be valid even when the daughter PCA short axes do
    // not align with the previous parent short axis. Negative keeps the normal
    // probability.split_axis_alignment_* behavior.
    float fusionSplitPriorHighConfidenceAxisAlignmentDegrees = -1.0f;
    float fusionSplitPriorDaughterVolumeScale = 0.68f;
    float fusionSplitPriorPrefilterMaxValleyRatio = 1.15f;
    float fusionSplitPriorBridgeMaxValleyRatio = 1.15f;
    // Optional CellLumen-prior-only bridge geometry gate. A negative value
    // disables it. When enabled, the two proposed daughter ellipsoids must
    // leave at least this much surface gap along the split axis; this prevents
    // splitting one bright cell core into two heavily overlapping daughters.
    float fusionSplitPriorMinBridgeGapWidth = -1.0f;
    float fusionSplitPriorMinEdgeBrightness = 0.025f;
    float fusionSplitPriorMaxDaughterOverlapFraction = 0.20f;
    // Soft gate mode for CellLumen split priors. These gates are evaluated
    // after daughter PCA refit. They should not reject a strong raw-intensity
    // split just because one geometry metric is slightly outside a fixed
    // threshold; instead they add a cost penalty and let the final raw image
    // gate decide. The hard limits below remain as biological impossibility
    // checks for extreme overlap or no visible valley.
    bool fusionSplitPriorSoftGateEnabled = false;
    float fusionSplitPriorSoftAxisPenaltyFraction = 0.03f;
    float fusionSplitPriorSoftDaughterOverlapPenaltyFraction = 0.12f;
    float fusionSplitPriorSoftValleyPenaltyFraction = 0.08f;
    float fusionSplitPriorSoftBridgeGapPenaltyFraction = 0.04f;
    float fusionSplitPriorSoftOverlapCostPenaltyWeight = 0.50f;
    bool fusionSplitPriorBridgeEvidenceWaivesOverlapSoftPenalty = false;
    float fusionSplitPriorHardMaxDaughterOverlapFraction = 0.80f;
    float fusionSplitPriorHardMaxValleyRatio = 2.0f;
    float fusionSplitPriorHardMaxOverlapCostFraction = 1.20f;
    float fusionSplitPriorMinPostRefitLateralSeparation = -1.0f;
    float fusionSplitPriorMinPostRefitLateralSeparationRadiusScale = 0.0f;
    float fusionSplitPriorMaxZDominanceForLowLateralSeparation = 0.85f;
    // Same-parent split-prior tie-breaker. When two CellLumen pairs have
    // nearly identical ranking scores, prefer the pair whose daughter axis is
    // less like a same-XY z-stack duplicate.
    float fusionSplitPriorZDominantTieBreakScoreMargin = 0.0f;
    float fusionSplitPriorZDominantTieBreakMinZDominance = 0.85f;
    float fusionSplitPriorZDominantTieBreakMaxLateralRadiusScale = 0.85f;
    bool fusionSplitPriorDynamicOverlapEnabled = false;
    float fusionSplitPriorLocalDensityRadiusScale = 2.50f;
    float fusionSplitPriorLocalDensityOverlapBonus = 0.035f;
    float fusionSplitPriorMaxDynamicDaughterOverlapFraction = 0.65f;
    bool fusionDensityMetricsEnabled = false;
    bool fusionDensityAdaptiveGateEnabled = false;
    bool fusionBrightnessAdaptiveGateEnabled = false;
    float fusionDensityMetricsRadiusScale = 2.50f;
    int fusionDensityMetricsK = 3;
    float fusionBrightnessMetricsRadiusScale = 1.0f;
    float fusionBrightnessMetricsShellScale = 1.55f;
    bool fusionSplitPriorSkipExistingCellBuriedCheck = true;
    bool fusionSplitPriorSkipNeighborBridgeCheck = false;
    int fusionSplitPriorBurnInIterations = 6;
    int fusionSplitPriorRefineIterations = 8;
    bool fusionSplitPriorPrepassFallbackEnabled = false;
    int fusionSplitPriorPrepassFallbackMaxPriors = 0;
    int fusionSplitPriorPrepassFallbackMinKeptPixels = 50000;
    float fusionSplitPriorPrepassFallbackMinShape = 1.20f;
    float fusionSplitPriorPrepassFallbackMinSeparationRadiusScale = 0.90f;
    float fusionSplitPriorPrepassFallbackMaxSeparationRadiusScale = 1.05f;
    float fusionSplitPriorPrepassFallbackParentClaimMargin = 8.0f;
    float fusionSplitPriorPrepassFallbackMaxScore = 8.0f;
    // The pre-pass fallback should only fill true CellLumen blind spots. If
    // CellLumen already found a parent pair but ranked it as unsafe because the
    // centers are better claimed by neighbors, do not resurrect that parent
    // through fallback.
    bool fusionSplitPriorPrepassFallbackRejectBadLumenParent = false;
    float fusionSplitPriorPrepassFallbackBadLumenMaxScore = -1.0f;
    float fusionSplitPriorPrepassFallbackBadLumenMaxNeighborClaimPenalty = -1.0f;
    int fusionSplitPriorPrepassFallbackMinParentAgeFrames = 0;
    // If image-grounded PCA pulls one daughter far away from the original
    // snapshot split seeds, the PCA direction is probably contaminated by a
    // neighboring bright cloud. In that narrow case we can fall back to the
    // original snapshot seeds, then let the normal split path's burn-in,
    // PCA refit, valley, and cost gates decide. This is only for true
    // CellLumen blind spots; it does not reopen classic random split search.
    bool fusionSplitPriorPrepassFallbackUseSnapshotSeedOnLargeDrift = false;
    float fusionSplitPriorPrepassFallbackSeedMaxShift = 25.0f;
    float fusionSplitPriorPrepassFallbackSeedMinSeparationRadiusScale = 0.55f;
    float fusionSplitPriorPrepassFallbackSeedMaxSeparationRadiusScale = 0.95f;
    float fusionSplitPriorPrepassFallbackSeedMaxScore = 16.0f;
    float fusionSplitPriorPrepassFallbackSeedMinShape = 1.40f;
    // Snapshot-seed fallback is only trusted when daughter PCA refit remains
    // near the original snapshot geometry. Negative disables this gate.
    float fusionSplitPriorSnapshotSeedMaxRefitDrift = -1.0f;
    bool fusionReducePostSplitPerturbEnabled = false;
    int fusionPostSplitPerturbItersPerCell = 2;

    float maxCandidateMeanVoxelCount = -1.0f;
    float candidateCellCountWeight = 0.30f;
    float candidateMeanVoxelPenaltyWeight = 1.0f;
    float candidateNearPairPenaltyWeight = 1.50f;
    float candidatePercentilePreferenceWeight = 0.0f;
    int candidateOvergrowthAlwaysBelowFirstCount = -1;
    int candidateOvergrowthMinFirstCount = -1;
    float maxCandidateCountGrowthFromFirst = -1.0f;
    float candidateOvergrowthPenaltyWeight = 0.0f;

    void explodeConfig(const YAML::Node& node) {
        if (!node) return;
        if (node["enabled"]) enabled = node["enabled"].as<bool>();
        if (node["percentiles"] && node["percentiles"].IsSequence()) {
            percentiles.clear();
            for (const auto &value : node["percentiles"]) {
                percentiles.push_back(value.as<float>());
            }
        }
        if (node["geometricCenterBlend"]) geometricCenterBlend = node["geometricCenterBlend"].as<float>();

        if (node["minComponentVoxels"]) minComponentVoxels = node["minComponentVoxels"].as<int>();
        if (node["minHighSeedVoxels"]) minHighSeedVoxels = node["minHighSeedVoxels"].as<int>();
        if (node["seedMergeDistance"]) seedMergeDistance = node["seedMergeDistance"].as<float>();
        if (node["adaptiveSeedMergeEnabled"]) adaptiveSeedMergeEnabled = node["adaptiveSeedMergeEnabled"].as<bool>();
        if (node["adaptiveSeedMergeModerateHighSeedCount"]) adaptiveSeedMergeModerateHighSeedCount = node["adaptiveSeedMergeModerateHighSeedCount"].as<int>();
        if (node["adaptiveSeedMergeDenseHighSeedCount"]) adaptiveSeedMergeDenseHighSeedCount = node["adaptiveSeedMergeDenseHighSeedCount"].as<int>();
        if (node["adaptiveSeedMergeModerateDistance"]) adaptiveSeedMergeModerateDistance = node["adaptiveSeedMergeModerateDistance"].as<float>();
        if (node["adaptiveSeedMergeDenseDistance"]) adaptiveSeedMergeDenseDistance = node["adaptiveSeedMergeDenseDistance"].as<float>();
        if (node["seedMergeUseScaledZ"]) seedMergeUseScaledZ = node["seedMergeUseScaledZ"].as<bool>();
        if (node["seedSplitSeparation"]) seedSplitSeparation = node["seedSplitSeparation"].as<float>();
        if (node["dedupDistance"]) dedupDistance = node["dedupDistance"].as<float>();
        if (node["dedupRadiusScale"]) dedupRadiusScale = node["dedupRadiusScale"].as<float>();
        if (node["adaptiveDedupEnabled"]) adaptiveDedupEnabled = node["adaptiveDedupEnabled"].as<bool>();
        if (node["adaptiveDedupModerateHighSeedCount"]) adaptiveDedupModerateHighSeedCount = node["adaptiveDedupModerateHighSeedCount"].as<int>();
        if (node["adaptiveDedupDenseHighSeedCount"]) adaptiveDedupDenseHighSeedCount = node["adaptiveDedupDenseHighSeedCount"].as<int>();
        if (node["adaptiveDedupModerateDistance"]) adaptiveDedupModerateDistance = node["adaptiveDedupModerateDistance"].as<float>();
        if (node["adaptiveDedupModerateRadiusScale"]) adaptiveDedupModerateRadiusScale = node["adaptiveDedupModerateRadiusScale"].as<float>();
        if (node["adaptiveDedupDenseDistance"]) adaptiveDedupDenseDistance = node["adaptiveDedupDenseDistance"].as<float>();
        if (node["adaptiveDedupDenseRadiusScale"]) adaptiveDedupDenseRadiusScale = node["adaptiveDedupDenseRadiusScale"].as<float>();
        if (node["fragmentMergeEnabled"]) fragmentMergeEnabled = node["fragmentMergeEnabled"].as<bool>();
        if (node["fragmentMergeMaxInputCells"]) fragmentMergeMaxInputCells = node["fragmentMergeMaxInputCells"].as<int>();
        if (node["fragmentMergeSparseMinDistance"]) fragmentMergeSparseMinDistance = node["fragmentMergeSparseMinDistance"].as<float>();
        if (node["fragmentMergeSparseMajorScale"]) fragmentMergeSparseMajorScale = node["fragmentMergeSparseMajorScale"].as<float>();
        if (node["fragmentMergeModerateMinDistance"]) fragmentMergeModerateMinDistance = node["fragmentMergeModerateMinDistance"].as<float>();
        if (node["fragmentMergeModerateMajorScale"]) fragmentMergeModerateMajorScale = node["fragmentMergeModerateMajorScale"].as<float>();
        if (node["allowSeededSplitInSparseField"]) allowSeededSplitInSparseField = node["allowSeededSplitInSparseField"].as<bool>();
        if (node["seededSplitMaxMajorRadius"]) seededSplitMaxMajorRadius = node["seededSplitMaxMajorRadius"].as<float>();
        if (node["seededSplitMinSeedBalance"]) seededSplitMinSeedBalance = node["seededSplitMinSeedBalance"].as<float>();
        if (node["seededSplitMinSeedSeparation"]) seededSplitMinSeedSeparation = node["seededSplitMinSeedSeparation"].as<float>();
        if (node["seededSplitUseScaled3DSeparation"]) seededSplitUseScaled3DSeparation = node["seededSplitUseScaled3DSeparation"].as<bool>();
        if (node["seededSplitMinScaled3DSeparation"]) seededSplitMinScaled3DSeparation = node["seededSplitMinScaled3DSeparation"].as<float>();
        if (node["seededSplitMinScaledZSeparation"]) seededSplitMinScaledZSeparation = node["seededSplitMinScaledZSeparation"].as<float>();

        if (node["useScaledVoxelVolumeForRadius"]) useScaledVoxelVolumeForRadius = node["useScaledVoxelVolumeForRadius"].as<bool>();
        if (node["radiusInflationScale"]) radiusInflationScale = node["radiusInflationScale"].as<float>();
        if (node["zCenterOffsetScaled"]) zCenterOffsetScaled = node["zCenterOffsetScaled"].as<float>();
        if (node["minOutputMajorRadius"]) minOutputMajorRadius = node["minOutputMajorRadius"].as<float>();
        if (node["minOutputBRadius"]) minOutputBRadius = node["minOutputBRadius"].as<float>();
        if (node["minOutputMinorRadius"]) minOutputMinorRadius = node["minOutputMinorRadius"].as<float>();
        if (node["maxOutputMajorRadius"]) maxOutputMajorRadius = node["maxOutputMajorRadius"].as<float>();
        if (node["maxOutputBRadius"]) maxOutputBRadius = node["maxOutputBRadius"].as<float>();
        if (node["maxOutputMinorRadius"]) maxOutputMinorRadius = node["maxOutputMinorRadius"].as<float>();
        if (node["minOutputCenterZScaled"]) minOutputCenterZScaled = node["minOutputCenterZScaled"].as<float>();
        if (node["maxOutputCenterZScaled"]) maxOutputCenterZScaled = node["maxOutputCenterZScaled"].as<float>();

        if (node["signalStatsEnabled"]) signalStatsEnabled = node["signalStatsEnabled"].as<bool>();
        if (node["shellInnerScale"]) shellInnerScale = node["shellInnerScale"].as<float>();
        if (node["shellOuterScale"]) shellOuterScale = node["shellOuterScale"].as<float>();
        if (node["minTop10MinusShell"]) minTop10MinusShell = node["minTop10MinusShell"].as<float>();
        if (node["minMeanMinusShell"]) minMeanMinusShell = node["minMeanMinusShell"].as<float>();
        if (node["minBiologicalVoxelCount"]) minBiologicalVoxelCount = node["minBiologicalVoxelCount"].as<int>();
        if (node["minBiologicalMajorRadius"]) minBiologicalMajorRadius = node["minBiologicalMajorRadius"].as<float>();
        if (node["smallArtifactSparseMaxCells"]) smallArtifactSparseMaxCells = node["smallArtifactSparseMaxCells"].as<int>();
        if (node["smallArtifactMaxVoxels"]) smallArtifactMaxVoxels = node["smallArtifactMaxVoxels"].as<int>();
        if (node["smallArtifactMaxTop10Intensity"]) smallArtifactMaxTop10Intensity = node["smallArtifactMaxTop10Intensity"].as<float>();
        if (node["smallArtifactMaxMeanIntensity"]) smallArtifactMaxMeanIntensity = node["smallArtifactMaxMeanIntensity"].as<float>();
        if (node["rejectTinyBrightArtifacts"]) rejectTinyBrightArtifacts = node["rejectTinyBrightArtifacts"].as<bool>();

        if (node["biologicalNearDuplicateMergeEnabled"]) biologicalNearDuplicateMergeEnabled = node["biologicalNearDuplicateMergeEnabled"].as<bool>();
        if (node["minBiologicalCenterDistance"]) minBiologicalCenterDistance = node["minBiologicalCenterDistance"].as<float>();
        if (node["adaptiveNearDuplicateMergeEnabled"]) adaptiveNearDuplicateMergeEnabled = node["adaptiveNearDuplicateMergeEnabled"].as<bool>();
        if (node["adaptiveNearDuplicateMinCells"]) adaptiveNearDuplicateMinCells = node["adaptiveNearDuplicateMinCells"].as<int>();
        if (node["adaptiveNearDuplicateMaxCells"]) adaptiveNearDuplicateMaxCells = node["adaptiveNearDuplicateMaxCells"].as<int>();
        if (node["adaptiveNearDuplicateDistance"]) adaptiveNearDuplicateDistance = node["adaptiveNearDuplicateDistance"].as<float>();
        if (node["adaptiveNearDuplicateMinPairRatio"]) adaptiveNearDuplicateMinPairRatio = node["adaptiveNearDuplicateMinPairRatio"].as<float>();
        if (node["localMaxSeedEnabled"]) localMaxSeedEnabled = node["localMaxSeedEnabled"].as<bool>();
        if (node["localMaxSeedMinStrongSeedCount"]) localMaxSeedMinStrongSeedCount = node["localMaxSeedMinStrongSeedCount"].as<int>();
        if (node["localMaxSeedMinDistance"]) localMaxSeedMinDistance = node["localMaxSeedMinDistance"].as<float>();
        if (node["localMaxSeedMaxPerFrameFactor"]) localMaxSeedMaxPerFrameFactor = node["localMaxSeedMaxPerFrameFactor"].as<float>();
        if (node["useHighSeedCenterForSplitCells"]) useHighSeedCenterForSplitCells = node["useHighSeedCenterForSplitCells"].as<bool>();
        if (node["seededWatershedEnabled"]) seededWatershedEnabled = node["seededWatershedEnabled"].as<bool>();
        if (node["seededWatershedPreferOverPercentile"]) seededWatershedPreferOverPercentile = node["seededWatershedPreferOverPercentile"].as<bool>();
        if (node["seededWatershedRescueEnabled"]) seededWatershedRescueEnabled = node["seededWatershedRescueEnabled"].as<bool>();
        if (node["seededWatershedPreferBaseCountMin"]) seededWatershedPreferBaseCountMin = node["seededWatershedPreferBaseCountMin"].as<int>();
        if (node["seededWatershedPreferBaseCountMax"]) seededWatershedPreferBaseCountMax = node["seededWatershedPreferBaseCountMax"].as<int>();
        if (node["seededWatershedSignalSigma"]) seededWatershedSignalSigma = node["seededWatershedSignalSigma"].as<float>();
        if (node["seededWatershedBackgroundSigma"]) seededWatershedBackgroundSigma = node["seededWatershedBackgroundSigma"].as<float>();
        if (node["seededWatershedSeedPercentile"]) seededWatershedSeedPercentile = node["seededWatershedSeedPercentile"].as<float>();
        if (node["seededWatershedMaskPercentile"]) seededWatershedMaskPercentile = node["seededWatershedMaskPercentile"].as<float>();
        if (node["seededWatershedSeedThresholdFloor"]) seededWatershedSeedThresholdFloor = node["seededWatershedSeedThresholdFloor"].as<float>();
        if (node["seededWatershedMaskThresholdFloor"]) seededWatershedMaskThresholdFloor = node["seededWatershedMaskThresholdFloor"].as<float>();
        if (node["seededWatershedMinSeedDistance"]) seededWatershedMinSeedDistance = node["seededWatershedMinSeedDistance"].as<float>();
        if (node["seededWatershedCloseSeedMinDistance"]) seededWatershedCloseSeedMinDistance = node["seededWatershedCloseSeedMinDistance"].as<float>();
        if (node["seededWatershedMinValleyDrop"]) seededWatershedMinValleyDrop = node["seededWatershedMinValleyDrop"].as<float>();
        if (node["seededWatershedMaxGrowDistance"]) seededWatershedMaxGrowDistance = node["seededWatershedMaxGrowDistance"].as<float>();
        if (node["seededWatershedUseDistanceTransform"]) seededWatershedUseDistanceTransform = node["seededWatershedUseDistanceTransform"].as<bool>();
        if (node["seededWatershedDistancePriorityWeight"]) seededWatershedDistancePriorityWeight = node["seededWatershedDistancePriorityWeight"].as<float>();
        if (node["seededWatershedContrastPriorityWeight"]) seededWatershedContrastPriorityWeight = node["seededWatershedContrastPriorityWeight"].as<float>();
        if (node["seededWatershedSeedDistancePenaltyWeight"]) seededWatershedSeedDistancePenaltyWeight = node["seededWatershedSeedDistancePenaltyWeight"].as<float>();
        if (node["seededWatershedCenterSeedBlend"]) seededWatershedCenterSeedBlend = node["seededWatershedCenterSeedBlend"].as<float>();
        if (node["seededWatershedMinVoxels"]) seededWatershedMinVoxels = node["seededWatershedMinVoxels"].as<int>();
        if (node["seededWatershedAdaptiveMinVoxelsEnabled"]) seededWatershedAdaptiveMinVoxelsEnabled = node["seededWatershedAdaptiveMinVoxelsEnabled"].as<bool>();
        if (node["seededWatershedAdaptiveMinVoxelsCellCount"]) seededWatershedAdaptiveMinVoxelsCellCount = node["seededWatershedAdaptiveMinVoxelsCellCount"].as<int>();
        if (node["seededWatershedAdaptiveDenseMinVoxels"]) seededWatershedAdaptiveDenseMinVoxels = node["seededWatershedAdaptiveDenseMinVoxels"].as<int>();
        if (node["seededWatershedMaxSeeds"]) seededWatershedMaxSeeds = node["seededWatershedMaxSeeds"].as<int>();
        if (node["seededWatershedRescueMinDistance"]) seededWatershedRescueMinDistance = node["seededWatershedRescueMinDistance"].as<float>();
        if (node["seededWatershedRescueMaxAddedFraction"]) seededWatershedRescueMaxAddedFraction = node["seededWatershedRescueMaxAddedFraction"].as<float>();
        if (node["seededWatershedRescueMaxAdded"]) seededWatershedRescueMaxAdded = node["seededWatershedRescueMaxAdded"].as<int>();
        if (node["seededWatershedLowRescueEnabled"]) seededWatershedLowRescueEnabled = node["seededWatershedLowRescueEnabled"].as<bool>();
        if (node["seededWatershedLowRescueSeedPercentile"]) seededWatershedLowRescueSeedPercentile = node["seededWatershedLowRescueSeedPercentile"].as<float>();
        if (node["seededWatershedLowRescueMinVoxels"]) seededWatershedLowRescueMinVoxels = node["seededWatershedLowRescueMinVoxels"].as<int>();
        if (node["seededWatershedLowRescueMinSeedDistance"]) seededWatershedLowRescueMinSeedDistance = node["seededWatershedLowRescueMinSeedDistance"].as<float>();
        if (node["seededWatershedLowRescueCloseSeedMinDistance"]) seededWatershedLowRescueCloseSeedMinDistance = node["seededWatershedLowRescueCloseSeedMinDistance"].as<float>();
        if (node["seededWatershedLowRescueMinValleyDrop"]) seededWatershedLowRescueMinValleyDrop = node["seededWatershedLowRescueMinValleyDrop"].as<float>();
        if (node["seededWatershedLowRescueRefineEnabled"]) seededWatershedLowRescueRefineEnabled = node["seededWatershedLowRescueRefineEnabled"].as<bool>();
        if (node["seededWatershedLowRescueRefineDistance"]) seededWatershedLowRescueRefineDistance = node["seededWatershedLowRescueRefineDistance"].as<float>();
        if (node["seededWatershedLowRescueMinDistance"]) seededWatershedLowRescueMinDistance = node["seededWatershedLowRescueMinDistance"].as<float>();
        if (node["seededWatershedLowRescueMinCandidateVoxels"]) seededWatershedLowRescueMinCandidateVoxels = node["seededWatershedLowRescueMinCandidateVoxels"].as<int>();
        if (node["seededWatershedLowRescueMinTop10MinusShell"]) seededWatershedLowRescueMinTop10MinusShell = node["seededWatershedLowRescueMinTop10MinusShell"].as<float>();
        if (node["seededWatershedLowRescueClusterDistance"]) seededWatershedLowRescueClusterDistance = node["seededWatershedLowRescueClusterDistance"].as<float>();
        if (node["seededWatershedLowRescueSortByDistance"]) seededWatershedLowRescueSortByDistance = node["seededWatershedLowRescueSortByDistance"].as<bool>();
        if (node["seededWatershedLowRescueMaxAdded"]) seededWatershedLowRescueMaxAdded = node["seededWatershedLowRescueMaxAdded"].as<int>();
        if (node["seededWatershedLowRescueMinBaseCells"]) seededWatershedLowRescueMinBaseCells = node["seededWatershedLowRescueMinBaseCells"].as<int>();
        if (node["seededWatershedLowRescueMaxBaseCells"]) seededWatershedLowRescueMaxBaseCells = node["seededWatershedLowRescueMaxBaseCells"].as<int>();
        if (node["finalDuplicateMergeDistance"]) finalDuplicateMergeDistance = node["finalDuplicateMergeDistance"].as<float>();
        if (node["finalLocalRefineEnabled"]) finalLocalRefineEnabled = node["finalLocalRefineEnabled"].as<bool>();
        if (node["finalLocalRefineRadius"]) finalLocalRefineRadius = node["finalLocalRefineRadius"].as<float>();
        if (node["finalLocalRefineQuantile"]) finalLocalRefineQuantile = node["finalLocalRefineQuantile"].as<float>();
        if (node["finalLocalRefineBlend"]) finalLocalRefineBlend = node["finalLocalRefineBlend"].as<float>();
        if (node["finalZColumnRefineEnabled"]) finalZColumnRefineEnabled = node["finalZColumnRefineEnabled"].as<bool>();
        if (node["finalZColumnRefineRadiusXY"]) finalZColumnRefineRadiusXY = node["finalZColumnRefineRadiusXY"].as<float>();
        if (node["finalZColumnRefineHalfWindowScaled"]) finalZColumnRefineHalfWindowScaled = node["finalZColumnRefineHalfWindowScaled"].as<float>();
        if (node["finalZColumnRefineQuantile"]) finalZColumnRefineQuantile = node["finalZColumnRefineQuantile"].as<float>();
        if (node["finalZColumnRefineMinScoreFraction"]) finalZColumnRefineMinScoreFraction = node["finalZColumnRefineMinScoreFraction"].as<float>();
        if (node["finalZColumnRefineMaxMoveScaled"]) finalZColumnRefineMaxMoveScaled = node["finalZColumnRefineMaxMoveScaled"].as<float>();
        if (node["finalZColumnRefineBlend"]) finalZColumnRefineBlend = node["finalZColumnRefineBlend"].as<float>();
        if (node["finalZColumnRefineMinCells"]) finalZColumnRefineMinCells = node["finalZColumnRefineMinCells"].as<int>();
        if (node["finalZColumnRefineMaxCells"]) finalZColumnRefineMaxCells = node["finalZColumnRefineMaxCells"].as<int>();
        if (node["finalZPeakSplitEnabled"]) finalZPeakSplitEnabled = node["finalZPeakSplitEnabled"].as<bool>();
        if (node["finalZPeakSplitMinCells"]) finalZPeakSplitMinCells = node["finalZPeakSplitMinCells"].as<int>();
        if (node["finalZPeakSplitMaxCells"]) finalZPeakSplitMaxCells = node["finalZPeakSplitMaxCells"].as<int>();
        if (node["finalZPeakSplitMaxAdded"]) finalZPeakSplitMaxAdded = node["finalZPeakSplitMaxAdded"].as<int>();
        if (node["finalZPeakSplitAddOnlyEnabled"]) finalZPeakSplitAddOnlyEnabled = node["finalZPeakSplitAddOnlyEnabled"].as<bool>();
        if (node["finalZPeakSplitMinMajorRadius"]) finalZPeakSplitMinMajorRadius = node["finalZPeakSplitMinMajorRadius"].as<float>();
        if (node["finalZPeakSplitRadiusXY"]) finalZPeakSplitRadiusXY = node["finalZPeakSplitRadiusXY"].as<float>();
        if (node["finalZPeakSplitQuantile"]) finalZPeakSplitQuantile = node["finalZPeakSplitQuantile"].as<float>();
        if (node["finalZPeakSplitMinPeakScoreFraction"]) finalZPeakSplitMinPeakScoreFraction = node["finalZPeakSplitMinPeakScoreFraction"].as<float>();
        if (node["finalZPeakSplitMaxParentTop10MinusShell"]) finalZPeakSplitMaxParentTop10MinusShell = node["finalZPeakSplitMaxParentTop10MinusShell"].as<float>();
        if (node["finalZPeakSplitLocal3DFallbackEnabled"]) finalZPeakSplitLocal3DFallbackEnabled = node["finalZPeakSplitLocal3DFallbackEnabled"].as<bool>();
        if (node["finalZPeakSplitPreferLocal3D"]) finalZPeakSplitPreferLocal3D = node["finalZPeakSplitPreferLocal3D"].as<bool>();
        if (node["finalZPeakSplitLocal3DCentroidRadiusXY"]) finalZPeakSplitLocal3DCentroidRadiusXY = node["finalZPeakSplitLocal3DCentroidRadiusXY"].as<float>();
        if (node["finalZPeakSplitLocal3DCentroidHalfWindowScaled"]) finalZPeakSplitLocal3DCentroidHalfWindowScaled = node["finalZPeakSplitLocal3DCentroidHalfWindowScaled"].as<float>();
        if (node["finalZPeakSplitMinSeparationScaled"]) finalZPeakSplitMinSeparationScaled = node["finalZPeakSplitMinSeparationScaled"].as<float>();
        if (node["finalZPeakSplitMaxSeparationScaled"]) finalZPeakSplitMaxSeparationScaled = node["finalZPeakSplitMaxSeparationScaled"].as<float>();
        if (node["finalZPeakSplitMaxCenterShiftXY"]) finalZPeakSplitMaxCenterShiftXY = node["finalZPeakSplitMaxCenterShiftXY"].as<float>();
        if (node["finalZPeakSplitRadiusScale"]) finalZPeakSplitRadiusScale = node["finalZPeakSplitRadiusScale"].as<float>();
        if (node["finalBrightPairMidpointRescueEnabled"]) finalBrightPairMidpointRescueEnabled = node["finalBrightPairMidpointRescueEnabled"].as<bool>();
        if (node["finalBrightPairMidpointRescueMinCells"]) finalBrightPairMidpointRescueMinCells = node["finalBrightPairMidpointRescueMinCells"].as<int>();
        if (node["finalBrightPairMidpointRescueMaxCells"]) finalBrightPairMidpointRescueMaxCells = node["finalBrightPairMidpointRescueMaxCells"].as<int>();
        if (node["finalBrightPairMidpointRescueMaxAdded"]) finalBrightPairMidpointRescueMaxAdded = node["finalBrightPairMidpointRescueMaxAdded"].as<int>();
        if (node["finalBrightPairMidpointRescueMinDistance"]) finalBrightPairMidpointRescueMinDistance = node["finalBrightPairMidpointRescueMinDistance"].as<float>();
        if (node["finalBrightPairMidpointRescueMaxDistance"]) finalBrightPairMidpointRescueMaxDistance = node["finalBrightPairMidpointRescueMaxDistance"].as<float>();
        if (node["finalBrightPairMidpointRescueMinTop10MinusShell"]) finalBrightPairMidpointRescueMinTop10MinusShell = node["finalBrightPairMidpointRescueMinTop10MinusShell"].as<float>();
        if (node["finalBrightPairMidpointRescueRadiusScale"]) finalBrightPairMidpointRescueRadiusScale = node["finalBrightPairMidpointRescueRadiusScale"].as<float>();
        if (node["finalBrightPairMidpointRescueMinAddedDistance"]) finalBrightPairMidpointRescueMinAddedDistance = node["finalBrightPairMidpointRescueMinAddedDistance"].as<float>();
        if (node["finalBrightPairMidpointRescueMinExistingDistance"]) finalBrightPairMidpointRescueMinExistingDistance = node["finalBrightPairMidpointRescueMinExistingDistance"].as<float>();
        if (node["finalBrightPairMidpointRescueSortByExistingDistance"]) finalBrightPairMidpointRescueSortByExistingDistance = node["finalBrightPairMidpointRescueSortByExistingDistance"].as<bool>();
        if (node["finalClusterCentroidRecallRescueEnabled"]) finalClusterCentroidRecallRescueEnabled = node["finalClusterCentroidRecallRescueEnabled"].as<bool>();
        if (node["finalClusterCentroidRecallRescueMinCells"]) finalClusterCentroidRecallRescueMinCells = node["finalClusterCentroidRecallRescueMinCells"].as<int>();
        if (node["finalClusterCentroidRecallRescueMaxCells"]) finalClusterCentroidRecallRescueMaxCells = node["finalClusterCentroidRecallRescueMaxCells"].as<int>();
        if (node["finalClusterCentroidRecallRescueMaxAdded"]) finalClusterCentroidRecallRescueMaxAdded = node["finalClusterCentroidRecallRescueMaxAdded"].as<int>();
        if (node["finalClusterCentroidRecallRescueMinClusterSize"]) finalClusterCentroidRecallRescueMinClusterSize = node["finalClusterCentroidRecallRescueMinClusterSize"].as<int>();
        if (node["finalClusterCentroidRecallRescueClusterDistance"]) finalClusterCentroidRecallRescueClusterDistance = node["finalClusterCentroidRecallRescueClusterDistance"].as<float>();
        if (node["finalClusterCentroidRecallRescueMinAddedDistance"]) finalClusterCentroidRecallRescueMinAddedDistance = node["finalClusterCentroidRecallRescueMinAddedDistance"].as<float>();
        if (node["finalClusterCentroidRecallRescueMinTop10MinusShell"]) finalClusterCentroidRecallRescueMinTop10MinusShell = node["finalClusterCentroidRecallRescueMinTop10MinusShell"].as<float>();
        if (node["finalClusterCentroidRecallRescueRadiusScale"]) finalClusterCentroidRecallRescueRadiusScale = node["finalClusterCentroidRecallRescueRadiusScale"].as<float>();
        if (node["finalClusterCentroidCollapseEnabled"]) finalClusterCentroidCollapseEnabled = node["finalClusterCentroidCollapseEnabled"].as<bool>();
        if (node["finalClusterCentroidCollapseMinCells"]) finalClusterCentroidCollapseMinCells = node["finalClusterCentroidCollapseMinCells"].as<int>();
        if (node["finalClusterCentroidCollapseMaxCells"]) finalClusterCentroidCollapseMaxCells = node["finalClusterCentroidCollapseMaxCells"].as<int>();
        if (node["finalClusterCentroidCollapseMinClusterSize"]) finalClusterCentroidCollapseMinClusterSize = node["finalClusterCentroidCollapseMinClusterSize"].as<int>();
        if (node["finalClusterCentroidCollapseLinkDistance"]) finalClusterCentroidCollapseLinkDistance = node["finalClusterCentroidCollapseLinkDistance"].as<float>();
        if (node["finalClusterCentroidCollapseRadiusScale"]) finalClusterCentroidCollapseRadiusScale = node["finalClusterCentroidCollapseRadiusScale"].as<float>();
        if (node["finalClusterCentroidCollapseUseSignalWeights"]) finalClusterCentroidCollapseUseSignalWeights = node["finalClusterCentroidCollapseUseSignalWeights"].as<bool>();
        if (node["finalDominatedDuplicateFilterEnabled"]) finalDominatedDuplicateFilterEnabled = node["finalDominatedDuplicateFilterEnabled"].as<bool>();
        if (node["finalDominatedDuplicateFilterMinCells"]) finalDominatedDuplicateFilterMinCells = node["finalDominatedDuplicateFilterMinCells"].as<int>();
        if (node["finalDominatedDuplicateFilterMaxCells"]) finalDominatedDuplicateFilterMaxCells = node["finalDominatedDuplicateFilterMaxCells"].as<int>();
        if (node["finalDominatedDuplicateFilterDistance"]) finalDominatedDuplicateFilterDistance = node["finalDominatedDuplicateFilterDistance"].as<float>();
        if (node["finalDominatedDuplicateFilterMinVoxelRatio"]) finalDominatedDuplicateFilterMinVoxelRatio = node["finalDominatedDuplicateFilterMinVoxelRatio"].as<float>();
        if (node["finalDominatedDuplicateFilterMinSignalRatio"]) finalDominatedDuplicateFilterMinSignalRatio = node["finalDominatedDuplicateFilterMinSignalRatio"].as<float>();
        if (node["finalDominatedDuplicateFilterMinRadiusRatio"]) finalDominatedDuplicateFilterMinRadiusRatio = node["finalDominatedDuplicateFilterMinRadiusRatio"].as<float>();
        if (node["finalDominatedDuplicateFilterMaxLoserTop10MinusShell"]) finalDominatedDuplicateFilterMaxLoserTop10MinusShell = node["finalDominatedDuplicateFilterMaxLoserTop10MinusShell"].as<float>();
        if (node["finalDominatedDuplicateFilterRequireTwoSignals"]) finalDominatedDuplicateFilterRequireTwoSignals = node["finalDominatedDuplicateFilterRequireTwoSignals"].as<bool>();
        if (node["finalPostRefineDuplicateMergeDistance"]) finalPostRefineDuplicateMergeDistance = node["finalPostRefineDuplicateMergeDistance"].as<float>();
        if (node["finalWeakSatelliteFilterEnabled"]) finalWeakSatelliteFilterEnabled = node["finalWeakSatelliteFilterEnabled"].as<bool>();
        if (node["finalWeakSatelliteNeighborDistance"]) finalWeakSatelliteNeighborDistance = node["finalWeakSatelliteNeighborDistance"].as<float>();
        if (node["finalWeakSatelliteMaxVoxels"]) finalWeakSatelliteMaxVoxels = node["finalWeakSatelliteMaxVoxels"].as<int>();
        if (node["finalWeakSatelliteMaxTop10MinusShell"]) finalWeakSatelliteMaxTop10MinusShell = node["finalWeakSatelliteMaxTop10MinusShell"].as<float>();
        if (node["finalWeakSatelliteMaxMinorRadius"]) finalWeakSatelliteMaxMinorRadius = node["finalWeakSatelliteMaxMinorRadius"].as<float>();
        if (node["finalWeakSatelliteNeighborMinVoxels"]) finalWeakSatelliteNeighborMinVoxels = node["finalWeakSatelliteNeighborMinVoxels"].as<int>();
        if (node["finalWeakSatelliteNeighborVoxelRatio"]) finalWeakSatelliteNeighborVoxelRatio = node["finalWeakSatelliteNeighborVoxelRatio"].as<float>();
        if (node["finalSparseIsolatedFloorFilterEnabled"]) finalSparseIsolatedFloorFilterEnabled = node["finalSparseIsolatedFloorFilterEnabled"].as<bool>();
        if (node["finalSparseIsolatedFloorFilterMaxCells"]) finalSparseIsolatedFloorFilterMaxCells = node["finalSparseIsolatedFloorFilterMaxCells"].as<int>();
        if (node["finalSparseIsolatedFloorMaxMajorRadius"]) finalSparseIsolatedFloorMaxMajorRadius = node["finalSparseIsolatedFloorMaxMajorRadius"].as<float>();
        if (node["finalSparseIsolatedFloorMaxMinorRadius"]) finalSparseIsolatedFloorMaxMinorRadius = node["finalSparseIsolatedFloorMaxMinorRadius"].as<float>();
        if (node["finalSparseIsolatedFloorMinNearestDistance"]) finalSparseIsolatedFloorMinNearestDistance = node["finalSparseIsolatedFloorMinNearestDistance"].as<float>();
        if (node["finalSparseIsolatedFloorWeakSignalMinCells"]) finalSparseIsolatedFloorWeakSignalMinCells = node["finalSparseIsolatedFloorWeakSignalMinCells"].as<int>();
        if (node["finalSparseIsolatedFloorWeakSignalMaxCells"]) finalSparseIsolatedFloorWeakSignalMaxCells = node["finalSparseIsolatedFloorWeakSignalMaxCells"].as<int>();
        if (node["finalSparseIsolatedFloorMaxTop10MinusShell"]) finalSparseIsolatedFloorMaxTop10MinusShell = node["finalSparseIsolatedFloorMaxTop10MinusShell"].as<float>();
        if (node["fusionEnabled"]) fusionEnabled = node["fusionEnabled"].as<bool>();
        if (node["fusionStartFrame"]) fusionStartFrame = node["fusionStartFrame"].as<int>();
        if (node["fusionEveryNFrames"]) fusionEveryNFrames = std::max(1, node["fusionEveryNFrames"].as<int>());
        if (node["fusionMinDistanceToExisting"]) fusionMinDistanceToExisting = node["fusionMinDistanceToExisting"].as<float>();
        if (node["fusionMinVoxels"]) fusionMinVoxels = node["fusionMinVoxels"].as<int>();
        if (node["fusionMinTop10MinusShell"]) fusionMinTop10MinusShell = node["fusionMinTop10MinusShell"].as<float>();
        if (node["fusionMaxAddedPerFrame"]) fusionMaxAddedPerFrame = node["fusionMaxAddedPerFrame"].as<int>();
        if (node["fusionRadiusScale"]) fusionRadiusScale = node["fusionRadiusScale"].as<float>();
        if (node["fusionBrightness"]) fusionBrightness = node["fusionBrightness"].as<float>();
        if (node["fusionNamePrefix"]) fusionNamePrefix = node["fusionNamePrefix"].as<std::string>();
        if (node["fusionRepairCloseCellsEnabled"]) fusionRepairCloseCellsEnabled = node["fusionRepairCloseCellsEnabled"].as<bool>();
        if (node["fusionRepairMaxDistance"]) fusionRepairMaxDistance = node["fusionRepairMaxDistance"].as<float>();
        if (node["fusionRepairRadiusRatio"]) fusionRepairRadiusRatio = node["fusionRepairRadiusRatio"].as<float>();
        if (node["fusionRepairPositionBlend"]) fusionRepairPositionBlend = node["fusionRepairPositionBlend"].as<float>();
        if (node["fusionRepairRadiusBlend"]) fusionRepairRadiusBlend = node["fusionRepairRadiusBlend"].as<float>();
        if (node["fusionCenterPriorEnabled"]) fusionCenterPriorEnabled = node["fusionCenterPriorEnabled"].as<bool>();
        if (node["fusionCenterPriorMaxDistance"]) fusionCenterPriorMaxDistance = node["fusionCenterPriorMaxDistance"].as<float>();
        if (node["fusionCenterPriorPositionBlend"]) fusionCenterPriorPositionBlend = node["fusionCenterPriorPositionBlend"].as<float>();
        if (node["fusionCenterPriorRadiusBlend"]) fusionCenterPriorRadiusBlend = node["fusionCenterPriorRadiusBlend"].as<float>();
        if (node["fusionSplitPriorEnabled"]) fusionSplitPriorEnabled = node["fusionSplitPriorEnabled"].as<bool>();
        if (node["fusionSplitPriorForceSchedule"]) fusionSplitPriorForceSchedule = node["fusionSplitPriorForceSchedule"].as<bool>();
        if (node["fusionSplitPriorSkipRandomSplits"]) fusionSplitPriorSkipRandomSplits = node["fusionSplitPriorSkipRandomSplits"].as<bool>();
        if (node["fusionSplitPriorGuidedOnly"]) fusionSplitPriorGuidedOnly = node["fusionSplitPriorGuidedOnly"].as<bool>();
        if (node["fusionSplitPriorMaxParentDistance"]) fusionSplitPriorMaxParentDistance = node["fusionSplitPriorMaxParentDistance"].as<float>();
        if (node["fusionSplitPriorParentRadiusScale"]) fusionSplitPriorParentRadiusScale = node["fusionSplitPriorParentRadiusScale"].as<float>();
        if (node["fusionSplitPriorMinSeparation"]) fusionSplitPriorMinSeparation = node["fusionSplitPriorMinSeparation"].as<float>();
        if (node["fusionSplitPriorMinSeparationRadiusScale"]) fusionSplitPriorMinSeparationRadiusScale = node["fusionSplitPriorMinSeparationRadiusScale"].as<float>();
        if (node["fusionSplitPriorRoundParentMaxShape"]) fusionSplitPriorRoundParentMaxShape = node["fusionSplitPriorRoundParentMaxShape"].as<float>();
        if (node["fusionSplitPriorRoundParentMinSeparationRadiusScale"]) fusionSplitPriorRoundParentMinSeparationRadiusScale = node["fusionSplitPriorRoundParentMinSeparationRadiusScale"].as<float>();
        if (node["fusionSplitPriorMaxSeparation"]) fusionSplitPriorMaxSeparation = node["fusionSplitPriorMaxSeparation"].as<float>();
        if (node["fusionSplitPriorMaxSeparationRadiusScale"]) fusionSplitPriorMaxSeparationRadiusScale = node["fusionSplitPriorMaxSeparationRadiusScale"].as<float>();
        if (node["fusionSplitPriorMaxMidpointDistance"]) fusionSplitPriorMaxMidpointDistance = node["fusionSplitPriorMaxMidpointDistance"].as<float>();
        if (node["fusionSplitPriorMaxMidpointRadiusScale"]) fusionSplitPriorMaxMidpointRadiusScale = node["fusionSplitPriorMaxMidpointRadiusScale"].as<float>();
        if (node["fusionSplitPriorMaxScore"]) fusionSplitPriorMaxScore = node["fusionSplitPriorMaxScore"].as<float>();
        if (node["fusionSplitPriorMinVoxels"]) fusionSplitPriorMinVoxels = node["fusionSplitPriorMinVoxels"].as<int>();
        if (node["fusionSplitPriorMinTop10MinusShell"]) fusionSplitPriorMinTop10MinusShell = node["fusionSplitPriorMinTop10MinusShell"].as<float>();
        if (node["fusionSplitPriorElongatedParentRescueEnabled"]) fusionSplitPriorElongatedParentRescueEnabled = node["fusionSplitPriorElongatedParentRescueEnabled"].as<bool>();
        if (node["fusionSplitPriorElongatedParentMinShape"]) fusionSplitPriorElongatedParentMinShape = node["fusionSplitPriorElongatedParentMinShape"].as<float>();
        if (node["fusionSplitPriorElongatedParentMaxScore"]) fusionSplitPriorElongatedParentMaxScore = node["fusionSplitPriorElongatedParentMaxScore"].as<float>();
        if (node["fusionSplitPriorElongatedParentScoreBonus"]) fusionSplitPriorElongatedParentScoreBonus = node["fusionSplitPriorElongatedParentScoreBonus"].as<float>();
        if (node["fusionSplitPriorElongatedParentMaxNeighborClaimPenalty"]) fusionSplitPriorElongatedParentMaxNeighborClaimPenalty = node["fusionSplitPriorElongatedParentMaxNeighborClaimPenalty"].as<float>();
        if (node["fusionSplitPriorMidpointWeight"]) fusionSplitPriorMidpointWeight = node["fusionSplitPriorMidpointWeight"].as<float>();
        if (node["fusionSplitPriorSeparationPenaltyWeight"]) fusionSplitPriorSeparationPenaltyWeight = node["fusionSplitPriorSeparationPenaltyWeight"].as<float>();
        if (node["fusionSplitPriorSignalBonusWeight"]) fusionSplitPriorSignalBonusWeight = node["fusionSplitPriorSignalBonusWeight"].as<float>();
        if (node["fusionSplitPriorConflictReplacementEnabled"]) fusionSplitPriorConflictReplacementEnabled = node["fusionSplitPriorConflictReplacementEnabled"].as<bool>();
        if (node["fusionSplitPriorConflictCloseParentRadiusScale"]) fusionSplitPriorConflictCloseParentRadiusScale = node["fusionSplitPriorConflictCloseParentRadiusScale"].as<float>();
        if (node["fusionSplitPriorConflictMinNewSeparationRadiusScale"]) fusionSplitPriorConflictMinNewSeparationRadiusScale = node["fusionSplitPriorConflictMinNewSeparationRadiusScale"].as<float>();
        if (node["fusionSplitPriorConflictMinSeparationDrop"]) fusionSplitPriorConflictMinSeparationDrop = node["fusionSplitPriorConflictMinSeparationDrop"].as<float>();
        if (node["fusionSplitPriorConflictMinOldScoreForReplacement"]) fusionSplitPriorConflictMinOldScoreForReplacement = node["fusionSplitPriorConflictMinOldScoreForReplacement"].as<float>();
        if (node["fusionSplitPriorSuppressConflictPenaltiesForAsymmetricClose"]) fusionSplitPriorSuppressConflictPenaltiesForAsymmetricClose = node["fusionSplitPriorSuppressConflictPenaltiesForAsymmetricClose"].as<bool>();
        if (node["fusionSplitPriorMaxPriorsPerFrame"]) fusionSplitPriorMaxPriorsPerFrame = node["fusionSplitPriorMaxPriorsPerFrame"].as<int>();
        if (node["fusionSplitPriorWindowEnabled"]) fusionSplitPriorWindowEnabled = node["fusionSplitPriorWindowEnabled"].as<bool>();
        if (node["fusionSplitPriorWindowSize"]) fusionSplitPriorWindowSize = node["fusionSplitPriorWindowSize"].as<int>();
        if (node["fusionSplitPriorWindowMatchDistance"]) fusionSplitPriorWindowMatchDistance = node["fusionSplitPriorWindowMatchDistance"].as<float>();
        if (node["fusionSplitPriorWindowMatchDistancePerFrame"]) fusionSplitPriorWindowMatchDistancePerFrame = node["fusionSplitPriorWindowMatchDistancePerFrame"].as<float>();
        if (node["fusionSplitPriorWindowDaughterSupportBonus"]) fusionSplitPriorWindowDaughterSupportBonus = node["fusionSplitPriorWindowDaughterSupportBonus"].as<float>();
        if (node["fusionSplitPriorWindowMissingDaughterPenalty"]) fusionSplitPriorWindowMissingDaughterPenalty = node["fusionSplitPriorWindowMissingDaughterPenalty"].as<float>();
        if (node["fusionSplitPriorWindowParentPersistencePenalty"]) fusionSplitPriorWindowParentPersistencePenalty = node["fusionSplitPriorWindowParentPersistencePenalty"].as<float>();
        if (node["fusionSplitPriorWindowBalancedDaughterBonus"]) fusionSplitPriorWindowBalancedDaughterBonus = node["fusionSplitPriorWindowBalancedDaughterBonus"].as<float>();
        if (node["fusionSplitPriorWindowBalancedMinParentDistanceBalance"]) fusionSplitPriorWindowBalancedMinParentDistanceBalance = node["fusionSplitPriorWindowBalancedMinParentDistanceBalance"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealWindowPairMinParentDistanceBalance"]) fusionSplitPriorCleanTwoRealWindowPairMinParentDistanceBalance = node["fusionSplitPriorCleanTwoRealWindowPairMinParentDistanceBalance"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealWindowPairMaxScore"]) fusionSplitPriorCleanTwoRealWindowPairMaxScore = node["fusionSplitPriorCleanTwoRealWindowPairMaxScore"].as<float>();
        if (node["fusionSplitPriorWindowBackedConflictSelectionBonus"]) fusionSplitPriorWindowBackedConflictSelectionBonus = node["fusionSplitPriorWindowBackedConflictSelectionBonus"].as<float>();
        if (node["fusionSplitPriorWindowBalancedMinNearParentRadiusScale"]) fusionSplitPriorWindowBalancedMinNearParentRadiusScale = node["fusionSplitPriorWindowBalancedMinNearParentRadiusScale"].as<float>();
        if (node["fusionSplitPriorWindowHighConfidenceMaxOverlapCostFraction"]) fusionSplitPriorWindowHighConfidenceMaxOverlapCostFraction = node["fusionSplitPriorWindowHighConfidenceMaxOverlapCostFraction"].as<float>();
        if (node["fusionSplitPriorRankingSoftGateEnabled"]) fusionSplitPriorRankingSoftGateEnabled = node["fusionSplitPriorRankingSoftGateEnabled"].as<bool>();
        if (node["fusionSplitPriorRankingSoftSeparationPenalty"]) fusionSplitPriorRankingSoftSeparationPenalty = node["fusionSplitPriorRankingSoftSeparationPenalty"].as<float>();
        if (node["fusionSplitPriorRankingSoftMidpointPenalty"]) fusionSplitPriorRankingSoftMidpointPenalty = node["fusionSplitPriorRankingSoftMidpointPenalty"].as<float>();
        if (node["fusionSplitPriorRankingSoftScorePenalty"]) fusionSplitPriorRankingSoftScorePenalty = node["fusionSplitPriorRankingSoftScorePenalty"].as<float>();
        if (node["fusionSplitPriorRankingSoftNeighborPenalty"]) fusionSplitPriorRankingSoftNeighborPenalty = node["fusionSplitPriorRankingSoftNeighborPenalty"].as<float>();
        if (node["fusionSplitPriorGlobalSelectMaxCost"]) fusionSplitPriorGlobalSelectMaxCost = node["fusionSplitPriorGlobalSelectMaxCost"].as<float>();
        if (node["fusionSplitPriorEarlyLargeSeparationRescueEnabled"]) fusionSplitPriorEarlyLargeSeparationRescueEnabled = node["fusionSplitPriorEarlyLargeSeparationRescueEnabled"].as<bool>();
        if (node["fusionSplitPriorEarlyLargeSeparationRescueFirstFrameOnly"]) fusionSplitPriorEarlyLargeSeparationRescueFirstFrameOnly = node["fusionSplitPriorEarlyLargeSeparationRescueFirstFrameOnly"].as<bool>();
        if (node["fusionSplitPriorEarlyLargeSeparationRescueMaxFrame"]) fusionSplitPriorEarlyLargeSeparationRescueMaxFrame = node["fusionSplitPriorEarlyLargeSeparationRescueMaxFrame"].as<int>();
        if (node["fusionSplitPriorEarlyLargeSeparationMinSeparation"]) fusionSplitPriorEarlyLargeSeparationMinSeparation = node["fusionSplitPriorEarlyLargeSeparationMinSeparation"].as<float>();
        if (node["fusionSplitPriorEarlyLargeSeparationMinParentDistanceBalance"]) fusionSplitPriorEarlyLargeSeparationMinParentDistanceBalance = node["fusionSplitPriorEarlyLargeSeparationMinParentDistanceBalance"].as<float>();
        if (node["fusionSplitPriorEarlyLargeSeparationMaxScore"]) fusionSplitPriorEarlyLargeSeparationMaxScore = node["fusionSplitPriorEarlyLargeSeparationMaxScore"].as<float>();
        if (node["fusionSplitPriorEarlyLargeSeparationMinWindowBoth"]) fusionSplitPriorEarlyLargeSeparationMinWindowBoth = node["fusionSplitPriorEarlyLargeSeparationMinWindowBoth"].as<int>();
        if (node["fusionSplitPriorEarlyLargeSeparationMaxWindowMissing"]) fusionSplitPriorEarlyLargeSeparationMaxWindowMissing = node["fusionSplitPriorEarlyLargeSeparationMaxWindowMissing"].as<int>();
        if (node["fusionSplitPriorEarlyLargeSeparationMaxWindowParentPersists"]) fusionSplitPriorEarlyLargeSeparationMaxWindowParentPersists = node["fusionSplitPriorEarlyLargeSeparationMaxWindowParentPersists"].as<int>();
        if (node["fusionSplitPriorTemporalRepairEnabled"]) fusionSplitPriorTemporalRepairEnabled = node["fusionSplitPriorTemporalRepairEnabled"].as<bool>();
        if (node["fusionSplitPriorTemporalRepairMinWindowBoth"]) fusionSplitPriorTemporalRepairMinWindowBoth = node["fusionSplitPriorTemporalRepairMinWindowBoth"].as<int>();
        if (node["fusionSplitPriorTemporalRepairMaxWindowMissing"]) fusionSplitPriorTemporalRepairMaxWindowMissing = node["fusionSplitPriorTemporalRepairMaxWindowMissing"].as<int>();
        if (node["fusionSplitPriorTemporalRepairMaxWindowParentPersists"]) fusionSplitPriorTemporalRepairMaxWindowParentPersists = node["fusionSplitPriorTemporalRepairMaxWindowParentPersists"].as<int>();
        if (node["fusionSplitPriorTemporalRepairMinSeparation"]) fusionSplitPriorTemporalRepairMinSeparation = node["fusionSplitPriorTemporalRepairMinSeparation"].as<float>();
        if (node["fusionSplitPriorTemporalRepairMinSeparationRadiusScale"]) fusionSplitPriorTemporalRepairMinSeparationRadiusScale = node["fusionSplitPriorTemporalRepairMinSeparationRadiusScale"].as<float>();
        if (node["fusionSplitPriorTemporalRepairMaxParentDistance"]) fusionSplitPriorTemporalRepairMaxParentDistance = node["fusionSplitPriorTemporalRepairMaxParentDistance"].as<float>();
        if (node["fusionSplitPriorTemporalRepairParentRadiusScale"]) fusionSplitPriorTemporalRepairParentRadiusScale = node["fusionSplitPriorTemporalRepairParentRadiusScale"].as<float>();
        if (node["fusionSplitPriorTemporalRepairMaxCandidateSeparation"]) fusionSplitPriorTemporalRepairMaxCandidateSeparation = node["fusionSplitPriorTemporalRepairMaxCandidateSeparation"].as<float>();
        if (node["fusionSplitPriorTemporalRepairMaxCandidateSeparationRadiusScale"]) fusionSplitPriorTemporalRepairMaxCandidateSeparationRadiusScale = node["fusionSplitPriorTemporalRepairMaxCandidateSeparationRadiusScale"].as<float>();
        if (node["fusionSplitPriorTemporalRepairMinParentDistanceBalance"]) fusionSplitPriorTemporalRepairMinParentDistanceBalance = node["fusionSplitPriorTemporalRepairMinParentDistanceBalance"].as<float>();
        if (node["fusionSplitPriorTemporalRepairMinNearParentDistance"]) fusionSplitPriorTemporalRepairMinNearParentDistance = node["fusionSplitPriorTemporalRepairMinNearParentDistance"].as<float>();
        if (node["fusionSplitPriorTemporalRepairMaxScore"]) fusionSplitPriorTemporalRepairMaxScore = node["fusionSplitPriorTemporalRepairMaxScore"].as<float>();
        if (node["fusionSplitPriorTemporalRepairSelectionBonus"]) fusionSplitPriorTemporalRepairSelectionBonus = node["fusionSplitPriorTemporalRepairSelectionBonus"].as<float>();
        if (node["fusionSplitPriorTemporalRepairMaxNeighborClaimPenalty"]) fusionSplitPriorTemporalRepairMaxNeighborClaimPenalty = node["fusionSplitPriorTemporalRepairMaxNeighborClaimPenalty"].as<float>();
        if (node["fusionSplitPriorTemporalRepairMaxContinuationClaimSoftPenalty"]) fusionSplitPriorTemporalRepairMaxContinuationClaimSoftPenalty = node["fusionSplitPriorTemporalRepairMaxContinuationClaimSoftPenalty"].as<float>();
        if (node["fusionSplitPriorTemporalRepairRejectClaimBlockers"]) fusionSplitPriorTemporalRepairRejectClaimBlockers = node["fusionSplitPriorTemporalRepairRejectClaimBlockers"].as<bool>();
        if (node["fusionSplitPriorTemporalRepairStrongMinVoxels"]) fusionSplitPriorTemporalRepairStrongMinVoxels = node["fusionSplitPriorTemporalRepairStrongMinVoxels"].as<int>();
        if (node["fusionSplitPriorTemporalRepairStrongMinSignal"]) fusionSplitPriorTemporalRepairStrongMinSignal = node["fusionSplitPriorTemporalRepairStrongMinSignal"].as<float>();
        if (node["fusionSplitPriorTemporalRepairStrongMaxNeighborClaimPenalty"]) fusionSplitPriorTemporalRepairStrongMaxNeighborClaimPenalty = node["fusionSplitPriorTemporalRepairStrongMaxNeighborClaimPenalty"].as<float>();
        if (node["fusionSplitPriorTemporalRepairStrongMinParentDistanceBalance"]) fusionSplitPriorTemporalRepairStrongMinParentDistanceBalance = node["fusionSplitPriorTemporalRepairStrongMinParentDistanceBalance"].as<float>();
        if (node["fusionSplitPriorTemporalRepairStrongAsymmetryBalance"]) fusionSplitPriorTemporalRepairStrongAsymmetryBalance = node["fusionSplitPriorTemporalRepairStrongAsymmetryBalance"].as<float>();
        if (node["fusionSplitPriorTemporalRepairMinParentAgeFrames"]) fusionSplitPriorTemporalRepairMinParentAgeFrames = node["fusionSplitPriorTemporalRepairMinParentAgeFrames"].as<int>();
        if (node["fusionSplitPriorMinParentAgeFrames"]) fusionSplitPriorMinParentAgeFrames = node["fusionSplitPriorMinParentAgeFrames"].as<int>();
        if (node["fusionSplitPriorParentAnchorMinShape"]) fusionSplitPriorParentAnchorMinShape = node["fusionSplitPriorParentAnchorMinShape"].as<float>();
        if (node["fusionTemporalCenterRepairEnabled"]) fusionTemporalCenterRepairEnabled = node["fusionTemporalCenterRepairEnabled"].as<bool>();
        if (node["fusionTemporalCenterRepairMinDistance"]) fusionTemporalCenterRepairMinDistance = node["fusionTemporalCenterRepairMinDistance"].as<float>();
        if (node["fusionTemporalCenterRepairMaxDistance"]) fusionTemporalCenterRepairMaxDistance = node["fusionTemporalCenterRepairMaxDistance"].as<float>();
        if (node["fusionTemporalCenterRepairMinWindowSupport"]) fusionTemporalCenterRepairMinWindowSupport = node["fusionTemporalCenterRepairMinWindowSupport"].as<int>();
        if (node["fusionTemporalCenterRepairMaxOldWindowSupport"]) fusionTemporalCenterRepairMaxOldWindowSupport = node["fusionTemporalCenterRepairMaxOldWindowSupport"].as<int>();
        if (node["fusionTemporalCenterRepairMinWindowDistanceGain"]) fusionTemporalCenterRepairMinWindowDistanceGain = node["fusionTemporalCenterRepairMinWindowDistanceGain"].as<float>();
        if (node["fusionTemporalCenterRepairClaimMargin"]) fusionTemporalCenterRepairClaimMargin = node["fusionTemporalCenterRepairClaimMargin"].as<float>();
        if (node["fusionTemporalCenterRepairMinCellAgeFrames"]) fusionTemporalCenterRepairMinCellAgeFrames = node["fusionTemporalCenterRepairMinCellAgeFrames"].as<int>();
        if (node["fusionTemporalCenterRepairOldSupportedMinVoxels"]) fusionTemporalCenterRepairOldSupportedMinVoxels = node["fusionTemporalCenterRepairOldSupportedMinVoxels"].as<int>();
        if (node["fusionTemporalCenterRepairOldSupportedMinSignal"]) fusionTemporalCenterRepairOldSupportedMinSignal = node["fusionTemporalCenterRepairOldSupportedMinSignal"].as<float>();
        if (node["fusionTemporalCenterRepairParentAnchorReanchorEnabled"]) fusionTemporalCenterRepairParentAnchorReanchorEnabled = node["fusionTemporalCenterRepairParentAnchorReanchorEnabled"].as<bool>();
        if (node["fusionTemporalCenterRepairParentAnchorReanchorMinShape"]) fusionTemporalCenterRepairParentAnchorReanchorMinShape = node["fusionTemporalCenterRepairParentAnchorReanchorMinShape"].as<float>();
        if (node["fusionTemporalCenterRepairParentAnchorReanchorMinDistance"]) fusionTemporalCenterRepairParentAnchorReanchorMinDistance = node["fusionTemporalCenterRepairParentAnchorReanchorMinDistance"].as<float>();
        if (node["fusionSplitPriorContinuationClaimGuardEnabled"]) fusionSplitPriorContinuationClaimGuardEnabled = node["fusionSplitPriorContinuationClaimGuardEnabled"].as<bool>();
        if (node["fusionSplitPriorContinuationClaimRadiusScale"]) fusionSplitPriorContinuationClaimRadiusScale = node["fusionSplitPriorContinuationClaimRadiusScale"].as<float>();
        if (node["fusionSplitPriorContinuationClaimTieMargin"]) fusionSplitPriorContinuationClaimTieMargin = node["fusionSplitPriorContinuationClaimTieMargin"].as<float>();
        if (node["fusionSplitPriorContinuationClaimCloseParentRadiusScale"]) fusionSplitPriorContinuationClaimCloseParentRadiusScale = node["fusionSplitPriorContinuationClaimCloseParentRadiusScale"].as<float>();
        if (node["fusionSplitPriorContinuationClaimCloseParentPenalty"]) fusionSplitPriorContinuationClaimCloseParentPenalty = node["fusionSplitPriorContinuationClaimCloseParentPenalty"].as<float>();
        if (node["fusionSplitPriorAllowWindowBackedDuplicateHandoff"]) fusionSplitPriorAllowWindowBackedDuplicateHandoff = node["fusionSplitPriorAllowWindowBackedDuplicateHandoff"].as<bool>();
        if (node["fusionSplitPriorParentAnchorCleanFutureDriftRescueMinImageGain"]) fusionSplitPriorParentAnchorCleanFutureDriftRescueMinImageGain = node["fusionSplitPriorParentAnchorCleanFutureDriftRescueMinImageGain"].as<float>();
        if (node["fusionSplitPriorDuplicateHandoffMinContinuationPenalty"]) fusionSplitPriorDuplicateHandoffMinContinuationPenalty = node["fusionSplitPriorDuplicateHandoffMinContinuationPenalty"].as<float>();
        if (node["fusionSplitPriorDuplicateHandoffNormLimit"]) fusionSplitPriorDuplicateHandoffNormLimit = node["fusionSplitPriorDuplicateHandoffNormLimit"].as<float>();
        if (node["fusionSplitPriorDuplicateHandoffDistanceLimit"]) fusionSplitPriorDuplicateHandoffDistanceLimit = node["fusionSplitPriorDuplicateHandoffDistanceLimit"].as<float>();
        if (node["fusionStaleContinuationPruneEnabled"]) fusionStaleContinuationPruneEnabled = node["fusionStaleContinuationPruneEnabled"].as<bool>();
        if (node["fusionStaleContinuationPruneMaxCurrentDistance"]) fusionStaleContinuationPruneMaxCurrentDistance = node["fusionStaleContinuationPruneMaxCurrentDistance"].as<float>();
        if (node["fusionStaleContinuationPruneMaxFutureSupport"]) fusionStaleContinuationPruneMaxFutureSupport = node["fusionStaleContinuationPruneMaxFutureSupport"].as<int>();
        if (node["fusionStaleContinuationPruneMinCellAgeFrames"]) fusionStaleContinuationPruneMinCellAgeFrames = node["fusionStaleContinuationPruneMinCellAgeFrames"].as<int>();
        if (node["fusionStaleContinuationPruneReanchorParentAnchorOneRealEnabled"]) fusionStaleContinuationPruneReanchorParentAnchorOneRealEnabled = node["fusionStaleContinuationPruneReanchorParentAnchorOneRealEnabled"].as<bool>();
        if (node["fusionStaleContinuationPruneReanchorMinWindowBoth"]) fusionStaleContinuationPruneReanchorMinWindowBoth = node["fusionStaleContinuationPruneReanchorMinWindowBoth"].as<int>();
        if (node["fusionStaleContinuationPruneReanchorMaxWindowMissing"]) fusionStaleContinuationPruneReanchorMaxWindowMissing = node["fusionStaleContinuationPruneReanchorMaxWindowMissing"].as<int>();
        if (node["fusionStaleContinuationPruneReanchorMaxWindowParentPersists"]) fusionStaleContinuationPruneReanchorMaxWindowParentPersists = node["fusionStaleContinuationPruneReanchorMaxWindowParentPersists"].as<int>();
        if (node["fusionStaleContinuationPruneReanchorMinVoxels"]) fusionStaleContinuationPruneReanchorMinVoxels = node["fusionStaleContinuationPruneReanchorMinVoxels"].as<int>();
        if (node["fusionStaleContinuationPruneReanchorMinSignal"]) fusionStaleContinuationPruneReanchorMinSignal = node["fusionStaleContinuationPruneReanchorMinSignal"].as<float>();
        if (node["fusionStaleContinuationPruneReanchorMinMove"]) fusionStaleContinuationPruneReanchorMinMove = node["fusionStaleContinuationPruneReanchorMinMove"].as<float>();
        if (node["fusionStaleContinuationPruneLowQualityEnabled"]) fusionStaleContinuationPruneLowQualityEnabled = node["fusionStaleContinuationPruneLowQualityEnabled"].as<bool>();
        if (node["fusionStaleContinuationPruneLowQualityMaxBrightness"]) fusionStaleContinuationPruneLowQualityMaxBrightness = node["fusionStaleContinuationPruneLowQualityMaxBrightness"].as<float>();
        if (node["fusionStaleContinuationPruneLowQualityMaxMajorRadius"]) fusionStaleContinuationPruneLowQualityMaxMajorRadius = node["fusionStaleContinuationPruneLowQualityMaxMajorRadius"].as<float>();
        if (node["fusionStaleContinuationPruneLowQualityMaxMinorRadius"]) fusionStaleContinuationPruneLowQualityMaxMinorRadius = node["fusionStaleContinuationPruneLowQualityMaxMinorRadius"].as<float>();
        if (node["fusionStaleContinuationPruneLowQualityMaxCellAgeFrames"]) fusionStaleContinuationPruneLowQualityMaxCellAgeFrames = node["fusionStaleContinuationPruneLowQualityMaxCellAgeFrames"].as<int>();
        if (node["fusionStaleContinuationPruneLowQualityIgnoresProtection"]) fusionStaleContinuationPruneLowQualityIgnoresProtection = node["fusionStaleContinuationPruneLowQualityIgnoresProtection"].as<bool>();
        if (node["fusionStaleContinuationPruneClaimedCenterEnabled"]) fusionStaleContinuationPruneClaimedCenterEnabled = node["fusionStaleContinuationPruneClaimedCenterEnabled"].as<bool>();
        if (node["fusionStaleContinuationPruneClaimedCenterMargin"]) fusionStaleContinuationPruneClaimedCenterMargin = node["fusionStaleContinuationPruneClaimedCenterMargin"].as<float>();
        if (node["fusionStaleContinuationPruneClaimedCenterRequireGuardOrNoFuture"]) fusionStaleContinuationPruneClaimedCenterRequireGuardOrNoFuture = node["fusionStaleContinuationPruneClaimedCenterRequireGuardOrNoFuture"].as<bool>();
        if (node["fusionStaleContinuationPruneGuardedNoFutureIgnoresCurrentCenterEnabled"]) fusionStaleContinuationPruneGuardedNoFutureIgnoresCurrentCenterEnabled = node["fusionStaleContinuationPruneGuardedNoFutureIgnoresCurrentCenterEnabled"].as<bool>();
        if (node["fusionStaleContinuationPruneCrowdedBridgeEnabled"]) fusionStaleContinuationPruneCrowdedBridgeEnabled = node["fusionStaleContinuationPruneCrowdedBridgeEnabled"].as<bool>();
        if (node["fusionStaleContinuationPruneCrowdedBridgeMaxNeighborDistance"]) fusionStaleContinuationPruneCrowdedBridgeMaxNeighborDistance = node["fusionStaleContinuationPruneCrowdedBridgeMaxNeighborDistance"].as<float>();
        if (node["fusionStaleContinuationPruneCrowdedBridgeMinBrightnessAdvantage"]) fusionStaleContinuationPruneCrowdedBridgeMinBrightnessAdvantage = node["fusionStaleContinuationPruneCrowdedBridgeMinBrightnessAdvantage"].as<float>();
        if (node["fusionStaleContinuationPruneCrowdedBridgeMinGuardedBrightness"]) fusionStaleContinuationPruneCrowdedBridgeMinGuardedBrightness = node["fusionStaleContinuationPruneCrowdedBridgeMinGuardedBrightness"].as<float>();
        if (node["fusionStaleContinuationPruneCrowdedBridgeMaxGuardedBrightness"]) fusionStaleContinuationPruneCrowdedBridgeMaxGuardedBrightness = node["fusionStaleContinuationPruneCrowdedBridgeMaxGuardedBrightness"].as<float>();
        if (node["fusionStaleContinuationPruneProtectedOneRealSyntheticEnabled"]) fusionStaleContinuationPruneProtectedOneRealSyntheticEnabled = node["fusionStaleContinuationPruneProtectedOneRealSyntheticEnabled"].as<bool>();
        if (node["fusionStaleContinuationPruneProtectedOneRealSyntheticMaxBrightness"]) fusionStaleContinuationPruneProtectedOneRealSyntheticMaxBrightness = node["fusionStaleContinuationPruneProtectedOneRealSyntheticMaxBrightness"].as<float>();
        if (node["fusionStaleContinuationPruneProtectedOneRealSyntheticMaxMajorRadius"]) fusionStaleContinuationPruneProtectedOneRealSyntheticMaxMajorRadius = node["fusionStaleContinuationPruneProtectedOneRealSyntheticMaxMajorRadius"].as<float>();
        if (node["fusionStaleContinuationPruneProtectedOneRealSyntheticMaxMinorRadius"]) fusionStaleContinuationPruneProtectedOneRealSyntheticMaxMinorRadius = node["fusionStaleContinuationPruneProtectedOneRealSyntheticMaxMinorRadius"].as<float>();
        if (node["fusionStaleContinuationPruneProtectedOneRealSyntheticMaxSiblingDistance"]) fusionStaleContinuationPruneProtectedOneRealSyntheticMaxSiblingDistance = node["fusionStaleContinuationPruneProtectedOneRealSyntheticMaxSiblingDistance"].as<float>();
        if (node["fusionStaleContinuationPruneProtectedOneRealSyntheticMaxIndependentCenterDistance"]) fusionStaleContinuationPruneProtectedOneRealSyntheticMaxIndependentCenterDistance = node["fusionStaleContinuationPruneProtectedOneRealSyntheticMaxIndependentCenterDistance"].as<float>();
        if (node["fusionSplitPriorParentAnchorContinuationMergeMinShape"]) fusionSplitPriorParentAnchorContinuationMergeMinShape = node["fusionSplitPriorParentAnchorContinuationMergeMinShape"].as<float>();
        if (node["fusionSplitPriorParentAnchorContinuationMergeNormLimit"]) fusionSplitPriorParentAnchorContinuationMergeNormLimit = node["fusionSplitPriorParentAnchorContinuationMergeNormLimit"].as<float>();
        if (node["fusionSplitPriorParentAnchorContinuationMergeDistanceLimit"]) fusionSplitPriorParentAnchorContinuationMergeDistanceLimit = node["fusionSplitPriorParentAnchorContinuationMergeDistanceLimit"].as<float>();
        if (node["fusionSplitPriorParentAnchorContinuationMergeLineageBonus"]) fusionSplitPriorParentAnchorContinuationMergeLineageBonus = node["fusionSplitPriorParentAnchorContinuationMergeLineageBonus"].as<float>();
        if (node["fusionSplitPriorParentAnchorContinuationMergeRejectClaimConflict"]) fusionSplitPriorParentAnchorContinuationMergeRejectClaimConflict = node["fusionSplitPriorParentAnchorContinuationMergeRejectClaimConflict"].as<bool>();
        if (node["fusionTemporalCenterRepairAllowSharedClaimPcaRelax"]) fusionTemporalCenterRepairAllowSharedClaimPcaRelax = node["fusionTemporalCenterRepairAllowSharedClaimPcaRelax"].as<bool>();
        if (node["fusionTemporalCenterRepairSharedClaimPcaRelaxMinShape"]) fusionTemporalCenterRepairSharedClaimPcaRelaxMinShape = node["fusionTemporalCenterRepairSharedClaimPcaRelaxMinShape"].as<float>();
        if (node["fusionSplitPriorPartialParentAnchorRescueEnabled"]) fusionSplitPriorPartialParentAnchorRescueEnabled = node["fusionSplitPriorPartialParentAnchorRescueEnabled"].as<bool>();
        if (node["fusionSplitPriorPartialParentAnchorMinShape"]) fusionSplitPriorPartialParentAnchorMinShape = node["fusionSplitPriorPartialParentAnchorMinShape"].as<float>();
        if (node["fusionSplitPriorPartialParentAnchorMaxScore"]) fusionSplitPriorPartialParentAnchorMaxScore = node["fusionSplitPriorPartialParentAnchorMaxScore"].as<float>();
        if (node["fusionSplitPriorPartialParentAnchorMinRealVoxels"]) fusionSplitPriorPartialParentAnchorMinRealVoxels = node["fusionSplitPriorPartialParentAnchorMinRealVoxels"].as<int>();
        if (node["fusionSplitPriorPartialParentAnchorMinRealSignal"]) fusionSplitPriorPartialParentAnchorMinRealSignal = node["fusionSplitPriorPartialParentAnchorMinRealSignal"].as<float>();
        if (node["fusionSplitPriorPartialParentAnchorMinWindowBoth"]) fusionSplitPriorPartialParentAnchorMinWindowBoth = node["fusionSplitPriorPartialParentAnchorMinWindowBoth"].as<int>();
        if (node["fusionSplitPriorPartialParentAnchorMaxWindowMissing"]) fusionSplitPriorPartialParentAnchorMaxWindowMissing = node["fusionSplitPriorPartialParentAnchorMaxWindowMissing"].as<int>();
        if (node["fusionSplitPriorPartialParentAnchorMaxWindowParentPersists"]) fusionSplitPriorPartialParentAnchorMaxWindowParentPersists = node["fusionSplitPriorPartialParentAnchorMaxWindowParentPersists"].as<int>();
        if (node["fusionSplitPriorParentAnchorWeakGainDuplicateRescueEnabled"]) fusionSplitPriorParentAnchorWeakGainDuplicateRescueEnabled = node["fusionSplitPriorParentAnchorWeakGainDuplicateRescueEnabled"].as<bool>();
        if (node["fusionSplitPriorParentAnchorWeakGainCleanMinShape"]) fusionSplitPriorParentAnchorWeakGainCleanMinShape = node["fusionSplitPriorParentAnchorWeakGainCleanMinShape"].as<float>();
        if (node["fusionSplitPriorParentAnchorWeakGainPartialMinShape"]) fusionSplitPriorParentAnchorWeakGainPartialMinShape = node["fusionSplitPriorParentAnchorWeakGainPartialMinShape"].as<float>();
        if (node["fusionSplitPriorParentAnchorWeakGainCleanMaxScore"]) fusionSplitPriorParentAnchorWeakGainCleanMaxScore = node["fusionSplitPriorParentAnchorWeakGainCleanMaxScore"].as<float>();
        if (node["fusionSplitPriorParentAnchorWeakGainPartialMaxScore"]) fusionSplitPriorParentAnchorWeakGainPartialMaxScore = node["fusionSplitPriorParentAnchorWeakGainPartialMaxScore"].as<float>();
        if (node["fusionSplitPriorParentAnchorWeakGainCleanMaxOverlapCostFraction"]) fusionSplitPriorParentAnchorWeakGainCleanMaxOverlapCostFraction = node["fusionSplitPriorParentAnchorWeakGainCleanMaxOverlapCostFraction"].as<float>();
        if (node["fusionSplitPriorParentAnchorWeakGainPartialMaxOverlapCostFraction"]) fusionSplitPriorParentAnchorWeakGainPartialMaxOverlapCostFraction = node["fusionSplitPriorParentAnchorWeakGainPartialMaxOverlapCostFraction"].as<float>();
        if (node["fusionSplitPriorParentAnchorWeakGainCleanMinImageGain"]) fusionSplitPriorParentAnchorWeakGainCleanMinImageGain = node["fusionSplitPriorParentAnchorWeakGainCleanMinImageGain"].as<float>();
        if (node["fusionSplitPriorParentAnchorWeakGainPartialMinImageGain"]) fusionSplitPriorParentAnchorWeakGainPartialMinImageGain = node["fusionSplitPriorParentAnchorWeakGainPartialMinImageGain"].as<float>();
        if (node["fusionSplitPriorParentAnchorWeakGainMaxBridgeValleyRatio"]) fusionSplitPriorParentAnchorWeakGainMaxBridgeValleyRatio = node["fusionSplitPriorParentAnchorWeakGainMaxBridgeValleyRatio"].as<float>();
        if (node["fusionSplitPriorParentAnchorWeakGainPartialRequireConflictEvidence"]) fusionSplitPriorParentAnchorWeakGainPartialRequireConflictEvidence = node["fusionSplitPriorParentAnchorWeakGainPartialRequireConflictEvidence"].as<bool>();
        if (node["fusionSplitPriorParentAnchorCompactPositiveOneRealRescueEnabled"]) fusionSplitPriorParentAnchorCompactPositiveOneRealRescueEnabled = node["fusionSplitPriorParentAnchorCompactPositiveOneRealRescueEnabled"].as<bool>();
        if (node["fusionSplitPriorParentAnchorCompactPositiveOneRealMaxParentShape"]) fusionSplitPriorParentAnchorCompactPositiveOneRealMaxParentShape = node["fusionSplitPriorParentAnchorCompactPositiveOneRealMaxParentShape"].as<float>();
        if (node["fusionSplitPriorParentAnchorCompactPositiveOneRealMinPriorScore"]) fusionSplitPriorParentAnchorCompactPositiveOneRealMinPriorScore = node["fusionSplitPriorParentAnchorCompactPositiveOneRealMinPriorScore"].as<float>();
        if (node["fusionSplitPriorParentAnchorCompactPositiveOneRealMaxPriorScore"]) fusionSplitPriorParentAnchorCompactPositiveOneRealMaxPriorScore = node["fusionSplitPriorParentAnchorCompactPositiveOneRealMaxPriorScore"].as<float>();
        if (node["fusionSplitPriorParentAnchorCompactPositiveOneRealMinImageGain"]) fusionSplitPriorParentAnchorCompactPositiveOneRealMinImageGain = node["fusionSplitPriorParentAnchorCompactPositiveOneRealMinImageGain"].as<float>();
        if (node["fusionSplitPriorParentAnchorCompactPositiveOneRealMaxOverlapCostFraction"]) fusionSplitPriorParentAnchorCompactPositiveOneRealMaxOverlapCostFraction = node["fusionSplitPriorParentAnchorCompactPositiveOneRealMaxOverlapCostFraction"].as<float>();
        if (node["fusionSplitPriorParentAnchorCompactPositiveOneRealMaxBridgeValleyRatio"]) fusionSplitPriorParentAnchorCompactPositiveOneRealMaxBridgeValleyRatio = node["fusionSplitPriorParentAnchorCompactPositiveOneRealMaxBridgeValleyRatio"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealPostRefitGuardEnabled"]) fusionSplitPriorParentAnchorOneRealPostRefitGuardEnabled = node["fusionSplitPriorParentAnchorOneRealPostRefitGuardEnabled"].as<bool>();
        if (node["fusionSplitPriorParentAnchorOneRealMaxRefitDrift"]) fusionSplitPriorParentAnchorOneRealMaxRefitDrift = node["fusionSplitPriorParentAnchorOneRealMaxRefitDrift"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealRefitDriftRescueEnabled"]) fusionSplitPriorParentAnchorOneRealRefitDriftRescueEnabled = node["fusionSplitPriorParentAnchorOneRealRefitDriftRescueEnabled"].as<bool>();
        if (node["fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinImageGain"]) fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinImageGain = node["fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinImageGain"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealRefitDriftRescueMaxBridgeValleyRatio"]) fusionSplitPriorParentAnchorOneRealRefitDriftRescueMaxBridgeValleyRatio = node["fusionSplitPriorParentAnchorOneRealRefitDriftRescueMaxBridgeValleyRatio"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinBridgeGapWidth"]) fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinBridgeGapWidth = node["fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinBridgeGapWidth"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinParentShape"]) fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinParentShape = node["fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinParentShape"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealRefitDriftRescueMaxScore"]) fusionSplitPriorParentAnchorOneRealRefitDriftRescueMaxScore = node["fusionSplitPriorParentAnchorOneRealRefitDriftRescueMaxScore"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealCleanHighOverlapMinCost"]) fusionSplitPriorParentAnchorOneRealCleanHighOverlapMinCost = node["fusionSplitPriorParentAnchorOneRealCleanHighOverlapMinCost"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealCleanHighOverlapMinImageGain"]) fusionSplitPriorParentAnchorOneRealCleanHighOverlapMinImageGain = node["fusionSplitPriorParentAnchorOneRealCleanHighOverlapMinImageGain"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealCleanHighOverlapMaxBridgeValleyRatio"]) fusionSplitPriorParentAnchorOneRealCleanHighOverlapMaxBridgeValleyRatio = node["fusionSplitPriorParentAnchorOneRealCleanHighOverlapMaxBridgeValleyRatio"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateEnabled"]) fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateEnabled = node["fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateEnabled"].as<bool>();
        if (node["fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinOverlapCost"]) fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinOverlapCost = node["fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinOverlapCost"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMaxBridgeGapWidth"]) fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMaxBridgeGapWidth = node["fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMaxBridgeGapWidth"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinBridgeValleyRatio"]) fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinBridgeValleyRatio = node["fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinBridgeValleyRatio"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMaxParentShape"]) fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMaxParentShape = node["fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMaxParentShape"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinImageGain"]) fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinImageGain = node["fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinImageGain"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealSeedLockOnRefitCollapse"]) fusionSplitPriorParentAnchorOneRealSeedLockOnRefitCollapse = node["fusionSplitPriorParentAnchorOneRealSeedLockOnRefitCollapse"].as<bool>();
        if (node["fusionSplitPriorParentAnchorOneRealSeedLockMaxScore"]) fusionSplitPriorParentAnchorOneRealSeedLockMaxScore = node["fusionSplitPriorParentAnchorOneRealSeedLockMaxScore"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealSeedLockMinSeedSeparation"]) fusionSplitPriorParentAnchorOneRealSeedLockMinSeedSeparation = node["fusionSplitPriorParentAnchorOneRealSeedLockMinSeedSeparation"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealSeedLockMaxFinalSeedAxisRatio"]) fusionSplitPriorParentAnchorOneRealSeedLockMaxFinalSeedAxisRatio = node["fusionSplitPriorParentAnchorOneRealSeedLockMaxFinalSeedAxisRatio"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealMinImageGainGuardEnabled"]) fusionSplitPriorParentAnchorOneRealMinImageGainGuardEnabled = node["fusionSplitPriorParentAnchorOneRealMinImageGainGuardEnabled"].as<bool>();
        if (node["fusionSplitPriorParentAnchorOneRealCleanMinImageGain"]) fusionSplitPriorParentAnchorOneRealCleanMinImageGain = node["fusionSplitPriorParentAnchorOneRealCleanMinImageGain"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealPartialMinImageGain"]) fusionSplitPriorParentAnchorOneRealPartialMinImageGain = node["fusionSplitPriorParentAnchorOneRealPartialMinImageGain"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealPositiveWindowRescueEnabled"]) fusionSplitPriorParentAnchorOneRealPositiveWindowRescueEnabled = node["fusionSplitPriorParentAnchorOneRealPositiveWindowRescueEnabled"].as<bool>();
        if (node["fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMinShape"]) fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMinShape = node["fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMinShape"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxShape"]) fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxShape = node["fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxShape"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxScore"]) fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxScore = node["fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxScore"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxImageWorsening"]) fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxImageWorsening = node["fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxImageWorsening"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxTotalWorsening"]) fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxTotalWorsening = node["fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxTotalWorsening"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxOverlapCostFraction"]) fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxOverlapCostFraction = node["fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxOverlapCostFraction"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxSoftPenaltyFraction"]) fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxSoftPenaltyFraction = node["fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxSoftPenaltyFraction"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxBridgeValleyRatio"]) fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxBridgeValleyRatio = node["fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMaxBridgeValleyRatio"].as<float>();
        if (node["fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMinRealVoxels"]) fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMinRealVoxels = node["fusionSplitPriorParentAnchorOneRealPositiveWindowRescueMinRealVoxels"].as<int>();
        if (node["fusionSplitPriorCompactParentAnchorWindowRescueEnabled"]) fusionSplitPriorCompactParentAnchorWindowRescueEnabled = node["fusionSplitPriorCompactParentAnchorWindowRescueEnabled"].as<bool>();
        if (node["fusionSplitPriorCompactParentAnchorMinShape"]) fusionSplitPriorCompactParentAnchorMinShape = node["fusionSplitPriorCompactParentAnchorMinShape"].as<float>();
        if (node["fusionSplitPriorCompactParentAnchorMaxScore"]) fusionSplitPriorCompactParentAnchorMaxScore = node["fusionSplitPriorCompactParentAnchorMaxScore"].as<float>();
        if (node["fusionSplitPriorCompactParentAnchorMinRealVoxels"]) fusionSplitPriorCompactParentAnchorMinRealVoxels = node["fusionSplitPriorCompactParentAnchorMinRealVoxels"].as<int>();
        if (node["fusionSplitPriorCompactParentAnchorMinRealSignal"]) fusionSplitPriorCompactParentAnchorMinRealSignal = node["fusionSplitPriorCompactParentAnchorMinRealSignal"].as<float>();
        if (node["fusionSplitPriorCompactParentAnchorMinFarDistanceRadiusScale"]) fusionSplitPriorCompactParentAnchorMinFarDistanceRadiusScale = node["fusionSplitPriorCompactParentAnchorMinFarDistanceRadiusScale"].as<float>();
        if (node["fusionSplitPriorParentAnchorSingleBlockerRescueEnabled"]) fusionSplitPriorParentAnchorSingleBlockerRescueEnabled = node["fusionSplitPriorParentAnchorSingleBlockerRescueEnabled"].as<bool>();
        if (node["fusionSplitPriorParentAnchorSingleBlockerMinShape"]) fusionSplitPriorParentAnchorSingleBlockerMinShape = node["fusionSplitPriorParentAnchorSingleBlockerMinShape"].as<float>();
        if (node["fusionSplitPriorParentAnchorSingleBlockerMaxScore"]) fusionSplitPriorParentAnchorSingleBlockerMaxScore = node["fusionSplitPriorParentAnchorSingleBlockerMaxScore"].as<float>();
        if (node["fusionSplitPriorParentAnchorSingleBlockerMinRealVoxels"]) fusionSplitPriorParentAnchorSingleBlockerMinRealVoxels = node["fusionSplitPriorParentAnchorSingleBlockerMinRealVoxels"].as<int>();
        if (node["fusionSplitPriorParentAnchorSingleBlockerMinRealSignal"]) fusionSplitPriorParentAnchorSingleBlockerMinRealSignal = node["fusionSplitPriorParentAnchorSingleBlockerMinRealSignal"].as<float>();
        if (node["fusionSplitPriorParentAnchorSingleBlockerMinFarDistanceRadiusScale"]) fusionSplitPriorParentAnchorSingleBlockerMinFarDistanceRadiusScale = node["fusionSplitPriorParentAnchorSingleBlockerMinFarDistanceRadiusScale"].as<float>();
        if (node["fusionSplitPriorParentAnchorSingleBlockerMaxContinuationPenalty"]) fusionSplitPriorParentAnchorSingleBlockerMaxContinuationPenalty = node["fusionSplitPriorParentAnchorSingleBlockerMaxContinuationPenalty"].as<float>();
        if (node["fusionSplitPriorParentAnchorSingleBlockerSelectionBonus"]) fusionSplitPriorParentAnchorSingleBlockerSelectionBonus = node["fusionSplitPriorParentAnchorSingleBlockerSelectionBonus"].as<float>();
        if (node["fusionSplitPriorLockAcceptedDaughtersForFinalPca"]) fusionSplitPriorLockAcceptedDaughtersForFinalPca = node["fusionSplitPriorLockAcceptedDaughtersForFinalPca"].as<bool>();
        if (node["fusionSplitPriorCleanTwoRealSingleBlockerRescueEnabled"]) fusionSplitPriorCleanTwoRealSingleBlockerRescueEnabled = node["fusionSplitPriorCleanTwoRealSingleBlockerRescueEnabled"].as<bool>();
        if (node["fusionSplitPriorCleanTwoRealSingleBlockerMinParentDistanceBalance"]) fusionSplitPriorCleanTwoRealSingleBlockerMinParentDistanceBalance = node["fusionSplitPriorCleanTwoRealSingleBlockerMinParentDistanceBalance"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealSingleBlockerMaxScore"]) fusionSplitPriorCleanTwoRealSingleBlockerMaxScore = node["fusionSplitPriorCleanTwoRealSingleBlockerMaxScore"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealSingleBlockerMinVoxels"]) fusionSplitPriorCleanTwoRealSingleBlockerMinVoxels = node["fusionSplitPriorCleanTwoRealSingleBlockerMinVoxels"].as<int>();
        if (node["fusionSplitPriorCleanTwoRealSingleBlockerMinSignal"]) fusionSplitPriorCleanTwoRealSingleBlockerMinSignal = node["fusionSplitPriorCleanTwoRealSingleBlockerMinSignal"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealSingleBlockerSelectionBonus"]) fusionSplitPriorCleanTwoRealSingleBlockerSelectionBonus = node["fusionSplitPriorCleanTwoRealSingleBlockerSelectionBonus"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealAsymmetricRescueEnabled"]) fusionSplitPriorCleanTwoRealAsymmetricRescueEnabled = node["fusionSplitPriorCleanTwoRealAsymmetricRescueEnabled"].as<bool>();
        if (node["fusionSplitPriorCleanTwoRealAsymmetricMinParentDistanceBalance"]) fusionSplitPriorCleanTwoRealAsymmetricMinParentDistanceBalance = node["fusionSplitPriorCleanTwoRealAsymmetricMinParentDistanceBalance"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealAsymmetricMaxScore"]) fusionSplitPriorCleanTwoRealAsymmetricMaxScore = node["fusionSplitPriorCleanTwoRealAsymmetricMaxScore"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealAsymmetricMinVoxels"]) fusionSplitPriorCleanTwoRealAsymmetricMinVoxels = node["fusionSplitPriorCleanTwoRealAsymmetricMinVoxels"].as<int>();
        if (node["fusionSplitPriorCleanTwoRealAsymmetricMinSignal"]) fusionSplitPriorCleanTwoRealAsymmetricMinSignal = node["fusionSplitPriorCleanTwoRealAsymmetricMinSignal"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealAsymmetricMaxNeighborClaimPenalty"]) fusionSplitPriorCleanTwoRealAsymmetricMaxNeighborClaimPenalty = node["fusionSplitPriorCleanTwoRealAsymmetricMaxNeighborClaimPenalty"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealAsymmetricSelectionBonus"]) fusionSplitPriorCleanTwoRealAsymmetricSelectionBonus = node["fusionSplitPriorCleanTwoRealAsymmetricSelectionBonus"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealCompactRescueEnabled"]) fusionSplitPriorCleanTwoRealCompactRescueEnabled = node["fusionSplitPriorCleanTwoRealCompactRescueEnabled"].as<bool>();
        if (node["fusionSplitPriorCleanTwoRealCompactMinParentDistanceBalance"]) fusionSplitPriorCleanTwoRealCompactMinParentDistanceBalance = node["fusionSplitPriorCleanTwoRealCompactMinParentDistanceBalance"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealCompactMaxScore"]) fusionSplitPriorCleanTwoRealCompactMaxScore = node["fusionSplitPriorCleanTwoRealCompactMaxScore"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealCompactMinVoxels"]) fusionSplitPriorCleanTwoRealCompactMinVoxels = node["fusionSplitPriorCleanTwoRealCompactMinVoxels"].as<int>();
        if (node["fusionSplitPriorCleanTwoRealCompactMinSignal"]) fusionSplitPriorCleanTwoRealCompactMinSignal = node["fusionSplitPriorCleanTwoRealCompactMinSignal"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealCompactMaxParentPersistencePenalty"]) fusionSplitPriorCleanTwoRealCompactMaxParentPersistencePenalty = node["fusionSplitPriorCleanTwoRealCompactMaxParentPersistencePenalty"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealCompactMinNearParentDistance"]) fusionSplitPriorCleanTwoRealCompactMinNearParentDistance = node["fusionSplitPriorCleanTwoRealCompactMinNearParentDistance"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealCompactMinSeparationRadiusScale"]) fusionSplitPriorCleanTwoRealCompactMinSeparationRadiusScale = node["fusionSplitPriorCleanTwoRealCompactMinSeparationRadiusScale"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealCompactMaxMidpointDistance"]) fusionSplitPriorCleanTwoRealCompactMaxMidpointDistance = node["fusionSplitPriorCleanTwoRealCompactMaxMidpointDistance"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealCompactSelectionBonus"]) fusionSplitPriorCleanTwoRealCompactSelectionBonus = node["fusionSplitPriorCleanTwoRealCompactSelectionBonus"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealRescueBeforeDefaultNegativeGate"]) fusionSplitPriorCleanTwoRealRescueBeforeDefaultNegativeGate = node["fusionSplitPriorCleanTwoRealRescueBeforeDefaultNegativeGate"].as<bool>();
        if (node["fusionSplitPriorParentAnchorOneRealSharedCandidatePenalty"]) fusionSplitPriorParentAnchorOneRealSharedCandidatePenalty = node["fusionSplitPriorParentAnchorOneRealSharedCandidatePenalty"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealHighNeighborClaimRescueEnabled"]) fusionSplitPriorCleanTwoRealHighNeighborClaimRescueEnabled = node["fusionSplitPriorCleanTwoRealHighNeighborClaimRescueEnabled"].as<bool>();
        if (node["fusionSplitPriorCleanTwoRealHighNeighborClaimMinParentDistanceBalance"]) fusionSplitPriorCleanTwoRealHighNeighborClaimMinParentDistanceBalance = node["fusionSplitPriorCleanTwoRealHighNeighborClaimMinParentDistanceBalance"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealHighNeighborClaimMaxScore"]) fusionSplitPriorCleanTwoRealHighNeighborClaimMaxScore = node["fusionSplitPriorCleanTwoRealHighNeighborClaimMaxScore"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealHighNeighborClaimMinVoxels"]) fusionSplitPriorCleanTwoRealHighNeighborClaimMinVoxels = node["fusionSplitPriorCleanTwoRealHighNeighborClaimMinVoxels"].as<int>();
        if (node["fusionSplitPriorCleanTwoRealHighNeighborClaimMinSignal"]) fusionSplitPriorCleanTwoRealHighNeighborClaimMinSignal = node["fusionSplitPriorCleanTwoRealHighNeighborClaimMinSignal"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealHighNeighborClaimMinNeighborClaimPenalty"]) fusionSplitPriorCleanTwoRealHighNeighborClaimMinNeighborClaimPenalty = node["fusionSplitPriorCleanTwoRealHighNeighborClaimMinNeighborClaimPenalty"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealHighNeighborClaimMaxContinuationClaimPenalty"]) fusionSplitPriorCleanTwoRealHighNeighborClaimMaxContinuationClaimPenalty = node["fusionSplitPriorCleanTwoRealHighNeighborClaimMaxContinuationClaimPenalty"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealHighNeighborClaimSelectionBonus"]) fusionSplitPriorCleanTwoRealHighNeighborClaimSelectionBonus = node["fusionSplitPriorCleanTwoRealHighNeighborClaimSelectionBonus"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealDuplicateBypassEnabled"]) fusionSplitPriorCleanTwoRealDuplicateBypassEnabled = node["fusionSplitPriorCleanTwoRealDuplicateBypassEnabled"].as<bool>();
        if (node["fusionSplitPriorCleanTwoRealDuplicateBypassMinImageGain"]) fusionSplitPriorCleanTwoRealDuplicateBypassMinImageGain = node["fusionSplitPriorCleanTwoRealDuplicateBypassMinImageGain"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealDuplicateBypassMaxBridgeValleyRatio"]) fusionSplitPriorCleanTwoRealDuplicateBypassMaxBridgeValleyRatio = node["fusionSplitPriorCleanTwoRealDuplicateBypassMaxBridgeValleyRatio"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealDuplicateBypassMinBridgeGapWidth"]) fusionSplitPriorCleanTwoRealDuplicateBypassMinBridgeGapWidth = node["fusionSplitPriorCleanTwoRealDuplicateBypassMinBridgeGapWidth"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealDuplicateBypassMinParentDistanceBalance"]) fusionSplitPriorCleanTwoRealDuplicateBypassMinParentDistanceBalance = node["fusionSplitPriorCleanTwoRealDuplicateBypassMinParentDistanceBalance"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealDuplicateBypassMaxScore"]) fusionSplitPriorCleanTwoRealDuplicateBypassMaxScore = node["fusionSplitPriorCleanTwoRealDuplicateBypassMaxScore"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealCompactDuplicateBypassEnabled"]) fusionSplitPriorCleanTwoRealCompactDuplicateBypassEnabled = node["fusionSplitPriorCleanTwoRealCompactDuplicateBypassEnabled"].as<bool>();
        if (node["fusionSplitPriorCleanTwoRealCompactDuplicateBypassMinImageGain"]) fusionSplitPriorCleanTwoRealCompactDuplicateBypassMinImageGain = node["fusionSplitPriorCleanTwoRealCompactDuplicateBypassMinImageGain"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxBridgeValleyRatio"]) fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxBridgeValleyRatio = node["fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxBridgeValleyRatio"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxOverlapCostFraction"]) fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxOverlapCostFraction = node["fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxOverlapCostFraction"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxSoftPenaltyFraction"]) fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxSoftPenaltyFraction = node["fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxSoftPenaltyFraction"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealCompactDuplicateBypassMinParentDistanceBalance"]) fusionSplitPriorCleanTwoRealCompactDuplicateBypassMinParentDistanceBalance = node["fusionSplitPriorCleanTwoRealCompactDuplicateBypassMinParentDistanceBalance"].as<float>();
        if (node["fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxScore"]) fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxScore = node["fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxScore"].as<float>();
        if (node["fusionSplitPriorRejectLowShapeZDominantPairs"]) fusionSplitPriorRejectLowShapeZDominantPairs = node["fusionSplitPriorRejectLowShapeZDominantPairs"].as<bool>();
        if (node["fusionSplitPriorLowShapeZDominantMaxParentShape"]) fusionSplitPriorLowShapeZDominantMaxParentShape = node["fusionSplitPriorLowShapeZDominantMaxParentShape"].as<float>();
        if (node["fusionSplitPriorRejectNonWindowLowShapeOverlapDuplicate"]) fusionSplitPriorRejectNonWindowLowShapeOverlapDuplicate = node["fusionSplitPriorRejectNonWindowLowShapeOverlapDuplicate"].as<bool>();
        if (node["fusionSplitPriorNonWindowLowShapeMaxParentShape"]) fusionSplitPriorNonWindowLowShapeMaxParentShape = node["fusionSplitPriorNonWindowLowShapeMaxParentShape"].as<float>();
        if (node["fusionSplitPriorNonWindowLowShapeMinOverlapCost"]) fusionSplitPriorNonWindowLowShapeMinOverlapCost = node["fusionSplitPriorNonWindowLowShapeMinOverlapCost"].as<float>();
        if (node["fusionSplitPriorNonWindowLowShapeMaxBridgeGapWidth"]) fusionSplitPriorNonWindowLowShapeMaxBridgeGapWidth = node["fusionSplitPriorNonWindowLowShapeMaxBridgeGapWidth"].as<float>();
        if (node["fusionSplitPriorReanchorNonParentDuplicateToOtherDaughter"]) fusionSplitPriorReanchorNonParentDuplicateToOtherDaughter = node["fusionSplitPriorReanchorNonParentDuplicateToOtherDaughter"].as<bool>();
        if (node["fusionSplitPriorNonParentDuplicateReanchorMinImageGain"]) fusionSplitPriorNonParentDuplicateReanchorMinImageGain = node["fusionSplitPriorNonParentDuplicateReanchorMinImageGain"].as<float>();
        if (node["fusionSplitPriorNonParentDuplicateReanchorMaxBridgeValleyRatio"]) fusionSplitPriorNonParentDuplicateReanchorMaxBridgeValleyRatio = node["fusionSplitPriorNonParentDuplicateReanchorMaxBridgeValleyRatio"].as<float>();
        if (node["fusionSplitPriorNonParentDuplicateReanchorMinBridgeGapWidth"]) fusionSplitPriorNonParentDuplicateReanchorMinBridgeGapWidth = node["fusionSplitPriorNonParentDuplicateReanchorMinBridgeGapWidth"].as<float>();
        if (node["fusionSplitPriorNonParentDuplicateReanchorMinMove"]) fusionSplitPriorNonParentDuplicateReanchorMinMove = node["fusionSplitPriorNonParentDuplicateReanchorMinMove"].as<float>();
        if (node["fusionSplitPriorNonParentDuplicateReanchorMinParentDistanceBalance"]) fusionSplitPriorNonParentDuplicateReanchorMinParentDistanceBalance = node["fusionSplitPriorNonParentDuplicateReanchorMinParentDistanceBalance"].as<float>();
        if (node["fusionSplitPriorMinDaughterParentDistance"]) fusionSplitPriorMinDaughterParentDistance = node["fusionSplitPriorMinDaughterParentDistance"].as<float>();
        if (node["fusionSplitPriorMinDaughterParentDistanceRadiusScale"]) fusionSplitPriorMinDaughterParentDistanceRadiusScale = node["fusionSplitPriorMinDaughterParentDistanceRadiusScale"].as<float>();
        if (node["fusionSplitPriorMinParentDistanceBalance"]) fusionSplitPriorMinParentDistanceBalance = node["fusionSplitPriorMinParentDistanceBalance"].as<float>();
        if (node["fusionSplitPriorParentPersistencePenalty"]) fusionSplitPriorParentPersistencePenalty = node["fusionSplitPriorParentPersistencePenalty"].as<float>();
        if (node["fusionSplitPriorNeighborClaimPenalty"]) fusionSplitPriorNeighborClaimPenalty = node["fusionSplitPriorNeighborClaimPenalty"].as<float>();
        if (node["fusionSplitPriorNeighborClaimMargin"]) fusionSplitPriorNeighborClaimMargin = node["fusionSplitPriorNeighborClaimMargin"].as<float>();
        if (node["fusionSplitPriorMaxNeighborClaimPenalty"]) fusionSplitPriorMaxNeighborClaimPenalty = node["fusionSplitPriorMaxNeighborClaimPenalty"].as<float>();
        if (node["fusionSplitPriorDebugTopN"]) fusionSplitPriorDebugTopN = node["fusionSplitPriorDebugTopN"].as<int>();
        if (node["fusionSplitPriorFallbackToClassicOnReject"]) fusionSplitPriorFallbackToClassicOnReject = node["fusionSplitPriorFallbackToClassicOnReject"].as<bool>();
        if (node["fusionSplitPriorUseDedicatedCostGate"]) fusionSplitPriorUseDedicatedCostGate = node["fusionSplitPriorUseDedicatedCostGate"].as<bool>();
        if (node["fusionSplitPriorUseImageCostGate"]) fusionSplitPriorUseImageCostGate = node["fusionSplitPriorUseImageCostGate"].as<bool>();
        if (node["fusionSplitPriorCost"]) fusionSplitPriorCost = node["fusionSplitPriorCost"].as<float>();
        if (node["fusionSplitPriorCostFraction"]) fusionSplitPriorCostFraction = node["fusionSplitPriorCostFraction"].as<float>();
        if (node["fusionSplitPriorMaxPositiveCostFraction"]) fusionSplitPriorMaxPositiveCostFraction = node["fusionSplitPriorMaxPositiveCostFraction"].as<float>();
        if (node["fusionSplitPriorPositiveGateMinImageGain"]) fusionSplitPriorPositiveGateMinImageGain = node["fusionSplitPriorPositiveGateMinImageGain"].as<float>();
        if (node["fusionSplitPriorPositiveGateMinImageGainPenaltyRatio"]) fusionSplitPriorPositiveGateMinImageGainPenaltyRatio = node["fusionSplitPriorPositiveGateMinImageGainPenaltyRatio"].as<float>();
        if (node["fusionSplitPriorPositiveGateElongatedParentMinShape"]) fusionSplitPriorPositiveGateElongatedParentMinShape = node["fusionSplitPriorPositiveGateElongatedParentMinShape"].as<float>();
        if (node["fusionSplitPriorPositiveGateElongatedMaxRawWorsening"]) fusionSplitPriorPositiveGateElongatedMaxRawWorsening = node["fusionSplitPriorPositiveGateElongatedMaxRawWorsening"].as<float>();
        if (node["fusionSplitPriorPositiveGateElongatedMaxSoftPenaltyFraction"]) fusionSplitPriorPositiveGateElongatedMaxSoftPenaltyFraction = node["fusionSplitPriorPositiveGateElongatedMaxSoftPenaltyFraction"].as<float>();
        if (node["fusionSplitPriorPositiveGateElongatedMaxScore"]) fusionSplitPriorPositiveGateElongatedMaxScore = node["fusionSplitPriorPositiveGateElongatedMaxScore"].as<float>();
        if (node["fusionSplitPriorMaxOverlapCostFraction"]) fusionSplitPriorMaxOverlapCostFraction = node["fusionSplitPriorMaxOverlapCostFraction"].as<float>();
        if (node["fusionSplitPriorHighConfidenceMaxScore"]) fusionSplitPriorHighConfidenceMaxScore = node["fusionSplitPriorHighConfidenceMaxScore"].as<float>();
        if (node["fusionSplitPriorHighConfidenceMaxOverlapCostFraction"]) fusionSplitPriorHighConfidenceMaxOverlapCostFraction = node["fusionSplitPriorHighConfidenceMaxOverlapCostFraction"].as<float>();
        if (node["fusionSplitPriorHighConfidenceAxisAlignmentDegrees"]) fusionSplitPriorHighConfidenceAxisAlignmentDegrees = node["fusionSplitPriorHighConfidenceAxisAlignmentDegrees"].as<float>();
        if (node["fusionSplitPriorDaughterVolumeScale"]) fusionSplitPriorDaughterVolumeScale = node["fusionSplitPriorDaughterVolumeScale"].as<float>();
        if (node["fusionSplitPriorPrefilterMaxValleyRatio"]) fusionSplitPriorPrefilterMaxValleyRatio = node["fusionSplitPriorPrefilterMaxValleyRatio"].as<float>();
        if (node["fusionSplitPriorBridgeMaxValleyRatio"]) fusionSplitPriorBridgeMaxValleyRatio = node["fusionSplitPriorBridgeMaxValleyRatio"].as<float>();
        if (node["fusionSplitPriorMinBridgeGapWidth"]) fusionSplitPriorMinBridgeGapWidth = node["fusionSplitPriorMinBridgeGapWidth"].as<float>();
        if (node["fusionSplitPriorMinEdgeBrightness"]) fusionSplitPriorMinEdgeBrightness = node["fusionSplitPriorMinEdgeBrightness"].as<float>();
        if (node["fusionSplitPriorMaxDaughterOverlapFraction"]) fusionSplitPriorMaxDaughterOverlapFraction = node["fusionSplitPriorMaxDaughterOverlapFraction"].as<float>();
        if (node["fusionSplitPriorSoftGateEnabled"]) fusionSplitPriorSoftGateEnabled = node["fusionSplitPriorSoftGateEnabled"].as<bool>();
        if (node["fusionSplitPriorSoftAxisPenaltyFraction"]) fusionSplitPriorSoftAxisPenaltyFraction = node["fusionSplitPriorSoftAxisPenaltyFraction"].as<float>();
        if (node["fusionSplitPriorSoftDaughterOverlapPenaltyFraction"]) fusionSplitPriorSoftDaughterOverlapPenaltyFraction = node["fusionSplitPriorSoftDaughterOverlapPenaltyFraction"].as<float>();
        if (node["fusionSplitPriorSoftValleyPenaltyFraction"]) fusionSplitPriorSoftValleyPenaltyFraction = node["fusionSplitPriorSoftValleyPenaltyFraction"].as<float>();
        if (node["fusionSplitPriorSoftBridgeGapPenaltyFraction"]) fusionSplitPriorSoftBridgeGapPenaltyFraction = node["fusionSplitPriorSoftBridgeGapPenaltyFraction"].as<float>();
        if (node["fusionSplitPriorSoftOverlapCostPenaltyWeight"]) fusionSplitPriorSoftOverlapCostPenaltyWeight = node["fusionSplitPriorSoftOverlapCostPenaltyWeight"].as<float>();
        if (node["fusionSplitPriorBridgeEvidenceWaivesOverlapSoftPenalty"]) fusionSplitPriorBridgeEvidenceWaivesOverlapSoftPenalty = node["fusionSplitPriorBridgeEvidenceWaivesOverlapSoftPenalty"].as<bool>();
        if (node["fusionSplitPriorHardMaxDaughterOverlapFraction"]) fusionSplitPriorHardMaxDaughterOverlapFraction = node["fusionSplitPriorHardMaxDaughterOverlapFraction"].as<float>();
        if (node["fusionSplitPriorHardMaxValleyRatio"]) fusionSplitPriorHardMaxValleyRatio = node["fusionSplitPriorHardMaxValleyRatio"].as<float>();
        if (node["fusionSplitPriorHardMaxOverlapCostFraction"]) fusionSplitPriorHardMaxOverlapCostFraction = node["fusionSplitPriorHardMaxOverlapCostFraction"].as<float>();
        if (node["fusionSplitPriorMinPostRefitLateralSeparation"]) fusionSplitPriorMinPostRefitLateralSeparation = node["fusionSplitPriorMinPostRefitLateralSeparation"].as<float>();
        if (node["fusionSplitPriorMinPostRefitLateralSeparationRadiusScale"]) fusionSplitPriorMinPostRefitLateralSeparationRadiusScale = node["fusionSplitPriorMinPostRefitLateralSeparationRadiusScale"].as<float>();
        if (node["fusionSplitPriorMaxZDominanceForLowLateralSeparation"]) fusionSplitPriorMaxZDominanceForLowLateralSeparation = node["fusionSplitPriorMaxZDominanceForLowLateralSeparation"].as<float>();
        if (node["fusionSplitPriorZDominantTieBreakScoreMargin"]) fusionSplitPriorZDominantTieBreakScoreMargin = node["fusionSplitPriorZDominantTieBreakScoreMargin"].as<float>();
        if (node["fusionSplitPriorZDominantTieBreakMinZDominance"]) fusionSplitPriorZDominantTieBreakMinZDominance = node["fusionSplitPriorZDominantTieBreakMinZDominance"].as<float>();
        if (node["fusionSplitPriorZDominantTieBreakMaxLateralRadiusScale"]) fusionSplitPriorZDominantTieBreakMaxLateralRadiusScale = node["fusionSplitPriorZDominantTieBreakMaxLateralRadiusScale"].as<float>();
        if (node["fusionSplitPriorDynamicOverlapEnabled"]) fusionSplitPriorDynamicOverlapEnabled = node["fusionSplitPriorDynamicOverlapEnabled"].as<bool>();
        if (node["fusionSplitPriorLocalDensityRadiusScale"]) fusionSplitPriorLocalDensityRadiusScale = node["fusionSplitPriorLocalDensityRadiusScale"].as<float>();
        if (node["fusionSplitPriorLocalDensityOverlapBonus"]) fusionSplitPriorLocalDensityOverlapBonus = node["fusionSplitPriorLocalDensityOverlapBonus"].as<float>();
        if (node["fusionSplitPriorMaxDynamicDaughterOverlapFraction"]) fusionSplitPriorMaxDynamicDaughterOverlapFraction = node["fusionSplitPriorMaxDynamicDaughterOverlapFraction"].as<float>();
        if (node["fusionDensityMetricsEnabled"]) fusionDensityMetricsEnabled = node["fusionDensityMetricsEnabled"].as<bool>();
        if (node["fusionDensityAdaptiveGateEnabled"]) fusionDensityAdaptiveGateEnabled = node["fusionDensityAdaptiveGateEnabled"].as<bool>();
        if (node["fusionBrightnessAdaptiveGateEnabled"]) fusionBrightnessAdaptiveGateEnabled = node["fusionBrightnessAdaptiveGateEnabled"].as<bool>();
        if (node["fusionDensityMetricsRadiusScale"]) fusionDensityMetricsRadiusScale = node["fusionDensityMetricsRadiusScale"].as<float>();
        if (node["fusionDensityMetricsK"]) fusionDensityMetricsK = node["fusionDensityMetricsK"].as<int>();
        if (node["fusionBrightnessMetricsRadiusScale"]) fusionBrightnessMetricsRadiusScale = node["fusionBrightnessMetricsRadiusScale"].as<float>();
        if (node["fusionBrightnessMetricsShellScale"]) fusionBrightnessMetricsShellScale = node["fusionBrightnessMetricsShellScale"].as<float>();
        if (node["fusionSplitPriorSkipExistingCellBuriedCheck"]) fusionSplitPriorSkipExistingCellBuriedCheck = node["fusionSplitPriorSkipExistingCellBuriedCheck"].as<bool>();
        if (node["fusionSplitPriorSkipNeighborBridgeCheck"]) fusionSplitPriorSkipNeighborBridgeCheck = node["fusionSplitPriorSkipNeighborBridgeCheck"].as<bool>();
        if (node["fusionSplitPriorBurnInIterations"]) fusionSplitPriorBurnInIterations = node["fusionSplitPriorBurnInIterations"].as<int>();
        if (node["fusionSplitPriorRefineIterations"]) fusionSplitPriorRefineIterations = node["fusionSplitPriorRefineIterations"].as<int>();
        if (node["fusionSplitPriorPrepassFallbackEnabled"]) fusionSplitPriorPrepassFallbackEnabled = node["fusionSplitPriorPrepassFallbackEnabled"].as<bool>();
        if (node["fusionSplitPriorPrepassFallbackMaxPriors"]) fusionSplitPriorPrepassFallbackMaxPriors = node["fusionSplitPriorPrepassFallbackMaxPriors"].as<int>();
        if (node["fusionSplitPriorPrepassFallbackMinKeptPixels"]) fusionSplitPriorPrepassFallbackMinKeptPixels = node["fusionSplitPriorPrepassFallbackMinKeptPixels"].as<int>();
        if (node["fusionSplitPriorPrepassFallbackMinShape"]) fusionSplitPriorPrepassFallbackMinShape = node["fusionSplitPriorPrepassFallbackMinShape"].as<float>();
        if (node["fusionSplitPriorPrepassFallbackMinSeparationRadiusScale"]) fusionSplitPriorPrepassFallbackMinSeparationRadiusScale = node["fusionSplitPriorPrepassFallbackMinSeparationRadiusScale"].as<float>();
        if (node["fusionSplitPriorPrepassFallbackMaxSeparationRadiusScale"]) fusionSplitPriorPrepassFallbackMaxSeparationRadiusScale = node["fusionSplitPriorPrepassFallbackMaxSeparationRadiusScale"].as<float>();
        if (node["fusionSplitPriorPrepassFallbackParentClaimMargin"]) fusionSplitPriorPrepassFallbackParentClaimMargin = node["fusionSplitPriorPrepassFallbackParentClaimMargin"].as<float>();
        if (node["fusionSplitPriorPrepassFallbackMaxScore"]) fusionSplitPriorPrepassFallbackMaxScore = node["fusionSplitPriorPrepassFallbackMaxScore"].as<float>();
        if (node["fusionSplitPriorPrepassFallbackRejectBadLumenParent"]) fusionSplitPriorPrepassFallbackRejectBadLumenParent = node["fusionSplitPriorPrepassFallbackRejectBadLumenParent"].as<bool>();
        if (node["fusionSplitPriorPrepassFallbackBadLumenMaxScore"]) fusionSplitPriorPrepassFallbackBadLumenMaxScore = node["fusionSplitPriorPrepassFallbackBadLumenMaxScore"].as<float>();
        if (node["fusionSplitPriorPrepassFallbackBadLumenMaxNeighborClaimPenalty"]) fusionSplitPriorPrepassFallbackBadLumenMaxNeighborClaimPenalty = node["fusionSplitPriorPrepassFallbackBadLumenMaxNeighborClaimPenalty"].as<float>();
        if (node["fusionSplitPriorPrepassFallbackMinParentAgeFrames"]) fusionSplitPriorPrepassFallbackMinParentAgeFrames = node["fusionSplitPriorPrepassFallbackMinParentAgeFrames"].as<int>();
        if (node["fusionSplitPriorPrepassFallbackUseSnapshotSeedOnLargeDrift"]) fusionSplitPriorPrepassFallbackUseSnapshotSeedOnLargeDrift = node["fusionSplitPriorPrepassFallbackUseSnapshotSeedOnLargeDrift"].as<bool>();
        if (node["fusionSplitPriorPrepassFallbackSeedMaxShift"]) fusionSplitPriorPrepassFallbackSeedMaxShift = node["fusionSplitPriorPrepassFallbackSeedMaxShift"].as<float>();
        if (node["fusionSplitPriorPrepassFallbackSeedMinSeparationRadiusScale"]) fusionSplitPriorPrepassFallbackSeedMinSeparationRadiusScale = node["fusionSplitPriorPrepassFallbackSeedMinSeparationRadiusScale"].as<float>();
        if (node["fusionSplitPriorPrepassFallbackSeedMaxSeparationRadiusScale"]) fusionSplitPriorPrepassFallbackSeedMaxSeparationRadiusScale = node["fusionSplitPriorPrepassFallbackSeedMaxSeparationRadiusScale"].as<float>();
        if (node["fusionSplitPriorPrepassFallbackSeedMaxScore"]) fusionSplitPriorPrepassFallbackSeedMaxScore = node["fusionSplitPriorPrepassFallbackSeedMaxScore"].as<float>();
        if (node["fusionSplitPriorPrepassFallbackSeedMinShape"]) fusionSplitPriorPrepassFallbackSeedMinShape = node["fusionSplitPriorPrepassFallbackSeedMinShape"].as<float>();
        if (node["fusionSplitPriorSnapshotSeedMaxRefitDrift"]) fusionSplitPriorSnapshotSeedMaxRefitDrift = node["fusionSplitPriorSnapshotSeedMaxRefitDrift"].as<float>();
        if (node["fusionReducePostSplitPerturbEnabled"]) fusionReducePostSplitPerturbEnabled = node["fusionReducePostSplitPerturbEnabled"].as<bool>();
        if (node["fusionPostSplitPerturbItersPerCell"]) fusionPostSplitPerturbItersPerCell = node["fusionPostSplitPerturbItersPerCell"].as<int>();
        if (node["maxCandidateMeanVoxelCount"]) maxCandidateMeanVoxelCount = node["maxCandidateMeanVoxelCount"].as<float>();
        if (node["candidateCellCountWeight"]) candidateCellCountWeight = node["candidateCellCountWeight"].as<float>();
        if (node["candidateMeanVoxelPenaltyWeight"]) candidateMeanVoxelPenaltyWeight = node["candidateMeanVoxelPenaltyWeight"].as<float>();
        if (node["candidateNearPairPenaltyWeight"]) candidateNearPairPenaltyWeight = node["candidateNearPairPenaltyWeight"].as<float>();
        if (node["candidatePercentilePreferenceWeight"]) candidatePercentilePreferenceWeight = node["candidatePercentilePreferenceWeight"].as<float>();
        if (node["candidateOvergrowthAlwaysBelowFirstCount"]) candidateOvergrowthAlwaysBelowFirstCount = node["candidateOvergrowthAlwaysBelowFirstCount"].as<int>();
        if (node["candidateOvergrowthMinFirstCount"]) candidateOvergrowthMinFirstCount = node["candidateOvergrowthMinFirstCount"].as<int>();
        if (node["maxCandidateCountGrowthFromFirst"]) maxCandidateCountGrowthFromFirst = node["maxCandidateCountGrowthFromFirst"].as<float>();
        if (node["candidateOvergrowthPenaltyWeight"]) candidateOvergrowthPenaltyWeight = node["candidateOvergrowthPenaltyWeight"].as<float>();
    }

    void printConfig() const {
        std::cout << "CellLumen Config\n";
        std::cout << "enabled: " << enabled << '\n';
        std::cout << "minComponentVoxels: " << minComponentVoxels << '\n';
        std::cout << "minHighSeedVoxels: " << minHighSeedVoxels << '\n';
        std::cout << "seedMergeDistance: " << seedMergeDistance << '\n';
        std::cout << "adaptiveSeedMergeEnabled: " << adaptiveSeedMergeEnabled << '\n';
        std::cout << "seedMergeUseScaledZ: " << seedMergeUseScaledZ << '\n';
        std::cout << "seedSplitSeparation: " << seedSplitSeparation << '\n';
        std::cout << "seededSplitUseScaled3DSeparation: " << seededSplitUseScaled3DSeparation << '\n';
        std::cout << "seededSplitMinScaled3DSeparation: " << seededSplitMinScaled3DSeparation << '\n';
        std::cout << "seededSplitMinScaledZSeparation: " << seededSplitMinScaledZSeparation << '\n';
        std::cout << "seededWatershedLowRescueMinBaseCells: " << seededWatershedLowRescueMinBaseCells << '\n';
        std::cout << "seededWatershedLowRescueMaxBaseCells: " << seededWatershedLowRescueMaxBaseCells << '\n';
        std::cout << "dedupDistance: " << dedupDistance << '\n';
        std::cout << "dedupRadiusScale: " << dedupRadiusScale << '\n';
        std::cout << "adaptiveDedupEnabled: " << adaptiveDedupEnabled << '\n';
        std::cout << "fragmentMergeEnabled: " << fragmentMergeEnabled << '\n';
        std::cout << "fragmentMergeMaxInputCells: " << fragmentMergeMaxInputCells << '\n';
        std::cout << "fragmentMergeSparseMinDistance: " << fragmentMergeSparseMinDistance << '\n';
        std::cout << "fragmentMergeSparseMajorScale: " << fragmentMergeSparseMajorScale << '\n';
        std::cout << "fragmentMergeModerateMinDistance: " << fragmentMergeModerateMinDistance << '\n';
        std::cout << "fragmentMergeModerateMajorScale: " << fragmentMergeModerateMajorScale << '\n';
        std::cout << "finalZColumnRefineMinCells: " << finalZColumnRefineMinCells << '\n';
        std::cout << "finalZColumnRefineMaxCells: " << finalZColumnRefineMaxCells << '\n';
        std::cout << "finalZPeakSplitEnabled: " << finalZPeakSplitEnabled << '\n';
        std::cout << "finalZPeakSplitMinCells: " << finalZPeakSplitMinCells << '\n';
        std::cout << "finalZPeakSplitMaxCells: " << finalZPeakSplitMaxCells << '\n';
        std::cout << "finalZPeakSplitMaxAdded: " << finalZPeakSplitMaxAdded << '\n';
        std::cout << "finalZPeakSplitAddOnlyEnabled: " << finalZPeakSplitAddOnlyEnabled << '\n';
        std::cout << "finalZPeakSplitMinMajorRadius: " << finalZPeakSplitMinMajorRadius << '\n';
        std::cout << "finalZPeakSplitRadiusXY: " << finalZPeakSplitRadiusXY << '\n';
        std::cout << "finalZPeakSplitQuantile: " << finalZPeakSplitQuantile << '\n';
        std::cout << "finalZPeakSplitMinPeakScoreFraction: " << finalZPeakSplitMinPeakScoreFraction << '\n';
        std::cout << "finalZPeakSplitMaxParentTop10MinusShell: " << finalZPeakSplitMaxParentTop10MinusShell << '\n';
        std::cout << "finalZPeakSplitLocal3DFallbackEnabled: " << finalZPeakSplitLocal3DFallbackEnabled << '\n';
        std::cout << "finalZPeakSplitPreferLocal3D: " << finalZPeakSplitPreferLocal3D << '\n';
        std::cout << "finalZPeakSplitLocal3DCentroidRadiusXY: " << finalZPeakSplitLocal3DCentroidRadiusXY << '\n';
        std::cout << "finalZPeakSplitLocal3DCentroidHalfWindowScaled: " << finalZPeakSplitLocal3DCentroidHalfWindowScaled << '\n';
        std::cout << "finalZPeakSplitMinSeparationScaled: " << finalZPeakSplitMinSeparationScaled << '\n';
        std::cout << "finalZPeakSplitMaxSeparationScaled: " << finalZPeakSplitMaxSeparationScaled << '\n';
        std::cout << "finalZPeakSplitMaxCenterShiftXY: " << finalZPeakSplitMaxCenterShiftXY << '\n';
        std::cout << "finalZPeakSplitRadiusScale: " << finalZPeakSplitRadiusScale << '\n';
        std::cout << "finalBrightPairMidpointRescueEnabled: " << finalBrightPairMidpointRescueEnabled << '\n';
        std::cout << "finalBrightPairMidpointRescueMinCells: " << finalBrightPairMidpointRescueMinCells << '\n';
        std::cout << "finalBrightPairMidpointRescueMaxCells: " << finalBrightPairMidpointRescueMaxCells << '\n';
        std::cout << "finalBrightPairMidpointRescueMaxAdded: " << finalBrightPairMidpointRescueMaxAdded << '\n';
        std::cout << "finalBrightPairMidpointRescueMinDistance: " << finalBrightPairMidpointRescueMinDistance << '\n';
        std::cout << "finalBrightPairMidpointRescueMaxDistance: " << finalBrightPairMidpointRescueMaxDistance << '\n';
        std::cout << "finalBrightPairMidpointRescueMinTop10MinusShell: " << finalBrightPairMidpointRescueMinTop10MinusShell << '\n';
        std::cout << "finalBrightPairMidpointRescueMinExistingDistance: " << finalBrightPairMidpointRescueMinExistingDistance << '\n';
        std::cout << "finalBrightPairMidpointRescueSortByExistingDistance: " << finalBrightPairMidpointRescueSortByExistingDistance << '\n';
        std::cout << "finalClusterCentroidRecallRescueEnabled: " << finalClusterCentroidRecallRescueEnabled << '\n';
        std::cout << "finalClusterCentroidRecallRescueMinCells: " << finalClusterCentroidRecallRescueMinCells << '\n';
        std::cout << "finalClusterCentroidRecallRescueMaxCells: " << finalClusterCentroidRecallRescueMaxCells << '\n';
        std::cout << "finalClusterCentroidRecallRescueMaxAdded: " << finalClusterCentroidRecallRescueMaxAdded << '\n';
        std::cout << "finalClusterCentroidRecallRescueMinClusterSize: " << finalClusterCentroidRecallRescueMinClusterSize << '\n';
        std::cout << "finalClusterCentroidRecallRescueClusterDistance: " << finalClusterCentroidRecallRescueClusterDistance << '\n';
        std::cout << "finalClusterCentroidRecallRescueMinAddedDistance: " << finalClusterCentroidRecallRescueMinAddedDistance << '\n';
        std::cout << "finalClusterCentroidRecallRescueMinTop10MinusShell: " << finalClusterCentroidRecallRescueMinTop10MinusShell << '\n';
        std::cout << "finalClusterCentroidRecallRescueRadiusScale: " << finalClusterCentroidRecallRescueRadiusScale << '\n';
        std::cout << "finalClusterCentroidCollapseEnabled: " << finalClusterCentroidCollapseEnabled << '\n';
        std::cout << "finalClusterCentroidCollapseMinCells: " << finalClusterCentroidCollapseMinCells << '\n';
        std::cout << "finalClusterCentroidCollapseMaxCells: " << finalClusterCentroidCollapseMaxCells << '\n';
        std::cout << "finalClusterCentroidCollapseMinClusterSize: " << finalClusterCentroidCollapseMinClusterSize << '\n';
        std::cout << "finalClusterCentroidCollapseLinkDistance: " << finalClusterCentroidCollapseLinkDistance << '\n';
        std::cout << "finalClusterCentroidCollapseRadiusScale: " << finalClusterCentroidCollapseRadiusScale << '\n';
        std::cout << "finalClusterCentroidCollapseUseSignalWeights: " << finalClusterCentroidCollapseUseSignalWeights << '\n';
        std::cout << "finalDominatedDuplicateFilterEnabled: " << finalDominatedDuplicateFilterEnabled << '\n';
        std::cout << "finalDominatedDuplicateFilterDistance: " << finalDominatedDuplicateFilterDistance << '\n';
        std::cout << "finalDominatedDuplicateFilterMinVoxelRatio: " << finalDominatedDuplicateFilterMinVoxelRatio << '\n';
        std::cout << "finalDominatedDuplicateFilterMinSignalRatio: " << finalDominatedDuplicateFilterMinSignalRatio << '\n';
        std::cout << "finalDominatedDuplicateFilterMinRadiusRatio: " << finalDominatedDuplicateFilterMinRadiusRatio << '\n';
        std::cout << "finalDominatedDuplicateFilterMaxLoserTop10MinusShell: " << finalDominatedDuplicateFilterMaxLoserTop10MinusShell << '\n';
        std::cout << "finalSparseIsolatedFloorFilterEnabled: " << finalSparseIsolatedFloorFilterEnabled << '\n';
        std::cout << "finalSparseIsolatedFloorFilterMaxCells: " << finalSparseIsolatedFloorFilterMaxCells << '\n';
        std::cout << "finalSparseIsolatedFloorMaxMajorRadius: " << finalSparseIsolatedFloorMaxMajorRadius << '\n';
        std::cout << "finalSparseIsolatedFloorMaxMinorRadius: " << finalSparseIsolatedFloorMaxMinorRadius << '\n';
        std::cout << "finalSparseIsolatedFloorMinNearestDistance: " << finalSparseIsolatedFloorMinNearestDistance << '\n';
        std::cout << "finalSparseIsolatedFloorWeakSignalMinCells: " << finalSparseIsolatedFloorWeakSignalMinCells << '\n';
        std::cout << "finalSparseIsolatedFloorWeakSignalMaxCells: " << finalSparseIsolatedFloorWeakSignalMaxCells << '\n';
        std::cout << "finalSparseIsolatedFloorMaxTop10MinusShell: " << finalSparseIsolatedFloorMaxTop10MinusShell << '\n';
        std::cout << "radiusInflationScale: " << radiusInflationScale << '\n';
        std::cout << "signalStatsEnabled: " << signalStatsEnabled << '\n';
        std::cout << "minTop10MinusShell: " << minTop10MinusShell << '\n';
        std::cout << "smallArtifactMaxVoxels: " << smallArtifactMaxVoxels << '\n';
        std::cout << "minBiologicalCenterDistance: " << minBiologicalCenterDistance << '\n';
        std::cout << "seededWatershedEnabled: " << seededWatershedEnabled << '\n';
        std::cout << "fusionEnabled: " << fusionEnabled << '\n';
        std::cout << "fusionMinDistanceToExisting: " << fusionMinDistanceToExisting << '\n';
        std::cout << "fusionMaxAddedPerFrame: " << fusionMaxAddedPerFrame << '\n';
        std::cout << "fusionRepairCloseCellsEnabled: " << fusionRepairCloseCellsEnabled << '\n';
        std::cout << "fusionCenterPriorEnabled: " << fusionCenterPriorEnabled << '\n';
        std::cout << "fusionSplitPriorEnabled: " << fusionSplitPriorEnabled << '\n';
        std::cout << "fusionSplitPriorSkipRandomSplits: " << fusionSplitPriorSkipRandomSplits << '\n';
        std::cout << "fusionSplitPriorMaxPriorsPerFrame: " << fusionSplitPriorMaxPriorsPerFrame << '\n';
        std::cout << "fusionSplitPriorMaxScore: " << fusionSplitPriorMaxScore << '\n';
        std::cout << "fusionSplitPriorMinVoxels: " << fusionSplitPriorMinVoxels << '\n';
        std::cout << "fusionSplitPriorMinTop10MinusShell: " << fusionSplitPriorMinTop10MinusShell << '\n';
        std::cout << "fusionSplitPriorRoundParentMaxShape: " << fusionSplitPriorRoundParentMaxShape << '\n';
        std::cout << "fusionSplitPriorRoundParentMinSeparationRadiusScale: " << fusionSplitPriorRoundParentMinSeparationRadiusScale << '\n';
        std::cout << "fusionSplitPriorElongatedParentRescueEnabled: " << fusionSplitPriorElongatedParentRescueEnabled << '\n';
        std::cout << "fusionSplitPriorElongatedParentMinShape: " << fusionSplitPriorElongatedParentMinShape << '\n';
        std::cout << "fusionSplitPriorElongatedParentMaxScore: " << fusionSplitPriorElongatedParentMaxScore << '\n';
        std::cout << "fusionSplitPriorElongatedParentScoreBonus: " << fusionSplitPriorElongatedParentScoreBonus << '\n';
        std::cout << "fusionSplitPriorElongatedParentMaxNeighborClaimPenalty: " << fusionSplitPriorElongatedParentMaxNeighborClaimPenalty << '\n';
        std::cout << "fusionSplitPriorMidpointWeight: " << fusionSplitPriorMidpointWeight << '\n';
        std::cout << "fusionSplitPriorSeparationPenaltyWeight: " << fusionSplitPriorSeparationPenaltyWeight << '\n';
        std::cout << "fusionSplitPriorSignalBonusWeight: " << fusionSplitPriorSignalBonusWeight << '\n';
        std::cout << "fusionSplitPriorConflictReplacementEnabled: " << fusionSplitPriorConflictReplacementEnabled << '\n';
        std::cout << "fusionSplitPriorConflictCloseParentRadiusScale: " << fusionSplitPriorConflictCloseParentRadiusScale << '\n';
        std::cout << "fusionSplitPriorConflictMinNewSeparationRadiusScale: " << fusionSplitPriorConflictMinNewSeparationRadiusScale << '\n';
        std::cout << "fusionSplitPriorConflictMinSeparationDrop: " << fusionSplitPriorConflictMinSeparationDrop << '\n';
        std::cout << "fusionSplitPriorConflictMinOldScoreForReplacement: " << fusionSplitPriorConflictMinOldScoreForReplacement << '\n';
        std::cout << "fusionSplitPriorSuppressConflictPenaltiesForAsymmetricClose: " << fusionSplitPriorSuppressConflictPenaltiesForAsymmetricClose << '\n';
        std::cout << "fusionSplitPriorWindowEnabled: " << fusionSplitPriorWindowEnabled << '\n';
        std::cout << "fusionSplitPriorWindowSize: " << fusionSplitPriorWindowSize << '\n';
        std::cout << "fusionSplitPriorWindowMatchDistance: " << fusionSplitPriorWindowMatchDistance << '\n';
        std::cout << "fusionSplitPriorWindowMatchDistancePerFrame: " << fusionSplitPriorWindowMatchDistancePerFrame << '\n';
        std::cout << "fusionSplitPriorWindowDaughterSupportBonus: " << fusionSplitPriorWindowDaughterSupportBonus << '\n';
        std::cout << "fusionSplitPriorWindowMissingDaughterPenalty: " << fusionSplitPriorWindowMissingDaughterPenalty << '\n';
        std::cout << "fusionSplitPriorWindowParentPersistencePenalty: " << fusionSplitPriorWindowParentPersistencePenalty << '\n';
        std::cout << "fusionSplitPriorWindowBalancedDaughterBonus: " << fusionSplitPriorWindowBalancedDaughterBonus << '\n';
        std::cout << "fusionSplitPriorWindowBalancedMinParentDistanceBalance: " << fusionSplitPriorWindowBalancedMinParentDistanceBalance << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealWindowPairMinParentDistanceBalance: " << fusionSplitPriorCleanTwoRealWindowPairMinParentDistanceBalance << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealWindowPairMaxScore: " << fusionSplitPriorCleanTwoRealWindowPairMaxScore << '\n';
        std::cout << "fusionSplitPriorWindowBackedConflictSelectionBonus: " << fusionSplitPriorWindowBackedConflictSelectionBonus << '\n';
        std::cout << "fusionSplitPriorWindowBalancedMinNearParentRadiusScale: " << fusionSplitPriorWindowBalancedMinNearParentRadiusScale << '\n';
        std::cout << "fusionSplitPriorWindowHighConfidenceMaxOverlapCostFraction: " << fusionSplitPriorWindowHighConfidenceMaxOverlapCostFraction << '\n';
        std::cout << "fusionSplitPriorRankingSoftGateEnabled: " << fusionSplitPriorRankingSoftGateEnabled << '\n';
        std::cout << "fusionSplitPriorRankingSoftSeparationPenalty: " << fusionSplitPriorRankingSoftSeparationPenalty << '\n';
        std::cout << "fusionSplitPriorRankingSoftMidpointPenalty: " << fusionSplitPriorRankingSoftMidpointPenalty << '\n';
        std::cout << "fusionSplitPriorRankingSoftScorePenalty: " << fusionSplitPriorRankingSoftScorePenalty << '\n';
        std::cout << "fusionSplitPriorRankingSoftNeighborPenalty: " << fusionSplitPriorRankingSoftNeighborPenalty << '\n';
        std::cout << "fusionSplitPriorGlobalSelectMaxCost: " << fusionSplitPriorGlobalSelectMaxCost << '\n';
        std::cout << "fusionSplitPriorTemporalRepairEnabled: " << fusionSplitPriorTemporalRepairEnabled << '\n';
        std::cout << "fusionSplitPriorTemporalRepairMinWindowBoth: " << fusionSplitPriorTemporalRepairMinWindowBoth << '\n';
        std::cout << "fusionSplitPriorTemporalRepairMaxWindowMissing: " << fusionSplitPriorTemporalRepairMaxWindowMissing << '\n';
        std::cout << "fusionSplitPriorTemporalRepairMaxWindowParentPersists: " << fusionSplitPriorTemporalRepairMaxWindowParentPersists << '\n';
        std::cout << "fusionSplitPriorTemporalRepairMinSeparation: " << fusionSplitPriorTemporalRepairMinSeparation << '\n';
        std::cout << "fusionSplitPriorTemporalRepairMinSeparationRadiusScale: " << fusionSplitPriorTemporalRepairMinSeparationRadiusScale << '\n';
        std::cout << "fusionSplitPriorTemporalRepairMaxParentDistance: " << fusionSplitPriorTemporalRepairMaxParentDistance << '\n';
        std::cout << "fusionSplitPriorTemporalRepairParentRadiusScale: " << fusionSplitPriorTemporalRepairParentRadiusScale << '\n';
        std::cout << "fusionSplitPriorTemporalRepairMaxCandidateSeparation: " << fusionSplitPriorTemporalRepairMaxCandidateSeparation << '\n';
        std::cout << "fusionSplitPriorTemporalRepairMaxCandidateSeparationRadiusScale: " << fusionSplitPriorTemporalRepairMaxCandidateSeparationRadiusScale << '\n';
        std::cout << "fusionSplitPriorTemporalRepairMinParentDistanceBalance: " << fusionSplitPriorTemporalRepairMinParentDistanceBalance << '\n';
        std::cout << "fusionSplitPriorTemporalRepairMinNearParentDistance: " << fusionSplitPriorTemporalRepairMinNearParentDistance << '\n';
        std::cout << "fusionSplitPriorTemporalRepairMaxScore: " << fusionSplitPriorTemporalRepairMaxScore << '\n';
        std::cout << "fusionSplitPriorTemporalRepairSelectionBonus: " << fusionSplitPriorTemporalRepairSelectionBonus << '\n';
        std::cout << "fusionSplitPriorTemporalRepairMaxNeighborClaimPenalty: " << fusionSplitPriorTemporalRepairMaxNeighborClaimPenalty << '\n';
        std::cout << "fusionSplitPriorTemporalRepairMaxContinuationClaimSoftPenalty: " << fusionSplitPriorTemporalRepairMaxContinuationClaimSoftPenalty << '\n';
        std::cout << "fusionSplitPriorTemporalRepairRejectClaimBlockers: " << fusionSplitPriorTemporalRepairRejectClaimBlockers << '\n';
        std::cout << "fusionSplitPriorTemporalRepairStrongMinVoxels: " << fusionSplitPriorTemporalRepairStrongMinVoxels << '\n';
        std::cout << "fusionSplitPriorTemporalRepairStrongMinSignal: " << fusionSplitPriorTemporalRepairStrongMinSignal << '\n';
        std::cout << "fusionSplitPriorTemporalRepairStrongMaxNeighborClaimPenalty: " << fusionSplitPriorTemporalRepairStrongMaxNeighborClaimPenalty << '\n';
        std::cout << "fusionSplitPriorTemporalRepairStrongMinParentDistanceBalance: " << fusionSplitPriorTemporalRepairStrongMinParentDistanceBalance << '\n';
        std::cout << "fusionSplitPriorTemporalRepairStrongAsymmetryBalance: " << fusionSplitPriorTemporalRepairStrongAsymmetryBalance << '\n';
        std::cout << "fusionSplitPriorTemporalRepairMinParentAgeFrames: " << fusionSplitPriorTemporalRepairMinParentAgeFrames << '\n';
        std::cout << "fusionSplitPriorMinParentAgeFrames: " << fusionSplitPriorMinParentAgeFrames << '\n';
        std::cout << "fusionTemporalCenterRepairEnabled: " << fusionTemporalCenterRepairEnabled << '\n';
        std::cout << "fusionTemporalCenterRepairMinDistance: " << fusionTemporalCenterRepairMinDistance << '\n';
        std::cout << "fusionTemporalCenterRepairMaxDistance: " << fusionTemporalCenterRepairMaxDistance << '\n';
        std::cout << "fusionTemporalCenterRepairMinWindowSupport: " << fusionTemporalCenterRepairMinWindowSupport << '\n';
        std::cout << "fusionTemporalCenterRepairMaxOldWindowSupport: " << fusionTemporalCenterRepairMaxOldWindowSupport << '\n';
        std::cout << "fusionTemporalCenterRepairMinWindowDistanceGain: " << fusionTemporalCenterRepairMinWindowDistanceGain << '\n';
        std::cout << "fusionTemporalCenterRepairClaimMargin: " << fusionTemporalCenterRepairClaimMargin << '\n';
        std::cout << "fusionTemporalCenterRepairMinCellAgeFrames: " << fusionTemporalCenterRepairMinCellAgeFrames << '\n';
        std::cout << "fusionTemporalCenterRepairOldSupportedMinVoxels: " << fusionTemporalCenterRepairOldSupportedMinVoxels << '\n';
        std::cout << "fusionTemporalCenterRepairOldSupportedMinSignal: " << fusionTemporalCenterRepairOldSupportedMinSignal << '\n';
        std::cout << "fusionTemporalCenterRepairParentAnchorReanchorEnabled: " << fusionTemporalCenterRepairParentAnchorReanchorEnabled << '\n';
        std::cout << "fusionTemporalCenterRepairParentAnchorReanchorMinShape: " << fusionTemporalCenterRepairParentAnchorReanchorMinShape << '\n';
        std::cout << "fusionTemporalCenterRepairParentAnchorReanchorMinDistance: " << fusionTemporalCenterRepairParentAnchorReanchorMinDistance << '\n';
        std::cout << "fusionSplitPriorParentAnchorMinShape: " << fusionSplitPriorParentAnchorMinShape << '\n';
        std::cout << "fusionSplitPriorContinuationClaimGuardEnabled: " << fusionSplitPriorContinuationClaimGuardEnabled << '\n';
        std::cout << "fusionSplitPriorContinuationClaimRadiusScale: " << fusionSplitPriorContinuationClaimRadiusScale << '\n';
        std::cout << "fusionSplitPriorContinuationClaimTieMargin: " << fusionSplitPriorContinuationClaimTieMargin << '\n';
        std::cout << "fusionSplitPriorContinuationClaimCloseParentRadiusScale: " << fusionSplitPriorContinuationClaimCloseParentRadiusScale << '\n';
        std::cout << "fusionSplitPriorContinuationClaimCloseParentPenalty: " << fusionSplitPriorContinuationClaimCloseParentPenalty << '\n';
        std::cout << "fusionSplitPriorAllowWindowBackedDuplicateHandoff: " << fusionSplitPriorAllowWindowBackedDuplicateHandoff << '\n';
        std::cout << "fusionSplitPriorParentAnchorCleanFutureDriftRescueMinImageGain: " << fusionSplitPriorParentAnchorCleanFutureDriftRescueMinImageGain << '\n';
        std::cout << "fusionSplitPriorDuplicateHandoffMinContinuationPenalty: " << fusionSplitPriorDuplicateHandoffMinContinuationPenalty << '\n';
        std::cout << "fusionSplitPriorDuplicateHandoffNormLimit: " << fusionSplitPriorDuplicateHandoffNormLimit << '\n';
        std::cout << "fusionSplitPriorDuplicateHandoffDistanceLimit: " << fusionSplitPriorDuplicateHandoffDistanceLimit << '\n';
        std::cout << "fusionStaleContinuationPruneEnabled: " << fusionStaleContinuationPruneEnabled << '\n';
        std::cout << "fusionStaleContinuationPruneMaxCurrentDistance: " << fusionStaleContinuationPruneMaxCurrentDistance << '\n';
        std::cout << "fusionStaleContinuationPruneMaxFutureSupport: " << fusionStaleContinuationPruneMaxFutureSupport << '\n';
        std::cout << "fusionStaleContinuationPruneMinCellAgeFrames: " << fusionStaleContinuationPruneMinCellAgeFrames << '\n';
        std::cout << "fusionStaleContinuationPruneReanchorParentAnchorOneRealEnabled: " << fusionStaleContinuationPruneReanchorParentAnchorOneRealEnabled << '\n';
        std::cout << "fusionStaleContinuationPruneReanchorMinWindowBoth: " << fusionStaleContinuationPruneReanchorMinWindowBoth << '\n';
        std::cout << "fusionStaleContinuationPruneReanchorMaxWindowMissing: " << fusionStaleContinuationPruneReanchorMaxWindowMissing << '\n';
        std::cout << "fusionStaleContinuationPruneReanchorMaxWindowParentPersists: " << fusionStaleContinuationPruneReanchorMaxWindowParentPersists << '\n';
        std::cout << "fusionStaleContinuationPruneReanchorMinVoxels: " << fusionStaleContinuationPruneReanchorMinVoxels << '\n';
        std::cout << "fusionStaleContinuationPruneReanchorMinSignal: " << fusionStaleContinuationPruneReanchorMinSignal << '\n';
        std::cout << "fusionStaleContinuationPruneReanchorMinMove: " << fusionStaleContinuationPruneReanchorMinMove << '\n';
        std::cout << "fusionStaleContinuationPruneLowQualityEnabled: " << fusionStaleContinuationPruneLowQualityEnabled << '\n';
        std::cout << "fusionStaleContinuationPruneLowQualityMaxBrightness: " << fusionStaleContinuationPruneLowQualityMaxBrightness << '\n';
        std::cout << "fusionStaleContinuationPruneLowQualityMaxMajorRadius: " << fusionStaleContinuationPruneLowQualityMaxMajorRadius << '\n';
        std::cout << "fusionStaleContinuationPruneLowQualityMaxMinorRadius: " << fusionStaleContinuationPruneLowQualityMaxMinorRadius << '\n';
        std::cout << "fusionStaleContinuationPruneLowQualityMaxCellAgeFrames: " << fusionStaleContinuationPruneLowQualityMaxCellAgeFrames << '\n';
        std::cout << "fusionStaleContinuationPruneLowQualityIgnoresProtection: " << fusionStaleContinuationPruneLowQualityIgnoresProtection << '\n';
        std::cout << "fusionStaleContinuationPruneClaimedCenterEnabled: " << fusionStaleContinuationPruneClaimedCenterEnabled << '\n';
        std::cout << "fusionStaleContinuationPruneClaimedCenterMargin: " << fusionStaleContinuationPruneClaimedCenterMargin << '\n';
        std::cout << "fusionStaleContinuationPruneClaimedCenterRequireGuardOrNoFuture: " << fusionStaleContinuationPruneClaimedCenterRequireGuardOrNoFuture << '\n';
        std::cout << "fusionStaleContinuationPruneGuardedNoFutureIgnoresCurrentCenterEnabled: " << fusionStaleContinuationPruneGuardedNoFutureIgnoresCurrentCenterEnabled << '\n';
        std::cout << "fusionStaleContinuationPruneCrowdedBridgeEnabled: " << fusionStaleContinuationPruneCrowdedBridgeEnabled << '\n';
        std::cout << "fusionStaleContinuationPruneCrowdedBridgeMaxNeighborDistance: " << fusionStaleContinuationPruneCrowdedBridgeMaxNeighborDistance << '\n';
        std::cout << "fusionStaleContinuationPruneCrowdedBridgeMinBrightnessAdvantage: " << fusionStaleContinuationPruneCrowdedBridgeMinBrightnessAdvantage << '\n';
        std::cout << "fusionStaleContinuationPruneCrowdedBridgeMinGuardedBrightness: " << fusionStaleContinuationPruneCrowdedBridgeMinGuardedBrightness << '\n';
        std::cout << "fusionStaleContinuationPruneCrowdedBridgeMaxGuardedBrightness: " << fusionStaleContinuationPruneCrowdedBridgeMaxGuardedBrightness << '\n';
        std::cout << "fusionStaleContinuationPruneProtectedOneRealSyntheticEnabled: " << fusionStaleContinuationPruneProtectedOneRealSyntheticEnabled << '\n';
        std::cout << "fusionStaleContinuationPruneProtectedOneRealSyntheticMaxBrightness: " << fusionStaleContinuationPruneProtectedOneRealSyntheticMaxBrightness << '\n';
        std::cout << "fusionStaleContinuationPruneProtectedOneRealSyntheticMaxMajorRadius: " << fusionStaleContinuationPruneProtectedOneRealSyntheticMaxMajorRadius << '\n';
        std::cout << "fusionStaleContinuationPruneProtectedOneRealSyntheticMaxMinorRadius: " << fusionStaleContinuationPruneProtectedOneRealSyntheticMaxMinorRadius << '\n';
        std::cout << "fusionStaleContinuationPruneProtectedOneRealSyntheticMaxSiblingDistance: " << fusionStaleContinuationPruneProtectedOneRealSyntheticMaxSiblingDistance << '\n';
        std::cout << "fusionStaleContinuationPruneProtectedOneRealSyntheticMaxIndependentCenterDistance: " << fusionStaleContinuationPruneProtectedOneRealSyntheticMaxIndependentCenterDistance << '\n';
        std::cout << "fusionSplitPriorParentAnchorContinuationMergeMinShape: " << fusionSplitPriorParentAnchorContinuationMergeMinShape << '\n';
        std::cout << "fusionSplitPriorParentAnchorContinuationMergeNormLimit: " << fusionSplitPriorParentAnchorContinuationMergeNormLimit << '\n';
        std::cout << "fusionSplitPriorParentAnchorContinuationMergeDistanceLimit: " << fusionSplitPriorParentAnchorContinuationMergeDistanceLimit << '\n';
        std::cout << "fusionSplitPriorParentAnchorContinuationMergeLineageBonus: " << fusionSplitPriorParentAnchorContinuationMergeLineageBonus << '\n';
        std::cout << "fusionSplitPriorParentAnchorContinuationMergeRejectClaimConflict: " << fusionSplitPriorParentAnchorContinuationMergeRejectClaimConflict << '\n';
        std::cout << "fusionTemporalCenterRepairAllowSharedClaimPcaRelax: " << fusionTemporalCenterRepairAllowSharedClaimPcaRelax << '\n';
        std::cout << "fusionTemporalCenterRepairSharedClaimPcaRelaxMinShape: " << fusionTemporalCenterRepairSharedClaimPcaRelaxMinShape << '\n';
        std::cout << "fusionSplitPriorPartialParentAnchorRescueEnabled: " << fusionSplitPriorPartialParentAnchorRescueEnabled << '\n';
        std::cout << "fusionSplitPriorPartialParentAnchorMinShape: " << fusionSplitPriorPartialParentAnchorMinShape << '\n';
        std::cout << "fusionSplitPriorPartialParentAnchorMaxScore: " << fusionSplitPriorPartialParentAnchorMaxScore << '\n';
        std::cout << "fusionSplitPriorPartialParentAnchorMinRealVoxels: " << fusionSplitPriorPartialParentAnchorMinRealVoxels << '\n';
        std::cout << "fusionSplitPriorPartialParentAnchorMinRealSignal: " << fusionSplitPriorPartialParentAnchorMinRealSignal << '\n';
        std::cout << "fusionSplitPriorPartialParentAnchorMinWindowBoth: " << fusionSplitPriorPartialParentAnchorMinWindowBoth << '\n';
        std::cout << "fusionSplitPriorPartialParentAnchorMaxWindowMissing: " << fusionSplitPriorPartialParentAnchorMaxWindowMissing << '\n';
        std::cout << "fusionSplitPriorPartialParentAnchorMaxWindowParentPersists: " << fusionSplitPriorPartialParentAnchorMaxWindowParentPersists << '\n';
        std::cout << "fusionSplitPriorParentAnchorWeakGainDuplicateRescueEnabled: " << fusionSplitPriorParentAnchorWeakGainDuplicateRescueEnabled << '\n';
        std::cout << "fusionSplitPriorParentAnchorWeakGainCleanMinShape: " << fusionSplitPriorParentAnchorWeakGainCleanMinShape << '\n';
        std::cout << "fusionSplitPriorParentAnchorWeakGainPartialMinShape: " << fusionSplitPriorParentAnchorWeakGainPartialMinShape << '\n';
        std::cout << "fusionSplitPriorParentAnchorWeakGainCleanMaxScore: " << fusionSplitPriorParentAnchorWeakGainCleanMaxScore << '\n';
        std::cout << "fusionSplitPriorParentAnchorWeakGainPartialMaxScore: " << fusionSplitPriorParentAnchorWeakGainPartialMaxScore << '\n';
        std::cout << "fusionSplitPriorParentAnchorWeakGainCleanMaxOverlapCostFraction: " << fusionSplitPriorParentAnchorWeakGainCleanMaxOverlapCostFraction << '\n';
        std::cout << "fusionSplitPriorParentAnchorWeakGainPartialMaxOverlapCostFraction: " << fusionSplitPriorParentAnchorWeakGainPartialMaxOverlapCostFraction << '\n';
        std::cout << "fusionSplitPriorParentAnchorWeakGainCleanMinImageGain: " << fusionSplitPriorParentAnchorWeakGainCleanMinImageGain << '\n';
        std::cout << "fusionSplitPriorParentAnchorWeakGainPartialMinImageGain: " << fusionSplitPriorParentAnchorWeakGainPartialMinImageGain << '\n';
        std::cout << "fusionSplitPriorParentAnchorWeakGainMaxBridgeValleyRatio: " << fusionSplitPriorParentAnchorWeakGainMaxBridgeValleyRatio << '\n';
        std::cout << "fusionSplitPriorParentAnchorWeakGainPartialRequireConflictEvidence: " << fusionSplitPriorParentAnchorWeakGainPartialRequireConflictEvidence << '\n';
        std::cout << "fusionSplitPriorParentAnchorCompactPositiveOneRealRescueEnabled: " << fusionSplitPriorParentAnchorCompactPositiveOneRealRescueEnabled << '\n';
        std::cout << "fusionSplitPriorParentAnchorCompactPositiveOneRealMaxParentShape: " << fusionSplitPriorParentAnchorCompactPositiveOneRealMaxParentShape << '\n';
        std::cout << "fusionSplitPriorParentAnchorCompactPositiveOneRealMinPriorScore: " << fusionSplitPriorParentAnchorCompactPositiveOneRealMinPriorScore << '\n';
        std::cout << "fusionSplitPriorParentAnchorCompactPositiveOneRealMaxPriorScore: " << fusionSplitPriorParentAnchorCompactPositiveOneRealMaxPriorScore << '\n';
        std::cout << "fusionSplitPriorParentAnchorCompactPositiveOneRealMinImageGain: " << fusionSplitPriorParentAnchorCompactPositiveOneRealMinImageGain << '\n';
        std::cout << "fusionSplitPriorParentAnchorCompactPositiveOneRealMaxOverlapCostFraction: " << fusionSplitPriorParentAnchorCompactPositiveOneRealMaxOverlapCostFraction << '\n';
        std::cout << "fusionSplitPriorParentAnchorCompactPositiveOneRealMaxBridgeValleyRatio: " << fusionSplitPriorParentAnchorCompactPositiveOneRealMaxBridgeValleyRatio << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealPostRefitGuardEnabled: " << fusionSplitPriorParentAnchorOneRealPostRefitGuardEnabled << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealMaxRefitDrift: " << fusionSplitPriorParentAnchorOneRealMaxRefitDrift << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealRefitDriftRescueEnabled: " << fusionSplitPriorParentAnchorOneRealRefitDriftRescueEnabled << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinImageGain: " << fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinImageGain << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealRefitDriftRescueMaxBridgeValleyRatio: " << fusionSplitPriorParentAnchorOneRealRefitDriftRescueMaxBridgeValleyRatio << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinBridgeGapWidth: " << fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinBridgeGapWidth << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinParentShape: " << fusionSplitPriorParentAnchorOneRealRefitDriftRescueMinParentShape << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealRefitDriftRescueMaxScore: " << fusionSplitPriorParentAnchorOneRealRefitDriftRescueMaxScore << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealCleanHighOverlapMinCost: " << fusionSplitPriorParentAnchorOneRealCleanHighOverlapMinCost << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealCleanHighOverlapMinImageGain: " << fusionSplitPriorParentAnchorOneRealCleanHighOverlapMinImageGain << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealCleanHighOverlapMaxBridgeValleyRatio: " << fusionSplitPriorParentAnchorOneRealCleanHighOverlapMaxBridgeValleyRatio << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateEnabled: " << fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateEnabled << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinOverlapCost: " << fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinOverlapCost << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMaxBridgeGapWidth: " << fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMaxBridgeGapWidth << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinBridgeValleyRatio: " << fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinBridgeValleyRatio << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMaxParentShape: " << fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMaxParentShape << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinImageGain: " << fusionSplitPriorParentAnchorOneRealOverlapNoGapDuplicateMinImageGain << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealSeedLockOnRefitCollapse: " << fusionSplitPriorParentAnchorOneRealSeedLockOnRefitCollapse << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealSeedLockMaxScore: " << fusionSplitPriorParentAnchorOneRealSeedLockMaxScore << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealSeedLockMinSeedSeparation: " << fusionSplitPriorParentAnchorOneRealSeedLockMinSeedSeparation << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealSeedLockMaxFinalSeedAxisRatio: " << fusionSplitPriorParentAnchorOneRealSeedLockMaxFinalSeedAxisRatio << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealMinImageGainGuardEnabled: " << fusionSplitPriorParentAnchorOneRealMinImageGainGuardEnabled << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealCleanMinImageGain: " << fusionSplitPriorParentAnchorOneRealCleanMinImageGain << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealPartialMinImageGain: " << fusionSplitPriorParentAnchorOneRealPartialMinImageGain << '\n';
        std::cout << "fusionSplitPriorCompactParentAnchorWindowRescueEnabled: " << fusionSplitPriorCompactParentAnchorWindowRescueEnabled << '\n';
        std::cout << "fusionSplitPriorCompactParentAnchorMinShape: " << fusionSplitPriorCompactParentAnchorMinShape << '\n';
        std::cout << "fusionSplitPriorCompactParentAnchorMaxScore: " << fusionSplitPriorCompactParentAnchorMaxScore << '\n';
        std::cout << "fusionSplitPriorCompactParentAnchorMinRealVoxels: " << fusionSplitPriorCompactParentAnchorMinRealVoxels << '\n';
        std::cout << "fusionSplitPriorCompactParentAnchorMinRealSignal: " << fusionSplitPriorCompactParentAnchorMinRealSignal << '\n';
        std::cout << "fusionSplitPriorCompactParentAnchorMinFarDistanceRadiusScale: " << fusionSplitPriorCompactParentAnchorMinFarDistanceRadiusScale << '\n';
        std::cout << "fusionSplitPriorParentAnchorSingleBlockerRescueEnabled: " << fusionSplitPriorParentAnchorSingleBlockerRescueEnabled << '\n';
        std::cout << "fusionSplitPriorParentAnchorSingleBlockerMinShape: " << fusionSplitPriorParentAnchorSingleBlockerMinShape << '\n';
        std::cout << "fusionSplitPriorParentAnchorSingleBlockerMaxScore: " << fusionSplitPriorParentAnchorSingleBlockerMaxScore << '\n';
        std::cout << "fusionSplitPriorParentAnchorSingleBlockerMinRealVoxels: " << fusionSplitPriorParentAnchorSingleBlockerMinRealVoxels << '\n';
        std::cout << "fusionSplitPriorParentAnchorSingleBlockerMinRealSignal: " << fusionSplitPriorParentAnchorSingleBlockerMinRealSignal << '\n';
        std::cout << "fusionSplitPriorParentAnchorSingleBlockerMinFarDistanceRadiusScale: " << fusionSplitPriorParentAnchorSingleBlockerMinFarDistanceRadiusScale << '\n';
        std::cout << "fusionSplitPriorParentAnchorSingleBlockerMaxContinuationPenalty: " << fusionSplitPriorParentAnchorSingleBlockerMaxContinuationPenalty << '\n';
        std::cout << "fusionSplitPriorParentAnchorSingleBlockerSelectionBonus: " << fusionSplitPriorParentAnchorSingleBlockerSelectionBonus << '\n';
        std::cout << "fusionSplitPriorLockAcceptedDaughtersForFinalPca: " << fusionSplitPriorLockAcceptedDaughtersForFinalPca << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealSingleBlockerRescueEnabled: " << fusionSplitPriorCleanTwoRealSingleBlockerRescueEnabled << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealSingleBlockerMinParentDistanceBalance: " << fusionSplitPriorCleanTwoRealSingleBlockerMinParentDistanceBalance << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealSingleBlockerMaxScore: " << fusionSplitPriorCleanTwoRealSingleBlockerMaxScore << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealSingleBlockerMinVoxels: " << fusionSplitPriorCleanTwoRealSingleBlockerMinVoxels << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealSingleBlockerMinSignal: " << fusionSplitPriorCleanTwoRealSingleBlockerMinSignal << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealSingleBlockerSelectionBonus: " << fusionSplitPriorCleanTwoRealSingleBlockerSelectionBonus << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealAsymmetricRescueEnabled: " << fusionSplitPriorCleanTwoRealAsymmetricRescueEnabled << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealAsymmetricMinParentDistanceBalance: " << fusionSplitPriorCleanTwoRealAsymmetricMinParentDistanceBalance << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealAsymmetricMaxScore: " << fusionSplitPriorCleanTwoRealAsymmetricMaxScore << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealAsymmetricMinVoxels: " << fusionSplitPriorCleanTwoRealAsymmetricMinVoxels << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealAsymmetricMinSignal: " << fusionSplitPriorCleanTwoRealAsymmetricMinSignal << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealAsymmetricMaxNeighborClaimPenalty: " << fusionSplitPriorCleanTwoRealAsymmetricMaxNeighborClaimPenalty << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealAsymmetricSelectionBonus: " << fusionSplitPriorCleanTwoRealAsymmetricSelectionBonus << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealCompactRescueEnabled: " << fusionSplitPriorCleanTwoRealCompactRescueEnabled << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealCompactMinParentDistanceBalance: " << fusionSplitPriorCleanTwoRealCompactMinParentDistanceBalance << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealCompactMaxScore: " << fusionSplitPriorCleanTwoRealCompactMaxScore << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealCompactMinVoxels: " << fusionSplitPriorCleanTwoRealCompactMinVoxels << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealCompactMinSignal: " << fusionSplitPriorCleanTwoRealCompactMinSignal << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealCompactMaxParentPersistencePenalty: " << fusionSplitPriorCleanTwoRealCompactMaxParentPersistencePenalty << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealCompactMinNearParentDistance: " << fusionSplitPriorCleanTwoRealCompactMinNearParentDistance << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealCompactMinSeparationRadiusScale: " << fusionSplitPriorCleanTwoRealCompactMinSeparationRadiusScale << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealCompactMaxMidpointDistance: " << fusionSplitPriorCleanTwoRealCompactMaxMidpointDistance << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealCompactSelectionBonus: " << fusionSplitPriorCleanTwoRealCompactSelectionBonus << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealRescueBeforeDefaultNegativeGate: " << fusionSplitPriorCleanTwoRealRescueBeforeDefaultNegativeGate << '\n';
        std::cout << "fusionSplitPriorParentAnchorOneRealSharedCandidatePenalty: " << fusionSplitPriorParentAnchorOneRealSharedCandidatePenalty << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealHighNeighborClaimRescueEnabled: " << fusionSplitPriorCleanTwoRealHighNeighborClaimRescueEnabled << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealHighNeighborClaimMinParentDistanceBalance: " << fusionSplitPriorCleanTwoRealHighNeighborClaimMinParentDistanceBalance << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealHighNeighborClaimMaxScore: " << fusionSplitPriorCleanTwoRealHighNeighborClaimMaxScore << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealHighNeighborClaimMinVoxels: " << fusionSplitPriorCleanTwoRealHighNeighborClaimMinVoxels << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealHighNeighborClaimMinSignal: " << fusionSplitPriorCleanTwoRealHighNeighborClaimMinSignal << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealHighNeighborClaimMinNeighborClaimPenalty: " << fusionSplitPriorCleanTwoRealHighNeighborClaimMinNeighborClaimPenalty << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealHighNeighborClaimMaxContinuationClaimPenalty: " << fusionSplitPriorCleanTwoRealHighNeighborClaimMaxContinuationClaimPenalty << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealHighNeighborClaimSelectionBonus: " << fusionSplitPriorCleanTwoRealHighNeighborClaimSelectionBonus << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealDuplicateBypassEnabled: " << fusionSplitPriorCleanTwoRealDuplicateBypassEnabled << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealDuplicateBypassMinImageGain: " << fusionSplitPriorCleanTwoRealDuplicateBypassMinImageGain << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealDuplicateBypassMaxBridgeValleyRatio: " << fusionSplitPriorCleanTwoRealDuplicateBypassMaxBridgeValleyRatio << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealDuplicateBypassMinBridgeGapWidth: " << fusionSplitPriorCleanTwoRealDuplicateBypassMinBridgeGapWidth << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealDuplicateBypassMinParentDistanceBalance: " << fusionSplitPriorCleanTwoRealDuplicateBypassMinParentDistanceBalance << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealDuplicateBypassMaxScore: " << fusionSplitPriorCleanTwoRealDuplicateBypassMaxScore << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealCompactDuplicateBypassEnabled: " << fusionSplitPriorCleanTwoRealCompactDuplicateBypassEnabled << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealCompactDuplicateBypassMinImageGain: " << fusionSplitPriorCleanTwoRealCompactDuplicateBypassMinImageGain << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxBridgeValleyRatio: " << fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxBridgeValleyRatio << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxOverlapCostFraction: " << fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxOverlapCostFraction << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxSoftPenaltyFraction: " << fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxSoftPenaltyFraction << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealCompactDuplicateBypassMinParentDistanceBalance: " << fusionSplitPriorCleanTwoRealCompactDuplicateBypassMinParentDistanceBalance << '\n';
        std::cout << "fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxScore: " << fusionSplitPriorCleanTwoRealCompactDuplicateBypassMaxScore << '\n';
        std::cout << "fusionSplitPriorRejectLowShapeZDominantPairs: " << fusionSplitPriorRejectLowShapeZDominantPairs << '\n';
        std::cout << "fusionSplitPriorLowShapeZDominantMaxParentShape: " << fusionSplitPriorLowShapeZDominantMaxParentShape << '\n';
        std::cout << "fusionSplitPriorRejectNonWindowLowShapeOverlapDuplicate: " << fusionSplitPriorRejectNonWindowLowShapeOverlapDuplicate << '\n';
        std::cout << "fusionSplitPriorNonWindowLowShapeMaxParentShape: " << fusionSplitPriorNonWindowLowShapeMaxParentShape << '\n';
        std::cout << "fusionSplitPriorNonWindowLowShapeMinOverlapCost: " << fusionSplitPriorNonWindowLowShapeMinOverlapCost << '\n';
        std::cout << "fusionSplitPriorNonWindowLowShapeMaxBridgeGapWidth: " << fusionSplitPriorNonWindowLowShapeMaxBridgeGapWidth << '\n';
        std::cout << "fusionSplitPriorReanchorNonParentDuplicateToOtherDaughter: " << fusionSplitPriorReanchorNonParentDuplicateToOtherDaughter << '\n';
        std::cout << "fusionSplitPriorNonParentDuplicateReanchorMinImageGain: " << fusionSplitPriorNonParentDuplicateReanchorMinImageGain << '\n';
        std::cout << "fusionSplitPriorNonParentDuplicateReanchorMaxBridgeValleyRatio: " << fusionSplitPriorNonParentDuplicateReanchorMaxBridgeValleyRatio << '\n';
        std::cout << "fusionSplitPriorNonParentDuplicateReanchorMinBridgeGapWidth: " << fusionSplitPriorNonParentDuplicateReanchorMinBridgeGapWidth << '\n';
        std::cout << "fusionSplitPriorNonParentDuplicateReanchorMinMove: " << fusionSplitPriorNonParentDuplicateReanchorMinMove << '\n';
        std::cout << "fusionSplitPriorNonParentDuplicateReanchorMinParentDistanceBalance: " << fusionSplitPriorNonParentDuplicateReanchorMinParentDistanceBalance << '\n';
        std::cout << "fusionSplitPriorMinDaughterParentDistance: " << fusionSplitPriorMinDaughterParentDistance << '\n';
        std::cout << "fusionSplitPriorMinParentDistanceBalance: " << fusionSplitPriorMinParentDistanceBalance << '\n';
        std::cout << "fusionSplitPriorParentPersistencePenalty: " << fusionSplitPriorParentPersistencePenalty << '\n';
        std::cout << "fusionSplitPriorNeighborClaimPenalty: " << fusionSplitPriorNeighborClaimPenalty << '\n';
        std::cout << "fusionSplitPriorMaxNeighborClaimPenalty: " << fusionSplitPriorMaxNeighborClaimPenalty << '\n';
        std::cout << "fusionSplitPriorUseDedicatedCostGate: " << fusionSplitPriorUseDedicatedCostGate << '\n';
        std::cout << "fusionSplitPriorUseImageCostGate: " << fusionSplitPriorUseImageCostGate << '\n';
        std::cout << "fusionSplitPriorMaxPositiveCostFraction: " << fusionSplitPriorMaxPositiveCostFraction << '\n';
        std::cout << "fusionSplitPriorPositiveGateMinImageGain: " << fusionSplitPriorPositiveGateMinImageGain << '\n';
        std::cout << "fusionSplitPriorPositiveGateMinImageGainPenaltyRatio: " << fusionSplitPriorPositiveGateMinImageGainPenaltyRatio << '\n';
        std::cout << "fusionSplitPriorPositiveGateElongatedParentMinShape: " << fusionSplitPriorPositiveGateElongatedParentMinShape << '\n';
        std::cout << "fusionSplitPriorPositiveGateElongatedMaxRawWorsening: " << fusionSplitPriorPositiveGateElongatedMaxRawWorsening << '\n';
        std::cout << "fusionSplitPriorPositiveGateElongatedMaxSoftPenaltyFraction: " << fusionSplitPriorPositiveGateElongatedMaxSoftPenaltyFraction << '\n';
        std::cout << "fusionSplitPriorPositiveGateElongatedMaxScore: " << fusionSplitPriorPositiveGateElongatedMaxScore << '\n';
        std::cout << "fusionSplitPriorMaxOverlapCostFraction: " << fusionSplitPriorMaxOverlapCostFraction << '\n';
        std::cout << "fusionSplitPriorHighConfidenceMaxScore: " << fusionSplitPriorHighConfidenceMaxScore << '\n';
        std::cout << "fusionSplitPriorHighConfidenceMaxOverlapCostFraction: " << fusionSplitPriorHighConfidenceMaxOverlapCostFraction << '\n';
        std::cout << "fusionSplitPriorHighConfidenceAxisAlignmentDegrees: " << fusionSplitPriorHighConfidenceAxisAlignmentDegrees << '\n';
        std::cout << "fusionSplitPriorDaughterVolumeScale: " << fusionSplitPriorDaughterVolumeScale << '\n';
        std::cout << "fusionSplitPriorPrefilterMaxValleyRatio: " << fusionSplitPriorPrefilterMaxValleyRatio << '\n';
        std::cout << "fusionSplitPriorBridgeMaxValleyRatio: " << fusionSplitPriorBridgeMaxValleyRatio << '\n';
        std::cout << "fusionSplitPriorMinBridgeGapWidth: " << fusionSplitPriorMinBridgeGapWidth << '\n';
        std::cout << "fusionSplitPriorMinEdgeBrightness: " << fusionSplitPriorMinEdgeBrightness << '\n';
        std::cout << "fusionSplitPriorMaxDaughterOverlapFraction: " << fusionSplitPriorMaxDaughterOverlapFraction << '\n';
        std::cout << "fusionSplitPriorSoftGateEnabled: " << fusionSplitPriorSoftGateEnabled << '\n';
        std::cout << "fusionSplitPriorSoftAxisPenaltyFraction: " << fusionSplitPriorSoftAxisPenaltyFraction << '\n';
        std::cout << "fusionSplitPriorSoftDaughterOverlapPenaltyFraction: " << fusionSplitPriorSoftDaughterOverlapPenaltyFraction << '\n';
        std::cout << "fusionSplitPriorSoftValleyPenaltyFraction: " << fusionSplitPriorSoftValleyPenaltyFraction << '\n';
        std::cout << "fusionSplitPriorSoftBridgeGapPenaltyFraction: " << fusionSplitPriorSoftBridgeGapPenaltyFraction << '\n';
        std::cout << "fusionSplitPriorSoftOverlapCostPenaltyWeight: " << fusionSplitPriorSoftOverlapCostPenaltyWeight << '\n';
        std::cout << "fusionSplitPriorBridgeEvidenceWaivesOverlapSoftPenalty: " << fusionSplitPriorBridgeEvidenceWaivesOverlapSoftPenalty << '\n';
        std::cout << "fusionSplitPriorHardMaxDaughterOverlapFraction: " << fusionSplitPriorHardMaxDaughterOverlapFraction << '\n';
        std::cout << "fusionSplitPriorHardMaxValleyRatio: " << fusionSplitPriorHardMaxValleyRatio << '\n';
        std::cout << "fusionSplitPriorHardMaxOverlapCostFraction: " << fusionSplitPriorHardMaxOverlapCostFraction << '\n';
        std::cout << "fusionSplitPriorZDominantTieBreakScoreMargin: " << fusionSplitPriorZDominantTieBreakScoreMargin << '\n';
        std::cout << "fusionSplitPriorZDominantTieBreakMinZDominance: " << fusionSplitPriorZDominantTieBreakMinZDominance << '\n';
        std::cout << "fusionSplitPriorZDominantTieBreakMaxLateralRadiusScale: " << fusionSplitPriorZDominantTieBreakMaxLateralRadiusScale << '\n';
        std::cout << "fusionSplitPriorDynamicOverlapEnabled: " << fusionSplitPriorDynamicOverlapEnabled << '\n';
        std::cout << "fusionSplitPriorLocalDensityRadiusScale: " << fusionSplitPriorLocalDensityRadiusScale << '\n';
        std::cout << "fusionSplitPriorLocalDensityOverlapBonus: " << fusionSplitPriorLocalDensityOverlapBonus << '\n';
        std::cout << "fusionSplitPriorMaxDynamicDaughterOverlapFraction: " << fusionSplitPriorMaxDynamicDaughterOverlapFraction << '\n';
        std::cout << "fusionDensityMetricsEnabled: " << fusionDensityMetricsEnabled << '\n';
        std::cout << "fusionDensityAdaptiveGateEnabled: " << fusionDensityAdaptiveGateEnabled << '\n';
        std::cout << "fusionBrightnessAdaptiveGateEnabled: " << fusionBrightnessAdaptiveGateEnabled << '\n';
        std::cout << "fusionDensityMetricsRadiusScale: " << fusionDensityMetricsRadiusScale << '\n';
        std::cout << "fusionDensityMetricsK: " << fusionDensityMetricsK << '\n';
        std::cout << "fusionBrightnessMetricsRadiusScale: " << fusionBrightnessMetricsRadiusScale << '\n';
        std::cout << "fusionBrightnessMetricsShellScale: " << fusionBrightnessMetricsShellScale << '\n';
        std::cout << "fusionSplitPriorSkipExistingCellBuriedCheck: " << fusionSplitPriorSkipExistingCellBuriedCheck << '\n';
        std::cout << "fusionSplitPriorSkipNeighborBridgeCheck: " << fusionSplitPriorSkipNeighborBridgeCheck << '\n';
        std::cout << "fusionSplitPriorPrepassFallbackEnabled: " << fusionSplitPriorPrepassFallbackEnabled << '\n';
        std::cout << "fusionSplitPriorPrepassFallbackMaxPriors: " << fusionSplitPriorPrepassFallbackMaxPriors << '\n';
        std::cout << "fusionSplitPriorPrepassFallbackMinKeptPixels: " << fusionSplitPriorPrepassFallbackMinKeptPixels << '\n';
        std::cout << "fusionSplitPriorPrepassFallbackMinShape: " << fusionSplitPriorPrepassFallbackMinShape << '\n';
        std::cout << "fusionSplitPriorPrepassFallbackMinSeparationRadiusScale: " << fusionSplitPriorPrepassFallbackMinSeparationRadiusScale << '\n';
        std::cout << "fusionSplitPriorPrepassFallbackMaxSeparationRadiusScale: " << fusionSplitPriorPrepassFallbackMaxSeparationRadiusScale << '\n';
        std::cout << "fusionSplitPriorPrepassFallbackParentClaimMargin: " << fusionSplitPriorPrepassFallbackParentClaimMargin << '\n';
        std::cout << "fusionSplitPriorPrepassFallbackMaxScore: " << fusionSplitPriorPrepassFallbackMaxScore << '\n';
        std::cout << "fusionSplitPriorPrepassFallbackRejectBadLumenParent: " << fusionSplitPriorPrepassFallbackRejectBadLumenParent << '\n';
        std::cout << "fusionSplitPriorPrepassFallbackBadLumenMaxScore: " << fusionSplitPriorPrepassFallbackBadLumenMaxScore << '\n';
        std::cout << "fusionSplitPriorPrepassFallbackBadLumenMaxNeighborClaimPenalty: " << fusionSplitPriorPrepassFallbackBadLumenMaxNeighborClaimPenalty << '\n';
        std::cout << "fusionSplitPriorPrepassFallbackMinParentAgeFrames: " << fusionSplitPriorPrepassFallbackMinParentAgeFrames << '\n';
        std::cout << "fusionSplitPriorPrepassFallbackUseSnapshotSeedOnLargeDrift: " << fusionSplitPriorPrepassFallbackUseSnapshotSeedOnLargeDrift << '\n';
        std::cout << "fusionSplitPriorPrepassFallbackSeedMaxShift: " << fusionSplitPriorPrepassFallbackSeedMaxShift << '\n';
        std::cout << "fusionSplitPriorPrepassFallbackSeedMinSeparationRadiusScale: " << fusionSplitPriorPrepassFallbackSeedMinSeparationRadiusScale << '\n';
        std::cout << "fusionSplitPriorPrepassFallbackSeedMaxSeparationRadiusScale: " << fusionSplitPriorPrepassFallbackSeedMaxSeparationRadiusScale << '\n';
        std::cout << "fusionSplitPriorPrepassFallbackSeedMaxScore: " << fusionSplitPriorPrepassFallbackSeedMaxScore << '\n';
        std::cout << "fusionSplitPriorPrepassFallbackSeedMinShape: " << fusionSplitPriorPrepassFallbackSeedMinShape << '\n';
        std::cout << "fusionSplitPriorSnapshotSeedMaxRefitDrift: " << fusionSplitPriorSnapshotSeedMaxRefitDrift << '\n';
        std::cout << "fusionReducePostSplitPerturbEnabled: " << fusionReducePostSplitPerturbEnabled << '\n';
        std::cout << "fusionPostSplitPerturbItersPerCell: " << fusionPostSplitPerturbItersPerCell << '\n';
        std::cout << "maxCandidateMeanVoxelCount: " << maxCandidateMeanVoxelCount << '\n';
        std::cout << "maxCandidateCountGrowthFromFirst: " << maxCandidateCountGrowthFromFirst << '\n';
    }
};

class BaseConfig {
public:
    std::string cellType;
    std::unique_ptr<EllipsoidConfig> cell;
    SimulationConfig simulation;
    ProbabilityConfig prob;
    CellLumenConfig cellLumen;
    struct RuntimeDensityProfile {
        std::string name;
        float minMedianNearestNeighborPx = -1000000000.0f;
        float maxMedianNearestNeighborPx = 1000000000.0f;
        YAML::Node expandedConfig;
    };
    bool runtimeDensityProfileSelectionEnabled = false;
    std::string runtimeDensityProfileMetric = "median_nearest_neighbor_px";
    std::string runtimeDensityDefaultProfile;
    std::string runtimeDensityActiveProfile;
    std::vector<RuntimeDensityProfile> runtimeDensityProfiles;

    BaseConfig() = default;
    ~BaseConfig() = default;

    // Deep copy (unique_ptr requires explicit copy)
    BaseConfig(const BaseConfig& other)
        : cellType(other.cellType),
          cell(other.cell ? std::make_unique<EllipsoidConfig>(*other.cell) : nullptr),
          simulation(other.simulation),
          prob(other.prob),
          cellLumen(other.cellLumen),
          runtimeDensityProfileSelectionEnabled(other.runtimeDensityProfileSelectionEnabled),
          runtimeDensityProfileMetric(other.runtimeDensityProfileMetric),
          runtimeDensityDefaultProfile(other.runtimeDensityDefaultProfile),
          runtimeDensityActiveProfile(other.runtimeDensityActiveProfile),
          runtimeDensityProfiles(other.runtimeDensityProfiles) {}

    BaseConfig& operator=(const BaseConfig& other) {
        if (this != &other) {
            cellType = other.cellType;
            cell = other.cell ? std::make_unique<EllipsoidConfig>(*other.cell) : nullptr;
            simulation = other.simulation;
            prob = other.prob;
            cellLumen = other.cellLumen;
            runtimeDensityProfileSelectionEnabled = other.runtimeDensityProfileSelectionEnabled;
            runtimeDensityProfileMetric = other.runtimeDensityProfileMetric;
            runtimeDensityDefaultProfile = other.runtimeDensityDefaultProfile;
            runtimeDensityActiveProfile = other.runtimeDensityActiveProfile;
            runtimeDensityProfiles = other.runtimeDensityProfiles;
        }
        return *this;
    }

    // Move operations auto-generated correctly by unique_ptr
    BaseConfig(BaseConfig&&) noexcept = default;
    BaseConfig& operator=(BaseConfig&&) noexcept = default;

    void explodeConfig(const YAML::Node& node) {
        cellType = node["cellType"].as<std::string>();
        cell = std::make_unique<EllipsoidConfig>();
        cell->explodeConfig(node["cell"]);
        simulation.explodeConfig(node["simulation"]);
        prob.explodeConfig(node["prob"]);
        if (node["cell_lumen"]) cellLumen.explodeConfig(node["cell_lumen"]);
        if (node["runtime_density_profiles"]) {
            const YAML::Node runtime = node["runtime_density_profiles"];
            if (runtime["enabled"]) {
                runtimeDensityProfileSelectionEnabled =
                    runtime["enabled"].as<bool>();
            }
            if (runtime["metric"]) {
                runtimeDensityProfileMetric = runtime["metric"].as<std::string>();
            }
            if (runtime["default_profile"]) {
                runtimeDensityDefaultProfile =
                    runtime["default_profile"].as<std::string>();
            }
            if (runtime["active_profile"]) {
                runtimeDensityActiveProfile =
                    runtime["active_profile"].as<std::string>();
            } else {
                runtimeDensityActiveProfile = runtimeDensityDefaultProfile;
            }
            runtimeDensityProfiles.clear();
            if (runtime["profiles"] && runtime["profiles"].IsSequence()) {
                for (const auto &entry : runtime["profiles"]) {
                    if (!entry["name"] || !entry["expanded_config"]) {
                        continue;
                    }
                    RuntimeDensityProfile profile;
                    profile.name = entry["name"].as<std::string>();
                    if (entry["min_median_nearest_neighbor_px"]) {
                        profile.minMedianNearestNeighborPx =
                            entry["min_median_nearest_neighbor_px"].as<float>();
                    }
                    if (entry["max_median_nearest_neighbor_px"]) {
                        profile.maxMedianNearestNeighborPx =
                            entry["max_median_nearest_neighbor_px"].as<float>();
                    }
                    profile.expandedConfig =
                        YAML::Clone(entry["expanded_config"]);
                    runtimeDensityProfiles.push_back(profile);
                }
            }
        }
    }

    void printConfig() const {
        simulation.printConfig();
        prob.printConfig();
        cellLumen.printConfig();
        std::cout << "runtimeDensityProfileSelectionEnabled: "
                  << runtimeDensityProfileSelectionEnabled << '\n';
        if (runtimeDensityProfileSelectionEnabled) {
            std::cout << "runtimeDensityProfileMetric: "
                      << runtimeDensityProfileMetric << '\n';
            std::cout << "runtimeDensityActiveProfile: "
                      << runtimeDensityActiveProfile << '\n';
            std::cout << "runtimeDensityProfileCount: "
                      << runtimeDensityProfiles.size() << '\n';
        }
    }
};

#endif
