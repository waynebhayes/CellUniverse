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

class SimulationConfig {
public:
    int iterations_per_cell;
    float z_scaling;
    float blur_sigma;
    int z_slices;
    float iterative_penalty = 0.1f;
    float iterative_min_penalty = 0.005f;
    float iterative_collapse_backoff = 0.99f;
    float iterative_penalty_range = 0.9f;
    float iterative_reward_gate = 1.0f;
    float iterative_reward_gate_decrement = 0.008f;
    float iterative_reward_gate_min = 0.025f;
    float iterative_reward = 0.1f;
    float iterative_score_max = 1.5f;
    int iterative_max_count = 300;
    int iterative_no_improvement_patience = 10;
    float iterative_improvement_tolerance = 0.01f;
    float iterative_score_drop_stop_threshold = 0.1f;
    float iterative_score_percentile = 0.05f;
    float iterative_score_percentile_max = 0.90f;
    float iterative_score_percentile_increment = 0.025f;
    int contrast_inner_window_size = 51;
    int contrast_outer_window_size = 101;
    float contrast_structure_threshold = 0.02f;
    float contrast_eps = 1e-6f;
    float post_process_blur_sigma = 2.5f;
    float post_process_amplification = 15.0f;
    float post_process_black_percentile = 0.005f;
    float post_process_white_percentile = 0.3f;
    float michelson_low_percentile = 0.10f;
    float michelson_high_percentile = 0.90f;
    float michelson_eps = 1e-6f;
    float weber_background_percentile = 0.10f;
    float weber_signal_percentile = 0.90f;
    float weber_background_floor = 1.0f / 255.0f;
    float weber_eps = 1e-6f;
    bool export_preprocessed_images = false;
    bool quit_after_preprocessing = false;
    bool enable_lineage_tree_window = false;
    float adaptive_background_expand_factor = 1.1f;
    float adaptive_background_top_fraction = 0.4f;

    // Number of OpenCV workers used for independent z-slice work. The default
    // stays at one so laptops do not accidentally oversubscribe memory; OpenLab
    // jobs override it with CELLUNIVERSE_THREADS after Slurm grants the cores.
    int parallel_threads = 1;

    // Very small slice ranges are faster in a plain loop than through the
    // OpenCV scheduler. This threshold keeps partial redraws cheap when a
    // small cell touches only a few z slices.
    int parallel_min_slices = 8;

    // How initial CSV z coordinates should be interpreted.
    // auto: preserve the old heuristic, where theta columns mean a resume CSV
    // scaled: CSV z is already in interpolated optimizer space
    // raw: CSV z comes from the raw TIFF or GT stack and must be multiplied by z_scaling
    std::string initial_z_space = "auto";

    // Constructor with default values
    SimulationConfig() : iterations_per_cell(0),
                         z_scaling(1.0), blur_sigma(0.0f), z_slices(-1) {
    }
    void explodeConfig(const YAML::Node& node) {
        iterations_per_cell = node["iterations_per_cell"].as<int>();
        z_scaling = node["z_scaling"].as<float>();
        blur_sigma = node["blur_sigma"].as<float>();
        if (node["iterative_penalty"]) iterative_penalty = node["iterative_penalty"].as<float>();
        if (node["iterative_min_penalty"]) iterative_min_penalty = node["iterative_min_penalty"].as<float>();
        if (node["iterative_collapse_backoff"]) iterative_collapse_backoff = node["iterative_collapse_backoff"].as<float>();
        if (node["iterative_penalty_range"]) iterative_penalty_range = node["iterative_penalty_range"].as<float>();
        if (node["iterative_reward_gate"]) iterative_reward_gate = node["iterative_reward_gate"].as<float>();
        if (node["iterative_reward_gate_decrement"]) iterative_reward_gate_decrement = node["iterative_reward_gate_decrement"].as<float>();
        if (node["iterative_reward_gate_min"]) iterative_reward_gate_min = node["iterative_reward_gate_min"].as<float>();
        if (node["iterative_reward"]) iterative_reward = node["iterative_reward"].as<float>();
        if (node["iterative_score_max"]) iterative_score_max = node["iterative_score_max"].as<float>();
        if (node["iterative_max_count"]) iterative_max_count = node["iterative_max_count"].as<int>();
        if (node["iterative_no_improvement_patience"]) iterative_no_improvement_patience = node["iterative_no_improvement_patience"].as<int>();
        if (node["iterative_improvement_tolerance"]) iterative_improvement_tolerance = node["iterative_improvement_tolerance"].as<float>();
        if (node["iterative_score_drop_stop_threshold"]) iterative_score_drop_stop_threshold = node["iterative_score_drop_stop_threshold"].as<float>();
        if (node["iterative_score_percentile"]) iterative_score_percentile = node["iterative_score_percentile"].as<float>();
        if (node["iterative_score_percentile_max"]) iterative_score_percentile_max = node["iterative_score_percentile_max"].as<float>();
        if (node["iterative_score_percentile_increment"]) iterative_score_percentile_increment = node["iterative_score_percentile_increment"].as<float>();
        if (node["contrast_inner_window_size"]) contrast_inner_window_size = node["contrast_inner_window_size"].as<int>();
        if (node["contrast_outer_window_size"]) contrast_outer_window_size = node["contrast_outer_window_size"].as<int>();
        if (node["contrast_structure_threshold"]) contrast_structure_threshold = node["contrast_structure_threshold"].as<float>();
        if (node["contrast_eps"]) contrast_eps = node["contrast_eps"].as<float>();
        if (node["post_process_blur_sigma"]) post_process_blur_sigma = node["post_process_blur_sigma"].as<float>();
        if (node["post_process_amplification"]) post_process_amplification = node["post_process_amplification"].as<float>();
        if (node["post_process_black_percentile"]) post_process_black_percentile = node["post_process_black_percentile"].as<float>();
        if (node["post_process_white_percentile"]) post_process_white_percentile = node["post_process_white_percentile"].as<float>();
        if (node["michelson_low_percentile"]) michelson_low_percentile = node["michelson_low_percentile"].as<float>();
        if (node["michelson_high_percentile"]) michelson_high_percentile = node["michelson_high_percentile"].as<float>();
        if (node["michelson_eps"]) michelson_eps = node["michelson_eps"].as<float>();
        if (node["weber_background_percentile"]) weber_background_percentile = node["weber_background_percentile"].as<float>();
        if (node["weber_signal_percentile"]) weber_signal_percentile = node["weber_signal_percentile"].as<float>();
        if (node["weber_background_floor"]) weber_background_floor = node["weber_background_floor"].as<float>();
        if (node["weber_eps"]) weber_eps = node["weber_eps"].as<float>();
        if (node["export_preprocessed_images"]) export_preprocessed_images = node["export_preprocessed_images"].as<bool>();
        if (node["quit_after_preprocessing"]) quit_after_preprocessing = node["quit_after_preprocessing"].as<bool>();
        if (node["enable_lineage_tree_window"]) enable_lineage_tree_window = node["enable_lineage_tree_window"].as<bool>();
        if (node["lineage_tree_window"]) enable_lineage_tree_window = node["lineage_tree_window"].as<bool>();
        if (node["adaptive_background_expand_factor"]) adaptive_background_expand_factor = node["adaptive_background_expand_factor"].as<float>();
        if (node["adaptive_background_top_fraction"]) adaptive_background_top_fraction = node["adaptive_background_top_fraction"].as<float>();
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
    }
    void printConfig() const {
        std::cout << "Simulation Config\n";
        std::cout << "iterations_per_cell: " << iterations_per_cell << '\n';
        std::cout << "z_scaling: " << z_scaling << '\n';
        std::cout << "blur_sigma: " << blur_sigma << '\n';
        std::cout << "iterative_penalty: " << iterative_penalty << '\n';
        std::cout << "iterative_min_penalty: " << iterative_min_penalty << '\n';
        std::cout << "iterative_collapse_backoff: " << iterative_collapse_backoff << '\n';
        std::cout << "iterative_penalty_range: " << iterative_penalty_range << '\n';
        std::cout << "iterative_reward_gate: " << iterative_reward_gate << '\n';
        std::cout << "iterative_reward_gate_decrement: " << iterative_reward_gate_decrement << '\n';
        std::cout << "iterative_reward_gate_min: " << iterative_reward_gate_min << '\n';
        std::cout << "iterative_reward: " << iterative_reward << '\n';
        std::cout << "iterative_score_max: " << iterative_score_max << '\n';
        std::cout << "iterative_max_count: " << iterative_max_count << '\n';
        std::cout << "iterative_no_improvement_patience: " << iterative_no_improvement_patience << '\n';
        std::cout << "iterative_improvement_tolerance: " << iterative_improvement_tolerance << '\n';
        std::cout << "iterative_score_drop_stop_threshold: " << iterative_score_drop_stop_threshold << '\n';
        std::cout << "iterative_score_percentile: " << iterative_score_percentile << '\n';
        std::cout << "iterative_score_percentile_max: " << iterative_score_percentile_max << '\n';
        std::cout << "iterative_score_percentile_increment: " << iterative_score_percentile_increment << '\n';
        std::cout << "contrast_inner_window_size: " << contrast_inner_window_size << '\n';
        std::cout << "contrast_outer_window_size: " << contrast_outer_window_size << '\n';
        std::cout << "contrast_structure_threshold: " << contrast_structure_threshold << '\n';
        std::cout << "contrast_eps: " << contrast_eps << '\n';
        std::cout << "post_process_blur_sigma: " << post_process_blur_sigma << '\n';
        std::cout << "post_process_amplification: " << post_process_amplification << '\n';
        std::cout << "post_process_black_percentile: " << post_process_black_percentile << '\n';
        std::cout << "post_process_white_percentile: " << post_process_white_percentile << '\n';
        std::cout << "michelson_low_percentile: " << michelson_low_percentile << '\n';
        std::cout << "michelson_high_percentile: " << michelson_high_percentile << '\n';
        std::cout << "weber_background_percentile: " << weber_background_percentile << '\n';
        std::cout << "weber_signal_percentile: " << weber_signal_percentile << '\n';
        std::cout << "weber_background_floor: " << weber_background_floor << '\n';
        std::cout << "export_preprocessed_images: " << export_preprocessed_images << '\n';
        std::cout << "quit_after_preprocessing: " << quit_after_preprocessing << '\n';
        std::cout << "enable_lineage_tree_window: " << enable_lineage_tree_window << '\n';
        std::cout << "adaptive_background_expand_factor: " << adaptive_background_expand_factor << '\n';
        std::cout << "adaptive_background_top_fraction: " << adaptive_background_top_fraction << '\n';
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
    float shape_elongation_classify_threshold = 1.20f;

    float overlap_penalty_weight = 500.0f;
    float size_reduction_penalty_weight = 30.0f;
    float split_cost = 80.0f;
    bool split_cost_rescue_enabled = true;
    float split_cost_rescue_min_fraction = 0.60f;
    float split_cost_rescue_max_gap_density = 0.05f;
    float split_cost_rescue_max_valley_ratio = 0.35f;
    float split_cost_rescue_max_drift_fraction = 0.25f;
    float split_cost_rescue_max_overlap_penalty = 2.0f;
    bool split_cost_perfect_bridge_rescue_enabled = true;
    float split_cost_perfect_bridge_rescue_min_fraction = 0.25f;
    float split_cost_perfect_bridge_rescue_max_gap_density = 0.005f;
    float split_cost_perfect_bridge_rescue_max_valley_ratio = 0.15f;
    float split_cost_perfect_bridge_rescue_max_overlap_penalty = 0.50f;

    float split_direction_agreement_degrees = 20.0f;
    int expected_daughter_pre_pass_iterations = 1;

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
    bool split_far_candidate_enabled = false;
    float split_far_candidate_min_distance_fraction = 0.70f;
    float split_far_candidate_quantile = 0.70f;
    float split_far_candidate_min_weight_fraction = 0.01f;
    int split_reject_retry_limit = 2;
    int split_reject_retry_cooldown_iterations = 500;
    bool split_snapshot_xy_candidate_enabled = true;
    float split_snapshot_xy_separation_scale = 1.8f;
    float split_snapshot_xy_min_parent_radius_fraction = 1.5f;

    float bio_daughter_size_ratio_max = 1.5f;
    float bio_combined_volume_min_fraction = 0.6f;
    float bio_combined_volume_max_fraction = 1.3f;

    // Post-burn-in drift gate. The best candidate's final daughter centers
    // must not have drifted more than `max(bio_max_drift_parent_fraction *
    // parent_maxR, bio_max_drift_daughter_fraction * daughter_maxR)` from
    // their initial seed positions. Catches daughters escaping the parent
    // footprint during burn-in (the f4/f5 false-split pathology where one
    // daughter wanders to absorb a neighbor cell).
    float bio_max_drift_parent_fraction = 0.4f;
    float bio_max_drift_daughter_fraction = 1.0f;
    float bio_min_daughter_separation_parent_fraction = 1.70f;

    // Midpoint-near-parent gate. Reject when the daughter midpoint lies
    // further than `bio_max_midpoint_parent_fraction * srcMaxR` from the
    // snapshot parent center. Uses SNAPSHOT (last-frame) position, not
    // live — live drifts during Phase A and can park on one daughter
    // rather than between them. A real division happens at the cell's
    // previous-frame location; if PCA finds a bright region 1+ parent-
    // diameters away, it's parent drift (e3d03 run 072416 f5: midpoint
    // 1.36 x parent maxR) or inflation (1f2ed f4: 1.04), not a real split.
    float bio_max_midpoint_parent_fraction = 0.95f;

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

    // Multiplier applied to the Spheroid x/y/z perturbation sigmas during
    // candidate burn-in. The main-loop sigmas (x=5, y=5, z=8) let daughters
    // wander 15-25 voxels across a 20-iter burn-in, enough to leave the
    // parent footprint. 0.4 scales them to (2, 2, 3.2), limiting drift to
    // refinement distances (<10 voxels) while still letting daughters
    // settle to the local image optimum.
    float split_burn_in_pos_sigma_scale = 0.4f;

    // Multiplier applied to the Spheroid majorRadius/bRadius/minorRadius
    // perturbation sigmas during candidate burn-in. At 0.1, radii can
    // only drift ~10% of their configured sigma per iteration, preserving
    // the snapshot-based daughter sizing (`0.794 * src`) through burn-in
    // instead of letting burn-in collapse one daughter to the radius
    // floor. Fixes the "real daughter + collapsed sliver" false-split
    // pattern where one daughter shrinks to (~r_min) at a distant image
    // spot while the other stays near the parent.
    float split_burn_in_radius_sigma_scale = 0.1f;

    ProbabilityConfig() = default;

    void explodeConfig(const YAML::Node& node) {
        if (node["P_split_base"]) P_split_base = node["P_split_base"].as<float>();
        if (node["P_split_max"]) P_split_max = node["P_split_max"].as<float>();
        if (node["shape_elongation_classify_threshold"]) shape_elongation_classify_threshold = node["shape_elongation_classify_threshold"].as<float>();
        if (node["overlap_penalty_weight"]) overlap_penalty_weight = node["overlap_penalty_weight"].as<float>();
        if (node["size_reduction_penalty_weight"]) size_reduction_penalty_weight = node["size_reduction_penalty_weight"].as<float>();
        if (node["split_cost"]) split_cost = node["split_cost"].as<float>();
        if (node["split_cost_rescue_enabled"]) split_cost_rescue_enabled = node["split_cost_rescue_enabled"].as<bool>();
        if (node["split_cost_rescue_min_fraction"]) split_cost_rescue_min_fraction = node["split_cost_rescue_min_fraction"].as<float>();
        if (node["split_cost_rescue_max_gap_density"]) split_cost_rescue_max_gap_density = node["split_cost_rescue_max_gap_density"].as<float>();
        if (node["split_cost_rescue_max_valley_ratio"]) split_cost_rescue_max_valley_ratio = node["split_cost_rescue_max_valley_ratio"].as<float>();
        if (node["split_cost_rescue_max_drift_fraction"]) split_cost_rescue_max_drift_fraction = node["split_cost_rescue_max_drift_fraction"].as<float>();
        if (node["split_cost_rescue_max_overlap_penalty"]) split_cost_rescue_max_overlap_penalty = node["split_cost_rescue_max_overlap_penalty"].as<float>();
        if (node["split_cost_perfect_bridge_rescue_enabled"]) split_cost_perfect_bridge_rescue_enabled = node["split_cost_perfect_bridge_rescue_enabled"].as<bool>();
        if (node["split_cost_perfect_bridge_rescue_min_fraction"]) split_cost_perfect_bridge_rescue_min_fraction = node["split_cost_perfect_bridge_rescue_min_fraction"].as<float>();
        if (node["split_cost_perfect_bridge_rescue_max_gap_density"]) split_cost_perfect_bridge_rescue_max_gap_density = node["split_cost_perfect_bridge_rescue_max_gap_density"].as<float>();
        if (node["split_cost_perfect_bridge_rescue_max_valley_ratio"]) split_cost_perfect_bridge_rescue_max_valley_ratio = node["split_cost_perfect_bridge_rescue_max_valley_ratio"].as<float>();
        if (node["split_cost_perfect_bridge_rescue_max_overlap_penalty"]) split_cost_perfect_bridge_rescue_max_overlap_penalty = node["split_cost_perfect_bridge_rescue_max_overlap_penalty"].as<float>();
        if (node["split_direction_agreement_degrees"]) split_direction_agreement_degrees = node["split_direction_agreement_degrees"].as<float>();
        if (node["expected_daughter_pre_pass_iterations"]) expected_daughter_pre_pass_iterations = node["expected_daughter_pre_pass_iterations"].as<int>();
        if (node["split_candidates_per_attempt"]) split_candidates_per_attempt = node["split_candidates_per_attempt"].as<int>();
        if (node["split_candidate_burn_in_iterations"]) split_candidate_burn_in_iterations = node["split_candidate_burn_in_iterations"].as<int>();
        if (node["split_final_refine_iterations"]) split_final_refine_iterations = node["split_final_refine_iterations"].as<int>();
        if (node["split_calibration_iterations_per_cell"]) split_calibration_iterations_per_cell = node["split_calibration_iterations_per_cell"].as<int>();
        if (node["split_candidate_rotation_delta_degrees"]) split_candidate_rotation_delta_degrees = node["split_candidate_rotation_delta_degrees"].as<float>();
        if (node["split_candidate_translation_delta_fraction"]) split_candidate_translation_delta_fraction = node["split_candidate_translation_delta_fraction"].as<float>();
        if (node["split_far_candidate_enabled"]) split_far_candidate_enabled = node["split_far_candidate_enabled"].as<bool>();
        if (node["split_far_candidate_min_distance_fraction"]) split_far_candidate_min_distance_fraction = node["split_far_candidate_min_distance_fraction"].as<float>();
        if (node["split_far_candidate_quantile"]) split_far_candidate_quantile = node["split_far_candidate_quantile"].as<float>();
        if (node["split_far_candidate_min_weight_fraction"]) split_far_candidate_min_weight_fraction = node["split_far_candidate_min_weight_fraction"].as<float>();
        if (node["split_reject_retry_limit"]) split_reject_retry_limit = node["split_reject_retry_limit"].as<int>();
        if (node["split_reject_retry_cooldown_iterations"]) split_reject_retry_cooldown_iterations = node["split_reject_retry_cooldown_iterations"].as<int>();
        if (node["split_snapshot_xy_candidate_enabled"]) split_snapshot_xy_candidate_enabled = node["split_snapshot_xy_candidate_enabled"].as<bool>();
        if (node["split_snapshot_xy_separation_scale"]) split_snapshot_xy_separation_scale = node["split_snapshot_xy_separation_scale"].as<float>();
        if (node["split_snapshot_xy_min_parent_radius_fraction"]) split_snapshot_xy_min_parent_radius_fraction = node["split_snapshot_xy_min_parent_radius_fraction"].as<float>();
        if (node["bio_daughter_size_ratio_max"]) bio_daughter_size_ratio_max = node["bio_daughter_size_ratio_max"].as<float>();
        if (node["bio_combined_volume_min_fraction"]) bio_combined_volume_min_fraction = node["bio_combined_volume_min_fraction"].as<float>();
        if (node["bio_combined_volume_max_fraction"]) bio_combined_volume_max_fraction = node["bio_combined_volume_max_fraction"].as<float>();
        if (node["bio_max_drift_parent_fraction"]) bio_max_drift_parent_fraction = node["bio_max_drift_parent_fraction"].as<float>();
        if (node["bio_max_drift_daughter_fraction"]) bio_max_drift_daughter_fraction = node["bio_max_drift_daughter_fraction"].as<float>();
        if (node["bio_min_daughter_separation_parent_fraction"]) bio_min_daughter_separation_parent_fraction = node["bio_min_daughter_separation_parent_fraction"].as<float>();
        if (node["bio_max_midpoint_parent_fraction"]) bio_max_midpoint_parent_fraction = node["bio_max_midpoint_parent_fraction"].as<float>();
        if (node["bio_max_single_daughter_volume_fraction"]) bio_max_single_daughter_volume_fraction = node["bio_max_single_daughter_volume_fraction"].as<float>();
        if (node["bio_bridge_max_gap_density"]) bio_bridge_max_gap_density = node["bio_bridge_max_gap_density"].as<float>();
        if (node["bio_bridge_max_valley_ratio"]) bio_bridge_max_valley_ratio = node["bio_bridge_max_valley_ratio"].as<float>();
        if (node["split_burn_in_pos_sigma_scale"]) split_burn_in_pos_sigma_scale = node["split_burn_in_pos_sigma_scale"].as<float>();
        if (node["split_burn_in_radius_sigma_scale"]) split_burn_in_radius_sigma_scale = node["split_burn_in_radius_sigma_scale"].as<float>();

        // Legacy YAML aliases — silently map to the new names.
        if (node["split"]) P_split_base = node["split"].as<float>();
        if (node["max_split_probability"]) P_split_max = node["max_split_probability"].as<float>();
    }

    void printConfig() const {
        std::cout << "Probability Config\n";
        std::cout << "P_split_base: " << P_split_base << '\n';
        std::cout << "P_split_max: " << P_split_max << '\n';
        std::cout << "shape_elongation_classify_threshold: " << shape_elongation_classify_threshold << '\n';
        std::cout << "split_cost: " << split_cost << '\n';
        std::cout << "split_cost_rescue_enabled: " << split_cost_rescue_enabled << '\n';
        std::cout << "split_cost_rescue_min_fraction: " << split_cost_rescue_min_fraction << '\n';
        std::cout << "split_cost_rescue_max_gap_density: " << split_cost_rescue_max_gap_density << '\n';
        std::cout << "split_cost_rescue_max_valley_ratio: " << split_cost_rescue_max_valley_ratio << '\n';
        std::cout << "split_cost_rescue_max_drift_fraction: " << split_cost_rescue_max_drift_fraction << '\n';
        std::cout << "split_cost_rescue_max_overlap_penalty: " << split_cost_rescue_max_overlap_penalty << '\n';
        std::cout << "split_cost_perfect_bridge_rescue_enabled: " << split_cost_perfect_bridge_rescue_enabled << '\n';
        std::cout << "split_cost_perfect_bridge_rescue_min_fraction: " << split_cost_perfect_bridge_rescue_min_fraction << '\n';
        std::cout << "split_cost_perfect_bridge_rescue_max_gap_density: " << split_cost_perfect_bridge_rescue_max_gap_density << '\n';
        std::cout << "split_cost_perfect_bridge_rescue_max_valley_ratio: " << split_cost_perfect_bridge_rescue_max_valley_ratio << '\n';
        std::cout << "split_cost_perfect_bridge_rescue_max_overlap_penalty: " << split_cost_perfect_bridge_rescue_max_overlap_penalty << '\n';
        std::cout << "overlap_penalty_weight: " << overlap_penalty_weight << '\n';
        std::cout << "size_reduction_penalty_weight: " << size_reduction_penalty_weight << '\n';
        std::cout << "split_candidates_per_attempt: " << split_candidates_per_attempt << '\n';
        std::cout << "split_candidate_burn_in_iterations: " << split_candidate_burn_in_iterations << '\n';
        std::cout << "split_final_refine_iterations: " << split_final_refine_iterations << '\n';
        std::cout << "split_calibration_iterations_per_cell: " << split_calibration_iterations_per_cell << '\n';
        std::cout << "split_candidate_rotation_delta_degrees: " << split_candidate_rotation_delta_degrees << '\n';
        std::cout << "split_candidate_translation_delta_fraction: " << split_candidate_translation_delta_fraction << '\n';
        std::cout << "split_far_candidate_enabled: " << split_far_candidate_enabled << '\n';
        std::cout << "split_far_candidate_min_distance_fraction: " << split_far_candidate_min_distance_fraction << '\n';
        std::cout << "split_far_candidate_quantile: " << split_far_candidate_quantile << '\n';
        std::cout << "split_far_candidate_min_weight_fraction: " << split_far_candidate_min_weight_fraction << '\n';
        std::cout << "split_reject_retry_limit: " << split_reject_retry_limit << '\n';
        std::cout << "split_reject_retry_cooldown_iterations: " << split_reject_retry_cooldown_iterations << '\n';
        std::cout << "split_snapshot_xy_candidate_enabled: " << split_snapshot_xy_candidate_enabled << '\n';
        std::cout << "split_snapshot_xy_separation_scale: " << split_snapshot_xy_separation_scale << '\n';
        std::cout << "split_snapshot_xy_min_parent_radius_fraction: " << split_snapshot_xy_min_parent_radius_fraction << '\n';
        std::cout << "bio_daughter_size_ratio_max: " << bio_daughter_size_ratio_max << '\n';
        std::cout << "bio_combined_volume_min_fraction: " << bio_combined_volume_min_fraction << '\n';
        std::cout << "bio_combined_volume_max_fraction: " << bio_combined_volume_max_fraction << '\n';
        std::cout << "bio_max_drift_parent_fraction: " << bio_max_drift_parent_fraction << '\n';
        std::cout << "bio_max_drift_daughter_fraction: " << bio_max_drift_daughter_fraction << '\n';
        std::cout << "bio_min_daughter_separation_parent_fraction: " << bio_min_daughter_separation_parent_fraction << '\n';
        std::cout << "bio_max_midpoint_parent_fraction: " << bio_max_midpoint_parent_fraction << '\n';
        std::cout << "bio_max_single_daughter_volume_fraction: " << bio_max_single_daughter_volume_fraction << '\n';
        std::cout << "bio_bridge_max_gap_density: " << bio_bridge_max_gap_density << '\n';
        std::cout << "bio_bridge_max_valley_ratio: " << bio_bridge_max_valley_ratio << std::endl;
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

class SpheroidConfig {
public:
    PerturbParams x{};
    PerturbParams y{};
    PerturbParams z{};
    PerturbParams majorRadius{};
    PerturbParams bRadius{};
    PerturbParams minorRadius{};
    PerturbParams abRatio{};
    PerturbParams thetaX{};
    PerturbParams thetaY{};
    PerturbParams thetaZ{};
    PerturbParams brightness{};
    double minMajorRadius{};
    double maxMajorRadius{};
    double minBRadius{};
    double maxBRadius{};
    double minMinorRadius{};
    double maxMinorRadius{};
    float initialBrightness{0.2f};
    float initialRadiusScale{1.0f};
    float backgroundColor{0.0f};
    double minBrightness{0.1};
    double maxBrightness{1.0};
    float brightnessProbabilityStep{0.02f};
    float brightnessProbabilityTrust{1.0f};
    float majorRadiusProbabilityStep{0.02f};
    float majorRadiusProbabilityTrust{1.0f};
    float minorRadiusProbabilityStep{0.02f};
    float minorRadiusProbabilityTrust{1.0f};
    float abRatioProbabilityStep{0.02f};
    float abRatioProbabilityTrust{1.0f};
    float brightnessUpdateBlend{0.2f};
    float brightnessMeanAmplification{1.0f};
    // Maximum valid z position in interpolated optimizer space. It starts open
    // because initial cells are constructed before the first frame has loaded,
    // so using an embryo-sized default would clip deeper datasets such as
    // Drosophila. CellUniverse::ensureFrameLoaded updates this to the actual
    // slice count before optimization and later perturbations.
    float maxZ{std::numeric_limits<float>::max()};
    ~SpheroidConfig() = default;

    void explodeConfig(const YAML::Node& node)
    {
        x.explodeParams(node["x"]);
        y.explodeParams(node["y"]);
        z.explodeParams(node["z"]);
        majorRadius.explodeParams(node["majorRadius"]);
        if (node["bRadius"]) bRadius.explodeParams(node["bRadius"]);
        minorRadius.explodeParams(node["minorRadius"]);
        if (node["abRatio"]) abRatio.explodeParams(node["abRatio"]);
        thetaX.explodeParams(node["thetaX"]);
        thetaY.explodeParams(node["thetaY"]);
        thetaZ.explodeParams(node["thetaZ"]);
        if (node["brightness"]) brightness.explodeParams(node["brightness"]);

        minMajorRadius = node["minMajorRadius"].as<double>();
        maxMajorRadius = node["maxMajorRadius"].as<double>();
        if (node["minBRadius"]) minBRadius = node["minBRadius"].as<double>();
        if (node["maxBRadius"]) maxBRadius = node["maxBRadius"].as<double>();
        minMinorRadius = node["minMinorRadius"].as<double>();
        maxMinorRadius = node["maxMinorRadius"].as<double>();
        if (node["initialBrightness"]) initialBrightness = node["initialBrightness"].as<float>();
        if (node["initialRadiusScale"]) initialRadiusScale = node["initialRadiusScale"].as<float>();
        if (node["backgroundColor"]) backgroundColor = node["backgroundColor"].as<float>();
        if (node["minBrightness"]) minBrightness = node["minBrightness"].as<double>();
        if (node["maxBrightness"]) maxBrightness = node["maxBrightness"].as<double>();
        if (node["brightnessProbabilityStep"]) brightnessProbabilityStep = node["brightnessProbabilityStep"].as<float>();
        if (node["brightnessProbabilityTrust"]) brightnessProbabilityTrust = node["brightnessProbabilityTrust"].as<float>();
        majorRadiusProbabilityStep = brightnessProbabilityStep;
        majorRadiusProbabilityTrust = brightnessProbabilityTrust;
        minorRadiusProbabilityStep = brightnessProbabilityStep;
        minorRadiusProbabilityTrust = brightnessProbabilityTrust;
        abRatioProbabilityStep = brightnessProbabilityStep;
        abRatioProbabilityTrust = brightnessProbabilityTrust;
        if (node["majorRadiusProbabilityStep"]) majorRadiusProbabilityStep = node["majorRadiusProbabilityStep"].as<float>();
        if (node["majorRadiusProbabilityTrust"]) majorRadiusProbabilityTrust = node["majorRadiusProbabilityTrust"].as<float>();
        if (node["minorRadiusProbabilityStep"]) minorRadiusProbabilityStep = node["minorRadiusProbabilityStep"].as<float>();
        if (node["minorRadiusProbabilityTrust"]) minorRadiusProbabilityTrust = node["minorRadiusProbabilityTrust"].as<float>();
        if (node["abRatioProbabilityStep"]) abRatioProbabilityStep = node["abRatioProbabilityStep"].as<float>();
        if (node["abRatioProbabilityTrust"]) abRatioProbabilityTrust = node["abRatioProbabilityTrust"].as<float>();
        if (node["brightnessUpdateBlend"]) brightnessUpdateBlend = node["brightnessUpdateBlend"].as<float>();
        if (node["brightnessMeanAmplification"]) brightnessMeanAmplification = node["brightnessMeanAmplification"].as<float>();
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
