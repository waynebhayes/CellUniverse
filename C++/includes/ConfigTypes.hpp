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
    int signal_guided_iterations_per_cell = -1;
    int random_iterations_per_cell = -1;
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
    float iterative_improvement_tolerance = 0.01f;
    float iterative_score_drop_stop_threshold = 0.1f;
    float iterative_penalty_score_drop_stop_threshold = 0.15f;
    float iterative_score_percentile = 0.05f;
    float iterative_score_percentile_max = 0.90f;
    float iterative_score_percentile_increment = 0.025f;
    int contrast_inner_window_size = 51;
    int contrast_outer_window_size = 101;
    float contrast_structure_threshold = 0.02f;
    float contrast_background_floor = 0.02f;
    float contrast_bright_fraction = 0.02f;
    float contrast_brightness_weight = 0.25f;
    float contrast_hollow_penalty_weight = 0.15f;
    float contrast_eps = 1e-6f;
    float global_intensity_scale_low_percentile = 0.01f;
    float global_intensity_scale_high_percentile = 0.995f;
    float global_intensity_hard_max = 0.0f;
    float post_process_blur_sigma = 2.5f;
    float post_process_final_blur_sigma = 0.0f;
    float post_process_final_direct_weight = 1.0f;
    float post_process_final_direct_amplification = 1.0f;
    float post_process_final_blurred_amplification = 1.0f;
    float post_process_amplification = 15.0f;
    float post_process_black_percentile = 0.005f;
    float post_process_white_percentile = 0.3f;
    bool brightness_alignment_enabled = true;
    bool export_preprocessed_images = false;
    bool quit_after_preprocessing = false;

    // Checkpoint resume (Approach 2): skip frames 0..resume_from-1 and
    // load state from {resume_source_dir}/checkpoints/frame_{resume_from-1:03d}.txt.
    // Set resume_from=0 or -1 to disable. resume_source_dir is the output
    // folder of the run being resumed FROM.
    int resume_from = 0;
    std::string resume_source_dir = "";
    bool export_post_localization_images = false;
    bool adaptive_cube_pooling_enabled = false;
    bool adaptive_cube_pooling_cost_comparison_enabled = true;
    float adaptive_cube_pooling_cube_size_scale = 0.25f;
    float adaptive_cube_pooling_zero_threshold = 0.02f;
    float adaptive_cube_pooling_majority_threshold = 0.7f;
    bool adaptive_cube_pooling_remove_isolated_bright_cubes = false;
    float adaptive_cube_pooling_isolated_bright_cube_threshold = 0.08f;
    float adaptive_background_expand_factor = 1.1f;
    float adaptive_background_top_fraction = 0.4f;
    bool signal_guided_position_enabled = false;
    float signal_guided_box_size_scale = 1.0f;
    float signal_guided_min_box_brightness_delta = 0.0f;
    float signal_guided_min_sigma_scale = 0.35f;
    float signal_guided_sigma_range_multiplier = 1.0f;

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
        if (node["iterative_improvement_tolerance"]) iterative_improvement_tolerance = node["iterative_improvement_tolerance"].as<float>();
        if (node["iterative_score_drop_stop_threshold"]) iterative_score_drop_stop_threshold = node["iterative_score_drop_stop_threshold"].as<float>();
        if (node["iterative_penalty_score_drop_stop_threshold"]) iterative_penalty_score_drop_stop_threshold = node["iterative_penalty_score_drop_stop_threshold"].as<float>();
        if (node["iterative_score_percentile"]) iterative_score_percentile = node["iterative_score_percentile"].as<float>();
        if (node["iterative_score_percentile_max"]) iterative_score_percentile_max = node["iterative_score_percentile_max"].as<float>();
        if (node["iterative_score_percentile_increment"]) iterative_score_percentile_increment = node["iterative_score_percentile_increment"].as<float>();
        if (node["contrast_inner_window_size"]) contrast_inner_window_size = node["contrast_inner_window_size"].as<int>();
        if (node["contrast_outer_window_size"]) contrast_outer_window_size = node["contrast_outer_window_size"].as<int>();
        if (node["contrast_structure_threshold"]) contrast_structure_threshold = node["contrast_structure_threshold"].as<float>();
        if (node["contrast_background_floor"]) contrast_background_floor = node["contrast_background_floor"].as<float>();
        if (node["contrast_bright_fraction"]) contrast_bright_fraction = node["contrast_bright_fraction"].as<float>();
        if (node["contrast_brightness_weight"]) contrast_brightness_weight = node["contrast_brightness_weight"].as<float>();
        if (node["contrast_hollow_penalty_weight"]) contrast_hollow_penalty_weight = node["contrast_hollow_penalty_weight"].as<float>();
        if (node["contrast_eps"]) contrast_eps = node["contrast_eps"].as<float>();
        if (node["global_intensity_scale_low_percentile"]) global_intensity_scale_low_percentile = node["global_intensity_scale_low_percentile"].as<float>();
        if (node["global_intensity_scale_high_percentile"]) global_intensity_scale_high_percentile = node["global_intensity_scale_high_percentile"].as<float>();
        if (node["global_intensity_hard_max"]) global_intensity_hard_max = node["global_intensity_hard_max"].as<float>();
        if (node["post_process_blur_sigma"]) post_process_blur_sigma = node["post_process_blur_sigma"].as<float>();
        if (node["post_process_final_blur_sigma"]) post_process_final_blur_sigma = node["post_process_final_blur_sigma"].as<float>();
        if (node["post_process_final_direct_weight"]) post_process_final_direct_weight = node["post_process_final_direct_weight"].as<float>();
        if (node["post_process_final_direct_amplification"]) post_process_final_direct_amplification = node["post_process_final_direct_amplification"].as<float>();
        if (node["post_process_final_blurred_amplification"]) post_process_final_blurred_amplification = node["post_process_final_blurred_amplification"].as<float>();
        if (node["post_process_amplification"]) post_process_amplification = node["post_process_amplification"].as<float>();
        if (node["post_process_black_percentile"]) post_process_black_percentile = node["post_process_black_percentile"].as<float>();
        if (node["post_process_white_percentile"]) post_process_white_percentile = node["post_process_white_percentile"].as<float>();
        if (node["brightness_alignment_enabled"]) brightness_alignment_enabled = node["brightness_alignment_enabled"].as<bool>();
        if (node["export_preprocessed_images"]) export_preprocessed_images = node["export_preprocessed_images"].as<bool>();
        if (node["quit_after_preprocessing"]) quit_after_preprocessing = node["quit_after_preprocessing"].as<bool>();
        if (node["resume_from"]) resume_from = node["resume_from"].as<int>();
        if (node["resume_source_dir"]) resume_source_dir = node["resume_source_dir"].as<std::string>();
        if (node["export_post_localization_images"]) export_post_localization_images = node["export_post_localization_images"].as<bool>();
        if (node["adaptive_cube_pooling_enabled"]) adaptive_cube_pooling_enabled = node["adaptive_cube_pooling_enabled"].as<bool>();
        if (node["adaptive_cube_pooling_cost_comparison_enabled"]) adaptive_cube_pooling_cost_comparison_enabled = node["adaptive_cube_pooling_cost_comparison_enabled"].as<bool>();
        if (node["adaptive_cube_pooling_cube_size_scale"]) adaptive_cube_pooling_cube_size_scale = node["adaptive_cube_pooling_cube_size_scale"].as<float>();
        if (node["adaptive_cube_pooling_zero_threshold"]) adaptive_cube_pooling_zero_threshold = node["adaptive_cube_pooling_zero_threshold"].as<float>();
        if (node["adaptive_cube_pooling_majority_threshold"]) adaptive_cube_pooling_majority_threshold = node["adaptive_cube_pooling_majority_threshold"].as<float>();
        if (node["adaptive_cube_pooling_remove_isolated_bright_cubes"]) adaptive_cube_pooling_remove_isolated_bright_cubes = node["adaptive_cube_pooling_remove_isolated_bright_cubes"].as<bool>();
        if (node["adaptive_cube_pooling_isolated_bright_cube_threshold"]) adaptive_cube_pooling_isolated_bright_cube_threshold = node["adaptive_cube_pooling_isolated_bright_cube_threshold"].as<float>();
        if (node["adaptive_background_expand_factor"]) adaptive_background_expand_factor = node["adaptive_background_expand_factor"].as<float>();
        if (node["adaptive_background_top_fraction"]) adaptive_background_top_fraction = node["adaptive_background_top_fraction"].as<float>();
        if (node["signal_guided_position_enabled"]) signal_guided_position_enabled = node["signal_guided_position_enabled"].as<bool>();
        if (node["signal_guided_box_size_scale"]) signal_guided_box_size_scale = node["signal_guided_box_size_scale"].as<float>();
        if (node["signal_guided_min_box_brightness_delta"]) signal_guided_min_box_brightness_delta = node["signal_guided_min_box_brightness_delta"].as<float>();
        if (node["signal_guided_min_sigma_scale"]) signal_guided_min_sigma_scale = node["signal_guided_min_sigma_scale"].as<float>();
        if (node["signal_guided_sigma_range_multiplier"]) signal_guided_sigma_range_multiplier = node["signal_guided_sigma_range_multiplier"].as<float>();
        if (node["asymmetric_cost_weight"]) asymmetric_cost_weight = node["asymmetric_cost_weight"].as<float>();
        if (node["asymmetric_cost_threshold"]) asymmetric_cost_threshold = node["asymmetric_cost_threshold"].as<float>();
        if (node["voronoi_cost_enabled"]) voronoi_cost_enabled = node["voronoi_cost_enabled"].as<bool>();
        if (node["voronoi_bleed_penalty_enabled"]) voronoi_bleed_penalty_enabled = node["voronoi_bleed_penalty_enabled"].as<bool>();
        if (node["voronoi_bleed_penalty_weight"]) voronoi_bleed_penalty_weight = node["voronoi_bleed_penalty_weight"].as<float>();
    }
    void printConfig() const {
        std::cout << "Simulation Config\n";
        std::cout << "iterations_per_cell: " << iterations_per_cell << '\n';
        std::cout << "signal_guided_iterations_per_cell: " << signal_guided_iterations_per_cell << '\n';
        std::cout << "random_iterations_per_cell: " << random_iterations_per_cell << '\n';
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
        std::cout << "iterative_improvement_tolerance: " << iterative_improvement_tolerance << '\n';
        std::cout << "iterative_score_drop_stop_threshold: " << iterative_score_drop_stop_threshold << '\n';
        std::cout << "iterative_penalty_score_drop_stop_threshold: " << iterative_penalty_score_drop_stop_threshold << '\n';
        std::cout << "iterative_score_percentile: " << iterative_score_percentile << '\n';
        std::cout << "iterative_score_percentile_max: " << iterative_score_percentile_max << '\n';
        std::cout << "iterative_score_percentile_increment: " << iterative_score_percentile_increment << '\n';
        std::cout << "contrast_inner_window_size: " << contrast_inner_window_size << '\n';
        std::cout << "contrast_outer_window_size: " << contrast_outer_window_size << '\n';
        std::cout << "contrast_structure_threshold: " << contrast_structure_threshold << '\n';
        std::cout << "contrast_background_floor: " << contrast_background_floor << '\n';
        std::cout << "contrast_bright_fraction: " << contrast_bright_fraction << '\n';
        std::cout << "contrast_brightness_weight: " << contrast_brightness_weight << '\n';
        std::cout << "contrast_hollow_penalty_weight: " << contrast_hollow_penalty_weight << '\n';
        std::cout << "contrast_eps: " << contrast_eps << '\n';
        std::cout << "global_intensity_scale_low_percentile: " << global_intensity_scale_low_percentile << '\n';
        std::cout << "global_intensity_scale_high_percentile: " << global_intensity_scale_high_percentile << '\n';
        std::cout << "global_intensity_hard_max: " << global_intensity_hard_max << '\n';
        std::cout << "post_process_blur_sigma: " << post_process_blur_sigma << '\n';
        std::cout << "post_process_final_blur_sigma: " << post_process_final_blur_sigma << '\n';
        std::cout << "post_process_final_direct_weight: " << post_process_final_direct_weight << '\n';
        std::cout << "post_process_final_direct_amplification: " << post_process_final_direct_amplification << '\n';
        std::cout << "post_process_final_blurred_amplification: " << post_process_final_blurred_amplification << '\n';
        std::cout << "post_process_amplification: " << post_process_amplification << '\n';
        std::cout << "post_process_black_percentile: " << post_process_black_percentile << '\n';
        std::cout << "post_process_white_percentile: " << post_process_white_percentile << '\n';
        std::cout << "brightness_alignment_enabled: " << brightness_alignment_enabled << '\n';
        std::cout << "export_preprocessed_images: " << export_preprocessed_images << '\n';
        std::cout << "quit_after_preprocessing: " << quit_after_preprocessing << '\n';
        std::cout << "export_post_localization_images: " << export_post_localization_images << '\n';
        std::cout << "adaptive_cube_pooling_enabled: " << adaptive_cube_pooling_enabled << '\n';
        std::cout << "adaptive_cube_pooling_cost_comparison_enabled: " << adaptive_cube_pooling_cost_comparison_enabled << '\n';
        std::cout << "adaptive_cube_pooling_cube_size_scale: " << adaptive_cube_pooling_cube_size_scale << '\n';
        std::cout << "adaptive_cube_pooling_zero_threshold: " << adaptive_cube_pooling_zero_threshold << '\n';
        std::cout << "adaptive_cube_pooling_majority_threshold: " << adaptive_cube_pooling_majority_threshold << '\n';
        std::cout << "adaptive_cube_pooling_remove_isolated_bright_cubes: " << adaptive_cube_pooling_remove_isolated_bright_cubes << '\n';
        std::cout << "adaptive_cube_pooling_isolated_bright_cube_threshold: " << adaptive_cube_pooling_isolated_bright_cube_threshold << '\n';
        std::cout << "adaptive_background_expand_factor: " << adaptive_background_expand_factor << '\n';
        std::cout << "adaptive_background_top_fraction: " << adaptive_background_top_fraction << '\n';
        std::cout << "signal_guided_position_enabled: " << signal_guided_position_enabled << '\n';
        std::cout << "signal_guided_box_size_scale: " << signal_guided_box_size_scale << '\n';
        std::cout << "signal_guided_min_box_brightness_delta: " << signal_guided_min_box_brightness_delta << '\n';
        std::cout << "signal_guided_min_sigma_scale: " << signal_guided_min_sigma_scale << '\n';
        std::cout << "signal_guided_sigma_range_multiplier: " << signal_guided_sigma_range_multiplier << '\n';
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

    float bio_daughter_size_ratio_max = 1.5f;
    float bio_combined_volume_min_fraction = 0.6f;
    float bio_combined_volume_max_fraction = 1.3f;

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

    ProbabilityConfig() = default;

    void explodeConfig(const YAML::Node& node) {
        if (node["P_split_base"]) P_split_base = node["P_split_base"].as<float>();
        if (node["P_split_max"]) P_split_max = node["P_split_max"].as<float>();
        if (node["overlap_penalty_weight"]) overlap_penalty_weight = node["overlap_penalty_weight"].as<float>();
        if (node["split_cost"]) split_cost = node["split_cost"].as<float>();
        if (node["split_cost_fraction"]) split_cost_fraction = node["split_cost_fraction"].as<float>();
        if (node["position_prior_weight"]) position_prior_weight = node["position_prior_weight"].as<float>();
        if (node["position_prior_threshold"]) position_prior_threshold = node["position_prior_threshold"].as<float>();
        if (node["max_perturb_drift_xy"]) max_perturb_drift_xy = node["max_perturb_drift_xy"].as<float>();
        if (node["max_perturb_drift_z"]) max_perturb_drift_z = node["max_perturb_drift_z"].as<float>();
        if (node["birth_growth_cap_factor"]) birth_growth_cap_factor = node["birth_growth_cap_factor"].as<float>();
        if (node["birth_growth_cap_elong_threshold"]) birth_growth_cap_elong_threshold = node["birth_growth_cap_elong_threshold"].as<float>();
        if (node["expected_daughter_pre_pass_iterations"]) expected_daughter_pre_pass_iterations = node["expected_daughter_pre_pass_iterations"].as<int>();
        if (node["split_candidates_per_attempt"]) split_candidates_per_attempt = node["split_candidates_per_attempt"].as<int>();
        if (node["split_candidate_burn_in_iterations"]) split_candidate_burn_in_iterations = node["split_candidate_burn_in_iterations"].as<int>();
        if (node["split_final_refine_iterations"]) split_final_refine_iterations = node["split_final_refine_iterations"].as<int>();
        if (node["split_calibration_iterations_per_cell"]) split_calibration_iterations_per_cell = node["split_calibration_iterations_per_cell"].as<int>();
        if (node["split_candidate_rotation_delta_degrees"]) split_candidate_rotation_delta_degrees = node["split_candidate_rotation_delta_degrees"].as<float>();
        if (node["split_candidate_translation_delta_fraction"]) split_candidate_translation_delta_fraction = node["split_candidate_translation_delta_fraction"].as<float>();
        if (node["bio_daughter_size_ratio_max"]) bio_daughter_size_ratio_max = node["bio_daughter_size_ratio_max"].as<float>();
        if (node["bio_combined_volume_min_fraction"]) bio_combined_volume_min_fraction = node["bio_combined_volume_min_fraction"].as<float>();
        if (node["bio_combined_volume_max_fraction"]) bio_combined_volume_max_fraction = node["bio_combined_volume_max_fraction"].as<float>();
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

        // Legacy YAML aliases — silently map to the new names.
        if (node["split"]) P_split_base = node["split"].as<float>();
        if (node["max_split_probability"]) P_split_max = node["max_split_probability"].as<float>();
    }

    void printConfig() const {
        std::cout << "Probability Config\n";
        std::cout << "P_split_base: " << P_split_base << '\n';
        std::cout << "P_split_max: " << P_split_max << '\n';
        std::cout << "split_cost: " << split_cost << '\n';
        std::cout << "overlap_penalty_weight: " << overlap_penalty_weight << '\n';
        std::cout << "split_candidates_per_attempt: " << split_candidates_per_attempt << '\n';
        std::cout << "split_candidate_burn_in_iterations: " << split_candidate_burn_in_iterations << '\n';
        std::cout << "bio_daughter_size_ratio_max: " << bio_daughter_size_ratio_max << std::endl;
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
        thread_local std::mt19937 gen{std::random_device{}()};
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
    // Reference radius for proportional perturbation sigma scaling.
    // positionScale = max(cell.a, cell.b, cell.c) / perturbSigmaReferenceRadius.
    // Cells larger than refR take bigger steps; smaller cells take smaller
    // steps. Set to 0 to disable (all cells use base sigma unchanged).
    float perturbSigmaReferenceRadius{0.0f};
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
        if (node["pcaShapeWeightExponent"]) pcaShapeWeightExponent = node["pcaShapeWeightExponent"].as<float>();
        if (node["pcaShapeAdaptiveExponent"]) pcaShapeAdaptiveExponent = node["pcaShapeAdaptiveExponent"].as<bool>();
        if (node["pcaShapeWeightExponentBright"]) pcaShapeWeightExponentBright = node["pcaShapeWeightExponentBright"].as<float>();
        if (node["pcaShapeCoreBrightnessThreshold"]) pcaShapeCoreBrightnessThreshold = node["pcaShapeCoreBrightnessThreshold"].as<float>();
        if (node["pcaShapeCoreFractionLow"]) pcaShapeCoreFractionLow = node["pcaShapeCoreFractionLow"].as<float>();
        if (node["pcaShapeCoreFractionHigh"]) pcaShapeCoreFractionHigh = node["pcaShapeCoreFractionHigh"].as<float>();
        if (node["pcaShapeRadiusInflationBright"]) pcaShapeRadiusInflationBright = node["pcaShapeRadiusInflationBright"].as<float>();
        if (node["pcaShapeRadiusPercentile"]) pcaShapeRadiusPercentile = node["pcaShapeRadiusPercentile"].as<float>();
        if (node["perturbSigmaReferenceRadius"]) perturbSigmaReferenceRadius = node["perturbSigmaReferenceRadius"].as<float>();
    }
};

class BaseConfig {
public:
    std::string cellType;
    std::unique_ptr<EllipsoidConfig> cell;
    SimulationConfig simulation;
    ProbabilityConfig prob;

    BaseConfig() = default;
    ~BaseConfig() = default;

    // Deep copy (unique_ptr requires explicit copy)
    BaseConfig(const BaseConfig& other)
        : cellType(other.cellType),
          cell(other.cell ? std::make_unique<EllipsoidConfig>(*other.cell) : nullptr),
          simulation(other.simulation),
          prob(other.prob) {}

    BaseConfig& operator=(const BaseConfig& other) {
        if (this != &other) {
            cellType = other.cellType;
            cell = other.cell ? std::make_unique<EllipsoidConfig>(*other.cell) : nullptr;
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
        cell = std::make_unique<EllipsoidConfig>();
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
