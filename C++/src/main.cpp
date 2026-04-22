#include <iostream>
#include <cstdio>
#include "types.hpp"
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "ConfigTypes.hpp"
#include "CellFactory.hpp"
#include "yaml-cpp/yaml.h"
#include "Ellipsoid.hpp"
#include "CellUniverse.hpp"
#include "ImageHandler.hpp"
#include <chrono>
#include <algorithm>


class Args
{
public:
    std::string config{};
    std::string input{};
    int firstFrame = 0;
    int lastFrame = 0;
    std::string initial{};
    std::string output{};
    int continueFrom = -1;
    // Optional checkpoint-resume args (2026-04-22). When provided via the
    // run_celluniverse.sh launcher (sourced from the INI preset), these
    // override config.simulation.resume_from / resume_source_dir. Absent
    // (argc < 9) → resume defaults to disabled (0 / "").
    int resumeFromFrame = 0;
    std::string resumeSourceDir{};
};

// helper function to load the config
void loadConfig(const std::string &path, BaseConfig &config)
{
    YAML::Node node = YAML::LoadFile(path);
    config.explodeConfig(node);
}

Args initArgs(int argc, char *argv[]) {
    // parse args here
    Args args;

    args.firstFrame = std::stoi(argv[ff]);
    std::cout << "Loading args:\n";
    std::cout << "First frame: " << args.firstFrame << '\n'
              << std::flush;
    args.lastFrame = std::stoi(argv[lf]);
    std::cout << "Last frame: " << args.lastFrame << '\n'
              << std::flush;
    args.initial = argv[initial];
    std::cout << "Initial CSV path: " << args.initial << '\n'
              << std::flush;
    args.input = argv[input];
    std::cout << "Input: " << args.input << '\n'
              << std::flush;
    args.output = argv[output];
    std::cout << "Output folder: " << args.output << '\n'
              << std::flush;
    args.config = argv[config];
    std::cout << "Config file: " << args.config << '\n'
              << std::flush;
    args.continueFrom = -1;

    // Optional resume args (positions 7 and 8). Both must be present to
    // activate resume; either missing → leave defaults (resume disabled).
    // argc indexing: argv[0]=binary, argv[1]=ff, ..., argv[6]=initial,
    // argv[7]=resumeFrom, argv[8]=resumeSourceDir.
    if (argc > static_cast<int>(resumeFrom)) {
        try {
            args.resumeFromFrame = std::stoi(argv[resumeFrom]);
        } catch (const std::exception &) {
            args.resumeFromFrame = 0;
        }
        std::cout << "Resume from frame (arg): " << args.resumeFromFrame
                  << '\n' << std::flush;
    }
    if (argc > static_cast<int>(resumeSourceDir)) {
        args.resumeSourceDir = argv[resumeSourceDir];
        std::cout << "Resume source dir (arg): " << args.resumeSourceDir
                  << '\n' << std::flush;
    }

    return args;
}

int main(int argc, char *argv[])
{
    // Suppress OpenCV TIFF warnings (ColorMap tag noise from microscopy TIFFs)
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    // Note: progress/status lines use std::endl for immediate visibility
    // through pipes (tee). Inner-loop diagnostics use '\n' for performance.

    // check user input
    if (argc < 7)
    {
        std::cerr << "Usage: celluniverse <firstFrame> <lastFrame> <input_pattern_or_dir_or_file> <output_dir> <config.yaml> <initial.csv>\n";
        return 1;
    }


    // parse args
    Args args = initArgs(argc, argv);

    // load config
    BaseConfig config;
    loadConfig(args.config, config);

    // CLI resume args (from run_celluniverse.sh INI preset) override whatever
    // was parsed from the YAML. Use argv-based detection so "absent CLI arg"
    // (argc < 8 / 9) does NOT clobber a YAML-set value.
    if (argc > static_cast<int>(resumeFrom)) {
        config.simulation.resume_from = args.resumeFromFrame;
    }
    if (argc > static_cast<int>(resumeSourceDir)) {
        config.simulation.resume_source_dir = args.resumeSourceDir;
    }

    config.printConfig();

    // load file paths
    PathVec imageFilePaths = ImageHandler::getImageFilePaths(args.input, args.firstFrame, args.lastFrame, config);

    // load cells
    std::string firstFrameFile;
    if (!imageFilePaths.empty()) {
        firstFrameFile = imageFilePaths.front().filename().string();
        std::cout << "[INFO] firstFrameFile=" << firstFrameFile << '\n';
    } else {
        std::cerr << "[WARN] imageFilePaths is empty; cannot determine initial frame filename." << '\n';
    }

    if (config.simulation.quit_after_preprocessing) {
        CellUniverse preprocessOnlyLineage({}, imageFilePaths, config, args.output, args.firstFrame, args.continueFrom);
        std::cout << "[DEBUG] quit_after_preprocessing=true; exiting after preprocessing/load phase." << std::endl;
        return 0;
    }

    // load cells here
    CellFactory cellFactory(config);
    std::map<Path, std::vector<Ellipsoid>> cells = cellFactory.createCells(args.initial, config.simulation.z_slices / 2,
                                                                        config.simulation.z_scaling, firstFrameFile);
    // create lineage
    CellUniverse lineage = CellUniverse(cells, imageFilePaths, config, args.output, args.firstFrame, args.continueFrom);

    // Checkpoint resume (Approach 2): if config.simulation.resume_from > 0
    // and resume_source_dir is set, load the checkpoint file from
    // `{resume_source_dir}/checkpoints/frame_{resume_from - 1:03d}.txt` and
    // skip the main loop up to resume_from. Requires the source run's
    // checkpoint to have been written with saveCheckpoint.
    int loopStart = 0;
    if (config.simulation.resume_from > 0 && !config.simulation.resume_source_dir.empty()) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "frame_%03d.txt", config.simulation.resume_from - 1);
        const std::string ckptPath =
            config.simulation.resume_source_dir + "/checkpoints/" + buf;
        if (lineage.loadCheckpoint(config.simulation.resume_from - 1, ckptPath)) {
            loopStart = config.simulation.resume_from;
            std::cout << "[Resume] skipping frames 0.." << (loopStart - 1)
                      << " — loaded checkpoint from " << ckptPath << std::endl;
        } else {
            std::cerr << "[Resume] checkpoint load failed, running from frame 0\n";
        }
    }

    // Run
    auto start = std::chrono::steady_clock::now();
    for (int frame = loopStart; frame < lineage.length(); ++frame)
    {
        // M2 Option A: lazy-load this frame's images (raw TIFF load +
        // percentile normalize + iterative preprocess). Constructor only
        // sampled for percentiles; actual frame data is loaded here.
        lineage.prepareFrame(frame);

        lineage.optimize(frame);

        lineage.copyCellsForward(frame + 1);

        lineage.saveImages(frame);

        lineage.saveCells(frame);

        // M1 memory optimization: release this frame's image stacks now that
        // we've captured its snapshot, saved its outputs, and copied cells
        // forward. Downstream only needs snapshot metadata + per-cell state.
        lineage.releaseFrameImages(frame);

        // Checkpoint for potential future resume.
        lineage.saveCheckpoint(frame);
    }
    auto end = std::chrono::steady_clock::now(); // timer end


    // end this program
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "Time elapsed: " << elapsed_seconds.count() << " seconds" << std::endl;

    return 0;
}
