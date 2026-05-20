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
#include "CellGroundTruthBuilder.hpp"
#include "ImageHandler.hpp"
#include "LineageTreeCreator.hpp"
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

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

class GroundTruthArgs
{
public:
    std::string inputFile{};
    std::string output{};
    std::string config{};
    std::string csvOutput{};
};

// helper function to load the config
void loadConfig(const std::string &path, BaseConfig &config)
{
    YAML::Node node = YAML::LoadFile(path);
    config.explodeConfig(node);
}

void applyRuntimeOverrides(BaseConfig &config)
{
    const char *threadEnv = std::getenv("CELLUNIVERSE_THREADS");
    std::string threadSource = "config";

    if (threadEnv != nullptr && std::string(threadEnv).size() > 0)
    {
        try
        {
            config.simulation.parallel_threads = std::stoi(threadEnv);
            threadSource = "CELLUNIVERSE_THREADS";
        }
        catch (const std::exception &)
        {
            std::cerr << "[WARN] Ignoring invalid CELLUNIVERSE_THREADS="
                      << threadEnv << "; using config value "
                      << config.simulation.parallel_threads << '\n';
        }
    }

    config.simulation.parallel_threads = std::max(1, config.simulation.parallel_threads);
    config.simulation.parallel_min_slices = std::max(1, config.simulation.parallel_min_slices);

    const unsigned int hardwareThreads = std::thread::hardware_concurrency();
    if (hardwareThreads > 0 &&
        config.simulation.parallel_threads > static_cast<int>(hardwareThreads))
    {
        std::cerr << "[WARN] Requested parallel_threads="
                  << config.simulation.parallel_threads
                  << " but hardware_concurrency=" << hardwareThreads
                  << "; clamping to hardware_concurrency." << '\n';
        config.simulation.parallel_threads = static_cast<int>(hardwareThreads);
    }

    cv::setNumThreads(config.simulation.parallel_threads);
#ifdef _OPENMP
    omp_set_num_threads(config.simulation.parallel_threads);
#endif

    const char *seedEnv = std::getenv("CELLUNIVERSE_SEED");
    std::cout << "[Runtime Parallelism] mode="
              << (config.simulation.parallel_threads > 1 ? "parallel_z_slices" : "single_thread")
              << " threads=" << config.simulation.parallel_threads
              << " source=" << threadSource
              << " hardware_concurrency=" << hardwareThreads
              << " parallel_min_slices=" << config.simulation.parallel_min_slices
              << " opencv_threads=" << cv::getNumThreads()
              << std::endl;
    if (seedEnv != nullptr && std::string(seedEnv).size() > 0)
    {
        std::cout << "[Runtime Random] CELLUNIVERSE_SEED=" << seedEnv << std::endl;
    }
    std::cout << "[Efficiency Metric] primary=seconds_per_frame"
              << " normalized=seconds_per_cell_iteration"
              << " realtime=iterations_per_second"
              << " note=split_attempts_are_reported_separately_because_they_are_much_more_expensive"
              << std::endl;
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

GroundTruthArgs initGroundTruthArgs(char *argv[])
{
    GroundTruthArgs args;
    args.inputFile = argv[2];
    args.output = argv[3];
    args.config = argv[4];
    args.csvOutput = argv[5];

    std::cout << "Loading ground-truth builder args:\n";
    std::cout << "Input frame: " << args.inputFile << '\n' << std::flush;
    std::cout << "Output folder: " << args.output << '\n' << std::flush;
    std::cout << "Config file: " << args.config << '\n' << std::flush;
    std::cout << "CSV output: " << args.csvOutput << '\n' << std::flush;
    return args;
}

int main(int argc, char *argv[])
{
    // Suppress OpenCV TIFF warnings (ColorMap tag noise from microscopy TIFFs)
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    // Note: progress/status lines use std::endl for immediate visibility
    // through pipes (tee). Inner-loop diagnostics use '\n' for performance.

    if (argc >= 2 && std::string(argv[1]) == "--lineage-tree")
    {
        LineageTreeCreator::RenderOptions options;
        int argi = 2;
        while (argi < argc)
        {
            const std::string opt = argv[argi];
            if (opt == "--first-frame" && argi + 1 < argc)
            {
                options.firstFrame = std::stoi(argv[argi + 1]);
                argi += 2;
            }
            else if (opt == "--last-frame" && argi + 1 < argc)
            {
                options.lastFrame = std::stoi(argv[argi + 1]);
                argi += 2;
            }
            else if (opt == "--fps" && argi + 1 < argc)
            {
                options.fps = std::stod(argv[argi + 1]);
                argi += 2;
            }
            else
            {
                break;
            }
        }

        if (argc - argi < 2)
        {
            std::cerr << "Usage: celluniverse --lineage-tree [--first-frame N] [--last-frame N] [--fps F] "
                      << "<output.png|output.mp4> <cells.csv> [cells.csv ...]\n"
                      << "GIF output is handled by C++/scripts/make_lineage_tree_demo.py.\n";
            return 1;
        }

        std::vector<std::string> csvPaths;
        const std::string lineageOutput = argv[argi++];
        for (int i = argi; i < argc; ++i)
        {
            csvPaths.emplace_back(argv[i]);
        }

        const bool ok = LineageTreeCreator::renderCsvFiles(csvPaths, lineageOutput, options);
        return ok ? 0 : 1;
    }

    if (argc >= 2 && std::string(argv[1]) == "--build-ground-truth")
    {
        if (argc < 6)
        {
            std::cerr << "Usage: celluniverse --build-ground-truth <input_frame.tif> <output_dir> <config.yaml> <csv_output>\n";
            return 1;
        }

        GroundTruthArgs args = initGroundTruthArgs(argv);
        BaseConfig config;
        loadConfig(args.config, config);
        applyRuntimeOverrides(config);
        config.printConfig();

        fs::create_directories(args.output);
        CellGroundTruthBuilder builder(config, fs::path(args.output));
        builder.buildInitialCsvForFrame(fs::path(args.inputFile), fs::path(args.csvOutput));
        return 0;
    }

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

    applyRuntimeOverrides(config);
    config.printConfig();

    if (config.cellType == "ellipsoid" && config.cell) {
        Ellipsoid::cellConfig = *config.cell;
    }

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
        preprocessOnlyLineage.preprocessAllFramesAlignedToMinimumBackground(false);
        std::cout << "[DEBUG] quit_after_preprocessing=true; exiting after preprocessing/load phase." << std::endl;
        return 0;
    }

    // load cells here
    CellFactory cellFactory(config);
    std::map<Path, std::vector<Ellipsoid>> cells = cellFactory.createCells(args.initial, config.simulation.z_slices / 2,
                                                                        config.simulation.z_scaling, firstFrameFile,
                                                                        config.simulation.initial_z_space);
    // create lineage
    CellUniverse lineage = CellUniverse(cells, imageFilePaths, config, args.output, args.firstFrame, args.continueFrom);
    const bool prepareAnalyzeOneFrame =
        config.simulation.prepare_analyze_one_frame &&
        !config.simulation.quit_after_preprocessing;
    if (prepareAnalyzeOneFrame) {
        std::cout << "[INFO] prepare_analyze_one_frame=true; each frame will be prepared immediately before optimize()."
                  << std::endl;
    } else {
        lineage.preprocessAllFramesAlignedToMinimumBackground(true);
    }

    // Checkpoint resume (Approach 2): resume_from is the absolute dataset
    // frame to analyze next. Load `{resume_source_dir}/checkpoints/
    // frame_{resume_from - 1:03d}.txt`, install its copied-forward cells into
    // the local frame corresponding to resume_from, then start the loop there.
    int loopStart = 0;
    if (config.simulation.resume_from > 0 && !config.simulation.resume_source_dir.empty()) {
        const int resumeFrame = config.simulation.resume_from;
        const int checkpointFrame = resumeFrame - 1;
        const int targetLocalFrame = resumeFrame - args.firstFrame;
        if (targetLocalFrame < 0 ||
            targetLocalFrame >= static_cast<int>(lineage.length())) {
            std::cerr << "[Resume] invalid resume_from=" << resumeFrame
                      << " for requested frame range [" << args.firstFrame
                      << "," << args.lastFrame << "]; running from local frame 0\n";
        } else {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "frame_%03d.txt", checkpointFrame);
            const std::string ckptPath =
                config.simulation.resume_source_dir + "/checkpoints/" + buf;
            if (lineage.loadCheckpoint(checkpointFrame, targetLocalFrame, ckptPath)) {
                loopStart = targetLocalFrame;
                std::cout << "[Resume] skipping absolute frames "
                          << args.firstFrame << ".." << (resumeFrame - 1)
                          << " (local 0.." << (loopStart - 1) << ")"
                          << " — loaded checkpoint from " << ckptPath << std::endl;
            } else {
                std::cerr << "[Resume] checkpoint load failed, running from frame 0\n";
            }
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

        if (config.simulation.release_analyzed_exported_frames) {
            // Release image-heavy stacks after exported outputs and saved cells.
            // Later frames use copied cells, checkpoints, and cached summaries.
            lineage.releaseFrameImages(frame);
        }

        // Checkpoint for potential future resume.
        lineage.saveCheckpoint(frame);
    }
    auto end = std::chrono::steady_clock::now(); // timer end


    // end this program
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "Time elapsed: " << elapsed_seconds.count() << " seconds" << std::endl;

    return 0;
}
