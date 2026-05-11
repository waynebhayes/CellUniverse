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
#include "Spheroid.hpp"
#include "CellUniverse.hpp"
#include "CellGroundTruthBuilder.hpp"
#include "ImageHandler.hpp"
#include "LineageTreeCreator.hpp"
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <thread>


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
    // Keep the YAML file conservative for laptop runs, then let batch jobs
    // opt into parallel execution through the environment. This makes the
    // command line self-contained on OpenLab while preserving the default
    // single-thread behavior for memory-limited local machines.
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

    // OpenCV owns the worker pool used by cv::parallel_for_. Matching the
    // OpenCV thread count to the Cell Universe setting keeps the z-slice
    // renderer and cost evaluator inside the CPU allocation requested from
    // Slurm. The algorithm still schedules the same perturbations and split
    // candidates; only independent slice work is distributed across workers.
    cv::setNumThreads(config.simulation.parallel_threads);

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

Args initArgs(char *argv[]) {
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
    Args args = initArgs(argv);

    // load config
    BaseConfig config;
    loadConfig(args.config, config);

    ImageHandler::applyDatasetRuntimeProfile(args.input, config);
    applyRuntimeOverrides(config);

    // load file paths
    PathVec imageFilePaths = ImageHandler::getImageFilePaths(args.input, args.firstFrame, args.lastFrame, config);
    config.printConfig();

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
    std::map<Path, std::vector<Spheroid>> cells = cellFactory.createCells(args.initial, config.simulation.z_slices / 2,
                                                                        config.simulation.z_scaling, firstFrameFile);
    // create lineage
    CellUniverse lineage = CellUniverse(cells, imageFilePaths, config, args.output, args.firstFrame, args.continueFrom);
    LineageTreeCreator lineageTree;
    const bool showLineageTreeWindow = config.simulation.enable_lineage_tree_window;

    // Run
    auto start = std::chrono::steady_clock::now();
    for (int frame = 0; frame < lineage.length(); ++frame)
    {
        const auto frameWallStart = std::chrono::steady_clock::now();
        const int displayFrame = args.firstFrame + frame;

        auto stageStart = std::chrono::steady_clock::now();
        lineage.optimize(frame);
        const double optimizeSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - stageStart).count();

        if (showLineageTreeWindow)
        {
            lineageTree.update(args.firstFrame + frame,
                               LineageTreeCreator::makeCellViz(lineage.getCells(frame)));
        }

        stageStart = std::chrono::steady_clock::now();
        lineage.saveImages(frame);
        const double saveImagesSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - stageStart).count();

        stageStart = std::chrono::steady_clock::now();
        lineage.saveCells(frame);
        const double saveCellsSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - stageStart).count();

        stageStart = std::chrono::steady_clock::now();
        lineage.copyCellsForward(frame + 1);
        const double carrySeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - stageStart).count();

        const double frameWallSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - frameWallStart).count();
        std::cout << "[Frame Wall Timing] frame=" << displayFrame
                  << " wall_sec=" << frameWallSeconds
                  << " optimize_sec=" << optimizeSeconds
                  << " save_images_sec=" << saveImagesSeconds
                  << " save_cells_sec=" << saveCellsSeconds
                  << " carry_forward_sec=" << carrySeconds
                  << std::endl;
    }
    lineageTree.close();
    auto end = std::chrono::steady_clock::now(); // timer end


    // end this program
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "Time elapsed: " << elapsed_seconds.count() << " seconds" << std::endl;

    return 0;
}
