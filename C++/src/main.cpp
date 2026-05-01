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
        lineage.optimize(frame);

        if (showLineageTreeWindow)
        {
            lineageTree.update(args.firstFrame + frame,
                               LineageTreeCreator::makeCellViz(lineage.getCells(frame)));
        }

        lineage.saveImages(frame);

        lineage.saveCells(frame);

        lineage.copyCellsForward(frame + 1);
    }
    lineageTree.close();
    auto end = std::chrono::steady_clock::now(); // timer end


    // end this program
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "Time elapsed: " << elapsed_seconds.count() << " seconds" << std::endl;

    return 0;
}
