#include <iostream>
#include "types.hpp"
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "ConfigTypes.hpp"
#include "CellFactory.hpp"
#include "yaml-cpp/yaml.h"
#include "Spheroid.hpp"
#include "Lineage.hpp"
#include <chrono>
#include <algorithm>

#include "LineageViewer.hpp"

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

static void updateTiffConfigIfNeeded(const fs::path &file, BaseConfig &config)
{
    if (!(file.extension() == ".tif" || file.extension() == ".tiff"))
    {
        return;
    }

    std::vector<cv::Mat> images;
    cv::imreadmulti(file.string(), images, cv::IMREAD_UNCHANGED);
    int slices = static_cast<int>(images.size());
    config.simulation.z_slices = slices;
    config.simulation.z_values.clear();
    for (int j = 0; j < slices; ++j)
    {
        config.simulation.z_values.push_back(j - slices / 2);
    }
}

// helper function to get all image file paths
PathVec getImageFilePaths(const std::string &input, int firstFrame, int lastFrame, BaseConfig &config)
{
    PathVec imagePaths;

    // printf-style pattern input, e.g. frame%03d.tif
    if (input.find('%') != std::string::npos)
    {
        for (int i = firstFrame; lastFrame == -1 || i <= lastFrame; ++i)
        {
            char buffer[1024];
            std::snprintf(buffer, sizeof(buffer), input.c_str(), i);
            fs::path file(buffer);

            if (fs::exists(file) && fs::is_regular_file(file))
            {
                imagePaths.push_back(file);
                continue;
            }

            std::cerr << "Input file not found \"" << file << "\"" << std::endl;
            throw std::runtime_error("Input file not found");
        }
    }
    // Directory input, auto-detect image files
    else if (fs::is_directory(input))
    {
        PathVec allFiles;
        for (const auto &entry : fs::directory_iterator(input))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }
            const fs::path &p = entry.path();
            if (p.extension() == ".tif" || p.extension() == ".tiff")
            {
                allFiles.push_back(p);
            }
        }

        if (allFiles.empty())
        {
            throw std::runtime_error("No .tif/.tiff files found in directory: " + input);
        }

        std::sort(allFiles.begin(), allFiles.end());

        if (firstFrame < 0)
        {
            throw std::runtime_error("firstFrame must be >= 0 for directory input");
        }

        int start = firstFrame;
        int end = (lastFrame < 0) ? static_cast<int>(allFiles.size()) - 1
                                  : std::min(lastFrame, static_cast<int>(allFiles.size()) - 1);

        if (start >= static_cast<int>(allFiles.size()))
        {
            throw std::runtime_error("firstFrame is out of range for directory input");
        }
        if (start > end)
        {
            throw std::runtime_error("Invalid frame range for directory input");
        }

        for (int i = start; i <= end; ++i)
        {
            imagePaths.push_back(allFiles[i]);
        }
    }

    // Single file input
    else if (fs::exists(input) && fs::is_regular_file(input))
    {
        imagePaths.push_back(input);
    }
    else
    {
        throw std::runtime_error("Input is neither a pattern, directory, nor file: " + input);
    }

    if (!imagePaths.empty())
    {
        updateTiffConfigIfNeeded(imagePaths.front(), config);
    }

    // Print paths for verification
    for (const auto &path : imagePaths)
    {
        std::cout << path << std::endl;
    }

    return imagePaths;
}

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
    std::cout << "First frame: " << args.firstFrame << std::endl
              << std::flush;
    args.lastFrame = std::stoi(argv[lf]);
    std::cout << "Last frame: " << args.lastFrame << std::endl
              << std::flush;
    args.initial = argv[initial];
    std::cout << "Initial CSV path: " << args.initial << std::endl
              << std::flush;
    args.input = argv[input];
    std::cout << "Input: " << args.input << std::endl
              << std::flush;
    args.output = argv[output];
    std::cout << "Output folder: " << args.output << std::endl
              << std::flush;
    args.config = argv[config];
    std::cout << "Config file: " << args.config << std::endl
              << std::flush;
    args.continueFrom = -1;

    return args;
}

int main(int argc, char *argv[])
{
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
    config.printConfig();

    // load file paths
    PathVec imageFilePaths = getImageFilePaths(args.input, args.firstFrame, args.lastFrame, config);

    // load cells
    // [PATCH] Provide the first-frame filename for 4-column initial CSV (cell_type,z,y,x).
    // CellFactory needs a frame key (e.g., "t000.tif") to attach initial cells.
    // We pass it via an environment variable to avoid changing function signatures.
    if (!imageFilePaths.empty()) {
        const std::string firstFrameFile = imageFilePaths.front().filename().string();
        setenv("CELLUNIVERSE_INITIAL_FRAME_FILE", firstFrameFile.c_str(), 1);
        std::cout << "[INFO] CELLUNIVERSE_INITIAL_FRAME_FILE=" << firstFrameFile << std::endl;
    } else {
        std::cerr << "[WARN] imageFilePaths is empty; cannot set initial frame filename." << std::endl;
    }

    // load cells here
    CellFactory cellFactory(config);
    std::map<Path, std::vector<Spheroid>> cells = cellFactory.createCells(args.initial, config.simulation.z_slices / 2,
                                                                        config.simulation.z_scaling);
    // create lineage
    Lineage lineage = Lineage(cells, imageFilePaths, config, args.output, args.firstFrame, args.continueFrom);

    LineageViewer viewer;

    // Run
    auto start = std::chrono::steady_clock::now();
    for (int frame = 0; frame < lineage.length(); ++frame)
    {
        lineage.optimize(frame);


        // Build 2D viz list: rawName + x,y from current detected cells
        std::vector<LineageViewer::CellViz> viz;
        const auto &cellsNow = lineage.getCells(frame);
        viz.reserve(cellsNow.size());
        for (const auto &cell : cellsNow)
        {
            const auto params = cell.getCellParams();
            LineageViewer::CellViz c;
            c.rawName = params.name;
            c.x = (float)params.x;
            c.y = (float)params.y;
            viz.push_back(c);
        }
        // viewer.update(args.firstFrame + frame, viz);

        lineage.copyCellsForward(frame + 1);

        lineage.saveImages(frame);

        lineage.saveCells(frame);
    }
    auto end = std::chrono::steady_clock::now(); // timer end


    // end this program
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "Time elapsed: " << elapsed_seconds.count() << " seconds" << std::endl;

    std::cout << "Processing finished. Close the window manually to exit." << std::endl;
    // Keep window alive until user closes it
    while (true)
    {
        int key = cv::waitKey(30);
        // If window was manually closed
        if (cv::getWindowProperty("Cell Lineage (Realtime)", cv::WND_PROP_VISIBLE) < 1)
        {
            break;
        }
    }

    return 0;
}
