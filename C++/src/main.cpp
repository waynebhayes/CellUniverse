#include <iostream>
#include "types.hpp"
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "ConfigTypes.hpp"
#include "CellFactory.hpp"
#include "yaml-cpp/yaml.h"
#include "Sphere.hpp"
#include "Lineage.hpp"
#include <chrono>

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
    // Add other arguments as necessary
};

// helper function to get all image file paths
PathVec getImageFilePaths(const std::string &inputPattern, int firstFrame, int lastFrame, BaseConfig &config)
{
    PathVec imagePaths;
    for (int i = firstFrame; lastFrame == -1 || i <= lastFrame; ++i)
    {
        char buffer[100];
        sprintf(buffer, inputPattern.c_str(), i);
        fs::path file(buffer);

        if (fs::exists(file) && fs::is_regular_file(file))
        {
            imagePaths.push_back(file);

            // Setup some configurations automatically if they are tif files
            if (file.extension() == ".tif" || file.extension() == ".tiff")
            {
                std::vector<cv::Mat> images;
                cv::imreadmulti(file.string(), images, cv::IMREAD_UNCHANGED);
                //                std::cout << "Loaded " << images.size() << " images from the TIFF file." << std::endl;
                int slices = images.size(); // Assuming the first dimension is the number of slices
                // set the uninitialized z_slices and z_values
                config.simulation.z_slices = slices;
                config.simulation.z_values.clear();
                for (int j = 0; j < slices; ++j)
                {
                    config.simulation.z_values.push_back(j - slices / 2);
                }
            }
        }
        else
        {
            std::cerr << "Input file not found \"" << file << "\"" << std::endl;
            throw std::runtime_error("Input file not found");
        }
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

int main(int argc, char *argv[])
{
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
    std::cout << "Input folder: " << args.input << std::endl
              << std::flush;
    args.output = argv[output];
    std::cout << "Output folder: " << args.output << std::endl
              << std::flush;
    args.config = argv[config];
    std::cout << "Config file: " << args.config << std::endl
              << std::flush;
    args.continueFrom = -1;

    // load config here
    BaseConfig config;
    loadConfig(args.config, config);
    config.printConfig();
    // load file paths here
    PathVec imageFilePaths = getImageFilePaths(args.input, args.firstFrame, args.lastFrame, config);

    // load cells here
    CellFactory cellFactory(config);
    std::map<Path, std::vector<Sphere>> cells = cellFactory.createCells(args.initial, config.simulation.z_slices / 2,
                                                                        config.simulation.z_scaling);
    // create lineage here
    Lineage lineage = Lineage(cells, imageFilePaths, config, args.output, args.continueFrom);

    // Run
    auto start = std::chrono::steady_clock::now();
    for (int frame = 0; frame < lineage.length(); ++frame)
    {
        lineage.optimize(frame);
        lineage.copyCellsForward(frame + 1);
        lineage.saveImages(frame);
        // lineage.saveCells(frame); // TODO: Fix this
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "Time elapsed: " << elapsed_seconds.count() << " seconds" << std::endl;

    return 0;
}