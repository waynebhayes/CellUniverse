#include <iostream>
#include "types.hpp"
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "ConfigTypes.hpp"
#include "CellFactory.hpp"
#include "yaml-cpp/yaml.h"

struct Args {
    std::string config;
    std::string input;
    int first_frame;
    int last_frame;
    std::string initial;
    std::string output;
    int continue_from;
    // Add other arguments as necessary
};

// helper function to get tall image file paths
PathVec get_image_file_paths(const std::string& input_pattern, int first_frame, int last_frame, BaseConfig& config) {
    PathVec image_paths;
    for (int i = first_frame; last_frame == -1 || i <= last_frame; ++i) {
        char buffer[100];
        sprintf(buffer, input_pattern.c_str(), i);
        fs::path file(buffer);

        if (fs::exists(file) && fs::is_regular_file(file)) {
            image_paths.push_back(file);

            // Setup some configurations automatically if they are tif files
            if (file.extension() == ".tif" || file.extension() == ".tiff") {
                cv::Mat img = cv::imread(file.string(), cv::IMREAD_UNCHANGED); // Use UNCHANGED to read the image as is, including alpha channel
                int slices = img.size[0]; // Assuming the first dimension is the number of slices
                // set the uninitialized z_slices and z_values
                config.simulation.z_slices = slices;
                config.simulation.z_values.clear();
                for (int j = 0; j < slices; ++j) {
                    config.simulation.z_values.push_back(j - slices / 2);
                }
            }
        } else {
            std::cerr << "Input file not found \"" << file << "\"" << std::endl;
            throw std::runtime_error("Input file not found");
        }
    }

    // Print paths for verification
    for (const auto& path : image_paths) {
        std::cout << path << std::endl;
    }

    return image_paths;
}

// helper function to load the config
void loadConfig(const std::string& path, BaseConfig& config) {
    YAML::Node node = YAML::LoadFile(path);
    config.explodeConfig(node);
}


CellMap create_cells(const Path &init_params_path, int z_offset, float z_scaling) {
    std::ifstream file(init_params_path.c_str());
    std::string line;
    std::string firstLine;
    std::getline(file, firstLine); // remove the header
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        float x, y, z, radius;
        std::string floatStr;
        std::string filePath;
        std::string cellName;
        std::getline(ss, filePath, ',');
        std::getline(ss, cellName, ',');
        std::getline(ss, floatStr, ',');
        x = std::stof(floatStr);
        std::getline(ss, floatStr, ',');
        y = std::stof(floatStr);
        std::getline(ss, floatStr, ',');
        z = std::stof(floatStr);
        std::getline(ss, floatStr, ',');
        radius = std::stof(floatStr);
        continue;

        // Further processing to create cells
        // This is a placeholder. You need to parse each field and construct cells accordingly.
    }
    return {};
}


int main(int argc, char* argv[])
{
    // parse args here
    Args args;
    
    args.first_frame = std::stoi(argv[ff]);
    std::cout << "Loading args:\n";
    std::cout << "First frame: " << args.first_frame << std::endl << std::flush;
    args.last_frame = std::stoi(argv[lf]);
    std::cout << "Last frame: " << args.last_frame << std::endl<< std::flush;
    args.initial = argv[initial];
    std::cout << "Initial CSV path: " << args.initial << std::endl << std::flush;
    args.input = argv[input];
    std::cout << "Input folder: " << args.input << std::endl << std::flush;
    args.output = argv[output];
    std::cout << "Output folder: " << args.output << std::endl << std::flush;
    args.config = argv[config];
    std::cout << "Config file: "<< args.config << std::endl << std::flush;
    args.continue_from = -1;

    // load config here
    BaseConfig config;
    loadConfig(args.config, config);

    // load file paths here
    PathVec imageFilePaths = get_image_file_paths(args.input, args.first_frame, args.last_frame, config);

    // load cells here
    CellFactory cellFactory(config);
    CellMap cells = create_cells(args.initial, config.simulation.z_slices / 2, config.simulation.z_scaling);
    return 0;
}