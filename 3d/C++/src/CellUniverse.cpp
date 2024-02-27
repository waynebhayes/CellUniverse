#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include "ConfigTypes.hpp"

// Helper function to get image file paths
std::vector<std::filesystem::path> getImageFilePaths(const std::string& inputPattern, int firstFrame, int lastFrame, BaseConfig& config) {
    std::vector<std::filesystem::path> imagePaths;
    int i = firstFrame;
    try {
        while (lastFrame == -1 || i <= lastFrame) {
            std::filesystem::path file = std::filesystem::path(inputPattern).replace_filename(std::to_string(i));

            if (std::filesystem::exists(file) && std::filesystem::is_regular_file(file)) {
                imagePaths.push_back(file);

                if (file.extension() == ".tif" || file.extension() == ".tiff") {
                    // Logic to set up configurations if it's a tif file
                }
            }
            else {
                throw std::invalid_argument("Input file not found: " + file.string());
            }
            i++;
        }
    }
    catch (const std::invalid_argument& e) {
        if (lastFrame != -1 && imagePaths.size() != lastFrame - firstFrame + 1) {
            throw;
        }
    }

    for (const auto& path : imagePaths) {
        std::cout << path.string() << std::endl;
    }

    return imagePaths;
}

class CellUniverse {
private:
    Lineage lineage;

public:
    CellUniverse(Args args) {

        // Config
        BaseConfig config = loadConfig(args.config);
        std::vector<std::filesystem::path> imageFilePaths = getImageFilePaths(args.input, args.firstFrame, args.lastFrame, config);

        // Cells
        CellFactory cellFactory(config);
        std::unordered_map<std::string, std::vector<Cell*>> cells = cellFactory.createCells(args.initial, config.simulation.z_slices / 2, config.simulation.z_scaling);

        // Lineage
        this->lineage = Lineage(cells, imageFilePaths, config, args.output, args.continueFrom);
    }

    void run() {
        double currentTime = time(nullptr);
        for (int frame = 0; frame < this->lineage.size(); frame++) {
            this->lineage.optimize(frame);
            this->lineage.copyCellsForward(frame + 1);
            this->lineage.saveImages(frame);
            // this->lineage.saveCells(frame); // TODO: Figure out why this isn't working
        }

        std::cout << "Time elapsed: " << time(nullptr) - currentTime << " seconds" << std::endl;
    }
};
