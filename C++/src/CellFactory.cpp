#include "CellFactory.hpp"

CellFactory::CellFactory(const BaseConfig &config) {
    std::string cellType = config.cellType;
    // TODO: add more else if branches for more cell Types
    if (cellType == "sphere") {
        // Sphere::cellConfig = *config.cell; // this is ideal, but entire code base must be changed to run it this way
        Spheroid::cellConfig = *config.cell;
    } if (cellType == "spheroid") {
        Spheroid::cellConfig = *config.cell;
    }
    else {
        throw std::invalid_argument("Invalid cell type: " + config.cellType);
    }
}

// TODO: use abstract base class in the future to handle different cell types
std::map<Path, std::vector<Spheroid>> CellFactory::createCells(const Path &init_params_path, int z_offset, float z_scaling) {
    std::ifstream file(init_params_path);
    std::string line;
    std::string firstLine;
    std::getline(file, firstLine); // remove the header
    std::map<Path, std::vector<Spheroid>> initialCells;
    int line_cnt = 0;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        float x, y, z, majorRadius, minorRadius;
        std::string floatStr;
        std::string filePath;
        std::string cellName;
        std::getline(ss, filePath, ',');
//        std::cout << "File Path: " << filePath << std::endl;
        std::getline(ss, cellName, ',');
        std::getline(ss, floatStr, ',');
        x = std::stof(floatStr);
        std::getline(ss, floatStr, ',');
        y = std::stof(floatStr);
        std::getline(ss, floatStr, ',');
        z = std::stof(floatStr);
        std::getline(ss, floatStr, ',');
        majorRadius = std::stof(floatStr);
        std::getline(ss, floatStr, ',');
        minorRadius = std::stof(floatStr);
        z *= z_scaling;
        // std::cout << "One new cell added!" << std::endl;
        // std::cout << "Input Major : " << majorRadius << std::endl;
        // std::cout << "Input Minor : " << minorRadius << std::endl;
        // std::cout << "Output Major : " << Spheroid::paramClass.majorRadius << std::endl;
        // std::cout << "Output Minor : " << Spheroid::paramClass.minorRadius << std::endl;


        initialCells[filePath].push_back(
        Spheroid(SpheroidParams(cellName, x, y, z, majorRadius, minorRadius)));
        
        ++line_cnt;
        continue;
    }
    std::cout << "Input Line Count : " << line_cnt << std::endl;
    std::cout << "initCells Size : " << initialCells["frame001.tif"].size() << std::endl;
    return initialCells;
}
