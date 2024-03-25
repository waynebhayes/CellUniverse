#include "CellFactory.hpp"

CellFactory::CellFactory(const BaseConfig &config) {
    std::string cellType = config.cellType;
    // TODO: add more else if branches for more cell Types
    if (cellType == "sphere") {
        Sphere::cellConfig = *config.cell;
    }
    else {
        throw std::invalid_argument("Invalid cell type: " + config.cellType);
    }
}

// TODO: use abstract base class in the future to handle different cell types
std::map<Path, std::vector<Sphere>> CellFactory::createCells(const Path &init_params_path, int z_offset, float z_scaling) {
    std::ifstream file(init_params_path);
    std::string line;
    std::string firstLine;
    std::getline(file, firstLine); // remove the header
    std::map<Path, std::vector<Sphere>> initialCells;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        float x, y, z, radius;
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
        radius = std::stof(floatStr);
        z -= z_offset;
        z *= z_scaling;
        Sphere::paramClass = SphereParams(cellName, x, y, z, radius);
        initialCells[filePath].push_back(Sphere(SphereParams(cellName, x, y, z, radius)));
        continue;
    }
    return initialCells;
}
