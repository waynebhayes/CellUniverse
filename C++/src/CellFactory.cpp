#include "../includes/CellFactory.hpp"

CellFactory::CellFactory(const BaseConfig &config)
{
    std::string cellType = config.cellType;

    // TODO: add more else-if branches for more cell types
    if (cellType == "sphere")
    {
        Spheroid::cellConfig = *config.cell;
    }
    else
    {
        throw std::invalid_argument("Invalid cell type: " + config.cellType);
    }
}

// TODO: use abstract base class in the future to handle different cell types
std::map<Path, std::vector<Spheroid>>
CellFactory::createCells(const Path &init_params_path, int z_offset, float z_scaling)
{
    std::ifstream file(init_params_path);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open initial csv: " + init_params_path);
    }

    std::string line;
    std::string firstLine;
    std::getline(file, firstLine); // remove the header

    std::map<Path, std::vector<Spheroid>> initialCells;

    while (std::getline(file, line))
    {
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

        // If you still want z_offset later:
        // z -= z_offset;

        z *= z_scaling;

        initialCells[filePath].push_back(
            Spheroid(SpheroidParams(cellName, x, y, z, radius)));

        continue;
    }

    return initialCells;
}