#include "CellFactory.hpp"
#include "CsvHandler.hpp"

CellFactory::CellFactory(const BaseConfig &config) {
    std::string cellType = config.cellType;
    // Frame-1 seed for per-cell _brightness. After frame 1, the per-cell EMA
    // update (measureMeanBrightness * brightnessMeanAmplification blended via
    // brightnessUpdateBlend) takes over.
    initialBrightness = config.cell ? config.cell->initialBrightness : 0.2f;
    initialRadiusScale = config.cell ? config.cell->initialRadiusScale : 1.0f;
    // TODO: add more else if branches for more cell Types
    if (cellType == "ellipsoid") {
        Ellipsoid::cellConfig = *config.cell;
    } else {
        throw std::invalid_argument("Invalid cell type: " + config.cellType);
    }
}

// TODO: use abstract base class in the future to handle different cell types
std::map<Path, std::vector<Ellipsoid>> CellFactory::createCells(const Path &init_params_path, int z_offset, float z_scaling,
                                                               const std::string& firstFrameFile,
                                                               const std::string& initialZSpace) {
    std::map<Path, std::vector<Ellipsoid>> initialCells;
    const std::vector<InitialCellRecord> records =
        CsvHandler::loadInitialCells(init_params_path, firstFrameFile, z_scaling, initialRadiusScale, initialZSpace);

    for (const InitialCellRecord &record : records) {
        EllipsoidParams params(record.cellName,
                              record.x,
                              record.y,
                              record.z,
                              record.aRadius,
                              record.cRadius,
                              0.0f,
                              0.0f,
                              0.0f,
                              initialBrightness);
        params.bRadius = record.bRadius;
        params.isTrash = record.isTrash;
        initialCells[record.filePath].push_back(Ellipsoid(params));
    }

    std::cout << "Input Line Count : " << records.size() << '\n';
    std::cout << "Initial z space : " << initialZSpace
              << "  z_scaling : " << z_scaling << '\n';
    std::cout << "Initial frame keys loaded : " << initialCells.size() << '\n';
    if (!initialCells.empty()) {
        const auto& firstKey = initialCells.begin()->first;
        std::cout << "Example key : " << firstKey << "  cell count : " << initialCells.begin()->second.size() << '\n';
    }
    return initialCells;
}
