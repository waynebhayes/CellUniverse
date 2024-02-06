#ifndef CELLFACTORY_HPP
#define CELLFACTORY_HPP

#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "Cell.cpp"

class CellFactory {
private:
    std::unordered_map<std::string, std::function<Cell* ()>> cellTypes{
        {"sphere", []() { return new Sphere(); }},
        {"bacilli", []() { return new Bacilli(); }}
    };

public:
    CellFactory(const BaseConfig& config) {
        const std::string& cellType = config.cellType;
        if (cellTypes.find(cellType) == cellTypes.end()) {
            throw std::invalid_argument("Invalid cell type: " + cellType);
        }
        // Type error checking
        cellClass = cellTypes[cellType];
    }

    std::vector<Cell*> create_cells(const std::string& params_path, float z_offset = 0, float z_scaling = 1) {
        std::ifstream file(params_path);
        std::vector<Cell*> initial_cells;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string name;
            float x, y, z;
            z -= z_offset;
            z *= z_scaling;
            Cell* cell = cellClass();
            initial_cells.push_back(cell);
        }
        return initial_cells;
    }

private:
    std::function<Cell* ()> cellClass;
};


#endif // CELLFACTORY_HPP