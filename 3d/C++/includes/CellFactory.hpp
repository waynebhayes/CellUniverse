//
// Created by yuant on 2/26/2024.
//

#ifndef CELLUNIVERSE_CELLFACTORY_HPP
#define CELLUNIVERSE_CELLFACTORY_HPP

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include "ConfigTypes.hpp"
#include "Cell.hpp"
#include "Sphere.hpp"
#include "types.hpp"

class CellFactory {
public:
    explicit CellFactory(const BaseConfig& config);
    std::map<Path, std::vector<Sphere>> createCells(const Path &init_params_path, int z_offset = 0, float z_scaling = 1.0);
};


#endif //CELLUNIVERSE_CELLFACTORY_HPP
