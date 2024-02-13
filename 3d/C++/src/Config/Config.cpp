#include <iostream>
#include <fstream>
#include "yaml-cpp/yaml.h"
#include "Sphere.cpp"

//Update 2/12:
//move BaseConfig to ConfigType
//remove templetes and replace by a new attribute CellType (char)


BaseConfig* loadConfig(const std::string& path) {
    YAML::Node config = YAML::LoadFile(path);
    BaseConfig* Basecf = new BaseConfig();
    if (config["cellType"].as<std::string>() == "sphere") {
        Basecf->CellType = 's';
        return Basecf;
    } else if (config["cellType"].as<std::string>() == "bacilli") {
        Basecf->CellType = 'b';
        return Basecf;
    } else {
        delete(Basecf);
        throw std::invalid_argument("Invalid cell type: " + config["cellType"].as<std::string>());
    }
}
