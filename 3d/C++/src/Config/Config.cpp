// Config.cpp
#include "Config.hpp"

// Commit Krishna's config file
#include <iostream>
#include <vector>

class BaseModel {
    // TODO must find a replacement to pydantic python library BaseModel
};

class SimulationConfig : public BaseModel {
public:
    int iterationsPerCell;
    float backgroundColor;
    float cellColor;
    int padding = 0;
    float zScaling = 1;
    float blurSigma = 0;
    int zSlices = -1;
    std::vector<int> zValues;

    SimulationConfig() {
        iterationsPerCell = 0;
        backgroundColor = 0.0f;
        cellColor = 0.0f;
    }

    void checkZValues() const {
        if (!zValues.empty()) {
            throw std::invalid_argument("zValues should not be set manually");
        }
    }

    void checkZSlices() const {
        if (zSlices != -1) {
            throw std::invalid_argument("zSlices should not be set manually");
        }
    }
};
