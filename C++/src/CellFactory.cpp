#include "CellFactory.hpp"

CellFactory::CellFactory(const BaseConfig &config) {
    std::string cellType = config.cellType;
    // TODO: add more else if branches for more cell Types
    if (cellType == "spheroid") {
        Spheroid::cellConfig = *config.cell;
    } else {
        throw std::invalid_argument("Invalid cell type: " + config.cellType);
    }
}

// TODO: use abstract base class in the future to handle different cell types
std::map<Path, std::vector<Spheroid>> CellFactory::createCells(const Path &init_params_path, int z_offset, float z_scaling,
                                                               const std::string& firstFrameFile) {
    std::ifstream file(init_params_path);
    std::string line;
    std::string firstLine;
    std::getline(file, firstLine); // remove the header
    std::map<Path, std::vector<Spheroid>> initialCells;
    int line_cnt = 0;

while (std::getline(file, line)) {
    if (line.empty()) {
        continue;
    }

    // Split CSV row into tokens (supports both 7-col and 4-col formats).
    std::vector<std::string> tokens;
    {
        std::istringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            // Trim spaces (very simple trim)
            while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t')) tok.erase(tok.begin());
            while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t' || tok.back() == '\r')) tok.pop_back();
            tokens.push_back(tok);
        }
    }

    // ----------------------------
    // Case A: Original 7-column format:
    // filePath, cellName, x, y, z, majorRadius, minorRadius
    // ----------------------------
    if (tokens.size() >= 7) {
        std::string filePath = tokens[0];
        std::string cellName = tokens[1];

        float x = std::stof(tokens[2]);
        float y = std::stof(tokens[3]);
        float z = std::stof(tokens[4]);
        float majorRadius = std::stof(tokens[5]);
        float minorRadius = std::stof(tokens[6]);

        // Optional brightness in column 8 (measured from real image)
        float brightness = 0.5f;
        if (tokens.size() >= 8 && tokens[7] != "None" && !tokens[7].empty()) {
            brightness = std::stof(tokens[7]);
        }

        z *= z_scaling;

        initialCells[filePath].push_back(
            Spheroid(SpheroidParams(cellName, x, y, z, majorRadius, minorRadius,
                                   0.0f, 0.0f, 0.0f, brightness))
        );

        ++line_cnt;
        continue;
    }

    // ----------------------------
    // Case B: 4-column Napari format:
    // cell_type, z, y, x
    // We must attach cells to a frame key, e.g., "t000.tif".
    // main.cpp passes firstFrameFile as a parameter.
    // ----------------------------
    if (tokens.size() == 4) {
        std::string filePath = !firstFrameFile.empty() ? firstFrameFile : std::string("t000.tif");

        const std::string cellType = tokens[0];

        float z = std::stof(tokens[1]);
        float y = std::stof(tokens[2]);
        float x = std::stof(tokens[3]);

        // Apply z scaling (index -> scaled space used by the C++ tracker)
        z *= z_scaling;

        // Default radii for embryo initial points (tunable)
        // You can adjust these later if needed.
        const float defaultMajorRadius = 10.0f;
        const float defaultMinorRadius = 10.0f;

        // Generate a stable name
        const std::string cellName = cellType + "_" + std::to_string(line_cnt + 1);

        initialCells[filePath].push_back(
            Spheroid(SpheroidParams(cellName, x, y, z, defaultMajorRadius, defaultMinorRadius))
        );

        ++line_cnt;
        continue;
    }

    // Unknown/invalid line
    std::cerr << "[WARN] Skipping invalid initial CSV row (expected 7 or 4 columns): " << line << '\n';
}

    std::cout << "Input Line Count : " << line_cnt << '\n';
    std::cout << "Initial frame keys loaded : " << initialCells.size() << '\n';
    if (!initialCells.empty()) {
        const auto& firstKey = initialCells.begin()->first;
        std::cout << "Example key : " << firstKey << "  cell count : " << initialCells.begin()->second.size() << '\n';
    }
    return initialCells;
}
