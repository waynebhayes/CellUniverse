#include "CellFactory.hpp"

#include <cctype>
#include <unordered_map>

namespace
{
std::string trim(std::string value)
{
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

std::vector<std::string> splitCsvLine(const std::string &line)
{
    std::vector<std::string> tokens;
    std::string token;
    bool inQuotes = false;

    for (char ch : line) {
        if (ch == '"') {
            inQuotes = !inQuotes;
            continue;
        }
        if (ch == ',' && !inQuotes) {
            tokens.push_back(trim(token));
            token.clear();
            continue;
        }
        token.push_back(ch);
    }
    tokens.push_back(trim(token));
    return tokens;
}

std::string normalizeHeaderName(const std::string &header)
{
    std::string normalized;
    for (char ch : header) {
        if (std::isalnum(static_cast<unsigned char>(ch))) {
            normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
    }
    return normalized;
}

std::unordered_map<std::string, size_t> buildHeaderIndex(const std::vector<std::string> &headers)
{
    std::unordered_map<std::string, size_t> index;
    for (size_t i = 0; i < headers.size(); ++i) {
        index[normalizeHeaderName(headers[i])] = i;
    }
    return index;
}

bool findColumn(const std::unordered_map<std::string, size_t> &index,
                std::initializer_list<const char *> names,
                size_t &column)
{
    for (const char *name : names) {
        auto it = index.find(normalizeHeaderName(name));
        if (it != index.end()) {
            column = it->second;
            return true;
        }
    }
    return false;
}

bool getToken(const std::vector<std::string> &tokens, size_t column, std::string &value)
{
    if (column >= tokens.size()) {
        return false;
    }
    value = tokens[column];
    const std::string normalized = normalizeHeaderName(value);
    return !value.empty() && normalized != "none" && normalized != "null" && normalized != "nan";
}

float parseScaledFloat(const std::string &value, float scale)
{
    return std::stof(value) * scale;
}
}

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
                                                               const std::string& firstFrameFile) {
    std::ifstream file(init_params_path);
    std::string line;
    std::string firstLine;
    std::getline(file, firstLine); // remove the header
    const std::unordered_map<std::string, size_t> headerIndex = buildHeaderIndex(splitCsvLine(firstLine));
    std::map<Path, std::vector<Ellipsoid>> initialCells;
    int line_cnt = 0;

while (std::getline(file, line)) {
    if (line.empty()) {
        continue;
    }

    const std::vector<std::string> tokens = splitCsvLine(line);

    try {
        size_t filePathCol = 0;
        size_t cellNameCol = 0;
        size_t xCol = 0;
        size_t yCol = 0;
        size_t zCol = 0;
        size_t aRadiusCol = 0;
        size_t bRadiusCol = 0;
        size_t cRadiusCol = 0;
        const bool hasNamedCellColumns =
            findColumn(headerIndex, {"filePath", "file", "frame", "frameName", "image", "imageFile", "imagePath"}, filePathCol) &&
            findColumn(headerIndex, {"cellName", "name", "id", "cellId", "label"}, cellNameCol) &&
            findColumn(headerIndex, {"x"}, xCol) &&
            findColumn(headerIndex, {"y"}, yCol) &&
            findColumn(headerIndex, {"z"}, zCol) &&
            findColumn(headerIndex, {"aRadius", "majorRadius", "radiusA", "radius", "r"}, aRadiusCol) &&
            findColumn(headerIndex, {"cRadius", "minorRadius", "radiusC", "zRadius"}, cRadiusCol);

        if (hasNamedCellColumns) {
            std::string filePath;
            std::string cellName;
            std::string xText;
            std::string yText;
            std::string zText;
            std::string aRadiusText;
            std::string cRadiusText;
            if (!getToken(tokens, filePathCol, filePath) ||
                !getToken(tokens, cellNameCol, cellName) ||
                !getToken(tokens, xCol, xText) ||
                !getToken(tokens, yCol, yText) ||
                !getToken(tokens, zCol, zText) ||
                !getToken(tokens, aRadiusCol, aRadiusText) ||
                !getToken(tokens, cRadiusCol, cRadiusText)) {
                std::cerr << "[WARN] Skipping invalid initial CSV row (missing named columns): " << line << '\n';
                continue;
            }

            float x = std::stof(xText);
            float y = std::stof(yText);
            float z = std::stof(zText);
            float aRadius = parseScaledFloat(aRadiusText, initialRadiusScale);
            float bRadius = aRadius;
            if (findColumn(headerIndex, {"bRadius", "radiusB", "middleRadius", "intermediateRadius"}, bRadiusCol)) {
                std::string bRadiusText;
                if (getToken(tokens, bRadiusCol, bRadiusText)) {
                    bRadius = parseScaledFloat(bRadiusText, initialRadiusScale);
                }
            }
            float cRadius = parseScaledFloat(cRadiusText, initialRadiusScale);

            z *= z_scaling;

            EllipsoidParams params(cellName, x, y, z, aRadius, cRadius,
                                  0.0f, 0.0f, 0.0f, initialBrightness);
            params.bRadius = bRadius;
            initialCells[filePath].push_back(Ellipsoid(params));

            ++line_cnt;
            continue;
        }

        size_t cellTypeCol = 0;
        const bool hasNamedNapariColumns =
            findColumn(headerIndex, {"cell_type", "cellType", "type"}, cellTypeCol) &&
            findColumn(headerIndex, {"z"}, zCol) &&
            findColumn(headerIndex, {"y"}, yCol) &&
            findColumn(headerIndex, {"x"}, xCol);

        if (hasNamedNapariColumns && tokens.size() >= 4) {
            std::string cellType;
            std::string zText;
            std::string yText;
            std::string xText;
            if (!getToken(tokens, cellTypeCol, cellType) ||
                !getToken(tokens, zCol, zText) ||
                !getToken(tokens, yCol, yText) ||
                !getToken(tokens, xCol, xText)) {
                std::cerr << "[WARN] Skipping invalid initial CSV row (missing Napari columns): " << line << '\n';
                continue;
            }

            std::string filePath = !firstFrameFile.empty() ? firstFrameFile : std::string("t000.tif");
            float z = std::stof(zText);
            float y = std::stof(yText);
            float x = std::stof(xText);

            z *= z_scaling;

            const float defaultMajorRadius = 10.0f * initialRadiusScale;
            const float defaultMinorRadius = 10.0f * initialRadiusScale;
            const std::string cellName = cellType + "_" + std::to_string(line_cnt + 1);

            EllipsoidParams params(cellName, x, y, z, defaultMajorRadius, defaultMinorRadius,
                                  0.0f, 0.0f, 0.0f, initialBrightness);
            params.bRadius = defaultMajorRadius;
            initialCells[filePath].push_back(Ellipsoid(params));

            ++line_cnt;
            continue;
        }

    // ----------------------------
    // Case A: Original 7-column format:
    // filePath, cellName, x, y, z, aRadius, cRadius
    //
    // Triaxial extension (2026-04-11): optional 8-column format adds bRadius
    // between aRadius and cRadius:
    // filePath, cellName, x, y, z, aRadius, bRadius, cRadius
    // If absent, bRadius defaults to aRadius (oblate-compatible fallback).
    // ----------------------------
    if (tokens.size() >= 7) {
        std::string filePath = tokens[0];
        std::string cellName = tokens[1];

        float x = std::stof(tokens[2]);
        float y = std::stof(tokens[3]);
        float z = std::stof(tokens[4]);
        float aRadius = std::stof(tokens[5]) * initialRadiusScale;
        float bRadius;
        float cRadius;
        if (tokens.size() >= 8) {
            bRadius     = std::stof(tokens[6]) * initialRadiusScale;
            cRadius = std::stof(tokens[7]) * initialRadiusScale;
        } else {
            bRadius     = aRadius; // oblate fallback
            cRadius = std::stof(tokens[6]) * initialRadiusScale;
        }

        float brightness = initialBrightness;

        z *= z_scaling;

        EllipsoidParams params(cellName, x, y, z, aRadius, cRadius,
                              0.0f, 0.0f, 0.0f, brightness);
        params.bRadius = bRadius;
        initialCells[filePath].push_back(Ellipsoid(params));

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
        const float defaultMajorRadius = 10.0f * initialRadiusScale;
        const float defaultMinorRadius = 10.0f * initialRadiusScale;

        // Generate a stable name
        const std::string cellName = cellType + "_" + std::to_string(line_cnt + 1);

        EllipsoidParams params(cellName, x, y, z, defaultMajorRadius, defaultMinorRadius,
                              0.0f, 0.0f, 0.0f, initialBrightness);
        params.bRadius = defaultMajorRadius; // oblate fallback for Napari format
        initialCells[filePath].push_back(Ellipsoid(params));

        ++line_cnt;
        continue;
    }

	    // Unknown/invalid line
	    std::cerr << "[WARN] Skipping invalid initial CSV row (expected 7 or 4 columns): " << line << '\n';
    } catch (const std::exception &e) {
        std::cerr << "[WARN] Skipping invalid initial CSV row (" << e.what() << "): " << line << '\n';
    }
	}

    std::cout << "Input Line Count : " << line_cnt << '\n';
    std::cout << "Initial frame keys loaded : " << initialCells.size() << '\n';
    if (!initialCells.empty()) {
        const auto& firstKey = initialCells.begin()->first;
        std::cout << "Example key : " << firstKey << "  cell count : " << initialCells.begin()->second.size() << '\n';
    }
    return initialCells;
}
