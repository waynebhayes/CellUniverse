#include "CsvHandler.hpp"

#include <cctype>
#include <fstream>
#include <iostream>
#include <stdexcept>
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

bool parseBoolText(const std::string &value)
{
    const std::string normalized = normalizeHeaderName(value);
    return normalized == "1" ||
           normalized == "true" ||
           normalized == "yes" ||
           normalized == "y" ||
           normalized == "trash";
}

void parseOptionalTrashColumn(const std::vector<std::string> &tokens,
                              const std::unordered_map<std::string, size_t> &headerIndex,
                              InitialCellRecord &record)
{
    size_t trashCol = 0;
    if (!findColumn(headerIndex, {"isTrash", "trash", "is_trash"}, trashCol)) {
        record.isTrash = false;
        return;
    }

    std::string trashText;
    record.isTrash = getToken(tokens, trashCol, trashText) && parseBoolText(trashText);
}

float parseScaledFloat(const std::string &value, float scale)
{
    return std::stof(value) * scale;
}

bool parseNamedInitialCell(const std::vector<std::string> &tokens,
                           const std::unordered_map<std::string, size_t> &headerIndex,
                           float zScaling,
                           float initialRadiusScale,
                           InitialCellRecord &record)
{
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

    if (!hasNamedCellColumns) {
        return false;
    }

    std::string xText;
    std::string yText;
    std::string zText;
    std::string aRadiusText;
    std::string cRadiusText;
    if (!getToken(tokens, filePathCol, record.filePath) ||
        !getToken(tokens, cellNameCol, record.cellName) ||
        !getToken(tokens, xCol, xText) ||
        !getToken(tokens, yCol, yText) ||
        !getToken(tokens, zCol, zText) ||
        !getToken(tokens, aRadiusCol, aRadiusText) ||
        !getToken(tokens, cRadiusCol, cRadiusText)) {
        throw std::runtime_error("missing named columns");
    }

    record.x = std::stof(xText);
    record.y = std::stof(yText);
    record.z = std::stof(zText) * zScaling;
    record.aRadius = parseScaledFloat(aRadiusText, initialRadiusScale);
    record.bRadius = record.aRadius;
    if (findColumn(headerIndex, {"bRadius", "radiusB", "middleRadius", "intermediateRadius"}, bRadiusCol)) {
        std::string bRadiusText;
        if (getToken(tokens, bRadiusCol, bRadiusText)) {
            record.bRadius = parseScaledFloat(bRadiusText, initialRadiusScale);
        }
    }
    record.cRadius = parseScaledFloat(cRadiusText, initialRadiusScale);
    parseOptionalTrashColumn(tokens, headerIndex, record);
    return true;
}

bool parseNamedNapariCell(const std::vector<std::string> &tokens,
                          const std::unordered_map<std::string, size_t> &headerIndex,
                          const std::string &firstFrameFile,
                          float zScaling,
                          float initialRadiusScale,
                          int lineCount,
                          InitialCellRecord &record)
{
    size_t cellTypeCol = 0;
    size_t xCol = 0;
    size_t yCol = 0;
    size_t zCol = 0;
    const bool hasNamedNapariColumns =
        findColumn(headerIndex, {"cell_type", "cellType", "type"}, cellTypeCol) &&
        findColumn(headerIndex, {"z"}, zCol) &&
        findColumn(headerIndex, {"y"}, yCol) &&
        findColumn(headerIndex, {"x"}, xCol);

    if (!hasNamedNapariColumns) {
        return false;
    }

    std::string cellType;
    std::string xText;
    std::string yText;
    std::string zText;
    if (!getToken(tokens, cellTypeCol, cellType) ||
        !getToken(tokens, xCol, xText) ||
        !getToken(tokens, yCol, yText) ||
        !getToken(tokens, zCol, zText)) {
        throw std::runtime_error("missing Napari columns");
    }

    record.filePath = !firstFrameFile.empty() ? firstFrameFile : std::string("t000.tif");
    record.cellName = cellType + "_" + std::to_string(lineCount + 1);
    record.x = std::stof(xText);
    record.y = std::stof(yText);
    record.z = std::stof(zText) * zScaling;
    record.aRadius = 10.0f * initialRadiusScale;
    record.bRadius = record.aRadius;
    record.cRadius = 10.0f * initialRadiusScale;
    parseOptionalTrashColumn(tokens, headerIndex, record);
    return true;
}

bool parsePositionalCell(const std::vector<std::string> &tokens,
                         float zScaling,
                         float initialRadiusScale,
                         InitialCellRecord &record)
{
    if (tokens.size() < 7) {
        return false;
    }

    record.filePath = tokens[0];
    record.cellName = tokens[1];
    record.x = std::stof(tokens[2]);
    record.y = std::stof(tokens[3]);
    record.z = std::stof(tokens[4]) * zScaling;
    record.aRadius = parseScaledFloat(tokens[5], initialRadiusScale);
    record.bRadius = record.aRadius;
    if (tokens.size() >= 8) {
        const std::string normalizedB = normalizeHeaderName(tokens[6]);
        if (!tokens[6].empty() && normalizedB != "none" && normalizedB != "null" && normalizedB != "nan") {
            record.bRadius = parseScaledFloat(tokens[6], initialRadiusScale);
        }
        record.cRadius = parseScaledFloat(tokens[7], initialRadiusScale);
    } else {
        record.cRadius = parseScaledFloat(tokens[6], initialRadiusScale);
    }
    return true;
}

bool parsePositionalNapariCell(const std::vector<std::string> &tokens,
                               const std::string &firstFrameFile,
                               float zScaling,
                               float initialRadiusScale,
                               int lineCount,
                               InitialCellRecord &record)
{
    if (tokens.size() != 4) {
        return false;
    }

    record.filePath = !firstFrameFile.empty() ? firstFrameFile : std::string("t000.tif");
    record.cellName = tokens[0] + "_" + std::to_string(lineCount + 1);
    record.z = std::stof(tokens[1]) * zScaling;
    record.y = std::stof(tokens[2]);
    record.x = std::stof(tokens[3]);
    record.aRadius = 10.0f * initialRadiusScale;
    record.bRadius = record.aRadius;
    record.cRadius = 10.0f * initialRadiusScale;
    return true;
}
}

std::vector<InitialCellRecord> CsvHandler::loadInitialCells(const Path &initParamsPath,
                                                            const std::string &firstFrameFile,
                                                            float zScaling,
                                                            float initialRadiusScale,
                                                            const std::string &initialZSpace)
{
    std::ifstream file(initParamsPath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open initial CSV: " + initParamsPath);
    }

    std::string firstLine;
    std::getline(file, firstLine);
    const bool resumeStateCsv =
        firstLine.find("theta_x") != std::string::npos &&
        firstLine.find("theta_y") != std::string::npos &&
        firstLine.find("theta_z") != std::string::npos;
    const bool zAlreadyScaled =
        (initialZSpace == "scaled") ||
        (initialZSpace == "auto" && resumeStateCsv);
    const float effectiveZScaling = zAlreadyScaled ? 1.0f : zScaling;
    const std::unordered_map<std::string, size_t> headerIndex = buildHeaderIndex(splitCsvLine(firstLine));

    std::vector<InitialCellRecord> records;
    std::string line;
    int lineCount = 0;
    while (std::getline(file, line)) {
        if (trim(line).empty()) {
            continue;
        }

        const std::vector<std::string> tokens = splitCsvLine(line);
        try {
            InitialCellRecord record;
            if (parseNamedInitialCell(tokens, headerIndex, effectiveZScaling, initialRadiusScale, record) ||
                parseNamedNapariCell(tokens, headerIndex, firstFrameFile, effectiveZScaling, initialRadiusScale, lineCount, record) ||
                parsePositionalCell(tokens, effectiveZScaling, initialRadiusScale, record) ||
                parsePositionalNapariCell(tokens, firstFrameFile, effectiveZScaling, initialRadiusScale, lineCount, record)) {
                records.push_back(record);
                ++lineCount;
                continue;
            }

            throw std::runtime_error(
                "Invalid initial CSV row at data line " + std::to_string(lineCount + 1) +
                " (expected named, 7/8-column, or Napari format): " + line);
        } catch (const std::exception &e) {
            throw std::runtime_error(
                "Invalid initial CSV row at data line " + std::to_string(lineCount + 1) +
                " (" + e.what() + "): " + line);
        }
    }

    return records;
}
