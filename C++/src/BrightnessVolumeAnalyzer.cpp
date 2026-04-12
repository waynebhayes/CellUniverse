#include "EmbryoBrightTracker.hpp"
#include "yaml-cpp/yaml.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Args
{
    int firstFrame = 0;
    int lastFrame = -1;
    std::string input;
    std::string output;
    std::string config;
    std::string initial;
};

struct Observation
{
    int frameIndex = 0;
    std::string frameName;
    std::string cellId;
    std::string parentId;
    std::string source;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float diameter = 0.0f;
    int voxelCount = 0;
    float meanIntensity = 0.0f;
    float integratedIntensity = 0.0f;
    float thresholdHigh = 0.0f;
    float thresholdLow = 0.0f;
    float thresholdSplitLow = 0.0f;
};

struct SplitRecord
{
    int frameIndex = 0;
    std::string frameName;
    std::string parentId;
    std::string child1Id;
    std::string child2Id;
};

struct RegressionSummary
{
    std::size_t count = 0;
    double volumeMean = std::numeric_limits<double>::quiet_NaN();
    double volumeMin = std::numeric_limits<double>::quiet_NaN();
    double volumeMax = std::numeric_limits<double>::quiet_NaN();
    double brightnessMean = std::numeric_limits<double>::quiet_NaN();
    double brightnessMin = std::numeric_limits<double>::quiet_NaN();
    double brightnessMax = std::numeric_limits<double>::quiet_NaN();
    double integratedMean = std::numeric_limits<double>::quiet_NaN();
    double pearsonVolumeVsBrightness = std::numeric_limits<double>::quiet_NaN();
    double pearsonVolumeVsIntegrated = std::numeric_limits<double>::quiet_NaN();
    double slopeBrightnessPerVolume = std::numeric_limits<double>::quiet_NaN();
    double slopeIntegratedPerVolume = std::numeric_limits<double>::quiet_NaN();
};

struct RegressionLine
{
    double slope = std::numeric_limits<double>::quiet_NaN();
    double intercept = std::numeric_limits<double>::quiet_NaN();
    bool valid = false;
};

static float clampf(float v, float lo, float hi)
{
    return std::max(lo, std::min(v, hi));
}

static float dist2_3d(const cv::Point3f& a, const cv::Point3f& b)
{
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

static float estimateDiameterFromVoxelCount(int voxelCount)
{
    if (voxelCount <= 0)
    {
        return 0.0f;
    }
    const float volume = static_cast<float>(voxelCount);
    const float radius = std::cbrt((3.0f * volume) / (4.0f * static_cast<float>(M_PI)));
    return 2.0f * radius;
}

static bool isTiffPath(const fs::path& path)
{
    const std::string ext = path.extension().string();
    return ext == ".tif" || ext == ".tiff";
}

static std::vector<std::string> splitCsvRow(const std::string& line)
{
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        while (!token.empty() && (token.front() == ' ' || token.front() == '\t'))
        {
            token.erase(token.begin());
        }
        while (!token.empty() && (token.back() == ' ' || token.back() == '\t' || token.back() == '\r'))
        {
            token.pop_back();
        }
        tokens.push_back(token);
    }
    return tokens;
}

static void loadConfig(const std::string& path, BaseConfig& config)
{
    const YAML::Node node = YAML::LoadFile(path);
    config.explodeConfig(node);
}

static void updateTiffConfigIfNeeded(const fs::path& file, BaseConfig& config)
{
    if (!isTiffPath(file))
    {
        return;
    }

    std::vector<cv::Mat> images;
    if (!cv::imreadmulti(file.string(), images, cv::IMREAD_UNCHANGED))
    {
        return;
    }
    config.simulation.z_slices = static_cast<int>(images.size());
}

static bool naturalStringLess(const std::string& lhs, const std::string& rhs)
{
    std::size_t i = 0;
    std::size_t j = 0;

    while (i < lhs.size() && j < rhs.size())
    {
        const unsigned char cl = static_cast<unsigned char>(lhs[i]);
        const unsigned char cr = static_cast<unsigned char>(rhs[j]);

        if (std::isdigit(cl) && std::isdigit(cr))
        {
            std::size_t iEnd = i;
            std::size_t jEnd = j;
            while (iEnd < lhs.size() && std::isdigit(static_cast<unsigned char>(lhs[iEnd])))
            {
                ++iEnd;
            }
            while (jEnd < rhs.size() && std::isdigit(static_cast<unsigned char>(rhs[jEnd])))
            {
                ++jEnd;
            }

            std::string ln = lhs.substr(i, iEnd - i);
            std::string rn = rhs.substr(j, jEnd - j);

            const auto lnNonZero = ln.find_first_not_of('0');
            const auto rnNonZero = rn.find_first_not_of('0');
            const std::string lnTrimmed = (lnNonZero == std::string::npos) ? "0" : ln.substr(lnNonZero);
            const std::string rnTrimmed = (rnNonZero == std::string::npos) ? "0" : rn.substr(rnNonZero);

            if (lnTrimmed.size() != rnTrimmed.size())
            {
                return lnTrimmed.size() < rnTrimmed.size();
            }
            if (lnTrimmed != rnTrimmed)
            {
                return lnTrimmed < rnTrimmed;
            }
            if (ln.size() != rn.size())
            {
                return ln.size() < rn.size();
            }

            i = iEnd;
            j = jEnd;
            continue;
        }

        const char l = static_cast<char>(std::tolower(cl));
        const char r = static_cast<char>(std::tolower(cr));
        if (l != r)
        {
            return l < r;
        }

        ++i;
        ++j;
    }

    return lhs.size() < rhs.size();
}

static std::vector<fs::path> getImageFilePaths(
    const std::string& input,
    int firstFrame,
    int lastFrame,
    BaseConfig& config)
{
    std::vector<fs::path> imagePaths;

    if (input.find('%') != std::string::npos)
    {
        for (int frame = firstFrame; ; ++frame)
        {
            char buffer[1024];
            std::snprintf(buffer, sizeof(buffer), input.c_str(), frame);
            fs::path candidate(buffer);

            if (!fs::exists(candidate) || !fs::is_regular_file(candidate))
            {
                if (lastFrame >= 0)
                {
                    throw std::runtime_error("Input file not found: " + candidate.string());
                }
                break;
            }

            imagePaths.push_back(candidate);
            if (lastFrame >= 0 && frame >= lastFrame)
            {
                break;
            }
        }
    }
    else if (fs::is_directory(input))
    {
        std::vector<fs::path> allFiles;
        for (const auto& entry : fs::directory_iterator(input))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }
            if (isTiffPath(entry.path()))
            {
                allFiles.push_back(entry.path());
            }
        }

        if (allFiles.empty())
        {
            throw std::runtime_error("No .tif/.tiff files found in directory: " + input);
        }

        std::sort(allFiles.begin(), allFiles.end(), [](const fs::path& a, const fs::path& b) {
            return naturalStringLess(a.filename().string(), b.filename().string());
        });

        if (firstFrame < 0 || firstFrame >= static_cast<int>(allFiles.size()))
        {
            throw std::runtime_error("firstFrame is out of range for directory input");
        }

        const int end = (lastFrame < 0) ? static_cast<int>(allFiles.size()) - 1
                                        : std::min(lastFrame, static_cast<int>(allFiles.size()) - 1);

        if (firstFrame > end)
        {
            throw std::runtime_error("Invalid frame range for directory input");
        }

        for (int i = firstFrame; i <= end; ++i)
        {
            imagePaths.push_back(allFiles[i]);
        }
    }
    else if (fs::exists(input) && fs::is_regular_file(input))
    {
        imagePaths.push_back(fs::path(input));
    }
    else
    {
        throw std::runtime_error("Input is neither a pattern, directory, nor file: " + input);
    }

    if (imagePaths.empty())
    {
        throw std::runtime_error("No input frames selected.");
    }

    updateTiffConfigIfNeeded(imagePaths.front(), config);
    return imagePaths;
}

static double pearsonCorrelation(const std::vector<double>& x, const std::vector<double>& y)
{
    if (x.size() != y.size() || x.size() < 2)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double sumX = 0.0;
    double sumY = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        sumX += x[i];
        sumY += y[i];
    }
    const double meanX = sumX / static_cast<double>(x.size());
    const double meanY = sumY / static_cast<double>(y.size());

    double cov = 0.0;
    double varX = 0.0;
    double varY = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        const double dx = x[i] - meanX;
        const double dy = y[i] - meanY;
        cov += dx * dy;
        varX += dx * dx;
        varY += dy * dy;
    }

    if (varX <= 1e-12 || varY <= 1e-12)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return cov / std::sqrt(varX * varY);
}

static double slopeYOnX(const std::vector<double>& x, const std::vector<double>& y)
{
    if (x.size() != y.size() || x.size() < 2)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double sumX = 0.0;
    double sumY = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        sumX += x[i];
        sumY += y[i];
    }
    const double meanX = sumX / static_cast<double>(x.size());
    const double meanY = sumY / static_cast<double>(y.size());

    double cov = 0.0;
    double varX = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        const double dx = x[i] - meanX;
        cov += dx * (y[i] - meanY);
        varX += dx * dx;
    }

    if (varX <= 1e-12)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return cov / varX;
}

static RegressionLine fitLineYOnX(const std::vector<double>& x, const std::vector<double>& y)
{
    RegressionLine line;
    if (x.size() != y.size() || x.size() < 2)
    {
        return line;
    }

    double sumX = 0.0;
    double sumY = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        sumX += x[i];
        sumY += y[i];
    }
    const double meanX = sumX / static_cast<double>(x.size());
    const double meanY = sumY / static_cast<double>(y.size());

    line.slope = slopeYOnX(x, y);
    if (!std::isfinite(line.slope))
    {
        return line;
    }

    line.intercept = meanY - line.slope * meanX;
    line.valid = std::isfinite(line.intercept);
    return line;
}

static std::string formatDouble(double value, int precision = 4)
{
    if (!std::isfinite(value))
    {
        return "nan";
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

static std::string insertThousandsSeparators(const std::string& digits)
{
    if (digits.empty())
    {
        return digits;
    }

    std::string sign;
    std::string work = digits;
    if (!work.empty() && (work.front() == '-' || work.front() == '+'))
    {
        sign.push_back(work.front());
        work.erase(work.begin());
    }

    const std::size_t dotPos = work.find('.');
    std::string integerPart = (dotPos == std::string::npos) ? work : work.substr(0, dotPos);
    const std::string fractionalPart = (dotPos == std::string::npos) ? "" : work.substr(dotPos);

    std::string withCommas;
    withCommas.reserve(integerPart.size() + integerPart.size() / 3);
    for (std::size_t i = 0; i < integerPart.size(); ++i)
    {
        const std::size_t remaining = integerPart.size() - i;
        withCommas.push_back(integerPart[i]);
        if (remaining > 1 && remaining % 3 == 1)
        {
            withCommas.push_back(',');
        }
    }

    return sign + withCommas + fractionalPart;
}

static std::string formatAxisNumber(double value, int precision, bool useThousandsSeparators)
{
    const std::string formatted = formatDouble(value, precision);
    return useThousandsSeparators ? insertThousandsSeparators(formatted) : formatted;
}

static std::string shortenCellId(const std::string& cellId, std::size_t head = 8, std::size_t tail = 6)
{
    if (cellId.size() <= head + tail + 3)
    {
        return cellId;
    }
    return cellId.substr(0, head) + "..." + cellId.substr(cellId.size() - tail);
}

static void putTextBlock(
    cv::Mat& canvas,
    const std::vector<std::string>& lines,
    const cv::Point& origin,
    double fontScale,
    const cv::Scalar& color,
    int thickness,
    int lineStep)
{
    int y = origin.y;
    for (const auto& line : lines)
    {
        cv::putText(canvas, line, cv::Point(origin.x, y), cv::FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv::LINE_AA);
        y += lineStep;
    }
}

static RegressionSummary summarizeObservations(const std::vector<Observation>& observations)
{
    RegressionSummary summary;
    summary.count = observations.size();
    if (observations.empty())
    {
        return summary;
    }

    std::vector<double> volumes;
    std::vector<double> brightness;
    std::vector<double> integrated;
    volumes.reserve(observations.size());
    brightness.reserve(observations.size());
    integrated.reserve(observations.size());

    double volumeSum = 0.0;
    double brightnessSum = 0.0;
    double integratedSum = 0.0;

    summary.volumeMin = std::numeric_limits<double>::infinity();
    summary.volumeMax = -std::numeric_limits<double>::infinity();
    summary.brightnessMin = std::numeric_limits<double>::infinity();
    summary.brightnessMax = -std::numeric_limits<double>::infinity();

    for (const auto& obs : observations)
    {
        const double volume = static_cast<double>(obs.voxelCount);
        const double bright = static_cast<double>(obs.meanIntensity);
        const double integ = static_cast<double>(obs.integratedIntensity);

        volumes.push_back(volume);
        brightness.push_back(bright);
        integrated.push_back(integ);

        volumeSum += volume;
        brightnessSum += bright;
        integratedSum += integ;

        summary.volumeMin = std::min(summary.volumeMin, volume);
        summary.volumeMax = std::max(summary.volumeMax, volume);
        summary.brightnessMin = std::min(summary.brightnessMin, bright);
        summary.brightnessMax = std::max(summary.brightnessMax, bright);
    }

    const double n = static_cast<double>(observations.size());
    summary.volumeMean = volumeSum / n;
    summary.brightnessMean = brightnessSum / n;
    summary.integratedMean = integratedSum / n;
    summary.pearsonVolumeVsBrightness = pearsonCorrelation(volumes, brightness);
    summary.pearsonVolumeVsIntegrated = pearsonCorrelation(volumes, integrated);
    summary.slopeBrightnessPerVolume = slopeYOnX(volumes, brightness);
    summary.slopeIntegratedPerVolume = slopeYOnX(volumes, integrated);
    return summary;
}

static float pickTrackingBoxDiameter(
    const EmbryoBrightTracker::CellState& prev,
    const EmbryoBrightTracker::CellState& cur,
    float matureDiamAvg)
{
    float boxDiam = std::max(prev.diameter, cur.diameter);
    if (matureDiamAvg > 1e-3f)
    {
        boxDiam = std::max(boxDiam, 0.85f * matureDiamAvg);
    }
    return clampf(boxDiam, 20.0f, 120.0f);
}

static void pushUniqueTracked(
    std::vector<EmbryoBrightTracker::CellState>& nextCells,
    std::unordered_map<std::string, std::string>& sourceById,
    const EmbryoBrightTracker::CellState& candidate,
    const std::string& source)
{
    const float boxDiam = clampf(candidate.diameter, 20.0f, 120.0f);
    const float mergeR2 = (0.25f * boxDiam) * (0.25f * boxDiam);

    for (auto& existing : nextCells)
    {
        if (!existing.alive)
        {
            continue;
        }
        if (dist2_3d(existing.center, candidate.center) <= mergeR2)
        {
            if (candidate.voxelCount > existing.voxelCount)
            {
                sourceById.erase(existing.id);
                existing = candidate;
                sourceById[candidate.id] = source;
            }
            return;
        }
    }

    nextCells.push_back(candidate);
    sourceById[candidate.id] = source;
}

static bool detectSplitFromComponents(
    int frameIndex,
    const EmbryoBrightTracker::CellState& prev,
    const std::vector<EmbryoBrightTracker::CellState>& currentCells,
    const std::vector<EmbryoBrightTracker::Comp3DStat>& compsForSplit,
    float matureDiamAvg,
    EmbryoBrightTracker::CellState& outC1,
    EmbryoBrightTracker::CellState& outC2)
{
    if (compsForSplit.size() < 2)
    {
        return false;
    }

    int maxVox = 0;
    int sumVox = 0;
    for (const auto& comp : compsForSplit)
    {
        maxVox = std::max(maxVox, comp.vox);
        sumVox += comp.vox;
    }

    int minDaughterVox = 200;
    if (maxVox > 0)
    {
        minDaughterVox = std::max(minDaughterVox, static_cast<int>(std::lround(maxVox * 0.22)));
    }
    if (sumVox > 0)
    {
        minDaughterVox = std::min(minDaughterVox, static_cast<int>(std::lround(sumVox * 0.12)));
    }
    minDaughterVox = std::max(minDaughterVox, 200);

    std::vector<EmbryoBrightTracker::Comp3DStat> bigComps;
    bigComps.reserve(compsForSplit.size());
    for (const auto& comp : compsForSplit)
    {
        if (comp.vox >= minDaughterVox)
        {
            bigComps.push_back(comp);
        }
    }

    if (bigComps.size() < 2)
    {
        return false;
    }

    std::sort(bigComps.begin(), bigComps.end(), [](const auto& a, const auto& b) {
        return a.vox > b.vox;
    });

    const cv::Point3f p1 = bigComps[0].center();
    const cv::Point3f p2 = bigComps[1].center();
    const float sep2 = dist2_3d(p1, p2);

    const float scale = (matureDiamAvg > 1e-3f) ? matureDiamAvg : std::max(prev.diameter, 20.0f);
    const float nearR2 = scale * scale;
    const bool near1 = dist2_3d(p1, prev.center) <= nearR2;
    const bool near2 = dist2_3d(p2, prev.center) <= nearR2;

    const float boxDiam = clampf(prev.diameter, 30.0f, 90.0f);
    const float avoidR2 = (0.55f * boxDiam) * (0.55f * boxDiam);
    const float minSep = 0.30f * boxDiam;

    auto minDist2ToOther = [&](const cv::Point3f& point) -> float {
        float best = std::numeric_limits<float>::max();
        for (const auto& other : currentCells)
        {
            if (!other.alive || other.id == prev.id)
            {
                continue;
            }
            best = std::min(best, dist2_3d(point, other.center));
        }
        return best;
    };

    const bool farFromOthers1 = (minDist2ToOther(p1) >= avoidR2);
    const bool farFromOthers2 = (minDist2ToOther(p2) >= avoidR2);
    const bool separated = sep2 >= (minSep * minSep);

    if (!(near1 && near2 && separated && farFromOthers1 && farFromOthers2))
    {
        return false;
    }

    outC1 = prev;
    outC2 = prev;
    outC1.id = prev.id + ".1";
    outC2.id = prev.id + ".2";
    outC1.parentId = prev.id;
    outC2.parentId = prev.id;
    outC1.center = p1;
    outC2.center = p2;
    outC1.voxelCount = bigComps[0].vox;
    outC2.voxelCount = bigComps[1].vox;
    outC1.meanIntensity = bigComps[0].meanI();
    outC2.meanIntensity = bigComps[1].meanI();
    outC1.diameter = estimateDiameterFromVoxelCount(outC1.voxelCount);
    outC2.diameter = estimateDiameterFromVoxelCount(outC2.voxelCount);
    outC1.alive = true;
    outC2.alive = true;
    outC1.lastSeenFrame = frameIndex;
    outC2.lastSeenFrame = frameIndex;
    outC1.lastSplitFrame = frameIndex;
    outC2.lastSplitFrame = frameIndex;
    return true;
}

static void writeObservationsCsv(const fs::path& path, const std::vector<Observation>& observations)
{
    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("Failed to open output file: " + path.string());
    }

    out << std::fixed << std::setprecision(6);
    out << "frame_index,frame_name,cell_id,parent_id,source,x,y,z,diameter,volume_voxels,mean_intensity,integrated_intensity,threshold_high,threshold_low,threshold_split_low\n";
    for (const auto& obs : observations)
    {
        out << obs.frameIndex << ','
            << obs.frameName << ','
            << obs.cellId << ','
            << obs.parentId << ','
            << obs.source << ','
            << obs.x << ','
            << obs.y << ','
            << obs.z << ','
            << obs.diameter << ','
            << obs.voxelCount << ','
            << obs.meanIntensity << ','
            << obs.integratedIntensity << ','
            << obs.thresholdHigh << ','
            << obs.thresholdLow << ','
            << obs.thresholdSplitLow << '\n';
    }
}

static void writeSplitEventsCsv(const fs::path& path, const std::vector<SplitRecord>& splitEvents)
{
    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("Failed to open output file: " + path.string());
    }

    out << "frame_index,frame_name,parent_id,child1_id,child2_id\n";
    for (const auto& event : splitEvents)
    {
        out << event.frameIndex << ','
            << event.frameName << ','
            << event.parentId << ','
            << event.child1Id << ','
            << event.child2Id << '\n';
    }
}

static void writeSummaryCsv(const fs::path& path, const std::vector<Observation>& observations)
{
    std::map<std::string, std::vector<Observation>> grouped;
    for (const auto& obs : observations)
    {
        grouped[obs.cellId].push_back(obs);
    }

    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("Failed to open output file: " + path.string());
    }

    out << std::fixed << std::setprecision(6);
    out << "cell_id,parent_id,num_observations,first_frame,last_frame,volume_mean,volume_min,volume_max,mean_intensity_mean,mean_intensity_min,mean_intensity_max,integrated_intensity_mean,pearson_r_volume_vs_mean_intensity,slope_mean_intensity_per_volume,pearson_r_volume_vs_integrated_intensity,slope_integrated_intensity_per_volume\n";

    for (const auto& [cellId, rows] : grouped)
    {
        const RegressionSummary summary = summarizeObservations(rows);
        int firstFrame = rows.front().frameIndex;
        int lastFrame = rows.front().frameIndex;
        std::string parentId = rows.front().parentId;
        for (const auto& row : rows)
        {
            firstFrame = std::min(firstFrame, row.frameIndex);
            lastFrame = std::max(lastFrame, row.frameIndex);
            if (parentId.empty() && !row.parentId.empty())
            {
                parentId = row.parentId;
            }
        }

        out << cellId << ','
            << parentId << ','
            << summary.count << ','
            << firstFrame << ','
            << lastFrame << ','
            << summary.volumeMean << ','
            << summary.volumeMin << ','
            << summary.volumeMax << ','
            << summary.brightnessMean << ','
            << summary.brightnessMin << ','
            << summary.brightnessMax << ','
            << summary.integratedMean << ','
            << summary.pearsonVolumeVsBrightness << ','
            << summary.slopeBrightnessPerVolume << ','
            << summary.pearsonVolumeVsIntegrated << ','
            << summary.slopeIntegratedPerVolume << '\n';
    }

    const RegressionSummary global = summarizeObservations(observations);
    out << "__ALL__,,"
        << global.count << ",,,"
        << global.volumeMean << ','
        << global.volumeMin << ','
        << global.volumeMax << ','
        << global.brightnessMean << ','
        << global.brightnessMin << ','
        << global.brightnessMax << ','
        << global.integratedMean << ','
        << global.pearsonVolumeVsBrightness << ','
        << global.slopeBrightnessPerVolume << ','
        << global.pearsonVolumeVsIntegrated << ','
        << global.slopeIntegratedPerVolume << '\n';
}

static void writeReport(
    const fs::path& path,
    const Args& args,
    const std::vector<fs::path>& imagePaths,
    float matureDiamAvg,
    const std::vector<Observation>& observations,
    const std::vector<SplitRecord>& splitEvents)
{
    std::map<std::string, std::vector<Observation>> grouped;
    for (const auto& obs : observations)
    {
        grouped[obs.cellId].push_back(obs);
    }

    const RegressionSummary global = summarizeObservations(observations);

    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("Failed to open output file: " + path.string());
    }

    out << std::fixed << std::setprecision(6);
    out << "Brightness/Volume Analysis Report\n";
    out << "input=" << args.input << '\n';
    out << "config=" << args.config << '\n';
    out << "requested_frame_range=[" << args.firstFrame << "," << args.lastFrame << "]\n";
    out << "selected_frame_count=" << imagePaths.size() << '\n';
    out << "first_selected_frame=" << imagePaths.front().filename().string() << '\n';
    out << "last_selected_frame=" << imagePaths.back().filename().string() << '\n';
    out << "detected_cell_ids=" << grouped.size() << '\n';
    out << "split_events=" << splitEvents.size() << '\n';
    out << "mature_diameter_used=" << matureDiamAvg << '\n';
    out << '\n';
    out << "Global relation summary\n";
    out << "mean(volume_voxels)=" << global.volumeMean << '\n';
    out << "mean(mean_intensity)=" << global.brightnessMean << '\n';
    out << "pearson(volume, mean_intensity)=" << global.pearsonVolumeVsBrightness << '\n';
    out << "slope(mean_intensity vs volume)=" << global.slopeBrightnessPerVolume << '\n';
    out << "pearson(volume, integrated_intensity)=" << global.pearsonVolumeVsIntegrated << '\n';
    out << "slope(integrated_intensity vs volume)=" << global.slopeIntegratedPerVolume << '\n';
    out << '\n';
    out << "Metric definitions\n";
    out << "- volume_voxels = number of voxels in the tracked 3D connected component after thresholding.\n";
    out << "- mean_intensity = average normalized voxel intensity inside that component.\n";
    out << "- integrated_intensity = mean_intensity * volume_voxels, i.e. total normalized brightness inside the component.\n";
    out << "- Each TIFF frame is min-max normalized independently to [0,1], so brightness is relative within frame, not absolute raw fluorescence.\n";
    out << "- volume_voxels is a segmented bright-region volume, not a physical micron^3 measurement.\n";
    out << '\n';
    out << "How to read the outputs\n";
    out << "- per_cell_observations.csv = one row per cell per frame.\n";
    out << "- per_cell_summary.csv = one summary row per cell plus a global __ALL__ row.\n";
    out << "- volume_vs_mean_intensity_scatter.png = best plot for the question: when segmented cell volume changes, does mean brightness tend to move in the opposite direction?\n";
    out << "- volume_vs_integrated_intensity_scatter.png = total segmented brightness, which often increases with volume because more voxels are included.\n";
    out << "- per_cell_normalized_time_series.png = easiest plot to inspect trend shape cell by cell.\n";
    out << '\n';
    out << "Interpretation\n";
    out << "- Negative pearson(volume, mean_intensity) suggests larger cells tend to be dimmer.\n";
    out << "- Positive pearson(volume, mean_intensity) suggests larger cells tend to stay brighter.\n";
    out << "- If your new embryo only becomes clear around t013-t015, run this program with firstFrame=13 (or 14/15) so the first selected frame becomes the automatic initialization frame.\n";
    out << "- Generated plots: volume_vs_mean_intensity_scatter.png, volume_vs_integrated_intensity_scatter.png, per_cell_normalized_time_series.png\n";
}

static std::vector<EmbryoBrightTracker::CellState> loadInitialCellsFromCsv(
    const fs::path& csvPath,
    const std::string& firstFrameName,
    const std::vector<cv::Mat>& volume,
    EmbryoBrightTracker& tracker,
    float thresholdLow)
{
    std::ifstream in(csvPath);
    if (!in)
    {
        throw std::runtime_error("Failed to open initial CSV: " + csvPath.string());
    }

    std::string line;
    std::getline(in, line); // header

    std::vector<EmbryoBrightTracker::CellState> cells;
    const int Z = static_cast<int>(volume.size());
    const int Y = volume[0].rows;
    const int X = volume[0].cols;

    while (std::getline(in, line))
    {
        if (line.empty())
        {
            continue;
        }

        const std::vector<std::string> tokens = splitCsvRow(line);
        if (tokens.size() < 7)
        {
            continue;
        }

        if (!firstFrameName.empty() && !tokens[0].empty() && tokens[0] != firstFrameName)
        {
            continue;
        }

        EmbryoBrightTracker::CellState seed;
        seed.id = tokens[1];
        seed.parentId = "";
        seed.center = cv::Point3f(
            std::stof(tokens[2]),
            std::stof(tokens[3]),
            std::stof(tokens[4]));

        const float majorRadius = std::stof(tokens[5]);
        const float minorRadius = std::stof(tokens[6]);
        seed.diameter = 2.0f * std::max(majorRadius, minorRadius);
        seed.meanIntensity = 0.0f;
        if (tokens.size() >= 8 && !tokens[7].empty() && tokens[7] != "None")
        {
            seed.meanIntensity = std::stof(tokens[7]);
        }
        seed.alive = true;
        seed.lastSeenFrame = 0;
        seed.bbox = tracker.makeBBoxForAnalysis(
            seed.center,
            clampf(seed.diameter, 20.0f, 120.0f),
            Z, Y, X);

        bool ok = false;
        std::vector<EmbryoBrightTracker::Comp3DStat> comps;
        EmbryoBrightTracker::CellState refined =
            tracker.trackSingleCellByCCInBBoxForAnalysis(0, volume, seed, thresholdLow, ok, &comps);

        if (!ok || !refined.alive)
        {
            const float retryThreshold = std::max(0.02f, thresholdLow * 0.25f);
            refined = tracker.trackSingleCellByCCInBBoxForAnalysis(0, volume, seed, retryThreshold, ok, &comps);
        }

        if (ok && refined.alive)
        {
            refined.id = seed.id;
            refined.parentId = seed.parentId;
            if (refined.diameter <= 1e-3f)
            {
                refined.diameter = seed.diameter;
            }
            cells.push_back(refined);
        }
        else
        {
            cells.push_back(seed);
        }
    }

    if (cells.empty())
    {
        throw std::runtime_error(
            "No initial cells were loaded from CSV for frame " + firstFrameName + ": " + csvPath.string());
    }
    return cells;
}

static std::vector<cv::Scalar> plotPalette()
{
    return {
        cv::Scalar(231, 76, 60),
        cv::Scalar(52, 152, 219),
        cv::Scalar(46, 204, 113),
        cv::Scalar(241, 196, 15),
        cv::Scalar(155, 89, 182),
        cv::Scalar(230, 126, 34),
        cv::Scalar(26, 188, 156),
        cv::Scalar(52, 73, 94),
        cv::Scalar(127, 140, 141),
        cv::Scalar(192, 57, 43)
    };
}

static void drawAxes(
    cv::Mat& canvas,
    const cv::Rect& plotRect,
    double xMin,
    double xMax,
    double yMin,
    double yMax,
    const std::string& xLabel,
    const std::string& yLabel,
    int xPrecision,
    int yPrecision,
    bool xThousandsSeparators,
    bool yThousandsSeparators)
{
    cv::rectangle(canvas, plotRect, cv::Scalar(210, 210, 210), 1, cv::LINE_AA);

    for (int i = 0; i <= 5; ++i)
    {
        const double xt = xMin + (xMax - xMin) * static_cast<double>(i) / 5.0;
        const int px = plotRect.x + static_cast<int>(std::lround(plotRect.width * static_cast<double>(i) / 5.0));
        cv::line(canvas, cv::Point(px, plotRect.y), cv::Point(px, plotRect.y + plotRect.height), cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
        const std::string xTickLabel = formatAxisNumber(xt, xPrecision, xThousandsSeparators);
        cv::putText(canvas, xTickLabel, cv::Point(px - 38, plotRect.y + plotRect.height + 28),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);

        const double yt = yMin + (yMax - yMin) * static_cast<double>(5 - i) / 5.0;
        const int py = plotRect.y + static_cast<int>(std::lround(plotRect.height * static_cast<double>(i) / 5.0));
        cv::line(canvas, cv::Point(plotRect.x, py), cv::Point(plotRect.x + plotRect.width, py), cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
        const std::string yTickLabel = formatAxisNumber(yt, yPrecision, yThousandsSeparators);
        cv::putText(canvas, yTickLabel, cv::Point(plotRect.x - 96, py + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
    }

    int baseline = 0;
    const cv::Size xSize = cv::getTextSize(xLabel, cv::FONT_HERSHEY_SIMPLEX, 0.72, 2, &baseline);
    cv::putText(canvas, xLabel, cv::Point(plotRect.x + (plotRect.width - xSize.width) / 2, plotRect.y + plotRect.height + 72),
                cv::FONT_HERSHEY_SIMPLEX, 0.72, cv::Scalar(40, 40, 40), 2, cv::LINE_AA);
    cv::putText(canvas, yLabel, cv::Point(plotRect.x, plotRect.y - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.72, cv::Scalar(40, 40, 40), 2, cv::LINE_AA);
}

static void drawScatterPlot(
    const fs::path& path,
    const std::vector<Observation>& observations,
    bool useIntegratedIntensity)
{
    if (observations.empty())
    {
        return;
    }

    std::map<std::string, std::vector<Observation>> grouped;
    for (const auto& obs : observations)
    {
        grouped[obs.cellId].push_back(obs);
    }

    std::vector<double> xs;
    std::vector<double> ys;
    xs.reserve(observations.size());
    ys.reserve(observations.size());

    double xMin = std::numeric_limits<double>::infinity();
    double xMax = -std::numeric_limits<double>::infinity();
    double yMin = std::numeric_limits<double>::infinity();
    double yMax = -std::numeric_limits<double>::infinity();

    for (const auto& obs : observations)
    {
        const double x = static_cast<double>(obs.voxelCount);
        const double y = useIntegratedIntensity
            ? static_cast<double>(obs.integratedIntensity)
            : static_cast<double>(obs.meanIntensity);

        xs.push_back(x);
        ys.push_back(y);
        xMin = std::min(xMin, x);
        xMax = std::max(xMax, x);
        yMin = std::min(yMin, y);
        yMax = std::max(yMax, y);
    }

    if (std::abs(xMax - xMin) <= 1e-9)
    {
        xMin -= 1.0;
        xMax += 1.0;
    }
    if (std::abs(yMax - yMin) <= 1e-9)
    {
        yMin -= 1.0;
        yMax += 1.0;
    }

    const double xPad = 0.08 * (xMax - xMin);
    const double yPad = 0.10 * (yMax - yMin);
    xMin = std::max(0.0, xMin - xPad);
    xMax += xPad;
    yMin = std::max(0.0, yMin - yPad);
    yMax += yPad;
    if (!useIntegratedIntensity)
    {
        yMax = std::min(1.0, std::max(yMax, 0.5));
    }

    cv::Mat canvas(1220, 1820, CV_8UC3, cv::Scalar(250, 250, 250));
    const cv::Rect plotRect(140, 220, 1020, 780);
    const cv::Rect legendRect(1210, 220, 560, 820);

    const std::string title = useIntegratedIntensity
        ? "Segmented Cell Volume vs Integrated Normalized Brightness"
        : "Segmented Cell Volume vs Mean Normalized Brightness";
    const std::vector<std::string> subtitleLines = useIntegratedIntensity
        ? std::vector<std::string>{
            "X axis = segmented cell volume: number of voxels inside the tracked 3D connected component.",
            "Y axis = integrated normalized brightness: sum of normalized voxel intensities in that same component.",
            "Each point = one cell in one frame. Each TIFF frame is min-max normalized independently before measurement."}
        : std::vector<std::string>{
            "X axis = segmented cell volume: number of voxels inside the tracked 3D connected component.",
            "Y axis = mean normalized brightness: average voxel intensity inside that component after per-frame normalization.",
            "Each point = one cell in one frame. Each TIFF frame is min-max normalized independently before measurement."};

    cv::putText(canvas, title, cv::Point(140, 55), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(30, 30, 30), 2, cv::LINE_AA);
    putTextBlock(canvas, subtitleLines, cv::Point(140, 88), 0.62, cv::Scalar(70, 70, 70), 1, 28);

    drawAxes(
        canvas,
        plotRect,
        xMin,
        xMax,
        yMin,
        yMax,
        "Segmented cell volume (voxel count)",
        useIntegratedIntensity ? "Integrated normalized brightness" : "Mean normalized brightness (0 to 1)",
        0,
        useIntegratedIntensity ? 0 : 3,
        true,
        useIntegratedIntensity);

    auto toPixel = [&](double x, double y) -> cv::Point {
        const double tx = (x - xMin) / (xMax - xMin);
        const double ty = (y - yMin) / (yMax - yMin);
        const int px = plotRect.x + static_cast<int>(std::lround(tx * plotRect.width));
        const int py = plotRect.y + plotRect.height - static_cast<int>(std::lround(ty * plotRect.height));
        return cv::Point(
            std::clamp(px, plotRect.x, plotRect.x + plotRect.width),
            std::clamp(py, plotRect.y, plotRect.y + plotRect.height));
    };

    const auto palette = plotPalette();
    int colorIndex = 0;
    for (const auto& [cellId, rows] : grouped)
    {
        const cv::Scalar color = palette[colorIndex % palette.size()];
        for (const auto& obs : rows)
        {
            const double x = static_cast<double>(obs.voxelCount);
            const double y = useIntegratedIntensity
                ? static_cast<double>(obs.integratedIntensity)
                : static_cast<double>(obs.meanIntensity);
            const cv::Point p = toPixel(x, y);
            cv::circle(canvas, p, 6, color, -1, cv::LINE_AA);
            cv::circle(canvas, p, 7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
        ++colorIndex;
    }

    const RegressionLine line = fitLineYOnX(xs, ys);
    if (line.valid)
    {
        const cv::Point p0 = toPixel(xMin, line.slope * xMin + line.intercept);
        const cv::Point p1 = toPixel(xMax, line.slope * xMax + line.intercept);
        cv::line(canvas, p0, p1, cv::Scalar(20, 20, 20), 2, cv::LINE_AA);
    }

    cv::rectangle(canvas, legendRect, cv::Scalar(225, 225, 225), 1, cv::LINE_AA);
    cv::putText(canvas, "Legend", cv::Point(legendRect.x + 20, legendRect.y + 35),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(40, 40, 40), 2, cv::LINE_AA);
    putTextBlock(
        canvas,
        {
            "Cell IDs are shortened in this figure.",
            "Use the CSV files for the full IDs."
        },
        cv::Point(legendRect.x + 20, legendRect.y + 65),
        0.52,
        cv::Scalar(100, 100, 100),
        1,
        22);

    colorIndex = 0;
    int legendY = legendRect.y + 120;
    for (const auto& [cellId, rows] : grouped)
    {
        const cv::Scalar color = palette[colorIndex % palette.size()];
        cv::circle(canvas, cv::Point(legendRect.x + 22, legendY - 5), 7, color, -1, cv::LINE_AA);
        cv::putText(
            canvas,
            "cell " + shortenCellId(cellId) + " (" + std::to_string(rows.size()) + " pts)",
            cv::Point(legendRect.x + 42, legendY),
            cv::FONT_HERSHEY_SIMPLEX,
            0.58,
            cv::Scalar(60, 60, 60),
            1,
            cv::LINE_AA);
        legendY += 28;
        ++colorIndex;
    }

    const double corr = pearsonCorrelation(xs, ys);
    cv::putText(canvas, "pearson r = " + formatDouble(corr, 4), cv::Point(legendRect.x + 20, legendRect.y + 350),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(40, 40, 40), 2, cv::LINE_AA);
    cv::putText(canvas, "slope = " + formatDouble(line.slope, 6), cv::Point(legendRect.x + 20, legendRect.y + 388),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(40, 40, 40), 2, cv::LINE_AA);
    putTextBlock(
        canvas,
        useIntegratedIntensity
            ? std::vector<std::string>{
                "Interpretation note:",
                "Integrated brightness often rises with volume",
                "because it sums over more voxels.",
                "Use the mean-brightness plot for",
                "the question: does a larger cell become dimmer?"}
            : std::vector<std::string>{
                "Interpretation note:",
                "Negative r or slope: larger segmented cells",
                "tend to have lower mean brightness.",
                "Positive r or slope: larger segmented cells",
                "tend to remain brighter."},
        cv::Point(legendRect.x + 20, legendRect.y + 445),
        0.52,
        cv::Scalar(90, 90, 90),
        1,
        22);
    putTextBlock(
        canvas,
        {
            "Technical note:",
            "Segmented volume is not physical micron^3 volume.",
            "Brightness is relative within each frame.",
            "See CSV/report for the full metric definitions."
        },
        cv::Point(legendRect.x + 20, legendRect.y + 560),
        0.52,
        cv::Scalar(90, 90, 90),
        1,
        22);

    cv::imwrite(path.string(), canvas);
}

static void drawPerCellNormalizedTimeSeries(const fs::path& path, const std::vector<Observation>& observations)
{
    if (observations.empty())
    {
        return;
    }

    std::map<std::string, std::vector<Observation>> grouped;
    for (const auto& obs : observations)
    {
        grouped[obs.cellId].push_back(obs);
    }

    for (auto& [cellId, rows] : grouped)
    {
        std::sort(rows.begin(), rows.end(), [](const Observation& a, const Observation& b) {
            return a.frameIndex < b.frameIndex;
        });
    }

    const int cols = std::min(3, std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(grouped.size()))))));
    const int rowsCount = static_cast<int>((grouped.size() + cols - 1) / cols);
    const int panelW = 560;
    const int panelH = 340;
    const cv::Scalar volumeColor(219, 152, 52);
    const cv::Scalar intensityColor(44, 127, 255);

    cv::Mat canvas(230 + rowsCount * panelH, 100 + cols * panelW, CV_8UC3, cv::Scalar(250, 250, 250));
    cv::putText(canvas, "Per-cell normalized time series", cv::Point(40, 45),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(30, 30, 30), 2, cv::LINE_AA);
    putTextBlock(
        canvas,
        {
            "Blue line = segmented cell volume normalized within this cell across the selected frames.",
            "Orange line = mean normalized brightness normalized within this cell across the selected frames.",
            "Y axis is re-scaled to 0 to 1 separately for each cell and each metric. Compare trend shape, not absolute magnitude.",
            "In each panel, r = pearson correlation between segmented volume and mean brightness for that cell."
        },
        cv::Point(40, 78),
        0.62,
        cv::Scalar(70, 70, 70),
        1,
        28);

    cv::line(canvas, cv::Point(46, 182), cv::Point(96, 182), volumeColor, 3, cv::LINE_AA);
    cv::circle(canvas, cv::Point(71, 182), 4, volumeColor, -1, cv::LINE_AA);
    cv::putText(canvas, "normalized segmented volume", cv::Point(108, 188),
                cv::FONT_HERSHEY_SIMPLEX, 0.54, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
    cv::line(canvas, cv::Point(410, 182), cv::Point(460, 182), intensityColor, 3, cv::LINE_AA);
    cv::circle(canvas, cv::Point(435, 182), 4, intensityColor, -1, cv::LINE_AA);
    cv::putText(canvas, "normalized mean brightness", cv::Point(472, 188),
                cv::FONT_HERSHEY_SIMPLEX, 0.54, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);

    int panelIndex = 0;
    for (const auto& [cellId, cellRows] : grouped)
    {
        const int col = panelIndex % cols;
        const int row = panelIndex / cols;
        const int x0 = 30 + col * panelW;
        const int y0 = 220 + row * panelH;
        const cv::Rect panelRect(x0, y0, panelW - 40, panelH - 35);
        const cv::Rect plotRect(x0 + 78, y0 + 58, panelRect.width - 112, panelRect.height - 118);

        cv::rectangle(canvas, panelRect, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
        cv::putText(canvas, "cell " + shortenCellId(cellId, 6, 4), cv::Point(panelRect.x + 15, panelRect.y + 28),
                    cv::FONT_HERSHEY_SIMPLEX, 0.64, cv::Scalar(40, 40, 40), 2, cv::LINE_AA);

        const RegressionSummary summary = summarizeObservations(cellRows);
        cv::putText(canvas, "r = " + formatDouble(summary.pearsonVolumeVsBrightness, 4),
                    cv::Point(panelRect.x + 320, panelRect.y + 28),
                    cv::FONT_HERSHEY_SIMPLEX, 0.56, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);

        cv::rectangle(canvas, plotRect, cv::Scalar(215, 215, 215), 1, cv::LINE_AA);

        const int frameMin = cellRows.front().frameIndex;
        const int frameMax = cellRows.back().frameIndex;
        const double xMin = static_cast<double>(frameMin);
        const double xMax = static_cast<double>((frameMax == frameMin) ? frameMin + 1 : frameMax);

        double volumeMin = std::numeric_limits<double>::infinity();
        double volumeMax = -std::numeric_limits<double>::infinity();
        double intensityMin = std::numeric_limits<double>::infinity();
        double intensityMax = -std::numeric_limits<double>::infinity();

        for (const auto& obs : cellRows)
        {
            volumeMin = std::min(volumeMin, static_cast<double>(obs.voxelCount));
            volumeMax = std::max(volumeMax, static_cast<double>(obs.voxelCount));
            intensityMin = std::min(intensityMin, static_cast<double>(obs.meanIntensity));
            intensityMax = std::max(intensityMax, static_cast<double>(obs.meanIntensity));
        }

        auto normalize = [](double value, double lo, double hi) -> double {
            if (std::abs(hi - lo) <= 1e-12)
            {
                return 0.5;
            }
            return (value - lo) / (hi - lo);
        };

        auto mapPoint = [&](double frame, double normalizedY) -> cv::Point {
            const double tx = (frame - xMin) / (xMax - xMin);
            const int px = plotRect.x + static_cast<int>(std::lround(tx * plotRect.width));
            const int py = plotRect.y + plotRect.height - static_cast<int>(std::lround(normalizedY * plotRect.height));
            return cv::Point(
                std::clamp(px, plotRect.x, plotRect.x + plotRect.width),
                std::clamp(py, plotRect.y, plotRect.y + plotRect.height));
        };

        for (int t = 0; t <= 4; ++t)
        {
            const int py = plotRect.y + static_cast<int>(std::lround(plotRect.height * t / 4.0));
            cv::line(canvas, cv::Point(plotRect.x, py), cv::Point(plotRect.x + plotRect.width, py),
                     cv::Scalar(238, 238, 238), 1, cv::LINE_AA);
            const double yNorm = static_cast<double>(4 - t) / 4.0;
            cv::putText(canvas, formatDouble(yNorm, 2), cv::Point(plotRect.x - 42, py + 4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(90, 90, 90), 1, cv::LINE_AA);
        }

        for (int f = frameMin; f <= frameMax; ++f)
        {
            const cv::Point p = mapPoint(static_cast<double>(f), 0.0);
            cv::putText(canvas, std::to_string(f), cv::Point(p.x - 8, plotRect.y + plotRect.height + 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(90, 90, 90), 1, cv::LINE_AA);
        }

        std::vector<cv::Point> volumePoints;
        std::vector<cv::Point> intensityPoints;
        for (const auto& obs : cellRows)
        {
            const double frame = static_cast<double>(obs.frameIndex);
            volumePoints.push_back(mapPoint(frame, normalize(static_cast<double>(obs.voxelCount), volumeMin, volumeMax)));
            intensityPoints.push_back(mapPoint(frame, normalize(static_cast<double>(obs.meanIntensity), intensityMin, intensityMax)));
        }

        for (std::size_t i = 1; i < volumePoints.size(); ++i)
        {
            cv::line(canvas, volumePoints[i - 1], volumePoints[i], volumeColor, 2, cv::LINE_AA);
            cv::line(canvas, intensityPoints[i - 1], intensityPoints[i], intensityColor, 2, cv::LINE_AA);
        }

        for (const auto& p : volumePoints)
        {
            cv::circle(canvas, p, 4, volumeColor, -1, cv::LINE_AA);
        }
        for (const auto& p : intensityPoints)
        {
            cv::circle(canvas, p, 4, intensityColor, -1, cv::LINE_AA);
        }

        cv::putText(canvas, "normalized value within this cell (0 to 1)", cv::Point(plotRect.x - 10, plotRect.y - 16),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(80, 80, 80), 1, cv::LINE_AA);
        cv::putText(canvas, "frame index", cv::Point(plotRect.x + plotRect.width / 2 - 40, plotRect.y + plotRect.height + 48),
                    cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(80, 80, 80), 1, cv::LINE_AA);

        ++panelIndex;
    }

    cv::imwrite(path.string(), canvas);
}

} // namespace

int main(int argc, char* argv[])
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    if (argc < 6)
    {
        std::cerr << "Usage: brightness_volume_analyzer <firstFrame> <lastFrame> <input_pattern_or_dir_or_file> <output_dir> <config.yaml> [initial.csv]\n";
        std::cerr << "If initial.csv is provided, the first selected frame uses CSV positions as initialization seeds.\n";
        return 1;
    }

    Args args;
    args.firstFrame = std::stoi(argv[1]);
    args.lastFrame = std::stoi(argv[2]);
    args.input = argv[3];
    args.output = argv[4];
    args.config = argv[5];
    if (argc >= 7)
    {
        args.initial = argv[6];
    }

    BaseConfig config;
    loadConfig(args.config, config);

    std::vector<fs::path> imagePaths = getImageFilePaths(args.input, args.firstFrame, args.lastFrame, config);
    fs::path analysisDir = fs::path(args.output) / "brightness_volume_analysis";
    fs::create_directories(analysisDir);

    EmbryoBrightTracker tracker(config, analysisDir.string());
    tracker.setDebugVerbose(false);

    std::vector<Observation> observations;
    std::vector<SplitRecord> splitEvents;

    const std::vector<cv::Mat> vol0 = tracker.loadVolumeForAnalysis(imagePaths.front());
    if (vol0.empty())
    {
        throw std::runtime_error("Failed to load initial volume.");
    }

    const int Z = static_cast<int>(vol0.size());
    const int Y = vol0[0].rows;
    const int X = vol0[0].cols;

    float initThresholdHigh = tracker.percentileThresholdForAnalysis(vol0, 99.3f);
    initThresholdHigh = clampf(initThresholdHigh, 0.2f, 0.95f);
    const float initThresholdLow = clampf(initThresholdHigh * 0.50f, 0.06f, 0.95f);
    const float initThresholdSplitLow = clampf(initThresholdHigh * 0.60f, 0.08f, 0.95f);

    std::vector<EmbryoBrightTracker::CellState> cells;
    const std::string initialFrameName = imagePaths.front().filename().string();

    if (!args.initial.empty())
    {
        cells = loadInitialCellsFromCsv(args.initial, initialFrameName, vol0, tracker, initThresholdLow);
    }
    else
    {
        cells = tracker.detectInitialCellsGlobalForAnalysis(vol0, initThresholdHigh);
    }

    if (cells.empty())
    {
        throw std::runtime_error(
            "Automatic initial-cell detection returned no cells. Try a later firstFrame where the cells are clearer.");
    }

    float diamSum = 0.0f;
    int diamCount = 0;
    for (const auto& cell : cells)
    {
        if (cell.diameter > 1e-3f)
        {
            diamSum += cell.diameter;
            ++diamCount;
        }
    }

    float matureDiamAvg = (diamCount > 0) ? (diamSum / static_cast<float>(diamCount)) : 40.0f;
    matureDiamAvg = clampf(matureDiamAvg, 20.0f, 120.0f);
    tracker.setMatureDiameterForAnalysis(matureDiamAvg);

    const int initialFrameIndex = args.firstFrame;

    for (auto& cell : cells)
    {
        cell.bbox = tracker.makeBBoxForAnalysis(cell.center, matureDiamAvg, Z, Y, X);

        Observation obs;
        obs.frameIndex = initialFrameIndex;
        obs.frameName = initialFrameName;
        obs.cellId = cell.id;
        obs.parentId = cell.parentId;
        obs.source = "initial";
        obs.x = cell.center.x;
        obs.y = cell.center.y;
        obs.z = cell.center.z;
        obs.diameter = cell.diameter;
        obs.voxelCount = cell.voxelCount;
        obs.meanIntensity = cell.meanIntensity;
        obs.integratedIntensity = cell.meanIntensity * static_cast<float>(cell.voxelCount);
        obs.thresholdHigh = initThresholdHigh;
        obs.thresholdLow = initThresholdLow;
        obs.thresholdSplitLow = initThresholdSplitLow;
        observations.push_back(obs);
    }

    for (int localFrame = 1; localFrame < static_cast<int>(imagePaths.size()); ++localFrame)
    {
        const int frameIndex = args.firstFrame + localFrame;
        const std::string frameName = imagePaths[localFrame].filename().string();

        const std::vector<cv::Mat> volume = tracker.loadVolumeForAnalysis(imagePaths[localFrame]);
        float thresholdHigh = tracker.percentileThresholdForAnalysis(volume, (localFrame <= 7) ? 99.3f : 98.8f);
        thresholdHigh = clampf(thresholdHigh, 0.2f, 0.95f);
        const float thresholdLow = clampf(thresholdHigh * 0.50f, 0.06f, 0.95f);
        const float thresholdSplitLow = clampf(thresholdHigh * 0.60f, 0.08f, 0.95f);

        std::vector<EmbryoBrightTracker::CellState> nextCells;
        nextCells.reserve(cells.size() * 2);
        std::unordered_map<std::string, std::string> sourceById;

        for (const auto& prev : cells)
        {
            if (!prev.alive)
            {
                continue;
            }

            bool ok = false;
            std::vector<EmbryoBrightTracker::Comp3DStat> compsInBox;
            EmbryoBrightTracker::CellState tracked =
                tracker.trackSingleCellByCCInBBoxForAnalysis(frameIndex, volume, prev, thresholdLow, ok, &compsInBox);

            if (!ok || !tracked.alive)
            {
                continue;
            }

            const int splitCooldown = 8;
            if (frameIndex - prev.lastSplitFrame >= splitCooldown)
            {
                std::vector<EmbryoBrightTracker::Comp3DStat> compsForSplit =
                    tracker.extractConnectedComponents3DForAnalysis(
                        volume,
                        thresholdSplitLow,
                        prev.bbox.z0, prev.bbox.z1,
                        prev.bbox.y0, prev.bbox.y1,
                        prev.bbox.x0, prev.bbox.x1,
                        true);

                EmbryoBrightTracker::CellState c1;
                EmbryoBrightTracker::CellState c2;
                if (detectSplitFromComponents(frameIndex, prev, cells, compsForSplit, matureDiamAvg, c1, c2))
                {
                    const float childBoxDiam1 = clampf(std::max(c1.diameter, 0.75f * matureDiamAvg), 20.0f, 120.0f);
                    const float childBoxDiam2 = clampf(std::max(c2.diameter, 0.75f * matureDiamAvg), 20.0f, 120.0f);
                    c1.bbox = tracker.makeBBoxForAnalysis(c1.center, childBoxDiam1, static_cast<int>(volume.size()), volume[0].rows, volume[0].cols);
                    c2.bbox = tracker.makeBBoxForAnalysis(c2.center, childBoxDiam2, static_cast<int>(volume.size()), volume[0].rows, volume[0].cols);

                    pushUniqueTracked(nextCells, sourceById, c1, "split_child_1");
                    pushUniqueTracked(nextCells, sourceById, c2, "split_child_2");

                    SplitRecord split;
                    split.frameIndex = frameIndex;
                    split.frameName = frameName;
                    split.parentId = prev.id;
                    split.child1Id = c1.id;
                    split.child2Id = c2.id;
                    splitEvents.push_back(split);
                    continue;
                }
            }

            const float boxDiam = pickTrackingBoxDiameter(prev, tracked, matureDiamAvg);
            tracked.bbox = tracker.makeBBoxForAnalysis(
                tracked.center,
                boxDiam,
                static_cast<int>(volume.size()),
                volume[0].rows,
                volume[0].cols);
            pushUniqueTracked(nextCells, sourceById, tracked, "tracked");
        }

        for (const auto& cell : nextCells)
        {
            Observation obs;
            obs.frameIndex = frameIndex;
            obs.frameName = frameName;
            obs.cellId = cell.id;
            obs.parentId = cell.parentId;
            obs.source = sourceById[cell.id];
            obs.x = cell.center.x;
            obs.y = cell.center.y;
            obs.z = cell.center.z;
            obs.diameter = cell.diameter;
            obs.voxelCount = cell.voxelCount;
            obs.meanIntensity = cell.meanIntensity;
            obs.integratedIntensity = cell.meanIntensity * static_cast<float>(cell.voxelCount);
            obs.thresholdHigh = thresholdHigh;
            obs.thresholdLow = thresholdLow;
            obs.thresholdSplitLow = thresholdSplitLow;
            observations.push_back(obs);
        }

        cells = std::move(nextCells);
    }

    const fs::path observationsCsv = analysisDir / "per_cell_observations.csv";
    const fs::path summaryCsv = analysisDir / "per_cell_summary.csv";
    const fs::path splitCsv = analysisDir / "split_events.csv";
    const fs::path reportTxt = analysisDir / "report.txt";
    const fs::path scatterMeanPng = analysisDir / "volume_vs_mean_intensity_scatter.png";
    const fs::path scatterIntegratedPng = analysisDir / "volume_vs_integrated_intensity_scatter.png";
    const fs::path perCellTimeSeriesPng = analysisDir / "per_cell_normalized_time_series.png";

    writeObservationsCsv(observationsCsv, observations);
    writeSummaryCsv(summaryCsv, observations);
    writeSplitEventsCsv(splitCsv, splitEvents);
    writeReport(reportTxt, args, imagePaths, matureDiamAvg, observations, splitEvents);
    drawScatterPlot(scatterMeanPng, observations, false);
    drawScatterPlot(scatterIntegratedPng, observations, true);
    drawPerCellNormalizedTimeSeries(perCellTimeSeriesPng, observations);

    std::cout << "Brightness/volume analysis finished.\n";
    std::cout << "Frames analyzed: " << imagePaths.size() << '\n';
    std::cout << "Observation rows: " << observations.size() << '\n';
    std::cout << "Split events: " << splitEvents.size() << '\n';
    std::cout << "Output directory: " << analysisDir << '\n';
    std::cout << "Plots:\n";
    std::cout << "  - " << scatterMeanPng << '\n';
    std::cout << "  - " << scatterIntegratedPng << '\n';
    std::cout << "  - " << perCellTimeSeriesPng << '\n';
    return 0;
}
