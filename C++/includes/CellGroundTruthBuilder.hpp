#ifndef CELL_GROUND_TRUTH_BUILDER_HPP
#define CELL_GROUND_TRUTH_BUILDER_HPP

#include "ConfigTypes.hpp"
#include "EmbryoBrightTracker.hpp"
#include "Frame.hpp"
#include "Spheroid.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

class CellGroundTruthBuilder
{
public:
    struct DetectedCell
    {
        std::string name;
        cv::Point3f centerScaled;
        float zForCsv = 0.0f;
        float majorRadius = 0.0f;
        float bRadius = 0.0f;
        float minorRadius = 0.0f;
        int voxelCount = 0;
        float meanIntensity = 0.0f;
        EmbryoBrightTracker::Comp3DStat component{};
    };

    CellGroundTruthBuilder(BaseConfig config, const fs::path &outputDir);

    std::vector<DetectedCell> buildInitialCsvForFrame(const fs::path &imageFile,
                                                      const fs::path &csvOutputPath);

private:
    BaseConfig config;
    fs::path outputDir;
    EmbryoBrightTracker tracker;

    static float clampf(float value, float lo, float hi);
    static std::string extractFrameFolderName(const fs::path &imageFile);
    static std::string makeCellName(const std::string &frameStem, int index);

    float estimateBackgroundValue(const std::vector<cv::Mat> &volume) const;
    int computeMinComponentVoxels() const;
    std::vector<DetectedCell> detectCellsInVolume(const std::vector<cv::Mat> &volume,
                                                  const std::string &frameStem);
    std::vector<DetectedCell> detectCellsAtPercentile(const std::vector<cv::Mat> &volume,
                                                      const std::string &frameStem,
                                                      float percentileHigh) const;
    DetectedCell makeDetectedCellFromComponent(const EmbryoBrightTracker::Comp3DStat &component) const;
    std::optional<DetectedCell> detectLocalSeededCell(const std::vector<cv::Mat> &volume,
                                                      const EmbryoBrightTracker::Comp3DStat &seedComponent,
                                                      float thresholdLow) const;
    bool componentContainsBrightSeed(const EmbryoBrightTracker::Comp3DStat &component,
                                     const std::vector<EmbryoBrightTracker::Comp3DStat> &highComponents) const;
    std::vector<Spheroid> makeSpheroids(const std::vector<DetectedCell> &cells) const;
    void saveInitialCsv(const fs::path &csvOutputPath,
                        const std::string &frameFileName,
                        const std::vector<DetectedCell> &cells) const;
    void saveFrameOutputs(const fs::path &imageFile,
                          const std::vector<cv::Mat> &realFrame,
                          const std::vector<DetectedCell> &cells) const;
};

#endif
