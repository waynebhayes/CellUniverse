#ifndef CELL_LUMEN_HPP
#define CELL_LUMEN_HPP

#include "ConfigTypes.hpp"
#include "Ellipsoid.hpp"
#include "EmbryoBrightTracker.hpp"
#include "Frame.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

class CellLumen
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
        int shellVoxelCount = 0;
        float top10Intensity = 0.0f;
        float shellMeanIntensity = 0.0f;
        float meanMinusShell = 0.0f;
        float top10MinusShell = 0.0f;
        EmbryoBrightTracker::Comp3DStat component{};
    };

    CellLumen(BaseConfig config, const fs::path &outputDir);

    std::vector<DetectedCell> detectCellsForFrame(const fs::path &imageFile,
                                                  bool printCellDetails = false,
                                                  bool allowTraMask = true);

    std::vector<DetectedCell> buildInitialCsvForFrame(const fs::path &imageFile,
                                                      const fs::path &csvOutputPath);

private:
    struct DatasetProfile
    {
        std::string label = "default";
        float effectiveZScaling = 1.0f;
        int minHighSeedVoxels = 8;
        float seedMergeDistance = 8.0f;
        float seedSplitSeparation = 10.0f;
        float dedupDistance = 8.0f;
    };

    BaseConfig config;
    fs::path outputDir;
    EmbryoBrightTracker tracker;
    DatasetProfile activeProfile;

    static float clampf(float value, float lo, float hi);
    static std::string toLowerCopy(std::string value);
    static std::string extractFrameFolderName(const fs::path &imageFile);
    static std::string makeCellName(const std::string &frameStem, int index);

    DatasetProfile inferDatasetProfile(const fs::path &imageFile) const;
    float effectiveZScaling() const;
    float estimateBackgroundValue(const std::vector<cv::Mat> &volume) const;
    int computeMinComponentVoxels() const;
    std::vector<DetectedCell> detectCellsInVolume(const std::vector<cv::Mat> &volume,
                                                  const std::string &frameStem);
    std::vector<DetectedCell> detectCellsFromTraMask(const fs::path &traMaskFile,
                                                     const std::vector<cv::Mat> &rawVolume,
                                                     const std::string &frameStem) const;
    std::vector<DetectedCell> detectCellsAtPercentile(const std::vector<cv::Mat> &volume,
                                                      const std::string &frameStem,
                                                      float percentileHigh) const;
    std::vector<DetectedCell> detectCellsBySeededWatershed(const std::vector<cv::Mat> &volume,
                                                           const std::string &frameStem) const;
    std::vector<cv::Mat> buildLocalContrastVolume(const std::vector<cv::Mat> &volume) const;
    DetectedCell makeDetectedCellFromComponent(const EmbryoBrightTracker::Comp3DStat &component) const;
    void estimateAdaptiveRadii(const EmbryoBrightTracker::Comp3DStat &component,
                               float &majorRadius,
                               float &bRadius,
                               float &minorRadius) const;
    std::optional<DetectedCell> detectLocalSeededCell(const std::vector<cv::Mat> &volume,
                                                      const EmbryoBrightTracker::Comp3DStat &seedComponent,
                                                      float thresholdLow) const;
    std::vector<EmbryoBrightTracker::Comp3DStat> extractLocalMaximumSeeds(
        const std::vector<cv::Mat> &volume,
        float thresholdHigh,
        const std::vector<EmbryoBrightTracker::Comp3DStat> &componentSeeds) const;
    std::vector<EmbryoBrightTracker::Comp3DStat> collapseNearbySeeds(
        const std::vector<EmbryoBrightTracker::Comp3DStat> &seeds,
        float mergeDistance) const;
    bool shouldSplitCoarseComponent(const DetectedCell &coarseCell,
                                    const std::vector<EmbryoBrightTracker::Comp3DStat> &containedSeeds) const;
    std::vector<DetectedCell> pruneLikelySatelliteCells(const std::vector<DetectedCell> &cells) const;
    void annotateSignalStats(const std::vector<cv::Mat> &volume,
                             DetectedCell &cell) const;
    std::vector<DetectedCell> applyBiologicalPriors(const std::vector<DetectedCell> &cells,
                                                    const std::vector<cv::Mat> &volume,
                                                    const std::string &stage) const;
    std::vector<DetectedCell> filterWeakSatelliteCells(const std::vector<cv::Mat> &volume,
                                                       const std::vector<DetectedCell> &cells) const;
    void refineCentersByZColumn(const std::vector<cv::Mat> &volume,
                                std::vector<DetectedCell> &cells) const;
    std::vector<DetectedCell> mergeLikelySameCellFragments(const std::vector<DetectedCell> &cells,
                                                           const std::vector<cv::Mat> &volume,
                                                           float thresholdLow,
                                                           float thresholdHigh) const;
    float scoreCandidateCells(const std::vector<DetectedCell> &cells,
                              int &totalVoxels,
                              int &clampedMinorCount,
                              int &verySmallCount,
                              int &veryLargeCount,
                              int &nearDuplicatePairs,
                              int &flattenedCount,
                              int &tinyFragmentCount) const;
    bool componentContainsBrightSeed(const EmbryoBrightTracker::Comp3DStat &component,
                                     const std::vector<EmbryoBrightTracker::Comp3DStat> &highComponents) const;
    std::vector<Ellipsoid> makeEllipsoids(const std::vector<DetectedCell> &cells) const;
    void saveInitialCsv(const fs::path &csvOutputPath,
                        const std::string &frameFileName,
                        const std::vector<DetectedCell> &cells) const;
    void saveFrameOutputs(const fs::path &imageFile,
                          const std::vector<cv::Mat> &realFrame,
                          const std::vector<DetectedCell> &cells) const;
};

#endif
