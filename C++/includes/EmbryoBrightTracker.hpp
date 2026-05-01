// EmbryoBrightTracker.hpp
#ifndef EMBRYO_BRIGHT_TRACKER_HPP
#define EMBRYO_BRIGHT_TRACKER_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <filesystem>
#include <iostream>
#include <cmath>

#include "ConfigTypes.hpp"
#include "LineageTreeCreator.hpp"

namespace fs = std::filesystem;

class EmbryoBrightTracker
{
public:
    void setDebugVerbose(bool v) { dbgVerbose = v; }

    struct PendingSplit {
        bool active = false;
        int firstFrame = -1;
        cv::Point3f c1, c2;
    };

    std::unordered_map<std::string, PendingSplit> pendingSplits;

    struct BBox3D {
        int z0, z1;
        int y0, y1;
        int x0, x1;
    };

    struct CellState {
        std::string id;
        cv::Point3f center;
        float diameter = 0.0f;
        float meanIntensity = 0.0f;

        int voxelCount = 0; // voxel count of the chosen connected component (for split logic & noise filtering)

        int lastSeenFrame = -1;

        int lastSplitFrame = -100000; // split cooldown guard

        bool alive = true;
        BBox3D bbox{};
        std::string parentId;
    };

    struct SplitEvent {
        int frame = -1;
        std::string parent;
        std::string child1;
        std::string child2;
        float child1Diameter = 0.0f;
        float child2Diameter = 0.0f;
    };

    struct LostEvent {
        int frame = -1;
        std::string cellId;
        cv::Point3f lastCenter;
    };

    struct Comp3DStat {
        int vox = 0;
        double sumW = 0.0;
        double sx = 0.0, sy = 0.0, sz = 0.0;
        double sumI = 0.0;
        int z0=0,z1=0,y0=0,y1=0,x0=0,x1=0;
        cv::Point3f center() const {
            if (sumW <= 1e-9) return cv::Point3f(0,0,0);
            return cv::Point3f((float)(sx/sumW),(float)(sy/sumW),(float)(sz/sumW));
        }

        float meanI() const {
            if (vox <= 0) return 0.0f;
            return (float)(sumI / (double)vox);
        }

        float diamXY() const {
            int dx = (x1 - x0 + 1);
            int dy = (y1 - y0 + 1);
            return (float)std::max(dx, dy);
        }
    };

public:
    EmbryoBrightTracker(const BaseConfig& cfg, const std::string& outputDir);

    void run(const std::vector<fs::path>& imagePaths);

    // Reuse EmbryoBrightTracker internals from standalone analysis tools
    std::vector<cv::Mat> loadVolumeForAnalysis(const fs::path& imageFile) { return loadVolume(imageFile); }
    float percentileThresholdForAnalysis(const std::vector<cv::Mat>& vol, float pct) const { return percentileThreshold(vol, pct); }
    std::vector<CellState> detectInitialCellsGlobalForAnalysis(const std::vector<cv::Mat>& vol, float thresh) { return detectInitialCellsGlobal(vol, thresh); }
    std::vector<Comp3DStat> extractConnectedComponents3DForAnalysis(
        const std::vector<cv::Mat>& vol,
        float threshLow,
        int z0, int z1, int y0, int y1, int x0, int x1,
        bool use26) const {
        return extractConnectedComponents3D(vol, threshLow, z0, z1, y0, y1, x0, x1, use26);
    }
    CellState trackSingleCellByCCInBBoxForAnalysis(
        int frameIdx,
        const std::vector<cv::Mat>& vol,
        const CellState& prev,
        float threshLow,
        bool& ok,
        std::vector<Comp3DStat>* outComps) const {
        return trackSingleCellByCCInBBox(frameIdx, vol, prev, threshLow, ok, outComps);
    }
    BBox3D makeBBoxForAnalysis(const cv::Point3f& c, float matureDiam, int Z, int Y, int X) const {
        return makeBBox(c, matureDiam, Z, Y, X);
    }
    void setMatureDiameterForAnalysis(float value) { matureDiamAvg = value; }
    float getMatureDiameterForAnalysis() const { return matureDiamAvg; }

private:
    std::vector<cv::Mat> loadVolume(const fs::path& imageFile);
    cv::Mat zSliceToU8(const cv::Mat& zf) const;

    float percentileThreshold(const std::vector<cv::Mat>& vol, float pct) const;

    std::vector<CellState> detectInitialCellsGlobal(
        const std::vector<cv::Mat>& vol,
        float thresh);

    std::vector<Comp3DStat> extractConnectedComponents3D(
        const std::vector<cv::Mat>& vol,
        float threshLow,
        int z0, int z1, int y0, int y1, int x0, int x1,
        bool use26) const;

    CellState trackSingleCellByCCInBBox(
        int frameIdx,
        const std::vector<cv::Mat>& vol,
        const CellState& prev,
        float threshLow,
        bool& ok,
        std::vector<Comp3DStat>* outComps) const;

    CellState trackSingleCellInBBox(
        int frameIdx,
        const std::vector<cv::Mat>& vol,
        const CellState& prev,
        float thresh,
        bool& ok) const;

    bool detectSplitInBBox(
        int frameIdx,
        const std::vector<cv::Mat>& vol,
        const CellState& parentPrev,
        float thresh,
        CellState& outC1,
        CellState& outC2) const;

    BBox3D makeBBox(const cv::Point3f& c, float matureDiam, int Z, int Y, int X) const;

    void printFrameSummary(
        int frameIdx,
        const std::vector<CellState>& cells,
        float matureDiam) const;

    void printBBox(const CellState& c) const;

    void saveNewSynth(
        int frameIdx,
        const std::vector<cv::Mat>& realVol,
        const std::vector<CellState>& cells,
        float thresh) const;

    void saveRealStretchedZ255(
    int frameIdx,
    const std::vector<cv::Mat>& realVol) const;

    void updateViewer(
        int frameIdx,
        const std::vector<CellState>& cells);

private:
    BaseConfig config;
    std::string outDir;

    bool dbgVerbose = true;   // set false to silence per-cell debug logs

    LineageTreeCreator viewer;

    bool debugEnabled = true;   // now default ON (same behavior as current)
    int  debugEveryNFrames = 1; // print every N frames
    inline bool dbgFrame(int f) const {
        return debugEnabled && (debugEveryNFrames <= 1 || (f % debugEveryNFrames) == 0);
    }

    float matureDiamAvg = 0.0f;
    int splitCounter = 0;

    std::vector<SplitEvent> splitEvents;
    std::vector<LostEvent> lostEvents;
};

#endif
