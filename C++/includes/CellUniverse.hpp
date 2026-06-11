#ifndef CELLUNIVERSE_HPP
#define CELLUNIVERSE_HPP

#include <opencv2/opencv.hpp>
#include "ConfigTypes.hpp"
#include "Frame.hpp"
#include "types.hpp"
#include "Ellipsoid.hpp"

#include <array>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <set>
#include <unordered_map>

namespace fs = std::filesystem;

class CellUniverse
{
public:
    CellUniverse(std::map<std::string, std::vector<Ellipsoid>> initialCells,
                 PathVec imagePaths,
                 BaseConfig &config,
                 std::string outputPath,
                 int firstFrame = 0,
                 int continueFrom = -1);

    void optimize(int frameIndex);
    void saveImages(int frameIndex, const std::string &stage = "");
    void saveCells(int frameIndex);
    void copyCellsForward(size_t to);
    // Memory optimization (M1): after this frame has been optimized, saved,
    // and its snapshot captured, release its image stacks. Cells and
    // snapshot metadata are retained. Enables long-horizon runs (60+ frames)
    // without 13+ GB memory peaks.
    void releaseFrameImages(int frameIndex);
    // Memory optimization (M2 — Option A): lazy per-frame load. Constructor
    // only samples percentiles; per-frame TIFF load + normalize + preprocess
    // happens on demand in this method. Main loop calls `prepareFrame(i)`
    // before `optimize(i)`. Keeps peak memory at ~1-2 frames (<1 GB for
    // 100+ frame runs vs 25+ GB before).
    void prepareFrame(int frameIndex);
    void preprocessAllFramesAlignedToMinimumBackground(bool loadIntoFrames);
    // Checkpoint save/load (Approach 2 — full state serialization).
    // saveCheckpoint(N) writes all state needed to resume AT frame N+1
    // to `{outputPath}/checkpoints/frame_{N:03d}.txt`. Includes:
    // cells of frame N+1 (already copied forward), previousSnapshots,
    // cellShapeReference, cellShapeBirth, firstSeen anti-cascade metadata,
    // RNG state, edge-alignment state, perFrameAdaptiveBackground[N],
    // perFrameMeanBrightness[N], frame N+1 backgroundValue, z_slices/maxZ.
    // loadCheckpoint(N, targetFrameIndex) reads checkpoint N and populates
    // the local frame that should be optimized next. Call BEFORE the main
    // loop starts. After loading, set the loop start to targetFrameIndex.
    void saveCheckpoint(int frameIndex);
    bool loadCheckpoint(int frameIndex, const std::string &checkpointPath);
    bool loadCheckpoint(int checkpointFrameIndex, int targetFrameIndex,
                        const std::string &checkpointPath);

    unsigned int length();

    // ---- Added for realtime viewer ----
    const std::vector<Ellipsoid> &getCells(int frameIndex) const;
    std::vector<std::string> getCellNames(int frameIndex) const;

private:
   BaseConfig config;
   std::vector<Frame> frames;
   std::string outputPath;
   int firstFrame;
   std::map<std::string, PreviousFrameSnapshot> previousSnapshots;
   // Frozen per-cell shape reference (a, b, c radii). Captured at cell
   // birth (frame 1 for initial cells; post-refit at split-accept for
   // daughters) and NEVER updated. Used as the pixel-collection mask
   // basis in subsequent frames' shape fits, decoupled from snap radii,
   // so a bloated fit in frame N can't compound into an even bigger
   // mask for frame N+1. See 2026-04-15 compounding-bloat analysis.
   std::map<std::string, std::array<float, 3>> cellShapeReference;
   // Birth-time radii. Captured once at first appearance, NEVER updated.
   // Used as the pixel-collection mask basis: mask = birth × maskScale.
   // Decoupled from the bounded ref so the mask can't participate in
   // feedback loops (neither upward bloat nor downward thinning).
   // The bounded ref is used ONLY for the fit-side growth cap.
   std::map<std::string, std::array<float, 3>> cellShapeBirth;
   // Absolute frame where each live cell first appeared in this run. Initial
   // or checkpoint-loaded cells are treated as old unless checkpoint metadata
   // says otherwise. Used to block rescue/fallback cascades that immediately
   // re-split newborn daughters.
   std::map<std::string, int> cellFirstSeenFrame;

   // M2 state: per-frame paths retained for lazy load and initial-cells map.
   PathVec imagePaths;
   std::map<std::string, std::vector<Ellipsoid>> initialCells;
   float edgeBrightnessAlignmentTarget = 0.0f;
   bool edgeBrightnessAlignmentTargetInitialized = false;
   int continueFrom = -1;
   // Per-frame CellLumen split proposals. applyCellLumenRescue() builds these
   // from raw current-frame CellLumen centers; optimize() consumes them before
   // random split scheduling.
   std::unordered_map<int, std::unordered_map<std::string, BridgeSplitProposal>> cellLumenSplitPriors;
   // Strong one-real parent-anchor CellLumen centers that are allowed to
   // reanchor an existing parent center during temporal center repair. These
   // are intentionally separate from cellLumenSplitPriors: the global split
   // selector may skip a split candidate for count/overlap reasons, while the
   // same single real center can still be valid evidence to move the parent.
   std::unordered_map<int, std::unordered_map<std::string, std::set<int>>> cellLumenCenterReanchorCandidateIds;
   // Parents whose best CellLumen split pair looked unsafe. The pre-pass
   // fallback may not resurrect these parents, otherwise a weak Lumen pair can
   // be rejected in ranking and then re-enter through the fallback path.
   std::unordered_map<int, std::set<std::string>> cellLumenSplitPriorRejectedBadParents;
   struct CellLumenCenterCandidate {
       cv::Point3f position{0.0f, 0.0f, 0.0f};
       float distance = 0.0f;
       int voxelCount = 0;
       float signal = 0.0f;
       int candidateId = -1;
   };
   std::unordered_map<int, std::unordered_map<std::string, CellLumenCenterCandidate>> cellLumenCenterCandidates;
   struct CellLumenLookaheadCandidate {
       cv::Point3f position{0.0f, 0.0f, 0.0f};
       int voxelCount = 0;
       float signal = 0.0f;
       int candidateId = -1;
   };
   std::unordered_map<int, std::vector<CellLumenLookaheadCandidate>> cellLumenLookaheadCandidates;

   // Per-frame cached summaries for adaptive background (computed at end of
   // optimize(N); consumed by optimize(N+1) without needing frames[N]'s
   // image data.
   std::vector<float> perFrameAdaptiveBackground;
   std::vector<float> perFrameMeanBrightness;
   bool resumePreviousFrameSummaryValid = false;
   float resumePreviousAdaptiveBackground = 0.0f;
   float resumePreviousMeanBrightness = 0.0f;
   size_t resumePreviousCellCount = 0;

   void prepareSignalCentersForFrame(int frameIndex,
                                     const std::vector<cv::Mat> &realFrame,
                                     bool keepLoaded);
   float computeFrameMedianNearestNeighbor(int frameIndex) const;
   void applyRuntimeDensityProfileForFrame(int frameIndex,
                                           const std::string &phase);
   void applyCellLumenRescue(int frameIndex);
   void writeDensityBrightnessMetrics(int frameIndex, const std::string &phase);
   const std::vector<CellLumenLookaheadCandidate> &getCellLumenLookaheadCandidates(int frameIndex);
};

#endif
