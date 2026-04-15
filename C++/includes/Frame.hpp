// Frame.hpp
#ifndef FRAME_HPP
#define FRAME_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "types.hpp"
#include "ConfigTypes.hpp"
#include <random>
#include <functional>
#include "Ellipsoid.hpp"
#include <opencv2/core/mat.hpp>

void interpolateSlices(const cv::Mat& slice1, const cv::Mat& slice2, std::vector<cv::Mat>& processedSlices, int numInterpolations);

class Frame
{
public:
    // Single-pipeline constructor — the analysis-frame / dual-pipeline
    // variant was removed on 2026-04-11 when the new ImageHandler preprocessing
    // replaced the sigmoid-first / raw-analysis split.
    Frame(const std::vector<cv::Mat> &realFrame, const SimulationConfig &simulationConfig, const std::vector<Ellipsoid> &cells, const Path &outputPath, const std::string &imageName);

    // Rendering
    std::vector<cv::Mat> generateSynthFrame();
    std::vector<cv::Mat> generateSynthFrameFast(Ellipsoid &oldCell, Ellipsoid &newCell,
                                                int *outAffectedZMin = nullptr,
                                                int *outAffectedZMax = nullptr);
    std::vector<cv::Mat> generateOutputFrame();
    std::vector<cv::Mat> generateOutputSynthFrame();

    // Cost and optimization
    Cost calculateCost(const std::vector<cv::Mat> &synthFrame);
    size_t length() const;
    CostCallbackPair perturbCell(size_t index, float overlapWeight = 1000.0f);
    double computeOverlapPenalty(float weight) const;
    double computeOverlapForCell(size_t cellIdx, float weight) const;

    // ---- Triaxial split pipeline (2026-04-11 redesign) ----
    //
    // Voronoi claim-set — each cell (identified by name) contributes zero or
    // more "claim points" in world space. When gathering bright pixels for a
    // split attempt, a pixel is kept only if its nearest claim point across
    // ALL cells belongs to the cell being split.
    using ClaimSet = std::map<std::string, std::vector<cv::Point3f>>;

    // Split-attempt result: callback pair that either commits the split
    // (daughters replace parent) or reverts (parent restored). Returns the
    // (costDiff, callback) in the same contract as perturbCell.
    CostCallbackPair trySplitCellPhased(
        size_t cellIndex,
        const PreviousFrameSnapshot &snapshot,
        const ClaimSet &otherCellsClaimSets,
        bool useSnapshotDirection,
        const ProbabilityConfig &probConfig);

    // Frame-start pre-pass helper. For a pre-classified cell, gathers
    // bright pixels in a snapshot-centered bounding box, Voronoi-filters
    // them using the caller-supplied claim sets, and returns the two PCA
    // centroids as image-grounded expected-daughter positions. Used by
    // CellUniverse::optimize at frame start to overwrite the raw snapshot
    // extrapolation seeds with image-grounded estimates, per the plan's
    // Pre-Pass block. Returns false when the pixel set is too small or
    // PCA fails; in that case the caller should keep the snapshot seeds.
    bool imageGroundExpectedDaughters(
        size_t cellIndex,
        const PreviousFrameSnapshot &snapshot,
        const ClaimSet &otherCellsClaimSets,
        cv::Point3f &outD1,
        cv::Point3f &outD2,
        int *outKeptPixels = nullptr) const;

    // Calibrate a cell's POSITION (radii + rotation unchanged) by trying
    // the weighted-mean centroid of Voronoi-filtered bright pixels as a
    // candidate position. The centroid of bright pixels inside a dividing
    // cell is the midpoint between the two daughters — a direct analytic
    // estimate of where the "whole cell" is centered. Compares L2 cost at
    // the current position vs the centroid position and installs whichever
    // is lower. Used by CellUniverse::optimize's calibration pass to seed
    // each cell at a good single-ellipsoid-fit starting position before
    // the perturbation refinement runs.
    //
    // Returns true iff the cell was moved to the centroid (i.e., centroid
    // position gave a lower cost than the current position).
    bool calibrateCellPositionViaCentroid(
        size_t cellIndex,
        const PreviousFrameSnapshot &snapshot,
        const ClaimSet &otherCellsClaimSets);

    // Iteratively fit a cell's SHAPE (rotation + 3 radii, and optionally
    // centroid position) to the bright pixel cloud around it. Each iteration:
    //   1. Gather bright pixels inside (maskScale * current ellipsoid),
    //      Voronoi-filtered against otherCellsClaimSets.
    //   2. Weighted 3D PCA → centroid, 3 eigenvectors, 3 eigenvalues.
    //   3. Greedy-match eigenvectors to current a/b/c axes (identity-stable),
    //      enforce proper rotation (det=+1), decompose to Euler (R=Rz·Ry·Rx).
    //   4. Target radii = radiusScale * sqrt(eigenvalue).
    //   5. If updatePosition, shift centroid toward PCA centroid capped by
    //      maxPosShiftFraction * maxR.
    //   6. Apply directly (no EMA). Stop when radius delta < convergeRadius
    //      AND max axis rotation < convergeAngleDeg.
    // Skips rotation update on eigenvalue degeneracy (λ1/λ3 < 1.1).
    // Returns true iff at least one iteration applied an update.
    // maskA/B/C: fixed radii used to build the pixel-collection mask (sphere
    //   + ellipsoid) each iteration. Keeping this fixed across iterations
    //   prevents the mask-feedback collapse where shrinking fitted radii
    //   tighten the mask, reveal less of the bright cloud, feed smaller
    //   fitted radii, etc. Pass snapshot radii (previous frame's fit) so
    //   the mask always covers the true bright extent. If any is <=0, the
    //   caller's current cell radii are used as fallback.
    bool calibrateCellShapeViaPca(
        size_t cellIndex,
        const ClaimSet &otherCellsClaimSets,
        int maxIters,
        float radiusScale,
        int minPixels,
        float maskScale,
        float convergeRadius,
        float convergeAngleDeg,
        bool  updatePosition,
        float maxPosShiftFraction,
        float maskA = 0.0f,
        float maskB = 0.0f,
        float maskC = 0.0f);


    std::vector<cv::Mat> getSynthFrame();
    const std::vector<cv::Mat>& getRealFrame() const { return _realFrame; }
    void setBackgroundColor(float backgroundColor) { _backgroundValue = backgroundColor; }
    float getBackgroundValue() const { return _backgroundValue; }
    void regenerateSynthFrame() { _synthFrame = generateSynthFrame(); refreshFullCostCache(); }
    std::string getImageName() const { return imageName; }
    std::vector<Ellipsoid> cells;

private:
    std::vector<double> z_slices;
    SimulationConfig simulationConfig;
    std::string outputPath;
    std::string imageName;
    std::vector<cv::Mat> _realFrame;
    std::vector<cv::Mat> _synthFrame;
    double _currentCost = -1.0; // cached L2 image cost of _synthFrame
    // Per-slice L2 contribution of _synthFrame to the total image cost. Kept
    // in sync with _synthFrame / _currentCost so that a perturbation touching
    // only a few z-slices can rebuild the total by recomputing those slices'
    // cv::norm and summing the unchanged cached values. sum(_currentCostPerSlice)
    // equals _currentCost (up to summation order, which is always 0..n-1).
    std::vector<double> _currentCostPerSlice;
    // Runtime-mutable synth frame background and PCA noise floor. Starts at 0.0 (post-sigmoid
    // background invariant). Updated per-frame by the adaptive background path in
    // CellUniverse::optimize via setBackgroundColor().
    float _backgroundValue = 0.0f;
    cv::Size getImageShape();

    // Rebuild _currentCostPerSlice and _currentCost from scratch by walking
    // every slice of _synthFrame vs _realFrame. Used after a full render
    // (constructor, regenerateSynthFrame) where the incremental cache cannot
    // be updated delta-wise.
    void refreshFullCostCache();

    // Given a new synth frame that differs from _synthFrame only in slices
    // [affectedZMin, affectedZMax], compute the new total image cost and
    // write the new per-slice contributions into outNewPerSlice. Slices
    // outside the affected range inherit their cached values unchanged
    // (the underlying cv::Mat buffer is shared — shallow copy — so the
    // L2 norm is bit-identical to the cached value). Passing
    // affectedZMin < 0 or affectedZMax < 0 means "no slice changed"; the
    // cached total is returned verbatim and outNewPerSlice is a copy of
    // _currentCostPerSlice.
    double calculateIncrementalCost(const std::vector<cv::Mat> &newSynthFrame,
                                    int affectedZMin, int affectedZMax,
                                    std::vector<double> &outNewPerSlice) const;
};
#endif // FRAME_H
