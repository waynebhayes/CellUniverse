// Frame.hpp
#ifndef FRAME_HPP
#define FRAME_HPP

#include <vector>
#include <string>
#include <ostream>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "types.hpp"
#include "ConfigTypes.hpp"
#include <random>
#include <functional>
#include "Ellipsoid.hpp"
#include <opencv2/core/mat.hpp>

void interpolateSlices(const cv::Mat& slice1, const cv::Mat& slice2, std::vector<cv::Mat>& processedSlices, int numInterpolations);

// Axis-aligned 3D bounding box in pixel coordinates (inclusive bounds).
// Used for per-cell bbox cost evaluation: cost is measured over voxels
// inside the bbox, with Voronoi neighbor exclusion applied via a mask of
// matching size. See Frame::computeCellBbox / calculateBboxCost.
struct BoundingBox3D
{
    int xMin = 0, xMax = -1;
    int yMin = 0, yMax = -1;
    int zMin = 0, zMax = -1;

    bool isValid() const {
        return xMin <= xMax && yMin <= yMax && zMin <= zMax;
    }
    int nx() const { return xMax - xMin + 1; }
    int ny() const { return yMax - yMin + 1; }
    int nz() const { return zMax - zMin + 1; }
    size_t volume() const {
        if (!isValid()) return 0;
        return static_cast<size_t>(nx()) * ny() * nz();
    }
};

class Frame
{
public:
    // Signal center descriptor used by ImageHandler's signal-center
    // localization (`localizeSignalCentersInStack`). Defined here as a
    // public nested type so ImageHandler.cpp can compile without coupling
    // to the rest of the signal-guided perturbation feature (which is
    // currently unused in our pipeline). Kept for future re-enablement.
    struct SignalCenter {
        cv::Point3f position{0.0f, 0.0f, 0.0f};
        float brightness = 0.0f;
        float sigmaScale = 1.0f;
        int boxes = 0;
    };

    // Single-pipeline constructor — the analysis-frame / dual-pipeline
    // variant was removed on 2026-04-11 when the new ImageHandler preprocessing
    // replaced the sigmoid-first / raw-analysis split.
    Frame(const std::vector<cv::Mat> &realFrame, const SimulationConfig &simulationConfig, const std::vector<Ellipsoid> &cells, const Path &outputPath, const std::string &imageName);

    // Lazy-load constructor (M2): builds a placeholder Frame without image
    // data. Cells and metadata are stored; `_realFrame` / `_synthFrame` are
    // empty until `loadImageStacks()` is called. Used by CellUniverse to
    // defer per-frame preprocessing until just before `optimize()` runs,
    // keeping memory peak at ~1-2 frames' worth for long-horizon runs.
    Frame(const SimulationConfig &simulationConfig,
          const std::vector<Ellipsoid> &cells,
          const Path &outputPath, const std::string &imageName);

    // Populates `_realFrame` and regenerates `_synthFrame` + cost cache.
    // Call after the lazy-load constructor and just before the first
    // image-dependent operation (optimize / regenerateSynthFrame / etc.).
    void loadImageStacks(const std::vector<cv::Mat> &realFrame);

    // Test: has this Frame's image data been loaded yet?
    bool hasImageStacks() const { return !_realFrame.empty(); }

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
    CostCallbackPair perturbCell(size_t index, float overlapWeight = 1000.0f,
                                 bool useSignalGuidance = false,
                                 float randomPerturbRadiusRatio = 1.0f);
    double computeOverlapPenalty(float weight) const;
    double computeOverlapForCell(size_t cellIdx, float weight) const;

    // ---- Triaxial split pipeline (2026-04-11 redesign) ----
    //
    // Voronoi claim-set — each cell (identified by name) contributes zero or
    // more "claim points" in world space. When gathering bright pixels for a
    // split attempt, a pixel is kept only if its nearest claim point across
    // ALL cells belongs to the cell being split.
    using ClaimSet = std::map<std::string, std::vector<cv::Point3f>>;

    // ---- Bounding-box cost infrastructure (Universal Bbox Plan) ----
    //
    // Per-cell cost computation over a finite 3D bbox with Voronoi-based
    // neighbor exclusion. Replaces full-image L2 for per-cell decisions
    // (perturbation, split). See plans/2026-04-14-universal-bbox-cost.md.
    //
    // Bbox half-extent per axis = marginScale * max(a,b,c) of the cell.
    // marginScale matches existing boxRadius convention in
    // gatherBrightPixelsVoronoi and trySplitCellPhased.
    BoundingBox3D computeCellBbox(size_t cellIdx, float marginScale) const;

    // Generic bbox centered at an arbitrary (center, radius) pair. Used to
    // build snap-anchored bboxes from PreviousFrameSnapshot{position, maxR}.
    // Half-extent per axis = marginScale * radius, clamped to image bounds.
    BoundingBox3D computeBboxAtPoint(const cv::Point3f &center,
                                     float radius,
                                     float marginScale) const;

    // Union of per-cell bboxes for a list of cells. Used for split where
    // parent + both daughter candidates share a single voxel set for
    // apples-to-apples baseline vs candidate comparison.
    BoundingBox3D computeUnionBbox(const std::vector<size_t> &cellIndices,
                                    float marginScale) const;

    // Union that includes extra explicit points (e.g. daughter seeds that
    // don't yet exist as cells) each contributing a sphere of pointRadius.
    BoundingBox3D computeUnionBboxWithPoints(
        const std::vector<size_t> &cellIndices,
        float marginScale,
        const std::vector<cv::Point3f> &extraPoints,
        float pointRadius) const;

    // Asymmetric L2 cost over bbox voxels where mask[v]=1. Uses the same
    // asymmetric_cost_weight as calculateCost. synthFrame must be the same
    // size as _realFrame. Inlined voxel loop (no SIMD) is acceptable because
    // a typical bbox is ~5M voxels (6× smaller than full image) and masked
    // skipping is irregular.
    double calculateBboxCost(
        const BoundingBox3D &bbox,
        const std::vector<cv::Mat> &synthFrame,
        const std::vector<uint8_t> &mask,
        int voronoiCellIdx = -1) const;

    // Static-Voronoi cost territory. When enabled, each pixel is assigned
    // to the nearest cell's snap-anchor (or live-center fallback for
    // cells without a snap). `calculateBboxCost(..., cellIdx)` only sums
    // residuals at pixels assigned to cellIdx. Anchors are fixed for the
    // whole frame (SNAP positions, captured once) so the Voronoi boundary
    // does NOT shift when a cell is perturbed — unlike the earlier live-
    // center attempt, the snap-anchor's claim region stays put. Prevents
    // cell X from inflating to cover neighbor Y's bright pixels, because
    // those pixels are never scored against X regardless of X's shape.
    // rebuildVoronoiMap is called at frame start after snap install, and
    // again after each split accept in the optimize loop (daughter anchors
    // derive from the post-split live positions).
    void enableVoronoiCost(bool on) { _voronoiEnabled = on; }
    bool isVoronoiCostEnabled() const { return _voronoiEnabled; }
    void rebuildVoronoiMap();

    // Additive Voronoi bleed penalty. Count of voxels inside the passed
    // cell's ellipsoid that are NOT assigned to `cellIdx` in the
    // snap-anchored Voronoi map. perturbCell multiplies this by
    // `_voronoiBleedWeight` and adds to the perturbation cost delta.
    // Returns 0 when the map is empty / _voronoiEnabled is off / weight
    // is zero, so the penalty is a no-op in those cases.
    std::size_t computeVoronoiBleedVoxels(const Ellipsoid &cell,
                                          int cellIdx) const;
    void setVoronoiBleedWeight(float w) { _voronoiBleedWeight = w; }
    float getVoronoiBleedWeight() const { return _voronoiBleedWeight; }

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
        float maskC = 0.0f,
        // Optional log sink — when non-null, all per-iter log lines are
        // appended here instead of std::cout. Used by the parallelized
        // shape-fit caller to accumulate per-cell logs and emit them in
        // deterministic cell-index order after the parallel region.
        std::ostream *logSink = nullptr);

    std::vector<cv::Mat> getSynthFrame();
    const std::vector<cv::Mat>& getRealFrame() const { return _realFrame; }

    // Memory optimization (M1): release the real + synth image stacks after
    // the frame has been optimized, its snapshot captured, and its outputs
    // saved. Downstream only needs `cells` + snapshot metadata. Cuts peak
    // memory from O(N_frames × 288 MB) to O(2-3 × 288 MB) for long horizons.
    // After calling this, do not call perturbCell/calculateCost/generateSynthFrame
    // etc. on this frame — the image data is gone.
    void releaseImageStacks() {
        _realFrame.clear();
        _realFrame.shrink_to_fit();
        _synthFrame.clear();
        _synthFrame.shrink_to_fit();
        _signalProbability.clear();
        _signalProbability.shrink_to_fit();
        _currentCostPerSlice.clear();
        _currentCostPerSlice.shrink_to_fit();
    }
    void setBackgroundColor(float backgroundColor) { _backgroundValue = backgroundColor; }
    float getBackgroundValue() const { return _backgroundValue; }
    // Signal centers for signal-guided perturbation (yp ffc1917). Populated
    // during frame preparation/preload after preprocessing is loaded.
    void setSignalCenters(std::vector<SignalCenter> centers) { _signalCenters = std::move(centers); }
    const std::vector<SignalCenter>& getSignalCenters() const { return _signalCenters; }
    void setSignalProbability(std::vector<cv::Mat> probability) { _signalProbability = std::move(probability); }
    const std::vector<cv::Mat>& getSignalProbability() const { return _signalProbability; }
    void setMeanCellBrightness(float mean) { _meanCellBrightness = mean; }
    // Bbox-cost mode: perturb/split use a per-cell bbox with Voronoi
    // neighbor exclusion instead of full-image L2. Set at frame start
    // from ProbabilityConfig.use_bbox_cost; forwarded to per-cell paths.
    void setUseBboxCost(bool enable, float marginScale) {
        _useBboxCost = enable;
        _bboxMarginScale = marginScale;
    }
    bool getUseBboxCost() const { return _useBboxCost; }
    float getBboxMarginScale() const { return _bboxMarginScale; }

    // Snap-anchored bbox installation. One bbox per cell-name, fixed for
    // the whole frame, centered on the snapshot position. Restores the
    // position anchor lost when a follow-the-cell bbox dropped voxels at
    // the abandoned snap position (see 2026-04-15 Option A). Cells without
    // a snap (frame 1, newborn daughters post-split) fall back to the
    // legacy live pre/post-union bbox.
    void setSnapBbox(const std::string &name, const BoundingBox3D &bbox) {
        _snapBboxes[name] = bbox;
    }
    void clearSnapBboxes() { _snapBboxes.clear(); }
    bool hasSnapBbox(const std::string &name) const {
        return _snapBboxes.find(name) != _snapBboxes.end();
    }

    // Position prior (2026-04-18, re-introduced after Phase A edge-fit
    // shape stabilized). Quadratic penalty on ||cell.pos - snap.pos||
    // beyond a threshold. Addresses the downstream drift pathology
    // where cells escape the snap bbox during perturbation (e.g. e3d03
    // drifting 94 px in f3 of run 063143). The snap bbox undershoot
    // penalty saturates once the cell fully exits — the quadratic prior
    // doesn't.
    //
    // Formula:
    //   d = ||cell.pos - snap.pos||
    //   penalty = position_prior_weight × max(0, d - threshold)²
    // Below threshold: 0 (allow legitimate biological motion).
    // Above threshold: grows quadratically, dominating any image gain.
    //
    // setSnapPosition installs one snap position per cell name.
    // setPositionPriorWeight sets the global weight (typically 10-50).
    // setPositionPriorThreshold sets the free-motion threshold in px.
    void setSnapPosition(const std::string &name, const cv::Point3f &pos) {
        _snapPositions[name] = pos;
    }
    void clearSnapPositions() { _snapPositions.clear(); }
    void setPositionPriorWeight(float w) { _positionPriorWeight = w; }
    void setPositionPriorThreshold(float t) { _positionPriorThreshold = t; }
    void setMaxPerturbDriftXY(float v) { _maxPerturbDriftXY = v; }
    void setMaxPerturbDriftZ(float v)  { _maxPerturbDriftZ  = v; }

    // _sharedMasks removed — cost path now uses empty mask (no Voronoi
    // exclusion). The shared mask was never read after perturbCell
    // switched to empty mask. Saves ~100M distance comparisons per
    // split attempt that were wasted building and storing the mask.
    void regenerateSynthFrame() {
        _synthFrame = generateSynthFrame();
        // Skip the full-image L2 cache when bbox cost is active — the
        // cache is never read for decisions (perturbCell + split use
        // calculateBboxCost directly). Saves ~32M pixel ops per call.
        if (!_useBboxCost) refreshFullCostCache();
    }
    std::string getImageName() const { return imageName; }
    std::vector<Ellipsoid> cells;

private:
    std::vector<double> z_slices;
    SimulationConfig simulationConfig;
    std::string outputPath;
    std::string imageName;
    std::vector<cv::Mat> _realFrame;
    std::vector<cv::Mat> _synthFrame;
    // Signal centers (yp ffc1917) — bright clusters in the real image that
    // signal-guided perturbation snaps cells onto.
    std::vector<SignalCenter> _signalCenters;
    // Normalized center-guided perturbation probability stack, computed
    // during frame preparation/preload and reused by debug export/future use.
    std::vector<cv::Mat> _signalProbability;
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
    // Universal bbox cost mode — set once per frame via setUseBboxCost().
    bool  _useBboxCost = false;
    float _bboxMarginScale = 3.0f;
    // Mean cell brightness for brightness-proportional overlap scaling.
    // Set once per frame by CellUniverse::optimize. When > 0, perturbCell
    // scales overlap weight by (cellBrightness / mean)².
    float _meanCellBrightness = 0.0f;
    // Snap-anchored bboxes keyed by cell name. Installed once per frame by
    // CellUniverse::optimize from PreviousFrameSnapshot{position,maxRadius}.
    // When present, perturbCell uses the stored bbox as a fixed evaluation
    // window for the whole frame, so voxels at the snap position are always
    // included in the cost sum — drifting away from snap pays an undershoot
    // cost, anchoring the cell to its real-cell location. Missing entry
    // (frame 1, newborn daughters) → legacy live pre/post-union bbox.
    std::unordered_map<std::string, BoundingBox3D> _snapBboxes;
    // Snap positions keyed by cell name — used for the position prior
    // penalty in perturbCell. Populated once per frame by
    // CellUniverse::optimize from previousSnapshots alongside snap bboxes.
    std::unordered_map<std::string, cv::Point3f> _snapPositions;
    // Position prior weight (0 disables). Quadratic penalty on distance
    // from snap beyond threshold. Set once per frame from config.
    float _positionPriorWeight = 0.0f;
    float _positionPriorThreshold = 20.0f;
    // Per-frame velocity cap on drift from snap position. Rejects perturbs
    // that move the cell further than these thresholds. -1 disables.
    float _maxPerturbDriftXY = -1.0f;
    float _maxPerturbDriftZ  = -1.0f;
    // _sharedMasks member removed — see comment above.

    // Static-Voronoi cost territory — per-pixel cell index, one CV_32S slice
    // per z. Rebuilt by rebuildVoronoiMap() using snap positions (or live
    // centers for cells without a snap). Empty when voronoi cost disabled.
    bool _voronoiEnabled = false;
    std::vector<cv::Mat> _voronoiMap;
    std::vector<cv::Point3f> _voronoiAnchors;  // parallel to cells[]
    // Bleed penalty weight: 0 disables the penalty (default). Any >0
    // value is multiplied by the count of ellipsoid voxels outside own
    // Voronoi territory during perturbation cost computation.
    float _voronoiBleedWeight = 0.0f;
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
