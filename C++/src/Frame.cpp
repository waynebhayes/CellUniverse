#include "../includes/Frame.hpp"

#include <sstream>

// OpenMP pragmas are parsed by the compiler when -fopenmp is passed;
// no header include needed unless runtime functions (omp_get_thread_num, etc.)
// are used. The #pragma directives below become no-ops without -fopenmp.

// Asymmetric-L2 per-slice cost (Fix E).
// Returns sqrt(sum(w_i * (synth_i - real_i)^2)) where w_i = k when synth>real
// and w_i = 1 when synth<=real. With k=1.0 this is identical to
// cv::norm(real, synth, NORM_L2). Penalizes a cell covering a dark image
// region more heavily than a cell undershooting a bright image region —
// forces split-vs-no-split cost comparison to prefer daughters that cover
// only bright blobs over a parent that also covers the inter-daughter valley.
//
// Uses SIMD-optimized OpenCV primitives (subtract, multiply, compare, sum)
// to stay close to cv::norm performance. Decomposition:
//   sumSq       = sum(diff^2)              (all pixels)
//   posSumSq    = sum(diff^2 where diff>0) (overshoot pixels only)
//   asymSumSq   = sumSq + (k-1) * posSumSq
static double asymmetricL2Slice(const cv::Mat &real, const cv::Mat &synth, float k)
{
    if (k <= 1.0f + 1e-6f) {
        return cv::norm(real, synth, cv::NORM_L2);
    }
    CV_Assert(real.type() == CV_32F && synth.type() == CV_32F);
    CV_Assert(real.size() == synth.size());

    cv::Mat diff;
    cv::subtract(synth, real, diff);
    cv::Mat diffSq;
    cv::multiply(diff, diff, diffSq);
    const double sumSq = cv::sum(diffSq)[0];

    // Mask of pixels where diff > 0 (synth overshoots real).
    cv::Mat posMask;
    cv::compare(diff, 0.0f, posMask, cv::CMP_GT);   // 8U mask, 255 or 0

    // Copy squared diffs only at overshoot pixels; sum those.
    cv::Mat posSq = cv::Mat::zeros(diffSq.size(), diffSq.type());
    diffSq.copyTo(posSq, posMask);
    const double posSumSq = cv::sum(posSq)[0];

    const double asymSumSq = sumSq + static_cast<double>(k - 1.0f) * posSumSq;
    return std::sqrt(std::max(0.0, asymSumSq));
}

// Function to interpolate between two slices
void interpolateSlices(const cv::Mat& slice1, const cv::Mat& slice2, 
                       std::vector<cv::Mat>& processedSlices, int numInterpolations) {
    // Ensure the two slices have the same size and type
    if (slice1.size() != slice2.size() || slice1.type() != slice2.type()) {
        throw std::invalid_argument("Slices must have the same size and type for interpolation!");
    }

    // Perform interpolation
    for (int i = 1; i <= numInterpolations; ++i) {
        double t = static_cast<double>(i) / (numInterpolations + 1);
        cv::Mat interpolatedSlice = (1.0 - t) * slice1 + t * slice2;
        processedSlices.push_back(interpolatedSlice);
    }
}

Frame::Frame(const std::vector<cv::Mat> &realFrame, const SimulationConfig &simulationConfig, const std::vector<Ellipsoid> &cells,
             const Path &outputPath, const std::string &imageName)
    : cells(cells),
      simulationConfig(simulationConfig),
      outputPath(outputPath),
      imageName(imageName),
      _realFrame(realFrame)
{
    // Calculate z_slices
    for (int i = 0; i < simulationConfig.z_slices; ++i)
    {
        double zValue = i;
        z_slices.push_back(zValue);
    }
    _synthFrame = generateSynthFrame();
    refreshFullCostCache();
}

void Frame::refreshFullCostCache()
{
    if (_realFrame.size() != _synthFrame.size())
    {
        throw std::runtime_error("Mismatch in image stack sizes");
    }

    _currentCostPerSlice.assign(_realFrame.size(), 0.0);
    const float asymK = simulationConfig.asymmetric_cost_weight;
    const int nSlices = static_cast<int>(_realFrame.size());
    double totalCost = 0.0;
    #pragma omp parallel for reduction(+:totalCost) schedule(static)
    for (int i = 0; i < nSlices; ++i)
    {
        const double sliceCost = asymmetricL2Slice(_realFrame[i], _synthFrame[i], asymK);
        _currentCostPerSlice[i] = sliceCost;
        totalCost += sliceCost;
    }
    _currentCost = totalCost;
}

double Frame::calculateIncrementalCost(const std::vector<cv::Mat> &newSynthFrame,
                                       int affectedZMin, int affectedZMax,
                                       std::vector<double> &outNewPerSlice) const
{
    if (_realFrame.size() != newSynthFrame.size())
    {
        throw std::runtime_error("Mismatch in image stack sizes");
    }
    if (_currentCostPerSlice.size() != _realFrame.size())
    {
        throw std::runtime_error("Per-slice cost cache not initialized");
    }

    outNewPerSlice = _currentCostPerSlice;

    if (affectedZMin >= 0 && affectedZMax >= 0)
    {
        const int nSlices = static_cast<int>(_realFrame.size());
        const int zMin = std::max(0, affectedZMin);
        const int zMax = std::min(nSlices - 1, affectedZMax);
        const float asymK = simulationConfig.asymmetric_cost_weight;
        for (int i = zMin; i <= zMax; ++i)
        {
            outNewPerSlice[i] = asymmetricL2Slice(_realFrame[i], newSynthFrame[i], asymK);
        }
    }

    // Always sum in slice-index order so the result is bit-identical to
    // what calculateCost(newSynthFrame) would return (same operands, same
    // summation order).
    double totalCost = 0.0;
    for (size_t i = 0; i < outNewPerSlice.size(); ++i)
    {
        totalCost += outNewPerSlice[i];
    }
    return totalCost;
}

std::vector<cv::Mat> Frame::generateSynthFrame()
{
    cv::Size shape = getImageShape();
    std::vector<cv::Mat> frame;

    for (double z : z_slices)
    {
        Image synthImage = cv::Mat(shape, CV_32F, cv::Scalar(_backgroundValue));
        const float zf = static_cast<float>(z);
        for (const auto &cell : cells)
        {
            const float cellMaxR = std::max({cell.getARadius(),
                                             cell.getBRadius(),
                                             cell.getCRadius()});
            if (std::abs(zf - cell.getZ()) > cellMaxR) continue;
            cell.draw(synthImage, simulationConfig, z);
        }
        frame.push_back(synthImage);
    }
    return frame;
}

cv::Size Frame::getImageShape()
{
    if (_realFrame.empty())
    {
        throw std::runtime_error("Real image stack is empty");
    }
    return _realFrame[0].size(); // Returns the size of the first image in the stack
}

Cost Frame::calculateCost(const std::vector<cv::Mat> &synthFrame)
{
    if (_realFrame.size() != synthFrame.size())
    {
        throw std::runtime_error("Mismatch in image stack sizes");
    }

    const float asymK = simulationConfig.asymmetric_cost_weight;
    const int nSlices = static_cast<int>(_realFrame.size());
    double totalCost = 0.0;
    #pragma omp parallel for reduction(+:totalCost) schedule(static)
    for (int i = 0; i < nSlices; ++i)
    {
        totalCost += asymmetricL2Slice(_realFrame[i], synthFrame[i], asymK);
    }
    return totalCost;
}

// =============================================================================
// Bounding-box cost infrastructure
// =============================================================================
//
// Per-cell bbox cost is computed as asymmetric-L2 over voxels inside a 3D
// box around the cell, with voxels claimed by other cells (Voronoi) excluded.
// This concentrates the cost signal on the cell's own territory and makes
// split / perturbation decisions independent of unrelated image regions.

BoundingBox3D Frame::computeCellBbox(size_t cellIdx, float marginScale) const
{
    BoundingBox3D bbox;
    if (cellIdx >= cells.size() || _realFrame.empty()) return bbox;
    const Ellipsoid &cell = cells[cellIdx];
    const float maxR = std::max({cell.getARadius(), cell.getBRadius(), cell.getCRadius()});
    const float r = marginScale * maxR;
    const int cols = _realFrame[0].cols;
    const int rows = _realFrame[0].rows;
    const int slices = static_cast<int>(_realFrame.size());
    bbox.xMin = std::max(0,        static_cast<int>(std::floor(cell.getX() - r)));
    bbox.xMax = std::min(cols - 1, static_cast<int>(std::ceil (cell.getX() + r)));
    bbox.yMin = std::max(0,        static_cast<int>(std::floor(cell.getY() - r)));
    bbox.yMax = std::min(rows - 1, static_cast<int>(std::ceil (cell.getY() + r)));
    bbox.zMin = std::max(0,          static_cast<int>(std::floor(cell.getZ() - r)));
    bbox.zMax = std::min(slices - 1, static_cast<int>(std::ceil (cell.getZ() + r)));
    return bbox;
}

BoundingBox3D Frame::computeBboxAtPoint(const cv::Point3f &center,
                                         float radius,
                                         float marginScale) const
{
    BoundingBox3D bbox;
    if (_realFrame.empty() || radius <= 0.0f) return bbox;
    // Floor on absolute half-extent: small cells (R=10, margin=2.5 → 25 px)
    // get at least 40 px of bbox regardless of their radius, ensuring the
    // position anchor has enough voxels to function.
    constexpr float kMinBboxHalfExtent = 40.0f;
    const float r = std::max(kMinBboxHalfExtent, marginScale * radius);
    const int cols = _realFrame[0].cols;
    const int rows = _realFrame[0].rows;
    const int slices = static_cast<int>(_realFrame.size());
    bbox.xMin = std::max(0,        static_cast<int>(std::floor(center.x - r)));
    bbox.xMax = std::min(cols - 1, static_cast<int>(std::ceil (center.x + r)));
    bbox.yMin = std::max(0,        static_cast<int>(std::floor(center.y - r)));
    bbox.yMax = std::min(rows - 1, static_cast<int>(std::ceil (center.y + r)));
    bbox.zMin = std::max(0,          static_cast<int>(std::floor(center.z - r)));
    bbox.zMax = std::min(slices - 1, static_cast<int>(std::ceil (center.z + r)));
    return bbox;
}

BoundingBox3D Frame::computeUnionBbox(const std::vector<size_t> &cellIndices,
                                       float marginScale) const
{
    BoundingBox3D result;
    bool first = true;
    for (size_t idx : cellIndices) {
        BoundingBox3D b = computeCellBbox(idx, marginScale);
        if (!b.isValid()) continue;
        if (first) { result = b; first = false; continue; }
        result.xMin = std::min(result.xMin, b.xMin);
        result.xMax = std::max(result.xMax, b.xMax);
        result.yMin = std::min(result.yMin, b.yMin);
        result.yMax = std::max(result.yMax, b.yMax);
        result.zMin = std::min(result.zMin, b.zMin);
        result.zMax = std::max(result.zMax, b.zMax);
    }
    return result;
}

BoundingBox3D Frame::computeUnionBboxWithPoints(
    const std::vector<size_t> &cellIndices,
    float marginScale,
    const std::vector<cv::Point3f> &extraPoints,
    float pointRadius) const
{
    BoundingBox3D result = computeUnionBbox(cellIndices, marginScale);
    if (_realFrame.empty()) return result;
    const int cols = _realFrame[0].cols;
    const int rows = _realFrame[0].rows;
    const int slices = static_cast<int>(_realFrame.size());
    bool first = !result.isValid();
    for (const auto &p : extraPoints) {
        const int px0 = std::max(0,        static_cast<int>(std::floor(p.x - pointRadius)));
        const int px1 = std::min(cols - 1, static_cast<int>(std::ceil (p.x + pointRadius)));
        const int py0 = std::max(0,        static_cast<int>(std::floor(p.y - pointRadius)));
        const int py1 = std::min(rows - 1, static_cast<int>(std::ceil (p.y + pointRadius)));
        const int pz0 = std::max(0,          static_cast<int>(std::floor(p.z - pointRadius)));
        const int pz1 = std::min(slices - 1, static_cast<int>(std::ceil (p.z + pointRadius)));
        if (px0 > px1 || py0 > py1 || pz0 > pz1) continue;
        if (first) {
            result.xMin = px0; result.xMax = px1;
            result.yMin = py0; result.yMax = py1;
            result.zMin = pz0; result.zMax = pz1;
            first = false;
            continue;
        }
        result.xMin = std::min(result.xMin, px0);
        result.xMax = std::max(result.xMax, px1);
        result.yMin = std::min(result.yMin, py0);
        result.yMax = std::max(result.yMax, py1);
        result.zMin = std::min(result.zMin, pz0);
        result.zMax = std::max(result.zMax, pz1);
    }
    return result;
}

double Frame::calculateBboxCost(
    const BoundingBox3D &bbox,
    const std::vector<cv::Mat> &synthFrame,
    const std::vector<uint8_t> &mask) const
{
    if (!bbox.isValid()) return 0.0;
    if (synthFrame.size() != _realFrame.size()) {
        throw std::runtime_error("bbox cost: synth/real stack size mismatch");
    }
    const float asymK = simulationConfig.asymmetric_cost_weight;
    // Threshold: only apply asymK when overshoot exceeds this value.
    // Below threshold, boundary rendering artifacts (d ≈ 0.01-0.05) are
    // penalized at 1x (symmetric). Above, genuine overshoot (cell covering
    // dark background, d ≈ 0.1+) gets the full asymK penalty.
    // This eliminates the double-boundary bias: two daughters have 2×
    // boundary artifacts but the same valley-coverage signal as one parent.
    // Without the threshold, asymK amplifies boundary artifacts 8x,
    // making two daughters structurally more expensive than one parent.
    const float asymThreshold = simulationConfig.asymmetric_cost_threshold;
    const int nx = bbox.nx();
    const int ny = bbox.ny();
    const bool useAsym = asymK > 1.0f + 1e-6f;
    // When mask is empty, ALL voxels in the bbox contribute to cost —
    // no Voronoi neighbor exclusion. This is intentional: during any
    // single perturbCell call, neighbors' synth is constant between old
    // and new → cancels out in the cost delta. Exclusion was actively
    // HARMFUL because the Voronoi boundary shifts with the perturbed
    // cell, hiding the cost of abandoning voxels at the snap position
    // and weakening the position anchor. Without exclusion, every voxel
    // in the snap-anchored bbox contributes to the delta → drift from
    // snap costs reliably.
    const bool useMask = !mask.empty();
    double totalCost = 0.0;

    // Two loop variants: with-mask (rare, includes the per-voxel mask check
    // that kills auto-vectorization) and no-mask (default in current code,
    // straight-line inner body that the compiler can SIMD-vectorize). The
    // useAsym flag is loop-invariant so the compiler hoists it; the asym
    // multiply itself is per-voxel data-dependent and stays in the loop.
    if (!useMask) {
        #pragma omp parallel for reduction(+:totalCost) schedule(static)
        for (int z = bbox.zMin; z <= bbox.zMax; ++z) {
            const cv::Mat &realSlice  = _realFrame[z];
            const cv::Mat &synthSlice = synthFrame[z];
            if (realSlice.type() != CV_32F || synthSlice.type() != CV_32F) continue;
            double sliceCost = 0.0;
            for (int y = bbox.yMin; y <= bbox.yMax; ++y) {
                const float *rr = realSlice.ptr<float>(y);
                const float *ss = synthSlice.ptr<float>(y);
                if (useAsym) {
                    for (int x = bbox.xMin; x <= bbox.xMax; ++x) {
                        const float d = ss[x] - rr[x];
                        const float d2 = d * d;
                        const float mul = (d > asymThreshold) ? asymK : 1.0f;
                        sliceCost += d2 * mul;
                    }
                } else {
                    for (int x = bbox.xMin; x <= bbox.xMax; ++x) {
                        const float d = ss[x] - rr[x];
                        sliceCost += d * d;
                    }
                }
            }
            totalCost += sliceCost;
        }
    } else {
        #pragma omp parallel for reduction(+:totalCost) schedule(static)
        for (int z = bbox.zMin; z <= bbox.zMax; ++z) {
            const cv::Mat &realSlice  = _realFrame[z];
            const cv::Mat &synthSlice = synthFrame[z];
            if (realSlice.type() != CV_32F || synthSlice.type() != CV_32F) continue;
            const int zOff = (z - bbox.zMin) * nx * ny;
            double sliceCost = 0.0;
            for (int y = bbox.yMin; y <= bbox.yMax; ++y) {
                const float *rr = realSlice.ptr<float>(y);
                const float *ss = synthSlice.ptr<float>(y);
                const int yOff = zOff + (y - bbox.yMin) * nx;
                for (int x = bbox.xMin; x <= bbox.xMax; ++x) {
                    if (!mask[yOff + (x - bbox.xMin)]) continue;
                    const float d = ss[x] - rr[x];
                    float d2 = d * d;
                    if (useAsym && d > asymThreshold) d2 *= asymK;
                    sliceCost += d2;
                }
            }
            totalCost += sliceCost;
        }
    }
    return totalCost;
}

std::vector<cv::Mat> Frame::generateSynthFrameFast(Ellipsoid &oldCell, Ellipsoid &newCell,
                                                   int *outAffectedZMin, int *outAffectedZMax)
{
    if (cells.empty())
    {
        std::cerr << "Cells are not set\n";
    }

    cv::Size shape = getImageShape(); // Assuming getImageShape() returns a cv::Size
    std::vector<cv::Mat> synthFrame;

    // Calculate the smallest box that contains both the old and new cell
    MinBox minBox = oldCell.calculateMinimumBox(newCell);
    Corner &minCorner = minBox.first;
    Corner &maxCorner = minBox.second;

    // Track which slices were actually re-rendered so callers can drive
    // incremental cost updates without recomputing cv::norm on unchanged
    // slices. -1/-1 means nothing was re-rendered (move entirely outside
    // the cached z range).
    int affectedMin = -1;
    int affectedMax = -1;

    // preallocate space to avoid reallocation
    synthFrame.reserve(z_slices.size());
    for (size_t i = 0; i < z_slices.size(); ++i)
    {
        double z = z_slices[i];
        // If the z-slice is outside the min/max box, append the existing synthetic image to the stack
        // index 2 is representing the z parameter
        if (z < minCorner[2] || z > maxCorner[2])
        {
            synthFrame.push_back(_synthFrame[i]);
            continue;
        }

        if (affectedMin < 0) affectedMin = static_cast<int>(i);
        affectedMax = static_cast<int>(i);

        cv::Mat synthImage = cv::Mat(shape, CV_32F, cv::Scalar(_backgroundValue));

        for (const auto &cell : cells)
        {
            // Skip cells that can't contribute to this z-slice.
            // A cell at z=100 with maxR=25 only affects slices 75-125.
            // Without this check, ALL cells are drawn on every affected
            // slice — 80%+ of draw calls produce zero pixels.
            const float cellMaxR = std::max({cell.getARadius(),
                                             cell.getBRadius(),
                                             cell.getCRadius()});
            if (std::abs(static_cast<float>(z) - cell.getZ()) > cellMaxR) continue;
            cell.draw(synthImage, simulationConfig, z);
        }

        synthFrame.push_back(synthImage);
    }

    if (outAffectedZMin) *outAffectedZMin = affectedMin;
    if (outAffectedZMax) *outAffectedZMax = affectedMax;

    return synthFrame;
}

std::vector<cv::Mat> Frame::generateOutputFrame()
{
    std::vector<cv::Mat> realFrameWithOutlines;

    for (size_t i = 0; i < _realFrame.size(); ++i)
    {
        const cv::Mat &realImage = _realFrame[i];
        double z = z_slices[i];
        const float outlineIntensity = std::min(1.0f, _backgroundValue * 1.6f);

        cv::Mat outputFrame = realImage.clone();

        // Draw outlines for each cell
        for (const auto &cell : cells)
        {
            cell.drawOutline(outputFrame, outlineIntensity, z);
        }

        // Convert to 8-bit image if necessary
        if (outputFrame.depth() != CV_8U)
        {
            outputFrame.convertTo(outputFrame, CV_8U, 255.0);
        }

        realFrameWithOutlines.push_back(outputFrame);
    }

    return realFrameWithOutlines;
}

std::vector<cv::Mat> Frame::generateOutputSynthFrame()
{
    std::vector<cv::Mat> outputSynthFrame;

    for (const auto &synthImage : _synthFrame)
    {
        cv::Mat outputImage;
        if (synthImage.depth() != CV_8U)
        {
            // Convert to 8-bit image if necessary, scaling pixel values by 255
            synthImage.convertTo(outputImage, CV_8U, 255.0);
        }
        else
        {
            outputImage = synthImage.clone();
        }

        outputSynthFrame.push_back(outputImage);
    }

    return outputSynthFrame;
}

size_t Frame::length() const
{
    return cells.size();
}

CostCallbackPair Frame::perturbCell(size_t index, float overlapWeight)
{
    if (index >= cells.size()) {
        return {0.0, [](bool) {}};
    }

    Ellipsoid oldCell = cells[index];
    PerturbDirections perturbDirections;

    // Brightness-proportional overlap weight: scale the penalty by
    // (cellBrightness / meanBrightness)² so dim cells get lower overlap
    // weight (their image cost is lower → overlap shouldn't dominate).
    // Prevents dim cells from being pushed out of position by overlap
    // with brighter neighbors. Disabled when _meanCellBrightness <= 0.
    float effectiveOverlapWeight = overlapWeight;
    if (_meanCellBrightness > 1e-6f) {
        const float bRatio = oldCell.getBrightness() / _meanCellBrightness;
        effectiveOverlapWeight = overlapWeight * bRatio * bRatio;
    }

    // O(n) overlap for just this cell before perturbation
    double oldOverlapCell = computeOverlapForCell(index, effectiveOverlapWeight);

    // Radius-proportional sigma: large cells take bigger position steps,
    // small cells take smaller steps. Scale = maxR / referenceR.
    // When referenceR <= 0, scaling is disabled (posScale = 1.0).
    const float refR = Ellipsoid::cellConfig.perturbSigmaReferenceRadius;
    float posScale = 1.0f;
    if (refR > 1e-3f) {
        const float maxR = std::max({oldCell.getARadius(),
                                     oldCell.getBRadius(),
                                     oldCell.getCRadius()});
        posScale = maxR / refR;
    }
    cells[index] = cells[index].getPerturbedCell(&perturbDirections, posScale);

    // Min-radius hard clamp (2026-04-09): prevent cells from ratcheting down to
    // minimum radius bounds via unconstrained perturbation. The Ellipsoid ctor
    // silently clamps a/b/c radii to their configured minima,
    // so a decrease-biased perturbation sequence parks cells at the floor where
    // the L2 cost rewards the tiny footprint (see 12345...3400 at (10,5) in
    // run 074740 f22 — the degenerate crumpled ellipse visible in the f8
    // screenshot of run 101212 was the same failure mode on 12345...341).
    // Revert any proposal that would take either radius FROM above the floor
    // TO the floor; proposals that were already at the floor are still allowed
    // (those cells are already parked there and need a different recovery
    // path — see the deferred volume recovery work in config).
    {
        const float newAR = cells[index].getARadius();
        const float newBR = cells[index].getBRadius();
        const float newCR = cells[index].getCRadius();
        const float oldAR = oldCell.getARadius();
        const float oldBR = oldCell.getBRadius();
        const float oldCR = oldCell.getCRadius();
        const float newMinorR = cells[index].getMinorRadius();
        const float oldMinorR = oldCell.getMinorRadius();
        const float minAR = static_cast<float>(Ellipsoid::cellConfig.minARadius);
        const float minBR = static_cast<float>(
            Ellipsoid::cellConfig.maxBRadius > 0.0 ? Ellipsoid::cellConfig.minBRadius
                                                   : Ellipsoid::cellConfig.minARadius);
        const float minCR = static_cast<float>(Ellipsoid::cellConfig.minCRadius);
        const float minMinorR = std::min({minAR, minBR, minCR});
        constexpr float kClampEpsilon = 1e-3f;
        const bool hitAFloor = (newAR <= minAR + kClampEpsilon) &&
                               (oldAR >  minAR + kClampEpsilon);
        const bool hitBFloor = (newBR <= minBR + kClampEpsilon) &&
                               (oldBR >  minBR + kClampEpsilon);
        const bool hitCFloor = (newCR <= minCR + kClampEpsilon) &&
                               (oldCR >  minCR + kClampEpsilon);
        const bool hitMinorFloor = (newMinorR <= minMinorR + kClampEpsilon) &&
                                   (oldMinorR >  minMinorR + kClampEpsilon);
        if (hitAFloor || hitBFloor || hitCFloor || hitMinorFloor) {
            cells[index] = oldCell;
            return {0.0, [](bool) {}};
        }
    }

    // O(n) overlap for this cell after perturbation
    double newOverlapCell = computeOverlapForCell(index, effectiveOverlapWeight);

    // Render only the affected z-slice range; generateSynthFrameFast writes
    // [affectedMin, affectedMax] (inclusive) for us to drive incremental
    // cost. Unchanged slices alias _synthFrame[i] — same pixel buffer, so
    // the cached per-slice L2 for those slices is bit-exact.
    int affectedMin = -1;
    int affectedMax = -1;
    auto newSynthFrame = generateSynthFrameFast(oldCell, cells[index],
                                                &affectedMin, &affectedMax);

    double newImageCost = 0.0;
    double oldImageCost = 0.0;
    std::vector<double> newCostPerSlice;

    if (_useBboxCost) {
        // Bbox cost path: measure asymmetric L2 over a fixed 3D bbox, with
        // Voronoi exclusion of voxels claimed by any other cell.
        //
        // Option A — snap-anchored bbox: if this cell has a snap bbox
        // installed (CellUniverse::optimize populates once per frame from
        // PreviousFrameSnapshot.position + maxRadius), use it unchanged for
        // every perturbation of this cell during the frame. That way, voxels
        // at the snap position are always scored — if the cell drifts away
        // from snap, synth at the snap position is empty while real is
        // bright, producing an undershoot penalty that anchors the cell to
        // its real-cell location. Without the anchor, the bbox follows the
        // cell and the abandoned snap voxels drop out of scope entirely.
        //
        // Cells without a snap (frame 1, daughters just created by a split
        // this frame) fall back to the legacy live pre/post-union bbox.
        BoundingBox3D bboxUnion;
        const std::string &cellName = cells[index].getName();
        auto snapIt = _snapBboxes.find(cellName);
        const bool haveSnapBbox = (snapIt != _snapBboxes.end()
                                   && snapIt->second.isValid());
        if (haveSnapBbox) {
            bboxUnion = snapIt->second;
        } else {
            BoundingBox3D bboxPre = computeCellBbox(index, _bboxMarginScale);
            bboxUnion = bboxPre;
            // Old-cell bbox derived inline (cell is currently post-perturb).
            const float maxRold = std::max({oldCell.getARadius(),
                                            oldCell.getBRadius(),
                                            oldCell.getCRadius()});
            const float r = _bboxMarginScale * maxRold;
            const int cols = _realFrame[0].cols;
            const int rows = _realFrame[0].rows;
            const int slices = static_cast<int>(_realFrame.size());
            BoundingBox3D b;
            b.xMin = std::max(0,        static_cast<int>(std::floor(oldCell.getX() - r)));
            b.xMax = std::min(cols - 1, static_cast<int>(std::ceil (oldCell.getX() + r)));
            b.yMin = std::max(0,        static_cast<int>(std::floor(oldCell.getY() - r)));
            b.yMax = std::min(rows - 1, static_cast<int>(std::ceil (oldCell.getY() + r)));
            b.zMin = std::max(0,          static_cast<int>(std::floor(oldCell.getZ() - r)));
            b.zMax = std::min(slices - 1, static_cast<int>(std::ceil (oldCell.getZ() + r)));
            if (!bboxUnion.isValid()) {
                bboxUnion = b;
            } else {
                bboxUnion.xMin = std::min(bboxUnion.xMin, b.xMin);
                bboxUnion.xMax = std::max(bboxUnion.xMax, b.xMax);
                bboxUnion.yMin = std::min(bboxUnion.yMin, b.yMin);
                bboxUnion.yMax = std::max(bboxUnion.yMax, b.yMax);
                bboxUnion.zMin = std::min(bboxUnion.zMin, b.zMin);
                bboxUnion.zMax = std::max(bboxUnion.zMax, b.zMax);
            }
        }

        // No Voronoi exclusion mask for bbox cost. Neighbors' synth is
        // constant between old and new (only the perturbed cell changed)
        // → their contribution cancels in the cost delta. Building a
        // Voronoi mask was actively harmful: the mask shifts with the
        // cell's new position, hiding the cost of abandoning voxels at
        // the snap position (the anchor mechanism that prevents drift).
        // Without exclusion, every voxel in the snap-anchored bbox
        // contributes to the delta → drift from snap costs reliably.
        const std::vector<uint8_t> noMask;  // empty = all voxels count
        oldImageCost = calculateBboxCost(bboxUnion, _synthFrame, noMask);
        newImageCost = calculateBboxCost(bboxUnion, newSynthFrame, noMask);
    } else {
        // Legacy full-image path.
        newImageCost = calculateIncrementalCost(newSynthFrame,
                                                affectedMin, affectedMax,
                                                newCostPerSlice);
        oldImageCost = _currentCost;
    }

    // Position prior penalty (2026-04-18): quadratic penalty on distance
    // from snap beyond a free-motion threshold. Prevents the drift
    // escape-hatch where the snap bbox undershoot penalty saturates once
    // the cell fully exits its bbox, leaving a flat cost landscape
    // outside. With the prior, drift past threshold is quadratically
    // penalized, independent of image evidence.
    //
    // Formula:
    //   d = ||cell.pos - snap.pos||
    //   penalty = weight × max(0, d - threshold)²
    //
    // Evaluated for both old and new cell positions; only the delta
    // matters for the accept/reject decision.
    double oldPositionPrior = 0.0;
    double newPositionPrior = 0.0;
    if (_positionPriorWeight > 0.0f) {
        auto snapPosIt = _snapPositions.find(cells[index].getName());
        if (snapPosIt != _snapPositions.end()) {
            const cv::Point3f &snapPos = snapPosIt->second;
            auto priorOf = [&](const Ellipsoid &c) -> double {
                const float dx = c.getX() - snapPos.x;
                const float dy = c.getY() - snapPos.y;
                const float dz = c.getZ() - snapPos.z;
                const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
                const float over = std::max(0.0f, d - _positionPriorThreshold);
                return static_cast<double>(_positionPriorWeight) *
                       static_cast<double>(over) * static_cast<double>(over);
            };
            oldPositionPrior = priorOf(oldCell);
            newPositionPrior = priorOf(cells[index]);
        }
    }

    double costDiff = (newImageCost + newOverlapCell + newPositionPrior)
                    - (oldImageCost + oldOverlapCell + oldPositionPrior);

    const bool useBboxLocal = _useBboxCost;
    CallBackFunc callback = [this,
                             newSynthFrame = std::move(newSynthFrame),
                             newCostPerSlice = std::move(newCostPerSlice),
                             oldCell, index, newImageCost, perturbDirections,
                             useBboxLocal](bool accept) mutable
    {
        const float brightnessStep = std::max(0.0f, Ellipsoid::cellConfig.brightnessProbabilityStep);
        const float aRadiusStep = std::max(0.0f, Ellipsoid::cellConfig.aRadiusProbabilityStep);
        const float bRadiusStep = std::max(0.0f, Ellipsoid::cellConfig.bRadiusProbabilityStep);
        const float cRadiusStep = std::max(0.0f, Ellipsoid::cellConfig.cRadiusProbabilityStep);
        if (accept) {
            if (perturbDirections.brightness != 0) this->cells[index].adjustBrightnessPerturbProbability(perturbDirections.brightness, brightnessStep);
            if (perturbDirections.aRadius != 0) this->cells[index].adjustARadiusPerturbProbability(perturbDirections.aRadius, aRadiusStep);
            if (perturbDirections.bRadius != 0) this->cells[index].adjustBRadiusPerturbProbability(perturbDirections.bRadius, bRadiusStep);
            if (perturbDirections.cRadius != 0) this->cells[index].adjustCRadiusPerturbProbability(perturbDirections.cRadius, cRadiusStep);
            this->_synthFrame = std::move(newSynthFrame);
            if (!useBboxLocal) {
                this->_currentCost = newImageCost;
                this->_currentCostPerSlice = std::move(newCostPerSlice);
            }
            // Under bbox cost, _currentCost/_currentCostPerSlice are stale
            // and unused for decisions. They remain populated from the
            // initial refreshFullCostCache for diagnostic logging only.
        } else {
            Ellipsoid revertedCell = oldCell;
            if (perturbDirections.brightness != 0) revertedCell.adjustBrightnessPerturbProbability(perturbDirections.brightness, -brightnessStep);
            if (perturbDirections.aRadius != 0) revertedCell.adjustARadiusPerturbProbability(perturbDirections.aRadius, -aRadiusStep);
            if (perturbDirections.bRadius != 0) revertedCell.adjustBRadiusPerturbProbability(perturbDirections.bRadius, -bRadiusStep);
            if (perturbDirections.cRadius != 0) revertedCell.adjustCRadiusPerturbProbability(perturbDirections.cRadius, -cRadiusStep);
            this->cells[index] = revertedCell;
        }
    };
    return {costDiff, callback};
}

namespace {
// Bounding-sphere radius for a triaxial ellipsoid. Conservative overlap
// detection: treats the cell as the smallest enclosing sphere so no pair of
// touching cells is missed. Trades some over-penalization of non-overlapping
// but elongated cells for correctness. Runtime cost is one std::max.
inline float boundingSphereRadius(const Ellipsoid &cell)
{
    return std::max({cell.getARadius(), cell.getBRadius(), cell.getCRadius()});
}
}

// Non-saturating overlap penalty (barrier form). Diverges as cells approach
// full coincidence, so any finite image-cost gain cannot overcome stacking.
//   penalty = weight × ratio² / (1 − ratio + epsilon)
// Properties:
//   ratio=0.0  → 0                (no overlap, no penalty)
//   ratio=0.3  → ~1.4× old        (barely stronger for light overlap)
//   ratio=0.5  → ~2× old
//   ratio=0.85 → ~5.6× old        (serious overlap, strong pushback)
//   ratio=0.95 → ~17× old
//   ratio=0.99 → ~100× old
//   ratio=1.0  → huge (bounded by 1/epsilon)    (full coincidence forbidden)
// EPS keeps numerical stability; choose small enough that ratio=0.95 barrier
// is already effective but not so small the near-coincidence penalty overflows.
static inline double nonSaturatingOverlap(float overlapRatio, float weight)
{
    constexpr float EPS = 0.01f;
    const float denom = std::max(EPS, 1.0f - overlapRatio + EPS);
    return static_cast<double>(weight) *
           static_cast<double>(overlapRatio) *
           static_cast<double>(overlapRatio) /
           static_cast<double>(denom);
}

double Frame::computeOverlapPenalty(float weight) const
{
    double totalPenalty = 0.0;
    for (size_t i = 0; i < cells.size(); ++i) {
        float ri = boundingSphereRadius(cells[i]);
        float xi = cells[i].getX();
        float yi = cells[i].getY();
        float zi = cells[i].getZ();
        for (size_t j = i + 1; j < cells.size(); ++j) {
            float rj = boundingSphereRadius(cells[j]);
            float dx = xi - cells[j].getX();
            float dy = yi - cells[j].getY();
            float dz = zi - cells[j].getZ();
            float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            float combinedR = ri + rj;
            if (dist < combinedR) {
                float overlapRatio = (combinedR - dist) / combinedR;
                totalPenalty += nonSaturatingOverlap(overlapRatio, weight);
            }
        }
    }
    return totalPenalty;
}

double Frame::computeOverlapForCell(size_t cellIdx, float weight) const
{
    double penalty = 0.0;
    float ri = boundingSphereRadius(cells[cellIdx]);
    float xi = cells[cellIdx].getX();
    float yi = cells[cellIdx].getY();
    float zi = cells[cellIdx].getZ();
    for (size_t j = 0; j < cells.size(); ++j) {
        if (j == cellIdx) continue;
        float rj = boundingSphereRadius(cells[j]);
        float dx = xi - cells[j].getX();
        float dy = yi - cells[j].getY();
        float dz = zi - cells[j].getZ();
        float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        float combinedR = ri + rj;
        if (dist < combinedR) {
            float overlapRatio = (combinedR - dist) / combinedR;
            penalty += nonSaturatingOverlap(overlapRatio, weight);
        }
    }
    return penalty;
}


// ---------------------------------------------------------------------------
// Triaxial split pipeline — Phase A / Phase B helper (2026-04-11 redesign).
// ---------------------------------------------------------------------------

namespace {

struct BrightPixel
{
    cv::Point3f pos;   // world coordinates (x, y, z in interpolated-z space)
    float weight;      // pixel intensity above background
};

// Diagnostic counts for the voxel filter pipeline.
struct GatherStats
{
    int boxVoxels = 0;          // voxels in the axis-aligned bounding box
    int inSphere = 0;           // voxels inside the spherical box
    int aboveBrightness = 0;    // voxels above the brightness cutoff
    int voronoiRejected = 0;    // rejected because a non-self claim point was closer
    int voronoiKept = 0;        // final kept (== returned.size())
};

// Gather bright pixels inside a 3D bounding box around `center`, keeping
// only those whose nearest claim point across ALL cells belongs to the
// splitting cell (`selfClaimPoints`). Returns the kept pixels plus their
// raw intensities above background.
std::vector<BrightPixel> gatherBrightPixelsVoronoi(
    const std::vector<cv::Mat> &realFrame,
    float backgroundValue,
    const cv::Point3f &center,
    float radius,
    const std::vector<cv::Point3f> &selfClaimPoints,
    const Frame::ClaimSet &otherClaimSets,
    GatherStats *stats = nullptr)
{
    std::vector<BrightPixel> kept;
    if (realFrame.empty() || radius <= 0.0f || selfClaimPoints.empty()) {
        return kept;
    }

    const int rows = realFrame[0].rows;
    const int cols = realFrame[0].cols;
    const int slices = static_cast<int>(realFrame.size());

    const int minX = std::max(0, static_cast<int>(std::floor(center.x - radius)));
    const int maxX = std::min(cols - 1, static_cast<int>(std::ceil(center.x + radius)));
    const int minY = std::max(0, static_cast<int>(std::floor(center.y - radius)));
    const int maxY = std::min(rows - 1, static_cast<int>(std::ceil(center.y + radius)));
    const int minZ = std::max(0, static_cast<int>(std::floor(center.z - radius)));
    const int maxZ = std::min(slices - 1, static_cast<int>(std::ceil(center.z + radius)));

    const float radiusSq = radius * radius;

    if (stats) {
        stats->boxVoxels = std::max(0, (maxX - minX + 1)) *
                           std::max(0, (maxY - minY + 1)) *
                           std::max(0, (maxZ - minZ + 1));
    }

    const auto distSq = [](const cv::Point3f &a, const cv::Point3f &b) {
        const float dx = a.x - b.x;
        const float dy = a.y - b.y;
        const float dz = a.z - b.z;
        return dx * dx + dy * dy + dz * dz;
    };

    // Brightness cutoff: pixels above (background + small margin). The margin
    // keeps us above sensor noise without being strict enough to lose dim
    // inter-daughter regions.
    const float brightnessCutoff = std::max(0.05f, backgroundValue + 0.02f);

    for (int z = minZ; z <= maxZ; ++z) {
        const cv::Mat &slice = realFrame[z];
        if (slice.type() != CV_32F || slice.empty()) continue;

        const float dz = static_cast<float>(z) - center.z;
        const float dzSq = dz * dz;

        for (int y = minY; y <= maxY; ++y) {
            const float *row = slice.ptr<float>(y);
            const float dy = static_cast<float>(y) - center.y;
            const float dySq = dy * dy;
            for (int x = minX; x <= maxX; ++x) {
                const float dx = static_cast<float>(x) - center.x;
                const float r2 = dx * dx + dySq + dzSq;
                if (r2 > radiusSq) continue;
                if (stats) ++stats->inSphere;

                const float v = row[x];
                if (v <= brightnessCutoff) continue;
                if (stats) ++stats->aboveBrightness;

                const cv::Point3f p{
                    static_cast<float>(x),
                    static_cast<float>(y),
                    static_cast<float>(z)};

                // Hard Voronoi exclusion: keep pixel only if self is the
                // nearest claim across all cells.
                float selfBest = std::numeric_limits<float>::infinity();
                for (const auto &sp : selfClaimPoints) {
                    const float d2 = distSq(p, sp);
                    if (d2 < selfBest) selfBest = d2;
                }

                float otherBest = std::numeric_limits<float>::infinity();
                for (const auto &kv : otherClaimSets) {
                    for (const auto &op : kv.second) {
                        const float d2 = distSq(p, op);
                        if (d2 < otherBest) otherBest = d2;
                    }
                }

                // Hard Voronoi exclusion: keep pixel only if self is the
                // nearest claim across ALL cells. Simple, proven in best22.
                bool keep = (otherBest >= selfBest);
                if (keep) {
                    kept.push_back({p, v - backgroundValue});
                } else if (stats) {
                    ++stats->voronoiRejected;
                }
            }
        }
    }

    if (stats) stats->voronoiKept = static_cast<int>(kept.size());
    return kept;
}

// PCA on weighted 3D points in the CELL'S LOCAL FRAME. Transforms world-
// space pixel positions into the cell's local coordinate system (inverse
// rotation + normalization by radii a, b, c) before computing PCA. This
// makes the analysis invariant to cell orientation and shape — the PCA sees
// brightness distribution relative to the cell, not absolute image geometry.
// The resulting eigenvector is rotated back to world space. D1/D2 centroids
// are returned in world space (computed from the original world positions).
//
// When cellCenter is null or radii are zero, falls back to world-frame PCA
// with isotropic normalization (divide all axes by maxR).
bool pca3DWithCentroids(
    const std::vector<BrightPixel> &points,
    cv::Point3f &eigvec1Out,
    cv::Point3f &d1Out,
    cv::Point3f &d2Out,
    const cv::Point3f *cellCenter = nullptr,
    const std::array<double, 9> *invRotation = nullptr,
    float radiusA = 0.0f, float radiusB = 0.0f, float radiusC = 0.0f)
{
    if (points.size() < 8) return false;

    const bool useLocalFrame = cellCenter && invRotation
        && radiusA > 1e-3f && radiusB > 1e-3f && radiusC > 1e-3f;

    // Weighted mean in world space.
    cv::Point3f mean{0.0f, 0.0f, 0.0f};
    double wsum = 0.0;
    for (const auto &bp : points) {
        mean.x += bp.pos.x * bp.weight;
        mean.y += bp.pos.y * bp.weight;
        mean.z += bp.pos.z * bp.weight;
        wsum += bp.weight;
    }
    if (wsum < 1e-6) return false;
    mean.x /= static_cast<float>(wsum);
    mean.y /= static_cast<float>(wsum);
    mean.z /= static_cast<float>(wsum);

    // Normalization factors.
    const double invA = useLocalFrame ? (1.0 / radiusA) : 1.0;
    const double invB = useLocalFrame ? (1.0 / radiusB) : 1.0;
    const double invC = useLocalFrame ? (1.0 / radiusC) : 1.0;
    const auto &R = invRotation; // R_T: inverse rotation (world→local)

    // Helper: transform a world-space displacement into the normalized local
    // frame. If not using local frame, just returns the displacement as-is.
    auto toLocal = [&](double wx, double wy, double wz,
                       double &lx, double &ly, double &lz) {
        if (useLocalFrame) {
            // Inverse rotation: world → local
            const auto &M = *R;
            const double rx = M[0] * wx + M[1] * wy + M[2] * wz;
            const double ry = M[3] * wx + M[4] * wy + M[5] * wz;
            const double rz = M[6] * wx + M[7] * wy + M[8] * wz;
            // Normalize by radii
            lx = rx * invA;
            ly = ry * invB;
            lz = rz * invC;
        } else {
            lx = wx; ly = wy; lz = wz;
        }
    };

    // Weighted covariance in local (or world) frame.
    double cxx = 0, cxy = 0, cxz = 0, cyy = 0, cyz = 0, czz = 0;
    for (const auto &bp : points) {
        double lx, ly, lz;
        toLocal(bp.pos.x - mean.x, bp.pos.y - mean.y, bp.pos.z - mean.z,
                lx, ly, lz);
        const double w = bp.weight;
        cxx += w * lx * lx;
        cxy += w * lx * ly;
        cxz += w * lx * lz;
        cyy += w * ly * ly;
        cyz += w * ly * lz;
        czz += w * lz * lz;
    }
    cxx /= wsum; cxy /= wsum; cxz /= wsum;
    cyy /= wsum; cyz /= wsum; czz /= wsum;

    cv::Matx33d cov(
        cxx, cxy, cxz,
        cxy, cyy, cyz,
        cxz, cyz, czz);

    cv::Matx33d eigvecs;
    cv::Vec3d eigvals;
    cv::eigen(cov, eigvals, eigvecs);
    // cv::eigen returns eigenvectors as rows of `eigvecs`, sorted by
    // descending eigenvalue. Row 0 is the principal direction.
    // This eigenvector is in LOCAL frame (or world if not using local).
    cv::Point3f ev1Local(
        static_cast<float>(eigvecs(0, 0)),
        static_cast<float>(eigvecs(0, 1)),
        static_cast<float>(eigvecs(0, 2)));

    // Rotate eigenvector back to world space if using local frame.
    // Forward rotation R maps local→world. R_T is R^T stored row-major,
    // so column i of R = (R_T[i], R_T[3+i], R_T[6+i]).
    // R * v = (col0*vx + col1*vy + col2*vz).
    cv::Point3f ev1World;
    if (useLocalFrame) {
        const auto &M = *R;
        // Un-normalize: scale eigenvector by radii to undo the 1/r normalization
        // before rotating back. This ensures the world-space direction reflects
        // actual spatial extent, not the normalized unit-sphere.
        const double sx = ev1Local.x * radiusA;
        const double sy = ev1Local.y * radiusB;
        const double sz = ev1Local.z * radiusC;
        // Forward rotation (local→world): columns of R = rows of R^T transposed
        // R * v = (R_T[0]*sx + R_T[1]*sy + R_T[2]*sz, ...)
        // Wait — R_T is R^T row-major. R = (R_T)^T. So R[i][j] = R_T[j][i].
        // R * v:  row i of R = column i of R_T = (R_T[i], R_T[i+3], R_T[i+6])? No.
        // R_T stored row-major: R_T[row*3+col]. R_T[row][col] = R^T[row][col] = R[col][row].
        // So R[i][j] = R_T[j*3+i].
        // R*v: (R*v)_i = sum_j R[i][j]*v_j = sum_j R_T[j*3+i]*v_j
        //             = R_T[0*3+i]*vx + R_T[1*3+i]*vy + R_T[2*3+i]*vz
        //             = R_T[i]*vx + R_T[3+i]*vy + R_T[6+i]*vz
        ev1World.x = static_cast<float>(M[0]*sx + M[3]*sy + M[6]*sz);
        ev1World.y = static_cast<float>(M[1]*sx + M[4]*sy + M[7]*sz);
        ev1World.z = static_cast<float>(M[2]*sx + M[5]*sy + M[8]*sz);
    } else {
        ev1World = ev1Local;
    }

    const double n = std::sqrt(ev1World.x * ev1World.x +
                               ev1World.y * ev1World.y +
                               ev1World.z * ev1World.z);
    if (n < 1e-9) return false;
    ev1World.x = static_cast<float>(ev1World.x / n);
    ev1World.y = static_cast<float>(ev1World.y / n);
    ev1World.z = static_cast<float>(ev1World.z / n);
    eigvec1Out = ev1World;

    // Project every pixel onto the WORLD-SPACE eigenvector, find the median,
    // partition into two groups, compute weighted centroid of each group.
    // Centroids are returned in world space.
    std::vector<float> projections;
    projections.reserve(points.size());
    for (const auto &bp : points) {
        const float px = bp.pos.x - mean.x;
        const float py = bp.pos.y - mean.y;
        const float pz = bp.pos.z - mean.z;
        projections.push_back(px * ev1World.x + py * ev1World.y + pz * ev1World.z);
    }

    std::vector<float> sortedProj = projections;
    std::nth_element(
        sortedProj.begin(),
        sortedProj.begin() + sortedProj.size() / 2,
        sortedProj.end());
    const float median = sortedProj[sortedProj.size() / 2];

    cv::Point3f sumLo{0, 0, 0}, sumHi{0, 0, 0};
    double wLo = 0.0, wHi = 0.0;
    for (size_t i = 0; i < points.size(); ++i) {
        const auto &bp = points[i];
        if (projections[i] < median) {
            sumLo.x += bp.pos.x * bp.weight;
            sumLo.y += bp.pos.y * bp.weight;
            sumLo.z += bp.pos.z * bp.weight;
            wLo += bp.weight;
        } else {
            sumHi.x += bp.pos.x * bp.weight;
            sumHi.y += bp.pos.y * bp.weight;
            sumHi.z += bp.pos.z * bp.weight;
            wHi += bp.weight;
        }
    }
    if (wLo < 1e-6 || wHi < 1e-6) return false;

    d1Out = cv::Point3f(
        sumLo.x / static_cast<float>(wLo),
        sumLo.y / static_cast<float>(wLo),
        sumLo.z / static_cast<float>(wLo));
    d2Out = cv::Point3f(
        sumHi.x / static_cast<float>(wHi),
        sumHi.y / static_cast<float>(wHi),
        sumHi.z / static_cast<float>(wHi));
    return true;
}

// Project bright pixels onto a given direction, split at the median, and
// return the weighted centroid of each half. This gives a data-driven
// midpoint and separation for any candidate split axis without relying on
// PCA to choose the direction.
bool centroidsAlongAxis(
    const std::vector<BrightPixel> &points,
    const cv::Point3f &axis,
    cv::Point3f &d1Out,
    cv::Point3f &d2Out)
{
    if (points.size() < 8) return false;

    // Weighted mean for centering.
    cv::Point3f mean{0, 0, 0};
    double wsum = 0.0;
    for (const auto &bp : points) {
        mean.x += bp.pos.x * bp.weight;
        mean.y += bp.pos.y * bp.weight;
        mean.z += bp.pos.z * bp.weight;
        wsum += bp.weight;
    }
    if (wsum < 1e-6) return false;
    mean.x /= static_cast<float>(wsum);
    mean.y /= static_cast<float>(wsum);
    mean.z /= static_cast<float>(wsum);

    // Project onto axis.
    std::vector<float> proj;
    proj.reserve(points.size());
    for (const auto &bp : points) {
        proj.push_back(
            (bp.pos.x - mean.x) * axis.x +
            (bp.pos.y - mean.y) * axis.y +
            (bp.pos.z - mean.z) * axis.z);
    }

    // Median split.
    std::vector<float> sorted = proj;
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
    const float median = sorted[sorted.size() / 2];

    cv::Point3f sumLo{0, 0, 0}, sumHi{0, 0, 0};
    double wLo = 0.0, wHi = 0.0;
    for (size_t i = 0; i < points.size(); ++i) {
        const auto &bp = points[i];
        if (proj[i] < median) {
            sumLo += cv::Point3f(bp.pos.x * bp.weight, bp.pos.y * bp.weight, bp.pos.z * bp.weight);
            wLo += bp.weight;
        } else {
            sumHi += cv::Point3f(bp.pos.x * bp.weight, bp.pos.y * bp.weight, bp.pos.z * bp.weight);
            wHi += bp.weight;
        }
    }
    if (wLo < 1e-6 || wHi < 1e-6) return false;

    d1Out = sumLo * (1.0f / static_cast<float>(wLo));
    d2Out = sumHi * (1.0f / static_cast<float>(wHi));
    return true;
}

// Rotate a unit vector by `angleRad` around `axis` (Rodrigues' formula).
cv::Point3f rotateAroundAxis(const cv::Point3f &v, const cv::Point3f &axis, float angleRad)
{
    const float c = std::cos(angleRad);
    const float s = std::sin(angleRad);
    const float oneMinusC = 1.0f - c;
    const float ax = axis.x, ay = axis.y, az = axis.z;
    const float dot = v.x * ax + v.y * ay + v.z * az;
    cv::Point3f out;
    out.x = v.x * c + (ay * v.z - az * v.y) * s + ax * dot * oneMinusC;
    out.y = v.y * c + (az * v.x - ax * v.z) * s + ay * dot * oneMinusC;
    out.z = v.z * c + (ax * v.y - ay * v.x) * s + az * dot * oneMinusC;
    return out;
}

// Build an arbitrary orthonormal frame whose +z aligns with `primary`. Used
// for generating rotation-candidate axes perpendicular to the primary.
void orthonormalFrame(const cv::Point3f &primary, cv::Point3f &u, cv::Point3f &v)
{
    cv::Point3f seed = (std::abs(primary.x) < 0.9f)
        ? cv::Point3f(1, 0, 0) : cv::Point3f(0, 1, 0);
    u.x = primary.y * seed.z - primary.z * seed.y;
    u.y = primary.z * seed.x - primary.x * seed.z;
    u.z = primary.x * seed.y - primary.y * seed.x;
    const float un = std::sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
    if (un > 1e-6f) { u.x /= un; u.y /= un; u.z /= un; }
    v.x = primary.y * u.z - primary.z * u.y;
    v.y = primary.z * u.x - primary.x * u.z;
    v.z = primary.x * u.y - primary.y * u.x;
}

// Bio check — generic sphere-like geometric sanity. Called before cost check.
// Returns true if the split is geometrically plausible, false to reject.
bool bioCheckDaughters(
    const Ellipsoid &daughter1,
    const Ellipsoid &daughter2,
    double refParentVolume,
    const std::vector<Ellipsoid> &allCells,
    size_t d1Idx,
    size_t d2Idx,
    const ProbabilityConfig &probConfig,
    std::string &reasonOut)
{
    const auto cellVolume = [](const Ellipsoid &c) {
        return static_cast<double>(c.getARadius()) *
               static_cast<double>(c.getBRadius()) *
               static_cast<double>(c.getCRadius());
    };

    const float d1R = std::max({daughter1.getARadius(),
                                 daughter1.getBRadius(),
                                 daughter1.getCRadius()});
    const float d2R = std::max({daughter2.getARadius(),
                                 daughter2.getBRadius(),
                                 daughter2.getCRadius()});

    // 1. Size ratio check (catches shrink-grow degenerate splits).
    if (d1R < 1e-3f || d2R < 1e-3f) {
        reasonOut = "degenerate_radii";
        return false;
    }
    const float sizeRatio = std::max(d1R, d2R) / std::min(d1R, d2R);
    if (sizeRatio > probConfig.bio_daughter_size_ratio_max) {
        reasonOut = "size_ratio_" + std::to_string(sizeRatio);
        return false;
    }

    // 2. Combined volume fraction. The reference volume is supplied by the
    // caller — typically computed from the snapshot (last-frame) parent radii,
    // not from the live parent, so Phase B shrinkage doesn't distort the
    // ratio.
    const double d1Vol = cellVolume(daughter1);
    const double d2Vol = cellVolume(daughter2);
    const double combinedVol = d1Vol + d2Vol;
    if (refParentVolume < 1e-6) {
        reasonOut = "parent_zero_volume";
        return false;
    }
    const double volFraction = combinedVol / refParentVolume;
    if (volFraction < probConfig.bio_combined_volume_min_fraction ||
        volFraction > probConfig.bio_combined_volume_max_fraction) {
        reasonOut = "volume_fraction_" + std::to_string(volFraction);
        return false;
    }

    // 2b. Single-daughter volume gate. For a real division each daughter is
    // ~0.5 * parent vol. If one daughter > bio_max_single_daughter_volume_fraction
    // of parent, it's inheriting the parent ("asymmetric mimic" pattern).
    const double maxDaughterFraction =
        std::max(d1Vol, d2Vol) / refParentVolume;
    if (maxDaughterFraction > probConfig.bio_max_single_daughter_volume_fraction) {
        reasonOut = "single_daughter_volume_" + std::to_string(maxDaughterFraction);
        return false;
    }

    // 3. Daughters not buried in each other or in any other cell.
    for (size_t i = 0; i < allCells.size(); ++i) {
        if (i == d1Idx || i == d2Idx) continue;
        const Ellipsoid &other = allCells[i];
        if (other.isPointInsideEllipsoid(cv::Point3f(
                daughter1.getX(), daughter1.getY(), daughter1.getZ()), 1.0f)) {
            reasonOut = "d1_buried_in_" + other.getName();
            return false;
        }
        if (other.isPointInsideEllipsoid(cv::Point3f(
                daughter2.getX(), daughter2.getY(), daughter2.getZ()), 1.0f)) {
            reasonOut = "d2_buried_in_" + other.getName();
            return false;
        }
    }
    // Sibling buried check — treat the other daughter as "another cell".
    if (daughter2.isPointInsideEllipsoid(cv::Point3f(
            daughter1.getX(), daughter1.getY(), daughter1.getZ()), 1.0f)) {
        reasonOut = "d1_buried_in_sibling";
        return false;
    }
    if (daughter1.isPointInsideEllipsoid(cv::Point3f(
            daughter2.getX(), daughter2.getY(), daughter2.getZ()), 1.0f)) {
        reasonOut = "d2_buried_in_sibling";
        return false;
    }

    // 4. Neighbor-bridging check: reject if either daughter's center is
    // closer to an existing neighbor than to its sibling. This catches
    // false splits where the cell stretches to a neighbor's bright blob
    // and the bridge gate sees a valley between them — the "split" is
    // really the cell covering two separate cells, not dividing.
    const cv::Point3f d1Pos(daughter1.getX(), daughter1.getY(), daughter1.getZ());
    const cv::Point3f d2Pos(daughter2.getX(), daughter2.getY(), daughter2.getZ());
    const float siblingDist = static_cast<float>(cv::norm(d2Pos - d1Pos));
    for (size_t i = 0; i < allCells.size(); ++i) {
        if (i == d1Idx || i == d2Idx) continue;
        const Ellipsoid &other = allCells[i];
        const cv::Point3f oPos(other.getX(), other.getY(), other.getZ());
        const float d1ToOther = static_cast<float>(cv::norm(oPos - d1Pos));
        const float d2ToOther = static_cast<float>(cv::norm(oPos - d2Pos));
        if (d1ToOther < siblingDist * 0.5f) {
            reasonOut = "d1_bridging_to_" + other.getName();
            return false;
        }
        if (d2ToOther < siblingDist * 0.5f) {
            reasonOut = "d2_bridging_to_" + other.getName();
            return false;
        }
    }

    return true;
}

// Build a daughter Ellipsoid at position `center` with radii scaled from the
// given source radii by `volumeScale` and clamped to the config bounds.
// `srcMajor/srcB/srcMinor` are the reference dimensions the daughter should
// inherit from — typically the parent's last-frame snapshot values, so that
// Phase B perturbations that shrink the live parent do not also shrink the
// daughters below what the image actually supports.
Ellipsoid buildDaughter(
    const std::string &name,
    const cv::Point3f &center,
    const Ellipsoid &parent,
    float volumeScale,
    float srcMajor,
    float srcB,
    float srcMinor)
{
    const auto &cfg = Ellipsoid::cellConfig;
    const float dMajor = std::clamp(
        srcMajor * volumeScale,
        static_cast<float>(cfg.minARadius),
        static_cast<float>(cfg.maxARadius));
    const float dB = std::clamp(
        srcB * volumeScale,
        static_cast<float>(cfg.maxBRadius > 0.0 ? cfg.minBRadius : cfg.minARadius),
        static_cast<float>(cfg.maxBRadius > 0.0 ? cfg.maxBRadius : cfg.maxARadius));
    const float dMinor = std::clamp(
        srcMinor * volumeScale,
        static_cast<float>(cfg.minCRadius),
        static_cast<float>(cfg.maxCRadius));

    EllipsoidParams dp(
        name,
        center.x, center.y, center.z,
        dMajor, dMinor,
        parent.getCellParams().theta_x,
        parent.getCellParams().theta_y,
        parent.getCellParams().theta_z,
        parent.getBrightness());
    dp.bRadius = dB;
    return Ellipsoid(dp);
}

} // namespace

// Frame-start pre-pass: PCA-ground the expected daughter positions.
// See plan 2026-04-10-triaxial-pipeline-redesign.md "PRE-PASS" block.
// Shares the gather + PCA helpers with trySplitCellPhased above (both
// defined in the anonymous namespace of this TU) — this is the "cheap
// PCA-only path, no candidate burn-in" version.
bool Frame::imageGroundExpectedDaughters(
    size_t cellIndex,
    const PreviousFrameSnapshot &snapshot,
    const ClaimSet &otherCellsClaimSets,
    cv::Point3f &outD1,
    cv::Point3f &outD2,
    int *outKeptPixels) const
{
    if (outKeptPixels) *outKeptPixels = 0;
    if (cellIndex >= cells.size()) return false;
    if (!snapshot.valid) return false;

    // Self claim points: same two-seed pattern as trySplitCellPhased uses.
    //   D1_seed = snapshot.center - 0.5 * splitAxisLength * splitAxisDir
    //   D2_seed = snapshot.center + 0.5 * splitAxisLength * splitAxisDir
    std::vector<cv::Point3f> selfClaim;
    if (snapshot.splitAxisLength > 1e-3f) {
        const float half = 0.5f * snapshot.splitAxisLength;
        selfClaim.push_back(cv::Point3f(
            snapshot.position.x - half * snapshot.splitAxisDir.x,
            snapshot.position.y - half * snapshot.splitAxisDir.y,
            snapshot.position.z - half * snapshot.splitAxisDir.z));
        selfClaim.push_back(cv::Point3f(
            snapshot.position.x + half * snapshot.splitAxisDir.x,
            snapshot.position.y + half * snapshot.splitAxisDir.y,
            snapshot.position.z + half * snapshot.splitAxisDir.z));
    } else {
        selfClaim.push_back(snapshot.position);
    }

    // Bounding box radius: 3x the largest snapshot semi-axis, same as the
    // split path's box radius. Uses snapshot (not live) radii so the
    // pre-pass sees the same region of space regardless of how Phase A
    // has since moved the parent.
    const float srcMajor = snapshot.aRadius;
    const float srcB = (snapshot.bRadius > 1e-3f) ? snapshot.bRadius : snapshot.aRadius;
    const float srcMinor = snapshot.cRadius;
    const float boxRadius = 3.0f * std::max({srcMajor, srcB, srcMinor});

    GatherStats gstats;
    auto pixels = gatherBrightPixelsVoronoi(
        _realFrame,
        _backgroundValue,
        snapshot.position,
        boxRadius,
        selfClaim,
        otherCellsClaimSets,
        &gstats);

    if (outKeptPixels) *outKeptPixels = gstats.voronoiKept;
    if (pixels.size() < 20) return false;

    // PCA in cell's local frame using snapshot rotation + radii.
    const Ellipsoid &cell = cells[cellIndex];
    std::array<double, 9> R_T;
    cell.generateInverseRotationMatrix(R_T);
    const cv::Point3f center(cell.getX(), cell.getY(), cell.getZ());

    cv::Point3f dirPca;
    if (!pca3DWithCentroids(pixels, dirPca, outD1, outD2,
                            &center, &R_T,
                            cell.getARadius(), cell.getBRadius(), cell.getCRadius()))
        return false;
    return true;
}

// Position-only calibration via weighted bright-pixel centroid.
// The centroid is the analytic midpoint-between-daughters for a dividing
// cell (pixels split into two clusters contribute equally to the mean)
// and the true center for a non-dividing cell. This is the same quantity
// pca3DWithCentroids computes internally before the eigenvector split.
bool Frame::calibrateCellPositionViaCentroid(
    size_t cellIndex,
    const PreviousFrameSnapshot &snapshot,
    const ClaimSet &otherCellsClaimSets)
{
    if (cellIndex >= cells.size()) return false;
    if (!snapshot.valid) return false;

    // Self claim points — same two-seed pattern as the split path.
    std::vector<cv::Point3f> selfClaim;
    if (snapshot.splitAxisLength > 1e-3f) {
        const float half = 0.5f * snapshot.splitAxisLength;
        selfClaim.push_back(cv::Point3f(
            snapshot.position.x - half * snapshot.splitAxisDir.x,
            snapshot.position.y - half * snapshot.splitAxisDir.y,
            snapshot.position.z - half * snapshot.splitAxisDir.z));
        selfClaim.push_back(cv::Point3f(
            snapshot.position.x + half * snapshot.splitAxisDir.x,
            snapshot.position.y + half * snapshot.splitAxisDir.y,
            snapshot.position.z + half * snapshot.splitAxisDir.z));
    } else {
        selfClaim.push_back(snapshot.position);
    }

    const float srcMajor = snapshot.aRadius;
    const float srcB = (snapshot.bRadius > 1e-3f) ? snapshot.bRadius : snapshot.aRadius;
    const float srcMinor = snapshot.cRadius;
    const float boxRadius = 3.0f * std::max({srcMajor, srcB, srcMinor});

    GatherStats gstats;
    auto pixels = gatherBrightPixelsVoronoi(
        _realFrame,
        _backgroundValue,
        snapshot.position,
        boxRadius,
        selfClaim,
        otherCellsClaimSets,
        &gstats);

    if (pixels.size() < 20) {
        std::cout << "  [Centroid Calibration] cell=" << cells[cellIndex].getName()
                  << " skipped (too_few_pixels=" << pixels.size() << ")" << std::endl;
        return false;
    }

    // Weighted mean of bright pixel positions — the analytic midpoint.
    double sumWx = 0.0, sumWy = 0.0, sumWz = 0.0, sumW = 0.0;
    for (const auto &bp : pixels) {
        sumWx += static_cast<double>(bp.pos.x) * bp.weight;
        sumWy += static_cast<double>(bp.pos.y) * bp.weight;
        sumWz += static_cast<double>(bp.pos.z) * bp.weight;
        sumW  += bp.weight;
    }
    if (sumW < 1e-6) {
        std::cout << "  [Centroid Calibration] cell=" << cells[cellIndex].getName()
                  << " skipped (sum_weight=0)" << std::endl;
        return false;
    }
    const cv::Point3f centroid(
        static_cast<float>(sumWx / sumW),
        static_cast<float>(sumWy / sumW),
        static_cast<float>(sumWz / sumW));

    const Ellipsoid savedCell = cells[cellIndex];
    const cv::Point3f currentPos(savedCell.getX(), savedCell.getY(), savedCell.getZ());
    const float moveDist = static_cast<float>(cv::norm(centroid - currentPos));

    // Skip the cost comparison if the centroid is essentially where the
    // cell already is — no point moving.
    if (moveDist < 0.5f) {
        std::cout << "  [Centroid Calibration] cell=" << savedCell.getName()
                  << " centroid=(" << centroid.x << "," << centroid.y << "," << centroid.z << ")"
                  << " currentPos=(" << currentPos.x << "," << currentPos.y << "," << currentPos.z << ")"
                  << " dist=" << moveDist << " kept=" << gstats.voronoiKept
                  << " result=no_move" << std::endl;
        return false;
    }

    // Build candidate at centroid position — radii/rotation/brightness
    // inherited from the current cell (only position changes).
    EllipsoidParams cp = savedCell.getCellParams();
    cp.x = centroid.x;
    cp.y = centroid.y;
    cp.z = centroid.z;
    Ellipsoid candidate(cp);

    const double savedCost = _currentCost;
    const std::vector<cv::Mat> savedSynth = _synthFrame;
    const std::vector<double> savedPerSlice = _currentCostPerSlice;

    // Install candidate and compute incremental cost
    cells[cellIndex] = candidate;
    int affMin = -1, affMax = -1;
    Ellipsoid savedMutable = savedCell;
    Ellipsoid candidateMutable = candidate;
    auto newSynth = generateSynthFrameFast(savedMutable, candidateMutable, &affMin, &affMax);
    std::vector<double> newPerSlice;
    const double newCost = calculateIncrementalCost(newSynth, affMin, affMax, newPerSlice);

    if (newCost < savedCost) {
        _synthFrame = newSynth;
        _currentCost = newCost;
        _currentCostPerSlice = newPerSlice;
        std::cout << "  [Centroid Calibration] cell=" << savedCell.getName()
                  << " ACCEPTED centroid=(" << centroid.x << "," << centroid.y << "," << centroid.z << ")"
                  << " currentPos=(" << currentPos.x << "," << currentPos.y << "," << currentPos.z << ")"
                  << " dist=" << moveDist
                  << " costDelta=" << (newCost - savedCost)
                  << " kept=" << gstats.voronoiKept
                  << std::endl;
        return true;
    }

    // Revert — keep current position (which may or may not be the raw
    // snapshot; we preserve whatever state was live at entry).
    cells[cellIndex] = savedCell;
    _synthFrame = savedSynth;
    _currentCost = savedCost;
    _currentCostPerSlice = savedPerSlice;
    std::cout << "  [Centroid Calibration] cell=" << savedCell.getName()
              << " REJECTED centroid=(" << centroid.x << "," << centroid.y << "," << centroid.z << ")"
              << " currentPos=(" << currentPos.x << "," << currentPos.y << "," << currentPos.z << ")"
              << " dist=" << moveDist
              << " costDelta=" << (newCost - savedCost)
              << " kept=" << gstats.voronoiKept
              << std::endl;
    return false;
}

// ----- PCA shape-from-image helpers -----
//
// Decompose a proper-rotation matrix R (world <- local columns) into Euler
// angles with convention R = Rz(tz) * Ry(ty) * Rx(tx). Matches
// Ellipsoid::generateInverseRotationMatrix. Handles gimbal lock.
static void rotationMatrixToEulerZYX(const cv::Matx33d &R,
                                     double &tx, double &ty, double &tz)
{
    const double s = std::clamp(-R(2, 0), -1.0, 1.0);
    ty = std::asin(s);
    const double c = std::cos(ty);
    if (std::abs(c) > 1e-6) {
        tx = std::atan2(R(2, 1), R(2, 2));
        tz = std::atan2(R(1, 0), R(0, 0));
    } else {
        tx = std::atan2(-R(1, 2), R(1, 1));
        tz = 0.0;
    }
}


bool Frame::calibrateCellShapeViaPca(
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
    float maskA,
    float maskB,
    float maskC,
    std::ostream *logSink)
{
    if (cellIndex >= cells.size()) return false;
    if (maxIters <= 0) return false;
    Ellipsoid &cell = cells[cellIndex];

    // Route per-iter logs to the optional sink (per-thread accumulator)
    // when provided, else to std::cout. Lets the parallelized caller emit
    // deterministic per-cell log blocks in cell-index order.
    std::ostream &log = logSink ? *logSink : std::cout;

    const float convergeAngleRad =
        convergeAngleDeg * static_cast<float>(M_PI) / 180.0f;

    // Fixed mask radii (birth radii, or fallback to live). Mask stays
    // constant across iterations so the pixel-collection scope does not
    // collapse as the fitted radii shrink — prevents the mask-feedback
    // collapse where a shrinking cell latches onto one emerging daughter.
    const float effMaskA = (maskA > 0.0f) ? maskA : cell.getARadius();
    const float effMaskB = (maskB > 0.0f) ? maskB : cell.getBRadius();
    const float effMaskC = (maskC > 0.0f) ? maskC : cell.getCRadius();
    const float maskMaxR = std::max({effMaskA, effMaskB, effMaskC});
    const float sphereR = maskScale * maskMaxR;
    const double invA2Fixed = 1.0 / (maskScale * effMaskA * maskScale * effMaskA);
    const double invB2Fixed = 1.0 / (maskScale * effMaskB * maskScale * effMaskB);
    const double invC2Fixed = 1.0 / (maskScale * effMaskC * maskScale * effMaskC);

    bool anyUpdate = false;

    // P4 — bright-pixel gather cache.
    //
    // gatherBrightPixelsVoronoi scans a 3D box of `sphereR` around `center`,
    // applies brightness cutoff, then Voronoi-filters against neighbor
    // claim points. The expensive step is the box scan + Voronoi test —
    // O(boxVolume * nNeighbors). Per shape-fit iteration the gather inputs
    // are mostly invariant: `_realFrame`, `_backgroundValue`, `sphereR`,
    // `otherCellsClaimSets` never change; only `center` can change (and
    // only when `updatePosition=true`). Cache the raw pixel set keyed on
    // `center`; re-gather only when the centroid moves.
    cv::Point3f cachedCenter{std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN()};
    std::vector<BrightPixel> cachedRaw;
    int cachedHits = 0;
    int cachedMisses = 0;

    // Adaptive exponent: bright core-dominated cells get a lower exponent
    // (more halo-inclusive fit, counters core-only shrinkage); dim/uniform
    // cells keep the default exponent. Recomputed on each cache miss (same
    // scope as the raw gather — invariant across iterations otherwise).
    const bool   adaptiveExp = Ellipsoid::cellConfig.pcaShapeAdaptiveExponent;
    const float  expDim      = std::max(0.0f, Ellipsoid::cellConfig.pcaShapeWeightExponent);
    const float  expBright   = std::max(0.0f, Ellipsoid::cellConfig.pcaShapeWeightExponentBright);
    const float  coreT       = Ellipsoid::cellConfig.pcaShapeCoreBrightnessThreshold;
    const float  coreLo      = Ellipsoid::cellConfig.pcaShapeCoreFractionLow;
    const float  coreHi      = std::max(coreLo + 1e-6f,
                                        Ellipsoid::cellConfig.pcaShapeCoreFractionHigh);
    const float  radInflBright = std::max(1.0f, Ellipsoid::cellConfig.pcaShapeRadiusInflationBright);
    float cellWeightExponent = expDim;
    // Per-cell radius inflation multiplier. 1.0 for uniform/dim cells
    // (PCA variance formula is analytically exact for them); ramps up to
    // radInflBright for peaked/bright-core cells whose visible halo
    // extends beyond the 97% containment radius. Driven by the same
    // pCore metric as adaptive exponent.
    float cellRadiusInflation = 1.0f;

    for (int iter = 0; iter < maxIters; ++iter) {
        const cv::Point3f center(cell.getX(), cell.getY(), cell.getZ());
        const float curA = cell.getARadius();
        const float curB = cell.getBRadius();
        const float curC = cell.getCRadius();
        const float maxR = std::max({curA, curB, curC});
        if (maxR <= 1e-3f) break;

        // Cache hit when center hasn't moved since last gather. With
        // updatePosition=false (recommended), this hits on every iter
        // after the first → 14× speedup on the gather phase.
        const bool cacheHit = (cachedRaw.size() > 0)
            && (std::abs(center.x - cachedCenter.x) < 1e-4f)
            && (std::abs(center.y - cachedCenter.y) < 1e-4f)
            && (std::abs(center.z - cachedCenter.z) < 1e-4f);
        if (!cacheHit) {
            std::vector<cv::Point3f> selfClaim{center};
            GatherStats gstats;
            cachedRaw = gatherBrightPixelsVoronoi(
                _realFrame, _backgroundValue, center, sphereR,
                selfClaim, otherCellsClaimSets, &gstats);
            cachedCenter = center;
            ++cachedMisses;

            // Adaptive exponent: measure core-dominance on the freshly
            // gathered raw pixels. Dim cells stay at expDim; bright cells
            // ramp down toward expBright to give halo a fairer vote in
            // the weighted moments.
            if (adaptiveExp && !cachedRaw.empty()) {
                // Relative pCore: fraction of pixels whose weight exceeds
                // 1.5× the cell's own mean weight. Scale-invariant — a
                // peaked cell (bright core, dim halo) has many pixels
                // above 1.5×mean; a uniform cell has few. Doesn't depend
                // on absolute brightness scale or preprocessing pipeline.
                //
                // Replaces the previous absolute-threshold metric
                // (pcaShapeCoreBrightnessThreshold) which failed for this
                // dataset: dim cells got HIGH pCore (many pixels barely
                // above threshold) while bright cells got LOW pCore
                // (pixels spread across a wide range). The relative
                // metric correctly classifies both.
                float meanW = 0.0f;
                for (const auto &bp : cachedRaw) meanW += bp.weight;
                meanW /= static_cast<float>(cachedRaw.size());

                int coreCount = 0;
                const float relativeThreshold = 1.5f * meanW;
                for (const auto &bp : cachedRaw) {
                    if (bp.weight > relativeThreshold) ++coreCount;
                }
                const float pCore = static_cast<float>(coreCount) /
                                    static_cast<float>(cachedRaw.size());
                const float t = std::clamp(
                    (pCore - coreLo) / (coreHi - coreLo), 0.0f, 1.0f);
                cellWeightExponent = expDim + t * (expBright - expDim);
                // Same pCore ramp drives radius inflation: 1.0 for uniform
                // (t=0), radInflBright for peaked (t=1).
                cellRadiusInflation = 1.0f + t * (radInflBright - 1.0f);
                log << "  [PCA Shape Exp] cell=" << cell.getName()
                    << " pCore=" << pCore
                    << " meanW=" << meanW
                    << " relThresh=" << relativeThreshold
                    << " exp=" << cellWeightExponent
                    << " radInfl=" << cellRadiusInflation
                    << std::endl;
            }
        } else {
            ++cachedHits;
        }
        const std::vector<BrightPixel> &raw = cachedRaw;

        // Ellipsoid mask uses FIXED radii (snap), not live, so it cannot
        // tighten between iterations — prevents mask-feedback collapse.
        // Rotation still uses the cell's CURRENT rotation so the mask
        // orientation follows the fit as it converges.
        std::array<double, 9> R_T;
        cell.generateInverseRotationMatrix(R_T);
        const double invA2 = invA2Fixed;
        const double invB2 = invB2Fixed;
        const double invC2 = invC2Fixed;

        std::vector<BrightPixel> pixels;
        pixels.reserve(raw.size());
        for (const auto &bp : raw) {
            const double dx = bp.pos.x - center.x;
            const double dy = bp.pos.y - center.y;
            const double dz = bp.pos.z - center.z;
            const double lx = R_T[0] * dx + R_T[1] * dy + R_T[2] * dz;
            const double ly = R_T[3] * dx + R_T[4] * dy + R_T[5] * dz;
            const double lz = R_T[6] * dx + R_T[7] * dy + R_T[8] * dz;
            const double val = lx * lx * invA2 + ly * ly * invB2 + lz * lz * invC2;
            if (val <= 1.0) pixels.push_back(bp);
        }

        if (static_cast<int>(pixels.size()) < minPixels) {
            log << "  [PCA Shape] cell=" << cell.getName()
                      << " iter=" << iter
                      << " stop_too_few pixels=" << pixels.size()
                      << " min=" << minPixels << std::endl;
            break;
        }

        // Intensity-weighted centroid + covariance with configurable exponent.
        //   exp=1.0: linear (historical behavior, halo can bloat radii).
        //   exp=2.0: quadratic (current default; suppresses halo ~25× vs core).
        //   exp=higher: stronger core emphasis.
        // Cached per-pixel weight is `weight_eff = pow(bp.weight, exp)`.
        // Fast paths for exp∈{1, 2} avoid the std::pow call.
        // `cellWeightExponent` is per-cell adaptive when enabled (set above
        // on cache miss), else equals `pcaShapeWeightExponent`.
        const float weightExponent = cellWeightExponent;
        const bool exp1 = std::abs(weightExponent - 1.0f) < 1e-6f;
        const bool exp2 = std::abs(weightExponent - 2.0f) < 1e-6f;

        auto effectiveWeight = [&](float w) -> double {
            if (exp1) return static_cast<double>(w);
            if (exp2) return static_cast<double>(w) * w;
            return std::pow(static_cast<double>(w), static_cast<double>(weightExponent));
        };

        // Cache per-pixel effective weights to avoid recomputing pow()
        // in the covariance loop below (saves N pow() calls where N is
        // typically 500-5000 pixels per PCA iteration).
        std::vector<double> pixelWeights(pixels.size());
        double sx = 0, sy = 0, sz = 0, wsum = 0;
        for (size_t pi = 0; pi < pixels.size(); ++pi) {
            const double we = effectiveWeight(pixels[pi].weight);
            pixelWeights[pi] = we;
            sx += pixels[pi].pos.x * we;
            sy += pixels[pi].pos.y * we;
            sz += pixels[pi].pos.z * we;
            wsum += we;
        }
        if (wsum < 1e-6) break;
        const double mx = sx / wsum, my = sy / wsum, mz = sz / wsum;

        double cxx = 0, cxy = 0, cxz = 0, cyy = 0, cyz = 0, czz = 0;
        for (size_t pi = 0; pi < pixels.size(); ++pi) {
            const double dx = pixels[pi].pos.x - mx;
            const double dy = pixels[pi].pos.y - my;
            const double dz = pixels[pi].pos.z - mz;
            const double we = pixelWeights[pi];
            cxx += we * dx * dx; cxy += we * dx * dy; cxz += we * dx * dz;
            cyy += we * dy * dy; cyz += we * dy * dz;
            czz += we * dz * dz;
        }
        cxx /= wsum; cxy /= wsum; cxz /= wsum;
        cyy /= wsum; cyz /= wsum; czz /= wsum;

        cv::Matx33d cov(cxx, cxy, cxz, cxy, cyy, cyz, cxz, cyz, czz);
        cv::Matx33d eigvecs;
        cv::Vec3d eigvals;
        cv::eigen(cov, eigvals, eigvecs);

        cv::Point3f pcaAxis[3];
        double pcaVariance[3];
        for (int i = 0; i < 3; ++i) {
            pcaAxis[i] = cv::Point3f(
                static_cast<float>(eigvecs(i, 0)),
                static_cast<float>(eigvecs(i, 1)),
                static_cast<float>(eigvecs(i, 2)));
            pcaVariance[i] = std::max(0.0, eigvals[i]);
        }

        // Eigenvalue degeneracy: cell is near-spherical. PCA rotation is
        // noisy. Skip rotation update and only refresh radii/position.
        const double maxVar = std::max({pcaVariance[0], pcaVariance[1], pcaVariance[2]});
        const double minVar = std::min({pcaVariance[0], pcaVariance[1], pcaVariance[2]});
        const bool degenerate = (minVar <= 1e-6) || (maxVar / minVar < 1.1);

        // Current cell axes (columns of R) in world frame.
        // R_T is R^T stored row-major (R_T[r*3+c] = R[c,r]); column i of R
        // = (R_T[3i], R_T[3i+1], R_T[3i+2]). See worldSplitAxis comment for
        // the 2026-04-19 indexing bug fix history.
        cv::Point3f curAxis[3];
        for (int i = 0; i < 3; ++i) {
            const int base = 3 * i;
            curAxis[i] = cv::Point3f(
                static_cast<float>(R_T[base]),
                static_cast<float>(R_T[base + 1]),
                static_cast<float>(R_T[base + 2]));
        }

        // Strict eigenvalue-rank assignment: a = largest variance, b = middle,
        // c = smallest. Greedy |dot| matching oscillated when eigenvalue
        // ordering differed from current slot labeling, because matched
        // variances cycled between slots each iteration. Rank assignment is
        // stable — physical variance ranks are invariant to the label rotation.
        cv::Point3f matchedAxis[3];
        double matchedVariance[3];
        double maxAxisAngle = 0.0;
        for (int ci = 0; ci < 3; ++ci) {
            cv::Point3f v = pcaAxis[ci];
            // Sign-align with current slot direction for rotation continuity.
            const double dot = curAxis[ci].x * v.x + curAxis[ci].y * v.y + curAxis[ci].z * v.z;
            if (dot < 0.0) { v.x = -v.x; v.y = -v.y; v.z = -v.z; }
            matchedAxis[ci] = v;
            matchedVariance[ci] = pcaVariance[ci];
            const double ang = std::acos(std::clamp(std::abs(dot), 0.0, 1.0));
            if (ang > maxAxisAngle) maxAxisAngle = ang;
        }

        double targetTx = cell.getThetaX();
        double targetTy = cell.getThetaY();
        double targetTz = cell.getThetaZ();
        if (!degenerate) {
            cv::Matx33d R(
                matchedAxis[0].x, matchedAxis[1].x, matchedAxis[2].x,
                matchedAxis[0].y, matchedAxis[1].y, matchedAxis[2].y,
                matchedAxis[0].z, matchedAxis[1].z, matchedAxis[2].z);
            if (cv::determinant(R) < 0.0) {
                R(0, 2) = -R(0, 2); R(1, 2) = -R(1, 2); R(2, 2) = -R(2, 2);
            }
            rotationMatrixToEulerZYX(R, targetTx, targetTy, targetTz);
        } else {
            maxAxisAngle = 0.0;
        }

        // Variance-based radii: radius = radiusScale × sqrt(weighted variance).
        //
        // The weighted variance along each axis is ALREADY computed by the
        // PCA covariance above (matchedVariance[i] = eigenvalue for axis i).
        // radiusScale = sqrt(5) ≈ 2.236 is the analytically correct
        // containment radius for a uniformly-filled ellipsoid.
        //
        // Why variance over percentile: the weight exponent (1.3) correctly
        // de-emphasizes halo for PCA rotation, but that same weighting
        // makes the weighted variance genuinely smaller along thin axes
        // (fewer bright pixels → lower weighted variance). This preserves
        // the elongation signal: e.g. 12345 at f2 gets c-axis variance
        // much smaller than a/b → elongation ~1.4 → correct split detection.
        //
        // Unweighted percentile lost this signal (halo extends equally on
        // all axes → cell looks round, elong=1.07). Weighted percentile
        // over-corrected (radii 40-50% too small, cells hit min floor).
        // Variance + radiusScale is the calibrated middle ground that the
        // best22 reference run (8/8 GT splits) validated.
        //
        // Per-cell adaptive inflation (cellRadiusInflation, from the pCore
        // ramp above) compensates for peaked cells where sqrt(5) × sqrt(var)
        // under-estimates the visible halo.
        //
        // Birth-based mask (introduced after the original variance formula)
        // prevents the mask-feedback loops that motivated the percentile
        // experiment. The mask is frozen at birth radii and never participates
        // in the fit → no compounding bloat.
        const float targetA = cellRadiusInflation * radiusScale *
                              std::sqrt(static_cast<float>(matchedVariance[0]));
        const float targetB = cellRadiusInflation * radiusScale *
                              std::sqrt(static_cast<float>(matchedVariance[1]));
        const float targetC = cellRadiusInflation * radiusScale *
                              std::sqrt(static_cast<float>(matchedVariance[2]));
        // No floor. Collapse prevented by birth-based mask (above).

        // Position update: centroid, capped.
        cv::Point3f newCenter = center;
        if (updatePosition) {
            const float dx = static_cast<float>(mx) - center.x;
            const float dy = static_cast<float>(my) - center.y;
            const float dz = static_cast<float>(mz) - center.z;
            const float shift = std::sqrt(dx * dx + dy * dy + dz * dz);
            const float cap = maxPosShiftFraction * maxR;
            if (shift > cap && shift > 1e-6f) {
                const float s = cap / shift;
                newCenter = cv::Point3f(center.x + s * dx,
                                        center.y + s * dy,
                                        center.z + s * dz);
            } else {
                newCenter = cv::Point3f(static_cast<float>(mx),
                                        static_cast<float>(my),
                                        static_cast<float>(mz));
            }
        }

        // Apply.
        cell.setRadii(targetA, targetB, targetC);
        cell.setRotation(static_cast<float>(targetTx),
                         static_cast<float>(targetTy),
                         static_cast<float>(targetTz));
        if (updatePosition) {
            EllipsoidParams p = cell.getCellParams();
            p.x = newCenter.x; p.y = newCenter.y; p.z = newCenter.z;
            cell = Ellipsoid(p);
        }
        anyUpdate = true;

        // Convergence check.
        const float dA = std::abs(targetA - curA);
        const float dB = std::abs(targetB - curB);
        const float dC = std::abs(targetC - curC);
        const float maxDR = std::max({dA, dB, dC});

        log << "  [PCA Shape] cell=" << cell.getName()
                  << " iter=" << iter
                  << " n=" << pixels.size()
                  << " degen=" << degenerate
                  << " R=(" << targetA << "," << targetB << "," << targetC << ")"
                  << " dR=" << maxDR
                  << " axisAng=" << (maxAxisAngle * 180.0 / M_PI)
                  << " posShift=" << cv::norm(newCenter - center)
                  << std::endl;

        if (maxDR < convergeRadius &&
            maxAxisAngle < convergeAngleRad) {
            break;
        }
    }

    if (cachedHits + cachedMisses > 0) {
        log << "  [PCA Shape Cache] cell=" << cell.getName()
                  << " hits=" << cachedHits
                  << " misses=" << cachedMisses
                  << " hitRate="
                  << (static_cast<float>(cachedHits) /
                      static_cast<float>(cachedHits + cachedMisses))
                  << std::endl;
    }

    return anyUpdate;
}

// Triaxial split attempt with candidate refinement + bio/cost gates.
// Implements the Phase A/B split-attempt flow from the plan. Caller supplies
// the Voronoi claim-sets for all OTHER cells (not this one) and whether to
// trust the snapshot direction as a candidate primary.
CostCallbackPair Frame::trySplitCellPhased(
    size_t cellIndex,
    const PreviousFrameSnapshot &snapshot,
    const ClaimSet &otherCellsClaimSets,
    bool useSnapshotDirection,
    const ProbabilityConfig &probConfig)
{
    const auto noop = [](bool) {};
    if (cellIndex >= cells.size()) return {0.0, noop};

    // --- 0. Save live parent and install snapshot-state parent ---
    //
    // The split attempt must compare daughters against a FULL-SIZE parent
    // at the snapshot position/rotation, not against the Phase A/B-drifted
    // live parent. Otherwise the cost delta is dominated by "daughters vs
    // already-collapsed parent" which is small even for real divisions.
    //
    // Strategy: save the live parent, construct a snapshot-state Ellipsoid,
    // install it at cells[cellIndex], update _synthFrame + _currentCost
    // incrementally via generateSynthFrameFast. Everything downstream
    // (parent local, baseline cost, savedCells, candidate loop) then sees
    // the snapshot-state parent. On rejection we restore the live parent
    // to keep Phase B's perturbation progress.
    const Ellipsoid liveParent = cells[cellIndex];
    const std::string parentName = liveParent.getName();

    const bool snapshotValid = snapshot.valid &&
        snapshot.aRadius > 1e-3f &&
        snapshot.cRadius > 1e-3f;
    const float srcMajor = snapshotValid ? snapshot.aRadius : liveParent.getARadius();
    const float srcB     = (snapshotValid && snapshot.bRadius > 1e-3f)
        ? snapshot.bRadius : liveParent.getBRadius();
    const float srcMinor = snapshotValid ? snapshot.cRadius : liveParent.getCRadius();

    // Build the snapshot-state parent: position, radii, rotation, and
    // brightness all come from the snapshot (falling back to live values
    // when snapshot is missing a field).
    Ellipsoid snapshotParent = liveParent;
    if (snapshotValid) {
        EllipsoidParams snapParams(parentName,
                                  snapshot.position.x, snapshot.position.y, snapshot.position.z,
                                  srcMajor, srcMinor,
                                  snapshot.thetaX, snapshot.thetaY, snapshot.thetaZ,
                                  snapshot.brightness > 0.0f ? snapshot.brightness
                                                             : liveParent.getBrightness());
        snapParams.bRadius = srcB;
        snapshotParent = Ellipsoid(snapParams);

        // Swap in snapshot parent, render the affected z-range, and
        // compare costs to decide which baseline (live vs snap) is better.
        cells[cellIndex] = snapshotParent;
        int affMinS = -1, affMaxS = -1;
        Ellipsoid liveMutable = liveParent;
        Ellipsoid snapshotMutable = snapshotParent;
        auto swappedSynth = generateSynthFrameFast(liveMutable, snapshotMutable,
                                                    &affMinS, &affMaxS);

        // Compare live vs snap using the CORRECT cost mode for this frame.
        // In bbox mode, _currentCost is stale (perturbCell doesn't update it),
        // so we compute a fresh bbox cost over a temporary bbox covering
        // both live and snap positions. In legacy mode, _currentCost is
        // maintained by perturbCell and calculateIncrementalCost is exact.
        double liveCostForComparison = 0.0;
        double snapCostForComparison = 0.0;
        std::vector<double> swappedPerSlice;
        double swappedImageCost = 0.0;
        if (_useBboxCost) {
            // Build a temporary bbox covering both live and snap positions.
            const float maxR = std::max({srcMajor, srcB, srcMinor,
                                         liveParent.getARadius(),
                                         liveParent.getBRadius(),
                                         liveParent.getCRadius()});
            // Union of live-parent bbox and snap-parent bbox.
            BoundingBox3D liveBbox = computeBboxAtPoint(
                cv::Point3f(liveParent.getX(), liveParent.getY(), liveParent.getZ()),
                maxR, _bboxMarginScale);
            BoundingBox3D snapBbox = computeBboxAtPoint(
                snapshot.position, maxR, _bboxMarginScale);
            BoundingBox3D cmpBbox;
            cmpBbox.xMin = std::min(liveBbox.xMin, snapBbox.xMin);
            cmpBbox.xMax = std::max(liveBbox.xMax, snapBbox.xMax);
            cmpBbox.yMin = std::min(liveBbox.yMin, snapBbox.yMin);
            cmpBbox.yMax = std::max(liveBbox.yMax, snapBbox.yMax);
            cmpBbox.zMin = std::min(liveBbox.zMin, snapBbox.zMin);
            cmpBbox.zMax = std::max(liveBbox.zMax, snapBbox.zMax);
            const std::vector<uint8_t> noMask;
            liveCostForComparison = calculateBboxCost(cmpBbox, _synthFrame, noMask);
            snapCostForComparison = calculateBboxCost(cmpBbox, swappedSynth, noMask);
        } else {
            liveCostForComparison = _currentCost;
            swappedImageCost = calculateIncrementalCost(swappedSynth,
                                                        affMinS, affMaxS,
                                                        swappedPerSlice);
            snapCostForComparison = swappedImageCost;
        }

        // Use min(liveCost, snapCost) as baseline. If the snapshot parent
        // is much worse than live (cell drifted between frames), keeping
        // the inflated snapshot baseline would make ANY daughter placement
        // look like an improvement — causing false splits. Using the
        // minimum ensures the split must beat the TIGHTER of the two fits.
        const bool useSnapshotBaseline = (snapCostForComparison <= liveCostForComparison);
        if (useSnapshotBaseline) {
            _synthFrame = swappedSynth;
            if (!_useBboxCost) {
                _currentCost = swappedImageCost;
                _currentCostPerSlice = swappedPerSlice;
            }
        } else {
            // Revert: live parent was a better fit, keep it as baseline.
            cells[cellIndex] = liveParent;
            // _synthFrame already reflects the live parent (never changed).
        }

        std::cout << "  [Split Snapshot Parent] " << parentName
                  << " livePos=(" << liveParent.getX() << "," << liveParent.getY() << "," << liveParent.getZ() << ")"
                  << " snapPos=(" << snapshot.position.x << "," << snapshot.position.y << "," << snapshot.position.z << ")"
                  << " liveR=(" << liveParent.getARadius() << "," << liveParent.getBRadius() << "," << liveParent.getCRadius() << ")"
                  << " snapR=(" << srcMajor << "," << srcB << "," << srcMinor << ")"
                  << " liveCost=" << liveCostForComparison
                  << " snapCost=" << snapCostForComparison
                  << " baseline=" << (useSnapshotBaseline ? "snapshot" : "live")
                  << (_useBboxCost ? " (bbox)" : " (full)")
                  << std::endl;

        // Update snapshotParent to match what was actually installed so
        // restoreLiveParent works correctly on rejection paths.
        if (!useSnapshotBaseline) {
            snapshotParent = liveParent;
        }
    }

    // Geometric reference for split candidate generation: ALWAYS use the
    // snapshot-state parent (per 2026-04-19 design rule: split candidates
    // use SNAP or PCA-derived data only, never LIVE).
    //
    // Rationale: live position/radii drift during in-frame perturbation. If
    // axis directions, radii, or fallback midpoints were derived from the
    // live cell, that drift would cascade into different candidate seeds
    // each iteration, making the split decision sensitive to perturbation
    // history (frame-3 12345 / e9077 regression in run 084534).
    //
    // cells[cellIndex] still holds whatever the cost comparison above chose
    // (snapshot or live) — that is the RENDERING baseline for the cost
    // delta, intentionally separate from the GEOMETRIC reference here.
    //
    // Falls back to the live cell only when no snapshot exists (frame 1,
    // newborn daughter post-split — but those are filtered earlier).
    Ellipsoid parent = snapshotValid ? snapshotParent : cells[cellIndex];

    // Restore-live-parent helper. Used on every rejection path (early
    // returns inside this function AND the callback's reject branch) to
    // undo the snapshot-state install so Phase B's live state isn't
    // lost. No-op if snapshot wasn't valid (no install happened).
    auto restoreLiveParent = [&]() {
        // Cleanup snap-bboxes + shared masks installed for daughter
        // candidates — safe to erase unconditionally (erase of absent key
        // is a no-op). Covers every reject path through restoreLiveParent.
        _snapBboxes.erase(parentName + "0");
        _snapBboxes.erase(parentName + "1");
        // _sharedMasks removed — no longer used (cost uses empty mask)
        if (!snapshotValid) return;
        if (cellIndex >= cells.size()) return;
        cells[cellIndex] = liveParent;
        int affMinR = -1, affMaxR = -1;
        Ellipsoid snapshotMutable = snapshotParent;
        Ellipsoid liveMutable = liveParent;
        auto revertedSynth = generateSynthFrameFast(snapshotMutable, liveMutable,
                                                     &affMinR, &affMaxR);
        _synthFrame = revertedSynth;
        // In bbox mode the full-image cache is never read for decisions and
        // is intentionally left stale (Change 1). Skip the incremental
        // recompute and the stale-seeded write entirely.
        if (!_useBboxCost) {
            std::vector<double> revertedPerSlice;
            const double revertedCost = calculateIncrementalCost(revertedSynth,
                                                                    affMinR, affMaxR,
                                                                    revertedPerSlice);
            _currentCost = revertedCost;
            _currentCostPerSlice = revertedPerSlice;
        }
    };

    // --- 1. Gather bright pixels in a snapshot-centered bounding box ---

    const float parentMajor = std::max(srcMajor, parent.getARadius());
    const float parentB     = std::max(srcB,     parent.getBRadius());
    const float parentMinor = std::max(srcMinor, parent.getCRadius());
    const float boxRadius = 3.0f * std::max({parentMajor, parentB, parentMinor});

    // Reference parent volume used by the bio volume-fraction check and by
    // the drift gate. Uses source (snapshot when available) radii so a
    // shrunken live parent doesn't skew either ratio.
    const double refParentVolume =
        static_cast<double>(srcMajor) *
        static_cast<double>(srcB) *
        static_cast<double>(srcMinor);
    const float srcMaxR = std::max({srcMajor, srcB, srcMinor});

    std::cout << "[Split Attempt] " << parentName
              << " useSnapshotDir=" << (useSnapshotDirection ? 1 : 0)
              << " snapValid=" << (snapshotValid ? 1 : 0)
              << " snapElong=" << snapshot.shapeElongation
              << " snapLongLen=" << snapshot.splitAxisLength
              << " src=(" << srcMajor << "," << srcB << "," << srcMinor << ")"
              << " liveR=(" << liveParent.getARadius() << "," << liveParent.getBRadius() << "," << liveParent.getCRadius() << ")"
              << " livePos=(" << liveParent.getX() << "," << liveParent.getY() << "," << liveParent.getZ() << ")"
              << " parentNow=(" << parent.getARadius() << "," << parent.getBRadius() << "," << parent.getCRadius() << ")"
              << " parentPos=(" << parent.getX() << "," << parent.getY() << "," << parent.getZ() << ")"
              << std::endl;

    // Self claim points for Voronoi test: the two expected-daughter seeds
    // along the snapshot long axis (if we have one) or just the snapshot
    // center (for round cells).
    //
    //   D1_seed = snapshot.center - 0.5 * splitAxisLength * splitAxisDir
    //   D2_seed = snapshot.center + 0.5 * splitAxisLength * splitAxisDir
    std::vector<cv::Point3f> selfClaim;
    if (snapshot.splitAxisLength > 1e-3f) {
        const float half = 0.5f * snapshot.splitAxisLength;
        selfClaim.push_back(cv::Point3f(
            snapshot.position.x - half * snapshot.splitAxisDir.x,
            snapshot.position.y - half * snapshot.splitAxisDir.y,
            snapshot.position.z - half * snapshot.splitAxisDir.z));
        selfClaim.push_back(cv::Point3f(
            snapshot.position.x + half * snapshot.splitAxisDir.x,
            snapshot.position.y + half * snapshot.splitAxisDir.y,
            snapshot.position.z + half * snapshot.splitAxisDir.z));
        std::cout << "  [Split Seeds] " << parentName
                  << " snapCenter=(" << snapshot.position.x << "," << snapshot.position.y << "," << snapshot.position.z << ")"
                  << " splitAxisDir=(" << snapshot.splitAxisDir.x << "," << snapshot.splitAxisDir.y << "," << snapshot.splitAxisDir.z << ")"
                  << " splitAxisLen=" << snapshot.splitAxisLength
                  << " D1_seed=(" << selfClaim[0].x << "," << selfClaim[0].y << "," << selfClaim[0].z << ")"
                  << " D2_seed=(" << selfClaim[1].x << "," << selfClaim[1].y << "," << selfClaim[1].z << ")"
                  << " boxRadius=" << boxRadius
                  << std::endl;
    } else {
        selfClaim.push_back(snapshot.position);
        std::cout << "  [Split Seeds] " << parentName
                  << " snapCenter=(" << snapshot.position.x << "," << snapshot.position.y << "," << snapshot.position.z << ")"
                  << " splitAxisLen=0 (round cell, single seed = snapCenter)"
                  << " boxRadius=" << boxRadius
                  << std::endl;
    }

    // Voronoi exclusion diagnostic — summarize the other-cell claim set.
    {
        size_t otherCellCount = 0;
        size_t otherPointCount = 0;
        std::ostringstream oc;
        for (const auto &kv : otherCellsClaimSets) {
            ++otherCellCount;
            otherPointCount += kv.second.size();
            oc << " " << kv.first << "[" << kv.second.size() << "]";
        }
        std::cout << "  [Voronoi In] " << parentName
                  << " otherCells=" << otherCellCount
                  << " otherPoints=" << otherPointCount
                  << " selfPoints=" << selfClaim.size()
                  << " others=" << oc.str()
                  << std::endl;
    }

    GatherStats gstats;
    std::vector<BrightPixel> pixels = gatherBrightPixelsVoronoi(
        _realFrame,
        _backgroundValue,
        snapshot.position,
        boxRadius,
        selfClaim,
        otherCellsClaimSets,
        &gstats);

    std::cout << "  [Voronoi Out] " << parentName
              << " box=" << gstats.boxVoxels
              << " inSphere=" << gstats.inSphere
              << " aboveBright=" << gstats.aboveBrightness
              << " voronoiRejected=" << gstats.voronoiRejected
              << " kept=" << gstats.voronoiKept
              << std::endl;

    if (pixels.size() < 20) {
        std::cout << "[Split Reject] " << parentName
                  << " too_few_bright_pixels=" << pixels.size() << std::endl;
        restoreLiveParent();
        return {0.0, noop};
    }

    // --- 2. Local-axis directions + data-driven daughter placement ---
    //   Instead of PCA (whose direction is dominated by the z-brightness
    //   gradient for flat cells), try ALL THREE cell-local axes (a, b, c)
    //   rotated to world frame. For each axis, project bright pixels onto
    //   that direction and compute the centroid of each half — this gives a
    //   data-driven midpoint and separation. Cost picks the winning axis.
    std::array<double, 9> parentR_T;
    parent.generateInverseRotationMatrix(parentR_T);
    const cv::Point3f parentCenter(parent.getX(), parent.getY(), parent.getZ());

    // Extract all three local axes in world frame from the inverse rotation
    // matrix. R_T is R^T stored row-major (R_T[r*3+c] = R[c,r]); the world
    // direction of local axis i is column i of R = (R_T[3i], R_T[3i+1],
    // R_T[3i+2]). See worldSplitAxis comment for the 2026-04-19 indexing
    // bug fix history.
    auto extractWorldAxis = [&](int axisIdx) -> cv::Point3f {
        const int base = 3 * axisIdx;
        const double dx = parentR_T[base];
        const double dy = parentR_T[base + 1];
        const double dz = parentR_T[base + 2];
        const double norm = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (norm < 1e-9) return {0.0f, 0.0f, 1.0f};
        return cv::Point3f(
            static_cast<float>(dx / norm),
            static_cast<float>(dy / norm),
            static_cast<float>(dz / norm));
    };
    const cv::Point3f axisA = extractWorldAxis(0); // local x (a-axis) in world
    const cv::Point3f axisB = extractWorldAxis(1); // local y (b-axis) in world
    const cv::Point3f axisC = extractWorldAxis(2); // local z (c-axis) in world

    // Only try axes that are close to the shortest radius. Cells split
    // through their thin dimension. Axes much longer than the shortest
    // create false splits by placing daughters end-to-end along a long
    // direction, matching the z-brightness gradient instead of a real
    // division (e3d03 false splits at f2 via axA and axC).
    //
    // Include ONLY the single shortest local axis as a primary direction.
    //
    // Data analysis from run 002144 (196 split winners across burn-in,
    // 6 accepted splits):
    //   Axis   | Wins | Accepts
    //   -------|------|--------
    //   axC    | 100  |   1
    //   imgPca |  75  |   5
    //   axB    |  19  |   0
    //   axA    |   2  |   0
    //
    // ZERO accepts came from axB or axA — they only competed in near-
    // round cells (where the 1.2× threshold admitted them) and never
    // beat axC or imgPca in the final accept. Biologically, cells
    // divide along their SHORTEST axis; axA and axB were candidate-set
    // bloat. Dropping them reduces near-round-cell candidates from 3
    // axes × 10 variants = 30 to 2 axes × 10 = 20 (~33% saving), with
    // zero accuracy loss.
    const float rA = parent.getARadius();
    const float rB = parent.getBRadius();
    const float rC = parent.getCRadius();
    const float radii3[] = {rA, rB, rC};
    const cv::Point3f axes3[] = {axisA, axisB, axisC};
    const std::string names3[] = {"axA", "axB", "axC"};

    int shortIdx = 0;
    for (int i = 1; i < 3; ++i) {
        if (radii3[i] < radii3[shortIdx]) shortIdx = i;
    }
    std::vector<cv::Point3f> primaryDirs{axes3[shortIdx]};
    std::vector<std::string> primaryNames{names3[shortIdx]};

    // Add the image-PCA direction (from pre-pass, supplied via
    // snapshot.splitAxisDir) as an additional primary axis. For near-round
    // snap cells, parent-rotation axes are arbitrary — pre-pass finds the
    // real direction connecting the two bright blobs in the current frame.
    // Only add if it's sufficiently different from the existing axes
    // (|dot| < 0.95 against all).
    if (useSnapshotDirection && snapshotValid) {
        const cv::Point3f pcaDir = snapshot.splitAxisDir;
        const double pcaNorm = cv::norm(pcaDir);
        if (pcaNorm > 1e-3) {
            const cv::Point3f pcaUnit(
                static_cast<float>(pcaDir.x / pcaNorm),
                static_cast<float>(pcaDir.y / pcaNorm),
                static_cast<float>(pcaDir.z / pcaNorm));
            bool duplicate = false;
            for (const auto &existing : primaryDirs) {
                const double dot = std::abs(
                    static_cast<double>(existing.x) * pcaUnit.x +
                    static_cast<double>(existing.y) * pcaUnit.y +
                    static_cast<double>(existing.z) * pcaUnit.z);
                if (dot > 0.95) { duplicate = true; break; }
            }
            if (!duplicate) {
                primaryDirs.push_back(pcaUnit);
                primaryNames.push_back("imgPca");
            }
        }
    }

    {
        std::ostringstream selAxes;
        for (size_t i = 0; i < primaryNames.size(); ++i) {
            if (i > 0) selAxes << ",";
            selAxes << primaryNames[i];
            selAxes << "=(" << primaryDirs[i].x << "," << primaryDirs[i].y
                    << "," << primaryDirs[i].z << ")";
        }
        std::cout << "  [Split Dirs] " << parentName
                  << " mode=shortest_local+imgPca"
                  << " radii=(" << rA << "," << rB << "," << rC << ")"
                  << " shortestAxis=" << names3[shortIdx]
                  << " shortestR=" << radii3[shortIdx]
                  << " selected=[" << selAxes.str() << "]"
                  << " nPrimaries=" << primaryDirs.size()
                  << " (expect 2 midpoints × 5 variants each = "
                  << (primaryDirs.size() * 10) << " candidates before cap)"
                  << " nPixels=" << pixels.size()
                  << std::endl;
    }

    // --- 3. Generate K candidate placements around each (midpoint, direction) pair ---
    //
    // For each axis, centroidsAlongAxis projects bright pixels onto that
    // direction and splits at the median to get two daughter centroids.
    // This gives a data-driven midpoint and separation per axis. Snapshot
    // center is also tried as an alternative midpoint if it differs.
    struct Candidate {
        cv::Point3f d1Pos;
        cv::Point3f d2Pos;
        std::string label;
    };

    // Daughter built radii = volumeScale × snapshot parent radii. Default
    // ∛(0.5) ≈ 0.7937 preserves total cell volume across the split. Tunable
    // via probConfig.split_daughter_volume_scale — raise toward 1.0 to make
    // daughters cover more material on first cost eval (helps when parent
    // PCA fit is tight and undercounts the real cell extent), lower toward
    // 0.5 to start tight and rely on per-daughter PCA refit to grow back.
    const float volumeScale = std::max(0.1f, probConfig.split_daughter_volume_scale);
    const float daughterR = std::max(0.5f * parentMajor, 5.0f);
    const float rotDeltaRad =
        probConfig.split_candidate_rotation_delta_degrees * static_cast<float>(M_PI) / 180.0f;
    const float transDelta =
        probConfig.split_candidate_translation_delta_fraction * daughterR;

    // For each axis direction, project bright pixels onto that axis and
    // compute centroids of the two halves. This gives a data-driven midpoint
    // and separation tuned to where the brightness actually is along each
    // axis, rather than using a fixed radius as the initial separation.
    struct AxisPlacement {
        cv::Point3f d1, d2;     // data-driven daughter centroids
        cv::Point3f midpoint;
        float separation;
        bool valid;
    };
    const size_t nDirs = primaryDirs.size();
    std::vector<AxisPlacement> axisPlace(nDirs);
    // Fallback separation: use the minimum radius as a conservative default.
    const float fallbackSep = std::min({rA, rB, rC});
    for (size_t i = 0; i < nDirs; ++i) {
        axisPlace[i].valid = centroidsAlongAxis(
            pixels, primaryDirs[i], axisPlace[i].d1, axisPlace[i].d2);
        if (axisPlace[i].valid) {
            axisPlace[i].separation = static_cast<float>(
                cv::norm(axisPlace[i].d1 - axisPlace[i].d2));
            axisPlace[i].midpoint = 0.5f * (axisPlace[i].d1 + axisPlace[i].d2);
        } else {
            axisPlace[i].separation = fallbackSep;
            axisPlace[i].midpoint = parentCenter;
        }
        std::cout << "  [Split AxisPlace] " << parentName
                  << " axis=" << primaryNames[i]
                  << " sep=" << axisPlace[i].separation
                  << " mid=(" << axisPlace[i].midpoint.x << "," << axisPlace[i].midpoint.y << "," << axisPlace[i].midpoint.z << ")"
                  << " valid=" << axisPlace[i].valid
                  << std::endl;
    }

    std::vector<Candidate> candidates;
    for (size_t di = 0; di < nDirs; ++di) {
        const auto &dir0 = primaryDirs[di];
        const std::string &axLabel = primaryNames[di];
        cv::Point3f perpU, perpV;
        orthonormalFrame(dir0, perpU, perpV);

        // Two midpoint options for this axis:
        // 1. Data-driven: centroid midpoint from projecting pixels onto this axis
        // 2. Snapshot center (if available and different)
        struct AxisMidOption {
            cv::Point3f center;
            float separation;
            std::string label;
        };
        std::vector<AxisMidOption> axisMids;
        const auto &ap = axisPlace[di];
        axisMids.push_back({ap.midpoint, ap.separation, "data_" + axLabel});

        if (snapshotValid && snapshot.splitAxisLength > 1e-3f) {
            const float dist = static_cast<float>(cv::norm(ap.midpoint - snapshot.position));
            if (dist > 0.5f) {
                axisMids.push_back({snapshot.position, ap.separation, "snap_" + axLabel});
            }
        }

        for (const auto &mp : axisMids) {
            const cv::Point3f midpoint = mp.center;
            const float sep = mp.separation;
            const float half = 0.5f * sep;
            const std::string baseLabel = mp.label;

            // Primary candidate for this (midpoint, direction) pair.
            cv::Point3f d1(midpoint.x - half * dir0.x,
                           midpoint.y - half * dir0.y,
                           midpoint.z - half * dir0.z);
            cv::Point3f d2(midpoint.x + half * dir0.x,
                           midpoint.y + half * dir0.y,
                           midpoint.z + half * dir0.z);
            candidates.push_back({d1, d2, baseLabel + "_primary"});

            // Rotation variants around an axis perpendicular to dir0.
            for (float sign : {-1.0f, 1.0f}) {
                const float angle = sign * rotDeltaRad;
                const cv::Point3f rDir = rotateAroundAxis(dir0, perpU, angle);
                candidates.push_back({
                    cv::Point3f(midpoint.x - half * rDir.x,
                                midpoint.y - half * rDir.y,
                                midpoint.z - half * rDir.z),
                    cv::Point3f(midpoint.x + half * rDir.x,
                                midpoint.y + half * rDir.y,
                                midpoint.z + half * rDir.z),
                    baseLabel + (sign < 0 ? "_rot-" : "_rot+")
                });
            }

            // Translation variants along dir0.
            for (float sign : {-1.0f, 1.0f}) {
                const float t = sign * transDelta;
                candidates.push_back({
                    cv::Point3f(d1.x + t * dir0.x, d1.y + t * dir0.y, d1.z + t * dir0.z),
                    cv::Point3f(d2.x + t * dir0.x, d2.y + t * dir0.y, d2.z + t * dir0.z),
                    baseLabel + (sign < 0 ? "_trans-" : "_trans+")
                });
            }
        }
    }

    const int Kmax = std::max(1, probConfig.split_candidates_per_attempt);
    if (static_cast<int>(candidates.size()) > Kmax) {
        candidates.resize(Kmax);
    }

    // --- 4. Evaluate each candidate via a short burn-in ---
    // Save the pre-split state to revert cheaply.
    //
    // When _useBboxCost is true, build a single union bbox + exclusion mask
    // ONCE here covering parent + all candidate seed positions + drift
    // margin. All cost evaluations during burn-in / refine / final gate use
    // this same bbox + mask, so baseline and every candidate are compared on
    // the same voxel set (apples-to-apples). Neighbor positions don't
    // change during the split attempt, so the mask stays valid throughout.
    BoundingBox3D splitBbox;
    if (_useBboxCost) {
        std::vector<cv::Point3f> seedPoints;
        seedPoints.reserve(candidates.size() * 2);
        for (const auto &cand : candidates) {
            seedPoints.push_back(cand.d1Pos);
            seedPoints.push_back(cand.d2Pos);
        }
        const float pointR = _bboxMarginScale * std::max({srcMajor, srcB, srcMinor});
        splitBbox = computeUnionBboxWithPoints({cellIndex}, _bboxMarginScale,
                                               seedPoints, pointR);

        std::cout << "  [Split Bbox Init] " << parentName
                  << " bboxXYZ=(" << splitBbox.xMin << "-" << splitBbox.xMax
                  << ", " << splitBbox.yMin << "-" << splitBbox.yMax
                  << ", " << splitBbox.zMin << "-" << splitBbox.zMax << ")"
                  << " volume=" << splitBbox.volume()
                  << " maskSeedPoints=" << seedPoints.size()
                  << std::endl;

        // Install the shared split union bbox under each daughter
        // candidate name so perturbCell anchors daughters to the SAME
        // union bbox covering both lobes during burn-in + refine.
        // No Voronoi mask installed (cost uses empty mask; see below).
        if (splitBbox.isValid()) {
            _snapBboxes[parentName + "0"] = splitBbox;
            _snapBboxes[parentName + "1"] = splitBbox;
        }
    }

    // No Voronoi mask for split cost eval either — same reasoning as
    // perturbCell: neighbors' synth is constant across candidates and
    // baseline, cancels in all comparisons. Dropping the mask keeps
    // cost accounting honest about abandoned voxels.
    const std::vector<uint8_t> noSplitMask;
    auto evalImageCost = [&](const std::vector<cv::Mat> &synth) -> double {
        if (_useBboxCost) return calculateBboxCost(splitBbox, synth, noSplitMask);
        return _currentCost;  // legacy path: cached after refreshFullCostCache
    };

    const double baselineImageCost = _useBboxCost
        ? calculateBboxCost(splitBbox, _synthFrame, noSplitMask)
        : _currentCost;
    const double baselineOverlap = computeOverlapPenalty(probConfig.overlap_penalty_weight);
    const double baselineTotal = baselineImageCost + baselineOverlap;
    // Non-const: moved into callback copies at the end of the function
    // (std::move on const silently falls back to copy).
    std::vector<cv::Mat> savedSynth = _synthFrame;
    std::vector<double> savedPerSlice = _currentCostPerSlice;
    const double savedCost = _currentCost;
    std::vector<Ellipsoid> savedCells = cells;

    std::cout << "  [Split Baseline] " << parentName
              << " imageCost=" << baselineImageCost
              << " overlap=" << baselineOverlap
              << " total=" << baselineTotal
              << " threshold=" << -probConfig.split_cost
              << " nCandidates=" << candidates.size()
              << " burnIters=" << probConfig.split_candidate_burn_in_iterations
              << std::endl;

    int bestIdx = -1;
    double bestTotal = std::numeric_limits<double>::infinity();
    std::vector<Ellipsoid> bestCells;
    std::vector<cv::Mat> bestSynth;
    std::vector<double> bestPerSlice;
    double bestImageCost = 0.0;
    cv::Point3f bestSeedD1{0, 0, 0};
    cv::Point3f bestSeedD2{0, 0, 0};
    std::string bestLabel;

    const int burnIters = std::max(0, probConfig.split_candidate_burn_in_iterations);

    // Install tight position sigmas for candidate burn-in.
    //
    // Main-loop position sigmas (x=5, y=5, z=8) let a daughter drift
    // 15-25 voxels across 20 iters, far enough to escape the parent
    // footprint. Scaling by split_burn_in_pos_sigma_scale (0.4 default)
    // restricts burn-in to refinement distances (<10 voxels).
    //
    // Radii are not perturbed anywhere (config sigmas are zero; radii
    // are fit by iterative PCA each frame), so there is no radius-sigma
    // scaling here.
    //
    // Global static state mutation is safe here — single-threaded
    // optimizer, restored on every exit path below.
    const float posScale = std::max(0.0f, probConfig.split_burn_in_pos_sigma_scale);
    PerturbParams savedPerturbX = Ellipsoid::cellConfig.x;
    PerturbParams savedPerturbY = Ellipsoid::cellConfig.y;
    PerturbParams savedPerturbZ = Ellipsoid::cellConfig.z;
    Ellipsoid::cellConfig.x.sigma = savedPerturbX.sigma * posScale;
    Ellipsoid::cellConfig.y.sigma = savedPerturbY.sigma * posScale;
    Ellipsoid::cellConfig.z.sigma = savedPerturbZ.sigma * posScale;

    std::cout << "  [Split Sigmas] " << parentName
              << " posScale=" << posScale
              << " xSigma=" << savedPerturbX.sigma << "->" << Ellipsoid::cellConfig.x.sigma
              << " ySigma=" << savedPerturbY.sigma << "->" << Ellipsoid::cellConfig.y.sigma
              << " zSigma=" << savedPerturbZ.sigma << "->" << Ellipsoid::cellConfig.z.sigma
              << std::endl;

    for (size_t ci = 0; ci < candidates.size(); ++ci) {
        const auto &cand = candidates[ci];
        Ellipsoid child1 = buildDaughter(parentName + "0", cand.d1Pos, parent,
                                         volumeScale, srcMajor, srcB, srcMinor);
        Ellipsoid child2 = buildDaughter(parentName + "1", cand.d2Pos, parent,
                                         volumeScale, srcMajor, srcB, srcMinor);

        // Replace parent with daughters.
        cells.erase(cells.begin() + cellIndex);
        cells.push_back(child1);
        cells.push_back(child2);
        const size_t d1Idx = cells.size() - 2;
        const size_t d2Idx = cells.size() - 1;

        // Render the synth with the new cell configuration (parent
        // removed, daughters added). Under bbox cost, only re-render
        // the z-slices affected by the parent + both daughters —
        // unaffected slices retain their existing render from savedSynth
        // (which is correct because those slices have no parent pixels
        // and the cost bbox doesn't reach them anyway).
        // For legacy full-image mode, do a full render (needed for
        // correct full-image cost).
        if (_useBboxCost) {
            // Compute the z-range touched by parent + both daughters.
            const float pMaxR = std::max({parent.getARadius(),
                                          parent.getBRadius(),
                                          parent.getCRadius()});
            const float d1MaxR = std::max({child1.getARadius(),
                                           child1.getBRadius(),
                                           child1.getCRadius()});
            const float d2MaxR = std::max({child2.getARadius(),
                                           child2.getBRadius(),
                                           child2.getCRadius()});
            const int nSlices = static_cast<int>(z_slices.size());
            const int zLo = std::max(0, static_cast<int>(std::floor(
                std::min({parent.getZ() - pMaxR,
                          child1.getZ() - d1MaxR,
                          child2.getZ() - d2MaxR}))));
            const int zHi = std::min(nSlices - 1, static_cast<int>(std::ceil(
                std::max({parent.getZ() + pMaxR,
                          child1.getZ() + d1MaxR,
                          child2.getZ() + d2MaxR}))));

            // Re-render only affected slices. Unaffected slices retain
            // the savedSynth reference (cv::Mat shallow copy = free).
            const cv::Size shape = getImageShape();
            for (int z = zLo; z <= zHi; ++z) {
                cv::Mat synthImage = cv::Mat(shape, CV_32F,
                                            cv::Scalar(_backgroundValue));
                const float zf = static_cast<float>(z_slices[z]);
                for (const auto &cell : cells) {
                    const float cmr = std::max({cell.getARadius(),
                                                cell.getBRadius(),
                                                cell.getCRadius()});
                    if (std::abs(zf - cell.getZ()) > cmr) continue;
                    cell.draw(synthImage, simulationConfig, zf);
                }
                _synthFrame[z] = synthImage;
            }
        } else {
            _synthFrame = generateSynthFrame();
            refreshFullCostCache();
        }

        // Short alternating burn-in on each daughter.
        for (int it = 0; it < burnIters; ++it) {
            const size_t target = (it % 2 == 0) ? d1Idx : d2Idx;
            CostCallbackPair cp = perturbCell(target,
                                              probConfig.overlap_penalty_weight);
            const bool accept = cp.first < 0.0;
            cp.second(accept);
        }

        // Under bbox flag, _currentCost is stale (perturbCell's bbox path
        // doesn't update it). Use the cached bbox+mask to score the
        // candidate's post-burn-in synth on the same voxel set as baseline.
        const double candImageCost = evalImageCost(_synthFrame);
        const double candOverlap = computeOverlapPenalty(probConfig.overlap_penalty_weight);
        const double candTotal = candImageCost + candOverlap;

        const Ellipsoid &candD1 = cells[d1Idx];
        const Ellipsoid &candD2 = cells[d2Idx];
        const float candDrift1 = static_cast<float>(cv::norm(
            cv::Point3f(candD1.getX(), candD1.getY(), candD1.getZ()) - cand.d1Pos));
        const float candDrift2 = static_cast<float>(cv::norm(
            cv::Point3f(candD2.getX(), candD2.getY(), candD2.getZ()) - cand.d2Pos));
        std::cout << "  [Split Cand] " << parentName
                  << " idx=" << ci << "/" << candidates.size()
                  << " label=" << cand.label
                  << " seed1=(" << cand.d1Pos.x << "," << cand.d1Pos.y << "," << cand.d1Pos.z << ")"
                  << " final1=(" << candD1.getX() << "," << candD1.getY() << "," << candD1.getZ() << ")"
                  << " drift1=" << candDrift1
                  << " seed2=(" << cand.d2Pos.x << "," << cand.d2Pos.y << "," << cand.d2Pos.z << ")"
                  << " final2=(" << candD2.getX() << "," << candD2.getY() << "," << candD2.getZ() << ")"
                  << " drift2=" << candDrift2
                  << " total=" << candTotal
                  << " (image=" << candImageCost << " overlap=" << candOverlap << ")"
                  << std::endl;

        // Per-candidate edge brightness pre-filter. A daughter whose
        // position in the real image is below the edge_too_dim threshold
        // would fail the bridge gate anyway — filter it HERE so it can't
        // win the cost comparison and block a better candidate.
        //
        // This is the same biological signal as edge_too_dim (is there a
        // real cell at the daughter's position?) applied earlier in the
        // pipeline. Catches the f11 1f2ed pattern where a Z-direction
        // candidate placed d2 at z=57 (brightness 0.029) and won on cost
        // over the correct Y-direction candidate (daughters on bright
        // lobes, brightness >0.10). Without this filter, the Z-candidate
        // won, hit edge_too_dim at the bridge, and the whole split was
        // rejected — 1 frame late.
        //
        // Measure: average real-image brightness in a small 3D neighborhood
        // around each daughter center (3×3×3 voxels). Cheap and local.
        const float kMinDaughterBright = probConfig.bio_bridge_min_edge_brightness_absolute;
        auto measureLocalBrightness = [&](float cx, float cy, float cz) -> float {
            const int ix = static_cast<int>(std::round(cx));
            const int iy = static_cast<int>(std::round(cy));
            const int iz = static_cast<int>(std::round(cz));
            float sum = 0.0f;
            int cnt = 0;
            for (int dz = -1; dz <= 1; ++dz) {
                const int zz = iz + dz;
                if (zz < 0 || zz >= static_cast<int>(_realFrame.size())) continue;
                const cv::Mat &sl = _realFrame[zz];
                if (sl.type() != CV_32F) continue;
                for (int dy = -1; dy <= 1; ++dy) {
                    const int yy = iy + dy;
                    if (yy < 0 || yy >= sl.rows) continue;
                    const float *row = sl.ptr<float>(yy);
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int xx = ix + dx;
                        if (xx < 0 || xx >= sl.cols) continue;
                        sum += row[xx];
                        ++cnt;
                    }
                }
            }
            return (cnt > 0) ? (sum / cnt) : 0.0f;
        };

        const float d1LocalBright = measureLocalBrightness(
            candD1.getX(), candD1.getY(), candD1.getZ());
        const float d2LocalBright = measureLocalBrightness(
            candD2.getX(), candD2.getY(), candD2.getZ());
        const bool bothDaughtersBright =
            (d1LocalBright >= kMinDaughterBright) &&
            (d2LocalBright >= kMinDaughterBright);

        // Quick valley pre-filter: project the parent's pixel cloud onto
        // the d1→d2 axis and measure gap vs max(edges). Same logic as the
        // final bridge gate but computed per-candidate BEFORE cost ranking.
        // Candidates with no valley (valleyFromBright > 0.85) or dim
        // daughters are filtered out so they can't win on cost and block
        // better candidates. The full bridge runs again on the winner
        // after refine+refit (Pass 2).
        float candValleyFromBright = 1.0f;
        {
            const cv::Point3f d1Pos(candD1.getX(), candD1.getY(), candD1.getZ());
            const cv::Point3f d2Pos(candD2.getX(), candD2.getY(), candD2.getZ());
            const cv::Point3f axVec = d2Pos - d1Pos;
            const float axLen = static_cast<float>(cv::norm(axVec));
            if (axLen > 1e-3f) {
                const cv::Point3f axDir(axVec.x/axLen, axVec.y/axLen, axVec.z/axLen);
                const cv::Point3f mid(
                    0.5f*(d1Pos.x+d2Pos.x), 0.5f*(d1Pos.y+d2Pos.y), 0.5f*(d1Pos.z+d2Pos.z));
                const float halfLen = 0.5f * axLen;
                const float gapHalf = std::max(0.15f * halfLen, 1.0f);

                double gapSum=0, e1Sum=0, e2Sum=0;
                int gapN=0, e1N=0, e2N=0;
                for (const auto &bp : pixels) {
                    const float dx = bp.pos.x - mid.x;
                    const float dy = bp.pos.y - mid.y;
                    const float dz = bp.pos.z - mid.z;
                    const float proj = dx*axDir.x + dy*axDir.y + dz*axDir.z;
                    if (std::abs(proj) > 1.5f*halfLen) continue;
                    if (std::abs(proj) < gapHalf) {
                        gapSum += bp.weight; ++gapN;
                    } else if (proj < -gapHalf && proj > -halfLen*1.1f) {
                        e1Sum += bp.weight; ++e1N;
                    } else if (proj > gapHalf && proj < halfLen*1.1f) {
                        e2Sum += bp.weight; ++e2N;
                    }
                }
                const float gB = (gapN>0) ? static_cast<float>(gapSum/gapN) : 0.0f;
                const float e1B = (e1N>0) ? static_cast<float>(e1Sum/e1N) : 0.0f;
                const float e2B = (e2N>0) ? static_cast<float>(e2Sum/e2N) : 0.0f;
                const float maxE = std::max(e1B, e2B);
                if (maxE > 1e-6f) candValleyFromBright = gB / maxE;
            }
        }
        const float valleyLimit = probConfig.bio_bridge_max_valley_ratio;
        const bool candPassesPreFilter = bothDaughtersBright &&
                                         (candValleyFromBright < valleyLimit);

        if (!candPassesPreFilter) {
            std::cout << "  [Split Cand PreFilter] " << parentName
                      << " idx=" << ci << " label=" << cand.label
                      << " d1Bright=" << d1LocalBright
                      << " d2Bright=" << d2LocalBright
                      << " valley=" << candValleyFromBright
                      << (bothDaughtersBright ? "" : " EDGE_DIM")
                      << (candValleyFromBright >= valleyLimit ? " NO_VALLEY" : "")
                      << std::endl;
        }

        if (candPassesPreFilter && candTotal < bestTotal) {
            bestTotal = candTotal;
            bestIdx = static_cast<int>(ci);
            // Move instead of copy — cells, _synthFrame, _currentCostPerSlice
            // are immediately overwritten from savedCells/savedSynth/savedPerSlice
            // below, so we can steal their contents here.
            bestCells = std::move(cells);
            bestSynth = std::move(_synthFrame);
            bestPerSlice = std::move(_currentCostPerSlice);
            bestImageCost = candImageCost;
            bestSeedD1 = cand.d1Pos;
            bestSeedD2 = cand.d2Pos;
            bestLabel = cand.label;
        }

        // Revert to pre-split state for the next candidate.
        cells = savedCells;
        _synthFrame = savedSynth;
        _currentCost = savedCost;
        _currentCostPerSlice = savedPerSlice;
    }

    if (bestIdx < 0) {
        Ellipsoid::cellConfig.x = savedPerturbX;
        Ellipsoid::cellConfig.y = savedPerturbY;
        Ellipsoid::cellConfig.z = savedPerturbZ;
        restoreLiveParent();
        return {0.0, noop};
    }

    // Log which candidate won the burn-in competition.
    const double preCostDiff = bestTotal - baselineTotal;
    std::cout << "  [Split Winner] " << parentName
              << " bestIdx=" << bestIdx << "/" << candidates.size()
              << " label=" << bestLabel
              << " preCostDiff=" << preCostDiff
              << " bestTotal=" << bestTotal
              << " baseline=" << baselineTotal
              << " seed1=(" << bestSeedD1.x << "," << bestSeedD1.y << "," << bestSeedD1.z << ")"
              << " seed2=(" << bestSeedD2.x << "," << bestSeedD2.y << "," << bestSeedD2.z << ")"
              << std::endl;

    // --- 4b. Final refine burn-in on the winning candidate ---
    // The candidate loop runs a short (~20 iter) burn-in per candidate so
    // the K=5 comparison is cheap. Now that we've picked a winner, give it
    // an extra refine pass with the same tight sigmas so the chosen
    // daughters can settle before bio/cost gates fire. This runs on the
    // best candidate's state (reinstalled), and the post-refine state is
    // re-captured as bestCells / bestSynth / etc.
    const int refineIters = std::max(0, probConfig.split_final_refine_iterations);
    if (refineIters > 0) {
        // Reinstall the winning candidate's state.
        cells = bestCells;
        _synthFrame = bestSynth;
        _currentCostPerSlice = bestPerSlice;
        _currentCost = bestImageCost;

        const size_t d1IdxRefine = cells.size() - 2;
        const size_t d2IdxRefine = cells.size() - 1;
        const cv::Point3f preRefineD1(cells[d1IdxRefine].getX(),
                                        cells[d1IdxRefine].getY(),
                                        cells[d1IdxRefine].getZ());
        const cv::Point3f preRefineD2(cells[d2IdxRefine].getX(),
                                        cells[d2IdxRefine].getY(),
                                        cells[d2IdxRefine].getZ());
        // Pre-refine baseline: bestImageCost was set from candidate's
        // post-burn-in cost (bbox or full per flag), reinstalled into
        // _currentCost above for the legacy path.
        const double preRefineTotal = bestImageCost +
            computeOverlapPenalty(probConfig.overlap_penalty_weight);

        int refineAccepts = 0;
        for (int it = 0; it < refineIters; ++it) {
            const size_t target = (it % 2 == 0) ? d1IdxRefine : d2IdxRefine;
            CostCallbackPair cp = perturbCell(target,
                                              probConfig.overlap_penalty_weight);
            const bool accept = cp.first < 0.0;
            if (accept) ++refineAccepts;
            cp.second(accept);
        }

        // --- A1: per-daughter PCA radius refit ---
        // After burn-in + positional refine pins daughter centers, each
        // daughter's radii (inherited from parent as 0.794 * src) are
        // still generic — a real daughter is usually smaller and may have
        // different aspect. Run a short PCA shape fit on each daughter,
        // using its built-in radii as the FIXED mask (same snap-mask
        // pattern as the per-frame shape fit), with Voronoi exclusion so
        // the sibling daughter and every other cell are claimants.
        //
        // Clamp fitted radii to [min * built, max * built] per axis:
        //   floor (min_fraction × built) — phantom daughter can't collapse
        //   ceiling (max_fraction × built) — newborn daughter can't bloat
        //     past ~1.1× built due to immature sibling-Voronoi boundary
        //     absorbing neighbor/halo pixels during refit.
        const int daughterRefitIters = std::max(0, probConfig.split_daughter_refit_iterations);
        if (daughterRefitIters > 0) {
            const float minFrac = std::max(0.0f, std::min(1.0f,
                probConfig.split_daughter_refit_min_radius_fraction));
            const float maxFrac = std::max(1.0f,
                probConfig.split_daughter_refit_max_radius_fraction);
            const float dBuiltA = volumeScale * srcMajor;
            const float dBuiltB = volumeScale * srcB;
            const float dBuiltC = volumeScale * srcMinor;
            const float floorA = minFrac * dBuiltA;
            const float floorB = minFrac * dBuiltB;
            const float floorC = minFrac * dBuiltC;
            const float ceilA  = maxFrac * dBuiltA;
            const float ceilB  = maxFrac * dBuiltB;
            const float ceilC  = maxFrac * dBuiltC;

            auto buildRefitClaimSet = [&](size_t selfIdx) -> ClaimSet {
                ClaimSet others;
                for (size_t oi = 0; oi < cells.size(); ++oi) {
                    if (oi == selfIdx) continue;
                    const std::string oname = cells[oi].getName();
                    others[oname].push_back(cv::Point3f(
                        cells[oi].getX(), cells[oi].getY(), cells[oi].getZ()));
                }
                return others;
            };

            auto refitOne = [&](size_t idx, const char *label) {
                const float preA = cells[idx].getARadius();
                const float preB = cells[idx].getBRadius();
                const float preC = cells[idx].getCRadius();
                const cv::Point3f prePos(cells[idx].getX(),
                                         cells[idx].getY(),
                                         cells[idx].getZ());
                ClaimSet others = buildRefitClaimSet(idx);
                // Position update ENABLED for daughter refit (distinct
                // from mature cells' shape fit, which keeps it off to
                // let calibration own position). At birth the daughter's
                // centroid from burn-in is an estimate — letting the PCA
                // slide it toward the actual pixel centroid fixes the
                // "daughter drawn off-center" artifact.
                calibrateCellShapeViaPca(
                    idx, others,
                    daughterRefitIters,
                    Ellipsoid::cellConfig.pcaShapeRadiusScale,
                    Ellipsoid::cellConfig.pcaShapeMinPixels,
                    Ellipsoid::cellConfig.pcaShapeMaskScale,
                    Ellipsoid::cellConfig.pcaShapeConvergeRadius,
                    Ellipsoid::cellConfig.pcaShapeConvergeAngleDeg,
                    /*updatePosition=*/true,
                    Ellipsoid::cellConfig.pcaShapeMaxPosShiftFraction,
                    dBuiltA, dBuiltB, dBuiltC);
                const float fitA = std::clamp(cells[idx].getARadius(), floorA, ceilA);
                const float fitB = std::clamp(cells[idx].getBRadius(), floorB, ceilB);
                const float fitC = std::clamp(cells[idx].getCRadius(), floorC, ceilC);
                cells[idx].setRadii(fitA, fitB, fitC);
                const cv::Point3f postPos(cells[idx].getX(),
                                          cells[idx].getY(),
                                          cells[idx].getZ());
                std::cout << "  [Split Daughter Refit] " << parentName
                          << " " << label
                          << " iters=" << daughterRefitIters
                          << " built=(" << dBuiltA << "," << dBuiltB << "," << dBuiltC << ")"
                          << " floor=(" << floorA << "," << floorB << "," << floorC << ")"
                          << " ceil=(" << ceilA << "," << ceilB << "," << ceilC << ")"
                          << " pre=(" << preA << "," << preB << "," << preC << ")"
                          << " post=(" << fitA << "," << fitB << "," << fitC << ")"
                          << " prePos=(" << prePos.x << "," << prePos.y << "," << prePos.z << ")"
                          << " postPos=(" << postPos.x << "," << postPos.y << "," << postPos.z << ")"
                          << " posShift=" << cv::norm(postPos - prePos)
                          << std::endl;
            };

            refitOne(d1IdxRefine, "d1");
            refitOne(d2IdxRefine, "d2");

            // Radii changed → regenerate synth. Under bbox mode, only
            // re-render the z-slices affected by the two daughters (whose
            // radii just changed). Under legacy mode, full render.
            if (_useBboxCost) {
                const Ellipsoid &rd1 = cells[d1IdxRefine];
                const Ellipsoid &rd2 = cells[d2IdxRefine];
                const float rd1MaxR = std::max({rd1.getARadius(), rd1.getBRadius(), rd1.getCRadius()});
                const float rd2MaxR = std::max({rd2.getARadius(), rd2.getBRadius(), rd2.getCRadius()});
                // Include the PRE-refit extent too (built radii) in case
                // the refit shrunk the cell — old render needs clearing.
                const float builtMaxR = std::max({dBuiltA, dBuiltB, dBuiltC});
                const float extentR = std::max({rd1MaxR, rd2MaxR, builtMaxR});
                const int nSlices = static_cast<int>(z_slices.size());
                const int zLo = std::max(0, static_cast<int>(std::floor(
                    std::min(rd1.getZ(), rd2.getZ()) - extentR)));
                const int zHi = std::min(nSlices - 1, static_cast<int>(std::ceil(
                    std::max(rd1.getZ(), rd2.getZ()) + extentR)));
                const cv::Size shape = getImageShape();
                for (int z = zLo; z <= zHi; ++z) {
                    cv::Mat synthImage = cv::Mat(shape, CV_32F,
                                                cv::Scalar(_backgroundValue));
                    const float zf = static_cast<float>(z_slices[z]);
                    for (const auto &cell : cells) {
                        const float cmr = std::max({cell.getARadius(),
                                                    cell.getBRadius(),
                                                    cell.getCRadius()});
                        if (std::abs(zf - cell.getZ()) > cmr) continue;
                        cell.draw(synthImage, simulationConfig, zf);
                    }
                    _synthFrame[z] = synthImage;
                }
            } else {
                _synthFrame = generateSynthFrame();
                refreshFullCostCache();
            }
        }

        // Re-capture refined state as new best. Capture diagnostic data
        // BEFORE moving cells (use-after-move is UB).
        bestImageCost = evalImageCost(_synthFrame);
        bestTotal = bestImageCost + computeOverlapPenalty(probConfig.overlap_penalty_weight);

        const cv::Point3f postRefineD1(cells[d1IdxRefine].getX(),
                                         cells[d1IdxRefine].getY(),
                                         cells[d1IdxRefine].getZ());
        const cv::Point3f postRefineD2(cells[d2IdxRefine].getX(),
                                         cells[d2IdxRefine].getY(),
                                         cells[d2IdxRefine].getZ());
        // Daughter radii diagnostic. Built = 0.794 * src; post-refit radii
        // reflect the PCA fit on each daughter's pixel cloud, clamped at
        // the configured floor fraction.
        // Read daughter state BEFORE moving cells — references into a
        // moved-from vector are dangling (UB). Copy radii by value.
        const Ellipsoid refinedD1 = cells[d1IdxRefine];
        const Ellipsoid refinedD2 = cells[d2IdxRefine];
        const float builtMajor = volumeScale * srcMajor;
        const float builtB     = volumeScale * srcB;
        const float builtMinor = volumeScale * srcMinor;
        std::cout << "  [Split Refine] " << parentName
                  << " iters=" << refineIters
                  << " accepts=" << refineAccepts
                  << " preTotal=" << preRefineTotal
                  << " postTotal=" << bestTotal
                  << " delta=" << (bestTotal - preRefineTotal)
                  << " refineDrift1=" << cv::norm(postRefineD1 - preRefineD1)
                  << " refineDrift2=" << cv::norm(postRefineD2 - preRefineD2)
                  << " d1=(" << postRefineD1.x << "," << postRefineD1.y << "," << postRefineD1.z << ")"
                  << " d2=(" << postRefineD2.x << "," << postRefineD2.y << "," << postRefineD2.z << ")"
                  << " builtR=(" << builtMajor << "," << builtB << "," << builtMinor << ")"
                  << " d1R=(" << refinedD1.getARadius() << "," << refinedD1.getBRadius() << "," << refinedD1.getCRadius() << ")"
                  << " d2R=(" << refinedD2.getARadius() << "," << refinedD2.getBRadius() << "," << refinedD2.getCRadius() << ")"
                  << std::endl;

        bestCells = std::move(cells);
        bestSynth = std::move(_synthFrame);
        bestPerSlice = std::move(_currentCostPerSlice);

        // Revert to pre-split state — gates run on savedCells baseline
        // against bestCells (the refined winner).
        cells = savedCells;
        _synthFrame = savedSynth;
        _currentCost = savedCost;
        _currentCostPerSlice = savedPerSlice;
    }

    // Restore main-loop perturbation sigmas before the gate sequence.
    Ellipsoid::cellConfig.x = savedPerturbX;
    Ellipsoid::cellConfig.y = savedPerturbY;
    Ellipsoid::cellConfig.z = savedPerturbZ;

    // --- 5. Bio checks on the best candidate's final state ---
    // Rebuild daughter indices from bestCells (parent was at cellIndex,
    // daughters are the last two entries in bestCells since that's how we
    // replaced them during evaluation).
    const size_t d1IdxBest = bestCells.size() - 2;
    const size_t d2IdxBest = bestCells.size() - 1;
    const Ellipsoid &bestD1 = bestCells[d1IdxBest];
    const Ellipsoid &bestD2 = bestCells[d2IdxBest];

    // Drift from seed (diagnostic only, no rejection gate).
    const float drift1 = static_cast<float>(cv::norm(
        cv::Point3f(bestD1.getX(), bestD1.getY(), bestD1.getZ()) - bestSeedD1));
    const float drift2 = static_cast<float>(cv::norm(
        cv::Point3f(bestD2.getX(), bestD2.getY(), bestD2.getZ()) - bestSeedD2));

    // Daughter midpoint (shared by the bridge gate below for axis
    // projection and diagnostic logging). Previously also used by a
    // midpoint-near-snapshot-parent gate (2026-04-11 afternoon), but
    // that gate was removed because (a) the pre-pass grounds
    // snapshot.position onto the image bright region, which pulls the
    // reference toward the daughters and makes the check trivially
    // pass (see e3d03 f3 run 082245), and (b) the bridge brightness
    // gate catches the same false-split patterns directly via the
    // image content.
    const cv::Point3f daughterMidpoint(
        0.5f * (bestD1.getX() + bestD2.getX()),
        0.5f * (bestD1.getY() + bestD2.getY()),
        0.5f * (bestD1.getZ() + bestD2.getZ()));

    // 5a''. Bridge brightness gate — project the Voronoi-filtered bright
    // pixels (the same set PCA used) onto the daughter split axis. Real
    // divisions have a dim gap in the middle (dividing groove): low pixel
    // density and low mean brightness in the central bin. A fake split
    // over one continuous cell has uniform pixels across the middle.
    //
    // Normalized axis coordinate t in [-1, +1]:
    //   t = -1 ⇔ bestD1 center
    //   t =  0 ⇔ midpoint
    //   t = +1 ⇔ bestD2 center
    //
    // Only pixels with |t| < 1.5 are considered (ignore pixels way
    // outside the daughter span, which inflate statistics without
    // helping).
    //
    // Gap:  |t| < 0.3 (middle ~30% of the span)
    // Edge: 0.6 < |t| < 1.1 (near each daughter center)
    //
    // Single-metric rejection: gap / max(edge1, edge2) > bio_bridge_max_valley_ratio.
    // Adaptive bridge gate: measure brightness in the actual gap between
    // the two daughters' surfaces, not a fixed fraction of the axis.
    //
    // For each daughter, compute the ellipsoid radius along the split axis
    // (how far the surface extends toward the other daughter). The gap is
    // the region between those two surfaces. Edge zones are the regions
    // inside each daughter, away from the gap.
    {
        const cv::Point3f bestD1Pos(bestD1.getX(), bestD1.getY(), bestD1.getZ());
        const cv::Point3f bestD2Pos(bestD2.getX(), bestD2.getY(), bestD2.getZ());
        const cv::Point3f axisVec = bestD2Pos - bestD1Pos;
        const float axisLen = static_cast<float>(cv::norm(axisVec));
        const int totalBridgeCandidates = static_cast<int>(pixels.size());

        if (axisLen > 1e-3f && totalBridgeCandidates >= 1000) {
            const cv::Point3f axisDir(
                axisVec.x / axisLen,
                axisVec.y / axisLen,
                axisVec.z / axisLen);

            // Ellipsoid radius along an arbitrary direction: for an ellipsoid
            // with semi-axes (a,b,c) and rotation R, the support distance
            // along world direction d is ||diag(a,b,c) * R^T * d||.
            auto ellipsoidRadiusAlongDir = [](const Ellipsoid &e,
                                              const cv::Point3f &dir) -> float {
                std::array<double, 9> RT;
                e.generateInverseRotationMatrix(RT);
                // R^T * dir (world → local)
                const double lx = RT[0]*dir.x + RT[1]*dir.y + RT[2]*dir.z;
                const double ly = RT[3]*dir.x + RT[4]*dir.y + RT[5]*dir.z;
                const double lz = RT[6]*dir.x + RT[7]*dir.y + RT[8]*dir.z;
                // Scale by semi-axes
                const double sa = e.getARadius() * lx;
                const double sb = e.getBRadius() * ly;
                const double sc = e.getCRadius() * lz;
                return static_cast<float>(std::sqrt(sa*sa + sb*sb + sc*sc));
            };

            const float r1Along = ellipsoidRadiusAlongDir(bestD1, axisDir);
            const float r2Along = ellipsoidRadiusAlongDir(bestD2, axisDir);
            const float gapWidth = axisLen - r1Along - r2Along;

            // Coordinate system: project onto axisDir, origin at midpoint.
            // d1 is at -halfLen, d2 is at +halfLen.
            const float halfLen = 0.5f * axisLen;

            // ALWAYS measure the central region between daughters, even
            // when they geometrically overlap (gapWidth <= 0). A real split
            // produces a brightness valley at the midpoint; a false split
            // on a non-dividing cell has continuous brightness through the
            // center. The minimum gap half-width of 15% of halfLen
            // guarantees we sample enough pixels to detect this.
            const float minGapHalf = 0.30f * halfLen;
            const float surfaceGapHalf = 0.5f * gapWidth; // negative if overlapping
            const float effectiveGapHalf = std::max(minGapHalf, surfaceGapHalf);
            const float gapLo = -effectiveGapHalf;
            const float gapHi =  effectiveGapHalf;

            // Edge zones: near each daughter center, outside the gap.
            // d1 is at -halfLen, so its edge zone is [-halfLen-r1Along, gapLo]
            // d2 is at +halfLen, so its edge zone is [gapHi, halfLen+r2Along]
            const float edge1Lo = -halfLen - r1Along;
            const float edge1Hi = gapLo;
            const float edge2Lo = gapHi;
            const float edge2Hi = halfLen + r2Along;

            int totalInRange = 0;
            int gapCount = 0;
            int edge1Count = 0, edge2Count = 0;
            double gapBrightSum = 0.0;
            double edge1BrightSum = 0.0, edge2BrightSum = 0.0;

            for (const auto &bp : pixels) {
                const cv::Point3f delta(
                    bp.pos.x - daughterMidpoint.x,
                    bp.pos.y - daughterMidpoint.y,
                    bp.pos.z - daughterMidpoint.z);
                const float proj =
                    delta.x * axisDir.x +
                    delta.y * axisDir.y +
                    delta.z * axisDir.z;

                if (proj < edge1Lo || proj > edge2Hi) continue;
                ++totalInRange;

                if (proj >= gapLo && proj <= gapHi) {
                    ++gapCount;
                    gapBrightSum += bp.weight;
                } else if (proj >= edge1Lo && proj <= edge1Hi) {
                    ++edge1Count;
                    edge1BrightSum += bp.weight;
                } else if (proj >= edge2Lo && proj <= edge2Hi) {
                    ++edge2Count;
                    edge2BrightSum += bp.weight;
                }
            }

            const int edgeCount = edge1Count + edge2Count;
            const double edgeBrightSum = edge1BrightSum + edge2BrightSum;

            const float gapDensity = (totalInRange > 0)
                ? static_cast<float>(gapCount) / static_cast<float>(totalInRange)
                : 0.0f;
            const float gapBright = (gapCount > 0)
                ? static_cast<float>(gapBrightSum / gapCount)
                : 0.0f;
            const float edgeBright = (edgeCount > 0)
                ? static_cast<float>(edgeBrightSum / edgeCount)
                : 0.0f;
            const float edge1Bright = (edge1Count > 0)
                ? static_cast<float>(edge1BrightSum / edge1Count)
                : 0.0f;
            const float edge2Bright = (edge2Count > 0)
                ? static_cast<float>(edge2BrightSum / edge2Count)
                : 0.0f;
            // Per-daughter valley ratios. Pooling edges (gap/edgeAvg)
            // hides asymmetry — a bright real daughter averages with a
            // dim phantom daughter and the gap still looks like a valley.
            // Checking gap against EACH edge independently catches the
            // phantom case: if gap >= edge on one side, that daughter is
            // in near-empty space, not a real cell body.
            const float valleyRatio1 = (edge1Bright > 1e-6f)
                ? (gapBright / edge1Bright)
                : 1.0f;
            const float valleyRatio2 = (edge2Bright > 1e-6f)
                ? (gapBright / edge2Bright)
                : 1.0f;
            const float worstValleyRatio = std::max(valleyRatio1, valleyRatio2);
            // Pooled ratio kept for logging only (diagnostic continuity).
            const float valleyRatio = (edgeBright > 1e-6f)
                ? (gapBright / edgeBright)
                : 0.0f;
            // Valley metric based on the BRIGHTER daughter edge only.
            //
            // Asymmetric division (one daughter inherits more cytoplasm
            // and renders brighter) is biologically normal. The dim-
            // daughter's edge ≈ gap is expected — it doesn't indicate
            // "no valley", just that this daughter is small. The signal
            // that actually matters is whether the brighter daughter
            // shows a drop from its cell body into the midpoint.
            //
            // Taking max(edge1, edge2) as the reference naturally handles
            // both symmetric and asymmetric cases: for symmetric daughters
            // the two edges are equal, `gap/max(edges) = gap/either_edge`;
            // for asymmetric daughters the brighter side drives the
            // decision and the dim side doesn't punish it.
            //
            // Replaces the previous worstValleyRatio-based tiered decision
            // (2026-04-15). The legacy two-tier path (gap density + worst
            // valley ratio) was retired — it punished legitimate asymmetric
            // division.
            const float brighterEdge = std::max(edge1Bright, edge2Bright);
            const float valleyFromBright = (brighterEdge > 1e-6f)
                ? (gapBright / brighterEdge)
                : 1.0f;

            std::cout << "  [Split Bridge] " << parentName
                      << " axisLen=" << axisLen
                      << " r1Along=" << r1Along
                      << " r2Along=" << r2Along
                      << " gapWidth=" << gapWidth
                      << " effGapHalf=" << effectiveGapHalf
                      << " totalInRange=" << totalInRange
                      << " gapCount=" << gapCount
                      << " edgeCount=" << edgeCount
                      << " gapDensity=" << gapDensity
                      << " gapBright=" << gapBright
                      << " edge1Bright=" << edge1Bright
                      << " edge2Bright=" << edge2Bright
                      << " edgeBright=" << edgeBright
                      << " valleyRatio1=" << valleyRatio1
                      << " valleyRatio2=" << valleyRatio2
                      << " worstValleyRatio=" << worstValleyRatio
                      << " valleyRatioPooled=" << valleyRatio
                      << " valleyFromBright=" << valleyFromBright
                      << std::endl;

            // Absolute edge-brightness gate: a daughter whose edge zone
            // is at or near background is sitting in empty space, even
            // if the other edge is bright enough to make a favorable
            // cost delta. Measured in the same real-image units as the
            // sigmoid-calibrated background (~0.0), so ~0.05 is ~5% above
            // background — well below any real cell body (~0.1-0.3).
            // Tunable via probConfig.bio_bridge_min_edge_brightness_absolute.
            const float kMinEdgeBrightAbsolute =
                probConfig.bio_bridge_min_edge_brightness_absolute;
            if (edge1Count > 0 && edge2Count > 0 &&
                std::min(edge1Bright, edge2Bright) < kMinEdgeBrightAbsolute) {
                std::cout << "[Split Reject bio] " << parentName
                          << " reason=edge_too_dim"
                          << " edge1Bright=" << edge1Bright
                          << " edge2Bright=" << edge2Bright
                          << " minEdgeBright=" << std::min(edge1Bright, edge2Bright)
                          << " threshold=" << kMinEdgeBrightAbsolute
                          << std::endl;
                restoreLiveParent();
                return {0.0, noop};
            }

            // Single-metric valley gate (2026-04-15 redesign):
            //   reject when gap brightness ≥ valleyLimit × max(edge1, edge2)
            //
            // This measures the brightness drop from the brighter daughter
            // edge into the gap. A real split (symmetric or asymmetric) has
            // valleyFromBright < 0.85 (gap is clearly darker than the
            // brighter edge). A phantom split has valleyFromBright ≈ 1 or
            // above (gap at least as bright as any edge).
            //
            // Replaces the previous two-tier logic that combined
            // worstValleyRatio + gapDensity. The old worst-of-two-sides
            // metric punished legitimate asymmetric division because the
            // dimmer daughter's edge ≈ gap inflated its ratio. The new
            // metric ignores that side correctly.
            const float valleyLimit = probConfig.bio_bridge_max_valley_ratio;
            const bool bridgeFlat = valleyFromBright > valleyLimit;
            if (edgeCount > 0 && bridgeFlat) {
                std::cout << "[Split Reject bio] " << parentName
                          << " reason=bridge_flat"
                          << " valleyFromBright=" << valleyFromBright
                          << " brighterEdge=" << brighterEdge
                          << " gapBright=" << gapBright
                          << " valleyRatio1=" << valleyRatio1
                          << " valleyRatio2=" << valleyRatio2
                          << " gapDensity=" << gapDensity
                          << " gapWidth=" << gapWidth
                          << " effGapHalf=" << effectiveGapHalf
                          << " valleyLimit=" << valleyLimit
                          << std::endl;
                restoreLiveParent();
                return {0.0, noop};
            }
        }
    }

    // 5b. Size ratio, volume fraction, and buried checks.
    std::string bioReason;
    if (!bioCheckDaughters(bestD1, bestD2, refParentVolume, bestCells, d1IdxBest, d2IdxBest,
                           probConfig, bioReason)) {
        std::cout << "[Split Reject bio] " << parentName
                  << " reason=" << bioReason
                  << " d1=(" << bestD1.getX() << "," << bestD1.getY() << "," << bestD1.getZ() << ")"
                  << " r1=(" << bestD1.getARadius() << "," << bestD1.getBRadius() << "," << bestD1.getCRadius() << ")"
                  << " d2=(" << bestD2.getX() << "," << bestD2.getY() << "," << bestD2.getZ() << ")"
                  << " r2=(" << bestD2.getARadius() << "," << bestD2.getBRadius() << "," << bestD2.getCRadius() << ")"
                  << " refParentVolume=" << refParentVolume
                  << std::endl;
        restoreLiveParent();
        return {0.0, noop};
    }

    // --- 6. Cost check ---
    // costDiff is bbox-based when _useBboxCost is true (every cost
    // evaluation in burn-in / refine used the splitBbox + splitMask built
    // at the top of this function), or full-image L2 otherwise.
    // Both baseline and candidate were measured on the same voxel set,
    // so this is an apples-to-apples comparison.
    const double costDiff = bestTotal - baselineTotal;

    // Adaptive split cost threshold: the larger of:
    // (1) the fixed split_cost from config
    // (2) split_cost_fraction × baselineImageCost (proportional to cell)
    // This prevents marginal splits on dim/small cells (low baseline cost)
    // from passing the fixed threshold while requiring the same fractional
    // improvement from bright/large cells.
    const double adaptiveThreshold = std::max(
        static_cast<double>(probConfig.split_cost),
        static_cast<double>(probConfig.split_cost_fraction) * baselineImageCost);

    if (costDiff >= -adaptiveThreshold) {
        std::cout << "[Split Reject cost] " << parentName
                  << " diff=" << costDiff
                  << " mode=" << (_useBboxCost ? "bbox" : "full")
                  << " threshold=" << -adaptiveThreshold
                  << " (fixed=" << probConfig.split_cost
                  << " frac=" << probConfig.split_cost_fraction << "×" << baselineImageCost << ")"
                  << " bestIdx=" << bestIdx
                  << " bestLabel=" << bestLabel
                  << " d1=(" << bestD1.getX() << "," << bestD1.getY() << "," << bestD1.getZ() << ")"
                  << " r1=(" << bestD1.getARadius() << "," << bestD1.getBRadius() << "," << bestD1.getCRadius() << ")"
                  << " drift1=" << drift1
                  << " d2=(" << bestD2.getX() << "," << bestD2.getY() << "," << bestD2.getZ() << ")"
                  << " r2=(" << bestD2.getARadius() << "," << bestD2.getBRadius() << "," << bestD2.getCRadius() << ")"
                  << " drift2=" << drift2
                  << std::endl;
        restoreLiveParent();
        return {0.0, noop};
    }

    // Accept: install the best candidate state. The callback applies on
    // accept; the caller uses perturbCell's contract where the callback is
    // invoked after the decision. To stay consistent with that contract we
    // return the (costDiff, callback) pair that installs bestCells state.
    // Move saved* into copies for the callback — savedCells/savedSynth/
    // savedPerSlice are not read again after this point.
    auto savedCellsCopy = std::move(savedCells);
    auto savedSynthCopy = std::move(savedSynth);
    auto savedPerSliceCopy = std::move(savedPerSlice);
    double savedCostCopy = savedCost;

    const cv::Point3f acceptedD1Pos(bestD1.getX(), bestD1.getY(), bestD1.getZ());
    const cv::Point3f acceptedD2Pos(bestD2.getX(), bestD2.getY(), bestD2.getZ());
    const cv::Point3f acceptedD1R(bestD1.getARadius(), bestD1.getBRadius(), bestD1.getCRadius());
    const cv::Point3f acceptedD2R(bestD2.getARadius(), bestD2.getBRadius(), bestD2.getCRadius());
    const float acceptedDrift1 = drift1;
    const float acceptedDrift2 = drift2;
    const cv::Point3f acceptedSeed1 = bestSeedD1;
    const cv::Point3f acceptedSeed2 = bestSeedD2;

    // Capture extras for the callback's reject branch — it needs to
    // undo the snapshot-state install so Phase B's live state is restored.
    const Ellipsoid liveParentCopy = liveParent;
    const Ellipsoid snapshotParentCopy = snapshotParent;
    const size_t cellIndexCopy = cellIndex;
    const bool snapshotValidCopy = snapshotValid;

    const std::string acceptedLabel = bestLabel;

    CallBackFunc callback = [this,
                             bestCells = std::move(bestCells),
                             bestSynth = std::move(bestSynth),
                             bestPerSlice = std::move(bestPerSlice),
                             bestImageCost,
                             savedCellsCopy = std::move(savedCellsCopy),
                             savedSynthCopy = std::move(savedSynthCopy),
                             savedPerSliceCopy = std::move(savedPerSliceCopy),
                             savedCostCopy,
                             parentName, costDiff, acceptedD1Pos, acceptedD2Pos,
                             acceptedD1R, acceptedD2R, acceptedSeed1, acceptedSeed2,
                             acceptedDrift1, acceptedDrift2, acceptedLabel,
                             liveParentCopy, snapshotParentCopy, cellIndexCopy,
                             snapshotValidCopy](bool accept) mutable {
        // Snap bbox + snap position handling for daughter names.
        //
        // On ACCEPT: replace the burn-in-time shared splitBbox with per-
        // daughter snap bboxes centered on the final positions, and set
        // snap positions so the position-prior penalty (Change 37)
        // activates. Without this, newborn daughters had NO anchor post-
        // split — observed in run 205041 f3 where 12345..0 drifted from
        // (142, 176, 105) at split-accept to (-27, 265, 90) by f3 end
        // (175 px drift, exited image). Position prior shows priorWeight=30
        // in the log but was a no-op because _snapPositions had no entry
        // for the newborn daughter name.
        //
        // On REJECT: erase the stale daughter-name entries. The parent
        // keeps its own snap (never modified).
        const std::string d0Name = parentName + "0";
        const std::string d1Name = parentName + "1";
        if (accept) {
            this->cells = std::move(bestCells);
            this->_synthFrame = std::move(bestSynth);
            this->_currentCostPerSlice = std::move(bestPerSlice);
            this->_currentCost = bestImageCost;
            // Install snap anchors at the accepted daughter positions.
            // Use the daughter max-radius × bbox_margin_scale for the bbox.
            const float d0MaxR = std::max({acceptedD1R.x, acceptedD1R.y, acceptedD1R.z});
            const float d1MaxR = std::max({acceptedD2R.x, acceptedD2R.y, acceptedD2R.z});
            if (d0MaxR > 1e-3f) {
                BoundingBox3D b0 = this->computeBboxAtPoint(
                    acceptedD1Pos, d0MaxR, this->_bboxMarginScale);
                if (b0.isValid()) this->_snapBboxes[d0Name] = b0;
                this->_snapPositions[d0Name] = acceptedD1Pos;
            }
            if (d1MaxR > 1e-3f) {
                BoundingBox3D b1 = this->computeBboxAtPoint(
                    acceptedD2Pos, d1MaxR, this->_bboxMarginScale);
                if (b1.isValid()) this->_snapBboxes[d1Name] = b1;
                this->_snapPositions[d1Name] = acceptedD2Pos;
            }
            std::cout << "[Split Accepted] " << parentName
                      << " costDiff=" << costDiff
                      << " bestLabel=" << acceptedLabel
                      << " seed1=(" << acceptedSeed1.x << "," << acceptedSeed1.y << "," << acceptedSeed1.z << ")"
                      << " d1=(" << acceptedD1Pos.x << "," << acceptedD1Pos.y << "," << acceptedD1Pos.z << ")"
                      << " r1=(" << acceptedD1R.x << "," << acceptedD1R.y << "," << acceptedD1R.z << ")"
                      << " drift1=" << acceptedDrift1
                      << " seed2=(" << acceptedSeed2.x << "," << acceptedSeed2.y << "," << acceptedSeed2.z << ")"
                      << " d2=(" << acceptedD2Pos.x << "," << acceptedD2Pos.y << "," << acceptedD2Pos.z << ")"
                      << " r2=(" << acceptedD2R.x << "," << acceptedD2R.y << "," << acceptedD2R.z << ")"
                      << " drift2=" << acceptedDrift2
                      << std::endl;
        } else {
            // Reject path: restore the snapshot-parent-state first (which
            // is what savedCellsCopy holds), then swap cells[cellIndexCopy]
            // back to the live parent and re-render the affected z-range
            // so Phase B's live state isn't lost. Also erase any stale
            // daughter-name entries in snap maps (from burn-in installation).
            this->_snapBboxes.erase(d0Name);
            this->_snapBboxes.erase(d1Name);
            this->_snapPositions.erase(d0Name);
            this->_snapPositions.erase(d1Name);
            this->cells = std::move(savedCellsCopy);  // NOLINT: move in mutable lambda
            this->_synthFrame = std::move(savedSynthCopy);
            this->_currentCostPerSlice = std::move(savedPerSliceCopy);
            this->_currentCost = savedCostCopy;

            if (snapshotValidCopy && cellIndexCopy < this->cells.size()) {
                this->cells[cellIndexCopy] = liveParentCopy;
                int affMinR = -1, affMaxR = -1;
                Ellipsoid snapshotMutable = snapshotParentCopy;
                Ellipsoid liveMutable = liveParentCopy;
                auto revertedSynth = this->generateSynthFrameFast(snapshotMutable, liveMutable,
                                                                    &affMinR, &affMaxR);
                this->_synthFrame = revertedSynth;
                // Bbox mode keeps the full-image cache stale (Change 1).
                // Skip the incremental recompute on this reject path too.
                if (!this->_useBboxCost) {
                    std::vector<double> revertedPerSlice;
                    const double revertedCost = this->calculateIncrementalCost(revertedSynth,
                                                                                 affMinR, affMaxR,
                                                                                 revertedPerSlice);
                    this->_currentCost = revertedCost;
                    this->_currentCostPerSlice = revertedPerSlice;
                }
            }
        }
    };

    return {costDiff, callback};
}

// Snapshot-driven daughter placement. Daughters built from previous frame's

std::vector<cv::Mat> Frame::getSynthFrame()
{
    return _synthFrame;
}
