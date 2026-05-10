#include "../includes/Frame.hpp"

#include <chrono>
#include <sstream>

namespace {
template <typename Fn>
void forEachSliceIndex(const SimulationConfig &config, int count, const Fn &fn)
{
    if (count <= 0) {
        return;
    }

    const bool useParallel = config.parallel_threads > 1 &&
                             count >= config.parallel_min_slices;
    if (!useParallel) {
        for (int i = 0; i < count; ++i) {
            fn(i);
        }
        return;
    }

    cv::parallel_for_(cv::Range(0, count), [&](const cv::Range &range) {
        for (int i = range.start; i < range.end; ++i) {
            fn(i);
        }
    });
}

double elapsedSeconds(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

double computeSizeReductionPenalty(const Spheroid &oldCell, const Spheroid &newCell, float weight)
{
    if (weight <= 0.0f) {
        return 0.0;
    }

    const auto accumulateReductionPenalty = [weight](float oldRadius, float newRadius) {
        if (oldRadius <= 0.0f || newRadius >= oldRadius) {
            return 0.0;
        }
        const double reductionRatio = static_cast<double>(oldRadius - newRadius) / oldRadius;
        return static_cast<double>(weight) * reductionRatio * reductionRatio;
    };

    return accumulateReductionPenalty(oldCell.getMajorRadius(), newCell.getMajorRadius())
         + accumulateReductionPenalty(oldCell.getBRadius(),     newCell.getBRadius())
         + accumulateReductionPenalty(oldCell.getMinorRadius(), newCell.getMinorRadius());
}
// Vincent's old split helpers (computeEquivalentSphereRadius,
} // namespace

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

Frame::Frame(const std::vector<cv::Mat> &realFrame, const SimulationConfig &simulationConfig, const std::vector<Spheroid> &cells,
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
    forEachSliceIndex(simulationConfig, static_cast<int>(_realFrame.size()), [&](int i) {
        _currentCostPerSlice[static_cast<size_t>(i)] =
            cv::norm(_realFrame[static_cast<size_t>(i)],
                     _synthFrame[static_cast<size_t>(i)],
                     cv::NORM_L2);
    });

    double totalCost = 0.0;
    for (double sliceCost : _currentCostPerSlice)
    {
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
        const int affectedCount = zMax - zMin + 1;
        forEachSliceIndex(simulationConfig, affectedCount, [&](int localIndex) {
            const int i = zMin + localIndex;
            outNewPerSlice[static_cast<size_t>(i)] =
                cv::norm(_realFrame[static_cast<size_t>(i)],
                         newSynthFrame[static_cast<size_t>(i)],
                         cv::NORM_L2);
        });
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
    std::vector<cv::Mat> frame(z_slices.size());

    forEachSliceIndex(simulationConfig, static_cast<int>(z_slices.size()), [&](int i) {
        const double z = z_slices[static_cast<size_t>(i)];
        Image synthImage = cv::Mat(shape, CV_32F, cv::Scalar(_backgroundValue));
        for (const auto &cell : cells)
        {
            cell.draw(synthImage, simulationConfig, z);
        }
        frame[static_cast<size_t>(i)] = synthImage;
    });
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

    double totalCost = 0.0;
    for (size_t i = 0; i < _realFrame.size(); ++i)
    {
        totalCost += cv::norm(_realFrame[i], synthFrame[i], cv::NORM_L2);
    }
    return totalCost;
}

std::vector<cv::Mat> Frame::generateSynthFrameFast(Spheroid &oldCell, Spheroid &newCell,
                                                   int *outAffectedZMin, int *outAffectedZMax)
{
    if (cells.empty())
    {
        std::cerr << "Cells are not set\n";
    }

    cv::Size shape = getImageShape(); // Assuming getImageShape() returns a cv::Size
    std::vector<cv::Mat> synthFrame = _synthFrame;

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

    for (size_t i = 0; i < z_slices.size(); ++i)
    {
        const double z = z_slices[i];
        if (z < minCorner[2] || z > maxCorner[2]) {
            continue;
        }
        if (affectedMin < 0) affectedMin = static_cast<int>(i);
        affectedMax = static_cast<int>(i);
    }

    if (affectedMin >= 0 && affectedMax >= affectedMin) {
        const int affectedCount = affectedMax - affectedMin + 1;
        forEachSliceIndex(simulationConfig, affectedCount, [&](int localIndex) {
            const int sliceIndex = affectedMin + localIndex;
            const double z = z_slices[static_cast<size_t>(sliceIndex)];
            cv::Mat synthImage = cv::Mat(shape, CV_32F, cv::Scalar(_backgroundValue));

            for (const auto &cell : cells)
            {
                cell.draw(synthImage, simulationConfig, z);
            }

            synthFrame[static_cast<size_t>(sliceIndex)] = synthImage;
        });
    }

    if (outAffectedZMin) *outAffectedZMin = affectedMin;
    if (outAffectedZMax) *outAffectedZMax = affectedMax;

    return synthFrame;
}

std::vector<cv::Mat> Frame::generateOutputFrame()
{
    std::vector<cv::Mat> realFrameWithOutlines(_realFrame.size());

    forEachSliceIndex(simulationConfig, static_cast<int>(_realFrame.size()), [&](int sliceIndex) {
        const size_t i = static_cast<size_t>(sliceIndex);
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

        realFrameWithOutlines[i] = outputFrame;
    });

    return realFrameWithOutlines;
}

std::vector<cv::Mat> Frame::generateOutputSynthFrame()
{
    std::vector<cv::Mat> outputSynthFrame(_synthFrame.size());

    forEachSliceIndex(simulationConfig, static_cast<int>(_synthFrame.size()), [&](int sliceIndex) {
        const size_t i = static_cast<size_t>(sliceIndex);
        const auto &synthImage = _synthFrame[i];
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

        outputSynthFrame[i] = outputImage;
    });

    return outputSynthFrame;
}

size_t Frame::length() const
{
    return cells.size();
}

CostCallbackPair Frame::perturbCell(size_t index, float overlapWeight, float sizeReductionWeight)
{
    if (index >= cells.size()) {
        return {0.0, [](bool) {}};
    }

    Spheroid oldCell = cells[index];
    PerturbDirections perturbDirections;

    // O(n) overlap for just this cell before perturbation
    double oldOverlapCell = computeOverlapForCell(index, overlapWeight);

    cells[index] = cells[index].getPerturbedCell(&perturbDirections);

    // Min-radius hard clamp (2026-04-09): prevent cells from ratcheting down to
    // minimum radius bounds via unconstrained perturbation. The Spheroid ctor
    // silently clamps majorRadius/minorRadius to minMajorRadius/minMinorRadius,
    // so a decrease-biased perturbation sequence parks cells at the floor where
    // the L2 cost rewards the tiny footprint (see 12345...3400 at (10,5) in
    // run 074740 f22 — the degenerate crumpled ellipse visible in the f8
    // screenshot of run 101212 was the same failure mode on 12345...341).
    // Revert any proposal that would take either radius FROM above the floor
    // TO the floor; proposals that were already at the floor are still allowed
    // (those cells are already parked there and need a different recovery
    // path — see the deferred volume recovery work in config).
    {
        const float newMajorR = cells[index].getMajorRadius();
        const float newMinorR = cells[index].getMinorRadius();
        const float oldMajorR = oldCell.getMajorRadius();
        const float oldMinorR = oldCell.getMinorRadius();
        const float minMajorR = static_cast<float>(Spheroid::cellConfig.minMajorRadius);
        const float minMinorR = static_cast<float>(Spheroid::cellConfig.minMinorRadius);
        constexpr float kClampEpsilon = 1e-3f;
        const bool hitMajorFloor = (newMajorR <= minMajorR + kClampEpsilon) &&
                                   (oldMajorR  >  minMajorR + kClampEpsilon);
        const bool hitMinorFloor = (newMinorR <= minMinorR + kClampEpsilon) &&
                                   (oldMinorR  >  minMinorR + kClampEpsilon);
        if (hitMajorFloor || hitMinorFloor) {
            cells[index] = oldCell;
            return {0.0, [](bool) {}};
        }
    }

    // O(n) overlap for this cell after perturbation
    double newOverlapCell = computeOverlapForCell(index, overlapWeight);
    double sizeReductionPenalty = computeSizeReductionPenalty(oldCell, cells[index], sizeReductionWeight);

    // Render only the affected z-slice range; generateSynthFrameFast writes
    // [affectedMin, affectedMax] (inclusive) for us to drive incremental
    // cost. Unchanged slices alias _synthFrame[i] — same pixel buffer, so
    // the cached per-slice L2 for those slices is bit-exact.
    int affectedMin = -1;
    int affectedMax = -1;
    auto newSynthFrame = generateSynthFrameFast(oldCell, cells[index],
                                                &affectedMin, &affectedMax);
    std::vector<double> newCostPerSlice;
    double newImageCost = calculateIncrementalCost(newSynthFrame,
                                                   affectedMin, affectedMax,
                                                   newCostPerSlice);
    // Use cached cost instead of recalculating L2 over all 225 slices
    double oldImageCost = _currentCost;

    double costDiff = (newImageCost + newOverlapCell + sizeReductionPenalty)
                    - (oldImageCost + oldOverlapCell);

    CallBackFunc callback = [this, newSynthFrame, newCostPerSlice,
                             oldCell, index, newImageCost, perturbDirections](bool accept)
    {
        const float brightnessStep = std::max(0.0f, Spheroid::cellConfig.brightnessProbabilityStep);
        const float majorRadiusStep = std::max(0.0f, Spheroid::cellConfig.majorRadiusProbabilityStep);
        const float minorRadiusStep = std::max(0.0f, Spheroid::cellConfig.minorRadiusProbabilityStep);
        const float abRatioStep = std::max(0.0f, Spheroid::cellConfig.abRatioProbabilityStep);
        if (accept) {
            if (perturbDirections.brightness != 0) this->cells[index].adjustBrightnessPerturbProbability(perturbDirections.brightness, brightnessStep);
            if (perturbDirections.majorRadius != 0) this->cells[index].adjustMajorRadiusPerturbProbability(perturbDirections.majorRadius, majorRadiusStep);
            if (perturbDirections.minorRadius != 0) this->cells[index].adjustMinorRadiusPerturbProbability(perturbDirections.minorRadius, minorRadiusStep);
            if (perturbDirections.abRatio != 0) this->cells[index].adjustABRatioPerturbProbability(perturbDirections.abRatio, abRatioStep);
            this->_synthFrame = newSynthFrame;
            this->_currentCost = newImageCost;
            this->_currentCostPerSlice = newCostPerSlice;
        } else {
            Spheroid revertedCell = oldCell;
            if (perturbDirections.brightness != 0) revertedCell.adjustBrightnessPerturbProbability(perturbDirections.brightness, -brightnessStep);
            if (perturbDirections.majorRadius != 0) revertedCell.adjustMajorRadiusPerturbProbability(perturbDirections.majorRadius, -majorRadiusStep);
            if (perturbDirections.minorRadius != 0) revertedCell.adjustMinorRadiusPerturbProbability(perturbDirections.minorRadius, -minorRadiusStep);
            if (perturbDirections.abRatio != 0) revertedCell.adjustABRatioPerturbProbability(perturbDirections.abRatio, -abRatioStep);
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
inline float boundingSphereRadius(const Spheroid &cell)
{
    return std::max({cell.getMajorRadius(), cell.getBRadius(), cell.getMinorRadius()});
}
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
                totalPenalty += weight * overlapRatio * overlapRatio;
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
            penalty += weight * overlapRatio * overlapRatio;
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

                // Voronoi test: find nearest claim point across all cells.
                // Keep this pixel only if the nearest point belongs to self.
                float selfBest = std::numeric_limits<float>::infinity();
                for (const auto &sp : selfClaimPoints) {
                    const float d2 = distSq(p, sp);
                    if (d2 < selfBest) selfBest = d2;
                }

                bool keep = true;
                for (const auto &kv : otherClaimSets) {
                    for (const auto &op : kv.second) {
                        const float d2 = distSq(p, op);
                        if (d2 < selfBest) {
                            keep = false;
                            break;
                        }
                    }
                    if (!keep) break;
                }

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

// PCA on weighted 3D points. Returns the principal eigenvector (longest
// direction) and fills D1/D2 with the centroids of the two halves obtained
// by projecting onto the eigenvector and splitting at the median.
bool pca3DWithCentroids(
    const std::vector<BrightPixel> &points,
    cv::Point3f &eigvec1Out,
    cv::Point3f &d1Out,
    cv::Point3f &d2Out)
{
    if (points.size() < 8) return false;

    // Weighted mean.
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

    // Weighted covariance.
    double cxx = 0, cxy = 0, cxz = 0, cyy = 0, cyz = 0, czz = 0;
    for (const auto &bp : points) {
        const double dx = bp.pos.x - mean.x;
        const double dy = bp.pos.y - mean.y;
        const double dz = bp.pos.z - mean.z;
        const double w = bp.weight;
        cxx += w * dx * dx;
        cxy += w * dx * dy;
        cxz += w * dx * dz;
        cyy += w * dy * dy;
        cyz += w * dy * dz;
        czz += w * dz * dz;
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
    cv::Point3f ev1(
        static_cast<float>(eigvecs(0, 0)),
        static_cast<float>(eigvecs(0, 1)),
        static_cast<float>(eigvecs(0, 2)));
    const double n = std::sqrt(ev1.x * ev1.x + ev1.y * ev1.y + ev1.z * ev1.z);
    if (n < 1e-9) return false;
    ev1.x = static_cast<float>(ev1.x / n);
    ev1.y = static_cast<float>(ev1.y / n);
    ev1.z = static_cast<float>(ev1.z / n);
    eigvec1Out = ev1;

    // Project every pixel onto ev1, find the median projection, partition
    // points into two groups, compute weighted centroid of each group.
    std::vector<float> projections;
    projections.reserve(points.size());
    for (const auto &bp : points) {
        const float px = bp.pos.x - mean.x;
        const float py = bp.pos.y - mean.y;
        const float pz = bp.pos.z - mean.z;
        projections.push_back(px * ev1.x + py * ev1.y + pz * ev1.z);
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
    const Spheroid &daughter1,
    const Spheroid &daughter2,
    double refParentVolume,
    const std::vector<Spheroid> &allCells,
    size_t d1Idx,
    size_t d2Idx,
    const ProbabilityConfig &probConfig,
    std::string &reasonOut)
{
    const auto cellVolume = [](const Spheroid &c) {
        return static_cast<double>(c.getMajorRadius()) *
               static_cast<double>(c.getBRadius()) *
               static_cast<double>(c.getMinorRadius());
    };

    const float d1R = std::max({daughter1.getMajorRadius(),
                                 daughter1.getBRadius(),
                                 daughter1.getMinorRadius()});
    const float d2R = std::max({daughter2.getMajorRadius(),
                                 daughter2.getBRadius(),
                                 daughter2.getMinorRadius()});

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
        const Spheroid &other = allCells[i];
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

    return true;
}

// Build a daughter Spheroid at position `center` with radii scaled from the
// given source radii by `volumeScale` and clamped to the config bounds.
// `srcMajor/srcB/srcMinor` are the reference dimensions the daughter should
// inherit from — typically the parent's last-frame snapshot values, so that
// Phase B perturbations that shrink the live parent do not also shrink the
// daughters below what the image actually supports.
Spheroid buildDaughter(
    const std::string &name,
    const cv::Point3f &center,
    const Spheroid &parent,
    float volumeScale,
    float srcMajor,
    float srcB,
    float srcMinor)
{
    const auto &cfg = Spheroid::cellConfig;
    const float dMajor = std::clamp(
        srcMajor * volumeScale,
        static_cast<float>(cfg.minMajorRadius),
        static_cast<float>(cfg.maxMajorRadius));
    const float dB = std::clamp(
        srcB * volumeScale,
        static_cast<float>(cfg.maxBRadius > 0.0 ? cfg.minBRadius : cfg.minMajorRadius),
        static_cast<float>(cfg.maxBRadius > 0.0 ? cfg.maxBRadius : cfg.maxMajorRadius));
    const float dMinor = std::clamp(
        srcMinor * volumeScale,
        static_cast<float>(cfg.minMinorRadius),
        static_cast<float>(cfg.maxMinorRadius));

    SpheroidParams dp(
        name,
        center.x, center.y, center.z,
        dMajor, dMinor,
        parent.getCellParams().theta_x,
        parent.getCellParams().theta_y,
        parent.getCellParams().theta_z,
        parent.getBrightness());
    dp.bRadius = dB;
    return Spheroid(dp);
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
    //   D1_seed = snapshot.center - 0.5 * longAxisLength * longAxisDir
    //   D2_seed = snapshot.center + 0.5 * longAxisLength * longAxisDir
    std::vector<cv::Point3f> selfClaim;
    if (snapshot.longAxisLength > 1e-3f) {
        const float half = 0.5f * snapshot.longAxisLength;
        selfClaim.push_back(cv::Point3f(
            snapshot.position.x - half * snapshot.longAxisDir.x,
            snapshot.position.y - half * snapshot.longAxisDir.y,
            snapshot.position.z - half * snapshot.longAxisDir.z));
        selfClaim.push_back(cv::Point3f(
            snapshot.position.x + half * snapshot.longAxisDir.x,
            snapshot.position.y + half * snapshot.longAxisDir.y,
            snapshot.position.z + half * snapshot.longAxisDir.z));
    } else {
        selfClaim.push_back(snapshot.position);
    }

    // Bounding box radius: 3x the largest snapshot semi-axis, same as the
    // split path's box radius. Uses snapshot (not live) radii so the
    // pre-pass sees the same region of space regardless of how Phase A
    // has since moved the parent.
    const float srcMajor = snapshot.majorRadius;
    const float srcB = (snapshot.bRadius > 1e-3f) ? snapshot.bRadius : snapshot.majorRadius;
    const float srcMinor = snapshot.minorRadius;
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

    cv::Point3f dirPca;
    if (!pca3DWithCentroids(pixels, dirPca, outD1, outD2)) return false;
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
    if (snapshot.longAxisLength > 1e-3f) {
        const float half = 0.5f * snapshot.longAxisLength;
        selfClaim.push_back(cv::Point3f(
            snapshot.position.x - half * snapshot.longAxisDir.x,
            snapshot.position.y - half * snapshot.longAxisDir.y,
            snapshot.position.z - half * snapshot.longAxisDir.z));
        selfClaim.push_back(cv::Point3f(
            snapshot.position.x + half * snapshot.longAxisDir.x,
            snapshot.position.y + half * snapshot.longAxisDir.y,
            snapshot.position.z + half * snapshot.longAxisDir.z));
    } else {
        selfClaim.push_back(snapshot.position);
    }

    const float srcMajor = snapshot.majorRadius;
    const float srcB = (snapshot.bRadius > 1e-3f) ? snapshot.bRadius : snapshot.majorRadius;
    const float srcMinor = snapshot.minorRadius;
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

    const Spheroid savedCell = cells[cellIndex];
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
    SpheroidParams cp = savedCell.getCellParams();
    cp.x = centroid.x;
    cp.y = centroid.y;
    cp.z = centroid.z;
    Spheroid candidate(cp);

    const double savedCost = _currentCost;
    const std::vector<cv::Mat> savedSynth = _synthFrame;
    const std::vector<double> savedPerSlice = _currentCostPerSlice;

    // Install candidate and compute incremental cost
    cells[cellIndex] = candidate;
    int affMin = -1, affMax = -1;
    Spheroid savedMutable = savedCell;
    Spheroid candidateMutable = candidate;
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
    const auto splitTimerStart = std::chrono::steady_clock::now();
    const auto noop = [](bool) {};
    if (cellIndex >= cells.size()) return {0.0, noop};

    // --- 0. Save live parent and install snapshot-state parent ---
    //
    // The split attempt must compare daughters against a FULL-SIZE parent
    // at the snapshot position/rotation, not against the Phase A/B-drifted
    // live parent. Otherwise the cost delta is dominated by "daughters vs
    // already-collapsed parent" which is small even for real divisions.
    //
    // Strategy: save the live parent, construct a snapshot-state Spheroid,
    // install it at cells[cellIndex], update _synthFrame + _currentCost
    // incrementally via generateSynthFrameFast. Everything downstream
    // (parent local, baseline cost, savedCells, candidate loop) then sees
    // the snapshot-state parent. On rejection we restore the live parent
    // to keep Phase B's perturbation progress.
    const Spheroid liveParent = cells[cellIndex];
    const std::string parentName = liveParent.getName();

    const bool snapshotValid = snapshot.valid &&
        snapshot.majorRadius > 1e-3f &&
        snapshot.minorRadius > 1e-3f;
    const float srcMajor = snapshotValid ? snapshot.majorRadius : liveParent.getMajorRadius();
    const float srcB     = (snapshotValid && snapshot.bRadius > 1e-3f)
        ? snapshot.bRadius : liveParent.getBRadius();
    const float srcMinor = snapshotValid ? snapshot.minorRadius : liveParent.getMinorRadius();

    // Build the snapshot-state parent: position, radii, rotation, and
    // brightness all come from the snapshot (falling back to live values
    // when snapshot is missing a field).
    Spheroid snapshotParent = liveParent;
    if (snapshotValid) {
        SpheroidParams snapParams(parentName,
                                  snapshot.position.x, snapshot.position.y, snapshot.position.z,
                                  srcMajor, srcMinor,
                                  snapshot.thetaX, snapshot.thetaY, snapshot.thetaZ,
                                  snapshot.brightness > 0.0f ? snapshot.brightness
                                                             : liveParent.getBrightness());
        snapParams.bRadius = srcB;
        snapshotParent = Spheroid(snapParams);

        const double liveCostBeforeSwap = _currentCost;

        cells[cellIndex] = snapshotParent;
        int affMinS = -1, affMaxS = -1;
        Spheroid liveMutable = liveParent;
        Spheroid snapshotMutable = snapshotParent;
        auto swappedSynth = generateSynthFrameFast(liveMutable, snapshotMutable,
                                                    &affMinS, &affMaxS);
        std::vector<double> swappedPerSlice;
        const double swappedImageCost = calculateIncrementalCost(swappedSynth,
                                                                   affMinS, affMaxS,
                                                                   swappedPerSlice);
        // Use min(liveCost, snapCost) as baseline. If the snapshot parent
        // is much worse than live (cell drifted between frames), keeping
        // the inflated snapshot baseline would make ANY daughter placement
        // look like an improvement — causing false splits. Using the
        // minimum ensures the split must beat the TIGHTER of the two fits.
        const bool useSnapshotBaseline = (swappedImageCost <= liveCostBeforeSwap);
        if (useSnapshotBaseline) {
            _synthFrame = swappedSynth;
            _currentCost = swappedImageCost;
            _currentCostPerSlice = swappedPerSlice;
        } else {
            // Revert: live parent was a better fit, keep it as baseline.
            cells[cellIndex] = liveParent;
            // _synthFrame / _currentCost / _currentCostPerSlice already
            // reflect the live parent (never changed).
        }

        std::cout << "  [Split Snapshot Parent] " << parentName
                  << " livePos=(" << liveParent.getX() << "," << liveParent.getY() << "," << liveParent.getZ() << ")"
                  << " snapPos=(" << snapshot.position.x << "," << snapshot.position.y << "," << snapshot.position.z << ")"
                  << " liveR=(" << liveParent.getMajorRadius() << "," << liveParent.getBRadius() << "," << liveParent.getMinorRadius() << ")"
                  << " snapR=(" << srcMajor << "," << srcB << "," << srcMinor << ")"
                  << " liveCost=" << liveCostBeforeSwap
                  << " snapCost=" << swappedImageCost
                  << " baseline=" << (useSnapshotBaseline ? "snapshot" : "live")
                  << std::endl;

        // Update snapshotParent to match what was actually installed so
        // restoreLiveParent works correctly on rejection paths.
        if (!useSnapshotBaseline) {
            snapshotParent = liveParent;
        }
    }

    // parent now reflects the installed snapshot-state (or live fallback
    // when snapshot is invalid). Everything downstream treats this as the
    // baseline parent.
    Spheroid parent = cells[cellIndex];

    // Restore-live-parent helper. Used on every rejection path (early
    // returns inside this function AND the callback's reject branch) to
    // undo the snapshot-state install so Phase B's live state isn't
    // lost. No-op if snapshot wasn't valid (no install happened).
    auto restoreLiveParent = [&]() {
        if (!snapshotValid) return;
        if (cellIndex >= cells.size()) return;
        cells[cellIndex] = liveParent;
        int affMinR = -1, affMaxR = -1;
        Spheroid snapshotMutable = snapshotParent;
        Spheroid liveMutable = liveParent;
        auto revertedSynth = generateSynthFrameFast(snapshotMutable, liveMutable,
                                                     &affMinR, &affMaxR);
        std::vector<double> revertedPerSlice;
        const double revertedCost = calculateIncrementalCost(revertedSynth,
                                                                affMinR, affMaxR,
                                                                revertedPerSlice);
        _synthFrame = revertedSynth;
        _currentCost = revertedCost;
        _currentCostPerSlice = revertedPerSlice;
    };

    // --- 1. Gather bright pixels in a snapshot-centered bounding box ---

    const float parentMajor = std::max(srcMajor, parent.getMajorRadius());
    const float parentB     = std::max(srcB,     parent.getBRadius());
    const float parentMinor = std::max(srcMinor, parent.getMinorRadius());
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
              << " snapLongLen=" << snapshot.longAxisLength
              << " src=(" << srcMajor << "," << srcB << "," << srcMinor << ")"
              << " liveR=(" << liveParent.getMajorRadius() << "," << liveParent.getBRadius() << "," << liveParent.getMinorRadius() << ")"
              << " livePos=(" << liveParent.getX() << "," << liveParent.getY() << "," << liveParent.getZ() << ")"
              << " parentNow=(" << parent.getMajorRadius() << "," << parent.getBRadius() << "," << parent.getMinorRadius() << ")"
              << " parentPos=(" << parent.getX() << "," << parent.getY() << "," << parent.getZ() << ")"
              << std::endl;

    // Self claim points for Voronoi test: the two expected-daughter seeds
    // along the snapshot long axis (if we have one) or just the snapshot
    // center (for round cells).
    //
    //   D1_seed = snapshot.center - 0.5 * longAxisLength * longAxisDir
    //   D2_seed = snapshot.center + 0.5 * longAxisLength * longAxisDir
    std::vector<cv::Point3f> selfClaim;
    if (snapshot.longAxisLength > 1e-3f) {
        const float half = 0.5f * snapshot.longAxisLength;
        selfClaim.push_back(cv::Point3f(
            snapshot.position.x - half * snapshot.longAxisDir.x,
            snapshot.position.y - half * snapshot.longAxisDir.y,
            snapshot.position.z - half * snapshot.longAxisDir.z));
        selfClaim.push_back(cv::Point3f(
            snapshot.position.x + half * snapshot.longAxisDir.x,
            snapshot.position.y + half * snapshot.longAxisDir.y,
            snapshot.position.z + half * snapshot.longAxisDir.z));
        std::cout << "  [Split Seeds] " << parentName
                  << " snapCenter=(" << snapshot.position.x << "," << snapshot.position.y << "," << snapshot.position.z << ")"
                  << " longAxisDir=(" << snapshot.longAxisDir.x << "," << snapshot.longAxisDir.y << "," << snapshot.longAxisDir.z << ")"
                  << " longAxisLen=" << snapshot.longAxisLength
                  << " D1_seed=(" << selfClaim[0].x << "," << selfClaim[0].y << "," << selfClaim[0].z << ")"
                  << " D2_seed=(" << selfClaim[1].x << "," << selfClaim[1].y << "," << selfClaim[1].z << ")"
                  << " boxRadius=" << boxRadius
                  << std::endl;
    } else {
        selfClaim.push_back(snapshot.position);
        std::cout << "  [Split Seeds] " << parentName
                  << " snapCenter=(" << snapshot.position.x << "," << snapshot.position.y << "," << snapshot.position.z << ")"
                  << " longAxisLen=0 (round cell, single seed = snapCenter)"
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
    const auto gatherPcaTimerStart = std::chrono::steady_clock::now();
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

    // --- 2. Primary placement: PCA direction + centroids ---
    //   D1_exp = centroid of group 1 (pixels below projection median on dirPca)
    //   D2_exp = centroid of group 2 (pixels above projection median on dirPca)
    cv::Point3f dirPca, d1Pca, d2Pca;
    if (!pca3DWithCentroids(pixels, dirPca, d1Pca, d2Pca)) {
        std::cout << "[Split Reject] " << parentName << " pca_failed" << std::endl;
        restoreLiveParent();
        return {0.0, noop};
    }

    std::cout << "  [Split PCA] " << parentName
              << " dirPca=(" << dirPca.x << "," << dirPca.y << "," << dirPca.z << ")"
              << " D1_exp=(" << d1Pca.x << "," << d1Pca.y << "," << d1Pca.z << ")"
              << " D2_exp=(" << d2Pca.x << "," << d2Pca.y << "," << d2Pca.z << ")"
              << " expSep=" << cv::norm(d1Pca - d2Pca)
              << std::endl;

    // Choose primary direction(s). For pre-classified cells with a trusted
    // snapshot direction, use it; if PCA disagrees by more than the
    // configured angle, try BOTH as separate primaries.
    std::vector<cv::Point3f> primaryDirs;
    const char *dirMode = "pca_only";
    float dirAngleDeg = 0.0f;
    if (useSnapshotDirection && snapshot.longAxisLength > 1e-3f) {
        primaryDirs.push_back(snapshot.longAxisDir);
        const float cosAngle = std::clamp(
            snapshot.longAxisDir.x * dirPca.x +
            snapshot.longAxisDir.y * dirPca.y +
            snapshot.longAxisDir.z * dirPca.z, -1.0f, 1.0f);
        const float angle = std::acos(std::abs(cosAngle)); // axis is undirected
        dirAngleDeg = angle * 180.0f / static_cast<float>(M_PI);
        const float agreeRad = probConfig.split_direction_agreement_degrees * static_cast<float>(M_PI) / 180.0f;
        if (angle >= agreeRad) {
            primaryDirs.push_back(dirPca);
            dirMode = "snapshot+pca";
        } else {
            dirMode = "snapshot_only";
        }
    } else {
        primaryDirs.push_back(dirPca);
    }
    std::cout << "  [Split Dirs] " << parentName
              << " mode=" << dirMode
              << " snapDir=(" << snapshot.longAxisDir.x << "," << snapshot.longAxisDir.y << "," << snapshot.longAxisDir.z << ")"
              << " pcaDir=(" << dirPca.x << "," << dirPca.y << "," << dirPca.z << ")"
              << " angleDeg=" << dirAngleDeg
              << " agreeThresh=" << probConfig.split_direction_agreement_degrees
              << " nPrimaries=" << primaryDirs.size()
              << " nPixels=" << pixels.size()
              << std::endl;

    // --- 3. Generate K candidate placements around each (midpoint, direction) pair ---
    //
    // Two midpoint options, each with its own length:
    //   pca_mid  = 0.5 * (d1Pca + d2Pca)         ← analytic image midpoint
    //   snap_mid = snapshot.position             ← last-frame cell center
    //
    // And one or two directions from the earlier direction-agreement logic.
    // For each (midpoint, direction) pair we generate a primary candidate
    // plus rotation and translation variants. Both midpoints are tried and
    // burn-in / cost evaluation picks the winner. If pca_mid and snap_mid
    // are essentially the same point (<0.5 voxels apart), we only use one.
    struct Candidate {
        cv::Point3f d1Pos;
        cv::Point3f d2Pos;
        std::string label; // "pca_mid" / "snap_mid" with direction suffix
    };

    const float volumeScale = std::cbrt(0.5f);
    const float daughterR = std::max(0.5f * parentMajor, 5.0f);
    const float rotDeltaRad =
        probConfig.split_candidate_rotation_delta_degrees * static_cast<float>(M_PI) / 180.0f;
    const float transDelta =
        probConfig.split_candidate_translation_delta_fraction * daughterR;

    const cv::Point3f pcaMidpoint(
        0.5f * (d1Pca.x + d2Pca.x),
        0.5f * (d1Pca.y + d2Pca.y),
        0.5f * (d1Pca.z + d2Pca.z));
    const float pcaSep = static_cast<float>(cv::norm(d1Pca - d2Pca));

    struct MidpointOption {
        cv::Point3f center;
        float separation;
        std::string label;
    };
    std::vector<MidpointOption> midpoints;
    midpoints.push_back({pcaMidpoint, pcaSep, "pca_mid"});

    if (snapshotValid && snapshot.longAxisLength > 1e-3f) {
        const cv::Point3f snapMid = snapshot.position;
        const float midpointDist = static_cast<float>(cv::norm(pcaMidpoint - snapMid));
        if (midpointDist > 0.5f) {
            midpoints.push_back({snapMid, snapshot.longAxisLength, "snap_mid"});
        }
    }

    std::cout << "  [Split Midpoints] " << parentName
              << " pcaMid=(" << pcaMidpoint.x << "," << pcaMidpoint.y << "," << pcaMidpoint.z << ")"
              << " pcaSep=" << pcaSep
              << " snapMid=(" << snapshot.position.x << "," << snapshot.position.y << "," << snapshot.position.z << ")"
              << " snapLen=" << snapshot.longAxisLength
              << " nMidpoints=" << midpoints.size()
              << std::endl;

    std::vector<Candidate> candidates;
    for (const auto &dir0 : primaryDirs) {
        cv::Point3f perpU, perpV;
        orthonormalFrame(dir0, perpU, perpV);

        for (const auto &mp : midpoints) {
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
    const double gatherPcaSeconds = elapsedSeconds(gatherPcaTimerStart);

    // --- 4. Evaluate each candidate via a short burn-in ---
    // Save the pre-split state to revert cheaply. Use the current _synthFrame
    // and per-slice cost cache as the baseline.
    const double baselineImageCost = _currentCost;
    const double baselineOverlap = computeOverlapPenalty(probConfig.overlap_penalty_weight);
    const double baselineTotal = baselineImageCost + baselineOverlap;
    const std::vector<cv::Mat> savedSynth = _synthFrame;
    const std::vector<double> savedPerSlice = _currentCostPerSlice;
    const double savedCost = _currentCost;
    const std::vector<Spheroid> savedCells = cells;

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
    std::vector<Spheroid> bestCells;
    std::vector<cv::Mat> bestSynth;
    std::vector<double> bestPerSlice;
    double bestImageCost = 0.0;
    double bestOverlap = 0.0;
    cv::Point3f bestSeedD1{0, 0, 0};
    cv::Point3f bestSeedD2{0, 0, 0};
    std::string bestLabel;

    const int burnIters = std::max(0, probConfig.split_candidate_burn_in_iterations);

    // Install tight position AND radius sigmas for candidate burn-in.
    //
    // Main-loop position sigmas (x=5, y=5, z=8) let a daughter drift
    // 15-25 voxels across 20 iters, far enough to escape the parent
    // footprint. Scaling by split_burn_in_pos_sigma_scale (0.4 default)
    // restricts burn-in to refinement distances (<10 voxels).
    //
    // Main-loop radius sigmas (~2 per axis) let a daughter's minorR
    // collapse 10+ voxels over 50 iters of burn-in + refine, producing
    // the "collapsed sliver" pattern where one daughter drops to the
    // radius floor and hides at a distant image spot. Scaling by
    // split_burn_in_radius_sigma_scale (0.1 default) freezes the
    // snapshot-based daughter sizing through burn-in.
    //
    // Global static state mutation is safe here — single-threaded
    // optimizer, restored on every exit path below.
    const float posScale = std::max(0.0f, probConfig.split_burn_in_pos_sigma_scale);
    const float radiusScale = std::max(0.0f, probConfig.split_burn_in_radius_sigma_scale);
    PerturbParams savedPerturbX = Spheroid::cellConfig.x;
    PerturbParams savedPerturbY = Spheroid::cellConfig.y;
    PerturbParams savedPerturbZ = Spheroid::cellConfig.z;
    PerturbParams savedPerturbMajor = Spheroid::cellConfig.majorRadius;
    PerturbParams savedPerturbB = Spheroid::cellConfig.bRadius;
    PerturbParams savedPerturbMinor = Spheroid::cellConfig.minorRadius;
    Spheroid::cellConfig.x.sigma = savedPerturbX.sigma * posScale;
    Spheroid::cellConfig.y.sigma = savedPerturbY.sigma * posScale;
    Spheroid::cellConfig.z.sigma = savedPerturbZ.sigma * posScale;
    Spheroid::cellConfig.majorRadius.sigma = savedPerturbMajor.sigma * radiusScale;
    Spheroid::cellConfig.bRadius.sigma     = savedPerturbB.sigma     * radiusScale;
    Spheroid::cellConfig.minorRadius.sigma = savedPerturbMinor.sigma * radiusScale;

    std::cout << "  [Split Sigmas] " << parentName
              << " posScale=" << posScale
              << " xSigma=" << savedPerturbX.sigma << "->" << Spheroid::cellConfig.x.sigma
              << " ySigma=" << savedPerturbY.sigma << "->" << Spheroid::cellConfig.y.sigma
              << " zSigma=" << savedPerturbZ.sigma << "->" << Spheroid::cellConfig.z.sigma
              << " radiusScale=" << radiusScale
              << " majorSigma=" << savedPerturbMajor.sigma << "->" << Spheroid::cellConfig.majorRadius.sigma
              << " bSigma=" << savedPerturbB.sigma << "->" << Spheroid::cellConfig.bRadius.sigma
              << " minorSigma=" << savedPerturbMinor.sigma << "->" << Spheroid::cellConfig.minorRadius.sigma
              << std::endl;

    const auto candidateTimerStart = std::chrono::steady_clock::now();
    for (size_t ci = 0; ci < candidates.size(); ++ci) {
        const auto &cand = candidates[ci];
        Spheroid child1 = buildDaughter(parentName + "0", cand.d1Pos, parent,
                                         volumeScale, srcMajor, srcB, srcMinor);
        Spheroid child2 = buildDaughter(parentName + "1", cand.d2Pos, parent,
                                         volumeScale, srcMajor, srcB, srcMinor);

        // Replace parent with daughters.
        cells.erase(cells.begin() + cellIndex);
        cells.push_back(child1);
        cells.push_back(child2);
        const size_t d1Idx = cells.size() - 2;
        const size_t d2Idx = cells.size() - 1;

        // Full render + cost refresh (candidate set is small, K<=5).
        _synthFrame = generateSynthFrame();
        refreshFullCostCache();

        // Short alternating burn-in on each daughter.
        for (int it = 0; it < burnIters; ++it) {
            const size_t target = (it % 2 == 0) ? d1Idx : d2Idx;
            CostCallbackPair cp = perturbCell(target,
                                              probConfig.overlap_penalty_weight,
                                              probConfig.size_reduction_penalty_weight);
            const bool accept = cp.first < 0.0;
            cp.second(accept);
        }

        const double candImageCost = _currentCost;
        const double candOverlap = computeOverlapPenalty(probConfig.overlap_penalty_weight);
        const double candTotal = candImageCost + candOverlap;

        const Spheroid &candD1 = cells[d1Idx];
        const Spheroid &candD2 = cells[d2Idx];
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

        if (candTotal < bestTotal) {
            bestTotal = candTotal;
            bestIdx = static_cast<int>(ci);
            bestCells = cells;
            bestSynth = _synthFrame;
            bestPerSlice = _currentCostPerSlice;
            bestImageCost = candImageCost;
            bestOverlap = candOverlap;
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
    const double candidateSeconds = elapsedSeconds(candidateTimerStart);

    if (bestIdx < 0) {
        // Restore main-loop perturbation sigmas before the early return.
        Spheroid::cellConfig.x = savedPerturbX;
        Spheroid::cellConfig.y = savedPerturbY;
        Spheroid::cellConfig.z = savedPerturbZ;
        Spheroid::cellConfig.majorRadius = savedPerturbMajor;
        Spheroid::cellConfig.bRadius     = savedPerturbB;
        Spheroid::cellConfig.minorRadius = savedPerturbMinor;
        restoreLiveParent();
        return {0.0, noop};
    }

    // --- 4b. Final refine burn-in on the winning candidate ---
    // The candidate loop runs a short (~20 iter) burn-in per candidate so
    // the K=5 comparison is cheap. Now that we've picked a winner, give it
    // an extra refine pass with the same tight sigmas so the chosen
    // daughters can settle before bio/cost gates fire. This runs on the
    // best candidate's state (reinstalled), and the post-refine state is
    // re-captured as bestCells / bestSynth / etc.
    const int refineIters = std::max(0, probConfig.split_final_refine_iterations);
    double refineSeconds = 0.0;
    if (refineIters > 0) {
        const auto refineTimerStart = std::chrono::steady_clock::now();
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
        const double preRefineTotal = _currentCost +
            computeOverlapPenalty(probConfig.overlap_penalty_weight);

        int refineAccepts = 0;
        for (int it = 0; it < refineIters; ++it) {
            const size_t target = (it % 2 == 0) ? d1IdxRefine : d2IdxRefine;
            CostCallbackPair cp = perturbCell(target,
                                              probConfig.overlap_penalty_weight,
                                              probConfig.size_reduction_penalty_weight);
            const bool accept = cp.first < 0.0;
            if (accept) ++refineAccepts;
            cp.second(accept);
        }

        // Re-capture refined state as new best.
        bestCells = cells;
        bestSynth = _synthFrame;
        bestPerSlice = _currentCostPerSlice;
        bestImageCost = _currentCost;
        bestOverlap = computeOverlapPenalty(probConfig.overlap_penalty_weight);
        bestTotal = _currentCost + bestOverlap;

        const cv::Point3f postRefineD1(cells[d1IdxRefine].getX(),
                                         cells[d1IdxRefine].getY(),
                                         cells[d1IdxRefine].getZ());
        const cv::Point3f postRefineD2(cells[d2IdxRefine].getX(),
                                         cells[d2IdxRefine].getY(),
                                         cells[d2IdxRefine].getZ());
        // Capture daughter radii for the radius-drift diagnostic. Built-in
        // radii from buildDaughter are `0.794 * src`; after burn-in +
        // refine these should stay near the built-in values because
        // split_burn_in_radius_sigma_scale freezes them. If a daughter's
        // minor/b radius has drifted more than a few voxels from the
        // built-in, the radius sigma lock isn't working as intended.
        const Spheroid &refinedD1 = cells[d1IdxRefine];
        const Spheroid &refinedD2 = cells[d2IdxRefine];
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
                  << " d1R=(" << refinedD1.getMajorRadius() << "," << refinedD1.getBRadius() << "," << refinedD1.getMinorRadius() << ")"
                  << " d2R=(" << refinedD2.getMajorRadius() << "," << refinedD2.getBRadius() << "," << refinedD2.getMinorRadius() << ")"
                  << std::endl;

        // Revert to pre-split state — gates run on savedCells baseline
        // against bestCells (the refined winner).
        cells = savedCells;
        _synthFrame = savedSynth;
        _currentCost = savedCost;
        _currentCostPerSlice = savedPerSlice;
        refineSeconds = elapsedSeconds(refineTimerStart);
    }

    // Restore main-loop perturbation sigmas before the gate sequence.
    Spheroid::cellConfig.x = savedPerturbX;
    Spheroid::cellConfig.y = savedPerturbY;
    Spheroid::cellConfig.z = savedPerturbZ;
    Spheroid::cellConfig.majorRadius = savedPerturbMajor;
    Spheroid::cellConfig.bRadius     = savedPerturbB;
    Spheroid::cellConfig.minorRadius = savedPerturbMinor;

    std::cout << "  [Split Timing] " << parentName
              << " total_so_far_sec=" << elapsedSeconds(splitTimerStart)
              << " gather_pca_sec=" << gatherPcaSeconds
              << " candidate_eval_sec=" << candidateSeconds
              << " final_refine_sec=" << refineSeconds
              << " candidates=" << candidates.size()
              << " burn_iters=" << burnIters
              << " refine_iters=" << refineIters
              << " parallel_threads=" << simulationConfig.parallel_threads
              << std::endl;

    // --- 5. Bio checks on the best candidate's final state ---
    // Rebuild daughter indices from bestCells (parent was at cellIndex,
    // daughters are the last two entries in bestCells since that's how we
    // replaced them during evaluation).
    const size_t d1IdxBest = bestCells.size() - 2;
    const size_t d2IdxBest = bestCells.size() - 1;
    const Spheroid &bestD1 = bestCells[d1IdxBest];
    const Spheroid &bestD2 = bestCells[d2IdxBest];

    // 5a. Drift-from-seed gate. Reject if either daughter center has
    // wandered too far from its initial candidate placement during burn-in.
    // The limit is max(parent_frac * srcMaxR, daughter_frac * daughterMaxR).
    const float bestD1MaxR = std::max({bestD1.getMajorRadius(),
                                         bestD1.getBRadius(),
                                         bestD1.getMinorRadius()});
    const float bestD2MaxR = std::max({bestD2.getMajorRadius(),
                                         bestD2.getBRadius(),
                                         bestD2.getMinorRadius()});
    const float daughterMaxR = std::max(bestD1MaxR, bestD2MaxR);
    const float driftLimit = std::max(
        probConfig.bio_max_drift_parent_fraction * srcMaxR,
        probConfig.bio_max_drift_daughter_fraction * daughterMaxR);
    const float drift1 = static_cast<float>(cv::norm(
        cv::Point3f(bestD1.getX(), bestD1.getY(), bestD1.getZ()) - bestSeedD1));
    const float drift2 = static_cast<float>(cv::norm(
        cv::Point3f(bestD2.getX(), bestD2.getY(), bestD2.getZ()) - bestSeedD2));
    if (drift1 > driftLimit || drift2 > driftLimit) {
        std::cout << "[Split Reject bio] " << parentName
                  << " reason=drift_from_seed d1=" << drift1
                  << " d2=" << drift2
                  << " limit=" << driftLimit << std::endl;
        restoreLiveParent();
        return {0.0, noop};
    }

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
    bool bridgeStatsValid = false;
    float bridgeGapDensity = 1.0f;
    float bridgeValleyRatio = 1.0f;

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
    // Rejection requires BOTH signals to indicate a flat profile:
    //   gap_density > bio_bridge_max_gap_density  AND
    //   valley_ratio > bio_bridge_max_valley_ratio
    //
    // This deliberately requires both to fire so real divisions with
    // only partial grooves still pass.
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
            const float halfLen = 0.5f * axisLen;

            int totalInRange = 0;
            int gapCount = 0;
            int edgeCount = 0;
            double gapBrightSum = 0.0;
            double edgeBrightSum = 0.0;

            for (const auto &bp : pixels) {
                const cv::Point3f delta(
                    bp.pos.x - daughterMidpoint.x,
                    bp.pos.y - daughterMidpoint.y,
                    bp.pos.z - daughterMidpoint.z);
                const float signedProj =
                    delta.x * axisDir.x +
                    delta.y * axisDir.y +
                    delta.z * axisDir.z;
                const float t = signedProj / halfLen;
                const float absT = std::abs(t);
                if (absT > 1.5f) continue;
                ++totalInRange;

                if (absT < 0.3f) {
                    ++gapCount;
                    gapBrightSum += bp.weight;
                } else if (absT > 0.6f && absT < 1.1f) {
                    ++edgeCount;
                    edgeBrightSum += bp.weight;
                }
            }

            const float gapDensity = (totalInRange > 0)
                ? static_cast<float>(gapCount) / static_cast<float>(totalInRange)
                : 0.0f;
            const float gapBright = (gapCount > 0)
                ? static_cast<float>(gapBrightSum / gapCount)
                : 0.0f;
            const float edgeBright = (edgeCount > 0)
                ? static_cast<float>(edgeBrightSum / edgeCount)
                : 0.0f;
            const float valleyRatio = (edgeBright > 1e-6f)
                ? (gapBright / edgeBright)
                : 0.0f;
            bridgeStatsValid = edgeCount > 0;
            bridgeGapDensity = gapDensity;
            bridgeValleyRatio = valleyRatio;

            std::cout << "  [Split Bridge] " << parentName
                      << " totalInRange=" << totalInRange
                      << " gapCount=" << gapCount
                      << " edgeCount=" << edgeCount
                      << " gapDensity=" << gapDensity
                      << " gapBright=" << gapBright
                      << " edgeBright=" << edgeBright
                      << " valleyRatio=" << valleyRatio
                      << std::endl;

            const bool densityFlat = gapDensity > probConfig.bio_bridge_max_gap_density;
            const bool brightnessFlat = valleyRatio > probConfig.bio_bridge_max_valley_ratio;
            if (densityFlat && brightnessFlat && edgeCount > 0) {
                std::cout << "[Split Reject bio] " << parentName
                          << " reason=bridge_flat"
                          << " gapDensity=" << gapDensity
                          << " valleyRatio=" << valleyRatio
                          << " densityLimit=" << probConfig.bio_bridge_max_gap_density
                          << " valleyLimit=" << probConfig.bio_bridge_max_valley_ratio
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
                  << " r1=(" << bestD1.getMajorRadius() << "," << bestD1.getBRadius() << "," << bestD1.getMinorRadius() << ")"
                  << " d2=(" << bestD2.getX() << "," << bestD2.getY() << "," << bestD2.getZ() << ")"
                  << " r2=(" << bestD2.getMajorRadius() << "," << bestD2.getBRadius() << "," << bestD2.getMinorRadius() << ")"
                  << " refParentVolume=" << refParentVolume
                  << std::endl;
        restoreLiveParent();
        return {0.0, noop};
    }

    // --- 6. Cost check ---
    const double costDiff = bestTotal - baselineTotal;
    const double standardSplitThreshold = -static_cast<double>(probConfig.split_cost);
    const double rescueSplitThreshold =
        -static_cast<double>(probConfig.split_cost) *
        static_cast<double>(probConfig.split_cost_rescue_min_fraction);
    const double perfectBridgeRescueSplitThreshold =
        -static_cast<double>(probConfig.split_cost) *
        static_cast<double>(probConfig.split_cost_perfect_bridge_rescue_min_fraction);
    const float rescueDriftLimit =
        driftLimit * std::max(0.0f, probConfig.split_cost_rescue_max_drift_fraction);

    const bool rescueNearMissCost =
        costDiff < 0.0 &&
        costDiff >= standardSplitThreshold &&
        costDiff < rescueSplitThreshold;
    const bool perfectBridgeRescueNearMissCost =
        costDiff < 0.0 &&
        costDiff >= standardSplitThreshold &&
        costDiff < perfectBridgeRescueSplitThreshold;
    const bool rescueStrongBridge =
        bridgeStatsValid &&
        bridgeGapDensity < probConfig.split_cost_rescue_max_gap_density &&
        bridgeValleyRatio < probConfig.split_cost_rescue_max_valley_ratio;
    const bool perfectBridgeRescueStrongBridge =
        bridgeStatsValid &&
        bridgeGapDensity < probConfig.split_cost_perfect_bridge_rescue_max_gap_density &&
        bridgeValleyRatio < probConfig.split_cost_perfect_bridge_rescue_max_valley_ratio;
    const bool rescueStableDaughters =
        drift1 <= rescueDriftLimit &&
        drift2 <= rescueDriftLimit;
    const bool rescueLowOverlap =
        bestOverlap <= probConfig.split_cost_rescue_max_overlap_penalty;
    const bool perfectBridgeRescueLowOverlap =
        bestOverlap <= probConfig.split_cost_perfect_bridge_rescue_max_overlap_penalty;
    const bool rescueAccept =
        probConfig.split_cost_rescue_enabled &&
        rescueNearMissCost &&
        rescueStrongBridge &&
        rescueStableDaughters &&
        rescueLowOverlap;
    const bool perfectBridgeRescueAccept =
        probConfig.split_cost_perfect_bridge_rescue_enabled &&
        perfectBridgeRescueNearMissCost &&
        perfectBridgeRescueStrongBridge &&
        rescueStableDaughters &&
        perfectBridgeRescueLowOverlap;

    if (costDiff >= standardSplitThreshold && !rescueAccept && !perfectBridgeRescueAccept) {
        std::cout << "[Split Reject cost] " << parentName
                  << " diff=" << costDiff
                  << " threshold=" << standardSplitThreshold
                  << " rescueThreshold=" << rescueSplitThreshold
                  << " perfectBridgeRescueThreshold=" << perfectBridgeRescueSplitThreshold
                  << " rescueNearMissCost=" << rescueNearMissCost
                  << " perfectBridgeRescueNearMissCost=" << perfectBridgeRescueNearMissCost
                  << " bridgeStatsValid=" << bridgeStatsValid
                  << " gapDensity=" << bridgeGapDensity
                  << " valleyRatio=" << bridgeValleyRatio
                  << " rescueStrongBridge=" << rescueStrongBridge
                  << " perfectBridgeRescueStrongBridge=" << perfectBridgeRescueStrongBridge
                  << " overlap=" << bestOverlap
                  << " rescueLowOverlap=" << rescueLowOverlap
                  << " perfectBridgeRescueLowOverlap=" << perfectBridgeRescueLowOverlap
                  << " rescueDriftLimit=" << rescueDriftLimit
                  << " rescueStableDaughters=" << rescueStableDaughters
                  << " bestIdx=" << bestIdx
                  << " bestLabel=" << bestLabel
                  << " d1=(" << bestD1.getX() << "," << bestD1.getY() << "," << bestD1.getZ() << ")"
                  << " r1=(" << bestD1.getMajorRadius() << "," << bestD1.getBRadius() << "," << bestD1.getMinorRadius() << ")"
                  << " drift1=" << drift1
                  << " d2=(" << bestD2.getX() << "," << bestD2.getY() << "," << bestD2.getZ() << ")"
                  << " r2=(" << bestD2.getMajorRadius() << "," << bestD2.getBRadius() << "," << bestD2.getMinorRadius() << ")"
                  << " drift2=" << drift2
                  << std::endl;
        restoreLiveParent();
        return {0.0, noop};
    }
    if (rescueAccept) {
        std::cout << "[Split Cost Rescue Accepted] " << parentName
                  << " diff=" << costDiff
                  << " standardThreshold=" << standardSplitThreshold
                  << " rescueThreshold=" << rescueSplitThreshold
                  << " gapDensity=" << bridgeGapDensity
                  << " valleyRatio=" << bridgeValleyRatio
                  << " overlap=" << bestOverlap
                  << " drift1=" << drift1
                  << " drift2=" << drift2
                  << " rescueDriftLimit=" << rescueDriftLimit
                  << " bestIdx=" << bestIdx
                  << " bestLabel=" << bestLabel
                  << std::endl;
    } else if (perfectBridgeRescueAccept) {
        std::cout << "[Split Perfect Bridge Rescue Accepted] " << parentName
                  << " diff=" << costDiff
                  << " standardThreshold=" << standardSplitThreshold
                  << " rescueThreshold=" << rescueSplitThreshold
                  << " perfectBridgeRescueThreshold=" << perfectBridgeRescueSplitThreshold
                  << " gapDensity=" << bridgeGapDensity
                  << " valleyRatio=" << bridgeValleyRatio
                  << " overlap=" << bestOverlap
                  << " drift1=" << drift1
                  << " drift2=" << drift2
                  << " rescueDriftLimit=" << rescueDriftLimit
                  << " bestIdx=" << bestIdx
                  << " bestLabel=" << bestLabel
                  << std::endl;
    }

    // Accept: install the best candidate state. The callback applies on
    // accept; the caller uses perturbCell's contract where the callback is
    // invoked after the decision. To stay consistent with that contract we
    // return the (costDiff, callback) pair that installs bestCells state.
    auto savedCellsCopy = savedCells;
    auto savedSynthCopy = savedSynth;
    auto savedPerSliceCopy = savedPerSlice;
    double savedCostCopy = savedCost;

    const cv::Point3f acceptedD1Pos(bestD1.getX(), bestD1.getY(), bestD1.getZ());
    const cv::Point3f acceptedD2Pos(bestD2.getX(), bestD2.getY(), bestD2.getZ());
    const cv::Point3f acceptedD1R(bestD1.getMajorRadius(), bestD1.getBRadius(), bestD1.getMinorRadius());
    const cv::Point3f acceptedD2R(bestD2.getMajorRadius(), bestD2.getBRadius(), bestD2.getMinorRadius());
    const float acceptedDrift1 = drift1;
    const float acceptedDrift2 = drift2;
    const cv::Point3f acceptedSeed1 = bestSeedD1;
    const cv::Point3f acceptedSeed2 = bestSeedD2;

    // Capture extras for the callback's reject branch — it needs to
    // undo the snapshot-state install so Phase B's live state is restored.
    const Spheroid liveParentCopy = liveParent;
    const Spheroid snapshotParentCopy = snapshotParent;
    const size_t cellIndexCopy = cellIndex;
    const bool snapshotValidCopy = snapshotValid;

    const std::string acceptedLabel = bestLabel;

    CallBackFunc callback = [this, bestCells, bestSynth, bestPerSlice, bestImageCost,
                             savedCellsCopy, savedSynthCopy, savedPerSliceCopy, savedCostCopy,
                             parentName, costDiff, acceptedD1Pos, acceptedD2Pos,
                             acceptedD1R, acceptedD2R, acceptedSeed1, acceptedSeed2,
                             acceptedDrift1, acceptedDrift2, acceptedLabel,
                             liveParentCopy, snapshotParentCopy, cellIndexCopy,
                             snapshotValidCopy](bool accept) mutable {
        if (accept) {
            this->cells = bestCells;
            this->_synthFrame = bestSynth;
            this->_currentCostPerSlice = bestPerSlice;
            this->_currentCost = bestImageCost;
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
            // so Phase B's live state isn't lost.
            this->cells = savedCellsCopy;
            this->_synthFrame = savedSynthCopy;
            this->_currentCostPerSlice = savedPerSliceCopy;
            this->_currentCost = savedCostCopy;

            if (snapshotValidCopy && cellIndexCopy < this->cells.size()) {
                this->cells[cellIndexCopy] = liveParentCopy;
                int affMinR = -1, affMaxR = -1;
                Spheroid snapshotMutable = snapshotParentCopy;
                Spheroid liveMutable = liveParentCopy;
                auto revertedSynth = this->generateSynthFrameFast(snapshotMutable, liveMutable,
                                                                    &affMinR, &affMaxR);
                std::vector<double> revertedPerSlice;
                const double revertedCost = this->calculateIncrementalCost(revertedSynth,
                                                                             affMinR, affMaxR,
                                                                             revertedPerSlice);
                this->_synthFrame = revertedSynth;
                this->_currentCost = revertedCost;
                this->_currentCostPerSlice = revertedPerSlice;
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
