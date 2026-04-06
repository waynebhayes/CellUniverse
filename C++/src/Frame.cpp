#include "../includes/Frame.hpp"

namespace {
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
         + accumulateReductionPenalty(oldCell.getMinorRadius(), newCell.getMinorRadius());
}

double computeEquivalentSphereRadius(const Spheroid &cell)
{
    const double a = static_cast<double>(cell.getMajorRadius());
    const double c = static_cast<double>(cell.getMinorRadius());
    if (a <= 0.0 || c <= 0.0) {
        return 0.0;
    }
    return std::cbrt(a * a * c);
}

double computeSphereIntersectionVolume(double r1, double r2, double dist)
{
    if (r1 <= 0.0 || r2 <= 0.0) {
        return 0.0;
    }
    if (dist >= r1 + r2) {
        return 0.0;
    }
    if (dist <= std::abs(r1 - r2)) {
        const double rMin = std::min(r1, r2);
        return (4.0 / 3.0) * M_PI * rMin * rMin * rMin;
    }

    const double term1 = r1 + r2 - dist;
    const double term2 = dist * dist + 2.0 * dist * (r1 + r2) - 3.0 * (r1 - r2) * (r1 - r2);
    return M_PI * term1 * term1 * term2 / (12.0 * dist);
}

double computeOverlapVolumeFractionApprox(const Spheroid &cell1, const Spheroid &cell2)
{
    const double r1 = computeEquivalentSphereRadius(cell1);
    const double r2 = computeEquivalentSphereRadius(cell2);
    const double v1 = (4.0 / 3.0) * M_PI * r1 * r1 * r1;
    const double v2 = (4.0 / 3.0) * M_PI * r2 * r2 * r2;
    const double minVolume = std::min(v1, v2);
    if (minVolume <= 0.0) {
        return 0.0;
    }

    const double dx = static_cast<double>(cell1.getX()) - static_cast<double>(cell2.getX());
    const double dy = static_cast<double>(cell1.getY()) - static_cast<double>(cell2.getY());
    const double dz = static_cast<double>(cell1.getZ()) - static_cast<double>(cell2.getZ());
    const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double overlapVolume = computeSphereIntersectionVolume(r1, r2, dist);
    return overlapVolume / minVolume;
}

double computeMaxDaughterRadiusRatio(const Spheroid &cell1, const Spheroid &cell2)
{
    const double major1 = static_cast<double>(cell1.getMajorRadius());
    const double major2 = static_cast<double>(cell2.getMajorRadius());
    const double minor1 = static_cast<double>(cell1.getMinorRadius());
    const double minor2 = static_cast<double>(cell2.getMinorRadius());

    const auto safeRatio = [](double a, double b) {
        const double minValue = std::min(a, b);
        const double maxValue = std::max(a, b);
        if (minValue <= 1e-9) {
            return std::numeric_limits<double>::infinity();
        }
        return maxValue / minValue;
    };

    return std::max(safeRatio(major1, major2), safeRatio(minor1, minor2));
}

double computeSpheroidVolume(const Spheroid &cell)
{
    const double a = static_cast<double>(cell.getMajorRadius());
    const double c = static_cast<double>(cell.getMinorRadius());
    if (a <= 0.0 || c <= 0.0) {
        return 0.0;
    }
    return (4.0 / 3.0) * M_PI * a * a * c;
}

double computeCylinderMeanBrightnessAlongSegment(const std::vector<cv::Mat> &frame,
                                                const cv::Point3f &center,
                                                const cv::Point3f &axisUnit,
                                                double cylinderLength,
                                                double cylinderRadius)
{
    if (frame.empty() || cylinderLength <= 0.0 || cylinderRadius <= 0.0) {
        return 0.0;
    }

    const double axisX = static_cast<double>(axisUnit.x);
    const double axisY = static_cast<double>(axisUnit.y);
    const double axisZ = static_cast<double>(axisUnit.z);
    const double axisNorm = std::sqrt(axisX * axisX + axisY * axisY + axisZ * axisZ);
    if (axisNorm <= 1e-6) {
        return 0.0;
    }

    const double radiusSq = cylinderRadius * cylinderRadius;
    const double halfLength = 0.5 * cylinderLength;
    const double unitX = axisX / axisNorm;
    const double unitY = axisY / axisNorm;
    const double unitZ = axisZ / axisNorm;

    const int rows = frame[0].rows;
    const int cols = frame[0].cols;
    const int slices = static_cast<int>(frame.size());

    const double axisExtentX = std::abs(unitX) * halfLength;
    const double axisExtentY = std::abs(unitY) * halfLength;
    const double axisExtentZ = std::abs(unitZ) * halfLength;
    const int minX = std::max(0, static_cast<int>(std::floor(center.x - axisExtentX - cylinderRadius)));
    const int maxX = std::min(cols - 1, static_cast<int>(std::ceil(center.x + axisExtentX + cylinderRadius)));
    const int minY = std::max(0, static_cast<int>(std::floor(center.y - axisExtentY - cylinderRadius)));
    const int maxY = std::min(rows - 1, static_cast<int>(std::ceil(center.y + axisExtentY + cylinderRadius)));
    const int minZ = std::max(0, static_cast<int>(std::floor(center.z - axisExtentZ - cylinderRadius)));
    const int maxZ = std::min(slices - 1, static_cast<int>(std::ceil(center.z + axisExtentZ + cylinderRadius)));

    double brightnessSum = 0.0;
    double sampleCount = 0.0;
    for (int z = minZ; z <= maxZ; ++z) {
        for (int y = minY; y <= maxY; ++y) {
            const float *row = frame[z].ptr<float>(y);
            for (int x = minX; x <= maxX; ++x) {
                const double relX = static_cast<double>(x) - center.x;
                const double relY = static_cast<double>(y) - center.y;
                const double relZ = static_cast<double>(z) - center.z;
                const double axialDistance = relX * unitX + relY * unitY + relZ * unitZ;
                if (std::abs(axialDistance) > halfLength) {
                    continue;
                }
                const double radialX = relX - axialDistance * unitX;
                const double radialY = relY - axialDistance * unitY;
                const double radialZ = relZ - axialDistance * unitZ;
                const double distX = radialX;
                const double distY = radialY;
                const double distZ = radialZ;
                const double distSq = distX * distX + distY * distY + distZ * distZ;
                if (distSq <= radiusSq) {
                    brightnessSum += row[x];
                    sampleCount += 1.0;
                }
            }
        }
    }

    return (sampleCount > 0.0) ? (brightnessSum / sampleCount) : 0.0;
}

double computeBridgeCylinderRadius(const Spheroid &cell)
{
    const double major = static_cast<double>(cell.getMajorRadius());
    const double minor = static_cast<double>(cell.getMinorRadius());
    if (major <= 0.0 || minor <= 0.0) {
        return 0.0;
    }

    // Use a cell-shaped effective cross-section radius; length then controls the probe volume.
    return std::sqrt(major * minor);
}
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
    _currentCost = calculateCost(_synthFrame);
}

std::vector<cv::Mat> Frame::generateSynthFrame()
{
    cv::Size shape = getImageShape();
    std::vector<cv::Mat> frame;

    for (double z : z_slices)
    {
        Image synthImage = cv::Mat(shape, CV_32F, cv::Scalar(simulationConfig.background_color));
        for (const auto &cell : cells)
        {
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

    double totalCost = 0.0;
    for (size_t i = 0; i < _realFrame.size(); ++i)
    {
        totalCost += cv::norm(_realFrame[i], synthFrame[i], cv::NORM_L2);
    }
    return totalCost;
}

std::vector<cv::Mat> Frame::generateSynthFrameFast(Spheroid &oldCell, Spheroid &newCell)
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

        cv::Mat synthImage = cv::Mat(shape, CV_32F, cv::Scalar(simulationConfig.background_color));

        for (const auto &cell : cells)
        {
            cell.draw(synthImage, simulationConfig, z);
        }

        synthFrame.push_back(synthImage);
    }

    return synthFrame;
}

std::vector<cv::Mat> Frame::generateOutputFrame()
{
    std::vector<cv::Mat> realFrameWithOutlines;

    for (size_t i = 0; i < _realFrame.size(); ++i)
    {
        const cv::Mat &realImage = _realFrame[i];
        double z = z_slices[i];

        // Convert grayscale to RGB
        cv::Mat outputFrame;
        cv::cvtColor(realImage, outputFrame, cv::COLOR_GRAY2BGR);

        // Draw outlines for each cell
        for (const auto &cell : cells)
        {
            cell.drawOutline(outputFrame, 1.0, z); // Assuming drawOutline takes a cv::Scalar for color
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

CostCallbackPair Frame::perturbCell(size_t index, float overlapWeight, float sizeReductionWeight)
{
    if (index >= cells.size()) {
        return {0.0, [](bool) {}};
    }

    Spheroid oldCell = cells[index];

    // O(n) overlap for just this cell before perturbation
    double oldOverlapCell = computeOverlapForCell(index, overlapWeight);

    cells[index] = cells[index].getPerturbedCell();

    // O(n) overlap for this cell after perturbation
    double newOverlapCell = computeOverlapForCell(index, overlapWeight);
    double sizeReductionPenalty = computeSizeReductionPenalty(oldCell, cells[index], sizeReductionWeight);

    auto newSynthFrame = generateSynthFrameFast(oldCell, cells[index]);
    double newImageCost = calculateCost(newSynthFrame);
    // Use cached cost instead of recalculating L2 over all 225 slices
    double oldImageCost = _currentCost;

    double costDiff = (newImageCost + newOverlapCell + sizeReductionPenalty)
                    - (oldImageCost + oldOverlapCell);

    CallBackFunc callback = [this, newSynthFrame, oldCell, index, newImageCost](bool accept)
    {
        if (accept) {
            this->_synthFrame = newSynthFrame;
            this->_currentCost = newImageCost;
        } else {
            this->cells[index] = oldCell;
        }
    };
    return {costDiff, callback};
}

double Frame::computeOverlapPenalty(float weight) const
{
    double totalPenalty = 0.0;
    for (size_t i = 0; i < cells.size(); ++i) {
        float ri = cells[i].getMajorRadius();
        float xi = cells[i].getX();
        float yi = cells[i].getY();
        float zi = cells[i].getZ();
        for (size_t j = i + 1; j < cells.size(); ++j) {
            float rj = cells[j].getMajorRadius();
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
    float ri = cells[cellIdx].getMajorRadius();
    float xi = cells[cellIdx].getX();
    float yi = cells[cellIdx].getY();
    float zi = cells[cellIdx].getZ();
    for (size_t j = 0; j < cells.size(); ++j) {
        if (j == cellIdx) continue;
        float rj = cells[j].getMajorRadius();
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

std::map<std::string, float> Frame::computeElongationRatios() const
{
    std::map<std::string, float> ratios;
    for (size_t i = 0; i < cells.size(); ++i) {
        // Collect neighbor centers (exclude self)
        std::vector<cv::Point3f> neighbors;
        for (size_t j = 0; j < cells.size(); ++j) {
            if (j != i) neighbors.push_back(cells[j].get_center());
        }
        // getSplitCells returns (d1, d2, valid, elongationRatio)
        auto [d1, d2, valid, elongation] = cells[i].getSplitCells(
            _realFrame, simulationConfig.z_scaling, simulationConfig.background_color, neighbors);
        ratios[cells[i].getName()] = valid ? elongation : 1.0f;
    }
    return ratios;
}

float Frame::computeElongationForCell(size_t cellIdx) const
{
    if (cellIdx >= cells.size()) return 1.0f;

    std::vector<cv::Point3f> neighbors;
    for (size_t j = 0; j < cells.size(); ++j) {
        if (j != cellIdx) neighbors.push_back(cells[j].get_center());
    }

    auto [d1, d2, valid, elongation] = cells[cellIdx].getSplitCells(
        _realFrame, simulationConfig.z_scaling, simulationConfig.background_color, neighbors);

    return valid ? elongation : 1.0f;
}

CostCallbackPair Frame::trySplitCell(size_t index, float preOptMajorR, float preOptMinorR,
                                     float preOptX, float preOptY, float preOptZ,
                                     float splitElongationThreshold,
                                     float overlapWeight,
                                     float fakeSplitOverlapVolumeFractionThreshold,
                                     float fakeSplitRadiusRatioThreshold,
                                     float splitMinorAxisAlignmentToleranceDegrees,
                                     float splitMinorAxisAlignmentFlatnessRatioThreshold,
                                     float splitFakeBridgeBrightnessSimilarityThreshold)
{
    if (index >= cells.size()) {
        return {0.0, [](bool accept) {}};
    }

    Spheroid oldCell = cells[index];

    // Collect neighbor cell centers (all cells except the one being split)
    std::vector<cv::Point3f> neighborCenters;
    for (size_t i = 0; i < cells.size(); ++i) {
        if (i != index) {
            neighborCenters.push_back(cells[i].get_center());
        }
    }

    auto [child1, child2, valid, elongationRatio] = oldCell.getSplitCells(_realFrame, simulationConfig.z_scaling,
                                                                          simulationConfig.background_color,
                                                                          neighborCenters, preOptMajorR, preOptMinorR,
                                                                          preOptX, preOptY, preOptZ,
                                                                          splitMinorAxisAlignmentToleranceDegrees,
                                                                          splitMinorAxisAlignmentFlatnessRatioThreshold);
    if (!valid)
    {
        std::cout << "[Split Skip] " << oldCell.getName()
                  << " getSplitCells returned invalid" << std::endl;
        return {0.0, [](bool accept) {}};
    }

    if (splitElongationThreshold > 0.0f && elongationRatio < splitElongationThreshold) {
        std::cout << "[Split Skip] " << oldCell.getName()
                  << " elongation_ratio=" << elongationRatio
                  << " < threshold=" << splitElongationThreshold << std::endl;
        return {0.0, [](bool accept) {}};
    }

    cells.erase(cells.begin() + index);
    cells.push_back(child1);
    cells.push_back(child2);

    size_t d1Idx = cells.size() - 2;
    size_t d2Idx = cells.size() - 1;

    // No hard overlap rejection — overlap penalty in cost handles it.
    // Daughters near other cells get penalized, not blocked.

    // --- Post-split burn-in: optimize daughters using overlap penalty ---
    // overlapWeight is passed from config via parameter (same as perturbCell)
    auto bestSynthFrame = generateSynthFrame();
    double bestImageCost = calculateCost(bestSynthFrame);
    double bestOverlap = computeOverlapPenalty(overlapWeight);
    double bestTotalCost = bestImageCost + bestOverlap;

    // Recalculate old overlap with parent still present for fair comparison
    // (parent was already removed, so we need to re-add temporarily)
    cells.pop_back(); cells.pop_back();
    cells.insert(cells.begin() + index, oldCell);
    double oldOverlap = computeOverlapPenalty(overlapWeight);
    double oldTotalCost = _currentCost + oldOverlap;
    cells.erase(cells.begin() + index);
    cells.push_back(child1); cells.push_back(child2);
    d1Idx = cells.size() - 2;
    d2Idx = cells.size() - 1;

    auto savedSynthFrame = _synthFrame;
    _synthFrame = bestSynthFrame;

    const int BURN_IN_ITERATIONS = 500;
    int accepted = 0;
    for (int iter = 0; iter < BURN_IN_ITERATIONS; ++iter) {
        size_t dIdx = (iter % 2 == 0) ? d1Idx : d2Idx;

        Spheroid saved = cells[dIdx];

        // Measure overlap for this cell before perturbation
        double oldCellOverlap = computeOverlapForCell(dIdx, overlapWeight);

        // Perturb the daughter
        cells[dIdx] = cells[dIdx].getPerturbedCell();

        // Measure overlap after perturbation
        double newCellOverlap = computeOverlapForCell(dIdx, overlapWeight);

        // Render and compute image cost
        auto trialFrame = generateSynthFrameFast(saved, cells[dIdx]);
        double trialImageCost = calculateCost(trialFrame);

        // Total cost comparison: image cost + overlap delta
        double improvement = (trialImageCost + newCellOverlap) - (bestImageCost + oldCellOverlap);

        if (improvement < 0) {
            bestSynthFrame = trialFrame;
            bestImageCost = trialImageCost;
            _synthFrame = trialFrame;
            accepted++;
        } else {
            cells[dIdx] = saved;
        }
    }

    _synthFrame = savedSynthFrame;

    const double daughterOverlapFraction =
        computeOverlapVolumeFractionApprox(cells[d1Idx], cells[d2Idx]);
    const double daughterRadiusRatio =
        computeMaxDaughterRadiusRatio(cells[d1Idx], cells[d2Idx]);
    const double daughter1MeanBrightness = cells[d1Idx].measureMeanBrightness(_realFrame);
    const double daughter2MeanBrightness = cells[d2Idx].measureMeanBrightness(_realFrame);
    const cv::Point3f daughterCenter1 = cells[d1Idx].get_center();
    const cv::Point3f daughterCenter2 = cells[d2Idx].get_center();
    const cv::Point3f bridgeCenter(
        0.5f * (daughterCenter1.x + daughterCenter2.x),
        0.5f * (daughterCenter1.y + daughterCenter2.y),
        0.5f * (daughterCenter1.z + daughterCenter2.z));
    const cv::Point3f bridgeAxis(
        daughterCenter2.x - daughterCenter1.x,
        daughterCenter2.y - daughterCenter1.y,
        daughterCenter2.z - daughterCenter1.z);
    const double daughter1BridgeVolume = 0.5 * computeSpheroidVolume(cells[d1Idx]);
    const double daughter2BridgeVolume = 0.5 * computeSpheroidVolume(cells[d2Idx]);
    const double daughter1BridgeRadius = computeBridgeCylinderRadius(cells[d1Idx]);
    const double daughter2BridgeRadius = computeBridgeCylinderRadius(cells[d2Idx]);
    const double daughter1BridgeLength =
        (daughter1BridgeRadius > 1e-6)
            ? (daughter1BridgeVolume / (M_PI * daughter1BridgeRadius * daughter1BridgeRadius))
            : 0.0;
    const double daughter2BridgeLength =
        (daughter2BridgeRadius > 1e-6)
            ? (daughter2BridgeVolume / (M_PI * daughter2BridgeRadius * daughter2BridgeRadius))
            : 0.0;
    const double bridgeBrightness1 = computeCylinderMeanBrightnessAlongSegment(
        _realFrame, bridgeCenter, bridgeAxis, daughter1BridgeLength, daughter1BridgeRadius);
    const double bridgeBrightness2 = computeCylinderMeanBrightnessAlongSegment(
        _realFrame, bridgeCenter, bridgeAxis, daughter2BridgeLength, daughter2BridgeRadius);
    const double bridgeSimilarity1 = (daughter1MeanBrightness > 1e-6)
        ? (bridgeBrightness1 / daughter1MeanBrightness)
        : 0.0;
    const double bridgeSimilarity2 = (daughter2MeanBrightness > 1e-6)
        ? (bridgeBrightness2 / daughter2MeanBrightness)
        : 0.0;
    const double averageBridgeSimilarity = 0.5 * (bridgeSimilarity1 + bridgeSimilarity2);
    const bool bridgeLooksContinuous =
        averageBridgeSimilarity >= splitFakeBridgeBrightnessSimilarityThreshold;
    if (daughterOverlapFraction > fakeSplitOverlapVolumeFractionThreshold ||
        daughterRadiusRatio > fakeSplitRadiusRatioThreshold ||
        bridgeLooksContinuous) {
        cells.pop_back();
        cells.pop_back();
        cells.insert(cells.begin() + index, oldCell);

        std::cout << "[Split Rejected Fake] " << oldCell.getName()
                  << " daughter_overlap_fraction=" << daughterOverlapFraction
                  << " threshold=" << fakeSplitOverlapVolumeFractionThreshold
                  << " daughter_radius_ratio=" << daughterRadiusRatio
                  << " ratio_threshold=" << fakeSplitRadiusRatioThreshold
                  << " bridge_similarity=(" << bridgeSimilarity1 << "," << bridgeSimilarity2 << ")"
                  << " bridge_similarity_avg=" << averageBridgeSimilarity
                  << " bridge_threshold=" << splitFakeBridgeBrightnessSimilarityThreshold
                  << std::endl;
        return {0.0, [](bool accept) {}};
    }

    // Recompute final total cost after burn-in
    bestOverlap = computeOverlapPenalty(overlapWeight);
    bestTotalCost = bestImageCost + bestOverlap;
    double costDiff = bestTotalCost - oldTotalCost;

    std::cout << "[Split Burn-in] " << oldCell.getName()
              << " burn_in_accepted=" << accepted << "/" << BURN_IN_ITERATIONS
              << " oldCost=" << oldTotalCost << " newCost=" << bestTotalCost
              << " diff=" << costDiff << std::endl;

    CallBackFunc callback = [this, bestSynthFrame, bestImageCost, oldCell, index](bool accept)
    {
        if (accept)
        {
            this->_synthFrame = bestSynthFrame;
            this->_currentCost = bestImageCost;
        }
        else
        {
            this->cells.pop_back();
            this->cells.pop_back();
            this->cells.insert(this->cells.begin() + index, oldCell);
        }
    };

    return {costDiff, callback};
}

std::vector<cv::Mat> Frame::getSynthFrame()
{
    return _synthFrame;
}
