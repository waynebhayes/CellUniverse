#include "../includes/Frame.hpp"

namespace {

bool shouldRejectSplitPreBurnIn(const std::string &cellName,
                                const SplitDiagnostics &diag,
                                float splitElongationThreshold)
{
    const float weakElongationCutoff = splitElongationThreshold + 0.20f;
    if (diag.separationOverDaughterMajor < 1.0f) {
        std::cout << "[Split HeuristicReject] " << cellName
                  << " reason=overlapping_daughters"
                  << " elongation_ratio=" << diag.elongationRatio
                  << " sepOverDaughterMajor=" << diag.separationOverDaughterMajor
                  << " driftOverParentMajor=" << diag.driftOverParentMajor
                  << " axisAbsZ=" << diag.axisAbsZ
                  << std::endl;
        return true;
    }

    if (diag.elongationRatio < weakElongationCutoff &&
        diag.separationOverDaughterMajor < 1.10f) {
        std::cout << "[Split HeuristicReject] " << cellName
                  << " reason=weak_geometry"
                  << " elongation_ratio=" << diag.elongationRatio
                  << " weak_cutoff=" << weakElongationCutoff
                  << " sepOverDaughterMajor=" << diag.separationOverDaughterMajor
                  << " driftOverParentMajor=" << diag.driftOverParentMajor
                  << " axisAbsZ=" << diag.axisAbsZ
                  << std::endl;
        return true;
    }

    if (diag.axisAbsZ > 0.92f &&
        diag.separationOverDaughterMajor < 1.30f &&
        diag.driftOverParentMajor > 0.40f) {
        std::cout << "[Split HeuristicReject] " << cellName
                  << " reason=z_axis_internal_structure"
                  << " elongation_ratio=" << diag.elongationRatio
                  << " sepOverDaughterMajor=" << diag.separationOverDaughterMajor
                  << " driftOverParentMajor=" << diag.driftOverParentMajor
                  << " axisAbsZ=" << diag.axisAbsZ
                  << std::endl;
        return true;
    }

    return false;
}

bool shouldRejectSplitPostBurnIn(const std::string &cellName,
                                 const SplitDiagnostics &diag,
                                 double costDiff)
{
    if (diag.driftOverParentMajor > 0.85f && costDiff > -40.0) {
        std::cout << "[Split HeuristicReject] " << cellName
                  << " reason=large_recenter_marginal_gain"
                  << " costDiff=" << costDiff
                  << " driftOverParentMajor=" << diag.driftOverParentMajor
                  << " sepOverDaughterMajor=" << diag.separationOverDaughterMajor
                  << " elongation_ratio=" << diag.elongationRatio
                  << " axisAbsZ=" << diag.axisAbsZ
                  << std::endl;
        return true;
    }

    return false;
}

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

CostCallbackPair Frame::perturbCell(size_t index, float overlapWeight)
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

    auto newSynthFrame = generateSynthFrameFast(oldCell, cells[index]);
    double newImageCost = calculateCost(newSynthFrame);
    // Use cached cost instead of recalculating L2 over all 225 slices
    double oldImageCost = _currentCost;

    double costDiff = (newImageCost + newOverlapCell) - (oldImageCost + oldOverlapCell);

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
        const auto splitResult = cells[i].getSplitCells(
            _realFrame, simulationConfig.z_scaling, neighbors);
        ratios[cells[i].getName()] = std::get<2>(splitResult) ? std::get<3>(splitResult) : 1.0f;
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

    const auto splitResult = cells[cellIdx].getSplitCells(
        _realFrame, simulationConfig.z_scaling, neighbors);

    return std::get<2>(splitResult) ? std::get<3>(splitResult) : 1.0f;
}

CostCallbackPair Frame::trySplitCell(size_t index, float preOptMajorR, float preOptMinorR,
                                     float preOptX, float preOptY, float preOptZ,
                                     float splitElongationThreshold,
                                     float overlapWeight)
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

    auto [child1, child2, valid, elongationRatio, splitDiagnostics] =
        oldCell.getSplitCells(_realFrame, simulationConfig.z_scaling, neighborCenters,
                              preOptMajorR, preOptMinorR, preOptX, preOptY, preOptZ);
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

    if (shouldRejectSplitPreBurnIn(oldCell.getName(), splitDiagnostics, splitElongationThreshold)) {
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

    // Recompute final total cost after burn-in
    bestOverlap = computeOverlapPenalty(overlapWeight);
    bestTotalCost = bestImageCost + bestOverlap;
    double costDiff = bestTotalCost - oldTotalCost;

    std::cout << "[Split Burn-in] " << oldCell.getName()
              << " burn_in_accepted=" << accepted << "/" << BURN_IN_ITERATIONS
              << " oldImageCost=" << _currentCost
              << " oldOverlap=" << oldOverlap
              << " newImageCost=" << bestImageCost
              << " newOverlap=" << bestOverlap
              << " oldCost=" << oldTotalCost
              << " newCost=" << bestTotalCost
              << " diff=" << costDiff << std::endl;

    if (shouldRejectSplitPostBurnIn(oldCell.getName(), splitDiagnostics, costDiff)) {
        _synthFrame = savedSynthFrame;
        cells.pop_back();
        cells.pop_back();
        cells.insert(cells.begin() + index, oldCell);
        return {0.0, [](bool accept) {}};
    }

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
