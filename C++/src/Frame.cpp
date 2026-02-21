#include "../includes/Frame.hpp"

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
      _realFrameCopy(realFrame), // Make a copy of the input image stack
      _realFrame(realFrame)
{

    // Calculate z_slices
    for (int i = 0; i < simulationConfig.z_slices; ++i)
    {
        double zValue = i;
        z_slices.push_back(zValue);
    }
    // TODO: Fix padding
    //    padRealImage();
    _synthFrame = generateSynthFrame();
}

void Frame::padRealFrame()
{
    // Return: Generates a new frame from an old one, standarizes all images within one frame so that
    // image sizes do not differ between frames
    int padding = simulationConfig.padding;
    cv::Scalar paddingColor(simulationConfig.background_color);

    cv::Mat paddedFrame;

    // pad the real frame with a border
    cv::copyMakeBorder(_realFrame, paddedFrame, padding, padding, padding, padding, cv::BORDER_CONSTANT, paddingColor);

    // update the _realFrame to the padded frame
    _realFrame = paddedFrame;
}

std::vector<cv::Mat> Frame::generateSynthFrame()
{
    //    if (cells.empty()) {
    //        throw std::runtime_error("Cells are not set");
    //    }

    cv::Size shape = getImageShape(); // Assuming getImageShape returns a cv::Size
    std::vector<cv::Mat> frame;

    unsigned int x = 0;
    // std::cout << "Num of Cells to draw : " << cells.size() << std::endl;
    for (double z : z_slices)
    {
        Image synthImage = cv::Mat(shape, CV_32F, cv::Scalar(simulationConfig.background_color)); // Assuming background color is in cv::Scalar format
        for (const auto &cell : cells)
        {
            cell.printCellInfo();
            // cell.print();
            cell.draw(synthImage, simulationConfig, nullptr, z);
        }
        // if(!frame.empty() && x > 0)
        // {
        //     unsigned num_interpolated_slices = 7;
        //     std::vector<cv::Mat> interSlices{interpolateSlices(frame.front(), synthImage, num_interpolated_slices)};
        //     for (unsigned i = 0; i < num_interpolated_slices; ++i)
        //     {
        //         frame.push_back(interSlices[i]);
        //     }
        // }
        x += 1;
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
            cell.draw(synthImage, simulationConfig, nullptr, z);
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

// size_t Frame::length() const
// {
//     return cells.size();
// }
size_t Frame::length() const
{
    return cells.size();
}

CostCallbackPair Frame::perturb()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, cells.size() - 1);

    // Randomly pick an index for a cell
    size_t index = distrib(gen);

    // Store old cell // no reference
    Spheroid oldCell = cells[index];

    // Replace the cell at that index with a new cell
    cells[index] = cells[index].getPerturbedCell();

    bool areCellsValid = oldCell.checkIfCellsValid(cells);
    if (!areCellsValid)
    {
        cells[index] = oldCell;
        return {0.0, [](bool accept) {}};
    }

    // Synthesize new synthetic image
    auto newSynthFrame = generateSynthFrameFast(oldCell, cells[index]);

    // Get the cost of the new synthetic image
    double newCost = calculateCost(newSynthFrame);

    // If the difference is greater than the threshold, revert to the old cell
    double oldCost = calculateCost(_synthFrame);

    CallBackFunc callback = [this, newSynthFrame, oldCell, index](bool accept)
    {
        if (accept)
        {
            this->_synthFrame = newSynthFrame;
        }
        else
        {
            this->cells[index] = oldCell;
        }
    };
    if (newCost - oldCost < 0){
        std::cout << " New Residual Accepted: " << newCost << std::endl;
    } else {
        // std::cout << "Residual Too High: " << newCost-oldCost << std::endl;
    }
    return {newCost - oldCost, callback};
}

CostCallbackPair Frame::split()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, cells.size() - 1);
    size_t index = distrib(gen);
    return trySplitCell(index);
}

CostCallbackPair Frame::trySplitCell(size_t index)
{
    if (index >= cells.size()) {
        return {0.0, [](bool accept) {}};
    }

    Spheroid oldCell = cells[index];

    Spheroid child1;
    Spheroid child2;
    bool valid;
    std::tie(child1, child2, valid) = oldCell.getSplitCells(_realFrame, simulationConfig.z_scaling);
    if (!valid)
    {
        return {0.0, [](bool accept) {}};
    }

    cells.erase(cells.begin() + index);
    cells.push_back(child1);
    cells.push_back(child2);

    // Check daughters against existing cells only (not each other).
    // Daughters split from a single parent so they naturally overlap each
    // other initially. The cost function and burn-in handle separation.
    size_t d1Idx = cells.size() - 2;
    size_t d2Idx = cells.size() - 1;
    bool overlapWithExisting = false;
    for (size_t i = 0; i < d1Idx; ++i) {
        for (size_t di : {d1Idx, d2Idx}) {
            cv::Point3f diff = cells[i].get_center() - cells[di].get_center();
            float dist = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
            auto pi = cells[i].getCellParams();
            auto pd = cells[di].getCellParams();
            if (dist < (pi.majorRadius + pd.majorRadius) * 0.95f &&
                dist < (pi.minorRadius + pd.minorRadius) * 0.95f) {
                overlapWithExisting = true;
                break;
            }
        }
        if (overlapWithExisting) break;
    }
    if (overlapWithExisting) {
        cells.pop_back();
        cells.pop_back();
        cells.insert(cells.begin() + index, oldCell);
        return {0.0, [](bool accept) {}};
    }

    // --- Post-split burn-in: optimize daughters before cost comparison ---
    auto bestSynthFrame = generateSynthFrame();
    double bestCost = calculateCost(bestSynthFrame);
    double oldCost = calculateCost(_synthFrame);

    auto savedSynthFrame = _synthFrame;
    _synthFrame = bestSynthFrame;

    const int BURN_IN_ITERATIONS = 100;
    int accepted = 0;
    for (int iter = 0; iter < BURN_IN_ITERATIONS; ++iter) {
        size_t dIdx = (iter % 2 == 0) ? d1Idx : d2Idx;

        Spheroid saved = cells[dIdx];
        cells[dIdx] = cells[dIdx].getPerturbedCell();

        // Check perturbed daughter against non-daughter cells only.
        // Daughter-daughter overlap is allowed — cost function handles it.
        bool burnInValid = true;
        for (size_t i = 0; i < d1Idx; ++i) {
            cv::Point3f diff = cells[i].get_center() - cells[dIdx].get_center();
            float dist = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
            auto pi = cells[i].getCellParams();
            auto pd = cells[dIdx].getCellParams();
            if (dist < (pi.majorRadius + pd.majorRadius) * 0.95f &&
                dist < (pi.minorRadius + pd.minorRadius) * 0.95f) {
                burnInValid = false;
                break;
            }
        }
        if (!burnInValid) {
            cells[dIdx] = saved;
            continue;
        }

        auto trialFrame = generateSynthFrameFast(saved, cells[dIdx]);
        double trialCost = calculateCost(trialFrame);

        if (trialCost < bestCost) {
            bestSynthFrame = trialFrame;
            bestCost = trialCost;
            _synthFrame = trialFrame;
            accepted++;
        } else {
            cells[dIdx] = saved;
        }
    }

    _synthFrame = savedSynthFrame;

    std::cout << "[Split Burn-in] " << oldCell.getCellParams().name
              << " burn_in_accepted=" << accepted << "/" << BURN_IN_ITERATIONS
              << " oldCost=" << oldCost << " newCost=" << bestCost
              << " diff=" << (bestCost - oldCost) << std::endl;

    CallBackFunc callback = [this, bestSynthFrame, oldCell, index](bool accept)
    {
        if (accept)
        {
            this->_synthFrame = bestSynthFrame;
        }
        else
        {
            this->cells.pop_back();
            this->cells.pop_back();
            this->cells.insert(this->cells.begin() + index, oldCell);
        }
    };

    return {bestCost - oldCost, callback};
}

Cost Frame::costOfPerturb(const std::string &perturbParam, float perturbVal, size_t index, const Cell &oldCell)
{
    std::unordered_map<std::string, float> perturbParams;
    perturbParams[perturbParam] = perturbVal;

    // Perturb cell
    Spheroid perturbedCell = cells[index].getParameterizedCell(perturbParams);
    Spheroid originalCell = cells[index]; // Store the original cell
    cells[index] = perturbedCell;       // Replace with the perturbed cell

    // Generate new image stack and get new cost
    auto newSynthFrame = generateSynthFrame();
    double newCost = calculateCost(newSynthFrame);

    // Reset cell to its old state
    cells[index] = originalCell;

    return newCost;
}

ParamImageMap
Frame::getSynthPerturbedCells(
    size_t index,
    const ParamValMap &params,
    float perturbLength,
    const Cell &oldCell)
{
    std::unordered_map<std::string, std::vector<cv::Mat>> perturbedCells;

    for (const auto &[param, val] : params)
    {
        if (param == "name")
        {
            continue;
        }

        std::unordered_map<std::string, float> perturbParams;
        perturbParams[param] = perturbLength;

        Spheroid perturbedCell = cells[index].getParameterizedCell(perturbParams);
        Spheroid originalCell = cells[index]; // Store the original cell
        cells[index] = perturbedCell;       // Replace with the perturbed cell

        perturbedCells[param] = generateSynthFrame();

        cells[index] = originalCell; // Reset cell to its old state
    }

    return perturbedCells;
}

// Cost Frame::gradientDescent() {
//     // Hyperparameters
//     float movingDelta = 1.0f;
//     float delta = 1e-3f;
//     float alpha = 0.2f;
//
//     std::vector<Cell*> cellList = cells; // Assuming cells is a std::vector<Cell>
//     double origCost = calculateCost(_synthImageStack);
//
//     std::cout << "Original cost: " << origCost << std::endl;
//
//     std::vector<std::vector<double>> cellsGrad;
//     std::vector<std::string> paramNames;
//
//     // Get gradient for each cell
//     for (size_t index = 0; index < cellList.size(); ++index) {
//         Cell* cell = cellList[index];
//         Cell* oldCell = cell; // Deep copy of the cell
//
//         auto params = cell->getCellParams(); // Assuming getCellParams returns a map or similar structure
//
//         // Get params that are changing
//         if (paramNames.empty()) {
//             for (const auto& [param, _] : params) {
//                 if (param != "name") {
//                     paramNames.push_back(param);
//                 }
//             }
//         }
//
//         std::unordered_map<std::string, std::vector<cv::Mat>> perturbedCells =
//                 getSynthPerturbedCells(index, params, movingDelta, oldCell); // Assuming implementation
//
//         std::vector<double> costs;
//         for (const auto& [_, synthImageStack] : perturbedCells) {
//             costs.push_back(calculateCost(synthImageStack)); // Assuming calculateCost implementation
//         }
//
//         cellsGrad.push_back(costs);
//     }
//
//     // Calculating gradient
//     for (auto& grad : cellsGrad) {
//         std::transform(grad.begin(), grad.end(), grad.begin(),
//                        [origCost, delta](double cost) { return (cost - origCost) / delta; });
//     }
//
//
//     std::unordered_map<size_t, std::unordered_map<std::string, float>> directions;
//
//     // Line search and parameter update
//     for (size_t index = 0; index < cellList.size(); ++index) {
//         const auto& grad = cellsGrad[index];
//         Cell* cell = cellList[index];
//         Cell* oldCell = cell; // Deep copy of the cell
//
//         std::unordered_map<std::string, float> paramGradients;
//         for (size_t i = 0; i < paramNames.size(); ++i) {
//             std::string param = paramNames[i];
//             float gradient = grad[i];
//
//             float tolerance = 1e-2f;
//             float direction = -1.0f * alpha;
//             float lower = gradient * direction;
//             float upper = 3 * lower;
//             double lowerCost = costOfPerturb(param, lower, index, *oldCell);
//             double upperCost = costOfPerturb(param, upper, index, *oldCell);
//             double bestCost = lowerCost;
//
//             // Line search loop
//             while (upperCost < bestCost && std::abs(upperCost - bestCost) > tolerance) {
//                 upper *= 3;
//                 bestCost = upperCost;
//                 upperCost = costOfPerturb(param, upper, index, *oldCell);
//             }
//
//             float mid;
//             // Finding the minimal cost
//             while (true) {
//                 mid = (lower + upper) / 2.0f;
//                 double midCost = costOfPerturb(param, mid, index, *oldCell);
//
//                 if (std::abs(midCost - bestCost) < tolerance) {
//                     break;
//                 }
//
//                 if (midCost < lowerCost) {
//                     lower = mid;
//                     lowerCost = midCost;
//                     bestCost = midCost;
//                 } else if (midCost > lowerCost) {
//                     upper = mid;
//                 }
//             }
//
//             directions[index][param] = mid;
//         }
//     }
//
//// Updating cells based on calculated directions
//    for (size_t index = 0; index < cellList.size(); ++index) {
//        cells[index] = cells[index]->getParameterizedCell(directions[index]);
//    }
//
//    _synthImageStack = generateSynthImages();
//    double newCost = calculateCost(_synthImageStack);
//
//    std::cout << "Current cost: " << newCost << std::endl;
//    return newCost;
//
//}

std::vector<cv::Mat> Frame::getSynthFrame()
{
    return _synthFrame;
}
