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

Frame::Frame(const std::vector<cv::Mat> &realFrame, const SimulationConfig &simulationConfig, const std::vector<Sphere> &cells,
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
        double zValue = simulationConfig.z_scaling * (i - simulationConfig.z_slices / 2);
        z_slices.push_back(zValue);
    }
    // TODO: Fix padding
    //    padRealImage();
    _synthFrame = generateSynthFrame();
    //std::cout << " SYNTH FRAME SIZE: " << _synthFrame.size();

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
    for (double z : z_slices)
    {
        Image synthImage = cv::Mat(shape, CV_32F, cv::Scalar(simulationConfig.background_color)); // Assuming background color is in cv::Scalar format
        for (const auto &cell : cells)
        {
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
        //throw std::runtime_error("Mismatch in image stack sizes");
        std::cerr << " real frame size " << _realFrame.size() << " synth frame size " << synthFrame.size();
    }

    double totalCost = 0.0;
    for (size_t i = 0; i < _realFrame.size(); ++i)
    {
        totalCost += cv::norm(_realFrame[i], synthFrame[i], cv::NORM_L2);
    }
    return totalCost;
}

std::vector<cv::Mat> Frame::generateSynthFrameFast(Sphere &oldCell, Sphere &newCell)
{
    if (cells.empty())
    {
        std::cerr << "Cells are not set\n";
    }
#define SLICE_WISE 1 // this shall be zero eventually
#if SLICE_WISE
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
#else
// Using a 3D universe volume to store the synthetic image data
    #define X_SPAN 450
    #define Y_SPAN 550
    #define Z_SPAN 225
    // declare 3D array
    unsigned char UNIVERSE[Z_SPAN][Y_SPAN][Z_SPAN];
    
    // Initialize the 3D array universe with background color
    memset(UNIVERSE, static_cast<unsigned char>(simulationConfig.background_color), sizeof(UNIVERSE));

    // draw each cell into the universe
    for (const auto &cell : cells)
    {
	    cell.draw(UNIVERSE, simulationConfig);
    }

    // extract each z-slice and store as cv::Mat
    std::vector<cv::Mat> synthFrame;
    for (int z = 0; z < Z_SPAN; ++z)
    {
        // using the z-th slice of the universe, create a cv::Mat
        cv::Mat slice(Y_SPAN, X_SPAN, CV_8UC1, UNIVERSE[z]);
        synthFrame.push_back(slice.clone());
    }
    return synthFrame;
#endif
}

/**
 * Generates a vector of output frames with outlines drawn around cells.
 * 
 * This function processes a series of grayscale 3D TIFF image slices (_realFrame) by:
 * 1. Converting each grayscale image to an RGB image.
 * 2. Drawing outlines for detected cells on the respective slice.
 * 3. Ensuring the output frames are converted to 8-bit images if necessary.
 * 
 * @return A std::vector<cv::Mat> containing the processed RGB frames with cell outlines.
 */
std::vector<cv::Mat> Frame::generateOutputFrame()
{
    // Vector to store the processed output frames with outlines
    std::vector<cv::Mat> realFrameWithOutlines;

    // Loop through each slice (image) in the 3D TIFF volume
    for (size_t i = 0; i < _realFrame.size(); ++i)
    {
        const cv::Mat &realImage = _realFrame[i]; // Get the current grayscale image slice
        double z = z_slices[i]; // Get the corresponding z-coordinate for the slice

        // Step 1: Convert the grayscale image to an RGB image
        cv::Mat outputFrame;
        cv::cvtColor(realImage, outputFrame, cv::COLOR_GRAY2BGR);

        // Step 2: Draw outlines for detected cells on the RGB image
        for (const auto &cell : cells)
        {
            // Draw the outline of the cell on the image.
            // Passes the z-coordinate of the current slice for 3D context.
            cell.drawOutline(outputFrame, 0, z);
        }

        // Step 3: Ensure the output frame is an 8-bit image
        // If the image is not already 8-bit, convert it to 8-bit with appropriate scaling
        if (outputFrame.depth() != CV_8U)
        {
            outputFrame.convertTo(outputFrame, CV_8U, 255.0);
        }

        // Add the processed frame to the result vector
        realFrameWithOutlines.push_back(outputFrame);
    }

    // Return the vector of processed frames
    return realFrameWithOutlines;
}

/**
 * Generates a vector of output synthetic frames, ensuring all images are 8-bit.
 * 
 * This function processes synthetic images stored in `_synthFrame` by:
 * 1. Checking each image's depth.
 * 2. Converting images to 8-bit format if necessary, with appropriate scaling.
 * 3. Returning a vector of converted or cloned images.
 * 
 * @return A std::vector<cv::Mat> containing 8-bit synthetic image frames.
 */
std::vector<cv::Mat> Frame::generateOutputSynthFrame()
{
    // Vector to store the processed synthetic frames
    std::vector<cv::Mat> outputSynthFrame;

    // Loop through each synthetic image in `_synthFrame`
    for (const auto &synthImage : _synthFrame)
    {
        cv::Mat outputImage; // Placeholder for the processed image

        // Step 1: Check if the image is already 8-bit
        if (synthImage.depth() != CV_8U)
        {
            // Step 2: Convert to 8-bit if necessary
            // Scale pixel values by 255.0 to map the original range to 8-bit
            synthImage.convertTo(outputImage, CV_8U, 255.0);
        }
        else
        {
            // Step 3: If already 8-bit, clone the image to avoid modifying the original
            outputImage = synthImage.clone();
        }

        // Add the processed image to the output vector
        outputSynthFrame.push_back(outputImage);
    }

    // Return the vector of processed synthetic frames
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
    Sphere oldCell = cells[index];

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
    }
    return {newCost - oldCost, callback};
}

CostCallbackPair Frame::split()
{
    // std::cout << " real " << _realFrame.size() << " synth " << _synthFrame.size();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, cells.size() - 1);

    // Randomly pick an index for a cell
    size_t index = distrib(gen);

    // Store old cell
    Sphere oldCell = cells[index];

    // Emilio + Atif: here is where you want to call a function that creates the bounding 3D rectangle, calls OpenCV PCA,
    // and returns the 3 Eigenvalues + 3rd Eigenvector. Call it "CellPCA". Once that's working, we'll decide what to do
    // with the returned values in terms of deciding (randomly) whether it's worth even attempting a split. If we don't
    // attempt a split, we'll return (I think) the same value as below under if(!valid).

    // Replace the cell at that index with new cells
    Sphere child1;
    Sphere child2;
    bool valid;
    //     std::tie(child1, child2, valid) = oldCell.getSplitCells();
    std::tie(child1, child2, valid) = oldCell.getSplitCells(_realFrame);
    if (!valid)
    {
        return {0.0, [](bool accept) {}};
    }

    cells.erase(cells.begin() + index);
    cells.push_back(child1);

    bool areCellsValid = oldCell.checkIfCellsValid(cells);
    if (!areCellsValid)
    {
        cells.pop_back();
        cells.insert(cells.begin() + index, oldCell);
        return {0.0, [](bool accept) {}};
    }

    auto newSynthFrame = generateSynthFrame();
    double newCost = calculateCost(newSynthFrame);
    double oldCost = calculateCost(_synthFrame);

    CallBackFunc callback = [this, newSynthFrame, oldCell, index](bool accept)
    {
        if (accept)
        {
            this->_synthFrame = newSynthFrame;
        }
        else
        {
            this->cells.pop_back();
            this->cells.pop_back();
            this->cells.insert(this->cells.begin() + index, oldCell);
        }
    };

    return {newCost - oldCost, callback};
}

Cost Frame::costOfPerturb(const std::string &perturbParam, float perturbVal, size_t index, const Cell &oldCell)
{
    std::unordered_map<std::string, float> perturbParams;
    perturbParams[perturbParam] = perturbVal;

    // Perturb cell
    Sphere perturbedCell = cells[index].getParameterizedCell(perturbParams);
    Sphere originalCell = cells[index]; // Store the original cell
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

        Sphere perturbedCell = cells[index].getParameterizedCell(perturbParams);
        Sphere originalCell = cells[index]; // Store the original cell
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
