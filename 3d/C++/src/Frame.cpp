#include "../includes/Frame.hpp"

Frame::Frame(const ImageStack &realImageStack, const SimulationConfig &simulationConfig, const std::vector<Cell> &cells,
             const Path &outputPath, const std::string &imageName)
    : cells(cells),
      simulationConfig(simulationConfig),
      outputPath(outputPath),
      imageName(imageName),
      _realImageStackCopy(realImageStack), // Make a copy of the input image stack
      _realImageStack(realImageStack) {

    // Calculate z_slices
    for (int i = 0; i < simulationConfig.zSlices; ++i) {
        double zValue = simulationConfig.zScaling * (i - simulationConfig.zSlices / 2);
        zSlices.push_back(zValue);
    }
    padRealImage();
    _synthImageStack = generateSynthImages();
}



void Frame::padRealImage() {
    int padding = simulationConfig.padding;
    cv::Scalar paddingColor(simulationConfig.backgroundColor);

    cv::Mat paddedImage;
    std::vector<cv::Mat> paddedImageStack;

    // Assuming RealImageStack is a vector of cv::Mat
    for (auto& image : _realImageStack) {
        // Pad each image in the stack
        cv::copyMakeBorder(image, paddedImage, padding, padding, padding, padding, cv::BORDER_CONSTANT, paddingColor);
        paddedImageStack.push_back(paddedImage);
    }
    _realImageStack = paddedImageStack;
}

ImageStack Frame::generateSynthImages() {
    if (cells.empty()) {
        throw std::runtime_error("Cells are not set");
    }

    cv::Size shape = getImageShape(); // Assuming getImageShape returns a cv::Size
    ImageStack imageStack;

    for (double z : zSlices) {
        Image synthImage = cv::Mat(shape, CV_8UC3, cv::Scalar(simulationConfig.backgroundColor)); // Assuming background color is in cv::Scalar format

        for (const Cell& cell : cells) {
            cell.draw(synthImage, simulationConfig, z);
        }

        imageStack.push_back(synthImage);
    }
    return imageStack;
}

cv::Size Frame::getImageShape() {
    if (_realImageStack.empty()) {
        throw std::runtime_error("Real image stack is empty");
    }
    return _realImageStack[0].size(); // Returns the size of the first image in the stack
}

Cost Frame::calculateCost(const ImageStack &synthImageStack) {
    if (_realImageStack.size() != synthImageStack.size()) {
        throw std::runtime_error("Mismatch in image stack sizes");
    }

    double totalCost = 0.0;
    for (size_t i = 0; i < _realImageStack.size(); ++i) {
        totalCost += cv::norm(_realImageStack[i], synthImageStack[i], cv::NORM_L2);
    }
    return totalCost;
}

ImageStack Frame::generateSynthImagesFast(Cell &oldCell, Cell &newCell) {
    if (cells.empty()) {
        throw std::runtime_error("Cells are not set");
    }

    cv::Size shape = getImageShape(); // Assuming getImageShape() returns a cv::Size
    ImageStack synthImageStack;

    // Calculate the smallest box that contains both the old and new cell
    cv::Rect minBox = oldCell.calculateMinimumBox(newCell);

    for (size_t i = 0; i < zSlices.size(); ++i) {
        double z = zSlices[i];

        // If the z-slice is outside the min/max box, append the existing synthetic image to the stack
        if (z < minBox.z || z > maxBox.z) {
            synthImageStack.push_back(synthImageStack[i]);
            continue;
        }

        Image synthImage = cv::Mat(shape, CV_8UC3, cv::Scalar(simulationConfig.backgroundColor));

        for (const Cell& cell : cells) {
            cell.draw(synthImage, simulationConfig, z);
        }

        synthImageStack.push_back(synthImage);
    }

    return synthImageStack;
}

ImageStack Frame::generateOutputImage() {
    ImageStack realImagesWithOutlines;

    for (size_t i = 0; i < _realImageStack.size(); ++i) {
        const cv::Mat& realImage = _realImageStack[i];
        double z = zSlices[i];

        // Convert grayscale to RGB
        cv::Mat outputFrame;
        cv::cvtColor(realImage, outputFrame, cv::COLOR_GRAY2BGR);

        // Draw outlines for each cell
        for (const Cell& cell : cells) {
            cell.drawOutline(outputFrame, cv::Scalar(255, 0, 0), z); // Assuming drawOutline takes a cv::Scalar for color
        }

        // Convert to 8-bit image if necessary
        if (outputFrame.depth() != CV_8U) {
            outputFrame.convertTo(outputFrame, CV_8U, 255.0);
        }

        realImagesWithOutlines.push_back(outputFrame);
    }

    return realImagesWithOutlines;
}

ImageStack Frame::generateOutputSynthImage() {
    ImageStack outputSynthImages;

    for (const auto& synthImage : synthImageStack) {
        cv::Mat outputImage;
        if (synthImage.depth() != CV_8U) {
            // Convert to 8-bit image if necessary, scaling pixel values by 255
            synthImage.convertTo(outputImage, CV_8U, 255.0);
        } else {
            outputImage = synthImage.clone();
        }

        outputSynthImages.push_back(outputImage);
    }

    return outputSynthImages;
}

size_t Frame::length() const {
    return cells.size();
}

CostCallbackPair Frame::perturb() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, cells.size() - 1);

    // Randomly pick an index for a cell
    size_t index = distrib(gen);

    // Store old cell
    Cell oldCell = cells[index];

    // Replace the cell at that index with a new cell
    cells[index] = *cells[index].getPerturbedCell();

    bool areCellsValid = oldCell.checkIfCellsValid(cells);
    if (!areCellsValid) {
        cells[index] = oldCell;
        return {0.0, [](bool accept) {}};
    }

    // Synthesize new synthetic image
    auto newSynthImageStack = generateSynthImagesFast(oldCell, cells[index]);

    // Get the cost of the new synthetic image
    double newCost = calculateCost(newSynthImageStack);

    // If the difference is greater than the threshold, revert to the old cell
    double oldCost = calculateCost(_synthImageStack);

    CallBackFunc callback = [this, newSynthImageStack, oldCell, index](bool accept) {
        if (accept) {
            this->_synthImageStack = newSynthImageStack;
        } else {
            this->cells[index] = oldCell;
        }
    };

    return {newCost - oldCost, callback};
}

CostCallbackPair Frame::split() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, cells.size() - 1);

    // Randomly pick an index for a cell
    size_t index = distrib(gen);

    // Store old cell
    Cell oldCell = cells[index];

    // Replace the cell at that index with new cells
    Cell child1, child2;
    bool valid;
    std::tie(child1, child2, valid) = oldCell.getSplitCells();
    if (!valid) {
        return {0.0, [](bool accept) {}};
    }

    cells.erase(cells.begin() + index);
    cells.push_back(child1);
    cells.push_back(child2);

    bool areCellsValid = oldCell.checkIfCellsValid(cells);
    if (!areCellsValid) {
        cells.pop_back();
        cells.pop_back();
        cells.insert(cells.begin() + index, oldCell);
        return {0.0, [](bool accept) {}};
    }

    auto newSynthImageStack = generateSynthImages();
    double newCost = calculateCost(newSynthImageStack);
    double oldCost = calculateCost(_synthImageStack);

    CallBackFunc callback = [this, newSynthImageStack, oldCell, index](bool accept) {
        if (accept) {
            this->_synthImageStack = newSynthImageStack;
        } else {
            this->cells.pop_back();
            this->cells.pop_back();
            this->cells.insert(this->cells.begin() + index, oldCell);
        }
    };

    return {newCost - oldCost, callback};
}

Cost Frame::costOfPerturb(const std::string &perturbParam, float perturbVal, size_t index, const Cell &oldCell) {
    std::unordered_map<std::string, float> perturbParams;
    perturbParams[perturbParam] = perturbVal;

    // Perturb cell
    Cell perturbedCell = cells[index].getParameterizedCell(perturbParams);
    Cell originalCell = cells[index]; // Store the original cell
    cells[index] = perturbedCell; // Replace with the perturbed cell

    // Generate new image stack and get new cost
    auto newSynthImageStack = generateSynthImages();
    double newCost = calculateCost(newSynthImageStack);

    // Reset cell to its old state
    cells[index] = originalCell;

    return newCost;
}

ParamImageMap
Frame::getSynthPerturbedCells(
        size_t index,
        const ParamValMap &params,
        float perturbLength,
        const Cell &oldCell) {
    std::unordered_map<std::string, ImageStack> perturbedCells;

    for (const auto& [param, val] : params) {
        if (param == "name") {
            continue;
        }

        std::unordered_map<std::string, float> perturbParams;
        perturbParams[param] = perturbLength;

        Cell perturbedCell = cells[index].getParameterizedCell(perturbParams);
        Cell originalCell = cells[index]; // Store the original cell
        cells[index] = perturbedCell; // Replace with the perturbed cell

        perturbedCells[param] = generateSynthImages();

        cells[index] = originalCell; // Reset cell to its old state
    }

    return perturbedCells;
}















