// Lineage.cpp
#include "../includes/Lineage.hpp"

namespace utils
{
    template <typename T>
    void printMat(const cv::Mat &mat)
    {
        // Check if the matrix is empty
        if (mat.empty())
        {
            std::cout << "The matrix is empty." << std::endl;
            return;
        }

        // Iterate over matrix rows
        for (int i = 0; i < mat.rows; ++i)
        {
            for (int j = 0; j < mat.cols; ++j)
            {
                // Print each element. Use mat.at<T>(i,j) to access the element at [i,j] and cast it to the type T.
                // The type T should match the type of the elements in the matrix.
                std::cout << mat.at<T>(i, j) << " ";
            }
            std::cout << std::endl; // Newline for each row
        }
    }
}

Image processImage(const Image &image, const BaseConfig &config)
{
    Image processedImage;

    if (image.channels() == 3)
    {
        cv::cvtColor(image, processedImage, cv::COLOR_RGB2GRAY);
    }
    else
    {
        processedImage = image.clone();
    }

    processedImage.convertTo(processedImage, CV_32F, 1.0 / 255.0);

    // Gaussian blur the image
    // use 1.5 temporarily
    // TODO: use GaussianBlur with custom sigma
    SimulationConfig simConfig = config.simulation;
    //    std::cout << "The blur sigma is: " <<  simConfig.blur_sigma << std::endl;
    cv::GaussianBlur(processedImage, processedImage, cv::Size(0, 0), 1.5);

    return processedImage;
}

std::vector<cv::Mat> loadFrame(const std::string &imageFile, const BaseConfig &config)
{
    std::vector<cv::Mat> processedZSlices; // vector of matrices, each matrix is a 2D image
    std::vector<cv::Mat> interpolatedZSlices;

    // Get the file extension
    std::string extension = imageFile.substr(imageFile.find_last_of('.') + 1);
    if (extension == "tiff" || extension == "tif")
    {
        std::vector<cv::Mat> tiffImage;
        cv::imreadmulti(imageFile, tiffImage, cv::IMREAD_ANYDEPTH | cv::IMREAD_COLOR);
        long unsigned numTiffSlices {tiffImage.size()};
	    assert(numTiffSlices == 33); // FIXME: just for the Pavak's test files
        cv::Mat img = tiffImage[0];

        if (img.empty())
        {
            std::cout << "Error: Could not read the TIFF image" << std::endl;
            return processedZSlices;
        }

        // Iterate through tiffImage, begin coversion to black and white, blurring
        for (unsigned i = 0; i < numTiffSlices; ++i) // should we end at == slices?
        {
            cv::Mat slice = tiffImage[i].clone();
            cv::cvtColor(slice, slice, cv::COLOR_BGR2GRAY);
            cv::Mat processedImg = processImage(slice, config);
            processedZSlices.push_back(processedImg);
        }

        const int expandFactor = config.simulation.z_scaling; 
        // there will be (expandFactor-1) interpolated slices between each "real" one.
        // we need one extra at the very top to hold the top "real" z-Slice.
        unsigned numSynthSlices = expandFactor * (numTiffSlices-1) + 1; // 225 for 33 slices

        // for checking
        // std::cout << "Number of synthetic slices: " << numSynthSlices << std::endl;
        
        // iterate through synthslices and interpolate between each "real" slice
        for (int synthSlice = 0; synthSlice < numSynthSlices; ++synthSlice) {
            int tiffSlice = int(synthSlice / expandFactor); // "real" slice index 
            if(synthSlice % expandFactor == 0) 
            { // copy the real slice to the synth one, verbatim
            interpolatedZSlices.push_back(processedZSlices[tiffSlice]);
            } 
            else if (synthSlice % expandFactor == 1) {
                // Interpolate between realTiff[tiffSlice] and realTiff[tiffSlice + 1]
                interpolateSlices(processedZSlices[tiffSlice], 
                                processedZSlices[tiffSlice + 1], 
                                interpolatedZSlices, 
                                expandFactor - 1);
            }
        }
        // here do one FINAL copy of the very top tiff [number 32] to the very top interp [225, not 224!]
        // interpolatedZSlices.push_back(processedZSlices[32]);
    }
    else
    {
        // TODO: fix this
        cv::Mat img = cv::imread(imageFile);
        if (img.empty())
        {
            std::cout << "Error: Could not read the image" << std::endl;
            return processedZSlices;
        }

        if (img.channels() == 3)
        {
            cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        }

        processedZSlices.push_back(processImage(img, config));
    }
    if (interpolatedZSlices.size() != 225) {
        std::string errorMessage = "interpolatedZSlices must have exactly 255 slices, but has " +
                                std::to_string(interpolatedZSlices.size()) + " slices";
        throw std::runtime_error(errorMessage);
    }
    std::cout << std::to_string(interpolatedZSlices.size()) << "slices built successfully" << std::endl;
    return interpolatedZSlices;
}


Lineage::Lineage(std::map<std::string, std::vector<Spheroid>> initialCells, PathVec imagePaths, BaseConfig &config, std::string outputPath, int continueFrom)
    : config(config), outputPath(outputPath)
{
    for (size_t i = 0; i < imagePaths.size(); ++i)
    {
        std::vector<Image> real_frame;
        real_frame = loadFrame(imagePaths[i], config);
        // loadFrame interpolate frames, update to config is needed
        config.simulation.z_slices = real_frame.size();

        fs::path path(imagePaths[i]);
        //        std::cout << "Filename: " << path.filename() << std::endl;
        std::string file_name = path.filename();

        if ((continueFrom == -1 || i < continueFrom) && initialCells.find(file_name) != initialCells.end())
        {
            const std::vector<Spheroid> &cells = initialCells.at(file_name);
            frames.emplace_back(real_frame, config.simulation, cells, outputPath, file_name);
        }
        else
        {
            frames.emplace_back(real_frame, config.simulation, std::vector<Spheroid>(), outputPath, file_name);
        }
    }
}
void Lineage::optimize(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("Invalid frame index");
    }

    Frame &frame = frames[frameIndex];
    size_t totalIterations = frame.length() * config.simulation.iterations_per_cell;
    std::cout << "Total iterations: " << totalIterations << std::endl;

    Cost costDiff = 0;
    double residSum = 0;
    double residCount = 0;
    double ovrResidual = 0;

    // ============================================================
    // Phase 1: Perturbation-only optimization
    // Settle all existing cells into their best positions first.
    // ============================================================
    std::cout << "[Phase 1] Perturbation optimization for frame " << frameIndex
              << " (" << frame.cells.size() << " cells, " << totalIterations << " iterations)" << std::endl;

    for (size_t i = 0; i < totalIterations; ++i) {
        if (costDiff < 0) {
            residSum += costDiff;
            residCount++;
        }
        if (i % 100 == 0) {
            ovrResidual = residSum / residCount;
            if (residCount > 0) {
                std::cout << "Frame " << frameIndex << ", iteration " << i
                          << " Difference of Residuals " << ovrResidual << std::endl;
            } else {
                std::cout << "Frame " << frameIndex << ", iteration " << i
                          << " -- No synthezised images selected" << std::endl;
            }
            residSum = 0;
            residCount = 0;
        }

        auto result = frame.perturb();
        costDiff = result.first;
        std::function<void(bool)> accept = result.second;
        accept(costDiff < 0);
    }

    // ============================================================
    // Phase 2: Post-optimization split detection
    // Try splitting each original cell exactly once. Accept only if
    // cost improves by more than split_cost. After each accepted
    // split, run extra perturbation iterations so daughters settle.
    // ============================================================
    std::vector<std::string> cellNames;
    for (const auto &cell : frame.cells) {
        cellNames.push_back(cell.getCellParams().name);
    }

    std::cout << "[Phase 2] Split detection for frame " << frameIndex
              << " (" << cellNames.size() << " cells)" << std::endl;

    for (const auto &name : cellNames) {
        // Find current index of this cell by name
        size_t idx = SIZE_MAX;
        for (size_t j = 0; j < frame.cells.size(); ++j) {
            if (frame.cells[j].getCellParams().name == name) {
                idx = j;
                break;
            }
        }
        if (idx == SIZE_MAX) continue;

        auto result = frame.trySplitCell(idx);
        costDiff = result.first;
        std::function<void(bool)> accept = result.second;

        if (costDiff < -config.prob.split_cost) {
            accept(true);
            std::cout << "[Split Accepted] " << name << " split in frame "
                      << frameIndex << " (diff=" << costDiff << ")" << std::endl;

            // Run extra perturbation iterations so daughters can settle
            size_t postSplitIters = 2 * config.simulation.iterations_per_cell;
            for (size_t j = 0; j < postSplitIters; ++j) {
                auto presult = frame.perturb();
                presult.second(presult.first < 0);
            }
        } else {
            accept(false);
        }
    }
}

void Lineage::saveImages(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("Invalid frame index");
    }

    std::vector<Image> realImages = frames[frameIndex].generateOutputFrame();
    std::vector<Image> synthImages = frames[frameIndex].generateOutputSynthFrame();
    std::cout << "Saving images for frame " << frameIndex << "..." << std::endl;
    std::cout << "Real Image Type: " << realImages[frameIndex].type() << std::endl;
    std::cout << "Synth Image Type: " << synthImages[frameIndex].type() << std::endl;

    std::string realOutputPath = outputPath + "/real/" + std::to_string(frameIndex);
    if (!std::filesystem::exists(realOutputPath))
    {
        std::filesystem::create_directories(realOutputPath);
    }
    for (size_t i = 0; i < realImages.size(); ++i)
    {
        // Save real images
        cv::imwrite(realOutputPath + "/" + std::to_string(i) + ".png", realImages[i]);
    }

    std::string synthOutputPath = outputPath + "/synth/" + std::to_string(frameIndex);
    if (!std::filesystem::exists(synthOutputPath))
    {
        std::filesystem::create_directories(synthOutputPath);
    }
    for (size_t i = 0; i < synthImages.size(); ++i)
    {
        // Save synthetic images
        cv::imwrite(synthOutputPath + "/" + std::to_string(i) + ".png", synthImages[i]);
    }

    std::cout << "Done" << std::endl;
}

void Lineage::saveCells(int frameIndex)
{
    std::string cellsPath = outputPath + "/cells.csv";
    bool fileExists = std::filesystem::exists(cellsPath);

    // Append mode: each frame adds its rows as it finishes optimizing
    std::ofstream file(cellsPath, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << cellsPath << " for writing" << std::endl;
        return;
    }

    // Write header only for the first frame
    if (!fileExists || frameIndex == 0) {
        // Truncate if frame 0 (fresh run)
        if (frameIndex == 0) {
            file.close();
            file.open(cellsPath, std::ios::trunc);
        }
        file << "file,name,x,y,z,majorRadius,minorRadius,theta_x,theta_y,theta_z" << std::endl;
    }

    Frame &frame = frames[frameIndex];
    std::string imageName = frame.getImageName();

    for (const auto &cell : frame.cells) {
        SpheroidParams params = cell.getCellParams();
        cell.printCellInfo();
        file << imageName << ","
             << params.name << ","
             << params.x << ","
             << params.y << ","
             << params.z << ","
             << params.majorRadius << ","
             << params.minorRadius << ","
             << params.theta_x << ","
             << params.theta_y << ","
             << params.theta_z
             << std::endl;
    }

    std::cout << "Saved " << frame.cells.size() << " cells for frame " << frameIndex
              << " to " << cellsPath << std::endl;
}

void Lineage::copyCellsForward(int to)
{
    if (to >= frames.size())
    {
        return;
    }
    // assumes cells have deepcopy copy constructors
    frames[to].cells = frames[to - 1].cells;
}

unsigned int Lineage::length()
{
    return frames.size();
}
    