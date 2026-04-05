#include "../includes/CellUniverse.hpp"
#include <set>
#include <cmath>

namespace utils
{
    template <typename T>
    void printMat(const cv::Mat &mat)
    {
        // Check if the matrix is empty
        if (mat.empty())
        {
            std::cout << "The matrix is empty." << '\n';
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
            std::cout << '\n'; // Newline for each row
        }
    }
}

// Apply sigmoid contrast: output = 1 / (1 + exp(-k * (input - center)))
// Makes cells near-white and background near-black, giving the L2 cost
// function clear gradient signal for cell boundaries.
static void applySigmoid(cv::Mat &image, float k, float center)
{
    image.forEach<float>([k, center](float &pixel, const int *) {
        pixel = 1.0f / (1.0f + std::exp(-k * (pixel - center)));
    });
}

// Blur only across valid image content so zero-valued border padding does not
// bleed into the specimen near the edges.
static void applySafeGaussianBlur(cv::Mat &image, double sigma)
{
    CV_Assert(image.type() == CV_32F);

    cv::Mat validMask;
    cv::compare(image, 0.0f, validMask, cv::CMP_GT);

    if (cv::countNonZero(validMask) == 0) {
        return;
    }

    cv::Mat validMaskFloat;
    validMask.convertTo(validMaskFloat, CV_32F, 1.0 / 255.0);

    cv::Mat weightedImage = image.mul(validMaskFloat);
    cv::Mat blurredWeighted;
    cv::Mat blurredWeights;

    cv::GaussianBlur(weightedImage, blurredWeighted, cv::Size(0, 0), sigma, sigma, cv::BORDER_CONSTANT);
    cv::GaussianBlur(validMaskFloat, blurredWeights, cv::Size(0, 0), sigma, sigma, cv::BORDER_CONSTANT);

    image.setTo(0.0f);
    cv::divide(blurredWeighted, blurredWeights, image, 1.0, CV_32F);
    image.setTo(0.0f, blurredWeights <= 1e-6f);
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

    // Gaussian blur for noise reduction
    if (config.simulation.blur_sigma > 0) {
        applySafeGaussianBlur(processedImage, config.simulation.blur_sigma);
    }

    return processedImage;
}

std::vector<cv::Mat> loadFrame(const std::string &imageFile, BaseConfig &config)
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
        if (numTiffSlices == 0) {
            throw std::runtime_error("TIFF has 0 slices: " + imageFile);
        }

        cv::Mat img = tiffImage[0];

        if (img.empty())
        {
            std::cout << "Error: Could not read the TIFF image" << '\n';
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

        // --- Calibrate sigmoid center from background zone, then apply sigmoid ---
        // Measure mean brightness in a cell-free zone to determine sigmoid center.
        // Sigmoid: output = 1 / (1 + exp(-k * (input - center)))
        // This amplifies contrast: cells → near 1.0, background → near 0.0.
        // The L2 cost function needs this contrast to correctly fit cell boundaries.
        float sigmoidCenter = config.simulation.sigmoid_center; // default from config
        if (!processedZSlices.empty()) {
            int calZ = std::clamp(config.simulation.calibration_z, 0, static_cast<int>(numTiffSlices) - 1);
            int calX = config.simulation.calibration_x;
            int calY = config.simulation.calibration_y;
            int calW = std::min(config.simulation.calibration_width,
                               processedZSlices[calZ].cols - calX);
            int calH = std::min(config.simulation.calibration_height,
                               processedZSlices[calZ].rows - calY);

            if (calW > 0 && calH > 0) {
                cv::Rect roi(calX, calY, calW, calH);
                float bgMean = static_cast<float>(cv::mean(processedZSlices[calZ](roi))[0]);
                sigmoidCenter = bgMean + config.simulation.sigmoid_center_offset;
                std::cout << "[Calibration] bgMean=" << bgMean
                          << " sigmoidCenter=" << sigmoidCenter
                          << " sigmoid_k=" << config.simulation.sigmoid_k << std::endl;
            }
        }

        // Apply sigmoid to all slices — amplifies cell/background contrast
        if (config.simulation.sigmoid_k > 0) {
            for (auto &slice : processedZSlices) {
                applySigmoid(slice, config.simulation.sigmoid_k, sigmoidCenter);
            }
        }

        const int expandFactor = config.simulation.z_scaling;
        // there will be (expandFactor-1) interpolated slices between each "real" one.
        // we need one extra at the very top to hold the top "real" z-Slice.

        unsigned numSynthSlices = expandFactor * (numTiffSlices-1) + 1; // 225 for 33 slices

        // iterate through synthslices and interpolate between each "real" slice
        for (int synthSlice = 0; synthSlice < numSynthSlices; ++synthSlice) {
            int tiffSlice = int(synthSlice / expandFactor); // "real" slice index
            if (synthSlice % expandFactor == 0) {
                interpolatedZSlices.push_back(processedZSlices[tiffSlice]);
            } else if (synthSlice % expandFactor == 1) {
                interpolateSlices(processedZSlices[tiffSlice],
                                  processedZSlices[tiffSlice + 1],
                                  interpolatedZSlices,
                                  expandFactor - 1);
            }
        }

        // [PATCH] Validate synthetic slice count (only for TIFF branch)
        if (interpolatedZSlices.size() != numSynthSlices) {
            std::string errorMessage =
                "interpolatedZSlices must have exactly " + std::to_string(numSynthSlices) +
                " slices, but has " + std::to_string(interpolatedZSlices.size()) + " slices";
            throw std::runtime_error(errorMessage);
        }
    }
    else
    {
        // TODO: fix this
        cv::Mat img = cv::imread(imageFile);
        if (img.empty())
        {
            std::cout << "Error: Could not read the image" << '\n';
            return processedZSlices;
        }

        if (img.channels() == 3)
        {
            cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        }

        processedZSlices.push_back(processImage(img, config));
    }

    std::cout << std::to_string(interpolatedZSlices.size()) << "slices built successfully" << std::endl;
    return interpolatedZSlices;
}


CellUniverse::CellUniverse(std::map<std::string, std::vector<Spheroid>> initialCells,
                           PathVec imagePaths,
                           BaseConfig &config,
                           std::string outputPath,
                           int firstFrame,
                           int continueFrom)
: config(config), outputPath(outputPath), firstFrame(firstFrame)
{
    for (size_t i = 0; i < imagePaths.size(); ++i)
    {
        std::vector<Image> real_frame;
        real_frame = loadFrame(imagePaths[i], config);
        // loadFrame interpolate frames, update to config is needed
        config.simulation.z_slices = real_frame.size();

        fs::path path(imagePaths[i]);
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
void CellUniverse::optimize(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("Invalid frame index");
    }

    Frame &frame = frames[frameIndex];
    frame.regenerateSynthFrame();

    size_t totalIterations = frame.length() * config.simulation.iterations_per_cell;
    int displayFrame = firstFrame + frameIndex;

    std::cout << "[Optimize] frame " << displayFrame
              << " (" << frame.cells.size() << " cells, " << totalIterations << " iterations)" << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

    const float overlapWeight = config.prob.overlap_penalty_weight;
    const float baseSplitProb = config.prob.split; // base probability (e.g., 0.03)

    // No splits on the first frame — cells can't divide before any time has passed
    bool allowSplits = (frameIndex > 0);

    // Save pre-optimization shapes for each cell before Phase 1 perturbation.
    // Phase 1 can collapse a cell toward one blob of a splitting pair;
    // pre-opt position feeds PCA center and pre-opt radii set daughter sizing.
    struct PreOptShape { float majorR, minorR, x, y, z; };
    std::map<std::string, PreOptShape> preOptShapes;
    for (const auto &cell : frame.cells) {
        auto p = cell.getCellParams();
        preOptShapes[p.name] = {p.majorRadius, p.minorRadius, p.x, p.y, p.z};
    }

    // Cells that already failed a burn-in this frame — skip all further split attempts.
    std::set<std::string> splitBlacklist;

    // P(split) is driven by PREVIOUS frame's PCA elongation (stored in previousElongations).
    // If a cell looked elongated last frame, it's likely dividing now → higher P(split).
    // Frame 1 has no previous data → all cells use base rate.
    // The current frame's PCA (via trySplitCell→getSplitCells) is used for the split AXIS.
    std::cout << "[P(split)] frame " << displayFrame
              << (allowSplits ? " (from previous frame PCA)" : " (splits disabled)") << std::endl;
    for (const auto &cell : frame.cells) {
        auto p = cell.getCellParams();
        float prevElong = 1.0f;
        auto it = previousElongations.find(p.name);
        if (it != previousElongations.end()) {
            prevElong = it->second;
        }
        float ps = allowSplits
            ? std::min(config.prob.max_split_probability, baseSplitProb + std::max(0.0f, 1.0f - 1.0f / prevElong))
            : 0.0f;
        std::cout << "  " << p.name << " prevElong=" << prevElong << " P(split)=" << ps << std::endl;
    }

    double residSum = 0;
    double residCount = 0;
    int splitAccepted = 0;
    int splitAttempted = 0;
    int perturbAccepted = 0;

    for (size_t i = 0; i < totalIterations; ++i) {
        if (frame.cells.empty()) break;

        // Pick random cell
        std::uniform_int_distribution<size_t> cellDist(0, frame.cells.size() - 1);
        size_t cellIdx = cellDist(gen);
        auto params = frame.cells[cellIdx].getCellParams();

        // P(split) driven by PREVIOUS frame's PCA elongation for this cell.
        // Current frame's PCA is used for split axis (inside trySplitCell), not probability.
        float prevElongation = 1.0f;
        {
            auto it = previousElongations.find(params.name);
            if (it != previousElongations.end()) {
                prevElongation = it->second;
            }
        }
        float pSplit = allowSplits
            ? std::min(config.prob.max_split_probability, baseSplitProb + std::max(0.0f, 1.0f - 1.0f / prevElongation))
            : 0.0f;

        if (pSplit > 0.0f && uniform01(gen) < pSplit
            && splitBlacklist.find(params.name) == splitBlacklist.end()) {
            // --- Try split (PCA + 500-iter burn-in via trySplitCell) ---
            splitAttempted++;
            float preOptMajorR = 0.0f, preOptMinorR = 0.0f;
            float preOptX = 0.0f, preOptY = 0.0f, preOptZ = 0.0f;
            auto pit = preOptShapes.find(params.name);
            if (pit != preOptShapes.end()) {
                preOptMajorR = pit->second.majorR;
                preOptMinorR = pit->second.minorR;
                preOptX = pit->second.x;
                preOptY = pit->second.y;
                preOptZ = pit->second.z;
            }
            auto result = frame.trySplitCell(cellIdx, preOptMajorR, preOptMinorR,
                                             preOptX, preOptY, preOptZ,
                                             config.prob.split_elongation_threshold,
                                             overlapWeight);
            double costDiff = result.first;
            auto callback = result.second;

            if (costDiff < -config.prob.split_cost) {
                callback(true);
                splitAccepted++;
                // Remove parent's previous elongation — daughters are new cells
                previousElongations.erase(params.name);
                std::cout << "[Split Accepted] " << params.name << " frame=" << displayFrame
                          << " iter=" << i << " diff=" << costDiff
                          << " threshold=" << -config.prob.split_cost
                          << " prevElong=" << prevElongation
                          << " P(split)=" << pSplit << std::endl;
            } else {
                callback(false);
                // Blacklist this cell from further split attempts this frame.
                // trySplitCell recomputes PCA internally (ignores cache),
                // so cache reset alone doesn't prevent repeated burn-ins.
                splitBlacklist.insert(params.name);
            }
        } else {
            // --- Try perturbation ---
            auto result = frame.perturbCell(cellIdx, overlapWeight);
            double costDiff = result.first;
            auto callback = result.second;

            if (costDiff < 0) {
                callback(true);
                perturbAccepted++;
                residSum += costDiff;
                residCount++;
            } else {
                callback(false);
            }
        }

        // Progress logging
        if (i % 500 == 0 && i > 0) {
            double avgResid = residCount > 0 ? residSum / residCount : 0.0;
            std::cout << "Frame " << displayFrame << " iter=" << i
                      << " perturb_accepted=" << perturbAccepted
                      << " split_attempts=" << splitAttempted
                      << " split_accepted=" << splitAccepted
                      << " avg_resid=" << avgResid
                      << " cells=" << frame.cells.size() << std::endl;
            residSum = 0;
            residCount = 0;
        }
    }

    // End of frame: compute PCA elongation for each cell on THIS frame's image.
    // Store in previousElongations for the NEXT frame's P(split) computation.
    previousElongations.clear();
    std::cout << "[Elongation for next frame]" << std::endl;
    for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
        auto p = frame.cells[ci].getCellParams();
        float elong = frame.computeElongationForCell(ci);
        previousElongations[p.name] = elong;
        std::cout << "  " << p.name << " elongation=" << elong << std::endl;
    }

    std::cout << "[Optimize Done] frame " << displayFrame
              << " perturb_accepted=" << perturbAccepted
              << " split_attempts=" << splitAttempted
              << " split_accepted=" << splitAccepted
              << " final_cells=" << frame.cells.size() << std::endl;
}

void CellUniverse::saveImages(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("Invalid frame index");
    }

    std::vector<Image> realImages = frames[frameIndex].generateOutputFrame();
    std::vector<Image> synthImages = frames[frameIndex].generateOutputSynthFrame();
    int displayFrame = firstFrame + frameIndex;
    std::cout << "Saving images for frame " << displayFrame << "..." << '\n';
    std::cout << "Real Image Type: " << realImages[0].type() << '\n';
    std::cout << "Synth Image Type: " << synthImages[0].type() << '\n';

    std::string realOutputPath = outputPath + "/real/" + std::to_string(displayFrame);
    if (!std::filesystem::exists(realOutputPath))
    {
        std::filesystem::create_directories(realOutputPath);
    }
    for (size_t i = 0; i < realImages.size(); ++i)
    {
        // Save real images
        cv::imwrite(realOutputPath + "/" + std::to_string(i) + ".png", realImages[i]);
    }

    std::string synthOutputPath = outputPath + "/synth/" + std::to_string(displayFrame);
    if (!std::filesystem::exists(synthOutputPath))
    {
        std::filesystem::create_directories(synthOutputPath);
    }
    for (size_t i = 0; i < synthImages.size(); ++i)
    {
        // Save synthetic images
        cv::imwrite(synthOutputPath + "/" + std::to_string(i) + ".png", synthImages[i]);
    }

    std::cout << "Done" << '\n';
}

void CellUniverse::saveCells(int frameIndex)
{
    std::string cellsPath = outputPath + "/cells.csv";
    bool fileExists = std::filesystem::exists(cellsPath);

    // Append mode: each frame adds its rows as it finishes optimizing
    std::ofstream file(cellsPath, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << cellsPath << " for writing" << '\n';
        return;
    }

    // Write header only for the first frame
    if (!fileExists || frameIndex == 0) {
        // Truncate if frame 0 (fresh run)
        if (frameIndex == 0) {
            file.close();
            file.open(cellsPath, std::ios::trunc);
        }
        file << "file,name,x,y,z,majorRadius,minorRadius,theta_x,theta_y,theta_z" << '\n';
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
             << '\n';
    }

    std::cout << "Saved " << frame.cells.size() << " cells for frame " << (firstFrame + frameIndex)
              << " to " << cellsPath << std::endl;
}

void CellUniverse::copyCellsForward(size_t to)
{
    if (to >= frames.size())
    {
        return;
    }
    // assumes cells have deepcopy copy constructors
    frames[to].cells = frames[to - 1].cells;
}

unsigned int CellUniverse::length()
{
    return frames.size();
}

const std::vector<Spheroid> &CellUniverse::getCells(int frameIndex) const
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("CellUniverse::getCells - invalid frameIndex");
    }
    return frames[frameIndex].cells;
}

std::vector<std::string> CellUniverse::getCellNames(int frameIndex) const
{
    const auto &cells = getCells(frameIndex);
    std::vector<std::string> names;
    names.reserve(cells.size());
    for (const auto &c : cells)
    {
        names.push_back(c.getCellParams().name);
    }
    return names;
}
