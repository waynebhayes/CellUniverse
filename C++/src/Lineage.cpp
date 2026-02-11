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
    cv::GaussianBlur(processedImage, processedImage, cv::Size(0, 0), 3);

    return processedImage;
}

std::vector<cv::Mat> loadFrame(const std::string &imageFile, const BaseConfig &config)
{
    std::vector<cv::Mat> processedZSlices;
    std::vector<cv::Mat> interpolatedZSlices;

    size_t numTiffSlices = 0;                 // visible after the if-block
    const int expandFactor = config.simulation.z_scaling;

    std::string extension = imageFile.substr(imageFile.find_last_of('.') + 1);
    if (extension == "tiff" || extension == "tif")
    {
        std::vector<cv::Mat> tiffImage;
        cv::imreadmulti(imageFile, tiffImage, cv::IMREAD_ANYDEPTH | cv::IMREAD_COLOR);

        numTiffSlices = tiffImage.size();
        if (numTiffSlices == 0)
        {
            throw std::runtime_error("TIFF has 0 slices (imreadmulti failed?)");
        }
        std::cout << "[INFO] TIFF slices detected: " << numTiffSlices << std::endl;

        cv::Mat img = tiffImage[0];
        if (img.empty())
        {
            std::cout << "Error: Could not read the TIFF image" << std::endl;
            return processedZSlices;
        }

        for (size_t i = 0; i < numTiffSlices; ++i)
        {
            cv::Mat slice = tiffImage[i].clone();
            cv::cvtColor(slice, slice, cv::COLOR_BGR2GRAY);
            cv::Mat processedImg = processImage(slice, config);
            processedZSlices.push_back(processedImg);
        }

        const size_t numSynthSlices = static_cast<size_t>(expandFactor) * (numTiffSlices - 1) + 1;

        for (size_t synthSlice = 0; synthSlice < numSynthSlices; ++synthSlice)
        {
            size_t tiffSlice = synthSlice / static_cast<size_t>(expandFactor);

            if (synthSlice % static_cast<size_t>(expandFactor) == 0)
            {
                interpolatedZSlices.push_back(processedZSlices[tiffSlice]);
            }
            else if (synthSlice % static_cast<size_t>(expandFactor) == 1)
            {
                interpolateSlices(
                    processedZSlices[tiffSlice],
                    processedZSlices[tiffSlice + 1],
                    interpolatedZSlices,
                    expandFactor - 1);
            }
        }

        if (interpolatedZSlices.empty())
        {
            throw std::runtime_error("interpolatedZSlices is empty (TIFF read failed?)");
        }

        const size_t expected = static_cast<size_t>(expandFactor) * (numTiffSlices - 1) + 1;
        if (interpolatedZSlices.size() != expected)
        {
            throw std::runtime_error(
                "interpolatedZSlices size mismatch: expected " + std::to_string(expected) +
                ", got " + std::to_string(interpolatedZSlices.size()) +
                " (numTiffSlices=" + std::to_string(numTiffSlices) +
                ", z_scaling=" + std::to_string(expandFactor) + ")");
        }

        std::cout << "[INFO] Interpolated Z slices built: " << interpolatedZSlices.size()
                  << " (expected=" << expected
                  << ", numTiffSlices=" << numTiffSlices
                  << ", z_scaling=" << expandFactor << ")\n";

        std::cout << std::to_string(interpolatedZSlices.size()) << " slices built successfully" << std::endl;
        return interpolatedZSlices;
    }
    else
    {
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

        if (processedZSlices.empty())
        {
            throw std::runtime_error("processedZSlices is empty (image read failed?)");
        }

        return processedZSlices;
    }
}

Lineage::Lineage(std::map<std::string, std::vector<Spheroid>> initialCells, PathVec imagePaths, BaseConfig &config, std::string outputPath, int continueFrom)
    : config(config), outputPath(outputPath)
{
    for (size_t i = 0; i < imagePaths.size(); ++i)
    {
        std::vector<Image> real_frame;
        real_frame = loadFrame(imagePaths[i], config);
        // loadFrame interpolates frames, update to config is needed
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
    std::string algorithm = "hill"; // Set default algorithm
    size_t totalIterations = frame.length() * config.simulation.iterations_per_cell;
    std::cout << "Total iterations: " << totalIterations << std::endl;

    double tolerance = 0.5;
    bool minimaReached = false;
   // Cost curCost = 0;
 //   Cost newCost = 0;
    Cost costDiff = 0;
    double residSum = 0;
    double residCount = 0;
    double ovrResidual = 0;

    for (size_t i = 0; i < totalIterations; ++i) {
        if(costDiff<0){
            residSum += costDiff;
            residCount++;
        }
        if (i % 100 == 0) {
            ovrResidual = residSum/residCount;
            if (residCount>0){
            std::cout << "Frame " << frameIndex << ", iteration " << i << " Difference of Residuals " << ovrResidual << std::endl;
            }else{
               std::cout << "Frame " << frameIndex << ", iteration " << i << " -- No synthezised images selected" << std::endl; 
            }
            residSum = 0;
            residCount = 0;
        }
        if (algorithm == "simulated annealing") {
            // Simulated annealing logic
        }
        else if (algorithm == "gradient descent")
        {
            //            std::cout << "Current iteration: " << i + 1 << std::endl;
            //            if (minimaReached) {
            //                continue;
            //            }
            //            curCost = frame.calculateCost(frame.getSynthImageStack());
            //            newCost = frame.gradientDescent();
            //
            //            if ((curCost - newCost) < tolerance) {
            //                minimaReached = true;
            //            }
        }
        else
        {
            std::vector<std::string> options = {"split", "perturbation"};
            std::vector<double> probabilities = {config.prob.split, config.prob.perturbation};

#if USE_MERSENNE
            int chosenIndex = std::discrete_distribution<>(probabilities.begin(), probabilities.end())(get_mt_generator());
#else
            int chosenIndex = std::discrete_distribution<>(probabilities.begin(), probabilities.end())(get_lc_generator());
#endif
            std::string chosenOption = options[chosenIndex];

            CostCallbackPair result;
            if (chosenOption == "perturbation")
            {
                // std::cout << "attempting to perturb..." << std::endl;
                result = frame.perturb();
            }
            else if (chosenOption == "split")
            {
                // std::cout << "attempting to split..." << std::endl;
                result = frame.split();
            }
            else
            {
                throw std::invalid_argument("Invalid option");
            }
            costDiff = result.first;
            std::function<void(bool)> accept = result.second;

            accept(costDiff < 0);
            // Hill climbing logic
        }
    }
}

void Lineage::saveFrame(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("Invalid frame index");
    }

    std::vector<cv::Mat> realFrame = frames[frameIndex].generateOutputFrame();
    std::vector<cv::Mat> synthFrame = frames[frameIndex].generateOutputSynthFrame();
    std::cout << "Saving images for frame " << frameIndex << "..." << std::endl;

    std::string realOutputPath = outputPath + "/real/" + std::to_string(frameIndex);
    if (!std::filesystem::exists(realOutputPath))
    {
        std::filesystem::create_directories(realOutputPath);
    }
    for (size_t i = 0; i < realFrame.size(); ++i)
    {
        // Save real images
        cv::imwrite(realOutputPath + "/" + std::to_string(i) + ".png", realFrame[i]);
    }

    std::string synthOutputPath = outputPath + "/synth/" + std::to_string(frameIndex);
    if (!std::filesystem::exists(synthOutputPath))
    {
        std::filesystem::create_directories(synthOutputPath);
    }
    for (size_t i = 0; i < synthFrame.size(); ++i)
    {
        // Save synthetic images
        cv::imwrite(synthOutputPath + "/" + std::to_string(i) + ".png", synthFrame[i]);
    }

    std::cout << "Done" << std::endl;
}

// void Lineage::saveCells(int frameIndex)
//{
//     std::vector<CellParams> all_cells;
//
//     // Concatenating cell data from each frame
//     for (int i = 0; i <= frameIndex && i < frames.size(); ++i) {
//         auto frame_cells = frames[i].get_cells_as_params();
//         all_cells.insert(all_cells.end(), frame_cells.begin(), frame_cells.end());
//     }
//
//     // Sorting cells by frame and then by cell ID
//     std::sort(all_cells.begin(), all_cells.end(), [](const CellParams& a, const CellParams& b) {
//         return a.file < b.file || (a.file == b.file && a.name < b.name);
//     });
//
//     // Writing to CSV
//     std::ofstream file(outputPath + "cells.csv");
//     if (file.is_open()) {
//         // Assuming you want to write the file and name fields
//         file << "file,name\n";
//         for (const auto& cell : all_cells) {
//             file << cell.file << "," << cell.name << "\n";
//         }
//     }
// }

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
    
