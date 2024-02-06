// Lineage.cpp
#include "../includes/Frame.hpp"
#include "../includes/Lineage.hpp"


Image processImage(const Image& image, const BaseConfig & config)
{
    Image processedImage;

    if (image.channels() == 3) {
        cv::cvtColor(image, processedImage, cv::COLOR_RGB2GRAY);
    }
    else {
        processedImage = image.clone();
    }

    processedImage.convertTo(processedImage, CV_32F, 1.0 / 255.0);

    // Gaussian blur the image
    // use 1.5 temporarily
    cv::GaussianBlur(processedImage, processedImage, cv::Size(0, 0), 1.5);

    return processedImage;
}

std::vector<Image> loadImage(const std::string & imageFile, const BaseConfig & config)
{
    std::vector<Image> imgs;
    // Get the file extension
    std::string extension = imageFile.substr(imageFile.find_last_of('.') + 1);
    if (extension == "tiff" || extension == "tif") {
        Image img = cv::imread(imageFile, cv::IMREAD_ANYDEPTH | cv::IMREAD_COLOR);
        cv::imshow("Tif file", img);
        cv::waitKey(0);
        if (img.empty()) {
            std::cout << "Error: Could not read the TIFF image" << std::endl;
            return imgs;
        }

        int slices = img.size[0]; // Assuming the TIFF contains a stack of images
        std::cout << img.size[0] << " " << img.size[1] << " " << std::endl;

        for (int i = 0; i < slices; ++i) {
            Image slice = img.row(i).clone();
            cv::cvtColor(slice, slice, cv::COLOR_BGR2GRAY);
            imgs.push_back(processImage(slice, config));
        }
    } else {
        Image img = cv::imread(imageFile);
        if (img.empty()) {
            std::cout << "Error: Could not read the image" << std::endl;
            return imgs;
        }

        if (img.channels() == 3) {
            cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        }

        imgs.push_back(processImage(img, config));
    }

    return imgs;
}

Lineage::Lineage(std::map<std::string, std::vector<Cell>> initialCells, std::vector<std::string> imagePaths, BaseConfig config, std::string outputPath, int continueFrom)
    : config(config), outputPath(outputPath)
{
    for (size_t i = 0; i < imagePaths.size(); ++i) {
        std::vector<Image> real_images;
        real_images = loadImage(imagePaths[i], config);

        std::string file_name = imagePaths[i];

        if ((continueFrom == -1 || i < continueFrom) && initialCells.find(file_name) != initialCells.end()) {
            const std::vector<Cell>& cells = initialCells.at(file_name);
//            frames.emplace_back(real_images, config.simulation, cells, outputPath, file_name);
        }
        else {
//            frames.emplace_back(real_images, config.simulation, std::vector<Cell>(), outputPath, file_name);
        }
    }
}
void Lineage::optimize(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size()) {
        throw std::invalid_argument("Invalid frame index");
    }

    Frame& frame = frames[frameIndex];
    std::string algorithm = "hill"; // Set default algorithm
    size_t totalIterations = frame.size() * config.simulation.iterationsPerCell;
    std::cout << "Total iterations: " << totalIterations << std::endl;

    double tolerance = 0.5;
    bool minimaReached = false;

    for (size_t i = 0; i < totalIterations; ++i) {
        if (i % 100 == 0) {
            std::cout << "Frame " << frameIndex << ", iteration " << i << std::endl;
        }

        if (algorithm == "simulated annealing") {
            // Simulated annealing logic
        } else if (algorithm == "gradient descent") {
            std::cout << "Current iteration: " << i + 1 << std::endl;
            if (minimaReached) {
                continue;
            }

            // Gradient descent logic
        } else {
            // Hill climbing logic
        }
    }
}

void Lineage::saveImages(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size()) {
        throw std::invalid_argument("Invalid frame index");
    }

    std::vector<Image> realImages = frames[frameIndex].generateOutputImages();
    std::vector<Image> synthImages = frames[frameIndex].generateOutputSynthImages();
    std::cout << "Saving images for frame " << frameIndex << "..." << std::endl;

    std::string realOutputPath = outputPath + "/real/" + std::to_string(frameIndex);
    if (!std::filesystem::exists(realOutputPath)) {
        std::filesystem::create_directories(realOutputPath);
    }
    for (size_t i = 0; i < realImages.size(); ++i) {
        // Save real images
        cv::imwrite(realOutputPath + "/" + std::to_string(i) + ".png", realImages[i]);
    }

    std::string synthOutputPath = outputPath + "/synth/" + std::to_string(frameIndex);
    if (!std::filesystem::exists(synthOutputPath)) {
        std::filesystem::create_directories(synthOutputPath);
    }
    for (size_t i = 0; i < synthImages.size(); ++i) {
        // Save synthetic images
        cv::imwrite(synthOutputPath + "/" + std::to_string(i) + ".png", synthImages[i]);
    }

    std::cout << "Done" << std::endl;
}

void Lineage::saveCells(int frameIndex)
{

}

void Lineage::copyCellsForward(int to)
{
    if (to >= frames.size()) {
        return;
    }
    // assumes cells have deepcopy copy constructors
//    frames[to].cells = frames[to - 1].cells;
}

unsigned int Lineage::getLength()
{
    return frames.size();
}

