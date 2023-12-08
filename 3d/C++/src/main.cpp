#include <iostream>
#include <fstream>
#include "Lineage.hpp"
#include <direct.h>

void testProcesedImage(const std::string& imagePath)
{
    BaseConfig config;
    cv::Mat inputImage = cv::imread(imagePath);
    if (inputImage.empty())
    {
        std::cout << "Could not open or find the image" << std::endl;
    }
    float blurSigma = 1.5; // Set the blur sigma value
    cv::Mat processedImage = processImage(inputImage, config);

    // Display the processed image
    cv::imshow("Processed Image", processedImage);
    cv::waitKey(0);
}

void testLoadImage(const std::string& imagePath)
{
    BaseConfig config;
    loadImage(imagePath, config);
}

int main(int argc, char** argv )
{
    std::string rootPath = "E:\\CS\\ResearchRepo\\CellUniverse\\3d\\C++";
    std::string imagePath = rootPath + "\\lenna.png";
    std::string tifPath = rootPath + "\\frame000.tif";
    testProcesedImage(imagePath);
    testLoadImage(tifPath);
    return 0;
}