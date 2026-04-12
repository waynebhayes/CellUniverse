#include <gtest/gtest.h>

#include "ImageHandler.hpp"

namespace {
BaseConfig MakeConfig(float backgroundColor = 0.2f) {
    BaseConfig cfg;
    cfg.simulation.background_color = backgroundColor;
    cfg.simulation.z_scaling = 2.0f;
    return cfg;
}
}

TEST(ProcessImageTest, ConvertsRgbInputToSingleChannelFloatImage) {
    BaseConfig cfg = MakeConfig();
    cv::Mat rgb(5, 7, CV_8UC3, cv::Scalar(10, 20, 30));

    cv::Mat out = ImageHandler::processImage(rgb, cfg);

    EXPECT_EQ(out.rows, 5);
    EXPECT_EQ(out.cols, 7);
    EXPECT_EQ(out.channels(), 1);
    EXPECT_EQ(out.type(), CV_32F);
}

TEST(ProcessImageTest, OutputValuesStayWithinUnitRange) {
    BaseConfig cfg = MakeConfig();
    cv::Mat gray = cv::Mat::zeros(7, 7, CV_8U);
    gray.at<unsigned char>(3, 3) = 255;

    cv::Mat out = ImageHandler::processImage(gray, cfg);

    double minVal = 0.0;
    double maxVal = 0.0;
    cv::minMaxLoc(out, &minVal, &maxVal);

    EXPECT_GE(minVal, 0.0);
    EXPECT_LE(maxVal, 1.0);
}

TEST(LoadFrameTest, MissingNonTiffFileReturnsEmptyVector) {
    BaseConfig cfg = MakeConfig();

    std::vector<cv::Mat> out = ImageHandler::loadFrame("/path/that/does/not/exist.png", cfg);

    EXPECT_TRUE(out.empty());
}
