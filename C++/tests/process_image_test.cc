#include <gtest/gtest.h>

#include "ImageHandler.hpp"

namespace {
BaseConfig MakeConfig(float backgroundColor = 0.2f) {
    BaseConfig cfg;
    cfg.simulation.z_scaling = 2.0f;
    cfg.cell = std::make_unique<EllipsoidConfig>();
    cfg.cell->backgroundColor = backgroundColor;
    cfg.cell->minARadius = 4.0;
    cfg.cell->minBRadius = 4.0;
    cfg.cell->minCRadius = 4.0;
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

TEST(PreprocessLoadedFrameTest, CubePoolingUsesMeanForMostlyZeroBackgroundCubes) {
    BaseConfig cfg = MakeConfig();
    cfg.simulation.adaptive_cube_pooling_enabled = true;
    cfg.simulation.adaptive_cube_pooling_cube_size_scale = 0.5f; // min radius=4 -> cube=2
    cfg.simulation.adaptive_cube_pooling_zero_threshold = 0.02f;
    cfg.simulation.adaptive_cube_pooling_majority_threshold = 0.7f;

    std::vector<cv::Mat> slices(2, cv::Mat::zeros(4, 4, CV_32F));
    slices[0].at<float>(0, 0) = 0.8f;

    const std::vector<cv::Mat> out =
        ImageHandler::preprocessLoadedFrame(slices, "synthetic.tif", cfg);

    ASSERT_EQ(out.size(), 3u);
    EXPECT_NEAR(out[0].at<float>(0, 0), 0.1f, 1e-5f);
    EXPECT_NEAR(out[0].at<float>(1, 1), 0.1f, 1e-5f);
}

TEST(PreprocessLoadedFrameTest, CubePoolingUsesMaxWhenCubeContainsSignal) {
    BaseConfig cfg = MakeConfig();
    cfg.simulation.adaptive_cube_pooling_enabled = true;
    cfg.simulation.adaptive_cube_pooling_cube_size_scale = 0.5f; // min radius=4 -> cube=2
    cfg.simulation.adaptive_cube_pooling_zero_threshold = 0.02f;
    cfg.simulation.adaptive_cube_pooling_majority_threshold = 0.7f;

    std::vector<cv::Mat> slices(2, cv::Mat::zeros(4, 4, CV_32F));
    slices[0].at<float>(0, 0) = 0.8f;
    slices[0].at<float>(0, 1) = 0.6f;
    slices[0].at<float>(1, 0) = 0.6f;

    const std::vector<cv::Mat> out =
        ImageHandler::preprocessLoadedFrame(slices, "synthetic.tif", cfg);

    ASSERT_EQ(out.size(), 3u);
    EXPECT_NEAR(out[0].at<float>(0, 0), 0.8f, 1e-5f);
    EXPECT_NEAR(out[0].at<float>(1, 1), 0.8f, 1e-5f);
}

TEST(PreprocessLoadedFrameTest, RemovesIsolatedBrightCubeAfterPooling) {
    BaseConfig cfg = MakeConfig();
    cfg.simulation.adaptive_cube_pooling_enabled = true;
    cfg.simulation.adaptive_cube_pooling_cube_size_scale = 0.5f; // min radius=4 -> cube=2
    cfg.simulation.adaptive_cube_pooling_zero_threshold = 0.02f;
    cfg.simulation.adaptive_cube_pooling_majority_threshold = 0.7f;
    cfg.simulation.adaptive_cube_pooling_remove_isolated_bright_cubes = true;
    cfg.simulation.adaptive_cube_pooling_isolated_bright_cube_threshold = 0.1f;

    std::vector<cv::Mat> slices(2, cv::Mat::zeros(4, 4, CV_32F));
    slices[0].at<float>(0, 0) = 0.8f;
    slices[0].at<float>(0, 1) = 0.6f;
    slices[0].at<float>(1, 0) = 0.6f;

    const std::vector<cv::Mat> out =
        ImageHandler::preprocessLoadedFrame(slices, "synthetic.tif", cfg);

    ASSERT_EQ(out.size(), 3u);
    EXPECT_NEAR(out[0].at<float>(0, 0), 0.0f, 1e-5f);
    EXPECT_NEAR(out[0].at<float>(1, 1), 0.0f, 1e-5f);
}
