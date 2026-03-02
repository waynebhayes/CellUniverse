#include <gtest/gtest.h>

#include "Frame.hpp"

#include <stdexcept>
#include <vector>

namespace {
SimulationConfig MakeFrameConfig(int zSlices, float background) {
    SimulationConfig cfg;
    cfg.z_slices = zSlices;
    cfg.background_color = background;
    cfg.cell_color = 1.0f;
    cfg.padding = 0;
    cfg.z_scaling = 1.0f;
    return cfg;
}
}

TEST(FrameTest, CalculateCostReturnsZeroForIdenticalStacks) {
    const SimulationConfig cfg = MakeFrameConfig(1, 0.0f);
    std::vector<cv::Mat> real = {cv::Mat::zeros(2, 2, CV_32F)};
    Frame frame(real, cfg, {}, "", "f0");

    EXPECT_DOUBLE_EQ(frame.calculateCost(real), 0.0);
}

TEST(FrameTest, CalculateCostSumsL2NormAcrossSlices) {
    const SimulationConfig cfg = MakeFrameConfig(2, 0.0f);
    std::vector<cv::Mat> real = {
        cv::Mat::zeros(2, 2, CV_32F),
        cv::Mat::zeros(2, 2, CV_32F)
    };
    Frame frame(real, cfg, {}, "", "f1");

    std::vector<cv::Mat> synth = {
        cv::Mat::ones(2, 2, CV_32F),
        cv::Mat(2, 2, CV_32F, cv::Scalar(2.0f))
    };

    EXPECT_NEAR(frame.calculateCost(synth), 6.0, 1e-9);
}

TEST(FrameTest, CalculateCostThrowsOnMismatchedStackSize) {
    const SimulationConfig cfg = MakeFrameConfig(2, 0.0f);
    std::vector<cv::Mat> real = {
        cv::Mat::zeros(2, 2, CV_32F),
        cv::Mat::zeros(2, 2, CV_32F)
    };
    Frame frame(real, cfg, {}, "", "f2");

    std::vector<cv::Mat> synth = {cv::Mat::zeros(2, 2, CV_32F)};
    EXPECT_THROW(frame.calculateCost(synth), std::runtime_error);
}

TEST(FrameTest, GenerateOutputSynthFrameConvertsFloatTo8Bit) {
    const SimulationConfig cfg = MakeFrameConfig(1, 0.5f);
    std::vector<cv::Mat> real = {cv::Mat::zeros(3, 3, CV_32F)};
    Frame frame(real, cfg, {}, "", "f3");

    std::vector<cv::Mat> out = frame.generateOutputSynthFrame();

    ASSERT_EQ(out.size(), 1U);
    EXPECT_EQ(out[0].type(), CV_8U);
    EXPECT_NEAR(static_cast<double>(out[0].at<unsigned char>(0, 0)), 128.0, 1.0);
}
