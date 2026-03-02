#include <gtest/gtest.h>

#include "Frame.hpp"

#include <stdexcept>
#include <vector>

TEST(InterpolateSlicesTest, ProducesExpectedLinearIntermediateSlices) {
    cv::Mat slice1 = cv::Mat::zeros(2, 2, CV_32F);
    cv::Mat slice2 = cv::Mat::ones(2, 2, CV_32F);

    std::vector<cv::Mat> out;
    interpolateSlices(slice1, slice2, out, 2);

    ASSERT_EQ(out.size(), 2U);
    EXPECT_NEAR(out[0].at<float>(0, 0), 1.0f / 3.0f, 1e-6f);
    EXPECT_NEAR(out[1].at<float>(0, 0), 2.0f / 3.0f, 1e-6f);
}

TEST(InterpolateSlicesTest, ThrowsWhenSlicesDoNotMatch) {
    cv::Mat slice1 = cv::Mat::zeros(2, 2, CV_32F);
    cv::Mat slice2 = cv::Mat::zeros(3, 2, CV_32F);

    std::vector<cv::Mat> out;
    EXPECT_THROW(interpolateSlices(slice1, slice2, out, 1), std::invalid_argument);
}

TEST(InterpolateSlicesTest, ZeroInterpolationsProducesNoSlices) {
    cv::Mat slice1 = cv::Mat::zeros(2, 2, CV_32F);
    cv::Mat slice2 = cv::Mat::ones(2, 2, CV_32F);

    std::vector<cv::Mat> out;
    interpolateSlices(slice1, slice2, out, 0);

    EXPECT_TRUE(out.empty());
}
