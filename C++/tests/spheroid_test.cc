#include <gtest/gtest.h>

#include "Ellipsoid.hpp"

namespace {
void ConfigureEllipsoidBounds() {
    Ellipsoid::cellConfig.minARadius = 1.0;
    Ellipsoid::cellConfig.maxARadius = 10.0;
    Ellipsoid::cellConfig.minCRadius = 0.5;
    Ellipsoid::cellConfig.maxCRadius = 8.0;
}
}

TEST(EllipsoidTest, ConstructorClampsRadiiToConfiguredBounds) {
    ConfigureEllipsoidBounds();

    Ellipsoid spheroid(EllipsoidParams("cellA", 0.0f, 0.0f, 0.0f, 20.0f, 0.1f));
    EllipsoidParams params = spheroid.getCellParams();

    EXPECT_DOUBLE_EQ(params.aRadius, 10.0);
    EXPECT_DOUBLE_EQ(params.cRadius, 0.5);
    EXPECT_TRUE(spheroid.checkConstraints());
}

TEST(EllipsoidTest, ConstructorEnforcesMinorRadiusNotGreaterThanMajor) {
    ConfigureEllipsoidBounds();

    Ellipsoid spheroid(EllipsoidParams("cellB", 0.0f, 0.0f, 0.0f, 2.0f, 5.0f));
    EllipsoidParams params = spheroid.getCellParams();

    EXPECT_DOUBLE_EQ(params.aRadius, 2.0);
    EXPECT_DOUBLE_EQ(params.cRadius, 2.0);
}

TEST(EllipsoidTest, DrawColorsCenterPixelAndLeavesFarPixelUnchanged) {
    ConfigureEllipsoidBounds();

    SimulationConfig simulationConfig;
    simulationConfig.cell_color = 0.9f;
    simulationConfig.background_color = 0.1f;

    cv::Mat image(21, 21, CV_32F, cv::Scalar(simulationConfig.background_color));
    Ellipsoid spheroid(EllipsoidParams("cellC", 10.0f, 10.0f, 0.0f, 3.0f, 3.0f));

    spheroid.draw(image, simulationConfig, 0.0f);

    EXPECT_NEAR(image.at<float>(10, 10), simulationConfig.cell_color, 1e-6f);
    EXPECT_NEAR(image.at<float>(0, 0), simulationConfig.background_color, 1e-6f);
}

