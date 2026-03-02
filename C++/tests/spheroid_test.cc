#include <gtest/gtest.h>

#include "Spheroid.hpp"

namespace {
void ConfigureSpheroidBounds() {
    Spheroid::cellConfig.minMajorRadius = 1.0;
    Spheroid::cellConfig.maxMajorRadius = 10.0;
    Spheroid::cellConfig.minMinorRadius = 0.5;
    Spheroid::cellConfig.maxMinorRadius = 8.0;
}
}

TEST(SpheroidTest, ConstructorClampsRadiiToConfiguredBounds) {
    ConfigureSpheroidBounds();

    Spheroid spheroid(SpheroidParams("cellA", 0.0f, 0.0f, 0.0f, 20.0f, 0.1f));
    SpheroidParams params = spheroid.getCellParams();

    EXPECT_DOUBLE_EQ(params.majorRadius, 10.0);
    EXPECT_DOUBLE_EQ(params.minorRadius, 0.5);
    EXPECT_TRUE(spheroid.checkConstraints());
}

TEST(SpheroidTest, ConstructorEnforcesMinorRadiusNotGreaterThanMajor) {
    ConfigureSpheroidBounds();

    Spheroid spheroid(SpheroidParams("cellB", 0.0f, 0.0f, 0.0f, 2.0f, 5.0f));
    SpheroidParams params = spheroid.getCellParams();

    EXPECT_DOUBLE_EQ(params.majorRadius, 2.0);
    EXPECT_DOUBLE_EQ(params.minorRadius, 2.0);
}

TEST(SpheroidTest, DrawColorsCenterPixelAndLeavesFarPixelUnchanged) {
    ConfigureSpheroidBounds();

    SimulationConfig simulationConfig;
    simulationConfig.cell_color = 0.9f;
    simulationConfig.background_color = 0.1f;

    cv::Mat image(21, 21, CV_32F, cv::Scalar(simulationConfig.background_color));
    Spheroid spheroid(SpheroidParams("cellC", 10.0f, 10.0f, 0.0f, 3.0f, 3.0f));

    spheroid.draw(image, simulationConfig, nullptr, 0.0f);

    EXPECT_NEAR(image.at<float>(10, 10), simulationConfig.cell_color, 1e-6f);
    EXPECT_NEAR(image.at<float>(0, 0), simulationConfig.background_color, 1e-6f);
}

TEST(SpheroidTest, DetectsOverlapAndNonOverlap) {
    ConfigureSpheroidBounds();

    Spheroid nearA(SpheroidParams("nearA", 0.0f, 0.0f, 0.0f, 3.0f, 3.0f));
    Spheroid nearB(SpheroidParams("nearB", 2.0f, 0.0f, 0.0f, 3.0f, 3.0f));
    Spheroid farB(SpheroidParams("farB", 50.0f, 0.0f, 0.0f, 3.0f, 3.0f));

    EXPECT_TRUE(Spheroid::checkIfCellsOverlap({nearA, nearB}));
    EXPECT_FALSE(Spheroid::checkIfCellsOverlap({nearA, farB}));
}
