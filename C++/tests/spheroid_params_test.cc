#include <gtest/gtest.h>

#include "Spheroid.hpp"

#include <unordered_map>

namespace {
void ConfigureSpheroidBounds() {
    Spheroid::cellConfig.minMajorRadius = 1.0;
    Spheroid::cellConfig.maxMajorRadius = 10.0;
    Spheroid::cellConfig.minMinorRadius = 1.0;
    Spheroid::cellConfig.maxMinorRadius = 8.0;
}

void ExpectNear(float a, float b) {
    EXPECT_NEAR(a, b, 1e-6f);
}
}

TEST(SpheroidParamsTest, GetParameterizedCellAppliesOffsets) {
    ConfigureSpheroidBounds();
    Spheroid base(SpheroidParams("p0", 10.0f, 20.0f, 30.0f, 4.0f, 3.0f, 0.0f, 0.0f, 0.0f));

    std::unordered_map<std::string, float> p = {
        {"x", 1.0f},
        {"y", -2.0f},
        {"z", 3.0f},
        {"majorRadius", 2.0f},
        {"minorRadius", -1.0f},
        {"thetaX", 0.1f},
        {"thetaY", 0.2f},
        {"thetaZ", 0.3f}
    };

    Spheroid updated = base.getParameterizedCell(p);
    SpheroidParams params = updated.getCellParams();

    ExpectNear(params.x, 11.0f);
    ExpectNear(params.y, 18.0f);
    ExpectNear(params.z, 33.0f);
    ExpectNear(params.majorRadius, 6.0f);
    ExpectNear(params.minorRadius, 2.0f);
    ExpectNear(params.theta_x, 0.1f);
    ExpectNear(params.theta_y, 0.2f);
    ExpectNear(params.theta_z, 0.3f);
}

TEST(SpheroidParamsTest, GetParameterizedCellClampsRadiiToBounds) {
    ConfigureSpheroidBounds();
    Spheroid base(SpheroidParams("p1", 0.0f, 0.0f, 0.0f, 9.0f, 1.5f));

    std::unordered_map<std::string, float> p = {
        {"majorRadius", 100.0f},
        {"minorRadius", -100.0f}
    };

    Spheroid updated = base.getParameterizedCell(p);
    SpheroidParams params = updated.getCellParams();

    ExpectNear(params.majorRadius, 10.0f);
    ExpectNear(params.minorRadius, 1.0f);
}

TEST(SpheroidParamsTest, ParameterizationStillEnforcesMinorNotGreaterThanMajor) {
    ConfigureSpheroidBounds();
    Spheroid base(SpheroidParams("p2", 0.0f, 0.0f, 0.0f, 2.0f, 2.0f));

    std::unordered_map<std::string, float> p = {
        {"majorRadius", -5.0f},
        {"minorRadius", 10.0f}
    };

    Spheroid updated = base.getParameterizedCell(p);
    SpheroidParams params = updated.getCellParams();

    ExpectNear(params.majorRadius, 1.0f);
    ExpectNear(params.minorRadius, 1.0f);
}

TEST(SpheroidParamsTest, CalculateCornersUsesMaxRadiusAroundCenter) {
    ConfigureSpheroidBounds();
    Spheroid spheroid(SpheroidParams("p3", 10.0f, 20.0f, 30.0f, 5.0f, 2.0f));

    auto [minCorner, maxCorner] = spheroid.calculateCorners();

    ASSERT_EQ(minCorner.size(), 3U);
    ASSERT_EQ(maxCorner.size(), 3U);

    ExpectNear(minCorner[0], 5.0f);
    ExpectNear(minCorner[1], 15.0f);
    ExpectNear(minCorner[2], 25.0f);

    ExpectNear(maxCorner[0], 15.0f);
    ExpectNear(maxCorner[1], 25.0f);
    ExpectNear(maxCorner[2], 35.0f);
}

TEST(SpheroidParamsTest, CalculateMinimumBoxContainsBothCells) {
    ConfigureSpheroidBounds();
    Spheroid a(SpheroidParams("a", 10.0f, 20.0f, 30.0f, 5.0f, 2.0f));
    Spheroid b(SpheroidParams("b", 20.0f, 10.0f, 40.0f, 3.0f, 1.0f));

    auto [minCorner, maxCorner] = a.calculateMinimumBox(b);

    ASSERT_EQ(minCorner.size(), 3U);
    ASSERT_EQ(maxCorner.size(), 3U);

    ExpectNear(minCorner[0], 5.0f);
    ExpectNear(minCorner[1], 7.0f);
    ExpectNear(minCorner[2], 25.0f);

    ExpectNear(maxCorner[0], 23.0f);
    ExpectNear(maxCorner[1], 25.0f);
    ExpectNear(maxCorner[2], 43.0f);
}
