#include <gtest/gtest.h>

#include "Ellipsoid.hpp"

namespace {
void ConfigureEllipsoidBounds() {
    Ellipsoid::cellConfig.minARadius = 1.0;
    Ellipsoid::cellConfig.maxARadius = 10.0;
    Ellipsoid::cellConfig.minCRadius = 1.0;
    Ellipsoid::cellConfig.maxCRadius = 8.0;
}

void ExpectNear(float a, float b) {
    EXPECT_NEAR(a, b, 1e-6f);
}
}

TEST(EllipsoidParamsTest, CalculateCornersUsesMaxRadiusAroundCenter) {
    ConfigureEllipsoidBounds();
    Ellipsoid spheroid(EllipsoidParams("p3", 10.0f, 20.0f, 30.0f, 5.0f, 2.0f));

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

TEST(EllipsoidParamsTest, CalculateMinimumBoxContainsBothCells) {
    ConfigureEllipsoidBounds();
    Ellipsoid a(EllipsoidParams("a", 10.0f, 20.0f, 30.0f, 5.0f, 2.0f));
    Ellipsoid b(EllipsoidParams("b", 20.0f, 10.0f, 40.0f, 3.0f, 1.0f));

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
