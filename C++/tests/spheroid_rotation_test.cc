#include <gtest/gtest.h>

#include "Ellipsoid.hpp"

#include <cmath>

namespace {
void ConfigureEllipsoidBounds() {
    Ellipsoid::cellConfig.minARadius = 1.0;
    Ellipsoid::cellConfig.maxARadius = 20.0;
    Ellipsoid::cellConfig.minCRadius = 0.5;
    Ellipsoid::cellConfig.maxCRadius = 20.0;
}

constexpr float kTestBrightness = 1.0f;

SimulationConfig MakeSimConfig() {
    SimulationConfig cfg;
    cfg.background_color = 0.0f;
    cfg.cell_color = kTestBrightness;
    return cfg;
}
} // namespace

TEST(EllipsoidRotationTest, NoRotationFillsAlongMajorXAxis) {
    ConfigureEllipsoidBounds();
    const SimulationConfig cfg = MakeSimConfig();

    cv::Mat image(41, 41, CV_32F, cv::Scalar(cfg.background_color));
    Ellipsoid spheroid(EllipsoidParams("rot0", 20.0f, 20.0f, 0.0f, 5.0f, 2.0f, 0.0f, 0.0f, 0.0f, kTestBrightness));

    spheroid.draw(image, cfg, 0.0f);

    EXPECT_NEAR(image.at<float>(20, 24), kTestBrightness, 1e-6f);
}

TEST(EllipsoidRotationTest, YAxisQuarterTurnShrinksXAxisCrossSection) {
    ConfigureEllipsoidBounds();
    const SimulationConfig cfg = MakeSimConfig();

    cv::Mat image(41, 41, CV_32F, cv::Scalar(cfg.background_color));
    const float kPiOver2 = static_cast<float>(M_PI) / 2.0f;
    Ellipsoid spheroid(EllipsoidParams("rotY", 20.0f, 20.0f, 0.0f, 5.0f, 2.0f, 0.0f, kPiOver2, 0.0f, kTestBrightness));

    spheroid.draw(image, cfg, 0.0f);

    EXPECT_NEAR(image.at<float>(20, 24), cfg.background_color, 1e-6f);
    EXPECT_NEAR(image.at<float>(24, 20), kTestBrightness, 1e-6f);
}

TEST(EllipsoidRotationTest, XAxisQuarterTurnShrinksYAxisCrossSection) {
    ConfigureEllipsoidBounds();
    const SimulationConfig cfg = MakeSimConfig();

    cv::Mat image(41, 41, CV_32F, cv::Scalar(cfg.background_color));
    const float kPiOver2 = static_cast<float>(M_PI) / 2.0f;
    Ellipsoid spheroid(EllipsoidParams("rotX", 20.0f, 20.0f, 0.0f, 5.0f, 2.0f, kPiOver2, 0.0f, 0.0f, kTestBrightness));

    spheroid.draw(image, cfg, 0.0f);

    EXPECT_NEAR(image.at<float>(24, 20), cfg.background_color, 1e-6f);
    EXPECT_NEAR(image.at<float>(20, 24), kTestBrightness, 1e-6f);
}

TEST(EllipsoidRotationTest, ZAxisRotationPreservesShapeForOblateEllipsoid) {
    ConfigureEllipsoidBounds();
    const SimulationConfig cfg = MakeSimConfig();

    cv::Mat unrotated(41, 41, CV_32F, cv::Scalar(cfg.background_color));
    cv::Mat zRotated(41, 41, CV_32F, cv::Scalar(cfg.background_color));

    Ellipsoid noRotation(EllipsoidParams("baseZ", 20.0f, 20.0f, 0.0f, 5.0f, 2.0f, 0.0f, 0.0f, 0.0f));
    Ellipsoid rotatedZ(EllipsoidParams("rotZ", 20.0f, 20.0f, 0.0f, 5.0f, 2.0f, 0.0f, 0.0f, static_cast<float>(M_PI) / 3.0f));

    noRotation.draw(unrotated, cfg, 0.0f);
    rotatedZ.draw(zRotated, cfg, 0.0f);

    const double diff = cv::norm(unrotated, zRotated, cv::NORM_L1);
    // Rasterization can shift a few boundary pixels despite rotational symmetry (a == b).
    EXPECT_LT(diff, 10.0);
}

TEST(EllipsoidRotationTest, DrawnRegionChangesAfterRotation) {
    ConfigureEllipsoidBounds();
    const SimulationConfig cfg = MakeSimConfig();

    cv::Mat unrotated(41, 41, CV_32F, cv::Scalar(cfg.background_color));
    cv::Mat rotated(41, 41, CV_32F, cv::Scalar(cfg.background_color));

    Ellipsoid noRotation(EllipsoidParams("base", 20.0f, 20.0f, 0.0f, 5.0f, 2.0f, 0.0f, 0.0f, 0.0f));
    const float kPiOver2 = static_cast<float>(M_PI) / 2.0f;
    Ellipsoid yRotated(EllipsoidParams("rot", 20.0f, 20.0f, 0.0f, 5.0f, 2.0f, 0.0f, kPiOver2, 0.0f));

    noRotation.draw(unrotated, cfg, 0.0f);
    yRotated.draw(rotated, cfg, 0.0f);

    const double diff = cv::norm(unrotated, rotated, cv::NORM_L1);
    EXPECT_GT(diff, 0.0);
}
