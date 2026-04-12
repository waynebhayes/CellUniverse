#include <gtest/gtest.h>

#include "Spheroid.hpp"

#include <cmath>

namespace {
void ConfigureSpheroidBounds() {
    Spheroid::cellConfig.minMajorRadius = 1.0;
    Spheroid::cellConfig.maxMajorRadius = 20.0;
    Spheroid::cellConfig.minMinorRadius = 0.5;
    Spheroid::cellConfig.maxMinorRadius = 20.0;
}

constexpr float kTestBrightness = 1.0f;

SimulationConfig MakeSimConfig() {
    SimulationConfig cfg;
    cfg.background_color = 0.0f;
    cfg.cell_color = kTestBrightness;
    return cfg;
}
} // namespace

TEST(SpheroidRotationTest, NoRotationFillsAlongMajorXAxis) {
    ConfigureSpheroidBounds();
    const SimulationConfig cfg = MakeSimConfig();

    cv::Mat image(41, 41, CV_32F, cv::Scalar(cfg.background_color));
    Spheroid spheroid(SpheroidParams("rot0", 20.0f, 20.0f, 0.0f, 5.0f, 2.0f, 0.0f, 0.0f, 0.0f, kTestBrightness));

    spheroid.draw(image, cfg, 0.0f);

    EXPECT_NEAR(image.at<float>(20, 24), kTestBrightness, 1e-6f);
}

TEST(SpheroidRotationTest, YAxisQuarterTurnShrinksXAxisCrossSection) {
    ConfigureSpheroidBounds();
    const SimulationConfig cfg = MakeSimConfig();

    cv::Mat image(41, 41, CV_32F, cv::Scalar(cfg.background_color));
    const float kPiOver2 = static_cast<float>(M_PI) / 2.0f;
    Spheroid spheroid(SpheroidParams("rotY", 20.0f, 20.0f, 0.0f, 5.0f, 2.0f, 0.0f, kPiOver2, 0.0f, kTestBrightness));

    spheroid.draw(image, cfg, 0.0f);

    EXPECT_NEAR(image.at<float>(20, 24), cfg.background_color, 1e-6f);
    EXPECT_NEAR(image.at<float>(24, 20), kTestBrightness, 1e-6f);
}

TEST(SpheroidRotationTest, XAxisQuarterTurnShrinksYAxisCrossSection) {
    ConfigureSpheroidBounds();
    const SimulationConfig cfg = MakeSimConfig();

    cv::Mat image(41, 41, CV_32F, cv::Scalar(cfg.background_color));
    const float kPiOver2 = static_cast<float>(M_PI) / 2.0f;
    Spheroid spheroid(SpheroidParams("rotX", 20.0f, 20.0f, 0.0f, 5.0f, 2.0f, kPiOver2, 0.0f, 0.0f, kTestBrightness));

    spheroid.draw(image, cfg, 0.0f);

    EXPECT_NEAR(image.at<float>(24, 20), cfg.background_color, 1e-6f);
    EXPECT_NEAR(image.at<float>(20, 24), kTestBrightness, 1e-6f);
}

TEST(SpheroidRotationTest, ZAxisRotationPreservesShapeForOblateSpheroid) {
    ConfigureSpheroidBounds();
    const SimulationConfig cfg = MakeSimConfig();

    cv::Mat unrotated(41, 41, CV_32F, cv::Scalar(cfg.background_color));
    cv::Mat zRotated(41, 41, CV_32F, cv::Scalar(cfg.background_color));

    Spheroid noRotation(SpheroidParams("baseZ", 20.0f, 20.0f, 0.0f, 5.0f, 2.0f, 0.0f, 0.0f, 0.0f));
    Spheroid rotatedZ(SpheroidParams("rotZ", 20.0f, 20.0f, 0.0f, 5.0f, 2.0f, 0.0f, 0.0f, static_cast<float>(M_PI) / 3.0f));

    noRotation.draw(unrotated, cfg, 0.0f);
    rotatedZ.draw(zRotated, cfg, 0.0f);

    const double diff = cv::norm(unrotated, zRotated, cv::NORM_L1);
    // Rasterization can shift a few boundary pixels despite rotational symmetry (a == b).
    EXPECT_LT(diff, 10.0);
}

TEST(SpheroidRotationTest, DrawnRegionChangesAfterRotation) {
    ConfigureSpheroidBounds();
    const SimulationConfig cfg = MakeSimConfig();

    cv::Mat unrotated(41, 41, CV_32F, cv::Scalar(cfg.background_color));
    cv::Mat rotated(41, 41, CV_32F, cv::Scalar(cfg.background_color));

    Spheroid noRotation(SpheroidParams("base", 20.0f, 20.0f, 0.0f, 5.0f, 2.0f, 0.0f, 0.0f, 0.0f));
    const float kPiOver2 = static_cast<float>(M_PI) / 2.0f;
    Spheroid yRotated(SpheroidParams("rot", 20.0f, 20.0f, 0.0f, 5.0f, 2.0f, 0.0f, kPiOver2, 0.0f));

    noRotation.draw(unrotated, cfg, 0.0f);
    yRotated.draw(rotated, cfg, 0.0f);

    const double diff = cv::norm(unrotated, rotated, cv::NORM_L1);
    EXPECT_GT(diff, 0.0);
}
