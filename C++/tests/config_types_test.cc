#include <gtest/gtest.h>

#include "ConfigTypes.hpp"

TEST(ConfigTypesTest, FlatCellRotationRefineEnabledDefaultsToTrue) {
    const YAML::Node node = YAML::Load(R"(
cellType: spheroid
cell:
  x: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  y: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  z: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  majorRadius: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  minorRadius: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  thetaX: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  thetaY: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  thetaZ: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  minMajorRadius: 1.0
  maxMajorRadius: 10.0
  minMinorRadius: 1.0
  maxMinorRadius: 10.0
simulation:
  iterations_per_cell: 1
  background_color: 0.0
  z_scaling: 1.0
  blur_sigma: 0.0
prob: {}
)");

    BaseConfig config;
    config.explodeConfig(node);

    ASSERT_TRUE(config.cell);
    EXPECT_TRUE(config.cell->flatCellRotationRefineEnabled);
}

TEST(ConfigTypesTest, FlatCellRotationRefineEnabledCanBeDisabledFromYaml) {
    const YAML::Node node = YAML::Load(R"(
cellType: spheroid
cell:
  x: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  y: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  z: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  majorRadius: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  minorRadius: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  thetaX: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  thetaY: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  thetaZ: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  minMajorRadius: 1.0
  maxMajorRadius: 10.0
  minMinorRadius: 1.0
  maxMinorRadius: 10.0
  flatCellRotationRefineEnabled: false
simulation:
  iterations_per_cell: 1
  background_color: 0.0
  z_scaling: 1.0
  blur_sigma: 0.0
prob: {}
)");

    BaseConfig config;
    config.explodeConfig(node);

    ASSERT_TRUE(config.cell);
    EXPECT_FALSE(config.cell->flatCellRotationRefineEnabled);
}
