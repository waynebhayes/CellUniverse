#include <gtest/gtest.h>

#include "ConfigTypes.hpp"

TEST(ConfigTypesTest, FlatCellRotationRefineEnabledDefaultsToTrue) {
    const YAML::Node node = YAML::Load(R"(
cellType: ellipsoid
cell:
  x: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  y: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  z: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  aRadius: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  cRadius: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  thetaX: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  thetaY: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  thetaZ: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  minARadius: 1.0
  maxARadius: 10.0
  minCRadius: 1.0
  maxCRadius: 10.0
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
cellType: ellipsoid
cell:
  x: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  y: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  z: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  aRadius: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  cRadius: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  thetaX: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  thetaY: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  thetaZ: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  minARadius: 1.0
  maxARadius: 10.0
  minCRadius: 1.0
  maxCRadius: 10.0
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
