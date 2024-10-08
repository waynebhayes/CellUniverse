from .Change import Change, useDistanceObjective
from .utils import objective, dist_objective, generate_synthetic_image
import numpy as np


class BackGroundLuminosityOffset(Change):
    def __init__(self, frame, realimage, synthimage, cellmap, config, distmap=None):
        self.frame = frame
        self.realimage = realimage
        self.old_synthimage = synthimage
        self.cellmap = cellmap
        self.old_simulation_config = frame.simulation_config
        self.new_simulation_config = frame.simulation_config.copy()
        self.config = config

        offset_mu = config["background_offset.mu"]
        offset_sigma = config["background_offset.sigma"]

        cell_brightness_mu = config["cell_brightness.mu"]
        cell_brightness_sigma = config["cell_brightness.sigma"]

        self.new_simulation_config["background.color"] += np.random.normal(offset_mu, offset_sigma)
        self.new_simulation_config["cell.color"] += np.random.normal(cell_brightness_mu, cell_brightness_sigma)
        self.new_synthimage, _ = generate_synthetic_image(frame.nodes, realimage.shape, self.new_simulation_config)

    @property
    def is_valid(self) -> bool:
        return self.frame.simulation_config["background.color"] > 0 and self.frame.simulation_config["cell.color"] - self.frame.simulation_config["background.color"] > 0.2

    @property
    def costdiff(self) -> float:
        overlap_cost = self.config["overlap.cost"]
        if useDistanceObjective:
            start_cost = dist_objective(self.realimage, self.old_synthimage, self.distmap, self.cellmap, overlap_cost)
            end_cost = dist_objective(self.realimage, self.new_synthimage, self.distmap, self.cellmap, overlap_cost)
        else:
            start_cost = objective(self.realimage, self.old_synthimage, self.cellmap, overlap_cost, self.config["cell.importance"])
            end_cost = objective(self.realimage, self.new_synthimage, self.cellmap, overlap_cost, self.config["cell.importance"])
        return end_cost - start_cost

    def apply(self):
        self.old_synthimage[:] = self.new_synthimage
        self.old_simulation_config["background.color"] = self.new_simulation_config["background.color"]
        self.old_simulation_config["cell.color"] = self.new_simulation_config["cell.color"]
