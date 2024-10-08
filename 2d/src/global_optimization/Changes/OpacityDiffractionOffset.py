from .Change import Change, useDistanceObjective
from .utils import objective, dist_objective, generate_synthetic_image
import numpy as np


class OpacityDiffractionOffset(Change):
    def __init__(self, frame, realimage, synthimage, cellmap, config, distmap=None):
        self.frame = frame
        self.realimage = realimage
        self.old_synthimage = synthimage
        self.cellmap = cellmap
        self.old_simulation_config = frame.simulation_config
        self.new_simulation_config = frame.simulation_config.copy()
        self.config = config

        opacity_offset_mu = config["opacity_offset.mu"]
        opacity_offset_sigma = config["opacity_offset.sigma"]

        diffraction_strength_offset_mu = config["diffraction_strength_offset.mu"]
        diffraction_strength_offset_sigma = config["diffraction_strength_offset.sigma"]

        diffraction_sigma_offset_mu = config["diffraction_sigma_offset.mu"]
        diffraction_sigma_offset_sigma = config["diffraction_sigma_offset.sigma"]

        self.new_simulation_config["cell.opacity"] += np.random.normal(opacity_offset_mu, opacity_offset_sigma)
        self.new_simulation_config["light.diffraction.strength"] += np.random.normal(diffraction_strength_offset_mu, diffraction_strength_offset_sigma)
        self.new_simulation_config["light.diffraction.sigma"] += np.random.normal(diffraction_sigma_offset_mu, diffraction_sigma_offset_sigma)
        self.new_synthimage, _ = generate_synthetic_image(frame.nodes, realimage.shape, self.new_simulation_config)

    @property
    def is_valid(self) -> bool:
        return self.new_simulation_config["cell.opacity"] >= 0 and \
            self.new_simulation_config["light.diffraction.strength"] >= 0 and \
            self.new_simulation_config["light.diffraction.sigma"] >= 0

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
        self.old_simulation_config["cell.opacity"] = self.new_simulation_config["cell.opacity"]
        self.old_simulation_config["light.diffraction.strength"] = self.new_simulation_config["light.diffraction.strength"]
        self.old_simulation_config["light.diffraction.sigma"] = self.new_simulation_config["light.diffraction.sigma"]
