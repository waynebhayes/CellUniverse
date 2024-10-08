from .Change import Change, useDistanceObjective
from .utils import objective, dist_objective, generate_synthetic_image
import numpy as np
from copy import deepcopy


class CameraShift(Change):
    def __init__(self, frame, realimage, synthimage, cellmap, config, distmap=None):
        self.frame = frame
        self.realimage = realimage
        self.old_synthimage = synthimage
        self.old_cellmap = cellmap
        self.simulation_config = frame.simulation_config
        self.config = config
        self.new_node_map = deepcopy(frame.node_map)

        camera_shift_x_sigma = config["camera"]["modification.x.sigma"]
        camera_shift_y_sigma = config["camera"]["modification.y.sigma"]

        shift_x = np.random.normal(0, camera_shift_x_sigma)
        shift_y = np.random.normal(0, camera_shift_y_sigma)

        for node in self.new_node_map.values():
            node.cell.x += shift_x
            node.cell.y += shift_y

    @property
    def is_valid(self) -> bool:
        for node in self.new_node_map.values():
            if not (0 <= node.cell.x < self.realimage.shape[1] and 0 <= node.cell.y < self.realimage.shape[0]):
                return False
        return True

    @property
    def costdiff(self) -> float:
        overlap_cost = self.config["overlap.cost"]

        self.new_synthimage, self.new_cellmap = generate_synthetic_image(self.new_node_map.values(), self.realimage.shape, self.simulation_config)

        if useDistanceObjective:
            start_cost = dist_objective(self.realimage, self.old_synthimage, self.distmap, self.old_cellmap, overlap_cost)
            end_cost = dist_objective(self.realimage, self.new_synthimage, self.distmap, self.new_cellmap, overlap_cost)
        else:
            start_cost = objective(self.realimage, self.old_synthimage, self.old_cellmap, overlap_cost, self.config["cell.importance"])
            end_cost = objective(self.realimage, self.new_synthimage, self.new_cellmap, overlap_cost, self.config["cell.importance"])
        return end_cost - start_cost

    def apply(self):
        self.old_synthimage[:] = self.new_synthimage
        self.old_cellmap[:] = self.new_cellmap
        self.frame.node_map = self.new_node_map
