from .Change import Change, useDistanceObjective
from .utils import is_cell, is_background, objective, dist_objective, check_constraints
import numpy as np
from copy import deepcopy
from typing import Any, Dict, List, Tuple
from global_optimization.Modules import CellNodeM, FrameM, LineageM


class Perturbation(Change):
    def __init__(self, node: CellNodeM, config: Dict[str, Any], frame: FrameM):
        self.node = node
        self.config = config
        self._checks = []
        self.frame = frame
        cell = node.cell
        new_cell = deepcopy(cell)
        self.replacement_cell = new_cell
        valid = False
        badcount = 0

        perturb_conf = config["perturbation"]
        p_x = perturb_conf["prob.x"]
        p_y = perturb_conf["prob.y"]
        p_z = perturb_conf["prob.z"]
        p_width = perturb_conf["prob.width"]
        p_length = perturb_conf["prob.length"]
        p_rotation = perturb_conf["prob.rotation"]

        x_mu = perturb_conf["modification.x.mu"]
        y_mu = perturb_conf["modification.y.mu"]
        width_mu = perturb_conf["modification.width.mu"]
        length_mu = perturb_conf["modification.length.mu"]
        rotation_mu = perturb_conf["modification.rotation.mu"]

        x_sigma = perturb_conf["modification.x.sigma"]
        y_sigma = perturb_conf["modification.y.sigma"]
        width_sigma = perturb_conf["modification.width.sigma"]
        length_sigma = perturb_conf["modification.length.sigma"]
        rotation_sigma = perturb_conf["modification.rotation.sigma"]

        # functionality to change individual cell opacity commented.
        # simulation_config = config["simulation"]
        # if simulation_config["image.type"] == "graySynthetic":
        #     p_opacity = perturb_conf["prob.opacity"]
        #     opacity_mu = perturb_conf["modification.opacity.mu"]
        #     opacity_sigma = perturb_conf["modification.opacity.sigma"]

        # set starting properties
        # if simulation_config["image.type"] == "graySynthetic":
        #     p_decision = np.array([p_x, p_y, p_width, p_length, p_rotation, p_opacity])
        # else:
        #     p_decision = np.array([p_x, p_y, p_width, p_length, p_rotation])
        p_decision = np.array([p_x, p_y, p_width, p_length, p_rotation, p_z])

        p = np.random.uniform(0.0, 1.0, size=p_decision.size)
        # generate a sequence such that at least an attribute must be modified
        while not valid and badcount < 50:
            while (p > p_decision).all():
                p = np.random.uniform(0.0, 1.0, size=p_decision.size)

            if p[0] < p_decision[0]:  # perturb x
                new_cell.x = cell.x + np.random.normal(x_mu, x_sigma)

            if p[1] < p_decision[1]:  # perturb y
                new_cell.y = cell.y + np.random.normal(y_mu, y_sigma)

            if p[2] < p_decision[2]:  # perturb width
                new_cell.width = cell.width + np.random.normal(width_mu, width_sigma)

            if p[3] < p_decision[3]:  # perturb length
                new_cell.length = cell.length + np.random.normal(length_mu, length_sigma)

            if p[4] < p_decision[4]:  # perturb rotation
                new_cell.rotation = cell.rotation + np.random.normal(rotation_mu, rotation_sigma)

            # if p[5] < p_decision[5]:  # perturb z
            #     new_cell.z = cell.z + int(np.random.normal(0, 0.1))

            # if simulation_config["image.type"] == "graySynthetic" and p[5] < p_decision[5]:
                # new_cell.opacity = cell.opacity + (np.random.normal(opacity_mu, opacity_sigma))

            # ensure that those changes fall within constraints
            valid = self.is_valid

            if not valid:
                badcount += 1

    def is_valid(self, real_image_shape: Tuple[int, int]) -> bool:
        return check_constraints(self.config, real_image_shape, [self.replacement_cell], self.get_checks())

    def costdiff(self, frame_stacks, z_slices) -> float:
        overlap_cost = self.config["overlap.cost"]
        total_diff = 0
        for synth_image, cell_map, dist_map, real_image, z in zip(*frame_stacks, z_slices):
            new_synth = synth_image.copy()
            new_cellmap = cell_map.copy()
            region = self.node.cell.simulated_region(self.frame.simulation_config).\
                union(self.replacement_cell.simulated_region(self.frame.simulation_config))
            self.node.cell.draw(new_synth, new_cellmap, is_background, self.frame.simulation_config)
            self.replacement_cell.draw(new_synth, new_cellmap, is_cell, self.frame.simulation_config, z = z)

            if useDistanceObjective:
                start_cost = dist_objective(real_image[region.top:region.bottom, region.left:region.right],
                                            synth_image[region.top:region.bottom, region.left:region.right],
                                            dist_map[region.top:region.bottom, region.left:region.right],
                                            cell_map[region.top:region.bottom, region.left:region.right],
                                            overlap_cost)
                end_cost = dist_objective(real_image[region.top:region.bottom, region.left:region.right],
                                          new_synth[region.top:region.bottom, region.left:region.right],
                                          dist_map[region.top:region.bottom, region.left:region.right],
                                          new_cellmap[region.top:region.bottom, region.left:region.right],
                                          overlap_cost)
            else:
                start_cost = objective(real_image[region.top:region.bottom, region.left:region.right],
                                       synth_image[region.top:region.bottom, region.left:region.right],
                                       cell_map[region.top:region.bottom, region.left:region.right],
                                       overlap_cost, self.config["cell.importance"])
                end_cost = objective(real_image[region.top:region.bottom, region.left:region.right],
                                     new_synth[region.top:region.bottom, region.left:region.right],
                                     new_cellmap[region.top:region.bottom, region.left:region.right],
                                     overlap_cost, self.config["cell.importance"])
            total_diff += end_cost - start_cost

        return total_diff

    def apply(self, frame_stacks, z_slices):
        for synth_image, cell_map, dist_map, real_image, z in zip(*frame_stacks, z_slices):
            self.node.cell.draw(synth_image, cell_map, is_background, self.frame.simulation_config)
            self.replacement_cell.draw(synth_image, cell_map, is_cell, self.frame.simulation_config, z = z)
        self.frame.add_cell(self.replacement_cell)

    def get_checks(self) -> List[Tuple['cell.Bacilli', 'cell.Bacilli']]:
        if not self._checks:
            if self.node.parent:
                if len(self.node.parent.children) == 1:
                    self._checks.append((self.node.parent.cell, self.replacement_cell))
                elif len(self.node.parent.children) == 2:
                    p1, p2 = self.node.parent.cell.split(self.node.cell.split_alpha)

                    if p1.name == self.replacement_cell.name:
                        self._checks.append((p1, self.replacement_cell))
                    elif p2.name == self.replacement_cell.name:
                        self._checks.append((p2, self.replacement_cell))

            if len(self.node.children) == 1:
                self._checks.append((self.replacement_cell, self.node.children[0].cell))
            elif len(self.node.children) == 2:
                p1, p2 = self.replacement_cell.split(self.node.children[0].cell.split_alpha)
                for c in self.node.children:
                    if c.cell.name == p1.name:
                        self._checks.append((p1, c.cell))
                    elif c.cell.name == p2.name:
                        self._checks.append((p2, c.cell))

        return self._checks
