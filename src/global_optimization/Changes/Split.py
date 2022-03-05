from .Change import Change, useDistanceObjective
from .utils import is_cell, is_background, objective, dist_objective, check_constraints
import numpy as np
from global_optimization.Modules import CellNodeM, FrameM


class Split(Change):
    """Move split backward: o-o<8 -> o<8=8"""
    def __init__(self, node: CellNodeM, config, child_realimage, child_synthimage, child_cellmap, child_frame: FrameM, distmap=None):
        self.node = node
        self.config = config
        self.realimage = child_realimage
        self.synthimage = child_synthimage
        self.cellmap = child_cellmap
        self.frame = child_frame
        self._checks = []
        self.s1 = self.s2 = None
        self.distmap = distmap

        if len(self.node.children) == 1:
            alpha = np.random.uniform(0.2, 0.8)
            self.s1, self.s2 = self.node.children[0].cell.split(alpha)

    def get_checks(self):
        if len(self.node.children) == 1 and not self._checks:
            p1, p2 = self.node.cell.split(self.s1.split_alpha)

            if p1.name == self.s1.name:
                self._checks.append((p1, self.s1))
            elif p1.name == self.s2.name:
                self._checks.append((p1, self.s2))

            if p2.name == self.s1.name:
                self._checks.append((p2, self.s1))
            elif p2.name == self.s2.name:
                self._checks.append((p2, self.s2))

            for child in self.node.grandchildren:
                if child.cell.name == self.s1.name:
                    self._checks.append((self.s1, child.cell))
                elif child.cell.name == self.s2.name:
                    self._checks.append((self.s2, child.cell))

        return self._checks

    @property
    def is_valid(self) -> bool:
        return len(self.node.children) == 1 and len(self.node.grandchildren) != 1 and \
            check_constraints(self.config, self.realimage.shape, [self.s1, self.s2], self.get_checks())

    @property
    def costdiff(self) -> float:
        overlap_cost = self.config["overlap.cost"]
        new_synth = self.synthimage.copy()
        new_cellmap = self.cellmap.copy()
        region = self.node.children[0].cell.simulated_region(self.frame.simulation_config).\
            union(self.s1.simulated_region(self.frame.simulation_config)).\
            union(self.s2.simulated_region(self.frame.simulation_config))
        self.node.children[0].cell.draw(new_synth, new_cellmap, is_background, self.frame.simulation_config)
        self.s1.draw(new_synth, new_cellmap, is_cell, self.frame.simulation_config)
        self.s2.draw(new_synth, new_cellmap, is_cell, self.frame.simulation_config)

        if useDistanceObjective:
            start_cost = dist_objective(self.realimage[region.top:region.bottom, region.left:region.right],
                                        self.synthimage[region.top:region.bottom, region.left:region.right],
                                        self.distmap[region.top:region.bottom, region.left:region.right],
                                        self.cellmap[region.top:region.bottom, region.left:region.right],
                                        overlap_cost)
            end_cost = dist_objective(self.realimage[region.top:region.bottom, region.left:region.right],
                                      new_synth[region.top:region.bottom, region.left:region.right],
                                      self.distmap[region.top:region.bottom, region.left:region.right],
                                      new_cellmap[region.top:region.bottom, region.left:region.right],
                                      overlap_cost)
        else:
            start_cost = objective(self.realimage[region.top:region.bottom, region.left:region.right],
                                   self.synthimage[region.top:region.bottom, region.left:region.right],
                                   self.cellmap[region.top:region.bottom, region.left:region.right],
                                   overlap_cost, self.config["cell.importance"])
            end_cost = objective(self.realimage[region.top:region.bottom, region.left:region.right],
                                 new_synth[region.top:region.bottom, region.left:region.right],
                                 new_cellmap[region.top:region.bottom, region.left:region.right],
                                 overlap_cost, self.config["cell.importance"])

        return end_cost - start_cost + self.config["split.cost"]

    def apply(self) -> None:
        self.node.children[0].cell.draw(self.synthimage, self.cellmap, is_background, self.frame.simulation_config)
        self.s1.draw(self.synthimage, self.cellmap, is_cell, self.frame.simulation_config)
        self.s2.draw(self.synthimage, self.cellmap, is_cell, self.frame.simulation_config)
        del self.frame.node_map[self.node.children[0].cell.name]
        grandchildren = self.node.grandchildren
        self.node.children = []
        s1_node = self.node.make_child(self.s1)
        s2_node = self.node.make_child(self.s2)
        self.frame.node_map[self.s1.name] = s1_node
        self.frame.node_map[self.s2.name] = s2_node

        for gc in grandchildren:
            if gc.cell.name == self.s1.name:
                gc.parent = s1_node
                s1_node.children = [gc]
            elif gc.cell.name == self.s2.name:
                gc.parent = s2_node
                s2_node.children = [gc]
