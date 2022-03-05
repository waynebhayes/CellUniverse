from .Change import Change, useDistanceObjective
from .utils import is_cell, is_background, objective, dist_objective, check_constraints
from global_optimization.Modules import CellNodeM, FrameM


class Combination(Change):
    """Move split forward: o<8=8 -> o-o<8"""
    def __init__(self, node: CellNodeM, config, child_realimage, child_synthimage, child_cellmap, child_frame: FrameM, distmap=None):
        self.node = node
        self.config = config
        self.realimage = child_realimage
        self.synthimage = child_synthimage
        self.cellmap = child_cellmap
        self.frame = child_frame
        self._checks = []
        self.combination = None
        self.distmap = distmap

        if len(self.node.children) == 2:
            self.combination = self.node.children[0].cell.combine(self.node.children[1].cell)

    def get_checks(self):
        if self.combination and not self._checks:
            self._checks.append((self.node.cell, self.combination))
            p1, p2 = self.combination.split(self.node.children[0].cell.split_alpha)

            for gc in self.node.grandchildren:
                if gc.cell.name == p1.name:
                    self._checks.append((p1, gc.cell))
                elif gc.cell.name == p2.name:
                    self._checks.append((p2, gc.cell))

        return self._checks

    @property
    def is_valid(self) -> bool:
        return len(self.node.children) == 2 and len(self.node.grandchildren) <= 2 and \
            check_constraints(self.config, self.realimage.shape, [self.combination], self.get_checks())

    @property
    def costdiff(self) -> float:
        overlap_cost = self.config["overlap.cost"]
        new_synth = self.synthimage.copy()
        new_cellmap = self.cellmap.copy()
        region = self.combination.simulated_region(self.frame.simulation_config)

        for child in self.node.children:
            region = region.union(child.cell.simulated_region(self.frame.simulation_config))

        for child in self.node.children:
            child.cell.draw(new_synth, new_cellmap, is_background, self.frame.simulation_config)

        self.combination.draw(new_synth, new_cellmap, is_cell, self.frame.simulation_config)

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

        return end_cost - start_cost + self.config["combine.cost"]

    def apply(self) -> None:
        self.combination.draw(self.synthimage, self.cellmap, is_cell, self.frame.simulation_config)
        grandchildren = self.node.grandchildren

        for child in self.node.children:
            del self.frame.node_map[child.cell.name]
            child.cell.draw(self.synthimage, self.cellmap, is_background, self.frame.simulation_config)

        self.node.children = []
        combination_node = self.node.make_child(self.combination)
        self.frame.node_map[self.combination.name] = combination_node

        for gc in grandchildren:
            combination_node.children.append(gc)
            gc.parent = combination_node
