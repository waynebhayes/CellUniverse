from copy import deepcopy
from math import sqrt
from time import time
from typing import List, Dict, Any, Tuple
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image
import pandas as pd

import cell
import optimization

useDistanceObjective = False
totalCostDiff = 0.0

def check_constraints(config, imageshape, cells: List[cell.Bacilli], pairs: List[Tuple[cell.Bacilli, cell.Bacilli]] = None):
    max_displacement = config['bacilli.maxSpeed'] / config['global.framesPerSecond']
    max_rotation = config['bacilli.maxSpin'] / config['global.framesPerSecond']
    min_growth = config['bacilli.minGrowth']
    max_growth = config['bacilli.maxGrowth']
    min_width = config['bacilli.minWidth']
    max_width = config['bacilli.maxWidth']
    min_length = config['bacilli.minLength']
    max_length = config['bacilli.maxLength']

    for cell in cells:
        if not (0 <= cell.x < imageshape[1] and 0 <= cell.y < imageshape[0]):
            #print("shape")
            return False
        elif cell.width < min_width or cell.width > max_width:
            #print("width")
            return False
        elif not (min_length < cell.length < max_length):
            #print("length")
            return False
        elif config["simulation"]["image.type"] == "graySynthetic" and cell.opacity < 0:
            return False

    for cell1, cell2 in pairs:
        displacement = sqrt(np.sum((cell1.position - cell2.position)) ** 2)
        if displacement > max_displacement:
            return False
        elif abs(cell2.rotation - cell1.rotation) > max_rotation:
            return False
        elif not (min_growth < cell2.length - cell1.length < max_growth):
            return False

    return True


class CellNodeM:
    def __init__(self, cell: cell.Bacilli, parent: 'CellNodeM' = None):
        self.cell = cell
        self.parent = parent
        self.children: List[CellNodeM] = []

    def __repr__(self):
        return f'<name={self.cell.name}, parent={self.parent.cell.name if self.parent else None}, children={[node.cell.name for node in self.children]}>'

    @property
    def grandchildren(self):
        grandchildren = []
        for child in self.children:
            grandchildren.extend(child.children)
        return grandchildren

    def make_child(self, cell: cell.Bacilli):
        child = CellNodeM(cell, self)
        self.children.append(child)
        return child


class FrameM:
    def __init__(self, simulation_config=None, prev: 'FrameM' = None):
        #simulation_config is passed by value and modified within the object
        self.node_map: Dict[str, CellNodeM] = {}
        self.prev = prev
        self.simulation_config = simulation_config.copy()
     
    def __repr__(self):
        return str(list(self.node_map.values()))

    @property
    def nodes(self) -> List[CellNodeM]:
        return list(self.node_map.values())

    def add_cell(self, cell: cell.Bacilli):
        if cell.name in self.node_map:
            self.node_map[cell.name].cell = cell
        elif self.prev and cell.name in self.prev.node_map:
            self.node_map[cell.name] = self.prev.node_map[cell.name].make_child(cell)
        elif self.prev and cell.name[:-1] in self.prev.node_map:
            self.node_map[cell.name] = self.prev.node_map[cell.name[:-1]].make_child(cell)
        else:
            self.node_map[cell.name] = CellNodeM(cell)


class LineageM:
    def __init__(self, simulation_config=None):
        self.frames = [FrameM(simulation_config)]

    def __repr__(self):
        return '\n'.join([str(frame) for frame in self.frames])

    @property
    def total_cell_count(self):
        return sum(len(frame.node_map) for frame in self.frames)

    def count_cells_in(self, start, end):
        if start is None or start < 0:
            start = 0
        if end is None or end > len(self.frames):
            end = len(self.frames)
        return sum(len(frame.node_map) for frame in self.frames[start:end])

    def forward(self):
        self.frames.append(FrameM(self.frames[-1].simulation_config, self.frames[-1]))

    def copy_forward(self):
        self.forward()
        for cell_node in self.frames[-2].nodes:
            self.frames[-1].add_cell(cell_node.cell)

    def choose_random_frame_index(self, start=None, end=None) -> int:
        if start is None or start < 0:
            start = 0

        if end is None or end > len(self.frames):
            end = len(self.frames)

        threshold = int(np.random.random_sample() * self.count_cells_in(start, end))

        for i in range(start, end):
            frame = self.frames[i]
            if len(frame.nodes) > threshold:
                return i
            else:
                threshold -= len(frame.nodes)

        raise RuntimeError('this should not have happened')


class Change:
    @property
    def is_valid(self) -> bool:
        pass

    @property
    def costdiff(self) -> float:
        pass

    def apply(self) -> None:
        pass

class Perturbation(Change):
    def __init__(self, node: CellNodeM, config: Dict[str, Any], realimage, synthimage, cellmap, frame, distmap=None):
        self.node = node
        self.realimage = realimage
        self.synthimage = synthimage
        self.cellmap = cellmap
        self.config = config
        self._checks = []
        self.frame = frame
        cell = node.cell
        new_cell = deepcopy(cell)
        self.replacement_cell = new_cell
        valid = False
        badcount = 0
        self.distmap = distmap
        
        perturb_conf = config["perturbation"]
        p_x = perturb_conf["prob.x"]
        p_y = perturb_conf["prob.y"]
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
        
        simulation_config = config["simulation"]
        if simulation_config["image.type"] == "graySynthetic":
            p_opacity = perturb_conf["prob.opacity"]
            opacity_mu = perturb_conf["modification.opacity.mu"]
            opacity_sigma = perturb_conf["modification.opacity.sigma"]
            
        # set starting properties
        if simulation_config["image.type"] == "graySynthetic":
            p_decision = np.array([p_x, p_y, p_width, p_length, p_rotation, p_opacity])
        else:
            p_decision = np.array([p_x, p_y, p_width, p_length, p_rotation])
            
        p = np.random.uniform(0.0, 1.0, size= p_decision.size)
        # generate a sequence such that at least an attribute must be modified
        while not valid and badcount < 50:
            while (p > p_decision).all():
                p = np.random.uniform(0.0, 1.0, size= p_decision.size)
        
            if p[0] < p_decision[0]: #perturb x
                new_cell.x = cell.x + np.random.normal(x_mu, x_sigma)
    
            if p[1] < p_decision[1]: #perturb y
                new_cell.y = cell.y + np.random.normal(y_mu, y_sigma)
        
            if p[2] < p_decision[2]: #perturb width
                new_cell.width = cell.width + np.random.normal(width_mu, width_sigma)
        
            if p[3] < p_decision[3]: #perturb length
                new_cell.length = cell.length + np.random.normal(length_mu, length_sigma)
                
            if p[4] < p_decision[4]: #perturb rotation
                new_cell.rotation = cell.rotation + np.random.normal(rotation_mu, rotation_sigma)
                
            #if simulation_config["image.type"] == "graySynthetic" and p[5] < p_decision[5]:
                #new_cell.opacity = cell.opacity + (np.random.normal(opacity_mu, opacity_sigma))

            # ensure that those changes fall within constraints
            valid = self.is_valid

            if not valid:
                badcount += 1

    @property
    def is_valid(self):
        return check_constraints(self.config, self.realimage.shape, [self.replacement_cell], self.get_checks())

    @property
    def costdiff(self):
        overlap_cost = self.config["overlap.cost"]
        new_synth = self.synthimage.copy()
        new_cellmap = self.cellmap.copy()
        region = self.node.cell.simulated_region(self.frame.simulation_config).\
            union(self.replacement_cell.simulated_region(self.frame.simulation_config))
        self.node.cell.draw(new_synth, new_cellmap, optimization.is_background, self.frame.simulation_config)
        self.replacement_cell.draw(new_synth, new_cellmap, optimization.is_cell, self.frame.simulation_config)

        if useDistanceObjective:
            start_cost = optimization.dist_objective(self.realimage[region.top:region.bottom, region.left:region.right],
                                                     self.synthimage[region.top:region.bottom, region.left:region.right],
                                                     self.distmap[region.top:region.bottom, region.left:region.right],
                                                     self.cellmap[region.top:region.bottom, region.left:region.right],
                                                     overlap_cost)
            end_cost = optimization.dist_objective(self.realimage[region.top:region.bottom, region.left:region.right],
                                                   new_synth[region.top:region.bottom, region.left:region.right],
                                                   self.distmap[region.top:region.bottom, region.left:region.right],
                                                   new_cellmap[region.top:region.bottom, region.left:region.right],
                                                   overlap_cost)
        else:
            start_cost = optimization.objective(self.realimage[region.top:region.bottom, region.left:region.right],
                                self.synthimage[region.top:region.bottom, region.left:region.right],
                                self.cellmap[region.top:region.bottom, region.left:region.right],
                                overlap_cost, self.config["cell.importance"])
            end_cost = optimization.objective(self.realimage[region.top:region.bottom, region.left:region.right],
                              new_synth[region.top:region.bottom, region.left:region.right],
                              new_cellmap[region.top:region.bottom, region.left:region.right],
                              overlap_cost, self.config["cell.importance"])

        return end_cost - start_cost + \
              0 * (np.sqrt((self.node.cell.x - self.replacement_cell.x)**2 + (self.node.cell.y - self.replacement_cell.y)**2) 
            + 0 * abs(self.node.cell.rotation - self.replacement_cell.rotation))

    def apply(self):
        self.node.cell.draw(self.synthimage, self.cellmap, optimization.is_background, self.frame.simulation_config)
        self.replacement_cell.draw(self.synthimage, self.cellmap, optimization.is_cell, self.frame.simulation_config)
        self.frame.add_cell(self.replacement_cell)

    def get_checks(self) -> List[Tuple[cell.Bacilli, cell.Bacilli]]:
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
            child.cell.draw(new_synth, new_cellmap, optimization.is_background, self.frame.simulation_config)

        self.combination.draw(new_synth, new_cellmap, optimization.is_cell, self.frame.simulation_config)

        if useDistanceObjective:
            start_cost = optimization.dist_objective(self.realimage[region.top:region.bottom, region.left:region.right],
                                                     self.synthimage[region.top:region.bottom, region.left:region.right],
                                                     self.distmap[region.top:region.bottom, region.left:region.right],
                                                     self.cellmap[region.top:region.bottom, region.left:region.right],
                                                     overlap_cost)
            end_cost = optimization.dist_objective(self.realimage[region.top:region.bottom, region.left:region.right],
                                                   new_synth[region.top:region.bottom, region.left:region.right],
                                                   self.distmap[region.top:region.bottom, region.left:region.right],
                                                   new_cellmap[region.top:region.bottom, region.left:region.right],
                                                   overlap_cost)
        else:
            start_cost = optimization.objective(self.realimage[region.top:region.bottom, region.left:region.right],
                                self.synthimage[region.top:region.bottom, region.left:region.right],
                                self.cellmap[region.top:region.bottom, region.left:region.right],
                                overlap_cost, self.config["cell.importance"])
            end_cost = optimization.objective(self.realimage[region.top:region.bottom, region.left:region.right],
                              new_synth[region.top:region.bottom, region.left:region.right],
                              new_cellmap[region.top:region.bottom, region.left:region.right],
                              overlap_cost, self.config["cell.importance"])

        return end_cost - start_cost + self.config["combine.cost"]

    def apply(self) -> None:
        self.combination.draw(self.synthimage, self.cellmap, optimization.is_cell, self.frame.simulation_config)
        grandchildren = self.node.grandchildren

        for child in self.node.children:
            del self.frame.node_map[child.cell.name]
            child.cell.draw(self.synthimage, self.cellmap, optimization.is_background, self.frame.simulation_config)

        self.node.children = []
        combination_node = self.node.make_child(self.combination)
        self.frame.node_map[self.combination.name] = combination_node

        for gc in grandchildren:
            combination_node.children.append(gc)
            gc.parent = combination_node


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
        self.node.children[0].cell.draw(new_synth, new_cellmap, optimization.is_background, self.frame.simulation_config)
        self.s1.draw(new_synth, new_cellmap, optimization.is_cell, self.frame.simulation_config)
        self.s2.draw(new_synth, new_cellmap, optimization.is_cell, self.frame.simulation_config)

        if useDistanceObjective:
            start_cost = optimization.dist_objective(self.realimage[region.top:region.bottom, region.left:region.right],
                                                     self.synthimage[region.top:region.bottom, region.left:region.right],
                                                     self.distmap[region.top:region.bottom, region.left:region.right],
                                                     self.cellmap[region.top:region.bottom, region.left:region.right],
                                                     overlap_cost)
            end_cost = optimization.dist_objective(self.realimage[region.top:region.bottom, region.left:region.right],
                                                   new_synth[region.top:region.bottom, region.left:region.right],
                                                   self.distmap[region.top:region.bottom, region.left:region.right],
                                                   new_cellmap[region.top:region.bottom, region.left:region.right],
                                                   overlap_cost)
        else:
            start_cost = optimization.objective(self.realimage[region.top:region.bottom, region.left:region.right],
                                self.synthimage[region.top:region.bottom, region.left:region.right],
                                self.cellmap[region.top:region.bottom, region.left:region.right],
                                overlap_cost, self.config["cell.importance"])
            end_cost = optimization.objective(self.realimage[region.top:region.bottom, region.left:region.right],
                              new_synth[region.top:region.bottom, region.left:region.right],
                              new_cellmap[region.top:region.bottom, region.left:region.right],
                              overlap_cost, self.config["cell.importance"])

        return end_cost - start_cost + self.config["split.cost"]

    def apply(self) -> None:
        self.node.children[0].cell.draw(self.synthimage, self.cellmap, optimization.is_background, self.frame.simulation_config)
        self.s1.draw(self.synthimage, self.cellmap, optimization.is_cell, self.frame.simulation_config)
        self.s2.draw(self.synthimage, self.cellmap, optimization.is_cell, self.frame.simulation_config)
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

class Opacity_Diffraction_offset(Change):
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
        
        self.new_simulation_config["cell.opacity"] += random.gauss(mu=opacity_offset_mu, sigma=opacity_offset_sigma)
        self.new_simulation_config["light.diffraction.strength"] += random.gauss(mu = diffraction_strength_offset_mu, sigma = diffraction_strength_offset_sigma)
        self.new_simulation_config["light.diffraction.sigma"] += random.gauss(mu = diffraction_sigma_offset_mu, sigma = diffraction_sigma_offset_sigma)
        self.new_synthimage, _ = optimization.generate_synthetic_image(frame.nodes, realimage.shape, self.new_simulation_config)
        
    @property
    def is_valid(self) -> bool:
        return self.new_simulation_config["cell.opacity"] >= 0 and \
        self.new_simulation_config["light.diffraction.strength"] >= 0 and \
        self.new_simulation_config["light.diffraction.sigma"] >= 0
    @property
    def costdiff(self) -> float:
        overlap_cost = self.config["overlap.cost"]
        if useDistanceObjective:
            start_cost = optimization.dist_objective(self.realimage,self.old_synthimage,self.distmap,self.cellmap,overlap_cost)
            end_cost = optimization.dist_objective(self.realimage,self.new_synthimage,self.distmap,self.cellmap,overlap_cost)
        else:
            start_cost = optimization.objective(self.realimage,self.old_synthimage,self.cellmap,overlap_cost, self.config["cell.importance"])
            end_cost = optimization.objective(self.realimage,self.new_synthimage,self.cellmap,overlap_cost, self.config["cell.importance"])
        return end_cost-start_cost
    
    def apply(self):
        self.old_synthimage[:]= self.new_synthimage
        self.old_simulation_config["cell.opacity"] = self.new_simulation_config["cell.opacity"]
        self.old_simulation_config["light.diffraction.strength"] = self.new_simulation_config["light.diffraction.strength"]
        self.old_simulation_config["light.diffraction.sigma"] = self.new_simulation_config["light.diffraction.sigma"]

class BackGround_luminosity_offset(Change):
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
        self.new_synthimage, _ = optimization.generate_synthetic_image(frame.nodes, realimage.shape, self.new_simulation_config)
        
    @property
    def is_valid(self) -> bool:
        return self.frame.simulation_config["background.color"] > 0
    @property
    def costdiff(self) -> float:
        overlap_cost = self.config["overlap.cost"]
        if useDistanceObjective:
            start_cost = optimization.dist_objective(self.realimage,self.old_synthimage,self.distmap,self.cellmap,overlap_cost)
            end_cost = optimization.dist_objective(self.realimage,self.new_synthimage,self.distmap,self.cellmap,overlap_cost)
        else:
            start_cost = optimization.objective(self.realimage,self.old_synthimage,self.cellmap,overlap_cost, self.config["cell.importance"])
            end_cost = optimization.objective(self.realimage,self.new_synthimage,self.cellmap,overlap_cost, self.config["cell.importance"])
        return end_cost-start_cost
    
    def apply(self):
        self.old_synthimage[:]= self.new_synthimage
        self.old_simulation_config["background.color"] = self.new_simulation_config["background.color"]
        self.old_simulation_config["cell.color"] = self.new_simulation_config["cell.color"]
        
def save_lineage(filename, cellnodes: List[CellNodeM],  lineagefile):
        for node in cellnodes:
            properties = [filename, node.cell.name]
            properties.extend([
                str(node.cell.x),
                str(node.cell.y),
                str(node.cell.width),
                str(node.cell.length),
                str(node.cell.rotation),
                str(node.cell.split_alpha),
                str(node.cell.opacity)])
            print(','.join(properties), file=lineagefile)

def build_initial_lineage(imagefiles, lineagefile, continue_from, simulation_config):
    #create a lineage given the path of the lineagefile and requiring imagefiles. The output lineage will contain frame number up to the continue_from(exclude)
    cells_data = pd.read_csv(lineagefile)
    cells_data = cells_data.replace('None', None)
    lineage = LineageM(simulation_config)
    for i in range(len(imagefiles)):
        filename = imagefiles[i].name
        #this is some what a ugly way to find out frame number contained in a string. Should be improved later?
        current_frame_number = int(filename.split('.')[0][-3:])
        if current_frame_number > continue_from - 1:
            break
        if i > 0:
            lineage.forward()
        for index, row in cells_data[cells_data["file"]==filename].iterrows():
            acell = cell.Bacilli(row["name"], row["x"], row["y"], row["width"], row["length"], row["rotation"], row["split_alpha"], row["opacity"])
            lineage.frames[-1].add_cell(acell) 
    return lineage

def find_optimal_simulation_confs(imagefiles, lineage, realimages, up_to_frame):
    for i in range(len(imagefiles)):
        filename = imagefiles[i].name
        current_frame_number = int(filename.split('.')[0][-3:])
        if current_frame_number >  up_to_frame - 1:
            break
        lineage.frames[i].simulation_config = optimization.find_optimal_simulation_conf(lineage.frames[i].simulation_config, realimages[i], lineage.frames[i].nodes)
    return lineage

def save_output(image_name, synthimage, realimage, cellnodes, args, config):
    residual_vmin = config["residual.vmin"]
    residual_vmax = config["residual.vmax"]
    if args.residual:
        colormap = cm.ScalarMappable(norm = Normalize(vmin=residual_vmin, vmax=residual_vmax), cmap = "bwr")
    bestfit_frame = Image.fromarray(np.uint8(255*synthimage), "L")
    bestfit_frame.save(args.bestfit / image_name)
    shape = realimage.shape
    output_frame = np.empty((shape[0], shape[1], 3))
    output_frame[..., 0] = realimage
    output_frame[..., 1] = output_frame[..., 0]
    output_frame[..., 2] = output_frame[..., 0]
    for node in cellnodes:
        node.cell.drawoutline(output_frame, (1, 0, 0))
    output_frame = Image.fromarray(np.uint8(255*output_frame))
    output_frame.save(args.output / image_name)
        
    if args.residual:
        residual_frame = Image.fromarray(np.uint8(255*colormap.to_rgba(np.clip(realimage - synthimage,
                                                                               residual_vmin, residual_vmax))), "RGB")
        residual_frame.save(args.residual / image_name)

def gerp(a, b, t):
    """Geometric interpolation"""
    return a * (b / a) ** t


def optimize(imagefiles, lineage, realimages, synthimages, cellmaps, distmaps, window_start, window_end, lineagefile, args, config,
             iteration_per_cell, in_auto_temp_schedule=False, const_temp=None):
    
    global totalCostDiff
        
    if in_auto_temp_schedule:
        lineage = deepcopy(lineage)
        synthimages = deepcopy(synthimages)
        distmaps = deepcopy(distmaps)
        cellmaps = deepcopy(cellmaps)
        
    debugfile = None
    if args.debug:
        debugfile = open(args.debug/'debug.csv', 'a')
        
    pbad_total = 0
    circular_buffer_capacity = config["pbad_max_size"]
    circular_buffer = np.empty(circular_buffer_capacity, float)
    circular_buffer_cursor = 0
    
    perturbation_prob = config["prob.perturbation"]
    combine_prob = config["prob.combine"]
    split_prob = config["prob.split"]
    background_offset_prob = config["prob.background_offset"]
    opacity_diffraction_offset_prob = config["prob.opacity_diffraction_offset"]
    #block_comb_remaining = 0
    window = window_end - window_start
    
    # simulated annealing
    total_iterations = iteration_per_cell*lineage.count_cells_in(window_start, window_end)//window
    bad_count = 0
    current_iteration = 1
    while current_iteration < total_iterations:
        frame_index = lineage.choose_random_frame_index(window_start, window_end)
        if in_auto_temp_schedule:
            temperature = const_temp
        else:
            frame_start_temp = gerp(args.end_temp, args.start_temp, (frame_index - window_start + 1)/window)
            frame_end_temp = gerp(args.end_temp, args.start_temp, (frame_index - window_start)/window)
            temperature = gerp(frame_start_temp, frame_end_temp, current_iteration/(total_iterations))                
            updated_iterations = iteration_per_cell * lineage.count_cells_in(window_start, window_end) // window
            if (updated_iterations > total_iterations): total_iterations = updated_iterations
        frame = lineage.frames[frame_index]
        node = np.random.choice(frame.nodes)
        change_option = np.random.choice(["split", "perturbation", "combine", "background_offset", "opacity_diffraction_offset"], 
                                         p=[split_prob, perturbation_prob, combine_prob, background_offset_prob, opacity_diffraction_offset_prob])
        change = None
        if change_option == "split" and np.random.random_sample() < optimization.split_proba(node.cell.length) and frame_index > 0:
            #print("split")
            change = Split(node.parent, config, realimages[frame_index], synthimages[frame_index], cellmaps[frame_index], lineage.frames[frame_index], distmaps[frame_index])
            
        elif change_option == "perturbation":
            change = Perturbation(node, config, realimages[frame_index], synthimages[frame_index], cellmaps[frame_index], lineage.frames[frame_index], distmaps[frame_index])
            
        elif change_option == "combine" and frame_index > 0: #and block_comb_remaining ==0:
            #print("combine")
            change = Combination(node.parent, config, realimages[frame_index], synthimages[frame_index], cellmaps[frame_index], lineage.frames[frame_index], distmaps[frame_index])

        elif change_option == "background_offset" and frame_index > 0 and config["simulation"]["image.type"] == "graySynthetic":
            #print("background")
            change = BackGround_luminosity_offset(lineage.frames[frame_index], realimages[frame_index], synthimages[frame_index], cellmaps[frame_index], config)
        
        elif change_option == "opacity_diffraction_offset" and frame_index > 0 and config["simulation"]["image.type"] == "graySynthetic":
            #print("opacity change")
            change = Opacity_Diffraction_offset(lineage.frames[frame_index], realimages[frame_index], synthimages[frame_index], cellmaps[frame_index], config)
        if change and change.is_valid:
            # apply if acceptable
            costdiff = change.costdiff

            if costdiff <= 0:
                acceptance = 1.0
            else:
                acceptance = np.exp(-costdiff / temperature)
                pbad_total += acceptance
                if (bad_count >= circular_buffer_capacity):
                    pbad_total -= circular_buffer[circular_buffer_cursor]
                else:
                    bad_count += 1
                circular_buffer[circular_buffer_cursor] = acceptance
                circular_buffer_cursor = (circular_buffer_cursor + 1) % circular_buffer_capacity

            if acceptance > np.random.random_sample():
                totalCostDiff += costdiff
                change.apply()
                if type(change) == Split:
                    total_iterations = lineage.count_cells_in(window_start, window_end)

                #if type(change) == Combination:
                    #total_iterations -= iteration_per_cell

        if debugfile and not in_auto_temp_schedule:
            print("{},{},{},{},{},{},{},{}".format(window_start, window_end, pbad_total, bad_count, temperature, totalCostDiff, current_iteration, total_iterations), file=debugfile)
        current_iteration += 1
        #print(current_iteration, total_iterations)

    if in_auto_temp_schedule:
        print("pbad is ", pbad_total/bad_count)
        return pbad_total/bad_count

    if debugfile:
        debugfile.close()

        #output module

def auto_temp_schedule(imagefiles, lineage, realimages, synthimages, cellmaps, distmaps, window_start, window_end, lineagefile, args, config):
    initial_temp = 1
    iteration_per_cell = config["auto_temp_scheduler.iteration_per_cell"]
    count=0
    
    while(optimize(imagefiles, lineage, realimages, synthimages, cellmaps, distmaps, window_start, window_end, lineagefile, args, config,
                   iteration_per_cell=iteration_per_cell, in_auto_temp_schedule=True, const_temp=initial_temp)<0.3):
        count += 1
        initial_temp *= 10.0
    #print("finished < 0.4")
    while(optimize(imagefiles, lineage, realimages, synthimages, cellmaps, distmaps, window_start, window_end, lineagefile, args, config,
                   iteration_per_cell=iteration_per_cell, in_auto_temp_schedule=True, const_temp=initial_temp)>0.3):
        count += 1 
        initial_temp /= 10.0
    #print("finished > 0.4")
    while(optimize(imagefiles, lineage, realimages, synthimages, cellmaps, distmaps, window_start, window_end, lineagefile, args, config,
                   iteration_per_cell=iteration_per_cell, in_auto_temp_schedule=True, const_temp=initial_temp)<0.3):
        count += 1
        initial_temp *= 1.1
    end_temp = initial_temp
    #print("finished < 0.4")
    while(optimize(imagefiles, lineage, realimages, synthimages, cellmaps, distmaps, window_start, window_end, lineagefile, args, config,
                   iteration_per_cell=iteration_per_cell, in_auto_temp_schedule=True, const_temp=end_temp)>1e-10):
        count += 1
        end_temp /= 10.0

    return initial_temp, end_temp
