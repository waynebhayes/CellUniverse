import random
from copy import deepcopy
from math import sqrt
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image

import cell
import optimization


grey_synthetic_image = False


# noinspection DuplicatedCode
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
            return False
        elif cell.width < min_width or cell.width > max_width:
            return False
        elif not (min_length < cell.length < max_length):
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
    def __init__(self, prev: 'FrameM' = None):
        self.node_map: Dict[str, CellNodeM] = {}
        self.prev = prev

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
    def __init__(self):
        self.frames = [FrameM()]

    def __repr__(self):
        return '\n'.join([str(frame) for frame in self.frames])

    @property
    def total_cell_count(self):
        return sum(len(frame.node_map) for frame in self.frames)

    def forward(self):
        self.frames.append(FrameM(self.frames[-1]))

    def choose_random_frame_index(self) -> int:
        threshold = int(random.random()*self.total_cell_count)

        for i, frame in enumerate(self.frames):
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
    def __init__(self, node: CellNodeM, config: Dict[str, Any], diffimage):
        self.node = node
        self.diffimage = diffimage
        self.config = config
        self._checks = []
        cell = node.cell
        new_cell = deepcopy(cell)
        self.replacement_cell = new_cell
        modified = False
        badcount = 0

        while not modified and badcount < 100:
            if random.random() < 0.35:
                new_cell.x = cell.x + random.gauss(mu=0, sigma=0.5)
                modified = True

            if random.random() < 0.35:
                new_cell.y = cell.y + random.gauss(mu=0, sigma=0.5)
                modified = True

            if random.random() < 0.1:
                new_cell.width = cell.width + random.gauss(mu=0, sigma=0.1)
                modified = True

            if random.random() < 0.2:
                new_cell.length = cell.length + random.gauss(mu=0, sigma=1)
                modified = True

            if random.random() < 0.2:
                new_cell.rotation = cell.rotation + random.gauss(mu=0, sigma=0.2)
                modified = True

            # ensure that those changes fall within constraints
            if modified:
                modified = self.is_valid

            badcount += 1

    @property
    def is_valid(self):
        return check_constraints(self.config, self.diffimage.shape, [self.replacement_cell], self.get_checks())

    @property
    def costdiff(self):
        new_diff = self.diffimage.copy()
        region = self.node.cell.region.union(self.replacement_cell.region)
        start_cost = np.sum(self.diffimage[region.top:region.bottom, region.left:region.right] ** 2)
        self.node.cell.draw(new_diff, True, grey_synthetic_image)
        self.replacement_cell.draw(new_diff, False, grey_synthetic_image)
        end_cost = np.sum(new_diff[region.top:region.bottom, region.left:region.right] ** 2)
        return end_cost - start_cost

    def apply(self):
        self.node.cell.draw(self.diffimage, True, grey_synthetic_image)
        self.replacement_cell.draw(self.diffimage, False, grey_synthetic_image)
        self.node.cell = self.replacement_cell

    def get_checks(self) -> List[Tuple[cell.Bacilli, cell.Bacilli]]:
        if not self._checks:
            if self.node.parent:
                if len(self.node.parent.children) == 1:
                    self._checks.append((self.node.parent.cell, self.replacement_cell))
                elif len(self.node.parent.children) == 2:
                    p1, p2 = self.node.parent.cell.split(.5)

                    if p1.name == self.replacement_cell.name:
                        self._checks.append((p1, self.replacement_cell))
                    elif p2.name == self.replacement_cell.name:
                        self._checks.append((p2, self.replacement_cell))

            if len(self.node.children) == 1:
                self._checks.append((self.replacement_cell, self.node.children[0].cell))
            elif len(self.node.children) == 2:
                p1, p2 = self.replacement_cell.split(.5)
                for c in self.node.children:
                    if c.cell.name == p1.name:
                        self._checks.append((p1, c.cell))
                    elif c.cell.name == p2.name:
                        self._checks.append((p2, c.cell))

        return self._checks


class Combination(Change):
    """Move split forward: o<8=8 -> o-o<8"""
    def __init__(self, node: CellNodeM, config, child_diffimage, child_frame: FrameM):
        self.node = node
        self.config = config
        self.diffimage = child_diffimage
        self.frame = child_frame
        self._checks = []
        self.combination = None

        if len(self.node.children) == 2:
            self.combination = self.node.children[0].cell.combine(self.node.children[1].cell)

    def get_checks(self):
        if self.combination and not self._checks:
            self._checks.append((self.node.cell, self.combination))
            p1, p2 = self.combination.split(0.5)

            for gc in self.node.grandchildren:
                if gc.cell.name == p1.name:
                    self._checks.append((p1, gc.cell))
                elif gc.cell.name == p2.name:
                    self._checks.append((p2, gc.cell))

        return self._checks

    @property
    def is_valid(self) -> bool:
        return len(self.node.children) == 2 and len(self.node.grandchildren) <= 2 and \
               check_constraints(self.config, self.diffimage.shape, [self.combination], self.get_checks())

    @property
    def costdiff(self) -> float:
        new_diff = self.diffimage.copy()
        region = self.combination.region

        for child in self.node.children:
            region = region.union(child.cell.region)

        start_cost = np.sum(self.diffimage[region.top:region.bottom, region.left:region.right] ** 2)

        for child in self.node.children:
            child.cell.draw(new_diff, True, grey_synthetic_image)

        self.combination.draw(new_diff, False, grey_synthetic_image)
        end_cost = np.sum(new_diff[region.top:region.bottom, region.left:region.right] ** 2)
        return end_cost - start_cost

    def apply(self) -> None:
        self.combination.draw(self.diffimage, False, grey_synthetic_image)
        grandchildren = self.node.grandchildren

        for child in self.node.children:
            del self.frame.node_map[child.cell.name]
            child.cell.draw(self.diffimage, True, grey_synthetic_image)

        self.node.children = []
        combination_node = self.node.make_child(self.combination)
        self.frame.node_map[self.combination.name] = combination_node

        for gc in grandchildren:
            combination_node.children.append(gc)
            gc.parent = combination_node


class Split(Change):
    """Move split backward: o-o<8 -> o<8=8"""
    def __init__(self, node: CellNodeM, config, child_diffimage, child_frame: FrameM):
        self.node = node
        self.config = config
        self.diffimage = child_diffimage
        self.frame = child_frame
        self._checks = []
        self.s1 = self.s2 = None

        if len(self.node.children) == 1:
            self.s1, self.s2 = self.node.children[0].cell.split(.5)

    def get_checks(self):
        if len(self.node.children) == 1 and not self._checks:
            p1, p2 = self.node.cell.split(.5)

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
               check_constraints(self.config, self.diffimage.shape, [self.s1, self.s2], self.get_checks())

    @property
    def costdiff(self) -> float:
        new_diff = self.diffimage.copy()
        region = self.node.children[0].cell.region.union(self.s1.region).union(self.s2.region)
        start_cost = np.sum(self.diffimage[region.top:region.bottom, region.left:region.right] ** 2)
        self.node.children[0].cell.draw(new_diff, True, grey_synthetic_image)
        self.s1.draw(new_diff, False, grey_synthetic_image)
        self.s2.draw(new_diff, False, grey_synthetic_image)
        end_cost = np.sum(new_diff[region.top:region.bottom, region.left:region.right] ** 2)
        return end_cost - start_cost

    def apply(self) -> None:
        self.node.children[0].cell.draw(self.diffimage, True, grey_synthetic_image)
        self.s1.draw(self.diffimage, False, grey_synthetic_image)
        self.s2.draw(self.diffimage, False, grey_synthetic_image)
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


def build_initial_lineage(imagefiles, lineageframes, args, config):
    lineage = LineageM()

    colony = lineageframes.latest
    for cellnode in colony:
        lineage.frames[0].add_cell(cellnode.cell)

    for imagefile in imagefiles[1:]:
        lineage.forward()
        colony, _, debugimage = optimization.optimize_core(imagefile, lineageframes.forward(), args, config, iterations_per_cell=500)
        colony.flatten()
        # debugimage.save(args.output / imagefile.name)

        for cellnode in colony:
            lineage.frames[-1].add_cell(cellnode.cell)
            properties = [
                imagefile.name, cellnode.cell.name,
                str(cellnode.cell.x),
                str(cellnode.cell.y),
                str(cellnode.cell.width),
                str(cellnode.cell.length),
                str(cellnode.cell.rotation)]
            # print(','.join(properties), file=lineagefile)

        # print(str(lineage))

    return lineage


def save_output(imagefiles, realimages, lineage: LineageM, args):
    shape = realimages[0].shape
    for frame_index in range(len(lineage.frames)):
        realimage = realimages[frame_index]
        cellnodes = lineage.frames[frame_index].nodes
        synthimage = optimization.generate_synthetic_image(cellnodes, realimage.shape, grey_synthetic_image)
        cost = optimization.objective(realimage, synthimage)
        print('Final Cost:', cost)

        frame = np.empty((shape[0], shape[1], 3))
        frame[..., 0] = realimage
        frame[..., 1] = frame[..., 0]
        frame[..., 2] = frame[..., 0]

        for node in cellnodes:
            node.cell.drawoutline(frame, (1, 0, 0))

        frame = np.clip(frame, 0, 1)

        debugimage = Image.fromarray((255 * frame).astype(np.uint8))
        debugimage.save(args.output / imagefiles[frame_index].name)


def optimize(imagefiles, lineageframes, lineagefile, args, config):
    # optimize normally and copy to my data structures
    lineage = build_initial_lineage(imagefiles, lineageframes, args, config)
    realimages = [optimization.load_image(imagefile) for imagefile in imagefiles]
    shape = realimages[0].shape
    diffimages = []

    for frame_index, realimage in enumerate(realimages):
        synthimage = optimization.generate_synthetic_image(lineage.frames[frame_index].nodes, shape, grey_synthetic_image)
        diffimages.append(realimage - synthimage)

    # simulated annealing
    run_count = 2000*lineage.total_cell_count
    temperature = args.temp
    end_temperature = args.endtemp
    alpha = (end_temperature/temperature)**(1/run_count)

    bad_count = 0
    bad_accepted = 0

    for iteration in range(run_count):
        frame_index = lineage.choose_random_frame_index()
        frame = lineage.frames[frame_index]
        node = random.choice(frame.nodes)
        change = None

        if frame_index < len(lineage.frames) - 1 and random.random() < 1/3:
            change = Combination(node, config, diffimages[frame_index + 1], lineage.frames[frame_index + 1])
            if not change.is_valid:
                change = None

        if not change and frame_index < len(lineage.frames) - 1 and random.random() < 2/3:
            change = Split(node, config, diffimages[frame_index + 1], lineage.frames[frame_index + 1])
            if not change.is_valid:
                change = None

        if not change:
            change = Perturbation(node, config, diffimages[frame_index])
            if not change.is_valid:
                continue

        # apply if acceptable
        costdiff = change.costdiff

        if costdiff <= 0:
            acceptance = 1.0
        else:
            bad_count += 1
            acceptance = np.exp(-costdiff / temperature)

        if acceptance > random.random():
            if acceptance < 1:
                bad_accepted += 1
            change.apply()

        if iteration % 1000 == 0 and bad_count > 0:
            print('pbad:', bad_accepted/bad_count)
            bad_count = bad_accepted = 0

        temperature *= alpha

    save_output(imagefiles, realimages, lineage, args)
