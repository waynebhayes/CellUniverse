import numpy as np
from math import sqrt
from typing import List, Tuple

# Code taken from optimization.py
# -->

is_cell = True
is_background = False


def objective(realimage, synthimage, cellmap, overlap_cost, cell_importance):
    """Full objective function between two images."""
    overlap_map = cellmap[cellmap > 1] - 1
    return np.sum(np.square((realimage - synthimage))) \
        + overlap_cost * np.sum(np.square(overlap_map))


def dist_objective(realimage, synthimage, distmap, cellmap, overlap_cost):
    overlap_map = cellmap[cellmap > 1] - 1
    return np.sum(np.square((realimage - synthimage) * distmap)) + overlap_cost * np.sum(np.square(overlap_map))


def generate_synthetic_image(cellnodes, shape, simulation_config):
    image_type = simulation_config["image.type"]
    cellmap = np.zeros(shape, dtype=int)
    if image_type == "graySynthetic" or image_type == "phaseContrast":
        background_color = simulation_config["background.color"]
        synthimage = np.full(shape, background_color)
        for node in cellnodes:
            node.cell.draw(synthimage, cellmap, is_cell, simulation_config)
        return synthimage, cellmap
    else:
        synthimage = np.zeros(shape)
        for node in cellnodes:
            node.cell.draw(synthimage, cellmap, is_cell, simulation_config)
        return synthimage, cellmap

# <--


def check_constraints(config, imageshape, cells: List['Cell.Bacilli'], pairs: List[Tuple['cell.Bacilli', 'cell.Bacilli']] = None):
    max_displacement = config['sphere.maxSpeed'] / config['global.framesPerSecond']
    max_rotation = config['sphere.maxSpin'] / config['global.framesPerSecond']
    min_growth = config['sphere.minGrowth']
    max_growth = config['sphere.maxGrowth']
    min_width = config['sphere.minWidth']
    max_width = config['sphere.maxWidth']
    min_length = config['sphere.minLength']
    max_length = config['sphere.maxLength']

    for cell in cells:
        if not (0 <= cell.x < imageshape[1] and 0 <= cell.y < imageshape[0]):
            return False
        elif cell.width < min_width or cell.width > max_width:
            return False
        elif not (min_length < cell.length < max_length):
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
