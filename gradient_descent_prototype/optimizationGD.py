"""
optimizationGD.py is the prototype of Cell Univserse with 2D gradient descent optimizer.
"""
import time
import argparse
import multiprocessing
from pathlib import Path
import optimization
from global_optimization import global_optimize, auto_temp_schedule
# from scipy.ndimage import distance_transform_edt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image
from lineage_funcs import create_lineage, save_lineage
import jsonc
import sys
from itertools import count
from cell import Bacilli
import matplotlib.pyplot as plt
from copy import deepcopy
import os
from scipy.ndimage import distance_transform_edt
from scipy.optimize import leastsq

FRAME = 0


def parse_args():
    """Reads and parses the command-line arguments."""
    parser = argparse.ArgumentParser()

    # optional arguments
    parser.add_argument('-d', '--debug', metavar='DIRECTORY', type=Path, default=None,
                        help='path to the debug directory (enables debug mode)')
    parser.add_argument('-ff', '--frame_first', metavar='N', type=int, default=0,
                        help='starting image (default: %(default)s)')
    parser.add_argument('-lf', '--frame_last', metavar='N', type=int, default=-1,
                        help='final image (defaults to until last image)')
    parser.add_argument('--dist', action='store_true', default=False,
                        help='use distance-based objective function')
    parser.add_argument('-w', '--workers', type=int, default=-1,
                        help='number of parallel workers (defaults to number of processors)')
    parser.add_argument('-j', '--jobs', type=int, default=-1,
                        help='number of jobs per frame (defaults to --workers/-w)')
    parser.add_argument('--keep', type=int, default=1,
                        help='number of top solutions kept (must be equal or less than --jobs/-j)')
    parser.add_argument('--strategy', type=str, default='best-wins',
                        help='one of "best-wins", "worst-wins", "extreme-wins"')
    parser.add_argument('--cluster', type=str, default='',
                        help='dask cluster address (defaults to local cluster)')
    parser.add_argument('--no_parallel', action='store_true',
                        default=False, help='disable parallelism')
    parser.add_argument('--global_optimization', action='store_true',
                        default=False, help='global optimization')
    parser.add_argument('--binary', action='store_true', default=True,
                        help="input image is binary")
    parser.add_argument('--graySynthetic', action='store_true', default=False,
                        help='enables the use of the grayscale synthetic image for use with non-thresholded images')
    parser.add_argument('--phaseContrast', action='store_true', default=False,
                        help='enables the use of the grayscale synthetic image for phase contract images')
    parser.add_argument('-ta', '--auto_temp', metavar='TEMP', type=int, default=1,
                        help='auto temperature scheduling for the simulated annealing')
    parser.add_argument('-ts', '--start_temp', type=float,
                        help='starting temperature for the simulated annealing')
    parser.add_argument('-te', '--end_temp', type=float,
                        help='ending temperature for the simulated annealing')
    parser.add_argument('-am', '--auto_meth', type=str, default='none', choices=('none', 'frame', 'factor', 'const', 'cost'),
                        help='method for auto-temperature scheduling')
    parser.add_argument('-r', "--residual", metavar="FILE", type=Path, required=False,
                        help="path to the residual image output directory")
    parser.add_argument('--lineage_file', metavar='FILE', type=Path, required=False,
                        help='path to previous lineage file')
    parser.add_argument('--continue_from', metavar='N', type=int, default=0,
                        help="load already found orientation of cells and start from the continue_from frame")
    parser.add_argument('--seed', metavar='N', type=int,
                        default=None, help='seed for random number generation')
    parser.add_argument('--batches', metavar='N', type=int, default=1,
                        help='number of batches to split each frame into for multithreading')

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--input', metavar='PATTERN', type=str, required=True,
                          help='input filename pattern (e.g. "image%%03d.png")')
    required.add_argument('-o', '--output', metavar='DIRECTORY', type=Path, required=True,
                          help='path to the output directory')
    required.add_argument('-c', '--config', metavar='FILE', type=Path, required=True,
                          help='path to the configuration file')
    required.add_argument('-x', '--initial', metavar='FILE', type=Path, required=True,
                          help='path to the initial cell configuration')
    required.add_argument('-b', "--bestfit", metavar="FILE", type=Path, required=True,
                          help="path to the best fit synthetic image output directory")

    parsed = parser.parse_args()

    if parsed.workers == -1:
        parsed.workers = multiprocessing.cpu_count()

    if parsed.jobs == -1:
        if parsed.cluster:
            raise ValueError('-j/--jobs is required for non-local clusters')
        else:
            parsed.jobs = parsed.workers

    return parsed


def load_config(config_file):
    """Loads the configuration file."""
    with open(config_file) as fp:
        config = jsonc.load(fp)

    if not isinstance(config, dict):
        raise ValueError('Invalid config: must be a dictionary')
    elif 'global.cellType' not in config:
        raise ValueError('Invalid config: missing "global.cellType"')
    elif 'global.pixelsPerMicron' not in config:
        raise ValueError('Invalid config: missing "global.pixelsPerMicron"')
    elif 'global.framesPerSecond' not in config:
        raise ValueError('Invalid config: missing "global.framesPerSecond"')

    if config['global.cellType'].lower() == 'bacilli':
        celltype = Bacilli
    else:
        raise ValueError('Invalid config: unsupported cell type')

    celltype.checkconfig(config)

    return config


def get_inputfiles(args):
    """Gets the list of images that are to be analyzed."""
    inputfiles = []

    if args.frame_first > args.frame_last and args.frame_last >= 0:
        raise ValueError(
            'Invalid interval: frame_first must be less than frame_last')
    elif args.frame_first < 0:
        raise ValueError(
            'Invalid interval: frame_first must be greater or equal to 0')

    for i in count(args.frame_first):
        # check to see if the file exists
        file = Path(args.input % i)
        if file.exists() and file.is_file():
            inputfiles.append(file)
            if i == args.frame_last:
                break
        elif args.frame_last < 0 and args.frame_first != i:
            break
        else:
            raise ValueError(f'Input file not found "{file}"')
    return inputfiles


def objective(realimage, synthimage):
    """Full objective function between two images."""
    return np.sum(np.square((realimage - synthimage)))


def get_loss(cellNodes, realimage, config):
    """Get the current loss function's value"""
    synthimage, _ = optimization.generate_synthetic_image(
        cellNodes, shape, config["simulation"])
    loss = objective(realimage, synthimage)
    return loss


def show_synth_real(cellNodes, realimage, i, config):
    """Generate and save the real and synthetic images"""
    synthimage, _ = optimization.generate_synthetic_image(
        cellNodes, shape, config["simulation"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(synthimage, cmap="gray")
    ax[1].imshow(realimage, cmap="gray")
    relative_path = os.path.join("gradient_descent_prototype", "video0",
                                 "real_synth")
    file_path = os.path.join(relative_path, "frame{:03d}.png".format(i))
    plt.savefig(file_path)
    plt.close(fig)


def show_bestfit(cellNodes, realimage, i, config):
    """Generate the best fit synthetic image of each frame"""
    synthimage, _ = optimization.generate_synthetic_image(
        cellNodes, shape, config["simulation"])
    bestfit_frame = Image.fromarray(np.uint8(255 * synthimage), "L")
    file_path = os.path.join("gradient_descent_prototype",
                             "video0", "output", "bestfit", "frame{:03d}.png".format(i))
    bestfit_frame.save(file_path)


def drawOutlines(cellNodes, realimage, i):
    """Draw the outlines of cells in synthetic image on the real images"""
    shape = realimage.shape
    output_frame = np.empty((shape[0], shape[1], 3))
    output_frame[..., 0] = realimage
    output_frame[..., 1] = output_frame[..., 0]
    output_frame[..., 2] = output_frame[..., 0]
    for node in cellNodes:
        node.cell.drawoutline(output_frame, (1, 0, 0))
    output_frame = Image.fromarray(np.uint8(255 * output_frame))
    file_path = os.path.join("gradient_descent_prototype", "video0",
                             "output", "frame{:03d}.png".format(i))
    output_frame.save(file_path)


def get_gradient(cellNodes, realimage, config):
    """Get the gradient of current cell parameters. The gradient is based on cell's x, y, rotation, width and length"""
    nodes_copy = deepcopy(cellNodes)
    cell_cnt = len(nodes_copy)
    gradient = np.array([[0]*5 for i in range(cell_cnt)], dtype=np.float64)

    delta = 1
    f0 = get_loss(nodes_copy, realimage, config)
    for i in range(cell_cnt):

        nodes_copy[i].cell.x += delta
        f1 = get_loss(nodes_copy, realimage, config)
        gradient[i][0] = (f1 - f0) / delta
        nodes_copy[i].cell.x = cellNodes[i].cell.x

        nodes_copy[i].cell.y += delta
        f1 = get_loss(nodes_copy, realimage, config)
        gradient[i][1] = (f1 - f0) / delta
        nodes_copy[i].cell.y = cellNodes[i].cell.y

        nodes_copy[i].cell.rotation += 0.1
        f1 = get_loss(nodes_copy, realimage, config)
        gradient[i][2] = (f1 - f0) / 0.1
        nodes_copy[i].cell.rotation = cellNodes[i].cell.rotation

        f1 = 1000
        d = 0.01
        for delta in [0.7, 1.3, 2]:
            nodes_copy[i].cell.length = cellNodes[i].cell.length + delta
            loss = get_loss(nodes_copy, realimage, config)
            if f1 > loss:
                f1 = loss
                d = delta
        gradient[i][3] = (f1 - f0) / d
        nodes_copy[i].cell.length = cellNodes[i].cell.length

        d = nodes_copy[i].cell.width * 0.2
        nodes_copy[i].cell.width *= 1.2
        f1 = get_loss(nodes_copy, realimage, config)
        gradient[i][4] = (f1 - f0) / d
        nodes_copy[i].cell.width = cellNodes[i].cell.width

    return gradient


def modify_cells(cellNodes, step):
    """Modify the cells' parameters using the step vector"""
    nodes_copy = deepcopy(cellNodes)
    cell_cnt = len(nodes_copy)
    for i in range(cell_cnt):
        nodes_copy[i].cell.x += step[i][0]
        nodes_copy[i].cell.y += step[i][1]
        nodes_copy[i].cell.rotation += step[i][2]
        nodes_copy[i].cell.length += step[i][3]
        nodes_copy[i].cell.width += step[i][4]
    return nodes_copy


def modify_cell(cellNodes, step, i):
    """Modify a particular cell i's parameter using the step vector"""
    nodes_copy = deepcopy(cellNodes)
    nodes_copy[i].cell.x += step[i][0]
    nodes_copy[i].cell.y += step[i][1]
    nodes_copy[i].cell.rotation += step[i][2]
    nodes_copy[i].cell.length += step[i][3]
    nodes_copy[i].cell.width += step[i][4]
    return nodes_copy


def get_derivative(cellNodes, realimage, a, direction, config):
    """Get the derivative of the loss function with a particular step size"""
    delta = 0.03
    f1 = get_loss(modify_cells(cellNodes, direction *
                  (a + delta)), realimage, config)
    f0 = get_loss(modify_cells(cellNodes, direction * a), realimage, config)
    return (f1 - f0)/delta


def secant_method(cellNodes, realimage, direction, config):
    """alternative method for finding the optimal step size"""
    a0 = 0.0
    a1 = 0.1
    while abs(a1 - a0) > 0.03:
        df0 = get_derivative(cellNodes, realimage, a0, direction, config)
        df1 = get_derivative(cellNodes, realimage, a1, direction, config)
        ddf = (df1 - df0) / (a1 - a0)
        a = a1 - df1/ddf
        a0 = a1
        a1 = a
    return a1


def backtrackLS(cellNodes, realimage, direction, config):
    """The backtrack line search method for finding a good step size"""
    cell_cnt = len(cellNodes)
    t = [2] * cell_cnt
    f0 = get_loss(cellNodes, realimage, config)
    beta = 0.8
    for i in range(cell_cnt):
        LHS = get_loss(modify_cell(
            cellNodes, t[i] * direction, i), realimage, config)
        RHS = f0 - t[i]/2*(np.linalg.norm(direction[i]))**2
        while LHS > RHS and t[i] > 0.05:
            t[i] = beta * t[i]
            LHS = get_loss(modify_cell(
                cellNodes, t[i] * direction, i), realimage, config)
            RHS = f0 - t[i]/2*(np.linalg.norm(direction[i]))**2
        direction[i] = direction[i] * t[i]


def normalize(v):
    """Normalize the vector v"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def find_optimal_simulation_conf(simulation_config, realimage1, cellnodes):
    shape = realimage1.shape

    def cost(values, target, simulation_config):
        for i in range(len(target)):
            simulation_config[target[i]] = values[i]
        synthimage, cellmap = optimization.generate_synthetic_image(
            cellnodes, shape, simulation_config)
        return (realimage1 - synthimage).flatten()

    initial_values = []
    variables = []

    variables.append("background.color")
    initial_values.append(simulation_config["background.color"])

    variables.append("cell.color")
    initial_values.append(simulation_config["cell.color"])

    variables.append("light.diffraction.sigma")
    initial_values.append(simulation_config["light.diffraction.sigma"])

    variables.append("light.diffraction.strength")
    initial_values.append(simulation_config["light.diffraction.strength"])

    auto_opacity = True
    variables.append("cell.opacity")
    initial_values.append(simulation_config["cell.opacity"])
    if len(variables) != 0:
        def residues(x): return cost(x, variables, simulation_config)
        optimal_values, _ = leastsq(residues, initial_values)

        for i, param in enumerate(variables):
            simulation_config[param] = optimal_values[i]
        simulation_config["cell.opacity"] = max(
            0, simulation_config["cell.opacity"])
        simulation_config["light.diffraction.sigma"] = max(
            0, simulation_config["light.diffraction.sigma"])

        if auto_opacity:
            for node in cellnodes:
                node.cell.opacity = simulation_config["cell.opacity"]

    print(simulation_config)
    return simulation_config


def gradient_descent(cellNodes, realimages, config):
    """The main function body of gradient descent optimizer"""
    for i in range(len(realimages)):
        global FRAME
        FRAME = i
        loss0 = get_loss(cellNodes, realimages[i], config)
        print("Step {}\nLoss before GD is:{:.5f}".format(i, loss0))
        epoch = 60
        while epoch > 0:
            gradient = get_gradient(cellNodes, realimages[i], config)
            direction = np.array([-1 * normalize(v) for v in gradient])
            backtrackLS(cellNodes, realimages[i], direction, config)
            cellNodes = modify_cells(cellNodes, direction)
            loss1 = get_loss(cellNodes, realimages[i], config)
            if abs(loss1 - loss0) < 0.001:
                break
            loss0 = loss1
            epoch -= 1
        drawOutlines(cellNodes, realimages[i], i)
        show_bestfit(cellNodes, realimages[i], i, config)
        print("Loss after GD is: {:.5f}".format(loss1))


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    simulation_config = config["simulation"]
    if args.graySynthetic:
        simulation_config["image.type"] = "graySynthetic"
    elif args.phaseContrast:
        simulation_config["image.type"] = "phaseContrastImage"
    elif args.binary:
        simulation_config["image.type"] = "binary"

    imagefiles = get_inputfiles(args)
    realimages = [optimization.load_image(
        imagefile) for imagefile in imagefiles]
    shape = realimages[0].shape
    lineage = create_lineage(imagefiles, realimages, config, args)
    config["simulation"] = lineage.frames[0].simulation_config
    cellNodes = lineage.frames[0].nodes

    gradient_descent(cellNodes, realimages, config)
