#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

"""
cellanneal
~~~~~~~~~~

The entry point for CellAnneal program.
"""
import argparse
import multiprocessing
from pathlib import Path
import optimization
from global_optimization import global_optimize, auto_temp_schedule
from scipy.ndimage import distance_transform_edt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image
from copy import deepcopy

from global_optimization.CellUniverse import CellUniverse
from global_optimization.Modules import LineageM
from lineage_funcs import create_lineage, save_lineage
from Cells.Bacilli import Bacilli
from Cells.Sphere import Sphere

import sys
sys.setrecursionlimit(10000)
# bootleg fix to prevent recursion error when pickling large lineages


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
    parser.add_argument('--no_parallel', action='store_true', default=False, help='disable parallelism')
    parser.add_argument('--global_optimization', action='store_true', default=False, help='global optimization')
    parser.add_argument('--binary', action='store_true', default=True,
                        help="input image is binary")
    parser.add_argument('--graySynthetic', action='store_true', default=False,
                        help='enables the use of the grayscale synthetic image for use with non-thresholded images')
    parser.add_argument('--phaseContrast', action='store_true', default=False,
                        help='enables the use of the grayscale synthetic image for phase contract images')
    parser.add_argument('-ta', '--auto_temp', metavar='TEMP', type=int, default=1,
                        help='auto temperature scheduling for the simulated annealing')
    parser.add_argument('-ts', '--start_temp', type=float, help='starting temperature for the simulated annealing')
    parser.add_argument('-te', '--end_temp', type=float, help='ending temperature for the simulated annealing')
    parser.add_argument('-am', '--auto_meth', type=str, default='none', choices=('none', 'frame', 'factor', 'const', 'cost'),
                        help='method for auto-temperature scheduling')
    parser.add_argument('-r', "--residual", metavar="FILE", type=Path, required=False,
                        help="path to the residual image output directory")
    parser.add_argument('--lineage_file', metavar='FILE', type=Path, required=False,
                        help='path to previous lineage file')
    parser.add_argument('--continue_from', metavar='N', type=int, default=0,
                        help="load already found orientation of cells and start from the continue_from frame")
    parser.add_argument('--seed', metavar='N', type=int, default=None, help='seed for random number generation')
    parser.add_argument('--batches', metavar='N', type=int, default=1, help='number of batches to split each frame into for multithreading')

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
    elif config['global.cellType'].lower() == 'sphere':
        celltype = Sphere
    else:
        raise ValueError('Invalid config: unsupported cell type')

    celltype.checkconfig(config)

    return config

def save_output(image_name, synthimage, realimage, cellnodes, args, config):
    residual_vmin = config["residual.vmin"]
    residual_vmax = config["residual.vmax"]
    if args.residual:
        colormap = cm.ScalarMappable(norm=Normalize(vmin=residual_vmin, vmax=residual_vmax), cmap="bwr")
    if 'image.slices' in config['simulation']:
        zs = range(-1 * (config['simulation']['image.slices'] // 2), config['simulation']['image.slices'] // 2 + 1)
        shape = realimage.shape
        for i, z in enumerate(zs):
            bestfit_frame = Image.fromarray(np.uint8(255 * synthimage[i]), "L")
            bestfit_frame.save(args.bestfit / (image_name[:-4] + '_' + str(z) + image_name[-4:]))
            output_frame = np.empty((shape[0], shape[1], 3))
            output_frame[..., 0] = realimage
            output_frame[..., 1] = output_frame[..., 0]
            output_frame[..., 2] = output_frame[..., 0]
            for node in cellnodes:
                if node.cell.dormant:
                    continue
                node.cell.drawoutline(output_frame, (1, 0, 0), z)
            output_frame = Image.fromarray(np.uint8(255 * output_frame))
            output_frame.save(args.output / (image_name[:-4] + '_' + str(z) + image_name[-4:] ))
            print('saved', str(z) + image_name)
    else:
        bestfit_frame = Image.fromarray(np.uint8(255 * synthimage), "L")
        bestfit_frame.save(args.bestfit / image_name)
        shape = realimage.shape
        output_frame = np.empty((shape[0], shape[1], 3))
        output_frame[..., 0] = realimage
        output_frame[..., 1] = output_frame[..., 0]
        output_frame[..., 2] = output_frame[..., 0]
        for node in cellnodes:
            if node.cell.dormant:
                continue
            node.cell.drawoutline(output_frame, (1, 0, 0))
        output_frame = Image.fromarray(np.uint8(255 * output_frame))
        output_frame.save(args.output / image_name)

    if args.residual:
        residual_frame = Image.fromarray(np.uint8(255 * colormap.to_rgba(np.clip(realimage - synthimage,
                                                                                 residual_vmin, residual_vmax))), "RGB")
        residual_frame.save(args.residual / image_name)


def main(args):
    """Main function of cellanneal."""

    test = CellUniverse(args)
    print(test.config)
    return 0

    if (args.start_temp is not None or args.end_temp is not None) and args.auto_temp == 1:
        raise Exception("when auto_temp is set to 1(default value), starting temperature or ending temperature should not be set manually")

    if not args.no_parallel:
        from dask.distributed import Client, LocalCluster
        if not args.cluster:
            cluster = LocalCluster(
                n_workers=args.workers, threads_per_worker=1,
            )
            client = Client(cluster)
        else:
            cluster = args.cluster
            client = Client(cluster)
            client.restart()
    else:
        client = None

    lineagefile = None
    start = time.time()

    # TODO: Change back to try/except
    if 1:
        config = load_config(args.config)

        simulation_config = config["simulation"]
        if args.graySynthetic:
            simulation_config["image.type"] = "graySynthetic"
        elif args.phaseContrast:
            simulation_config["image.type"] = "phaseContrastImage"
        elif args.binary:
            simulation_config["image.type"] = "binary"
        else:
            raise ValueError("Invalid Command: Synthetic image type must be specified")

        if not args.output.is_dir():
            args.output.mkdir()
        if not args.bestfit.is_dir():
            args.bestfit.mkdir()
        if args.residual and not args.residual.is_dir():
            args.residual.mkdir()

        seed = int(start * 1000) % (2**32)
        if args.seed is not None:
            seed = args.seed
        np.random.seed(seed)
        print("Seed: {}".format(seed))

        celltype = config['global.cellType'].lower()

        # open the lineage file for writing
        lineagefile = open(args.output / 'lineage.csv', 'w')
        header = ['file', 'name']
        if celltype == 'bacilli':
            header.extend(['x', 'y', 'width', 'length', 'rotation', "split_alpha", "opacity"])
        print(','.join(header), file=lineagefile)

        if args.debug:
            with open(args.debug / 'debug.csv', 'w') as debugfile:
                print(','.join(['window_start', 'window_end', 'pbad_total', 'bad_count', 'temperature', 'total_cost_diff', 'current_iteration', 'total_iterations']), file=debugfile)

        if args.global_optimization:
            global useDistanceObjective

            useDistanceObjective = args.dist
            # realimages = [optimization.load_image(imagefile) for imagefile in imagefiles]

            # setup the colony from a file with the initial properties
            lineage = LineageM(config, args)
            # lineage = create_lineage(imagefiles, realimages, config, args)

            window = config["global_optimizer.window_size"]
            sim_start = args.continue_from - args.frame_first
            print(sim_start)
            iteration_per_cell = config["iteration_per_cell"]
            for window_start in range(1 - window, len(lineage.real_images)):
                window_end = window_start + window
                print(window_start, window_end)
                if window_end <= len(lineage.real_images):
                    # get initial estimate
                    if window_start >= sim_start:
                        if window_end > 1:
                            lineage.copy_forward()
                    # if useDistanceObjective:
                    #     distmap = distance_transform_edt(realimage < .5)
                    #     distmap /= config[f'{config["global.cellType"].lower()}.distanceCostDivisor'] * config[
                    #         'global.pixelsPerMicron']
                    #     distmap += 1
                    # if args.auto_temp == 1 and window_end == 1:
                    #     print("auto temperature schedule started")
                    #     args.start_temp, args.end_temp = \
                    #         auto_temp_schedule(imagefiles, lineage, realimages, synthimages, cellmaps, distmaps, 0, 1, lineagefile, args, config)
                    #     print("auto temperature schedule finished")
                    #     print("starting temperature is ", args.start_temp, "ending temperature is ", args.end_temp)
                    # if args.auto_meth == "frame" and optimization.auto_temp_schedule_frame(window_end, 3):
                    #     print("auto temperature schedule restarted")
                    #     args.start_temp, args.end_temp = \
                    #         auto_temp_schedule(imagefiles, lineage, realimages, synthimages, cellmaps, distmaps, window_start, window_end, lineagefile, args, config)
                    #     print("auto temperature schedule finished")
                    #     print("starting temperature is ", args.start_temp, "ending temperature is ", args.end_temp)
                if window_start >= sim_start:
                    # if useDistanceObjective:
                    #     global_optimize.totalCostDiff = optimization.dist_objective(realimage, synthimage, distmap, cellmap, config["overlap.cost"])
                    # else:
                    #     global_optimize.totalCostDiff = optimization.objective(realimage, synthimage, cellmap, config["overlap.cost"], config["cell.importance"])
                    backup_lineage = deepcopy(lineage)
                    lineage = global_optimize(lineage, window_start, window_end, args, iteration_per_cell, client=client)
                    lineage.generate_synthetic_images(window_start)
                if window_start >= 0:
                    lineage.save_to_file(window_start, lineagefile)
                    lineage.save_output(window_start)
            return 0

        # local optimization
        # optimization.local_optimize(imagefiles, config, args, lineagefile, client)

    # except KeyboardInterrupt as error:
    #     raise error
    # finally:
    #     if lineagefile:
    #         lineagefile.close()

    print(f'{time.time() - start} seconds')
    if client and not cluster:
        client.shutdown()

    return 0


if __name__ == '__main__':
    args = parse_args()

    import csv
    import time
    from itertools import count

    import jsonc
    from Cells.Bacilli import Bacilli
    from sys import exit
    # pr = cProfile.Profile()
    # pr.enable()
    print('CHECKPOINT, {}, {}, {}'.format(time.time(), -1, -1), flush=True)
    # exit(main(args))
    main(args)
    # pr.disable()
    # pr.dump_stats('./main.profile')
