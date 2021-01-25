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
import global_optimization
from scipy.ndimage import distance_transform_edt
from colony import LineageFrames
import numpy as np

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
                        help = "input image is binary")
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
    parser.add_argument('--continue_from', metavar='N', type=int, default=1,
                        help="load already found orientation of cells and start from the continue_from frame")
    parser.add_argument('--seed', metavar='N', type=int, default=None, help='seed for random number generation')

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


def load_colony(colony, initial_file, config, initial_frame = None):
    """Loads the initial colony of cells."""
    with open(initial_file, newline='') as fp:
        reader = csv.DictReader(fp, skipinitialspace=True)
        for row in reader:
            if initial_frame != None and row['file'] != initial_frame:
                continue
            name = row['name']
            celltype = config['global.cellType'].lower()
            if celltype == 'bacilli':
                x = float(row['x'])
                y = float(row['y'])
                width = float(row['width'])
                length = float(row['length'])
                rotation = float(row['rotation'])
                if config["simulation"]["image.type"] == "graySynthetic":
                    opacity = config["simulation"]["cell.opacity"]
                    cell = Bacilli(name, x, y, width, length, rotation, opacity = opacity)
                else:
                    cell = Bacilli(name, x, y, width, length, rotation)
            colony.add(optimization.CellNode(cell))


def get_inputfiles(args):
    """Gets the list of images that are to be analyzed."""
    inputfiles = []

    if args.frame_first > args.frame_last and args.frame_last >= 0:
        raise ValueError('Invalid interval: frame_first must be less than frame_last')
    elif args.frame_first < 0:
        raise ValueError('Invalid interval: frame_first must be greater or equal to 0')

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


def main(args):
    """Main function of cellanneal."""
    if (args.start_temp is not None or args.end_temp is not None) and args.auto_temp == 1:
        raise Exception("when auto_temp is set to 1(default value), starting temperature or ending temperature should not be set manually")

    # if not args.no_parallel:
    #     import dask
    #     from dask.distributed import Client, LocalCluster
    #     if not args.cluster:
    #         cluster = LocalCluster(
    #             n_workers=args.workers,local_dir="/tmp/CellUniverse/dask-worker-space"
    #         )
    #     else:
    #         cluster = args.cluster
    #
    #     client = Client(cluster)
    # else:
    client = None

    lineagefile = None
    start = time.time()

    try:
        config = load_config(args.config)
        
        simulation_config = config["simulation"]
        #Maybe better to store the image type in the config file in the first place, instead of using cmd?
        if args.graySynthetic == True:
            simulation_config["image.type"] = "graySynthetic"
        elif args.phaseContrast == True:
            simulation_config["image.type"] = "phaseContrastImage"
        elif args.binary == True:
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
        if args.seed != None:
            seed = args.seed
        np.random.seed(seed)
        print("Seed: {}".format(seed))
        
        
        celltype = config['global.cellType'].lower()

        # setup the colony from a file with the initial properties
        lineageframes = LineageFrames()
        colony = lineageframes.forward()
        imagefiles =  get_inputfiles(args)
        if args.lineage_file:
            load_colony(colony, args.lineage_file, config, initial_frame = imagefiles[0].name)
        else:
            load_colony(colony, args.initial, config)
        cost_diff = (-1, -1)

        # open the lineage file for writing
        lineagefile = open(args.output/'lineage.csv', 'w')
        header = ['file', 'name']
        if celltype == 'bacilli':
            header.extend(['x', 'y', 'width', 'length', 'rotation', "split_alpha", "opacity"])
        print(','.join(header), file=lineagefile)

        if args.debug:
            with open(args.debug/'debug.csv', 'w') as debugfile:
                print(','.join(['window_start', 'window_end', 'pbad_total', 'bad_count', 'temperature', 'total_cost_diff', 'current_iteration', 'total_iterations']), file=debugfile)


        if args.global_optimization:
            global useDistanceObjective
            
            useDistanceObjective = args.dist
            realimages = [optimization.load_image(imagefile) for imagefile in imagefiles]
            window = config["global_optimizer.window_size"]
            if args.lineage_file:
                lineage = global_optimization.build_initial_lineage(imagefiles, args.lineage_file, args.continue_from, config["simulation"])
            else:
                lineage = global_optimization.build_initial_lineage(imagefiles, args.initial, args.continue_from, config["simulation"])
            lineage = global_optimization.find_optimal_simulation_confs(imagefiles, lineage, realimages, args.continue_from)
            sim_start = args.continue_from - args.frame_first
            shape = realimages[0].shape
            synthimages = []
            cellmaps = []
            distmaps = []
            iteration_per_cell = config["iteration_per_cell"]
            if not useDistanceObjective:
                distmaps = [None] * len(realimages)
            for window_start in range(1 - window, len(realimages)):
                 window_end = window_start + window
                 print(window_start, window_end)
                 if window_end <= len(realimages):
                        # get initial estimate
                    if window_start >= sim_start:
                        if window_end > 1:
                            lineage.copy_forward()
                    realimage = realimages[window_end - 1]
                    synthimage, cellmap = optimization.generate_synthetic_image(lineage.frames[window_end - 1].nodes, shape, lineage.frames[window_end - 1].simulation_config)
                    synthimages.append(synthimage)
                    cellmaps.append(cellmap)
                    if useDistanceObjective:
                        distmap = distance_transform_edt(realimage < .5)
                        distmap /= config[f'{config["global.cellType"].lower()}.distanceCostDivisor'] * config[
                            'global.pixelsPerMicron']
                        distmap += 1
                        distmaps.append(distmap)
                    if args.auto_temp == 1 and window_end == 1:
                        print("auto temperature schedule started")
                        args.start_temp, args.end_temp = \
                            global_optimization.auto_temp_schedule(imagefiles, lineage, realimages, synthimages, cellmaps, distmaps, 0, 1, lineagefile, args, config)
                        print("auto temperature schedule finished")
                        print("starting temperature is ", args.start_temp, "ending temperature is ", args.end_temp)
                    if args.auto_meth == "frame" and optimization.auto_temp_schedule_frame(window_end, 3):
                        print("auto temperature schedule restarted")
                        args.start_temp, args.end_temp = \
                            global_optimization.auto_temp_schedule(imagefiles, lineage, realimages, synthimages, cellmaps, distmaps, window_start, window_end, lineagefile, args, config)
                        print("auto temperature schedule finished")
                        print("starting temperature is ", args.start_temp, "ending temperature is ", args.end_temp)
                 if window_start >= sim_start:
                    if useDistanceObjective:
                        global_optimization.totalCostDiff = optimization.dist_objective(realimage, synthimage, distmap, cellmap, config["overlap.cost"])
                    else:
                        global_optimization.totalCostDiff = optimization.objective(realimage, synthimage, cellmap, config["overlap.cost"], config["cell.importance"])
                    global_optimization.optimize(imagefiles, lineage, realimages, synthimages, cellmaps, distmaps, window_start, window_end, lineagefile, args, config, iteration_per_cell)
                 if window_start >= 0:
                    global_optimization.save_lineage(imagefiles[window_start].name, lineage.frames[window_start].nodes, lineagefile)
                    global_optimization.save_output(imagefiles[window_start].name, synthimages[window_start], realimages[window_start], lineage.frames[window_start].nodes, args, config)
            return 0
        
        config["simulation"] = optimization.find_optimal_simulation_conf(config["simulation"], optimization.load_image(imagefiles[0]), list(colony))
        if args.auto_temp == 1:
            print("auto temperature schedule started")
            args.start_temp, args.end_temp = optimization.auto_temp_schedule(imagefiles[0], lineageframes.forward(), args, config)
            print("auto temperature schedule finished")
            print("starting temperature is ", args.start_temp, "ending temperature is ", args.end_temp)

        frame_num = 0
        prev_cell_num = len(colony)
        for imagefile in imagefiles: # Recomputing temperature when needed

            frame_num += 1

            if args.auto_meth == "frame":
                if optimization.auto_temp_schedule_frame(frame_num, 8):
                    print("auto temperature schedule started (recomputed)")
                    args.start_temp, args.end_temp = optimization.auto_temp_schedule(imagefile, colony, args, config)
                    print("auto temperature schedule finished")
                    print("starting temperature is ", args.start_temp, "ending temperature is ", args.end_temp)

            elif args.auto_meth == "factor":
                if optimization.auto_temp_schedule_factor(len(colony), prev_cell_num, 1.1):
                    print("auto temperature schedule started (recomputed)")
                    args.start_temp, args.end_temp = optimization.auto_temp_schedule(imagefile, colony, args, config)
                    print("auto temperature schedule finished")
                    print("starting temperature is ", args.start_temp, "ending temperature is ", args.end_temp)
                    prev_cell_num = len(colony)

            elif args.auto_meth == "const":
                if optimization.auto_temp_schedule_const(len(colony), prev_cell_num, 10):
                    print("auto temperature schedule started (recomputed)")
                    args.start_temp, args.end_temp = optimization.auto_temp_schedule(imagefile, colony, args, config)
                    print("auto temperature schedule finished")
                    print("starting temperature is ", args.start_temp, "ending temperature is ", args.end_temp)
                    prev_cell_num = len(colony)

            elif args.auto_meth == "cost":
                print(cost_diff, frame_num, optimization.auto_temp_shcedule_cost(cost_diff))
                if frame_num >= 2 and optimization.auto_temp_shcedule_cost(cost_diff):
                    print("auto temperature schedule started cost_diff (recomputed)")
                    args.start_temp, args.end_temp = optimization.auto_temp_schedule(imagefile, colony, args, config)
                    print("auto temperature schedule finished")
                    print("starting temperature is ", args.start_temp, "ending temperature is ", args.end_temp)

            colony = optimize(imagefile, lineageframes, args, config, client)

            cost_diff = optimization.update_cost_diff(colony, cost_diff)

            # flatten modifications and save cell properties

            colony.flatten()
            for cellnode in colony:
                properties = [imagefile.name, cellnode.cell.name]
                if celltype == 'bacilli':                                   
                    properties.extend([
                        str(cellnode.cell.x),
                        str(cellnode.cell.y),
                        str(cellnode.cell.width),
                        str(cellnode.cell.length),
                        str(cellnode.cell.rotation)])
                print(','.join(properties), file=lineagefile)

    except KeyboardInterrupt as error:
        raise error
    finally:
        if lineagefile:
            lineagefile.close()

    print(f'{time.time() - start} seconds')

    return 0


if __name__ == '__main__':
    args = parse_args()

    import csv
    import time
    from itertools import count

    import jsonc
    from cell import Bacilli
    from optimization import optimize
    from sys import exit
    # pr = cProfile.Profile()
    # pr.enable()
    print('CHECKPOINT, {}, {}, {}'.format(time.time(), -1, -1), flush=True)
    exit(main(args))
    # main(args)
    # pr.disable()
    # pr.dump_stats('./main.profile')
