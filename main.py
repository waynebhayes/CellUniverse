#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

"""
cellanneal
~~~~~~~~~~

The entry point for CellAnneal program.
"""

import cProfile
import argparse
import multiprocessing
from pathlib import Path
from optimization import auto_temp_schedule, auto_temp_schedule_frame, auto_temp_schedule_factor, auto_temp_schedule_const, find_optimal_simulation_conf, load_image


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
    parser.add_argument('-am', '--auto_meth', type=str, default='none', choices=('none', 'frame', 'factor', 'const'),
                        help='method for auto-temperature scheduling')

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


def load_colony(colony, initial_file, config):
    """Loads the initial colony of cells."""
    with open(initial_file, newline='') as fp:
        reader = csv.DictReader(fp, skipinitialspace=True)
        for row in reader:
            name = row['name']
            celltype = config['global.cellType'].lower()
            if celltype == 'bacilli':
                x = float(row['x'])
                y = float(row['y'])
                width = float(row['width'])
                length = float(row['length'])
                rotation = float(row['rotation'])
                cell = Bacilli(name, x, y, width, length, rotation)
            colony.add(CellNode(cell))


def get_inputfiles(args):
    """Gets the list of images that are to be analyzed."""
    inputfiles = []

    if args.frame_first < args.frame_first and args.frame_last >= 0:
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
            
        if args.debug:
            debugmode = True

        
        celltype = config['global.cellType'].lower()

        # setup the colony from a file with the initial properties
        lineageframes = LineageFrames()
        colony = lineageframes.forward()
        load_colony(colony, args.initial, config)

        # open the lineage file for writing
        lineagefile = open(args.output/'lineage.csv', 'w')
        header = ['file', 'name']
        if celltype == 'bacilli':
            header.extend(['x', 'y', 'width', 'length', 'rotation'])
        print(','.join(header), file=lineagefile)

        #optimze configuration
        config["simulation"] = find_optimal_simulation_conf(config["simulation"], load_image(get_inputfiles(args)[0]), list(colony))
        if args.global_optimization:
            import global_optimization
            global_optimization.optimize(get_inputfiles(args), lineageframes, lineagefile, args, config)
            return 0

        if args.auto_temp == 1:
            print("auto temperature schedule started")
            args.start_temp, args.end_temp = auto_temp_schedule(get_inputfiles(args)[0], lineageframes.forward(), args, config)
            print("auto temperature schedule finished")
            print("starting temperature is ", args.start_temp, "ending temperature is ", args.end_temp)

        frame_num = 0
        prev_cell_num = len(colony)
        for imagefile in get_inputfiles(args): # Recomputing temperature when needed

            frame_num += 1

            if args.auto_meth == "frame":
                if auto_temp_schedule_frame(frame_num, 8):
                    print("auto temperature schedule started (recomputed)")
                    args.start_temp, args.end_temp = auto_temp_schedule(imagefile, colony, args, config)
                    print("auto temperature schedule finished")
                    print("starting temperature is ", args.start_temp, "ending temperature is ", args.end_temp)

            elif args.auto_meth == "factor":
                if auto_temp_schedule_factor(len(colony), prev_cell_num, 1.1):
                    print("auto temperature schedule started (recomputed)")
                    args.start_temp, args.end_temp = auto_temp_schedule(imagefile, colony, args, config)
                    print("auto temperature schedule finished")
                    print("starting temperature is ", args.start_temp, "ending temperature is ", args.end_temp)
                    prev_cell_num = len(colony)

            elif args.auto_meth == "const":
                if auto_temp_schedule_const(len(colony), prev_cell_num, 10):
                    print("auto temperature schedule started (recomputed)")
                    args.start_temp, args.end_temp = auto_temp_schedule(imagefile, colony, args, config)
                    print("auto temperature schedule finished")
                    print("starting temperature is ", args.start_temp, "ending temperature is ", args.end_temp)
                    prev_cell_num = len(colony)


            colony = optimize(imagefile, lineageframes, args, config, client)

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

    import numpy as np
    from PIL import Image

    import jsonc
    from cell import Bacilli
    from colony import CellNode, Colony, LineageFrames
    from optimization import optimize
    from sys import exit
    # pr = cProfile.Profile()
    # pr.enable()
    print('CHECKPOINT, {}, {}, {}'.format(time.time(), -1, -1), flush=True)
    exit(main(args))
    # main(args)
    # pr.disable()
    # pr.dump_stats('./main.profile')
