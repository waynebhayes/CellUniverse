#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

"""
cellanneal
~~~~~~~~~~

The entry point for CellAnneal program.
"""

import cProfile
import argparse
from pathlib import Path


def parse_args():
    """Reads and parses the command-line arguments."""
    parser = argparse.ArgumentParser()

    # optional arguments
    parser.add_argument('-d', '--debug', metavar='DIRECTORY', type=Path, default=None,
                        help='path to the debug directory (enables debug mode)')
    parser.add_argument('-s', '--start', metavar='N', type=int, default=0,
                        help='starting image (default: %(default)s)')
    parser.add_argument('-f', '--finish', metavar='N', type=int, default=-1,
                        help='final image (defaults to until last image)')
    parser.add_argument('--dist', action='store_true', default=False,
                        help='use distance-based objective function')

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
    required.add_argument('-t', '--temp', metavar='TEMP', type=float, required=True,
                          help='starting temperature for the simulated annealing')
    required.add_argument('-e', '--endtemp', metavar='TEMP', type=float, required=True,
                          help='ending temperature for the simulated annealing')

    return parser.parse_args()


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

    if args.finish < args.start and args.finish >= 0:
        raise ValueError('Invalid interval: start must be less than finish')
    elif args.start < 0:
        raise ValueError('Invalid interval: start must be greater or equal to 0')

    for i in count(args.start):
        # check to see if the file exists
        file = Path(args.input % i)
        if file.exists() and file.is_file():
            inputfiles.append(file)
            if i == args.finish:
                break
        elif args.finish < 0 and args.start != i:
            break
        else:
            raise ValueError(f'Input file not found "{file}"')

    return inputfiles


def main(args):
    """Main function of cellanneal."""

    lineagefile = None
    start = time.time()

    try:
        if args.debug:
            debugmode = True

        config = load_config(args.config)
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

        for imagefile in get_inputfiles(args):

            colony = optimize(imagefile, lineageframes, args, config)

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

    # pr = cProfile.Profile()
    # pr.enable()
    exit(main(args))
    # main(args)
    # pr.disable()
    # pr.dump_stats('./main.profile')