#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

"""
cellannealer
~~~~~~~~~~~~
"""

import argparse
import csv
import logging
import re
from collections import namedtuple
from itertools import count
from pathlib import Path

import numpy as np
import yaml
from skimage.io import imread

import logconfig
from cell import Bacilli
from colony import Colony, LineageNode
from errorinfo import CellAnnealerError, ExitCode
from optimizer import simulated_annealing

logger = logging.getLogger(__name__)

# regex patterns
RE_DIV = re.compile(r'^---\s*$')

# structures
Options = namedtuple('Options', ['input_files', 'output_dir', 'config_file'])


def parse_args() -> argparse.Namespace:
    """Reads and parses command-line arguments."""

    parser = argparse.ArgumentParser()

    # Optional options
    parser.add_argument(
        '-v', '--verbose', action='count',
        default=0,
        help='give more output (additive)')
    parser.add_argument(
        '-q', '--quiet', action='count',
        default=0,
        help='give less output (additive)')
    parser.add_argument(
        '-l', '--log', metavar='FILE',
        type=str, default='',
        help='path to a verbose appending log')
    parser.add_argument(
        '-p', '--processes', metavar='N',
        type=int, default=1,
        help='number of extra processes allowed')
    parser.add_argument(
        '-s', '--start', metavar='N',
        type=int, default=0,
        help='starting image number (default: %(default)s)')
    parser.add_argument(
        '-f', '--finish', metavar='N',
        type=int, default=-1,
        help='final image number (default until last image')

    # Required options
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-i', '--input', metavar='PATTERN',
        type=str, required=True,
        help='input filename pattern (example: "image%%03d.png")')
    required.add_argument(
        '-o', '--output', metavar='DIRECTORY',
        type=Path, required=True,
        help='path to the output directory')
    required.add_argument(
        '-c', '--config', metavar='FILE',
        type=Path, required=True,
        help='filename of the configuration file')

    return parser.parse_args()


def get_options() -> Options:
    """Retrieve command-line arguments and options."""

    args = parse_args()

    # configure logger
    verbosity = args.verbose - args.quiet
    logconfig.configure(args.log, verbosity)


    # get a list of the input files
    input_files = []

    if args.finish < args.start and args.finish >= 0:
        raise CellAnnealerError(f'Invalid interval: start number must be less than finish number')

    if args.start < 0:
        raise CellAnnealerError(f'Invalid interval: start number must be greater than or equal to zero')

    for i in count(args.start):

        # check to see if the file exists
        file = Path(args.input % i)
        if file.exists() and file.is_file():
            input_files.append(file)

            # break out if reached finish option
            if i == args.finish:
                break
        elif args.finish < 0 and args.start != i:
            break
        else:
            raise CellAnnealerError(f'Input file not found: \'{file}\'')


    # ensure output directory exists
    if not (args.output.exists() and args.output.is_dir()):
        raise CellAnnealerError(f'Output directory not found: \'{args.output}\'')


    # ensure config file exists
    if not (args.config.exists() and args.config.is_file()):
        raise CellAnnealerError(f'Config file not found: \'{args.config}\'')

    return Options(input_files, args.output, args.config)


def load_config(filename: Path) -> (dict, Colony):
    """Loads the configuration file with cell table data."""

    with open(filename) as fd:

        # load yaml config
        yaml_text = ''
        while True:
            line = fd.readline()

            # check for unexpected EOF
            if not line:
                raise CellAnnealerError('Invalid config: missing cell table')

            # check for document divider
            if RE_DIV.match(line):
                break

            yaml_text += line

        config = yaml.load(yaml_text)


        # check for necessary configuration values
        if 'timestep' not in config:
            raise CellAnnealerError('Invalid config: missing timestep (example: \'timestep: 1\')')

        if 'delimiter' not in config:
            raise CellAnnealerError('Invalid config: missing delimiter (example: \'delimiter: " "\')')

        if 'cellType' not in config:
            raise CellAnnealerError('Invalid config: missing cell type (example: \'cellType: bacilli\'')

        # determine the cell type
        if config['cellType'] == 'bacilli':
            cell_type = Bacilli
        else:
            raise CellAnnealerError('Invalid config: unrecognized cell type')

        # confirm that the config file has the necessary info
        cell_type.check_config(config)

        # load the cell table
        cell_colony = Colony()
        reader = csv.DictReader(fd, delimiter=config['delimiter'], skipinitialspace=True)
        for row in reader:
            cell = cell_type(**row)
            node = LineageNode(cell)
            cell_colony.roots.append(node)

    return config, cell_colony


def main() -> int:
    """Starting point for cellannealer."""

    output_fd = None

    try:
        # get the command-line options and file configuration
        options = get_options()
        config, cell_colony = load_config(options.config_file)

        # write the output data header
        output_fd = open(options.output_dir/'results.csv', 'w')
        print('file,name,x,y,width,length,rotation', file=output_fd)

        # show config
        logger.info('Configuration:')
        for key, value in config.items():
            logger.info(f'{key}: {value}')
        logger.info('')

        # run the optimizer on all frames
        for i, filename in enumerate(options.input_files):
            real_image = imread(filename, as_gray=True).astype(np.float64)

            max_value = np.max(real_image)
            if max_value > 255:
                real_image /= 65535
            elif max_value > 1:
                real_image /= 255

            simulated_annealing(cell_colony, real_image, config, filename, offset=1000*i)

            cell_colony.flatten()

            # output the current cell colony
            for leaf in cell_colony.leaves:
                print(','.join(
                    map(str, [
                        filename,
                        leaf.cell.name,
                        leaf.cell.position.x,
                        leaf.cell.position.y,
                        leaf.cell.dimensions.width,
                        leaf.cell.dimensions.length,
                        leaf.cell.rotation
                    ])), file=output_fd)

    except CellAnnealerError as error:
        logger.error(error)
        return ExitCode.ERROR
    except Exception as error:
        logger.exception(error)
        return ExitCode.UNKNOWN
    except KeyboardInterrupt:
        return ExitCode.INTERRUPTED
    finally:
        if output_fd:
            output_fd.close()

    logger.info(f'Found {len(options.input_files)} files.')

    return ExitCode.SUCCESS


if __name__ == '__main__':
    exit(main())
