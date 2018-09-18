#!/usr/bin/env python
# Authors: Huy Pham, Emile Shehada, Shane Stahlheber
# Date: July 11, 2017
# Bacterial Growth Simulation Project

from __future__ import print_function

import argparse
import os
import sys
import time
from multiprocessing import Lock, cpu_count

# To prevent crashing during a keyboard interrupt (must be before numpy/scipy)
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import cv2
import numpy as np
from scipy import misc

from constants import Config, Globals
from findconsistentpaths import create_consistent
from helperMethods import (collision_matrix, deepcopy_list, find_k_best_moves,
                           generate_image_edge_cv2, generate_universes,
                           get_frames, improve, init_space, process_init,
                           write_state)
from mphelper import InterruptablePool, kwargs


__version__ = "2.2"


def main():

    default_processes = max(cpu_count() // 2, 1)

    parser = argparse.ArgumentParser(description="Cell-Universe Cell Tracker.")

    parser.add_argument("-f", "--frames",
                        metavar="DIR",
                        type=str,
                        default='frames',
                        help="directory of the frames (default: 'frames')")

    parser.add_argument("-v", "--version",
                        action="version",
                        version="%(prog)s {}".format(__version__))

    parser.add_argument("-s", "--start",
                        metavar="FRAME",
                        type=int,
                        default=0,
                        help="start from specific frame (default: 0)")

    parser.add_argument("-p", "--processes",
                        metavar="COUNT",
                        type=int,
                        default=default_processes,
                        help="number of concurrent processes to run "
                             "(default: {})".format(default_processes))

    parser.add_argument("-o", "--output",
                        metavar="OUTDIR",
                        type=str,
                        default="Output",
                        help="directory of output states and images "
                             "(default: 'Output')")

    parser.add_argument("initial",
                        type=str,
                        help="initial properties file ('example.init.txt')")

    cli_args = parser.parse_args()

    # get starting frame
    t = cli_args.start
    frames_dir = cli_args.frames
    output_dir = cli_args.output
    frames = get_frames(frames_dir, t)

    # Initialize the Space S
    with open(cli_args.initial, 'r') as file:
        S = init_space(t, file)

    # Creating directories
    image_dirs = []
    state_dirs = []
    for i in range(Config.K):
        image_dirs.append(os.path.join(output_dir, Config.images_dir, str(i)))
        if not os.path.exists(image_dirs[i]):
            os.makedirs(image_dirs[i])

        state_dirs.append(os.path.join(output_dir, Config.states_dir, str(i)))
        if not os.path.exists(state_dirs[i]):
            os.makedirs(state_dirs[i])

    # Creating a pool of processes
    lock = Lock()
    pool = InterruptablePool(cli_args.processes,
                             initializer=process_init,
                             initargs=(lock,
                                       Globals.image_width,
                                       Globals.image_height,
                                       cli_args.initial))

    # Processing
    for frame in frames:
        print("Processing frame {}...".format(t))
        sys.stdout.flush()

        start = time.time()
        frame_array = (cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) > 0)*np.int16(1)

        # generate the list of arguments for find_k_best_moves_mapped
        args = []
        for index, U in enumerate(S):

            # calculate U's matrix for collision detection
            M = collision_matrix(U)

            for bacterium_index in range(len(U)):
                args.append(kwargs(
                    U=deepcopy_list(U),         # deep copy of universes
                    frame_array=frame_array,    # current image
                    index=index,                # index of universe
                    i=bacterium_index,          # index of bacterium
                    count=len(S),               # number of universes
                    M=M,                        # the collision matrix
                    start=start))               # start time for current frame

        # Find best moves for each bacterium in each universe
        moves_list = pool.map(find_k_best_moves, args)

        # initialize the best move lists
        best_moves = [[None for _ in universe] for universe in S]

        # organize the best move dictionary into a list
        for moves in moves_list:
            best_moves[moves[0]][moves[1]] = moves[2]

        # generate the list of arguments for generate_universes
        args = []
        for index, U in enumerate(S):
            args.append(kwargs(
                U=deepcopy_list(U),             # deepcopy of universes
                frame_array=frame_array,        # current image
                index=index,                    # index of universe
                count=len(S),                   # number of universes
                best_moves=best_moves[index],   # list of best moves
                start=start))                   # start time for current frame

        # Generate future universes from S
        new_S = pool.map(generate_universes, args)

        # flatten new_S in to a list of universes's (S)
        S = [universe for universe_list in new_S for universe in universe_list]

        # Pulling stage
        S.sort(key=lambda x: x[0])

        # improve the top 3K universes
        k3 = min(3*Config.K, len(S))
        args = []
        for index in range(k3):
            args.append(kwargs(
                Si=S[i],
                frame_array=frame_array,
                index=index,
                count=k3,
                start=start))
        S = pool.map(improve, args)

        # pick the K best universes
        S.sort(key=lambda x: x[0])
        S = S[:Config.K]

        # Combine all best-match bacteria into 1 new universe
        # best_U = best_bacteria(S, frame_array)
        # S.append(best_U)

        # Output to files
        # runtime = str(int(time.time() - start))
        for i, (c, U, index, _) in enumerate(S):
            new_frame = np.array(frame)
            # TODO: change the name of the output images and place this info
            #   somewhere else.
            # file_name = "{}{} - {} - {} - {}.png".format(image_dirs[i],
            #                                              str(t),
            #                                              str(c),
            #                                              str(index),
            #                                              runtime)
            image_filename = "{}.png".format(t)
            image_path = os.path.join(image_dirs[i], image_filename)
            generate_image_edge_cv2(U, new_frame)
            misc.imsave(image_path, new_frame)

            state_filename = "{}.txt".format(t)
            state_path = os.path.join(state_dirs[i], state_filename)
            write_state(state_path, index, U)

        S = [U for _, U, _, _ in S[:Config.K]]

        # next frame
        t += 1

    # make consistent
    print("Making consistent universes...")
    create_consistent(cli_args.start, t-1, output_dir)

    # finished
    print("Finished!")
    parser.exit(0)


if __name__ == '__main__':
    main()
