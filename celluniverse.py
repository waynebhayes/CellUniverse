#!/usr/bin/env python
# Authors: Huy Pham, Emile Shehada, Shane Stahlheber
# Date: July 11, 2017
# Bacterial Growth Simulation Project

from __future__ import print_function

import argparse
import os
import sys
import time
from multiprocessing import Lock, Pool, cpu_count

import cv2
import numpy as np
from scipy import misc

from constants import Config, Globals
from helperMethods import (collision_matrix, deepcopy_list,
                           find_k_best_moves_mapped, generate_image_edge_cv2,
                           generate_universes, get_frames, improve_mapped,
                           init_space, process_init, write_state)

__version__ = "2.2"




#------------------
# Main
#------------------
def main():

    default_processes = max(cpu_count() // 2, 1)

    parser = argparse.ArgumentParser(description="Cell-Universe Cell Tracker.")
    parser.add_argument("-v", "--version", action="version",
                        version="%(prog)s {}".format(__version__))
    parser.add_argument("-s", "--start", type=int, default=0, metavar="FRAME",
                        help="start from specific frame (default: 0)")
    parser.add_argument("-p", "--processes", type=int, default=default_processes, metavar="COUNT",
                        help="number of concurrent processes to run (default: {})".format(default_processes))
    parser.add_argument("initial", type=argparse.FileType('r'),
                        help="initial properties file ('example.init.txt')")

    args = parser.parse_args()

    # get starting frame
    t = args.start

    # Image set from the real universe
    frames = get_frames('frames/', t)

    # Initialize the Space S
    S = init_space(t, args.initial)
    args.initial.close()

    # Creating directories
    image_dirs = []
    state_dirs = []
    for i in range(Config.K):
        image_dirs.append('Output/Images/' + str(i) + '/')
        if not os.path.exists(image_dirs[i]):
            os.makedirs(image_dirs[i])

        state_dirs.append('Output/States/' + str(i) + '/')
        if not os.path.exists(state_dirs[i]):
            os.makedirs(state_dirs[i])
            
    # Creating a pool of processes
    lock = Lock()
    pool = Pool(args.processes, initializer=process_init, initargs=(lock, Globals.image_width, Globals.image_height))

    # Processing
    for frame in frames:
        print("Processing frame {}...".format(t))
        sys.stdout.flush()

        start = time.time()
        frame_array = (cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) > 0)*np.int16(1)

        # generate the list of arguments to be passed to find_k_best_moves_mapped
        args = []
        for index, U in enumerate(S):

            # calculate U's matrix for collision detection
            M = collision_matrix(U)

            for bacterium_index in range(len(U)):
                args.append((deepcopy_list(U), frame_array, index, bacterium_index, len(S), M, start))

        # Find best moves for each bacterium in each universe
        moves_list = pool.map(find_k_best_moves_mapped, args)

        # initialize the best move lists
        best_moves = []
        for universe_index in range(len(S)):
            best_moves_per_universe = []
            for bacterium_index in range(len(S[universe_index])):
                best_moves_per_universe.append(None)
            best_moves.append(best_moves_per_universe)

        # organize the best move dictionary into a list
        for moves in moves_list:
            best_moves[moves[0]][moves[1]] = moves[2]

        # Generate future universes from S
        new_S = pool.map(generate_universes, [(deepcopy_list(U), frame_array, index, len(S), best_moves[index], start) for index, U in enumerate(S)])

        # unroll new_S in to a list of U's
        S = []
        for Us in new_S:
            S += Us
        
        # Pulling stage
        S.sort(key=lambda x: x[0])

        # improve the top 3K universes
        k3 = min(3*Config.K, len(S))
        S = pool.map(improve_mapped, [(S[i], frame_array, i, k3, start) for i in range(k3)])
        
        # pick the K best universes
        S.sort(key=lambda x: x[0])
        S = S[:Config.K]

        # Combine all best-match bacteria into 1 new universe
        # best_U = best_bacteria(S, frame_array)
        # S.append(best_U)

        # Output to files
        runtime = str(int(time.time() - start))
        for i, (c, U, index, _) in enumerate(S):
            new_frame = np.array(frame)
            file_name = image_dirs[i] + str(t) + ' - ' + str(c) + ' - ' + str(index) + ' - ' + runtime + '.png'
            generate_image_edge_cv2(U, new_frame)
            misc.imsave(file_name, new_frame)
            
            f = open(state_dirs[i] + str(t) + '.txt', 'w')
            write_state(index, U, f)
            f.close()

        S = [U for _, U, _, _ in S[:Config.K]]

        # next frame
        t += 1

    # finished
    parser.exit(0)


if __name__ == '__main__':
    main()
