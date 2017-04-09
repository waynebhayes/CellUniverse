# Authors: Huy Pham, Emile Shehada, Shane Stahlheber
# Date: April 6, 2017
# Bacterial Growth Simulation Project

from __future__ import print_function

from helperMethods import *
from scipy import misc
from multiprocessing import Pool, Lock
import time
import sys

#------------------
# Main
#------------------
if __name__ == '__main__':
    # Initializations
    # Initialize the Space S
    t = 1
    S = init_space(t)

    # Image set from the real universe
    frames = get_frames('frames/', t)

    # Creating directories
    image_dirs = []
    state_dirs = []
    for i in range(K + 1):
        image_dirs.append('Output/Images/' + str(i) + '/')
        if not os.path.exists(image_dirs[i]):
            os.makedirs(image_dirs[i])

        state_dirs.append('Output/States/' + str(i) + '/')
        if not os.path.exists(state_dirs[i]):
            os.makedirs(state_dirs[i])

    debug_dir = "Output/Debug/"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    profile_dir = "Output/Profile"
    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir) 
            
    # Creating a pool of processes
    lock = Lock()
    pool = Pool(NUMBER_OF_PROCESSES, initializer=process_init, initargs=(lock, profile_dir))

    # Processing
    start_program = time.time()
    for frame in frames:
        print("Processing frame {}...".format(t))
        sys.stdout.flush()

        start = time.time()
        frame_array = (cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) > 0)*np.int16(1)

        # Generate future universes from S
        new_S = pool.map(generate_universes, [(deepcopy_list(U), frame_array, index, len(S), start, t) for index, U in enumerate(S)])

        # unroll new_S in to a list of U's
        S = []
        for Us in new_S:
            S += Us
        
        # Pulling stage
        S.sort(key=lambda x: x[0])

        # debug statistics: Which of the newly generated universes have the lowest cost?
        with open("{}prepulling {}.csv".format(debug_dir, t), "w") as fp:
            print("Old_Universe, New_Universe, Cost", file=fp)
            for s in S:
                print("{}, {}, {}".format(s[2], s[3], s[0]), file=fp)

        # improve the top 3K universes
        k3 = min(3*K, len(S))
        S = pool.map(improve_mapped_wrapper, [(S[i], frame_array, i, k3, start, t) for i in range(k3)])
        
        # pick the K best universes
        S.sort(key=lambda x: x[0])
        S = S[:K]

        # debug statistics: Which of the universes won?
        with open("{}postpulling {}.csv".format(debug_dir, t), "w") as fp:
            print("Old_Universe, New_Universe, Cost", file=fp)
            for s in S:
                print("{}, {}, {}".format(s[2], s[3], s[0]), file=fp)

        # Combine all best-match bacteria into 1 new universe
        best_U = best_bacteria(S, frame_array)
        S.append(best_U)

        # Output to files
        runtime = str(int(time.time() - start))
        for i, (c, U, index, _) in enumerate(S):
            new_frame = np.array(frame)
            file_name = image_dirs[i] + str(t) + ' - ' + str(c) + ' - ' + str(index) + ' - ' + runtime + '.png'
            generate_image_edge_cv2(U, new_frame)
            misc.imsave(file_name, new_frame)
            
            f = open(state_dirs[i] + str(t) + '.txt', 'w')
            write_state(U, f)
            f.close()

        S = [U for _, U, _, _ in S[:K+1]]

        # next frame
        t += 1

