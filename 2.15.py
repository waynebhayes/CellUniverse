# Author: Huy Pham
# Secondary author: Emile Shehada
# Date: September 21, 2016
# Bacterial Growth Simulation Project

from helperMethods import *
from scipy import misc
from multiprocessing import Pool
import time
import pickle


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
            
        # Creating a pool of processes
    pool = Pool(NUMBER_OF_PROCESSES)


    # Processing
    start_program = time.time()
    for frame in frames:
        print t
        start = time.time()
        frame_array = (cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) > 0)*1

        # Generate future universes from S
        #generate_universes((S[0], frame_array, 0))
        new_S = pool.map(generate_universes, [(deepcopy_list(U), frame_array, index) for index, U in enumerate(S)])
        S = []
        for Us in new_S:    # unroll new_S in to a list of U's
            S += Us
        
        # Pulling stage
        S.sort(key=lambda x: x[0])
        k3 = min(3*K, len(S))
        S = pool.map(improve_mapped, [(S[i], frame_array) for i in range(k3)])
        S.sort(key=lambda x: x[0])
        S = S[:K]

        # Combine all best-match bacteria into 1 new universe
        best_U = best_bacteria(S, frame_array)
        S.append(best_U)

        # Output to files
        runtime = str(int(time.time() - start))
        for i, (c, U, index) in enumerate(S):
            new_frame = np.array(frame)
            file_name = image_dirs[i] + str(t) + ' - ' + str(c) + ' - ' + str(index) + ' - ' + runtime + '.png'
            generate_image_edge_cv2(U, new_frame)
            misc.imsave(file_name, new_frame)
            
            f = open(state_dirs[i] + str(t) + '.txt', 'w')
            write_state(U, f)
            f.close()

        S = [U for c, U, index in S[:K+1]]
        t += 1







        
