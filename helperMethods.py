import os
from Bacterium import *
import cv2
import collisionEventsHandler
import mahotas
import bisect
import time
import sys

# ran once per process in the pool
def process_init(l):
    global lock
    lock = l

# syncronized print function
def safe_print(msg):
    with lock:
        print(msg)
        sys.stdout.flush()

# Combine best-match bacteria across all U in S
#   into 1 new universe
#   return format: cost, U, index
def best_bacteria(S, frame_array):
    # find the greatest common ancestors
    # and put them in a map {name_of_bacterium -> cost}
    # and another map {name_of_bacterium -> bacterium}

    # Init the map with unique names of the greates ancestors
    name_to_cost = dict()
    for _, U, _, _ in S:
        for bacterium in U:
            name_to_cost[bacterium.name] = sys.maxsize

    # Keep only greatest ancestors
    to_remove = set()
    for name, c in name_to_cost.items():
        for i in range(1, len(name)):
            if name[:i] in name_to_cost:
                to_remove.add(name)
                break

    name_to_cost = {name: c for name, c in name_to_cost.items() if name not in to_remove}

    # Finding the minimum cost for each entry in the map
    name_to_bacteria = dict()
    for _, U, _, _ in S:
        for bacterium in U:
            # Find the bacterium's ancestor name in the 'names' map
            ancestor = bacterium.name
            for i in range(1,len(bacterium.name)):
                if bacterium.name[:i] in name_to_cost:
                    ancestor = bacterium.name[:i]
                    break

            # Find all descendants of the ancestor
            descendants = []
            for bacterium in U:
                if ancestor == bacterium.name[:len(ancestor)]:
                    descendants.append(bacterium)

            # Update the cost and the bacterium
            c = cost(frame_array, descendants)
            if c < name_to_cost[ancestor]:
                name_to_cost[ancestor] = c
                name_to_bacteria[ancestor] = descendants

    U = []
    for value in name_to_bacteria.values():
        U += value

    return (cost(frame_array, U), U, K, -1)


# find the best k moves mapped for parallel processing
def find_k_best_moves_mapped(args):

    U = args[0]
    frame_array = args[1]
    index = args[2]
    i = args[3]
    count = args[4]
    M = args[5]
    start = args[6]

    # Find best moves for each bacterium in U
    safe_print("[PID: {}] find_k_best_moves: Bacterium {} of {} in Universe {} of {} ({:.2f} sec)".format(os.getpid(), i + 1, len(U), index + 1, count, time.time() - start))

    bacterium = U[i]
    bacterium.v = np.zeros(2)
    bacterium.w = np.zeros(2)
    k_best_moves = []
    U_copy = deepcopy_list(U)

    for x in np.linspace(-3,3,7):
        for y in np.linspace(-3,3,7):
            for d_theta in np.linspace(-pi/10, pi/10, 21):
                for dh in np.linspace(0,3,4):

                    bacterium_copy = deepcopy(bacterium)

                    bacterium_copy.pos += np.array([x,y,0])
                    bacterium_copy.theta += d_theta
                    bacterium_copy.length += dh

                    bacterium_copy.update()
                    temp = U_copy[i]
                    U_copy[i] = bacterium_copy
                    collisionEventsHandler.run2(U_copy, i, M)
                    U_copy[i] = temp
                    
                    c = cost(frame_array, [bacterium_copy])
                    bisect.insort_left(k_best_moves, (c, x, y, d_theta, dh))

    # splitting
    to_insert = []
    for c,x,y,d_theta,dh in k_best_moves:
        bacterium_copy = deepcopy(bacterium)

        bacterium_copy.pos += np.array([x,y,0])
        bacterium_copy.theta += d_theta
        bacterium_copy.length += dh
        bacterium_copy.update()
        if bacterium_copy.length > 31:
            for split_ratio in np.linspace(0.25,0.75,20):
                bacterium_copy_2 = deepcopy(bacterium_copy)

                new_bacterium = split(bacterium_copy_2, split_ratio)
                if bacterium_copy_2.length < 13 or new_bacterium.length < 13:
                    continue

                c = cost(frame_array, [bacterium_copy_2, new_bacterium]) + 5
                to_insert.append((c, x, y, d_theta, dh, split_ratio))
    for e in to_insert:
        bisect.insort_left(k_best_moves, e)

    # result extracted from k_best_moves
    results = [e[1:] for e in k_best_moves]

    return (index, i, results)


# Generate future universes from an input universe
def generate_universes(args):

    U = args[0]
    frame_array = args[1]
    index = args[2]
    count = args[3]
    best_moves = args[4]
    start = args[5]

    M = collision_matrix(U) # calculate U's matrix for collision detection
    Us = [deepcopy_list(U) for i in range(4*K)]

    # Find best moves for each bacterium in U
    safe_print("[PID: {}] generate_universes: Universe {} of {} ({:.2f} sec)".format(os.getpid(), index + 1, count, time.time() - start))

    # Apply the best moves to all bacteria in Us
    for i in range(4*K):
        correspondence = []
        for j in range(len(U)):
            bacterium = Us[i][j]
            move = best_moves[j][i]
            
            bacterium.pos += np.array([move[0], move[1], 0])
            bacterium.theta += move[2]
            bacterium.length += move[3]
            bacterium.update()
            if len(move) == 5:
                Us[i].append(split(bacterium, move[4]))
                correspondence.append(j)

        # expand the collision matrix if bacteria have split
        new_M = M
        if len(correspondence) > 0:
            new_M = expand_collision_matrix(U, M, correspondence)

        collisionEventsHandler.run(Us[i], new_M)
        
    return [(cost(frame_array, Us[i]), Us[i], index, i) for i in xrange(len(Us))]


# parallelized improve() method in pulling stage
def improve_mapped(args):
    Si = args[0]
    frame_array = args[1]

    index = args[2]
    count = args[3]
    start = args[4]

    safe_print("[PID: {}] improve_mapped: Universe {} of {} ({:.02f} sec)".format(os.getpid(), index + 1, count, time.time() - start))

    w, U, index, index2 = Si
    improved = True
    M = collision_matrix(U)
    
    while improved:
        improved = False
        for j, bacterium in enumerate(U):

            # shrink
            for dh in [-4,-2, -1]:
                if U[j].length + dh < 10: continue
                U_copy = deepcopy_list(U)
                U_copy[j].length += dh
                U_copy[j].update()
                collisionEventsHandler.run(U_copy, M)
                s = cost(frame_array, U_copy)
                if s < w:
                    w = s
                    U = U_copy
                    Si = (s, U, index, index2)
                    improved = True
                    
            # horizontal
            for x in [-1,1]:
                U_copy = deepcopy_list(U)
                U_copy[j].pos[0] += x
                U_copy[j].update()
                collisionEventsHandler.run(U_copy, M)
                s = cost(frame_array, U_copy)
                if s < w:
                    w = s
                    U = U_copy
                    Si = (s, U, index, index2)
                    improved = True
                
            # vertical
            for y in [-1,1]:
                U_copy = deepcopy_list(U)
                U_copy[j].pos[1] += y
                U_copy[j].update()
                collisionEventsHandler.run(U_copy, M)
                s = cost(frame_array, U_copy)
                if s < w:
                    w = s
                    U = U_copy
                    Si = (s, U, index, index2)
                    improved = True

            # rotating
            for dtheta in [-pi/100, pi/100]:
                U_copy = deepcopy_list(U)
                U_copy[j].theta += dtheta
                U_copy[j].update()
                collisionEventsHandler.run(U_copy, M)
                s = cost(frame_array, U_copy)
                if s < w:
                    w = s
                    U = U_copy
                    Si = (s, U, index, index2)
                    improved = True

    return Si


# find greatest common ancestor name
def find_greatest_common_ancestor_name(bacterium, names):
    for i in range(1, len(bacterium.name)):
        if bacterium.name[:i] in names:
            return bacterium.name[:i]

    return bacterium.name

# find relatives
def find_relatives(U, bacterium, names):
    relatives = []
    ancestor_name = find_greatest_common_ancestor_name(bacterium, names)
    for a_bacterium in U:
        if ancestor_name in a_bacterium.name:
            relatives.append(a_bacterium)

    return relatives

# find states starting from time t
def find_states(t):
    file_names = []
    for root, dirs, files in os.walk('Input/states/'):
        for name in files:
            if int(name.split('.')[0]) == t - 1:
                file_names.append(root + '/' + name)

    return file_names

# Initial space S
def init_space(t):
    if t == 0:
        N = 4   # init number of bacteria
        U = []
        for i in range(N):
            U.append(Bacterium())

        # bacterium 1
        U[0].name = '00'
        U[0].pos = np.array([160, 105, 0])
        U[0].theta = pi/2 + pi/93
        U[0].length = 20
        
        # bacterium 2
        U[1].name = '01'
        U[1].pos = np.array([156, 125, 0])
        U[1].theta = pi/2 + pi/8 + pi/93
        U[1].length = 17
        
        # bacterium 3
        U[2].name = '10'
        U[2].pos = np.array([165, 130, 0])
        U[2].theta = pi/2 + pi/8 + pi/93
        U[2].length = 14

        # bacterium 4
        U[3].name = '11'
        U[3].pos = np.array([170, 113, 0])
        U[3].theta = pi/2 + pi/93
        U[3].length = 15


        for i in range(N):
            U[i].update()
            
        return [U]
    
    else:
        file_names = find_states(t)
        file_names.sort(key=lambda x: int(x.split('/')[2]))
        return [read_state(open(file_name)) for file_name in file_names]


# Advance bacterium, including moving and spinning
def advance(bacterium):
    bound_velocity(bacterium, MAX_SPEED, 2*pi)
    bacterium.pos += bacterium.v*dt
    bacterium.theta += bacterium.w.z*dt

    bacterium.update()

def bound_velocity(bacterium, max_v, max_w):
    bacterium.v[0] = min(max_v, bacterium.v[0])
    bacterium.v[1] = min(max_v, bacterium.v[1])
    bacterium.v[0] = max(-max_v, bacterium.v[0])
    bacterium.v[1] = max(-max_v, bacterium.v[1])

    bacterium.w.z = min(max_w, bacterium.w.z)
    bacterium.w.z = max(-max_w, bacterium.w.z)

# Return a copy of the input image
# with red pixels
def red(im):
    red_im = np.zeros((im.shape[0], im.shape[1], 3))
    red_im[:,:,1] = im

    return red_im



def get_frames(directory, t):
    frames = []
    file_names = []
    
    for f in os.listdir(directory):
        if not f.endswith('.png'): continue
        if int(f.split('.')[0]) < t: continue
        file_names.append(f)

    file_names = sorted(file_names, key=lambda f: int(f.split('.')[0]))

    for f in file_names:
        frame = cv2.imread(directory + f)
        frames.append(frame)

    return frames

def two_tuple(v):
    return tuple(list(v)[:2])



def generate_image_cv2(U, im=None):
    if im is None:
        im = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.int16)
    for bacterium in U:
        # head and tail
        cv2.circle(im, tuple(bacterium.head_pos[:2].astype(int)), bacterium.radius, 1, -1)
        cv2.circle(im, tuple(bacterium.tail_pos[:2].astype(int)), bacterium.radius, 1, -1)

        # body
        points = [tuple(bacterium.end_point_1[:2]), tuple(bacterium.end_point_2[:2]), tuple(bacterium.end_point_3[:2]), tuple(bacterium.end_point_4[:2])]
        points = np.array([(int(point[0]), int(point[1])) for point in points])
        cv2.fillConvexPoly(im, points, 1, 1)

    return im


def generate_image_edge(U, im=None):
    if im is None:
        im = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.int16)
    for bacterium in U:
        p1 = (int(bacterium.end_point_1[0]), int(bacterium.end_point_1[1]))
        p2 = (int(bacterium.end_point_2[0]), int(bacterium.end_point_2[1]))
        p3 = (int(bacterium.end_point_3[0]), int(bacterium.end_point_3[1]))
        p4 = (int(bacterium.end_point_4[0]), int(bacterium.end_point_4[1]))

        # body lines
        cv2.line(im, p1, p4, 1)
        cv2.line(im, p2, p3, 1)

        # head
        center = (int(bacterium.head_pos[0]), int(bacterium.head_pos[1]))
        axes = (bacterium.radius, bacterium.radius)
        cv2.ellipse(im, (center[0], center[1]), axes, bacterium.theta*180/3.14, 90, 270,1)
        
        # tail
        center = (int(bacterium.tail_pos[0]), int(bacterium.tail_pos[1]))
        axes = (bacterium.radius, bacterium.radius)
        cv2.ellipse(im, (center[0], center[1]), axes, bacterium.theta*180/3.14, -90, 90,1)

    return im

# This method is for drawing on the existing frame at the end of
#   each iteration in the simulation
def generate_image_edge_cv2(U, im):
    for bacterium in U:
        p1 = (int(bacterium.end_point_1[0]), int(bacterium.end_point_1[1]))
        p2 = (int(bacterium.end_point_2[0]), int(bacterium.end_point_2[1]))
        p3 = (int(bacterium.end_point_3[0]), int(bacterium.end_point_3[1]))
        p4 = (int(bacterium.end_point_4[0]), int(bacterium.end_point_4[1]))

        # body lines
        cv2.line(im, p1, p4, (255,0,0))
        cv2.line(im, p2, p3, (255,0,0))

        # head
        center = (int(bacterium.head_pos[0]), int(bacterium.head_pos[1]))
        axes = (bacterium.radius, bacterium.radius)
        cv2.ellipse(im, (center[0], center[1]), axes, bacterium.theta*180/3.14, 90, 270,255)
        
        # tail
        center = (int(bacterium.tail_pos[0]), int(bacterium.tail_pos[1]))
        axes = (bacterium.radius, bacterium.radius)
        cv2.ellipse(im, (center[0], center[1]), axes, bacterium.theta*180/3.14, -90, 90,255)
        


    return im
    
# Normalize a list of weights
def normalize(W):
    if len(W) < 2:
        return W
    W1 = []
    for e in W:
        W1.append((W[-1] - e)/(W[-1] - W[0]))
    s = sum(W1)
    return [x/s for x in W1]


# Calculate matrix from a bacteria set for collision detection
def collision_matrix(U):
    N = len(U)
    M = [[False for i in range(N)] for i in range(N)]

    for i in range(N):
        for j in range(i+1, N):
            if (np.linalg.norm(U[i].pos - U[j].pos) < U[i].length/2 + U[j].length/2 + 6):
                M[i][j] = True

    return M

        
def expand_collision_matrix(U, M, correspondence):
    """Expands the collision matrix after splits have occured.

    When a bacterium splits, their bodies overlap the same space that the 
    original bacterium covered (they don't spontaneously move far away).  
    Therefore, the entries of the collision matrix for the original bacterium
    corresponds to the entries of the two new bacteria (they collide with the
    same bacteria as the original did).  We can use this fact to save time by
    reusing the existing values in the collision matrix, since computing the
    distance between two bacterium is computationally more expensive.

    Args:
        U (Universe): The universe for which the collision matrix is made.
        M (list): The old collision matrix (list of lists of booleans).
        correspondence (list): A list of integers representing the bacteria
            index which corresponds to the other split bacterium.

    Returns:
        A boolean square matrix of length len(U) indicating if a pair of
        bacteria are in proximity to each other.
    """

    N = len(U)

    new_M = [[False for i in range(N)] for i in range(N)]

    # Copy over the existing collision matrix
    for i in xrange(N):
        for j in xrange(i + 1, N):

            # get the corresponding indices of the newly split bacteria
            ci = i if i < len(M) else correspondence[i - len(M)]
            cj = j if j < len(M) else correspondence[j - len(M)]

            # swap so that the smallest index is first
            if ci > cj: 
                ci, cj = cj, ci

            new_M[i][j] = M[ci][cj]

    return new_M


# deep copy function
#   returns a copy of a bacterium
def deepcopy(bacterium):
    new_bacterium = Bacterium()
    new_bacterium.pos = np.array(bacterium.pos)
    new_bacterium.theta = bacterium.theta
    new_bacterium.length = bacterium.length
    new_bacterium.bending = bacterium.bending
    new_bacterium.bend_ratio = bacterium.bend_ratio
    new_bacterium.bend_angle = bacterium.bend_angle
    new_bacterium.name = bacterium.name
    new_bacterium.update()
    return new_bacterium

def deepcopy_list(U):
    U_copy = []
    for bacterium in U:
        U_copy.append(deepcopy(bacterium))

    return U_copy

# Read from file f
#   return a set of Bacteria from the file
def read_state(f):
    U = []
    f.readline()
    for line in f:
        line = line.split(' ')
        bacterium = Bacterium()
        bacterium.name      = str(line[0])
        bacterium.pos[0]    = float(line[1])
        bacterium.pos[1]    = float(line[2])
        bacterium.length    = float(line[3])
        bacterium.theta     = float(line[4])

        bacterium.update()
        U.append(bacterium)

    return U

# Write the state of bacteria B to the file f
def write_state(U, f):
    f.write(str(len(U)) + '\n')
    for bacterium in sorted(U, key=lambda x: x.pos[0]):
        f.write(str(bacterium.name)  + ' ')
        f.write(str(bacterium.pos[0]) + ' ')
        f.write(str(bacterium.pos[1]) + ' ')
        f.write(str(bacterium.length)+ ' ')
        f.write(str(bacterium.theta) +'\n')

def write_state2(U, f):
    f.write(str(len(U)) + '\n')
    for bacterium in sorted(U, key=lambda x: x.pos[0]):
        f.write(str(bacterium.name)  + ' ')
        f.write(str(bacterium.index) + ' ')
        f.write(str(bacterium.pos[0]) + ' ')
        f.write(str(bacterium.pos[1]) + ' ')
        f.write(str(bacterium.length)+ ' ')
        f.write(str(bacterium.theta) +'\n')


# cost function
def cost(base_im_array, U, xform=None):
    im = generate_image_cv2(U)
    dif = cv2.absdiff(base_im_array, im)

    #xform = True
    if xform != None:
        xform = mahotas.distance(generate_image_edge(U) == 0)
        xform[xform > 5] = 5
        xform = xform + 1
        dif *= xform
    
    return int(cv2.sumElems(dif)[0])

# bacterium splits into 2 bacteria
def split(bacterium, split_ratio):
    # Positioning the first one
    L_0 = bacterium.length
    bacterium.length = split_ratio*L_0 - 2
    x_0 = bacterium.pos[0]
    y_0 = bacterium.pos[1]
    x = x_0 + (L_0-bacterium.length)*cos(bacterium.theta)/2
    y = y_0 + (L_0-bacterium.length)*sin(bacterium.theta)/2
    bacterium.pos = np.array([x,y,0])
    bacterium.name += '0'
    bacterium.bending = False
    bacterium.update()

    # Positioning the second one
    new_bacterium = Bacterium()
    x = x_0 - bacterium.length*cos(bacterium.theta)/2
    y = y_0 - bacterium.length*sin(bacterium.theta)/2
    new_bacterium.pos = np.array([x, y, 0])
    new_bacterium.theta = bacterium.theta
    new_bacterium.length = L_0 - bacterium.length - 2
    new_bacterium.name = bacterium.name[:-1] + '1'

    # Update the new bacterium
    new_bacterium.update()

    return new_bacterium


