from __future__ import print_function

import bisect
import os
import re
import shlex
import signal
import sys
import time
from collections import defaultdict
from math import cos, pi, sin

import cv2
import numpy as np

import collisionEventsHandler
from Bacterium import Bacterium
from constants import Config, Globals
from mphelper import doublestar_function, HandleExceptions

LINE_PATTERN = re.compile(r"""
        ^(?P<line>[^#]*)  # anything before a comment marker is the line
        (?:\#.*)?$        # ignore comments if they exist until the end of line
    """, re.VERBOSE)


# ran once per process in the pool
def process_init(l, width, height, initial):
    global lock, safe_print
    lock = l

    safe_print = _safe_print

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    HandleExceptions.set_print(safe_print)

    Globals.image_width = width
    Globals.image_height = height

    with lock:
        with open(initial, 'r') as file:
            init_space(0, file)


safe_print = print


# syncronized print function
def _safe_print(*args, **kwargs):
    with lock:
        print(*args, **kwargs)
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

    name_to_cost = {name: c
                    for name, c in name_to_cost.items()
                    if name not in to_remove}

    # Finding the minimum cost for each entry in the map
    name_to_bacteria = dict()
    for _, U, _, _ in S:
        for bacterium in U:
            # Find the bacterium's ancestor name in the 'names' map
            ancestor = bacterium.name
            for i in range(1, len(bacterium.name)):
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

    return (cost(frame_array, U), U, Config.K, -1)


# find the best k moves mapped for parallel processing
@doublestar_function
@HandleExceptions.decorate
def find_k_best_moves(U, frame_array, index, i, count, M, start):

    # display update
    details = {"process_id": os.getpid(),
               "bacterium_index": i + 1,
               "bacteria_count": len(U),
               "universe_index": index + 1,
               "universe_count": count,
               "current_runtime": time.time() - start}

    safe_print("[PID: {process_id}] find_k_best_moves: "
               "Bacterium {bacterium_index} of {bacteria_count} in "
               "Universe {universe_index} of {universe_count} "
               "({current_runtime:.2f} sec)".format(**details))

    # Find best moves for each bacterium in U
    bacterium = U[i]
    bacterium.v = np.zeros(2)
    bacterium.w = np.zeros(2)
    k_best_moves = []
    U_copy = deepcopy_list(U)

    dx_space = np.linspace(-Config.MAX_X_MOTION,
                           Config.MAX_X_MOTION,
                           Config.MAX_X_RESOLUTION)

    dy_space = np.linspace(-Config.MAX_Y_MOTION,
                           Config.MAX_Y_MOTION,
                           Config.MAX_Y_RESOLUTION)

    dtheta_space = np.linspace(-Config.MAX_ROTATION,
                               Config.MAX_ROTATION,
                               Config.MAX_ROTATION_RESOLUTION)

    dlength_space = np.linspace(Config.MIN_LENGTH_INCREASE,
                                Config.MAX_LENGTH_INCREASE,
                                Config.LENGTH_INCREASE_RESOLUTION)

    split_ratio_space = np.linspace(Config.SPLIT_RATIO_BEGINNING,
                                    Config.SPLIT_RATIO_END,
                                    Config.SPLIT_RATIO_RESOLUTION)

    for dx in dx_space:
        for dy in dy_space:
            for dtheta in dtheta_space:
                for dlength in dlength_space:

                    bacterium_copy = deepcopy(bacterium)

                    bacterium_copy.pos += np.array([dx, dy, 0])
                    bacterium_copy.theta += dtheta
                    bacterium_copy.length += dlength

                    bacterium_copy.update()
                    temp = U_copy[i]
                    U_copy[i] = bacterium_copy
                    collisionEventsHandler.run2(U_copy, i, M)
                    U_copy[i] = temp

                    c = cost(frame_array, [bacterium_copy])
                    bisect.insort_left(k_best_moves,
                                       (c, dx, dy, dtheta, dlength))

    # splitting
    to_insert = []
    for c, dx, dy, dtheta, dlength in k_best_moves:
        bacterium_copy = deepcopy(bacterium)

        bacterium_copy.pos += np.array([dx, dy, 0])
        bacterium_copy.theta += dtheta
        bacterium_copy.length += dlength
        bacterium_copy.update()

        if bacterium_copy.length > Config.MAX_LENGTH_BEFORE_SPLIT:
            for split_ratio in split_ratio_space:
                bacterium_copy_2 = deepcopy(bacterium_copy)

                new_bacterium = split(bacterium_copy_2, split_ratio)
                if (bacterium_copy_2.length < Config.MIN_LENGTH or
                        new_bacterium.length < Config.MIN_LENGTH):
                    continue

                c = cost(frame_array, [bacterium_copy_2, new_bacterium]) + 5
                to_insert.append((c, dx, dy, dtheta, dlength, split_ratio))

    for e in to_insert:
        bisect.insort_left(k_best_moves, e)

    # result extracted from k_best_moves
    results = [e[1:] for e in k_best_moves]

    return (index, i, results)


# Generate future universes from an input universe
@doublestar_function
@HandleExceptions.decorate
def generate_universes(U, frame_array, index, count, best_moves, start):

    M = collision_matrix(U)  # calculate U's matrix for collision detection
    Us = [deepcopy_list(U) for i in range(4*Config.K)]

    # display update
    details = {"process_id": os.getpid(),
               "universe_index": index + 1,
               "universe_count": count,
               "current_runtime": time.time() - start}

    safe_print("[PID: {process_id}] generate_universes: "
               "Universe {universe_index} of {universe_count} "
               "({current_runtime:.2f} sec)".format(**details))

    # Find best moves for each bacterium in U
    # Apply the best moves to all bacteria in Us
    for i in range(4*Config.K):
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
            new_M = expand_collision_matrix(M, correspondence)

        assert len(new_M) == len(Us[i]), 'M and Us[i] differ in length'
        # collisionEventsHandler.run(Us[i], new_M)
        collisionEventsHandler.run(Us[i], M)

    return [(cost(frame_array, Us[i]), Us[i], index, i)
            for i in xrange(len(Us))]


# parallelized improve() method in pulling stage
@doublestar_function
@HandleExceptions.decorate
def improve(Si, frame_array, index, count, start):

    # display update
    details = {"process_id": os.getpid(),
               "universe_index": index + 1,
               "universe_count": count,
               "current_runtime": time.time() - start}

    safe_print("[PID: {process_id}] improve: "
               "Universe {universe_index} of {universe_count} "
               "({current_runtime:.02f} sec)".format(**details))

    w, U, index, index2 = Si
    improved = True
    M = collision_matrix(U)

    while improved:
        improved = False
        for j, bacterium in enumerate(U):

            # shrink
            for dh in [-4, -2, -1]:
                if U[j].length + dh < 10:
                    continue
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
            for x in [-1, 1]:
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
            for y in [-1, 1]:
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


def trim_comments(line):
    return LINE_PATTERN.match(line).group("line")


def get_next_nonempty_line(initial):
    line = trim_comments(initial.next()).strip()
    while not line:
        line = trim_comments(initial.next().strip())
    return line


# Initial space S
def init_space(t, initial):
    # FIXME: A malformed initial config file can cause an exception without any
    #  explaination.
    Config.dt = float(get_next_nonempty_line(initial))
    Config.init_length = int(get_next_nonempty_line(initial))
    Config.init_width = int(get_next_nonempty_line(initial))
    Config.max_speed = int(get_next_nonempty_line(initial))
    Config.max_spin = float(get_next_nonempty_line(initial))
    Config.K = int(get_next_nonempty_line(initial))
    Config.MAX_X_MOTION = float(get_next_nonempty_line(initial))
    Config.MAX_Y_MOTION = float(get_next_nonempty_line(initial))
    Config.MAX_X_RESOLUTION = int(get_next_nonempty_line(initial))
    Config.MAX_Y_RESOLUTION = int(get_next_nonempty_line(initial))
    Config.MAX_ROTATION = float(get_next_nonempty_line(initial))
    Config.MAX_ROTATION_RESOLUTION = int(get_next_nonempty_line(initial))
    Config.MIN_LENGTH_INCREASE = int(get_next_nonempty_line(initial))
    Config.MAX_LENGTH_INCREASE = int(get_next_nonempty_line(initial))
    Config.LENGTH_INCREASE_RESOLUTION = float(get_next_nonempty_line(initial))
    Config.MAX_LENGTH_BEFORE_SPLIT = int(get_next_nonempty_line(initial))
    Config.MIN_LENGTH = int(get_next_nonempty_line(initial))
    Config.SPLIT_RATIO_BEGINNING = float(get_next_nonempty_line(initial))
    Config.SPLIT_RATIO_END = float(get_next_nonempty_line(initial))
    Config.SPLIT_RATIO_RESOLUTION = int(get_next_nonempty_line(initial))

    # define attribute names and types
    schema = {"name": str,
              "pos:x": float,
              "pos:y": float,
              "length": float,
              "rotation": float}

    # list required header columns
    requirements = ["pos:x", "pos:y", "length", "rotation"]

    # find header
    for line in initial:
        header = shlex.split(trim_comments(line))
        if header:
            break
    else:
        raise Exception("No data found in '{}'!".format(initial.name))

    # check for valid header columns
    invalid_attributes = [column.lower()
                          for column in header
                          if column not in schema]

    if invalid_attributes:
        raise Exception("Header column(s) '{}' not valid attribute(s)! "
                        "Must be one of the following: {}"
                        .format(", ".join(invalid_attributes),
                                ", ".join(schema.keys())))

    # check for duplicates
    duplicate_attributes = set([column
                                for column in header
                                if header.count(column) > 1])

    if duplicate_attributes:
        raise Exception("Cannot have duplicate header columns! "
                        "Too many: {}"
                        .format(", ".join(duplicate_attributes)))

    # check for required header columns
    missing_attributes = [required
                          for required in requirements
                          if required not in header]

    if missing_attributes:
        raise Exception("Missing header column(s): {}"
                        .format(", ".join(missing_attributes)))

    # get initial data
    rows = []
    for line in initial:
        data = shlex.split(trim_comments(line))

        # check for and skip empty lines
        if not data:
            continue

        # check that row column count matches header column count
        if len(data) != len(header):
            raise Exception("Row '{}' has an incorrect number of columns!"
                            .format(trim_comments(line)))

        # append row of data
        row = {header[i]: schema[header[i]](data[i]) for i in range(len(data))}
        rows.append(defaultdict(str, row))

    # check if we have any rows
    if not rows:
        raise Exception("No initial data found!")

    # create a name template for automatically generating names
    name_template = "{{:0{}b}}".format(len("{:b}".format(len(rows) - 1)))

    # create starting universe
    universe = []
    for index, row in enumerate(rows):
        # create bacterium
        bacterium = Bacterium()
        bacterium.name = row["name"] or name_template.format(index)
        bacterium.pos = np.array([row["pos:x"], row["pos:y"], 0])
        bacterium.theta = row["rotation"]
        bacterium.length = row["length"]
        bacterium.update()

        # add to the universe
        universe.append(bacterium)

    return [universe]


# Advance bacterium, including moving and spinning
def advance(bacterium):
    bound_velocity(bacterium, Config.max_speed, 2*pi)
    bacterium.pos += bacterium.v*Config.dt
    bacterium.theta += bacterium.w.z*Config.dt

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
    red_im[:, :, 1] = im

    return red_im


def get_frames(directory, t):
    frames = []
    file_names = []

    for f in os.listdir(directory):
        if not f.endswith('.png'):
            continue
        if int(f.split('.')[0]) < t:
            continue
        file_names.append(f)

    file_names = sorted(file_names, key=lambda f: int(f.split('.')[0]))

    for f in file_names:
        path = os.path.join(directory, f)
        frame = cv2.imread(path)
        height, width, _ = frame.shape

        if Globals.image_width is None or Globals.image_height is None:
            Globals.image_width = width
            Globals.image_height = height
        elif Globals.image_width != width or Globals.image_height != height:
            raise Exception("Images must be of the same size! "
                            "'{}' doesn't match previous frames.".format(f))

        frames.append(frame)

    return frames


def two_tuple(v):
    return tuple(list(v)[:2])


def generate_image_cv2(U, im=None):
    if im is None:
        im = np.zeros((Globals.image_height, Globals.image_width),
                      dtype=np.int16)
    for bacterium in U:

        # head and tail
        cv2.circle(im, tuple(bacterium.head_pos[:2].astype(int)),
                   bacterium.radius, 1, -1)
        cv2.circle(im, tuple(bacterium.tail_pos[:2].astype(int)),
                   bacterium.radius, 1, -1)

        # body
        points = [tuple(bacterium.end_point_1[:2]),
                  tuple(bacterium.end_point_2[:2]),
                  tuple(bacterium.end_point_3[:2]),
                  tuple(bacterium.end_point_4[:2])]
        points = np.array([(int(point[0]), int(point[1])) for point in points])
        cv2.fillConvexPoly(im, points, 1, 1)

    return im


def generate_image_edge(U, im=None):
    if im is None:
        im = np.zeros((Globals.image_height, Globals.image_width),
                      dtype=np.int16)
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
        cv2.ellipse(im, (center[0], center[1]), axes,
                    bacterium.theta*180/3.14, 90, 270, 1)

        # tail
        center = (int(bacterium.tail_pos[0]), int(bacterium.tail_pos[1]))
        axes = (bacterium.radius, bacterium.radius)
        cv2.ellipse(im, (center[0], center[1]), axes,
                    bacterium.theta*180/3.14, -90, 90, 1)

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
        cv2.line(im, p1, p4, (255, 0, 0))
        cv2.line(im, p2, p3, (255, 0, 0))

        # head
        center = (int(bacterium.head_pos[0]), int(bacterium.head_pos[1]))
        axes = (bacterium.radius, bacterium.radius)
        cv2.ellipse(im, (center[0], center[1]), axes,
                    bacterium.theta*180/3.14, 90, 270, 255)

        # tail
        center = (int(bacterium.tail_pos[0]), int(bacterium.tail_pos[1]))
        axes = (bacterium.radius, bacterium.radius)
        cv2.ellipse(im, (center[0], center[1]), axes,
                    bacterium.theta*180/3.14, -90, 90, 255)

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
            distance_between_centers = np.linalg.norm(U[i].pos - U[j].pos)
            if distance_between_centers < U[i].length/2 + U[j].length/2 + 6:
                M[i][j] = True

    return M


def expand_collision_matrix(old_matrix, correspondence):
    """Expands the collision matrix after splits have occured.

    When a bacterium splits, their bodies overlap the same space that the
    original bacterium covered (they don't spontaneously move far away).
    Therefore, the entry of the collision matrix for the original bacterium
    corresponds to the entries of the two new bacteria (one or both will still
    collide with the same bacterium as the original did).  We can use this
    fact to save time by reusing the existing values in the collision matrix,
    since computing the distance between two bacterium is computationally
    more expensive.

    Args:
        old_matrix (list): The old collision matrix (list of lists of
            booleans).
        correspondence (list): A list of integers representing the bacteria
            index which corresponds to the other split bacterium.

    Returns:
        A boolean square matrix of length len(old_matrix) + len(correspondence)
        indicating if a pair of bacteria are in proximity to each other.
    """
    old_count = len(old_matrix)
    new_count = len(old_matrix) + len(correspondence)

    new_matrix = [[False]*new_count]*new_count

    # Copy over the existing collision matrix
    for i in xrange(new_count):
        for j in xrange(i + 1, new_count):

            ci = i
            cj = j

            # get the corresponding indices of the newly split bacteria
            if i >= old_count:
                ci = correspondence[i - old_count]
            if j >= old_count:
                cj = correspondence[j - old_count]

            # swap so that the smallest index is first
            if ci > cj:
                ci, cj = cj, ci

            new_matrix[i][j] = old_matrix[ci][cj]

    return new_matrix


# deep copy function
#   returns a copy of a bacterium
def deepcopy(bacterium):
    new_bacterium = Bacterium()
    new_bacterium.pos = np.array(bacterium.pos)
    new_bacterium.theta = bacterium.theta
    new_bacterium.length = bacterium.length
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
        bacterium.name = str(line[0])
        bacterium.pos[0] = float(line[1])
        bacterium.pos[1] = float(line[2])
        bacterium.length = float(line[3])
        bacterium.theta = float(line[4])

        bacterium.update()
        U.append(bacterium)

    return U


# Write the state of bacteria B to the file f
def write_state(path, index, U):
    with open(path, "w") as file:
        file.write("{index} {count}\n".format(index=index, count=len(U)))
        for bacterium in sorted(U, key=lambda x: x.pos[0]):
            file.write("{name} {x} {y} {length} {theta}\n"
                       .format(name=bacterium.name,
                               x=bacterium.pos[0],
                               y=bacterium.pos[1],
                               length=bacterium.length,
                               theta=bacterium.theta))


def write_state2(path, U):
    with open(path, "w") as file:
        file.write("{count}\n".format(count=len(U)))
        for bacterium in sorted(U, key=lambda x: x.pos[0]):
            file.write("{name} {index} {x} {y} {length} {theta}\n"
                       .format(name=bacterium.name,
                               index=bacterium.index,
                               x=bacterium.pos[0],
                               y=bacterium.pos[1],
                               length=bacterium.length,
                               theta=bacterium.theta))


# cost function
def cost(base_im_array, U):
    im = generate_image_cv2(U)
    dif = cv2.absdiff(base_im_array, im)

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
    bacterium.pos = np.array([x, y, 0])
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
