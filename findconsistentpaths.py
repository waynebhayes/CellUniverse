"""Finds self-consistent universe paths from Cell Universe data."""

import os
import shutil

from constants import Config


def create_consistent(start, end, output_dir):
    universe_path = os.path.join(output_dir, Config.states_dir)
    universe_image_path = os.path.join(output_dir, Config.images_dir)

    # make output directory
    output_base = os.path.join(output_dir, Config.consistent_states_dir)
    if not os.path.exists(output_base):
        os.makedirs(output_base)

    output_image_base = os.path.join(output_dir, Config.consistent_images_dir)
    if not os.path.exists(output_image_base):
        os.makedirs(output_image_base)

    # create directories
    for universe_index in range(Config.K):
        universe_dir = str(universe_index)

        # make directories for output
        output_path = os.path.join(output_base, universe_dir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_image_path = os.path.join(output_image_base, universe_dir)
        if not os.path.exists(output_image_path):
            os.makedirs(output_image_path)

    parents = []

    # load the object information from the Cell Universe state files
    for universe_index in range(Config.K):
        universe_dir = str(universe_index)

        # get the states corresponding to the universe index
        states_path = os.path.join(universe_path, str(universe_index))
        previous = {}

        for frame_index in range(start, end+1):
            filename = "{}.txt".format(frame_index)
            path = os.path.join(states_path, filename)

            # read the states
            with open(path, "rb") as states_file:
                parent = int(states_file.readline().split(" ")[0])

            # store the states
            previous[frame_index] = parent

        parents.append(previous)

    for universe_index in range(Config.K):
        universe_dir = str(universe_index)

        # start from last frame
        parent = universe_index
        frame_index = end

        while frame_index >= start:
            parent_dir = str(parent)

            file_path = "{}.txt".format(frame_index)
            image_path = "{}.png".format(frame_index)

            # copy last frame states
            src_path = os.path.join(universe_path, parent_dir, file_path)
            dst_path = os.path.join(output_base, universe_dir, file_path)
            shutil.copyfile(src_path, dst_path)

            src_path = os.path.join(universe_image_path,
                                    parent_dir,
                                    image_path)
            dst_path = os.path.join(output_image_base,
                                    universe_dir,
                                    image_path)
            shutil.copyfile(src_path, dst_path)

            parent = parents[parent][frame_index]
            frame_index -= 1
