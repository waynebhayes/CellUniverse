from math import pi


class Config:
    images_dir = "Images"
    states_dir = "States"
    consistent_states_dir = "Consistent"
    consistent_images_dir = "Consistent Images"
    dt = 0.33
    init_length = 26
    init_width = 6
    max_speed = 3
    max_spin = pi/10
    K = 20
    MAX_X_MOTION = 3
    MAX_Y_MOTION = 3
    MAX_X_RESOLUTION = 7
    MAX_Y_RESOLUTION = 7
    MAX_ROTATION = pi/10
    MAX_ROTATION_RESOLUTION = 21
    MIN_LENGTH_INCREASE = 0
    MAX_LENGTH_INCREASE = 3
    LENGTH_INCREASE_RESOLUTION = 4
    MAX_LENGTH_BEFORE_SPLIT = 31
    MIN_LENGTH = 13
    SPLIT_RATIO_BEGINNING = 0.25
    SPLIT_RATIO_END = 0.75
    SPLIT_RATIO_RESOLUTION = 20


class Globals:
    image_width = None
    image_height = None
