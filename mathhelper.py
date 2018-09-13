# -*- coding: utf-8 -*-

"""
cellannealer.mathhelper
~~~~~~~~~~~~~~~~~~~~~~~
"""

import numpy as np


class Rectangle(np.ndarray):

    def __new__(cls, rectangle=None):
        obj = super().__new__(cls, shape=4)
        if rectangle is not None:
            obj[0], obj[1], obj[2], obj[3] = rectangle
        return obj

    @property
    def left(self):
        return self[0]

    @left.setter
    def left(self, left):
        self[0] = left

    @property
    def top(self):
        return self[1]

    @top.setter
    def top(self, top):
        self[1] = top

    @property
    def right(self):
        return self[2]

    @right.setter
    def right(self, right):
        self[2] = right

    @property
    def bottom(self):
        return self[3]

    @bottom.setter
    def bottom(self, bottom):
        self[3] = bottom


class Shape(np.ndarray):

    def __new__(cls, shape=None):
        obj = super().__new__(cls, shape=2)
        if shape is not None:
            obj[0], obj[1] = shape[:2]
        return obj

    @property
    def height(self):
        return self[0]

    @height.setter
    def height(self, height):
        self[0] = height

    @property
    def width(self):
        return self[1]

    @width.setter
    def width(self, width):
        self[1] = width


class Dimensions(np.ndarray):

    def __new__(cls, dimensions=None):
        obj = super().__new__(cls, shape=2)
        if dimensions is not None:
            obj[0], obj[1] = dimensions
        return obj

    @property
    def length(self):
        return self[0]

    @length.setter
    def length(self, length):
        self[0] = length

    @property
    def width(self):
        return self[1]

    @width.setter
    def width(self, width):
        self[1] = width


class Vector(np.ndarray):

    def __new__(cls, vector=None):
        obj = super().__new__(cls, shape=3)
        if vector is not None:
            obj[0], obj[1], obj[2] = vector
        return obj

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, x):
        self[0] = x

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, y):
        self[1] = y

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, z):
        self[2] = z
