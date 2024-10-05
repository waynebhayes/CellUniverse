# -*- coding: utf-8 -*-

"""
cellanneal.mathhelper
~~~~~~~~~~~~~~~~~~~~~

Contains classes and functions to make calculations easier to read.
"""

import numpy as np


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


class Rectangle(object):
    """Rectangle represents a 2-D region."""

    def __init__(self, left, top, right, bottom):
        self._left = left
        self._top = top
        self._right = right
        self._bottom = bottom

    def union(self, rectangle):
        return Rectangle(
            min(self._left, rectangle._left),
            min(self._top, rectangle._top),
            max(self._right, rectangle._right),
            max(self._bottom, rectangle._bottom))

    def __repr__(self):
        return (f'Rectangle('
                f'left={self._left}, '
                f'top={self._top}, '
                f'right={self._right}, '
                f'bottom={self._bottom})')

    @property
    def left(self):
        return self._left

    @property
    def top(self):
        return self._top

    @property
    def right(self):
        return self._right

    @property
    def bottom(self):
        return self._bottom
