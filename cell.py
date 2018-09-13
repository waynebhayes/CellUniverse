# -*- coding: utf-8 -*-

"""
cellannealer.cell
~~~~~~~~~~~~~~~~~
"""

import colorsys
import hashlib

import numpy as np
from skimage.draw import circle, polygon

from errorinfo import CellAnnealerError
from mathhelper import Dimensions, Rectangle, Vector


class Cell(object):

    _REQUIRED_CONFIG = []

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return f'Cell(name={self.name})'

    def __str__(self):
        return repr(self)

    @classmethod
    def check_config(cls, config):
        for required in cls._REQUIRED_CONFIG:
            if required not in config:
                raise CellAnnealerError(f'Invalid config: missing \'{required}\'')

    def update(self, *args, **kwargs):
        raise NotImplementedError()

    def draw(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def name(self):
        return self._name


class Bacilli(Cell):

    _REQUIRED_CONFIG = [
        'maxSpeed',
        'maxSpin',
        'minGrowth',
        'maxGrowth',
        'minWidth',
        'maxWidth',
        'minLength',
        'maxLength'
    ]

    def __init__(self, name, x, y, width, length, rotation, in_flux=False):
        super().__init__(name)

        self.position = Vector([x, y, 0])
        self.dimensions = Dimensions([length, width])
        self.rotation = float(rotation)
        self.in_flux = in_flux

        self._head = None
        self._tail = None

        self._head_left = None
        self._head_right = None
        self._tail_left = None
        self._tail_right = None

        self._region = None

        self._update()

    def _update(self):

        # head and tail
        direction = Vector([np.cos(self.rotation), np.sin(self.rotation), 0])
        distance = self.dimensions.length - self.dimensions.width

        self._head = self.position + distance*direction/2
        self._tail = self.position - distance*direction/2

        # body
        right = Vector([-np.sin(self.rotation), np.cos(self.rotation), 0])
        radius = self.dimensions.width/2

        self._head_right = self._head + radius*right
        self._head_left = self._head - radius*right
        self._tail_right = self._tail + radius*right
        self._tail_left = self._tail - radius*right

        # region of interest
        self._region = Rectangle()
        self._region.left = min(self._head.x, self._tail.x) - radius
        self._region.right = max(self._head.x, self._tail.x) + radius
        self._region.top = min(self._head.y, self._tail.y) - radius
        self._region.bottom = max(self._head.y, self._tail.y) + radius

    def update(self, position=None, rotation=None, dimensions=None):
        changes = False

        if position is not None:
            self.position = Vector(position)
            changes = True

        if rotation is not None:
            self.rotation = float(rotation)
            changes = True

        if dimensions is not None:
            self.dimensions = Dimensions(dimensions)
            changes = True

        if changes is not None:
            self._update()

    def draw(self, synthetic_image: np.ndarray):
        mask = np.zeros_like(synthetic_image, dtype=np.bool)

        body_mask = polygon(
            r=(
                self._head_left.y, self._head_right.y,
                self._tail_right.y, self._tail_left.y
            ),
            c=(
                self._head_left.x, self._head_right.x,
                self._tail_right.x, self._tail_left.x
            ),
            shape=synthetic_image.shape)

        head_mask = circle(
            r=self._head.y,
            c=self._head.x,
            radius=self.dimensions.width/2,
            shape=synthetic_image.shape)

        tail_mask = circle(
            r=self._tail.y,
            c=self._tail.x,
            radius=self.dimensions.width/2,
            shape=synthetic_image.shape)

        mask[body_mask] = True
        mask[head_mask] = True
        mask[tail_mask] = True

        synthetic_image[mask] += 1

    def debug_draw(self, frame: np.ndarray):
        mask = np.zeros_like(frame[:, :, 0], dtype=np.bool)

        body_mask = polygon(
            r=(
                self._head_left.y, self._head_right.y,
                self._tail_right.y, self._tail_left.y
            ),
            c=(
                self._head_left.x, self._head_right.x,
                self._tail_right.x, self._tail_left.x
            ),
            shape=frame.shape[:2])

        head_mask = circle(
            r=self._head.y,
            c=self._head.x,
            radius=self.dimensions.width/2,
            shape=frame.shape[:2])

        tail_mask = circle(
            r=self._tail.y,
            c=self._tail.x,
            radius=self.dimensions.width/2,
            shape=frame.shape[:2])

        mask[body_mask] = True
        mask[head_mask] = True
        mask[tail_mask] = True

        h = hashlib.blake2b()   # pylint: disable=E1101
        h.update(self._name.encode('utf-8'))
        hue = int(h.hexdigest()[2:4], base=16)/255
        r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)

        frame[:, :, 0][mask] += r*0.3
        frame[:, :, 1][mask] += g*0.3
        frame[:, :, 2][mask] += b*0.3

    def __repr__(self) -> str:
        return (f'Bacilli(name={self.name}, x={self.position.x}, y={self.position.y}, '
                f'width={self.dimensions.width}, length={self.dimensions.length}, '
                f'rotation={self.rotation})')

    def split(self, alpha: float) -> (Cell, Cell):
        assert 0 < alpha < 1

        direction = Vector([np.cos(self.rotation), np.sin(self.rotation), 0])
        unit = self.dimensions.length*direction

        front = self.position + unit/2
        back = self.position - unit/2
        center = self.position + (1/2 - alpha)*unit

        position_1 = (front + center)/2
        position_2 = (center + back)/2

        cell_1 = Bacilli(
            self.name + '0',
            position_1.x, position_1.y,
            self.dimensions.width,
            alpha*self.dimensions.length,
            self.rotation,
            in_flux=True)

        cell_2 = Bacilli(
            self.name + '1',
            position_2.x, position_2.y,
            self.dimensions.width,
            (1 - alpha)*self.dimensions.length,
            self.rotation,
            in_flux=True)

        return cell_1, cell_2

    def combine(self, cell_1: Cell, cell_2: Cell):

        separation = cell_1.position - cell_2.position
        direction = separation/np.sqrt(separation@separation)

        # get combined front
        direction_1 = Vector([np.cos(cell_1.rotation), np.sin(cell_1.rotation), 0])
        distance_1 = cell_1.dimensions.length - cell_1.dimensions.width
        if direction_1@direction >= 0:
            head_1 = cell_1.position + distance_1*direction_1/2
        else:
            head_1 = cell_1.position - distance_1*direction_1/2
        extent_1 = head_1 + cell_1.dimensions.width*direction/2
        front = cell_1.position + ((extent_1 - cell_1.position)@direction)*direction

        # get combined back
        direction_2 = Vector([np.cos(cell_2.rotation), np.sin(cell_2.rotation), 0])
        distance_2 = cell_2.dimensions.length - cell_2.dimensions.width
        if direction_2@direction >= 0:
            tail_2 = cell_2.position - distance_2*direction_2/2
        else:
            tail_2 = cell_2.position + distance_2*direction_2/2
        extent_2 = tail_2 - cell_2.dimensions.width*direction/2
        back = cell_2.position + ((extent_2 - cell_2.position)@direction)*direction

        # update cell
        self.position = (front + back)/2
        self.rotation = np.arctan2(direction.y, direction.x)
        self.dimensions.width = (cell_1.dimensions.width + cell_2.dimensions.width)/2
        self.dimensions.length = np.sqrt((front - back)@(front - back))
        self._update()


# for testing purposes
if __name__ == '__main__':
    cell = Bacilli('foo', 23.2, 96.3, 6, 25, 1.231)
    print(cell)
    cell_1, cell_2 = cell.split(0.4)
    print(cell_1)
    print(cell_2)
    cell.combine(cell_1, cell_2)
    print(cell)
