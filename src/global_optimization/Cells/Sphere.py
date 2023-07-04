from __future__ import annotations
from math import sqrt
from skimage.draw import disk, circle_perimeter_aa
from typing import DefaultDict

from .mathhelper import Vector

from .Cell import Cell, CellParams, PerturbParams, CellConfig


class SphereConfig(CellConfig):
    x: PerturbParams
    y: PerturbParams
    z: PerturbParams
    radius: PerturbParams
    minRadius: float
    maxRadius: float

class SphereParams(CellParams):
    x: float
    y: float
    z: float = 0.0
    radius: float
    # opacity: float = 1.0
    # split_alpha: float = 0.0


class Sphere(Cell):
    """The Sphere class represents a spherical bacterium."""
    paramClass = SphereParams
    cellConfig: SphereConfig

    def __init__(self, init_props: SphereParams):
        # destructure the properties of the sphere
        name = init_props.name
        x = init_props.x
        y = init_props.y
        z = init_props.z
        radius = init_props.radius
        # opacity = init_props.opacity
        # split_alpha = init_props.split_alpha

        self._name = name
        self._position = Vector([x, y, z])
        self._radius = radius
        self._rotation = 0
        # self._opacity = opacity
        # self._split_alpha = split_alpha
        self.dormant = False

    def draw(self, image, simulation_config, cell_map = None, z = 0):
        """Draws the cell by adding the given value to the image."""
        if self.dormant:
            return

        current_radius = self.get_radius_at(z)
        if current_radius <= 0:
            return

        background_color = simulation_config.background_color
        cell_color = simulation_config.cell_color

        rr, cc = disk((self._position.y, self._position.x), current_radius, shape=image.shape)
        image[rr, cc] = cell_color


    def draw_outline(self, image, color, z = 0):
        """Draws the outline of the cell over a color image."""
        if self.dormant:
            return

        current_radius = self.get_radius_at(z)
        if current_radius <= 0:
            return

        rr, cc, val = circle_perimeter_aa(round(self._position.y), round(self._position.x), round(current_radius), shape=image.shape)
        image[rr, cc] = color

    def split(self, alpha):
        """Splits a cell into two cells with a ratio determined by alpha."""
        pass
        # if self._needs_refresh:
        #     self._refresh()
        #
        # direction = Vector([cos(self._rotation), sin(self._rotation), 0])
        # unit = self._length*direction
        #
        # front = self._position + unit/2
        # back = self._position - unit/2
        # center = self._position + (0.5 - float(alpha))*unit
        #
        # position1 = (front + center)/2
        # position2 = (center + back)/2
        #
        # cell1 = Sphere(
        #     self._name + '0',
        #     position1.x, position1.y,
        #     self._width, self._length*float(alpha),
        #     self._rotation, alpha, self._opacity)
        #
        # cell2 = Sphere(
        #     self._name + '1',
        #     position2.x, position2.y,
        #     self._width, self._length*(1 - float(alpha)),
        #     self._rotation, alpha, self._opacity)
        #
        # return cell1, cell2

    def combine(self, cell):
        """Combines this cell with another cell."""
        pass
        # if self._needs_refresh:
        #     self._refresh()
        #
        # if cell._needs_refresh:
        #     cell._refresh()
        #
        # separation = self._position - cell._position
        # direction = separation/sqrt(separation@separation)
        #
        # # get combined front
        # direction1 = Vector([cos(self._rotation), sin(self._rotation), 0])
        # distance1 = self._length - self._width
        # if direction1@direction >= 0:
        #     head1 = self._position + distance1*direction1/2
        # else:
        #     head1 = self._position - distance1*direction1/2
        # extent1 = head1 + self._width*direction/2
        # front = self._position + ((extent1 - self._position)@direction)*direction
        #
        # # get combined back
        # direction2 = Vector([cos(cell._rotation), sin(cell._rotation), 0])
        # distance2 = cell._length - cell._width
        # if direction2@direction >= 0:
        #     tail2 = cell._position - distance2*direction2/2
        # else:
        #     tail2 = cell._position + distance2*direction2/2
        # extent2 = tail2 - cell._width*direction/2
        # back = cell._position + ((extent2 - cell._position)@direction)*direction
        #
        # # create new cell
        # position = (front + back)/2
        # rotation = atan2(direction.y, direction.x)
        # width = (self._width + cell._width)/2
        # length = sqrt((front - back)@(front - back))
        #
        # return Sphere(
        #     self._name[:-1],
        #     position.x, position.y,
        #     width, length,
        #     rotation, "combined alpha unknown", (self._opacity + cell.opacity)/2)

    def get_perturbed_cell(self):
        return Sphere(SphereParams(
            name=self._name,
            x=self._position.x + Sphere.cellConfig.x.get_perturb_offset(),
            y=self._position.y + Sphere.cellConfig.y.get_perturb_offset(),
            z=self._position.z + Sphere.cellConfig.z.get_perturb_offset(),
            radius=self._radius + Sphere.cellConfig.radius.get_perturb_offset(),
        ))
    
    def get_paramaterized_cell(self, params: DefaultDict[str, float]):
        # if no params, default to using perterb from files (used during simulated annealing)
        return Sphere(SphereParams(
            name=self._name,
            x=self._position.x + params['x'],
            y=self._position.y + params['y'],
            z=self._position.z + params['z'],
            radius=min(max(Sphere.cellConfig.minRadius, self._radius + params['radius']), Sphere.cellConfig.maxRadius),
        ))

    def get_radius_at(self, z: float):
        """Returns the radius of the sphere at a given z value."""
        if abs(self._position.z - z) > self._radius:
            return 0
        return sqrt((self._radius) ** 2 - (self._position.z - z) ** 2)

    def __repr__(self):
        return (f'Sphere('
                f'name="{self._name}", '
                f'x={self._position.x:.2f}, y={self._position.y:.2f}, z={self._position.z:.2f}, '
                f'radius={self._radius:.2f}')

    def get_cell_params(self):
        return SphereParams(
            name=self._name,
            x=self._position.x,
            y=self._position.y,
            z=self._position.z,
            radius=self._radius,
            # opacity=self._opacity,
            # split_alpha=self._split_alpha
        )
    
    def calculate_corners(self):
        """
        Calculate the minimum and maximum corners of the sphere.

        :return: A tuple of two lists, each of size 3. The first list represents the minimum corner 
                (x, y, z), and the second list represents the maximum corner (x, y, z).
        """
        min_corner = [self._position.x - self._radius, self._position.y - self._radius, self._position.z - self._radius]
        max_corner = [self._position.x + self._radius, self._position.y + self._radius, self._position.z + self._radius]
        return min_corner, max_corner

    def calculate_minimum_box(self, perturbed_cell: Sphere):
        """
        Calculate the minimum box that could encompass the current sphere and a given sphere.

        :param perturbed_cell: The second sphere that defines the dimensions of the box.
        :return: A tuple of two lists, each of size 3. The first list represents the minimum corner 
                (x, y, z) of the box, and the second list represents the maximum corner (x, y, z) of the box.
        """
        cell1_min_corner, cell1_max_corner = self.calculate_corners()
        cell2_min_corner, cell2_max_corner = perturbed_cell.calculate_corners()

        # Find the minimum and maximum coordinates among the spheres' corners
        min_corner = [min(cell1_min_corner[i], cell2_min_corner[i]) for i in range(3)]
        max_corner = [max(cell1_max_corner[i], cell2_max_corner[i]) for i in range(3)]
            
        # Return the minimum and maximum corners of the box
        return min_corner, max_corner
        