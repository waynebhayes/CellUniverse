from pathlib import Path
from typing import Dict, Optional, List, Generic, Tuple, Any
import numpy.typing as npt
import numpy as np
import pandas as pd
from PIL import Image
import random

from copy import deepcopy
from collections import defaultdict

from .Cells import Cell
from .Config import SimulationConfig
from .Cells import Sphere

class Frame:
    def __init__(self, real_image_stack: npt.NDArray, simulation_config: SimulationConfig, cells: List[Cell], output_path: Path, image_name: str):
        self.z_slices = [simulation_config.z_scaling * (i - simulation_config.z_slices // 2) for i in range(simulation_config.z_slices)]
        self.cells = cells
        self.simulation_config = simulation_config
        self.output_path = output_path
        self.image_name = image_name  # name of image file for saving cell data

        self._real_image_stack = real_image_stack  # original 3d array of images
        self.real_image_stack = np.array(self._real_image_stack)  # create a copy of the original image stack

        self.pad_real_image()
        # self.cell_map_stack = self.generate_cell_maps()
        self.synth_image_stack = self.generate_synth_images()

    # def update(self):
    #     """Update the frame."""
    #     self.pad_real_image()
    #     self.synth_image_stack = self.generate_synth_images()
    #     self.cell_map_stack = self.generate_cell_maps()
    #
    # def update_simulation_config(self, simulation_config: SimulationConfig):
    #     """Update the simulation config and regenerate the synthetic images and cell maps."""
    #     self.simulation_config = simulation_config
    #     self.update()

    def generate_synth_images(self):
        """Generate synthetic images from the cells in the frame."""
        if self.cells is None:
            raise ValueError("Cells are not set")

        shape = self.get_image_shape()
        synth_image_stack = []

        for i, z in enumerate(self.z_slices):
            synth_image = np.full(shape, self.simulation_config.background_color)
            for cell in self.cells:
                cell.draw(synth_image, self.simulation_config, z = z)
            synth_image_stack.append(synth_image)

        return np.array(synth_image_stack)
    

    def generate_synth_images_fast(self, old_cell: Cell, new_cell: Cell):
        """ 
            Generate the synthetic images after perturbing a cell.
            This is a faster version of a similar function that
            tries to reuse previously generated images if no changes
            have been made to them.
        """
        if self.cells is None:
            raise ValueError("Cells are not set")

        shape = self.get_image_shape()
        synth_image_stack = []

        # Calculate the smallest box that contains both the old and new cell
        min_corner, max_corner = old_cell.calculate_minimum_box(new_cell)

        for i, z in enumerate(self.z_slices):
            # If the z-slice is outside the min/max box, append the existing synthetic image to the stack
            if (z < min_corner[2] or z > max_corner[2]):
                synth_image_stack.append(self.synth_image_stack[i])
                continue
           
            synth_image = np.full(shape, self.simulation_config.background_color)
            for cell in self.cells:
                cell.draw(synth_image, self.simulation_config, z = z)
            synth_image_stack.append(synth_image)

        # Return the stack of synthetic images
        return synth_image_stack
        


    def calculate_cost(self, synth_image_stack: npt.NDArray):
        """Calculate the L2 cost of the synthetic images."""
        return float(np.linalg.norm(self.real_image_stack - synth_image_stack))

    # def generate_cell_maps(self):
    #     """Generate cell maps from the cells in the frame. This should only be for binary images"""
    #     # TODO: Implement this
    #     return np.zeros(self.real_image_stack.shape)

    def pad_real_image(self):
        """Pad the real image to account for the padding in the synthetic images."""
        padding = ((0, 0), (self.simulation_config.padding, self.simulation_config.padding), (self.simulation_config.padding, self.simulation_config.padding))
        self.real_image_stack = np.pad(self.real_image_stack, padding, mode='constant', constant_values=self.simulation_config.background_color)

    def get_image_shape(self):
        """Get the shape of an individual image in the frame."""
        return self.real_image_stack.shape[1:]

    def generate_output_images(self):
        """Generate the output images for the frame."""
        real_images_with_outlines: List[Image.Image] = []
        for real_image, z in zip(self.real_image_stack, self.z_slices):
            output_frame = np.stack((real_image,) * 3, axis=-1)
            for cell in self.cells:
                cell.draw_outline(output_frame, (1, 0, 0), z)
            output_frame = Image.fromarray(np.uint8(255 * output_frame))
            real_images_with_outlines.append(output_frame)
        return real_images_with_outlines

    def generate_output_synth_images(self):
        """Generate the output synthetic images for the frame."""
        return [Image.fromarray(np.uint8(255 * synth_image), "L") for synth_image in self.synth_image_stack]

    def get_cells_as_params(self):
        """Convert the cells in the frame to a pandas dataframe."""
        cell_params = pd.DataFrame([dict(cell.get_cell_params()) for cell in self.cells])
        # set the file name for all cells to the same file name
        cell_params["file"] = self.image_name
        return cell_params

    def __len__(self):
        return len(self.cells)

    def perturb(self):
        # randomly pick an index for a cell
        index = random.randint(0, len(self.cells) - 1)

        # store old cell
        old_cell = self.cells[index]

        # replace the cell at that index with a new cell
        self.cells[index] = self.cells[index].get_perturbed_cell()

        # synthesize new synthetic image
        # new_synth_image_stack = self.generate_synth_images()
        new_synth_image_stack = self.generate_synth_images_fast(old_cell, self.cells[index])
        
        # get the cost of the new synthetic image
        new_cost = self.calculate_cost(new_synth_image_stack)

        # if the difference is greater than the threshold, revert to the old cell
        old_cost = self.calculate_cost(self.synth_image_stack)

        def callback(accept: bool):
            if accept:
                self.cells[index] = old_cell
            else:
                self.synth_image_stack = new_synth_image_stack

        return old_cost - new_cost, callback

    def _cost_of_perterb(self, perterb_param: str, perterb_val: str, index: int, old_cell: Cell):
        # setup parameters to test perterb of size delta
        perterb_params = defaultdict(float)
        perterb_params[perterb_param] = perterb_val
        # perterb cell
        self.cells[index] = self.cells[index].get_paramaterized_cell(perterb_params)

        # generate new image stack
        new_synth_image_stack = self.generate_synth_images()
        # get new cost
        new_cost = self.calculate_cost(new_synth_image_stack)

        # reset cell
        self.cells[index] = old_cell

        return new_cost

    def _get_synth_perterbed_cells(
        self, 
        index: int, 
        params: Dict[str, Any], 
        perterb_length: float, 
        old_cell: Cell
    ) -> Dict[str, Any]:
        """
            Generates a dict in the format of {param: synth_image} where synth image is the synth image stack after
            some cell param was perterbed by some perterb_length
        """
        perterbed_cells = {}
        
        for param, val in params.items():
            if param == 'name':
                continue
            perterb_params = defaultdict(float)
            perterb_params[param] = perterb_length
            self.cells[index] = self.cells[index].get_paramaterized_cell(perterb_params)
            perterbed_cells[param] = self.generate_synth_images()
            self.cells[index] = old_cell
        
        return perterbed_cells


    # add a line search to figure out how much to move in the gradient direction
    # add a threshold so that stops calculating gradient when it converges
    def gradient_descent(self):

        directions = defaultdict(dict)

        # hyper parameters to tune for gradient descent
        moving_delta = 1
        delta = 1e-3
        alpha = 0.2

        cell_list = self.cells
        orig_cost = self.calculate_cost(self.synth_image_stack)

        cells_grad = []
        param_names = None

        print(f"original cost: {orig_cost}")

        # get gradient for each cell
        for index, cell in enumerate(cell_list):
            old_cell = deepcopy(cell)
        
            params = cell.get_cell_params().__dict__
            
            # get params that are changing
            if param_names is None:
                param_names = list(params.keys())
                param_names.remove("name")

            perterbed_cells = self._get_synth_perterbed_cells(index, params, moving_delta, old_cell)

            perterbed_cells_val = np.array(list(perterbed_cells.values()))
            costs = list(map(self.calculate_cost, perterbed_cells_val))

            # memory vs speed (if user needs more memory calculate grad here, otherwise vectorize and calculate 
            # grad after)
            cells_grad.append(costs)

            # cells_grad.append((costs - orig_cost) / delta)

        cells_grad = (np.array(cells_grad) - orig_cost) / delta

        # use this direction value if no line search
        #directions = {index:dict(zip(param_names, -1 * alpha * cells_grad[index])) for index, grad in enumerate(cells_grad)}
        
        # find optimal distance to move for each perterb using line search
        # TODO: if using line search try to find best tolerance and upper bound
        
        for index, cell in enumerate(cell_list):
            
            param_gradients = dict(zip(param_names, cells_grad[index]))
            old_cell = deepcopy(cell)

            for param, gradient in param_gradients.items():
                tolerance = 1e-2
                direction = -1 * alpha
                # line search to find the optimal amount to move
                lower = gradient * direction
                upper = 3 * lower
                lower_cost = self._cost_of_perterb(param, lower, index, old_cell)
                upper_cost = self._cost_of_perterb(param, upper, index, old_cell)
                # assume that the lower bound gradient is the best cost until new minima found
                best_cost = lower_cost
                old_upper_cost = 0

                # keep on finding the upper limit of line search until cost is negative
                while upper_cost < best_cost and (upper_cost - old_upper_cost) > tolerance:
                    upper = 3 * upper
                    old_upper_cost = upper_cost
                    upper_cost = self._cost_of_perterb(param, upper, index, old_cell)
                
                mid = None
                # try to find minimal cost by looking at the lower and upper cost
                while True:
                    mid = (lower + upper) / 2
                    mid_cost = self._cost_of_perterb(param, mid, index, old_cell)

                    if abs(mid_cost - best_cost) < tolerance:
                        break
                    
                    if mid_cost < lower_cost:
                        lower = mid
                        lower_cost = mid_cost
                        best_cost = mid_cost

                    elif mid_cost > lower_cost:
                        upper = mid
                    
                directions[index][param] = mid
        

        for index, cell in enumerate(cell_list):
            self.cells[index] = self.cells[index].get_paramaterized_cell(directions[index])
        
        self.synth_image_stack = self.generate_synth_images()
        new_cost = self.calculate_cost(self.synth_image_stack)
        
        print(f"current cost: {new_cost}")
        return new_cost
        