from pathlib import Path
from typing import Dict, Optional, List, Generic
import numpy.typing as npt
import numpy as np
import pandas as pd
from PIL import Image


from .Cells import Cell
from .Config import SimulationConfig


# Helper functions
def load_image(image_file: Path):
    """Open the image file and return a floating-point grayscale array."""
    with Image.open(image_file) as img:
        # Convert the image to grayscale if it's in RGB mode
        if img.mode == 'RGB':
            img = img.convert('L')
        # Convert the image to a NumPy array and normalize the pixel values
        arr = np.array(img, dtype=np.float32) / 255.0
    return arr


class Frame:


    def __init__(self, real_image_paths: List[Path], simulation_config: SimulationConfig, cells: List[Cell]):
        real_images: List[npt.NDArray] = []
        for path in real_image_paths:
            real_images.append(load_image(path))
        self._real_image_stack = np.array(real_images)  # original 3d array of images
        self.real_image_stack = np.array(self._real_image_stack)  # create a copy of the original image stack
        self.cells = cells
        self.simulation_config = simulation_config
        self.synth_image_stacks: Optional[npt.NDArray] = None  # 3D array of synthetic images
        self.cell_map_stacks: Optional[npt.NDArray] = None  # 3D array of cell maps
        self.update()

    def update(self):
        """Update the frame."""
        self.pad_real_image()
        self.update_synth_images()
        self.update_cell_maps()

    def update_simulation_config(self, simulation_config: SimulationConfig):
        """Update the simulation config and regenerate the synthetic images and cell maps."""
        self.simulation_config = simulation_config
        self.update()

    def update_synth_images(self):
        """Generate synthetic images from the cells in the frame."""
        if self.simulation_config is None:
            raise ValueError("Simulation config is not set")
        if self.cells is None:
            raise ValueError("Cells are not set")
        pass

    def update_cell_maps(self):
        """Generate cell maps from the cells in the frame. This should only be for binary images"""
        # TODO: Implement this
        self.cell_map_stacks = np.zeros((0, 0))

    def pad_real_image(self):
        """Pad the real image to account for the padding in the synthetic images."""
        padding = ((0, 0), (self.simulation_config.padding, self.simulation_config.padding), (self.simulation_config.padding, self.simulation_config.padding))
        self.real_image_stack = np.pad(self.real_image_stack, padding, mode='constant', constant_values=self.simulation_config.background_color)