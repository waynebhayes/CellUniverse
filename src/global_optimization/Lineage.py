from pathlib import Path

from scipy.ndimage import gaussian_filter

from .Cells import Cell

from .Config import BaseConfig
from .Frame import Frame
from typing import List, Dict

from PIL import Image
import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage import io
from copy import deepcopy

# Helper functions
def process_image(image: Image.Image, config: BaseConfig):
    """Convert the image to grayscale and normalize the pixel values."""
    if image.mode == 'RGB':
        image = image.convert('L')
    arr = np.array(image, dtype=np.float32) / 255.0
    # gaussian blur the image
    arr = gaussian_filter(arr, sigma=config.simulation.blur_sigma)
    return arr

def load_image(image_file: Path, config: BaseConfig):
    imgs = []

    # handle tif files
    if image_file.suffix in ['.tiff', '.tif']:
        img = io.imread(image_file)
        slices = img.shape[0]

        for i in range(slices):
            pil_img = Image.fromarray(img[i].astype(np.uint8))
            imgs.append(process_image(pil_img, config))
    else:
        """Open the image file and return a floating-point grayscale array."""
        with Image.open(image_file) as img:
            # Convert the image to grayscale if it's in RGB mode
            if img.mode == 'RGB':
                img = img.convert('L')
            # Convert the image to a NumPy array and normalize the pixel values
            imgs.append(process_image(img, config))
    return imgs


class Lineage:
    def __init__(self, initial_cells: Dict[str, List[Cell]], image_paths: List[Path], config: BaseConfig, output_path: Path, continue_from=-1):
        self.config = config
        self.frames: List[Frame] = []
        self.output_path = output_path

        for i, image_path in enumerate(image_paths):
            # Load original images
            real_images: List[npt.NDArray] = []
            real_images.extend(load_image(image_path, config))

            file_name = image_path.name

            if (continue_from == -1 or i < continue_from) and file_name in initial_cells:
                cells = initial_cells[str(file_name)]
            else:
                cells = []

            self.frames.append(Frame(np.array(real_images), config.simulation, cells, output_path, file_name))

    def optimize(self, frame_index: int):
        """Perturb the cells in the frame."""
        frame = self.frames[frame_index]
        algorithm = 'gradient descent'
        total_iterations = len(frame) * self.config.simulation.iterations_per_cell

        # add tolerance (do not need to calculate gradient descent if minima is reaced)
        tolerance = 0.5
        minima_reached = False

        for i in range(total_iterations):
            if i % 100 == 0:
                print(f"Frame {frame_index}, iteration {i}")

            if algorithm == 'simulated annealing':
                cost_diff, accept = frame.perturb()
                acceptance = np.exp(-cost_diff / ((i + 1) / total_iterations))
                accept(acceptance > np.random.random_sample())
            elif algorithm == 'gradient descent':
                print(f"Current iteration: {i + 1}")
                if minima_reached:
                    continue
                    
                cur_cost = frame.calculate_cost(frame.synth_image_stack)
                new_cost = frame.gradient_descent()

                if (cur_cost - new_cost) < tolerance:
                    minima_reached = True

            else:
                # Hill climbing
                cost_diff, accept = frame.perturb()
                accept(cost_diff < 0)

    def save_images(self, frame_index: int):
        """Save the images in the frame to the output path."""
        if frame_index < 0 or frame_index >= len(self.frames):
            raise ValueError("Invalid frame index")
        real_images = self.frames[frame_index].generate_output_images()
        synth_images = self.frames[frame_index].generate_output_synth_images()
        print(f"Saving images for frame {frame_index}...")

        real_output_path = self.output_path / f"real/{frame_index}"
        if not real_output_path.exists():
            real_output_path.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(real_images):
            # create a directory for the frame if it doesn't exist
            image.save(real_output_path / f"{i}.png")

        synth_output_path = self.output_path / f"synth/{frame_index}"
        if not synth_output_path.exists():
            synth_output_path.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(synth_images):
            # create a directory for the frame if it doesn't exist
            image.save(synth_output_path / f"{i}.png")

        print("Done")

    def save_cells(self, frame_index: int):
        """Save the cells up to the frame index to the output path."""
        output = pd.concat([frame.get_cells_as_params() for frame in self.frames[:frame_index+1]])
        # sort the cells by frame and then by cell ID
        output = output.sort_values(by=['file', 'name'])
        output.to_csv(self.output_path / "cells.csv", index=False)


    def copy_sim_config_forward(self, to: int):
        """Copy the simulation config from the previous frame to the next frame."""
        if to <= 0:
            raise ValueError("No previous frame to copy from")
        self.frames[to].update_simulation_config(self.frames[to-1].simulation_config)

    def copy_cells_forward(self, to: int):
        """Copy the cells from the previous frame to the next frame."""
        if to >= len(self.frames):
            return
        self.frames[to].cells = deepcopy((self.frames[to - 1].cells))

    def __len__(self):
        return len(self.frames)
