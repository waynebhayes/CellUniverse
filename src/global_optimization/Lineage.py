from pathlib import Path

from .Cells import Cell

from .Config import BaseConfig
from .Frame import Frame
from typing import List, Dict
import numpy.typing as npt
import pandas as pd


class Lineage:
    def __init__(self, initial_cells: Dict[str, List[Cell]], image_paths_stack: List[List[Path]], config: BaseConfig, continue_from=-1):
        self.config = config
        self.frames: List[Frame] = []

        for i, image_paths in enumerate(image_paths_stack):
            file_name = image_paths[0].name
            if (continue_from == -1 or i < continue_from) and file_name in initial_cells:
                cells = initial_cells[str(file_name)]
            else:
                cells = []
            self.frames.append(Frame(image_paths, config.simulation, cells))

    def copy_sim_config_forward(self, to: int):
        """Copy the simulation config from the previous frame to the next frame."""
        if to <= 0:
            raise ValueError("No previous frame to copy from")
        self.frames[to].update_simulation_config(self.frames[to-1].simulation_config)

