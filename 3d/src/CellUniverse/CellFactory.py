from pathlib import Path
from collections import defaultdict

import pandas as pd

from .Cells.Cell import Cell
from .Cells.Sphere import Sphere
from .Cells.Bacilli import Bacilli
from typing import Dict, Type

from .Config import BaseConfig

class CellFactory:
    _cell_types: Dict[str, Type[Cell]] = {
        'sphere': Sphere,
        'bacilli': Bacilli
    }

    def __init__(self, config: BaseConfig):
        cell_type = config.cellType
        if cell_type not in self._cell_types:
            raise ValueError(f'Invalid cell type: "{cell_type}"')
        self.cellClass = self._cell_types[cell_type]
        self.cellClass.cellConfig = config.cell

    def create_cells(self, init_params_path: Path, z_offset = 0, z_scaling = 1):
        initial_cell_params = pd.read_csv(init_params_path)
        initial_cells = defaultdict(list)
        paramClass = self.cellClass.paramClass

        for cell_data in initial_cell_params.to_dict(orient='records'):
            # The dictionary is typed as {Hashable: Any} but we know that the keys are strings
            cell_data['z'] -= z_offset
            cell_data['z'] *= z_scaling
            cell_params = paramClass(**cell_data)  # type: ignore
            initial_cells[cell_data['file']].append(self.cellClass(cell_params))
        return initial_cells
