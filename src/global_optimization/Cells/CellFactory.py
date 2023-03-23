from pathlib import Path
from collections import defaultdict

import pandas as pd

from .Cell import Cell
from .Sphere import Sphere
from .Bacilli import Bacilli
from typing import Dict, Type

class CellFactory:
    _cell_types: Dict[str, Type[Cell]] = {
        'sphere': Sphere,
        'bacilli': Bacilli
    }

    def __init__(self, cell_type: str):
        if cell_type not in self._cell_types:
            raise ValueError(f'Invalid cell type: "{cell_type}"')
        self.cellClass = self._cell_types[cell_type]

    def create_cells(self, init_params_path: Path):
        initial_cell_params = pd.read_csv(init_params_path)
        initial_cells = defaultdict(list)
        paramClass = self.cellClass.paramClass

        for cell_data in initial_cell_params.to_dict(orient='records'):
            # The dictionary is typed as {Hashable: Any} but we know that the keys are strings
            cell_params = paramClass(**cell_data)  # type: ignore
            initial_cells[cell_params.file].append(self.cellClass(cell_params))
        return initial_cells
