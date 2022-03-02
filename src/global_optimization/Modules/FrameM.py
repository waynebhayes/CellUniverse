from typing import Dict, List
from .CellNodeM import CellNodeM
import cell


class FrameM:
    def __init__(self, simulation_config=None, prev: 'FrameM' = None):
        # simulation_config is passed by value and modified within the object
        self.node_map: Dict[str, CellNodeM] = {}
        self.prev = prev
        self.simulation_config = simulation_config.copy()

    def __repr__(self):
        return str(list(self.node_map.values()))

    @property
    def nodes(self) -> List[CellNodeM]:
        return list(self.node_map.values())

    def add_cell(self, cell: cell.Bacilli):
        if cell.name in self.node_map:
            self.node_map[cell.name].cell = cell
        elif self.prev and cell.name in self.prev.node_map:
            self.node_map[cell.name] = self.prev.node_map[cell.name].make_child(cell)
        elif self.prev and cell.name[:-1] in self.prev.node_map:
            self.node_map[cell.name] = self.prev.node_map[cell.name[:-1]].make_child(cell)
        else:
            self.node_map[cell.name] = CellNodeM(cell)
