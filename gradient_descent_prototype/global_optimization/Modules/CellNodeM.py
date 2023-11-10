from typing import List
import cell


class CellNodeM:
    def __init__(self, cell: cell.Bacilli, parent: 'CellNodeM' = None):
        self.cell = cell
        self.parent = parent
        self.children: List[CellNodeM] = []

    def __repr__(self):
        return f'<name={self.cell.name}, parent={self.parent.cell.name if self.parent else None}, children={[node.cell.name for node in self.children]}>'

    @property
    def grandchildren(self):
        grandchildren = []
        for child in self.children:
            grandchildren.extend(child.children)
        return grandchildren

    def make_child(self, cell: cell.Bacilli):
        child = CellNodeM(cell, self)
        self.children.append(child)
        return child
