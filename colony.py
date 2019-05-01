# -*- coding: utf-8 -*-

"""
cellanneal.colony
~~~~~~~~~~~~~~~~~

This module includes classes that are meant to help organize cell in an easily
accessible form. This includes having a colony class that holds all of the
cells and a lineage frames class to hold the colonies of each frame of the
video.
"""

from copy import deepcopy


class CellNode(object):
    """The CellNode class keeps track of the modifications of cells."""

    def __init__(self, cell, parent=None, prior=None, split=False, alpha=None):
        self._cell = cell
        self._parent = parent
        self._children = None
        self._prior = prior
        self._split = split
        self._alpha = alpha
        self._ignore = False

    def push(self, cell):
        """Push a modified cell over the existing one."""
        cellnode = CellNode(cell, self, self._prior, split=self._split, alpha=self._alpha)
        self._children = [cellnode]
        return cellnode

    def push2(self, cell1, cell2, alpha):
        """Push two modified cells over the existing one."""
        cellnode1 = CellNode(cell1, self, self._prior, split=True, alpha=alpha)
        cellnode2 = CellNode(cell2, self, self._prior, split=True, alpha=alpha)
        self._children = [cellnode1, cellnode2]
        return cellnode1, cellnode2

    def pop(self):
        """Remove the child nodes."""
        self._children = None

    @property
    def leaves(self):
        if self._children is None:
            return [self]
        children = []
        for child in self._children:
            children.extend(child.leaves)
        return children

    @property
    def cell(self):
        return self._cell

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return self._children

    @property
    def prior(self):
        return self._prior

    @property
    def split(self):
        return self._split

    @property
    def alpha(self):
        return self._alpha

    @property
    def ignore(self):
        return self._ignore

    @ignore.setter
    def ignore(self, value):
        self._ignore = value


class Colony(object):
    """The Colony class holds the cells in the colony."""

    def __init__(self):
        self._nodes = []

    def add(self, cellnode):
        """Add the cell to the colony."""
        self._nodes.append(cellnode)

    def __iter__(self):
        for node in self._nodes:
            for leaf in node.leaves:
                yield leaf

    def flatten(self):
        """Flatten the colony in place."""
        nodes = []
        for node in self:
            if node.split:
                # get the cell node right before the split
                presplit = node.parent
                while len(presplit.children) < 2:
                    presplit = presplit.parent

                if presplit.ignore:
                    presplit.ignore = False
                    continue

                presplit.ignore = True

                # get the latest cell nodes after the split
                top_node1, top_node2 = presplit.children
                while top_node1.children:
                    top_node1 = top_node1.children[0]
                while top_node2.children:
                    top_node2 = top_node2.children[0]

                new_node = CellNode(presplit.cell, prior=node.prior)
                new_node.push2(top_node1.cell, top_node2.cell, node.alpha)
                nodes.append(new_node)
            else:
                nodes.append(CellNode(node.cell, prior=node.prior))
        #print(len(nodes))
        self._nodes = nodes


class LineageFrames(object):
    """The LineageFrames class keeps track of the colonies of each frame."""

    def __init__(self):
        self._frames = []

    def forward(self):
        colony = Colony()
        if self._frames:
            prior_colony = self._frames[-1]
            for node in list(prior_colony):
                colony.add(CellNode(deepcopy(node.cell), prior=node))
        self._frames.append(colony)
        return colony

    def __iter__(self):
        for colony in self._frames:
            yield colony

    @property
    def latest(self):
        return self._frames[-1]
