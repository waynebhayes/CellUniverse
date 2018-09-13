# -*- coding: utf-8 -*-

"""
cellannealer.colony
~~~~~~~~~~~~~~~~~~~
"""

from copy import deepcopy


class LineageNode(object):

    def __init__(self, cell, parent=None):
        self._cells = [cell]
        self._families = [[]]
        self._parent = parent

    @property
    def cell(self):
        return self._cells[-1]

    @property
    def children(self):
        return self._families[-1]

    @property
    def parent(self):
        return self._parent

    def push(self):
        self._cells.append(deepcopy(self.cell))
        self._families.append(deepcopy(self.children))

    def pop(self):
        self._cells.pop()
        self._families.pop()

    def flatten(self):
        self._cells = [self.cell]
        self._families = [self.children]


class Colony(object):

    def __init__(self):
        self._roots = []

    @property
    def roots(self):
        return self._roots

    @property
    def leaves(self):
        leaves = []
        for root in self.roots:
            stack = [root]
            while stack:
                if not stack[-1].children:
                    leaves.append(stack[-1])
                children = stack[-1].children
                stack.pop()
                for child in children:
                    stack.append(child)
        return leaves

    def flatten(self):
        self._roots = self.leaves
