# -*- coding: utf-8 -*-

"""
cellanneal.cell
~~~~~~~~~~~~~~~

This module contains the Cell class which stores the properties of cells and
related functions.
"""

class Cell():
    """The Cell class stores information about a particular cell."""

    _REQUIRED_CONFIG = []

    def __init__(self, name):
        self._name = name

    @classmethod
    def checkconfig(cls, config):
        for required in cls._REQUIRED_CONFIG:
            if required not in config:
                raise ValueError(f'Invalid config: missing "{required}"')

    @property
    def name(self):
        return self._name
