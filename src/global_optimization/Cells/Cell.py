# -*- coding: utf-8 -*-

"""
cellanneal.cell
~~~~~~~~~~~~~~~

This module contains the Cell class which stores the properties of cells and
related functions.
"""
from __future__ import annotations
from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import TypeVar, Type, DefaultDict
import random

class CellParams(BaseModel, ABC):
    """The CellParams class stores the parameters of a particular cell."""
    name: str

class CellConfig(BaseModel, ABC):
    """Abstract base class for cell configurations."""
    pass

class PerturbParams(BaseModel):
    """Used with a CellConfig to add perturb parameters."""
    prob: float
    mu: float
    sigma: float

    def get_perturb_offset(self):
        if random.random() < self.prob:
            return random.gauss(self.mu, self.sigma)
        else:
            return self.mu


class Cell(ABC):
    """The Cell class stores information about a particular cell."""
    paramClass: Type[CellParams]
    cellConfig: CellConfig

    @abstractmethod
    def __init__(self, initProps: CellParams):
        pass

    @abstractmethod
    def draw(self, image, simulation_config, cell_map = None, z = 0):
        pass

    @abstractmethod
    def draw_outline(self, image, color, z = 0):
        pass

    @abstractmethod
    def get_perturbed_cell(self) -> Cell:
        pass
    
    @abstractmethod
    def get_paramaterized_cell(self, params: DefaultDict[str, float] = None) -> Cell:
        pass

    @abstractmethod
    def get_cell_params(self) -> CellParams:
        pass
