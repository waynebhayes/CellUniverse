# -*- coding: utf-8 -*-

"""
cellanneal.cell
~~~~~~~~~~~~~~~

This module contains the Cell class which stores the properties of cells and
related functions.
"""
from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import TypeVar, Type

class CellParams(BaseModel):
    """The CellParams class stores the parameters of a particular cell."""
    file: str
    name: str

class Cell(ABC):
    """The Cell class stores information about a particular cell."""
    paramClass: Type[CellParams]

    @abstractmethod
    def __init__(self, initProps: CellParams):
        pass
