from typing import List, Union, Generic, TypeVar
from pydantic import BaseModel, root_validator, validator
from pydantic.generics import GenericModel

from colorama import Fore, Style

from ..Cells import BacilliConfig
from ..Cells import SphereConfig

class SimulationConfig(BaseModel, extra = 'forbid'):
    image_type: str
    background_color: float
    cell_color: float
    light_diffraction_sigma: Union[float, str]
    light_diffraction_strength: Union[float, str]
    light_diffraction_truncate: float
    cell_opacity: Union[float, str]
    padding = 0
    z_slices: int = None  # Number of z slices in 3d image
    z_scaling = 1
    z_values: List[int] = []  # List of z values to use for each image slice. This is set automatically, do not specify

    @validator('z_values')
    def check_z_values(cls, v, values):
        if v != []:
            raise ValueError('z_values should not be set manually')
    
    @validator('z_slices')
    def check_z_slices(cls, v, values):
        if v is not None:
            raise ValueError('z_slices should not be set manually')
        


class ProbabilityConfig(BaseModel, extra = 'forbid'):
    perturbation: float
    combine: float
    split: float
    camera_shift: float
    opacity_diffraction_offset: float
    background_offset: float

    @root_validator()
    def check_probability(cls, values):
        prob_sum = sum(values.values())
        if prob_sum != 1:
            for change in values:
                values[change] /= prob_sum
            print(Fore.YELLOW, end = '')
            print(f'WARNING: Probability sum is {prob_sum}, scaling to 1')
            print(f'New probabilities are {values}')
            print(Style.RESET_ALL, end='')
        return values

class CameraShiftConfig(BaseModel, extra = 'forbid'):
    modification_x_sigma = 0.0
    modification_y_sigma = 0.0

CellConfig = TypeVar('CellConfig', SphereConfig, BacilliConfig)

class BaseConfig(GenericModel, Generic[CellConfig], extra = 'forbid'):
    # Global settings
    cellType: str
    pixelsPerMicron: int
    framesPerSecond: int

    # Cell settings
    cell: CellConfig

    # Simulation settings
    simulation: SimulationConfig

    # Probability settings
    prob: ProbabilityConfig

    # Camera shift settings
    camera = CameraShiftConfig()

    # Misc settings
    # global_optimizer_window_size: int
    # pbad_max_size: int
    # auto_temp_scheduler_iteration_per_cell: int
    # output_format: str
    # output_quality: int
    # residual_vmin: float
    # residual_vmax: float
    # split_cost: float
    # combine_cost: float
    # overlap_cost: float
    # cell_importance: float
    # background_offset_mu: float
    # background_offset_sigma: float
    # cell_brightness_mu: float
    # cell_brightness_sigma: float
    # opacity_offset_mu: float
    # opacity_offset_sigma: float
    # diffraction_strength_offset_mu: float
    # diffraction_strength_offset_sigma: float
    # diffraction_sigma_offset_mu: float
    # diffraction_sigma_offset_sigma: float
