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
    z_slices = 1  # Number of z slices in 3d image
    z_scaling = 1
    z_values: List[int] = []  # List of z values to use for each image slice. This is set automatically, do not specify

    @validator('z_values')
    def set_z_values(cls, v, values):
        if v != []:
            raise ValueError('z_values should not be set manually')
        slices = values['z_slices']
        return [i - slices // 2 for i in range(slices)]

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

class PerturbationConfig(BaseModel, extra = 'forbid'):
    prob_x: float
    prob_y: float
    prob_z: float
    prob_width: float
    prob_length: float
    prob_rotation: float
    modification_x_mu: float
    modification_y_mu: float
    modification_width_mu: float
    modification_length_mu: float
    modification_rotation_mu: float
    modification_x_sigma: float
    modification_y_sigma: float
    modification_width_sigma: float
    modification_length_sigma: float
    modification_rotation_sigma: float
    prob_opacity: float
    modification_opacity_mu: float
    modification_opacity_sigma: float

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

    # Perturbation settings
    perturbation: PerturbationConfig

    # Camera shift settings
    camera = CameraShiftConfig()

    # Misc settings
    global_optimizer_window_size: int
    pbad_max_size: int
    auto_temp_scheduler_iteration_per_cell: int
    output_format: str
    output_quality: int
    residual_vmin: float
    residual_vmax: float
    split_cost: float
    combine_cost: float
    overlap_cost: float
    cell_importance: float
    background_offset_mu: float
    background_offset_sigma: float
    cell_brightness_mu: float
    cell_brightness_sigma: float
    opacity_offset_mu: float
    opacity_offset_sigma: float
    diffraction_strength_offset_mu: float
    diffraction_strength_offset_sigma: float
    diffraction_sigma_offset_mu: float
    diffraction_sigma_offset_sigma: float
