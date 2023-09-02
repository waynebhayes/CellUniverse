import yaml
from pathlib import Path

from ..Cells import BacilliConfig
from ..Cells import SphereConfig
from .ConfigTypes import BaseConfig


def load_config(path: Path):
    """Loads the configuration file and returns the appropriate config class and cell class."""
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    if config['cellType'] == 'sphere':
        return BaseConfig[SphereConfig](**config)
    elif config['cellType'] == 'bacilli':
        return BaseConfig[BacilliConfig](**config)
    else:
        raise ValueError(f'Invalid cell type: "{config["cellType"]}"')
