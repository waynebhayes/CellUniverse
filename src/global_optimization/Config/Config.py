import yaml

from Cells.Bacilli import BacilliConfig
from Cells.Sphere import SphereConfig
from .ConfigTypes import BaseConfig

def load_config(path: str):
    # Loads a config file from the path and returns a config object

    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    if config['cellType'] == 'sphere':
        config_type = SphereConfig
    elif config['cellType'] == 'bacilli':
        config_type = BacilliConfig
    else:
        raise ValueError(f'Invalid cell type: "{config["cellType"]}"')
    return BaseConfig[config_type](**config)
