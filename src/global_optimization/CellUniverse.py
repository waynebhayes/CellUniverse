import time
from pathlib import Path
from typing import List

import numpy as np

from .Cells import CellFactory
from .Config import load_config, BaseConfig
from .Lineage import Lineage
from .Args import Args
from skimage import io


# Helper functions
def get_image_file_paths(input_pattern: str, first_frame: int, last_frame: int, config: BaseConfig):
    """Gets the list of images that are to be analyzed."""
    image_paths: List[Path] = []
    i = first_frame
    try:
        while last_frame == -1 or i <= last_frame:
            file = Path(input_pattern % i)
            
            if file.exists() and file.is_file():
                image_paths.append(file)

                # setup some configurations automatically if they are tif files
                if file.suffix in ['.tif', '.tiff']:
                    img = io.imread(file)
                    slices = img.shape[0]

                    config.simulation.z_slices = slices
                    config.simulation.z_values = [i - slices // 2 for i in range(slices)]
            else:
                raise ValueError(f'Input file not found "{file}"')
            i += 1

    except ValueError as e:
        if last_frame != -1 and len(image_paths) != last_frame - first_frame + 1:
            raise e

    return image_paths


class CellUniverse:
    def __init__(self, args: Args):
        # set up dask client
        # if not args.no_parallel:
        #     from dask.distributed import Client, LocalCluster
        #     if not args.cluster:
        #         cluster = LocalCluster(
        #             n_workers=args.workers, threads_per_worker=1,
        #         )
        #         client = Client(cluster)
        #     else:
        #         cluster = args.cluster
        #         client = Client(cluster)
        #         client.restart()
        # else:
        #     client = None
        self.client = None

        # --------
        # Config
        # --------
        config = load_config(args.config)
        image_file_paths = get_image_file_paths(args.input, args.first_frame, args.last_frame, config)

        # --------
        # Cells
        # --------
        cellFactory = CellFactory(config)
        cells = cellFactory.create_cells(args.initial, z_offset = config.simulation.z_slices // 2, z_scaling = config.simulation.z_scaling)
        # --------
        # Lineage
        # --------
        self.lineage = Lineage(cells, image_file_paths, config, args.output, args.continue_from)

    def run(self):
        current_time = time.time()
        for frame in range(len(self.lineage)):
            self.lineage.optimize(frame)
            self.lineage.copy_cells_forward(frame + 1)
            self.lineage.save_images(frame)
            # self.lineage.save_cells(frame) // TODO: Figure out why this isn't working

        print(f"Time elapsed: {time.time() - current_time:.2f} seconds")
