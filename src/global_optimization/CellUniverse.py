import time
import numpy as np
import yaml
from .Config import load_config

class CellUniverse:
    def __init__(self, args):
        # --------
        #   Args
        # --------
        if (args.start_temp is not None or args.end_temp is not None) and args.auto_temp == 1:
            raise Exception("when auto_temp is set to 1(default value), starting temperature or ending temperature should not be set manually")

        # Make required folders
        if not args.output.is_dir():
            args.output.mkdir()
        if not args.bestfit.is_dir():
            args.bestfit.mkdir()
        if args.residual and not args.residual.is_dir():
            args.residual.mkdir()

        # set seed
        seed = int(time.time() * 1000) % (2**32)
        if args.seed is not None:
            seed = args.seed
        np.random.seed(seed)
        print(f"Seed: {seed}")

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
        self.config = load_config(args.config)

