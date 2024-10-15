from typing import Optional, TypeVar

import typed_argparse as tap

from pathlib import Path
import time
import numpy as np
import multiprocessing

class Args(tap.TypedArgs):
    # Required arguments
    input: str = tap.arg("-i", help="Input file pattern (e.g. 'input%03d.png')")
    output: Path = tap.arg("-o", help="Path to the output directory")
    config: Path = tap.arg("-c", help="Path to the config file")
    initial: Path = tap.arg("-I", help="Path to the initial cell configuration file")

    # Optional arguments
    debug: Optional[Path] = tap.arg('-d', help="Path to the debug directory", default=None)
    first_frame: int = tap.arg('-ff', help="First frame to analyze", default=0)
    last_frame: int = tap.arg('-lf', help="Last frame to analyze (defaults to the last frame)", default=-1)
    workers: int = tap.arg('-w', help="Number of workers to use (defaults to the number of cores)", default=-1)
    jobs: int = tap.arg('-j', help="Number of jobs to run in parallel (defaults to the number of workers)", default=-1)
    cluster: str = tap.arg('-C', help="Address of the cluster to connect to", default='')
    no_parallel: bool = tap.arg('--no_parallel', '-np', help="Disable parallelization", default=False)
    auto_temp: bool = tap.arg('--auto_temp' '-at', help="Automatically determine the starting and ending temperatures", default=False)
    start_temp: Optional[float] = tap.arg('-st', help="Starting temperature", default=None)
    end_temp: Optional[float] = tap.arg('-et', help="Ending temperature", default=None)
    residual: Optional[Path] = tap.arg('-r', help="Path to save the residual directory", default=None)
    continue_from: int = tap.arg('-cf', help="Frame to start from (defaults to first)", default=-1)
    seed: Optional[int] = tap.arg('-s', help="Random seed", default=None)
    batches: int = tap.arg('-b', help="Number of batches to run", default=1)

    def __post_init__(self):
        # Validate arguments
        if self.auto_temp and (self.start_temp is not None or self.end_temp is not None):
            raise Exception("when auto_temp is on (default value), starting temperature or ending temperature should not be set manually")
        elif not self.auto_temp and (self.start_temp is None or self.end_temp is None):
            raise Exception("when auto_temp is off, starting temperature and ending temperature should be set manually")
        if self.first_frame > self.last_frame and self.last_frame >= 0:
            raise ValueError('Invalid interval: frame_first must be less than frame_last')
        elif self.first_frame < 0:
            raise ValueError('Invalid interval: frame_first must be greater or equal to 0')

        # set seed
        seed = int(time.time() * 1000) % (2**32)
        if self.seed is not None:
            seed = self.seed
        np.random.seed(seed)
        print(f"Seed: {seed}")

        if self.workers == -1:
            self.workers = multiprocessing.cpu_count()

        if self.jobs == -1:
            if self.cluster:
                raise ValueError('-j/--jobs is required for non-local clusters')
            else:
                self.jobs = self.workers
