This is the OLD Python code
=============

The 2d Python code is comprehensive, and the 3d code is only prototype, but both have stopped working (Thanks, Python, for being so unstable!)

Prerequisites:
-----------
Requires Python 3.11+.

Recommendations
- [Conda](https://docs.conda.io/en/latest/miniconda.html)
- [pyenv](https://github.com/pyenv/pyenv#homebrew-in-macos)

Requires Pip.
https://pip.pypa.io/en/stable/quickstart/

Quickstart:

1. Install Pipenv
```bash
$ pip install pipenv
```

2. Install required packages using Pipenv
```bash
$ pipenv install
```
This will create a virtual environment with all the required Python packages

3. Activate Pipenv environment
```bash
$ pipenv shell
```

Steps to get a new video running:
---------------------------------
1. Separate your video into still images, one image per frame and put them in a directory with chronological names like `frame000.png`, `frame001.png`, etc.
2. Cell Universe needs a starting "guess" for the number, location, size, and orientation of the cells in the first frame. You can do this visually using the cell labeling tool Python program. This will create an initial properties file with all that information. See the `2d/tools/cell_labeling_tool/README.md` for more information about the cell labeling tool and `2d/examples/canonical/initial.csv` for an example initial properties file generated by the cell labeling tool.
```bash
(venv)$ python3 2d/tools/cell_labeling_tool/cell_labeling_tool.py
```
3. Create a Cell Universe config file and fill it in with settings that fits your cell video. See `2d/examples/canonical/global_optimizer_config.json` (and the comments in the file) and other config JSON files in the `examples` directory for examples.
    * It is important to change settings in the config file like the "Global Setttings" and the "Bacilli Settings" to match your cell video, or else Cell Universe will not function properly.
4. Run Cell Universe, giving it the input directory containing the frames, the initial properties file created above, the config file, as well as the other required arguments (see in the Usage section).
    * An example command is at the bottom of this README and more in the `run.sh` script files in the `2d/examples` directory.

There are several examples in subdirectories below the directory `2d/regression-tests` and `2d/examples`.

Usage
-----

Command line help:

``` sourceCode
usage: main.py [-h] [-d DIRECTORY] [-ff N] [-lf N] [--dist] [-w WORKERS] [-j JOBS] [--keep KEEP]
               [--strategy STRATEGY] [--cluster CLUSTER] [--no_parallel] [--global_optimization]
               [--binary] [--graySynthetic] [--phaseContrast] [-ta TEMP] [-ts START_TEMP]
               [-te END_TEMP] [-am {none,frame,factor,const,cost}] [-r FILE] [--lineage_file FILE]
               [--continue_from N] [--seed N] [--batches N] -i PATTERN -o DIRECTORY -c FILE -x FILE -b FILE
               
optional arguments:
  -h, --help            show this help message and exit
  -d DIRECTORY, --debug DIRECTORY
                        path to the debug directory (enables debug mode)
  -ff N, --frame_first N
                        starting image (default: 0)
  -lf N, --frame_last N
                        final image (defaults to until last image)
  --dist                use distance-based objective function
  -w WORKERS, --workers WORKERS
                        number of parallel workers (defaults to number of
                        processors)
  -j JOBS, --jobs JOBS  number of jobs per frame (defaults to --workers/-w)
  --keep KEEP           number of top solutions kept (must be equal or less
                        than --jobs/-j)
  --strategy STRATEGY   one of "best-wins", "worst-wins", "extreme-wins"
  --cluster CLUSTER     dask cluster address (defaults to local cluster)
  --no_parallel         disable parallelism
  --global_optimization
                        global optimization
  --binary              input image is binary
  --graySynthetic       enables the use of the grayscale synthetic image for
                        use with non-thresholded images
  --phaseContrast       enables the use of the grayscale synthetic image for
                        phase contract images
  -ta TEMP, --auto_temp TEMP
                        auto temperature scheduling for the simulated
                        annealing
  -ts START_TEMP, --start_temp START_TEMP
                        starting temperature for the simulated annealing
  -te END_TEMP, --end_temp END_TEMP
                        ending temperature for the simulated annealing
  -am {none,frame,factor,const,cost}, --auto_meth {none,frame,factor,const,cost}
                        method for auto-temperature scheduling
  -r FILE, --residual FILE
                        path to the residual image output directory
  --lineage_file FILE   path to previous lineage file
  --continue_from N     load already found orientation of cells and start from
                        the continue_from frame
  --seed N              seed for random number generation
  --batches N           number of batches to split each frame into for
                        multithreading

required arguments:
  -i PATTERN, --input PATTERN
                        input filename pattern (e.g. "image%03d.png")
  -o DIRECTORY, --output DIRECTORY
                        path to the output directory
  -c FILE, --config FILE
                        path to the configuration file
  -x FILE, --initial FILE
                        path to the initial cell configuration
  -b FILE, --bestfit FILE
                        path to the best fit synthetic image output directory
```

Examples
--------

``` sourceCode
python3 src/main.py --frame_first 0 --frame_last 13 --input "./examples/canonical/input/gray/frame%03d.png" \
  --output "./examples/canonical/output" --bestfit "./examples/canonical/output/bestfit" --config \
  "./examples/canonical/global_optimizer_config.json" --initial "./examples/canonical/initial.csv" --no_parallel \
  --graySynthetic --global_optimization
```
--------
Guide of running the examples
-----
There are currently four running examples located in the example folders. Each example is packed into a folder, which consists the input images, the initial cell configuration file, the configuration files, and the scripts to run the example. The scripts doesn't do anything magic. It's simply a wrapper of the command to run the program.

To run the examples:

Step 1: Activate the python environment. To do this, first locate to the root directory of this project, and run
```
pipenv shell
```

Step 2: Move to the folder consists the example that you want to run. Here, use the canonical one as an example.
```
cd 2d/examples/canonical
```
Step 3: Use the script to run the example

For Windows Users (Powershell):
```
.\run.ps1
```
For Linux Users (Bash):
```
.\run.sh
```

Step 4: Wait until the program finish. The output is located in the output folder in the example folder.

ps. if your ConfigTypes.py has errors, please replace the lines 18-21 with this:
padding: int = 0
z_scaling: float = 1.0
blur_sigma: float = 0.0
z_slices: int = -1