# CellUniverse

CellUniverse is a cell tracking system that fits geometric cell models to time-lapse microscopy images using Monte Carlo stochastic optimization. Given an initial guess of cell positions, sizes, and orientations, it iteratively perturbs a synthetic rendering to match the real image, detecting cell divisions along the way to produce a complete lineage map across all frames.

Published paper: [Cell Universe - MDPI Algorithms 2022](https://www.mdpi.com/1999-4893/15/2/51) (also available as [PDF](docs/Cell-Universe-published.pdf))

## How It Works

1. **Initialize** -- Provide the first frame's cell positions, sizes, and orientations in a CSV file.
2. **Render** -- CellUniverse generates a synthetic image from the current cell model parameters.
3. **Optimize** -- A stochastic loop randomly perturbs cell parameters (position, size, orientation) and keeps changes that reduce the L2 difference between the synthetic and real images.
4. **Detect Divisions** -- PCA analysis on bright regions detects elongated cells likely to divide. Candidate daughter cells are placed, refined through a burn-in phase, and accepted if cost improves.
5. **Advance** -- Cell states are carried forward to the next frame and the process repeats.

## Project Structure

```
CellUniverse/
  C++/              # Active 3D implementation (oblate ellipsoid models)
  Python/            # Legacy 2D/3D implementations (no longer maintained)
  docs/              # Published paper (PDF)
  debug-synthetic-images/   # Debug output directory
```

## C++ (Active)

The C++ implementation tracks cells in 3D using triaxial ellipsoid models fitted to multi-page TIFF z-stacks. It is the only actively developed version.

**Key features:**
- Triaxial ellipsoid cell models with independent radii (a, b, c) and full 3D rotation
- Sigmoid contrast preprocessing with percentile-based auto-calibration
- Z-interpolation (e.g., 33 raw slices to 225 interpolated slices)
- Unified stochastic optimization loop with adaptive split probability driven by PCA elongation
- Local-frame PCA split detection with dual-direction candidate generation
- Continuous overlap penalty (no hard rejection gates)
- Per-cell brightness tracking via exponential moving average from the real image
- Layered soft guards for split validation (pre/post burn-in)

**Dependencies:** CMake 3.10+, C++17, OpenCV, yaml-cpp

### Quick Start

```bash
# Clone and set up yaml-cpp
cd C++
mkdir -p lib && cd lib
git clone https://github.com/jbeder/yaml-cpp
cd ..

# Build
mkdir -p build && cd build
cmake -S .. -B .
cmake --build . -j $(nproc)
cd ..

# Run
scripts/run_celluniverse.sh config/user_input_configurations.ini default
```

See [C++/README.md](C++/README.md) for full build instructions (macOS, Linux, ICS Openlab), configuration details, and alternative setups (system-installed yaml-cpp, interactive mode).

### Configuration

Cell tracking behavior is controlled by `C++/config/config.yaml` with three sections:
- **cell** -- Perturbation parameters (position, size, orientation step sizes and probabilities)
- **simulation** -- Image preprocessing (blur, sigmoid contrast, z-scaling, adaptive background)
- **prob** -- Split detection thresholds, overlap penalty weight, burn-in iterations

Initial cell positions are specified in `C++/config/initial.csv`.

### Output

Each frame produces:
- `output/real/{frame}/` -- Real image slices with cell outlines overlaid
- `output/synth/{frame}/` -- Synthetic rendering from the fitted model
- `output/cells.csv` -- Per-frame cell states (name, position, radii, orientation)

Cell lineage is encoded in cell names: a parent cell `abc` produces daughters `abc0` and `abc1`.

## Python (Legacy)

The `Python/` directory contains the original implementations:
- **2d/** -- Comprehensive 2D cell tracking for bacteria-shaped (bacilli) cells. Includes regression tests, examples, a cell labeling tool for creating initial configurations, and parallel optimization support via Dask.
- **3d/** -- Experimental 3D prototype. Includes a cell labeling tool for 3D initial configurations.

These are **no longer maintained**. The 2D code once worked well with rod-shaped bacterial cells but has broken due to dependency drift. The 3D Python code was experimental and was abandoned in favor of the C++ rewrite. See [Python/README.md](Python/README.md) for historical reference.

## Citation

If you use CellUniverse in your research, please cite:

> Li, T.; Dong, H.; Feng, Y.; Yang, J.; Hayes, W.B. CellUniverse: An Automated Framework for Quantifying Cell Migration and Division from Time-Lapse Videos. *Algorithms* **2022**, *15*, 51.
