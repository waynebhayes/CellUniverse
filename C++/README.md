# 3D CellUniverse CPP-VERSION

## Directory Structure

```
C++/
  config/         # config.yaml, initial.csv, user_input_configurations.ini
  scripts/        # run_celluniverse.sh and other shell scripts
  data/input/     # input images (original_data, etc.)
  src/            # source files (main.cpp, Lineage.cpp, Spheroid.cpp, Frame.cpp, ...)
  includes/       # header files
  lib/            # external libraries (yaml-cpp)
  build/          # cmake build output
  deprecated/     # old unused files preserved for reference
  docs/           # changelogs and documentation
```

## Quick Start

### Dependencies
- yaml-cpp: clone into lib/ or install via package manager
- OpenCV: brew install opencv (macOS) or apt install libopencv-dev (Linux)
- CMake 3.10+

### Build
```bash
cd C++
mkdir -p build && cd build
cmake -S .. -B .
cmake --build . -j $(nproc)
```

### Run (from C++/)
```bash
scripts/run_celluniverse.sh config/user_input_configurations.ini default
# or interactive mode:
scripts/run_celluniverse.sh -i
# or direct:
./build/celluniverse 1 19 data/input/original_data output/ config/config.yaml config/initial.csv
```

### ICS Openlab
```bash
ssh <netid>@circinus-28.ics.uci.edu
# then follow build and run steps above
```
