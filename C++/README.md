# 3D CellUniverse CPP-VERSION

## Quick Start
### Package Required
- this project require package yaml-cpp
- option1: pull the package to local directory (default)
    - In the terminal type in `cd 3d/C++` to go to the root of C++ CellUniverse, then add the external YAML-CPP library
      ``` bash
        mkdir lib && cd lib
        git clone https://github.com/jbeder/yaml-cpp
      ```
- option2: install it through package manager such as homebrew(MacOS) or apt(Ubuntu), make it detect the one on this computer, no need for pull from github
  ```cmake
  # commend out the line bellow in CmakeLists.txt
   add_subdirectory(lib/yaml-cpp)
  
  # uncommend the line bellow in CmakeLists.txt
   find_package(yaml-cpp REQUIRED)
  ```
### Using ICS openlab
- ssh into your ICS openlab. (Recommend to use vscode remote development tool: https://code.visualstudio.com/docs/remote/ssh)
    - Notice you should ssh into the circinus-28 machine to avoid dependency issue (ssh <netid>@circinus-28.ics.uci.edu)
- clone the repo and cd into the project folder
- create build directory and compile the project using cmake (assume current directory is C++/lib created in previous step)
  ```bash
    cd ..
    mkdir build && cd build
    cmake -S .. -B .
    cmake --build . -j $(nproc) // build with all available 
  ```
- Run the bash script in the examples folder (assume current director is C++)
  ```bash
    cd examples
    ./runcpp.sh
  ```
- Change the mode of the bash script if you don't have the permission by running the following command
```bash
      chmod 755 runcpp.sh
   ```
- Wait for celluniverse to finish executing and you can check the result in the `output/` folder inside the current
  example

### Using Mac OS:
- clone the repo and cd into the project folder
- In the terminal type in `cd C++` to go to the root of C++ CellUniverse
- Add the external YAML-CPP library
  ```zsh
  mkdir lib && cd lib
  git clone https://github.com/jbeder/yaml-cpp
  ```

- download cmake and install
- use the following command to set make path:
  ```zsh
  PATH="/Applications/CMake.app/Contents/bin":"$PATH"
  ```
- install OpenCV using commend bellow (homebrew required)
  ```zsh
   brew install opencv
  ```
- create build directory and compile the project into `build` directory using cmake (assume current directory is C++/lib created in previous steps)
  ```zsh
  cd ..
  mkdir build && cd build
  cmake -S .. -B .
  cmake --build .
  ``` 


- Run the bash script in the examples folder
  ``` zsh
   cd ../../examples/3d
  ./runcpp.sh
  ```
    - Change the mode of the bash script if you don't have the permission by running the following command
  ```zsh
  chmod 755 runcpp.sh
  ```

- Wait for celluniverse to finish executing and you can check the result in the `output/` folder inside the current
  example

  ## 0211 Update

### Generate `initial_auto.csv` for new 3D datasets

I enhance the codes to read TIFF cell data with all kind of slices number & naming pattern. I created 2 commands script for easy testing the codes.

I added a small python script to automatically extract the initial 3D coordinates of cells from a TIFF stack and generate `initial_auto.csv`.

This script:
- loads a multi-slice TIFF stack
- finds the best z-slice (highest contrast / signal)
- detects connected components
- computes centroid (x, y), z index, and equivalent radius
- writes them into the same format required by `initial.csv`

```python
import tifffile as tiff
import numpy as np
import uuid
import csv
from skimage import filters, measure

TIFF_PATH = "t000.tif"   # starting frame
OUT_CSV = "initial_auto.csv"

stack = tiff.imread(TIFF_PATH)

# choose best z slice by simple variance score
scores = [np.var(slice_) for slice_ in stack]
best_z = int(np.argmax(scores))
best_slice = stack[best_z]

# threshold + connected components
thresh = filters.threshold_otsu(best_slice)
binary = best_slice > thresh
labels = measure.label(binary)
props = measure.regionprops(labels)

def equiv_radius(area):
    return np.sqrt(area / np.pi)

lines = []
for prop in props:
    cell_id = uuid.uuid4().hex
    y, x = prop.centroid
    r = equiv_radius(prop.area)
    height = 6.952   # keep consistent with config for now

    lines.append([
        TIFF_PATH,
        cell_id,
        float(x),
        float(y),
        float(best_z),
        float(r),
        height,
        "None",
        "None"
    ])

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(lines)

print("Generated:", OUT_CSV)
