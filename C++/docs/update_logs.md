## 02/11/2026 Update
1. analyzed the existing C++ file and provided report for the program structure and functionality
2. revised the program auto execution scripts to make it easier to use without conflict
3. everyone's path and parameter is managed in an .ini file
4. provided an script for remove all old outputs but only keep the latest one
5. revised the REDME.md, archived update log
## 02/11/2026 Update

### Generate `initial_auto.csv` for new 3D datasets

enhance the codes to read TIFF cell data with all kind of slices number & naming pattern. I created 2 commands script for easy testing the codes.

added a small python script to automatically extract the initial 3D coordinates of cells from a TIFF stack and generate `initial_auto.csv`.

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