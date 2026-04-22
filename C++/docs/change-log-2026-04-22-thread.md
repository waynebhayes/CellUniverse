# Change Log: 2026-04-22 Thread

This document records the code and behavior changes made during the 2026-04-22 working thread.

Unless another document is explicitly requested later in this same conversation, future "document the changes" requests should append to this file.

## OpenMP Adaptive Cube Pooling Fix And Refinement

### Problem

The build failed in `src/ImageHandler.cpp` with GCC/OpenMP errors like:

```text
error: not enough for loops to collapse
error: 'gz' was not declared in this scope
```

The failing code used `#pragma omp parallel for collapse(2)` above loops where statements such as:

```cpp
const int z0 = gz * cubeSize;
const int z1 = std::min(depth, z0 + cubeSize);
```

appeared between the `gz` loop and the nested `gy` loop. OpenMP `collapse(n)` requires the collapsed loops to be perfectly/directly nested, with no ordinary statements between the loop headers.

### Initial Fix

The illegal `collapse(2)` clauses were temporarily removed from the affected pragmas so GCC could compile the code. This confirmed the issue was the OpenMP loop structure, not the parallel safety of the work itself.

### Legal `collapse(2)` Trial

The per-`gz` setup variables were moved inside the `gy` loop body. That made the `gz` and `gy` loops directly nested and restored legal `collapse(2)` behavior.

### Final `collapse(3)` Trial

The setup variables were moved one level farther inward so the structure became:

```cpp
for (int gz = 0; gz < gridZ; ++gz)
{
    for (int gy = 0; gy < gridY; ++gy)
    {
        for (int gx = 0; gx < gridX; ++gx)
        {
            const int z0 = gz * cubeSize;
            const int z1 = std::min(depth, z0 + cubeSize);
            const int y0 = gy * cubeSize;
            const int y1 = std::min(rows, y0 + cubeSize);
            const int x0 = gx * cubeSize;
            const int x1 = std::min(cols, x0 + cubeSize);
            ...
        }
    }
}
```

The affected adaptive cube pooling loops now use legal `collapse(3)` forms:

- cube statistics computation
- cube reweighting and voxel fill
- isolated bright cube neighbor-check pass
- isolated bright cube zeroing pass

### Behavioral Assessment

Moving `z0/z1/y0/y1` inward does not change the mathematical logic. The same cube bounds are computed for each `(gz, gy, gx)` tuple. The values are recomputed more often, but the resulting voxel ranges are unchanged.

Parallel safety remains stable because each cube writes to its own `cubeStats`, `pooledCubeValues`, `clearCube`, or disjoint voxel region in `pooled`. The only reductions involved are integer counters, so there is no floating-point reduction nondeterminism introduced by this change.

### Verification

The project was rebuilt successfully with:

```bash
cmake --build build -j20
```

## Additional Parallelization Review

The rest of the OpenMP usage was inspected for possible similar refinements.

### Good Future Candidate

`src/Frame.cpp` contains bounding-box cost loops that currently parallelize only over `z`. These loops may benefit from `collapse(2)` over `(z, y)` because a cell bounding box can have too few z-slices to fully occupy many threads.

Recommended future shape:

```cpp
#pragma omp parallel for collapse(2) reduction(+:totalCost) schedule(static)
for (int z = bbox.zMin; z <= bbox.zMax; ++z) {
    for (int y = bbox.yMin; y <= bbox.yMax; ++y) {
        ...
        for (int x = bbox.xMin; x <= bbox.xMax; ++x) {
            ...
        }
    }
}
```

`collapse(3)` was not recommended as the first option there because keeping the inner `x` loop serial preserves contiguous row access and SIMD/vectorization opportunities.

### Larger Future Candidate

The bright-box scan in `src/ImageHandler.cpp` around the signal-guided center detection also has a 3D grid structure, but it pushes into a shared `boxes` vector. Parallelizing that safely would require thread-local vectors or a preallocated result array followed by compaction.

## Frame Export Pipeline Switches

### Goal

Add YAML-controlled export format switches so output frames can be exported as PNG slices, TIFF stacks, or both.

### Config Fields

Two booleans were added under `simulation`:

```yaml
export_frame_png: true
export_frame_tiff: false
```

Defaults preserve the prior behavior by keeping PNG enabled and TIFF disabled.

### Output Layout

PNG output is written under:

```text
output/png/real/<frame>/<slice>.png
output/png/synth/<frame>/<slice>.png
```

TIFF stack output is written under:

```text
output/tiff/real/<frame>.tif
output/tiff/synth/<frame>.tif
```

If both switches are enabled, both export branches run. If only TIFF is enabled, only stack files are written. If only PNG is enabled, only per-slice PNGs are written.

### `.tif` Extension

The TIFF stack export was changed from `.tiff` to `.tif` to match the requested extension and the project’s common TIFF input naming.

## Napari-Friendly TIFF Export

### Motivation

Napari can generally read TIFF, but OpenCV’s default `imwritemulti` output may not be ideal for all napari reader/plugin combinations. The initial TIFF writer used OpenCV defaults and ignored the write result.

### Export Improvements

The TIFF stack branch was made more reader-friendly by:

- converting pages to single-channel grayscale
- converting output to `uint8`
- validating that all slices have the same size
- writing uncompressed TIFF pages instead of relying on OpenCV’s default LZW compression
- checking `cv::imwritemulti(...)` return values
- throwing a runtime error if TIFF export fails

The exported TIFF intensity mapping is:

```text
stored_uint8 = clamp(internal_float * 255, 0, 255)
```

So exported PNG/TIFF files should be interpreted as 8-bit intensity images in the `0..255` range.

### Remaining Limitation

OpenCV does not write rich OME-TIFF/ImageJ metadata. If napari still has trouble reading the stacks, the next step would be an OME-TIFF export path using a TIFF writer that supports axis metadata.

## Initial CSV Parsing Robustness

### Problem

The original `CellFactory::createCells` reader skipped the first line as a header but parsed all data rows by fixed column position. This made it unsafe to reorder columns or insert new columns before expected fields.

### Header-Aware Parsing

The initial CSV parsing was made header-aware. It now recognizes columns by normalized header names and aliases rather than relying only on fixed positions.

Supported regular initial-cell aliases include:

```text
filePath / file / frame / frameName / image / imageFile / imagePath
cellName / name / id / cellId / label
x
y
z
aRadius / majorRadius / radiusA / radius / r
bRadius / radiusB / middleRadius / intermediateRadius
cRadius / minorRadius / radiusC / zRadius
```

Supported Napari-style aliases include:

```text
cell_type / cellType / type
z
y
x
```

### Fallback Compatibility

The parser still supports the older positional formats:

```text
filePath, cellName, x, y, z, aRadius, cRadius
filePath, cellName, x, y, z, aRadius, bRadius, cRadius
cell_type, z, y, x
```

Bad rows are skipped with warnings instead of crashing the whole program on parse errors.

### `bRadius` Compatibility

`bRadius` support was made explicit:

- if a `bRadius`/alias column exists and has a valid numeric value, it is used
- if the column is missing, empty, `None`, `null`, or `nan`, `bRadius` falls back to `aRadius`

This preserves oblate-compatible behavior while allowing triaxial initialization.

## Initial CSV File Reformatting

### `config/initial_auto.csv`

This file previously had no header, so the first data row would be consumed as the header and skipped. It also contained trailing `None` fields.

It was reformatted to:

```csv
file,name,x,y,z,aRadius,bRadius,cRadius
```

and each row now fills `bRadius` with the same value as `aRadius`.

### `config/initial_embryo.csv`

The Napari-style header was changed from:

```csv
cell_type,z,y,x
```

to:

```csv
cellType,z,y,x
```

Both spellings are supported by the parser, but `cellType` matches one of the explicit aliases cleanly.

## CSV Pipeline Extraction

### Motivation

After adding header aliases, robust field lookup, `bRadius` fallback, Napari support, and bad-row handling, CSV parsing became too large for `CellFactory.cpp`.

### New Handler

A new handler was introduced:

```text
includes/CsvHandler.hpp
src/CsvHandler.cpp
```

It follows the same architectural idea as `ImageHandler`: keep file-format parsing and preprocessing out of the factory/model logic.

### New Data Structure

`CsvHandler` returns `InitialCellRecord` objects:

```cpp
struct InitialCellRecord
{
    std::string filePath;
    std::string cellName;
    float x;
    float y;
    float z;
    float aRadius;
    float bRadius;
    float cRadius;
};
```

### Current Responsibility Split

`CsvHandler` owns:

- CSV splitting and trimming
- simple quoted field handling
- header normalization
- alias lookup
- named regular initial CSV parsing
- named Napari CSV parsing
- old positional regular CSV fallback
- old positional Napari fallback
- bad-row warnings
- `bRadius` fallback behavior

`CellFactory` now owns:

- reading `InitialCellRecord` objects from `CsvHandler`
- converting those records into `EllipsoidParams`
- constructing `Ellipsoid` instances
- grouping cells by frame path

### Build Integration

`src/CsvHandler.cpp` was added to the main `celluniverse` executable in `CMakeLists.txt`.

### Verification

The refactor was verified with:

```bash
cmake --build build -j20
```

The build completed successfully.
