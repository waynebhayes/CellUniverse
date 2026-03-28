# 3D CellUniverse C++ Project Analysis

## 1. What is it?

This project, named "3D CellUniverse," is a C++ application for 3D cell tracking in image sequences. It aims to initialize a state of cells and then iteratively optimize their properties (like position and size) across a series of image frames. It uses different optimization methods to match a rendered model of the cells to the actual image data.

## 2. What do the files do?

Here's a breakdown of the key files and their functionality:

*   **`src/main.cpp`**: The main entry point of the application. It parses command-line arguments, loads the initial configuration, and starts the main optimization loop.
*   **`includes/Lineage.hpp`**: Manages the entire state of the cell colony over time, including the relationships between cells (e.g., parent-daughter relationships after division) and their states at each frame.
*   **`includes/Frame.hpp`**: Represents a single frame in the time-series data. It holds the image data and the state of all cells at that specific time point.
*   **`includes/Cell.hpp`**: The base class for all cell types. It defines the common interface for a cell, such as its position, size, and methods for drawing or calculating its properties.
*   **`includes/CellFactory.hpp`**: A factory class responsible for creating instances of different cell types (e.g., `Sphere`, `Spheroid`) based on the configuration.
*   **`CMakeLists.txt`**: Defines how the project is compiled and linked. It specifies the source files, include directories, and external dependencies like OpenCV and yaml-cpp.
*   **`examples/runcpp.sh`**: An execution script that shows how to run the compiled program with the correct arguments and data.

The file `args.cpp` appears to be unused as it is not part of the main executable build defined in `CMakeLists.txt`.

## 3. How to use it?

To use this project, you need to:

1.  **Build the project:** Use CMake to build the project. The `CMakeLists.txt` file is configured to do this. You will need to have OpenCV and `yaml-cpp` installed.
2.  **Run the application:** The `examples/runcpp.sh` script shows how to run the compiled program with the correct arguments and data.

---

# File by File Analysis

## `C++/includes/Cell.hpp`

### `Cell()`
The default constructor.

### `~Cell()`
The default destructor.

### `draw(cv::Mat& image, SimulationConfig simulationConfig, cv::Mat* cellMap = nullptr, float z = 0) const`
A pure virtual function that, when implemented in a derived class, will draw the cell on a given image. It takes the image, simulation configuration, an optional cell map, and a z-coordinate as input.

### `drawOutline(cv::Mat& image, float color, float z = 0) const`
A pure virtual function that, when implemented, will draw the outline of the cell on a given image. It takes the image, a color, and a z-coordinate as input.

### `getPerturbedCell() const`
A pure virtual function that should return a new cell that is a small modification of the current cell. This is likely used for optimization algorithms.

### `getParameterizedCell(std::unordered_map<std::string, float> params = {}) const`
A pure virtual function that should return a new cell with parameters specified in the input map.

### `getSplitCells() const`
A pure virtual function that should return a tuple of two new cells and a boolean, representing the result of a cell division.

### `getCellParams() const`
A pure virtual function that should return the parameters of the cell.

### `checkIfCellsValid(const std::vector<Cell*>& cells)`
A static function that is supposed to check if a vector of cells is valid. The current implementation returns `false`.

### `calculateCorners() const`
A pure virtual function that should calculate the corners of the cell.

### `calculateMinimumBox(Cell& perturbed_cell) const`
A pure virtual function that should calculate the minimum bounding box that encloses the current cell and a perturbed cell.

## `C++/includes/Bacilli.hpp`
This file defines the `Bacilli` class, which seems to be a type of cell. However, this class does not correctly implement the `Cell` interface defined in `Cell.hpp`.

### `Bacilli(std::string name, double x, double y, double width, double length, double rotation, double split_alpha = 0, double opacity = 0)`
The constructor for the `Bacilli` class.

### `_refresh()`
A private helper function, likely to update the internal geometry variables.

### `draw()`
An override of a `draw` method, but with a signature that doesn't match the one in the `Cell` base class.

### `drawOutline()`
An override of a `drawOutline` method, but with a signature that doesn't match the one in the `Cell` base class.

### `split(double alpha)`
A method to split the cell. This method is not in the `Cell` base class.

### `combine(Bacilli* cell)`
A method to combine this cell with another `Bacilli` cell. This method is not in the `Cell` base class.

### `simulated_region()`
This method's purpose is unclear from the name, and it is not part of the `Cell` base class.

## `C++/includes/Sphere.hpp`
This file defines a `Sphere` class, which represents a spherical cell. It also defines a `SphereParams` class that inherits from `CellParams` and holds the parameters for a sphere. The `Sphere` class in this file is not fully compatible with the `Cell` interface defined in `Cell.hpp`.

### `Sphere(const SphereParams &init_props)`
The constructor, taking `SphereParams` as input.

### `Sphere()`
A default constructor.

### `printCellInfo() const`
Prints information about the sphere to the console.

### `getRadiusAt(double z) const`
Calculates the radius of the circle formed by the intersection of the sphere and a plane at a given `z` coordinate.

### `draw(cv::Mat &image, SimulationConfig simulationConfig, cv::Mat *cellMap = nullptr, float z = 0) const`
Draws the sphere on a given image.

### `drawOutline(cv::Mat &image, float color, float z = 0) const`
Draws the outline of the sphere.

### `getPerturbedCell() const`
Returns a perturbed `Sphere` object.

### `getParameterizedCell(std::unordered_map<std::string, float> params) const`
Returns a `Sphere` object with specified parameters.

### `performPCA(...) const`
Performs Principal Component Analysis on a set of points.

### `calculateContours(...)`
A static method to calculate 3D contours from image slices.

### `getSplitCells(...) const`
Returns two new `Sphere` objects and a boolean, representing a cell division.

### `checkConstraints() const`
Checks if the sphere's parameters are within allowed constraints.

### `getCellParams() const`
Returns the parameters of the sphere as a `CellParams` object.

### `calculateCorners() const`
Calculates the corners of the sphere's bounding box.

### `checkIfCellsValid(const std::vector<Sphere> &spheres)`
Checks if a vector of spheres is valid (i.e., they don't overlap).

### `calculateMinimumBox(Sphere &perturbed_cell) const`
Calculates the minimum bounding box enclosing the current sphere and a perturbed one.

### `checkIfCellsOverlap(const std::vector<Sphere> &spheres)`
A static method to check for overlaps in a vector of spheres.

## `C++/includes/Spheroid.hpp`
This file defines a `Spheroid` class, representing a spheroid-shaped cell, and a `SpheroidParams` class for its parameters. The `Spheroid` class is not fully compatible with the `Cell` interface from `Cell.hpp`.

### `Spheroid(const SpheroidParams &init_props)`
The constructor, taking `SpheroidParams` as input.

### `Spheroid()`
A default constructor.

### `printCellInfo() const`
Prints information about the spheroid to the console.

### `getRadiusAt(double z) const`
Calculates the radius of the ellipse formed by the intersection of the spheroid and a plane at a given `z` coordinate.

### `draw(...) const`
Draws the spheroid on a given image.

### `drawOutline(...) const`
Draws the outline of the spheroid.

### `getPerturbedCell() const`
Returns a perturbed `Spheroid` object.

### `getParameterizedCell(...) const`
Returns a `Spheroid` object with specified parameters.

### `performPCA(...) const`
Performs Principal Component Analysis.

### `getSplitCells(...) const`
Simulates cell division.

### `checkConstraints() const`
Checks if the spheroid's parameters are within allowed constraints.

### `getCellParams() const`
Returns the parameters of the spheroid as a `CellParams` object.

### `calculateCorners() const`
Calculates the corners of the spheroid's bounding box.

### `checkIfCellsValid(...)`
Checks if a vector of spheroids is valid.

### `calculateMinimumBox(...) const`
Calculates the minimum bounding box for the current and a perturbed spheroid.

### `checkIfCellsOverlap(...)`
A static method to check for overlaps in a vector of spheroids.

### `getAxisRadiusAt(double z, double axis_length) const`
An additional method specific to spheroids to get the radius along a specific axis at a given z-coordinate.

## `C++/includes/CellFactory.hpp`
This file defines a `CellFactory` class, which is responsible for creating cells.

### `CellFactory(const BaseConfig& config)`
The constructor, which takes a `BaseConfig` object.

### `createCells(const Path &init_params_path, int z_offset = 0, float z_scaling = 1.0)`
This method reads initial cell parameters from a file and creates a map of `Spheroid` cells.

## `C++/includes/Frame.hpp`
This file defines the `Frame` class, which represents a single frame in a time-series of images.

### `interpolateSlices(const cv::Mat& slice1, const cv::Mat& slice2, std::vector<cv::Mat>& processedSlices, int numInterpolations)`
This function interpolates between two image slices to create a number of intermediate slices.

### `Frame(const std::vector<cv::Mat> &realFrame, const SimulationConfig &simulationConfig, const std::vector<Spheroid> &cells, const Path &outputPath, const std::string &imageName)`
The constructor, which takes the real image data, simulation configuration, a vector of `Spheroid` cells, the output path, and the image name.

### `padRealFrame()`
Pads the real frame.

### `generateSynthFrame()`
Generates a synthetic frame by drawing all the cells.

### `generateSynthFrameFast(Spheroid &oldCell, Spheroid &newCell)`
A faster way to generate a synthetic frame, likely by only updating the regions affected by the changed cells.

### `calculateCost(const std::vector<cv::Mat> &synthFrame)`
Calculates the cost (or error) between the real frame and a synthetic frame.

### `generateOutputFrame()`
Generates an output frame for visualization.

### `generateOutputSynthFrame()`
Generates a synthetic frame for output.

### `length() const`
Returns the number of slices in the frame.

### `perturb()`
Perturbs a cell and returns the cost and a callback function.

### `split()`
Splits a cell and returns the cost and a callback function.

### `gradientDescent()`
Performs gradient descent to optimize cell parameters.

### `getSynthFrame()`
Returns the synthetic frame.

### `costOfPerturb(...)`
Calculates the cost of a specific perturbation.

### `getSynthPerturbedCells(...)`
Generates a synthetic image with perturbed cells.

## `C++/includes/Lineage.hpp`
This file defines the `Lineage` class, which manages the entire sequence of frames and the cells within them over time.

### `processImage(const cv::Mat &image, const BaseConfig &config)`
Processes a single image based on the given configuration.

### `loadFrame(const std::string &imageFile, const BaseConfig &config)`
Loads a frame from a file and processes it.

### `Lineage(std::map<std::string, std::vector<Spheroid>> initialCells, PathVec imagePaths, BaseConfig &config, std::string outputPath, int continueFrom = -1)`
The constructor, which takes the initial cells, a vector of image paths, the configuration, the output path, and an optional index to continue from.

### `optimize(int frameIndex)`
Optimizes the cells in a specific frame.

### `saveFrame(int frameIndex)`
Saves the output of a specific frame.

### `saveCells(int frameIndex)`
Saves the parameters of the cells in a specific frame.

### `copyCellsForward(int to)`
Copies the cells from one frame to the next.

### `length()`
Returns the number of frames in the lineage.

## `C++/src/main.cpp`
This file contains the `main` function, which is the entry point of the program.

### `main(int argc, char *argv[])`
The entry point of the program.

### `getImageFilePaths(const std::string &inputPattern, int firstFrame, int lastFrame, BaseConfig &config)`
This function takes an input pattern, a range of frame numbers, and the configuration and generates a vector of file paths for the images to be processed.

### `loadConfig(const std::string &path, BaseConfig &config)`
This function loads the configuration from a YAML file.

## `C++/src/CellFactory.cpp`
This file implements the `CellFactory` class.

### `CellFactory::CellFactory(const BaseConfig &config)`
The constructor takes a `BaseConfig` object and sets up the factory based on the cell type.

### `CellFactory::createCells(...)`
This method reads cell parameters from a CSV file and creates `Spheroid` cells.

## `C++/src/Frame.cpp`
This file provides the implementation for the `Frame` class.

### `interpolateSlices`
A free function that performs linear interpolation between two image slices to create intermediate slices.

### `Frame::Frame(...)`
Initializes a `Frame` object.

### `Frame::padRealFrame()`
Pads the real image with a border.

### `Frame::generateSynthFrame()`
Generates a synthetic image by drawing all the cells in the `cells` vector.

### `Frame::getImageShape()`
A helper function that returns the size of the images in the real frame.

### `Frame::calculateCost(...)`
Calculates the cost (difference) between the real and synthetic frames.

### `Frame::generateSynthFrameFast(...)`
An optimized version of `generateSynthFrame`.

### `Frame::generateOutputFrame()`
Generates an output image for visualization.

### `Frame::generateOutputSynthFrame()`
Generates an output synthetic image.

### `Frame::length()`
Returns the number of cells in the frame.

### `Frame::perturb()`
Performs a perturbation step of the optimization.

### `Frame::split()`
Performs a cell division step.

### `Frame::costOfPerturb(...)`
A helper function to calculate the cost of a specific perturbation of a single parameter of a cell.

### `Frame::getSynthPerturbedCells(...)`
A helper function that generates a map of synthetic frames, where each frame corresponds to a perturbation of a single parameter of a cell.

### `Frame::gradientDescent()`
This method is commented out, but it outlines a gradient descent optimization algorithm.

### `Frame::getSynthFrame()`
Returns the current synthetic frame.

## `C++/src/Lineage.cpp`
This file implements the `Lineage` class.

### `utils::printMat`
A templated helper function to print the contents of a `cv::Mat` to the console.

### `processImage`
A free function that takes an image and a configuration, converts the image to grayscale, normalizes it to the range [0, 1], and applies a Gaussian blur.

### `loadFrame`
A free function that loads a frame from a file.

### `Lineage::Lineage(...)`
The constructor initializes a `Lineage` object.

### `Lineage::optimize(...)`
This is the core of the optimization process.

### `Lineage::saveFrame(...)`
Saves the output of a specific frame.

### `Lineage::saveCells(...)`
This method is commented out.

### `Lineage::copyCellsForward(...)`
Copies the cells from a completed frame to the next one.

### `Lineage::length()`
Returns the total number of frames in the lineage.

## `C++/src/Cells/Sphere.cpp`
This file implements the methods for the `Sphere` class.

### `Sphere::getRadiusAt(double z) const`
Calculates the radius of the 2D circle that is the intersection of the 3D sphere with a plane at a given `z` coordinate.

### `Sphere::draw(...)`
Draws the sphere on a given image slice.

### `Sphere::drawOutline(...)`
Draws the outline of the sphere on a given image slice.

### `Sphere::getPerturbedCell() const`
Creates and returns a new `Sphere` object with its properties (position and radius) slightly changed.

### `Sphere::getParameterizedCell(...) const`
Creates and returns a new `Sphere` object with its properties changed by the values in the `params` map.

### `Sphere::performPCA(...) const`
Performs Principal Component Analysis (PCA) on a set of 3D points.

### `Sphere::calculateContours(...)`
A helper function for `getSplitCells`.

### `Sphere::getSplitCells(...) const`
This is a complex method that implements the logic for cell division.

### `Sphere::checkConstraints() const`
Checks if the sphere's radius is within the minimum and maximum allowed values defined in the configuration.

### `Sphere::getCellParams() const`
Returns the parameters of the sphere as a `SphereParams` object.

### `Sphere::calculateCorners() const`
Calculates the corners of the bounding box of the sphere.

### `Sphere::calculateMinimumBox(...) const`
Calculates the minimum bounding box that encloses the current sphere and a perturbed sphere.

### `Sphere::checkIfCellsOverlap(...)`
Checks if any two spheres in a given vector of spheres overlap.

## `C++/src/Cells/Spheroid.cpp`
This file implements the `Spheroid` class.

### `Spheroid::getRadiusAt(double z) const`
Calculates the average radius of the elliptical cross-section of the spheroid at a given `z` coordinate.

### `Spheroid::getAxisRadiusAt(double z, double axis_length) const`
Calculates the radius of the spheroid along a specific axis at a given `z` coordinate.

### `Spheroid::draw(...)`
Draws the spheroid on a given image slice.

### `Spheroid::drawOutline(...)`
Draws the outline of the spheroid on a given image slice.

### `Spheroid::getPerturbedCell() const`
Creates and returns a new `Spheroid` object with its properties (position and axes lengths) slightly changed.

### `Spheroid::getParameterizedCell(...) const`
Creates and returns a new `Spheroid` object with its properties changed by the values in the `params` map.

### `Spheroid::performPCA(...) const`
This method is identical to the one in the `Sphere` class.

### `Spheroid::getSplitCells(...) const`
This method is also very similar to the one in the `Sphere` class.

### `Spheroid::checkConstraints() const`
Checks if the spheroid's radius and axes lengths are within the allowed range.

### `Spheroid::getCellParams() const`
Returns the parameters of the spheroid as a `SpheroidParams` object.

### `Spheroid::calculateCorners() const`
Calculates the corners of the bounding box of the spheroid.

### `Spheroid::calculateMinimumBox(...) const`
Calculates the minimum bounding box that encloses the current spheroid and a perturbed one.

### `Spheroid::checkIfCellsOverlap(...)`
Checks if any two spheroids in a given vector overlap.
