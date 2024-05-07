# 3D CellUniverse CPP-VERSION

## Quick Start

- Using ICS openlab
  - ssh into your ICS openlab. (Recommend to use vscode remote development tool: https://code.visualstudio.com/docs/remote/ssh)
  - clone the repo and cd into the project folder
  - In the terminal type in `cd 3d/C++` to go to the root of C++ CellUniverse
  - Add the external YAML-CPP library
  ```
    mkdir lib && cd lib
    git clone https://github.com/jbeder/yaml-cpp
  ```
  - build and compile the project using cmake
  ```
    cd ..
    mkdir build && cd build
    cmake ..
    cmake --build .
  ```
  - Run the bash script in the examples folder
  ```
    cd ../../examples/3d
    ./runcpp.sh
  ```
    - Change the mode of the bash script if you don't have the permission by running the following command
    ```
      chmod 777 runcpp.sh
    ```
  - Wait for celluniverse to finish executing and you can check the result in the `output/` folder inside the current
  example