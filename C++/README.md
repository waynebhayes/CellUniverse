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
