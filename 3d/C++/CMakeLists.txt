cmake_minimum_required(VERSION 3.10)

project("celluniverse")

find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS}
        src
        src/Cells
        src/Config)

add_executable( ${PROJECT_NAME}
        src/main.cpp
        src/Lineage.cpp
        src/Frame.cpp
        src/Cells/Cell.cpp
        src/CellUniverse.cpp
        src/args.cpp
        src/Cells/Bacilli.cpp
        src/Cells/Sphere.cpp)

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})