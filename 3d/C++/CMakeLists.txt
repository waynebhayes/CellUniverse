cmake_minimum_required(VERSION 3.10)

project("celluniverse")

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories( ${OpenCV_INCLUDE_DIRS}
        includes)

add_executable( ${PROJECT_NAME}
        src/main.cpp
        src/Lineage.cpp
        src/Frame.cpp
        src/Cells/Cell.cpp
        src/CellUniverse.cpp
        src/args.cpp
        src/Cells/Bacilli.cpp
        src/Cells/Sphere.cpp
)

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})
add_subdirectory(tests)