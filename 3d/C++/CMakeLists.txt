cmake_minimum_required(VERSION 3.10)

project("celluniverse")
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(cwd ${CMAKE_CURRENT_SOURCE_DIR})

add_executable( ${PROJECT_NAME}
        ${cwd}/src/main.cpp
        ${cwd}/src/Lineage.cpp
        ${cwd}/src/Frame.cpp
        ${cwd}/src/Cells/Cell.cpp
        ${cwd}/src/CellUniverse.cpp
        ${cwd}/src/args.cpp
        ${cwd}/src/Cells/Bacilli.cpp
        ${cwd}/src/Cells/Sphere.cpp
)

target_include_directories( ${PROJECT_NAME} PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${cwd}/includes
        ${cwd}/src
        ${cwd}/src/Cells
        ${cwd}/src/Config
)

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})
add_subdirectory(tests)