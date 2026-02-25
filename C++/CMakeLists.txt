cmake_minimum_required(VERSION 3.10)

project("celluniverse")
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(cwd ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(${PROJECT_NAME}
        ${cwd}/src/main.cpp
        ${cwd}/src/CellFactory.cpp
        ${cwd}/src/Lineage.cpp
        ${cwd}/src/Frame.cpp
        ${cwd}/src/Cells/Sphere.cpp
        ${cwd}/src/Cells/Spheroid.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${cwd}/includes
        ${cwd}/src
        ${cwd}/src/Cells
)

# Build third-party libs
add_subdirectory(lib/yaml-cpp)
add_subdirectory(tests)

# Link dependencies (Eigen is header-only, but this supplies include dirs)
target_link_libraries(${PROJECT_NAME} PRIVATE
        ${OpenCV_LIBS}
        yaml-cpp
)

# If you vendor yaml-cpp, it should already export include dirs via the target;
# keep the manual include only if your yaml-cpp target doesn't.
# target_include_directories(${PROJECT_NAME} PRIVATE lib/yaml-cpp/include)