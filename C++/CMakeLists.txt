cmake_minimum_required(VERSION 3.10)

# Fix FetchContent robustness warning with CMake 3.24+/4.x
# CMP0135: DOWNLOAD_EXTRACT_TIMESTAMP default behavior
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

project(celluniverse LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

set(cwd ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(${PROJECT_NAME}
        ${cwd}/src/main.cpp
        ${cwd}/src/CellFactory.cpp
        ${cwd}/src/Lineage.cpp
        ${cwd}/src/Frame.cpp
        ${cwd}/src/LineageViewer.cpp
        ${cwd}/src/Cells/Sphere.cpp
        ${cwd}/src/Cells/Spheroid.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${cwd}/includes
        ${cwd}/src
        ${cwd}/src/Cells
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} yaml-cpp)
target_include_directories(${PROJECT_NAME} PRIVATE lib/yaml-cpp/include)
add_subdirectory(tests)
add_subdirectory(lib/yaml-cpp)