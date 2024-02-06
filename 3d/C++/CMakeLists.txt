cmake_minimum_required(VERSION 3.10)

project("celluniverse")
set(OpenCV_DIR	C:\Users/faisa/Downloads/opencv/build)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} src src/Cells src/Config)

add_executable( ${PROJECT_NAME} src/main.cpp src/Lineage.cpp src/Frame.cpp src/Cells/Cell.cpp)

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})