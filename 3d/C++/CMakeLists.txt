cmake_minimum_required(VERSION 3.10)

project("celluniverse")
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(cwd ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(${PROJECT_NAME}
        ${cwd}/src/main.cpp
        src/CellFactory.cpp
        #        ${cwd}/src/Lineage.cpp
#        ${cwd}/src/Frame.cpp
        ${cwd}/src/Cells/Sphere.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
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