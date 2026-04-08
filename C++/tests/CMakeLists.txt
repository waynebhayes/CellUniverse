include(FetchContent)

# Optional: silence deprecation warnings from third-party projects (like googletest)
# This affects configure-time warnings only.
set(_OLD_CMAKE_WARN_DEPRECATED ${CMAKE_WARN_DEPRECATED})
set(CMAKE_WARN_DEPRECATED OFF)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(googletest)

# Restore original setting
set(CMAKE_WARN_DEPRECATED ${_OLD_CMAKE_WARN_DEPRECATED})

add_executable(core_unit_test
        hello_test.cc
        config_types_test.cc
        interpolate_test.cc
        frame_test.cc
        process_image_test.cc
        spheroid_test.cc
        spheroid_params_test.cc
        spheroid_rotation_test.cc
        ${CMAKE_SOURCE_DIR}/src/Frame.cpp
        ${CMAKE_SOURCE_DIR}/src/CellUniverse.cpp
        ${CMAKE_SOURCE_DIR}/src/Spheroid.cpp
)

target_include_directories(core_unit_test PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/includes
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/lib/yaml-cpp/include
)

target_link_libraries(core_unit_test PRIVATE
        gtest_main
        ${OpenCV_LIBS}
        yaml-cpp
)

include(GoogleTest)
gtest_discover_tests(core_unit_test
        DISCOVERY_MODE PRE_TEST
        DISCOVERY_TIMEOUT 60
)
