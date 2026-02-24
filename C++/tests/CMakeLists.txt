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

add_executable(hello_test hello_test.cc)
target_link_libraries(hello_test PRIVATE gtest_main)

include(GoogleTest)
gtest_discover_tests(hello_test)