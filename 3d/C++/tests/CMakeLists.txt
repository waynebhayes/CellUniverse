include(FetchContent)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

add_executable(hello_test hello_test.cc)

target_link_libraries(hello_test gtest_main)

include(GoogleTest)

gtest_discover_tests(hello_test)
