# Include the src directory to access image_processor.h
include_directories(${CMAKE_SOURCE_DIR}/src)

# Download GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# Download doctest
include(FetchContent)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11
)
FetchContent_MakeAvailable(doctest)
include_directories(${doctest_SOURCE_DIR}/doctest)

# Unit tests with GoogleTest
add_executable(runGTests test_image_processor_gtest.cpp)
target_link_libraries(runGTests image_processor gtest gtest_main ${OpenCV_LIBS})

# Unit tests with doctest
add_executable(runDocTests test_image_processor_doctest.cpp)
target_link_libraries(runDocTests image_processor ${OpenCV_LIBS})

# Add testing
if (WITH_TESTS)
    add_test(NAME test_gtest
             COMMAND runGTests)
    add_test(NAME test_doctest
             COMMAND runDocTests)
endif()
