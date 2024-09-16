cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(Testing_Project VERSION 0.1.0 LANGUAGES CXX C)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

add_definitions(-DGTEST)

# Cmake option for test enabling
option(WITH_TESTS "Turn on testing" ON)
if (WITH_TESTS)
	# option responsible for target "test" in make
	enable_testing()
    message("Tests are enabled")
else()
    message("Tests are disabled")
endif()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)
