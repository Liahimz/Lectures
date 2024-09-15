cmake_minimum_required(VERSION 3.10)
project(OpenCV_Project)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

add_definitions(-DGTEST)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)
