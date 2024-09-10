cmake_minimum_required(VERSION 3.10)

# Set project name
project(OpenCV_Eigen_Project)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package (dynamically link OpenCV)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)  # For MyEigen.hpp

# Add Eigen include directory (update path if necessary)

include_directories(${CMAKE_SOURCE_DIR}/thirdparty/eigen)

# Add executable
add_executable(${PROJECT_NAME} src/main.cpp)

# Add subdirectory for src
add_subdirectory(src)

# Link OpenCV (dynamic linking)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})


if (EXISTS "${PROJECT_BINARY_DIR}/testdata")
    message(STATUS "testdata folder already in place")
else()
    message(STATUS "testdata folder not in place")
    if (EXISTS "${PROJECT_SOURCE_DIR}/testdata")
        message(STATUS "Moving testdata folder to executable directory")
        file(COPY "${PROJECT_SOURCE_DIR}/testdata" DESTINATION "${PROJECT_BINARY_DIR}")
        # file(REMOVE_RECURSE "${PROJECT_SOURCE_DIR}/testdata")
    else()
        message(FATAL_ERROR "Failed to find testdata folder! Try getting testdata folder from original source and place it where CMakelists.txt is located")
    endif()
endif()