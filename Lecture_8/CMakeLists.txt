cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(MNISTConvNet LANGUAGES CXX CUDA)

# Enable C++11 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
find_package(CUDA REQUIRED)

# OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
add_executable(mnist_convnet src/simple_convnet.cu)

# Link CUDA runtime library and OpenCV libraries
target_link_libraries(mnist_convnet ${CUDA_LIBRARIES} ${OpenCV_LIBS})

# Include the necessary CUDA directories
include_directories(${CUDA_INCLUDE_DIRS})





